#!/usr/bin/env python
"""WST of Gaussian random fields with the field power of an analytic covariance run, for its Gaussian-limit check.

Each mock is a Gaussian field on the measurement mesh with |delta_k|^2 = P_f(k) (window and shot noise included), read
from the output of scripts/analytic_covariance.py, so that ``--inputs gaussian`` there is the exact reference for the
sample covariance of these mocks. Only the coefficients of the analytic file are measured (torch backend), for the
exponents of the N-body files. One file per seed, mock_{seed:05d}.npz; existing files are skipped.

On Apple GPUs (MPS) single coefficients were occasionally corrupted (1-60% off, ~1 field in 6): run a second pass
with ``--pass2`` (written to <output>/pass2/) and covariance_vs_nbody.py keeps only the mocks both passes reproduce.

    python scripts/feasibility/gaussian_mocks_wst.py outputs/covariance/analytic_gaussian.npz --seeds 0-999 --device mps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from wstfast.config import WSTConfig
from wstfast.measure_torch import TorchLattice, default_device, wavelet_modulus

DATA = Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256")
QS = (0.5, 0.8, 1.0, 2.0)


def parse_label(label):
    parts = label.split("_")
    if parts[0] == "S1":
        return ("S1", int(parts[2][1:]), int(parts[1][1:]), None)
    return ("S21", int(parts[3][1:]), int(parts[1][1:]), int(parts[2][1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analytic", type=Path, help="output of scripts/analytic_covariance.py (field power, labels)")
    parser.add_argument("--data-dir", type=Path, default=DATA, help="N-body measurements (mesh, box, WST settings)")
    parser.add_argument("--seeds", default="0-999", help="inclusive range of seeds")
    parser.add_argument("--output", type=Path, default=Path("outputs/covariance/gaussian_mocks"))
    parser.add_argument("--pass2", action="store_true", help="recompute every seed into <output>/pass2/")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    analytic = np.load(args.analytic)
    with np.load(sorted((args.data_dir / "real").glob("wst_r*.npz"))[0]) as first:
        meta = json.loads(str(first["metadata"]))
    config = WSTConfig(**meta["config"])
    n, boxsize = meta["nmesh"], meta["boxsize"]
    device = default_device(args.device)
    lattice = TorchLattice(n, device=device)
    k = lattice.numpy.kmag.astype("f8") * n / boxsize
    logk, logp = np.log(analytic["pfield_k"]), np.log(analytic["pfield"])
    power = np.where(k > 0, np.exp(np.interp(np.log(np.where(k > 0, k, 1.0)), logk, logp)), 0.0)
    amplitude = torch.from_numpy(np.sqrt(power * n**3 / boxsize**3).astype("f4")).to(device)

    coefficients = [parse_label(label) for label in analytic["labels"]]
    first_layer = sorted({(j, ell) for _, ell, j, _ in coefficients})
    second = {}
    for kind, ell, j, j2 in coefficients:
        if kind == "S21":
            second.setdefault((j, ell), []).append(j2)
    output = args.output / "pass2" if args.pass2 else args.output
    output.mkdir(parents=True, exist_ok=True)
    first_seed, _, last_seed = args.seeds.partition("-")

    def moments(u):
        return [float((u.abs() ** q).mean(dtype=torch.float32)) for q in QS]

    for seed in range(int(first_seed), int(last_seed or first_seed) + 1):
        path = output / f"mock_{seed:05d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        white = torch.randn((n, n, n), generator=torch.Generator(device="cpu").manual_seed(seed)).to(device)
        fk = torch.fft.rfftn(white) * amplitude
        s1 = np.full((len(QS), config.J + 1, config.L + 1), np.nan)
        s2 = np.full((len(QS), config.J + 1, config.J + 1, config.lmax2 + 1), np.nan)
        for j, ell in first_layer:
            u = wavelet_modulus(fk, lattice, config.sigma0 * config.step**j, ell)
            s1[:, j, ell] = moments(u)
            if (j, ell) in second:
                mean = u.mean()
                fu = torch.fft.rfftn(u - mean)  # subtract the mean before the float32 FFT, as the estimator does
                fu[0, 0, 0] = mean * n**3
                for j2 in second[j, ell]:
                    s2[:, j, j2, ell] = moments(wavelet_modulus(fu, lattice, config.sigma0 * config.step**j2, ell))
        np.savez(path, S1=s1, S2=s2, q=np.array(QS))
        print(f"seed {seed}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
