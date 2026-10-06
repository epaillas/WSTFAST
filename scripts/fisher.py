#!/usr/bin/env python
"""Fisher information of WST coefficients versus the matter power spectrum, on one Quijote box.

Derivatives come from the theory models (the emulated WST basis, and linear + one-loop SPT with a
counterterm for P(k)); the covariance is the joint sample covariance of the measured data vectors,
which are stored in the same files, so P(k) + WST includes their cross-covariance. Nuisance
parameters are marginalised: cs2 and the noise amplitudes for the WST, cs2 for P(k) (the particle
shot noise V / N of matter is known; a free amplitude would be degenerate with A_s). Example:

    python scripts/fisher.py --q 0.8 --kmax 0.1 0.15 0.2
    python scripts/fisher.py --vary omega_cdm logA n_s h omega_b --emulator outputs/emulators/wst_basis_taylor_superset_5p.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from cosmoprimo import Cosmology
from desilike import build
from wstfast.calculators import WSTTheory
from wstfast.config import QUIJOTE_COSMOLOGY, select_coefficients
from wstfast.data import load_dataset, load_measurement
from wstfast.inference import load_emulated_basis
from wstfast.theory.perturbation import OneLoopMatter, log_interpolator

STEPS = {"omega_cdm": 0.002, "logA": 0.01, "n_s": 0.005, "h": 0.005, "omega_b": 0.0005, "cs2": 0.5, "noise": 0.02}


def load_power(files, kmax, rebin):
    """Monopole of delta (shot noise included, no window deconvolution) for k <= kmax, rebinned by modes."""
    with np.load(files[0]) as data:
        k, nmodes = data["k"], data["nmodes"]
    nk = (np.searchsorted(k, kmax, side="right") // rebin) * rebin
    weights = nmodes[:nk].reshape(-1, rebin)
    kbin = (k[:nk].reshape(-1, rebin) * weights).sum(axis=1) / weights.sum(axis=1)
    power = []
    for path in files:
        with np.load(path) as data:
            power.append((data["Pdd"][0, :nk].reshape(-1, rebin) * weights).sum(axis=1) / weights.sum(axis=1))
    return kbin, np.array(power)


class PowerModel:
    """W^2(k) [P_L + P_1loop - 2 cs2 k^2 P_L] + V / N at the measured bin centres."""

    def __init__(self, k, cellsize, shotnoise, z):
        self.k, self.shotnoise, self.z = k, shotnoise, z
        self.window2 = np.exp(-(k * cellsize) ** 2 / 6.0)
        self.klin = np.geomspace(1e-4, 10.0, 1024)
        self.loop = OneLoopMatter(k)

    def __call__(self, cs2=0.0, **cosmology):
        params = dict(QUIJOTE_COSMOLOGY, **cosmology)
        cosmo = Cosmology(engine="class", m_ncdm=0.0, **params)
        pklin = cosmo.get_fourier().pk_interpolator(of="delta_m")(self.klin, z=self.z)
        pk = log_interpolator(self.klin, pklin)
        plin = np.asarray(pk(self.k))
        return self.window2 * (plin + np.asarray(self.loop(pk)) - 2 * cs2 * self.k**2 * plin) \
            + self.shotnoise


def derivatives(model, center, names):
    """Central finite differences of model(**params) with respect to ``names``."""
    out = []
    for name in names:
        step = STEPS["noise" if name.startswith("noise") else name]
        plus, minus = dict(center, **{name: center[name] + step}), dict(center, **{name: center[name] - step})
        out.append((model(plus) - model(minus)) / (2 * step))
    return np.array(out)


def marginalised_errors(jacobian, covariance, nreal, keep):
    """1-sigma errors on the first ``keep`` parameters, with the Hartlap-corrected precision."""
    ndata = covariance.shape[0]
    precision = np.linalg.inv(covariance) * (nreal - ndata - 2.0) / (nreal - 1.0)
    fisher = jacobian @ precision @ jacobian.T
    inverse = np.linalg.inv(fisher)
    return np.sqrt(np.diag(inverse)[:keep]), inverse[:keep, :keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256"))
    parser.add_argument("--emulator", type=Path, default=Path("outputs/emulators/wst_basis_taylor_superset.h5"))
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--kmax", type=float, nargs="+", default=[0.1, 0.15, 0.2])
    parser.add_argument("--rebin", type=int, default=2, help="P(k) bins of rebin x k_f")
    parser.add_argument("--vary", nargs="+", default=["omega_cdm", "logA"], choices=sorted(QUIJOTE_COSMOLOGY),
                        help="cosmological parameters (must match the emulator's)")
    parser.add_argument("--fix-noise", action="store_true",
                        help="keep the WST noise amplitudes fixed (they fit to ~0 with the tree-level noise)")
    parser.add_argument("--output", type=Path, default=Path("outputs/fisher/fisher.json"))
    args = parser.parse_args()
    cosmology = tuple(args.vary)
    ncosmo = len(cosmology)

    first = sorted((args.data_dir / "real").glob("wst_r*.npz"))[0]
    config = load_measurement(first, q=args.q)["config"]
    selections = {"WST dyadic": [c for c in select_coefficients(config)
                                 if c.j % 2 == 0 and (c.j2 is None or c.j2 % 2 == 0)],
                  "WST half-octave": select_coefficients(config),
                  "WST half-octave, s21 >= 17.7": select_coefficients(config, s21_min_scale=17.6)}
    fiducial = {name: QUIJOTE_COSMOLOGY[name] for name in cosmology}

    # Fix the realizations once: measurements may still be arriving while this runs.
    files = [str(path) for path in sorted((args.data_dir / "real").glob("wst_r*.npz"))]
    datasets, wst = {}, {}
    for label, coefficients in selections.items():
        dataset = load_dataset(args.data_dir, "real", coefficients, q=args.q)
        rows = {path: i for i, path in enumerate(dataset.files)}
        dataset.vectors, dataset.files = dataset.vectors[[rows[path] for path in files]], files
        basis, _ = load_emulated_basis(args.emulator, dataset, cosmology)
        graph = build(WSTTheory(coefficients, config=dataset.config, basis=basis))
        names = list(cosmology) + ["cs2"]  # cosmology first: marginalised_errors keeps the leading rows
        if not args.fix_noise:
            names += sorted({f"noise_j{c.j}_l{c.ell}" for c in coefficients if c.kind == "S21"})
        center = dict(fiducial, **{name: 0.0 for name in names[2:]})
        datasets[label] = dataset
        wst[label] = (derivatives(lambda p: np.asarray(graph(p)), center, names), dataset.vectors)
    meta = datasets["WST dyadic"].metadata
    nreal = len(files)

    results = dict(q=args.q, nrealizations=nreal, volume="1 (Gpc/h)^3", parameters=list(cosmology),
                   fix_noise=args.fix_noise, cases={})

    def record(label, jacobian, vectors):
        errors, covariance = marginalised_errors(jacobian, np.cov(vectors, rowvar=False), nreal, len(cosmology))
        correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        results["cases"][label] = dict(ndata=int(vectors.shape[1]), sigma=dict(zip(cosmology, errors.tolist())),
                                       correlation=float(correlation))
        print(f"{label:46s} n={vectors.shape[1]:3d}  " + "  ".join(
            f"sigma({name})={value:.4g}" for name, value in zip(cosmology, errors)) + f"  r={correlation:+.2f}")

    for label, (jacobian, vectors) in wst.items():
        record(label, jacobian, vectors)
    for kmax in args.kmax:
        k, power = load_power(files, kmax, args.rebin)
        model = PowerModel(k, config.cellsize, meta["boxsize"] ** 3 / meta["nparticles"], meta["redshift"])
        jac_p = derivatives(lambda p: model(**p), dict(fiducial, cs2=0.0), list(cosmology) + ["cs2"])
        record(f"P(k), kmax={kmax}", jac_p, power)
        for label, (jac_w, vectors) in wst.items():
            # Block-diagonal Jacobian: cosmology shared, nuisance parameters separate.
            nw, npw = jac_w.shape[0] - ncosmo, jac_p.shape[0] - ncosmo
            joint = np.zeros((ncosmo + nw + npw, jac_w.shape[1] + jac_p.shape[1]))
            joint[:ncosmo] = np.concatenate([jac_w[:ncosmo], jac_p[:ncosmo]], axis=1)
            joint[ncosmo:ncosmo + nw, :jac_w.shape[1]] = jac_w[ncosmo:]
            joint[ncosmo + nw:, jac_w.shape[1]:] = jac_p[ncosmo:]
            record(f"P(k), kmax={kmax} + {label}", joint, np.concatenate([vectors, power], axis=1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
