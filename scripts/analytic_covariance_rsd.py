#!/usr/bin/env python
"""Analytic covariance of the redshift-space data vector [S1m blocks, P_0, P_2, P_4] (the halo baseline of fit_rsd.py).

The 2nd chaos of the line-of-sight-resolved S1m blocks and the Gaussian covariance of the multipoles, with their cross
covariance, are sums over the 3D lattice with the anisotropic field power P_s(k, mu)
(wstfast.theory.covariance.RSDGaussianCovariance); the 4th chaos of the S1m blocks is evaluated on FFT lattices
(chaos_covariances with |m| components). The multipoles are quadratic in the field: they have no 4th chaos.

``--inputs measured`` (the only one so far) takes P_s(k, mu) = sum_l P_l(k) L_l(mu) from the ensemble-mean stored
multipoles (l = 0, 2, 4; window and shot noise included), extrapolated as a power law beyond the last bin.
Writes the covariance with S1m in ln and P_l linear (``lncov``) and the labels; multiply the S1m rows by the mean S1m
for the covariance of the data vector (RSDGaussianCovariance.covariance). Example:

    python scripts/analytic_covariance_rsd.py --output outputs/covariance/halos_rsd_q0.8.npz
    python scripts/feasibility/covariance_vs_nbody_rsd.py outputs/covariance/halos_rsd_q0.8.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre

from wstfast.config import select_coefficients
from wstfast.data import load_measurement, load_power_dataset
from wstfast.theory.covariance import RSDGaussianCovariance, chaos_covariances

DATA = Path("data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los")


def measured_field_power(files):
    """P_s(k, mu) from the mean stored multipoles (l = 0, 2, 4) of the measurements."""
    pdd = []
    for path in files:
        with np.load(path) as m:
            pdd.append(m["Pdd"])
            k = m["k"]
    multipoles = np.mean(pdd, axis=0)  # (3, nk)
    p0 = multipoles[0]
    slope = np.log(p0[-1] / p0[-6]) / np.log(k[-1] / k[-6])
    ratio = multipoles[:, -1] / p0[-1]

    def pfield(kk, mu):
        kk, mu = np.asarray(kk), np.asarray(mu)
        inside = kk <= k[-1]
        out = np.zeros(np.broadcast(kk, mu).shape)
        for i, ell in enumerate((0, 2, 4)):
            tabulated = np.interp(kk, k, multipoles[i])
            extrapolated = ratio[i] * p0[-1] * (np.maximum(kk, k[-1]) / k[-1]) ** slope
            out = out + np.where(inside, tabulated, extrapolated) * eval_legendre(ell, mu)
        return np.maximum(out, 1e-3 * p0[-1])  # P_s > 0 (the truncation at l = 4 is noisy where P_s is small)

    return pfield


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--s1-min-scale", type=float, default=25.0)
    parser.add_argument("--s1-ells", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--kmin", type=float, default=0.0)
    parser.add_argument("--kmax", type=float, default=0.12)
    parser.add_argument("--rebin", type=int, default=2)
    parser.add_argument("--nmesh", type=int, default=128, help="lattice of the 2nd-chaos sums and cap of the 4th chaos")
    parser.add_argument("--chaos-nmesh", type=int, default=128, help="0: skip the 4th chaos")
    parser.add_argument("--output", type=Path, default=Path("outputs/covariance/halos_rsd.npz"))
    args = parser.parse_args()

    files = sorted((args.data_dir / "rsd").glob("wst_r*.npz"))
    first = load_measurement(files[0], q=args.q)
    config, meta = first["config"], first["metadata"]
    coefficients = [c for c in select_coefficients(config, s1_min_scale=args.s1_min_scale, s1_ells=args.s1_ells)
                    if c.kind == "S1"]
    power = load_power_dataset(args.data_dir, "rsd", kmin=args.kmin, kmax=args.kmax, rebin=args.rebin, files=files,
                               ells=(0, 2, 4))
    t0 = time.time()
    pfield = measured_field_power(files)
    print(f"field power from {len(files)} boxes ({time.time() - t0:.0f}s)", flush=True)

    t1 = time.time()
    gauss = RSDGaussianCovariance(config, coefficients, args.q, meta["boxsize"], pfield, pk_edges=power.edges,
                                  nmesh=args.nmesh)
    print(f"2nd chaos of {len(gauss.labels)} entries in {time.time() - t1:.1f}s", flush=True)
    lncov = gauss.lncov.copy()
    chaos4 = np.zeros_like(lncov)
    if args.chaos_nmesh:
        t2 = time.time()
        fields = [(config.sigma(c.j), c.ell, m) for c in coefficients for m in range(c.ell + 1)]
        _, cov4 = chaos_covariances(fields, args.q, meta["boxsize"], args.chaos_nmesh,
                                    lambda a, b, k, mu: pfield(k, mu), anisotropic=True)
        chaos4[:gauss.ns1m, :gauss.ns1m] = cov4
        lncov += chaos4
        print(f"4th chaos of {len(fields)} S1m blocks in {time.time() - t2:.0f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, labels=gauss.labels, ns1m=gauss.ns1m, q=args.q, data_dir=str(args.data_dir),
             inputs="measured", lncov=lncov, gauss=gauss.lncov, chaos4=chaos4, pk_edges=power.edges)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
