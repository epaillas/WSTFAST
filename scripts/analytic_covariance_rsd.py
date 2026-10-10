#!/usr/bin/env python
"""Analytic covariance of the redshift-space data vector [S1m blocks, P_0, P_2, P_4] (the halo baseline of fit_rsd.py).

The 2nd chaos of the line-of-sight-resolved S1m blocks and the Gaussian covariance of the multipoles, with their cross
covariance, are sums over the 3D lattice with the anisotropic field power P_s(k, mu)
(wstfast.theory.covariance.RSDGaussianCovariance); the 4th chaos of the S1m blocks is evaluated on FFT lattices
(chaos_covariances with |m| components). The multipoles are quadratic in the field: they have no 4th chaos.

The field power P_s(k, mu) (window and shot noise included) comes from

* ``--inputs measured``: sum_l P_l(k) L_l(mu) of the ensemble-mean stored multipoles (l = 0, 2, 4), extrapolated as
  a power law beyond the last bin;
* ``--inputs model``: the redshift-space model of fit_rsd.py (``--emulator``) at the Quijote cosmology and the
  nuisance values of ``--fiducial`` fit summaries: W^2 P_grid(k, mu) + (1 + alpha0) (V / N) S(k), with the tree, loop,
  P_l counterterms and alpha2 k^2 mu^2 of the grid, and the isotropic leading forms W^2 = S = exp(-(k H)^2 / 6) of the
  CIC window and aliased shot noise (H the cell size; < 1e-3 off for k < 0.25 h/Mpc).
Writes the covariance with S1m in ln and P_l linear (``lncov``) and the labels; multiply the S1m rows by the mean S1m
for the covariance of the data vector (RSDGaussianCovariance.covariance). Example:

    python scripts/analytic_covariance_rsd.py --inputs model --output outputs/covariance/halos_rsd_model_q0.8.npz
    python scripts/feasibility/covariance_vs_nbody_rsd.py outputs/covariance/halos_rsd_model_q0.8.npz \
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir.h5

With model inputs no measurement enters but the settings (mesh, box, shot noise V / N) of ``--data-dir``. The model
is only fitted to k <= 0.12 h/Mpc and departs from the measured multipoles above k ~ 0.15, where the S1m filters
(sigma >= 25 Mpc/h) have little weight: on Quijote halos both inputs give the same covariance to the sampling noise,
and the model removes the low-k P_4 noise of the measured inputs.
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


def model_field_power(data_dir, emulator, fiducial_files, cellsize):
    """P_field(k, mu) of the fit_rsd.py model: the P_s grid of RSDPowerTheory (identity projection) at the fiducial."""
    import json
    import sys

    import jax.numpy as jnp
    from desilike import build, get_params
    from scipy.interpolate import RegularGridInterpolator

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import wstfast.theory  # noqa: F401  (enables JAX double precision)
    from fit_rsd import load_basis
    from wstfast.calculators import RSDPowerTheory
    from wstfast.config import QUIJOTE_COSMOLOGY
    from wstfast.theory.rsd import RSDGrid

    basis, settings = load_basis(emulator, ("omega_cdm", "logA", "n_s", "h"))
    grid = RSDGrid(kmax=settings["kmax"])
    nk, nmu = grid.k.size, grid.mu.size

    class Identity:  # projection whose output is the P_s grid itself (no shot noise)
        matrix = np.eye(nk * nmu)
        shot = np.zeros(nk * nmu)
        k2mu2 = (grid.k[:, None] * grid.mu[None, :]) ** 2
        k2 = grid.k[:, None] ** 2 * np.ones_like(grid.mu)[None, :]

    first = sorted((data_dir / "rsd").glob("wst_r*.npz"))[0]
    meta = json.loads(str(np.load(first)["metadata"]))
    shotnoise = meta["boxsize"] ** 3 / meta["nparticles"]
    theory = RSDPowerTheory(Identity(), basis=basis, shotnoise=shotnoise, tracer=settings.get("tracer", "matter"))
    names = get_params(theory).select(varied=True, derived=False).names()
    values = {name: QUIJOTE_COSMOLOGY[name] for name in ("omega_cdm", "logA", "n_s", "h") if name in names}
    for path in fiducial_files:
        best = json.loads(Path(path).read_text())["bestfit"]
        values.update({k: float(v) for k, v in best.items() if k in names and k not in values})
    missing = [n for n in names if n not in values]
    if missing:
        print(f"  nuisances at their default values: {missing}")
    pgrid = np.asarray(build(theory)(values)).reshape(nk, nmu)
    alpha0 = values.get("alpha0", 0.0)
    print("  fiducial: " + ", ".join(f"{k}={v:.4g}" for k, v in values.items()))
    # P_s is even in mu: interpolate in (ln k, mu^2) on the grid, power-law in k beyond it (filters negligible there).
    interp = RegularGridInterpolator((np.log(grid.k), grid.mu**2), pgrid, bounds_error=False, fill_value=None)
    slope = np.log(pgrid[-1] / pgrid[-6]) / np.log(grid.k[-1] / grid.k[-6])

    def pfield(k, mu):
        k, mu2 = np.broadcast_arrays(np.asarray(k, "f8"), np.asarray(mu, "f8") ** 2)
        mu2 = np.clip(mu2, grid.mu[0] ** 2, grid.mu[-1] ** 2)
        kin = np.clip(k, grid.k[0], grid.k[-1])
        p = interp(np.stack([np.log(kin), mu2], axis=-1))
        last = interp(np.stack([np.full_like(k, np.log(grid.k[-1])), mu2], axis=-1))
        tail = last * (np.maximum(k, grid.k[-1]) / grid.k[-1]) ** np.interp(mu2, grid.mu**2, slope)
        p = np.where(k <= grid.k[-1], p, tail)
        damping = np.exp(-(k * cellsize) ** 2 / 6.0)
        return np.maximum(damping * p + (1 + alpha0) * shotnoise * damping, 1e-3 * shotnoise)

    return pfield


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--inputs", choices=("measured", "model"), default="measured")
    parser.add_argument("--emulator", type=Path, default=Path("outputs/emulators/rsd_biased_basis_taylor_4p_ir.h5"))
    parser.add_argument("--fiducial", type=Path, nargs="+",
                        default=[Path("outputs/inference/rsd/halos_bias_only_V60_pk/summary.json")],
                        help="fit summaries whose best fits set the nuisances (later files take precedence)")
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
    if args.inputs == "measured":
        pfield = measured_field_power(files)
        print(f"field power from {len(files)} boxes ({time.time() - t0:.0f}s)", flush=True)
    else:
        pfield = model_field_power(args.data_dir, args.emulator, args.fiducial, config.cellsize)
        print(f"model field power ({time.time() - t0:.0f}s)", flush=True)

    t1 = time.time()
    gauss = RSDGaussianCovariance(config, coefficients, args.q, meta["boxsize"], pfield, pk_edges=power.edges,
                                  nmesh=args.nmesh)
    print(f"2nd chaos of {len(gauss.labels)} entries in {time.time() - t1:.1f}s", flush=True)
    lncov = gauss.lncov.copy()
    chaos4 = np.zeros_like(lncov)
    if args.chaos_nmesh:
        t2 = time.time()
        fields = [(config.sigma(c.j), c.ell, m) for c in coefficients for m in range(c.ell + 1)]
        cache = {}  # every block sees the same field: evaluate P_s once per lattice (and per envelope grid)

        def spectra(a, b, k, mu):
            key = (k.shape, float(k.flat[0]), float(k.flat[-1]), float(mu.flat[-1]))
            if key not in cache:
                cache[key] = pfield(k, mu)
            return cache[key]

        _, cov4 = chaos_covariances(fields, args.q, meta["boxsize"], args.chaos_nmesh, spectra, anisotropic=True)
        chaos4[:gauss.ns1m, :gauss.ns1m] = cov4
        lncov += chaos4
        print(f"4th chaos of {len(fields)} S1m blocks in {time.time() - t2:.0f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, labels=gauss.labels, ns1m=gauss.ns1m, q=args.q, data_dir=str(args.data_dir),
             inputs=args.inputs, lncov=lncov, gauss=gauss.lncov, chaos4=chaos4, pk_edges=power.edges)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
