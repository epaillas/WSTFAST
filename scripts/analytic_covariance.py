#!/usr/bin/env python
"""Analytic covariance of the real-space WST data vector (wstfast.theory.covariance), one box.

The ingredients (field power, response and noise of the first-layer moduli) come from

* ``--inputs model`` (default): CLASS linear power, one-loop SPT with the counterterm cs2, CIC window and particle shot
  noise; tree-level response r and Gaussian noise g (moduli.py), tree-level noise correction dN (modulus_noise.py) and
  the noise amplitudes a of a fit (``--fit``), N = (1 + a) g + dN;
* ``--inputs measured``: the ensemble means of the spectra stored with the measurements (P_dd, P_Ud, P_UU, <U>);
* ``--inputs gaussian``: the Gaussian-field limit, r = 0 and N = g (the exact reference for Gaussian-field mocks,
  scripts/feasibility/gaussian_mocks_wst.py).

The cross noise of different first-layer fields is not measured: its shape is the Gaussian-chaos g_ab, rescaled by the
geometric mean of the auto ratios N / g. Writes the covariance of ln(data vector) and its parts to ``--output``;
multiply by outer(mean, mean) for the covariance of the data vector. Examples:

    python scripts/analytic_covariance.py --fit outputs/paper/fits.json --output outputs/covariance/analytic_q0.8.npz
    python scripts/analytic_covariance.py --inputs measured --chaos-nmesh 0
    python scripts/feasibility/covariance_vs_nbody.py outputs/covariance/analytic_q0.8.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from wstfast.config import QUIJOTE_COSMOLOGY, select_coefficients
from wstfast.data import load_measurement
from wstfast.theory.covariance import ModulusSpectra, analytic_covariance, cross_noise

DATA = Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256")


def linear_power(z):
    from cosmoprimo import Cosmology

    from wstfast.theory.perturbation import log_interpolator

    klin = np.geomspace(1e-4, 10.0, 1024)
    cosmo = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY)
    return log_interpolator(klin, cosmo.get_fourier().pk_interpolator(of="delta_m")(klin, z=z))


def rebin(x, weights, factor):
    n = (x.shape[-1] // factor) * factor
    x, w = x[..., :n].reshape(*x.shape[:-1], -1, factor), weights[:n].reshape(-1, factor)
    return (x * w).sum(axis=-1) / w.sum(axis=-1)


def measured_inputs(files, fields, config, shotnoise, factor=2):
    """Mean field power, and mean response and noise of the first-layer moduli (building_blocks.py estimators)."""
    first = np.load(files[0])
    k, nmodes = first["k"], first["nmodes"]
    kb = rebin(k, nmodes, factor)
    window = np.exp(-(kb * config.cellsize) ** 2 / 12.0)  # undoes the CIC window carried by delta
    pdd, response, noise = [], [], []
    for path in files:
        with np.load(path) as m:
            p = rebin(m["Pdd"][0], nmodes, factor) - shotnoise
            pud = np.array([rebin(m["PUd"][j, ell, 0], nmodes, factor) for j, ell in fields])
            puu = np.array([rebin(m["PUU"][j, ell, 0], nmodes, factor) for j, ell in fields])
            umean = np.array([m["Umean"][j, ell] for j, ell in fields])[:, None]
            pdd.append(m["Pdd"][0])
            response.append(pud * window / (umean * p))
            noise.append((puu - pud**2 / p) / umean**2)
    return k, np.mean(pdd, axis=0), kb, np.mean(response, axis=0), np.mean(noise, axis=0)


def power_law_interpolator(k, p):
    """log-log interpolation of a tabulated P(k), extrapolated with the slope of the last bins."""
    lk, lp = np.log(k), np.log(p)
    slope = (lp[-1] - lp[-6]) / (lk[-1] - lk[-6])

    def interp(x):
        lx = np.log(np.maximum(x, k[0]))
        return np.exp(np.where(lx <= lk[-1], np.interp(lx, lk, lp), lp[-1] + slope * (lx - lk[-1])))

    return interp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA, help="measurements (settings, shot noise, measured inputs)")
    parser.add_argument("--inputs", choices=("model", "measured", "gaussian"), default="model")
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--s1-min-scale", type=float, default=25.0)
    parser.add_argument("--s21-min-scale", type=float, default=12.5)
    parser.add_argument("--s21-min-ratio", type=float, default=2.8)
    parser.add_argument("--fit", type=Path, default=None,
                        help="fits.json of scripts/paper/run_fits.py: cs2 and noise amplitudes of --fit-run (default: 0)")
    parser.add_argument("--fit-run", default="baseline/model")
    parser.add_argument("--fit-volume", default="1")
    parser.add_argument("--chaos-nmesh", type=int, default=128, help="lattice of the 4th chaos (0: skip it)")
    parser.add_argument("--terms", nargs="+", default=["delta", "noise_level", "u_noise"])
    parser.add_argument("--output", type=Path, default=Path("outputs/covariance/analytic.npz"))
    args = parser.parse_args()

    files = sorted((args.data_dir / "real").glob("wst_r*.npz"))
    first = load_measurement(files[0], q=args.q)
    config, meta = first["config"], first["metadata"]
    boxsize, nmesh, z = meta["boxsize"], meta["nmesh"], meta["redshift"]
    shotnoise = boxsize**3 / meta["nparticles"]
    coefficients = select_coefficients(config, s1_min_scale=args.s1_min_scale, s21_min_scale=args.s21_min_scale,
                                       s21_min_ratio=args.s21_min_ratio)
    fields = sorted({(c.j, c.ell) for c in coefficients if c.kind == "S21"})
    best = {}
    if args.fit is not None:
        best = json.loads(args.fit.read_text())["runs"][args.fit_run]["fits"][args.fit_volume]["best"]
    cs2 = best.get("cs2", 0.0)
    amplitude = {key: best.get(f"noise_j{key[0]}_l{key[1]}", 0.0) for key in fields}

    t0 = time.time()
    pk = linear_power(z)
    plin = lambda k: np.asarray(pk(np.asarray(k)))  # noqa: E731
    sigmas = [(config.sigma(j), ell) for j, ell in fields]
    if args.inputs == "measured":
        kp, pmean, moduli_k, response, noise = measured_inputs(files, fields, config, shotnoise)
        pfield = power_law_interpolator(kp, pmean)
        gauss = cross_noise(moduli_k, plin, sigmas) if fields else None
        responses = {key: response[i] for i, key in enumerate(fields)}
        autos = {key: noise[i] for i, key in enumerate(fields)}
    else:
        kgrid = np.geomspace(2e-3, 0.4, 48)
        gauss = cross_noise(kgrid, plin, sigmas) if fields else None
        from wstfast.theory.perturbation import OneLoopMatter

        kloop = np.geomspace(1e-3, 2.0, 160)
        loop = np.asarray(OneLoopMatter(kloop)(pk))

        def pfield(k):
            pl = plin(k)
            pnl = pl + np.interp(np.log(k), np.log(kloop), loop, right=loop[-1]) - 2 * cs2 * k**2 * pl
            return pnl * np.exp(-(k * config.cellsize) ** 2 / 6.0) + shotnoise

        moduli_k = kgrid
        responses, autos = {}, {}
        if args.inputs == "gaussian":
            for i, key in enumerate(fields):
                responses[key] = np.zeros_like(kgrid)
                autos[key] = gauss[i, i]
        else:
            from wstfast.theory.moduli import ModulusClustering
            from wstfast.theory.modulus_noise import ModulusNoise

            def dpk(x):
                return np.interp(np.log(x), np.log(kloop), loop)

            for i, key in enumerate(fields):
                sigma, ell = config.sigma(key[0]), key[1]
                r_long, r_short, _ = ModulusClustering(kgrid, sigma, ell)(pk)
                noise_model = ModulusNoise(sigma, ell)
                correction = np.asarray(noise_model.interpolate(noise_model(pk, dpk), kgrid))
                responses[key] = np.asarray(r_long + r_short)
                autos[key] = (1 + amplitude[key]) * gauss[i, i] + correction
                print(f"  modulus inputs sigma = {sigma:5.1f}, l = {ell}  ({time.time() - t0:.0f}s)", flush=True)
    moduli = None
    if fields:
        # dN is tabulated for sigma K <= 3 only: the ratio N / g is clipped where g has decayed (no weight there).
        ratio = {key: np.clip(autos[key] / gauss[i, i], 0.1, 10.0) for i, key in enumerate(fields)}
        cross = {(a, b): (autos[a] if a == b else gauss[i, j] * np.sqrt(ratio[a] * ratio[b]))
                 for i, a in enumerate(fields) for j, b in enumerate(fields)}
        moduli = ModulusSpectra(moduli_k, responses, cross)
    print(f"inputs ({args.inputs}) ready in {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    result = analytic_covariance(config, coefficients, args.q, boxsize, nmesh, pfield, plin, moduli,
                                 chaos_nmesh=args.chaos_nmesh or None, include=tuple(args.terms))
    print(f"covariance of {len(coefficients)} coefficients in {time.time() - t1:.0f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ktab = np.geomspace(1e-3, 3.0, 4000)  # field power, for Gaussian-field mocks with the same spectrum
    np.savez(args.output, labels=[c.label for c in coefficients], q=args.q, inputs=args.inputs, terms=args.terms,
             data_dir=str(args.data_dir), fit=str(args.fit), cs2=cs2, pfield_k=ktab, pfield=pfield(ktab), **result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
