#!/usr/bin/env python
"""Analytic covariance of the redshift-space halo WST data vector [ln S1m, ln S21m, P_l] (wstfast.theory.covariance_rsd).

Model inputs at the nuisance values of a fit_rsd.py summary (``--fiducial``, e.g. the joint P_l + WST baseline fit):
the field power P_d(k, mu) of the fit model (as analytic_covariance_rsd.py --inputs model), the tree response and
Gaussian-chaos cross noise of the first-layer moduli at the fitted biases and shot noise, with the fitted noise
amplitudes and white terms W_j. Writes ``lncov`` (S1m and S21m in ln, P_l linear) and its parts. Example:

    python scripts/analytic_covariance_rsd_wst.py --fiducial outputs/inference/rsd/paper_c_joint_nsfixed_V60/summary.json \
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir_sv1_cubic.h5 --output outputs/covariance/halos_wst_model.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytic_covariance_rsd import DATA, model_field_power  # noqa: E402

from wstfast.config import QUIJOTE_COSMOLOGY, select_coefficients  # noqa: E402
from wstfast.data import load_measurement, load_power_dataset  # noqa: E402
from wstfast.theory.covariance_rsd import ModulusCrossRSD, ModulusTables, RSDWSTCovariance  # noqa: E402
from wstfast.theory.rsd_moduli import first_layer_fields, s21m_entries  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--emulator", type=Path, default=Path("outputs/emulators/rsd_biased_basis_taylor_4p_ir_sv1_cubic.h5"))
    parser.add_argument("--fiducial", type=Path, nargs="+", required=True)
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--s1-min-scale", type=float, default=25.0)
    parser.add_argument("--s21-min-scale", type=float, default=25.0)
    parser.add_argument("--s21-min-scale2", type=float, default=70.0)
    parser.add_argument("--s21-min-ratio", type=float, default=2.8)
    parser.add_argument("--kmax", type=float, default=0.12)
    parser.add_argument("--rebin", type=int, default=2)
    parser.add_argument("--nmesh", type=int, default=128)
    parser.add_argument("--chaos-nmesh", type=int, default=128, help="0: skip the 4th chaos")
    parser.add_argument("--output", type=Path, default=Path("outputs/covariance/halos_wst_model.npz"))
    args = parser.parse_args()

    files = sorted((args.data_dir / "rsd").glob("wst_r*.npz"))
    config = load_measurement(files[0], q=args.q)["config"]
    s1m = [c for c in select_coefficients(config, s1_min_scale=args.s1_min_scale) if c.kind == "S1"]
    s21 = [c for c in select_coefficients(config, s21_min_scale=args.s21_min_scale, s21_min_ratio=args.s21_min_ratio,
                                          s21_min_scale2=args.s21_min_scale2) if c.kind == "S21"]
    keys = first_layer_fields(s21)
    fields = [(config.sigma(j), ell, m) for j, ell, m in keys]
    entries = [(s21[i].j, s21[i].j2, s21[i].ell, m1, m2, keys.index((s21[i].j, s21[i].ell, m1)))
               for i, m1, m2 in s21m_entries(s21)]
    power = load_power_dataset(args.data_dir, "rsd", kmax=args.kmax, rebin=args.rebin, files=files, ells=(0, 2, 4))
    best = {}
    for path in args.fiducial:
        best.update(json.loads(path.read_text())["bestfit"])
    settings = json.loads(args.emulator.with_suffix(".json").read_text())

    t0 = time.time()
    pfield = model_field_power(args.data_dir, args.emulator, args.fiducial, config.cellsize)
    from cosmoprimo import Cosmology
    fo = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY).get_fourier()
    klin = np.geomspace(1e-4, 10.0, 1024)
    pklin = fo.pk_interpolator(of="delta_m")(klin, z=settings["z"])
    f = float(fo.sigma8_z(settings["z"], of="theta_cb") / fo.sigma8_z(settings["z"], of="delta_cb"))
    sigma_v = settings.get("s21_damping", 2.0) * float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
    plin = lambda k: np.where(np.asarray(k) > 0, np.exp(np.interp(np.log(np.maximum(k, 1e-30)), np.log(klin), np.log(pklin))), 0.0)  # noqa: E731
    beta = np.array([1.0] + [best[n] for n in ("b1", "b2", "bG2", "bGamma3")])
    meta = json.loads(str(np.load(files[0])["metadata"]))
    shot = float(np.mean([meta["boxsize"] ** 3 / json.loads(str(np.load(p)["metadata"]))["nparticles"] for p in files]))
    shot *= 1 + best.get("alpha0", 0.0)
    z1 = lambda k, mu: beta[1] + f * np.asarray(mu) ** 2  # noqa: E731
    print(f"model inputs ({time.time() - t0:.0f}s): b = {np.round(beta[1:], 3)}, N = {shot:.0f}, f = {f:.3f}, sigma_v = {sigma_v:.2f}",
          flush=True)

    t1 = time.time()
    x, _ = np.polynomial.legendre.leggauss(12)
    kgrid, mugrid = np.geomspace(4e-3, 0.12, 24), x[6:]
    r, g, _ = ModulusCrossRSD(fields, kgrid, mugrid).spectra(plin, f, sigma_v, beta, shot)
    amplitudes = [best.get(f"noise_j{j}_l{ell}_m{m}", 0.0) for j, ell, m in keys]
    white = [best.get(f"W_j{j}", 0.0) for j, _, _ in keys]
    moduli = ModulusTables(kgrid, mugrid, r, g, amplitudes, white)
    print(f"moduli spectra of {len(fields)} fields ({time.time() - t1:.0f}s)", flush=True)

    t2 = time.time()
    cov = RSDWSTCovariance(config, s1m, entries, fields, args.q, meta["boxsize"], pfield, plin, z1, moduli,
                           pk_edges=power.edges, nmesh=args.nmesh)
    print(f"2nd chaos of {len(cov.labels)} entries ({time.time() - t2:.0f}s)", flush=True)
    chaos4 = np.zeros_like(cov.lncov)
    if args.chaos_nmesh:
        t3 = time.time()
        n = cov.ns1m + cov.ns21m
        chaos4[:n, :n] = cov.chaos4(args.chaos_nmesh)
        print(f"4th chaos of {n} last layers ({time.time() - t3:.0f}s)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, labels=cov.labels, ns1m=cov.ns1m, ns21m=cov.ns21m, q=args.q, lncov=cov.lncov + chaos4,
             delta=cov.delta_part, u_noise=cov.u_part, chaos4=chaos4, pk_edges=power.edges,
             fiducial=[str(p) for p in args.fiducial])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
