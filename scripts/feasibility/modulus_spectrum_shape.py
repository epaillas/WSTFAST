#!/usr/bin/env python
"""K-shape of the first-layer modulus power spectra: measured P_UU(K) / <U>^2 (monopole) against the model.

The model is averaged over the lattice modes of each K bin, exactly as the measurement. Per field (j1, l, |m|),
over the lowest ``--nbins`` K bins (those the S21m second-layer filters weight), the data are
fitted as A_R x response(K) + A_N x noise(K), with the model response R^2 P_L and Gaussian noise g of
wstfast.theory.rsd_bias (monopoles over mu). A_R != 1 with the noise free, or K-trends in the residuals, mean that
the response / noise split of the model is wrong in shape, which a per-field noise amplitude cannot absorb.

    python scripts/feasibility/modulus_spectrum_shape.py --data-dir <..._los> --bias b1,b2,bG2,bGamma3 --sigma-v-factor 1
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

import wstfast.theory  # noqa: F401
from wstfast.config import QUIJOTE_COSMOLOGY, select_coefficients
from wstfast.data import load_measurement
from wstfast.theory.bias import monomials
from wstfast.theory.rsd_bias import BiasedModulusSpectraRSD, split
from wstfast.theory.rsd_moduli import first_layer_fields


def model(fields, edges, f, klin, pklin, beta, noise, sigma_v_factor, nmu=6, nk=28, **quadrature):
    """Monopoles of the response R^2 P_L and the noise g per field and K bin, (nfield, nbin) each, averaged over the
    lattice modes of each bin (as the measurement), from the model on a (K, mu) grid."""
    from types import SimpleNamespace

    from wstfast.theory.rsd import grid_interpolation, lattice_modes

    x, _ = np.polynomial.legendre.leggauss(2 * nmu)
    kgrid = np.geomspace(0.5 * edges[0] + 1e-3, edges[-1] * 1.05, nk)
    modes = lattice_modes(1000.0, 256, edges[-1], window=None)
    interp = grid_interpolation(SimpleNamespace(k=kgrid, mu=x[nmu:]), modes["k"], modes["mu"])
    index = np.digitize(modes["k"], edges) - 1
    nb = len(edges) - 1
    inside = (index >= 0) & (index < nb)
    binavg = np.zeros((nb, modes["k"].size))
    for b in range(nb):
        sel = inside & (index == b)
        binavg[b, sel] = modes["multiplicity"][sel] / modes["multiplicity"][sel].sum()
    project = lambda grid: binavg @ (interp @ grid.reshape(-1))  # noqa: E731
    logk, logp = np.log(klin), np.log(pklin)
    pk = lambda q: np.exp(np.interp(np.log(np.maximum(q, 1e-30)), logk, logp))  # noqa: E731
    sigma_v = sigma_v_factor * float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
    grids, norms = BiasedModulusSpectraRSD(fields, kgrid, x[nmu:], **quadrature)(pk, f, sigma_v)
    m2, m4 = np.asarray(monomials(beta, 2)), np.asarray(monomials(beta, 4))
    r0r0, r0r1, r1r1, g0, g1, g2 = split(grids, (70, 70, 70, 70, 15, 1))
    norm2 = (norms[:, :15] @ m2 + noise * norms[:, 15]) ** 2
    resp = (r0r0 @ m4 + noise * r0r1 @ m4 + noise**2 * r1r1 @ m4) / norm2[:, None, None]
    g = (g0 @ m4 + noise * g1 @ m2 + noise**2 * g2[..., 0]) / norm2[:, None, None]
    return np.array([project(r) for r in resp]), np.array([project(n) for n in g])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--bias", required=True, help="b1,b2,bG2,bGamma3")
    parser.add_argument("--alpha0", type=float, default=0.0)
    parser.add_argument("--sigma-v-factor", type=float, default=1.0)
    parser.add_argument("--nbins", type=int, default=10)
    parser.add_argument("--min-scale", type=float, default=25.0, help="smallest sigma_j1 of the S21m fields")
    parser.add_argument("--white", action="store_true", help="add a free K-independent term W per field")
    args = parser.parse_args()
    files = sorted(glob.glob(f"{args.data_dir}/rsd/wst_r*.npz"))
    config = load_measurement(files[0], q=0.8)["config"]
    s21 = [c for c in select_coefficients(config, s21_min_scale=args.min_scale, s21_min_ratio=2.8, s21_min_scale2=70.0)
           if c.kind == "S21"]
    keys = first_layer_fields(s21)
    puu, N = [], []
    for path in files:
        d = np.load(path)
        meta = json.loads(str(d["metadata"]))
        N.append(meta["boxsize"] ** 3 / meta["nparticles"])
        puu.append([d["PUU_m"][j, l, m, 0, :args.nbins] / d["Umean_m"][j, l, m] ** 2 for j, l, m in keys])
        k = d["k"][:args.nbins]
        edges = d["k_edges"][:args.nbins + 1]
    puu, nbox, sn = np.array(puu), len(files), float(np.mean(N))
    mean = puu.mean(0)
    from cosmoprimo import Cosmology
    fourier = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY).get_fourier()
    klin = np.geomspace(1e-4, 10.0, 1024)
    pklin = fourier.pk_interpolator(of="delta_m")(klin, z=0.5)
    f = float(fourier.sigma8_z(0.5, of="theta_cb") / fourier.sigma8_z(0.5, of="delta_cb"))
    beta = np.array([1.0, *(float(v) for v in args.bias.split(","))])
    resp, noise = model([(config.sigma(j), l, m) for j, l, m in keys], edges, f, klin, pklin, beta,
                        sn * (1 + args.alpha0), args.sigma_v_factor)
    print(f"{nbox} boxes, bias {args.bias}, alpha0 {args.alpha0}, damping {args.sigma_v_factor}; K {k[0]:.4f}-{k[-1]:.4f}")
    print(f"{'field':12s} | resp share | A_R (noise free)  A_N | chi2/dof  | data/model(A=1) at K bins")
    rng = np.random.default_rng(0)
    summary = []
    for i, (j, l, m) in enumerate(keys):
        cov = np.cov(puu[:, i], rowvar=False) / nbox  # covariance of the mean over boxes, across K bins
        prec = np.linalg.inv(cov)
        A = np.column_stack([resp[i], noise[i]] + ([np.ones_like(k)] if args.white else []))
        amp = np.linalg.solve(A.T @ prec @ A, A.T @ prec @ mean[i])
        err = np.sqrt(np.diag(np.linalg.inv(A.T @ prec @ A)))
        r = mean[i] - A @ amp
        chi2 = r @ prec @ r
        ratio = mean[i] / (resp[i] + noise[i])
        share = np.mean(resp[i] / (resp[i] + noise[i]))
        summary.append((j, l, m, amp[0], err[0], amp[1], chi2))
        white = f" W/noise {amp[2] / noise[i].mean():+.3f}" if args.white else ""
        print(f"j{j} l{l} m{m:<5d} | {share:9.2f} | {amp[0]:.3f} ± {err[0]:.3f}  {amp[1]:.3f} | {chi2:5.1f}/{len(k) - A.shape[1]:<3d}"
              + white + " | resid/err " + " ".join(f"{x:+.1f}" for x in r / np.sqrt(np.diag(cov))))
    s = np.array([x[3:] for x in summary])
    w = 1 / s[:, 1] ** 2
    print(f"weighted mean A_R {np.sum(s[:, 0] * w) / w.sum():.3f} ± {w.sum() ** -0.5:.3f}; mean A_N {s[:, 2].mean():.3f}; "
          f"total chi2 {s[:, 3].sum():.1f} / {len(keys) * (len(k) - 2 - args.white)}")


if __name__ == "__main__":
    main()
