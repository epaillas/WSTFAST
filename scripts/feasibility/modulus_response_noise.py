#!/usr/bin/env python
"""Model-free check of the S21m ingredients: response and noise of the first-layer moduli U_|m| at low K.

From the stored multipoles (monopoles) of each field (j1, l, |m|):
  response   P_Ud / <U> = s b_U^model(K) (P_dd - V/N) + C: s is the data / model ratio of the response,
  noise      N_U(K) = [P_UU - P_Ud^2 / P_dd] / <U>^2   (the stochasticity of U with respect to delta).
Model: wstfast.theory.rsd_bias (ModulusSpectraRSD with tracer kernels) at the Quijote cosmology and given biases:
  b_U = int R Z1 P_L / int Z1^2 P_L,   R = (r0 + N r1) / (n0 + N n1)   (response to delta_L, per field),
  N_U ~ g = (G0 + N G1 + N^2 G2) / (n0 + N n1)^2   (Gaussian noise of the model, before the free amplitude a).

    python scripts/feasibility/modulus_response_noise.py --data-dir <..._los> --bias 1.955,-0.99,-0.77,1.55 --alpha0 -0.14
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

import wstfast.theory  # noqa: F401
from wstfast.config import QUIJOTE_COSMOLOGY, select_coefficients
from wstfast.data import load_measurement
from wstfast.theory.bias import galaxy_kernel, monomial_product, monomials, product_coefficients
from wstfast.theory.rsd_bias import BiasedModulusSpectraRSD, linear_components, split
from wstfast.theory.rsd_moduli import SIGMA_V_FACTOR, _harmonic, first_layer_fields


def model(fields, kvals, f, klin, pklin, beta, noise, sigma_v_factor=SIGMA_V_FACTOR):
    """b_U(K) and g(K) (monopoles over mu) per field."""
    nmu = 6
    x, wx = np.polynomial.legendre.leggauss(2 * nmu)
    mu, wmu = x[nmu:], wx[nmu:]
    logk, logp = np.log(klin), np.log(pklin)
    pk = lambda q: np.exp(np.interp(np.log(np.maximum(q, 1e-30)), logk, logp))  # noqa: E731
    sigma_v = sigma_v_factor * float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
    spectra = BiasedModulusSpectraRSD(fields, kvals, mu)
    grids, norms = spectra(pk, f, sigma_v)  # grids (nfield, nk, nmu, 296), norms (nfield, 16)
    m2, m4 = np.asarray(monomials(beta, 2)), np.asarray(monomials(beta, 4))
    r0r0, r0r1, r1r1, g0, g1, g2 = split(grids, (70, 70, 70, 70, 15, 1))
    n0, n1 = norms[:, :15], norms[:, 15]
    norm = n0 @ m2 + noise * n1  # (nfield,)
    plin = pk(kvals)[None, :, None]
    # R^2 P_L = (r0 + N r1)^2 P_L / norm^2 from the stored squares: |R| = sqrt(R^2 P_L / P_L)
    r2pl = (r0r0 @ m4 + noise * r0r1 @ m4 + noise**2 * r1r1 @ m4) / norm[:, None, None] ** 2
    resp = np.sqrt(np.maximum(r2pl / plin, 0.0))  # sign: the response is positive for these fields
    z1 = (np.array([[b for b in (beta[1] + f * m**2 for m in mu)]]))  # (1, nmu)
    bU = np.sum(wmu * resp * z1[None] * plin, axis=2) / np.sum(wmu * z1[None] ** 2 * plin, axis=2)
    g = (g0 @ m4 + noise * g1 @ m2 + noise**2 * g2[..., 0]) / norm[:, None, None] ** 2
    return bU, np.sum(wmu * g, axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--bias", required=True, help="b1,b2,bG2,bGamma3")
    parser.add_argument("--alpha0", type=float, default=0.0)
    parser.add_argument("--nbins", type=int, default=10, help="lowest K bins used")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sigma-v-factor", type=float, default=SIGMA_V_FACTOR,
                        help="velocity damping of the modulus spectra, in units of the linear sigma_v")
    args = parser.parse_args()
    files = sorted(glob.glob(f"{args.data_dir}/rsd/wst_r*.npz"))[:args.max_files]
    config = load_measurement(files[0], q=0.8)["config"]
    s21 = [c for c in select_coefficients(config, s21_min_scale=17.6, s21_min_ratio=2.8, s21_min_scale2=70.0)
           if c.kind == "S21"]
    keys = first_layer_fields(s21)
    # Data: per box, per field, monopoles at the lowest bins
    PUd, PUU, U, Pdd, N = [], [], [], [], []
    for path in files:
        d = np.load(path)
        meta = json.loads(str(d["metadata"]))
        N.append(meta["boxsize"] ** 3 / meta["nparticles"])
        Pdd.append(d["Pdd"][0, :args.nbins])
        U.append([d["Umean_m"][j, l, m] for j, l, m in keys])
        PUd.append([d["PUd_m"][j, l, m, 0, :args.nbins] for j, l, m in keys])
        PUU.append([d["PUU_m"][j, l, m, 0, :args.nbins] for j, l, m in keys])
        k = d["k"][:args.nbins]
    PUd, PUU, U, Pdd, N = (np.array(a) for a in (PUd, PUU, U, Pdd, N))
    nbox = len(files)
    sn = N.mean()
    noise_data = ((PUU - PUd**2 / Pdd[:, None, :]) / U[:, :, None] ** 2).mean(0)  # (nfield, nbins)
    noise_err = ((PUU - PUd**2 / Pdd[:, None, :]) / U[:, :, None] ** 2).std(0) / np.sqrt(nbox)

    from cosmoprimo import Cosmology
    fourier = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY).get_fourier()
    klin = np.geomspace(1e-4, 10.0, 1024)
    pklin = fourier.pk_interpolator(of="delta_m")(klin, z=0.5)
    f = float(fourier.sigma8_z(0.5, of="theta_cb") / fourier.sigma8_z(0.5, of="delta_cb"))
    b = [float(v) for v in args.bias.split(",")]
    beta = np.array([1.0, *b])
    fields = [(config.sigma(j), l, m) for j, l, m in keys]
    bU_model, g_model = model(fields, k, f, klin, pklin, beta, sn * (1 + args.alpha0), args.sigma_v_factor)
    # response ratio s per field: P_Ud/<U> = s bU_model(K) (P_dd - N) + C, on the box mean; error by bootstrap
    x = (Pdd - N[:, None]).mean(0)
    yb = PUd / U[:, :, None]
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, nbox, nbox) for _ in range(200)]
    ratio, ratio_err = [], []
    for i in range(len(keys)):
        A = np.column_stack([bU_model[i] * x, np.ones_like(x)])
        ratio.append(np.linalg.lstsq(A, yb[:, i].mean(0), rcond=None)[0][0])
        ratio_err.append(np.std([np.linalg.lstsq(A, yb[d, i].mean(0), rcond=None)[0][0] for d in draws]))
    print(f"{nbox} boxes, V/N = {sn:.0f}, bias {b}, alpha0 {args.alpha0}; K in [{k[0]:.4f}, {k[-1]:.4f}] ({len(k)} bins)")
    print(f"{'field':14s} | response data/model | noise data/model, mean over the K bins")
    for i, (j, l, m) in enumerate(keys):
        nr = noise_data[i] / g_model[i]
        print(f"j{j} l{l} m{m:<8d} | {ratio[i]:6.3f} ± {ratio_err[i]:.3f}     | {nr.mean():.3f} (K-trend {nr[-1] - nr[0]:+.3f})")
    for j in sorted({j for j, _, _ in keys}):
        sel = [i for i, key in enumerate(keys) if key[0] == j]
        w = 1 / np.array(ratio_err)[sel] ** 2
        print(f"j{j}: weighted mean response ratio {np.sum(np.array(ratio)[sel] * w) / w.sum():.3f} ± {w.sum() ** -0.5:.3f};"
              f" mean noise ratio {np.mean([noise_data[i] / g_model[i] for i in sel]):.3f}")

if __name__ == "__main__":
    main()
