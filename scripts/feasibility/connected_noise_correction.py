#!/usr/bin/env python
"""Per-entry correction of the S21m Gaussian noise for the tree-level connected noise of the moduli.

For every S21m entry of an emulated basis, c_e = sum M_e g rho / sum M_e g, with the second-layer lattice weights M_e
of the entry (second_layer.S21mProjection), the Gaussian noise g of its first-layer field on the basis (K, mu) grid
and rho = Delta N / g from wstfast.theory.connected_noise_rsd, all at one cosmology (Quijote) and one set of biases.
The fit then uses (1 + a_field) (1 + c_e) for the noise of entry e (fit_rsd.py --s21-connected).

    python scripts/feasibility/connected_noise_correction.py --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir_sv1.h5 \
        --data-dir <halo _los dir> --bias 1.96,-0.87,-0.66,1.49 --alpha0 -0.135 --cubic prior --output <npz>
"""

from __future__ import annotations

import argparse
import glob
import json
import time

import numpy as np

import wstfast.theory  # noqa: F401
from cosmoprimo import Cosmology
from wstfast.config import QUIJOTE_COSMOLOGY, select_coefficients
from wstfast.data import load_measurement
from wstfast.theory.bias import cubic_priors, monomials
from wstfast.theory.connected_noise_rsd import TERMS, interpolate_ratio, noise_ratio
from wstfast.theory.layout import split
from wstfast.theory.second_layer import ModulusSpectra
from wstfast.theory.second_layer import S21mProjection, first_layer_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emulator", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--bias", required=True, help="b1,b2,bG2,bGamma3")
    parser.add_argument("--alpha0", type=float, default=0.0)
    parser.add_argument("--cubic", default="prior", help="b3,bG3,bdG2 or 'prior'")
    parser.add_argument("--terms", nargs="+", default=list(TERMS))
    parser.add_argument("--npoints", type=int, default=2**14)
    parser.add_argument("--connected-f", type=float, default=0.0,
                        help="scale of f in the kernels of the connected terms: 0 (default) uses the real-space tracer "
                             "kernels (the tree-level redshift-space velocity terms over-predict the connected noise)")
    parser.add_argument("--fog", action="store_true",
                        help="also the per-entry FoG counterterm part correction_fog (noise: c_e - c_FoG c_e^fog)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    settings = json.loads(open(args.emulator.replace(".h5", ".json")).read())
    files = sorted(glob.glob(f"{args.data_dir}/rsd/wst_r*.npz"))
    config = load_measurement(files[0], q=0.8)["config"]
    shot = float(np.mean([1e9 / json.loads(str(np.load(p)["metadata"]))["nparticles"] for p in files])) * (1 + args.alpha0)
    e21 = [c for c in select_coefficients(config, s21_min_scale=0.0, s21_min_ratio=0.0) if c.kind == "S21"
           and c.label in settings["s21_coefficients"]]
    e21.sort(key=lambda c: settings["s21_coefficients"].index(c.label))
    keys = first_layer_fields(e21)
    fields = [(config.sigma(j), l, m) for j, l, m in keys]
    beta = np.array([1.0, *(float(v) for v in args.bias.split(","))])
    cubic = cubic_priors(beta[1], beta[2]) if args.cubic == "prior" else np.array([float(v) for v in args.cubic.split(",")])
    fo = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY).get_fourier()
    klin = np.geomspace(1e-4, 10.0, 1024)
    pklin = fo.pk_interpolator(of="delta_m")(klin, z=settings["z"])
    f = float(fo.sigma8_z(settings["z"], of="theta_cb") / fo.sigma8_z(settings["z"], of="delta_cb"))
    sigma_v = settings.get("s21_damping", 2.0) * float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
    pk = lambda q: np.where(np.asarray(q) > 0, np.exp(np.interp(np.log(np.maximum(q, 1e-30)), np.log(klin), np.log(pklin))), 0.0)  # noqa
    # Gaussian noise of every field on the basis grid (as second_layer.S21mBasis)
    x, _ = np.polynomial.legendre.leggauss(12)
    kgrid, mugrid = np.geomspace(4e-3, 0.12, 24), x[6:]
    grids, norms = ModulusSpectra(fields, kgrid, mugrid)(pk, f, sigma_v)
    m2, m4 = np.asarray(monomials(beta, 2)), np.asarray(monomials(beta, 4))
    _, _, _, g0, g1, g2 = split(grids, (70, 70, 70, 70, 15, 1))
    norm2 = (norms[:, :15] @ m2 + shot * norms[:, 15]) ** 2
    g = (g0 @ m4 + shot * g1 @ m2 + shot**2 * g2[..., 0]) / norm2[:, None, None]  # (nfield, nk, nmu)
    t = time.time()
    rn = noise_ratio(fields, pk, args.connected_f * f, sigma_v, beta, cubic, shot, npoints=args.npoints,
                     terms=tuple(args.terms), fog=args.fog)
    print(f"connected noise of {len(fields)} fields: {time.time() - t:.0f} s")
    rho = interpolate_ratio(rn["rho"], rn["unodes"], rn["munodes"], [s for s, _, _ in fields], kgrid, mugrid)
    proj = S21mProjection(config, e21, keys, kgrid, mugrid)
    flat_g = g.reshape(len(fields), -1)
    flat_rho = rho.reshape(len(fields), -1)
    num = np.einsum("eg,eg->e", proj.matrix, (flat_g * flat_rho)[proj.field_index])
    den = np.einsum("eg,eg->e", proj.matrix, flat_g[proj.field_index])
    correction = num / den
    extra = {}
    if args.fog:
        rho_fog = interpolate_ratio(rn["rho_fog"], rn["unodes"], rn["munodes"], [s for s, _, _ in fields], kgrid, mugrid)
        extra["correction_fog"] = np.einsum("eg,eg->e", proj.matrix,
                                            (flat_g * rho_fog.reshape(len(fields), -1))[proj.field_index]) / den
    for j in sorted({c.j for c in e21}):
        sel = [n for n, (i, _, _) in enumerate(proj.entries) if e21[i].j == j]
        print(f"j1 = {j} (sigma1 {config.sigma(j):.1f}): c_e median {np.median(correction[sel]):+.3f}, "
              f"range {correction[sel].min():+.3f} .. {correction[sel].max():+.3f}")
    np.savez(args.output, correction=correction, labels=np.array([f"{e21[i].label}_m{m1}_m{m2}" for i, m1, m2 in proj.entries]),
             beta=beta, cubic=cubic, shotnoise=shot, terms=np.array(args.terms), fields=np.array(keys),
             connected_f=args.connected_f, **extra,
             **{f"rho_{k}": v for k, v in rn.items()})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
