#!/usr/bin/env python
"""Tree-level PT prediction of the large-scale bias of the WST modulus fields (real-space matter).

For l >= 1 the second-chaos projection U ~ <U> + <U>/(2 n s^2) (|X|^2 - n s^2) gives, at
leading non-Gaussian order,

    P_{U delta}(k) = <U>/(2 n s^2) * int_p B(p, k-p, -k) F(p, k-p),
    F(p, q) = sum_m psi_m(p) psi_m(q) = (-1)^l (s_j^2 p q)^l P_l(p.q) exp[-s_j^2 (p^2+q^2)/2],
    n s^2 = int_p P(p) (s_j p)^{2l} exp(-s_j^2 p^2),

with the tree-level matter bispectrum B. The script compares b(k)/<U> = P_{U delta}/(P <U>) with
the N-body measurement written by modulus_field_spectra.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre


def f2(k1, k2, mu):
    return 5 / 7 + 0.5 * mu * (k1 / k2 + k2 / k1) + 2 / 7 * mu**2


def linear_power(z=0.5):
    from cosmoprimo import Cosmology

    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175 - 0.049, n_s=0.9624, sigma8=0.834,
                      m_ncdm=[], N_eff=3.046, engine="class")
    return cosmo.get_fourier().pk_interpolator(of="delta_m").to_1d(z=z)


def tree_bias(kvals, sigma, ell, pk, npts=800, nmu=128):
    p = np.logspace(-4, np.log10(8.0 / sigma), npts)
    mu, wmu = leggauss(nmu)
    pp, mm = np.meshgrid(p, mu, indexing="ij")
    dlogp = np.gradient(np.log(p))
    wp = (p**3 * dlogp)[:, None] * wmu[None, :] / (4 * np.pi**2)  # d^3p/(2pi)^3 = p^2 dp dmu / (4 pi^2)
    norm = np.sum(wp * (pk(pp) * (sigma * pp) ** (2 * ell) * np.exp(-(sigma * pp) ** 2)))
    out = []
    for k in kvals:
        q = np.sqrt(np.maximum(k**2 + pp**2 - 2 * k * pp * mm, 1e-12))
        cos_pq = (k * pp * mm - pp**2) / (pp * q)
        filt = (-1) ** ell * (sigma**2 * pp * q) ** ell * eval_legendre(ell, np.clip(cos_pq, -1, 1)) \
            * np.exp(-0.5 * sigma**2 * (pp**2 + q**2))
        mu_qk = -(k**2 - k * pp * mm) / (k * q)
        bis = 2 * (f2(pp, q, cos_pq) * pk(pp) * pk(q)
                   + f2(q, k, mu_qk) * pk(q) * pk(k)
                   + f2(k, pp, -mm) * pk(k) * pk(pp))
        out.append(np.sum(wp * bis * filt) / (2 * norm * pk(k)))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, nargs="+",
                        default=sorted(Path("outputs/feasibility").glob("modulus_spectra_r*_real_n256.json")))
    parser.add_argument("--cellsize", type=float, default=1000.0 / 256)
    parser.add_argument("--sigma0", type=float, default=0.8)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [json.loads(fn.read_text()) for fn in args.measurements]
    k = np.array(data[0]["k"])
    pk = linear_power()
    trees = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    lines = [f"Realizations: {len(data)}", "",
             "| (j,l) | σ_j [Mpc/h] | b_U/⟨U⟩ N-body (k < 0.3/σ_j) | tree PT | ratio (mean ± scatter) |",
             "|---|---|---|---|---|"]
    for ax, ell in zip(axes, (1, 2, 4)):
        for j, ls in zip((0, 1, 2), ("-", "--", ":")):
            sigma = args.sigma0 * 2**j * args.cellsize
            trees[(j, ell)] = pred = tree_bias(k, sigma, ell, pk)
            meas = np.array([np.array(d["fields"]["nbody"][f"{j},{ell}"]["PUd"]) / np.array(d["fields"]["nbody"]["Pdd"])
                             / d["fields"]["nbody"][f"{j},{ell}"]["mean"] for d in data])
            ax.plot(k, meas.mean(0), ls, color="C0", label=f"N-body, σ={sigma:.1f}")
            ax.plot(k, pred, ls, color="C3", label=f"tree PT, σ={sigma:.1f}")
            sel = k < 0.3 / sigma
            ratios = (meas[:, sel] / pred[sel]).mean(1)
            lines.append(f"| {j},{ell} | {sigma:.1f} | {meas[:, sel].mean():.3f} | {pred[sel].mean():.3f} | "
                         f"{ratios.mean():.3f} ± {ratios.std(ddof=1) if len(ratios) > 1 else np.nan:.3f} |")
        ax.set_xscale("log")
        ax.set_title(f"l = {ell}", fontsize=10)
        ax.set_xlabel(r"$k$ [h/Mpc]")
    axes[0].set_ylabel(r"$b_U(k)/\langle U\rangle$")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out = args.measurements[0].parent / "tree_response.png"
    fig.savefig(out, dpi=130)
    lines += ["", variance_test(data, trees, args.cellsize, args.sigma0)]
    text = "\n".join(lines)
    (args.measurements[0].parent / "tree_response.md").write_text(text)
    print(text)
    print(f"wrote {out}")


def variance_test(data, trees, cellsize, sigma0):
    """Second-layer variance from tree-PT b_U(k), measured P_dd and a noise term, vs the measured P_UU band power."""
    k = np.array(data[0]["k"])
    nmodes = 4 * np.pi * k**2 * np.diff(np.array(data[0]["k_edges"]))
    lines = ["| (j1,j2,l) | σ_j1→σ_j2 | (tree b² P + N_N-body)/V | (tree b² P + N_chaos)/V | response share of V |",
             "|---|---|---|---|---|"]
    for ell in (1, 2, 4):
        for j1 in (0, 1, 2):
            s1 = sigma0 * 2**j1 * cellsize
            for j2 in range(j1 + 1, 5):
                s2 = sigma0 * 2**j2 * cellsize
                w = (s2 * k) ** (2 * ell) * np.exp(-((s2 * k) ** 2)) * nmodes
                if w[-3:].sum() / w.sum() > 0.01:
                    continue  # second-layer band not contained in the measured k range
                a, b, share = [], [], []
                for d in data:
                    nb, ga = d["fields"]["nbody"], d["fields"]["gauss"]
                    e, g = nb[f"{j1},{ell}"], ga[f"{j1},{ell}"]
                    pdd = np.array(nb["Pdd"])
                    puu, pud = np.array(e["PUU"]), np.array(e["PUd"])
                    resp = np.sum((trees[(j1, ell)] * e["mean"]) ** 2 * pdd * w)
                    meas = np.sum(puu * w)
                    a.append((resp + np.sum((puu - pud**2 / pdd) * w)) / meas)
                    b.append((resp + np.sum(np.array(g["PUU"]) * w)) / meas)
                    share.append(resp / meas)
                lines.append(f"| {j1},{j2},{ell} | {s1:.1f}→{s2:.1f} | {np.mean(a):.3f} | {np.mean(b):.3f} | "
                             f"{np.mean(share):.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
