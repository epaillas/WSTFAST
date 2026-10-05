#!/usr/bin/env python
"""Large-scale clustering of first-layer WST modulus fields U_{j,l} = |delta * psi_{j,l}|.

Measures, for one Quijote snapshot and its Gaussian twin, the cross-power with
delta and the auto-power of each modulus field, b(k) = P_Ud / P_dd and the
stochastic power N(k) = P_UU - P_Ud^2 / P_dd. This is the density-split-like
observable behind the second WST layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wst_feasibility as W


def binned(lat, a, b, boxsize, edges):
    kphys = lat.kmag * (lat.n / boxsize)
    idx = np.digitize(kphys.reshape(-1), edges) - 1
    w = lat.mult.reshape(-1)
    val = (a * np.conj(b)).real.reshape(-1) * w
    nb = len(edges) - 1
    ok = (idx >= 0) & (idx < nb)
    num = np.bincount(idx[ok], weights=val[ok], minlength=nb)
    cnt = np.bincount(idx[ok], weights=w[ok], minlength=nb)
    kmean = np.bincount(idx[ok], weights=(kphys.reshape(-1) * w)[ok], minlength=nb)
    norm = boxsize**3 / lat.n**6
    return kmean / cnt, num / cnt * norm


def plot(fn: Path, sigma0_mpc: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(fn.read_text())
    k = np.array(d["k"])
    nb, ga = d["fields"]["nbody"], d["fields"]["gauss"]
    pdd, pdd_g = np.array(nb["Pdd"]), np.array(ga["Pdd"])
    colors = dict(zip((0, 1, 2, 4), plt.cm.viridis(np.linspace(0, 0.9, 4))))
    styles = {0: "-", 1: "--", 2: ":"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ell in (0, 1, 2, 4):
        for j in (0, 1, 2):
            e, g = nb[f"{j},{ell}"], ga[f"{j},{ell}"]
            pud, puu = np.array(e["PUd"]), np.array(e["PUU"])
            lab = f"l={ell}, σ={sigma0_mpc * 2**j:.1f}" if ell in (1, 4) or j == 0 else None
            axes[0].plot(k, pud / pdd / e["mean"], styles[j], color=colors[ell], label=lab)
            axes[0].plot(k, np.array(g["PUd"]) / pdd_g / g["mean"], styles[j], color="0.6", lw=0.6)
            axes[1].plot(k, pud / np.sqrt(puu * pdd), styles[j], color=colors[ell])
            noise_nb = (puu - pud**2 / pdd) / e["mean"] ** 2
            noise_g = np.array(g["PUU"]) / g["mean"] ** 2
            axes[2].plot(k, noise_nb / noise_g, styles[j], color=colors[ell])
    axes[0].set_ylabel(r"$b_U(k)/\langle U\rangle=P_{U\delta}/(P_{\delta\delta}\langle U\rangle)$")
    axes[0].set_title("grey: Gaussian twin (same |δ(k)|)", fontsize=9)
    axes[1].set_ylabel(r"$r(k)=P_{U\delta}/\sqrt{P_{UU}P_{\delta\delta}}$")
    axes[2].set_ylabel(r"stochastic power: N-body / Gaussian-chaos")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ [h/Mpc]")
    axes[0].legend(fontsize=7, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.1), framealpha=0.9)
    fig.tight_layout()
    out = fn.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, help="only plot an existing measurement JSON")
    parser.add_argument("--realization", default="0")
    parser.add_argument("--nmesh", type=int, default=256)
    parser.add_argument("--sigma0", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("outputs/feasibility"))
    args = parser.parse_args()
    if args.plot:
        plot(args.plot, args.sigma0 * 1000.0 / args.nmesh)
        return
    lat = W.Lattice(args.nmesh)
    delta, header = W.paint_snapshot(W.SNAPSHOT_ROOT / args.realization / "snapdir_003", args.nmesh, rsd=False)
    box = header["boxsize"]
    edges = np.arange(0.5, 40.5, 1.0) * 2 * np.pi / box  # fundamental-mode bins up to k ~ 0.25
    out = {"k_edges": edges.tolist(), "fields": {}}
    for label, field in (("nbody", delta), ("gauss", W.phase_randomize(delta, seed=1000 + int(args.realization)))):
        fk = W.rfft(field)
        k, pdd = binned(lat, fk, fk, box, edges)
        out["k"] = k.tolist()
        out["fields"][label] = {"Pdd": pdd.tolist()}
        for ell in (0, 1, 2, 4):
            harm = lat.real_harmonics(ell)
            for j in (0, 1, 2):
                rad = lat.radial(args.sigma0 * 2**j, ell)
                z = np.zeros((lat.n,) * 3, dtype=np.float32)
                for y in harm:
                    x = W.irfft(fk * ((-1j) ** ell * rad * y), lat.n)
                    z += x * x
                u = np.sqrt(z)
                mean = float(u.mean(dtype=np.float64))
                fu = W.rfft(u - mean)
                _, pud = binned(lat, fu, fk, box, edges)
                _, puu = binned(lat, fu, fu, box, edges)
                out["fields"][label][f"{j},{ell}"] = {"mean": mean, "PUd": pud.tolist(), "PUU": puu.tolist()}
            print(f"{label} l={ell} done", flush=True)
    fn = args.output / f"modulus_spectra_r{args.realization}_real_n{args.nmesh}.json"
    fn.write_text(json.dumps(out))
    print(f"wrote {fn}")
    plot(fn, args.sigma0 * box / args.nmesh)


if __name__ == "__main__":
    main()
