#!/usr/bin/env python
"""Aggregate the WST feasibility diagnostics over Quijote realizations.

Writes markdown tables and a figure to outputs/feasibility/. Quantities:

* stat: single-box (1 Gpc/h)^3 fractional standard deviation of each coefficient;
* NG: S[N-body] / S[same |delta(k)|, random phases] - 1, i.e. everything that is
  not fixed by the measured power spectrum;
* NLO residual (S1): NG of the one-point distribution minus the Edgeworth
  next-to-leading prediction built from the measured cumulants K4, K33;
* S2 split: the part explained by the second-layer variance alone, the residual
  shape term, and the linear-bias + Gaussian-noise model of that variance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

Q = 0.8


def load(root: Path, space: str, tag: str):
    files = sorted(root.glob(f"wst_diag_r*_{space}_n256{tag}.json"))
    files = [f for f in files if (tag or "_boss" not in f.name)]
    return [json.loads(f.read_text()) for f in files]


def s1_table(runs, cell, sigma0):
    keys = list(runs[0]["nbody"]["S1"])
    rows = []
    for key in keys:
        j, ell = map(int, key.split(","))
        vals = np.array([r["nbody"]["S1"][key]["S1"] for r in runs])
        ng = np.array([r["nbody"]["S1"][key]["S1"] / r["gauss"]["S1"][key]["S1"] - 1 for r in runs])
        nlo = np.array([
            r["nbody"]["S1"][key]["S1"] / r["nbody"]["S1"][key]["S1_gauss"] - 1
            - r["nbody"]["S1"][key]["edgeworth_c4"] - r["nbody"]["S1"][key]["edgeworth_c33"]
            for r in runs
        ])
        w = runs[0]["nbody"]["S1"][key]["weights"]
        sd = np.sqrt(np.mean([r["nbody"]["S1"][f"{j},0"]["stats"]["s2"] for r in runs]))
        bias = [r["nbody"]["S1"][key].get("bias_U1_delta") for r in runs]
        bias = [b / r["nbody"]["S1"][key]["U1_mean"] for b, r in zip(bias, runs) if b is not None]
        rows.append(dict(
            key=key, j=j, l=ell, sigma=sigma0 * 2**j * cell, k50=w["k50"], k90=w["k90"], sig_delta=sd,
            stat=vals.std(ddof=1) / vals.mean(), ng=ng.mean(), ng_err=ng.std(ddof=1) / np.sqrt(len(ng)),
            nlo=nlo.mean(), nlo_err=nlo.std(ddof=1) / np.sqrt(len(nlo)),
            bias=np.mean(bias) if bias else np.nan, bias_err=np.std(bias, ddof=1) / np.sqrt(len(bias)) if len(bias) > 1 else np.nan,
        ))
    return rows


def s2_table(runs, cell, sigma0):
    keys = list(runs[0]["nbody"]["S2"])
    rows = []
    for key in keys:
        j1, j2, ell = map(int, key.split(","))
        vals = np.array([r["nbody"]["S2"][key]["S2"] for r in runs])
        ng = np.array([r["nbody"]["S2"][key]["S2"] / r["gauss"]["S2"][key]["S2"] - 1 for r in runs])
        if ell == 0:
            vpart = np.array([(r["nbody"]["S2"][key]["Y_mean"] / r["gauss"]["S2"][key]["Y_mean"]) ** Q - 1 for r in runs])
        else:
            vpart = np.array([(r["nbody"]["S2"][key]["Y_var"] / r["gauss"]["S2"][key]["Y_var"]) ** (Q / 2) - 1 for r in runs])
        shape = np.array([r["nbody"]["S2"][key]["S2"] / r["nbody"]["S2"][key]["S2_gauss_Y"] - 1 for r in runs])
        shape_nlo = np.array([
            r["nbody"]["S2"][key]["S2"] / r["nbody"]["S2"][key]["S2_gauss_Y"] - 1
            - r["nbody"]["S2"][key].get("edgeworth_c4", np.nan) - r["nbody"]["S2"][key].get("edgeworth_c33", np.nan)
            for r in runs
        ])
        model = []
        for r in runs:
            b = r["nbody"]["S1"][f"{j1},{ell}"].get("bias_U1_delta")
            if b is None or ell == 0:
                continue
            vlin = sum(r["nbody"]["S1"][f"{j2},{ell}"]["stats"]["eig"])
            model.append((b**2 * vlin + r["gauss"]["S2"][key]["Y_var"]) / r["nbody"]["S2"][key]["Y_var"])
        rows.append(dict(
            key=key, j1=j1, j2=j2, l=ell, s1=sigma0 * 2**j1 * cell, s2=sigma0 * 2**j2 * cell,
            stat=vals.std(ddof=1) / vals.mean(), ng=ng.mean(), vpart=vpart.mean(), shape=shape.mean(),
            shape_nlo=shape_nlo.mean(),
            linmodel=np.mean(model) if model else np.nan,
        ))
    return rows


def fmt_s1(rows, real_space):
    head = "| (j,l) | σ_j [Mpc/h] | k50–k90 [h/Mpc] | σ_δ(σ_j) | 1-box stat | NG | NLO residual |"
    if real_space:
        head += " b_U/⟨U⟩ |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        line = (f"| {r['j']},{r['l']} | {r['sigma']:.1f} | {r['k50']:.3f}–{r['k90']:.3f} | {r['sig_delta']:.3f} | "
                f"{100 * r['stat']:.2f}% | {100 * r['ng']:+.2f}% | {100 * r['nlo']:+.2f}% |")
        if real_space:
            line += f" {r['bias']:+.2f} |"
        lines.append(line)
    return "\n".join(lines)


def fmt_s2(rows, real_space):
    head = "| (j1,j2,l) | σ_j1→σ_j2 [Mpc/h] | 1-box stat | NG | variance part | shape part |"
    if real_space:
        head += " (b²V+N)/V |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        line = (f"| {r['j1']},{r['j2']},{r['l']} | {r['s1']:.0f}→{r['s2']:.0f} | {100 * r['stat']:.2f}% | "
                f"{100 * r['ng']:+.1f}% | {100 * r['vpart']:+.1f}% | {100 * r['shape']:+.2f}% |")
        if real_space:
            line += f" {r['linmodel']:.3f} |" if np.isfinite(r["linmodel"]) else " – |"
        lines.append(line)
    return "\n".join(lines)


def _rng(vals, scale=100, fmt="{:+.1f}"):
    vals = [scale * v for v in vals if np.isfinite(v)]
    if not vals:
        return "–"
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if abs(hi - lo) < 1e-12 else f"{fmt.format(lo)} … {fmt.format(hi)}"


def condensed(real_rows, rsd_rows):
    """Ranges over l=0..4 (S1) and l=1..4 (S2) per scale, real and redshift space."""
    rows1r, rows2r = real_rows
    rows1s, rows2s = rsd_rows
    lines = ["| j | σ_j [Mpc/h] | k90 [h/Mpc] | 1-box stat | NG real | NG RSD | NLO resid. real | NLO resid. RSD |",
             "|---|---|---|---|---|---|---|---|"]
    for j in sorted({r["j"] for r in rows1r}):
        a = [r for r in rows1r if r["j"] == j]
        b = [r for r in rows1s if r["j"] == j]
        lines.append(
            f"| {j} | {a[0]['sigma']:.1f} | {min(r['k90'] for r in a):.2f}–{max(r['k90'] for r in a):.2f} | "
            f"{_rng([r['stat'] for r in a], fmt='{:.2f}')}% | {_rng([r['ng'] for r in a])}% | {_rng([r['ng'] for r in b])}% | "
            f"{_rng([r['nlo'] for r in a], fmt='{:+.2f}')}% | {_rng([r['nlo'] for r in b], fmt='{:+.2f}')}% |")
    out = ["### S1, ranges over l = 0…4", "", *lines, ""]
    lines = ["| j1→j2 | σ [Mpc/h] | 1-box stat | NG real | NG RSD | shape real | shape RSD | (b²V+N)/V real |",
             "|---|---|---|---|---|---|---|---|"]
    pairs = sorted({(r["j1"], r["j2"]) for r in rows2r})
    for j1, j2 in pairs:
        a = [r for r in rows2r if (r["j1"], r["j2"]) == (j1, j2) and r["l"] > 0]
        b = [r for r in rows2s if (r["j1"], r["j2"]) == (j1, j2) and r["l"] > 0]
        lines.append(
            f"| {j1}→{j2} | {a[0]['s1']:.0f}→{a[0]['s2']:.0f} | {_rng([r['stat'] for r in a], fmt='{:.1f}')}% | "
            f"{_rng([r['ng'] for r in a], fmt='{:+.0f}')}% | {_rng([r['ng'] for r in b], fmt='{:+.0f}')}% | "
            f"{_rng([r['shape'] for r in a])}% | {_rng([r['shape'] for r in b])}% | "
            f"{_rng([r['linmodel'] for r in a], scale=1, fmt='{:.2f}')} |")
    out += ["### S2, ranges over l = 1…4", "", *lines, ""]
    return "\n".join(out)


def ladder(root: Path):
    """Merge the 3.9 Mpc/h (sigma0=0.8) and BOSS-scale (sigma0=2.048) runs into one scale ladder."""
    configs = []
    for tag, max_sigma in (("", 60.0), ("_boss_scales", 70.0)):
        spaces = {}
        for space in ("real", "rsd"):
            runs = load(root, space, tag)
            if runs:
                cell, sigma0 = runs[0]["config"]["cellsize"], runs[0]["config"]["sigma0"]
                spaces[space] = (s1_table(runs, cell, sigma0), s2_table(runs, cell, sigma0), len(runs))
        if len(spaces) == 2:
            configs.append((tag or "3.9", max_sigma, spaces))
    lines = ["| σ_j [Mpc/h] | k90 [h/Mpc] | 1-box stat | NG real | NG RSD | NLO resid. real | NLO resid. RSD | max abs(resid.)/stat |",
             "|---|---|---|---|---|---|---|---|"]
    rows = []
    for name, max_sigma, spaces in configs:
        r1, s1 = spaces["real"][0], spaces["rsd"][0]
        for j in sorted({r["j"] for r in r1}):
            a = [r for r in r1 if r["j"] == j]
            b = [r for r in s1 if r["j"] == j]
            if a[0]["sigma"] > max_sigma:
                continue
            worst = max(max(abs(r["nlo"]) / r["stat"] for r in a), max(abs(r["nlo"]) / r["stat"] for r in b))
            rows.append((a[0]["sigma"],
                         f"| {a[0]['sigma']:.1f} | {min(r['k90'] for r in a):.2f}–{max(r['k90'] for r in a):.2f} | "
                         f"{_rng([r['stat'] for r in a + b], fmt='{:.2f}')}% | {_rng([r['ng'] for r in a])}% | "
                         f"{_rng([r['ng'] for r in b])}% | {_rng([r['nlo'] for r in a], fmt='{:+.2f}')}% | "
                         f"{_rng([r['nlo'] for r in b], fmt='{:+.2f}')}% | {worst:.1f} |"))
    lines += [row for _, row in sorted(rows)]
    out = ["### S1 scale ladder (ranges over l = 0…4)", "", *lines, ""]
    lines = ["| σ_j1→σ_j2 [Mpc/h] | 1-box stat | NG real | NG RSD | variance share of NG | shape real | shape after Y-NLO |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    for name, max_sigma, spaces in configs:
        r2, s2 = spaces["real"][1], spaces["rsd"][1]
        for j1, j2 in sorted({(r["j1"], r["j2"]) for r in r2}):
            a = [r for r in r2 if (r["j1"], r["j2"]) == (j1, j2) and r["l"] > 0]
            b = [r for r in s2 if (r["j1"], r["j2"]) == (j1, j2) and r["l"] > 0]
            if a[0]["s2"] > max_sigma:
                continue
            share = [np.log1p(r["vpart"]) / np.log1p(r["ng"]) for r in a + b]
            rows.append(((a[0]["s1"], a[0]["s2"]),
                         f"| {a[0]['s1']:.1f}→{a[0]['s2']:.1f} | {_rng([r['stat'] for r in a + b], fmt='{:.1f}')}% | "
                         f"{_rng([r['ng'] for r in a], fmt='{:+.0f}')}% | {_rng([r['ng'] for r in b], fmt='{:+.0f}')}% | "
                         f"{_rng(share, fmt='{:.0f}')}% | {_rng([r['shape'] for r in a + b])}% | "
                         f"{_rng([r['shape_nlo'] for r in a + b], fmt='{:+.2f}')}% |"))
    lines += [row for _, row in sorted(rows)]
    out += ["### S2 scale ladder (l = 1…4, real and RSD)", "", *lines, ""]
    text = "\n".join(out)
    (root / "summary_ladder.md").write_text(text)
    ladder_figure(configs, root / "summary_ladder.png")
    return text


def ladder_figure(configs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for (name, max_sigma, spaces), marker in zip(configs, ("o", "s")):
        for space, ls in (("real", "-"), ("rsd", "--")):
            rows1, rows2, _ = spaces[space]
            for ell in range(5):
                sub = sorted((r for r in rows1 if r["l"] == ell and r["sigma"] <= max_sigma), key=lambda r: r["sigma"])
                sig = [r["sigma"] for r in sub]
                kw = dict(color=colors[ell], ls=ls, marker=marker, ms=3.5, lw=1)
                axes[0].plot(sig, [abs(r["ng"]) for r in sub], **kw)
                axes[1].plot(sig, [abs(r["nlo"]) / r["stat"] for r in sub], **kw)
                if ell > 0:
                    sub2 = sorted((r for r in rows2 if r["l"] == ell and r["j2"] == r["j1"] + 2 and r["s2"] <= max_sigma),
                                  key=lambda r: r["s1"])
                    axes[2].plot([r["s1"] for r in sub2], [r["ng"] for r in sub2], **kw)
            if space == "real":
                stat = sorted(((r["sigma"], r["stat"]) for r in rows1 if r["sigma"] <= max_sigma))
                axes[0].plot([a for a, _ in stat], [b for _, b in stat], ":", color="0.5", marker=marker, ms=2, lw=0.6)
    axes[1].axhline(1, color="0.4", lw=0.8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].set_xlabel(r"$\sigma_j$ [Mpc/h]")
    axes[1].set_xlabel(r"$\sigma_j$ [Mpc/h]")
    axes[2].set_xlabel(r"$\sigma_{j_1}$ [Mpc/h]  ($j_2=j_1+2$, $l\geq1$)")
    axes[0].set_ylabel(r"$|S_1^{\rm Nbody}/S_1^{\rm Gauss}-1|$  (grey: 1-box error)")
    axes[1].set_ylabel("S1 NLO residual / 1-box error")
    axes[2].set_ylabel(r"$S_2^{\rm Nbody}/S_2^{\rm Gauss}-1$")
    handles = [plt.Line2D([], [], color=colors[ell], label=f"l={ell}") for ell in range(5)]
    handles += [plt.Line2D([], [], color="k", ls="-", label="real"), plt.Line2D([], [], color="k", ls="--", label="RSD"),
                plt.Line2D([], [], color="k", marker="o", ls="", label="3.9 Mpc/h cells, σ0=0.8"),
                plt.Line2D([], [], color="k", marker="s", ls="", label="BOSS scales")]
    axes[0].legend(handles=handles, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def figure(tables, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for label, (rows1, rows2), ls in tables:
        for ell in range(5):
            sub = [r for r in rows1 if r["l"] == ell]
            sig = [r["sigma"] for r in sub]
            axes[0].plot(sig, [abs(r["ng"]) for r in sub], ls, color=colors[ell],
                         label=f"l={ell}" if ls == "-" else None)
            axes[0].plot(sig, [r["stat"] for r in sub], ":", color=colors[ell], lw=0.8)
            axes[1].plot(sig, [abs(r["nlo"]) for r in sub], ls, color=colors[ell])
            sub2 = [r for r in rows2 if r["l"] == ell and r["l"] > 0 and r["j2"] == r["j1"] + 2]
            axes[2].plot([r["s1"] for r in sub2], [r["ng"] for r in sub2], ls, color=colors[ell])
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma_j$ [Mpc/h]")
    axes[0].set_ylabel(r"$|S_1^{\rm Nbody}/S_1^{\rm Gauss}-1|$  (dotted: 1-box error)")
    axes[1].set_ylabel("|S1 residual after NLO Edgeworth|")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\sigma_{j_1}$ [Mpc/h]  ($j_2=j_1+2$)")
    axes[2].set_ylabel(r"$S_2^{\rm Nbody}/S_2^{\rm Gauss}-1$")
    axes[0].legend(fontsize=8)
    axes[0].set_title("solid: real space, dashed: redshift space", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/feasibility"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--ladder", action="store_true", help="merge both scale configurations")
    args = parser.parse_args()
    if args.ladder:
        print(ladder(args.root))
        return
    out = []
    tables = []
    for space, ls in (("real", "-"), ("rsd", "--")):
        runs = load(args.root, space, args.tag)
        if not runs:
            continue
        cell = runs[0]["config"]["cellsize"]
        sigma0 = runs[0]["config"]["sigma0"]
        rows1, rows2 = s1_table(runs, cell, sigma0), s2_table(runs, cell, sigma0)
        tables.append((space, (rows1, rows2), ls))
        out.append(f"## {space} space, {len(runs)} realizations (z=0.5, cell {cell:.2f} Mpc/h, sigma0={sigma0})\n")
        out.append(f"S0: NG = {100 * np.mean([r['nbody']['S0'] / r['gauss']['S0'] - 1 for r in runs]):+.1f}%\n")
        out.append("### S1\n\n" + fmt_s1(rows1, space == "real") + "\n")
        out.append("### S2\n\n" + fmt_s2(rows2, space == "real") + "\n")
    if len(tables) == 2:
        out.insert(0, condensed(tables[0][1], tables[1][1]))
    md = args.root / f"summary{args.tag}.md"
    md.write_text("\n".join(out))
    figure(tables, args.root / f"summary{args.tag}.png")
    print(md.read_text())


if __name__ == "__main__":
    main()
