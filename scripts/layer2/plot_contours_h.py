#!/usr/bin/env python
"""Contours of the three-parameter fits (omega_cdm, logA, h; n_s fixed) of scripts/layer2/fit.py: one row per data
vector, one column per parameter pair, Gaussian vs non-Gaussian second layer. Example:

    python scripts/layer2/plot_contours_h.py --volume 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/paper"))

from make_figures import FULL, SERIES, contour_levels  # noqa: E402

FITS = ROOT / "outputs/layer2/fits"
LABELS = {"omega_cdm": r"$\omega_{\rm cdm}$", "logA": r"$\ln(10^{10}A_s)$", "h": r"$h$"}
PAIRS = (("omega_cdm", "logA"), ("omega_cdm", "h"), ("logA", "h"))
ROWS = {"baseline": ("baseline_gauss_h", "baseline_pade_l_cal_h_fromadjacent"),
        "+ adjacent pairs": ("adjacent_gauss_h", "adjacent_pade_l_cal_h")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume", default="25")
    parser.add_argument("--nsigma", type=float, default=3.5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / f"outputs/layer2/contours_3p_V{args.volume}.pdf"
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(FULL, 2.3 * len(ROWS)))
    for ir, (title, runs) in enumerate(ROWS.items()):
        runs = [f"{r}_V{args.volume}" for r in runs]
        summaries = [json.loads((FITS / r / "summary.json").read_text()) for r in runs]
        samples = [np.load(FITS / r / "samples.npz") for r in runs]
        truth = summaries[0]["truth"]
        lims = {}
        for n in LABELS:
            post = [s["posterior"][n] for s in summaries]
            lims[n] = (min([p["mean"] - args.nsigma * p["std"] for p in post] + [truth[n]]),
                       max([p["mean"] + args.nsigma * p["std"] for p in post] + [truth[n]]))
        for ic, (px, py) in enumerate(PAIRS):
            ax = axes[ir, ic]
            for i, data in enumerate(samples):
                hist, xe, ye = np.histogram2d(data[px], data[py], bins=60, weights=data["weight"],
                                              range=[lims[px], lims[py]])
                hist = gaussian_filter(hist, 1.3)
                xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                ax.contour(xc, yc, hist.T, levels=sorted(contour_levels(hist)), colors=[SERIES[i]] * 2,
                           linewidths=[0.8, 1.3])
            ax.axvline(truth[px], color="0.5", lw=0.5, ls=":")
            ax.axhline(truth[py], color="0.5", lw=0.5, ls=":")
            ax.set_xlim(*lims[px])
            ax.set_ylim(*lims[py])
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_xlabel(LABELS[px])
            ax.set_ylabel(LABELS[py])
        axes[ir, 0].set_title(f"{title}, $V={args.volume}\\,(h^{{-1}}{{\\rm Gpc}})^3$, $n_s$ fixed", fontsize=7,
                              loc="left", pad=3)
    for i, label in enumerate(("Gaussian 2nd layer (paper)", "non-Gaussian 2nd layer")):
        axes[0, 2].plot([], [], color=SERIES[i], label=label)
    axes[0, 2].legend(loc="upper right", fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
