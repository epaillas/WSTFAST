#!/usr/bin/env python
"""Contours (68% and 95%) of omega_cdm and ln(10^10 A_s) for the second-layer models of scripts/layer2/fit.py.

Each panel is one data vector and volume; each colour one model. Example:

    python scripts/layer2/plot_contours.py --runs baseline_gauss_V25 baseline_pade_l_V25 \
        --labels "Gaussian layer" "non-Gaussian layer" --output outputs/layer2/contours_V25.pdf
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

from make_figures import COLUMN, SERIES, contour_levels  # noqa: E402

FITS = ROOT / "outputs/layer2/fits"
LABELS = {"omega_cdm": r"$\omega_{\rm cdm}$", "logA": r"$\ln(10^{10}A_s)$"}


def contours(ax, run, color, ls, truth, span):
    data = np.load(FITS / run / "samples.npz")
    x, y, w = data["omega_cdm"], data["logA"], data["weight"]
    ranges = [[truth["omega_cdm"] - span[0], truth["omega_cdm"] + span[0]], [truth["logA"] - span[1], truth["logA"] + span[1]]]
    hist, xe, ye = np.histogram2d(x, y, bins=70, weights=w, range=ranges)
    hist = gaussian_filter(hist, 1.3)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    ax.contour(xc, yc, hist.T, levels=sorted(contour_levels(hist)), colors=[color, color], linewidths=[0.8, 1.3],
               linestyles=[ls, ls])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run folders; ';' separates panels")
    parser.add_argument("--labels", nargs="+", required=True, help="one per model (colour)")
    parser.add_argument("--titles", nargs="+", default=None)
    parser.add_argument("--nsigma", type=float, default=4.0, help="half-width of each panel in posterior sigmas")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panels = [p.split(",") for p in " ".join(args.runs).split(";")]
    panels = [[r.strip() for r in p if r.strip()] for p in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(COLUMN * (0.55 + 0.5 * len(panels)), 2.5), squeeze=False)
    for ip, (ax, runs) in enumerate(zip(axes[0], panels)):
        summaries = [json.loads((FITS / r / "summary.json").read_text()) for r in runs]
        truth = summaries[0]["truth"]
        # Limits: the union of mean +- nsigma std of every run, and the truth.
        lims = {}
        for n in ("omega_cdm", "logA"):
            lo = min([s["posterior"][n]["mean"] - args.nsigma * s["posterior"][n]["std"] for s in summaries] + [truth[n]])
            hi = max([s["posterior"][n]["mean"] + args.nsigma * s["posterior"][n]["std"] for s in summaries] + [truth[n]])
            lims[n] = (lo, hi)
        span = (lims["omega_cdm"][1] - lims["omega_cdm"][0], lims["logA"][1] - lims["logA"][0])
        for i, run in enumerate(runs):
            contours(ax, run, SERIES[i], "-", truth, span)
            ax.plot([], [], color=SERIES[i], label=args.labels[i])
        ax.axvline(truth["omega_cdm"], color="0.5", lw=0.5, ls=":")
        ax.axhline(truth["logA"], color="0.5", lw=0.5, ls=":")
        ax.set_xlim(*lims["omega_cdm"])
        ax.set_ylim(*lims["logA"])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(LABELS["omega_cdm"])
        if ip == 0:
            ax.set_ylabel(LABELS["logA"])
            ax.legend(loc="upper right", fontsize=6, frameon=False)
        if args.titles:
            ax.set_title(args.titles[ip], fontsize=7, pad=3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
