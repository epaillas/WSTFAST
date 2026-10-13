#!/usr/bin/env python
"""Triangle plot (getdist) of omega_cdm, ln(10^10 A_s) and h for runs of scripts/layer2/fit.py. Example:

    python scripts/layer2/plot_triangle.py --runs baseline_pade_l_cal_h_fromadjacent_V25 \
        baseline_pade_l_prior0.5_h_fromadjacent_V25 --labels "c_l fixed" "c_l free, prior" \
        --title "baseline, V = 25" --output outputs/layer2/triangle_baseline_V25.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from getdist import MCSamples, plots

ROOT = Path(__file__).resolve().parents[2]
FITS = ROOT / "outputs/layer2/fits"
PARAMS = ("omega_cdm", "logA", "h")
LATEX = {"omega_cdm": r"\omega_{\rm cdm}", "logA": r"\ln(10^{10}A_s)", "h": "h"}
COLORS = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = []
    for run, label in zip(args.runs, args.labels):
        data = np.load(FITS / run / "samples.npz")
        samples.append(MCSamples(samples=np.column_stack([data[p] for p in PARAMS]), weights=data["weight"],
                                 names=list(PARAMS), labels=[LATEX[p] for p in PARAMS], label=label,
                                 settings=dict(smooth_scale_2D=0.4, smooth_scale_1D=0.4)))
    truth = json.loads((FITS / args.runs[0] / "summary.json").read_text())["truth"]
    g = plots.get_subplot_plotter(width_inch=4.2)
    g.settings.legend_fontsize = 9
    g.triangle_plot(samples, list(PARAMS), filled=[False] * len(samples), contour_colors=COLORS[:len(samples)],
                    contour_lws=[1.3] * len(samples), markers={p: truth[p] for p in PARAMS},
                    legend_loc="upper right")
    if args.title:
        g.fig.text(0.97, 0.84, args.title.replace("; ", "\n"), ha="right", va="top", fontsize=7, linespacing=1.4)
    g.export(str(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
