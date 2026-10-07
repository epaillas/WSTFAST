#!/usr/bin/env python
"""Triangle plot of Fisher forecasts written by fisher.py (Gaussian contours about the Quijote fiducial).

    python scripts/plot_fisher.py outputs/fisher/fisher_q0.8_5p_full.json \
        --cases "P(k), kmax=0.2" "P(k), kmax=0.2 + WST half-octave" --output outputs/fisher/triangle.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wstfast.config import QUIJOTE_COSMOLOGY

LABELS = {"omega_cdm": r"\omega_{\rm cdm}", "logA": r"\ln(10^{10}A_s)", "n_s": "n_s", "h": "h",
          "omega_b": r"\omega_{\rm b}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("forecast", type=Path, help="JSON written by fisher.py")
    parser.add_argument("--cases", nargs="+", required=True, help="case labels as printed by fisher.py")
    parser.add_argument("--volume", type=float, default=1.0, help="volume in (Gpc/h)^3 (errors scale as V^-1/2)")
    parser.add_argument("--output", type=Path, default=Path("outputs/fisher/triangle.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from getdist import gaussian_mixtures, plots

    forecast = json.loads(args.forecast.read_text())
    names = forecast["parameters"]
    mean = np.array([QUIJOTE_COSMOLOGY[name] for name in names])
    gaussians = []
    for case in args.cases:
        covariance = np.array(forecast["cases"][case]["covariance"]) / args.volume
        gaussians.append(gaussian_mixtures.GaussianND(mean, covariance, names=names,
                                                      labels=[LABELS.get(n, n) for n in names], label=case))
    plotter = plots.get_subplot_plotter(width_inch=8)
    plotter.settings.legend_fontsize = 11
    # The first case (the reference) as black outlines, so that it stays visible under the others.
    colors = ["k", "C3", "C0", "C2", "C1"][:len(gaussians)]
    plotter.triangle_plot(gaussians, filled=[index > 0 for index in range(len(gaussians))], contour_colors=colors,
                          contour_lws=[1.5] * len(gaussians), markers={name: value for name, value in zip(names, mean)})
    volume = f"{args.volume:g} (Gpc/h)$^3$"
    plotter.fig.suptitle(f"Fisher forecast, real-space matter, z = {forecast.get('redshift', 0.5)}, "
                         f"q = {forecast['q']}, V = {volume}", y=1.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plotter.export(str(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
