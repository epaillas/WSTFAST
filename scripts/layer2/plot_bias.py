#!/usr/bin/env python
"""Shift of the posterior mean of omega_cdm from the truth, in units of its error, against the covariance volume, for
the Gaussian and non-Gaussian second-layer models of scripts/layer2/fit.py (analytic covariance, dashed: mock).

    python scripts/layer2/plot_bias.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/paper"))

from make_figures import FULL, SERIES  # noqa: E402

FITS = ROOT / "outputs/layer2/fits"
VOLUMES = (1, 8, 25, 64)
PANELS = {"baseline": "pade_l_cal_fromadjacent", "conservative": "pade_l_cal_fromadjacent",
          "adjacent": "pade_l_cal_fromnonadjacent"}
TITLES = {"baseline": "baseline", "conservative": "conservative", "adjacent": "+ adjacent pairs"}


def series(vector, model, cov, param):
    out = []
    for v in VOLUMES:
        path = FITS / f"{vector}_{model}{cov}_V{v}" / "summary.json"
        out.append(json.loads(path.read_text())["bias_sigma"][param] if path.exists() else float("nan"))
    return out


def main():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), sharey=True)
    for ax, (vector, complete) in zip(axes, PANELS.items()):
        for i, (model, label) in enumerate((("gauss", "Gaussian 2nd layer (paper)"),
                                            (complete, "non-Gaussian 2nd layer\n(held-out calibration)"))):
            for cov, ls, marker in (("", "-", "o"), ("_mock", "--", "s")):
                ax.plot(VOLUMES, series(vector, model, cov, "omega_cdm"), color=SERIES[i], ls=ls, marker=marker,
                        mfc="white", ms=3, lw=1.0, label=label if cov == "" else None)
        ax.axhspan(-0.5, 0.5, color="0.9", lw=0, zorder=0)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xscale("log")
        ax.set_xticks(VOLUMES, [str(v) for v in VOLUMES])
        ax.set_xlabel(r"$V\ [(h^{-1}{\rm Gpc})^3]$")
        ax.set_title(TITLES[vector], fontsize=7, pad=3)
    axes[0].set_ylabel(r"$\Delta\omega_{\rm cdm}/\sigma$")
    axes[0].plot([], [], color="0.4", ls="--", marker="s", mfc="white", ms=3, lw=1.0, label="mock covariance")
    axes[0].legend(loc="lower left", fontsize=5, frameon=False)
    fig.tight_layout()
    out = ROOT / "outputs/layer2/bias_vs_volume.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
