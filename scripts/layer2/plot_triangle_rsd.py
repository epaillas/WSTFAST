#!/usr/bin/env python
"""Triangle plots (omega_cdm, ln A_s, h; n_s fixed) of the halo WST baseline with the Gaussian and the non-Gaussian
second layer (outputs/layer2/rsd_fits/<model>_V<volume>, written by outputs/layer2/rsd_fits/run.sh), one figure per
volume. Chains are read like scripts/paper/halos.py (a copy of each chain, first 30% dropped), so runs still
sampling can be plotted; missing runs are skipped. Example:

    python scripts/layer2/plot_triangle_rsd.py --volumes 25 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/paper"))

import halos  # noqa: E402

FITS = ROOT / "outputs/layer2/rsd_fits"
MODELS = {"gauss": "Gaussian 2nd layer (paper)", "halo_c": r"non-Gaussian, halo $c_\ell$ fixed",
          "matter_c": r"non-Gaussian, matter $c_\ell$ fixed", "halo_prior": r"non-Gaussian, halo $c_\ell$ + prior",
          "marg_global": r"non-Gaussian, amplitudes marginalised", "marg_per_l": r"non-Gaussian, amplitudes per $\ell$ marg.",
          "marg_per_l_ang": r"non-Gaussian, per-$\ell$ marg. + free $a_{\rm NG}$"}
TRUTH = {"omega_cdm": 0.120925743885, "logA": 3.061143632659324, "h": 0.6711}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volumes", nargs="+", default=["25", "60"])
    parser.add_argument("--models", nargs="+", default=["gauss", "halo_c", "matter_c", "halo_prior"], choices=list(MODELS))
    parser.add_argument("--tag", default="", help="suffix of the output file names")
    args = parser.parse_args()
    from getdist import plots

    halos.CHAINS = FITS
    for volume in args.volumes:
        samples, colors = [], []
        for i, model in enumerate(args.models):
            label = MODELS[model]
            try:
                samples.append(halos.load(f"{model}_V{volume}", label))
                colors.append(halos.SERIES[i])
            except FileNotFoundError:
                continue
        if not samples:
            print(f"V = {volume}: no chains yet")
            continue
        g = plots.get_subplot_plotter(width_inch=halos.FULL * 0.62)
        g.settings.legend_fontsize = 8
        g.triangle_plot(samples, halos.PARAMS, filled=False, contour_colors=colors, contour_lws=[1.2] * len(samples),
                        markers={p: TRUTH[p] for p in halos.PARAMS}, legend_loc="upper right",
                        title_limit=0)
        g.fig.text(0.97, 0.86, rf"halos, $V={volume}\,(h^{{-1}}{{\rm Gpc}})^3$, $n_s$ fixed", ha="right", fontsize=8)
        out = ROOT / f"outputs/layer2/triangle_halos{args.tag}_V{volume}.pdf"
        g.export(str(out))
        print(f"wrote {out} ({len(samples)} runs)")


if __name__ == "__main__":
    main()
