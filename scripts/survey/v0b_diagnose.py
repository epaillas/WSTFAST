"""Why is the edge effect of Quijote halos (v0b_halos.py, uniform) ~4% larger than the Gaussian-mock R predicts?

R is recomputed with mock variants on the V0b geometry (uniform nbar = 3.1e-4, the halo b and f). Each variant adds
one ingredient of the halo field: the CIC painting window, Poisson discreteness, lognormal non-Gaussianity. For each
variant, the edge effect is compared with the Gaussian one, (R_variant - 1) / (R_gauss - 1). This is set next to
the halos' (R_halo - 1) / (R_gauss - 1), read from outputs/survey/v0b.

    python scripts/survey/v0b_diagnose.py --output outputs/survey/v0b_diagnose
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from v0a_gaussian import Setup
from v0b_halos import BIAS, GROWTH_RATE, NBAR_HALO, quijote_power
from wstfast.survey import GaussianMocks, window_ratio

VARIANTS = {
    "gauss": {},
    "gauss_cic": dict(window="cic"),
    "poisson_cic": dict(window="cic", discrete="poisson"),
    "lognormal_cic": dict(window="cic", discrete="lognormal"),
    "gauss_cic_noise0.8": dict(window="cic", noise_factor=0.8),  # sub-Poisson noise, 0.8 / nbar
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nmesh", type=int, default=200)
    parser.add_argument("--mocks", type=int, default=16)
    parser.add_argument("--halos", type=Path, default=Path("outputs/survey/v0b/measurements.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/survey/v0b_diagnose"))
    args = parser.parse_args()
    for key, value in dict(boxsize=2000.0, cone=20.0, rmin=1100.0, rmax=2300.0, observer_depth=900.0,
                           nbar_peak=6e-4, bias=BIAS, growth_rate=GROWTH_RATE, q=0.8, L=4, p0=1e4).items():
        setattr(args, key, value)
    args.output.mkdir(parents=True, exist_ok=True)
    setup = Setup(args)
    nbar = np.where(setup.support, NBAR_HALO, 0.0)
    pk = quijote_power()
    common = dict(config=setup.config, support=setup.support, qs=[args.q], los="z", lattice=setup.lattice)

    halos = np.load(args.halos)
    ratios = {"halos": halos["masked"].mean(axis=0) / halos["reference"].mean(axis=0)}
    errors = {}
    for name, kwargs in VARIANTS.items():
        t0 = time.time()
        kwargs = dict(kwargs)
        scaled = nbar / kwargs.pop("noise_factor", 1.0)
        mocks = GaussianMocks(args.nmesh, setup.cell, pk, BIAS, GROWTH_RATE, scaled, setup.support, **kwargs)
        out = window_ratio(mocks, realizations=args.mocks, seed=0, **common)["S1m"]
        ratios[name], errors[name] = out["ratio"][0], out["error"][0]
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)

    gauss = ratios["gauss"]
    summary = {}
    print("\n(R - 1) / (R_gauss - 1): median over blocks per sigma_j [min..max]")
    for name, ratio in ratios.items():
        x = (ratio - 1) / (gauss - 1)
        per_scale = [(float(np.nanmedian(x[j])), float(np.nanmin(x[j])), float(np.nanmax(x[j]))) for j in range(len(x))]
        summary[name] = per_scale
        print(f"  {name:14s} " + "  ".join(f"{m:.3f} [{lo:.3f}..{hi:.3f}]" for m, lo, hi in per_scale))
    np.savez(args.output / "ratios.npz", **ratios, **{f"{k}_error": v for k, v in errors.items()})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
