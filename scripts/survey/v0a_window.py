"""V0a, window correction: is R = <S_masked> / <S_unmasked>, calibrated on Gaussian mocks at a fiducial spectrum,
valid for other spectra?

Geometry as in v0a_gaussian.py (cone with a DESI-LRG-like nbar(r), FKP weights, line of sight z). R is computed at
the fiducial spectrum (DESI cosmology at z = 0.5, b = 2, f = 0.75) for each erosion. Then, for every variant
spectrum, paired mocks are measured masked and unmasked on the same cells. The unmasked value stands for the box
model of that variant (validated in v0a_gaussian.py, test B). The residual of the survey model,
masked - R x unmasked, is reported in units of the scatter of one survey, next to the uncorrected masked - unmasked.

    python scripts/survey/v0a_window.py --output outputs/survey/v0a_window
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from v0a_gaussian import Setup
from wstfast.survey import GaussianMocks, window_ratio


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nmesh", type=int, default=200)
    parser.add_argument("--erosion", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--calibration", type=int, default=4, help="mocks of the calibration of R")
    parser.add_argument("--realizations", type=int, default=6, help="mocks per variant")
    parser.add_argument("--output", type=Path, default=Path("outputs/survey/v0a_window"))
    args = parser.parse_args()
    # Geometry and WST settings of v0a_gaussian.py.
    defaults = dict(boxsize=2000.0, cone=20.0, rmin=1100.0, rmax=2300.0, observer_depth=900.0, nbar_peak=6e-4,
                    nbar_uniform=3e-4, bias=2.0, growth_rate=0.75, q=0.8, L=4, p0=1e4)
    for key, value in defaults.items():
        setattr(args, key, value)
    return args


def spectrum(**params):
    from cosmoprimo.fiducial import DESI

    return DESI(**params).get_fourier().pk_interpolator().to_1d(z=0.5)


def variants(fiducial):
    """name: (pk, bias, f, nbar factor)."""
    return {
        "fiducial": (fiducial, 2.0, 0.75, 1.0),
        "bias_f": (fiducial, 1.6, 0.6, 1.0),
        "amplitude": (lambda k: 1.21 * fiducial(k), 2.0, 0.75, 1.0),
        "omega_cdm": (spectrum(omega_cdm=0.145), 2.0, 0.75, 1.0),
        "n_s": (spectrum(n_s=0.90), 2.0, 0.75, 1.0),
        "nbar": (fiducial, 2.0, 0.75, 0.5),
    }


def main():
    args = get_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup = Setup(args)
    sigmas = [s * setup.cell for _, s in setup.scales()]
    common = dict(config=setup.config, support=setup.support, weights=setup.weights, qs=[args.q], los="z",
                  lattice=setup.lattice)

    def mocks(pk, bias, f, nbar_factor):
        return GaussianMocks(args.nmesh, setup.cell, pk, bias, f, setup.nbar * nbar_factor, setup.support,
                             weights=setup.weights)

    fiducial = spectrum()
    ratios = {}
    for c in args.erosion:
        t0 = time.time()
        ratios[c] = window_ratio(mocks(fiducial, 2.0, 0.75, 1.0), erosion=c, realizations=args.calibration, seed=0,
                                 **common)["S1m"]
        r = ratios[c]["ratio"]
        print(f"R at c = {c:g}: {np.nanmin(r):.4f}..{np.nanmax(r):.4f}, error <= {np.nanmax(ratios[c]['error']):.1e}"
              f" ({time.time() - t0:.0f} s)", flush=True)

    summary = {"sigma": sigmas, "erosion": args.erosion, "variants": {}}
    save = {}
    for name, spec in variants(fiducial).items():
        summary["variants"][name] = {}
        for c in args.erosion:
            t0 = time.time()
            out = window_ratio(mocks(*spec), erosion=c, realizations=args.realizations, seed=1000, **common)["S1m"]
            masked, reference = out["masked"][:, 0], out["reference"][:, 0]
            scatter = masked.std(axis=0, ddof=1)
            raw = (masked - reference).mean(axis=0) / scatter
            residual = masked - ratios[c]["ratio"][0] * reference
            corrected = residual.mean(axis=0) / scatter
            error = residual.std(axis=0, ddof=1) / np.sqrt(len(masked)) / scatter
            save[f"{name}_c{c:g}_masked"], save[f"{name}_c{c:g}_reference"] = masked, reference
            row = {"uncorrected": np.nanmax(np.abs(raw), axis=(1, 2)).tolist(),
                   "corrected": np.nanmax(np.abs(corrected), axis=(1, 2)).tolist(),
                   "corrected_rms": float(np.sqrt(np.nanmean(corrected**2))),
                   "error_max": float(np.nanmax(error))}
            summary["variants"][name][f"c={c:g}"] = row
            print(f"{name:10s} c = {c:g}: |bias| / sigma_survey per sigma_j  uncorrected "
                  + " ".join(f"{x:6.2f}" for x in row["uncorrected"]) + "   corrected "
                  + " ".join(f"{x:5.2f}" for x in row["corrected"])
                  + f"   (rms {row['corrected_rms']:.2f}, error <= {row['error_max']:.2f}; {time.time() - t0:.0f} s)",
                  flush=True)
    np.savez(args.output / "measurements.npz", **save,
             **{f"ratio_c{c:g}": ratios[c]["ratio"] for c in args.erosion})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
