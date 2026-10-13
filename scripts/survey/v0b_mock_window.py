"""Mock-calibrated window correction: R from one set of survey-matched halo mocks, tested on an independent set.

This is the DESI-style procedure. Mocks of the tracer are cut to the survey geometry (``wstfast.survey.box_to_survey``):
the box is tiled, redshift-space distortions are applied along the line of sight, and the objects are thinned to
nbar(x). R = <S_masked> / <S_reference> is calibrated on them (``calibrate_window``), with the reference being the
unmasked mock on the same cells. The survey model is R x the box model. Here the mocks are Quijote FoF halo boxes
(z = 0.5) and the geometry is the V0a cone.

R is calibrated on ``--calibrate`` boxes and applied to the ``--test`` boxes. The residual masked - R reference of
the test boxes is reported in units of the scatter of one survey, with its error (test scatter and R error). It
should be consistent with zero.

    python scripts/survey/v0b_mock_window.py --mode radial --los z --output outputs/survey/mock_window/radial_z
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from v0a_gaussian import Setup
from wstfast.quijote import halo_catalogue
from wstfast.survey import SurveyGeometry, box_to_survey, calibrate_window


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("uniform", "radial"), default="radial")
    parser.add_argument("--los", choices=("z", "local"), default="z")
    parser.add_argument("--nmesh", type=int, default=200)
    parser.add_argument("--calibrate", type=int, nargs=2, default=[0, 40], metavar=("START", "STOP"))
    parser.add_argument("--test", type=int, nargs=2, default=[40, 80], metavar=("START", "STOP"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key, value in dict(boxsize=2000.0, cone=20.0, rmin=1100.0, rmax=2300.0, observer_depth=900.0,
                           nbar_peak=6e-4, bias=1.96, growth_rate=0.763, q=0.8, L=4, p0=1e4).items():
        setattr(args, key, value)
    return args


def main():
    args = get_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup = Setup(args)  # WST configuration and lattice of v0a_gaussian.py
    observer = np.array([args.boxsize / 2, args.boxsize / 2, -args.observer_depth])
    rmid, width = 0.5 * (args.rmin + args.rmax), 0.3 * (args.rmax - args.rmin)
    if args.mode == "uniform":
        def nbar_of_r(r):
            return np.full_like(r, 3.0e-4)
    else:
        def nbar_of_r(r):
            return 3.0e-4 * np.exp(-0.5 * ((r - rmid) / width) ** 2)
    geometry = SurveyGeometry.cone(args.nmesh, args.boxsize, observer, args.cone, args.rmin, args.rmax, nbar_of_r)
    los = geometry.line_of_sight(full=True) if args.los == "local" else "z"
    common = dict(config=setup.config, support=geometry.support, weights=geometry.weights, qs=[args.q], los=los,
                  lattice=setup.lattice)

    def fields(start, stop):
        for realization in range(start, stop):
            t0 = time.time()
            pos, disp = halo_catalogue(realization)
            yield box_to_survey(pos, 1000.0, geometry, displacements=disp, los=args.los, seed=[realization, 11])
            print(f"box {realization}: {time.time() - t0:.0f} s (cut)", flush=True)

    calibration = calibrate_window(fields(*args.calibrate), **common)["S1m"]
    test = calibrate_window(fields(*args.test), **common)["S1m"]
    ratio, ratio_error = calibration["ratio"][0], calibration["error"][0]
    masked, reference = test["masked"][:, 0], test["reference"][:, 0]
    np.savez(args.output / "measurements.npz", calibration_masked=calibration["masked"][:, 0],
             calibration_reference=calibration["reference"][:, 0], test_masked=masked, test_reference=reference,
             ratio=ratio, ratio_error=ratio_error, sigma=[s * setup.cell for _, s in setup.scales()])

    scatter = masked.std(axis=0, ddof=1)
    raw = (masked - reference).mean(axis=0) / scatter
    residual = masked - ratio * reference
    z = residual.mean(axis=0) / scatter
    error = np.sqrt((residual.std(axis=0, ddof=1) / np.sqrt(len(masked))) ** 2
                    + (ratio_error * reference.mean(axis=0)) ** 2) / scatter
    sigmas = [s * setup.cell for _, s in setup.scales()]
    summary = {"mode": args.mode, "los": args.los, "calibrate": args.calibrate, "test": args.test,
               "uncorrected": np.nanmax(np.abs(raw), axis=(1, 2)).tolist(),
               "corrected": np.nanmax(np.abs(z), axis=(1, 2)).tolist(),
               "corrected_rms": float(np.sqrt(np.nanmean(z**2))), "error_max": float(np.nanmax(error)),
               "chi2": float(np.nansum((z / error) ** 2)), "nblocks": int(np.isfinite(z).sum()),
               "edge_effect_max": np.nanmax(np.abs(1 - ratio), axis=(1, 2)).tolist()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{args.mode}, los {args.los}: R from boxes {args.calibrate}, tested on {args.test};"
          f" |bias| / sigma_survey per sigma_j " + " ".join(f"{s:.0f}" for s in sigmas))
    print("  uncorrected  " + " ".join(f"{x:6.2f}" for x in summary["uncorrected"]))
    print("  corrected    " + " ".join(f"{x:6.2f}" for x in summary["corrected"])
          + f"   rms {summary['corrected_rms']:.2f}, error <= {summary['error_max']:.2f},"
            f" chi2 {summary['chi2']:.0f} / {summary['nblocks']}")
    print("  |1 - R|      " + " ".join(f"{x:6.4f}" for x in summary["edge_effect_max"]))


if __name__ == "__main__":
    main()
