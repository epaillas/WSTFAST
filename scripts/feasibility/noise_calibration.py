#!/usr/bin/env python
"""Simulation-calibrated S21m noise: the per-field noise amplitudes a_i and the white term W as linear functions of
cosmology, for fit_rsd.py --s21-noise-calibration.

    a_i(theta) = a_i0 + grad_i . (theta - theta_0),   theta = (logA, omega_cdm) by default,

with a_i0 from a fit of the noise nuisances at the true fiducial cosmology (fit_rsd.py at fixed cosmology,
--covariance-of-mean; ideally on boxes outside any test set, --exclude-like) and grad_i from least squares over the
differences between Quijote derivative sets and the fiducial boxes of the same seeds (each fitted at its own truth).
Seed pairing cancels most of the cosmic variance in the differences. Every set (and the reference) is a fit_rsd.py
summary.json; the truths are given on the command line, as NAME=SUMMARY:logA,omega_cdm.

    python scripts/feasibility/noise_calibration.py --fiducial outputs/noise_calibration/v2/cal_holdout/summary.json \\
        --reference outputs/noise_calibration/v2/fid/summary.json \\
        --set s8m=outputs/noise_calibration/v2/s8m/summary.json:3.02484,0.120926 ... --output <npz>
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from wstfast.config import QUIJOTE_COSMOLOGY


def noise_names(bestfit):
    return sorted(name for name in bestfit if name.startswith("noise_") or name.startswith("W_j"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fiducial", required=True, help="summary of the noise fit at the fiducial truth (a_i0)")
    parser.add_argument("--reference", required=True, help="summary of the fiducial fit on the seeds of the sets")
    parser.add_argument("--set", nargs="+", required=True, metavar="NAME=SUMMARY:THETA",
                        help="derivative set: its summary and its true parameters (in the order of --params)")
    parser.add_argument("--params", nargs="+", default=["logA", "omega_cdm"])
    parser.add_argument("--leave-out", nargs="*", default=[], help="sets left out of the gradients (validation)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fiducial = json.loads(open(args.fiducial).read())["bestfit"]
    reference = json.loads(open(args.reference).read())["bestfit"]
    names = noise_names(fiducial)
    theta0 = np.array([QUIJOTE_COSMOLOGY[p] for p in args.params])
    rows, deltas, used = [], [], []
    for item in args.set:
        name, _, rest = item.partition("=")
        path, _, theta = rest.rpartition(":")
        if name in args.leave_out:
            continue
        best = json.loads(open(path).read())["bestfit"]
        rows.append(np.array([float(v) for v in theta.split(",")]) - theta0)
        deltas.append([best[n] - reference[n] for n in names])
        used.append(name)
    X, Y = np.array(rows), np.array(deltas)  # (nset, nparam), (nset, nname)
    gradient, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (nparam, nname)
    residual = Y - X @ gradient
    print(f"gradients of {len(names)} noise nuisances in {args.params} from {used}")
    for i, n in enumerate(names):
        print(f"  {n:16s} a0 {fiducial[n]:+10.4f}  grad " + " ".join(f"{g:+10.4f}" for g in gradient[:, i])
              + f"  | residual rms {np.sqrt(np.mean(residual[:, i] ** 2)):.4f}")
    np.savez(args.output, names=np.array(names), value=np.array([fiducial[n] for n in names]), gradient=gradient.T,
             params=np.array(args.params), reference=theta0, sets=np.array(used))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
