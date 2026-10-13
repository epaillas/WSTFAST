"""V0b/V0c of docs/survey_plan.md: the Gaussian-mock window correction R on Quijote halos (no erosion).

Each redshift-space Quijote FoF halo box (z = 0.5, line of sight z) is tiled 2 x 2 x 2 into the 2 Gpc/h box of
v0a_gaussian.py and masked by the same cone. The tiling repeats structure, which does not matter here: masked and
unmasked measurements share the field.

* ``--mode uniform`` (V0b): all halos, edges only.
* ``--mode radial`` (V0c): halos kept with probability p(r) = nbar(r) / nbar_halo inside the cone (the radial nbar of
  v0a_gaussian.py scaled to a peak of 3e-4) and with the FKP-weighted mean probability outside; FKP weights.
  delta = counts / (p(x) N / n^3) - 1.

R is calibrated once on Gaussian mocks (Quijote linear spectrum, b = 1.96, f = 0.763, the same nbar map and
weights). For each box, S1m is measured masked and unmasked on the same cells. The unmasked value stands for the box
model at the effective noise. Reported, in units of the scatter of one survey (the masked S1m over boxes):
uncorrected, masked - unmasked; corrected, masked - R unmasked; and the difference between the halo ratio
<masked> / <unmasked> and the Gaussian R, with its error.

    python scripts/survey/v0b_halos.py --mode uniform --output outputs/survey/v0b
    python scripts/survey/v0b_halos.py --mode radial --output outputs/survey/v0c
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from v0a_gaussian import Setup
from wstfast.config import QUIJOTE_COSMOLOGY
from wstfast.quijote import halo_positions, paint_cic
from wstfast.survey import GaussianMocks, measure_wst_survey, window_ratio

NBAR_HALO = 3.1e-4  # FoF halos (>= 20 particles) at z = 0.5
BIAS, GROWTH_RATE = 1.96, 0.763


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("uniform", "radial"), default="uniform")
    parser.add_argument("--nmesh", type=int, default=200)
    parser.add_argument("--realizations", type=int, nargs="+", default=list(range(40)))
    parser.add_argument("--calibration", type=int, default=32, help="Gaussian mocks of R")
    parser.add_argument("--radial-peak", type=float, default=3e-4, help="radial mode: peak nbar [(h/Mpc)^3]")
    parser.add_argument("--window", choices=("cic", "none"), default="cic", help="assignment window of the mocks")
    parser.add_argument("--reuse", action="store_true", help="recalibrate R only; halos from output/measurements.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    defaults = dict(boxsize=2000.0, cone=20.0, rmin=1100.0, rmax=2300.0, observer_depth=900.0, nbar_peak=6e-4,
                    bias=BIAS, growth_rate=GROWTH_RATE, q=0.8, L=4, p0=1e4)
    for key, value in defaults.items():
        setattr(args, key, value)
    return args


def quijote_power():
    from cosmoprimo import Cosmology

    c = QUIJOTE_COSMOLOGY
    cosmo = Cosmology(h=c["h"], omega_b=c["omega_b"], omega_cdm=c["omega_cdm"], n_s=c["n_s"], logA=c["logA"],
                      m_ncdm=[], engine="class")
    return cosmo.get_fourier().pk_interpolator().to_1d(z=0.5)


def main():
    args = get_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup = Setup(args)
    n = args.nmesh
    sigmas = [s * setup.cell for _, s in setup.scales()]

    # Sampling probability per cell, and the nbar map and weights of the measurement and of the mocks.
    if args.mode == "uniform":
        prob = np.ones(setup.support.shape)
    else:
        inside = setup.nbar * (args.radial_peak / args.nbar_peak) / NBAR_HALO
        x = inside * NBAR_HALO * args.p0
        weights = np.where(setup.support, (x / (1 + x)) ** 2, 0.0)
        effective = np.sum(weights * inside) / np.sum(weights)
        prob = np.where(setup.support, inside, effective)
    nbar = prob * NBAR_HALO
    x = nbar * args.p0
    weights = np.where(setup.support, (x / (1 + x)) ** 2, 0.0)
    print(f"{args.mode}: nbar {nbar[setup.support].min():.2e}..{nbar[setup.support].max():.2e}", flush=True)

    common = dict(config=setup.config, support=setup.support, weights=weights, qs=[args.q], los="z",
                  lattice=setup.lattice)
    t0 = time.time()
    mocks = GaussianMocks(n, setup.cell, quijote_power(), BIAS, GROWTH_RATE, np.where(setup.support, nbar, 0.0),
                          setup.support, weights=weights, window=None if args.window == "none" else args.window)
    calibration = window_ratio(mocks, realizations=args.calibration, seed=0, **common)["S1m"]
    ratio = calibration["ratio"][0]
    print(f"R: {np.nanmin(ratio):.4f}..{np.nanmax(ratio):.4f} ({time.time() - t0:.0f} s)", flush=True)

    # Cell of every halo, to apply the sampling probability.
    shifts = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * 1000.0
    masked, reference = [], []
    if args.reuse:
        saved = np.load(args.output / "measurements.npz")
        masked, reference = list(saved["masked"]), list(saved["reference"])
        args.realizations = []
    for realization in args.realizations:
        t0 = time.time()
        pos = halo_positions(realization, redshift=0.5, mmin=1e13, rsd=True, los=2)
        pos = (pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        total = len(pos)
        if args.mode == "radial":
            cells = np.minimum((pos / setup.cell).astype(int), n - 1)
            keep = np.random.default_rng([realization, 7]).random(total) < prob[cells[:, 0], cells[:, 1], cells[:, 2]]
            pos = pos[keep]
        counts = (paint_cic([pos], n, args.boxsize).astype(np.float64) + 1.0) * len(pos) / n**3
        delta = (counts / (prob * total / n**3) - 1.0).astype(np.float32)
        masked.append(measure_wst_survey(delta, support=setup.support, **{k: v for k, v in common.items()
                                                                           if k != "support"})["S1m"][0])
        reference.append(measure_wst_survey(delta, moment_support=setup.support,
                                            **{k: v for k, v in common.items() if k != "support"})["S1m"][0])
        print(f"box {realization}: {len(pos)} halos, {time.time() - t0:.0f} s", flush=True)
        np.savez(args.output / "measurements.npz", masked=np.array(masked), reference=np.array(reference),
                 ratio=ratio, ratio_error=calibration["error"][0], realizations=args.realizations[:len(masked)],
                 sigma=sigmas)

    masked, reference = np.array(masked), np.array(reference)
    np.savez(args.output / f"ratio_{args.window}.npz", ratio=ratio, ratio_error=calibration["error"][0])
    scatter = masked.std(axis=0, ddof=1)
    nbox = len(masked)
    raw = (masked - reference).mean(axis=0) / scatter
    residual = masked - ratio * reference
    corrected = residual.mean(axis=0) / scatter
    error = np.sqrt((residual.std(axis=0, ddof=1) / np.sqrt(nbox)) ** 2
                    + (calibration["error"][0] * reference.mean(axis=0)) ** 2) / scatter
    halo_ratio = masked.mean(axis=0) / reference.mean(axis=0)
    summary = {"mode": args.mode, "sigma": sigmas, "boxes": nbox,
               "uncorrected": np.nanmax(np.abs(raw), axis=(1, 2)).tolist(),
               "corrected": np.nanmax(np.abs(corrected), axis=(1, 2)).tolist(),
               "corrected_rms": float(np.sqrt(np.nanmean(corrected**2))),
               "error_max": float(np.nanmax(error)),
               "chi2": float(np.nansum((corrected / error) ** 2)), "nblocks": int(np.isfinite(corrected).sum()),
               "ratio_halo_minus_gauss_max": np.nanmax(np.abs(halo_ratio - ratio), axis=(1, 2)).tolist(),
               "edge_effect_max": np.nanmax(np.abs(1 - ratio), axis=(1, 2)).tolist()}
    summary["window"] = args.window
    (args.output / f"summary_{args.window}.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{args.mode}, {nbox} boxes; |bias| / sigma_survey per sigma_j " + " ".join(f"{s:.0f}" for s in sigmas))
    print("  uncorrected  " + " ".join(f"{x:6.2f}" for x in summary["uncorrected"]))
    print("  corrected    " + " ".join(f"{x:6.2f}" for x in summary["corrected"])
          + f"   rms {summary['corrected_rms']:.2f}, error <= {summary['error_max']:.2f},"
            f" chi2 {summary['chi2']:.0f} / {summary['nblocks']}")
    print("  |1 - R|      " + " ".join(f"{x:6.4f}" for x in summary["edge_effect_max"]))
    print("  |R_halo - R| " + " ".join(f"{x:6.4f}" for x in summary["ratio_halo_minus_gauss_max"]))


if __name__ == "__main__":
    main()
