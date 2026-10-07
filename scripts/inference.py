#!/usr/bin/env python
"""Fit WST coefficients of Quijote matter with the perturbative model (desilike, MH by default).

Examples:

    # profile + MH chains with the Taylor-emulated basis (train it first with train_emulator.py)
    python scripts/inference.py --output-dir outputs/inference/fiducial

    # exact model (CLASS at every step), profile only
    python scripts/inference.py --emulator none --method profile --output-dir outputs/inference/exact
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from desilike import setup_logging
from desilike.base import Posterior
from wstfast.config import QUIJOTE_COSMOLOGY, select_coefficients
from wstfast.data import load_dataset, load_measurement
from wstfast.inference import (build_likelihood, covariance_for, plot_fit, profile, sample_mh,
                                summarize)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/quijote/fiducial/z0.5/J4_L4_sigma0.8_n256"))
    parser.add_argument("--space", choices=("real",), default="real", help="the model is real-space only for now")
    parser.add_argument("--q", type=float, default=None, help="WST exponent (default: the measurement's first q)")
    parser.add_argument("--s1-min-scale", type=float, default=25.0, help="smallest sigma_j [Mpc/h] for S1")
    parser.add_argument("--s1-ells", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--s21-min-scale", type=float, default=12.5, help="smallest sigma_j1 [Mpc/h] for S2/S1")
    parser.add_argument("--s21-ells", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--s21-min-ratio", type=float, default=2.8,
                        help="smallest sigma_j2 / sigma_j1 for S2/S1 (2.8 drops adjacent pairs)")
    parser.add_argument("--scale-stride", type=int, default=1,
                        help="keep scales j divisible by this (2 on the half-octave superset = the dyadic subset)")
    parser.add_argument("--covariance", choices=("auto", "sample", "diagonal"), default="auto")
    parser.add_argument("--covariance-of-mean", action="store_true",
                        help="errors of the realization mean (default: one 1 (Gpc/h)^3 box)")
    parser.add_argument("--vary", nargs="+", default=["omega_cdm", "logA"], choices=sorted(QUIJOTE_COSMOLOGY))
    parser.add_argument("--emulator", default="outputs/emulators/wst_basis_taylor.h5",
                        help="trained basis emulator, or 'none' for the exact model")
    parser.add_argument("--method", choices=("profile", "sample"), default="sample")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--proposal-from", type=Path, default=None,
                        help="samples.h5 of an earlier run: use its chain covariance as the MH proposal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference/fiducial"))
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    first = next(iter(sorted((args.data_dir / args.space).glob("wst_r*.npz"))))
    config = load_measurement(first, q=args.q)["config"]
    coefficients = select_coefficients(config, s1_min_scale=args.s1_min_scale, s1_ells=args.s1_ells,
                                       s21_min_scale=args.s21_min_scale, s21_ells=args.s21_ells,
                                       s21_min_ratio=args.s21_min_ratio)
    coefficients = [c for c in coefficients
                    if c.j % args.scale_stride == 0 and (c.j2 is None or c.j2 % args.scale_stride == 0)]
    dataset = load_dataset(args.data_dir, args.space, coefficients, q=config.q)
    covariance, kind = covariance_for(dataset, args.covariance, of_mean=args.covariance_of_mean)
    print(f"{len(coefficients)} coefficients, {dataset.vectors.shape[0]} realizations, {kind} covariance")

    emulator = None if args.emulator == "none" else Path(args.emulator)
    likelihood = build_likelihood(dataset, covariance, vary=args.vary, emulator=emulator)
    posterior = Posterior(likelihood)

    profiles = profile(posterior, args.output_dir / "profiles.h5", seed=args.seed)
    print(profiles.to_stats(tablefmt="pretty"))
    samples = None
    if args.method == "sample":
        proposal = None
        if args.proposal_from is not None:
            from desilike.samples import Samples

            proposal = Samples.read(str(args.proposal_from))
        samples = sample_mh(posterior, profiles, args.output_dir / "chains", chains=args.chains, proposal=proposal,
                            seed=args.seed, max_steps=args.max_steps)
        samples.write(str(args.output_dir / "samples.h5"))
        print(samples.to_stats(tablefmt="pretty"))

    summary = summarize(likelihood, profiles, samples, dataset, kind)
    summary.update(vary=args.vary, emulator=str(emulator), covariance_of_mean=args.covariance_of_mean,
                   data_dir=str(args.data_dir), space=args.space)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_fit(args.output_dir / "bestfit.png", likelihood, dataset, covariance, config)
    print(f"chi2 = {summary['chi2']:.1f} for {summary['ndata']} data points and {summary['nvaried']} parameters")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
