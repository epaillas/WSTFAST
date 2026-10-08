#!/usr/bin/env python
"""Fit the real-space matter power spectrum of Quijote with dsc-model's tree / one-loop EFT (desilike).

The P(k) counterpart of inference.py, on the same realizations (P_dd is stored in the WST files):
P = P_L + L_Lambda - 2 cs2_pk k^2 P_L, averaged over the lattice modes of each bin with the CIC
window, plus the known particle shot noise. See wstfast/theory/power.py. Examples:

    # profile + MH chains with the Taylor-emulated basis (train_emulator.py --stat pk)
    python scripts/fit_power.py --kmax 0.15 --output-dir outputs/inference/pk/kmax0.15

    # exact model (CLASS + loop at every step), profile only; tree level; unregulated loop
    python scripts/fit_power.py --emulator none --method profile --output-dir outputs/inference/pk/exact
    python scripts/fit_power.py --order tree --emulator outputs/emulators/pk_tree_taylor.h5 --kmax 0.1 ...
    python scripts/fit_power.py --cutoff none --emulator outputs/emulators/pk_spt_taylor.h5 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from desilike import setup_logging
from desilike.base import Posterior
from wstfast.config import QUIJOTE_COSMOLOGY
from wstfast.data import load_power_dataset
from wstfast.inference import (build_power_likelihood, covariance_for, plot_power_fit, profile, sample_mh,
                                summarize)
from wstfast.theory.power import ORDERS


def cutoff_arg(value: str) -> float | None:
    """Loop regulator in h/Mpc, or 'none' for unregulated SPT."""
    return None if value.lower() == "none" else float(value)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256"))
    parser.add_argument("--kmin", type=float, default=0.0)
    parser.add_argument("--kmax", type=float, default=0.15)
    parser.add_argument("--rebin", type=int, default=2, help="P(k) bins of rebin x k_f")
    parser.add_argument("--order", choices=ORDERS, default="one-loop")
    parser.add_argument("--cutoff", type=cutoff_arg, default=0.5,
                        help="loop regulator Lambda in h/Mpc (dsc-model: 0.5), or 'none' for unregulated SPT")
    parser.add_argument("--counterterm", action=argparse.BooleanOptionalAction, default=None,
                        help="free cs2_pk (default: at one loop only)")
    parser.add_argument("--covariance", choices=("auto", "sample", "diagonal"), default="auto")
    parser.add_argument("--covariance-of-mean", action="store_true",
                        help="errors of the realization mean (default: one 1 (Gpc/h)^3 box)")
    parser.add_argument("--vary", nargs="+", default=["omega_cdm", "logA"], choices=sorted(QUIJOTE_COSMOLOGY))
    parser.add_argument("--emulator", default="outputs/emulators/pk_taylor.h5",
                        help="trained P(k) basis emulator, or 'none' for the exact model")
    parser.add_argument("--method", choices=("profile", "sample"), default="sample")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--proposal-from", type=Path, default=None,
                        help="samples.h5 of an earlier run: use its chain covariance as the MH proposal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference/pk/fiducial"))
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_power_dataset(args.data_dir, "real", kmin=args.kmin, kmax=args.kmax, rebin=args.rebin)
    covariance, kind = covariance_for(dataset, args.covariance, of_mean=args.covariance_of_mean)
    print(f"{dataset.k.size} bins in [{dataset.k[0]:.4f}, {dataset.k[-1]:.4f}] h/Mpc, "
          f"{dataset.vectors.shape[0]} realizations, {kind} covariance")

    emulator = None if args.emulator == "none" else Path(args.emulator)
    likelihood = build_power_likelihood(dataset, covariance, vary=args.vary, order=args.order, cutoff=args.cutoff,
                                        emulator=emulator, counterterm=args.counterterm)
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
    summary.update(vary=args.vary, order=args.order, cutoff=args.cutoff, counterterm=args.counterterm,
                   kmin=args.kmin, kmax=args.kmax, rebin=args.rebin, emulator=str(emulator),
                   covariance_of_mean=args.covariance_of_mean, data_dir=str(args.data_dir))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    cutoff = "unregulated" if args.cutoff is None else f"Lambda = {args.cutoff} h/Mpc"
    plot_power_fit(args.output_dir / "bestfit.png", likelihood, dataset, covariance,
                   title=f"P(k) fit ({args.order}, {cutoff}, kmax = {args.kmax})")
    print(f"chi2 = {summary['chi2']:.1f} for {summary['ndata']} data points and {summary['nvaried']} parameters")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
