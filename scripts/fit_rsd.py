#!/usr/bin/env python
"""Redshift-space fits of Quijote matter: P(k) multipoles, line-of-sight-resolved S1m, or both (desilike, MH).

Both statistics come from the same line-of-sight-resolved files, so the joint covariance (with cross terms)
is the sample covariance of the concatenated vectors [S1m blocks, P_0, P_2, P_4]. They share one emulated
RSDBasis (``train_emulator.py --stat rsd``); each keeps its own counterterms (c0, c2, c4 for S1m and
c0_pk, c2_pk, c4_pk for P(k)). Example (4 parameters, omega_b fixed, errors of a 60 (Gpc/h)^3 survey):

    python scripts/fit_rsd.py --stats pk s1m --kmax 0.12 --vary omega_cdm logA n_s h --volume 60 \
        --emulator outputs/emulators/rsd_basis_taylor_4p.h5 --output-dir outputs/inference/rsd/joint_V60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from desilike import build, get_params, setup_logging
from desilike.base import Posterior
from wstfast.calculators import JointTheory, RSDPowerTheory, S1mTheory, WSTLikelihood
from wstfast.config import QUIJOTE_COSMOLOGY, select_coefficients
from wstfast.data import load_measurement, load_power_dataset, load_s1m_dataset, sample_covariance
from wstfast.inference import (_bound_to_emulator, _fix_unvaried, bestfit_values, fix_parameters, parse_fixed,
                                profile, sample_mh)
from wstfast.theory.rsd import MultipoleProjection, RSDGrid, S1mProjection


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los"))
    parser.add_argument("--stats", nargs="+", choices=("pk", "s1m"), default=["pk", "s1m"])
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--s1-min-scale", type=float, default=25.0, help="smallest sigma_j [Mpc/h] of S1m")
    parser.add_argument("--s1-ells", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--kmin", type=float, default=0.0)
    parser.add_argument("--kmax", type=float, default=0.12)
    parser.add_argument("--rebin", type=int, default=2, help="P(k) bins of rebin x k_f")
    parser.add_argument("--vary", nargs="+", default=["omega_cdm", "logA"], choices=sorted(QUIJOTE_COSMOLOGY))
    parser.add_argument("--emulator", type=Path, required=True, help="RSDBasis emulator (train_emulator.py --stat rsd)")
    parser.add_argument("--volume", type=float, default=None,
                        help="errors of a survey of this volume in (Gpc/h)^3: single-box covariance / (V / V_box)")
    parser.add_argument("--covariance-of-mean", action="store_true", help="errors of the realization mean")
    parser.add_argument("--fix", nargs="+", default=None, metavar="NAME=VALUE")
    parser.add_argument("--method", choices=("profile", "sample"), default="sample")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--proposal-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_basis(path: Path, vary):
    from desilike.emulators import Emulator

    settings = json.loads(path.with_suffix(".json").read_text())
    if settings.get("stat") != "rsd":
        raise ValueError(f"{path} is not a redshift-space (RSDBasis) emulator")
    basis = Emulator.read(str(path)).to_calculator()
    _bound_to_emulator(basis, _fix_unvaried(basis, settings, vary, path))
    return basis, settings


def main():
    args = parse_args()
    setup_logging()
    if args.volume is not None and args.covariance_of_mean:
        raise SystemExit("--volume and --covariance-of-mean are exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    first = sorted((args.data_dir / "rsd").glob("wst_r*.npz"))[0]
    config = load_measurement(first, q=args.q)["config"]
    basis, settings = load_basis(args.emulator, args.vary)
    grid = RSDGrid(kmax=settings["kmax"])
    emulated = settings["coefficients"]

    theories, vectors, labels = [], [], []
    files = None
    if "s1m" in args.stats:
        coefficients = [c for c in select_coefficients(config, s1_min_scale=args.s1_min_scale, s1_ells=args.s1_ells)
                        if c.kind == "S1"]
        missing = [c.label for c in coefficients if c.label not in emulated]
        if missing:
            raise ValueError(f"the emulator has no cumulants for {missing}")
        s1m = load_s1m_dataset(args.data_dir, "rsd", coefficients, q=args.q)
        files = s1m.files
        theories.append(S1mTheory(S1mProjection(config, coefficients, grid), coefficients, q=args.q, basis=basis,
                                  shotnoise=s1m.shotnoise,
                                  cumulant_index=[emulated.index(c.label) for c in coefficients]))
        vectors.append(s1m.vectors)
        labels += [f"S1m_j{c.j}_l{c.ell}_m{m}" for c in coefficients for m in range(c.ell + 1)]
        meta = s1m.metadata
    if "pk" in args.stats:
        power = load_power_dataset(args.data_dir, "rsd", kmin=args.kmin, kmax=args.kmax, rebin=args.rebin, files=files,
                                   ells=(0, 2, 4))
        files = power.files
        theories.append(RSDPowerTheory(MultipoleProjection(power.edges, grid), basis=basis, shotnoise=power.shotnoise))
        vectors.append(power.vectors)
        labels += [f"P{ell}_k{k:.4f}" for ell in (0, 2, 4) for k in power.k]
        meta = power.metadata
    vectors = np.concatenate(vectors, axis=1)
    nreal, ndata = vectors.shape
    covariance = sample_covariance(vectors, of_mean=args.covariance_of_mean)
    if args.volume is not None:
        covariance = covariance * (meta["boxsize"] / 1000.0) ** 3 / args.volume
    print(f"{ndata} data points ({', '.join(args.stats)}), {nreal} realizations")

    theory = theories[0] if len(theories) == 1 else JointTheory(theories)
    likelihood = WSTLikelihood(theory, vectors.mean(axis=0), covariance)
    fix_parameters(likelihood, parse_fixed(args.fix))
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

    best = bestfit_values(profiles)
    varied = get_params(likelihood).select(varied=True, derived=False).names()
    build(likelihood)({name: value for name, value in best.items() if name in varied})
    residual = np.asarray(likelihood.flatdata) - np.asarray(likelihood.flattheory)
    chi2 = float(residual @ np.linalg.solve(covariance, residual))
    summary = dict(bestfit=best, chi2=chi2, ndata=int(ndata), nvaried=len(varied), nrealizations=int(nreal),
                   stats=args.stats, vary=args.vary, volume=args.volume, covariance_of_mean=args.covariance_of_mean,
                   kmax=args.kmax, s1_min_scale=args.s1_min_scale, labels=labels, emulator=str(args.emulator))
    if samples is not None:
        summary["posterior"] = {name: dict(mean=float(np.asarray(samples.mean(name))),
                                           std=float(np.asarray(samples.std(name))))
                                for name in varied if name in samples}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"chi2 = {chi2:.1f} for {ndata} data points and {len(varied)} parameters")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
