#!/usr/bin/env python
"""Joint fit of the WST coefficients and the matter P(k) of Quijote (desilike, MH by default).

Both statistics come from the same files, so the covariance is the joint sample covariance of the
concatenated data vectors [WST, P(k)], cross terms included. The two theories share the cosmological
parameters; each keeps its own nuisance parameters (cs2 and the noise amplitudes for the WST, cs2_pk for
P(k)). Example (4 parameters, omega_b fixed, errors of a 60 (Gpc/h)^3 survey):

    python scripts/fit_joint.py --vary omega_cdm logA n_s h --s21-min-scale 17.6 --kmax 0.1 --volume 60 \
        --wst-emulator outputs/emulators/wst_basis_taylor_superset_5p_wide.h5 \
        --pk-emulator outputs/emulators/pk_taylor_5p_wide.h5 --output-dir outputs/inference/joint/4p_V60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from desilike import setup_logging
from desilike.base import Posterior
from wstfast.config import QUIJOTE_COSMOLOGY, select_coefficients
from wstfast.data import load_dataset, load_measurement, load_power_dataset, sample_covariance
from wstfast.inference import (bestfit_values, build_joint_likelihood, fix_parameters, joint_vectors,
                                parse_fixed, plot_fit, plot_power_fit, profile, sample_mh)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256"))
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--s1-min-scale", type=float, default=25.0)
    parser.add_argument("--s21-min-scale", type=float, default=12.5)
    parser.add_argument("--s21-min-ratio", type=float, default=2.8)
    parser.add_argument("--kmin", type=float, default=0.0)
    parser.add_argument("--kmax", type=float, default=0.1)
    parser.add_argument("--rebin", type=int, default=2, help="P(k) bins of rebin x k_f")
    parser.add_argument("--cutoff", type=float, default=0.5, help="P(k) loop regulator [h/Mpc]")
    parser.add_argument("--vary", nargs="+", default=["omega_cdm", "logA"], choices=sorted(QUIJOTE_COSMOLOGY))
    parser.add_argument("--wst-emulator", type=Path, required=True)
    parser.add_argument("--pk-emulator", type=Path, required=True)
    parser.add_argument("--volume", type=float, default=None,
                        help="errors of a survey of this volume in (Gpc/h)^3: single-box covariance / (V / V_box)")
    parser.add_argument("--covariance-of-mean", action="store_true", help="errors of the realization mean")
    parser.add_argument("--layer2", action="store_true",
                        help="free second-layer non-Gaussianity constants C_l (needed for sigma_j1 = 12.5 Mpc/h)")
    parser.add_argument("--fix", nargs="+", default=None, metavar="NAME=VALUE",
                        help="fix parameters at these values (e.g. nuisances, for an upper bound on the information)")
    parser.add_argument("--method", choices=("profile", "sample"), default="sample")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--proposal-from", type=Path, default=None,
                        help="samples.h5 of an earlier run: use its chain covariance as the MH proposal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    if args.volume is not None and args.covariance_of_mean:
        raise SystemExit("--volume and --covariance-of-mean are exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    first = sorted((args.data_dir / "real").glob("wst_r*.npz"))[0]
    config = load_measurement(first, q=args.q)["config"]
    coefficients = select_coefficients(config, s1_min_scale=args.s1_min_scale, s21_min_scale=args.s21_min_scale,
                                       s21_min_ratio=args.s21_min_ratio)
    wst = load_dataset(args.data_dir, "real", coefficients, q=args.q)
    power = load_power_dataset(args.data_dir, "real", kmin=args.kmin, kmax=args.kmax, rebin=args.rebin, files=wst.files)
    vectors = joint_vectors(wst, power)
    nreal, ndata = vectors.shape
    covariance = sample_covariance(vectors, of_mean=args.covariance_of_mean)
    if args.volume is not None:
        covariance = covariance * (wst.metadata["boxsize"] / 1000.0) ** 3 / args.volume
    print(f"{len(coefficients)} WST coefficients + {power.k.size} P(k) bins up to k = {power.k[-1]:.3f} h/Mpc, "
          f"{nreal} realizations")

    likelihood = build_joint_likelihood(wst, power, covariance, vary=args.vary, wst_emulator=args.wst_emulator,
                                        power_emulator=args.pk_emulator, cutoff=args.cutoff, layer2=args.layer2)
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

    # Best fit: chi^2 of the joint vector and of each block, and the two fit plots.
    from desilike import build, get_params

    best = bestfit_values(profiles)
    varied = get_params(likelihood).select(varied=True, derived=False).names()
    build(likelihood)({name: value for name, value in best.items() if name in varied})
    data, theory = np.asarray(likelihood.flatdata), np.asarray(likelihood.flattheory)
    residual = data - theory
    nw = len(coefficients)
    blocks = {"joint": slice(None), "wst": slice(0, nw), "pk": slice(nw, None)}
    chi2 = {name: float(residual[s] @ np.linalg.solve(covariance[s, s], residual[s])) for name, s in blocks.items()}
    summary = dict(bestfit=best, chi2=chi2, ndata=int(ndata), nvaried=len(varied), nrealizations=int(nreal),
                   vary=args.vary, volume=args.volume, covariance_of_mean=args.covariance_of_mean,
                   coefficients=[c.label for c in coefficients], k=power.k.tolist(), kmax=args.kmax,
                   wst_emulator=str(args.wst_emulator), pk_emulator=str(args.pk_emulator))
    if samples is not None:
        summary["posterior"] = {name: dict(mean=float(np.asarray(samples.mean(name))),
                                           std=float(np.asarray(samples.std(name))))
                                for name in varied if name in samples}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    volume = (f"errors for V = {args.volume:g} (Gpc/h)$^3$" if args.volume else
              "errors of the mean" if args.covariance_of_mean else "errors of one box")
    part = lambda s: SimpleNamespace(flatdata=data[s], flattheory=theory[s])  # noqa: E731
    plot_fit(args.output_dir / "bestfit_wst.png", part(blocks["wst"]), wst, covariance[:nw, :nw], config,
             bestfit=best, title=f"Joint fit, WST part ({', '.join(args.vary)} varied), {volume}, q = {config.q}")
    plot_power_fit(args.output_dir / "bestfit_pk.png", part(blocks["pk"]), power, covariance[nw:, nw:],
                   title=f"Joint fit, P(k) part, kmax = {args.kmax}, {volume}")
    print("chi2: " + ", ".join(f"{name} = {value:.1f}" for name, value in chi2.items())
          + f" for {ndata} data points and {len(varied)} parameters")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
