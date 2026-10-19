#!/usr/bin/env python
"""Fit the line-of-sight-resolved WST (S1m, S21m) and the P(k) multipoles of box or survey measurements with the
emulated model (``wstfast.inference.build_fit``; NUTS or Metropolis-Hastings with desilike).

The model is that of the emulator (``train_emulator.py --stat rsd``; its ``ModelSpec``: tracer, real or redshift
space, Fingers-of-God counterterm, AP order, cubic biases). The statistics share the cosmology and the biases; S1m
and P(k) keep their own counterterms (c0, c2, c4 and c0_pk, c2_pk, c4_pk), and S21m adds one noise amplitude per
first-layer field. The halo baseline of the paper (scripts/fit_halo_baseline.sh):

    python scripts/fit.py --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los \\
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic_ap3_base.h5 --ap \\
        --stats s1m s21m --s21-min-scale 25 --s21-white --cubic-biases relation \\
        --s21-layer2 outputs/layer2/template_rsd.json --s21-layer2-marg per_l \\
        --vary omega_cdm logA h --volume 60 --output-dir outputs/inference/rsd/baseline_V60

A real-space emulator (``--space real``) fits the measurements of <data-dir>/real (P(k): the monopole). Survey fits:
``--survey NGC SGC --window ...`` (``FitOptions``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from desilike import build, get_params, setup_logging
from desilike.base import Posterior
from wstfast.config import QUIJOTE_COSMOLOGY
from wstfast.data import chi_squared, hartlap_factor
from wstfast.inference import FitOptions, bestfit_values, build_fit, profile, sample_mh, sample_nuts

#: Command-line options that are not FitOptions (the sampling).
SAMPLING = ("method", "sampler", "warmup", "chains", "max_steps", "proposal_from", "start_from", "seed", "output_dir")


def parse_args(argv=None):
    defaults = FitOptions(emulator=Path())
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emulator", type=Path, required=True, help="emulated Basis (train_emulator.py --stat rsd)")
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--stats", nargs="+", choices=("pk", "s1m", "s21m", "band"), default=list(defaults.stats),
                        help="band: the m-summed first-order coefficients at q = 2 (the band powers of the wavelet "
                             "filters), on their own cuts --band-min-scale / --band-ells")
    parser.add_argument("--q", type=float, nargs="+", default=list(defaults.q),
                        help="S1m exponent(s): several values concatenate the S1m data vectors (shared nuisances, joint "
                             "covariance); S21m uses the first")
    parser.add_argument("--s1-min-scale", type=float, default=defaults.s1_min_scale, help="smallest sigma_j [Mpc/h] of S1m")
    parser.add_argument("--s1-ells", type=int, nargs="+", default=list(defaults.s1_ells))
    parser.add_argument("--s21-min-scale", type=float, default=defaults.s21_min_scale,
                        help="smallest sigma_j1 [Mpc/h] of S21m")
    parser.add_argument("--s21-min-scale2", type=float, default=defaults.s21_min_scale2,
                        help="smallest sigma_j2 [Mpc/h] of S21m")
    parser.add_argument("--s21-min-ratio", type=float, default=defaults.s21_min_ratio,
                        help="smallest sigma_j2 / sigma_j1 of S21m")
    parser.add_argument("--band-min-scale", type=float, default=defaults.band_min_scale,
                        help="smallest sigma_j [Mpc/h] of the band powers (every band must lie within the emulator's kmax)")
    parser.add_argument("--band-max-scale", type=float, default=defaults.band_max_scale,
                        help="largest sigma_j of the band powers (keep them below --s1-min-scale when S1m is fitted too)")
    parser.add_argument("--band-ells", type=int, nargs="+", default=list(defaults.band_ells))
    parser.add_argument("--band-kmax", type=float, default=None,
                        help="keep the bands whose 84th-percentile wavenumber lies below this [h/Mpc]")
    parser.add_argument("--band-counterterm", choices=("shared", "separate"), default=defaults.band_counterterm,
                        help="counterterm of the band powers: the S1m c0 (the same spectrum) or its own c0_band")
    parser.add_argument("--band-cut", type=float, default=defaults.band_cut,
                        help="keep the principal components of the band powers with correlation eigenvalue above this "
                             "(the bands are numerically low-rank; 0: none)")
    parser.add_argument("--compress-m", action="store_true",
                        help="real space: average the |m| blocks of S1m and S21m within their equally distributed "
                             "classes (inference.m_compression); shorter data vector, smaller Hartlap and Percival factors")
    parser.add_argument("--kmin", type=float, default=defaults.kmin)
    parser.add_argument("--kmax", type=float, default=defaults.kmax)
    parser.add_argument("--rebin", type=int, default=defaults.rebin, help="P(k) bins of rebin x k_f")
    parser.add_argument("--vary", nargs="+", default=list(defaults.vary), choices=sorted(QUIJOTE_COSMOLOGY))
    parser.add_argument("--volume", type=float, default=None,
                        help="errors of a survey of this volume in (Gpc/h)^3: single-box covariance / (V / V_box)")
    parser.add_argument("--covariance-of-mean", action="store_true", help="errors of the realization mean")
    parser.add_argument("--survey", nargs="+", default=None, metavar="CAP",
                        help="survey fit: the caps (data in <data-dir>/<cap>/, e.g. NGC SGC; dr1_window.py --test "
                             "measure), one survey.SurveyWindow each (--window); shared parameters, the covariance from "
                             "the --covariance-data-dir boxes scaled to each cap's effective volume")
    parser.add_argument("--window", type=Path, nargs="+", default=None, help="survey: window file of each cap")
    parser.add_argument("--combine-caps", action="store_true",
                        help="survey: fit the caps combined as desi-clustering's GCcomb (calculators.combine_caps: the "
                             "FKP-weighted mean of the caps' vectors, the S21m ratios weighted by their first-layer "
                             "S1m) instead of concatenating them; the covariances are projected the same way")
    parser.add_argument("--covariance-file", type=Path, default=None,
                        help="survey: analytic covariance (analytic_covariance_rsd_wst.py --survey: labels, lncov of one "
                             "survey) instead of the --covariance-data-dir boxes; no Hartlap / Percival factors")
    parser.add_argument("--extra-covariance", type=Path, nargs="*", default=[],
                        help="npz files (labels, covariance in data units, or fractional) of modelling errors added "
                             "to the covariance as they are (not scaled with the volume), e.g. the window calibration "
                             "(dr1_window.py --test products)")
    parser.add_argument("--no-percival", action="store_true",
                        help="do not multiply the likelihood covariance by the Percival et al. (2014) factor m1 (the "
                             "posterior widening from the noise of the sample covariance; applied by default)")
    parser.add_argument("--covariance-data-dir", type=Path, default=None,
                        help="estimate the covariance from the realizations of this directory (e.g. the fiducial "
                             "ones for a derivative set with few boxes); the data are still those of --data-dir")
    parser.add_argument("--realizations-like", type=Path, default=None,
                        help="use only the realizations of --data-dir that also exist in this directory (paired seeds)")
    parser.add_argument("--exclude-like", type=Path, default=None,
                        help="leave out the realizations of --data-dir that exist in this directory (held-out seeds)")
    parser.add_argument("--fix", nargs="+", default=None, metavar="NAME=VALUE")
    parser.add_argument("--cumulant-points", type=int, default=defaults.cumulant_points,
                        help="Sobol points of the control-variate correction of the S1m cumulants (0: none)")
    parser.add_argument("--gaussian-prior", nargs="+", default=[], metavar="NAME=LOC,SCALE",
                        help="Gaussian prior on an emulated cosmological parameter, truncated to the emulator box, "
                             "e.g. n_s=0.9649,0.042 (Planck 2018 x 10, as the desi-clustering full-shape fits)")
    parser.add_argument("--s21-white", action="store_true",
                        help="free K-independent term W_j of the S21m modulus spectra per first-layer scale "
                             "(connected noise of the moduli)")
    parser.add_argument("--s21-layer2", type=Path, default=None,
                        help="template_rsd.json (scripts/layer2/template_rsd.py): non-Gaussian second layer, S21m x "
                             "(1 + d0 + R_l d2), R_l = 1 / (1 + c_l n s_lin^2)")
    parser.add_argument("--s21-layer2-c", choices=("halo", "matter"), default=defaults.s21_layer2_c,
                        help="c_l calibrated on the halo pairs outside the data vector, or transferred from matter")
    parser.add_argument("--s21-layer2-marg", choices=("global", "per_l"), default=None,
                        help="marginalise analytically over the amplitudes of the template pieces d0 and R d2 (one pair, or "
                             "one pair per l), Gaussian priors centred on the prediction (A = 1)")
    parser.add_argument("--s21-layer2-marg-width", type=float, default=defaults.s21_layer2_marg_width,
                        help="width of the amplitude priors, as a fraction of the predicted amplitude")
    parser.add_argument("--cubic-biases", choices=("zero", "relation"), default=None,
                        help="vary the cubic biases b3, bG3, bdG2 of the S1m tree trispectrum, with Gaussian priors "
                             "centred on zero or on the halo relations of bias.cubic_priors (then the parameters are the "
                             "offsets db3, dbG3, dbdG2); needs an emulator trained with --cubic-biases")
    parser.add_argument("--ap", action="store_true", help="Alcock-Paczynski distortions of the trial cosmology")
    parser.add_argument("--method", choices=("profile", "sample"), default="sample")
    parser.add_argument("--sampler", choices=("nuts", "mh"), default="nuts",
                        help="nuts: BlackJAX NUTS whitened by the exact Hessian at the best fit (minutes); mh: "
                             "Metropolis-Hastings with an adapted proposal (hours on the halo WST)")
    parser.add_argument("--warmup", type=int, default=1000, help="NUTS: step-size and mass-matrix adaptation steps")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="largest number of steps per chain (default 20000 for NUTS, 200000 for MH)")
    parser.add_argument("--proposal-from", type=Path, default=None)
    parser.add_argument("--start-from", type=Path, default=None, metavar="SUMMARY",
                        help="summary.json of another fit: its best fit is one more starting point of the profile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def fit_options(args) -> FitOptions:
    """The FitOptions of the command line."""
    values = {key: value for key, value in vars(args).items() if key not in SAMPLING}
    return FitOptions(**{key: tuple(value) if isinstance(value, list) else value for key, value in values.items()})


def main():
    args = parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = fit_options(args)
    fit = build_fit(options)
    likelihood, problem, covariance = fit["likelihood"], fit["problem"], fit["covariance"]
    posterior = Posterior(likelihood)
    start = json.loads(args.start_from.read_text())["bestfit"] if args.start_from is not None else None
    profiles = profile(posterior, args.output_dir / "profiles.h5", seed=args.seed, start=start)
    print(profiles.to_stats(tablefmt="pretty"))
    samples = None
    if args.method == "sample":
        proposal = None
        if args.proposal_from is not None:
            from desilike.samples import Samples

            proposal = Samples.read(str(args.proposal_from))
        if args.sampler == "nuts":
            samples = sample_nuts(posterior, profiles, args.output_dir / "chains", chains=args.chains,
                                  proposal=proposal, seed=args.seed, max_steps=args.max_steps or 20000,
                                  warmup=args.warmup)
        else:
            samples = sample_mh(posterior, profiles, args.output_dir / "chains", chains=args.chains,
                                proposal=proposal, seed=args.seed, max_steps=args.max_steps or 200000)
        samples.write(str(args.output_dir / "samples.h5"))
        print(samples.to_stats(tablefmt="pretty"))

    best = bestfit_values(profiles)
    varied = get_params(likelihood).select(varied=True, derived=False).names()
    build(likelihood)({name: value for name, value in best.items() if name in varied})
    residual = np.asarray(likelihood.flatdata) - np.asarray(likelihood.flattheory)
    chi2 = chi_squared(residual, covariance)
    ndata, meta = fit["ndata"], problem["meta"]
    options_out = {key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value)
                   for key, value in dataclasses.asdict(options).items() if key != "layer2_config"}
    summary = dict(options_out, bestfit=best, chi2=chi2, ndata=ndata, nvaried=len(varied),
                   nrealizations=fit["nreal"], labels=fit["labels"], model=problem["spec"].to_dict(),
                   tracer=problem["tracer"], sampler=args.sampler, hartlap_factor=hartlap_factor(fit["nsims"], ndata),
                   percival_m1=fit["m1"], nsims=fit["nsims"], shotnoise=meta["boxsize"] ** 3 / meta["nparticles"],
                   normalized_residual=(residual / np.sqrt(np.diag(covariance))).tolist())
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
