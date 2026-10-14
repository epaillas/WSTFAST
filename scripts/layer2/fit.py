#!/usr/bin/env python
"""Fits of the real-space matter WST with and without the non-Gaussian second-layer model.

The data are the mean of the Quijote realizations; the covariance is the analytic one (scripts/analytic_covariance.py,
covariance of ln data vector, sub-selected by label) scaled to a volume V, or with --cov mock the sample covariance
of the Quijote boxes. n_s, h and omega_b are fixed at the
fiducial (the emulator varies omega_cdm and logA; with --vary ... h, the 5-parameter emulator with n_s and omega_b
fixed). Second-layer models (--model):

    gauss       the paper's model: Gaussian second layer
    tree        S21 x (1 + d0 + d2 s_lin^2 / s_lin,fid^2): Gaussian-field and tree-level terms of
                scripts/layer2/template.py, no free parameter
    pade_l      tree-level term suppressed by 1 / (1 + c_l n s_lin^2(sigma_j1)), one free c_l per l
    per_l, global, per_field
                tree-level term times a free amplitude (per l, one in total, per first-layer field)

For the template models the Monte Carlo error of the template enters the covariance as a theory error.
Writes profiles, NUTS chains, flat samples (samples.npz) and a summary (summary.json) to
outputs/layer2/fits/<vector>_<model>_V<volume>/. Example:

    python scripts/layer2/fit.py --vector baseline --model pade_l --volume 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/paper"))

import common  # noqa: E402
import wstfast.theory  # noqa: E402,F401
from desilike import get_params  # noqa: E402
from desilike.base import Posterior  # noqa: E402
from wstfast.calculators import WSTLikelihood, WSTTheory  # noqa: E402
from wstfast.config import QUIJOTE_COSMOLOGY  # noqa: E402
from wstfast.inference import bestfit_values, profile, sample_nuts  # noqa: E402

VECTORS = {"baseline": ("baseline", {}), "conservative": ("conservative", {}),
           "adjacent": ("baseline", dict(s21_min_ratio=2.0))}
OUT = ROOT / "outputs/layer2"


def coefficients(vector):
    name, overrides = VECTORS[vector]
    return common.coefficients(name, **overrides)


def analytic_covariance(path, ds, volume):
    data = np.load(path, allow_pickle=True)
    labels = [str(x) for x in data["labels"]]
    index = [labels.index(c.label) for c in ds.coefficients]
    lncov = np.asarray(data["lncov"])[np.ix_(index, index)]
    return lncov * np.outer(ds.mean, ds.mean) / volume


def load_template(path):
    return json.loads(Path(path).read_text())["coefficients"]


EMULATOR_H = ROOT / "outputs/emulators/wst_basis_taylor_superset_5p_full.h5"


def build(vector, model, volume, template, covariance_path, vary=common.COSMOLOGY, cov="analytic",
          template_error=True, marg=None, marg_width=1.0, calibration="all"):
    coeffs = coefficients(vector)
    ds = common.dataset(coeffs)
    emulator = common.EMULATOR if set(vary) <= set(common.COSMOLOGY) else EMULATOR_H
    basis, bounds = common.emulated_basis(ds, emulator, vary=tuple(vary))
    params = get_params(basis)
    for name, (low, high) in bounds.items():
        params[name].update(prior=dict(limits=[low, high]))
    kwargs = {} if model == "gauss" else dict(layer2_template=load_template(template), layer2_mode=model)
    theory = WSTTheory(coeffs, config=ds.config, basis=basis, **kwargs)
    if cov == "analytic":
        covariance = analytic_covariance(covariance_path, ds, volume)
    else:  # sample covariance of the single-box realizations (Hartlap-corrected by sample_covariance), per volume
        covariance = ds.covariance("sample") / volume
    if model != "gauss" and template_error:
        covariance = covariance + np.diag(template_variance(template, ds, model))
    if model != "gauss" and marg is not None:  # analytic marginalisation over the template amplitudes
        for vector in template_shapes(template, ds, marg, calibration):
            covariance = covariance + marg_width**2 * np.outer(vector, vector)
    return ds, WSTLikelihood(theory, ds.mean, covariance), covariance


def template_shapes(path, ds, marg, calibration="all"):
    """Data-vector shapes S21 d0 and S21 R_l d2 of the template pieces at the calibrated c_l (fiducial cosmology), all
    l together ('global') or one pair per l ('per_l'), as in fit_rsd.py --s21-layer2-marg."""
    template = json.loads(Path(path).read_text())
    suffix = "" if calibration == "all" else f"_{calibration}"
    c_l = {int(k): max(v, 0.0) for k, v in template[f"pade_c{suffix}"].items()}
    rows, groups = template["coefficients"], {}
    for i, c in enumerate(ds.coefficients):
        if c.kind != "S21":
            continue
        r = rows[c.label]
        d0, d2 = groups.setdefault(0 if marg == "global" else c.ell, (np.zeros(len(ds.coefficients)),
                                                                     np.zeros(len(ds.coefficients))))
        d0[i] = ds.mean[i] * r["gauss"]
        d2[i] = ds.mean[i] * r["tree"] / (1.0 + c_l[c.ell] * r["var_ref"])
    return [v for pair in groups.values() for v in pair]


def template_variance(path, ds, model):
    """Theory-error variance of the data vector from the Monte Carlo error of the template (independent of the
    volume): (S21 dE)^2 with dE^2 = dd0^2 + (R dd2)^2, R from the simulation calibration of c_l ('pade_l') or 1."""
    template = json.loads(Path(path).read_text())
    rows, calibration = template["coefficients"], template.get("pade_c", {})
    out = np.zeros(len(ds.coefficients))
    for i, c in enumerate(ds.coefficients):
        if c.kind != "S21":
            continue
        r = rows[c.label]
        renorm = 1.0 / (1.0 + max(calibration.get(str(c.ell), 0.0), 0.0) * r["var_ref"]) if model == "pade_l" else 1.0
        out[i] = (ds.mean[i] * np.hypot(r["gauss_err"], renorm * r["tree_err"])) ** 2
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vector", choices=sorted(VECTORS), default="baseline")
    parser.add_argument("--model", choices=("gauss", "tree", "pade_l", "per_l", "global", "per_field"), default="gauss")
    parser.add_argument("--volume", type=float, default=1.0, help="(Gpc/h)^3")
    parser.add_argument("--template", type=Path, default=OUT / "template.json")
    parser.add_argument("--covariance", type=Path, default=OUT / "analytic_q0.8_adjacent.npz")
    parser.add_argument("--fix-calibrated", action="store_true",
                        help="pade_l: fix c_l at the simulation calibration stored in the template")
    parser.add_argument("--calibrated-prior", type=float, default=None, metavar="FRACTION",
                        help="pade_l: Gaussian prior on c_l centred on the calibration, of width FRACTION x c_l "
                             "(at least its calibration error), truncated at 0")
    parser.add_argument("--calibration", choices=("all", "adjacent", "nonadjacent"), default="all",
                        help="pairs the c_l calibration used (held-out tests)")
    parser.add_argument("--cov", choices=("analytic", "mock"), default="analytic",
                        help="analytic covariance, or the sample covariance of the Quijote boxes (Hartlap-corrected)")
    parser.add_argument("--vary", nargs="+", default=list(common.COSMOLOGY),
                        help="cosmological parameters (h needs the 5-parameter emulator; n_s, omega_b stay fixed)")
    parser.add_argument("--marg", choices=("global", "per_l"), default=None,
                        help="marginalise analytically over the amplitudes of the template pieces d0 and R d2")
    parser.add_argument("--marg-width", type=float, default=1.0, help="prior width of the amplitudes (fraction)")
    parser.add_argument("--no-template-error", action="store_true",
                        help="leave the Monte Carlo error of the template out of the covariance")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extra = "".join(f"_{n}" for n in args.vary if n not in common.COSMOLOGY)
    extra += "" if args.calibration == "all" else f"_from{args.calibration}"
    extra += "" if args.marg is None else f"_marg{args.marg}"
    extra += "_mock" if args.cov == "mock" else ""
    extra += "_nomcerr" if args.no_template_error else ""
    mode = "_cal" if args.fix_calibrated else (f"_prior{args.calibrated_prior:g}" if args.calibrated_prior else "")
    tag = f"{args.vector}_{args.model}{mode}{extra}_V{args.volume:g}"
    output = OUT / "fits" / tag
    output.mkdir(parents=True, exist_ok=True)
    ds, lik, covariance = build(args.vector, args.model, args.volume, args.template, args.covariance, args.vary,
                                args.cov, not args.no_template_error, args.marg, args.marg_width, args.calibration)
    truth = {n: QUIJOTE_COSMOLOGY[n] for n in args.vary}
    if args.fix_calibrated or args.calibrated_prior:
        suffix = "" if args.calibration == "all" else f"_{args.calibration}"
        template = json.loads(args.template.read_text())
        calibration, errors = template[f"pade_c{suffix}"], template[f"pade_c_err{suffix}"]
        params = get_params(lik)
        for ell, c in calibration.items():
            name = f"ng2c_l{ell}"
            if name not in params:
                continue
            if args.fix_calibrated:
                params[name].update(value=max(c, 0.0), fixed=True)
            else:
                width = max(args.calibrated_prior * abs(c), errors[ell], 0.05)
                params[name].update(value=max(c, 0.0), prior=dict(dist="norm", loc=c, scale=width, limits=[0.0, 500.0]),
                                    ref=dict(dist="norm", loc=max(c, 0.0), scale=0.1 * width))
    posterior = Posterior(lik)
    profiles = profile(posterior, output / "profiles.h5", seed=args.seed)
    best = bestfit_values(profiles)
    from desilike import build as dbuild
    graph = dbuild(lik)
    varied = get_params(lik).select(varied=True, derived=False).names()
    graph({name: best[name] for name in varied})
    residual = np.asarray(lik.flatdata) - np.asarray(lik.flattheory)
    chi2 = float(residual @ np.asarray(lik.precision) @ residual)
    summary = dict(vector=args.vector, model=args.model, volume=args.volume, ndata=int(residual.size),
                   nvaried=len(varied), chi2=chi2, bestfit=best, truth=truth)
    print(f"{tag}: chi2 = {chi2:.2f} for {residual.size} data, {len(varied)} parameters; "
          + ", ".join(f"{n} = {best[n]:.4f}" for n in args.vary), flush=True)
    if not args.profile_only:
        samples = sample_nuts(posterior, profiles, output / "chains", chains=4, seed=args.seed, gelman_rubin=1.02,
                              ess=1000.0)
        weights = np.asarray(samples.weight, dtype="f8").reshape(-1)
        flat = {str(n): np.asarray(v, dtype="f8").reshape(-1) for n, v in samples.to_dict().items()}
        flat = {n: v for n, v in flat.items() if v.size == weights.size}
        np.savez(output / "samples.npz", weight=weights, **flat)
        stats = {}
        for n, v in flat.items():
            mean = float(np.average(v, weights=weights))
            stats[n] = dict(mean=mean, std=float(np.sqrt(np.average((v - mean) ** 2, weights=weights))))
        summary["posterior"] = stats
        summary["bias_sigma"] = {n: (stats[n]["mean"] - truth[n]) / stats[n]["std"] for n in args.vary}
        print(f"{tag}: " + ", ".join(f"{n} = {stats[n]['mean']:.4f} +- {stats[n]['std']:.4f} "
                                     f"({summary['bias_sigma'][n]:+.2f} sigma)" for n in args.vary), flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
