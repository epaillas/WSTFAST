#!/usr/bin/env python
"""Fisher forecast of the redshift-space statistics (P_l, S1m, S21m) of fit_rsd.py: where does the information go?

The model and the data vectors are those of ``fit_rsd.py`` (``build_problem``), on the widest selection the emulator
supports. The derivatives of the joint model are taken once, by central finite differences at a fiducial point (the
Quijote cosmology and nuisance values from earlier fits, ``--fiducial``), and the covariance is the sample covariance
of the realizations (Hartlap-corrected per subset) for a survey volume ``--volume``. Each case is then a subset of
rows (data) and a treatment of the columns (nuisance parameters):

- ``fixed``: nuisances at their fiducial values (the information available in principle);
- ``marg``: nuisances marginalised, with the Gaussian priors of ``BiasParameters`` (flat priors enter as a Gaussian
  of the same variance, so that unconstrained directions stay finite);
- ``shared``: as ``marg``, but with one set of counterterms for P_l and S1m (c0 = c0_pk, ...).

Example (halos):

    python scripts/fisher_rsd.py --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los \\
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir.h5 --volume 25 --output outputs/fisher/halos_rsd_V25.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from desilike import build, get_params
from wstfast.calculators import JointTheory
from wstfast.config import QUIJOTE_COSMOLOGY

COSMOLOGY = ("omega_cdm", "logA", "n_s", "h")
STEPS = {"omega_cdm": 0.002, "logA": 0.01, "n_s": 0.005, "h": 0.005, "b1": 0.02, "b2": 0.05, "bG2": 0.05,
         "bGamma3": 0.05, "alpha0": 0.02, "alpha2": 1.0, "a_ng": 0.01}
DEFAULT_STEP = {"c": 1.0, "noise": 0.01}


def load_fit_rsd():
    spec = importlib.util.spec_from_file_location("fit_rsd", Path(__file__).with_name("fit_rsd.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--emulator", type=Path, required=True)
    parser.add_argument("--volume", type=float, default=25.0, help="survey volume in (Gpc/h)^3")
    parser.add_argument("--kmax", type=float, nargs="+", default=[0.12, 0.2], help="P_l kmax values to compare")
    parser.add_argument("--s1-min-scale", type=float, nargs="+", default=[25.0, 17.6],
                        help="smallest S1m sigma values to compare [Mpc/h]")
    parser.add_argument("--qs", type=float, nargs="+", default=[0.5, 0.8, 1.0], help="S1m exponents (0.8 first-class)")
    parser.add_argument("--fiducial", type=Path, nargs="*",
                        default=[Path("outputs/inference/rsd/halos_bias_only_V60_s1m/summary.json"),
                                 Path("outputs/inference/rsd/halos_bias_only_V60_pk/summary.json")],
                        help="fit summaries whose best fits set the nuisance values (later files take precedence)")
    parser.add_argument("--output", type=Path, default=Path("outputs/fisher/rsd.json"))
    return parser.parse_args()


def step_of(name):
    if name in STEPS:
        return STEPS[name]
    return DEFAULT_STEP["noise"] if name.startswith("noise") else DEFAULT_STEP["c"]


def prior_sigma(param):
    """Gaussian sigma of a parameter's prior; a flat prior of width w counts as sigma = w / sqrt(12)."""
    prior = param.prior
    if prior.dist == "norm":
        return float(prior.attrs["scale"])
    return float(prior.std()) if prior.is_proper() else np.inf


def main():
    args = parse_args()
    fr = load_fit_rsd()
    common = ["--data-dir", str(args.data_dir), "--emulator", str(args.emulator), "--output-dir", "unused",
              "--vary", *COSMOLOGY, "--rebin", "2"]
    # Widest selection: P_l to max(kmax), S1m to min(s1-min-scale) at every q, S21m with fit_rsd's default cuts.
    main_args = fr.parse_args(common + ["--stats", "pk", "s1m", "s21m", "--kmax", str(max(args.kmax)),
                                        "--s1-min-scale", str(min(args.s1_min_scale)), "--q", "0.8"])
    problem = fr.build_problem(main_args)
    theories, vectors, labels = list(problem["theories"]), [problem["vectors"]], list(problem["labels"])
    for q in args.qs:
        if q == 0.8:
            continue
        extra = fr.build_problem(fr.parse_args(common + ["--stats", "s1m", "--q", str(q),
                                                         "--s1-min-scale", str(min(args.s1_min_scale))]),
                                 basis=problem["basis"], settings=problem["settings"])
        theories += extra["theories"]
        vectors.append(extra["vectors"])
        labels += [f"q{q}:{label}" for label in extra["labels"]]
    vectors = np.concatenate(vectors, axis=1)
    theory = JointTheory(theories)
    first = sorted((args.data_dir / "rsd").glob("wst_r*.npz"))[0]
    config = fr.load_measurement(first, q=0.8)["config"]

    # Fiducial point and Jacobian.
    params = get_params(theory).select(varied=True, derived=False)
    names = params.names()
    fiducial = {name: float(params[name].value) for name in names}
    fiducial.update({name: QUIJOTE_COSMOLOGY[name] for name in COSMOLOGY})
    for path in args.fiducial:
        if path.exists():
            best = json.loads(path.read_text())["bestfit"]
            fiducial.update({name: float(value) for name, value in best.items() if name in names
                             and name not in COSMOLOGY})
    fiducial.update({name: 0.0 for name in names if name.startswith("noise")})
    model = build(theory)
    print(f"{len(labels)} data points, {len(vectors)} realizations, {len(names)} parameters")
    jacobian = np.zeros((len(names), len(labels)))
    for i, name in enumerate(names):
        h = step_of(name)
        up = np.asarray(model({**fiducial, name: fiducial[name] + h}))
        down = np.asarray(model({**fiducial, name: fiducial[name] - h}))
        jacobian[i] = (up - down) / (2 * h)
    raw = np.cov(vectors, rowvar=False, ddof=1) / args.volume  # boxes of 1 (Gpc/h)^3
    nreal = len(vectors)
    # Cosmology: no prior (the emulator box is not a prior of the forecast).
    priors = np.array([np.inf if name in COSMOLOGY else prior_sigma(params[name]) for name in names])

    # Row selections.
    def kind(label):
        return label.split(":")[-1].split("_")[0]

    def q_of(label):
        return float(label.split(":")[0][1:]) if ":" in label else 0.8

    def sigma_of(label):
        j = int(label.split(":")[-1].split("_")[1][1:])
        return config.sigma(j)

    def select(kmax=None, s1=None, qs=(0.8,), s21=False):
        rows = []
        for n, label in enumerate(labels):
            k = kind(label)
            if k.startswith("P") and kmax is not None and float(label.split("_k")[1]) <= kmax:
                rows.append(n)
            elif k == "S1m" and s1 is not None and q_of(label) in qs and sigma_of(label) >= s1 - 0.05:
                rows.append(n)
            elif k == "S21m" and s21:
                rows.append(n)
        return np.array(rows)

    shared = {"c0": "c0_pk", "c2": "c2_pk", "c4": "c4_pk"}

    def errors(rows, treatment):
        ndata = len(rows)
        hartlap = (nreal - ndata - 2.0) / (nreal - 1.0)
        precision = hartlap * np.linalg.inv(raw[np.ix_(rows, rows)])
        jac = jacobian[:, rows]
        cols = [i for i, name in enumerate(names) if name in COSMOLOGY or np.any(jac[i] != 0)]
        cosmo = [cols.index(names.index(name)) for name in COSMOLOGY]
        if treatment == "fixed":
            sub = jac[[names.index(name) for name in COSMOLOGY]]
            return np.sqrt(np.diag(np.linalg.inv(sub @ precision @ sub.T)))
        rows_j = jac[cols]
        prior = priors[cols].copy()
        if treatment == "shared":  # merge each S1m counterterm into its P_l twin
            keep, merged = list(range(len(cols))), rows_j.copy()
            for a, b in shared.items():
                if a in names and b in names and names.index(a) in cols and names.index(b) in cols:
                    ia, ib = cols.index(names.index(a)), cols.index(names.index(b))
                    merged[ib] += merged[ia]
                    keep.remove(ia)
            rows_j, prior = merged[keep], prior[keep]
            cosmo = [keep.index(c) for c in cosmo]
        fisher = rows_j @ precision @ rows_j.T + np.diag(1.0 / prior**2)
        return np.sqrt(np.diag(np.linalg.inv(fisher))[cosmo])

    kmin, kmaxes = min(args.kmax), sorted(args.kmax)
    s1_large, s1_small = max(args.s1_min_scale), min(args.s1_min_scale)
    cases = {f"P(k<={k})": dict(kmax=k) for k in kmaxes}
    cases.update({
        f"P(k<={kmin}) + S1m(sigma>={s1_large})": dict(kmax=kmin, s1=s1_large),
        f"P(k<={kmin}) + S1m(sigma>={s1_small})": dict(kmax=kmin, s1=s1_small),
        f"P(k<={kmin}) + S1m(sigma>={s1_large}, q={args.qs})": dict(kmax=kmin, s1=s1_large, qs=tuple(args.qs)),
        f"P(k<={kmin}) + S1m(sigma>={s1_large}) + S21m": dict(kmax=kmin, s1=s1_large, s21=True),
        f"P(k<={kmin}) + S1m(sigma>={s1_small}, q={args.qs}) + S21m": dict(kmax=kmin, s1=s1_small,
                                                                           qs=tuple(args.qs), s21=True),
        "S1m only (sigma>={}, q=0.8)".format(s1_large): dict(s1=s1_large),
        f"S1m only (sigma>={s1_small}, q={args.qs})": dict(s1=s1_small, qs=tuple(args.qs)),
    })
    if len(kmaxes) > 1:
        cases[f"P(k<={kmaxes[-1]}) + S1m(sigma>={s1_small}, q={args.qs}) + S21m"] = dict(
            kmax=kmaxes[-1], s1=s1_small, qs=tuple(args.qs), s21=True)
    results = {}
    for label, selection in cases.items():
        rows = select(**selection)
        results[label] = dict(ndata=int(len(rows)),
                              **{t: dict(zip(COSMOLOGY, errors(rows, t).tolist())) for t in ("fixed", "marg", "shared")})
    baseline = results[f"P(k<={kmin})"]
    width = max(len(label) for label in results) + 2
    for treatment in ("fixed", "marg", "shared"):
        print(f"\n{treatment}: sigma (ratio to P(k<={kmin}), same treatment), V = {args.volume} (Gpc/h)^3")
        print(f"{'case':{width}s}{'n':>5s}" + "".join(f"{name:>20s}" for name in COSMOLOGY))
        for label, result in results.items():
            cells = "".join(f"{result[treatment][name]:>11.4f} ({result[treatment][name] / baseline[treatment][name]:.2f})"
                            for name in COSMOLOGY)
            print(f"{label:{width}s}{result['ndata']:>5d}{cells}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(volume=args.volume, fiducial=fiducial, results=results,
                                           nrealizations=nreal), indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
