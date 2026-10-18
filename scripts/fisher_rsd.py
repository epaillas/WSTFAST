#!/usr/bin/env python
"""Fisher forecast of the redshift-space statistics (P_l, S1m, S21m) of fit.py: where does the information go?

The model and the data vectors are those of ``fit.py`` (``inference.build_problem``), on the widest selection the emulator
supports. The derivatives of the joint model are taken once, by central finite differences at a fiducial point (the
Quijote cosmology and nuisance values from earlier fits, ``--fiducial``), and the covariance is the sample covariance
of the realizations (Hartlap-corrected per subset) for a survey volume ``--volume``. Each case is then a subset of
rows (data) and a treatment of the columns (nuisance parameters):

- ``fixed``: nuisances at their fiducial values (the information available in principle);
- ``marg``: nuisances marginalised, with the Gaussian priors of ``NuisanceParameters`` (flat priors enter as a Gaussian
  of the same variance, so that unconstrained directions stay finite);
- ``shared``: as ``marg``, but with one set of counterterms for P_l and S1m (c0 = c0_pk, ...).

With ``--window-ratios`` (one npz per cap, from the survey window tests: per-mock ratios measured / window model of
the S1m and S21m entries, and the cap's FKP effective volume), it instead sizes the calibration of the S-level window
correction delta_b (docs/survey_plan.md section 8): the model and covariance are those of the fit with ``--fit-args``
(e.g. the halo baseline), the data covariance is that of the mean of ``--nmean`` surveys of the caps' effective
volumes, and the error of delta_b calibrated on N mocks, Cov(ratio) / N per cap, is propagated to the marginalised
parameters. It prints the rms parameter shift in units of the parameter errors for each N of ``--ncal``.

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
from wstfast import inference
from wstfast.calculators import JointTheory
from wstfast.config import QUIJOTE_COSMOLOGY
from wstfast.data import load_measurement

COSMOLOGY = ("omega_cdm", "logA", "n_s", "h")
STEPS = {"omega_cdm": 0.002, "logA": 0.01, "n_s": 0.005, "h": 0.005, "b1": 0.02, "b2": 0.05, "bG2": 0.05,
         "bGamma3": 0.05, "alpha0": 0.02, "alpha2": 1.0, "a_ng": 0.01}
DEFAULT_STEP = {"c": 1.0, "noise": 0.01}


def load_fit():
    spec = importlib.util.spec_from_file_location("fit", Path(__file__).with_name("fit.py"))
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
    parser.add_argument("--fit-args", default="", help="window mode: further fit.py arguments (the fit to size)")
    parser.add_argument("--window-ratios", type=Path, nargs="*", default=None,
                        help="window mode: npz per cap (s1m (nmock, J+1, L+1, L+1), s21m (nmock, nentry), s21m_keys, "
                             "sigma, veff) of measured / window-model ratios of fiducial survey mocks")
    parser.add_argument("--nmean", type=int, default=25, help="window mode: surveys averaged in the data")
    parser.add_argument("--ncal", type=int, nargs="+", default=[100, 300, 1000, 3000, 10000],
                        help="window mode: numbers of calibration mocks to size")
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


def window_calibration(args, fr):
    """Rms shift of the marginalised parameters from the calibration error of delta_b, per number of mocks."""
    import shlex

    fit = fr.fit_options(fr.parse_args(["--data-dir", str(args.data_dir), "--emulator", str(args.emulator), "--output-dir", "unused"]
                        + shlex.split(args.fit_args)))
    problem = inference.build_problem(fit)
    theory = problem["theories"][0] if len(problem["theories"]) == 1 else JointTheory(problem["theories"])
    labels, vectors, meta = problem["labels"], problem["vectors"], problem["meta"]
    config = load_measurement(sorted((args.data_dir / "rsd").glob("wst_r*.npz"))[0], q=0.8)["config"]
    params = get_params(theory).select(varied=True, derived=False)
    names = params.names()
    fiducial = {name: float(params[name].value) for name in names}
    for path in args.fiducial:
        if path.exists():
            best = json.loads(path.read_text())["bestfit"]
            fiducial.update({name: float(value) for name, value in best.items() if name in names})
    model = build(theory)
    mean = np.asarray(model(fiducial))
    jacobian = np.array([(np.asarray(model({**fiducial, n: fiducial[n] + step_of(n)}))
                          - np.asarray(model({**fiducial, n: fiducial[n] - step_of(n)}))) / (2 * step_of(n))
                         for n in names])
    box = np.cov(vectors, rowvar=False, ddof=1) * (meta["boxsize"] / 1000.0) ** 3  # one box of 1 (Gpc/h)^3
    common = np.zeros_like(box)  # volume-independent theory error, the same in every cap
    if getattr(fit, "s21_layer2", None) is not None:
        common += np.diag(inference.layer2_variance(fit, labels, vectors.mean(axis=0)))
        if fit.s21_layer2_marg is not None:
            templates = list(inference.layer2_templates(fit, labels, vectors.mean(axis=0)))
        else:
            templates = []
    else:
        templates = []

    def survey_index(label):
        """(kind, index) of a fit label in the survey ratio arrays."""
        parts = label.split("_")
        if parts[0] == "S1m":
            j, ell, m = int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:])
            jj = int(round(np.log(config.sigma(j) / 25.0) / np.log(np.sqrt(2.0))))
            return "s1m", (jj, ell, m)
        ell, m1, m2 = int(parts[3][1:]), int(parts[4][1:]), int(parts[5][1:])
        return "s21m", (ell, m1, m2)

    caps = []
    for path in args.window_ratios:
        f = np.load(path)
        keys = [tuple(k) for k in f["s21m_keys"]]
        columns, blocks = [], []
        for label in labels:
            kind, key = survey_index(label)
            column = f["s1m"][(slice(None),) + key] if kind == "s1m" else f["s21m"][:, keys.index(key)]
            columns.append(column / column.mean())
            blocks.append(kind)
        cov = np.zeros((len(labels), len(labels)))
        for kind in ("s1m", "s21m"):  # one mock set per statistic: no cross-covariance between the blocks
            rows = np.flatnonzero([b == kind for b in blocks])
            if len(rows):
                sub = np.array([columns[i] for i in rows]).T  # (nmock, nrow)
                cov[np.ix_(rows, rows)] = np.cov(sub[np.all(np.isfinite(sub), axis=1)], rowvar=False, ddof=1)
        caps.append(dict(name=path.stem, veff=float(f["veff"]), ratio_cov=cov * np.outer(mean, mean)))
    n = len(labels)
    data_cov = np.zeros((n * len(caps), n * len(caps)))
    for i, cap in enumerate(caps):
        data_cov[i * n:(i + 1) * n, i * n:(i + 1) * n] = box / (cap["veff"] * args.nmean)
    data_cov += np.kron(np.ones((len(caps), len(caps))), common)
    for vector in templates:
        stacked = np.tile(vector, len(caps))
        data_cov += fit.s21_layer2_marg_width**2 * np.outer(stacked, stacked)
    jac = np.tile(jacobian, (1, len(caps)))
    priors = np.array([np.inf if name in COSMOLOGY else prior_sigma(params[name]) for name in names])
    precision = np.linalg.inv(data_cov)
    fisher = jac @ precision @ jac.T + np.diag(1.0 / priors**2)
    inverse = np.linalg.inv(fisher)
    gain = inverse @ jac @ precision  # parameters per unit data shift (nuisances marginalised)
    cosmo = [names.index(name) for name in COSMOLOGY if name in names]
    sigma = np.sqrt(np.diag(inverse))[cosmo]
    calib = np.zeros_like(data_cov)
    for i, cap in enumerate(caps):
        calib[i * n:(i + 1) * n, i * n:(i + 1) * n] = cap["ratio_cov"]
    shift1 = np.sqrt(np.diag(gain @ calib @ gain.T))[cosmo]  # one calibration mock
    print(f"{n} data points per cap, caps {[c['name'] for c in caps]} (V_eff {[round(c['veff'], 3) for c in caps]}),"
          f" data = mean of {args.nmean}; {len(names)} parameters")
    print("sigma(param), data:", dict(zip([names[i] for i in cosmo], np.round(sigma, 5).tolist())))
    print(f"{'N_cal':>8s}" + "".join(f"{names[i]:>16s}" for i in cosmo) + "   (rms shift / sigma)")
    out = {}
    for ncal in args.ncal:
        ratio = shift1 / np.sqrt(ncal) / sigma
        out[ncal] = dict(zip([names[i] for i in cosmo], ratio.tolist()))
        print(f"{ncal:>8d}" + "".join(f"{r:>16.3f}" for r in ratio))
    need = {names[i]: float((shift1[k] / sigma[k] / 0.1) ** 2) for k, i in enumerate(cosmo)}
    print("N_cal for an rms shift of 0.1 sigma:", {k: int(np.ceil(v)) for k, v in need.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(fit_args=args.fit_args, nmean=args.nmean, fiducial=fiducial,
                                           sigma=dict(zip([names[i] for i in cosmo], sigma.tolist())),
                                           shift_over_sigma=out, ncal_for_0p1_sigma=need,
                                           caps={c["name"]: c["veff"] for c in caps}), indent=2))
    print(f"wrote {args.output}")


def main():
    args = parse_args()
    fr = load_fit()
    if args.window_ratios:
        return window_calibration(args, fr)
    common = ["--data-dir", str(args.data_dir), "--emulator", str(args.emulator), "--output-dir", "unused",
              "--vary", *COSMOLOGY, "--rebin", "2"]
    # Widest selection: P_l to max(kmax), S1m to min(s1-min-scale) at every q, S21m with fit.py's default cuts.
    main_args = fr.fit_options(fr.parse_args(common + ["--stats", "pk", "s1m", "s21m", "--kmax", str(max(args.kmax)),
                                        "--s1-min-scale", str(min(args.s1_min_scale)), "--q", "0.8"]))
    problem = inference.build_problem(main_args)
    theories, vectors, labels = list(problem["theories"]), [problem["vectors"]], list(problem["labels"])
    for q in args.qs:
        if q == 0.8:
            continue
        extra = inference.build_problem(fr.fit_options(fr.parse_args(common + ["--stats", "s1m", "--q", str(q),
                                                         "--s1-min-scale", str(min(args.s1_min_scale))])),
                                 basis=problem["basis"], settings=problem["settings"])
        theories += extra["theories"]
        vectors.append(extra["vectors"])
        labels += [f"q{q}:{label}" for label in extra["labels"]]
    vectors = np.concatenate(vectors, axis=1)
    theory = JointTheory(theories)
    first = sorted((args.data_dir / "rsd").glob("wst_r*.npz"))[0]
    config = load_measurement(first, q=0.8)["config"]

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
