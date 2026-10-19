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

With ``--neutrino M`` (fiducial sum of neutrino masses [eV], one massive species), it instead forecasts sigma(M_nu) for
the halo baseline (scripts/fit_halo_baseline.sh without AP; ``--fit-args`` adds or overrides fit.py arguments) plus
P_l, with the exact (CLASS) basis of ``--emulator``'s settings: no emulator varies m_ncdm. omega_cdm, logA, n_s, h,
omega_b and m_ncdm are varied, the nuisances marginalised with their priors; the layer-2 template uncertainty enters
the covariance as in the fit. Two prior sets: LSS only (BBN omega_b) and Planck-like (logA, n_s, omega_b). The
Jacobian (6 x 2 exact basis calls, ~4 min) is cached next to ``--output``:

    python scripts/fisher_rsd.py --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los \\
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic.h5 --neutrino 0.06 --volume 25 \\
        --output outputs/fisher/neutrino_V25.json

With ``--layer2-info``, it instead asks how much the halo second layer carries when its non-Gaussian part (the
layer-2 template) is modelled rather than marginalised: each case of ``LAYER2_CASES`` is a fit.py vector (``build_fit``:
its covariance with Hartlap and Percival, the template's Monte Carlo error and, with ``--s21-layer2-marg``, the
analytically marginalised template amplitudes), the Jacobian is taken by central differences of the emulated model at
the Quijote cosmology and the nuisances of ``--fiducial``, the nuisances are marginalised with their Gaussian priors
(flat ones add nothing), for vary = (omega_cdm, logA, h) and with n_s. The S21m-alone cases are also split by l and by
sigma_2, and the bands and P_l are also given with b2 and bG2 known (the ceiling of what breaking the bias degeneracies
could add). ``--fit-args`` adds fit.py arguments to every case:

    python scripts/fisher_rsd.py --layer2-info --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los \
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic_ap3_bands12.h5 --volume 60 \
        --fiducial outputs/inference/rsd/q2/pk02_V60/summary.json outputs/inference/rsd/q2/q2_12k18_s21_V60/summary.json \
        --output outputs/fisher/layer2_info_V60.json

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
STEPS = {"omega_cdm": 0.002, "logA": 0.01, "n_s": 0.005, "h": 0.005, "omega_b": 0.0005, "m_ncdm": 0.02, "b1": 0.02, "b2": 0.05, "bG2": 0.05,
         "bGamma3": 0.05, "alpha0": 0.02, "alpha2": 1.0, "a_ng": 0.01, "c_v": 0.2, "db3": 0.01, "dbG3": 0.01,
         "dbdG2": 0.01}
DEFAULT_STEP = {"c": 1.0, "noise": 0.01}
#: The cosmology of the neutrino forecast, and its prior sets (Gaussian sigmas): BBN omega_b alone, and Planck 2018
#: (TT,TE,EE+lowE+lensing) on the primordial parameters and omega_b.
NEUTRINO = ("omega_cdm", "logA", "n_s", "h", "omega_b", "m_ncdm")
NEUTRINO_PRIORS = {"LSS only (BBN omega_b)": {"omega_b": 0.00037},
                   "Planck-like (logA, n_s, omega_b)": {"logA": 0.014, "n_s": 0.0042, "omega_b": 0.00015}}
#: The halo baseline of scripts/fit_halo_baseline.sh, without AP (no emulator has m_ncdm, and the AP background is
#: massless), with P_l to k <= 0.12 h/Mpc.
BASELINE = ["--stats", "pk", "s1m", "s21m", "--kmax", "0.12", "--s21-min-scale", "25", "--s21-white",
            "--cubic-biases", "relation", "--s21-layer2", "outputs/layer2/template_rsd.json", "--s21-layer2-c", "halo",
            "--s21-layer2-marg", "per_l"]

#: The halo q = 2 vector of docs/paper_plan.md (bands to k84 = 0.18 + S21m, `q2_12k18_s21_V60`) and its pieces, for
#: --layer2-info. "marg" marginalises the template amplitudes per l (width 1, as fitted), "w0.1" with width 0.1, "model"
#: keeps them at the template (only its Monte Carlo error enters the covariance), "exact" also drops that error.
L2_BANDS = ["--q", "2.0", "--s1-min-scale", "12.5", "--s1-ells", "0", "1", "2", "3", "4", "5", "6", "--s1-kmax", "0.18"]
L2_S21 = ["--s21-min-scale", "25", "--s21-white", "--s21-layer2", "outputs/layer2/template_rsd.json",
          "--s21-layer2-c", "halo"]
L2_TREATMENTS = {"marg": ["--s21-layer2-marg", "per_l"],
                 "w0.1": ["--s21-layer2-marg", "per_l", "--s21-layer2-marg-width", "0.1"], "model": [], "exact": []}
LAYER2_CASES = {
    **{f"(a/b) bands + S21m [{t}]": (["--stats", "s1m", "s21m"] + L2_BANDS + L2_S21 + extra, t)
       for t, extra in L2_TREATMENTS.items()},
    "(c) bands alone": (["--stats", "s1m"] + L2_BANDS, None),
    **{f"(d) S21m alone [{t}]": (["--stats", "s21m"] + L2_S21 + extra, t) for t, extra in L2_TREATMENTS.items()},
    "(e) P_l to 0.2": (["--stats", "pk", "--kmax", "0.2"], None),
    **{f"(f) P_l to 0.2 + S21m [{t}]": (["--stats", "pk", "s21m", "--kmax", "0.2"] + L2_S21 + extra, t)
       for t, extra in L2_TREATMENTS.items() if t != "w0.1"},
}
L2_REPORT = ("omega_cdm", "logA", "n_s", "h", "b1", "b2", "bG2")


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
                        default=None,
                        help="fit summaries whose best fits set the nuisance values (later files take precedence)")
    parser.add_argument("--output", type=Path, default=Path("outputs/fisher/rsd.json"))
    parser.add_argument("--fit-args", default="", help="window mode: further fit.py arguments (the fit to size)")
    parser.add_argument("--window-ratios", type=Path, nargs="*", default=None,
                        help="window mode: npz per cap (s1m (nmock, J+1, L+1, L+1), s21m (nmock, nentry), s21m_keys, "
                             "sigma, veff) of measured / window-model ratios of fiducial survey mocks")
    parser.add_argument("--layer2-info", action="store_true",
                        help="layer-2 mode: information of the modelled vs marginalised halo second layer")
    parser.add_argument("--neutrino", type=float, default=None,
                        help="neutrino mode: fiducial sum of neutrino masses [eV]")
    parser.add_argument("--nmean", type=int, default=25, help="window mode: surveys averaged in the data")
    parser.add_argument("--ncal", type=int, nargs="+", default=[100, 300, 1000, 3000, 10000],
                        help="window mode: numbers of calibration mocks to size")
    args = parser.parse_args()
    if args.fiducial is None:  # neutrino mode: the no-AP baseline fits (WST nuisances take precedence)
        args.fiducial = ([Path("outputs/inference/rsd/fast_pk_noap_V60/summary.json"),
                          Path("outputs/inference/rsd/fast_noap_V60/summary.json")] if args.neutrino is not None else
                         [Path("outputs/inference/rsd/halos_bias_only_V60_s1m/summary.json"),
                          Path("outputs/inference/rsd/halos_bias_only_V60_pk/summary.json")])
    return args


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


def layer2_information(args, fr):
    """Fisher errors of the LAYER2_CASES with the template amplitudes marginalised or modelled (module docstring)."""
    import shlex

    from wstfast.data import hartlap_factor, precision_matrix

    fiducial_files = [json.loads(path.read_text())["bestfit"] for path in args.fiducial if path.exists()]
    results = {}
    for ns in (False, True):
        vary = ["omega_cdm", "logA", "n_s", "h"] if ns else ["omega_cdm", "logA", "h"]
        for case, (case_args, treatment) in LAYER2_CASES.items():
            fit = fr.fit_options(fr.parse_args(["--data-dir", str(args.data_dir), "--emulator", str(args.emulator),
                                                "--output-dir", "unused", "--volume", str(args.volume), "--vary", *vary,
                                                *case_args] + shlex.split(args.fit_args)))
            built = inference.build_fit(fit)
            theory, labels = built["theory"], built["labels"]
            params = get_params(theory).select(varied=True, derived=False)
            names = params.names()
            fiducial = {name: float(params[name].value) for name in names}
            for best in fiducial_files:
                fiducial.update({n: float(v) for n, v in best.items() if n in names and n not in COSMOLOGY})
            fiducial.update({n: QUIJOTE_COSMOLOGY[n] for n in COSMOLOGY if n in names})
            fiducial.update({n: 0.0 for n in names if n.startswith("noise")})
            model = build(theory)
            jacobian = np.array([(np.asarray(model({**fiducial, n: fiducial[n] + step_of(n)}))
                                  - np.asarray(model({**fiducial, n: fiducial[n] - step_of(n)}))) / (2 * step_of(n))
                                 for n in names])
            covariance = np.array(built["covariance"])
            if treatment == "exact":  # without the template's Monte Carlo error (S21m rows pass the compressions as is)
                mean = built["problem"]["vectors"].mean(axis=0)
                variance = dict(zip(built["problem"]["labels"],
                                    inference.layer2_variance(fit, built["problem"]["labels"], mean)))
                covariance -= np.diag([variance.get(label, 0.0) for label in labels])
            prior = np.array([prior_sigma(params[n]) if params[n].prior.dist == "norm" else np.inf for n in names])
            hartlap = hartlap_factor(built["nsims"], len(labels))

            def errors(rows, known=()):
                cov = covariance[np.ix_(rows, rows)] * hartlap_factor(built["nsims"], len(rows)) / hartlap
                jac = jacobian[:, rows]
                cols = [i for i, n in enumerate(names)
                        if n not in known and (n in COSMOLOGY or np.any(jac[i] != 0))]
                fisher = (jac[cols] @ precision_matrix(cov * built["m1"]) @ jac[cols].T
                          + np.diag(1.0 / prior[cols] ** 2))
                inverse = np.linalg.inv(fisher)
                return {names[i]: float(np.sqrt(inverse[k, k])) for k, i in enumerate(cols) if names[i] in L2_REPORT}

            key = ("[n_s free] " if ns else "") + case
            results[key] = dict(ndata=len(labels), nparams=len(names), sigma=errors(np.arange(len(labels))))
            print(f"{key:44s} n={len(labels):4d} np={len(names):3d} | "
                  + " ".join(f"{p} {results[key]['sigma'].get(p, np.nan):.4f}" for p in L2_REPORT), flush=True)
            subsets = {}
            if case.startswith(("(c)", "(e)")):
                subsets["b2, bG2 known"] = (np.arange(len(labels)), ("b2", "bG2"))
            if case.startswith("(d)"):  # where the S21m information sits: per l and per sigma_2 (sigma_1 = 25 only)
                parts = [label.split("_") for label in labels]
                for at in (3, 2):
                    for value in sorted({p[at] for p in parts}):
                        subsets[value] = (np.array([n for n, p in enumerate(parts) if p[at] == value]), ())
            for value, (rows, known) in subsets.items():
                sub = f"{key} {value}"
                results[sub] = dict(ndata=len(rows), sigma=errors(rows, known))
                print(f"  {sub:42s} n={len(rows):4d} | "
                      + " ".join(f"{p} {results[sub]['sigma'].get(p, np.nan):.4f}" for p in L2_REPORT), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(volume=args.volume, emulator=str(args.emulator), fit_args=args.fit_args,
                                           cases={k: v[0] for k, v in LAYER2_CASES.items()}, results=results), indent=2))
    print(f"wrote {args.output}")


def neutrino_forecast(args, fr):
    """sigma(M_nu) of P_l, the WST (S1m + S21m) and both, from exact derivatives (module docstring)."""
    import shlex

    fit = fr.fit_options(fr.parse_args(["--data-dir", str(args.data_dir), "--emulator", str(args.emulator),
                                         "--output-dir", "unused", *BASELINE] + shlex.split(args.fit_args)))
    settings = json.loads(args.emulator.with_suffix(".json").read_text())
    basis = inference.exact_basis(settings, NEUTRINO)
    problem = inference.build_problem(fit, basis=basis, settings=settings)
    theory = JointTheory(problem["theories"])
    labels, vectors = problem["labels"], problem["vectors"]
    params = get_params(theory).select(varied=True, derived=False)
    names = params.names()
    fiducial = {name: float(params[name].value) for name in names}
    for path in args.fiducial:
        if path.exists():
            best = json.loads(path.read_text())["bestfit"]
            fiducial.update({name: float(value) for name, value in best.items() if name in names})
    fiducial.update({name: QUIJOTE_COSMOLOGY[name] for name in NEUTRINO if name in QUIJOTE_COSMOLOGY})
    fiducial.update(m_ncdm=args.neutrino, **{name: 0.0 for name in names if name.startswith("noise")})
    cache = args.output.with_suffix(".jacobian.npz")
    if cache.exists() and list(np.load(cache)["names"]) == names and list(np.load(cache)["labels"]) == labels:
        jacobian = np.load(cache)["jacobian"]
    else:
        model = build(theory)
        jacobian = np.zeros((len(names), len(labels)))
        for i, name in enumerate(names):
            h = step_of(name)
            lo = max(fiducial[name] - h, 0.0) if name == "m_ncdm" else fiducial[name] - h  # M_nu >= 0
            up = np.asarray(model({**fiducial, name: fiducial[name] + h}))
            down = np.asarray(model({**fiducial, name: lo}))
            jacobian[i] = (up - down) / (fiducial[name] + h - lo)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, names=names, labels=labels, jacobian=jacobian)
    nreal = len(vectors)
    mean = vectors.mean(axis=0)
    raw = np.cov(vectors, rowvar=False, ddof=1) / args.volume  # boxes of 1 (Gpc/h)^3
    theory_cov = np.diag(inference.layer2_variance(fit, labels, mean))  # volume-independent, as in the fit
    for vector in inference.layer2_templates(fit, labels, mean):
        theory_cov += fit.s21_layer2_marg_width**2 * np.outer(vector, vector)
    nuisance = np.array([np.inf if name in NEUTRINO else prior_sigma(params[name]) for name in names])
    cases = {f"P_l (k <= {fit.kmax})": ("P",), "WST (S1m + S21m)": ("S1m", "S21m"), "P_l + WST": ("P", "S1m", "S21m")}
    cosmo_names = [name for name in NEUTRINO if name in names]
    results = {}
    for prior_label, cosmo_priors in NEUTRINO_PRIORS.items():
        for case, kinds in cases.items():
            rows = np.array([n for n, label in enumerate(labels) if label.split("_")[0].startswith(kinds)])
            hartlap = (nreal - len(rows) - 2.0) / (nreal - 1.0)
            cov = raw[np.ix_(rows, rows)] / hartlap + theory_cov[np.ix_(rows, rows)]
            jac = jacobian[:, rows]
            cols = [i for i, name in enumerate(names) if name in NEUTRINO or np.any(jac[i] != 0)]
            prior = np.array([cosmo_priors.get(names[i], nuisance[i]) for i in cols])
            fisher = jac[cols] @ np.linalg.solve(cov, jac[cols].T) + np.diag(1.0 / prior**2)
            inverse = np.linalg.inv(fisher)
            idx = [cols.index(names.index(name)) for name in cosmo_names]
            sub = inverse[np.ix_(idx, idx)]
            sigma = np.sqrt(np.diag(sub))
            corr = sub / np.outer(sigma, sigma)
            m = cosmo_names.index("m_ncdm")
            results.setdefault(prior_label, {})[case] = dict(
                ndata=int(len(rows)), sigma=dict(zip(cosmo_names, sigma.tolist())),
                corr_mnu=dict(zip(cosmo_names, corr[m].tolist())))
    width = max(len(case) for case in cases) + 2
    for prior_label, by_case in results.items():
        print(f"\n{prior_label}: V = {args.volume} (Gpc/h)^3, M_nu = {args.neutrino} eV, nuisances marginalised")
        print(f"{'case':{width}s}{'n':>5s}" + "".join(f"{name:>11s}" for name in cosmo_names)
              + f"{'r(Mnu,logA)':>13s}{'r(Mnu,wcdm)':>13s}")
        for case, result in by_case.items():
            print(f"{case:{width}s}{result['ndata']:>5d}" + "".join(f"{result['sigma'][n]:>11.5f}" for n in cosmo_names)
                  + f"{result['corr_mnu']['logA']:>13.3f}{result['corr_mnu']['omega_cdm']:>13.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(volume=args.volume, fiducial=fiducial, fit_args=BASELINE + [args.fit_args],
                                           priors=NEUTRINO_PRIORS, results=results, nrealizations=nreal), indent=2))
    print(f"\nwrote {args.output}")


def main():
    args = parse_args()
    fr = load_fit()
    if args.window_ratios:
        return window_calibration(args, fr)
    if args.neutrino is not None:
        return neutrino_forecast(args, fr)
    if args.layer2_info:
        return layer2_information(args, fr)
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
