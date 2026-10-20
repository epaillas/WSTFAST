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

With ``--edgeworth-info``, it instead asks whether the halo q != 2 first layer (the Edgeworth factor E(q)) next to the
q = 2 bands keeps its gain over P_l to 0.2 once the E nuisances are free (``EDGEWORTH_HALO``; ``--real-space`` runs the
real-space matter control ``EDGEWORTH_REAL`` against P_0 to 0.2). The parameters the refactor removed from the package
enter as extra Jacobian columns (``edgeworth_amplitudes``): A_TN, A_TN2 (priors N(1, 1)) scaling the N and N^2 Poisson
pieces of the tree kappa, and Edgeworth amplitudes of the q != 2 blocks, one, per sigma or per (sigma, l) (Gaussian
priors of width EDGEWORTH_NG_PRIOR); the cubic-bias priors (and bGamma3's) are also widened x3 or made flat, and the
q != 2 rows cut at sigma >= 12.5, 17.7 or 25:

    python scripts/fisher_rsd.py --edgeworth-info --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los \\
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic_ap3_bands12.h5 --volume 60 \\
        --fiducial outputs/inference/rsd/q2/pk02_V60/summary.json outputs/inference/rsd/q2/q2_12k18_s21_V60/summary.json \\
        --output outputs/fisher/edgeworth_info_V60.json

With ``--response-info``, it instead forecasts what the cross-spectra of the first-layer modulus fields with the density,
X_a = P_{U_a d} / <U_a> (a = sigma, l, |m|; monopole and quadrupole), add to P_l to 0.2: X_a is zero for a Gaussian field
(U_a is even in d) and its tree-level model is the response r_a Z1 P_L of the S21 model (``ModulusCrossRSD``). P_l has the
model, binning and priors of fit.py; X_a its tree-level model at the shared cosmology and b1, b2, bG2, a velocity
dispersion of the short modes, and per field a k^2 counterterm and a constant; the joint covariance is the sample
covariance of the boxes. Each field is first checked against the mean of the boxes at the fiducial biases
(``RESPONSE_FIDUCIAL``, the best point of a (b2, bG2, sigma_v) grid), k <= x / sigma, counterterm and constant fitted;
fields above chi^2/dof 2 at V = --volume are dropped from the "validated" rows:

    python scripts/fisher_rsd.py --response-info --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los \
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic.h5 --volume 60 \
        --fiducial outputs/inference/rsd/q2/pk02_V60/summary.json --output outputs/fisher/response_info_V60.json

Example (halos):

    python scripts/fisher_rsd.py --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los \\
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir.h5 --volume 25 --output outputs/fisher/halos_rsd_V25.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from contextlib import contextmanager
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
#: keeps them at the template (only its Monte Carlo error enters the covariance), "exact" also drops that error, "none"
#: has no template: S21m is taken at the first q (2 here), where the template, a deficit with respect to the Gaussian
#: prediction from S2m(q = 2), vanishes identically (scripts/layer2/template_rsd.py measures it at q = 0.8).
L2_BANDS = ["--q", "2.0", "--s1-min-scale", "12.5", "--s1-ells", "0", "1", "2", "3", "4", "5", "6", "--s1-kmax", "0.18"]
L2_S21 = ["--s21-min-scale", "25", "--s21-white"]
L2_TEMPLATE = ["--s21-layer2", "outputs/layer2/template_rsd.json", "--s21-layer2-c", "halo"]
L2_TREATMENTS = {"marg": L2_TEMPLATE + ["--s21-layer2-marg", "per_l"],
                 "w0.1": L2_TEMPLATE + ["--s21-layer2-marg", "per_l", "--s21-layer2-marg-width", "0.1"],
                 "model": L2_TEMPLATE, "exact": L2_TEMPLATE, "none": []}
LAYER2_CASES = {
    **{f"(a/b) bands + S21m [{t}]": (["--stats", "s1m", "s21m"] + L2_BANDS + L2_S21 + extra, t)
       for t, extra in L2_TREATMENTS.items()},
    "(c) bands alone": (["--stats", "s1m"] + L2_BANDS, None),
    **{f"(d) S21m(q = 0.8) alone [{t}]": (["--stats", "s21m"] + L2_S21 + extra, t)
       for t, extra in L2_TREATMENTS.items() if t != "none"},
    **{f"(d) S21m(q = 2) alone [{t}]": (["--stats", "s21m", "--q", "2.0"] + L2_S21 + L2_TREATMENTS[t], t)
       for t in ("marg", "none")},
    "(e) P_l to 0.2": (["--stats", "pk", "--kmax", "0.2"], None),
    **{f"(f) P_l to 0.2 + S21m(q = 0.8) [{t}]": (["--stats", "pk", "s21m", "--kmax", "0.2"] + L2_S21 + extra, t)
       for t, extra in L2_TREATMENTS.items() if t not in ("w0.1", "none")},
    **{f"(f) P_l to 0.2 + S21m(q = 2) [{t}]": (["--stats", "pk", "s21m", "--kmax", "0.2", "--q", "2.0"] + L2_S21
                                                + L2_TREATMENTS[t], t) for t in ("marg", "none")},
}
L2_REPORT = ("omega_cdm", "logA", "n_s", "h", "b1", "b2", "bG2")

#: --response-info: the modulus fields (j1, l, |m|) of X_a, the fiducial (b2, bG2) and short-mode velocity dispersion of
#: the tree response (best point of a grid against the box mean at k <= 0.75 / sigma), the finite-difference steps.
RESPONSE_FIELDS = [(j, l, m) for j in (4, 5, 6) for l in range(1, 5) for m in range(l + 1)]
RESPONSE_FIDUCIAL = dict(b2=-0.5, bG2=-0.3, sv=6.0)
RESPONSE_STEPS = dict(omega_cdm=0.002, logA=0.01, h=0.005, n_s=0.005, b1=0.02, b2=0.05, bG2=0.05, sv=0.5)

#: --edgeworth-info: the halo q = 2 bands (+ S21m) with the q = 0.8 S1m of the same blocks, against P_l to 0.2; the
#: real-space matter control (--real-space, bands12k + S21m + q = 0.8 at the emulated l <= 4, against P_0 to 0.2).
EDGEWORTH_HALO = {
    "reference": ["--stats", "pk", "--kmax", "0.2"],
    "q2": ["--stats", "s1m", "s21m", "--cubic-biases", "relation"] + L2_BANDS + L2_S21 + L2_TREATMENTS["marg"],
    "full": ["--stats", "s1m", "s21m", "--cubic-biases", "relation"] + L2_BANDS[:1] + ["2.0", "0.8"] + L2_BANDS[2:]
            + L2_S21 + L2_TREATMENTS["marg"],
}
EDGEWORTH_REAL_BANDS = ["--band-min-scale", "12.5", "--band-ells", "0", "1", "2", "3", "4", "5", "6", "--band-kmax", "0.2",
                        "--s21-min-scale", "17.6", "--s21-white", "--s21-layer2", "outputs/layer2/template_real.json",
                        "--s21-layer2-c", "matter", "--s21-layer2-marg", "per_l", "--compress-m"]
EDGEWORTH_REAL = {
    "reference": ["--stats", "pk", "--kmax", "0.2"],
    "q2": ["--stats", "band", "s21m"] + EDGEWORTH_REAL_BANDS,
    "full": ["--stats", "band", "s21m", "s1m", "--q", "0.8", "--s1-min-scale", "12.5"] + EDGEWORTH_REAL_BANDS,
}
#: Gaussian prior sigma of the free Edgeworth amplitudes (the flat [0.5, 1.5] of the old --ng-amplitude).
EDGEWORTH_NG_PRIOR = 0.29


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
    parser.add_argument("--edgeworth-info", action="store_true",
                        help="Edgeworth mode: gain of the q != 2 S1m next to the q = 2 bands, E(q) nuisances free")
    parser.add_argument("--real-space", action="store_true", help="Edgeworth mode: the real-space matter control")
    parser.add_argument("--response-info", action="store_true",
                        help="response mode: what the modulus cross-spectra P_Ud add to P_l to 0.2")
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


def case_fisher(args, fr, case_args, vary, exact=False, extra=None):
    """The Fisher pieces of one fit.py vector (``build_fit``: its covariance with Hartlap and Percival and the layer-2
    terms): the varied parameter names, the Jacobian by central differences of the emulated model at the Quijote
    cosmology and the nuisances of ``--fiducial``, the covariance (``exact``: without the layer-2 template's Monte
    Carlo error), the Gaussian prior sigmas (inf for flat ones) and the labels. ``extra`` = callable(theory, fiducial,
    labels) -> {name: (column, prior sigma)} adds Jacobian columns of parameters the theory does not have."""
    import shlex

    fit = fr.fit_options(fr.parse_args(["--data-dir", str(args.data_dir), "--emulator", str(args.emulator),
                                        "--output-dir", "unused", "--volume", str(args.volume), "--vary", *vary,
                                        *case_args] + shlex.split(args.fit_args)))
    built = inference.build_fit(fit)
    theory, labels = built["theory"], built["labels"]
    params = get_params(theory).select(varied=True, derived=False)
    names = params.names()
    fiducial = {name: float(params[name].value) for name in names}
    for path in args.fiducial:
        if path.exists():
            best = json.loads(path.read_text())["bestfit"]
            fiducial.update({n: float(v) for n, v in best.items() if n in names and n not in COSMOLOGY})
    fiducial.update({n: QUIJOTE_COSMOLOGY[n] for n in COSMOLOGY if n in names})
    fiducial.update({n: 0.0 for n in names if n.startswith("noise")})
    model = build(theory)
    jacobian = [(np.asarray(model({**fiducial, n: fiducial[n] + step_of(n)}))
                 - np.asarray(model({**fiducial, n: fiducial[n] - step_of(n)}))) / (2 * step_of(n)) for n in names]
    prior = [prior_sigma(params[n]) if params[n].prior.dist == "norm" else np.inf for n in names]
    if extra is not None:
        for name, (column, sigma) in extra(theory, fiducial, labels).items():
            names, jacobian, prior = names + [name], jacobian + [column], prior + [sigma]
    covariance = np.array(built["covariance"])
    if exact:  # without the template's Monte Carlo error (S21m rows pass the compressions as is)
        mean = built["problem"]["vectors"].mean(axis=0)
        variance = dict(zip(built["problem"]["labels"], inference.layer2_variance(fit, built["problem"]["labels"], mean)))
        covariance -= np.diag([variance.get(label, 0.0) for label in labels])
    return dict(names=list(names), jacobian=np.array(jacobian), covariance=covariance, prior=np.array(prior),
                labels=list(labels), nsims=built["nsims"], m1=built["m1"], fiducial=fiducial)


def fisher_errors(case, rows, known=(), report=None, prior_scale=None):
    """Marginalised Fisher errors of ``case`` (``case_fisher``) on ``rows``, with the parameters ``known`` fixed and
    the prior sigmas of ``prior_scale`` ({name: factor}) widened; every nuisance the rows depend on is marginalised."""
    from wstfast.data import hartlap_factor, precision_matrix

    names, jacobian = case["names"], case["jacobian"][:, rows]
    hartlap = hartlap_factor(case["nsims"], len(case["labels"]))
    cov = case["covariance"][np.ix_(rows, rows)] * hartlap_factor(case["nsims"], len(rows)) / hartlap
    prior = case["prior"] * np.array([(prior_scale or {}).get(n, 1.0) for n in names])
    cols = [i for i, n in enumerate(names) if n not in known and (n in COSMOLOGY or np.any(jacobian[i] != 0))]
    fisher = jacobian[cols] @ precision_matrix(cov * case["m1"]) @ jacobian[cols].T + np.diag(1.0 / prior[cols] ** 2)
    inverse = np.linalg.inv(fisher)
    return {names[i]: float(np.sqrt(inverse[k, k])) for k, i in enumerate(cols) if report is None or names[i] in report}


def layer2_information(args, fr):
    """Fisher errors of the LAYER2_CASES with the template amplitudes marginalised or modelled (module docstring)."""
    results = {}
    for ns in (False, True):
        vary = ["omega_cdm", "logA", "n_s", "h"] if ns else ["omega_cdm", "logA", "h"]
        for case, (case_args, treatment) in LAYER2_CASES.items():
            fc = case_fisher(args, fr, case_args, vary, exact=treatment == "exact")
            labels = fc["labels"]
            key = ("[n_s free] " if ns else "") + case
            results[key] = dict(ndata=len(labels), nparams=len(fc["names"]),
                                sigma=fisher_errors(fc, np.arange(len(labels)), report=L2_REPORT))
            print(f"{key:44s} n={len(labels):4d} np={len(fc['names']):3d} | "
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
                results[sub] = dict(ndata=len(rows), sigma=fisher_errors(fc, rows, known, report=L2_REPORT))
                print(f"  {sub:42s} n={len(rows):4d} | "
                      + " ".join(f"{p} {results[sub]['sigma'].get(p, np.nan):.4f}" for p in L2_REPORT), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(volume=args.volume, emulator=str(args.emulator), fit_args=args.fit_args,
                                           cases={k: v[0] for k, v in LAYER2_CASES.items()}, results=results), indent=2))
    print(f"wrote {args.output}")


@contextmanager
def edgeworth_amplitudes(A_TN=1.0, A_TN2=1.0, a_ng=1.0):
    """Within the block, the free E(q) nuisances that the refactor removed from the package (fit_rsd.py --kappa-shot and
    --ng-amplitude, b5436d2): A_TN, A_TN2 scale the N and N^2 Poisson pieces of the tree kappa (S1m and the S21m
    Edgeworth factors), a_ng the Edgeworth correction of the q != 2 S1m blocks."""
    from wstfast import calculators
    from wstfast.theory import first_layer
    from wstfast.theory.layout import CUBIC_KAPPA_SIZES, KAPPA_SIZES

    edges = np.cumsum((0,) + KAPPA_SIZES + CUBIC_KAPPA_SIZES)
    factor = np.ones(edges[-1])
    factor[edges[1]:edges[2]], factor[edges[2]:edges[3]] = A_TN, A_TN2
    cumulants, s1m = calculators.tracer_cumulants, first_layer.s1m_from_variances

    def tracer_cumulants(kappa, *rest, **kwargs):
        return cumulants(kappa * factor[:kappa.shape[-1]], *rest, **kwargs)

    def s1m_from_variances(*rest, **kwargs):
        if rest[5] != 2.0:
            kwargs["edgeworth_amplitude"] = a_ng
        return s1m(*rest, **kwargs)

    calculators.tracer_cumulants, first_layer.s1m_from_variances = tracer_cumulants, s1m_from_variances
    try:
        yield
    finally:
        calculators.tracer_cumulants, first_layer.s1m_from_variances = cumulants, s1m


def edgeworth_columns(config):
    """``case_fisher`` extra: the Jacobian columns of A_TN, A_TN2 (central differences, priors N(1, 1)) and of the
    Edgeworth amplitudes of the q != 2 S1m blocks, one for all blocks (a_ng), per sigma (a_ng_j{j}) and per (sigma, l)
    (a_ng_j{j}_l{l}); S1m is linear in a_ng, so a column is S1m(a_ng = 1) - S1m(a_ng = 0) on its blocks. Prior sigma
    of a_ng: EDGEWORTH_NG_PRIOR. The model is rebuilt within each patch (the compiled graph is jitted once)."""
    def extra(theory, fiducial, labels):
        def evaluate(**amplitudes):
            with edgeworth_amplitudes(**amplitudes):
                return np.asarray(build(theory)(fiducial))

        columns = {name: ((evaluate(**{name: 1.1}) - evaluate(**{name: 0.9})) / 0.2, 1.0) for name in ("A_TN", "A_TN2")}
        delta = evaluate() - evaluate(a_ng=0.0)
        blocks = edgeworth_blocks(labels, config)
        groups = {"a_ng": lambda b: True}
        groups.update({f"a_ng_j{j}": (lambda b, j=j: b[0] == j) for j in sorted({b[0] for b in blocks.values()})})
        groups.update({f"a_ng_j{j}_l{l}": (lambda b, j=j, l=l: b[:2] == (j, l))
                       for j, l in sorted({b[:2] for b in blocks.values()})})
        for name, member in groups.items():
            rows = [n for n, b in blocks.items() if member(b)]
            column = np.zeros(len(labels))
            column[rows] = delta[rows]
            columns[name] = (column, EDGEWORTH_NG_PRIOR)
        return columns
    return extra


def edgeworth_blocks(labels, config, q=0.8):
    """{row: (j, l, m, sigma)} of the q != 2 S1m rows of ``labels`` (``[q{q}:]S1m_j{j}_l{l}_m{m}``, m may be the
    ``m1+`` class of ``--compress-m``; an S1m label without a ``q`` prefix has exponent ``q``)."""
    out = {}
    for n, label in enumerate(labels):
        prefix, _, base = label.rpartition(":")
        exponent = float(prefix.rpartition("q")[-1]) if prefix else q
        if base.startswith("S1m_") and exponent != 2.0:
            j, l, m = base.split("_")[1:]
            out[n] = (int(j[1:]), int(l[1:]), m[1:], config.sigma(int(j[1:])))
    return out


def edgeworth_information(args, fr):
    """Fisher gain of the q != 2 S1m blocks next to the q = 2 bands over P_l to 0.2, with the E(q) nuisances free
    (module docstring, EDGEWORTH_CASES)."""
    first = sorted((args.data_dir / ("real" if args.real_space else "rsd")).glob("wst_r*.npz"))[0]
    config = load_measurement(first, q=2.0)["config"]
    vectors = EDGEWORTH_REAL if args.real_space else EDGEWORTH_HALO
    cubic = () if args.real_space else ("db3", "dbG3", "dbdG2", "bGamma3")
    results = {}
    for ns in (False, True):
        vary = ["omega_cdm", "logA", "n_s", "h"] if ns else ["omega_cdm", "logA", "h"]
        tag = "[n_s free] " if ns else ""
        ref = case_fisher(args, fr, vectors["reference"], vary)
        reference = fisher_errors(ref, np.arange(len(ref["labels"])), report=COSMOLOGY)
        results[tag + "reference"] = dict(ndata=len(ref["labels"]), sigma=reference)
        two = case_fisher(args, fr, vectors["q2"], vary)
        results[tag + "q = 2 alone"] = dict(ndata=len(two["labels"]),
                                            sigma=fisher_errors(two, np.arange(len(two["labels"])), report=COSMOLOGY))
        full = case_fisher(args, fr, vectors["full"], vary, extra=edgeworth_columns(config))
        blocks = edgeworth_blocks(full["labels"], config)
        ng = [n for n in full["names"] if n.startswith("a_ng")]
        per = {"global": ["a_ng"], "per sigma": [n for n in ng if n.count("_") == 2],
               "per (sigma, l)": [n for n in ng if n.count("_") == 3]}
        treatments = {  # name: (free E nuisances, prior widening of the cubic biases)
            "baseline E (tree K4, priors as fitted)": ((), {}),
            "+ A_TN, A_TN2": (("A_TN", "A_TN2"), {}),
            "+ a_ng": (tuple(per["global"]), {}),
            "+ a_ng per sigma": (tuple(per["per sigma"]), {}),
            "+ a_ng per (sigma, l)": (tuple(per["per (sigma, l)"]), {}),
            "+ A_TN, A_TN2, a_ng per sigma": (("A_TN", "A_TN2", *per["per sigma"]), {}),
        }
        if cubic:
            treatments.update({
                "cubic priors x3": ((), {n: 3.0 for n in cubic}),
                "cubic priors flat": ((), {n: np.inf for n in cubic}),
                "all: A_TN, A_TN2, a_ng per sigma, cubic x3": (("A_TN", "A_TN2", *per["per sigma"]),
                                                               {n: 3.0 for n in cubic}),
                "all: A_TN, A_TN2, a_ng per (sigma, l), cubic x3": (("A_TN", "A_TN2", *per["per (sigma, l)"]),
                                                                    {n: 3.0 for n in cubic}),
            })
        cuts = {"q != 2 at sigma >= 12.5": 0.0, "q != 2 at sigma >= 17.7": 17.6, "q != 2 at sigma >= 25": 24.9}
        for cut, smin in cuts.items():
            rows = np.array([n for n in range(len(full["labels"])) if n not in blocks or blocks[n][3] >= smin])
            for treatment, (free, widen) in treatments.items():
                known = [n for n in full["names"] if (n in ("A_TN", "A_TN2") or n.startswith("a_ng")) and n not in free]
                sigma = fisher_errors(full, rows, known=known, report=COSMOLOGY, prior_scale=widen)
                key = f"{tag}{cut} | {treatment}"
                gain = {p: 1 - sigma[p] / reference[p] for p in sigma}
                results[key] = dict(ndata=len(rows), sigma=sigma, gain=gain)
                print(f"{key:90s} n={len(rows):4d} | " + " ".join(f"{p} {sigma[p]:.4f} ({100 * gain[p]:+.0f}%)"
                                                               for p in sigma), flush=True)
        print(f"{tag}reference: " + " ".join(f"{p} {v:.4f}" for p, v in reference.items()), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(volume=args.volume, emulator=str(args.emulator), fit_args=args.fit_args,
                                           vectors=vectors, results=results), indent=2))
    print(f"wrote {args.output}")


def response_information(args, fr):
    """Fisher errors of P_l to 0.2 + the modulus cross-spectra X_a (module docstring)."""
    from cosmoprimo import Cosmology
    from numpy.polynomial.legendre import leggauss
    from scipy.special import eval_legendre

    from wstfast.data import load_power_dataset, precision_matrix
    from wstfast.theory.covariance_rsd import ModulusCrossRSD

    files = sorted((args.data_dir / "rsd").glob("wst_r*.npz"))
    n = len(files)
    pl = load_power_dataset(args.data_dir, "rsd", kmax=0.2, rebin=2, ells=(0, 2, 4)).vectors
    with np.load(files[0]) as z:
        k, nmodes, shot = z["k"], z["nmodes"], 1e9 / json.loads(str(z["metadata"]))["nparticles"]
    nk = k.size // 2 * 2
    w = nmodes[:nk].reshape(-1, 2)
    kb = (k[:nk].reshape(-1, 2) * w).sum(1) / w.sum(1)  # the mode-weighted bins of load_power_dataset
    kk = kb[kb <= 0.13]
    X = []
    for path in files:
        with np.load(path) as z:
            merged = (z["PUd_m"][..., :nk].reshape(*z["PUd_m"].shape[:-1], -1, 2) * w).sum(-1) / w.sum(1)
            X.append(merged[..., :kk.size] / z["Umean_m"][..., None, None])
    X = np.array(X)  # (n, j, l, |m|, ell, nk)
    sigma = lambda j: 3.125 * 2 ** (j / 2)  # noqa: E731
    mu, wmu = leggauss(16)
    mu, wmu = mu[8:], wmu[8:]
    model = ModulusCrossRSD([(sigma(j), l, m) for j, l, m in RESPONSE_FIELDS], kk, mu)

    def multipoles(grid):
        return np.concatenate([(2 * e + 1) * (grid * eval_legendre(e, mu)[None, :] * wmu).sum(1) for e in (0, 2)])

    q = QUIJOTE_COSMOLOGY
    b1 = json.loads(args.fiducial[0].read_text())["bestfit"]["b1"]
    fiducial = dict(omega_cdm=q["omega_cdm"], logA=q["logA"], h=q["h"], n_s=q["n_s"], b1=b1, **RESPONSE_FIDUCIAL)

    def tree(p):
        cosmo = Cosmology(omega_cdm=p["omega_cdm"], omega_b=q["omega_b"], h=p["h"], n_s=p["n_s"], logA=p["logA"],
                          m_ncdm=[], N_eff=3.046, engine="class")
        pk = cosmo.get_fourier().pk_interpolator(of="delta_m").to_1d(z=0.5)
        f = float(cosmo.growth_rate(0.5))
        r, _, _ = model.spectra(pk, f, p["sv"], np.array([1.0, p["b1"], p["b2"], p["bG2"], 0.0]), shot)
        z1 = p["b1"] + f * mu**2
        return np.array([multipoles(r[i] * z1[None, :] * pk(kk)[:, None]) for i in range(len(RESPONSE_FIELDS))]), pk

    base, pk0 = tree(fiducial)
    dtree = {name: (tree({**fiducial, name: fiducial[name] + h})[0] - tree({**fiducial, name: fiducial[name] - h})[0])
             / (2 * h) for name, h in RESPONSE_STEPS.items()}
    columns = dict(c=multipoles((kk**2 * pk0(kk))[:, None] * np.ones_like(mu)[None, :]),
                   e=multipoles(np.ones((kk.size, mu.size))))
    data = {a: X[:, j, l, m][:, [0, 1]].reshape(n, -1) for a, (j, l, m) in enumerate(RESPONSE_FIELDS)}

    def cut(a, x):
        return np.tile(kk <= x / sigma(RESPONSE_FIELDS[a][0]), 2)

    # Model check: per field, counterterm and constant fitted to the mean of the boxes at the fiducial biases.
    check = {}
    for x in (0.75, 1.0):
        for a, (j, l, m) in enumerate(RESPONSE_FIELDS):
            sel = cut(a, x)
            nb = int(sel.sum())
            cov = np.cov(data[a][:, sel], rowvar=False) / args.volume * (n - 1) / (n - nb - 2)
            design = np.array([columns["c"][sel], columns["e"][sel]]).T
            y = data[a][:, sel].mean(0) - base[a][sel]
            inv = np.linalg.inv(cov)
            res = y - design @ np.linalg.solve(design.T @ inv @ design, design.T @ inv @ y)
            check[x, a] = float(res @ inv @ res / (nb - 2))
    for x in (0.75, 1.0):
        print(f"model check, k <= {x} / sigma: chi^2/dof per field " + ", ".join(
            f"{sigma(j):.1f}/{l}/{m}: {check[x, a]:.1f}" for a, (j, l, m) in enumerate(RESPONSE_FIELDS)), flush=True)

    def forecast(vary, scales, x, nuisances, validated):
        case = case_fisher(args, fr, ["--stats", "pk", "--kmax", "0.2"], vary)
        names, prior = list(case["names"]), list(case["prior"])
        fields = [a for a, (j, _, _) in enumerate(RESPONSE_FIELDS) if j in scales
                  and (not validated or check[x, a] <= 2.0)]
        cuts = [cut(a, x) for a in fields]
        nx = int(sum(c.sum() for c in cuts))
        vector = np.concatenate([pl] + [data[a][:, c] for a, c in zip(fields, cuts)], axis=1)
        jac = {name: np.concatenate([np.zeros(0)] + [dtree[name][a][c] for a, c in zip(fields, cuts)])
               for name in [*vary, "b1", "b2", "bG2", "sv"]}
        offset = 0
        for a, c in zip(fields, cuts):
            for kind in nuisances:
                column = np.zeros(nx)
                column[offset:offset + c.sum()] = columns[kind][c]
                jac[f"{kind}_{a}"] = column
            offset += int(c.sum())
        allnames = names + [name for name in jac if name not in names]
        full = np.zeros((len(allnames), vector.shape[1]))
        for i, name in enumerate(allnames):
            if name in names:
                full[i, :pl.shape[1]] = case["jacobian"][names.index(name)]
            if name in jac:
                full[i, pl.shape[1]:] = jac[name]
        sig = np.array([prior[names.index(name)] if name in names else np.inf for name in allnames])
        cov = np.cov(vector, rowvar=False) / args.volume * (n - 1) / (n - vector.shape[1] - 2)
        used = [i for i, name in enumerate(allnames) if name in vary or np.any(full[i] != 0)]
        inverse = np.linalg.inv(full[used] @ precision_matrix(cov) @ full[used].T + np.diag(1 / sig[used] ** 2))
        return {allnames[i]: float(np.sqrt(inverse[b, b])) for b, i in enumerate(used) if allnames[i] in L2_REPORT}, \
            int(vector.shape[1])

    cases = {"P_l to 0.2": ((), 0.75, (), False)}
    for x in (0.75, 1.0):
        for scales in ((5, 6), (4, 5, 6)):
            for nuisances in (("c", "e"), ("e",)):
                for validated in (True, False):
                    label = (f"+ X sigma {'/'.join(f'{sigma(j):.1f}' for j in scales)}, k <= {x}/sigma, "
                             f"{'+'.join(nuisances)} free{', validated fields' if validated else ''}")
                    cases[label] = (scales, x, nuisances, validated)
    results = {}
    for ns in (False, True):
        vary = ["omega_cdm", "logA", "n_s", "h"] if ns else ["omega_cdm", "logA", "h"]
        for label, spec in cases.items():
            key = ("[n_s free] " if ns else "") + label
            errors, ndata = forecast(vary, *spec)
            results[key] = dict(ndata=ndata, sigma=errors)
            print(f"{key:72s} n={ndata:3d} | " + " ".join(f"{p} {errors.get(p, np.nan):.4f}" for p in L2_REPORT),
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(volume=args.volume, fiducial=fiducial, results=results,
                                           check={f"{x}:{RESPONSE_FIELDS[a]}": v for (x, a), v in check.items()}),
                                      indent=2))
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
    if args.edgeworth_info:
        return edgeworth_information(args, fr)
    if args.response_info:
        return response_information(args, fr)
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
