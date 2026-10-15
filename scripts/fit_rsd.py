#!/usr/bin/env python
"""Redshift-space fits of Quijote matter: P(k) multipoles, line-of-sight-resolved S1m and S21m (desilike, MH).

Both statistics come from the same line-of-sight-resolved files, so the joint covariance (with cross terms)
is the sample covariance of the concatenated vectors [S1m blocks, P_0, P_2, P_4]. They share one emulated
RSDBasis (``train_emulator.py --stat rsd``); each keeps its own counterterms (c0, c2, c4 for S1m and
c0_pk, c2_pk, c4_pk for P(k)). Example (4 parameters, omega_b fixed, errors of a 60 (Gpc/h)^3 survey):

    python scripts/fit_rsd.py --stats pk s1m --kmax 0.12 --vary omega_cdm logA n_s h --volume 60 \
        --emulator outputs/emulators/rsd_basis_taylor_4p.h5 --output-dir outputs/inference/rsd/joint_V60

With ``--ap``, the Alcock-Paczynski distortions of the trial cosmology (relative to the Quijote fiducial) are
applied at assembly, on top of the true-frame emulated basis (P(k) and S1m only).

S21m = S2m / S1m (``--stats s21m``) needs an emulator trained with the S21 terms; it adds one noise amplitude
noise_j{j1}_l{l}_m{m1} per first-layer field (``wstfast.theory.rsd_moduli``).

Halos: an emulator trained with ``--tracer biased`` (``wstfast.theory.rsd_bias``) and the halo measurements; the
tracer is read from the emulator. All statistics then share the Eulerian biases b1, b2, bG2, bGamma3 and the
stochastic alpha0, alpha2 (``BiasParameters``); the shot noise V / N is the mean over the realizations:

    python scripts/fit_rsd.py --stats pk s1m s21m --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los \
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir.h5 --vary omega_cdm logA n_s h --volume 8 --output-dir ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from desilike import build, get_params, setup_logging
from desilike.base import Posterior
from wstfast.calculators import APGeometry, JointTheory, RSDPowerTheory, S1mTheory, S21mTheory, WSTLikelihood
from wstfast.config import QUIJOTE_COSMOLOGY, Coefficient, WSTConfig, select_coefficients
from wstfast.data import (chi_squared, hartlap_factor, load_measurement, load_power_dataset, load_s1m_dataset,
                          load_s21m_dataset, percival_m1, sample_covariance)
from wstfast.inference import (_bound_to_emulator, _fix_unvaried, bestfit_values, fix_parameters, parse_fixed,
                                profile, sample_mh, sample_nuts)
from wstfast.theory.rsd import (APPolynomial, MultipoleProjection, RSDGrid, S1mProjection, multipole_ap_projection,
                                s1m_ap_projection)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los"))
    parser.add_argument("--stats", nargs="+", choices=("pk", "s1m", "s21m"), default=["pk", "s1m"])
    parser.add_argument("--q", type=float, nargs="+", default=[0.8],
                        help="S1m exponent(s): several values concatenate the S1m data vectors (shared nuisances, joint "
                             "covariance); S21m uses the first")
    parser.add_argument("--s1-min-scale", type=float, default=25.0, help="smallest sigma_j [Mpc/h] of S1m")
    parser.add_argument("--s1-ells", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--s21-min-scale", type=float, default=17.6, help="smallest sigma_j1 [Mpc/h] of S21m")
    parser.add_argument("--s21-min-scale2", type=float, default=70.0, help="smallest sigma_j2 [Mpc/h] of S21m")
    parser.add_argument("--s21-min-ratio", type=float, default=2.8, help="smallest sigma_j2 / sigma_j1 of S21m")
    parser.add_argument("--kmin", type=float, default=0.0)
    parser.add_argument("--kmax", type=float, default=0.12)
    parser.add_argument("--rebin", type=int, default=2, help="P(k) bins of rebin x k_f")
    parser.add_argument("--vary", nargs="+", default=["omega_cdm", "logA"], choices=sorted(QUIJOTE_COSMOLOGY))
    parser.add_argument("--emulator", type=Path, required=True, help="RSDBasis emulator (train_emulator.py --stat rsd)")
    parser.add_argument("--volume", type=float, default=None,
                        help="errors of a survey of this volume in (Gpc/h)^3: single-box covariance / (V / V_box)")
    parser.add_argument("--covariance-of-mean", action="store_true", help="errors of the realization mean")
    parser.add_argument("--survey", nargs="+", default=None, metavar="CAP",
                        help="survey fit: the caps (data in <data-dir>/<cap>/, e.g. NGC SGC; dr1_window.py --test "
                             "measure), one survey.SurveyWindow each (--window); shared parameters, the covariance from "
                             "the --covariance-data-dir boxes scaled to each cap's effective volume")
    parser.add_argument("--window", type=Path, nargs="+", default=None, help="survey: window file of each cap")
    parser.add_argument("--extra-covariance", type=Path, nargs="*", default=[],
                        help="npz files (labels, covariance) of modelling errors added to the covariance as they are "
                             "(not scaled with the volume), e.g. the window calibration (survey.window_correction)")
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
    parser.add_argument("--cumulant-points", type=int, default=2**20,
                        help="Sobol points of the control-variate correction of the S1m cumulants (0: none)")
    parser.add_argument("--fog", choices=("shared", "separate"), default=None,
                        help="Fingers-of-God counterterm of an emulator trained with --fog: one c_v for every tree-level "
                             "short-mode leg of S1m and S21m (shared), or c_v_s1 and c_v_s21 (separate)")
    parser.add_argument("--ng-amplitude", nargs="?", const="constant", default=None,
                        choices=("constant", "slope", "per-scale", "fog"),
                        help="free amplitude of the S1m Edgeworth correction: one a_ng (default when given without a "
                             "value), a_ng + b_ng [(25 / sigma)^2 - 1], or one a_ng_j per scale")
    parser.add_argument("--gaussian-prior", nargs="+", default=[], metavar="NAME=LOC,SCALE",
                        help="Gaussian prior on an emulated cosmological parameter, truncated to the emulator box, "
                             "e.g. n_s=0.9649,0.042 (Planck 2018 x 10, as the desi-clustering full-shape fits)")
    parser.add_argument("--k4-counterterm", action="store_true",
                        help="biased tracers: next-to-leading redshift-space counterterm -ck4 k^4 mu^4 (b1 + f mu^2)^2 P_L "
                             "(ck4 for S1m, ck4_pk for P_l unless --shared-counterterms)")
    parser.add_argument("--b-shot", action="store_true",
                        help="biased tracers: free amplitude B_shot of the Poisson bispectrum term of the S21m response")
    parser.add_argument("--noise-shot", action="store_true",
                        help="biased tracers: free amplitude alpha_N of the discreteness terms of the S21m modulus noise")
    parser.add_argument("--full-norm", action="store_true",
                        help="biased tracers: normalise the S21m modulus spectra by the one-loop first-layer variance "
                             "(with the S1m counterterms) instead of the tree-level one")
    parser.add_argument("--response-amplitude", action="store_true",
                        help="free S21m response amplitude a_R_j per first-layer scale (r -> (1 + a_R_j) r)")
    parser.add_argument("--k2-stochastic", action="store_true",
                        help="biased tracers: isotropic scale-dependent stochastic term alpha_k2 k^2 V/N (halo exclusion)")
    parser.add_argument("--s21-white", action="store_true",
                        help="free K-independent term W_j of the S21m modulus spectra per first-layer scale "
                             "(connected noise of the moduli)")
    parser.add_argument("--s21-k2", action="store_true",
                        help="free K^2 term WK2_j of the S21m modulus spectra per first-layer scale (with or without --s21-white)")
    parser.add_argument("--s21-connected", type=Path, default=None,
                        help="npz of per-entry connected-noise corrections c_e (connected_noise_correction.py)")
    parser.add_argument("--s21-connected-fog", type=float, default=None,
                        help="c_FoG: use c_e - c_FoG c_e^fog (an npz made with connected_noise_correction.py --fog)")
    parser.add_argument("--s21-noise-calibration", type=Path, default=None,
                        help="npz of the simulation-calibrated S21m noise (noise_calibration.py): the per-field noise "
                             "amplitudes and W follow the trial cosmology instead of being free")
    parser.add_argument("--s21-connected-amplitude", action="store_true",
                        help="free amplitude A_c_j of the connected-noise correction per first-layer scale")
    parser.add_argument("--s21-layer2", type=Path, default=None,
                        help="template_rsd.json (scripts/layer2/template_rsd.py): non-Gaussian second layer, S21m x "
                             "(1 + d0 + R_l d2), R_l = 1 / (1 + c_l n s_lin^2)")
    parser.add_argument("--s21-layer2-c", choices=("halo", "matter"), default="halo",
                        help="c_l calibrated on the halo pairs outside the data vector, or transferred from matter")
    parser.add_argument("--s21-layer2-marg", choices=("global", "per_l"), default=None,
                        help="marginalise analytically over the amplitudes of the template pieces d0 and R d2 (one pair, or "
                             "one pair per l), Gaussian priors centred on the prediction (A = 1)")
    parser.add_argument("--s21-layer2-marg-width", type=float, default=1.0,
                        help="width of the amplitude priors, as a fraction of the predicted amplitude")
    parser.add_argument("--s21-layer2-prior", type=float, default=None, metavar="FRACTION",
                        help="free c_l with a Gaussian prior of this fractional width around the calibration (default: fixed)")
    parser.add_argument("--kappa-shot", action="store_true",
                        help="biased tracers: free amplitudes A_TN, A_TN2 (priors N(1, 1)) of the N and N^2 Poisson pieces "
                             "of the tree trispectrum (S1m Edgeworth and the S21m one-point factor)")
    parser.add_argument("--cubic-biases", choices=("zero", "relation"), default=None,
                        help="biased tracers: vary the cubic biases b3, bG3, bdG2 of the S1m tree trispectrum, with "
                             "Gaussian priors centred on zero or on the halo relations of bias.cubic_priors (then the "
                             "parameters are the offsets db3, dbG3, dbdG2); needs an emulator trained with --cubic-biases")
    parser.add_argument("--shared-counterterms", action="store_true",
                        help="one set of counterterms (c0, c2, c4) for P_l and S1m: both are maps of the same P_s")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def load_basis(path: Path, vary, prepare=None):
    """RSDBasis emulator as a calculator and its settings; ``prepare(emulator, settings)`` acts on the emulator first
    (e.g. ``contract_s21m``), and its result is kept as settings["prepared"]."""
    from desilike.emulators import Emulator

    settings = json.loads(path.with_suffix(".json").read_text())
    if settings.get("stat") != "rsd":
        raise ValueError(f"{path} is not a redshift-space (RSDBasis) emulator")
    emulator = Emulator.read(str(path))
    if prepare is not None:
        settings["prepared"] = prepare(emulator, settings)
    basis = emulator.to_calculator()
    _bound_to_emulator(basis, _fix_unvaried(basis, settings, vary, path))
    return basis, settings


def s21m_selection(args, config, settings):
    """S21 coefficients of the fit and those of the emulator (in its order)."""
    coefficients = [c for c in select_coefficients(config, s21_min_scale=args.s21_min_scale,
                                                   s21_min_ratio=args.s21_min_ratio,
                                                   s21_min_scale2=args.s21_min_scale2) if c.kind == "S21"]
    emulated21 = [c for c in select_coefficients(config, s21_min_scale=0.0, s21_min_ratio=0.0)
                  if c.kind == "S21" and c.label in settings.get("s21_coefficients", [])]
    emulated21.sort(key=lambda c: settings["s21_coefficients"].index(c.label))
    return coefficients, emulated21


def contract_unprojected_s21m(args, config, emulator, settings):
    """For an emulator of the unprojected S21m basis and an S21m fit: fold the S21m projection (its AP polynomial with
    --ap) into the emulator (``contract_s21m``), so that a likelihood call costs what a projected emulator does."""
    from numpy.polynomial.legendre import leggauss

    from wstfast.calculators import contract_s21m
    from wstfast.theory.rsd_moduli import S21mAPProjection, S21mProjection, first_layer_fields

    if settings.get("s21_projected", True) or "s21m" not in args.stats or args.s21_connected:
        return None
    coefficients, emulated21 = s21m_selection(args, config, settings)
    if any(c not in emulated21 for c in coefficients):
        return None  # build_problem reports the missing terms
    x, _ = leggauss(12)  # the (K, mu_K) grid of BiasedRSDS21mBasis
    boxsize, nmesh = settings.get("lattice", [1000.0, 256])
    grid = dict(k=np.geomspace(4e-3, 0.12, 24), mu=x[6:], boxsize=float(boxsize), nmesh=int(nmesh))
    fields = first_layer_fields(coefficients)
    projection = (APPolynomial(S21mAPProjection(config, coefficients, fields, grid["k"], grid["mu"],
                                                boxsize=grid["boxsize"], nmesh=grid["nmesh"]))
                  if args.ap else S21mProjection(config, coefficients, fields, grid["k"], grid["mu"],
                                                 boxsize=grid["boxsize"], nmesh=grid["nmesh"]))
    return contract_s21m(emulator, coefficients, emulated21, projection)


def apply_gaussian_priors(basis, settings, items):
    """Gaussian priors ['n_s=0.9649,0.042', ...] on varied emulated parameters, within the emulator bounds."""
    params = get_params(basis)
    for item in items:
        name, _, values = item.partition("=")
        loc, scale = (float(v) for v in values.split(","))
        if name not in settings["vary"] or params[name].fixed:
            raise ValueError(f"--gaussian-prior: {name} is not a varied emulated parameter")
        params[name].update(prior=dict(dist="norm", loc=loc, scale=scale, limits=list(settings["bounds"][name])))


def load_correction(args, config, coefficients, settings):
    """Control-variate correction of the emulated S1m cumulants at the Quijote cosmology (cached next to the emulator)."""
    from cosmoprimo import Cosmology

    from wstfast.theory.rsd import cumulant_correction
    from wstfast.theory.rsd_bias import biased_cumulant_correction

    if settings.get("tracer", "matter") == "biased":
        cumulant_correction = biased_cumulant_correction  # noqa: F811
    cache = args.emulator.with_suffix(f".correction_{args.cumulant_points}.npz")
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        labels = list(stored["labels"])
        index = [labels.index(c.label) for c in coefficients]
        return _tile_ap(settings, stored["kappa"][index], stored["skewness2"][index])
    every = [c for c in select_coefficients(config, s1_min_scale=settings["s1_min_scale"]) if c.kind == "S1"]
    fourier = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY).get_fourier()
    klin = np.geomspace(1e-4, 10.0, 1024)
    pklin = fourier.pk_interpolator(of="delta_m")(klin, z=settings["z"])
    f = float(fourier.sigma8_z(settings["z"], of="theta_cb") / fourier.sigma8_z(settings["z"], of="delta_cb"))
    cubic = dict(cubic=True) if settings.get("cubic_biases") else {}
    if settings.get("fog"):
        cubic["fog"] = True
    kappa, skewness2 = cumulant_correction(config, every, klin, pklin, f, npoints_reference=args.cumulant_points, **cubic)
    np.savez(cache, kappa=kappa, skewness2=skewness2, labels=np.array([c.label for c in every]))
    print(f"wrote {cache}")
    labels = [c.label for c in every]
    index = [labels.index(c.label) for c in coefficients]
    return _tile_ap(settings, kappa[index], skewness2[index])


def _tile_ap(settings, kappa, skewness2):
    """The control-variate correction (computed without AP) for every AP-polynomial term of an --ap-order emulator."""
    from wstfast.theory.rsd_bias import AP_TERMS

    n = AP_TERMS[settings.get("ap_order", 0)]
    return np.tile(kappa, n), np.tile(skewness2, n)


def connected_correction(args):
    """Per-entry connected-noise correction c_e of the emulator's S21m entries (--s21-connected, --s21-connected-fog)."""
    if args.s21_connected is None:
        return None
    stored = np.load(args.s21_connected)
    if args.s21_connected_fog is None:
        return stored["correction"]
    return stored["correction"] - args.s21_connected_fog * stored["correction_fog"]


def load_layer2(args, projection, coefficients, first_layer, q, kwargs):
    """Template of the non-Gaussian second layer for S21mTheory(layer2=...): d0 and d2 per (j1, j2, l, |m1|, |m2|),
    the matter linear variances of the first-layer fields, the c_l (halo-calibrated or transferred from matter), and the
    fiducial linear block variances at the template's biases, from the basis evaluated at the Quijote cosmology."""
    from desilike import build

    from wstfast.calculators import S21mTheory, biased_rows, rsd_rows
    from wstfast.theory.bias import beta_vector, monomials
    from wstfast.theory.rsd_bias import TREE as TREE_INDEX
    from wstfast.theory.rsd_moduli import first_layer_fields

    template = layer2_template(args)
    rows = {tuple(r["key"]): (r["gauss"], r["tree"]) for r in template["rows"]}
    var = {(r["key"][0], r["key"][2]): r["var"] for r in template["rows"]}
    which = args.s21_layer2_c
    c = {int(k): v for k, v in template[f"{which}_c"].items()}
    c_err = {int(k): v for k, v in template.get(f"{which}_c_err", {}).items()} or {ell: 0.1 * abs(v) for ell, v in c.items()}
    b = template["biases"]
    beta = np.asarray(beta_vector(b["b1"], b["b2"], b["bG2"], b["bGamma3"]))
    probe = S21mTheory(projection, coefficients, first_layer, q=q, **kwargs)
    graph = build(probe)
    fiducial = {n: QUIJOTE_COSMOLOGY[n] for n in get_params(probe).select(varied=True, derived=False).names()
                if n in QUIJOTE_COSMOLOGY}
    graph(fiducial)
    if kwargs.get("tracer") == "biased":
        basis_rows = biased_rows(probe.basis)[0]
        tree = np.tensordot(np.asarray(monomials(beta, 2))[np.array(TREE_INDEX)], np.asarray(basis_rows[:len(TREE_INDEX)]), axes=1)
    else:
        tree = np.asarray(rsd_rows(probe.basis)[0][0])
    blocks = np.asarray(probe.matrix) @ tree.ravel()
    offsets = np.cumsum([0] + [f.ell + 1 for f in first_layer])
    lin_ref = {(f.j, f.ell, m1): float(blocks[offsets[i] + m1]) for i, f in enumerate(first_layer) for m1 in range(f.ell + 1)}
    print(f"non-Gaussian second layer: {which} c_l = " + ", ".join(f"{ell}: {v:.2f}" for ell, v in sorted(c.items()))
          + ("" if args.s21_layer2_prior is None else f" (free, {args.s21_layer2_prior:g} prior)"), flush=True)
    return dict(rows=rows, var=var, c=c, c_err=c_err, prior=args.s21_layer2_prior, beta_ref=beta.tolist(), lin_ref=lin_ref)


#: The measurement settings of the layer-2 template (scripts/layer2/template_rsd.py: the halo superset), whose (j1, j2)
#: keys are re-keyed by sigma for a fit of another configuration (a survey mesh).
TEMPLATE_CONFIG = WSTConfig(J=9, L=6, L2=4, sigma0=0.8, cellsize=1000.0 / 256, step=2**0.5, min_dj=2)


def layer2_template(args):
    """The --s21-layer2 template, with its rows keyed in the fit's configuration (``args.layer2_config``, set by
    build_problem): the template's scale indices mapped by sigma (rows of scales the fit lacks dropped)."""
    template = json.loads(args.s21_layer2.read_text())
    config = getattr(args, "layer2_config", None)
    if config is None or (np.isclose(config.sigma(0), TEMPLATE_CONFIG.sigma(0)) and config.step == TEMPLATE_CONFIG.step):
        return template

    def index(j):
        sigma = TEMPLATE_CONFIG.sigma(j)
        jj = int(round(np.log(sigma / config.sigma(0)) / np.log(config.step)))
        return jj if 0 <= jj <= config.J and np.isclose(config.sigma(jj), sigma, rtol=1e-6) else None

    rows = []
    for r in template["rows"]:
        j1, j2 = index(r["key"][0]), index(r["key"][1])
        if j1 is not None and j2 is not None:
            rows.append(dict(r, key=[j1, j2] + list(r["key"][2:])))
    return dict(template, rows=rows)


def layer2_templates(args, labels, mean):
    """Data-vector shapes S21m d0 and S21m R_l d2 of the template pieces (fiducial cosmology, calibrated c_l), all l
    together ('global') or one pair per l ('per_l'); zero outside S21m."""
    template = layer2_template(args)
    c = {int(k): max(v, 0.0) for k, v in template[f"{args.s21_layer2_c}_c"].items()}
    rows = {tuple(r["key"]): r for r in template["rows"]}
    groups = {}
    for i, label in enumerate(labels):
        label = label.split(":")[-1]  # without the cap / q prefixes
        if not label.startswith("S21m"):
            continue
        key = tuple(int(part[1:]) for part in label.split("_")[1:])
        r = rows[key]
        group = 0 if args.s21_layer2_marg == "global" else r["ell"]
        d0, d2 = groups.setdefault(group, (np.zeros(len(labels)), np.zeros(len(labels))))
        d0[i] = mean[i] * r["gauss"]
        d2[i] = mean[i] * r["tree"] / (1.0 + c[r["ell"]] * r["var"])
    return [v for pair in groups.values() for v in pair]


def layer2_variance(args, labels, mean):
    """(S21m dE)^2 per data point, dE^2 = dd0^2 + (R_l dd2)^2 with R_l at the calibrated c_l (zero for other statistics)."""
    template = layer2_template(args)
    c = {int(k): max(v, 0.0) for k, v in template[f"{args.s21_layer2_c}_c"].items()}
    rows = {tuple(r["key"]): r for r in template["rows"]}
    out = np.zeros(len(labels))
    for i, label in enumerate(labels):
        label = label.split(":")[-1]  # without the cap / q prefixes
        if label.startswith("S21m"):
            key = tuple(int(part[1:]) for part in label.split("_")[1:])
            r = rows[key]
            renorm = 1.0 / (1.0 + c[r["ell"]] * r["var"])
            out[i] = (mean[i] * np.hypot(r["gauss_err"], renorm * r["tree_err"])) ** 2
    return out


def fiducial_omega_b(settings) -> float:
    """omega_b of the emulator's fiducial (fixed in the AP ratios), with the massive neutrinos of AbacusSummit (matter at
    the redshifts of the fits)."""
    from wstfast.config import ABACUS_COSMOLOGY

    if settings.get("fiducial") == "abacus":
        return ABACUS_COSMOLOGY["omega_b"] + 0.0006442  # omega_ncdm of one 0.06 eV neutrino
    return QUIJOTE_COSMOLOGY["omega_b"]


def build_problem(args, basis=None, settings=None, space="rsd", window=None, prefix=""):
    """Theories, data vectors (nrealizations, ndata), labels and metadata of ``args.stats``, from the emulator
    ``args.emulator`` (or the given ``basis`` and ``settings``); also returns the basis, its settings and the tracer.
    Survey (``fit_rsd.py --survey``): ``space`` is the cap's data subdirectory, ``window`` its ``survey.SurveyWindow``
    and ``prefix`` that of its labels; the projections use the emulator's lattice (``settings["lattice"]``)."""
    first = sorted((args.data_dir / space).glob("wst_r*.npz"))[0]
    q0 = args.q[0]
    config = load_measurement(first, q=q0)["config"]
    args.layer2_config = config  # the layer-2 template is re-keyed to it (layer2_template)
    if basis is None:
        basis, settings = load_basis(args.emulator, args.vary, prepare=lambda e, st: contract_unprojected_s21m(
            args, config, e, st))
        apply_gaussian_priors(basis, settings, args.gaussian_prior)
    tracer = settings.get("tracer", "matter")
    cubic_biases = getattr(args, "cubic_biases", None)
    if cubic_biases is not None and not settings.get("cubic_biases"):
        raise SystemExit(f"--cubic-biases needs an emulator trained with --cubic-biases ({args.emulator} is not)")
    fog = getattr(args, "fog", None)
    if bool(fog) != bool(settings.get("fog")):
        raise SystemExit("--fog needs an emulator trained with --fog, and such an emulator needs --fog")
    grid = RSDGrid(kmax=settings["kmax"])
    emulated = settings["coefficients"]
    boxsize, nmesh = settings.get("lattice", [1000.0, 256])  # the periodic lattice the emulator describes
    lattice = dict(boxsize=float(boxsize), nmesh=int(nmesh))
    ap = APGeometry(basis, settings["z"], omega_b=fiducial_omega_b(settings)) if args.ap else None

    theories, vectors, labels = [], [], []
    files = None
    if getattr(args, "realizations_like", None) is not None:
        ids = {path.name for path in (args.realizations_like / space).glob("wst_r*.npz")}
        files = [path for path in sorted((args.data_dir / space).glob("wst_r*.npz")) if path.name in ids]
        if not files:
            raise SystemExit(f"no realizations of {args.data_dir} in {args.realizations_like}")
    if getattr(args, "exclude_like", None) is not None:
        ids = {path.name for path in (args.exclude_like / space).glob("wst_r*.npz")}
        files = [path for path in (files or sorted((args.data_dir / space).glob("wst_r*.npz"))) if path.name not in ids]
    if "s1m" in args.stats:
        coefficients = [c for c in select_coefficients(config, s1_min_scale=args.s1_min_scale, s1_ells=args.s1_ells)
                        if c.kind == "S1"]
        missing = [c.label for c in coefficients if c.label not in emulated]
        if missing:
            raise ValueError(f"the emulator has no cumulants for {missing}")
        correction = load_correction(args, config, coefficients, settings) if args.cumulant_points else None
        projection = (APPolynomial(s1m_ap_projection(config, coefficients, grid, **lattice)) if args.ap
                      else S1mProjection(config, coefficients, grid, **lattice))
        for q in args.q:
            s1m = load_s1m_dataset(args.data_dir, space, coefficients, q=q, files=files)
            files = s1m.files
            theories.append(S1mTheory(projection, coefficients, q=q, basis=basis, shotnoise=s1m.shotnoise,
                                      cumulant_index=[emulated.index(c.label) for c in coefficients],
                                      correction=correction, ng_amplitude=args.ng_amplitude, ap=ap, tracer=tracer,
                                      sigmas=[config.sigma(c.j) for c in coefficients],
                                      k4_counterterm=args.k4_counterterm, k2_stochastic=args.k2_stochastic,
                                      cubic_biases=cubic_biases, kappa_shot=args.kappa_shot, fog=fog, window=window))
            vectors.append(s1m.vectors)
            qprefix = f"q{q:g}:" if len(args.q) > 1 else ""
            labels += [f"{prefix}{qprefix}S1m_j{c.j}_l{c.ell}_m{m}" for c in coefficients for m in range(c.ell + 1)]
        meta = s1m.metadata
    if "s21m" in args.stats:
        from wstfast.theory.rsd_moduli import s21m_entries

        unprojected = not settings.get("s21_projected", True)
        if args.ap and not unprojected and not settings.get("ap_order"):
            raise SystemExit("--ap for s21m needs an emulator with the unprojected S21m basis (--s21-unprojected) "
                             "or with AP-polynomial terms (--ap-order)")
        coefficients = [c for c in select_coefficients(config, s21_min_scale=args.s21_min_scale,
                                                       s21_min_ratio=args.s21_min_ratio,
                                                       s21_min_scale2=args.s21_min_scale2) if c.kind == "S21"]
        emulated21 = [c for c in select_coefficients(config, s21_min_scale=0.0, s21_min_ratio=0.0)
                      if c.kind == "S21" and c.label in settings.get("s21_coefficients", [])]
        emulated21.sort(key=lambda c: settings["s21_coefficients"].index(c.label))
        missing = [c.label for c in coefficients if c not in emulated21]
        if missing:
            raise ValueError(f"the emulator has no S21m terms for {missing}")
        first_layer = [Coefficient("S1", ell, j) for j, ell in sorted({(c.j, c.ell) for c in coefficients})]
        missing = [c.label for c in first_layer if c.label not in emulated]
        if missing:
            raise ValueError(f"the emulator has no cumulants for the first-layer {missing}")
        basis_entries = {(emulated21[i].label, m1, m2): n for n, (i, m1, m2) in enumerate(s21m_entries(emulated21))}
        entry_index = [basis_entries[coefficients[i].label, m1, m2] for i, m1, m2 in s21m_entries(coefficients)]
        s21m = load_s21m_dataset(args.data_dir, space, coefficients, q=q0, files=files)
        files = s21m.files
        s21m_kwargs = dict(basis=basis, entry_index=entry_index,
                           cumulant_index=[emulated.index(c.label) for c in first_layer],
                           shotnoise=s21m.shotnoise, tracer=tracer, bispectrum_shot=args.b_shot,
                           noise_shot=args.noise_shot, full_norm=args.full_norm,
                           response_amplitude=args.response_amplitude, k2_stochastic=args.k2_stochastic,
                           white_noise=args.s21_white, config=config, k2_noise=args.s21_k2,
                           noise_correction=connected_correction(args), connected_amplitude=args.s21_connected_amplitude,
                           noise_calibration=(dict(np.load(args.s21_noise_calibration))
                                              if args.s21_noise_calibration is not None else None),
                           cubic_biases=cubic_biases, kappa_shot=args.kappa_shot, fog=fog, window=window)
        if settings.get("ap_order") and args.ap:  # AP-polynomial terms of the basis, at the trial AP ratios
            s21m_kwargs.update(ap=ap, first_ap=APPolynomial(s1m_ap_projection(config, first_layer, grid, **lattice)))
        if unprojected and settings.get("prepared") is not None:  # emulator contracted at load (contract_s21m)
            s21m_kwargs.update(entry_index=None, noise_correction=None, ap=ap, s21_projection=settings["prepared"])
        elif unprojected:  # lambda_2 projected at assembly from the emulated field grids, with AP if --ap
            from numpy.polynomial.legendre import leggauss

            from wstfast.theory.rsd_moduli import S21mAPProjection, S21mProjection, first_layer_fields

            if s21m_kwargs["noise_correction"] is not None:
                raise SystemExit("--s21-connected is not implemented for the unprojected S21m basis")
            fields, basis_fields = first_layer_fields(coefficients), first_layer_fields(emulated21)
            x, _ = leggauss(12)  # the (K, mu_K) grid of BiasedRSDS21mBasis
            s21m_kwargs.update(entry_index=None, noise_correction=None, ap=ap,
                               s21_projection=(S21mAPProjection if args.ap else S21mProjection)(
                                   config, coefficients, fields, np.geomspace(4e-3, 0.12, 24), x[6:],
                                   boxsize=float(boxsize), nmesh=int(nmesh)),
                               s21_field_rows=[basis_fields.index(field) for field in fields])
        layer2 = (load_layer2(args, S1mProjection(config, first_layer, grid, **lattice), coefficients, first_layer, q0,
                              s21m_kwargs) if getattr(args, "s21_layer2", None) is not None else None)
        theories.append(S21mTheory(S1mProjection(config, first_layer, grid, **lattice), coefficients, first_layer, q=q0,
                                   layer2=layer2, **s21m_kwargs))
        vectors.append(s21m.vectors)
        labels += [f"{prefix}S21m_j{coefficients[i].j}_j{coefficients[i].j2}_l{coefficients[i].ell}_m{m1}_m{m2}"
                   for i, m1, m2 in s21m_entries(coefficients)]
        meta = s21m.metadata
    if "pk" in args.stats:
        power = load_power_dataset(args.data_dir, space, kmin=args.kmin, kmax=args.kmax, rebin=args.rebin, files=files,
                                   ells=(0, 2, 4))
        files = power.files
        projection = (APPolynomial(multipole_ap_projection(power.edges, grid, **lattice)) if args.ap
                      else MultipoleProjection(power.edges, grid, **lattice))
        theories.append(RSDPowerTheory(projection, basis=basis, shotnoise=power.shotnoise, ap=ap, tracer=tracer,
                                       shared_counterterms=args.shared_counterterms,
                                       k4_counterterm=args.k4_counterterm, k2_stochastic=args.k2_stochastic))
        vectors.append(power.vectors)
        labels += [f"{prefix}P{ell}_k{k:.4f}" for ell in (0, 2, 4) for k in power.k]
        meta = power.metadata
    return dict(theories=theories, vectors=np.concatenate(vectors, axis=1), labels=labels, meta=meta, basis=basis,
                settings=settings, tracer=tracer)


def box_entries(box_dir: Path, labels, config: WSTConfig, q: float) -> tuple[np.ndarray, float]:
    """Box realizations (nbox, ndata) of the S1m / S21m entries ``labels`` (of a fit of ``config``, prefixes ignored),
    matched by sigma in the box measurements of ``box_dir``/rsd, and the box volume [(Gpc/h)^3]."""
    files = sorted((Path(box_dir) / "rsd").glob("wst_r*.npz"))
    first = load_measurement(files[0], q=q)
    box_config = first["config"]

    def j_box(j):
        sigma = config.sigma(j)
        matches = [jb for jb in range(box_config.J + 1) if np.isclose(box_config.sigma(jb), sigma, rtol=1e-6)]
        if not matches:
            raise SystemExit(f"no box scale for sigma = {sigma:.2f} Mpc/h in {box_dir}")
        return matches[0]

    keys = []
    for label in labels:
        parts = label.split(":")[-1].split("_")
        if parts[0] == "S1m":
            keys.append(("S1m", j_box(int(parts[1][1:])), int(parts[2][1:]), int(parts[3][1:])))
        elif parts[0] == "S21m":
            keys.append(("S21m", j_box(int(parts[1][1:])), j_box(int(parts[2][1:])), int(parts[3][1:]),
                         int(parts[4][1:]), int(parts[5][1:])))
        else:
            raise SystemExit(f"survey fits support S1m and S21m only ({label})")
    vectors = []
    for path in files:
        m = load_measurement(path, q=q)
        vectors.append([m["S1m"][k[1], k[2], k[3]] if k[0] == "S1m"
                        else m["S2m"][k[1], k[2], k[3], k[4], k[5]] / m["S1m"][k[1], k[3], k[4]] for k in keys])
    return np.array(vectors), (first["metadata"]["boxsize"] / 1000.0) ** 3


def survey_problem(args):
    """Theories, data, labels and covariance of a survey fit (--survey): one window per cap, shared parameters."""
    from wstfast.data import sample_covariance as sample
    from wstfast.survey import SurveyWindow

    if args.window is None or len(args.window) != len(args.survey):
        raise SystemExit("--survey needs one --window file per cap")
    if args.covariance_data_dir is None:
        raise SystemExit("--survey needs --covariance-data-dir (box realizations for the covariance)")
    basis = settings = None
    caps = []
    for cap, path in zip(args.survey, args.window):
        window = SurveyWindow.load(path)
        if window.veff is None:
            raise SystemExit(f"{path} has no effective volume (rebuild it with dr1_window.py --test products)")
        problem = build_problem(args, basis=basis, settings=settings, space=cap, window=window, prefix=f"{cap}:")
        basis, settings = problem["basis"], problem["settings"]
        caps.append((cap, window, problem))
    nreal = {p["vectors"].shape[0] for _, _, p in caps}
    if len(nreal) != 1:
        raise SystemExit(f"the caps have different numbers of realizations: {nreal}")
    nreal = nreal.pop()
    theories = [t for _, _, p in caps for t in p["theories"]]
    vectors = np.concatenate([p["vectors"] for _, _, p in caps], axis=1)
    labels = [label for _, _, p in caps for label in p["labels"]]
    blocks, nsims = [], None
    for cap, window, problem in caps:
        box, vbox = box_entries(args.covariance_data_dir, problem["labels"], args.layer2_config, args.q[0])
        nsims = box.shape[0]
        blocks.append(sample(box) * vbox / window.veff / (nreal if args.covariance_of_mean else 1))
    covariance = np.zeros((len(labels), len(labels)))
    start = 0
    for block in blocks:
        covariance[start:start + len(block), start:start + len(block)] = block
        start += len(block)
    if getattr(args, "s21_layer2", None) is not None:  # the template's error is the same in every cap
        mean = vectors.mean(axis=0)
        error = np.sqrt(layer2_variance(args, labels, mean))
        base = np.array([label.split(":", 1)[-1] for label in labels])
        covariance = covariance + np.where(base[:, None] == base[None, :], np.outer(error, error), 0.0)
        if args.s21_layer2_marg is not None:
            for vector in layer2_templates(args, labels, mean):
                covariance = covariance + args.s21_layer2_marg_width**2 * np.outer(vector, vector)
    meta = dict(caps[0][2]["meta"], caps=list(args.survey), veff={cap: w.veff for cap, w, _ in caps})
    return dict(theories=theories, vectors=vectors, labels=labels, meta=meta, basis=basis, settings=settings,
                tracer=caps[0][2]["tracer"], covariance=covariance, nsims=nsims)


def main():
    args = parse_args()
    setup_logging()
    if args.volume is not None and args.covariance_of_mean:
        raise SystemExit("--volume and --covariance-of-mean are exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    problem = survey_problem(args) if args.survey else build_problem(args)
    theories, vectors, labels, meta, tracer = (problem[k] for k in ("theories", "vectors", "labels", "meta", "tracer"))
    nreal, ndata = vectors.shape
    nsims = nreal  # realizations of the covariance estimate
    if args.survey:  # box covariance at each cap's effective volume, with the layer-2 terms (survey_problem)
        covariance, nsims = problem["covariance"], problem["nsims"]
    elif args.covariance_data_dir is None:
        covariance = sample_covariance(vectors, of_mean=args.covariance_of_mean)
    else:  # one-box covariance of another set; of the mean of this set's nreal boxes with --covariance-of-mean
        import copy

        other = copy.copy(args)
        other.data_dir, other.realizations_like, other.exclude_like = args.covariance_data_dir, None, None
        cov_vectors = build_problem(other, basis=problem["basis"], settings=problem["settings"])["vectors"]
        nsims = cov_vectors.shape[0]
        covariance = sample_covariance(cov_vectors)
        if args.covariance_of_mean:
            covariance = covariance / nreal
    if args.volume is not None and not args.survey:
        covariance = covariance * (meta["boxsize"] / 1000.0) ** 3 / args.volume
    if getattr(args, "s21_layer2", None) is not None and not args.survey:  # Monte Carlo error of the template, a volume-independent theory error
        covariance = covariance + np.diag(layer2_variance(args, labels, vectors.mean(axis=0)))
        if args.s21_layer2_marg is not None:  # analytic marginalisation over the template amplitudes
            for vector in layer2_templates(args, labels, vectors.mean(axis=0)):
                covariance = covariance + args.s21_layer2_marg_width**2 * np.outer(vector, vector)
    for path in args.extra_covariance:  # modelling errors in data units, matched by label
        extra = np.load(path)
        index = {str(label): i for i, label in enumerate(extra["labels"])}
        missing = [label for label in labels if label not in index]
        if missing:
            raise SystemExit(f"{path} has no covariance for {missing[:5]}...")
        rows = np.array([index[label] for label in labels])
        covariance = covariance + extra["covariance"][np.ix_(rows, rows)]
    print(f"{ndata} data points ({', '.join(args.stats)}), {nreal} realizations")

    theory = theories[0] if len(theories) == 1 else JointTheory(theories)
    fixed = parse_fixed(args.fix)
    nparams = len([n for n in get_params(theory).select(varied=True, derived=False).names() if n not in fixed])
    m1 = 1.0 if args.no_percival else percival_m1(nsims, ndata, nparams)
    print(f"Hartlap factor {hartlap_factor(nsims, ndata):.4f} ({nsims} realizations, {ndata} data points); "
          f"Percival m1 {m1:.4f} for {nparams} parameters", flush=True)
    likelihood = WSTLikelihood(theory, vectors.mean(axis=0), covariance * m1)
    fix_parameters(likelihood, fixed)
    posterior = Posterior(likelihood)
    profiles = profile(posterior, args.output_dir / "profiles.h5", seed=args.seed)
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
    summary = dict(bestfit=best, chi2=chi2, ndata=int(ndata), nvaried=len(varied), nrealizations=int(nreal),
                   stats=args.stats, vary=args.vary, volume=args.volume, covariance_of_mean=args.covariance_of_mean,
                   kmax=args.kmax, ap=args.ap, s21_min_scale=args.s21_min_scale,
                   s21_min_scale2=args.s21_min_scale2, s1_min_scale=args.s1_min_scale, labels=labels, emulator=str(args.emulator),
                   tracer=tracer, shared_counterterms=args.shared_counterterms, k4_counterterm=args.k4_counterterm, b_shot=args.b_shot, noise_shot=args.noise_shot, full_norm=args.full_norm, response_amplitude=args.response_amplitude, k2_stochastic=args.k2_stochastic, s21_white=args.s21_white, s21_k2=args.s21_k2, s21_connected=str(args.s21_connected), s21_connected_fog=args.s21_connected_fog, s21_connected_amplitude=args.s21_connected_amplitude, cubic_biases=args.cubic_biases, kappa_shot=args.kappa_shot, fog=args.fog, sampler=args.sampler, gaussian_prior=args.gaussian_prior, covariance_data_dir=str(args.covariance_data_dir), realizations_like=str(args.realizations_like), exclude_like=str(args.exclude_like), s21_noise_calibration=str(args.s21_noise_calibration), s21_layer2=str(args.s21_layer2), s21_layer2_c=args.s21_layer2_c, s21_layer2_prior=args.s21_layer2_prior, s21_layer2_marg=args.s21_layer2_marg, s21_layer2_marg_width=args.s21_layer2_marg_width, hartlap_factor=hartlap_factor(nsims, ndata), percival_m1=m1, nsims=int(nsims), shotnoise=meta["boxsize"] ** 3 / meta["nparticles"],
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
