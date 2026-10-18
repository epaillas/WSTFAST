"""Regression fixtures of the code-release refactor (docs/refactor_plan.md, step 0), frozen from the code of 2026-10-18.

- ``halo``: the halo baseline of the paper (scripts/fit_halo_baseline.sh: S1m + S21m with c_v, cubic biases, layer-2
  template and AP, on the ap3_base emulator), plus the P(k) multipoles from the same emulator, at the fiducial point
  and at the best fit of outputs/inference/rsd/fullap_ap_V60. The refactor must reproduce these to round-off
  (the emulators stay valid, D2).
- ``basis``: one non-emulated biased RSDBasis (fog, cubic, AP order 3, IR) at the Quijote cosmology, on a few
  coefficients: the pure model code, without the emulator. Also to round-off.
- ``real``: the real-space matter WSTMatterBasis at the Quijote cosmology and its Assembly at fixed nuisances, the
  reference the f = 0 limit is compared with in step 3 (agreement within the measurement errors, not round-off).
  ``test_real_space_is_the_f0_limit`` checks the f = 0 limit of the redshift-space basis against it.
- ``matter_rsd``: the redshift-space MATTER classes (RSDS1Basis cumulants, RSDS21mBasis, the tracer="matter"
  theories), frozen before step 2 deleted them, on the small configurations of test_bias.py: the matter limit of the
  biased code (beta = (1, 1, 0, 0, 0), no shot noise) must reproduce them. Not regenerable from this file.

The tests skip when the emulators or measurements (outside git) are missing. Regenerate only on purpose:

    python tests/test_regression.py [halo] [basis] [real]
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
HALO_DATA = ROOT / "data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los"
HALO_EMULATOR = ROOT / "outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic_ap3_base.h5"
LAYER2_TEMPLATE = ROOT / "outputs/layer2/template_rsd.json"
#: Best fit of the baseline at 60 (Gpc/h)^3 (outputs/inference/rsd/fullap_ap_V60/summary.json), frozen here.
HALO_BESTFIT = {
    "omega_cdm": 0.11855275407688619, "logA": 3.0809323306353344, "h": 0.6715176079415945, "c_v": 0.9404955575032141,
    "b1": 1.9200740815717343, "b2": -0.4940009277812223, "bG2": -0.49209491995535704, "bGamma3": 0.46128970092937305,
    "alpha0": 0.10299975170922426, "alpha2": -2.7310008046711545, "db3": 0.7339378203766374,
    "dbG3": 0.0712432728914183, "dbdG2": -0.03894042659402053, "c0": 24.63512573192721, "c2": 29.263813704946543,
    "c4": -9.57378295304558, "W_j6": 411.36979279931955, "noise_j6_l1_m0": -0.02749080596434006,
    "noise_j6_l1_m1": -0.059922886482247306, "noise_j6_l2_m0": 0.008454522273308451,
    "noise_j6_l2_m1": -0.0425740807624293, "noise_j6_l2_m2": -0.012042269648838033,
    "noise_j6_l3_m0": 0.03493221518682843, "noise_j6_l3_m1": -0.0273887256426818,
    "noise_j6_l3_m2": -0.03685271942439583, "noise_j6_l3_m3": 0.0019909298057141367,
    "noise_j6_l4_m0": 0.050193335862497634, "noise_j6_l4_m1": -0.012801198667800162,
    "noise_j6_l4_m2": -0.044104909582148855, "noise_j6_l4_m3": -0.04026501990797049,
    "noise_j6_l4_m4": 0.008971587731911201,
    # P(k) nuisances (not in the WST baseline): representative halo values
    "c0_pk": 5.0, "c2_pk": 10.0, "c4_pk": 0.0,
}
#: Coefficients of the non-emulated basis fixture: every l of the first layer at sigma_j6 = 25 Mpc/h, one larger scale,
#: and one S21 entry.
BASIS_S1 = ["S1_j6_l0", "S1_j6_l1", "S1_j6_l2", "S1_j6_l3", "S1_j6_l4", "S1_j8_l2"]
BASIS_S21 = ["S21_j6_j9_l2"]
REAL_DATA = ROOT / "data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256"
REAL_SHOTNOISE = 7.450580596923828  # (1000 Mpc/h)^3 / 512^3, as in outputs/emulators/wst_basis_taylor_superset.json
REAL_NUISANCE = dict(cs2=0.5, noise=0.1)

RTOL = 1e-10


def _halo_args(stats):
    sys.path.insert(0, str(ROOT / "scripts"))
    import fit_rsd

    output = tempfile.mkdtemp()
    argv = ["--data-dir", str(HALO_DATA), "--emulator", str(HALO_EMULATOR), "--ap", "--stats", *stats,
            "--s21-min-scale", "25", "--s21-white", "--cubic-biases", "relation", "--s21-layer2", str(LAYER2_TEMPLATE),
            "--s21-layer2-c", "halo", "--s21-layer2-marg", "per_l", "--fog", "shared", "--vary", "omega_cdm", "logA",
            "h", "--output-dir", output]
    return fit_rsd, fit_rsd.parse_args(argv)


def halo_predictions() -> dict:
    """Baseline WST and P_l predictions of the halo model at the fiducial point and at HALO_BESTFIT."""
    from desilike import build, get_params

    from wstfast.calculators import JointTheory

    fit_rsd, args = _halo_args(["s1m", "s21m", "pk"])
    problem = fit_rsd.build_problem(args)
    theory = JointTheory(problem["theories"])
    params = get_params(theory)
    varied = params.select(varied=True, derived=False).names()
    fiducial = {name: float(params[name].value) for name in varied}
    bestfit = {name: HALO_BESTFIT.get(name, fiducial[name]) for name in varied}
    missing = sorted(set(varied) - set(HALO_BESTFIT))
    if missing:
        print(f"halo fixture: no best-fit value for {missing}, fiducial used")
    graph = build(theory, output=lambda: theory.flattheory)
    return dict(labels=np.array(problem["labels"]), fiducial=np.asarray(graph(fiducial)),
                bestfit=np.asarray(graph(bestfit)), params=json.dumps(dict(fiducial=fiducial, bestfit=bestfit)))


def basis_outputs() -> dict:
    """Outputs of a non-emulated biased RSDBasis with the settings of the ap3_base emulator, at the Quijote cosmology."""
    from desilike import build

    from wstfast.calculators import RSDBasis, build_cosmology
    from wstfast.config import WSTConfig

    settings = json.loads(HALO_EMULATOR.with_suffix(".json").read_text())
    config = WSTConfig(**settings["config"])
    basis = RSDBasis(cosmo=build_cosmology(settings["vary"]), config=config, coefficients=_coefficients(config, BASIS_S1),
                     z=settings["z"], kmax=settings["kmax"], ir=settings["ir"],
                     s21_coefficients=_coefficients(config, BASIS_S21), tracer="biased",
                     cubic_biases=settings["cubic_biases"], fog=settings["fog"], s21_lattice=settings["lattice"],
                     s21_projected=settings["s21_projected"], ap_order=settings["ap_order"])
    keys = ("grid_terms", "kappa", "skewness2", "s21m_terms", "dlogA")
    out = build(basis, output=lambda: {key: getattr(basis, key) for key in keys})({})
    return {key: np.asarray(value) for key, value in out.items()}


def _coefficients(config, labels):
    from wstfast.config import select_coefficients

    every = {c.label: c for c in select_coefficients(config, s1_min_scale=0.0, s21_min_scale=0.0, s21_min_ratio=0.0)}
    return [every[label] for label in labels]


def real_outputs() -> dict:
    """Band integrals of WSTMatterBasis (all coefficients) at the Quijote cosmology, and their Assembly at REAL_NUISANCE."""
    import jax.numpy as jnp
    from desilike import build

    from wstfast.calculators import WSTBasis
    from wstfast.data import load_measurement
    from wstfast.theory import Assembly, all_coefficients

    config = load_measurement(sorted((REAL_DATA / "real").glob("wst_r*.npz"))[0], q=0.8)["config"]
    basis = WSTBasis(config=config, shotnoise=REAL_SHOTNOISE)
    s1_terms, s21_terms = build(basis, output=lambda: (basis.s1_terms, basis.s21_terms))({})
    coefficients = all_coefficients(config)
    assembly = Assembly(config, coefficients)
    noise = jnp.full(len(assembly.noise_keys), REAL_NUISANCE["noise"])
    prediction = assembly(s1_terms, s21_terms, REAL_NUISANCE["cs2"], noise)
    return dict(labels=np.array([c.label for c in coefficients]), s1_terms=np.asarray(s1_terms),
                s21_terms=np.asarray(s21_terms), prediction=np.asarray(prediction))


def matter_rsd_setup() -> dict:
    """Configurations of the matter-limit fixture (those of test_bias.py)."""
    from wstfast.config import SUPERSET, Coefficient, select_coefficients
    from wstfast.theory.spectrum import SpectrumGrid

    s1 = [c for c in select_coefficients(SUPERSET, s1_min_scale=35.0) if c.kind == "S1" and c.ell <= 1][:3]
    s21 = [c for c in select_coefficients(SUPERSET, s21_min_scale=35.0, s21_min_ratio=2.0, s21_min_scale2=70.0)
           if c.kind == "S21" and c.ell == 1][:1]
    first = [Coefficient("S1", c.ell, c.j) for c in s21]
    return dict(f=0.75, quad=dict(npts=16, nx=8, nphi=6),
                s1_cumulants=[c for c in select_coefficients(SUPERSET, s1_min_scale=25.0)
                              if c.kind == "S1" and c.ell in (0, 2)][:3],
                s21_terms=[c for c in select_coefficients(SUPERSET, s21_min_scale=25.0, s21_min_ratio=2.8,
                                                          s21_min_scale2=70.0) if c.kind == "S21"][:1],
                s1=s1 + [c for c in first if c not in s1], s21=s21, first=first, grid=SpectrumGrid(kmax=0.2),
                edges=np.linspace(0.01, 0.15, 8))


def matter_rsd_outputs(tracer: str = "biased") -> dict:
    """The matter limit of the biased code on ``matter_rsd_setup``: cumulants and S21m terms contracted with the matter
    beta, and the theories at b1 = 1 (other biases and stochastic terms at their zero defaults)."""
    from desilike import build

    from wstfast.calculators import JointTheory, RSDBasis, RSDPowerTheory, S1mTheory, S21mTheory, build_cosmology
    from wstfast.config import SUPERSET
    from wstfast.theory.bias import MATTER_BETA, monomials
    from wstfast.theory.spectrum import MultipoleProjection, S1mProjection
    from wstfast.theory.first_layer import S1mBasis
    from wstfast.theory.layout import KAPPA_SIZES, S21_SIZES, SKEW_SIZES, split
    from wstfast.theory.second_layer import S21mBasis

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_bias import linear_power

    setup = matter_rsd_setup()
    klin, pklin = linear_power()
    f = setup["f"]
    m = lambda d: np.asarray(monomials(MATTER_BETA, d))  # noqa: E731
    kappa, skewness = S1mBasis(SUPERSET, setup["s1_cumulants"], npoints=2**10,
                                       damping="linear").cumulants(klin, pklin, f)
    terms = split(S21mBasis(SUPERSET, setup["s21_terms"], nk=6, nmu=3, **setup["quad"])(klin, pklin, f),
                  S21_SIZES)
    norm2 = (terms[6] @ m(2)) ** 2
    s1, s21, first, grid, edges = (setup[k] for k in ("s1", "s21", "first", "grid", "edges"))
    basis = RSDBasis(cosmo=build_cosmology(()), config=SUPERSET, coefficients=s1, kmax=0.2, npoints=2**10,
                     s21_coefficients=s21, tracer="biased")
    theory = JointTheory([
        S1mTheory(S1mProjection(SUPERSET, s1, grid), s1, basis=basis, tracer=tracer),
        RSDPowerTheory(MultipoleProjection(edges, grid), basis=basis, tracer=tracer),
        S21mTheory(S1mProjection(SUPERSET, first, grid), s21, first, basis=basis,
                   cumulant_index=[s1.index(c) for c in first], tracer=tracer)])
    params = {"b1": 1.0} if tracer == "biased" else {}
    return dict(kappa=split(kappa, KAPPA_SIZES)[0] @ m(4), skewness2=15 * (split(skewness, SKEW_SIZES)[0] @ m(3)) ** 2,
                s21m=np.stack([terms[0] @ m(4) / norm2, terms[3] @ m(4) / norm2], axis=1),
                prediction=np.asarray(build(theory, output=lambda: theory.flattheory)(params)))


GENERATORS = {"halo": (halo_predictions, (HALO_DATA, HALO_EMULATOR, LAYER2_TEMPLATE)),
              "basis": (basis_outputs, (HALO_EMULATOR,)),
              "real": (real_outputs, (REAL_DATA,))}


def _reference(name):
    path = FIXTURES / f"regression_{name}.npz"
    if not path.exists():
        pytest.skip(f"no fixture {path.name}")
    missing = [str(p) for p in GENERATORS[name][1] if not p.exists()]
    if missing:
        pytest.skip(f"inputs outside git missing: {missing}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.slow
def test_halo_baseline_predictions():
    reference = _reference("halo")
    current = halo_predictions()
    assert current["labels"].tolist() == reference["labels"].tolist()
    assert json.loads(str(current["params"])) == json.loads(str(reference["params"]))
    for key in ("fiducial", "bestfit"):
        np.testing.assert_allclose(current[key], reference[key], rtol=RTOL, atol=0.0, err_msg=key)


@pytest.mark.slow
def test_biased_basis_outputs():
    reference = _reference("basis")
    current = basis_outputs()
    for key, value in reference.items():
        np.testing.assert_allclose(current[key], value, rtol=RTOL, atol=1e-300, err_msg=key)


@pytest.mark.slow
def test_real_space_matter_outputs():
    """Holds only until step 3 replaces WSTMatterBasis by the f = 0 limit; the fixture stays as its reference."""
    reference = _reference("real")
    current = real_outputs()
    assert current["labels"].tolist() == reference["labels"].tolist()
    for key in ("s1_terms", "s21_terms", "prediction"):
        np.testing.assert_allclose(current[key], reference[key], rtol=RTOL, atol=1e-300, err_msg=key)


def real_space_s1(labels, shotnoise=REAL_SHOTNOISE, cs2=REAL_NUISANCE["cs2"], q=0.8):
    """The real-space limit (``space="real"``: f = 0, matter) of the redshift-space basis at the Quijote cosmology,
    m-summed for the coefficients ``labels``: the |m| block variances lambda_M and Edgeworth factors E_M of every
    coefficient, and S1 = (2 lambda)^{q/2} Gamma((n + q) / 2) / Gamma(n / 2) (1 + E) of the multiplet (n = 2l + 1),
    with lambda and E the multiplicity-weighted means over the blocks (equal when the field is isotropic)."""
    from desilike import build
    from scipy.special import gammaln

    from wstfast.calculators import RSDBasis, basis_rows, build_cosmology, tracer_cumulants, tracer_grid
    from wstfast.data import load_measurement
    from wstfast.theory.bias import MATTER_BETA
    from wstfast.theory.first_layer import multiplicity
    from wstfast.theory.spectrum import S1mProjection, SpectrumGrid

    config = load_measurement(sorted((REAL_DATA / "real").glob("wst_r*.npz"))[0], q=q)["config"]
    coefficients = _coefficients(config, labels)
    kmax = 0.4
    basis = RSDBasis(cosmo=build_cosmology(()), config=config, coefficients=coefficients, kmax=kmax, tracer="matter",
                     space="real")
    keys = ("grid_terms", "kappa", "skewness2", "s21m_terms", "dlogA")
    out = build(basis, output=lambda: {key: getattr(basis, key) for key in keys})({})
    rows, kappa, skewness, _ = basis_rows(type("Basis", (), out)())
    projection = S1mProjection(config, coefficients, SpectrumGrid(kmax=kmax))
    beta = np.asarray(MATTER_BETA)
    full, tree = tracer_grid(rows, beta, (cs2, 0.0, 0.0), projection.k2mu2, shotnoise, 0.0)
    lam = np.asarray(projection.matrix @ np.asarray(full).ravel() + shotnoise * projection.shot)
    lin = np.asarray(projection.matrix @ np.asarray(tree).ravel() + shotnoise * projection.shot)
    kappa, skew2 = (np.asarray(x) for x in tracer_cumulants(kappa, skewness, beta, shotnoise))
    blocks, edgeworth, s1, start = [], [], [], 0
    for i, c in enumerate(coefficients):
        nb, n, w = c.ell + 1, 2 * c.ell + 1, multiplicity(c.ell)
        lm, ll = lam[start:start + nb], lin[start:start + nb]
        start += nb
        e_block = q * (q - 2) * kappa[i, :nb] / (8 * w * (w + 2) * ll**2)
        e = np.sum(w * e_block) / n
        if c.ell == 0:
            e = e + q * (q - 2) * (q - 4) * skew2[i] / ll[0] ** 3 / (72 * 15)
        blocks.append(lm)
        edgeworth.append(e_block)
        s1.append((2 * np.sum(w * lm) / n) ** (q / 2) * np.exp(gammaln((n + q) / 2) - gammaln(n / 2)) * (1 + e))
    return blocks, edgeworth, np.array(s1)


@pytest.mark.slow
def test_real_space_is_the_f0_limit():
    """At f = 0 the |m| blocks share one variance (up to the cubic lattice) and one Edgeworth factor (up to the Monte
    Carlo error of the cumulants), and the m-summed S1 equals the old real-space model (frozen in the ``real``
    fixture, cs2 = 0.5) to 2e-3: lattice sums against continuum integrals and the one-loop code (docs/refactor_plan.md,
    step 3: <= 0.05 sigma of one box at sigma >= 25 Mpc/h)."""
    reference = _reference("real")
    labels = [f"S1_j{j}_l{ell}" for j in (6, 7, 8, 9) for ell in (0, 1, 2, 4)]
    blocks, edgeworth, s1 = real_space_s1(labels)
    for lam, e in zip(blocks, edgeworth):
        assert np.ptp(lam) / lam.mean() < 1e-4
        assert np.ptp(e) < 3e-3
    old = np.array([reference["prediction"][reference["labels"].tolist().index(label)] for label in labels])
    np.testing.assert_allclose(s1, old, rtol=2e-3)


@pytest.mark.slow
def test_matter_limit_of_biased_code():
    path = FIXTURES / "regression_matter_rsd.npz"
    if not path.exists():
        pytest.skip(f"no fixture {path.name}")
    with np.load(path) as data:
        reference = {key: data[key] for key in data.files}
    current = matter_rsd_outputs()
    for key, rtol in (("kappa", 1e-10), ("skewness2", 1e-10), ("s21m", 1e-10), ("prediction", 1e-8)):
        np.testing.assert_allclose(current[key], reference[key], rtol=rtol, atol=1e-30, err_msg=key)
    # tracer="matter": the same theories with the biases fixed at the matter limit
    np.testing.assert_allclose(matter_rsd_outputs("matter")["prediction"], reference["prediction"], rtol=1e-8)


if __name__ == "__main__":
    import time

    import wstfast.theory  # noqa: F401  (double precision)

    FIXTURES.mkdir(exist_ok=True)
    for name in sys.argv[1:] or GENERATORS:
        start = time.time()
        np.savez(FIXTURES / f"regression_{name}.npz", **GENERATORS[name][0]())
        print(f"wrote regression_{name}.npz in {time.time() - start:.0f} s")
