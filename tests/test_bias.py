"""Biased-tracer extension: kernels, monomial algebra, matter limit of the basis and theories, Poisson terms."""

from pathlib import Path

import numpy as np
import pytest

import wstfast  # noqa: F401
from wstfast.config import SUPERSET, select_coefficients
from wstfast.theory.bias import (MATTER_BETA, galaxy_kernel, lift, monomial_exponents, monomial_product, monomials,
                                 product_coefficients)
from wstfast.theory.eft_loop import matter_kernel, spt_kernels


def linear_power():
    k = np.geomspace(1e-4, 10.0, 1024)
    pk = 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)  # smooth stand-in for P_L at z = 0.5
    return k, pk


def test_galaxy_kernel_matter_limit_and_z2():
    rng = np.random.default_rng(1)
    f = 0.7
    for n in (1, 2, 3):
        v = rng.normal(size=(40, n, 3))
        np.testing.assert_allclose(galaxy_kernel(v, f) @ MATTER_BETA, matter_kernel(v, f), atol=1e-13)
    v = rng.normal(size=(40, 2, 3))
    k1, k2 = v[:, 0], v[:, 1]
    n1, n2, nk = (np.linalg.norm(x, axis=-1) for x in (k1, k2, k1 + k2))
    mu1, mu2, mu = k1[:, 2] / n1, k2[:, 2] / n2, (k1 + k2)[:, 2] / nk
    F2, G2 = spt_kernels(v)
    b1, b2, bg2 = 1.7, 0.6, -0.4
    s2 = (np.sum(k1 * k2, axis=-1) / n1 / n2) ** 2 - 1
    z2 = (b1 * F2 + f * mu**2 * G2 + f * mu * nk / 2 * (mu1 / n1 * (b1 + f * mu2**2) + mu2 / n2 * (b1 + f * mu1**2))
          + b2 / 2 + bg2 * s2)
    np.testing.assert_allclose(galaxy_kernel(v, f) @ np.array([1, b1, b2, bg2, 0.3]), z2, atol=1e-13)


def test_gamma3_kernel_real_space():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(8, 3, 3))
    s2 = lambda a, b: np.sum(a * b, -1) ** 2 / np.sum(a * a, -1) / np.sum(b * b, -1) - 1  # noqa: E731
    expected = 0.0
    for i in range(3):
        j, l = (x for x in range(3) if x != i)
        F, G = spt_kernels(v[:, [j, l]])
        expected = expected + 2 * s2(v[:, i], v[:, j] + v[:, l]) * (F - G) / 3
    np.testing.assert_allclose(galaxy_kernel(v, 0.0)[:, 4], expected, atol=1e-14)


def test_monomial_algebra():
    rng = np.random.default_rng(3)
    beta = np.r_[1.0, rng.normal(size=4)]
    a, b = rng.normal(size=15), rng.normal(size=35)
    m = lambda d: np.asarray(monomials(beta, d))  # noqa: E731
    assert len(monomial_exponents(4)) == 70
    np.testing.assert_allclose(monomial_product(a, 2, b, 3) @ m(5), (a @ m(2)) * (b @ m(3)))
    np.testing.assert_allclose(lift(a, 2, 4) @ m(4), a @ m(2))
    z = [rng.normal(size=(6, 5)) for _ in range(3)]
    np.testing.assert_allclose(product_coefficients(*z) @ m(3), np.prod([x @ beta for x in z], axis=0))


def test_biased_loop_matter_limit_and_renormalization():
    from wstfast.theory.eft_loop import LinearSpectrum, Quadrature, loop_integrals
    from wstfast.theory.layout import LOOP
    from wstfast.theory.spectrum import renormalize
    from wstfast.theory.bias import monomial_exponents

    spectrum = LinearSpectrum(*linear_power())
    quadrature = Quadrature(nq=64, nx=12, nphi=9)
    k, mu = np.array([1e-3, 0.05, 0.1]), np.array([0.2, 0.8])
    matter = loop_integrals(k, mu, spectrum, 0.75, None, quadrature, rsd=True)
    biased = loop_integrals(k, mu, spectrum, 0.75, None, quadrature, rsd=True, biased=True)
    m2 = np.asarray(monomials(MATTER_BETA, 2))
    np.testing.assert_allclose(biased["loop"] @ m2, matter["loop"], rtol=1e-12, atol=1e-12)
    renormalized = renormalize(biased, spectrum, quadrature)
    np.testing.assert_allclose(renormalized["loop"] @ m2, matter["loop"], rtol=1e-12, atol=1e-12)
    b2sq = [tuple(e) for e in monomial_exponents(2)].index((0, 0, 2, 0, 0))
    assert abs(renormalized["22"][0, 0, b2sq]) < 1e-3 * abs(biased["22"][0, 0, b2sq])  # k -> 0 constant removed
    unused = set(range(15)) - set(LOOP)
    assert np.all(biased["loop"][..., sorted(unused)] == 0)


@pytest.mark.parametrize("biased", [False, True])
def test_loop_tables_equal_direct_integrals(biased):
    from wstfast.theory.eft_loop import LinearSpectrum, Quadrature, loop_integrals
    from wstfast.theory.loop_tables import LoopTables

    spectrum = LinearSpectrum(*linear_power())
    quadrature = Quadrature(nq=32, nx=8, nphi=9)
    k, mu = np.array([5e-3, 0.08]), np.array([0.3, 0.9])
    tables = LoopTables(k, mu, quadrature, biased=biased)
    for f in (0.6, 0.85):
        direct = loop_integrals(k, mu, spectrum, f, None, quadrature, rsd=True, biased=biased)
        for name in ("22", "13", "loop"):
            np.testing.assert_allclose(tables(spectrum, f)[name], direct[name], rtol=1e-11,
                                       atol=1e-12 * np.abs(direct[name]).max())


def test_poisson_cumulants_of_gaussian_wavelet():
    """Pure shot-noise kappa_4 and kappa_3 of the l = 0 block: n^-3 int psi^4 and n^-2 int psi^3 (CIC window ~1%)."""
    from wstfast.theory.first_layer import S1mBasis
    from wstfast.theory.layout import KAPPA_SIZES, SKEW_SIZES, split

    klin, pklin = linear_power()
    coefs = [c for c in select_coefficients(SUPERSET, s1_min_scale=35.0) if c.kind == "S1" and c.ell == 0][:1]
    sigma = SUPERSET.sigma(coefs[0].j)
    kappa, skewness = S1mBasis(SUPERSET, coefs, npoints=2**14).cumulants(klin, pklin, 0.7)
    k4 = (2 * np.pi * sigma**2) ** -6 * (np.pi * sigma**2 / 2) ** 1.5
    k3 = (2 * np.pi * sigma**2) ** -4.5 * (2 * np.pi * sigma**2 / 3) ** 1.5
    assert split(kappa, KAPPA_SIZES)[3][0, 0, 0] == pytest.approx(k4, rel=0.02)
    assert split(skewness, SKEW_SIZES)[2][0, 0] == pytest.approx(k3, rel=0.02)


FOF = Path("/Users/epaillas/data/quijote/halos/FoF/fiducial/0/groups_003")


@pytest.mark.skipif(not FOF.exists(), reason="no local Quijote FoF catalogue")
def test_read_fof():
    from wstfast.quijote import halo_positions, read_fof

    catalogue = read_fof(FOF)
    assert catalogue["pos"].shape == (len(catalogue["mass"]), 3)
    assert catalogue["npart"].min() >= 20 and np.all(catalogue["pos"] >= 0) and np.all(catalogue["pos"] <= 1000)
    real, rsd = halo_positions(0), halo_positions(0, rsd=True)
    assert np.array_equal(real[:, :2], rsd[:, :2]) and not np.array_equal(real[:, 2], rsd[:, 2])


def test_fog_counterterm_reproduces_damping():
    """At c_v = (f sigma_v k_NL)^2 / 2, the second-order Fingers-of-God counterterm columns reproduce the velocity-
    damped S1m cumulants and S21m terms (damping 1 x linear here) to second order; order 0 is the undamped basis."""
    from wstfast.theory.first_layer import S1mBasis
    from wstfast.theory.layout import FOG_KNL, KAPPA_SIZES, S21_SIZES, SKEW_SIZES, collapse_fog
    from wstfast.theory.second_layer import S21mBasis

    klin, pklin = linear_power()
    f = 0.76
    sigma_v = float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
    c = 0.5 * (f * sigma_v * FOG_KNL) ** 2
    s1 = [c_ for c_ in select_coefficients(SUPERSET, s1_min_scale=25.0) if c_.kind == "S1" and c_.ell in (0, 2)][:2]
    kwargs = dict(npoints=2**10, cubic=True)
    damped = S1mBasis(SUPERSET, s1, damping=sigma_v, **kwargs).cumulants(klin, pklin, f)
    undamped = S1mBasis(SUPERSET, s1, damping=None, **kwargs).cumulants(klin, pklin, f)
    fog = S1mBasis(SUPERSET, s1, fog=True, **kwargs).cumulants(klin, pklin, f)
    np.testing.assert_allclose(collapse_fog(*fog, c_s1=0.0)[0], undamped[0], rtol=1e-12)
    for got, want, n in zip(collapse_fog(*fog, c_s1=c)[:2], damped, (KAPPA_SIZES[0], SKEW_SIZES[0])):
        scale = np.max(np.abs(want[..., :n]))
        assert np.max(np.abs(got - want)) < 1e-4 * scale
    s21 = [c_ for c_ in select_coefficients(SUPERSET, s21_min_scale=25.0, s21_min_ratio=2.8, s21_min_scale2=70.0)
           if c_.kind == "S21"][:1]
    quad = dict(nk=6, nmu=3, npts=16, nx=8, nphi=6)
    damped = S21mBasis(SUPERSET, s21, sigma_v_factor=1.0, **quad)(klin, pklin, f)
    fog = S21mBasis(SUPERSET, s21, fog=True, **quad)(klin, pklin, f)
    np.testing.assert_allclose(fog[:, :sum(S21_SIZES)], S21mBasis(SUPERSET, s21, sigma_v_factor=0.0, **quad)(
        klin, pklin, f), rtol=1e-12)
    got = collapse_fog(s21m=fog, c_s21=c)[2]
    assert np.max(np.abs(got - damped)) < 1e-3 * np.max(np.abs(damped))


def test_ap_polynomial_bases_reduce_at_q1():
    """The AP-polynomial cumulants and S21m terms (rsd_bias, ap > 0): their alpha = 0 term is the basis without AP
    (exact at q = 1), and they evaluate through ap_evaluate / s21m_lambda_mask."""
    from wstfast.theory.first_layer import S1mBasis
    from wstfast.theory.layout import AP_TERMS, ap_evaluate, basis_layout, s21m_lambda_mask
    from wstfast.theory.second_layer import S21mBasis

    klin, pklin = linear_power()
    f = 0.76
    s1 = [c for c in select_coefficients(SUPERSET, s1_min_scale=25.0) if c.kind == "S1" and c.ell in (1, 2)][:2]
    kwargs = dict(npoints=2**10, cubic=True, fog=True)
    k0, s0 = S1mBasis(SUPERSET, s1, **kwargs).cumulants(klin, pklin, f)
    k3, s3 = S1mBasis(SUPERSET, s1, ap=3, **kwargs).cumulants(klin, pklin, f)
    assert basis_layout(k3.shape[-1]) == (True, True, AP_TERMS[3])
    np.testing.assert_allclose(ap_evaluate(k3, AP_TERMS[3]), k0, rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(ap_evaluate(s3, AP_TERMS[3]), s0, rtol=1e-12, atol=1e-300)
    s21 = [c for c in select_coefficients(SUPERSET, s21_min_scale=25.0, s21_min_ratio=2.8, s21_min_scale2=70.0)
           if c.kind == "S21"][:1]
    quad = dict(nk=4, nmu=3, npts=12, nx=6, nphi=4)
    t0 = S21mBasis(SUPERSET, s21, fog=True, **quad)(klin, pklin, f)
    t3 = S21mBasis(SUPERSET, s21, fog=True, ap=3, **quad)(klin, pklin, f)
    assert t3.shape[-1] == AP_TERMS[3] * t0.shape[-1] and s21m_lambda_mask(True).size == t0.shape[-1]
    np.testing.assert_allclose(ap_evaluate(t3, AP_TERMS[3]), t0, rtol=1e-4)  # the AP projection's grouping at q = 1


def test_real_space_theories_fix_the_anisotropic_terms():
    """ModelSpec(space="real"): c2, c4 and alpha2 fixed at 0, with tracer="matter" the biases too; AP is refused."""
    from desilike import get_params

    from wstfast.calculators import APGeometry, Basis, NuisanceParameters, PowerTheory, S1mTheory, build_cosmology
    from wstfast.config import ModelSpec
    from wstfast.theory.spectrum import MultipoleProjection, S1mProjection, SpectrumGrid

    s1 = [c for c in select_coefficients(SUPERSET, s1_min_scale=35.0) if c.kind == "S1" and c.ell <= 1][:2]
    grid = SpectrumGrid(kmax=0.2)
    basis = Basis(cosmo=build_cosmology(("omega_cdm", "h")), config=SUPERSET, coefficients=s1, kmax=0.2,
                  npoints=2**10, tracer="matter", space="real")
    matter = NuisanceParameters(ModelSpec.matter(space="real"))
    theory = S1mTheory(S1mProjection(SUPERSET, s1, grid), s1, basis=basis, nuisance=matter)
    varied = get_params(theory).select(varied=True, derived=False).names()
    assert "c0" in varied and not {"c2", "c4", "alpha2", "b1", "b2"} & set(varied)
    power = PowerTheory(MultipoleProjection(np.linspace(0.01, 0.15, 8), grid, ells=(0,)), basis=basis,
                        nuisance=NuisanceParameters(ModelSpec(space="real")))
    varied = get_params(power).select(varied=True, derived=False).names()
    assert {"c0_pk", "b1", "alpha0"} <= set(varied) and not {"c2_pk", "c4_pk", "alpha2"} & set(varied)
    with pytest.raises(ValueError):
        S1mTheory(S1mProjection(SUPERSET, s1, grid), s1, basis=basis, nuisance=matter, ap=APGeometry(basis, 0.5))


def test_band_powers_are_the_q2_first_layer():
    """The band powers (PowerTheory on a BandProjection) equal the sum over |m| of the S1m blocks at q = 2, where the
    Edgeworth factor vanishes; and the compression of real-space labels groups the equally distributed blocks."""
    from desilike import build

    from wstfast.calculators import Basis, JointTheory, NuisanceParameters, PowerTheory, S1mTheory, build_cosmology
    from wstfast.config import ModelSpec
    from wstfast.inference import m_compression
    from wstfast.theory.spectrum import BandProjection, S1mProjection, SpectrumGrid

    s1 = [c for c in select_coefficients(SUPERSET, s1_min_scale=35.0) if c.kind == "S1" and c.ell <= 2][:4]
    grid = SpectrumGrid(kmax=0.2)
    basis = Basis(cosmo=build_cosmology(("omega_cdm",)), config=SUPERSET, coefficients=s1, kmax=0.2, npoints=2**10,
                  tracer="matter", space="real")
    nuisance = NuisanceParameters(ModelSpec.matter(space="real"))
    bands = BandProjection(SUPERSET, s1, grid)
    assert np.all(bands.support < 1e-3)
    theory = JointTheory([PowerTheory(bands, basis=basis, nuisance=nuisance, shotnoise=7.45, counterterms=""),
                          S1mTheory(S1mProjection(SUPERSET, s1, grid), s1, basis=basis, nuisance=nuisance, q=2.0,
                                    shotnoise=7.45)])
    out = np.asarray(build(theory)({"omega_cdm": 0.125, "c0": 2.0}))
    band, blocks = out[:len(s1)], out[len(s1):]
    start, summed = 0, []
    for c in s1:
        summed.append(blocks[start:start + c.ell + 1].sum())
        start += c.ell + 1
    np.testing.assert_allclose(band, summed, rtol=1e-12)
    labels = ["S1m_j6_l2_m0", "S1m_j6_l2_m1", "S1m_j6_l2_m2", "S21m_j6_j9_l1_m0_m1", "S21m_j6_j9_l1_m1_m1", "S1q2_j4_l3"]
    matrix, names = m_compression(labels)
    assert names == ["S1m_j6_l2_m0", "S1m_j6_l2_m1+", "S21m_j6_j9_l1_m0_m1+", "S21m_j6_j9_l1_m1+_m1+", "S1q2_j4_l3"]
    np.testing.assert_allclose(matrix @ np.arange(1.0, 7.0), [1.0, 2.5, 4.0, 5.0, 6.0])


def test_band_compression_keeps_leading_components():
    """band_compression keeps the principal components of the band block above the cut, in standardized units, and
    passes the other entries through."""
    from wstfast.inference import band_compression

    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2))
    bands = np.column_stack([x[:, 0], x[:, 0] + 1e-9 * x[:, 1], 2 * x[:, 0] - x[:, 1]])  # rank 2 (numerically)
    vectors = np.column_stack([rng.normal(size=500), bands])
    labels = ["S1m_j6_l0_m0", "S1q2_j5_l0", "S1q2_j5_l1", "S1q2_j5_l2"]
    matrix, names = band_compression(vectors, labels, 1e-6)
    assert names == ["S1m_j6_l0_m0", "S1q2_pc0", "S1q2_pc1"]
    compressed = vectors @ matrix.T
    assert np.allclose(compressed[:, 0], vectors[:, 0])
    assert np.linalg.matrix_rank(np.cov(compressed, rowvar=False)) == 3
    assert band_compression(vectors, labels[:1], 1e-6)[0] is None
