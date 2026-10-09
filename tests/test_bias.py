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
    from wstfast.theory.rsd_bias import LOOP, renormalize
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


def test_biased_cumulants_and_s21m_matter_limit():
    from wstfast.theory.rsd import RSDS1Basis
    from wstfast.theory.rsd_bias import (KAPPA_SIZES, S21_SIZES, SKEW_SIZES, BiasedRSDS1Basis, BiasedRSDS21mBasis,
                                         split)
    from wstfast.theory.rsd_moduli import RSDS21mBasis

    klin, pklin = linear_power()
    f = 0.75
    m = lambda d: np.asarray(monomials(MATTER_BETA, d))  # noqa: E731
    s1 = [c for c in select_coefficients(SUPERSET, s1_min_scale=25.0) if c.kind == "S1" and c.ell in (0, 2)][:3]
    kappa, skew2 = RSDS1Basis(SUPERSET, s1, npoints=2**10, lattice=False, damping="linear").cumulants(klin, pklin, f)
    bk, bs = BiasedRSDS1Basis(SUPERSET, s1, npoints=2**10, lattice=False, damping="linear").cumulants(klin, pklin, f)
    np.testing.assert_allclose(split(bk, KAPPA_SIZES)[0] @ m(4), kappa, rtol=1e-10, atol=1e-30)
    np.testing.assert_allclose(15 * (split(bs, SKEW_SIZES)[0] @ m(3)) ** 2, skew2, rtol=1e-10, atol=1e-30)
    s21 = [c for c in select_coefficients(SUPERSET, s21_min_scale=25.0, s21_min_ratio=2.8, s21_min_scale2=70.0)
           if c.kind == "S21"][:1]
    quad = dict(npts=16, nx=8, nphi=6)
    matter = RSDS21mBasis(SUPERSET, s21, nk=6, nmu=3, **quad)(klin, pklin, f)
    terms = split(BiasedRSDS21mBasis(SUPERSET, s21, nk=6, nmu=3, **quad)(klin, pklin, f), S21_SIZES)
    norm2 = (terms[6] @ m(2)) ** 2
    np.testing.assert_allclose(terms[0] @ m(4) / norm2, matter[:, 0], rtol=1e-10)
    np.testing.assert_allclose(terms[3] @ m(4) / norm2, matter[:, 1], rtol=1e-10)


def test_poisson_cumulants_of_gaussian_wavelet():
    """Pure shot-noise kappa_4 and kappa_3 of the l = 0 block: n^-3 int psi^4 and n^-2 int psi^3 (CIC window ~1%)."""
    from wstfast.theory.rsd_bias import KAPPA_SIZES, SKEW_SIZES, BiasedRSDS1Basis, split

    klin, pklin = linear_power()
    coefs = [c for c in select_coefficients(SUPERSET, s1_min_scale=35.0) if c.kind == "S1" and c.ell == 0][:1]
    sigma = SUPERSET.sigma(coefs[0].j)
    kappa, skewness = BiasedRSDS1Basis(SUPERSET, coefs, npoints=2**14, lattice=False).cumulants(klin, pklin, 0.7)
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


@pytest.mark.slow
def test_biased_theories_matter_limit():
    """At beta = (1, 1, 0, 0, 0) and alpha = 0, the biased P_l, S1m and S21m equal the matter ones (no shot noise)."""
    from desilike import build

    from wstfast.calculators import (JointTheory, RSDBasis, RSDPowerTheory, S1mTheory, S21mTheory, build_cosmology)
    from wstfast.config import Coefficient
    from wstfast.theory.rsd import MultipoleProjection, RSDGrid, S1mProjection

    s1 = [c for c in select_coefficients(SUPERSET, s1_min_scale=35.0) if c.kind == "S1" and c.ell <= 1][:3]
    s21 = [c for c in select_coefficients(SUPERSET, s21_min_scale=35.0, s21_min_ratio=2.0, s21_min_scale2=70.0)
           if c.kind == "S21" and c.ell == 1][:1]
    first = [Coefficient("S1", c.ell, c.j) for c in s21]
    s1 = s1 + [c for c in first if c not in s1]
    grid = RSDGrid(kmax=0.2)
    edges = np.linspace(0.01, 0.15, 8)

    def theory(tracer):
        basis = RSDBasis(cosmo=build_cosmology(()), config=SUPERSET, coefficients=s1, kmax=0.2, npoints=2**10,
                         s21_coefficients=s21, tracer=tracer)
        return JointTheory([
            S1mTheory(S1mProjection(SUPERSET, s1, grid), s1, basis=basis, tracer=tracer),
            RSDPowerTheory(MultipoleProjection(edges, grid), basis=basis, tracer=tracer),
            S21mTheory(S1mProjection(SUPERSET, first, grid), s21, first, basis=basis,
                       cumulant_index=[s1.index(c) for c in first], tracer=tracer)])

    matter = np.asarray(build(theory("matter"))({}))
    biased = np.asarray(build(theory("biased"))({"b1": 1.0}))
    np.testing.assert_allclose(biased, matter, rtol=1e-8)
