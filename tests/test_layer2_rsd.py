"""The perturbation-theory tracer fields of scripts/feasibility/second_layer_rsd.py against the analytic redshift-space
kernels of wstfast.theory.bias: plane-wave inputs must give Z1, Z2 and Z3 (with the cubic operators) exactly."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/feasibility"))

from pt_modulus_noise import SPTFields  # noqa: E402
from second_layer_rsd import HaloRSDFields  # noqa: E402
from wstfast.measure import Lattice  # noqa: E402
from wstfast.theory.bias import cubic_kernel, galaxy_kernel  # noqa: E402

N = 32
BIASES = dict(b1=1.9, b2=-0.6, bG2=-0.6, bGamma3=0.6, b3=0.4, bG3=-0.3, bdG2=0.2, alpha0=0.0)
F = 0.76
BETA = np.array([1.0, BIASES["b1"], BIASES["b2"], BIASES["bG2"], BIASES["bGamma3"]])
CUBIC = np.array([BIASES["b3"], BIASES["bG3"], BIASES["bdG2"]])


def fields(modes):
    lattice = Lattice(N)
    spt = SPTFields.__new__(SPTFields)
    k, kz = 2 * np.pi * np.fft.fftfreq(N), 2 * np.pi * np.fft.rfftfreq(N)
    spt.kvec = np.meshgrid(k, k, kz, indexing="ij", sparse=True)
    spt.k2 = lattice.kmag.astype("f8") ** 2
    spt.k2[0, 0, 0] = 1.0
    spt.n = N
    delta1 = np.zeros((N, N, N // 2 + 1), complex)
    for index, value in modes:
        delta1[index] = value
    spt.delta1 = delta1
    spt.delta2, theta2 = spt._second(delta1)
    spt.delta3 = spt._third(delta1, spt.delta2, theta2)
    return HaloRSDFields(spt, F, BIASES)


def wavevector(index):
    i, j, l = index
    return 2 * np.pi * np.array([np.fft.fftfreq(N)[i], np.fft.fftfreq(N)[j], np.fft.rfftfreq(N)[l]])


@pytest.mark.parametrize("i1, i2", [((1, 2, 1), (3, 0, 2)), ((0, 1, 3), (2, 31, 1)), ((5, 3, 2), (30, 1, 1))])
def test_second_order_matches_z2(i1, i2):
    c1, c2 = 0.7 + 0.2j, -0.4 + 0.9j
    target = tuple((np.array(i1) + np.array(i2)) % N)
    z2 = galaxy_kernel(np.array([wavevector(i1), wavevector(i2)]), F) @ BETA
    assert fields([(i1, c1), (i2, c2)]).s2[target] == pytest.approx(2 * z2 * c1 * c2 / N**3, rel=1e-6)


@pytest.mark.parametrize("i1, i2, i3", [((1, 2, 1), (3, 0, 2), (0, 1, 1)), ((2, 1, 1), (0, 3, 2), (1, 30, 1))])
def test_third_order_matches_z3(i1, i2, i3):
    c1, c2, c3 = 0.7 + 0.2j, -0.4 + 0.9j, 0.3 - 0.5j
    target = tuple((np.array(i1) + np.array(i2) + np.array(i3)) % N)
    vectors = np.array([wavevector(i1), wavevector(i2), wavevector(i3)])
    z3 = galaxy_kernel(vectors, F) @ BETA + cubic_kernel(vectors) @ CUBIC
    s3 = fields([(i1, c1), (i2, c2), (i3, c3)]).s3[target]
    assert s3 == pytest.approx(6 * z3 * c1 * c2 * c3 / N**6, rel=1e-6)


def test_first_order_is_kaiser():
    k = wavevector((1, 2, 1))
    assert fields([((1, 2, 1), 1.0)]).s1[1, 2, 1] == pytest.approx(BIASES["b1"] + F * k[2] ** 2 / (k @ k), rel=1e-6)


def test_unprojected_s21m_columns_and_powers():
    from wstfast.theory.rsd_bias import S21_SIZES, biased_scalings, s21m_assemble, s21m_field_powers

    rng = np.random.default_rng(0)
    for fog in (False, True):
        plam, pnorm = s21m_field_powers(fog)
        lam, norms = rng.normal(size=(3, plam.size)), rng.normal(size=(3, pnorm.size))
        assembled = s21m_assemble(lam, norms, fog)
        assert assembled.shape[-1] == biased_scalings(2.0, fog=fog)["s21m"].size
        ratio = 1.37
        np.testing.assert_allclose(s21m_assemble(lam * ratio**plam, norms * ratio**pnorm, fog),
                                   assembled * np.asarray(biased_scalings(ratio, fog=fog)["s21m"]), rtol=1e-12)
    assert sum(S21_SIZES) == s21m_field_powers(False)[0].size + s21m_field_powers(False)[1].size


def test_s21m_ap_projection_matches_fixed_projection_and_scales_with_volume():
    from numpy.polynomial.legendre import leggauss

    from wstfast import WSTConfig
    from wstfast.config import Coefficient
    from wstfast.theory.rsd_moduli import S21mAPProjection, S21mProjection, first_layer_fields

    config = WSTConfig(J=3, L=2, L2=2, sigma0=2.5, step=2**0.5, cellsize=10.0, min_dj=3)
    coefficients = [Coefficient("S21", ell, 0, 3) for ell in (1, 2)]
    fields = first_layer_fields(coefficients)
    x, _ = leggauss(12)
    k, mu = np.geomspace(4e-3, 0.12, 24), x[6:]
    fixed = S21mProjection(config, coefficients, fields, k, mu, boxsize=2000.0, nmesh=128)
    ap = S21mAPProjection(config, coefficients, fields, k, mu, boxsize=2000.0, nmesh=128)
    grid = np.exp(-((k[:, None] / 0.05) ** 2)) * (1 + 0.5 * mu[None, :] ** 2) / (1 + (k[:, None] / 0.01) ** 2)
    a, b = fixed.matrix @ grid.ravel(), np.asarray(ap.at(1.0, 1.0)) @ grid.ravel()
    np.testing.assert_allclose(b, a, rtol=5e-3)
    ones = np.ones(grid.size)
    np.testing.assert_allclose(np.asarray(ap.at(1.02, 0.99)) @ ones, (np.asarray(ap.at(1.0, 1.0)) @ ones) / (1.02 * 0.99**2),
                               rtol=1e-10)
