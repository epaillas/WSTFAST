"""Consistency checks of the analytic WST covariance (wstfast.theory.covariance)."""

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from wstfast.config import SUPERSET, Coefficient
from wstfast.theory.covariance import (ModulusSpectra, ShellCovariance, chaos_covariances, cross_noise, last_layers,
                                       lattice_shells)
from wstfast.theory.moduli import ModulusClustering

BOX, NMESH = 1000.0, 32


def toy_power(k):
    k = np.asarray(k)
    return 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2) ** 1.4


def test_lattice_shells_count_every_mode():
    k, count = lattice_shells(NMESH, BOX, kmax=np.sqrt(3) * np.pi * NMESH / BOX)
    assert count.sum() == NMESH**3 - 1
    assert np.all(np.diff(k) > 0)


def test_cross_noise_diagonal_is_gaussian_noise():
    kout = np.geomspace(2e-3, 0.1, 6)
    fields = [(17.7, 1), (25.0, 3)]
    g = cross_noise(kout, toy_power, fields)
    for i, (sigma, ell) in enumerate(fields):
        _, _, auto = ModulusClustering(kout, sigma, ell)(toy_power)
        np.testing.assert_allclose(g[i, i], np.asarray(auto), rtol=2e-3)
    np.testing.assert_allclose(g[0, 1], g[1, 0])


def test_lattice_second_chaos_matches_shell_sums():
    """The FFT-lattice 2nd chaos and the |k|-shell sums are the same lattice sum (large scales: no corner modes)."""
    coefficients = [Coefficient("S1", ell, j) for j in (8, 9) for ell in (0, 2, 3)]
    q = 0.8
    shell = ShellCovariance(SUPERSET, coefficients, q, BOX, NMESH, toy_power, toy_power)
    fields, spectra = last_layers(SUPERSET, coefficients, toy_power, toy_power, None)
    cov2, cov4 = chaos_covariances(fields, q, BOX, NMESH, spectra)
    np.testing.assert_allclose(cov2, shell.lncov, rtol=1e-6)
    assert np.all(np.diag(cov4) > 0) and np.all(np.diag(cov4) < 0.1 * np.diag(cov2))
    _, cov4_q2 = chaos_covariances(fields, 2.0, BOX, NMESH, spectra)
    np.testing.assert_allclose(cov4_q2, 0.0, atol=1e-30)


def test_chaos_adaptive_mesh_and_single_precision_are_exact():
    """Pair meshes below nmesh (alias-free for the band limit) and float32 reproduce the fixed-mesh float64 result."""
    fields = [(50.0, 0), (50.0, 3), (70.7, 4), (35.4, 2)]
    spectra = lambda a, b, k: toy_power(k)  # noqa: E731
    ref2, ref4 = chaos_covariances(fields, 0.8, BOX, 96, spectra, tol=0, single=False)
    for tol, single in ((1e-8, False), (0, True), (1e-8, True)):
        cov2, cov4 = chaos_covariances(fields, 0.8, BOX, 96, spectra, tol=tol, single=single)
        np.testing.assert_allclose(cov2, ref2, rtol=1e-5)
        np.testing.assert_allclose(cov4, ref4, rtol=1e-5)


def test_s21_covariance_is_symmetric_positive():
    coefficients = [Coefficient("S1", 1, 9), Coefficient("S21", 1, 4, 8), Coefficient("S21", 1, 4, 9),
                    Coefficient("S21", 2, 5, 9)]
    keys = [(4, 1), (5, 2)]
    k = np.geomspace(2e-3, 0.4, 24)
    g = cross_noise(k, toy_power, [(SUPERSET.sigma(j), ell) for j, ell in keys])
    moduli = ModulusSpectra(k, {key: np.full_like(k, 1.2) for key in keys},
                            {(a, b): g[i, j] for i, a in enumerate(keys) for j, b in enumerate(keys)})
    cov = ShellCovariance(SUPERSET, coefficients, 0.8, BOX, 64, toy_power, toy_power, moduli).lncov
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
