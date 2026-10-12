"""Redshift-space analytic covariance of the WST (covariance_rsd): the cross moduli spectra and the covariance."""

import numpy as np

import wstfast  # noqa: F401


def linear_power():
    k = np.geomspace(1e-4, 10.0, 1024)
    pk = 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)  # smooth stand-in for P_L at z = 0.5
    return k, pk


def test_cross_moduli_reproduce_the_basis():
    """At fixed biases, the auto terms of ModulusCrossRSD equal the emulated basis (BiasedModulusSpectraRSD)."""
    from wstfast.theory.bias import monomials
    from wstfast.theory.covariance_rsd import ModulusCrossRSD
    from wstfast.theory.rsd_bias import BiasedModulusSpectraRSD, split

    k, pl = linear_power()
    pk = lambda x: np.exp(np.interp(np.log(np.maximum(x, 1e-30)), np.log(k), np.log(pl)))  # noqa: E731
    fields = [(25.0, 2, 0), (25.0, 2, 1), (25.0, 3, 2)]
    kg, mug = np.array([0.01, 0.03]), np.array([0.2, 0.8])
    beta, N, f, sv = np.array([1.0, 1.9, -0.7, -0.6, 0.8]), 3000.0, 0.72, 4.0
    grids, norms = BiasedModulusSpectraRSD(fields, kg, mug, npts=16, nx=8, nphi=6)(pk, f, sv)
    m2, m4 = monomials(beta, 2), monomials(beta, 4)
    r0r0, r0r1, r1r1, g0, g1, g2 = split(grids, (70, 70, 70, 70, 15, 1))
    norm2 = (norms[:, :15] @ m2 + N * norms[:, 15]) ** 2
    response = (r0r0 @ m4 + N * r0r1 @ m4 + N**2 * r1r1 @ m4) / norm2[:, None, None]
    noise = (g0 @ m4 + N * g1 @ m2 + N**2 * g2[..., 0]) / norm2[:, None, None]
    r, g, n = ModulusCrossRSD(fields, kg, mug, npts=16, nx=8, nphi=6).spectra(pk, f, sv, beta, N)
    np.testing.assert_allclose(n**2, norm2, rtol=1e-10)
    np.testing.assert_allclose(r**2 * pk(kg)[None, :, None], response, rtol=1e-8)
    np.testing.assert_allclose(np.array([g[i, i] for i in range(len(fields))]), noise, rtol=1e-8)
    np.testing.assert_allclose(g, np.swapaxes(g, 0, 1))  # symmetric cross noise


def test_modulus_tables_cross_rescaling():
    """The cross noise is g_ab rescaled by sqrt(N_aa N_bb / (g_aa g_bb)); the autos are (1 + a) g + W."""
    from wstfast.theory.covariance_rsd import ModulusTables

    x, _ = np.polynomial.legendre.leggauss(6)
    k, mu = np.geomspace(0.01, 0.1, 4), x[3:]
    rng = np.random.default_rng(0)
    g = np.abs(rng.normal(1.0, 0.1, (2, 2, 4, 3))) + 1.0
    g[0, 1] = g[1, 0] = 0.3 * np.sqrt(g[0, 0] * g[1, 1])
    t = ModulusTables(k, mu, np.ones((2, 4, 3)), g, amplitudes=[0.1, -0.2], white=[0.5, 0.0])
    auto0, auto1 = 1.1 * g[0, 0] + 0.5, 0.8 * g[1, 1]
    np.testing.assert_allclose(t.n[0, 0], auto0)
    np.testing.assert_allclose(t.n[0, 1], 0.3 * np.sqrt(auto0 * auto1))
    np.testing.assert_allclose(t.noise(0, 0, k[1], mu[2]), auto0[1, 2])
