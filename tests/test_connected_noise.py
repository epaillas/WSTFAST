"""Tree-level connected noise of the moduli of a biased tracer (connected_noise_rsd) and the cubic bias kernels."""

import numpy as np

import wstfast  # noqa: F401
from wstfast.theory.bias import cubic_kernel


def linear_power():
    k = np.geomspace(1e-4, 10.0, 1024)
    pk = 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6)  # smooth stand-in for P_L at z = 0.5
    return k, pk


def test_cubic_kernel_limits():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(20, 3, 3))
    out = cubic_kernel(v)
    np.testing.assert_allclose(out[:, 0], 1 / 6)
    np.testing.assert_allclose(cubic_kernel(v[:, :1].repeat(3, axis=1) * np.array([1, 2, -3])[None, :, None])[:, 1:], 0,
                               atol=1e-14)  # G3 and delta G2 vanish for collinear vectors
    np.testing.assert_allclose(cubic_kernel(np.eye(3)[None]), [[1 / 6, -0.5, -1.0]], atol=1e-14)  # orthogonal
    np.testing.assert_allclose(cubic_kernel(v[:, [2, 0, 1]]), out, atol=1e-14)  # symmetric
    np.testing.assert_allclose(cubic_kernel(v[:, :2]), 0)  # third order only


def test_real_space_matter_limit():
    """At f = 0, beta = matter, no shot noise and the whole multiplet, the terms equal modulus_noise.py's."""
    import jax.numpy as jnp

    from wstfast.theory.connected_noise_rsd import ConnectedNoiseRSD
    from wstfast.theory.modulus_noise import ModulusNoise
    from wstfast.theory.perturbation import log_interpolator

    k, pl = linear_power()
    pkj = log_interpolator(jnp.asarray(k), jnp.asarray(pl))
    pkn = lambda x: np.exp(np.interp(np.log(np.maximum(x, 1e-30)), np.log(k), np.log(pl)))  # noqa: E731
    sigma, ell = 25.0, 2
    ref = ModulusNoise(sigma, ell, npoints=2**14)
    terms = {name: np.asarray(value)[:2] for name, value in ref.terms(pkj, lambda x: 0.0 * x).items()}
    g = np.asarray(ref.clustering(pkj)[2])[:2]
    out = ConnectedNoiseRSD([(sigma, ell, -1)], ref.unodes[:2], [0.5], npoints=2**13)(
        pkn, 0.0, 0.0, [1, 1, 0, 0, 0], [0, 0, 0], 0.0)
    np.testing.assert_allclose(out["g"][0, :, 0], g, rtol=2e-3)
    for name in ("E", "T40", "T22", "T31", "B22", "G4"):
        np.testing.assert_allclose(out[name][0, :, 0] / g, terms[name] / g, atol=3e-3, err_msg=name)


def test_fog_counterterm_parts():
    """With a constant FoG weight and no shot noise, each FoG part is its term at first order (x 2 for products of
    two bispectra, none for the Gaussian G4)."""
    import wstfast.theory.connected_noise_rsd as module

    k, pl = linear_power()
    pk = lambda x: np.exp(np.interp(np.log(np.maximum(x, 1e-30)), np.log(k), np.log(pl)))  # noqa: E731
    cn = module.ConnectedNoiseRSD([(25.0, 2, 0), (25.0, 2, 1)], [0.4], [0.5], npoints=2**10)
    weight = module._Tracer.__dict__["fog"]  # the staticmethod itself
    try:
        module._Tracer.fog = staticmethod(lambda legs: np.ones_like(legs[0][..., 0]))
        out = cn(pk, 0.7, 4.0, [1, 1.9, -0.5, -0.3, 0.4], [0.5, -0.3, 0.1], 0.0, terms=("T40", "T22", "T31", "B22", "G4"),
                 fog=True)
    finally:
        module._Tracer.fog = weight
    for name, factor in (("T40", 1), ("T22", 1), ("T31", 1), ("B22", 2), ("G4", 0)):
        np.testing.assert_allclose(out[f"{name}_fog"], factor * out[name], rtol=1e-12, atol=1e-14, err_msg=name)
