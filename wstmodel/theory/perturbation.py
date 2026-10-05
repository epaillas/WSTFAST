"""Real-space matter perturbation theory on fixed quadrature grids (JAX).

Kernels are the EdS SPT kernels. Every function takes the linear spectrum through a
log-log interpolator so that P(k) can be evaluated at arbitrary |k - p|. Quadrature
geometry is built from small 1D grids inside the call, so that jit does not embed
large constant arrays.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss


def log_interpolator(k, pk):
    """Return P(x) interpolated in log-log space (constant beyond the tabulated range)."""
    logk, logpk = jnp.log(k), jnp.log(pk)

    def interp(x):
        return jnp.exp(jnp.interp(jnp.log(x), logk, logpk))

    return interp


def log_weights(x):
    """Weights for int dx f(x) = sum f(x_i) w_i on a logarithmic grid (trapezoid in ln x)."""
    return x * np.gradient(np.log(x))


def f2_kernel(k1, k2, mu12):
    """Symmetrised second-order density kernel F2 for magnitudes |k1|, |k2| and their cosine."""
    return 5.0 / 7.0 + 0.5 * mu12 * (k1 / k2 + k2 / k1) + 2.0 / 7.0 * mu12**2


def _p13_kernel(r):
    """Angle-integrated F3 kernel of P13, with its series expansions at small and large r."""
    r2 = r * r
    safe = jnp.clip(r, 1e-2, 1e2)
    s2 = safe * safe
    log = jnp.log(jnp.abs((1 + safe) / jnp.where(safe == 1.0, 1e-300, 1 - safe)))
    full = 12 / s2 - 158 + 100 * s2 - 42 * s2**2 + 3 / (s2 * safe) * (s2 - 1) ** 3 * (7 * s2 + 2) * log
    small = -168 + 928 / 5 * r2 - 4512 / 35 * r2**2 + 416 / 21 * r2**3
    x2 = 1 / jnp.maximum(r2, 1e-300)
    large = -488 / 5 + 96 / 5 * x2 - 160 / 21 * x2**2 - 1376 / 1155 * x2**3
    return jnp.where(r < 1e-2, small, jnp.where(r > 1e2, large, full))


class OneLoopMatter:
    """P22 + P13 of real-space matter on an output k grid.

    P22 is folded onto |k - q| > q (twice the half-space integral), so the only infrared
    region left is q -> 0, which the logarithmic q grid resolves.
    """

    def __init__(self, kout, qmin=1e-4, qmax=10.0, nq=320, nmu=48, nq13=2048):
        self.kout = np.asarray(kout, dtype="f8")
        self.q = np.geomspace(qmin, qmax, nq)
        self.wq = self.q**2 * log_weights(self.q) / (4 * np.pi**2)  # q^2 dq / (4 pi^2)
        self.x, self.wx = leggauss(nmu)
        self.q13 = np.geomspace(qmin, qmax, nq13)
        self.wq13 = log_weights(self.q13)

    def p22(self, pk):
        k = jnp.asarray(self.kout)[:, None, None]
        q = jnp.asarray(self.q)[None, :, None]
        half = 0.5 * (jnp.minimum(1.0, k / (2 * q)) + 1.0)
        mu = -1.0 + half * (jnp.asarray(self.x)[None, None, :] + 1.0)
        kq = jnp.sqrt(jnp.maximum(k**2 + q**2 - 2 * k * q * mu, 1e-30))
        f2 = f2_kernel(q, kq, (k * q * mu - q**2) / (q * kq))
        integrand = 4.0 * f2**2 * pk(q) * pk(kq) * half * jnp.asarray(self.wx)
        return jnp.sum(integrand * jnp.asarray(self.wq)[None, :, None], axis=(1, 2))

    def p13(self, pk):
        k = jnp.asarray(self.kout)[:, None]
        q = jnp.asarray(self.q13)[None, :]
        integral = jnp.sum(pk(q) * _p13_kernel(q / k) * jnp.asarray(self.wq13), axis=1)
        return k[:, 0] ** 2 * pk(k[:, 0]) * integral / (252.0 * 4 * np.pi**2)

    def __call__(self, pk):
        return self.p22(pk) + self.p13(pk)
