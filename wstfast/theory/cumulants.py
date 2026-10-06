"""Tree-level zero-lag cumulants of the wavelet vector X_a = (delta * psi_{sigma,l}^a)(x), a = 1..n = 2l + 1.

They enter the next-to-leading Edgeworth correction to S1 (docs/wst_eft_feasibility.md, Sec. 2.2):

    S1 / S1_G = 1 + q (q - 2) / (8 n (n + 2)) K4 + q (q - 2) (q - 4) / (72 n (n + 2) (n + 4)) (6 K33 + 9 K3v),
    K4 = sum_ab kappa_aabb / s^4,  K33 = sum_abc kappa_abc^2 / s^6,  K3v = sum_c (sum_a kappa_aac)^2 / s^6.

Both numerators are computed here in units where the per-leg filter is R(k) = W(k) (sigma k)^l exp(-sigma^2 k^2 / 2),
with the mass-assignment window W, so that n s^2 = int_k P(k) R(k)^2:

* ``bispectrum``: 6 K33 s^6 + 9 K3v s^6. By isotropy kappa_{m1 m2 m3} = A (l l l; m1 m2 m3) in the complex basis,
  so K33 s^6 = A^2, nonzero for even l only, and K3v vanishes unless l = 0 (where K3v = K33). A is a 3D integral
  of the tree bispectrum against the invariant I = sum_m (l l l; m) Y_lm1(k1) Y_lm2(k2) Y_lm3(k3).
* ``trispectrum``: sum_ab kappa_aabb = int_{k1 k2 k3} T(k1, k2, k3, k4) F(k1, k2) F(k3, k4), with the tree
  trispectrum T and the pair kernel F(p, q) = (-1)^l R(p) R(q) P_l(p.q). This 9D integral is done by
  scrambled-Sobol quasi-Monte Carlo with fixed points in u = sigma k, so it is a smooth, deterministic function
  of the linear spectrum (as the emulator needs).

Shot-noise contributions to the cumulants are neglected (Quijote has V / N = 7.5 (Mpc/h)^3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import sph_harm_y
from scipy.stats import norm, qmc

from .moduli import legendre
from .perturbation import f2_kernel, log_weights

_EPS = 1e-30


def _dot(a, b):
    return jnp.sum(a * b, axis=-1)


def _f2(a, b, c0=5.0 / 7.0, c2=2.0 / 7.0):
    """Symmetric F2 (default) or G2 (c0 = 3/7, c2 = 4/7) kernel for vectors a, b (last axis = components)."""
    a2, b2, ab = jnp.maximum(_dot(a, a), _EPS), jnp.maximum(_dot(b, b), _EPS), _dot(a, b)
    return c0 + 0.5 * ab * (1.0 / a2 + 1.0 / b2) + c2 * ab**2 / (a2 * b2)


def _g2(a, b):
    return _f2(a, b, c0=3.0 / 7.0, c2=4.0 / 7.0)


def _alpha(a, b):
    return _dot(a + b, a) / jnp.maximum(_dot(a, a), _EPS)


def _beta(a, b):
    s = a + b
    return _dot(s, s) * _dot(a, b) / (2.0 * jnp.maximum(_dot(a, a), _EPS) * jnp.maximum(_dot(b, b), _EPS))


def _f3(q1, q2, q3):
    """Symmetric EdS F3 kernel from the standard recursion."""
    total = 0.0
    for qi, qj, qk in ((q1, q2, q3), (q2, q3, q1), (q3, q1, q2)):
        qjk = qj + qk
        g2 = _g2(qj, qk)
        total = total + 7.0 * _alpha(qi, qjk) * _f2(qj, qk) + 2.0 * _beta(qi, qjk) * g2 \
            + g2 * (7.0 * _alpha(qjk, qi) + 2.0 * _beta(qjk, qi))
    return total / 54.0


def tree_trispectrum(k, pk):
    """Tree-level matter trispectrum T(k1, k2, k3, k4) for wavevectors k[i] (i = 0..3, summing to zero)."""
    norms = [jnp.sqrt(_dot(ki, ki)) for ki in k]
    power = [pk(x) for x in norms]
    total = 0.0
    legs = range(4)
    for i in legs:  # 3111: leg i at third order
        j, l, m = (x for x in legs if x != i)
        total = total + 6.0 * _f3(k[j], k[l], k[m]) * power[j] * power[l] * power[m]
    for i in legs:  # 2211: legs i < j at second order, c and d linear
        for j in range(i + 1, 4):
            c, d = (x for x in legs if x not in (i, j))
            for c, d in ((c, d), (d, c)):
                s = k[i] + k[c]
                total = total + 4.0 * power[c] * power[d] * pk(jnp.sqrt(_dot(s, s))) \
                    * _f2(-k[c], s) * _f2(-k[d], -s)
    return total


def _invariant(ell, u1, u2, mu):
    """I = Re sum_m (l l l; 0 m -m) Y_l0(k1) Y_lm(k2) Y_l-m(k3), k1 along z, k2 in the xz plane, k3 = -k1 - k2."""
    from sympy.physics.wigner import wigner_3j

    sin = np.sqrt(np.clip(1.0 - mu**2, 0.0, None))
    k3 = -np.stack([u2 * sin, 0.0 * mu, u1 + u2 * mu])
    u3 = np.sqrt(np.maximum(np.sum(k3**2, axis=0), _EPS))
    theta2, theta3 = np.arccos(mu), np.arccos(np.clip(k3[2] / u3, -1.0, 1.0))
    phi2, phi3 = 0.0, np.arctan2(k3[1], k3[0])
    y10 = sph_harm_y(ell, 0, 0.0, 0.0).real
    out = 0.0
    for m in range(-ell, ell + 1):
        w = float(wigner_3j(ell, ell, ell, 0, m, -m))
        if w:
            out = out + w * y10 * sph_harm_y(ell, m, theta2, phi2) * sph_harm_y(ell, -m, theta3, phi3)
    return np.real(out)


class ZeroLagCumulants:
    """Edgeworth numerators (``trispectrum``, ``bispectrum``) of S1(sigma, l) for several sigma at fixed l."""

    def __init__(self, sigmas, ell, cellsize=0.0, npoints=2**18, nu=96, nmu=48, seed=7):
        self.sigmas = np.asarray(sigmas, dtype="f8")
        self.ell, self.cellsize = int(ell), float(cellsize)
        # Bispectrum: (u1, u2, mu) grid in u = sigma k; 8 pi^2 from the trivial rotations.
        u = np.geomspace(1e-2, 8.0, nu)
        mu, wmu = leggauss(nmu)
        self.u1, self.u2, self.mu = np.meshgrid(u, u, mu, indexing="ij")
        weight = u**2 * log_weights(u)
        self.wbis = (8 * np.pi**2 / (2 * np.pi) ** 6 * weight[:, None, None] * weight[None, :, None] * wmu)
        c3 = (4 * np.pi / (2 * self.ell + 1)) ** 1.5
        self.wbis = self.wbis * c3 * (_invariant(self.ell, self.u1, self.u2, self.mu) if self.ell % 2 == 0 else 0.0)
        # Trispectrum: Gaussian importance sampling of (u1, u2, u3) with u4 = -u1 - u2 - u3, matched to
        # exp(-sum_i u_i^2 / 2) and widened for the (sigma k)^l factors.
        z = norm.ppf(qmc.Sobol(d=9, scramble=True, seed=seed).random(npoints)).reshape(npoints, 3, 3)
        scale2 = (self.ell + 3.0) / 2.0
        chol = np.linalg.cholesky(np.eye(3) - 0.25)  # covariance of (u1, u2, u3) per Cartesian component
        self.utri = np.sqrt(scale2) * np.einsum("ij,njc->nic", chol, z)
        quad = np.sum(self.utri**2, axis=(1, 2)) + np.sum(self.utri.sum(axis=1) ** 2, axis=-1)
        log_density = -quad / (2 * scale2) - 4.5 * np.log(2 * np.pi * scale2) + 1.5 * np.log(4.0)
        self.wtri = np.exp(-log_density) / npoints / (2 * np.pi) ** 9

    def _filter(self, u, sigma):
        """R(k) for u = sigma k."""
        window = jnp.exp(-((u / sigma * self.cellsize) ** 2) / 12.0)
        return window * u**self.ell * jnp.exp(-0.5 * u**2)

    def _bispectrum(self, pk, sigma):
        u1, u2, mu = (jnp.asarray(x) for x in (self.u1, self.u2, self.mu))
        u3 = jnp.sqrt(jnp.maximum(u1**2 + u2**2 + 2 * u1 * u2 * mu, _EPS))
        p1, p2, p3 = pk(u1 / sigma), pk(u2 / sigma), pk(u3 / sigma)
        cos23, cos31 = -(u1 * mu + u2) / u3, -(u1 + u2 * mu) / u3
        bis = 2 * (f2_kernel(u1, u2, mu) * p1 * p2 + f2_kernel(u2, u3, cos23) * p2 * p3
                   + f2_kernel(u3, u1, cos31) * p3 * p1)
        amplitude = jnp.sum(jnp.asarray(self.wbis) * bis * self._filter(u1, sigma) * self._filter(u2, sigma)
                            * self._filter(u3, sigma)) / sigma**6
        return (15.0 if self.ell == 0 else 6.0) * amplitude**2

    def _trispectrum(self, pk, sigma):
        u = jnp.asarray(self.utri)
        legs = [u[:, 0], u[:, 1], u[:, 2], -u.sum(axis=1)]
        norms = [jnp.sqrt(jnp.maximum(_dot(x, x), _EPS)) for x in legs]
        filters = [self._filter(x, sigma) for x in norms]
        pair12 = filters[0] * filters[1] * legendre(self.ell, _dot(legs[0], legs[1]) / (norms[0] * norms[1]))
        pair34 = filters[2] * filters[3] * legendre(self.ell, _dot(legs[2], legs[3]) / (norms[2] * norms[3]))
        trispectrum = tree_trispectrum([x / sigma for x in legs], pk)
        return jnp.sum(jnp.asarray(self.wtri) * trispectrum * pair12 * pair34) / sigma**9

    def __call__(self, pk):
        """Return (trispectrum[n_sigma], bispectrum[n_sigma]) numerators for the linear spectrum pk(k)."""
        def one(sigma):
            return jnp.stack([self._trispectrum(pk, sigma), self._bispectrum(pk, sigma)])

        return jax.lax.map(one, jnp.asarray(self.sigmas)).T
