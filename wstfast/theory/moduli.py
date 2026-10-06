"""Large-scale clustering of the first-layer WST modulus fields U_{j,l} = |delta * psi_{j,l}|.

For l >= 1 the second-chaos projection U ~ <U> + <U> / (2 n s^2) (|X|^2 - n s^2) gives, with the
pair kernel F(p, q) = sum_m psi^m(p) psi^m(q) = (-1)^l (sigma^2 p q)^l P_l(p.q) e^{-sigma^2 (p^2 + q^2) / 2}
and n s^2 = int_p P(p) (sigma p)^{2l} e^{-sigma^2 p^2},

    response   r(k) = P_{U delta}(k) / (<U> P(k)) = int_p B(p, k - p, -k) F(p, k - p) / (2 P(k) n s^2),
    noise      g(k) = P^G_{UU}(k) / <U>^2      = int_p P(p) P(|k - p|) F(p, k - p)^2 / (2 (n s^2)^2),

with the tree-level matter bispectrum B. Both are normalised by <U>, so they are independent of the
non-perturbative one-point amplitude of U (docs/wst_eft_feasibility.md, Secs. 2.3 and 3.3).

The response is returned in two parts, r = r_long + r_short: r_long holds the bispectrum terms with P(k)
(the long mode enters at linear order), r_short the term 2 F2(p, q) P(p) P(q) (the second-order density of
the short modes). The trispectrum noise of U (cumulants.py) needs them separately.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

from .perturbation import f2_kernel, log_weights


def legendre(ell, x):
    """Legendre polynomial P_l(x) by upward recursion."""
    p0, p1 = jnp.ones_like(x), x
    if ell == 0:
        return p0
    for n in range(1, ell):
        p0, p1 = p1, ((2 * n + 1) * x * p1 - n * p0) / (n + 1)
    return p1


class ModulusClustering:
    """Tree-level response r(k) = r_long + r_short and Gaussian noise g(k) of U_{sigma,l} (l >= 1) on a k grid."""

    def __init__(self, kout, sigma, ell, npts=192, nmu=64):
        if ell < 1:
            raise ValueError("the second-chaos response is implemented for l >= 1")
        self.kout = np.asarray(kout, dtype="f8")
        self.sigma, self.ell = float(sigma), int(ell)
        self.p = np.geomspace(1e-4, 8.0 / sigma, npts)
        self.wp = self.p**2 * log_weights(self.p) / (4 * np.pi**2)  # p^2 dp / (4 pi^2); x dmu below
        self.mu, self.wmu = leggauss(nmu)

    def normalisation(self, pk):
        """n s^2 = int_p P(p) (sigma p)^{2l} exp(-sigma^2 p^2)."""
        p = jnp.asarray(self.p)
        w = (self.sigma * p) ** (2 * self.ell) * jnp.exp(-((self.sigma * p) ** 2))
        return 2 * jnp.sum(pk(p) * w * jnp.asarray(self.wp))  # int dmu = 2

    def _geometry(self):
        k = jnp.asarray(self.kout)[:, None, None]
        p = jnp.asarray(self.p)[None, :, None]
        mu = jnp.asarray(self.mu)[None, None, :]
        q = jnp.sqrt(jnp.maximum(k**2 + p**2 - 2 * k * p * mu, 1e-30))
        cos_pq = jnp.clip((k * p * mu - p**2) / (p * q), -1.0, 1.0)
        sigma2 = self.sigma**2
        kernel = (-1) ** self.ell * (sigma2 * p * q) ** self.ell * legendre(self.ell, cos_pq) \
            * jnp.exp(-0.5 * sigma2 * (p**2 + q**2))
        weights = jnp.asarray(self.wp)[None, :, None] * jnp.asarray(self.wmu)[None, None, :]
        return k, p, mu, q, cos_pq, kernel, weights

    def noise_response(self, pk, dpk):
        """Linear change of g(k) when P -> P + dpk (in the propagators and in n s^2)."""
        _, p, _, q, _, kernel, weights = self._geometry()
        norm, dnorm = self.normalisation(pk), self.normalisation(dpk)
        noise = jnp.sum(weights * pk(p) * pk(q) * kernel**2, axis=(1, 2)) / (2 * norm**2)
        dnoise = jnp.sum(weights * (dpk(p) * pk(q) + pk(p) * dpk(q)) * kernel**2, axis=(1, 2)) / (2 * norm**2)
        return dnoise - 2 * noise * dnorm / norm

    def __call__(self, pk):
        """Return (r_long(k), r_short(k), g(k)) on the output grid."""
        k, p, mu, q, cos_pq, kernel, weights = self._geometry()
        cos_qk = -(k**2 - k * p * mu) / (k * q)
        pk_p, pk_q, pk_k = pk(p), pk(q), pk(k)
        long = 2 * (f2_kernel(q, k, cos_qk) * pk_q + f2_kernel(k, p, -mu) * pk_p) * pk_k
        short = 2 * f2_kernel(p, q, cos_pq) * pk_p * pk_q
        norm = self.normalisation(pk)
        r_long, r_short = (jnp.sum(weights * b * kernel, axis=(1, 2)) / (2 * pk_k[:, 0, 0] * norm) for b in (long, short))
        noise = jnp.sum(weights * pk_p * pk_q * kernel**2, axis=(1, 2)) / (2 * norm**2)
        return r_long, r_short, noise
