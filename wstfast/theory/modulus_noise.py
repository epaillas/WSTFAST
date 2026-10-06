"""Tree-level noise of the first-layer WST moduli U = |X|, X_a = (delta * psi_{sigma,l}^a)(x), l >= 1.

The large-scale noise of U, i.e. what is left of P_UU / <U>^2 after the part correlated with delta (the response
term of moduli.py), is the Gaussian second-chaos noise g(K) of moduli.py plus the corrections computed here.
They follow from the Edgeworth expansion of the joint distribution of X(x) and X(y), with the Gaussian moments
of the modulus E[d^{2p} U] = A_{2p} x (sum of pairings), A2 = <U> / (n s^2), A4 = -<U> / (n (n+2) s^4) and
A6 = 3 <U> / (n (n+2) (n+4) s^6), and the cross-covariance C_ab(x - y) of the two points:

* ``G1``: g evaluated with the one-loop instead of the linear power spectrum (linear response of g);
* ``E``: <U>^2 in the normalisation shifted by the Edgeworth correction of <U> (q = 1 in the S1 formula);
* ``T40``: zero-lag trispectrum with two links, 3 K4 / (4 n (n + 2)) g;
* ``T22``: A2^2 / 4 int T(k1, K - k1, k3, -K - k3) F F, without the terms whose internal propagator is P(K)
  (those make up r_long^2 P(K) in the response term), minus (2 r_long r_short + r_short^2) P(K), which the
  response term holds but the covariance of U does not;
* ``T31``: A2 A4 int T(k1, k2, k3, k4) P(k5) F(k1, k2) F(k3, k5) F(k4, -k5), k1 + k2 + k3 + k5 = K;
* ``B22``: A2 A4 / 4 (S1 + 2 S2), products of two tree bispectra with two legs at one point;
* ``B21``: A4^2 / 4 (W1 + 2 W2 + 2 W3 + 4 W4), the same with one link;
* ``G4``: the Gaussian fourth-chaos noise, A4^2 / 24 [3 (sum C^2)^2 + 6 tr (C C^T)^2].

F(p, q) = (-1)^l R(p) R(q) P_l(p.q) is the pair kernel of moduli.py, R(k) = (sigma k)^l exp(-sigma^2 k^2 / 2).
Terms with the zero-lag bispectrum (even l only) are left out: they scale with K33, which is ~1% of K4 for the
fields we use. The sum agrees with the exact tree-level noise measured on perturbation-theory fields
(scripts/feasibility/pt_modulus_noise.py) to 0.01-0.09 g for sigma = 12.5-25 Mpc/h and l = 1-4.

The 6D-12D integrals use scrambled-Sobol points in u = sigma k, importance-sampled on the Gaussian filter
factors, fixed once, so the result is a smooth deterministic function of the linear spectrum. As in moduli.py,
the filters omit the mass-assignment window.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm, qmc

from .cumulants import _EPS, ZeroLagCumulants, _dot, _f2, tree_trispectrum
from .moduli import ModulusClustering, legendre

#: Nodes in u = sigma K; the second layer of a pair with sigma_j2 >= 2 sigma_j1 probes u <~ 3.
UNODES = np.array([0.02, 0.1, 0.25, 0.45, 0.7, 0.95, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0])


def tree_bispectrum(k1, k2, k3, pk):
    """Tree-level matter bispectrum for wavevectors k1 + k2 + k3 = 0 (last axis = components)."""
    power = [pk(jnp.sqrt(jnp.maximum(_dot(x, x), _EPS))) for x in (k1, k2, k3)]
    return 2 * (_f2(k1, k2) * power[0] * power[1] + _f2(k2, k3) * power[1] * power[2]
                + _f2(k3, k1) * power[2] * power[0])


class LegSampler:
    """Quasi-Monte Carlo points for int prod_i d^3u_i / (2 pi)^3 f(legs), with legs = A v + c U zhat.

    v holds the free vectors; the points are Gaussian, matched to exp(-sum_legs |leg|^2 / 2) and widened by
    ``scale2`` for the (sigma k)^l factors of the filters.
    """

    def __init__(self, A, c, scale2, npoints, seed):
        self.A, self.c = np.asarray(A, "f8"), np.asarray(c, "f8")
        nfree = self.A.shape[1]
        precision = self.A.T @ self.A
        covariance = np.linalg.inv(precision) * scale2
        self.mean = -np.linalg.solve(precision, self.A.T @ self.c)  # per unit U, along z
        z = norm.ppf(qmc.Sobol(d=3 * nfree, scramble=True, seed=seed).random(npoints)).reshape(npoints, nfree, 3)
        self.dv = np.einsum("ij,njc->nic", np.linalg.cholesky(covariance), z)
        quad = np.einsum("nic,ij,njc->n", self.dv, np.linalg.inv(covariance), self.dv)
        log_density = -0.5 * quad - 1.5 * nfree * np.log(2 * np.pi) - 1.5 * np.linalg.slogdet(covariance)[1]
        self.weights = np.exp(-log_density) / npoints / (2 * np.pi) ** (3 * nfree)
        self.nfree = nfree

    def legs(self, unode):
        zhat = jnp.array([0.0, 0.0, 1.0])
        v = jnp.asarray(self.dv) + jnp.asarray(self.mean)[None, :, None] * unode * zhat
        A, c = jnp.asarray(self.A), jnp.asarray(self.c)
        return [jnp.einsum("j,njc->nc", A[i], v) + c[i] * unode * zhat for i in range(A.shape[0])]

    def integrate(self, integrand, unodes, sigma):
        """int prod d^3k_i / (2 pi)^3 of integrand(legs) at every node, with k = u / sigma."""
        def one(unode):
            return jnp.sum(jnp.asarray(self.weights) * integrand(self.legs(unode))) / sigma ** (3 * self.nfree)
        return jax.lax.map(one, jnp.asarray(unodes))


class ModulusNoise:
    """Tree-level correction to the noise of U_{sigma,l} (l >= 1), in units of <U>^2, at K = unodes / sigma."""

    def __init__(self, sigma, ell, unodes=UNODES, npoints=2**15, seed=11):
        if ell < 1:
            raise ValueError("the modulus noise is implemented for l >= 1")
        self.sigma, self.ell, self.n = float(sigma), int(ell), 2 * int(ell) + 1
        self.unodes = np.asarray(unodes, dtype="f8")
        self.clustering = ModulusClustering(self.unodes / self.sigma, self.sigma, self.ell)
        self.cumulants = ZeroLagCumulants([self.sigma], self.ell, cellsize=0.0)
        scale2 = (self.ell + 3.0) / 2.0
        # Free vectors and legs of each term (see the module docstring).
        self.t22 = LegSampler([[1, 0], [-1, 0], [0, 1], [0, -1]], [0, 1, 0, -1], scale2, 4 * npoints, seed)
        self.t31 = LegSampler([[1, 0, 0], [0, 1, 0], [-1, -1, -1], [0, 0, 1], [0, 0, 1], [0, 0, -1]],
                              [0, 0, 1, -1, 0, 0], scale2, npoints, seed + 1)
        self.b22 = LegSampler([[1, 0, 0], [0, 1, 0], [-1, -1, 0], [0, 0, 1], [-1, -1, -1], [1, 1, 0]],
                              [0, 0, 0, 0, 1, -1], scale2, npoints, seed + 2)
        self.b21 = LegSampler([[1, 0, 0, 0], [0, 1, 0, 0], [-1, -1, 0, 0], [0, 0, -1, -1], [0, 0, 1, 0],
                               [0, 0, 0, 1], [-1, -1, 1, 1], [1, 1, -1, -1]],
                              [0, 0, 0, 0, 0, 0, 1, -1], scale2, npoints, seed + 3)
        self.g4 = LegSampler([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]] * 2, [0, 0, 0, 1] * 2, scale2, npoints,
                             seed + 4)

    def _filter(self, u):
        return u**self.ell * jnp.exp(-0.5 * u**2)

    def _pair(self, a, b):
        """F(a, b) for u-space vectors."""
        na, nb = jnp.sqrt(jnp.maximum(_dot(a, a), _EPS)), jnp.sqrt(jnp.maximum(_dot(b, b), _EPS))
        return (-1) ** self.ell * self._filter(na) * self._filter(nb) * legendre(self.ell, _dot(a, b) / (na * nb))

    def terms(self, pk, dpk):
        """Every correction (dict of arrays at the nodes) for the linear spectrum pk and one-loop correction dpk."""
        s, n, F = self.sigma, self.n, self._pair
        power = lambda u: pk(jnp.sqrt(jnp.maximum(_dot(u, u), _EPS)) / s)  # noqa: E731
        bispectrum = lambda a, b, c: tree_bispectrum(a / s, b / s, c / s, pk)  # noqa: E731
        ns2 = self.clustering.normalisation(pk)
        s2 = ns2 / n
        r_long, r_short, gauss = self.clustering(pk)
        trispectrum4, bispectrum6 = self.cumulants(pk)[:, 0]
        k4, k33 = trispectrum4 / s2**2, bispectrum6 / (6 * s2**3)
        edgeworth = -k4 / (8 * n * (n + 2)) + 6 * k33 / (24 * n * (n + 2) * (n + 4))
        a2a4 = -1.0 / (n**2 * (n + 2) * s2**3)  # A2 A4 / <U>^2
        a4a4 = 1.0 / (n**2 * (n + 2) ** 2 * s2**4)  # A4^2 / <U>^2

        def t22(legs):
            u1, u2, u3, u4 = legs
            return tree_trispectrum([x / s for x in legs], pk, skip_internal=((0, 1), (2, 3))) * F(u1, u2) * F(u3, u4)

        def t31(legs):
            u1, u2, u3, u4, u5, m5 = legs
            return tree_trispectrum([x / s for x in (u1, u2, u3, u4)], pk) * power(u5) * F(u1, u2) * F(u3, u5) \
                * F(u4, m5)

        def b22(legs):
            k1, k2, k3, l1, l2, l3 = legs
            return bispectrum(k1, k2, k3) * bispectrum(l1, l2, l3) \
                * (F(k1, k2) * F(l1, l2) * F(k3, l3) + 2 * F(k1, l1) * F(k2, l2) * F(k3, l3))

        def b21(legs):
            k1, k2, k3, l1, l2, l3, k5, m5 = legs
            w = F(k1, k2) * F(l2, l3) * F(l1, k5) * F(k3, m5) + 2 * F(k1, k2) * F(k3, l2) * F(l1, k5) * F(l3, m5) \
                + 2 * F(k1, l1) * F(l2, l3) * F(k2, k5) * F(k3, m5) + 4 * F(k1, l1) * F(k3, l2) * F(k2, k5) * F(l3, m5)
            return bispectrum(k1, k2, k3) * bispectrum(l1, l2, l3) * power(k5) * w

        def g4(legs):
            k1, k2, k3, k4 = legs[:4]
            return power(k1) * power(k2) * power(k3) * power(k4) \
                * (3 * F(k1, k2) ** 2 * F(k3, k4) ** 2 + 6 * F(k1, k2) * F(k2, k3) * F(k3, -k4) * F(k1, -k4))

        u, kout = self.unodes, jnp.asarray(self.unodes / s)
        return {
            "G1": self.clustering.noise_response(pk, dpk),
            "E": -2 * edgeworth * gauss,
            "T40": 3 * k4 / (4 * n * (n + 2)) * gauss,
            "T22": self.t22.integrate(t22, u, s) / (2 * ns2) ** 2 - (2 * r_long * r_short + r_short**2) * pk(kout),
            "T31": a2a4 * self.t31.integrate(t31, u, s),
            "B22": a2a4 / 4 * self.b22.integrate(b22, u, s),
            "B21": a4a4 / 4 * self.b21.integrate(b21, u, s),
            "G4": a4a4 / 24 * self.g4.integrate(g4, u, s),
        }

    def __call__(self, pk, dpk):
        """Total correction at the nodes."""
        return sum(self.terms(pk, dpk).values())

    def interpolate(self, values, k):
        """Values at the nodes interpolated to wavenumbers k (white below the first node)."""
        return jnp.interp(self.sigma * jnp.asarray(k), jnp.asarray(self.unodes), values)
