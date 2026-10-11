"""Tree-level connected noise of the line-of-sight-resolved first-layer moduli of a biased, discrete tracer.

Redshift-space, biased-tracer version of modulus_noise.py. For the |m| block M of the wavelet vector X (n = n_M
components of variance lambda) and U = |X_M|, the noise of U beyond the Gaussian chaos value g (rsd_moduli.py) is,
in units of <U>^2 and with the Gaussian moments of the modulus A2 = <U> / (n lambda), A4 = -<U> / (n (n+2) lambda^2):

* ``E``:   -2 E_1 g, the shift of <U>^2 by its Edgeworth correction (q = 1);
* ``T40``: 3 K4 / (4 n (n + 2)) g, the zero-lag trispectrum with two links, K4 = sum_{a,b in M} kappa_aabb / lambda^2;
* ``T22``: A2^2 / 4 int T_d(k1, K - k1, k3, -K - k3) F F, without the terms that make up the response term of the
  model (internal P(K); the Poisson P(K) / n^2 and the P(K) parts of the Poisson B(K, ., .) / n);
* ``T31``: A2 A4 int T_d(k1, k2, k3, k4) P_d(k5) F(k1, k2) F(k3, k5) F(k4, -k5);
* ``B22``, ``B21``: products of two tracer bispectra B_d (with zero and one link);
* ``G4``:  the Gaussian fourth-chaos noise.

F(p, q) = (-1)^l c_l^2 w_M R(p) R(q) Re[Y_lM(p) Y_lM(q)^*] is the block pair kernel, R = (sigma k)^l e^{-sigma^2 k^2/2}.
The tracer correlators are those of a Poisson-sampled biased field: P_d = Z1^2 P_L D^2 + N, B_d = B + N (P + P + P)
+ N^2, T_d = T + N sum B + N^2 (sum P) + N^3, N = V / N_obj, with the tree-level Z_n of the Eulerian bias expansion
including the cubic operators (b3, bG3, bdG2) that the power spectrum does not see. The clustering parts are
velocity-damped on every leg, D(k) = exp(-(f k mu sigma_v)^2 / 2), the Poisson parts are not (as rsd_bias.py).

The terms are evaluated at fixed biases (no monomial expansion), on nodes in u = sigma K and mu_K, with scrambled
Sobol points importance-sampled on the filters (modulus_noise.LegSampler). The Z kernels depend on directions only,
so for a given (l, u, mu_K) they are shared by every sigma and |m|.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y

from .bias import cubic_kernel, galaxy_kernel
from .modulus_noise import LegSampler
from .rsd_bias import linear_components


def _norm(v):
    return np.sqrt(np.sum(v**2, axis=-1))


def _legs(sampler, kvec):
    """Legs (list of (N, 3)) of a LegSampler with the external wavevector kvec (3,) (in units of 1 / sigma)."""
    v = sampler.dv + sampler.mean[None, :, None] * kvec[None, None, :]
    return [np.einsum("j,njc->nc", sampler.A[i], v) + sampler.c[i] * kvec for i in range(sampler.A.shape[0])]


class _Tracer:
    """Tracer correlators at physical wavevectors for fixed biases (beta = (1, b1, b2, bG2, bGamma3), cubic)."""

    def __init__(self, pk, f, sigma_v, beta, cubic, shotnoise):
        self.pk, self.f, self.sigma_v = pk, float(f), float(sigma_v)
        self.beta, self.cubic, self.N = np.asarray(beta, "f8"), np.asarray(cubic, "f8"), float(shotnoise)

    def z1(self, k):
        return linear_components(k, self.f) @ self.beta

    def z2(self, a, b):
        return galaxy_kernel(np.stack([a, b], axis=-2), self.f) @ self.beta

    def z3(self, a, b, c):
        legs = np.stack([a, b, c], axis=-2)
        return galaxy_kernel(legs, self.f) @ self.beta + cubic_kernel(legs) @ self.cubic

    def damp(self, k):
        return np.exp(-0.5 * (self.f * k[..., 2] * self.sigma_v) ** 2)

    def plin(self, k):
        return self.pk(_norm(k))

    def pg(self, k):
        """Undamped tree spectrum Z1^2 P_L (the Poisson pieces)."""
        return self.z1(k) ** 2 * self.plin(k)

    def pd(self, k):
        """Gaussian link: damped tree spectrum plus shot noise."""
        return self.pg(k) * self.damp(k) ** 2 + self.N

    def bispectrum_g(self, q, skip=()):
        """Tree clustering bispectrum (undamped) of q[0] + q[1] + q[2] = 0, without the (a, b) terms in ``skip``."""
        z1 = [self.z1(x) for x in q]
        p = [self.plin(x) for x in q]
        out = 0.0
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (a, b) in skip:
                continue
            out = out + 2 * self.z2(q[a], q[b]) * z1[a] * z1[b] * p[a] * p[b]
        return out

    def bispectrum_d(self, q):
        """B_d = damped B + N (P_g + P_g + P_g) + N^2."""
        damp = np.prod([self.damp(x) for x in q], axis=0)
        return self.bispectrum_g(q) * damp + self.N * sum(self.pg(x) for x in q) + self.N**2

    def trispectrum_d(self, legs, response_pairs=()):
        """T_d at legs[0] + ... + legs[3] = 0. ``response_pairs``: leg pairs (i, j) with legs[i] + legs[j] = +-K whose
        response pieces are left out (internal P(K) in 2211, P_g(K) / n^2, P(K) parts of B(K, ., .) / n)."""
        idx = range(4)
        z1 = [self.z1(x) for x in legs]
        p = [self.plin(x) for x in legs]
        pairs = {frozenset(x) for x in response_pairs}
        tree = 0.0
        for i in idx:  # 3111
            j, l, m = (x for x in idx if x != i)
            tree = tree + 6 * self.z3(legs[j], legs[l], legs[m]) * z1[j] * z1[l] * z1[m] * p[j] * p[l] * p[m]
        for i in idx:  # 2211
            for j in range(i + 1, 4):
                c, d = (x for x in idx if x not in (i, j))
                for c_, d_ in ((c, d), (d, c)):
                    if frozenset((i, c_)) in pairs:
                        continue
                    s = legs[i] + legs[c_]
                    tree = tree + 4 * self.z2(-legs[c_], s) * self.z2(-legs[d_], -s) * z1[c_] * z1[d_] * p[c_] * p[d_] \
                        * self.plin(s)
        tree = tree * np.prod([self.damp(x) for x in legs], axis=0)
        shot = 0.0
        for i in idx:  # N B(k_i + k_j, k_c, k_d)
            for j in range(i + 1, 4):
                c, d = (x for x in idx if x not in (i, j))
                skip = ((0, 1), (2, 0)) if frozenset((i, j)) in pairs else ()  # the terms with P(k_i + k_j) = P(K)
                shot = shot + self.N * self.bispectrum_g((legs[i] + legs[j], legs[c], legs[d]), skip)
        for x in legs:  # N^2 P(k_i)
            shot = shot + self.N**2 * self.pg(x)
        for j in (1, 2, 3):  # N^2 P(k_0 + k_j)
            if frozenset((0, j)) in pairs:
                continue
            shot = shot + self.N**2 * self.pg(legs[0] + legs[j])
        return tree + shot + self.N**3


class ConnectedNoiseRSD:
    """Tree-level connected noise of the moduli U_{sigma, l, |m|} at K = u / sigma (nodes ``unodes``) and mu_K
    (``munodes``), for the fields (sigma, l, m) sharing one l. ``__call__`` returns a dict of terms, each
    (nfield, nu, nmu), in units of <U>^2, together with the Gaussian noise g and the block variance n lambda."""

    def __init__(self, fields, unodes, munodes, npoints=2**14, seed=11, gauss_points=(56, 24, 16)):
        ells = {ell for _, ell, _ in fields}
        if len(ells) != 1:
            raise ValueError("one l per instance")
        self.ell = ells.pop()
        if self.ell < 1:
            raise ValueError("l >= 1")
        self.fields = [(float(s), int(ell), int(m)) for s, ell, m in fields]
        self.unodes, self.munodes = np.asarray(unodes, "f8"), np.asarray(munodes, "f8")
        scale2 = (self.ell + 3.0) / 2.0
        self.t22 = LegSampler([[1, 0], [-1, 0], [0, 1], [0, -1]], [0, 1, 0, -1], scale2, 4 * npoints, seed)
        self.t31 = LegSampler([[1, 0, 0], [0, 1, 0], [-1, -1, -1], [0, 0, 1], [0, 0, 1], [0, 0, -1]],
                              [0, 0, 1, -1, 0, 0], scale2, npoints, seed + 1)
        self.b22 = LegSampler([[1, 0, 0], [0, 1, 0], [-1, -1, 0], [0, 0, 1], [-1, -1, -1], [1, 1, 0]],
                              [0, 0, 0, 0, 1, -1], scale2, npoints, seed + 2)
        self.b21 = LegSampler([[1, 0, 0, 0], [0, 1, 0, 0], [-1, -1, 0, 0], [0, 0, -1, -1], [0, 0, 1, 0],
                               [0, 0, 0, 1], [-1, -1, 1, 1], [1, 1, -1, -1]],
                              [0, 0, 0, 0, 0, 0, 1, -1], scale2, npoints, seed + 3)
        self.g4 = LegSampler([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], [0, 0, 0, 1], scale2, npoints, seed + 4)
        self.zl4 = LegSampler([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], [0, 0, 0, 0], scale2, 4 * npoints, seed + 5)
        self.zl3 = LegSampler([[1, 0], [0, 1], [-1, -1]], [0, 0, 0], scale2, 4 * npoints, seed + 6)
        self.gauss_points = gauss_points

    # ------------------------------------------------------------------------------------------------- kernels
    def _harmonics(self, legs):
        """Y_lm (m = 0..l) of every leg: list of (l + 1, N) complex arrays."""
        out = []
        for x in legs:
            n = np.maximum(_norm(x), 1e-30)
            theta = np.arccos(np.clip(x[:, 2] / n, -1, 1))
            phi = np.arctan2(x[:, 1], x[:, 0])
            out.append(np.array([sph_harm_y(self.ell, m, theta, phi) for m in range(self.ell + 1)]))
        return out

    def _pair(self, legs, ys, a, b, m, sign=(1, 1)):
        """F_M(sign_a legs[a], sign_b legs[b]) in u units; Y(-k) = (-1)^l Y(k). m = -1: the whole multiplet
        (all 2 l + 1 components, F = (-1)^l R R P_l(cos); for tests against the real-space modulus_noise.py)."""
        ell = self.ell
        ua, ub = _norm(legs[a]), _norm(legs[b])
        radial = ua**ell * np.exp(-0.5 * ua**2) * ub**ell * np.exp(-0.5 * ub**2)
        parity = ((-1) ** ell if sign[0] < 0 else 1) * ((-1) ** ell if sign[1] < 0 else 1)
        if m < 0:
            angular = 4 * np.pi / (2 * ell + 1) * (np.real(ys[a][0] * np.conj(ys[b][0]))
                                                   + 2 * sum(np.real(ys[a][k] * np.conj(ys[b][k])) for k in range(1, ell + 1)))
        else:
            angular = 4 * np.pi / (2 * ell + 1) * (1.0 if m == 0 else 2.0) * np.real(ys[a][m] * np.conj(ys[b][m]))
        return (-1) ** ell * radial * parity * angular

    def _size(self, m):
        return 2.0 * self.ell + 1 if m < 0 else (1.0 if m == 0 else 2.0)

    # -------------------------------------------------------------------------------------------- one-point
    def _zero_lag(self, tracer, sigma, m):
        """Block cumulants sum_{a,b in M} kappa_aabb and (m = 0, even l) the third cumulant of X_0."""
        legs = _legs(self.zl4, np.zeros(3))
        ys = self._harmonics(legs)
        phys = [x / sigma for x in legs]
        f12, f34 = self._pair(legs, ys, 0, 1, m), self._pair(legs, ys, 2, 3, m)
        kappa = np.sum(self.zl4.weights * tracer.trispectrum_d(phys) * f12 * f34) / sigma**9
        skew = 0.0
        if m == 0 and self.ell % 2 == 0:
            legs = _legs(self.zl3, np.zeros(3))
            ys = self._harmonics(legs)
            c = np.sqrt(4 * np.pi / (2 * self.ell + 1))
            factors = np.prod([_norm(x) ** self.ell * np.exp(-0.5 * _norm(x) ** 2) * c * np.real(y[0])
                               for x, y in zip(legs, ys)], axis=0)
            skew = np.sum(self.zl3.weights * tracer.bispectrum_d([x / sigma for x in legs]) * factors) / sigma**6
        return kappa, skew

    def _gaussian(self, tracer, sigma, m, kvec):
        """Gaussian noise g (in units of <U>^2) and n lambda at the physical wavevector kvec (direct quadrature)."""
        npts, nx, nphi = self.gauss_points
        p = np.geomspace(2e-3, 7.0 / sigma, npts)
        x, wx = np.polynomial.legendre.leggauss(nx)
        phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
        P, X, PH = np.meshgrid(p, x, phi, indexing="ij")
        sin = np.sqrt(1 - X**2)
        pv = np.stack([P * sin * np.cos(PH), P * sin * np.sin(PH), P * X], axis=-1).reshape(-1, 3)
        w = ((p**3 * np.gradient(np.log(p)))[:, None, None] * wx[None, :, None] * (2 * np.pi / nphi) / (2 * np.pi) ** 3
             * np.ones_like(P)).ravel()
        q = kvec[None, :] - pv
        ys = self._harmonics([pv * sigma, q * sigma])
        legs = [pv * sigma, q * sigma]
        up = _norm(legs[0])
        if m < 0:
            angular = 1.0  # c_l^2 sum_m |Y_lm|^2
        else:
            angular = 4 * np.pi / (2 * self.ell + 1) * (1.0 if m == 0 else 2.0) * np.abs(ys[0][m]) ** 2
        norm = np.sum(w * tracer.pd(pv) * angular * up ** (2 * self.ell) * np.exp(-up**2))
        F = self._pair(legs, ys, 0, 1, m)
        g = np.sum(w * tracer.pd(pv) * tracer.pd(q) * F**2) / (2 * norm**2)
        return g, norm

    # ------------------------------------------------------------------------------------------------- terms
    def __call__(self, pk, f, sigma_v, beta, cubic, shotnoise, terms=("E", "T40", "T22", "T31", "B22", "B21", "G4")):
        tracer = _Tracer(pk, f, sigma_v, beta, cubic, shotnoise)
        nf, nu, nm = len(self.fields), len(self.unodes), len(self.munodes)
        out = {name: np.zeros((nf, nu, nm)) for name in terms + ("g", "norm")}
        one_point = {}
        for i, (sigma, ell, m) in enumerate(self.fields):
            n = self._size(m)
            for a, u in enumerate(self.unodes):
                for b, mu in enumerate(self.munodes):
                    kvec = u / sigma * np.array([np.sqrt(1 - mu**2), 0.0, mu])
                    g, norm = self._gaussian(tracer, sigma, m, kvec)
                    out["g"][i, a, b], out["norm"][i, a, b] = g, norm
            if {"E", "T40"} & set(terms):
                kappa, skew = self._zero_lag(tracer, sigma, m)
                lam = out["norm"][i, 0, 0] / n
                one_point[i] = (kappa / lam**2, kappa, skew, lam)
                k4 = kappa / lam**2
                # NLO Edgeworth factor of <U> (q = 1) of one block, as rsd.block_edgeworth:
                # q (q - 2) K4 / (8 n (n + 2)) + [m = 0, even l] q (q - 2) (q - 4) kappa_3^2 / (72 lambda^3)
                e1 = -k4 / (8 * n * (n + 2)) + (3 * skew**2 / (72 * lam**3) if m == 0 and ell % 2 == 0 else 0.0)
                if "E" in terms:
                    out["E"][i] = -2 * e1 * out["g"][i]
                if "T40" in terms:
                    out["T40"][i] = 3 * k4 / (4 * n * (n + 2)) * out["g"][i]
        self.one_point = one_point
        for a, u in enumerate(self.unodes):
            for b, mu in enumerate(self.munodes):
                khat = np.array([np.sqrt(1 - mu**2), 0.0, mu])
                self._terms_at(tracer, u * khat, a, b, out, terms)
        return out

    def _terms_at(self, tracer, kvec_u, a, b, out, terms):
        """All 6- to 12-dimensional terms at one node, for every field (kernels shared across sigma and m)."""
        jobs = []
        if "T22" in terms:
            legs = _legs(self.t22, kvec_u)
            jobs.append(("T22", self.t22, legs, self._harmonics(legs)))
        if "T31" in terms:
            legs = _legs(self.t31, kvec_u)
            jobs.append(("T31", self.t31, legs, self._harmonics(legs)))
        if "B22" in terms:
            legs = _legs(self.b22, kvec_u)
            jobs.append(("B22", self.b22, legs, self._harmonics(legs)))
        if "B21" in terms:
            legs = _legs(self.b21, kvec_u)
            jobs.append(("B21", self.b21, legs, self._harmonics(legs)))
        if "G4" in terms:
            legs = _legs(self.g4, kvec_u)
            jobs.append(("G4", self.g4, legs, self._harmonics(legs)))
        for sigma in sorted({s for s, _, _ in self.fields}):
            members = [i for i, (s, _, _) in enumerate(self.fields) if s == sigma]
            for name, sampler, legs, ys in jobs:
                phys = [x / sigma for x in legs]
                # tracer correlators: shared by the |m| fields of this sigma
                if name == "T22":
                    common = tracer.trispectrum_d(phys, response_pairs=((0, 1), (2, 3)))
                elif name == "T31":
                    common = tracer.trispectrum_d(phys[:4]) * tracer.pd(phys[4])
                elif name in ("B22", "B21"):
                    common = tracer.bispectrum_d(phys[0:3]) * tracer.bispectrum_d(phys[3:6])
                    if name == "B21":
                        common = common * tracer.pd(phys[6])
                else:
                    common = tracer.pd(phys[0]) * tracer.pd(phys[1]) * tracer.pd(phys[2]) * tracer.pd(phys[3])
                common = sampler.weights * common / sigma ** (3 * sampler.nfree)
                for i in members:
                    m = self.fields[i][2]
                    n = self._size(m)
                    nlam = out["norm"][i, a, b]
                    lam = nlam / n
                    a2a4 = -1.0 / (n**2 * (n + 2) * lam**3)  # A2 A4 / <U>^2
                    a4a4 = 1.0 / (n**2 * (n + 2) ** 2 * lam**4)  # A4^2 / <U>^2
                    F = lambda x, y, s=(1, 1): self._pair(legs, ys, x, y, m, s)  # noqa: E731
                    if name == "T22":
                        kernel, scale = F(0, 1) * F(2, 3), 1.0 / (2 * nlam) ** 2
                    elif name == "T31":
                        kernel, scale = F(0, 1) * F(2, 4) * F(3, 5), a2a4
                    elif name == "B22":
                        kernel, scale = F(0, 1) * F(3, 4) * F(2, 5) + 2 * F(0, 3) * F(1, 4) * F(2, 5), a2a4 / 4
                    elif name == "B21":
                        kernel = (F(0, 1) * F(4, 5) * F(3, 6) * F(2, 7) + 2 * F(0, 1) * F(2, 4) * F(3, 6) * F(5, 7)
                                  + 2 * F(0, 3) * F(4, 5) * F(1, 6) * F(2, 7) + 4 * F(0, 3) * F(2, 4) * F(1, 6) * F(5, 7))
                        scale = a4a4 / 4
                    else:  # G4
                        kernel = (3 * F(0, 1) ** 2 * F(2, 3) ** 2
                                  + 6 * F(0, 1) * F(1, 2) * F(2, 3, (1, -1)) * F(0, 3, (1, -1)))
                        scale = a4a4 / 24
                    out[name][i, a, b] = scale * np.sum(common * kernel)


#: Default nodes in u = sigma K (the second-layer filters with sigma_2 >= 70 Mpc/h weight sigma_1 K <~ 1.5 for
#: sigma_1 = 25 Mpc/h) and the number of Gauss nodes in mu_K on [0, 1].
UNODES = np.array([0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.6])
NMU_NODES = 3
TERMS = ("E", "T40", "T22", "T31", "B22", "B21", "G4")


def noise_ratio(fields, pk, f, sigma_v, beta, cubic, shotnoise, unodes=UNODES, nmu=NMU_NODES, npoints=2**14,
                terms=TERMS):
    """rho = Delta N / g of every field (sigma, l, m) at the nodes: (nfield, nu, nmu) arrays of the total and of each
    term, the Gaussian noise g, and the nodes (unodes, munodes)."""
    munodes = np.polynomial.legendre.leggauss(2 * nmu)[0][nmu:]
    out = {name: np.zeros((len(fields), len(unodes), nmu)) for name in terms + ("g",)}
    for ell in sorted({ell for _, ell, _ in fields}):
        members = [i for i, (_, l, _) in enumerate(fields) if l == ell]
        cn = ConnectedNoiseRSD([fields[i] for i in members], unodes, munodes, npoints=npoints)
        values = cn(pk, f, sigma_v, beta, cubic, shotnoise, terms=terms)
        for name in out:
            out[name][members] = values[name]
    out["rho"] = sum(out[name] for name in terms) / out["g"]
    out["unodes"], out["munodes"] = np.asarray(unodes), munodes
    return out


def interpolate_ratio(rho, unodes, munodes, sigmas, k, mu):
    """rho (nfield, nu, nmu) at the nodes -> (nfield, nk, nmu_out) on the grid (k, mu): linear in u = sigma k (constant
    beyond the nodes), Lagrange in mu^2 through the Gauss nodes."""
    from .rsd import lagrange_matrix

    wmu = lagrange_matrix(np.asarray(munodes) ** 2, np.asarray(mu) ** 2)  # (nmu_out, nmu)
    out = []
    for r, sigma in zip(rho, sigmas):
        along = np.array([np.interp(sigma * np.asarray(k), unodes, r[:, b]) for b in range(r.shape[1])]).T  # (nk, nmu)
        out.append(along @ wmu.T)
    return np.array(out)
