"""First layer of the line-of-sight-resolved WST: S1m(j, l, |m|) of a biased, discrete tracer (line of sight z).

In redshift space the n = 2l + 1 components of X_a = (delta_s * psi_{sigma,l}^a)(x) are independent Gaussians at
leading order but not equally distributed: by symmetry about the line of sight, the variance of each component
depends on |m| only. Each |m| block M (n_M = 1 component for m = 0, 2 otherwise) is then an isotropic Gaussian of
variance lambda_M per component, the lattice sum of the filtered P_s (``spectrum.S1mProjection``), and

    S1m = (2 lambda_M)^{q/2} Gamma((n_M + q) / 2) / Gamma(n_M / 2) [1 + E_M],

with the NLO Edgeworth factor of an O(n_M)-invariant modulus (``block_edgeworth``),

    E_M = q (q - 2) K4_M / (8 n_M (n_M + 2)) + q (q - 2) (q - 4) 15 K33 / (72 * 15)   (the last for m = 0, even l),

K4_M = sum_{a,b in M} kappa_aabb / lambda_M^2 from the tree trispectrum of the tracer (plus its Poisson terms),
contracted with the block pair kernels F_M(p, q) = c_l^2 R R w_M Re[Y_lM(p) Y_lM(q)^*]. The squared zero-lag
bispectrum enters for m = 0 and even l only. ``S1mBasis`` computes these cumulants, resolved into the monomials of beta
(``layout``), by quasi-Monte Carlo integration (``LegSampler``); the Fingers-of-God counterterm (``fog``) and the AP
polynomial (``ap``) of the observed-frame cumulants are optional. Real space is f = 0, where the blocks of a
coefficient share one variance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln, sph_harm_y
from scipy.stats import norm, qmc

from ..config import WSTConfig
from .bias import CUBIC_NAMES, cubic_kernel, galaxy_kernel, linear_components, product_coefficients
from .layout import CUBIC_KAPPA_SIZES, FOG_KNL, KAPPA_SIZES, SKEW_SIZES, fog_sizes


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


def spherical_harmonics(keys, vec):
    """{(l, m): Y_lm} at the direction of ``vec`` (..., 3) for every (l, m) of ``keys``, equal to
    ``scipy.special.sph_harm_y(l, m, theta, phi)`` (same normalization and Condon-Shortley phase) but from the
    Cartesian components, about 10x faster: the normalized associated Legendre recurrence in cos(theta), shared by all l
    of a given |m|, times (sin(theta) e^{i phi})^|m| = ((x + i y) / r)^|m|. The zero vector takes the direction
    (1, 0, 0), as scipy at theta = pi / 2, phi = 0."""
    vec = np.asarray(vec, "f8")
    norm = np.sqrt(np.sum(vec**2, axis=-1))
    zero = norm == 0
    inv = 1.0 / np.where(zero, 1.0, norm)
    z = vec[..., 2] * inv
    w = np.where(zero, 1.0, (vec[..., 0] + 1j * vec[..., 1]) * inv)
    keys = [(int(ell), int(m)) for ell, m in keys]
    out = {}
    for a in sorted({abs(m) for _, m in keys}):
        ells = {ell for ell, m in keys if abs(m) == a}
        value = np.full(z.shape, (-1.0) ** a * np.sqrt((2 * a + 1) / (4 * np.pi)
                                                        * np.prod([(2 * k - 1) / (2 * k) for k in range(1, a + 1)])))
        phase = w**a if a else 1.0
        legendre = {a: value}
        if max(ells) > a:  # P_l^m from P_m^m and P_{m+1}^m
            previous, value = value, np.sqrt(2 * a + 3) * z * value
            legendre[a + 1] = value
            for ell in range(a + 2, max(ells) + 1):
                c = np.sqrt((4 * ell**2 - 1) / (ell**2 - a**2))
                c1 = np.sqrt((4 * (ell - 1) ** 2 - 1) / ((ell - 1) ** 2 - a**2))
                previous, value = value, c * (z * value - previous / c1)
                legendre[ell] = value
        for ell in ells:
            y = legendre[ell] * phase
            for m in {m for l, m in keys if l == ell and abs(m) == a}:
                out[ell, m] = (y if m >= 0 else (-1) ** a * np.conj(y)).astype(complex)
    return out


def spherical_harmonic(ell, m, vec):
    """Y_lm at the direction of ``vec`` (..., 3), as ``scipy.special.sph_harm_y`` (``spherical_harmonics``)."""
    return spherical_harmonics([(ell, m)], vec)[ell, m]


def angular_weights(ell, mu):
    """c_l^2 |Y_l|m|(mu)|^2 for |m| = 0..l, shape (l + 1, nmu); they sum (with multiplicity) to 1."""
    c2 = 4 * np.pi / (2 * ell + 1)
    theta = np.arccos(np.clip(mu, -1, 1))
    return np.array([c2 * np.abs(sph_harm_y(ell, m, theta, 0.0)) ** 2 for m in range(ell + 1)])


def multiplicity(ell):
    return np.array([1.0] + [2.0] * ell)


def _block_angles(ell, legs):
    """Angular factors at the four legs (sign conventions cancel in the products used):

    pairs (2, l+1, N): c_l^2 w_M Re[Y_lM(k_1) Y_lM(k_2)^*] and the same for legs 3, 4;
    components (l+1, N): c_l^4 sum_{a in block M} prod_i Y_a(k_i) over the real harmonics of block M.
    """
    c2 = 4 * np.pi / (2 * ell + 1)
    ys = []
    for leg in legs:
        y = spherical_harmonics([(ell, m) for m in range(ell + 1)], leg)
        ys.append(np.array([y[ell, m] for m in range(ell + 1)]))  # (l+1, N) complex
    w = multiplicity(ell)[:, None]
    pair12 = c2 * w * np.real(ys[0] * np.conj(ys[1]))
    pair34 = c2 * w * np.real(ys[2] * np.conj(ys[3]))
    comps = np.zeros((ell + 1, legs[0].shape[0]))
    for m in range(ell + 1):
        if m == 0:
            comps[0] = np.prod([np.real(y[0]) for y in ys], axis=0)
        else:
            re = np.prod([np.sqrt(2) * np.real(y[m]) for y in ys], axis=0)
            im = np.prod([np.sqrt(2) * np.imag(y[m]) for y in ys], axis=0)
            comps[m] = re + im
    return np.stack([pair12, pair34]), c2**2 * comps


class _ScaleFreeTerms:
    """Sum_t C_t(n) prod_{v in V_t} P_L(|v(n)| / sigma) over sample points n: the kernel coefficients C_t (scale-free:
    Z kernels depend on directions only) are built once, the spectra per sigma. Terms with the same degree are
    stacked: ``coefficients[d]`` (nterm, N, nmono(d)) and ``index[d]`` (nterm, nfactor) into the distinct leg norms
    ``unique`` (nunique, N): the terms share a few legs, so P_L is evaluated once per distinct norm and sigma (and
    reused across the degrees at the same sigma)."""

    def __init__(self):
        self.terms, self._lookup, self._unique = {}, {}, []

    def add(self, degree, coefficients, vectors):
        index = []
        for v in vectors:
            norm = np.sqrt(np.sum(v**2, axis=-1))
            key = norm.tobytes()
            if key not in self._lookup:
                self._lookup[key] = len(self._unique)
                self._unique.append(norm)
            index.append(self._lookup[key])
        self.terms.setdefault(degree, []).append((coefficients, index))

    def freeze(self):
        self.coefficients = {d: np.stack([c for c, _ in t]) for d, t in self.terms.items()}
        self.index = {d: np.array([i for _, i in t], dtype=int).reshape(len(t), -1) for d, t in self.terms.items()}
        self.unique = np.array(self._unique).reshape(len(self._unique), -1)
        self._cache = (None, None, None)
        del self.terms, self._lookup, self._unique

    def __call__(self, degree, scale, pk, weight):
        """sum_n weight(n, ...) sum_t C_t(n) prod P_L: weight (N, nout) -> (nout, nmono(degree))."""
        if self._cache[0] is not pk or self._cache[1] != scale:
            self._cache = (pk, scale, pk(self.unique / scale))
        factors = np.prod(self._cache[2][self.index[degree]], axis=1)  # (nterm, N); 1 for a term without spectra
        combined = np.einsum("tnm,tn->nm", self.coefficients[degree], factors)  # (N, nmono)
        return weight.T @ combined


def _trispectrum_terms(legs, f):
    """Tree trispectrum (degree 4) and Poisson terms B (3), P (2) and 1 (0) at unit legs k_0 + ... + k_3 = 0."""
    terms = _ScaleFreeTerms()
    idx = range(4)
    z1 = [linear_components(x, f) for x in legs]
    for i in idx:  # 3111
        j, l, m = (x for x in idx if x != i)
        z3 = galaxy_kernel(np.stack([legs[j], legs[l], legs[m]], axis=-2), f)
        terms.add(4, 6.0 * product_coefficients(z3, z1[j], z1[l], z1[m]), [legs[j], legs[l], legs[m]])
    for i in idx:  # 2211
        for j in range(i + 1, 4):
            c, d = (x for x in idx if x not in (i, j))
            for c_, d_ in ((c, d), (d, c)):
                s = legs[i] + legs[c_]
                z2a = galaxy_kernel(np.stack([-legs[c_], s], axis=-2), f)
                z2b = galaxy_kernel(np.stack([-legs[d_], -s], axis=-2), f)
                terms.add(4, 4.0 * product_coefficients(z2a, z2b, z1[c_], z1[d_]), [legs[c_], legs[d_], s])
    for i in idx:  # Poisson B(k_i + k_j, k_c, k_d) / n
        for j in range(i + 1, 4):
            c, d = (x for x in idx if x not in (i, j))
            _add_bispectrum(terms, (legs[i] + legs[j], legs[c], legs[d]), f)
    for v in list(legs) + [legs[0] + legs[1], legs[0] + legs[2], legs[0] + legs[3]]:  # Poisson P / n^2
        z = linear_components(v, f)
        terms.add(2, product_coefficients(z, z), [v])
    terms.add(0, np.ones((legs[0].shape[0], 1)), [])  # 1 / n^3
    terms.freeze()
    return terms


def _cubic_trispectrum_terms(legs, f):
    """T_3111 of the cubic operators at unit legs: one term group per operator (keyed by its name), each the
    coefficients of degree 3 in beta of sum_i 6 K_c(k_j, k_l, k_m) Z1(k_j) Z1(k_l) Z1(k_m) P P P."""
    terms = _ScaleFreeTerms()
    z1 = [linear_components(x, f) for x in legs]
    for i in range(4):
        j, l, m = (x for x in range(4) if x != i)
        kernels = cubic_kernel(np.stack([legs[j], legs[l], legs[m]], axis=-2))  # (N, 3)
        z111 = product_coefficients(z1[j], z1[l], z1[m])
        for c, name in enumerate(CUBIC_NAMES):
            terms.add(name, 6.0 * kernels[:, c, None] * z111, [legs[j], legs[l], legs[m]])
    terms.freeze()
    return terms


def _add_bispectrum(terms, q, f):
    """Tree bispectrum terms 2 Z2(q_a, q_b) Z1(q_a) Z1(q_b) P(q_a) P(q_b) (degree 3) for q_0 + q_1 + q_2 = 0."""
    z1 = [linear_components(x, f) for x in q]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        z2 = galaxy_kernel(np.stack([q[a], q[b]], axis=-2), f)
        terms.add(3, 2 * product_coefficients(z2, z1[a], z1[b]), [q[a], q[b]])


def _bispectrum_terms(legs, f):
    """Tree bispectrum (degree 3) and Poisson terms P (2) and 1 (0) at unit legs k_0 + k_1 + k_2 = 0."""
    terms = _ScaleFreeTerms()
    _add_bispectrum(terms, legs, f)
    for v in legs:
        z = linear_components(v, f)
        terms.add(2, product_coefficients(z, z), [v])
    terms.add(0, np.ones((legs[0].shape[0], 1)), [])
    terms.freeze()
    return terms


def _fog_weights(weight, legs):
    """[w | w S | w S^2 / 2] along the output axis, S = sum over legs of k_z^2 / k_NL^2 at each sample: orders 0, 1, 2
    of the Fingers-of-God factor exp(-c_v S) = 1 - c_v S + c_v^2 S^2 / 2 + ..., whose first order is the counterterm
    of Tucci & Philcox (2026) and whose expansion at c_v = (f sigma_v k_NL)^2 / 2 is the Gaussian velocity damping."""
    s = sum(x[:, 2] ** 2 for x in legs) / FOG_KNL**2
    return np.concatenate([weight, weight * s[:, None], weight * (0.5 * s**2)[:, None]], axis=1)


class S1mBasis:
    """Monomial-resolved cumulants of the S1m blocks of the (j, l) ``coefficients`` (see the module docstring and
    ``layout``): ``cumulants(klin, pklin, f)`` returns ``kappa`` (ncoef, L + 1, ncol), per block sum_{a,b in M}
    kappa_aabb in the columns [tree | B/n | P/n^2 | 1/n^3 [| b3 | bG3 | bdG2]], and ``skewness`` (ncoef, nskew), the
    m = 0 third cumulant (even l) in [tree | P/n | 1/n^2]. By azimuthal symmetry, blocks with |m| > 0 have no third
    cumulant, and neither has the m = 0 component of odd l (z -> -z parity).

    ``damping``: velocity damping of the clustering legs, exp(-(f k mu sigma_v)^2 / 2): None, "linear" (sigma_v^2 =
    int P_L dk / (6 pi^2)) or a fixed sigma_v [Mpc/h]. With ``cubic``, the kappa columns also hold the cubic-operator
    trispectrum (CUBIC_KAPPA_SIZES). With ``fog``, the clustering pieces (tree and cubic trispectrum, tree bispectrum)
    are not velocity-damped but carry the Fingers-of-God counterterm of every external leg instead, stored as the
    undamped columns (order 0) followed by the orders 1 and 2 of ``_fog_weights`` (``layout.fog_sizes``);
    ``layout.collapse_fog`` combines them at a given c_v. With ``ap``, the AP polynomial of the observed-frame
    cumulants (``_cumulants_ap``). The integrals use ``npoints`` scrambled Sobol points with fixed seeds.
    """

    def __init__(self, config: WSTConfig, coefficients, npoints: int = 2**16, damping: str | float | None = None,
                 cubic: bool = False, fog: bool = False, ap: int = 0, ap_nodes: int = 5):
        self.config, self.coefficients = config, list(coefficients)
        self.cubic, self.fog = bool(cubic), bool(fog)
        self.damping = None if self.fog else damping
        #: Order of the AP polynomial of the observed-frame cumulants (0: none; ``_cumulants_ap``).
        self.ap = int(ap)
        if self.ap and not self.fog:
            raise ValueError("ap needs fog=True")
        self.ap_nodes = int(ap_nodes)
        self.lmax = max(c.ell for c in self.coefficients)
        self.samplers, self.block_angles = {}, {}
        for ell in sorted({c.ell for c in self.coefficients}):
            sampler = LegSampler([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], [0, 0, 0, 0], (ell + 3) / 2, npoints,
                                 seed=17 + ell)
            self.samplers[ell] = sampler
            self.block_angles[ell] = _block_angles(ell, [np.asarray(x) for x in sampler.legs(0.0)])
        self.bis_samplers = {ell: LegSampler([[1, 0], [0, 1], [-1, -1]], [0, 0, 0], (ell + 3) / 2, npoints, seed=29 + ell)
                             for ell in sorted({c.ell for c in self.coefficients}) if ell % 2 == 0}

    def _filter(self, u, ell, sigma):
        return np.exp(-((u / sigma) * self.config.cellsize) ** 2 / 12.0) * u**ell * np.exp(-0.5 * u**2)

    def _damp(self, k):
        """exp(-(f k_z sigma_v)^2 / 2) for wavevectors k (N, 3)."""
        return np.exp(-0.5 * (self.f * k[:, 2] * self.sigma_v) ** 2)

    def cumulants(self, klin, pklin, f):
        """``kappa`` (ncoef, L + 1, sum KAPPA_SIZES [+ sum CUBIC_KAPPA_SIZES]): per block sum_{a,b in M} kappa_aabb in
        the columns [tree | B/n | P/n^2 | 1/n^3 [| b3 | bG3 | bdG2]]; ``skewness`` (ncoef, sum SKEW_SIZES): the m = 0
        third cumulant (even l) in [tree | P/n | 1/n^2]. With ``ap``, the AP polynomial of these (``_cumulants_ap``)."""
        if self.ap:
            return self._cumulants_ap(klin, pklin, f)
        logk, logp = np.log(klin), np.log(pklin)
        pk = lambda x: np.where(np.asarray(x) > 0, np.exp(np.interp(np.log(np.maximum(x, 1e-30)), logk, logp)), 0.0)  # noqa
        self.sigma_v = (0.0 if self.damping is None else float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
                        if self.damping == "linear" else float(self.damping))
        self.f = f
        nb = self.lmax + 1
        ncol = sum(KAPPA_SIZES) + (sum(CUBIC_KAPPA_SIZES) if self.cubic else 0)
        extra = fog_sizes(self.cubic) if self.fog else dict(kappa=0, skewness=0)
        kappa = np.zeros((len(self.coefficients), nb, ncol + extra["kappa"]))
        skewness = np.zeros((len(self.coefficients), sum(SKEW_SIZES) + extra["skewness"]))
        for ell in sorted({c.ell for c in self.coefficients}):  # kernels once per l, spectra per sigma
            members = [i for i, c in enumerate(self.coefficients) if c.ell == ell]
            legs = [np.asarray(x) for x in self.samplers[ell].legs(0.0)]
            terms = _trispectrum_terms(legs, f)
            cubic = _cubic_trispectrum_terms(legs, f) if self.cubic else None
            (pair12, pair34), _ = self.block_angles[ell]
            angles = (pair12 * pair34).T  # (N, l + 1): diagonal block sums
            for index in members:
                sigma = self.config.sigma(self.coefficients[index].j)
                filters = np.prod([self._filter(np.sqrt(np.sum(x**2, axis=-1)), ell, sigma) for x in legs], axis=0)
                damping = np.prod([self._damp(x / sigma) for x in legs], axis=0)
                weight = (self.samplers[ell].weights * filters / sigma**9)[:, None] * angles
                # velocity-damped legs (or the FoG counterterm) for the clustering piece only: the Poisson pieces are not
                if self.fog:
                    fog = _fog_weights(weight, [x / sigma for x in legs])
                    tree = np.split(terms(4, sigma, pk, fog), 3)
                    cubics = [np.split(cubic(name, sigma, pk, fog), 3) for name in CUBIC_NAMES] if cubic else []
                    columns = ([tree[0]] + [terms(d, sigma, pk, weight) for d in (3, 2, 0)] + [c[0] for c in cubics]
                               + [x for order in (1, 2) for x in [tree[order]] + [c[order] for c in cubics]])
                else:
                    columns = ([terms(4, sigma, pk, weight * damping[:, None])]
                               + [terms(d, sigma, pk, weight) for d in (3, 2, 0)])
                    if cubic is not None:  # clustering pieces: damped like the tree trispectrum
                        columns += [cubic(name, sigma, pk, weight * damping[:, None]) for name in CUBIC_NAMES]
                kappa[index, :ell + 1] = np.concatenate(columns, axis=-1)
            del terms, cubic
            if ell % 2:
                continue
            legs = [np.asarray(x) for x in self.bis_samplers[ell].legs(0.0)]
            terms = _bispectrum_terms(legs, f)
            c = np.sqrt(4 * np.pi / (2 * ell + 1))
            ys = [c * np.real(spherical_harmonic(ell, 0, x)) for x in legs]
            for index in members:
                sigma = self.config.sigma(self.coefficients[index].j)
                factors = np.prod([self._filter(np.sqrt(np.sum(x**2, axis=-1)), ell, sigma) * y for x, y in zip(legs, ys)],
                                  axis=0)
                damping = np.prod([self._damp(x / sigma) for x in legs], axis=0)
                weight = (self.bis_samplers[ell].weights * factors / sigma**6)[:, None]
                if self.fog:
                    tree = terms(3, sigma, pk, _fog_weights(weight, [x / sigma for x in legs]))
                    skewness[index] = np.concatenate([tree[0]] + [terms(d, sigma, pk, weight)[0] for d in (2, 0)]
                                                     + [tree[1], tree[2]])
                else:
                    skewness[index] = np.concatenate([terms(3, sigma, pk, weight * damping[:, None])[0]]
                                                     + [terms(d, sigma, pk, weight)[0] for d in (2, 0)])
        return kappa, skewness

    def _cumulants_ap(self, klin, pklin, f):
        """Observed-frame cumulants as an AP polynomial (``spectrum.APNodes``): kappa (ncoef, L + 1, nalpha * ncol) and
        skewness (ncoef, nalpha * nskew), the column layout of ``cumulants`` repeated per AP term alpha (alpha-major).

        In true-frame variables the Jacobians of the n-point functions cancel against the measure, so an observed-frame
        cumulant is the true-frame integral with every leg filtered (and projected on the harmonics) at its observed
        wavevector (q_perp k_x, q_perp k_y, q_par k_z): the physics (kernels, spectra, Fingers-of-God legs) is computed
        once, and only the filters and angular factors change with q. They are evaluated at the q = 1 point and the
        Chebyshev nodes, fitted per sample as a polynomial in x = ln q, and integrated against the physics in one pass.
        The shot noise of the assembly is the true-frame one, as for the AP projection of the block variances."""
        from .spectrum import APNodes

        logk, logp = np.log(klin), np.log(pklin)
        pk = lambda x: np.where(np.asarray(x) > 0, np.exp(np.interp(np.log(np.maximum(x, 1e-30)), logk, logp)), 0.0)  # noqa
        self.sigma_v, self.f = 0.0, f
        nodes = APNodes(self.ap, nodes=self.ap_nodes)
        nalpha = len(nodes.powers)
        stretches = [np.array([1.0, 1.0, 1.0])] + [np.exp([b, b, a]) for a, b in nodes.points]  # q=1, then the nodes
        nb = self.lmax + 1
        ncol = sum(KAPPA_SIZES) + (sum(CUBIC_KAPPA_SIZES) if self.cubic else 0) + fog_sizes(self.cubic)["kappa"]
        nskew = sum(SKEW_SIZES) + fog_sizes(self.cubic)["skewness"]
        kappa = np.zeros((len(self.coefficients), nb, nalpha * ncol))
        skewness = np.zeros((len(self.coefficients), nalpha * nskew))

        def fit(per_node):  # (1 + npoint, N, ...) -> (N, nalpha * prod(...)): alpha-major along the output axis
            terms = nodes.fit(per_node[0], per_node[1:])  # (nalpha, N, ...)
            terms = terms.reshape(nalpha, terms.shape[1], -1)
            return np.moveaxis(terms, 0, 1).reshape(terms.shape[1], -1)

        for ell in sorted({c.ell for c in self.coefficients}):
            members = [i for i, c in enumerate(self.coefficients) if c.ell == ell]
            legs = [np.asarray(x) for x in self.samplers[ell].legs(0.0)]  # true frame
            terms = _trispectrum_terms(legs, f)
            cubic = _cubic_trispectrum_terms(legs, f) if self.cubic else None
            observed = [[x * stretch for x in legs] for stretch in stretches]
            angles = []
            for obs in observed:
                (pair12, pair34), _ = _block_angles(ell, obs) if obs is not observed[0] else self.block_angles[ell]
                angles.append((pair12 * pair34).T)  # (N, l + 1)
            for index in members:
                sigma = self.config.sigma(self.coefficients[index].j)
                weights = np.array([(self.samplers[ell].weights / sigma**9
                                     * np.prod([self._filter(np.sqrt(np.sum(x**2, axis=-1)), ell, sigma) for x in obs],
                                               axis=0))[:, None] * angle for obs, angle in zip(observed, angles)])
                weight = fit(weights)  # (N, nalpha * (l + 1))
                fog = _fog_weights(weight, [x / sigma for x in legs])
                split3 = lambda values: [v.reshape(nalpha, ell + 1, -1) for v in np.split(values, 3)]  # noqa: E731
                tree = split3(terms(4, sigma, pk, fog))
                poisson = [terms(d, sigma, pk, weight).reshape(nalpha, ell + 1, -1) for d in (3, 2, 0)]
                cubics = [split3(cubic(name, sigma, pk, fog)) for name in CUBIC_NAMES] if cubic is not None else []
                blocks = []
                for a in range(nalpha):
                    blocks += ([tree[0][a]] + [x[a] for x in poisson] + [c[0][a] for c in cubics]
                               + [x for order in (1, 2) for x in [tree[order][a]] + [c[order][a] for c in cubics]])
                kappa[index, :ell + 1] = np.concatenate(blocks, axis=-1)
            del terms, cubic
            if ell % 2:
                continue
            legs = [np.asarray(x) for x in self.bis_samplers[ell].legs(0.0)]
            terms = _bispectrum_terms(legs, f)
            c = np.sqrt(4 * np.pi / (2 * ell + 1))
            observed = [[x * stretch for x in legs] for stretch in stretches]
            ys = [[c * np.real(spherical_harmonic(ell, 0, x)) for x in obs] for obs in observed]
            for index in members:
                sigma = self.config.sigma(self.coefficients[index].j)
                weights = np.array([(self.bis_samplers[ell].weights / sigma**6
                                     * np.prod([self._filter(np.sqrt(np.sum(x**2, axis=-1)), ell, sigma) * y
                                                for x, y in zip(obs, y_obs)], axis=0))[:, None]
                                    for obs, y_obs in zip(observed, ys)])
                weight = fit(weights)  # (N, nalpha)
                tree = np.split(terms(3, sigma, pk, _fog_weights(weight, [x / sigma for x in legs])), 3)
                poisson = [terms(d, sigma, pk, weight) for d in (2, 0)]
                skewness[index] = np.concatenate([x for a in range(nalpha)
                                                  for x in (tree[0][a], poisson[0][a], poisson[1][a], tree[1][a], tree[2][a])])
        return kappa, skewness


def s1m_from_variances(variances, linear, kappa_diag, skewness2, ells, q, edgeworth_amplitude=1.0, block_scale=None):
    """S1m of every block from its full variance ``variances`` and linear variance ``linear`` (flat over blocks,
    in the order coefficient-major, |m| = 0..l), the block cumulants ``kappa_diag`` (ncoef, L + 1) and the m = 0
    skewness ``skewness2`` (ncoef,); see the module docstring. ``edgeworth_amplitude`` scales the Edgeworth correction: a
    scalar, or one value per coefficient; ``block_scale`` (flat over blocks) scales it block by block."""
    amplitudes = jnp.broadcast_to(jnp.asarray(edgeworth_amplitude, dtype=float), (len(ells),))
    out, start = [], 0
    for index, ell in enumerate(ells):
        nb = ell + 1
        lam, lin = variances[start:start + nb], linear[start:start + nb]
        n = np.asarray(multiplicity(ell))
        gamma = np.exp(gammaln((n + q) / 2) - gammaln(n / 2))
        e = block_edgeworth(lin, kappa_diag[index, :nb], skewness2[index], ell, q)
        if block_scale is not None:
            e = e * block_scale[start:start + nb]
        out.append((2 * lam) ** (q / 2) * gamma * (1 + amplitudes[index] * e))
        start += nb
    return jnp.concatenate(out)


def block_edgeworth(linear, kappa_diag, skewness2, ell, q):
    """NLO Edgeworth factor E_M(q) of E|X_M|^q for the |m| blocks of one coefficient (module docstring), from the linear
    block variances, sum_{a,b in M} kappa_aabb and the m = 0 skewness^2 (used for even l)."""
    n = np.asarray(multiplicity(ell))
    e = q * (q - 2) * (kappa_diag / linear**2) / (8 * n * (n + 2))
    if ell % 2 == 0:
        e = jnp.asarray(e).at[0].add(q * (q - 2) * (q - 4) * skewness2 / linear[0] ** 3 / (72 * 15))
    return e


def cumulant_correction(config, coefficients, klin, pklin, f, npoints: int = 2**16, npoints_reference: int = 2**20,
                        damping="linear", cubic: bool = False, fog: bool = False):
    """Control-variate correction of the quasi-Monte Carlo block cumulants of an emulated basis.

    The basis cumulants use fixed ``npoints`` Sobol points, so their sampling error is a fixed, smooth function of
    cosmology (10% for l = 4, m = 0 at 2^16). Returns the column-wise ratios reference / base of kappa and skewness at
    one cosmology (``npoints_reference`` points, the same seeds; 1 where the base column is negligible), to multiply
    the emulated cumulants with.
    """
    base = S1mBasis(config, coefficients, damping=damping, npoints=npoints, cubic=cubic, fog=fog)
    reference = S1mBasis(config, coefficients, damping=damping, npoints=npoints_reference, cubic=cubic, fog=fog)
    out = []
    for b, r in zip(base.cumulants(klin, pklin, f), reference.cumulants(klin, pklin, f)):
        tiny = 1e-8 * np.max(np.abs(b), axis=tuple(range(b.ndim - 1)), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(np.where(np.abs(b) > tiny, r / b, 1.0))
    return tuple(out)
