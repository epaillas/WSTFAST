"""Redshift-space WST basis of Eulerian-biased tracers (P_s grid, S1m cumulants, S21m terms).

Every piece is the matter one (rsd.py, rsd_moduli.py) with the tracer kernels of ``bias.py``, resolved into the
coefficients of the monomials of beta = (1, b1, b2, bG2, bGamma3) so that the basis stays cosmology-only:

- P_s grid: TREE (beta degree 2, (b1 + f mu^2)^2 P_L), LOOP (degree 2, one loop) and the counterterm shapes.
  The loop is renormalized as in CLASS-PT: the b2 part of P13 (a constant times Z1(k) P_L(k)) is absorbed into b1
  and the k -> 0 constant of the b2^2 part of P22 into the shot noise.
- S1m cumulants: per block, the tree trispectrum (degree 4) and the Poisson terms of a discrete tracer,
  T + B / n + P / n^2 + 1 / n^3 (degrees 3, 2, 0), and the m = 0 third cumulant B + P / n + 1 / n^2. With ``cubic``,
  also the T_3111 of the cubic operators delta^3, G3 and delta G2 (``bias.CUBIC_NAMES``), which enter the tree
  trispectrum but not the one-loop P_s: one block per operator, linear in its bias times degree 3 in beta.
- S21m: per entry, the response R0 + N R1 + N^2 R2 (degree 4), the noise G0 + N G1 + N^2 G2 (degrees 4, 2, 0) and
  the first-layer normalization n0 + N n1 (degrees 2, 0):

      lambda_2 = [R0 + N R1 + N^2 R2 + (1 + a)(G0 + N G1 + N^2 G2)] / (n0 + N n1)^2,

  with N the tracer shot noise (1 + alpha0) / nbar. The response numerator r = r0 + N r1 holds the tree coupling of
  the matter model (r0) and the Poisson term of the discrete field, delta_d(p) delta_d(q) > delta_g(p + q) / nbar,
  which correlates |X|^2 with the long mode: r1 = Z1(K) int F_M(p, K - p) (lifted to degree 2 with beta_0).

With beta = (1, 1, 0, 0, 0) and no shot noise, all of these reduce to the matter basis.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y

from .bias import (CUBIC_NAMES, cubic_kernel, galaxy_kernel, monomial_exponents, monomial_position,
                   monomial_product, product_coefficients)
from .eft_loop import LinearSpectrum, Quadrature, radial_rule
from .rsd import RSDGrid, RSDS1Basis
from .rsd_moduli import SIGMA_V_FACTOR, ModulusSpectraRSD, S21mProjection, _harmonic, first_layer_fields

#: Degree-2 monomials of the tree spectrum: beta_0^2 (f^2 mu^4), beta_0 b1 (2 f mu^2), b1^2 (1).
TREE = tuple(monomial_position(e) for e in ([2, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 2, 0, 0, 0]))
#: Degree-2 monomials of the one-loop spectrum (bGamma3 only enters P13, with beta_0 or b1).
LOOP = tuple(i for i, e in enumerate(monomial_exponents(2)) if not (e[4] and (e[2] or e[3] or e[4] == 2)))
#: Rows of the biased P_s grid.
BIASED_GRID_TERMS = (tuple(f"tree_{i}" for i in TREE) + tuple(f"loop_{i}" for i in LOOP) + ("ct0", "ct2", "ct4"))
#: Columns of the per-block cumulants: tree (deg 4), B / n (deg 3), P / n^2 (deg 2), 1 / n^3.
KAPPA_SIZES = (len(monomial_exponents(4)), len(monomial_exponents(3)), len(monomial_exponents(2)), 1)
#: Columns of the cubic-operator trispectrum (``cubic``), appended to KAPPA_SIZES: one degree-3 block per cubic bias.
CUBIC_KAPPA_SIZES = (len(monomial_exponents(3)),) * len(CUBIC_NAMES)
#: Columns of the m = 0 third cumulant: tree (deg 3), P / n (deg 2), 1 / n^2.
SKEW_SIZES = (len(monomial_exponents(3)), len(monomial_exponents(2)), 1)
#: Columns of the S21m terms: R0, R1, R2 (deg 4), G0 (deg 4), G1 (deg 2), G2, n0 (deg 2), n1.
S21_SIZES = (70, 70, 70, 70, 15, 1, 15, 1)

#: k_NL of the Fingers-of-God counterterm [h/Mpc] (Tucci & Philcox 2026).
FOG_KNL = 0.3
#: Blocks of S21_SIZES that the Fingers-of-God counterterm acts on (R2, G2 and n1 have no short-mode spectra).
FOG_S21_BLOCKS = (0, 1, 3, 4, 6)


def fog_sizes(cubic: bool = False):
    """Columns appended by ``fog`` (orders 1 and 2 of the counterterm) to the kappa, skewness and S21m outputs."""
    kappa = KAPPA_SIZES[0] + (sum(CUBIC_KAPPA_SIZES) if cubic else 0)
    s21 = sum(S21_SIZES[i] for i in FOG_S21_BLOCKS)
    return dict(kappa=2 * kappa, skewness=2 * SKEW_SIZES[0], s21m=2 * s21)


def linear_components(vectors, f):
    """Z1 = b1 + f mu^2 along beta, for wavevectors (..., 3): (..., 5)."""
    vectors = np.asarray(vectors)
    norm2 = np.sum(vectors**2, axis=-1)
    mu2 = np.divide(vectors[..., 2] ** 2, norm2, out=np.zeros_like(norm2), where=norm2 > 0)
    out = np.zeros(vectors.shape[:-1] + (5,))
    out[..., 0], out[..., 1] = f * mu2, 1.
    return out


def renormalize(parts, spectrum: LinearSpectrum, quadrature: Quadrature):
    """Absorb the b2 part of P13 into b1 and the k -> 0 limit of the b2^2 part of P22 into the shot noise."""
    out = {name: np.array(value) for name, value in parts.items()}
    for e in ([1, 0, 1, 0, 0], [0, 1, 1, 0, 0]):
        out["13"][..., monomial_position(e)] = 0.
    p, w = radial_rule(quadrature.qmin, quadrature.qmax, quadrature.nq)
    constant = 0.5 * float(np.dot(w, p**3 * spectrum(p) ** 2)) / (2 * np.pi**2)  # b2^2 / 2 int P^2
    out["22"][..., monomial_position([0, 0, 2, 0, 0])] -= constant
    out["loop"] = out["22"] + out["13"]
    return out


class BiasedRSDGrid(RSDGrid):
    """``RSDGrid`` of a biased tracer: rows ``BIASED_GRID_TERMS`` (18, nk, nmu)."""

    _biased = True

    def _to_grid_components(self, coarse):
        return np.stack([self._to_grid(coarse[..., i]) for i in LOOP])

    def _loop(self, spectrum, f):
        return renormalize(self._loop_parts(spectrum, f), spectrum, self.quadrature)

    def __call__(self, klin, pklin, f, h=None, r_bao=None):
        spectrum = LinearSpectrum(np.asarray(klin, "f8"), np.asarray(pklin, "f8"))
        plin = spectrum(self.k)[:, None]
        mu2 = self.mu[None, :] ** 2 * np.ones_like(plin)
        k2 = self.k[:, None] ** 2
        shapes = np.stack([f**2 * mu2**2, 2 * f * mu2, np.ones_like(mu2)])  # TREE monomials
        parts = self._loop(spectrum, f)
        loop = self._to_grid_components(parts["loop"])
        if not self.ir:
            return np.concatenate([shapes * plin, loop, np.stack([k2 * plin * mu2**n for n in (0, 1, 2)])])
        from .ir import damping, split_linear

        nowiggle, sigma2, dsigma2, _ = split_linear(spectrum, h, r_bao / h)
        pnw = nowiggle(self.k)[:, None]
        parts_nw = self._loop(nowiggle, f)
        ratio = (spectrum(self.k_loop) / nowiggle(self.k_loop) - 1.0)[:, None, None]
        wiggle = self._to_grid_components(parts["22"] - parts_nw["22"] + parts_nw["13"] * ratio)
        d = np.asarray(damping(self.k, self.mu, f, sigma2, dsigma2))
        pw = plin - pnw
        damped = pnw + np.exp(-d) * pw
        return np.concatenate([shapes * (pnw + np.exp(-d) * (1 + d) * pw), loop + (np.exp(-d) - 1) * wiggle,
                               np.stack([k2 * damped * mu2**n for n in (0, 1, 2)])])


class _ScaleFreeTerms:
    """Sum_t C_t(n) prod_{v in V_t} P_L(|v(n)| / sigma) over sample points n: the kernel coefficients C_t (scale-free:
    Z kernels depend on directions only) are built once, the spectra per sigma. Terms with the same degree are
    stacked: ``coefficients[d]`` (nterm, N, nmono(d)) and ``norms[d]`` (nterm, nfactor, N)."""

    def __init__(self):
        self.terms = {}

    def add(self, degree, coefficients, vectors):
        norms = np.array([np.sqrt(np.sum(v**2, axis=-1)) for v in vectors]).reshape(len(vectors), len(coefficients))
        self.terms.setdefault(degree, []).append((coefficients, norms))

    def freeze(self):
        self.coefficients = {d: np.stack([c for c, _ in t]) for d, t in self.terms.items()}
        self.norms = {d: np.stack([n for _, n in t]) for d, t in self.terms.items()}
        del self.terms

    def __call__(self, degree, scale, pk, weight):
        """sum_n weight(n, ...) sum_t C_t(n) prod P_L: weight (N, nout) -> (nout, nmono(degree))."""
        norms = self.norms[degree]
        factors = np.prod(pk(norms / scale), axis=1)  # (nterm, N); 1 for a term without spectra
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


class BiasedRSDS1Basis(RSDS1Basis):
    """Monomial-resolved S1m cumulants of a biased tracer (``cumulants`` only; see the module docstring). With
    ``cubic``, the kappa columns also hold the cubic-operator trispectrum (CUBIC_KAPPA_SIZES).

    With ``fog``, the clustering pieces (tree and cubic trispectrum, tree bispectrum) are not velocity-damped but carry
    the Fingers-of-God counterterm of every external leg instead, stored as the undamped columns (order 0) followed by
    the orders 1 and 2 of ``_fog_weights`` (``fog_sizes``); ``collapse_fog`` combines them at a given c_v."""

    def __init__(self, *args, cubic: bool = False, fog: bool = False, ap: int = 0, ap_nodes: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.cubic, self.fog = bool(cubic), bool(fog)
        if self.fog:
            self.damping = None
        #: Order of the AP polynomial of the observed-frame cumulants (0: none; ``_cumulants_ap``).
        self.ap = int(ap)
        if self.ap and not self.fog:
            raise ValueError("ap needs fog=True")
        self.ap_nodes = int(ap_nodes)

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
            ys = [c * np.real(sph_harm_y(ell, 0, np.arccos(np.clip(x[:, 2] / np.sqrt(np.sum(x**2, axis=-1)), -1, 1)), 0.0))
                  for x in legs]
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
        """Observed-frame cumulants as an AP polynomial (``rsd.APNodes``): kappa (ncoef, L + 1, nalpha * ncol) and
        skewness (ncoef, nalpha * nskew), the column layout of ``cumulants`` repeated per AP term alpha (alpha-major).

        In true-frame variables the Jacobians of the n-point functions cancel against the measure, so an observed-frame
        cumulant is the true-frame integral with every leg filtered (and projected on the harmonics) at its observed
        wavevector (q_perp k_x, q_perp k_y, q_par k_z): the physics (kernels, spectra, Fingers-of-God legs) is computed
        once, and only the filters and angular factors change with q. They are evaluated at the q = 1 point and the
        Chebyshev nodes, fitted per sample as a polynomial in x = ln q, and integrated against the physics in one pass.
        The shot noise of the assembly is the true-frame one, as for the AP projection of the block variances."""
        from .rsd import APNodes, _block_angles

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
            ys = [[c * np.real(sph_harm_y(ell, 0, np.arccos(np.clip(x[:, 2] / np.sqrt(np.sum(x**2, axis=-1)), -1, 1)), 0.0))
                   for x in obs] for obs in observed]
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


class BiasedModulusSpectraRSD(ModulusSpectraRSD):
    """Monomial-resolved ``ModulusSpectraRSD``: ``__call__`` returns ``grids`` (nfield, nk, nmu, 296) with the
    columns [R0 | R1 | R2 | G0 | G1 | G2] and ``norms`` (nfield, 16) with [n0 | n1] (see the module docstring).

    With ``fog``, the short modes are not velocity-damped but carry the Fingers-of-God counterterm, one factor
    (1 - c_v k_z^2 / k_NL^2) per leg (two per P_L of a short mode, one per leg of the response), and the orders 1 and 2
    in c_v of the blocks R0, R1, G0, G1 (grids) and n0 (norms) are appended: grids
    [... | R0' | R1' | G0' | G1' | R0'' | R1'' | G0'' | G1''], norms [n0 | n1 | n0' | n0'']. R0 = r^2 is expanded to
    second order in c_v consistently: R0' = 2 r rho1, R0'' = rho1^2 + 2 r rho2 with r = r0 - c_v rho1 + c_v^2 rho2."""

    def __call__(self, pk, f: float, sigma_v: float, fog: bool = False, stretches=None):
        """``grids``, ``norms`` as in the class docstring. With ``stretches`` (a list of (q_perp, q_perp, q_par)), the
        first-layer filters and harmonics of every short leg are evaluated at its observed wavevector (the true one
        times the stretch) while the physics stays in the true frame (the Jacobians cancel in true-frame variables), and
        the outputs gain a leading axis over the stretches: grids (nstretch, nfield, nk, nmu, ncol), norms (nstretch,
        nfield, nnorm). Needs ``fog`` (no velocity damping)."""
        if fog:
            sigma_v = 0.0
        if stretches is not None and not fog:
            raise ValueError("stretches (AP in the first layer) need fog=True")
        single = stretches is None
        stretches = [np.ones(3)] if single else [np.asarray(x, "f8") for x in stretches]
        damp = lambda k, mu: np.exp(-0.5 * (f * k * mu * sigma_v) ** 2)  # noqa: E731
        p, w = self.p, self.weights
        dp, pkp = damp(p, self.mup), pk(p)
        sp = (p * self.mup) ** 2 / FOG_KNL**2  # k_z^2 / k_NL^2 of the leg p
        z1p = linear_components(self.pvec, f)
        psp = (pkp * dp**2)[:, None] * product_coefficients(z1p, z1p)  # (np, 15)
        norms, kernels, harmonics = [], [], []  # per stretch
        for stretch in stretches:
            pobs = self.pvec * stretch
            po = np.sqrt(np.sum(pobs**2, axis=-1))
            harm = self.harmonics if single or np.all(stretch == 1) else {
                key: _harmonic(key[0], key[1], pobs) for key in self.harmonics}
            radial = {s: np.exp(-0.5 * (s * po) ** 2) * self.window(po) for s, _, _ in self.fields}
            nrm, ker = [], []
            for s, ell, m in self.fields:
                c2w = 4 * np.pi / (2 * ell + 1) * (1.0 if m == 0 else 2.0)
                rp = (s * po) ** ell * radial[s]
                a = w * rp**2 * c2w * np.abs(harm[ell, m]) ** 2
                extra = [a @ (psp * (2 * sp)[:, None]), a @ (psp * (2 * sp**2)[:, None])] if fog else []
                nrm.append(np.concatenate([a @ psp, [np.sum(a)], *extra]))
                ker.append((-1) ** ell * c2w * rp)
            norms.append(nrm)
            kernels.append(ker)
            harmonics.append(harm)
        nk, nmu = self.k.size, self.mu.size
        ncol = sum(S21_SIZES[:6]) + (2 * sum(S21_SIZES[i] for i in (0, 1, 3, 4)) if fog else 0)
        grids = np.zeros((len(stretches), len(self.fields), nk, nmu, ncol))
        for i, kk in enumerate(self.k):
            pkk = pk(np.array([kk]))[0]
            for a, mk in enumerate(self.mu):
                kvec = np.array([kk * np.sqrt(1 - mk**2), 0.0, kk * mk])
                qvec = kvec - self.pvec
                qn = np.sqrt(np.sum(qvec**2, axis=-1))
                muq = qvec[:, 2] / np.maximum(qn, 1e-30)
                dq, pkq = damp(qn, muq), pk(qn)
                sq = qvec[:, 2] ** 2 / FOG_KNL**2
                z1q = linear_components(qvec, f)
                minus = np.broadcast_to(-kvec, qvec.shape)
                long = 2 * ((pkq * dp * dq)[:, None] * product_coefficients(galaxy_kernel(np.stack([qvec, minus], axis=-2), f), z1q)
                            + (pkp * dp * dq)[:, None] * product_coefficients(galaxy_kernel(np.stack([minus, self.pvec], axis=-2), f), z1p))
                long = w[:, None] * long
                psq = (pkq * dq**2)[:, None] * product_coefficients(z1q, z1q)
                pair = w[:, None] * monomial_product(psp, 2, psq, 2)
                z1k = monomial_product(linear_components(kvec, f), 1, np.eye(5)[0], 1)  # beta_0 Z1(K), degree 2
                shot = w[:, None] * (psp + psq)
                if fog:
                    sl, s2 = sp + sq, 2 * (sp + sq)  # legs of the response; of the pair of spectra
                    long1, long2 = long * sl[:, None], long * (0.5 * sl**2)[:, None]
                    pair1, pair2 = pair * s2[:, None], pair * (0.5 * s2**2)[:, None]
                    shot1 = w[:, None] * (psp * (2 * sp)[:, None] + psq * (2 * sq)[:, None])
                    shot2 = w[:, None] * (psp * (2 * sp**2)[:, None] + psq * (2 * sq**2)[:, None])
                for n, stretch in enumerate(stretches):
                    qobs = qvec * stretch
                    qo = np.sqrt(np.sum(qobs**2, axis=-1))
                    cache = {}
                    for index, (s, ell, m) in enumerate(self.fields):
                        if (ell, m) not in cache:
                            cache[ell, m] = np.real(harmonics[n][ell, m] * np.conj(_harmonic(ell, m, qobs)))
                        kernel = (kernels[n][index] * (s * qo) ** ell * np.exp(-0.5 * (s * qo) ** 2) * self.window(qo)
                                  * cache[ell, m])
                        r = kernel @ long / 2
                        r1 = np.sum(w * kernel) / 2 * z1k
                        k2 = kernel**2 / 2
                        columns = [monomial_product(r, 2, r, 2) * pkk, 2 * monomial_product(r, 2, r1, 2) * pkk,
                                   monomial_product(r1, 2, r1, 2) * pkk, k2 @ pair, k2 @ shot, [np.sum(w * k2)]]
                        if fog:
                            rho1, rho2 = kernel @ long1 / 2, kernel @ long2 / 2
                            columns += [2 * monomial_product(r, 2, rho1, 2) * pkk, 2 * monomial_product(rho1, 2, r1, 2) * pkk,
                                        k2 @ pair1, k2 @ shot1,
                                        (monomial_product(rho1, 2, rho1, 2) + 2 * monomial_product(r, 2, rho2, 2)) * pkk,
                                        2 * monomial_product(rho2, 2, r1, 2) * pkk, k2 @ pair2, k2 @ shot2]
                        grids[n, index, i, a] = np.concatenate(columns)
        if single:
            return grids[0], np.array(norms[0])
        return grids, np.array(norms)


class BiasedRSDS21mBasis:
    """S21m terms of a biased tracer: (nentry, sum S21_SIZES), columns [R0 | R1 | R2 | G0 | G1 | G2 | n0 | n1]. With
    ``fog`` (Fingers-of-God counterterm instead of the velocity damping, see ``BiasedModulusSpectraRSD``), followed by
    the orders 1 and 2 in c_v of the blocks FOG_S21_BLOCKS, [R0' | R1' | G0' | G1' | n0' | R0'' | ... | n0'']."""

    def __init__(self, config, coefficients, boxsize: float = 1000.0, nmesh: int = 256,
                 kmax: float = 0.12, nk: int = 24, nmu: int = 6, sigma_v_factor: float = SIGMA_V_FACTOR,
                 fog: bool = False, ap: int = 0, ap_nodes: int = 5, **quadrature):
        from numpy.polynomial.legendre import leggauss

        self.config, self.coefficients, self.fog = config, list(coefficients), bool(fog)
        #: Order of the AP polynomial (0: none): the first-layer filters of the modulus spectra in the observed frame
        #: (``BiasedModulusSpectraRSD(stretches=...)``) and the second-layer projection (``rsd.APPolynomial`` of an
        #: ``S21mAPProjection``), multiplied as polynomials; see ``__call__``.
        self.ap, self.ap_nodes = int(ap), int(ap_nodes)
        if self.ap and not self.fog:
            raise ValueError("ap needs fog=True")
        #: Velocity damping of the modulus spectra in units of the linear sigma_v. The matter value (2) is too strong
        #: for halos: 1 brings the measured response and noise of the halo moduli into agreement (and removes the
        #: S21m amplitude systematic; scripts/feasibility/modulus_response_noise.py --sigma-v-factor).
        self.sigma_v_factor = float(sigma_v_factor)
        self.fields = first_layer_fields(self.coefficients)
        x, _ = leggauss(2 * nmu)
        k, mu = np.geomspace(4e-3, kmax, nk), x[nmu:]
        self.spectra = BiasedModulusSpectraRSD([(config.sigma(j), ell, m) for j, ell, m in self.fields], k, mu,
                                               **quadrature)
        self.projection = S21mProjection(config, self.coefficients, self.fields, k, mu, boxsize=boxsize, nmesh=nmesh)
        if self.ap:
            from .rsd import APNodes, APPolynomial
            from .rsd_moduli import S21mAPProjection

            self.ap_poly = APPolynomial(S21mAPProjection(config, self.coefficients, self.fields, k, mu, boxsize=boxsize,
                                                         nmesh=nmesh), order=self.ap)
            self.nodes = APNodes(self.ap, nodes=self.ap_nodes)

    def __call__(self, klin, pklin, f):
        """(nentry, sum S21_SIZES [+ FoG orders]); with ``ap``, (nentry, nalpha * ncol), alpha-major: the AP polynomial
        of the S21m terms, sum_alpha x^alpha T_alpha, where the lambda_2 columns still need the volume factor
        1 / (q_par q_perp^2) of the second-layer sum (the norms do not; ``s21m_lambda_mask``). The lambda_2 terms are the
        product of the polynomials of the second-layer projection (D_beta) and of the first-layer grids (G_alpha),
        truncated at the order; the norms are the first-layer polynomial."""
        flat, norms = self.field_grids(klin, pklin, f)
        proj = self.projection
        if not self.ap:
            lam = np.einsum("eg,ecg->ec", proj.matrix, flat[proj.field_index])
            return s21m_assemble(lam, norms[proj.field_index], self.fog)
        nodes, index = self.nodes, proj.field_index
        projection = self.ap_poly.terms  # (nalpha, nentry, ngrid), without the volume factor
        lam = np.zeros((len(nodes.powers), len(index), flat.shape[2]))
        for i, (p1, r1) in enumerate(nodes.powers):
            for j, (p2, r2) in enumerate(nodes.powers):
                if p1 + p2 + r1 + r2 <= nodes.order:
                    lam[nodes.powers.index((p1 + p2, r1 + r2))] += np.einsum("eg,ecg->ec", projection[i], flat[j][index])
        return np.concatenate([s21m_assemble(lam[a], norms[a][index], self.fog) for a in range(len(nodes.powers))],
                              axis=1)

    def field_grids(self, klin, pklin, f):
        """The unprojected S21m pieces: modulus-spectrum grids (nfield, ncol, nk * nmu) of every first-layer field
        (``self.fields``) on the (K, mu_K) grid of ``self.spectra``, and their first-layer norms (nfield, nnorm). The
        projection to lambda_2 (fixed, or with AP) and ``s21m_assemble`` then give the S21m terms."""
        logk, logp = np.log(klin), np.log(pklin)
        pk = lambda x: np.exp(np.interp(np.log(np.maximum(x, 1e-30)), logk, logp))  # noqa: E731
        sigma_v = self.sigma_v_factor * float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
        if self.ap:  # the first-layer filters at q = 1 and the AP nodes, fitted as a polynomial in x = ln q
            stretches = [np.ones(3)] + [np.exp([b, b, a]) for a, b in self.nodes.points]
            grids, norms = self.spectra(pk, f, sigma_v, fog=self.fog, stretches=stretches)
            grids, norms = self.nodes.fit(grids[0], grids[1:]), self.nodes.fit(norms[0], norms[1:])
            nalpha, nfield, nk, nmu, ncol = grids.shape
            return np.moveaxis(grids, -1, 2).reshape(nalpha, nfield, ncol, nk * nmu), norms
        grids, norms = self.spectra(pk, f, sigma_v, fog=self.fog)
        nfield, nk, nmu, ncol = grids.shape
        return np.moveaxis(grids, -1, 1).reshape(nfield, ncol, nk * nmu), norms


def s21m_lambda_mask(fog: bool):
    """Boolean mask of the lambda_2 columns (the second-layer sums, which carry the AP volume factor) in the S21m column
    layout of ``s21m_assemble``; the others are the first-layer norms."""
    nlam = sum(S21_SIZES[:6]) + (2 * sum(S21_SIZES[i] for i in (0, 1, 3, 4)) if fog else 0)
    nnorm = sum(S21_SIZES[6:]) + (2 * S21_SIZES[6] if fog else 0)
    return s21m_assemble(np.ones((1, nlam)), np.zeros((1, nnorm)), fog)[0] > 0.5


def s21m_assemble(lam, norms, fog: bool):
    """S21m terms in the column layout of ``BiasedRSDS21mBasis`` from the projected grid columns ``lam`` (..., ncol) and
    the first-layer norms of each entry (..., nnorm): [lam | norms], or with ``fog`` the base blocks followed by the
    orders 1 and 2 of both. NumPy or JAX."""
    import jax.numpy as jnp

    xp = np if isinstance(lam, np.ndarray) and isinstance(norms, np.ndarray) else jnp
    if not fog:
        return xp.concatenate([lam, norms], axis=-1)
    nbase, norder = sum(S21_SIZES[:6]), sum(S21_SIZES[i] for i in (0, 1, 3, 4))
    n0 = S21_SIZES[6]
    base = [lam[..., :nbase], norms[..., :n0 + 1]]
    orders = [[lam[..., nbase + o * norder:nbase + (o + 1) * norder],
               norms[..., n0 + 1 + o * n0:n0 + 1 + (o + 1) * n0]] for o in range(2)]
    return xp.concatenate(base + orders[0] + orders[1], axis=-1)


def s21m_field_powers(fog: bool):
    """Powers of A_s / A_s,fid of the grid columns and of the norm columns of ``BiasedRSDS21mBasis.field_grids`` (those of
    ``biased_scalings``'s s21m columns, mapped back through ``s21m_assemble``)."""
    final = np.log(np.asarray(biased_scalings(np.e, fog=fog)["s21m"]))
    nlam, nnorm = (sum(S21_SIZES[:6]), sum(S21_SIZES[6:])) if not fog else (
        sum(S21_SIZES[:6]) + 2 * sum(S21_SIZES[i] for i in (0, 1, 3, 4)), S21_SIZES[6] + S21_SIZES[7] + 2 * S21_SIZES[6])
    source = s21m_assemble(np.arange(nlam, dtype=float), nlam + np.arange(nnorm, dtype=float), fog).astype(int)
    powers = np.empty(nlam + nnorm)
    powers[source] = final
    return powers[:nlam], powers[nlam:]


def split(values, sizes):
    """Split the last axis of ``values`` into blocks of ``sizes``."""
    edges = np.cumsum((0,) + tuple(sizes))
    return [values[..., a:b] for a, b in zip(edges[:-1], edges[1:])]



def biased_scalings(ratio, cubic: bool = False, fog: bool = False, ap_terms: int = 1):
    """Powers of A_s / A_s,fid of every biased basis output (grid rows, kappa / skewness / S21m columns), so that a
    Taylor emulator is nearly exact in logA. ``cubic``: the kappa columns include CUBIC_KAPPA_SIZES (P^3, like the
    tree trispectrum). ``fog``: with the counterterm orders of ``fog_sizes`` (the powers of the blocks they expand)."""
    import jax.numpy as jnp

    xp = np if isinstance(ratio, (float, np.floating)) else jnp
    powers = lambda counts: xp.concatenate([xp.full((n,), 1.0) * ratio**p for p, n in counts])  # noqa: E731
    clustering = [(3, KAPPA_SIZES[0])] + ([(3, sum(CUBIC_KAPPA_SIZES))] if cubic else [])
    s21 = list(zip((3, 2, 1, 2, 1, 0, 1, 0), S21_SIZES))
    s21_fog = [s21[i] for i in FOG_S21_BLOCKS]
    # with ``ap_terms`` AP-polynomial terms the columns repeat per term (alpha-major)
    return dict(grid=powers([(1, len(TREE)), (2, len(LOOP)), (1, 3)]),
                kappa=powers(ap_terms * (list(zip((3, 2, 1, 0), KAPPA_SIZES))
                                         + ([(3, sum(CUBIC_KAPPA_SIZES))] if cubic else []) + (2 * clustering if fog else []))),
                skewness=powers(ap_terms * (list(zip((2, 1, 0), SKEW_SIZES)) + (2 * [(2, SKEW_SIZES[0])] if fog else []))),
                s21m=powers(ap_terms * (s21 + (2 * s21_fog if fog else []))))


#: Number of terms of an AP polynomial of total degree 0, 1, 2, 3, 4 in (ln q_par, ln q_perp) (``rsd.APNodes``).
AP_TERMS = (1, 3, 6, 10, 15)


def basis_layout(kappa_columns: int):
    """(cubic, fog, nalpha) of a biased basis from the number of its kappa columns: with the cubic-operator blocks, with
    the Fingers-of-God orders, and the number of AP-polynomial terms (1 without AP)."""
    bases = {(c, f): sum(KAPPA_SIZES) + (sum(CUBIC_KAPPA_SIZES) if c else 0) + (fog_sizes(c)["kappa"] if f else 0)
             for c in (False, True) for f in (False, True)}
    for nalpha in AP_TERMS:
        for (cubic, fog), base in bases.items():
            if base * nalpha == kappa_columns:
                return cubic, fog, nalpha
    raise ValueError(f"{kappa_columns} kappa columns match no biased-basis layout")


def ap_powers(nalpha: int):
    """The (p, r) powers of x = (ln q_par, ln q_perp) of an AP polynomial with ``nalpha`` terms (``rsd.APNodes``)."""
    order = AP_TERMS.index(nalpha)
    return [(a, b) for a in range(order + 1) for b in range(order + 1 - a)]


def ap_evaluate(values, nalpha: int, qpar=1.0, qperp=1.0):
    """sum_alpha x^alpha values_alpha of alpha-major AP-polynomial columns (last axis nalpha * n) -> (..., n). NumPy or JAX."""
    import jax.numpy as jnp

    if nalpha == 1:
        return values
    x, y = jnp.log(qpar), jnp.log(qperp)
    weights = jnp.stack([x**p * y**r for p, r in ap_powers(nalpha)])
    return jnp.tensordot(values.reshape(*values.shape[:-1], nalpha, -1), weights, axes=([-2], [0]))


def is_fog(kappa=None, s21m=None) -> bool:
    """Whether basis outputs hold the Fingers-of-God counterterm orders (from their number of columns)."""
    if kappa is not None:
        return basis_layout(kappa.shape[-1])[1]
    n = s21m.shape[-1]
    base = sum(S21_SIZES) + fog_sizes()["s21m"]
    return n % base == 0 and n // base in AP_TERMS


def _collapse(values, base, blocks, c, order):
    """values[..., :base] with the orders 1 and 2 (the trailing columns, ``blocks`` the (start, size) of each block
    they expand, in order) combined as X0 - c X1 + c^2 X2 (order 2) or X0 - c X1 (order 1)."""
    import jax.numpy as jnp

    xp = np if isinstance(values, np.ndarray) and isinstance(c, (int, float, np.floating)) else jnp
    out = values[..., :base]
    n = sum(size for _, size in blocks)
    for o, factor in ((1, -c), (2, c**2)):
        if o > order:
            break
        column, pieces, last = base + (o - 1) * n, [], 0
        for start, size in blocks:
            pieces += [out[..., last:start], out[..., start:start + size] + factor * values[..., column:column + size]]
            last, column = start + size, column + size
        out = xp.concatenate(pieces + [out[..., last:]], axis=-1)
    return out


def collapse_fog(kappa=None, skewness=None, s21m=None, c_s1=0.0, c_s21=0.0, order: int = 2):
    """Standard-layout kappa, skewness and S21m columns from outputs with the Fingers-of-God counterterm orders (see
    ``fog_sizes``), at the counterterm c_s1 of the first-layer cumulants and c_s21 of the modulus spectra. Arguments
    left None are returned None."""
    out = []
    if kappa is not None:
        cubic = kappa.shape[-1] != sum(KAPPA_SIZES) + fog_sizes()["kappa"]
        base = sum(KAPPA_SIZES) + (sum(CUBIC_KAPPA_SIZES) if cubic else 0)
        blocks = [(0, KAPPA_SIZES[0])] + ([(sum(KAPPA_SIZES), sum(CUBIC_KAPPA_SIZES))] if cubic else [])
        out.append(_collapse(kappa, base, blocks, c_s1, order))
    else:
        out.append(None)
    out.append(None if skewness is None else _collapse(skewness, sum(SKEW_SIZES), [(0, SKEW_SIZES[0])], c_s1, order))
    if s21m is not None:
        edges = np.cumsum((0,) + S21_SIZES)
        out.append(_collapse(s21m, sum(S21_SIZES), [(edges[i], S21_SIZES[i]) for i in FOG_S21_BLOCKS], c_s21, order))
    else:
        out.append(None)
    return tuple(out)


def biased_cumulant_correction(config, coefficients, klin, pklin, f, npoints: int = 2**16,
                               npoints_reference: int = 2**20, damping="linear", cubic: bool = False,
                               fog: bool = False):
    """``rsd.cumulant_correction`` of the biased cumulants: column-wise reference / base ratios (1 where the base
    column is negligible)."""
    base = BiasedRSDS1Basis(config, coefficients, damping=damping, npoints=npoints, lattice=False, cubic=cubic, fog=fog)
    reference = BiasedRSDS1Basis(config, coefficients, damping=damping, npoints=npoints_reference, lattice=False,
                                 cubic=cubic, fog=fog)
    out = []
    for b, r in zip(base.cumulants(klin, pklin, f), reference.cumulants(klin, pklin, f)):
        tiny = 1e-8 * np.max(np.abs(b), axis=tuple(range(b.ndim - 1)), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(np.where(np.abs(b) > tiny, r / b, 1.0))
    return tuple(out)
