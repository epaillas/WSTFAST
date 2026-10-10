"""Redshift-space WST basis of Eulerian-biased tracers (P_s grid, S1m cumulants, S21m terms).

Every piece is the matter one (rsd.py, rsd_moduli.py) with the tracer kernels of ``bias.py``, resolved into the
coefficients of the monomials of beta = (1, b1, b2, bG2, bGamma3) so that the basis stays cosmology-only:

- P_s grid: TREE (beta degree 2, (b1 + f mu^2)^2 P_L), LOOP (degree 2, one loop) and the counterterm shapes.
  The loop is renormalized as in CLASS-PT: the b2 part of P13 (a constant times Z1(k) P_L(k)) is absorbed into b1
  and the k -> 0 constant of the b2^2 part of P22 into the shot noise.
- S1m cumulants: per block, the tree trispectrum (degree 4) and the Poisson terms of a discrete tracer,
  T + B / n + P / n^2 + 1 / n^3 (degrees 3, 2, 0), and the m = 0 third cumulant B + P / n + 1 / n^2.
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

from .bias import (galaxy_kernel, monomial_exponents, monomial_position, monomial_product,
                   product_coefficients)
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
#: Columns of the m = 0 third cumulant: tree (deg 3), P / n (deg 2), 1 / n^2.
SKEW_SIZES = (len(monomial_exponents(3)), len(monomial_exponents(2)), 1)
#: Columns of the S21m terms: R0, R1, R2 (deg 4), G0 (deg 4), G1 (deg 2), G2, n0 (deg 2), n1.
S21_SIZES = (70, 70, 70, 70, 15, 1, 15, 1)


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


class BiasedRSDS1Basis(RSDS1Basis):
    """Monomial-resolved S1m cumulants of a biased tracer (``cumulants`` only; see the module docstring)."""

    def cumulants(self, klin, pklin, f):
        """``kappa`` (ncoef, L + 1, sum KAPPA_SIZES): per block sum_{a,b in M} kappa_aabb in the columns
        [tree | B/n | P/n^2 | 1/n^3]; ``skewness`` (ncoef, sum SKEW_SIZES): the m = 0 third cumulant (even l) in
        [tree | P/n | 1/n^2]."""
        logk, logp = np.log(klin), np.log(pklin)
        pk = lambda x: np.where(np.asarray(x) > 0, np.exp(np.interp(np.log(np.maximum(x, 1e-30)), logk, logp)), 0.0)  # noqa
        self.sigma_v = (0.0 if self.damping is None else float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
                        if self.damping == "linear" else float(self.damping))
        self.f = f
        nb = self.lmax + 1
        kappa = np.zeros((len(self.coefficients), nb, sum(KAPPA_SIZES)))
        skewness = np.zeros((len(self.coefficients), sum(SKEW_SIZES)))
        for ell in sorted({c.ell for c in self.coefficients}):  # kernels once per l, spectra per sigma
            members = [i for i, c in enumerate(self.coefficients) if c.ell == ell]
            legs = [np.asarray(x) for x in self.samplers[ell].legs(0.0)]
            terms = _trispectrum_terms(legs, f)
            (pair12, pair34), _ = self.block_angles[ell]
            angles = (pair12 * pair34).T  # (N, l + 1): diagonal block sums
            for index in members:
                sigma = self.config.sigma(self.coefficients[index].j)
                filters = np.prod([self._filter(np.sqrt(np.sum(x**2, axis=-1)), ell, sigma) for x in legs], axis=0)
                damping = np.prod([self._damp(x / sigma) for x in legs], axis=0)
                weight = (self.samplers[ell].weights * filters / sigma**9)[:, None] * angles
                # velocity-damped legs for the clustering piece only: the Poisson pieces are not
                kappa[index, :ell + 1] = np.concatenate([terms(4, sigma, pk, weight * damping[:, None])]
                                                        + [terms(d, sigma, pk, weight) for d in (3, 2, 0)], axis=-1)
            del terms
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
                skewness[index] = np.concatenate([terms(3, sigma, pk, weight * damping[:, None])[0]]
                                                 + [terms(d, sigma, pk, weight)[0] for d in (2, 0)])
        return kappa, skewness


class BiasedModulusSpectraRSD(ModulusSpectraRSD):
    """Monomial-resolved ``ModulusSpectraRSD``: ``__call__`` returns ``grids`` (nfield, nk, nmu, 296) with the
    columns [R0 | R1 | R2 | G0 | G1 | G2] and ``norms`` (nfield, 16) with [n0 | n1] (see the module docstring)."""

    def __call__(self, pk, f: float, sigma_v: float):
        damp = lambda k, mu: np.exp(-0.5 * (f * k * mu * sigma_v) ** 2)  # noqa: E731
        p, w = self.p, self.weights
        dp, pkp = damp(p, self.mup), pk(p)
        z1p = linear_components(self.pvec, f)
        psp = (pkp * dp**2)[:, None] * product_coefficients(z1p, z1p)  # (np, 15)
        radial = {s: np.exp(-0.5 * (s * p) ** 2) for s, _, _ in self.fields}
        norms, kernels = [], []
        for s, ell, m in self.fields:
            c2w = 4 * np.pi / (2 * ell + 1) * (1.0 if m == 0 else 2.0)
            rp = (s * p) ** ell * radial[s]
            a = w * rp**2 * c2w * np.abs(self.harmonics[ell, m]) ** 2
            norms.append(np.concatenate([a @ psp, [np.sum(a)]]))
            kernels.append((-1) ** ell * c2w * rp)
        nk, nmu = self.k.size, self.mu.size
        grids = np.zeros((len(self.fields), nk, nmu, sum(S21_SIZES[:6])))
        for i, kk in enumerate(self.k):
            pkk = pk(np.array([kk]))[0]
            for a, mk in enumerate(self.mu):
                kvec = np.array([kk * np.sqrt(1 - mk**2), 0.0, kk * mk])
                qvec = kvec - self.pvec
                qn = np.sqrt(np.sum(qvec**2, axis=-1))
                muq = qvec[:, 2] / np.maximum(qn, 1e-30)
                dq, pkq = damp(qn, muq), pk(qn)
                z1q = linear_components(qvec, f)
                minus = np.broadcast_to(-kvec, qvec.shape)
                long = 2 * ((pkq * dp * dq)[:, None] * product_coefficients(galaxy_kernel(np.stack([qvec, minus], axis=-2), f), z1q)
                            + (pkp * dp * dq)[:, None] * product_coefficients(galaxy_kernel(np.stack([minus, self.pvec], axis=-2), f), z1p))
                long = w[:, None] * long
                psq = (pkq * dq**2)[:, None] * product_coefficients(z1q, z1q)
                pair = w[:, None] * monomial_product(psp, 2, psq, 2)
                z1k = monomial_product(linear_components(kvec, f), 1, np.eye(5)[0], 1)  # beta_0 Z1(K), degree 2
                shot = w[:, None] * (psp + psq)
                cache = {}
                for index, (s, ell, m) in enumerate(self.fields):
                    if (ell, m) not in cache:
                        cache[ell, m] = np.real(self.harmonics[ell, m] * np.conj(_harmonic(ell, m, qvec)))
                    kernel = kernels[index] * (s * qn) ** ell * np.exp(-0.5 * (s * qn) ** 2) * cache[ell, m]
                    r = kernel @ long / 2
                    r1 = np.sum(w * kernel) / 2 * z1k
                    k2 = kernel**2 / 2
                    grids[index, i, a] = np.concatenate([monomial_product(r, 2, r, 2) * pkk,
                                                         2 * monomial_product(r, 2, r1, 2) * pkk,
                                                         monomial_product(r1, 2, r1, 2) * pkk,
                                                         k2 @ pair, k2 @ shot, [np.sum(w * k2)]])
        return grids, np.array(norms)


class BiasedRSDS21mBasis:
    """S21m terms of a biased tracer: (nentry, sum S21_SIZES), columns [R0 | R1 | R2 | G0 | G1 | G2 | n0 | n1]."""

    def __init__(self, config, coefficients, boxsize: float = 1000.0, nmesh: int = 256,
                 kmax: float = 0.12, nk: int = 24, nmu: int = 6, sigma_v_factor: float = SIGMA_V_FACTOR, **quadrature):
        from numpy.polynomial.legendre import leggauss

        self.config, self.coefficients = config, list(coefficients)
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

    def __call__(self, klin, pklin, f):
        logk, logp = np.log(klin), np.log(pklin)
        pk = lambda x: np.exp(np.interp(np.log(np.maximum(x, 1e-30)), logk, logp))  # noqa: E731
        sigma_v = self.sigma_v_factor * float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
        grids, norms = self.spectra(pk, f, sigma_v)
        nfield, nk, nmu, ncol = grids.shape
        flat = np.moveaxis(grids, -1, 1).reshape(nfield, ncol, nk * nmu)
        proj = self.projection
        lam = np.einsum("eg,ecg->ec", proj.matrix, flat[proj.field_index])
        return np.concatenate([lam, norms[proj.field_index]], axis=1)


def split(values, sizes):
    """Split the last axis of ``values`` into blocks of ``sizes``."""
    edges = np.cumsum((0,) + tuple(sizes))
    return [values[..., a:b] for a, b in zip(edges[:-1], edges[1:])]



def biased_scalings(ratio):
    """Powers of A_s / A_s,fid of every biased basis output (grid rows, kappa / skewness / S21m columns), so that a
    Taylor emulator is nearly exact in logA."""
    import jax.numpy as jnp

    xp = np if isinstance(ratio, (float, np.floating)) else jnp
    powers = lambda counts: xp.concatenate([xp.full((n,), 1.0) * ratio**p for p, n in counts])  # noqa: E731
    return dict(grid=powers([(1, len(TREE)), (2, len(LOOP)), (1, 3)]),
                kappa=powers(zip((3, 2, 1, 0), KAPPA_SIZES)),
                skewness=powers(zip((2, 1, 0), SKEW_SIZES)),
                s21m=powers(zip((3, 2, 1, 2, 1, 0, 1, 0), S21_SIZES)))


def biased_cumulant_correction(config, coefficients, klin, pklin, f, npoints: int = 2**16,
                               npoints_reference: int = 2**20, damping="linear"):
    """``rsd.cumulant_correction`` of the biased cumulants: column-wise reference / base ratios (1 where the base
    column is negligible)."""
    base = BiasedRSDS1Basis(config, coefficients, damping=damping, npoints=npoints, lattice=False)
    reference = BiasedRSDS1Basis(config, coefficients, damping=damping, npoints=npoints_reference, lattice=False)
    out = []
    for b, r in zip(base.cumulants(klin, pklin, f), reference.cumulants(klin, pklin, f)):
        tiny = 1e-8 * np.max(np.abs(b), axis=tuple(range(b.ndim - 1)), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(np.where(np.abs(b) > tiny, r / b, 1.0))
    return tuple(out)
