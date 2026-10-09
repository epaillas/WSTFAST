"""First-order WST coefficients S1 of redshift-space matter (line of sight z), m-summed as measured.

Status. Line-of-sight-resolved S1m (``s1m_rsd``, one isotropic block of 1 or 2 components per |m|) with the
lattice-summed block variances and the velocity-damped tree cumulants (``damping="linear"``, sigma_v^2 =
int P_L / 6 pi^2) fits all 60 blocks with sigma >= 25 Mpc/h at the true cosmology with chi^2 = 29 for 57 dof
(diagonal errors of a 60 (Gpc/h)^3 survey) and counterterms (c0, c2, c4) = (0.7, 12.8, 2.9) (Mpc/h)^2; at
sigma = 17.7 it does not (chi^2 = 367 for 72). The m-summed S1 (``s1_rsd``) is not yet adequate: its data mix
block-variance and cumulant errors.

In redshift space the n = 2l + 1 components of X_a = (delta_s * psi_{sigma,l}^a)(x) are independent Gaussians
at leading order but no longer equally distributed: by symmetry about the line of sight, the variance of each
component depends on |m| only,

    lambda_|m| = int d^3k / (2 pi)^3 W^2(k) R(k)^2 c_l^2 |Y_l|m|(mu)|^2 P_s(k, mu),   R = (sigma k)^l e^{-sigma^2 k^2 / 2},

(m = 0 once, |m| > 0 twice), with the one-loop redshift-space matter spectrum and counterterms

    P_s(k, mu) = (1 + f mu^2)^2 P_L + P_1loop,s - 2 (c0 + c2 mu^2 + c4 mu^4) k^2 P_L + V / N.

S1 = E|X|^q is then the moment of an anisotropic Gaussian (``anisotropic_moment``) plus its NLO Edgeworth
correction for the block-diagonal covariance,

    (1/24) sum_abcd kappa_abcd E_G[d^4 |X|^q / dx_a dx_b dx_c dx_d]
        = (1/24) [ sum_a kappa_aaaa T_aaaa + 3 sum_{a != b} kappa_aabb T_aabb ],

where only index-paired terms survive and T_aaaa, T_aabb are 1D integrals over the block variances
(``edgeworth_tensors``). The tree cumulants are needed per pair of |m| blocks (``kappa_blocks``): the redshift-space
tree trispectrum (Z kernels) contracted with block pair kernels F_M(p, q) = c_l^2 R R w_M Re[Y_lM(p) Y_lM(q)^*].
The isotropic formula with the mean variance overshoots the measured correction by 10-15% at sigma = 25 Mpc/h.
The squared zero-lag bispectrum is kept for l = 0 only (it is ~1e-4 of S1 for l >= 1).

The block variances are sums over the Fourier modes of the measurement mesh (``boxsize``, ``nmesh``), not
continuum integrals: at the scales of S1 the low-k modes sample mu discretely, and redshift space (unlike real
space, where the m-sum is isotropic) is sensitive to that at the percent level. The same lattice treatment is
what makes the one-loop P_s fit the measured multipoles. The cumulants are continuum integrals.

The one-loop spectrum is the vendored dsc-model integral (eft_loop.py, unregulated as in the real-space WST
model, no IR resummation) and the tree kernels are its exact redshift-space Z_n. Both are NumPy: this basis is
evaluated per cosmology and emulated, like the real-space one.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, sph_harm_y

from ..config import WSTConfig
from .eft_loop import LinearSpectrum, Quadrature, loop_integrals, matter_kernel
from .modulus_noise import LegSampler

#: Columns of each |m| block: linear, one-loop, counterterms mu^0, mu^2, mu^4 (k^2 P_L) and shot noise.
BLOCK_TERMS = ("linear", "loop", "ct0", "ct2", "ct4", "shotnoise")

_T = np.exp(np.linspace(-25.0, 25.0, 1501))  # t * mean(lambda) grid of the moment integrals


def _laplace(eigenvalues, multiplicities):
    """Grid t (in units of 1 / scale), the scale (mean variance) and D(t) = prod_M (1 + 2 t lambda_M)^{-w_M / 2}."""
    lam = jnp.asarray(eigenvalues)
    w = jnp.asarray(multiplicities, dtype=lam.dtype)
    scale = jnp.sum(w * lam) / jnp.sum(w)
    t = jnp.asarray(_T)
    one_plus = 1 + 2 * t[:, None] * (lam / scale)[None, :]  # (nt, nblock)
    return t, scale, one_plus, jnp.exp(-0.5 * jnp.sum(w[None, :] * jnp.log(one_plus), axis=1))


def _integrate(values, t, a):
    """int values(t) t^{-1-a} dt on the logarithmic grid (trapezoid in ln t)."""
    integrand = values * t ** (-a)
    return jnp.sum(0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(jnp.log(t)))


def anisotropic_moment(eigenvalues, multiplicities, q):
    """E|X|^q, 0 < q < 2, for independent zero-mean Gaussian blocks of variance ``eigenvalues`` and sizes ``multiplicities``.

    Uses y^a = a / Gamma(1 - a) int_0^inf (1 - e^{-t y}) t^{-1-a} dt and E e^{-t |X|^2} = D(t).
    """
    a = q / 2
    t, scale, _, d = _laplace(eigenvalues, multiplicities)
    return a / np.exp(gammaln(1 - a)) * _integrate(1.0 - d, t, a) * scale**a


def edgeworth_tensors(eigenvalues, multiplicities, q):
    """E_G[|X|^q] and the per-block fourth derivatives T_aaaa (nblock,) and T_aabb (nblock, nblock), a != b.

    With He the Hermite polynomials of the block-diagonal Gaussian, E[He_aaaa e^{-t r^2}] = 12 t^2 D / (1 + 2 t l_a)^2
    and E[He_aabb e^{-t r^2}] = 4 t^2 D / ((1 + 2 t l_a)(1 + 2 t l_b)), both zero at t = 0, so that
    T = E_G[|X|^q He] = -a / Gamma(1 - a) int E[He e^{-t r^2}] t^{-1-a} dt.
    """
    a = q / 2
    const = a / np.exp(gammaln(1 - a))
    t, scale, one_plus, d = _laplace(eigenvalues, multiplicities)
    moment = const * _integrate(1.0 - d, t, a) * scale**a
    inv = 1.0 / one_plus  # (nt, nblock)
    base = t[:, None] ** 2 * d[:, None]
    t4 = jnp.stack([-const * _integrate(12 * base[:, 0] * inv[:, m] ** 2, t, a) for m in range(inv.shape[1])])
    t22 = jnp.stack([jnp.stack([-const * _integrate(4 * base[:, 0] * inv[:, m] * inv[:, n], t, a)
                                for n in range(inv.shape[1])]) for m in range(inv.shape[1])])
    return moment, t4 * scale ** (a - 2), t22 * scale ** (a - 2)


def angular_weights(ell, mu):
    """c_l^2 |Y_l|m|(mu)|^2 for |m| = 0..l, shape (l + 1, nmu); they sum (with multiplicity) to 1."""
    c2 = 4 * np.pi / (2 * ell + 1)
    theta = np.arccos(np.clip(mu, -1, 1))
    return np.array([c2 * np.abs(sph_harm_y(ell, m, theta, 0.0)) ** 2 for m in range(ell + 1)])


def multiplicity(ell):
    return np.array([1.0] + [2.0] * ell)


def tree_trispectrum_rsd(k, pk, f):
    """Tree-level redshift-space matter trispectrum (line of sight z) for wavevectors k[i] (N, 3), i = 0..3."""
    zhat = np.array([0.0, 0.0, 1.0])
    norms = [np.sqrt(np.sum(ki**2, axis=-1)) for ki in k]
    z1 = [1 + f * (ki @ zhat / n) ** 2 for ki, n in zip(k, norms)]
    power = [pk(n) * z for n, z in zip(norms, z1)]  # P(k) Z1(k)
    legs = range(4)
    total = 0.0
    for i in legs:  # 3111
        j, l, m = (x for x in legs if x != i)
        z3 = matter_kernel(np.stack([k[j], k[l], k[m]], axis=-2), f)
        total = total + 6.0 * z3 * power[j] * power[l] * power[m]
    for i in legs:  # 2211
        for j in range(i + 1, 4):
            c, d = (x for x in legs if x not in (i, j))
            for c, d in ((c, d), (d, c)):
                s = k[i] + k[c]
                z2a = matter_kernel(np.stack([-k[c], s], axis=-2), f)
                z2b = matter_kernel(np.stack([-k[d], -s], axis=-2), f)
                total = total + 4.0 * power[c] * power[d] * pk(np.sqrt(np.sum(s**2, axis=-1))) * z2a * z2b
    return total


def tree_bispectrum_rsd(k, pk, f):
    """Tree-level redshift-space matter bispectrum for wavevectors k[0] + k[1] + k[2] = 0."""
    zhat = np.array([0.0, 0.0, 1.0])
    norms = [np.sqrt(np.sum(ki**2, axis=-1)) for ki in k]
    power = [pk(n) * (1 + f * (ki @ zhat / n) ** 2) for ki, n in zip(k, norms)]
    total = 0.0
    for a, b in ((0, 1), (1, 2), (2, 0)):
        total = total + 2 * matter_kernel(np.stack([k[a], k[b]], axis=-2), f) * power[a] * power[b]
    return total


def _block_angles(ell, legs):
    """Angular factors at the four legs (sign conventions cancel in the products used):

    pairs (2, l+1, N): c_l^2 w_M Re[Y_lM(k_1) Y_lM(k_2)^*] and the same for legs 3, 4;
    components (l+1, N): c_l^4 sum_{a in block M} prod_i Y_a(k_i) over the real harmonics of block M.
    """
    c2 = 4 * np.pi / (2 * ell + 1)
    ys = []
    for leg in legs:
        norm = np.sqrt(np.sum(leg**2, axis=-1))
        theta = np.arccos(np.clip(leg[:, 2] / norm, -1, 1))
        phi = np.arctan2(leg[:, 1], leg[:, 0])
        ys.append(np.array([sph_harm_y(ell, m, theta, phi) for m in range(ell + 1)]))  # (l+1, N) complex
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


class RSDS1Basis:
    """Cosmology-dependent terms of S1(j, l) in redshift space for the given (j, l) coefficients.

    ``__call__(klin, pklin, f)`` returns
      ``blocks`` (ncoef, L + 1, 6): the |m| block variances split as BLOCK_TERMS (zero beyond |m| = l);
      ``kappa`` (ncoef, L + 1, L + 2): per block pair M, M' the tree sum_{a in M, b in M'} kappa_aabb (first L + 1
      columns) and per block sum_{a in M} kappa_aaaa (last column);
      ``skewness2`` (ncoef,): 15 kappa_3^2 of the m = 0 component for even l (zero otherwise). By azimuthal
      symmetry, blocks with |m| > 0 have no third cumulant, and neither has the m = 0 component of odd l
      (z -> -z parity).
    """

    def __init__(self, config: WSTConfig, coefficients, shotnoise: float = 0.0, window: str | None = "cic",
                 boxsize: float = 1000.0, nmesh: int = 256, kmax: float = 0.3, nk: int = 48, nmu: int = 12,
                 npoints: int = 2**16, quadrature: Quadrature = Quadrature(), damping: str | float | None = None):
        self.config, self.coefficients = config, list(coefficients)
        #: Velocity damping of the cumulant legs, exp(-(f k mu sigma_v)^2 / 2): None, "linear" (sigma_v^2 =
        #: int P_L dk / (6 pi^2)) or a fixed sigma_v [Mpc/h].
        self.damping = damping
        self.shotnoise, self.quadrature = float(shotnoise), quadrature
        sigmas = np.array([config.sigma(c.j) for c in self.coefficients])
        self.k = np.geomspace(2e-3, min(0.6, 7.0 / sigmas.min()), nk)
        x, w = leggauss(2 * nmu)  # P_s is even in mu: integrate over [0, 1] with doubled weights
        self.mu, self.wmu = x[nmu:], 2 * w[nmu:]
        self.window2 = (np.exp(-(self.k * config.cellsize) ** 2 / 6.0) if window == "cic" else np.ones_like(self.k))
        self.lmax = max(c.ell for c in self.coefficients)
        self.angles = {ell: angular_weights(ell, self.mu) for ell in range(self.lmax + 1)}
        self._lattice_modes(boxsize, nmesh, min(kmax, self.k[-1]), window)
        self.samplers, self.block_angles = {}, {}
        for ell in sorted({c.ell for c in self.coefficients}):
            sampler = LegSampler([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], [0, 0, 0, 0], (ell + 3) / 2, npoints,
                                 seed=17 + ell)
            self.samplers[ell] = sampler
            self.block_angles[ell] = _block_angles(ell, [np.asarray(x) for x in sampler.legs(0.0)])
        self.bis_samplers = {ell: LegSampler([[1, 0], [0, 1], [-1, -1]], [0, 0, 0], (ell + 3) / 2, npoints, seed=29 + ell)
                             for ell in sorted({c.ell for c in self.coefficients}) if ell % 2 == 0}

    def _lattice_modes(self, boxsize, nmesh, kmax, window):
        """Modes of the measurement mesh with |k| <= kmax: |k|, |mu| and the weight multiplicity W^2 / V per mode."""
        from ..measure import Lattice

        lattice = Lattice(nmesh)
        kmag = lattice.kmag.ravel() * (nmesh / boxsize)
        keep = (kmag > 0) & (kmag <= kmax)
        self.mode_k, self.mode_mu = kmag[keep].astype("f8"), np.abs(lattice.mu.ravel()[keep]).astype("f8")
        cell = boxsize / nmesh
        if window == "cic":
            k3 = [kc.ravel()[keep] for kc in np.meshgrid(*(2 * np.pi * np.fft.fftfreq(nmesh) / cell,) * 2,
                                                        2 * np.pi * np.fft.rfftfreq(nmesh) / cell, indexing="ij")]
            w2 = np.prod([np.sinc(kc * cell / (2 * np.pi)) ** 4 for kc in k3], axis=0)
        else:
            w2 = np.ones_like(self.mode_k)
        self.mode_weight = lattice.multiplicity.ravel()[keep] * w2 / boxsize**3
        theta = np.arccos(np.clip(self.mode_mu, -1, 1))
        self.mode_angles = {ell: np.array([4 * np.pi / (2 * ell + 1) * np.abs(sph_harm_y(ell, m, theta, 0.0)) ** 2
                                           for m in range(ell + 1)]) for ell in range(self.lmax + 1)}

    def _filter(self, u, ell, sigma):
        return np.exp(-((u / sigma) * self.config.cellsize) ** 2 / 12.0) * u**ell * np.exp(-0.5 * u**2)

    def __call__(self, klin, pklin, f):
        spectrum = LinearSpectrum(np.asarray(klin, "f8"), np.asarray(pklin, "f8"))
        logk, logp = np.log(klin), np.log(pklin)
        pk = lambda x: np.where(np.asarray(x) > 0, np.exp(np.interp(np.log(np.maximum(x, 1e-30)), logk, logp)), 0.0)  # noqa
        from scipy.interpolate import RegularGridInterpolator

        loop = loop_integrals(self.k, self.mu, spectrum, f, None, self.quadrature, rsd=True)["loop"]
        km, mum = self.mode_k, self.mode_mu
        points = np.column_stack([np.clip(km, self.k[0], self.k[-1]), np.clip(mum, self.mu[0], self.mu[-1])])
        loop_modes = RegularGridInterpolator((np.log(self.k), self.mu), loop, method="cubic")(
            np.column_stack([np.log(points[:, 0]), points[:, 1]]))
        plin = pk(km)
        k2 = km**2
        pieces = np.stack([(1 + f * mum**2) ** 2 * plin, loop_modes, k2 * plin, k2 * plin * mum**2,
                           k2 * plin * mum**4, np.full_like(plin, self.shotnoise)]) * self.mode_weight  # (6, nmodes)
        if self.damping is None:
            self.sigma_v = 0.0
        elif self.damping == "linear":
            self.sigma_v = float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
        else:
            self.sigma_v = float(self.damping)
        self.f = f
        ncoef, nb = len(self.coefficients), self.lmax + 1
        blocks = np.zeros((ncoef, nb, len(BLOCK_TERMS)))
        kappa = np.zeros((ncoef, nb, nb + 1))
        skewness2 = np.zeros(ncoef)
        trispectra = {}
        for index, c in enumerate(self.coefficients):
            sigma, ell = self.config.sigma(c.j), c.ell
            radial = (sigma * km) ** (2 * ell) * np.exp(-((sigma * km) ** 2))
            blocks[index, :ell + 1] = np.einsum("tn,n,an->at", pieces, radial, self.mode_angles[ell])
            kappa[index, :ell + 1, :ell + 1], kappa[index, :ell + 1, nb] = self._kappa(pk, f, sigma, ell, trispectra)
            if ell % 2 == 0:
                skewness2[index] = 15 * self._skewness(pk, f, sigma, ell) ** 2
        return blocks, kappa, skewness2

    def _kappa(self, pk, f, sigma, ell, cache):
        sampler = self.samplers[ell]
        legs = [np.asarray(x) for x in sampler.legs(0.0)]
        if (ell, sigma) not in cache:
            cache[ell, sigma] = tree_trispectrum_rsd([x / sigma for x in legs], pk, f)
        filters = np.prod([self._filter(np.sqrt(np.sum(x**2, axis=-1)), ell, sigma) * self._damp(x / sigma)
                           for x in legs], axis=0)
        weight = sampler.weights * cache[ell, sigma] * filters / sigma**9
        (pair12, pair34), comps = self.block_angles[ell]
        pairs = np.einsum("n,an,bn->ab", weight, pair12, pair34)
        return 0.5 * (pairs + pairs.T), comps @ weight

    def _skewness(self, pk, f, sigma, ell=0):
        """Third cumulant of the m = 0 component X_l0 (even l): c_l^3 int B_s prod_i R(k_i) Y_l0(k_i)."""
        sampler = self.bis_samplers[ell]
        legs = [np.asarray(x) for x in sampler.legs(0.0)]
        b = tree_bispectrum_rsd([x / sigma for x in legs], pk, f)
        c = np.sqrt(4 * np.pi / (2 * ell + 1))
        factors = []
        for x in legs:
            norm = np.sqrt(np.sum(x**2, axis=-1))
            y = np.real(sph_harm_y(ell, 0, np.arccos(np.clip(x[:, 2] / norm, -1, 1)), 0.0))
            factors.append(self._filter(norm, ell, sigma) * self._damp(x / sigma) * c * y)
        return float(np.sum(sampler.weights * b * np.prod(factors, axis=0))) / sigma**6

    def _damp(self, k):
        """exp(-(f k_z sigma_v)^2 / 2) for wavevectors k (N, 3)."""
        return np.exp(-0.5 * (self.f * k[:, 2] * self.sigma_v) ** 2)


def s1m_rsd(blocks, kappa, skewness2, ells, q, counterterms=(0.0, 0.0, 0.0), use_edgeworth=True,
            edgeworth_amplitude=1.0):
    """Line-of-sight-resolved S1m(j, l, |m|) for every coefficient, as a list of arrays of length l + 1.

    Each |m| block is an isotropic Gaussian of n_M = 1 (m = 0) or 2 components and variance lambda_M per component,
    so S1m = (2 lambda_M)^{q/2} Gamma((n_M + q)/2) / Gamma(n_M / 2) [1 + E_M], with the Edgeworth factor of an
    O(n_M)-invariant modulus: E_M = q (q-2) K4_M / (8 n_M (n_M+2)) + q (q-2)(q-4) 15 K33 / (72 * 15) for m = 0 and
    even l, K4_M = sum_{a,b in M} kappa_aabb / lambda_M^2 (linear lambda). ``edgeworth_amplitude`` scales E_M.
    """
    c0, c2, c4 = counterterms
    out = []
    for index, ell in enumerate(ells):
        nb = ell + 1
        n = jnp.asarray(multiplicity(ell))
        b = blocks[index, :nb]
        lam = b[:, 0] + b[:, 1] - 2 * (c0 * b[:, 2] + c2 * b[:, 3] + c4 * b[:, 4]) + b[:, 5]
        gamma = jnp.exp(gammaln((np.asarray(multiplicity(ell)) + q) / 2) - gammaln(np.asarray(multiplicity(ell)) / 2))
        s1 = (2 * lam) ** (q / 2) * gamma
        if use_edgeworth:
            lin = b[:, 0]
            k4 = jnp.diag(kappa[index, :nb, :nb]) / lin**2
            e = q * (q - 2) * k4 / (8 * n * (n + 2))
            e = e.at[0].add(q * (q - 2) * (q - 4) * skewness2[index] / lin[0] ** 3 / (72 * 15)) if ell % 2 == 0 else e
            s1 = s1 * (1 + edgeworth_amplitude * e)
        out.append(s1)
    return out


def s1_rsd(blocks, kappa, skewness2, ells, q, counterterms=(0.0, 0.0, 0.0), use_edgeworth=True):
    """S1 for every coefficient from ``RSDS1Basis`` outputs and the counterterms (c0, c2, c4) [(Mpc/h)^2]."""
    c0, c2, c4 = counterterms
    out = []
    for index, ell in enumerate(ells):
        nb = ell + 1
        w = multiplicity(ell)
        b = blocks[index, :nb]
        lam = b[:, 0] + b[:, 1] - 2 * (c0 * b[:, 2] + c2 * b[:, 3] + c4 * b[:, 4]) + b[:, 5]
        s1 = anisotropic_moment(lam, w, q)
        if use_edgeworth:
            moment, t4, t22 = edgeworth_tensors(b[:, 0], w, q)  # linear variances at NLO
            pairs, diagonal = kappa[index, :nb, :nb], kappa[index, :nb, -1]
            offdiagonal = pairs - jnp.diag(diagonal)  # sum over a in M, b in M', a != b
            correction = (jnp.sum(diagonal * t4) + 3 * jnp.sum(offdiagonal * t22)) / 24
            if ell == 0:
                s2 = b[0, 0]
                correction = correction + moment * q * (q - 2) * (q - 4) * skewness2[index] / s2**3 / (72 * 15)
            # (for even l > 0 the m = 0 skewness enters s1m_rsd; it is ~1e-4 of the m-summed S1)
            s1 = s1 * (1 + correction / moment)
        out.append(s1)
    return jnp.stack(out)
