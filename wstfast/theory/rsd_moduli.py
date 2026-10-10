"""Second layer of the line-of-sight-resolved WST of redshift-space matter (line of sight z).

The first-layer moduli U_|m| = |X_M| of each |m| block M of X_a = (delta_s * psi_{sigma1,l}^a)(x) (n_M = 1 component
for m = 0, 2 otherwise) are filtered again with psi_{sigma2,l} and resolved in |m2|:

    S21m(j1, j2, l, |m1|, |m2|) = S2m / S1m(j1, l, |m1|),   S2m = < |U_|m1| * psi_{sigma2,l}^{M2}|^q >.

Model. The block components are i.i.d. Gaussians of variance lambda_M, so the second-chaos projection
U ~ <U> + <U> / (2 n_M lambda_M) (|X_M|^2 - n_M lambda_M) holds block by block, with the pair kernel

    F_M(p, q) = (-1)^l R(p) R(q) c_l^2 w_M Re[Y_lM(p) Y_lM(q)^*],   R = (sigma1 k)^l e^{-sigma1^2 k^2 / 2},

and gives the power spectrum of U_|m| (normalised by <U>^2) as the tree-level response term plus the
Gaussian noise term, both anisotropic in K:

    P_UU(K) / <U>^2 = r_long(K)^2 P_L(K) + (1 + a) g(K),
    r_long(K) = int_p 2 [Z2(q, -K) Z1(q) P(q) + Z2(-K, p) Z1(p) P(p)] D(p) D(q) F_M(p, q) / (2 n_M lambda_M),
    g(K) = int_p P_s(p) P_s(q) F_M(p, q)^2 / (2 (n_M lambda_M)^2),   q = K - p,

with redshift-space Z kernels, damped Kaiser P_s = (1 + f mu^2)^2 P_L D^2 and D(k) = exp(-(f k mu sigma_v)^2 / 2).
The damping is effective: sigma_v = SIGMA_V_FACTOR x the linear sigma_v. With the linear sigma_v the noise of
the line-of-sight-oriented fields (small |m1|) has too much power along the line of sight; sigma_v ~ 2 sigma_v,lin
removes that anisotropy, which the one-loop EFT spectrum does not (it also stands for the missing non-Gaussian
noise terms). One free amplitude a per first-layer field absorbs what is left (~10%).

The second layer is taken Gaussian: Y = U * psi_{sigma2,l}^{M2} has n_2 = 1 or 2 components of variance

    lambda_2 = sum_K (mult / V) c_l^2 |Y_l|m2|(mu_K)|^2 (sigma2 K)^{2l} e^{-sigma2^2 K^2} P_UU(K) / <U>^2

(a sum over the modes of the measurement mesh), and with the one-point relation <U>^q / S1m of the first layer,

    S21m = Gamma((n2 + q) / 2) / Gamma(n2 / 2) (2 lambda_2)^{q/2}
           x [Gamma((n1 + 1) / 2) / Gamma(n1 / 2)]^q / [Gamma((n1 + q) / 2) / Gamma(n1 / 2)] x (1 + E_1(1))^q / (1 + E_1(q)),

where E_1(q) is the Edgeworth factor of S1m(j1, l, |m1|) (rsd.py).

Accuracy at the true cosmology (688 Quijote boxes, q = 0.8, l = 1-4, errors of a 60 (Gpc/h)^3 survey, full
covariance with the Hartlap factor). For sigma2 = 70.7 Mpc/h and sigma1 = 17.7, 25 Mpc/h (108 values) the
parameter-free model (a = 0) is within 3% of the data and with the 28 noise amplitudes chi^2 = 73 for 80 dof. The pair
sigma1 = 17.7, sigma2 = 50 Mpc/h is not described (chi^2 = 249 for its 54 values, a residual trend in |m2|), hence the
default sigma2 >= 70 Mpc/h of the selection.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, sph_harm_y

from ..config import WSTConfig
from .eft_loop import matter_kernel

#: Effective velocity dispersion of the modulus spectra, in units of the linear sigma_v.
SIGMA_V_FACTOR = 2.0
#: Columns of the S21m basis: response and Gaussian-noise parts of lambda_2.
S21M_TERMS = ("response", "noise")


def s21m_entries(coefficients):
    """(coefficient index, |m1|, |m2|) of every S21m entry of the S21 ``coefficients`` (coefficient-major)."""
    return [(index, m1, m2) for index, c in enumerate(coefficients)
            for m1 in range(c.ell + 1) for m2 in range(c.ell + 1)]


def first_layer_fields(coefficients):
    """Sorted first-layer fields (j1, l, |m1|) of the S21 ``coefficients``."""
    return sorted({(c.j, c.ell, m1) for c in coefficients for m1 in range(c.ell + 1)})


def _harmonic(ell, m, vec):
    norm = np.sqrt(np.sum(vec**2, axis=-1))
    theta = np.arccos(np.clip(vec[..., 2] / np.maximum(norm, 1e-30), -1, 1))
    return sph_harm_y(ell, m, theta, np.arctan2(vec[..., 1], vec[..., 0]))


class ModulusSpectraRSD:
    """Response r_long^2 P_L and noise g of every first-layer field (sigma1, l, |m1|) on a (K, mu_K >= 0) grid.

    The p integral uses a log-spaced radial grid, Gauss-Legendre in mu_p and the midpoint rule in phi_p over
    [0, pi] (the integrands are even in phi_p for K in the x-z plane). The geometry is shared by all fields.
    """

    def __init__(self, fields, k, mu, npts: int = 56, nx: int = 24, nphi: int = 16, pmin: float = 2e-3):
        self.fields = [(float(s), int(ell), int(m)) for s, ell, m in fields]
        self.k, self.mu = np.asarray(k, "f8"), np.asarray(mu, "f8")
        pmax = 7.0 / min(s for s, _, _ in self.fields)
        p = np.geomspace(pmin, pmax, npts)
        x, wx = leggauss(nx)
        phi = (np.arange(nphi) + 0.5) * np.pi / nphi
        P, X, PH = np.meshgrid(p, x, phi, indexing="ij")
        sin = np.sqrt(1 - X**2)
        self.p = P.ravel()
        self.pvec = np.stack([P * sin * np.cos(PH), P * sin * np.sin(PH), P * X], axis=-1).reshape(-1, 3)
        self.mup = X.ravel()
        self.weights = ((p**3 * np.gradient(np.log(p)))[:, None, None] * wx[None, :, None]
                        * (2 * np.pi / nphi) / (2 * np.pi) ** 3 * np.ones_like(P)).ravel()
        self.harmonics = {(ell, m): _harmonic(ell, m, self.pvec) for _, ell, m in self.fields}

    def __call__(self, pk, f: float, sigma_v: float):
        """Return (response, noise), each (nfield, nk, nmu); ``pk`` is a callable linear spectrum."""
        damp = lambda k, mu: np.exp(-0.5 * (f * k * mu * sigma_v) ** 2)  # noqa: E731
        z1 = lambda mu: 1 + f * mu**2  # noqa: E731
        p, w = self.p, self.weights
        dp, pkp = damp(p, self.mup), pk(p)
        psp = pkp * z1(self.mup) ** 2 * dp**2
        radial = {s: np.exp(-0.5 * (s * p) ** 2) for s, _, _ in self.fields}
        norms, kernels = [], []
        for s, ell, m in self.fields:
            c2w = 4 * np.pi / (2 * ell + 1) * (1.0 if m == 0 else 2.0)
            rp = (s * p) ** ell * radial[s]
            norms.append(np.sum(w * psp * rp**2 * c2w * np.abs(self.harmonics[ell, m]) ** 2))  # n_M lambda_M
            kernels.append((-1) ** ell * c2w * rp)
        response = np.zeros((len(self.fields), self.k.size, self.mu.size))
        noise = np.zeros_like(response)
        for i, kk in enumerate(self.k):
            pkk = pk(np.array([kk]))[0]
            for a, mk in enumerate(self.mu):
                kvec = np.array([kk * np.sqrt(1 - mk**2), 0.0, kk * mk])
                qvec = kvec - self.pvec
                qn = np.sqrt(np.sum(qvec**2, axis=-1))
                muq = qvec[:, 2] / np.maximum(qn, 1e-30)
                dq, pkq = damp(qn, muq), pk(qn)
                minus = np.broadcast_to(-kvec, qvec.shape)
                long = 2 * (matter_kernel(np.stack([qvec, minus], axis=-2), f) * z1(muq) * pkq
                            + matter_kernel(np.stack([minus, self.pvec], axis=-2), f) * z1(self.mup) * pkp) * dp * dq
                pair = w * psp * pkq * z1(muq) ** 2 * dq**2
                long = w * long
                cache = {}
                for index, (s, ell, m) in enumerate(self.fields):
                    if (ell, m) not in cache:
                        cache[ell, m] = np.real(self.harmonics[ell, m] * np.conj(_harmonic(ell, m, qvec)))
                    kernel = kernels[index] * (s * qn) ** ell * np.exp(-0.5 * (s * qn) ** 2) * cache[ell, m]
                    r = np.sum(long * kernel) / (2 * norms[index])
                    response[index, i, a] = r**2 * pkk
                    noise[index, i, a] = np.sum(pair * kernel**2) / (2 * norms[index] ** 2)
        return response, noise


class S21mProjection:
    """Lattice sums from the field grids of ``ModulusSpectraRSD`` to lambda_2 of every S21m entry.

    ``__call__(grids)`` maps (nfield, nk, nmu) grids to (nentry,) values; ``entries`` lists (coefficient index,
    |m1|, |m2|) and ``field_index`` the first-layer field of each entry.
    """

    def __init__(self, config: WSTConfig, coefficients, fields, k, mu, boxsize: float = 1000.0, nmesh: int = 256):
        from .rsd import angular_weights, grid_interpolation, lattice_modes

        modes = lattice_modes(boxsize, nmesh, float(k[-1]), window=None)
        interp = grid_interpolation(SimpleNamespace(k=np.asarray(k), mu=np.asarray(mu)), modes["k"], modes["mu"])
        self.entries = s21m_entries(coefficients)
        self.field_index = np.array([fields.index((coefficients[i].j, coefficients[i].ell, m1))
                                     for i, m1, _ in self.entries], dtype=int)
        rows = []
        for index, m1, m2 in self.entries:
            c = coefficients[index]
            sigma2 = config.sigma(c.j2)
            radial = (sigma2 * modes["k"]) ** (2 * c.ell) * np.exp(-((sigma2 * modes["k"]) ** 2))
            weight = modes["weight"] * radial * angular_weights(c.ell, modes["mu"])[m2]
            rows.append(interp.T @ weight)
        self.matrix = np.array(rows)  # (nentry, nk * nmu)

    def __call__(self, grids):
        flat = np.asarray(grids).reshape(len(grids), -1)
        return np.sum(self.matrix * flat[self.field_index], axis=1)


class RSDS21mBasis:
    """Cosmology-dependent S21m terms (``S21M_TERMS``) of the S21 ``coefficients``: (nentry, 2)."""

    def __init__(self, config: WSTConfig, coefficients, boxsize: float = 1000.0, nmesh: int = 256,
                 kmax: float = 0.12, nk: int = 24, nmu: int = 6, **quadrature):
        self.config, self.coefficients = config, list(coefficients)
        self.fields = first_layer_fields(self.coefficients)
        x, _ = leggauss(2 * nmu)
        k, mu = np.geomspace(4e-3, kmax, nk), x[nmu:]
        self.spectra = ModulusSpectraRSD([(config.sigma(j), ell, m) for j, ell, m in self.fields], k, mu, **quadrature)
        self.projection = S21mProjection(config, self.coefficients, self.fields, k, mu, boxsize=boxsize, nmesh=nmesh)

    def __call__(self, klin, pklin, f):
        logk, logp = np.log(klin), np.log(pklin)
        pk = lambda x: np.exp(np.interp(np.log(np.maximum(x, 1e-30)), logk, logp))  # noqa: E731
        sigma_v = SIGMA_V_FACTOR * float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
        response, noise = self.spectra(pk, f, sigma_v)
        return np.stack([self.projection(response), self.projection(noise)], axis=1)


def one_point_ratio(n, q):
    """Gaussian <U>^q / E[U^q] of an n-component modulus."""
    return np.exp(q * (gammaln((n + 1) / 2) - gammaln(n / 2)) - (gammaln((n + q) / 2) - gammaln(n / 2)))


def s21m_from_terms(terms, amplitudes, n1, n2, q, edgeworth_q=0.0, edgeworth_1=0.0):
    """S21m from the basis terms (nentry, 2), the noise amplitude of each entry's first-layer field, the block
    sizes n1, n2 of each entry and the first-layer Edgeworth factors E_1(q), E_1(1) of each entry."""
    lam2 = terms[:, 0] + (1 + amplitudes) * terms[:, 1]
    gamma2 = np.exp(gammaln((n2 + q) / 2) - gammaln(n2 / 2))
    return (gamma2 * one_point_ratio(n1, q) * (2 * lam2) ** (q / 2)
            * (1 + edgeworth_1) ** q / (1 + edgeworth_q))

