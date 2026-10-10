"""Analytic covariance of the WST data vector: ln S1(j, l) and ln S21 = ln S2(j1, j2, l) - ln S1(j1, l), one box.

Every coefficient is <|Z|^q> for a vector Z of n = 2l + 1 wavelet components of a "last-layer" field: delta for S1,
the first-layer modulus U = U_{j1,l} for the second layer of S21. Its fluctuation is expanded in Wiener chaos of the
Gaussian vector Z ~ N(0, s^2 I_n). With z = Z / s:

* 2nd chaos: |z|^q -> M_q q / (2n) (|z|^2 - n), i.e. ln S fluctuates as (q / 2) times the relative fluctuation of the
  band power Sigma = n s^2 = (1/V) sum_k P(k) w(k), w = sum_m |psi^m|^2 = (sigma k)^{2l} exp(-sigma^2 k^2);
* 4th chaos: |z|^q -> M_q q (q - 2) / (8 n (n + 2)) :|z|^4:, with E[:|z_a|^4: :|z_b|^4:] = 8 (tr rho rho^T)^2
  + 16 tr (rho rho^T)^2 for the normalised cross-covariance rho(r) of the two vectors. It vanishes at q = 2. It is a
  few per cent of the variance, but without it the S1 block is near-singular: all S1 are linear functionals of the
  same isotropic band powers, through strongly overlapping kernels.

The 2nd-chaos terms are lattice sums over |k| shells (``ShellCovariance``), with three sources:

* ``delta``: band powers of delta, Cov(dP(k), dP(k')) = 2 P_f(k)^2 per lattice mode (k and -k), with the field power
  P_f (window and shot noise included). S1 enters through w_{j,l}; the second layer of S21 through r(K)^2 w_{j2,l}
  (the large-scale modes U shares with delta, P_Ud = <U> r P), and S21 through -w_{j1,l} (its S1(j1) denominator);
* ``noise_level``: the small-scale band powers that set the noise of U. At K -> 0 the pair kernel reduces to w, so
  the noise is proportional to sum P^2 w1^2 / Sigma_1 and changes by
  2 P w1^2 / sum P^2 w1^2 - w1 / (V Sigma_1) per unit dP; it enters ln S21 weighted by the noise fraction f_N of the
  second-layer variance (delta method);
* ``u_noise``: band powers of the noise part of U at the second-layer scales, treated as Gaussian with cross
  spectra P_{UaUb} = r_a r_b P_L + N_ab (the part 2 r_a r_b P_L N_ab + N_ab^2 not already in ``delta``).

Spectra of U are in units of <U>^2 throughout: r(K) = P_Ud / (<U> P) and N(K) = noise / <U>^2, as in moduli.py.
The 4th chaos is evaluated exactly on a periodic FFT lattice for every last layer (``chaos_covariances``).

Not included (and responsible for the remaining ~10% deficit with respect to N-body): the connected trispectrum of
delta, and the non-Gaussianity of the second layer beyond its 4th chaos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft
from numpy.polynomial.legendre import leggauss
from scipy.special import sph_harm_y

from .moduli import legendre
from .perturbation import log_weights


def wavelet_weight(k, sigma, ell):
    """sum_m |psi_{sigma,l}^m(k)|^2 = (sigma k)^{2l} exp(-sigma^2 k^2)."""
    return (sigma * k) ** (2 * ell) * np.exp(-((sigma * k) ** 2))


def lattice_shells(nmesh: int, boxsize: float, kmax: float):
    """Distinct |k| (k != 0) of the full periodic lattice with |k| <= kmax, and the number of modes on each."""
    kf = 2 * np.pi / boxsize
    f = np.fft.fftfreq(nmesh, 1.0 / nmesh).astype(int)
    f = f[np.abs(f) <= int(np.ceil(kmax / kf))]
    n2 = (f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2).ravel()
    n2 = n2[(n2 > 0) & (n2 <= (kmax / kf) ** 2 * (1 + 1e-12))]
    values, counts = np.unique(n2, return_counts=True)
    return kf * np.sqrt(values), counts.astype("f8")


def cross_noise(kout, pk, fields, npts=192, nmu=64):
    """Gaussian-chaos cross noise of first-layer moduli, g_ab(K) = int_p P(p) P(|K - p|) F_a F_b / (2 ns2_a ns2_b).

    ``fields`` is a list of (sigma, l), l >= 1; F_l is the pair kernel of moduli.py. Returns (nfields, nfields, nk);
    the diagonal is g(K) of ``ModulusClustering``.
    """
    kout = np.asarray(kout, dtype="f8")
    p = np.geomspace(1e-4, 8.0 / min(s for s, _ in fields), npts)
    wp = p**2 * log_weights(p) / (4 * np.pi**2)
    mu, wmu = leggauss(nmu)
    k, pp, mm = kout[:, None, None], p[None, :, None], mu[None, None, :]
    q = np.sqrt(np.maximum(k**2 + pp**2 - 2 * k * pp * mm, 1e-30))
    cos = np.clip((k * pp * mm - pp**2) / (pp * q), -1, 1)
    weights = wp[None, :, None] * wmu[None, None, :] * np.asarray(pk(pp)) * np.asarray(pk(q))
    kernels, norms = [], []
    for sigma, ell in fields:
        kernels.append((-1) ** ell * (sigma**2 * pp * q) ** ell * np.asarray(legendre(ell, cos))
                       * np.exp(-0.5 * sigma**2 * (pp**2 + q**2)))
        norms.append(2 * np.sum(np.asarray(pk(p)) * wavelet_weight(p, sigma, ell) * wp))
    out = np.zeros((len(fields), len(fields), kout.size))
    for a in range(len(fields)):
        for b in range(a, len(fields)):
            out[a, b] = out[b, a] = np.sum(weights * kernels[a] * kernels[b], axis=(1, 2)) / (2 * norms[a] * norms[b])
    return out


@dataclass
class ModulusSpectra:
    """Large-scale spectra of the first-layer moduli used by S21, tabulated on ``k`` (constant beyond the table).

    ``response[(j1, l)]`` = r(K), ``cross[(a, b)]`` = N_ab(K) for every pair of fields (noise in units of <U_a><U_b>,
    amplitudes and corrections included; the diagonal is the auto noise).
    """

    k: np.ndarray
    response: dict
    cross: dict

    def __call__(self, values, k):
        return np.interp(k, self.k, values, left=values[0], right=values[-1])

    def r(self, key, k):
        return self(self.response[key], k)

    def noise(self, a, b, k):
        return self(self.cross[a, b], k)


class ShellCovariance:
    """2nd-chaos covariance of ln S for S1 and S21 coefficients by lattice sums over |k| shells.

    ``pfield(k)``: power of the analysed field (window and shot noise included); ``plin(k)``: linear power (the
    response term); ``moduli``: ``ModulusSpectra`` for every first-layer field of the S21 coefficients.
    """

    TERMS = ("delta", "noise_level", "u_noise")

    def __init__(self, config, coefficients, q, boxsize, nmesh, pfield, plin, moduli: ModulusSpectra | None = None,
                 include=TERMS):
        self.coefficients = list(coefficients)
        unknown = set(include) - set(self.TERMS)
        if unknown:
            raise ValueError(f"unknown terms {sorted(unknown)}")
        V = boxsize**3
        k, count = lattice_shells(nmesh, boxsize, kmax=min(np.pi * nmesh / boxsize, 8.0 / config.sigma(0)))
        P, PL = pfield(k), plin(k)
        band = lambda w, p: (w * p * count).sum() / V  # noqa: E731  (1/V) sum_k w p
        D = np.zeros((len(self.coefficients), k.size))  # d ln S_A / dP(k), per lattice mode
        second = {}
        for i, c in enumerate(self.coefficients):
            w1 = wavelet_weight(k, config.sigma(c.j), c.ell)
            sigma1 = band(w1, P)
            if c.kind == "S1":
                D[i] = q / 2 * w1 / (V * sigma1)
                continue
            if moduli is None:
                raise ValueError("S21 coefficients need the modulus spectra")
            key = (c.j, c.ell)
            w2 = wavelet_weight(k, config.sigma(c.j2), c.ell)
            r, noise = moduli.r(key, k), moduli.noise(key, key, k)
            sigma_y = band(w2, r**2 * PL + noise)
            fraction = band(w2, noise) / sigma_y
            if "delta" in include:
                D[i] += q / 2 * r**2 * w2 / (V * sigma_y) - q / 2 * w1 / (V * sigma1)
            if "noise_level" in include:
                D[i] += q / 2 * fraction * (2 * P * w1**2 / (count * P**2 * w1**2).sum() - w1 / (V * sigma1))
            second[i] = (key, w2, r, sigma_y)
        self.delta_part = (D * (2 * P**2 * count)) @ D.T
        self.u_part = np.zeros_like(self.delta_part)
        if "u_noise" in include:
            for i, (ka, w2a, ra, sa) in second.items():
                for j, (kb, w2b, rb, sb) in second.items():
                    nab = moduli.noise(ka, kb, k)
                    bracket = 2 * ra * rb * PL * nab + nab**2
                    self.u_part[i, j] = q**2 / 2 / V**2 * (count * w2a * w2b * bracket).sum() / (sa * sb)
        self.lncov = self.delta_part + self.u_part


def real_harmonics(ell, theta, phi):
    """Real spherical harmonics of degree l (m = -l..l), scaled so that sum_m Y^2 = 1 (as ``measure.Lattice``)."""
    norm = np.sqrt(4 * np.pi / (2 * ell + 1))
    out = []
    for m in range(-ell, ell + 1):
        y = sph_harm_y(ell, abs(m), theta, phi)
        y = y.real if m == 0 else np.sqrt(2) * (-1) ** m * (y.real if m > 0 else y.imag)
        out.append(norm * y)
    return np.array(out)


def chaos_covariances(fields, q, boxsize, nmesh, spectra, workers=-1):
    """2nd- and 4th-chaos covariance of ln <|Z_a|^q> for wavelet vectors Z_a, exactly on a periodic FFT lattice.

    ``fields`` is a list of (sigma, l); ``spectra(a, b, k)`` the cross power of the fields filtered by a and b. The
    cross-covariance C_{mm'}(r) = (1/V) sum_k P_ab psi_a^m psi_b^m'* e^{ikr} of every component pair is one FFT, then
        Cov_2 = q^2 / (4 n_a n_b) < 2 tr rho rho^T >_r,
        Cov_4 = q^2 (q - 2)^2 / (64 n_a (n_a + 2) n_b (n_b + 2)) < 8 (tr rho rho^T)^2 + 16 tr (rho rho^T)^2 >_r.
    Returns (cov2, cov4).
    """
    kf = 2 * np.pi / boxsize
    f = np.fft.fftfreq(nmesh, 1.0 / nmesh) * kf
    kx, ky, kz = np.meshgrid(f, f, f, indexing="ij")
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    positive = k > 0
    kpos = np.where(positive, k, 1.0)
    theta = np.arccos(np.clip(np.where(positive, kz / kpos, 1.0), -1, 1))
    phi = np.arctan2(ky, kx)
    V, ncell = boxsize**3, nmesh**3
    filters = []
    for sigma, ell in fields:
        x = sigma * k
        filters.append((-1j) ** ell * x**ell * np.exp(-0.5 * x**2) * real_harmonics(ell, theta, phi))
    power = lambda a, b: np.where(positive, spectra(a, b, kpos), 0.0)  # noqa: E731

    def cross(a, b):
        pab = power(a, b)
        fa, fb = filters[a], filters[b]
        out = np.empty((len(fa), len(fb)) + (nmesh,) * 3)
        for i in range(len(fa)):
            for j in range(len(fb)):
                out[i, j] = sfft.ifftn(pab * fa[i] * np.conj(fb[j]), workers=workers).real * ncell / V
        return out

    # Variance per component, s^2 = (1/V) sum_k P |psi^m|^2 averaged over m.
    variance = [(power(a, a) * np.sum(np.abs(filters[a]) ** 2, axis=0)).sum() / V / len(filters[a])
                for a in range(len(fields))]
    nf = len(fields)
    cov2, cov4 = np.zeros((nf, nf)), np.zeros((nf, nf))
    for a in range(nf):
        na = 2 * fields[a][1] + 1
        for b in range(a, nf):
            nb = 2 * fields[b][1] + 1
            rho = cross(a, b) / np.sqrt(variance[a] * variance[b])
            t2 = np.einsum("ijxyz,ijxyz->xyz", rho, rho)
            m = np.einsum("ijxyz,kjxyz->ikxyz", rho, rho)
            t4 = np.einsum("ikxyz,ikxyz->xyz", m, m)
            cov2[a, b] = cov2[b, a] = q**2 / (4 * na * nb) * 2 * t2.mean()
            cov4[a, b] = cov4[b, a] = q**2 * (q - 2) ** 2 / (64 * na * (na + 2) * nb * (nb + 2)) \
                * (8 * (t2**2).mean() + 16 * t4.mean())
    return cov2, cov4


def last_layers(config, coefficients, pfield, plin, moduli: ModulusSpectra | None):
    """Fields (sigma, l) and cross spectra of the last layer of every coefficient, for ``chaos_covariances``.

    S1 is the wavelet transform of delta (power ``pfield``); the second layer of S21 that of U_{j1,l} (power
    r^2 P_L + N, cross r P_L with delta), in units of <U>.
    """
    sources = [None if c.kind == "S1" else (c.j, c.ell) for c in coefficients]
    fields = [(config.sigma(c.j if c.kind == "S1" else c.j2), c.ell) for c in coefficients]
    cache = {}

    def spectra(a, b, k):
        key = (sources[a], sources[b])
        if key not in cache:
            ua, ub = key
            if ua is None and ub is None:
                cache[key] = pfield(k)
            elif ua is None or ub is None:
                cache[key] = moduli.r(ua or ub, k) * plin(k)
            else:
                cache[key] = moduli.r(ua, k) * moduli.r(ub, k) * plin(k) + moduli.noise(ua, ub, k)
        return cache[key]

    return fields, spectra


def analytic_covariance(config, coefficients, q, boxsize, nmesh, pfield, plin, moduli=None, chaos_nmesh=128,
                        include=ShellCovariance.TERMS):
    """Covariance of ln(data vector) of one box: 2nd-chaos lattice sums plus the 4th chaos of every last layer.

    Returns a dict with ``lncov`` and its parts (``delta``, ``u_noise``, ``chaos4``). Multiply by
    outer(mean, mean) for the covariance of the data vector itself. ``chaos_nmesh = None`` skips the 4th chaos.
    """
    shell = ShellCovariance(config, coefficients, q, boxsize, nmesh, pfield, plin, moduli, include=include)
    out = dict(delta=shell.delta_part, u_noise=shell.u_part, lncov=shell.lncov.copy())
    if chaos_nmesh:
        fields, spectra = last_layers(config, coefficients, pfield, plin, moduli)
        _, cov4 = chaos_covariances(fields, q, boxsize, chaos_nmesh, spectra)
        out["chaos4"] = cov4
        out["lncov"] += cov4
    return out
