"""Solid-harmonic wavelet scattering transform of a periodic density mesh (numpy + scipy.fft)."""

from __future__ import annotations

import os

import numpy as np
import scipy.fft as sfft
from scipy.special import eval_legendre, sph_harm_y

from .config import WSTConfig

WORKERS = int(os.environ.get("WST_FFT_WORKERS", os.cpu_count()))


class Lattice:
    """Fourier lattice of an n^3 periodic mesh, wavenumbers in radians per cell."""

    def __init__(self, nmesh: int):
        self.n = nmesh
        k = (2 * np.pi * np.fft.fftfreq(nmesh)).astype(np.float32)
        kz = (2 * np.pi * np.fft.rfftfreq(nmesh)).astype(np.float32)
        kx, ky, kz = np.meshgrid(k, k, kz, indexing="ij")
        self.kmag = np.sqrt(kx**2 + ky**2 + kz**2)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.mu = np.where(self.kmag > 0, kz / self.kmag, 0.0).astype(np.float32)
        self.theta = np.arccos(np.clip(np.where(self.kmag > 0, self.mu, 1.0), -1, 1)).astype(np.float64)
        self.phi = np.arctan2(ky, kx).astype(np.float64)
        # Number of Fourier modes each rfft cell stands for (Hermitian symmetry).
        self.multiplicity = np.full(self.kmag.shape, 2.0, dtype=np.float32)
        self.multiplicity[..., 0] = 1.0
        if nmesh % 2 == 0:
            self.multiplicity[..., -1] = 1.0

    def harmonics(self, ell: int) -> list[np.ndarray]:
        """Real spherical harmonics scaled so that sum_m Y_lm^2 = 1."""
        norm = np.sqrt(4 * np.pi / (2 * ell + 1))
        out = []
        for m in range(-ell, ell + 1):
            ylm = sph_harm_y(ell, abs(m), self.theta, self.phi)
            if m == 0:
                y = ylm.real
            elif m > 0:
                y = np.sqrt(2) * (-1) ** m * ylm.real
            else:
                y = np.sqrt(2) * (-1) ** m * ylm.imag
            out.append((norm * y).astype(np.float32))
        return out

    def radial(self, sigma: float, ell: int) -> np.ndarray:
        """(sigma k)^l exp(-sigma^2 k^2 / 2), sigma in cells."""
        x = sigma * self.kmag
        return (x**ell * np.exp(-0.5 * x**2)).astype(np.float32)


class PowerMultipoles:
    """Binned power-spectrum multipoles (line of sight z) of rfft fields of a periodic box."""

    def __init__(self, lattice: Lattice, boxsize: float, kmax: float = 0.4, ells=(0, 2, 4)):
        kfund = 2 * np.pi / boxsize
        self.edges = np.arange(0.5, kmax / kfund + 1.0, 1.0) * kfund
        self.ells = tuple(ells)
        kphys = (lattice.kmag * (lattice.n / boxsize)).ravel()
        self.index = np.digitize(kphys, self.edges) - 1
        self.inside = (self.index >= 0) & (self.index < len(self.edges) - 1)
        weight = lattice.multiplicity.ravel()[self.inside]
        self.nbins = len(self.edges) - 1
        self.nmodes = np.bincount(self.index[self.inside], weights=weight, minlength=self.nbins)
        self.k = np.bincount(self.index[self.inside], weights=weight * kphys[self.inside], minlength=self.nbins) / self.nmodes
        mu = lattice.mu.ravel()[self.inside]
        self.weights = [weight * (2 * ell + 1) * eval_legendre(ell, mu) for ell in self.ells]
        self.norm = boxsize**3 / lattice.n**6

    def __call__(self, fa, fb):
        """P_ell(k) of <a b*>, shape (len(ells), nbins)."""
        cross = (fa * np.conj(fb)).real.ravel()[self.inside]
        return np.array([np.bincount(self.index[self.inside], weights=w * cross, minlength=self.nbins)
                         for w in self.weights]) / self.nmodes * self.norm


def _irfft(arr, n):
    return sfft.irfftn(arr, s=(n, n, n), workers=WORKERS)


def _rfft(arr):
    return sfft.rfftn(arr, workers=WORKERS)


def wavelet_modulus(fk, lattice: Lattice, harmonics, sigma: float, ell: int):
    """|field * psi_{sigma,l}| on the mesh, given the field's rfft (sigma in cells)."""
    phase = (-1j) ** ell
    radial = lattice.radial(sigma, ell)
    power = None
    for y in harmonics:
        x = _irfft(fk * (phase * radial * y), lattice.n)
        power = x * x if power is None else power + x * x
    return np.sqrt(power)


def _moments(field, qs):
    """<|field|^q> for every q (float64 accumulation)."""
    absolute = np.abs(field)
    return np.array([np.mean(absolute**q, dtype=np.float64) for q in qs])


def measure_wst(delta: np.ndarray, config: WSTConfig, qs=None, lattice: Lattice | None = None,
                spectra: PowerMultipoles | None = None) -> dict:
    """WST coefficients of a density contrast mesh, for one or several exponents q.

    Returns ``q`` (nq,), ``S0`` (nq,), ``S1`` (nq, J+1, L+1) and ``S2`` (nq, J+1, J+1, L2+1), NaN where not
    measured. The second layer reuses l, as in kymatio. With ``spectra``, also the power-spectrum
    multipoles ``Pdd`` (nell, nk), ``PUd`` and ``PUU`` (J+1, L+1, nell, nk) of the first-layer moduli U
    with delta, their means ``Umean`` (J+1, L+1), and the binning ``k``, ``k_edges`` and ``nmodes``.
    """
    qs = np.atleast_1d(config.q if qs is None else qs).astype(float)
    nmesh = delta.shape[0]
    lattice = lattice or Lattice(nmesh)
    fk = _rfft(delta.astype(np.float32))
    nq, nj = len(qs), config.J + 1
    out = {"q": qs, "S0": _moments(delta.astype(np.float64), qs),
           "S1": np.full((nq, nj, config.L + 1), np.nan),
           "S2": np.full((nq, nj, nj, config.lmax2 + 1), np.nan)}
    if spectra is not None:
        nell, nk = len(spectra.ells), spectra.nbins
        out.update(k=spectra.k, k_edges=spectra.edges, nmodes=spectra.nmodes, Pdd=spectra(fk, fk),
                   PUd=np.full((nj, config.L + 1, nell, nk), np.nan), PUU=np.full((nj, config.L + 1, nell, nk), np.nan),
                   Umean=np.full((nj, config.L + 1), np.nan))
    pairs = config.second_layer_pairs()
    for ell in range(config.L + 1):
        harmonics = lattice.harmonics(ell)
        for j1 in range(nj):
            u1 = wavelet_modulus(fk, lattice, harmonics, config.sigma0 * config.step**j1, ell)
            out["S1"][:, j1, ell] = _moments(u1, qs)
            second_layer = ell <= config.lmax2 and any(pair[0] == j1 for pair in pairs)
            if spectra is None and not second_layer:
                continue
            mean = u1.mean(dtype=np.float64)
            fu = _rfft(u1 - mean)
            if spectra is not None:
                out["Umean"][j1, ell] = mean
                out["PUd"][j1, ell] = spectra(fu, fk)
                out["PUU"][j1, ell] = spectra(fu, fu)
            if not second_layer:
                continue
            fu[0, 0, 0] = mean * nmesh**3  # restore the mean for the low-pass (l = 0) second layer
            for _, j2 in (pair for pair in pairs if pair[0] == j1):
                u2 = wavelet_modulus(fu, lattice, harmonics, config.sigma0 * config.step**j2, ell)
                out["S2"][:, j1, j2, ell] = _moments(u2, qs)
    return out
