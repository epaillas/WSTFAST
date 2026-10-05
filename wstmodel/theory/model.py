"""Perturbative model of the WST coefficients of the real-space matter field.

Split in two, as in the density-split EFT:

* ``WSTMatterBasis.terms(k, P_L)``: everything that depends on cosmology, for the full set of
  coefficients. This is what the Taylor emulator replaces.
* ``assemble(...)``: closed-form combination with the nuisance parameters, exact and cheap.

First order (Gaussian limit, Sec. 2.1 of docs/wst_eft_feasibility.md):
    S1(j, l) = (2 s^2)^{q/2} Gamma((n + q) / 2) / Gamma(n / 2),  n = 2l + 1,
    n s^2 = int_k W^2(k) w_{j,l}(k) [P_L + P_1loop - 2 c_s^2 k^2 P_L + P_shot],
with the mass-assignment window W^2 ~ exp(-k^2 H^2 / 6) (CIC) and the particle shot noise.

Reduced second order (Sec. 2.3 and 3.3), S21 = S2(j1, j2, l) / S1(j1, l) for 1 <= l <= L2:
    S21 = [m_n (R + (1 + a_l) G) / n]^{q/2},  m_n = 2 [Gamma((n + 1) / 2) / Gamma(n / 2)]^2,
    R = int_k w_{j2,l} r_{j1,l}(k)^2 P_L(k),  G = int_k w_{j2,l} g_{j1,l}(k),
with the tree-level modulus response r and its Gaussian-chaos noise g (moduli.py), and a free
noise amplitude a_l per l. The second layer is taken Gaussian.

Not yet included: Edgeworth corrections (zero-lag tree trispectrum) to S1, one-loop responses,
redshift-space distortions.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

from ..config import Coefficient, WSTConfig
from .moduli import ModulusClustering
from .perturbation import OneLoopMatter, log_interpolator, log_weights

#: Columns of the S1 basis: linear, one-loop, counterterm (k^2 P_L) and shot-noise variances.
S1_TERMS = ("linear", "loop", "counterterm", "shotnoise")
#: Columns of the S21 basis: response (R) and Gaussian-noise (G) band powers.
S21_TERMS = ("response", "noise")


def all_coefficients(config: WSTConfig) -> list[Coefficient]:
    """Every S1 and every l >= 1 reduced S21 coefficient of a configuration."""
    out = [Coefficient("S1", ell, j) for ell in range(config.L + 1) for j in range(config.J + 1)]
    out += [Coefficient("S21", ell, j1, j2) for ell in range(1, config.lmax2 + 1)
            for j1, j2 in config.second_layer_pairs()]
    return out


def wavelet_weight(k, sigma, ell):
    """sum_m |psi_{sigma,l}^m(k)|^2 = (sigma k)^{2l} exp(-sigma^2 k^2)."""
    return (sigma * k) ** (2 * ell) * np.exp(-((sigma * k) ** 2))


def band(values, k, sigma, ell, window2=1.0):
    """int d^3k / (2 pi)^3 W^2(k) w_{sigma,l}(k) values(k) on a logarithmic grid k."""
    weights = wavelet_weight(k, sigma, ell) * window2 * k**2 * log_weights(k) / (2 * np.pi**2)
    return jnp.sum(values * weights)


class WSTMatterBasis:
    """Cosmology-dependent band integrals for every coefficient of a WST configuration."""

    def __init__(self, config: WSTConfig, shotnoise: float = 0.0, window: str | None = "cic", nk=160, nk_moduli=64):
        self.config = config
        self.coefficients = all_coefficients(config)
        self.s1 = [c for c in self.coefficients if c.kind == "S1"]
        self.s21 = [c for c in self.coefficients if c.kind == "S21"]
        self.klin = np.geomspace(1e-4, 10.0, 1024)
        self.kout = np.geomspace(1e-3, min(2.0, 6.0 / config.sigma(0)), nk)
        self.loop = OneLoopMatter(self.kout)
        if window == "cic":
            self.window2 = np.exp(-(self.kout * config.cellsize) ** 2 / 6.0)
        elif window is None:
            self.window2 = np.ones_like(self.kout)
        else:
            raise ValueError(f"unknown mass-assignment window {window!r}")
        self.shotnoise = float(shotnoise)
        # The second layer probes k <~ 6 / sigma_j2 for the smallest sigma_j2 paired with sigma_j1.
        self.moduli = {}
        for c in self.s21:
            if (c.j, c.ell) not in self.moduli:
                kgrid = np.geomspace(1e-3, 6.0 / config.sigma(c.j + config.min_dj), nk_moduli)
                self.moduli[c.j, c.ell] = ModulusClustering(kgrid, config.sigma(c.j), c.ell)

    def terms(self, pklin):
        """Return (s1_terms[n_s1, 4], s21_terms[n_s21, 2]) for the linear spectrum tabulated on ``klin``."""
        pk = log_interpolator(jnp.asarray(self.klin), pklin)
        k = self.kout
        plin = pk(jnp.asarray(k))
        spectra = (plin, self.loop(pk), k**2 * plin, jnp.full_like(plin, self.shotnoise))
        s1_terms = jnp.stack([jnp.stack([band(p, k, self.config.sigma(c.j), c.ell, self.window2) for p in spectra])
                              for c in self.s1])
        clustering = {key: modulus(pk) for key, modulus in self.moduli.items()}
        s21_terms = []
        for c in self.s21:
            modulus = self.moduli[c.j, c.ell]
            response, noise = clustering[c.j, c.ell]
            kgrid, sigma2 = modulus.kout, self.config.sigma(c.j2)
            s21_terms.append(jnp.stack([band(response**2 * pk(jnp.asarray(kgrid)), kgrid, sigma2, c.ell),
                                        band(noise, kgrid, sigma2, c.ell)]))
        return s1_terms, jnp.stack(s21_terms)


class Assembly:
    """Map basis terms and nuisance parameters to a selected data vector (all static indexing)."""

    def __init__(self, config: WSTConfig, coefficients):
        basis_coefficients = all_coefficients(config)
        s1_index = {c: i for i, c in enumerate(c for c in basis_coefficients if c.kind == "S1")}
        s21_index = {c: i for i, c in enumerate(c for c in basis_coefficients if c.kind == "S21")}
        self.q = config.q
        self.coefficients = list(coefficients)
        unknown = [c.label for c in self.coefficients if c not in s1_index and c not in s21_index]
        if unknown:
            raise ValueError(f"coefficients not modelled: {unknown}")
        self.s1_rows = np.array([s1_index[c] for c in self.coefficients if c.kind == "S1"], dtype=int)
        self.s21_rows = np.array([s21_index[c] for c in self.coefficients if c.kind == "S21"], dtype=int)
        s1_n = np.array([2 * c.ell + 1 for c in self.coefficients if c.kind == "S1"], dtype="f8")
        s21_n = np.array([2 * c.ell + 1 for c in self.coefficients if c.kind == "S21"], dtype="f8")
        self.s1_n, self.s21_n = s1_n, s21_n
        self.s1_gamma = np.exp(gammaln((s1_n + self.q) / 2) - gammaln(s1_n / 2))
        self.s21_mean2 = 2 * np.exp(2 * (gammaln((s21_n + 1) / 2) - gammaln(s21_n / 2)))
        self.noise_ells = sorted({c.ell for c in self.coefficients if c.kind == "S21"})
        self.s21_noise_index = np.array([self.noise_ells.index(c.ell) for c in self.coefficients if c.kind == "S21"],
                                        dtype=int)
        # Position of each output in the selected order.
        self.order = np.argsort([i for i, c in enumerate(self.coefficients) if c.kind == "S1"]
                                + [i for i, c in enumerate(self.coefficients) if c.kind == "S21"])

    def __call__(self, s1_terms, s21_terms, cs2, noise_amplitudes):
        q = self.q
        t1 = s1_terms[self.s1_rows]
        variance = (t1[:, 0] + t1[:, 1] - 2 * cs2 * t1[:, 2] + t1[:, 3]) / self.s1_n
        s1 = self.s1_gamma * (2 * variance) ** (q / 2)
        t21 = s21_terms[self.s21_rows]
        amplitude = 1.0 + jnp.asarray(noise_amplitudes)[self.s21_noise_index]
        s21 = (self.s21_mean2 * (t21[:, 0] + amplitude * t21[:, 1]) / self.s21_n) ** (q / 2)
        return jnp.concatenate([s1, s21])[self.order]
