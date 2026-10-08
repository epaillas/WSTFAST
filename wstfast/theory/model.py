"""Perturbative model of the WST coefficients of the real-space matter field.

Split in two, as in the density-split EFT:

* ``WSTMatterBasis.terms(k, P_L)``: everything that depends on cosmology, for the full set of
  coefficients. This is what the Taylor emulator replaces.
* ``assemble(...)``: closed-form combination with the nuisance parameters, exact and cheap.

S1 at one-loop order (Secs. 2.1-2.2 of docs/wst_eft_feasibility.md):
    S1(j, l) = (2 s^2)^{q/2} Gamma((n + q) / 2) / Gamma(n / 2) [1 + E],  n = 2l + 1,
    n s^2 = int_k W^2(k) w_{j,l}(k) [P_L + P_1loop - 2 c_s^2 k^2 P_L + P_shot],
with the mass-assignment window W^2 ~ exp(-k^2 H^2 / 6) (CIC) and the particle shot noise. The
next-to-leading Edgeworth correction
    E = q (q - 2) / (8 n (n + 2)) K4 + q (q - 2) (q - 4) / (72 n (n + 2) (n + 4)) (6 K33 + 9 K3v)
uses the zero-lag tree trispectrum and squared tree bispectrum of the wavelet vector (cumulants.py),
normalised by the linear variance.

Reduced second order (Sec. 2.3 and 3.3), S21 = S2(j1, j2, l) / S1(j1, l) for 1 <= l <= L2:
    S21 = [m_n (R + (1 + a_l) G + N) / n]^{q/2},  m_n = 2 [Gamma((n + 1) / 2) / Gamma(n / 2)]^2,
    R = int_k w_{j2,l} r_{j1,l}(k)^2 P_L(k),  G = int_k w_{j2,l} g_{j1,l}(k),  N = int_k w_{j2,l} dN_{j1,l}(k),
with the tree-level modulus response r and its Gaussian-chaos noise g (moduli.py), the tree-level
non-Gaussian and fourth-chaos correction dN to that noise (modulus_noise.py), and a free noise
amplitude a_{j1,l} per first-layer field U_{j1,l} that absorbs what is beyond tree level. dN is
computed for first-layer scales sigma_j1 >= ``noise_min_scale`` only (zero below, where the model is
not used). The second layer is taken Gaussian.

Not yet included: one-loop responses, redshift-space distortions.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

from ..config import Coefficient, WSTConfig
from .cumulants import ZeroLagCumulants
from .moduli import ModulusClustering
from .modulus_noise import ModulusNoise
from .perturbation import OneLoopMatter, log_interpolator, log_weights

#: Columns of the S1 basis: linear, one-loop, counterterm (k^2 P_L) and shot-noise variances (n s^2), and the
#: Edgeworth numerators sum_ab kappa_aabb and 6 K33 s^6 + 9 K3v s^6 (cumulants.py).
S1_TERMS = ("linear", "loop", "counterterm", "shotnoise", "trispectrum", "bispectrum")
#: Columns of the S21 basis: response (R), Gaussian-noise (G) and tree-level noise-correction (N) band powers.
S21_TERMS = ("response", "noise", "noise_correction")


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

    def __init__(self, config: WSTConfig, shotnoise: float = 0.0, window: str | None = "cic", nk=160, nk_moduli=64,
                 noise_min_scale: float = 12.0):
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
        sigmas = [config.sigma(j) for j in range(config.J + 1)]
        self.cumulants = [ZeroLagCumulants(sigmas, ell, cellsize=config.cellsize if window == "cic" else 0.0)
                          for ell in range(config.L + 1)]
        # The second layer probes k <~ 6 / sigma_j2 for the smallest sigma_j2 paired with sigma_j1.
        self.moduli = {}
        for c in self.s21:
            if (c.j, c.ell) not in self.moduli:
                kgrid = np.geomspace(1e-3, 6.0 / config.sigma(c.j + config.min_dj), nk_moduli)
                self.moduli[c.j, c.ell] = ModulusClustering(kgrid, config.sigma(c.j), c.ell)
        self.noise = {key: ModulusNoise(config.sigma(key[0]), key[1]) for key in self.moduli
                      if config.sigma(key[0]) >= noise_min_scale - 1e-6}

    def terms(self, pklin):
        """Return (s1_terms[n_s1, 6], s21_terms[n_s21, 3]) for the linear spectrum tabulated on ``klin``."""
        pk = log_interpolator(jnp.asarray(self.klin), pklin)
        k = self.kout
        plin = pk(jnp.asarray(k))
        loop = self.loop(pk)
        spectra = (plin, loop, k**2 * plin, jnp.full_like(plin, self.shotnoise))
        variances = jnp.stack([jnp.stack([band(p, k, self.config.sigma(c.j), c.ell, self.window2) for p in spectra])
                               for c in self.s1])
        cumulants = [cumulant(pk) for cumulant in self.cumulants]  # (2, J + 1) per l
        edgeworth = jnp.stack([cumulants[c.ell][:, c.j] for c in self.s1])
        s1_terms = jnp.concatenate([variances, edgeworth], axis=1)
        clustering = {key: modulus(pk) for key, modulus in self.moduli.items()}
        dpk = lambda x: jnp.interp(jnp.log(x), jnp.log(jnp.asarray(k)), loop)  # noqa: E731  (one-loop correction)
        corrections = {key: noise(pk, dpk) for key, noise in self.noise.items()}
        s21_terms = []
        for c in self.s21:
            modulus = self.moduli[c.j, c.ell]
            r_long, r_short, noise = clustering[c.j, c.ell]
            response = r_long + r_short
            kgrid, sigma2 = modulus.kout, self.config.sigma(c.j2)
            if (c.j, c.ell) in corrections:
                correction = band(self.noise[c.j, c.ell].interpolate(corrections[c.j, c.ell], kgrid), kgrid, sigma2,
                                  c.ell)
            else:
                correction = 0.0 * band(noise, kgrid, sigma2, c.ell)
            s21_terms.append(jnp.stack([band(response**2 * pk(jnp.asarray(kgrid)), kgrid, sigma2, c.ell),
                                        band(noise, kgrid, sigma2, c.ell), correction]))
        return s1_terms, jnp.stack(s21_terms)


def edgeworth(q, n, terms):
    """NLO Edgeworth correction E(q) of E|X|^q from S1 basis rows (s^2 = linear / n):
    K4 = trispectrum n^2 / linear^2 and 6 K33 + 9 K3v = bispectrum n^3 / linear^3."""
    k4 = terms[:, 4] * n**2 / terms[:, 0] ** 2
    k6 = terms[:, 5] * n**3 / terms[:, 0] ** 3
    return q * (q - 2) * k4 / (8 * n * (n + 2)) + q * (q - 2) * (q - 4) * k6 / (72 * n * (n + 2) * (n + 4))


class Assembly:
    """Map basis terms and nuisance parameters to a selected data vector (all static indexing).

    With ``edgeworth``, S1 gets its NLO Edgeworth factor 1 + E(q), and S21 = S2 / S1(j1) the one-point
    factor (1 + E(1))^q / (1 + E(q)) of its first-layer field: the Gaussian relation <U>^2 = m_n s^2 behind
    the S21 formula becomes <U>^2 = m_n s^2 (1 + E(1))^2, and S1(j1) = (Gaussian) (1 + E(q)).

    The second layer Y = U * psi_{j2} is not Gaussian either. Its Edgeworth factor is modelled as
        1 + C_l q (q - 2) / (8 n (n + 2)) K4(j1, l)^2 f_N^2 (sigma_j1 / sigma_j2)^3,
    with the kurtosis K4 of the first-layer field, the noise fraction f_N = (G' + N) / (R + G' + N) of Y's
    variance (G' including the noise amplitude) and the volume dilution (sigma_j1 / sigma_j2)^3, and one free
    constant C_l per l (``layer2_amplitudes``). This scaling describes the measured second-layer kurtosis of
    every pair with sigma_j2 / sigma_j1 >= 2.8 and sigma_j1 >= 12.5 Mpc/h to ~20% with C_l ~ 6-21.
    """

    def __init__(self, config: WSTConfig, coefficients, edgeworth: bool = True):
        self.edgeworth = edgeworth
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
        self.s21_s1_rows = np.array([s1_index[Coefficient("S1", c.ell, c.j)] for c in self.coefficients
                                     if c.kind == "S21"], dtype=int)
        s1_n = np.array([2 * c.ell + 1 for c in self.coefficients if c.kind == "S1"], dtype="f8")
        s21_n = np.array([2 * c.ell + 1 for c in self.coefficients if c.kind == "S21"], dtype="f8")
        self.s1_n, self.s21_n = s1_n, s21_n
        self.s1_gamma = np.exp(gammaln((s1_n + self.q) / 2) - gammaln(s1_n / 2))
        self.s21_mean2 = 2 * np.exp(2 * (gammaln((s21_n + 1) / 2) - gammaln(s21_n / 2)))
        s21 = [c for c in self.coefficients if c.kind == "S21"]
        self.s21_dilution = np.array([(config.sigma(c.j) / config.sigma(c.j2)) ** 3 for c in s21])
        #: Multipoles l of the S21 coefficients, one second-layer kurtosis constant C_l each.
        self.layer2_keys = sorted({c.ell for c in s21})
        self.s21_layer2_index = np.array([self.layer2_keys.index(c.ell) for c in s21], dtype=int)
        #: First-layer fields (j1, l) of the S21 coefficients, one noise amplitude each.
        self.noise_keys = sorted({(c.j, c.ell) for c in self.coefficients if c.kind == "S21"})
        self.s21_noise_index = np.array([self.noise_keys.index((c.j, c.ell)) for c in self.coefficients if c.kind == "S21"],
                                        dtype=int)
        # Position of each output in the selected order.
        self.order = np.argsort([i for i, c in enumerate(self.coefficients) if c.kind == "S1"]
                                + [i for i, c in enumerate(self.coefficients) if c.kind == "S21"])

    def __call__(self, s1_terms, s21_terms, cs2, noise_amplitudes, layer2_amplitudes=None):
        q = self.q
        t1 = s1_terms[self.s1_rows]
        variance = (t1[:, 0] + t1[:, 1] - 2 * cs2 * t1[:, 2] + t1[:, 3]) / self.s1_n
        s1 = self.s1_gamma * (2 * variance) ** (q / 2)
        t21 = s21_terms[self.s21_rows]
        amplitude = 1.0 + jnp.asarray(noise_amplitudes)[self.s21_noise_index]
        noise = amplitude * t21[:, 1] + t21[:, 2]
        s21 = (self.s21_mean2 * (t21[:, 0] + noise) / self.s21_n) ** (q / 2)
        if layer2_amplitudes is not None:
            n, first = self.s21_n, s1_terms[self.s21_s1_rows]
            k4 = first[:, 4] * n**2 / first[:, 0] ** 2
            fraction = noise / (t21[:, 0] + noise)
            constant = jnp.asarray(layer2_amplitudes)[self.s21_layer2_index]
            s21 = s21 * (1 + constant * q * (q - 2) / (8 * n * (n + 2)) * k4**2 * fraction**2 * self.s21_dilution)
        if self.edgeworth:
            s1 = s1 * (1 + edgeworth(q, self.s1_n, t1))
            first = s1_terms[self.s21_s1_rows]
            s21 = s21 * (1 + q * edgeworth(1.0, self.s21_n, first) - edgeworth(q, self.s21_n, first))
        return jnp.concatenate([s1, s21])[self.order]
