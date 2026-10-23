"""WST configuration and the coefficient labels shared by measurement, data and theory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

# Quijote fiducial cosmology (massless neutrinos), in the parameterisation desilike varies.
QUIJOTE_COSMOLOGY = dict(h=0.6711, omega_b=0.02206838529, omega_cdm=0.120925743885,
                         n_s=0.9624, logA=3.061143632659324)
# AbacusSummit c000 (Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing mean), the cosmology of the DESI DR1 Abacus
# mocks: A_s = 2.0830e-9, one massive neutrino of 0.06 eV (omega_ncdm = 0.00064420) on top of omega_cdm.
ABACUS_COSMOLOGY = dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200, n_s=0.9649, logA=3.036394)
ABACUS_M_NCDM = 0.06


@dataclass(frozen=True)
class WSTConfig:
    """Solid-harmonic WST settings.

    Filters follow kymatio's Fourier-space construction,
    psi_{j,l}^m(k) = c_l (-i)^l (sigma_j k)^l Y_l^m(k) exp(-sigma_j^2 k^2 / 2), sigma_j = sigma0 step^j,
    but with c_l = sqrt(4 pi / (2l + 1)) so that sum_m |psi_{j,l}^m|^2 = (sigma_j k)^{2l} exp(-sigma_j^2 k^2).
    This only rescales each l by a constant with respect to kymatio, whose scales are dyadic (step = 2).
    Coefficients are lattice means: S0 = <|delta|^q>, S1(j, l) = <U_{j,l}^q> for l <= L, and
    S2(j1, j2, l) = <|U_{j1,l} * psi_{j2,l}|^q> for l <= L2 and j2 >= j1 + min_dj.
    """

    J: int = 4
    L: int = 4
    sigma0: float = 0.8  # in cells
    q: float = 0.8
    cellsize: float = 1000.0 / 256  # Mpc/h
    step: float = 2.0  # ratio of consecutive scales
    L2: int | None = None  # largest l of the second layer (default L)
    min_dj: int = 1  # smallest j2 - j1 of the second layer

    @property
    def lmax2(self) -> int:
        return self.L if self.L2 is None else self.L2

    def sigma(self, j: int) -> float:
        """Gaussian width of scale j in Mpc/h."""
        return self.sigma0 * self.step**j * self.cellsize

    def second_layer_pairs(self):
        """(j1, j2) pairs of the second layer."""
        return [(j1, j2) for j1 in range(self.J + 1) for j2 in range(j1 + self.min_dj, self.J + 1)]

    def to_dict(self) -> dict:
        return asdict(self)

    def with_q(self, q: float) -> "WSTConfig":
        return replace(self, q=float(q))

    def tag(self, nmesh: int) -> str:
        """Directory name of the measurement settings, e.g. 'J4_L4_sigma0.8_n256' (q is stored inside files)."""
        tag = f"J{self.J}_L{self.L}"
        if self.L2 is not None and self.L2 != self.L:
            tag += f"_L2-{self.L2}"
        if self.min_dj != 1:
            tag += f"_dj{self.min_dj}"
        tag += f"_sigma{self.sigma0:g}"
        if self.step != 2.0:
            tag += f"_step{self.step:.4g}"
        return tag + f"_n{nmesh}"


#: Half-octave superset: sigma_j = 3.1 * 2^(j/2) Mpc/h on a 256^3 mesh of a 1 Gpc/h box, j = 0..9,
#: so that every dyadic configuration with sigma0 = 0.8 cells is the subset of even j.
SUPERSET = WSTConfig(J=9, L=6, L2=4, min_dj=2, sigma0=0.8, step=2**0.5)


#: Highest order of the AP polynomial of the basis (``theory.layout.AP_TERMS``).
MAX_AP_ORDER = 4


@dataclass(frozen=True)
class ModelSpec:
    """What the model of the WST and P(k) includes. There is one implementation, of a biased, discrete tracer in redshift
    space; the other cases are its limits:

    - ``tracer``: ``"biased"`` (Eulerian biases b1, b2, bG2, bGamma3 and stochastic terms) or ``"matter"`` (b1 = 1, the
      others 0, Poisson shot noise: the theories fix the biases, on the same basis);
    - ``space``: ``"redshift"`` (line of sight z) or ``"real"`` (the limit f = 0, without Fingers-of-God or AP);
    - ``ir``: BAO infrared resummation of the P_s grid;
    - ``fog``: the Fingers-of-God counterterm c_v of the S1m cumulants and S21m modulus spectra, in place of the linear
      velocity damping;
    - ``ap_order``: order of the Alcock-Paczynski polynomial of the observed-frame cumulants and S21m terms (needs
      ``fog``; 0: none, the projections still carry AP);
    - ``cubic``: the cubic biases of the tree trispectrum (biased tracers);
    - ``s21_loop``: the one-loop short modes of the S21m modulus spectra (``theory.second_layer.LoopModulusSpectra``):
      the P_s grid of the P(k) model, with its counterterms and stochastic terms, in place of the tree spectrum with
      the Fingers-of-God legs (needs ``fog``, which then acts on the S1m cumulants only).

    Validated once here; an emulator stores it in its settings (``to_dict``, read back by ``from_settings``).
    """

    tracer: str = "biased"
    space: str = "redshift"
    ir: bool = False
    fog: bool = False
    ap_order: int = 0
    cubic: bool = False
    s21_loop: bool = False

    def __post_init__(self):
        if self.tracer not in ("biased", "matter"):
            raise ValueError(f"unknown tracer {self.tracer!r}")
        if self.space not in ("redshift", "real"):
            raise ValueError(f"unknown space {self.space!r}")
        if self.space == "real" and (self.fog or self.ap_order):
            raise ValueError("real space (f = 0) has no Fingers-of-God or AP terms")
        if self.cubic and self.tracer == "matter":
            raise ValueError("cubic biases need tracer='biased' (they vanish for matter)")
        if not 0 <= int(self.ap_order) <= MAX_AP_ORDER:
            raise ValueError(f"ap_order must be in 0..{MAX_AP_ORDER}")
        if self.ap_order and not self.fog:
            raise ValueError("ap_order needs fog=True")
        if self.s21_loop and not self.fog:
            raise ValueError("s21_loop needs fog=True (no velocity damping of the short modes)")

    @classmethod
    def matter(cls, space: str = "redshift", **kwargs) -> "ModelSpec":
        """The matter limit (``tracer="matter"``)."""
        return cls(tracer="matter", space=space, **kwargs)

    @classmethod
    def from_settings(cls, settings: dict) -> "ModelSpec":
        """The spec of an emulator from its settings: ``settings["model"]``, or the separate keys of emulators trained
        before ModelSpec existed (tracer, ir, fog, ap_order, cubic_biases)."""
        if "model" in settings:
            return cls(**settings["model"])
        return cls(tracer=settings.get("tracer", "matter"), ir=bool(settings.get("ir", False)),
                   fog=bool(settings.get("fog")), ap_order=int(settings.get("ap_order") or 0),
                   cubic=bool(settings.get("cubic_biases")))

    def to_dict(self) -> dict:
        return asdict(self)


def s1_label(j: int, ell: int) -> str:
    return f"S1_j{j}_l{ell}"


def s21_label(j1: int, j2: int, ell: int) -> str:
    return f"S21_j{j1}_j{j2}_l{ell}"


@dataclass(frozen=True)
class Coefficient:
    """One entry of the WST data vector.

    kind 'S1' is S1(j, l); kind 'S21' is the reduced second-order coefficient
    S2(j1, j2, l) / S1(j1, l), which cancels most of the non-perturbative one-point
    normalisation of the first-layer modulus.
    """

    kind: str
    ell: int
    j: int
    j2: int | None = None

    @property
    def label(self) -> str:
        return s1_label(self.j, self.ell) if self.kind == "S1" else s21_label(self.j, self.j2, self.ell)


def select_coefficients(config: WSTConfig, s1_min_scale: float = 25.0, s1_ells=(0, 1, 2, 3, 4),
                        s21_min_scale: float = 12.5, s21_ells=(1, 2, 3, 4), s21_min_ratio: float = 2.8,
                        s21_min_scale2: float = 0.0):
    """Coefficients inside the perturbative regime (scales in Mpc/h, see docs/wst_eft_feasibility.md).

    ``s21_min_ratio`` is the smallest sigma_j2 / sigma_j1 kept in the second layer. The default 2.8 drops
    adjacent pairs (sigma_j2 = 2 sigma_j1), whose second-layer non-Gaussianity is not perturbative at
    sigma_j1 ~ 12.5 Mpc/h: tree level overshoots the N-body value by ~2x. ``s21_min_scale2`` is the smallest
    sigma_j2 (in redshift space, 70 Mpc/h with sigma_j1 >= 17.7 Mpc/h; see ``wstfast.theory.second_layer``).
    """
    out = [Coefficient("S1", ell, j) for ell in s1_ells if ell <= config.L for j in range(config.J + 1)
           if config.sigma(j) >= s1_min_scale - 1e-6]
    out += [Coefficient("S21", ell, j1, j2) for ell in s21_ells if ell <= config.lmax2
            for j1, j2 in config.second_layer_pairs()
            if config.sigma(j1) >= s21_min_scale - 1e-6 and config.sigma(j2) / config.sigma(j1) >= s21_min_ratio - 1e-6
            and config.sigma(j2) >= s21_min_scale2 - 1e-6]
    return out
