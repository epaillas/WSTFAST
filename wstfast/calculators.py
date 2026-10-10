"""desilike calculators: cosmology, WST and P(k) bases (emulated), theories and Gaussian likelihood.

The graphs are

    CosmoprimoCosmology -> WSTBasis -> WSTTheory -> WSTLikelihood
    CosmoprimoCosmology -> PowerBasis -> PowerTheory -> WSTLikelihood

The bases hold everything that depends on cosmology and are what the Taylor emulator
replaces; the theories add the nuisance parameters exactly.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from desilike import get_params
from desilike.base import Calculator, GaussianLikelihood
from desilike.parameter import Parameter
from desilike.theories import CosmoprimoCosmology

from .config import QUIJOTE_COSMOLOGY, WSTConfig
from .theory import Assembly, WSTMatterBasis
from .theory.power import LatticeBinning, matter_power_terms

#: Prior ranges used when cosmological parameters are varied.
COSMOLOGY_PRIORS = {"h": (0.5, 0.9), "omega_cdm": (0.08, 0.16), "logA": (2.5, 3.6),
                    "omega_b": (0.015, 0.03), "n_s": (0.85, 1.1)}


def build_cosmology(varied=("omega_cdm", "logA"), engine="class") -> CosmoprimoCosmology:
    """Quijote fiducial cosmology (massless neutrinos); only ``varied`` parameters are free."""
    params = CosmoprimoCosmology.propose_params(engine=engine, fiducial="DESI")
    for name, value in QUIJOTE_COSMOLOGY.items():
        params[name].update(value=value, fixed=name not in varied)
        if name in varied:
            params[name].update(prior=dict(limits=list(COSMOLOGY_PRIORS[name])),
                                ref=dict(dist="norm", loc=value, scale=params[name].ref.std()))
    params["m_ncdm"].update(value=0.0, fixed=True)
    params["tau_reio"].update(fixed=True)
    unknown = set(varied) - set(QUIJOTE_COSMOLOGY)
    if unknown:
        raise ValueError(f"cannot vary {sorted(unknown)}; choose among {sorted(QUIJOTE_COSMOLOGY)}")
    return CosmoprimoCosmology(engine=engine, fiducial="DESI", params=params)


class WSTBasis(Calculator):
    """Cosmology-dependent band integrals of the WST model, for every coefficient of ``config``.

    Outputs ``s1_terms`` (n_S1, 6) and ``s21_terms`` (n_S21, 2); see ``wstfast.theory.model``.
    """

    def __init__(self, cosmo=None, config: WSTConfig = WSTConfig(), z: float = 0.5, shotnoise: float = 0.0,
                 window: str | None = "cic"):
        self.cosmo = build_cosmology() if cosmo is None else cosmo

    def __post_init__(self, cosmo=None, config: WSTConfig = WSTConfig(), z: float = 0.5, shotnoise: float = 0.0,
                      window: str | None = "cic"):
        self.z = float(z)
        self.model = WSTMatterBasis(config, shotnoise=shotnoise, window=window)
        self.cosmo.add_requirements({"fourier.pk": [{"of": "delta_m", "z": self.z, "k": self.model.klin}]})

    def __call__(self):
        pk = self.cosmo.get("fourier.pk", of="delta_m", z=self.z, k=self.model.klin)
        self.s1_terms, self.s21_terms = self.model.terms(pk)
        return self

    def tree_flatten(self):
        return [self.s1_terms, self.s21_terms], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.s1_terms, obj.s21_terms = children
        return obj


class WSTTheory(Calculator):
    """WST data-vector prediction for selected coefficients.

    Nuisance parameters: ``cs2`` (counterterm, (Mpc/h)^2) and one second-layer noise amplitude
    ``noise_j{j1}_l{l}`` per first-layer field used in S21 (the noise of U_{j1,l} is (1 + noise) times
    its Gaussian-chaos value). ``edgeworth=False`` drops the NLO Edgeworth correction to S1.
    With ``layer2=True``, one constant ``layer2_l{l}`` per l scales the second-layer Edgeworth factor (see
    ``wstfast.theory.model.Assembly``).
    """

    def __init__(self, coefficients, config: WSTConfig = WSTConfig(), basis=None, edgeworth: bool = True,
                 layer2: bool = False, **basis_kwargs):
        self.basis = WSTBasis(config=config, **basis_kwargs) if basis is None else basis
        self.cs2 = Parameter("cs2", value=0.0, prior=dict(limits=[-20.0, 20.0]),
                             ref=dict(dist="norm", loc=0.0, scale=1.0), latex="c_s^2")
        fields = sorted({(c.j, c.ell) for c in coefficients if c.kind == "S21"})
        self.noise = {key: Parameter(f"noise_j{key[0]}_l{key[1]}", value=0.0, prior=dict(limits=[-1.0, 5.0]),
                                     ref=dict(dist="norm", loc=0.0, scale=0.1), latex=rf"a_{{N,{key[0]},{key[1]}}}")
                      for key in fields}
        ells = sorted({c.ell for c in coefficients if c.kind == "S21"}) if layer2 else []
        self.layer2 = {ell: Parameter(f"layer2_l{ell}", value=10.0, prior=dict(limits=[0.0, 100.0]),
                                      ref=dict(dist="norm", loc=10.0, scale=2.0), latex=rf"C_{{2,{ell}}}")
                       for ell in ells}

    def __post_init__(self, coefficients, config: WSTConfig = WSTConfig(), basis=None, edgeworth: bool = True,
                      layer2: bool = False, **basis_kwargs):
        self.assembly = Assembly(config, coefficients, edgeworth=edgeworth)

    def __call__(self):
        noise = jnp.stack([self.noise[key].value for key in self.assembly.noise_keys]) if self.noise else jnp.zeros(0)
        layer2 = (jnp.stack([self.layer2[ell].value for ell in self.assembly.layer2_keys]) if self.layer2 else None)
        self.flattheory = self.assembly(self.basis.s1_terms, self.basis.s21_terms, self.cs2.value, noise, layer2)
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


#: Primordial amplitude the P(k) basis is normalised to (Quijote).
POWER_AMPLITUDE = float(np.exp(QUIJOTE_COSMOLOGY["logA"]) * 1e-10)


class PowerBasis(Calculator):
    """Cosmology-dependent rows of dsc-model's real-space matter P(k) on ``knodes``.

    Outputs ``pk_terms`` (2, nnodes) = [P_L, L_Lambda] rescaled to the amplitude ``POWER_AMPLITUDE``
    (P_L ∝ A_s, L ∝ A_s^2), and ``dlogA`` = ln(A_s / POWER_AMPLITUDE), so that a Taylor emulator is exact
    in logA, as in dsc-model; ``power_rows`` undoes the scaling. The loop row is zero at tree level.
    ``cutoff`` is the loop regulator in h/Mpc (None: unregulated SPT); see ``wstfast.theory.power``.
    The loop is NumPy (dsc-model's reference quadrature), so desilike runs this node through a pure_callback.
    """

    _is_external = True

    def __init__(self, cosmo=None, knodes=None, z: float = 0.5, order: str = "one-loop", cutoff: float | None = 0.5):
        self.cosmo = build_cosmology() if cosmo is None else cosmo

    def __post_init__(self, cosmo=None, knodes=None, z: float = 0.5, order: str = "one-loop",
                      cutoff: float | None = 0.5):
        self.z, self.order, self.cutoff = float(z), order, cutoff
        self.knodes = np.asarray(knodes, dtype="f8")
        self.klin = np.geomspace(1e-4, 10.0, 1024)
        self.cosmo.add_requirements({"fourier.pk": [{"of": "delta_m", "z": self.z, "k": self.klin}],
                                     "params.A_s": None})

    def __call__(self):
        pk = self.cosmo.get("fourier.pk", of="delta_m", z=self.z, k=self.klin)
        ratio = float(self.cosmo.get("params.A_s")) / POWER_AMPLITUDE
        terms = matter_power_terms(self.klin, np.asarray(pk), self.knodes, order=self.order, cutoff=self.cutoff)
        self.pk_terms = terms / np.array([ratio, ratio**2])[:, None]
        self.dlogA = np.log(ratio)
        return self

    def tree_flatten(self):
        return [self.pk_terms, self.dlogA], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk_terms, obj.dlogA = children
        return obj


def power_rows(basis):
    """P_L and L_Lambda at the basis's amplitude, from a (possibly emulated) PowerBasis output."""
    amplitude = jnp.exp(basis.dlogA)
    return amplitude * basis.pk_terms[0], amplitude**2 * basis.pk_terms[1]


class PowerTheory(Calculator):
    """Binned real-space matter P(k): binning.matrix @ (P_L + L - 2 cs2_pk k^2 P_L) + shot noise.

    ``binning`` is a ``LatticeBinning`` on the basis's nodes; ``shotnoise`` (V / N) is known and fixed.
    The counterterm ``cs2_pk`` ((Mpc/h)^2) is free by default at one loop and absent at tree level.
    """

    def __init__(self, binning: LatticeBinning, basis=None, shotnoise: float = 0.0, order: str = "one-loop",
                 counterterm: bool | None = None, **basis_kwargs):
        self.basis = PowerBasis(knodes=binning.knodes, order=order, **basis_kwargs) if basis is None else basis
        if order == "one-loop" if counterterm is None else counterterm:
            self.cs2 = Parameter("cs2_pk", value=0.0, prior=dict(limits=[-20.0, 20.0]),
                                 ref=dict(dist="norm", loc=0.0, scale=1.0), latex=r"c_{s,P}^2")
        else:
            self.cs2 = None

    def __post_init__(self, binning: LatticeBinning, basis=None, shotnoise: float = 0.0, order: str = "one-loop",
                      counterterm: bool | None = None, **basis_kwargs):
        self.matrix = jnp.asarray(binning.matrix)
        self.noise = jnp.asarray(shotnoise * binning.noise)
        self.k2 = jnp.asarray(binning.knodes ** 2)
        self.loop = order == "one-loop"

    def __call__(self):
        linear, loop = power_rows(self.basis)
        power = linear + loop if self.loop else linear
        if self.cs2 is not None:
            power = power - 2 * self.cs2.value * self.k2 * linear
        self.flattheory = self.matrix @ power + self.noise
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


class RSDBasis(Calculator):
    """Cosmology-dependent redshift-space pieces shared by the P(k) multipoles and the S1m blocks.

    Outputs ``grid_terms`` (5, nk, nmu): GRID_TERMS of ``wstfast.theory.rsd.RSDGrid``; ``kappa`` (ncoef, L + 1) and
    ``skewness2`` (ncoef,): the velocity-damped tree cumulants of the S1m blocks of ``coefficients``; ``s21m_terms``
    (nentry, 2): the S21M_TERMS of every S21m entry of ``s21_coefficients`` (``wstfast.theory.rsd_moduli``); and
    ``dlogA``.
    With ``ir``, the grid is BAO IR-resummed (``wstfast.theory.rsd.RSDGrid``).
    Rows are rescaled by powers of A_s / POWER_AMPLITUDE (Kaiser and counterterms 1, loop 2, kappa 3, skewness^2 4,
    S21m response 1, noise 0)
    so that a Taylor emulator is nearly exact in logA (the damping breaks this slightly). NumPy (pure_callback).
    """

    _is_external = True

    def __init__(self, cosmo=None, config: WSTConfig = WSTConfig(), coefficients=(), z: float = 0.5,
                 damping: str | float | None = "linear", kmax: float = 0.3, npoints: int = 2**16, ir: bool = False,
                 s21_coefficients=(), tracer: str = "matter"):
        self.cosmo = build_cosmology() if cosmo is None else cosmo

    def __post_init__(self, cosmo=None, config: WSTConfig = WSTConfig(), coefficients=(), z: float = 0.5,
                      damping: str | float | None = "linear", kmax: float = 0.3, npoints: int = 2**16, ir: bool = False,
                      s21_coefficients=(), tracer: str = "matter"):
        from .theory.rsd import RSDGrid, RSDS1Basis
        from .theory.rsd_bias import BiasedRSDGrid, BiasedRSDS1Basis, BiasedRSDS21mBasis
        from .theory.rsd_moduli import RSDS21mBasis

        if tracer not in ("matter", "biased"):
            raise ValueError(f"unknown tracer {tracer!r}")
        self.z, self.tracer = float(z), tracer
        biased = tracer == "biased"
        self.grid = (BiasedRSDGrid if biased else RSDGrid)(kmax=kmax, ir=ir)
        self.coefficients = list(coefficients)
        self.cumulant_model = ((BiasedRSDS1Basis if biased else RSDS1Basis)(config, self.coefficients, damping=damping,
                                                                             npoints=npoints, lattice=False)
                               if self.coefficients else None)
        self.s21_model = ((BiasedRSDS21mBasis if biased else RSDS21mBasis)(config, list(s21_coefficients))
                          if len(s21_coefficients) else None)
        self.klin = np.geomspace(1e-4, 10.0, 1024)
        self.cosmo.add_requirements({"fourier.pk": [{"of": "delta_m", "z": self.z, "k": self.klin}],
                                     "fourier.sigma8_z": [{"of": "delta_cb", "z": self.z}, {"of": "theta_cb", "z": self.z}],
                                     "params.A_s": None, "params.h": None, "thermodynamics.rs_drag": None})

    def __call__(self):
        pk = np.asarray(self.cosmo.get("fourier.pk", of="delta_m", z=self.z, k=self.klin))
        f = float(self.cosmo.get("fourier.sigma8_z", of="theta_cb", z=self.z)
                  / self.cosmo.get("fourier.sigma8_z", of="delta_cb", z=self.z))
        ratio = float(self.cosmo.get("params.A_s")) / POWER_AMPLITUDE
        h, r_bao = float(self.cosmo.get("params.h")), float(self.cosmo.get("thermodynamics.rs_drag"))
        if self.tracer == "biased":
            from .theory.rsd_bias import biased_scalings

            scale = biased_scalings(ratio)
            self.grid_terms = self.grid(self.klin, pk, f, h=h, r_bao=r_bao) / scale["grid"][:, None, None]
            if self.cumulant_model is not None:
                kappa, skewness = self.cumulant_model.cumulants(self.klin, pk, f)
                self.kappa, self.skewness2 = kappa / scale["kappa"], skewness / scale["skewness"]
            else:
                self.kappa, self.skewness2 = np.zeros((0, 1, scale["kappa"].size)), np.zeros((0, scale["skewness"].size))
            self.s21m_terms = (self.s21_model(self.klin, pk, f) / scale["s21m"] if self.s21_model is not None
                               else np.zeros((0, scale["s21m"].size)))
            self.dlogA = np.log(ratio)
            return self
        self.grid_terms = (self.grid(self.klin, pk, f, h=h, r_bao=r_bao)
                           / np.array([ratio, ratio**2, ratio, ratio, ratio])[:, None, None])
        if self.cumulant_model is not None:
            kappa, skewness2 = self.cumulant_model.cumulants(self.klin, pk, f)
        else:
            kappa, skewness2 = np.zeros((0, 1)), np.zeros(0)
        self.kappa, self.skewness2 = kappa / ratio**3, skewness2 / ratio**4
        s21m = self.s21_model(self.klin, pk, f) if self.s21_model is not None else np.zeros((0, 2))
        self.s21m_terms = s21m / np.array([ratio, 1.0])
        self.dlogA = np.log(ratio)
        return self

    def tree_flatten(self):
        return [self.grid_terms, self.kappa, self.skewness2, self.s21m_terms, self.dlogA], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        if len(children) == 4:  # emulators trained before the S21m terms existed
            children = [*children[:3], jnp.zeros((0, 2)), children[3]]
        obj.grid_terms, obj.kappa, obj.skewness2, obj.s21m_terms, obj.dlogA = children
        return obj


def rsd_rows(basis):
    """Grid rows, kappa and skewness^2 at the basis's amplitude, from a (possibly emulated) RSDBasis output."""
    a = jnp.exp(basis.dlogA)
    scale = jnp.stack([a, a**2, a, a, a])[:, None, None]
    return basis.grid_terms * scale, basis.kappa * a**3, basis.skewness2 * a**4


def s21m_rows(basis):
    """S21m terms (nentry, 2) at the basis's amplitude, from a (possibly emulated) RSDBasis output."""
    return basis.s21m_terms * jnp.stack([jnp.exp(basis.dlogA), 1.0])


def biased_rows(basis):
    """Grid rows, kappa, skewness and S21m terms of a biased-tracer RSDBasis at the basis's amplitude."""
    from .theory.rsd_bias import biased_scalings

    scale = biased_scalings(jnp.exp(basis.dlogA))
    return (basis.grid_terms * scale["grid"][:, None, None], basis.kappa * scale["kappa"],
            basis.skewness2 * scale["skewness"], basis.s21m_terms * scale["s21m"])


class BiasParameters:
    """Eulerian bias (b1, b2, bG2, bGamma3) and stochastic (alpha0, alpha2) parameters of a tracer.

    Every theory of a joint fit makes its own instance; desilike shares the parameters by name. The shot noise
    is (1 + alpha0) V / N + alpha2 k^2 mu^2 V / N. Theories hold ``params`` as the attribute ``bias_params`` and
    must read the values from that attribute (desilike substitutes its own Parameter instances there):
    ``bias_beta(self.bias_params)``.
    """

    #: name: (fiducial value, limits, Gaussian prior sigma (None: flat within the limits), reference scale, latex).
    #: Gaussian priors centred on zero for the poorly constrained nuisances, as in EFT analyses of galaxies.
    SPECS = {"b1": (2.0, [0.0, 5.0], None, 0.1, "b_1"),
             "b2": (0.0, [-20.0, 20.0], 1.0, 0.5, "b_2"),
             "bG2": (0.0, [-20.0, 20.0], 1.0, 0.5, r"b_{\mathcal{G}_2}"),
             "bGamma3": (0.0, [-20.0, 20.0], 1.0, 0.5, r"b_{\Gamma_3}"),
             "alpha0": (0.0, [-1.0, 2.0], 0.3, 0.05, r"\alpha_0"),
             "alpha2": (0.0, [-100.0, 100.0], 10.0, 2.0, r"\alpha_2")}

    def __init__(self):
        self.params = {}
        for name, (value, limits, sigma, scale, latex) in self.SPECS.items():
            prior = dict(limits=limits) if sigma is None else dict(dist="norm", loc=0.0, scale=sigma, limits=limits)
            self.params[name] = Parameter(name, value=value, prior=prior, ref=dict(dist="norm", loc=value, scale=scale),
                                          latex=latex)



def bias_beta(params):
    """beta = (1, b1, b2, bG2, bGamma3) from a theory's ``bias_params``."""
    return jnp.stack([jnp.ones(()), *(jnp.asarray(params[name].value, dtype="f8")
                                      for name in ("b1", "b2", "bG2", "bGamma3"))])


from .theory.rsd_bias import TREE as TREE_INDEX  # noqa: E402


def biased_grid(rows, beta, counterterms, k2mu2, shotnoise, alpha2, ck4=None):
    """P_s grid and its tree part from the biased rows (BIASED_GRID_TERMS), with the stochastic k^2 mu^2 term and,
    with ``ck4`` [(Mpc/h)^4], the next-to-leading redshift-space counterterm -ck4 k^4 mu^4 (b1 + f mu^2)^2 P_L (the
    f^4 of the usual c~ (f mu)^4 k^4 form is absorbed into ck4)."""
    from .theory.bias import monomials
    from .theory.rsd_bias import LOOP, TREE

    m2 = monomials(beta, 2)
    nt, nl = len(TREE), len(LOOP)
    tree = jnp.tensordot(m2[np.array(TREE)], rows[:nt], axes=1)
    loop = jnp.tensordot(m2[np.array(LOOP)], rows[nt:nt + nl], axes=1)
    c0, c2, c4 = counterterms
    grid = tree + loop - 2 * (c0 * rows[-3] + c2 * rows[-2] + c4 * rows[-1]) + shotnoise * alpha2 * k2mu2
    if ck4 is not None:
        grid = grid - ck4 * k2mu2**2 * tree
    return grid, tree


def biased_cumulants(kappa, skewness, beta, shotnoise):
    """Per-block kappa (ncoef, L + 1) and 15 skewness^2 (ncoef,) of a tracer of Poisson shot noise ``shotnoise``."""
    from .theory.bias import monomials
    from .theory.rsd_bias import KAPPA_SIZES, SKEW_SIZES, split

    n = shotnoise
    k = split(kappa, KAPPA_SIZES)
    kappa = (k[0] @ monomials(beta, 4) + n * k[1] @ monomials(beta, 3) + n**2 * k[2] @ monomials(beta, 2)
             + n**3 * k[3][..., 0])
    sk = split(skewness, SKEW_SIZES)
    skew = sk[0] @ monomials(beta, 3) + n * sk[1] @ monomials(beta, 2) + n**2 * sk[2][..., 0]
    return kappa, 15 * skew**2


def _k4_counterterm(enabled, prefix, latex):
    """{'ck4': Parameter} of the k^4 mu^4 counterterm (``biased_grid``), or {} when not ``enabled``."""
    if not enabled:
        return {}
    return {"ck4": Parameter(f"ck4{prefix}", value=0.0, prior=dict(limits=[-1e4, 1e4]),
                             ref=dict(dist="norm", loc=0.0, scale=20.0), latex=rf"\tilde{{c}}{latex}")}


def _rsd_counterterms(prefix, latex):
    return {name: Parameter(f"{name}{prefix}", value=0.0, prior=dict(limits=[-100.0, 100.0]),
                            ref=dict(dist="norm", loc=0.0, scale=2.0), latex=rf"{label}{latex}")
            for name, label in (("c0", "c_0"), ("c2", "c_2"), ("c4", "c_4"))}


class APGeometry:
    """AP ratios of the trial cosmology. ``params`` are the basis's own ``h`` and ``omega_cdm`` Parameter instances
    (theories hold them as ``ap_params``, so that desilike sees one parameter each); ``omega_b`` is a constant (it is
    not an input of the emulated basis)."""

    def __init__(self, basis, z: float, omega_b: float = QUIJOTE_COSMOLOGY["omega_b"]):
        params = get_params(basis)
        self.params = {name: params[name] for name in ("h", "omega_cdm")}
        self.z, self.omega_b = float(z), float(omega_b)

    def ratios(self, params):
        from .theory.ap import ap_ratios

        return ap_ratios(self.z, params["h"].value, self.omega_b, params["omega_cdm"].value)


class RSDPowerTheory(Calculator):
    """Redshift-space matter P(k) multipoles, lattice-averaged: projection.matrix @ P_s grid + V/N projection.shot.

    Counterterms ``c0_pk``, ``c2_pk``, ``c4_pk`` [(Mpc/h)^2]: P_s -= 2 (c0 + c2 mu^2 + c4 mu^4) k^2 P_L.
    With ``ap`` (see ``APGeometry``), ``projection`` is an ``APProjection`` and the AP distortions are applied.
    With ``tracer="biased"`` (a biased-tracer basis), P_s is that of the tracer, with ``BiasParameters``.
    """

    def __init__(self, projection, basis=None, shotnoise: float = 0.0, ap=None, tracer: str = "matter",
                 shared_counterterms: bool = False, k4_counterterm: bool = False):
        self.basis = basis
        self.k4_params = _k4_counterterm(k4_counterterm, "" if shared_counterterms else "_pk",
                                         "" if shared_counterterms else r"^{P}")
        # Shared: the S1m names (c0, c2, c4), so that desilike merges them with the S1m counterterms of a joint fit.
        self.counterterms = _rsd_counterterms("", "") if shared_counterterms else _rsd_counterterms("_pk", r"^{P}")
        self.ap = ap
        self.ap_params = ap.params if ap is not None else {}
        self.bias_params = BiasParameters().params if tracer == "biased" else {}

    def __post_init__(self, projection, basis=None, shotnoise: float = 0.0, ap=None, tracer: str = "matter",
                      shared_counterterms: bool = False, k4_counterterm: bool = False):
        self.projection, self.shotnoise = projection, float(shotnoise)
        self.k2mu2 = jnp.asarray(projection.k2mu2)
        if ap is None:
            self.matrix = jnp.asarray(projection.matrix)
            self.shot = jnp.asarray(projection.shot)

    def __call__(self):
        c = [self.counterterms[name].value for name in ("c0", "c2", "c4")]
        noise = self.shotnoise
        if not self.bias_params:
            rows, _, _ = rsd_rows(self.basis)
            grid = rows[0] + rows[1] - 2 * (c[0] * rows[2] + c[1] * rows[3] + c[2] * rows[4])
        else:
            rows = biased_rows(self.basis)[0]
            ck4 = self.k4_params["ck4"].value if self.k4_params else None
            grid, _ = biased_grid(rows, bias_beta(self.bias_params), c, self.k2mu2, self.shotnoise,
                                  self.bias_params["alpha2"].value, ck4)
            noise = self.shotnoise * (1 + self.bias_params["alpha0"].value)
        if self.ap is None:
            self.flattheory = self.matrix @ grid.ravel() + noise * self.shot
        else:
            qpar, qperp = self.ap.ratios(self.ap_params)
            self.flattheory = self.projection(grid[None], qpar, qperp, noise)[0]
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


class S1mTheory(Calculator):
    """Line-of-sight-resolved S1m of the selected S1 coefficients (all |m| blocks), from the shared RSDBasis.

    Counterterms ``c0``, ``c2``, ``c4`` [(Mpc/h)^2] of the block variances (separate from the P(k) ones).
    ``correction`` = (kappa ratio, skewness^2 ratio) from ``wstfast.theory.rsd.cumulant_correction`` corrects the
    sampling error of the basis cumulants. ``ng_amplitude`` frees the amplitude of the Edgeworth correction (the
    damped tree cumulants overshoot it by ~7% at sigma = 25 Mpc/h, less at larger scales):

    - ``"constant"`` (or True): one factor ``a_ng`` for every block;
    - ``"slope"``: a(sigma) = a_ng + b_ng [(NG_PIVOT / sigma)^2 - 1], so ``a_ng`` is the amplitude at the pivot;
    - ``"per-scale"``: one factor ``a_ng_j{j}`` per filter scale.

    The last two need ``sigmas``, the filter scale of each coefficient.
    """

    def __init__(self, projection, coefficients, q: float = 0.8, basis=None, shotnoise: float = 0.0,
                 cumulant_index=None, correction=None, ng_amplitude=False, ap=None, tracer: str = "matter",
                 sigmas=None, k4_counterterm: bool = False):
        self.basis = basis
        self.bias_params = BiasParameters().params if tracer == "biased" else {}
        self.counterterms = _rsd_counterterms("", "")
        self.k4_params = _k4_counterterm(k4_counterterm, "", "")
        self.ap = ap
        self.ap_params = ap.params if ap is not None else {}
        self.ng_params = _ng_amplitudes(ng_amplitude, coefficients)

    def __post_init__(self, projection, coefficients, q: float = 0.8, basis=None, shotnoise: float = 0.0,
                      cumulant_index=None, correction=None, ng_amplitude=False, ap=None, tracer: str = "matter",
                      sigmas=None, k4_counterterm: bool = False):
        self.projection, self.shotnoise = projection, float(shotnoise)
        self.ng_mode = "constant" if ng_amplitude is True else ng_amplitude
        if self.ng_mode in ("slope", "per-scale") and sigmas is None:
            raise ValueError(f"ng_amplitude={ng_amplitude!r} needs the filter scales (sigmas)")
        self.ng_slope = None if sigmas is None else (NG_PIVOT / np.asarray(sigmas, dtype=float)) ** 2 - 1
        self.ng_names = [f"a_ng_j{c.j}" for c in coefficients]
        self.k2mu2 = jnp.asarray(projection.k2mu2)
        if ap is None:
            self.matrix = jnp.asarray(projection.matrix)
            self.noise = jnp.asarray(shotnoise * projection.shot)
        self.ells = [c.ell for c in coefficients]
        self.q = float(q)
        #: Rows of the basis cumulants for each selected coefficient (the basis may hold more coefficients).
        self.cumulant_index = np.arange(len(coefficients)) if cumulant_index is None else np.asarray(cumulant_index)
        self.kappa_ratio, self.skewness_ratio = ((jnp.asarray(correction[0]), jnp.asarray(correction[1]))
                                                 if correction is not None else (1.0, 1.0))

    def __call__(self):
        from .theory.rsd import s1m_from_variances

        amplitude = self.ng_amplitude()
        c = [self.counterterms[name].value for name in ("c0", "c2", "c4")]
        if not self.bias_params:
            rows, kappa, skewness2 = rsd_rows(self.basis)
            kappa = kappa[self.cumulant_index] * self.kappa_ratio
            skewness2 = skewness2[self.cumulant_index] * self.skewness_ratio
            grid = rows[0] + rows[1] - 2 * (c[0] * rows[2] + c[1] * rows[3] + c[2] * rows[4])
            linear_grid, noise, linear_noise = rows[0], 1.0, 0.0
        else:  # the Gaussian part of the Edgeworth normalisation includes the shot noise
            rows, kappa, skewness, _ = biased_rows(self.basis)
            beta = bias_beta(self.bias_params)
            kappa, skewness2 = biased_cumulants(kappa[self.cumulant_index] * self.kappa_ratio,
                                                skewness[self.cumulant_index] * self.skewness_ratio, beta,
                                                self.shotnoise)
            ck4 = self.k4_params["ck4"].value if self.k4_params else None
            grid, linear_grid = biased_grid(rows, beta, c, self.k2mu2, self.shotnoise,
                                            self.bias_params["alpha2"].value, ck4)
            noise = linear_noise = 1 + self.bias_params["alpha0"].value
        if self.ap is None:
            variances = self.matrix @ grid.ravel() + noise * self.noise
            linear = self.matrix @ linear_grid.ravel() + linear_noise * self.noise
        else:  # the cumulants (Edgeworth numerators) are left without AP: ~1e-4 of S1m
            qpar, qperp = self.ap.ratios(self.ap_params)  # one interpolation for both grids
            variances, linear = self.projection(jnp.stack([grid, linear_grid]), qpar, qperp,
                                                jnp.stack([noise, linear_noise]) * self.shotnoise)
        self.flattheory = s1m_from_variances(variances, linear, kappa, skewness2, self.ells, self.q,
                                             edgeworth_amplitude=amplitude)
        return self.flattheory

    def ng_amplitude(self):
        """Edgeworth amplitude of each coefficient (a scalar for "constant")."""
        if not self.ng_mode:
            return 1.0
        if self.ng_mode == "constant":
            return self.ng_params["a_ng"].value
        if self.ng_mode == "slope":
            return self.ng_params["a_ng"].value + self.ng_params["b_ng"].value * self.ng_slope
        return jnp.stack([self.ng_params[name].value for name in self.ng_names])

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


class S21mTheory(Calculator):
    """Line-of-sight-resolved S21m = S2m / S1m of the selected S21 coefficients (every |m1|, |m2|), from the shared
    RSDBasis (``wstfast.theory.rsd_moduli``).

    Nuisance parameters: one noise amplitude ``noise_j{j1}_l{l}_m{m1}`` per first-layer field (its Gaussian noise is
    (1 + noise) times the tree value). The one-point factor (1 + E_1(1))^q / (1 + E_1(q)) uses the linear block
    variances (``projection``: an ``S1mProjection`` of the first-layer S1 coefficients ``first_layer``) and the basis
    cumulants of those coefficients (rows ``cumulant_index``). ``entry_index`` selects the entries of the basis.
    """

    def __init__(self, projection, coefficients, first_layer, q: float = 0.8, basis=None, entry_index=None,
                 cumulant_index=None, shotnoise: float = 0.0, tracer: str = "matter"):
        from .theory.rsd_moduli import first_layer_fields

        self.basis = basis
        self.bias_params = BiasParameters().params if tracer == "biased" else {}
        self.noise = {key: Parameter(f"noise_j{key[0]}_l{key[1]}_m{key[2]}", value=0.0, prior=dict(limits=[-1.0, 5.0]),
                                     ref=dict(dist="norm", loc=0.0, scale=0.05),
                                     latex=rf"a_{{N,{key[0]},{key[1]},{key[2]}}}")
                      for key in first_layer_fields(coefficients)}

    def __post_init__(self, projection, coefficients, first_layer, q: float = 0.8, basis=None, entry_index=None,
                      cumulant_index=None, shotnoise: float = 0.0, tracer: str = "matter"):
        from .theory.rsd_moduli import first_layer_fields, s21m_entries

        self.q, self.shotnoise = float(q), float(shotnoise)
        self.shot = jnp.asarray(projection.shot)
        entries = s21m_entries(coefficients)
        self.entry_index = np.arange(len(entries)) if entry_index is None else np.asarray(entry_index)
        self.keys = first_layer_fields(coefficients)
        self.field_index = np.array([self.keys.index((coefficients[i].j, coefficients[i].ell, m1))
                                     for i, m1, _ in entries], dtype=int)
        self.n1 = np.array([1.0 if m1 == 0 else 2.0 for _, m1, _ in entries])
        self.n2 = np.array([1.0 if m2 == 0 else 2.0 for _, _, m2 in entries])
        # First-layer blocks: position of each entry's (j1, l, |m1|) block in the projection's flat block list.
        offsets = np.cumsum([0] + [c.ell + 1 for c in first_layer])
        position = {(c.j, c.ell): offsets[i] for i, c in enumerate(first_layer)}
        self.block_index = np.array([position[coefficients[i].j, coefficients[i].ell] + m1 for i, m1, _ in entries])
        self.first_ells = [c.ell for c in first_layer]
        self.matrix = jnp.asarray(projection.matrix)
        self.cumulant_index = np.arange(len(first_layer)) if cumulant_index is None else np.asarray(cumulant_index)

    def __call__(self):
        from .theory.rsd import block_edgeworth
        from .theory.rsd_moduli import s21m_from_terms

        if not self.bias_params:
            rows, kappa, skewness2 = rsd_rows(self.basis)
            linear = self.matrix @ rows[0].ravel()
            kappa, skewness2 = kappa[self.cumulant_index], skewness2[self.cumulant_index]
            terms = s21m_rows(self.basis)[self.entry_index]
        else:
            from .theory.bias import monomials
            from .theory.rsd_bias import S21_SIZES, split

            rows, kappa, skewness, s21 = biased_rows(self.basis)
            beta = bias_beta(self.bias_params)
            noise = self.shotnoise * (1 + self.bias_params["alpha0"].value)
            m2, m4 = monomials(beta, 2), monomials(beta, 4)
            tree = jnp.tensordot(m2[np.array(TREE_INDEX)], rows[:len(TREE_INDEX)], axes=1)
            linear = self.matrix @ tree.ravel() + noise * self.shot
            kappa, skewness2 = biased_cumulants(kappa[self.cumulant_index], skewness[self.cumulant_index], beta,
                                                self.shotnoise)
            r0, r1, r2, g0, g1, g2, n0, n1 = split(s21[self.entry_index], S21_SIZES)
            norm2 = (n0 @ m2 + noise * n1[:, 0]) ** 2
            terms = jnp.stack([(r0 + noise * r1 + noise**2 * r2) @ m4 / norm2,
                               (g0 @ m4 + noise * g1 @ m2 + noise**2 * g2[:, 0]) / norm2], axis=1)
        e_q, e_1, start = [], [], 0
        for index, ell in enumerate(self.first_ells):
            lin = linear[start:start + ell + 1]
            e_q.append(block_edgeworth(lin, kappa[index, :ell + 1], skewness2[index], ell, self.q))
            e_1.append(block_edgeworth(lin, kappa[index, :ell + 1], skewness2[index], ell, 1.0))
            start += ell + 1
        e_q, e_1 = jnp.concatenate(e_q)[self.block_index], jnp.concatenate(e_1)[self.block_index]
        amplitudes = jnp.stack([self.noise[key].value for key in self.keys])[self.field_index]
        self.flattheory = s21m_from_terms(terms, amplitudes, self.n1, self.n2, self.q, e_q, e_1)
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


#: Pivot scale [Mpc/h] of the "slope" Edgeworth amplitude (the smallest S1m scale of the baseline fits).
NG_PIVOT = 25.0


def _ng_amplitudes(mode, coefficients) -> dict:
    """Free Edgeworth amplitudes of S1mTheory (see its docstring)."""
    def amplitude(name, latex):
        return Parameter(name, value=1.0, prior=dict(limits=[0.5, 1.5]), ref=dict(dist="norm", loc=1.0, scale=0.02),
                         latex=latex)

    if not mode:
        return {}
    if mode in (True, "constant"):
        return {"a_ng": amplitude("a_ng", r"a_{\rm NG}")}
    if mode == "slope":
        slope = Parameter("b_ng", value=0.0, prior=dict(limits=[-1.0, 1.0]), ref=dict(dist="norm", loc=0.0, scale=0.02),
                          latex=r"b_{\rm NG}")
        return {"a_ng": amplitude("a_ng", r"a_{\rm NG}"), "b_ng": slope}
    if mode == "per-scale":
        return {f"a_ng_j{j}": amplitude(f"a_ng_j{j}", rf"a_{{\rm NG}}^{{({j})}}")
                for j in sorted({c.j for c in coefficients})}
    raise ValueError(f"unknown ng_amplitude {mode!r}")


class JointTheory(Calculator):
    """Concatenated predictions of several theories (e.g. WST and P(k)) that share the cosmological parameters.

    Parameters with the same name (the emulated cosmology) are shared; nuisance parameters stay separate.
    """

    def __init__(self, theories):
        self.theories = list(theories)

    def __call__(self):
        self.flattheory = jnp.concatenate([theory.flattheory for theory in self.theories])
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


class WSTLikelihood(GaussianLikelihood):
    """Gaussian likelihood of a data vector (WST or P(k)) with a fixed covariance."""

    def __init__(self, theory, data, covariance):
        self.theory = theory
        self.flatdata = jnp.asarray(data, dtype="f8")
        self._covariance = np.asarray(covariance, dtype="f8")

    def __post_init__(self, theory, data, covariance):
        from .data import precision_matrix

        self.precision = jnp.asarray(precision_matrix(self._covariance))

    def __call__(self):
        self.flattheory = self.theory.flattheory
        return super().__call__()
