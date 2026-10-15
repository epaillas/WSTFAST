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

from .config import ABACUS_COSMOLOGY, ABACUS_M_NCDM, QUIJOTE_COSMOLOGY, WSTConfig
from .theory import Assembly, WSTMatterBasis
from .theory.power import LatticeBinning, matter_power_terms

#: Prior ranges used when cosmological parameters are varied.
COSMOLOGY_PRIORS = {"h": (0.5, 0.9), "omega_cdm": (0.08, 0.16), "logA": (2.5, 3.6),
                    "omega_b": (0.015, 0.03), "n_s": (0.85, 1.1)}


def build_cosmology(varied=("omega_cdm", "logA"), engine="class", fiducial: str = "quijote") -> CosmoprimoCosmology:
    """Fiducial cosmology, Quijote (massless neutrinos) or AbacusSummit c000 (``fiducial="abacus"``: one 0.06 eV
    neutrino); only ``varied`` parameters are free."""
    if fiducial not in ("quijote", "abacus"):
        raise ValueError(f"unknown fiducial {fiducial!r}")
    values, m_ncdm = (QUIJOTE_COSMOLOGY, 0.0) if fiducial == "quijote" else (ABACUS_COSMOLOGY, ABACUS_M_NCDM)
    params = CosmoprimoCosmology.propose_params(engine=engine, fiducial="DESI")
    for name, value in values.items():
        params[name].update(value=value, fixed=name not in varied)
        if name in varied:
            params[name].update(prior=dict(limits=list(COSMOLOGY_PRIORS[name])),
                                ref=dict(dist="norm", loc=value, scale=params[name].ref.std()))
    params["m_ncdm"].update(value=m_ncdm, fixed=True)
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
                 layer2: bool = False, layer2_template: dict | None = None, layer2_mode: str = "tree",
                 **basis_kwargs):
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
        # Measured second-layer template (Assembly): renormalisation parameters of its tree-level term.
        self.ng2 = {}
        if layer2_template is not None:
            probe = Assembly(config, coefficients, edgeworth=edgeworth, layer2_template=layer2_template,
                             layer2_mode=layer2_mode)
            for key in probe.ng2_keys:
                tag = "all" if key == 0 else (f"l{key}" if isinstance(key, int) else f"j{key[0]}_l{key[1]}")
                if layer2_mode == "pade_l":
                    self.ng2[key] = Parameter(f"ng2c_{tag}", value=0.0, prior=dict(limits=[0.0, 500.0]),
                                              ref=dict(dist="norm", loc=1.0, scale=0.5), latex=rf"c_{{Y,{tag}}}")
                else:
                    self.ng2[key] = Parameter(f"ng2R_{tag}", value=1.0, prior=dict(limits=[-3.0, 5.0]),
                                              ref=dict(dist="norm", loc=0.5, scale=0.2), latex=rf"R_{{Y,{tag}}}")

    def __post_init__(self, coefficients, config: WSTConfig = WSTConfig(), basis=None, edgeworth: bool = True,
                      layer2: bool = False, layer2_template: dict | None = None, layer2_mode: str = "tree",
                      **basis_kwargs):
        self.assembly = Assembly(config, coefficients, edgeworth=edgeworth, layer2_template=layer2_template,
                                 layer2_mode=layer2_mode)

    def __call__(self):
        noise = jnp.stack([self.noise[key].value for key in self.assembly.noise_keys]) if self.noise else jnp.zeros(0)
        layer2 = (jnp.stack([self.layer2[ell].value for ell in self.assembly.layer2_keys]) if self.layer2 else None)
        ng2 = jnp.stack([self.ng2[key].value for key in self.assembly.ng2_keys]) if self.ng2 else None
        self.flattheory = self.assembly(self.basis.s1_terms, self.basis.s21_terms, self.cs2.value, noise, layer2, ng2)
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
                 s21_coefficients=(), tracer: str = "matter", s21_damping: float | None = None,
                 cubic_biases: bool = False, fog: bool = False, s21_lattice=(1000.0, 256), s21_cellsize: float = 0.0,
                 s21_projected: bool = True):
        self.cosmo = build_cosmology() if cosmo is None else cosmo

    def __post_init__(self, cosmo=None, config: WSTConfig = WSTConfig(), coefficients=(), z: float = 0.5,
                      damping: str | float | None = "linear", kmax: float = 0.3, npoints: int = 2**16, ir: bool = False,
                      s21_coefficients=(), tracer: str = "matter", s21_damping: float | None = None,
                      cubic_biases: bool = False, fog: bool = False, s21_lattice=(1000.0, 256),
                      s21_cellsize: float = 0.0, s21_projected: bool = True):
        from .theory.rsd import RSDGrid, RSDS1Basis
        from .theory.rsd_bias import BiasedRSDGrid, BiasedRSDS1Basis, BiasedRSDS21mBasis
        from .theory.rsd_moduli import RSDS21mBasis

        if tracer not in ("matter", "biased"):
            raise ValueError(f"unknown tracer {tracer!r}")
        self.z, self.tracer = float(z), tracer
        biased = tracer == "biased"
        if cubic_biases and not biased:
            raise ValueError("cubic_biases needs tracer='biased'")
        #: The kappa columns also hold the trispectrum of the cubic operators (``rsd_bias.CUBIC_KAPPA_SIZES``).
        self.cubic_biases = bool(cubic_biases)
        #: Fingers-of-God counterterm columns instead of the velocity damping, in the S1m cumulants and the S21m terms
        #: (``rsd_bias.fog_sizes``; combined at the counterterm c_v by S1mTheory / S21mTheory).
        self.fog = bool(fog)
        if self.fog and not biased:
            raise ValueError("fog needs tracer='biased'")
        if self.fog and s21_damping is not None:
            raise ValueError("fog replaces the velocity damping: drop s21_damping")
        self.grid = (BiasedRSDGrid if biased else RSDGrid)(kmax=kmax, ir=ir)
        self.coefficients = list(coefficients)
        cubic = dict(cubic=True) if self.cubic_biases else {}
        if self.fog:
            cubic["fog"] = True
        self.cumulant_model = ((BiasedRSDS1Basis if biased else RSDS1Basis)(config, self.coefficients, damping=damping,
                                                                             npoints=npoints, lattice=False, **cubic)
                               if self.coefficients else None)
        if s21_damping is not None and not biased:
            raise ValueError("s21_damping is only implemented for tracer='biased'")
        s21_kwargs = {} if s21_damping is None else dict(sigma_v_factor=s21_damping)
        if self.fog:
            s21_kwargs["fog"] = True
        # The lattice of the S21m mode sums (the periodic reference of the measurement) and the mass-assignment window
        # of the first-layer legs of the modulus spectra (0: none, as for the box fits).
        s21_kwargs.update(boxsize=float(s21_lattice[0]), nmesh=int(s21_lattice[1]), cellsize=float(s21_cellsize))
        self.s21_model = ((BiasedRSDS21mBasis if biased else RSDS21mBasis)(config, list(s21_coefficients), **s21_kwargs)
                          if len(s21_coefficients) else None)
        #: With ``s21_projected=False`` the S21m output is unprojected: the modulus-spectrum grids of every first-layer
        #: field and their norms (``s21m_grids``, ``s21m_norms``; ``BiasedRSDS21mBasis.field_grids``), projected to lambda_2 at
        #: assembly by S21mTheory (``s21_projection``, with AP or not); ``s21m_terms`` is then empty, with the column
        #: layout of the projected terms.
        self.s21_projected = bool(s21_projected)
        if not self.s21_projected and not biased:
            raise ValueError("the unprojected S21m basis is implemented for tracer='biased'")
        self.klin = np.geomspace(1e-4, 10.0, 1024)
        # cb (cold dark matter + baryons) power: the tracers follow it; equal to delta_m without massive neutrinos.
        self.cosmo.add_requirements({"fourier.pk": [{"of": "delta_cb", "z": self.z, "k": self.klin}],
                                     "fourier.sigma8_z": [{"of": "delta_cb", "z": self.z}, {"of": "theta_cb", "z": self.z}],
                                     "params.A_s": None, "params.h": None, "thermodynamics.rs_drag": None})

    def __call__(self):
        pk = np.asarray(self.cosmo.get("fourier.pk", of="delta_cb", z=self.z, k=self.klin))
        f = float(self.cosmo.get("fourier.sigma8_z", of="theta_cb", z=self.z)
                  / self.cosmo.get("fourier.sigma8_z", of="delta_cb", z=self.z))
        ratio = float(self.cosmo.get("params.A_s")) / POWER_AMPLITUDE
        h, r_bao = float(self.cosmo.get("params.h")), float(self.cosmo.get("thermodynamics.rs_drag"))
        if self.tracer == "biased":
            from .theory.rsd_bias import biased_scalings

            scale = biased_scalings(ratio, cubic=self.cubic_biases, fog=self.fog)
            self.grid_terms = self.grid(self.klin, pk, f, h=h, r_bao=r_bao) / scale["grid"][:, None, None]
            if self.cumulant_model is not None:
                kappa, skewness = self.cumulant_model.cumulants(self.klin, pk, f)
                self.kappa, self.skewness2 = kappa / scale["kappa"], skewness / scale["skewness"]
            else:
                self.kappa, self.skewness2 = np.zeros((0, 1, scale["kappa"].size)), np.zeros((0, scale["skewness"].size))
            if self.s21_projected:
                self.s21m_terms = (self.s21_model(self.klin, pk, f) / scale["s21m"] if self.s21_model is not None
                                   else np.zeros((0, scale["s21m"].size)))
            else:
                from .theory.rsd_bias import s21m_field_powers

                grids, norms = self.s21_model.field_grids(self.klin, pk, f)
                plam, pnorm = s21m_field_powers(self.fog)
                self.s21m_grids = grids / (ratio**plam)[None, :, None]
                self.s21m_norms = norms / ratio**pnorm
                self.s21m_terms = np.zeros((0, scale["s21m"].size))
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
        children = [self.grid_terms, self.kappa, self.skewness2, self.s21m_terms, self.dlogA]
        if not getattr(self, "s21_projected", True):
            children += [self.s21m_grids, self.s21m_norms]
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        if len(children) == 4:  # emulators trained before the S21m terms existed
            children = [*children[:3], jnp.zeros((0, 2)), children[3]]
        obj.grid_terms, obj.kappa, obj.skewness2, obj.s21m_terms, obj.dlogA = children[:5]
        if len(children) == 7:  # unprojected S21m (field grids and norms)
            obj.s21m_grids, obj.s21m_norms = children[5:]
        return obj


def rsd_rows(basis):
    """Grid rows, kappa and skewness^2 at the basis's amplitude, from a (possibly emulated) RSDBasis output."""
    a = jnp.exp(basis.dlogA)
    scale = jnp.stack([a, a**2, a, a, a])[:, None, None]
    return basis.grid_terms * scale, basis.kappa * a**3, basis.skewness2 * a**4


def s21m_rows(basis):
    """S21m terms (nentry, 2) at the basis's amplitude, from a (possibly emulated) RSDBasis output."""
    return basis.s21m_terms * jnp.stack([jnp.exp(basis.dlogA), 1.0])


def biased_s21m_fields(basis):
    """Field grids (nfield, ncol, ngrid) and norms (nfield, nnorm) of an unprojected biased RSDBasis output, at the
    basis's amplitude (``rsd_bias.s21m_field_powers``)."""
    from .theory.rsd_bias import is_fog, s21m_field_powers

    plam, pnorm = s21m_field_powers(is_fog(s21m=basis.s21m_terms))
    a = jnp.exp(basis.dlogA)
    return basis.s21m_grids * (a ** jnp.asarray(plam))[None, :, None], basis.s21m_norms * a ** jnp.asarray(pnorm)


def biased_rows(basis):
    """Grid rows, kappa, skewness and S21m terms of a biased-tracer RSDBasis at the basis's amplitude (with the
    Fingers-of-God counterterm orders of a ``fog`` basis; see ``rsd_bias.collapse_fog``)."""
    from .theory.rsd_bias import CUBIC_KAPPA_SIZES, KAPPA_SIZES, biased_scalings, fog_sizes, is_fog

    fog = is_fog(basis.kappa, basis.s21m_terms)
    cubic = basis.kappa.shape[-1] >= sum(KAPPA_SIZES) + sum(CUBIC_KAPPA_SIZES) + (fog_sizes(True)["kappa"] if fog else 0)
    scale = biased_scalings(jnp.exp(basis.dlogA), cubic=cubic, fog=fog)
    return (basis.grid_terms * scale["grid"][:, None, None], basis.kappa * scale["kappa"],
            basis.skewness2 * scale["skewness"], basis.s21m_terms * scale["s21m"])


class BiasParameters:
    """Eulerian bias (b1, b2, bG2, bGamma3) and stochastic (alpha0, alpha2) parameters of a tracer.

    Every theory of a joint fit makes its own instance; desilike shares the parameters by name. The shot noise
    is (1 + alpha0) V / N + alpha2 k^2 mu^2 V / N. Theories hold ``params`` as the attribute ``bias_params`` and
    must read the values from that attribute (desilike substitutes its own Parameter instances there):
    ``bias_beta(self.bias_params)``.

    ``cubic`` adds the cubic biases of the tree trispectrum (``bias.CUBIC_NAMES``), with Gaussian priors of widths
    CUBIC_WIDTHS: ``"zero"`` varies b3, bG3, bdG2 themselves, centred on zero; ``"relation"`` varies their offsets
    db3, dbG3, dbdG2 from the halo-calibrated and coevolution relations of ``bias.cubic_priors`` (b1, b2), so the
    priors are centred on the relations (``cubic_values``).
    """

    #: name: (fiducial value, limits, Gaussian prior sigma (None: flat within the limits), reference scale, latex).
    #: Gaussian priors centred on zero for the poorly constrained nuisances, as in EFT analyses of galaxies.
    SPECS = {"b1": (2.0, [0.0, 5.0], None, 0.1, "b_1"),
             "b2": (0.0, [-20.0, 20.0], 1.0, 0.5, "b_2"),
             "bG2": (0.0, [-20.0, 20.0], 1.0, 0.5, r"b_{\mathcal{G}_2}"),
             "bGamma3": (0.0, [-20.0, 20.0], 1.0, 0.5, r"b_{\Gamma_3}"),
             "alpha0": (0.0, [-1.0, 2.0], 0.3, 0.05, r"\alpha_0"),
             "alpha2": (0.0, [-100.0, 100.0], 10.0, 2.0, r"\alpha_2")}

    #: Gaussian prior widths of (b3, bG3, bdG2), or of their offsets from the relations.
    CUBIC_WIDTHS = (2.0, 1.0, 1.0)
    CUBIC_LATEX = ("b_3", r"b_{\mathcal{G}_3}", r"b_{\delta\mathcal{G}_2}")

    #: Stochastic amplitudes of the Poisson pieces of the tree trispectrum, kappa > k0 + A_TN N k1 + A_TN2 N^2 k2 + N^3 k3
    #: (1 = Poisson; halos are not Poisson), with Gaussian priors N(1, 1) as the shot amplitudes of Tucci & Philcox (2026).
    KAPPA_SHOT = ("A_TN", "A_TN2")

    def __init__(self, cubic: str | None = None, kappa_shot: bool = False):
        from .theory.bias import CUBIC_NAMES

        self.params = {}
        if kappa_shot:
            for name, latex in zip(self.KAPPA_SHOT, (r"A_{T,N}", r"A_{T,N^2}")):
                self.params[name] = Parameter(name, value=1.0, latex=latex,
                                              prior=dict(dist="norm", loc=1.0, scale=1.0, limits=[-5.0, 10.0]),
                                              ref=dict(dist="norm", loc=1.0, scale=0.1))
        for name, (value, limits, sigma, scale, latex) in self.SPECS.items():
            prior = dict(limits=limits) if sigma is None else dict(dist="norm", loc=0.0, scale=sigma, limits=limits)
            self.params[name] = Parameter(name, value=value, prior=prior, ref=dict(dist="norm", loc=value, scale=scale),
                                          latex=latex)
        if cubic not in (None, "zero", "relation"):
            raise ValueError(f"unknown cubic biases {cubic!r}")
        if cubic is not None:
            offset = cubic == "relation"
            for name, width, latex in zip(CUBIC_NAMES, self.CUBIC_WIDTHS, self.CUBIC_LATEX):
                name, latex = ("d" + name, r"\Delta " + latex) if offset else (name, latex)
                self.params[name] = Parameter(name, value=0.0, latex=latex,
                                              prior=dict(dist="norm", loc=0.0, scale=width, limits=[-30.0, 30.0]),
                                              ref=dict(dist="norm", loc=0.0, scale=0.1 * width))


def cubic_values(params):
    """(b3, bG3, bdG2) from a theory's ``bias_params`` (``BiasParameters(cubic=...)``), or None without them."""
    from .theory.bias import CUBIC_NAMES, cubic_priors

    if CUBIC_NAMES[0] in params:
        return jnp.stack([jnp.asarray(params[name].value, dtype="f8") for name in CUBIC_NAMES])
    if "d" + CUBIC_NAMES[0] in params:
        offsets = jnp.stack([jnp.asarray(params["d" + name].value, dtype="f8") for name in CUBIC_NAMES])
        return cubic_priors(jnp.asarray(params["b1"].value, dtype="f8"), jnp.asarray(params["b2"].value, dtype="f8")) + offsets
    return None



def kappa_shot_values(params):
    """(A_TN, A_TN2) from a theory's ``bias_params`` (``BiasParameters(kappa_shot=True)``), or None without them."""
    if "A_TN" not in params:
        return None
    return tuple(jnp.asarray(params[name].value, dtype="f8") for name in BiasParameters.KAPPA_SHOT)


def bias_beta(params):
    """beta = (1, b1, b2, bG2, bGamma3) from a theory's ``bias_params``."""
    return jnp.stack([jnp.ones(()), *(jnp.asarray(params[name].value, dtype="f8")
                                      for name in ("b1", "b2", "bG2", "bGamma3"))])


from .theory.rsd_bias import TREE as TREE_INDEX  # noqa: E402


def _value(params, name):
    return params[name].value if params else None


def biased_grid(rows, beta, counterterms, k2mu2, shotnoise, alpha2, ck4=None, k2=None, alpha_k2=None):
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
    if alpha_k2 is not None:  # isotropic scale-dependent stochasticity (halo exclusion): alpha_k2 k^2 V / N
        grid = grid + shotnoise * alpha_k2 * k2
    return grid, tree


def biased_cumulants(kappa, skewness, beta, shotnoise, cubic=None, shot=None):
    """Per-block kappa (ncoef, L + 1) and 15 skewness^2 (ncoef,) of a tracer of Poisson shot noise ``shotnoise``.
    ``cubic`` = (b3, bG3, bdG2) adds the cubic-operator trispectrum (needs the basis columns CUBIC_KAPPA_SIZES);
    ``shot`` = (A_TN, A_TN2) scales the N and N^2 pieces of kappa (non-Poisson stochasticity)."""
    from .theory.bias import monomials
    from .theory.rsd_bias import CUBIC_KAPPA_SIZES, KAPPA_SIZES, SKEW_SIZES, split

    n = shotnoise
    k = split(kappa, KAPPA_SIZES + CUBIC_KAPPA_SIZES)
    base = kappa
    a1, a2 = (1.0, 1.0) if shot is None else shot
    kappa = (k[0] @ monomials(beta, 4) + a1 * n * k[1] @ monomials(beta, 3) + a2 * n**2 * k[2] @ monomials(beta, 2)
             + n**3 * k[3][..., 0])
    if cubic is not None:
        if base.shape[-1] != sum(KAPPA_SIZES + CUBIC_KAPPA_SIZES):
            raise ValueError("cubic biases need an emulator trained with --cubic-biases")
        m3 = monomials(beta, 3)
        kappa = kappa + sum(cubic[c] * (k[4 + c] @ m3) for c in range(len(CUBIC_KAPPA_SIZES)))
    sk = split(skewness, SKEW_SIZES)
    skew = sk[0] @ monomials(beta, 3) + n * sk[1] @ monomials(beta, 2) + n**2 * sk[2][..., 0]
    return kappa, 15 * skew**2


def _k4_counterterm(enabled, prefix, latex):
    """{'ck4': Parameter} of the k^4 mu^4 counterterm (``biased_grid``), or {} when not ``enabled``."""
    if not enabled:
        return {}
    return {"ck4": Parameter(f"ck4{prefix}", value=0.0, prior=dict(limits=[-1e4, 1e4]),
                             ref=dict(dist="norm", loc=0.0, scale=20.0), latex=rf"\tilde{{c}}{latex}")}


def _k2_stochastic(enabled):
    """{'alpha_k2': Parameter} of the isotropic k^2 stochastic term (``biased_grid``) [(Mpc/h)^2], or {}."""
    if not enabled:
        return {}
    return {"alpha_k2": Parameter("alpha_k2", value=0.0, prior=dict(dist="norm", loc=0.0, scale=20.0, limits=[-200.0, 200.0]),
                                  ref=dict(dist="norm", loc=0.0, scale=1.0), latex=r"\alpha_{k^2}")}


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
                 shared_counterterms: bool = False, k4_counterterm: bool = False, k2_stochastic: bool = False):
        self.basis = basis
        self.k2_params = _k2_stochastic(k2_stochastic)
        self.k4_params = _k4_counterterm(k4_counterterm, "" if shared_counterterms else "_pk",
                                         "" if shared_counterterms else r"^{P}")
        # Shared: the S1m names (c0, c2, c4), so that desilike merges them with the S1m counterterms of a joint fit.
        self.counterterms = _rsd_counterterms("", "") if shared_counterterms else _rsd_counterterms("_pk", r"^{P}")
        self.ap = ap
        self.ap_params = ap.params if ap is not None else {}
        self.bias_params = BiasParameters().params if tracer == "biased" else {}

    def __post_init__(self, projection, basis=None, shotnoise: float = 0.0, ap=None, tracer: str = "matter",
                      shared_counterterms: bool = False, k4_counterterm: bool = False, k2_stochastic: bool = False):
        self.projection, self.shotnoise = projection, float(shotnoise)
        self.k2mu2 = jnp.asarray(projection.k2mu2)
        self.k2 = jnp.asarray(projection.k2)
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
                                  self.bias_params["alpha2"].value, ck4, self.k2, _value(self.k2_params, "alpha_k2"))
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
    - ``"per-scale"``: one factor ``a_ng_j{j}`` per filter scale;
    - ``"fog"``: the leading Fingers-of-God counterterm of the tree cumulants (Tucci & Philcox 2026), applied to
      each |m| block, E -> E [1 - c_T 4 <k^2 mu^2>_block / k_NL^2] with k_NL = FOG_KNL and <k^2 mu^2> the mean of
      k^2 mu^2 over the block's projection of the linear spectrum (the counterterm sum_i k_i^2 mu_i^2 over the four
      legs, each leg filtered like the block), on top of the linear velocity damping of the cumulants.

    ``fog`` (a basis trained with ``fog=True``): the Fingers-of-God counterterm c_v of every external leg of the tree
    cumulants in place of the velocity damping (``rsd_bias.collapse_fog``); "shared" (c_v, also the S21m one) or
    "separate" (c_v_s1). c_v = (f sigma_v k_NL)^2 / 2 reproduces the linear damping to second order.

    "slope" and "per-scale" need ``sigmas``, the filter scale of each coefficient.
    """

    def __init__(self, projection, coefficients, q: float = 0.8, basis=None, shotnoise: float = 0.0,
                 cumulant_index=None, correction=None, ng_amplitude=False, ap=None, tracer: str = "matter",
                 sigmas=None, k4_counterterm: bool = False, k2_stochastic: bool = False, cubic_biases: str | None = None,
                 kappa_shot: bool = False, fog: str | None = None):
        self.basis = basis
        self.fog_params = {k: v for k, v in _fog_counterterms(fog).items() if k == "s1"}
        self.k2_params = _k2_stochastic(k2_stochastic)
        self.bias_params = BiasParameters(cubic_biases, kappa_shot).params if tracer == "biased" else {}
        self.counterterms = _rsd_counterterms("", "")
        self.k4_params = _k4_counterterm(k4_counterterm, "", "")
        self.ap = ap
        self.ap_params = ap.params if ap is not None else {}
        self.ng_params = _ng_amplitudes(ng_amplitude, coefficients)

    def __post_init__(self, projection, coefficients, q: float = 0.8, basis=None, shotnoise: float = 0.0,
                      cumulant_index=None, correction=None, ng_amplitude=False, ap=None, tracer: str = "matter",
                      sigmas=None, k4_counterterm: bool = False, k2_stochastic: bool = False,
                      cubic_biases: str | None = None, kappa_shot: bool = False, fog: str | None = None):
        self.projection, self.shotnoise = projection, float(shotnoise)
        self.ng_mode = "constant" if ng_amplitude is True else ng_amplitude
        if self.ng_mode in ("slope", "per-scale") and sigmas is None:
            raise ValueError(f"ng_amplitude={ng_amplitude!r} needs the filter scales (sigmas)")
        self.ng_slope = None if sigmas is None else (NG_PIVOT / np.asarray(sigmas, dtype=float)) ** 2 - 1
        self.ng_names = [f"a_ng_j{c.j}" for c in coefficients]
        self.k2mu2 = jnp.asarray(projection.k2mu2)
        self.k2 = jnp.asarray(projection.k2)
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
            kappa, skewness = kappa[self.cumulant_index] * self.kappa_ratio, skewness[self.cumulant_index] * self.skewness_ratio
            if _fog_check(self.basis, self.fog_params):
                from .theory.rsd_bias import collapse_fog

                kappa, skewness, _ = collapse_fog(kappa, skewness, c_s1=self.fog_params["s1"].value)
            kappa, skewness2 = biased_cumulants(kappa, skewness, beta,
                                                self.shotnoise, cubic_values(self.bias_params),
                                                kappa_shot_values(self.bias_params))
            ck4 = self.k4_params["ck4"].value if self.k4_params else None
            grid, linear_grid = biased_grid(rows, beta, c, self.k2mu2, self.shotnoise,
                                            self.bias_params["alpha2"].value, ck4, self.k2,
                                            _value(self.k2_params, "alpha_k2"))
            noise = linear_noise = 1 + self.bias_params["alpha0"].value
        block_scale = None
        if self.ng_mode == "fog":  # 1 - c_T 4 <k^2 mu^2>_block / k_NL^2, from the block projection of the linear grid
            weight = self.matrix @ (linear_grid * self.k2mu2).ravel() / (self.matrix @ linear_grid.ravel())
            block_scale = 1.0 - self.ng_params["c_T"].value * 4 * weight / FOG_KNL**2
        if self.ap is None:
            variances = self.matrix @ grid.ravel() + noise * self.noise
            linear = self.matrix @ linear_grid.ravel() + linear_noise * self.noise
        else:  # the cumulants (Edgeworth numerators) are left without AP: ~1e-4 of S1m
            qpar, qperp = self.ap.ratios(self.ap_params)  # one interpolation for both grids
            variances, linear = self.projection(jnp.stack([grid, linear_grid]), qpar, qperp,
                                                jnp.stack([noise, linear_noise]) * self.shotnoise)
        self.flattheory = s1m_from_variances(variances, linear, kappa, skewness2, self.ells, self.q,
                                             edgeworth_amplitude=amplitude, block_scale=block_scale)
        return self.flattheory

    def ng_amplitude(self):
        """Edgeworth amplitude of each coefficient (a scalar for "constant")."""
        if not self.ng_mode or self.ng_mode == "fog":
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
                 cumulant_index=None, shotnoise: float = 0.0, tracer: str = "matter", bispectrum_shot: bool = False,
                 noise_shot: bool = False, full_norm: bool = False, response_amplitude: bool = False,
                 k2_stochastic: bool = False, white_noise: bool = False, config: WSTConfig | None = None,
                 k2_noise: bool = False, noise_correction=None, connected_amplitude: bool = False,
                 cubic_biases: str | None = None, noise_calibration=None, kappa_shot: bool = False, layer2=None,
                 fog: str | None = None, s21_projection=None, s21_field_rows=None, ap=None):
        from .theory.rsd_moduli import first_layer_fields

        self.basis = basis
        self.ap = ap
        self.ap_params = ap.params if ap is not None else {}
        # Fingers-of-God counterterm of a fog basis: c_s1 on the first-layer cumulants, c_s21 on the modulus spectra.
        self.fog_params = _fog_counterterms(fog)
        # Non-Gaussian second layer (``layer2``, see __call__): constants c_l of the beyond-tree suppression, fixed or
        # free with a Gaussian prior of width layer2['prior'] x c_l (at least its calibration error) around the calibration.
        self.ng2_params = {}
        if layer2 is not None and layer2.get("prior") is not None:
            for ell in sorted({c.ell for c in coefficients}):
                c, err = float(layer2["c"][ell]), float(layer2["c_err"][ell])
                width = max(layer2["prior"] * abs(c), err, 0.05)
                self.ng2_params[ell] = Parameter(f"ng2c_l{ell}", value=max(c, 0.0),
                                                 prior=dict(dist="norm", loc=c, scale=width, limits=[0.0, 500.0]),
                                                 ref=dict(dist="norm", loc=max(c, 0.0), scale=0.1 * width),
                                                 latex=rf"c_{{Y,{ell}}}")
        # Free amplitude A_c_j of the tree-level connected noise (``noise_correction``) per first-layer scale: the
        # predicted shape of the connected noise in (K, mu_K) and across fields, with a fitted amplitude (1 = predicted).
        self.connected_params = ({f"A_c_j{j}": Parameter(f"A_c_j{j}", value=1.0, prior=dict(limits=[-20.0, 20.0]),
                                                         ref=dict(dist="norm", loc=1.0, scale=0.1), latex=rf"A_{{c,{j}}}")
                                  for j in sorted({c.j for c in coefficients})} if connected_amplitude else {})
        # K^2 term of the modulus power spectra P_UU / <U>^2, per first-layer scale [(Mpc/h)^5]: the scale dependence
        # of the noise beyond the Gaussian one (the analog of c_s^2 for U), next to the white term W.
        self.k2noise_params = ({f"WK2_j{j}": Parameter(f"WK2_j{j}", value=0.0, prior=dict(limits=[-1e7, 1e7]),
                                                       ref=dict(dist="norm", loc=0.0, scale=1e3), latex=rf"W^{{(2)}}_{{{j}}}")
                                for j in sorted({c.j for c in coefficients})} if k2_noise else {})
        # K-independent (white) term of the modulus power spectra P_UU / <U>^2, one amplitude per first-layer scale
        # [(Mpc/h)^3]: stands for the connected (non-Gaussian) noise of the moduli, which the Gaussian noise lacks.
        self.white_params = ({f"W_j{j}": Parameter(f"W_j{j}", value=0.0, prior=dict(limits=[-1e5, 1e5]),
                                                   ref=dict(dist="norm", loc=0.0, scale=10.0), latex=rf"W_{{{j}}}")
                              for j in sorted({c.j for c in coefficients})} if white_noise else {})
        self.k2_params = _k2_stochastic(k2_stochastic and full_norm)
        # Free amplitude of the modulus response, r -> (1 + a_R_j) r, per first-layer scale j1 (measured: the model
        # response is low by a few percent, nearly uniformly in l and |m| at fixed j1).
        self.response_params = ({f"a_R_j{j}": Parameter(f"a_R_j{j}", value=0.0, prior=dict(dist="norm", loc=0.0, scale=0.2,
                                                                                         limits=[-0.5, 1.0]),
                                                        ref=dict(dist="norm", loc=0.0, scale=0.02), latex=rf"a_{{R,{j}}}")
                                 for j in sorted({c.j for c in coefficients})} if response_amplitude else {})
        # With full_norm, the S1m counterterms (shared by name) enter the first-layer variance.
        self.counterterms = _rsd_counterterms("", "") if full_norm and tracer == "biased" else {}
        self.bias_params = BiasParameters(cubic_biases, kappa_shot).params if tracer == "biased" else {}
        # Free amplitude of the Poisson ("same-object") bispectrum term of the response, B > (1 + B_shot) P(K) / nbar:
        # halos are not Poisson, and this stochastic term is independent of the power-spectrum one (alpha0).
        self.bshot_params = ({"B_shot": Parameter("B_shot", value=0.0, prior=dict(dist="norm", loc=0.0, scale=1.0,
                                                                                  limits=[-5.0, 5.0]),
                                                  ref=dict(dist="norm", loc=0.0, scale=0.1), latex=r"B_{\rm shot}")}
                             if bispectrum_shot and tracer == "biased" else {})
        # Free amplitude of the discreteness terms of the modulus noise, g > (1 + alpha_N) N [2 P + (1 + alpha_N) N]:
        # the 4-point stochasticity of halos need not be Poisson either.
        self.nshot_params = ({"alpha_N": Parameter("alpha_N", value=0.0, prior=dict(dist="norm", loc=0.0, scale=1.0,
                                                                                   limits=[-1.0, 5.0]),
                                                   ref=dict(dist="norm", loc=0.0, scale=0.05), latex=r"\alpha_N")}
                             if noise_shot and tracer == "biased" else {})
        self.noise = {key: Parameter(f"noise_j{key[0]}_l{key[1]}_m{key[2]}", value=0.0, prior=dict(limits=[-1.0, 5.0]),
                                     ref=dict(dist="norm", loc=0.0, scale=0.05),
                                     latex=rf"a_{{N,{key[0]},{key[1]},{key[2]}}}")
                      for key in first_layer_fields(coefficients)}
        # Simulation-calibrated noise: the per-field amplitudes and W are not free but a_i(theta) = a_i0 + grad_i . (theta -
        # theta_0) of the basis's own cosmological parameters (scripts/feasibility/noise_calibration.py), as dict(names,
        # value, gradient (nname, nparam), params, reference).
        self.calibration_params = {}
        if noise_calibration is not None:
            self.noise, self.white_params = {}, {}
            basis_params = get_params(basis)
            self.calibration_params = {str(name): basis_params[str(name)] for name in noise_calibration["params"]}

    def __post_init__(self, projection, coefficients, first_layer, q: float = 0.8, basis=None, entry_index=None,
                      cumulant_index=None, shotnoise: float = 0.0, tracer: str = "matter", bispectrum_shot: bool = False,
                      noise_shot: bool = False, full_norm: bool = False, response_amplitude: bool = False,
                      k2_stochastic: bool = False, white_noise: bool = False, config: WSTConfig | None = None,
                      k2_noise: bool = False, noise_correction=None, connected_amplitude: bool = False,
                      cubic_biases: str | None = None, noise_calibration=None, kappa_shot: bool = False, layer2=None,
                      fog: str | None = None, s21_projection=None, s21_field_rows=None, ap=None):
        from .theory.rsd_moduli import first_layer_fields, s21m_entries

        # Unprojected basis (RSDBasis(s21_projected=False)): lambda_2 is projected here from the field grids, with
        # ``s21_projection`` (S21mProjection over these coefficients, or S21mAPProjection with ``ap``); the basis field
        # of each of its fields is ``s21_field_rows`` (default: the same fields, in the same order).
        self.s21_projection = s21_projection
        if s21_projection is not None:
            if tracer != "biased":
                raise ValueError("s21_projection needs tracer='biased'")
            if noise_correction is not None or entry_index is not None:
                raise ValueError("s21_projection selects its own entries: no entry_index / noise_correction")
            if ap is not None and not hasattr(s21_projection, "at"):
                raise ValueError("ap needs an S21mAPProjection")
            nfield = len(first_layer_fields(coefficients))
            self.s21_field_rows = np.arange(nfield) if s21_field_rows is None else np.asarray(s21_field_rows)
            self.s21_matrix = None if hasattr(s21_projection, "at") else jnp.asarray(s21_projection.matrix)

        self.layer2 = layer2 is not None
        if layer2 is not None:
            entries = s21m_entries(coefficients)
            keys = [(coefficients[i].j, coefficients[i].j2, coefficients[i].ell, m1, m2) for i, m1, m2 in entries]
            missing = [k for k in keys if k not in layer2["rows"]]
            if missing:
                raise ValueError(f"no second-layer template for {missing}")
            self.ng2_d0 = jnp.asarray([layer2["rows"][k][0] for k in keys])
            self.ng2_d2 = jnp.asarray([layer2["rows"][k][1] for k in keys])
            self.ng2_var = jnp.asarray([layer2["var"][k[0], k[2]] for k in keys])
            self.ng2_ell = [k[2] for k in keys]
            self.ng2_c = jnp.asarray([max(float(layer2["c"][k[2]]), 0.0) for k in keys])
            self.ng2_beta = jnp.asarray(layer2["beta_ref"], dtype="f8")
            self.ng2_lin_ref = jnp.asarray([layer2["lin_ref"][k[0], k[2], k[3]] for k in keys])

        self.q, self.shotnoise = float(q), float(shotnoise)
        #: Fixed relative correction (1 + c_e) of the Gaussian noise of every basis entry, for the tree-level connected
        #: noise of the moduli (scripts/feasibility/connected_noise_correction.py); rows of the basis, as entry_index.
        self.noise_correction = None if noise_correction is None else jnp.asarray(noise_correction)
        if noise_calibration is not None:
            names = [str(name) for name in noise_calibration["names"]]
            self.calibration_names = [str(name) for name in noise_calibration["params"]]
            self.calibration_value = {name: float(v) for name, v in zip(names, noise_calibration["value"])}
            self.calibration_gradient = {name: jnp.asarray(g, dtype="f8") for name, g in zip(names, noise_calibration["gradient"])}
            self.calibration_reference = jnp.asarray(noise_calibration["reference"], dtype="f8")
        if white_noise or k2_noise or noise_calibration is not None:  # lattice sums of 1 and K^2 in P_UU / <U>^2
            from .theory.rsd_moduli import S21mProjection

            if config is None:
                raise ValueError("white_noise / k2_noise need the WST config")
            x, _ = np.polynomial.legendre.leggauss(12)
            kgrid = np.geomspace(4e-3, 0.12, 24)
            if s21_projection is not None:  # the lattice of the given projection (at q = 1)
                matrix = np.asarray(s21_projection.at(1.0, 1.0) if hasattr(s21_projection, "at")
                                    else s21_projection.matrix)
            else:
                matrix = S21mProjection(config, list(coefficients), first_layer_fields(coefficients),
                                        kgrid, x[6:]).matrix
            self.white_column = jnp.asarray(matrix.sum(axis=1))
            self.k2_column = jnp.asarray(matrix @ np.repeat(kgrid**2, 6))
        self.entry_j = np.array([coefficients[i].j for i, _, _ in s21m_entries(coefficients)])
        self.k2mu2 = jnp.asarray(projection.k2mu2)
        self.k2 = jnp.asarray(projection.k2)
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
            if self.s21_projection is not None:
                s21 = self._project_s21()
            if _fog_check(self.basis, self.fog_params):
                from .theory.rsd_bias import collapse_fog

                kappa, skewness, _ = collapse_fog(kappa, skewness, c_s1=self.fog_params["s1"].value)
                s21 = collapse_fog(s21m=s21, c_s21=_fog_value(self.fog_params, "s21"))[2]
            beta = bias_beta(self.bias_params)
            noise = self.shotnoise * (1 + self.bias_params["alpha0"].value)
            m2, m4 = monomials(beta, 2), monomials(beta, 4)
            tree = jnp.tensordot(m2[np.array(TREE_INDEX)], rows[:len(TREE_INDEX)], axes=1)
            linear = self.matrix @ tree.ravel() + noise * self.shot
            kappa, skewness2 = biased_cumulants(kappa[self.cumulant_index], skewness[self.cumulant_index], beta,
                                                self.shotnoise, cubic_values(self.bias_params),
                                                kappa_shot_values(self.bias_params))
            r0, r1, r2, g0, g1, g2, n0, n1 = split(s21[self.entry_index], S21_SIZES)
            norm2 = (n0 @ m2 + noise * n1[:, 0]) ** 2
            if self.counterterms:  # full_norm: tree -> one-loop + counterterms first-layer variance, block by block
                c = [self.counterterms[name].value for name in ("c0", "c2", "c4")]
                grid, _ = biased_grid(rows, beta, c, self.k2mu2, self.shotnoise, self.bias_params["alpha2"].value,
                                      k2=self.k2, alpha_k2=_value(self.k2_params, "alpha_k2"))
                full = self.matrix @ grid.ravel() + noise * self.shot
                norm2 = norm2 * (full / linear)[self.block_index] ** 2
            nr = noise * (1 + self.bshot_params["B_shot"].value) if self.bshot_params else noise
            ng = noise * (1 + self.nshot_params["alpha_N"].value) if self.nshot_params else noise
            terms = jnp.stack([(r0 + nr * r1 + nr**2 * r2) @ m4 / norm2,
                               (g0 @ m4 + ng * g1 @ m2 + ng**2 * g2[:, 0]) / norm2], axis=1)
        e_q, e_1, start = [], [], 0
        for index, ell in enumerate(self.first_ells):
            lin = linear[start:start + ell + 1]
            e_q.append(block_edgeworth(lin, kappa[index, :ell + 1], skewness2[index], ell, self.q))
            e_1.append(block_edgeworth(lin, kappa[index, :ell + 1], skewness2[index], ell, 1.0))
            start += ell + 1
        e_q, e_1 = jnp.concatenate(e_q)[self.block_index], jnp.concatenate(e_1)[self.block_index]
        white = None
        if self.calibration_params:  # noise nuisances from the simulation calibration at the trial cosmology
            theta = jnp.stack([jnp.asarray(self.calibration_params[n].value, dtype="f8") for n in self.calibration_names])
            value = lambda name: self.calibration_value[name] + self.calibration_gradient[name] @ (theta - self.calibration_reference)  # noqa: E731
            amplitudes = jnp.stack([value(f"noise_j{j}_l{l}_m{m}") for j, l, m in self.keys])[self.field_index]
            white = jnp.stack([value(f"W_j{j}") for j in self.entry_j]) * self.white_column
        else:
            amplitudes = jnp.stack([self.noise[key].value for key in self.keys])[self.field_index]
        if self.noise_correction is not None:
            correction = self.noise_correction[self.entry_index]
            if self.connected_params:
                correction = correction * jnp.stack([self.connected_params[f"A_c_j{j}"].value for j in self.entry_j])
            terms = jnp.stack([terms[:, 0], terms[:, 1] * (1 + correction)], axis=1)
        if self.white_params:
            white = jnp.stack([self.white_params[f"W_j{j}"].value for j in self.entry_j]) * self.white_column
        if white is not None:
            terms = jnp.stack([terms[:, 0] + white, terms[:, 1]], axis=1)
        if self.k2noise_params:
            k2 = jnp.stack([self.k2noise_params[f"WK2_j{j}"].value for j in self.entry_j]) * self.k2_column
            terms = jnp.stack([terms[:, 0] + k2, terms[:, 1]], axis=1)
        if self.response_params:
            scale = jnp.stack([(1 + self.response_params[f"a_R_j{j}"].value) ** 2 for j in self.entry_j])
            terms = jnp.stack([terms[:, 0] * scale, terms[:, 1]], axis=1)
        self.flattheory = s21m_from_terms(terms, amplitudes, self.n1, self.n2, self.q, e_q, e_1)
        if self.layer2:
            # Non-Gaussian second layer: S21m x (1 + d0 + R_l d2 s), with the Gaussian-field and tree-level terms of the
            # perturbation-theory template (scripts/layer2/template_rsd.py), s the ratio of the linear first-layer block
            # variance (fixed reference biases) to its fiducial value, and R_l = 1 / (1 + c_l n s_lin^2 s).
            if self.bias_params:
                from .theory.bias import monomials

                reference = jnp.tensordot(monomials(self.ng2_beta, 2)[np.array(TREE_INDEX)], rows[:len(TREE_INDEX)], axes=1)
                block = (self.matrix @ reference.ravel())[self.block_index]
            else:
                block = linear[self.block_index]
            scale = block / self.ng2_lin_ref
            c = (jnp.stack([self.ng2_params[ell].value for ell in self.ng2_ell]) if self.ng2_params else self.ng2_c)
            renorm = 1.0 / (1.0 + c * self.ng2_var * scale)
            self.flattheory = self.flattheory * (1 + self.ng2_d0 + renorm * self.ng2_d2 * scale)
        return self.flattheory

    def _project_s21(self):
        """S21m terms (nentry, ncol) of the coefficients from the unprojected basis: lambda_2 = matrix @ field grid, with
        the AP ratios of the trial cosmology when ``ap`` is set, then ``rsd_bias.s21m_assemble``."""
        from .theory.rsd_bias import is_fog, s21m_assemble

        grids, norms = biased_s21m_fields(self.basis)
        grids, norms = grids[self.s21_field_rows], norms[self.s21_field_rows]
        if self.s21_matrix is not None:
            matrix = self.s21_matrix
        else:
            qpar, qperp = self.ap.ratios(self.ap_params) if self.ap is not None else (1.0, 1.0)
            matrix = self.s21_projection.at(qpar, qperp)
        index = self.s21_projection.field_index
        lam = jnp.einsum("eg,ecg->ec", matrix, grids[index])
        return s21m_assemble(lam, norms[index], is_fog(s21m=self.basis.s21m_terms))

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


#: Pivot scale [Mpc/h] of the "slope" Edgeworth amplitude (the smallest S1m scale of the baseline fits).
NG_PIVOT = 25.0


#: k_NL^r of the Fingers-of-God counterterm of the S1m cumulants [h/Mpc] (Tucci & Philcox 2026).
FOG_KNL = 0.3


def _fog_counterterms(mode) -> dict:
    """Fingers-of-God counterterm of a ``fog`` basis (``rsd_bias.collapse_fog``): ``"shared"``, one c_v for the
    first-layer cumulants (S1m and the one-point factor of S21m) and the modulus spectra of S21m; ``"separate"``,
    c_v_s1 and c_v_s21. Keyed by role ("s1", "s21"); shared: "s1" only (see ``_fog_value``)."""
    if not mode:
        return {}
    def counterterm(name, latex):
        return Parameter(name, value=0.0, prior=dict(limits=[-5.0, 15.0]), ref=dict(dist="norm", loc=0.5, scale=0.2),
                         latex=latex)

    if mode == "shared":
        return {"s1": counterterm("c_v", r"c_v")}
    if mode == "separate":
        return {"s1": counterterm("c_v_s1", r"c_v^{(1)}"), "s21": counterterm("c_v_s21", r"c_v^{(2)}")}
    raise ValueError(f"unknown fog mode {mode!r}")


def _fog_value(params, role):
    return params.get(role, params["s1"]).value


def _fog_check(basis, params):
    from .theory.rsd_bias import is_fog

    fog = is_fog(basis.kappa, basis.s21m_terms)
    if fog and not params:
        raise ValueError("the basis holds the Fingers-of-God counterterm columns: set fog='shared' or 'separate'")
    if params and not fog:
        raise ValueError("fog needs a basis trained with fog=True")
    return fog


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
    if mode == "fog":
        return {"c_T": Parameter("c_T", value=0.0, prior=dict(limits=[-30.0, 30.0]), ref=dict(dist="norm", loc=0.0, scale=0.5),
                                 latex=r"c_T")}
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
