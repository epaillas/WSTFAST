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
    ``skewness2`` (ncoef,): the velocity-damped tree cumulants of the S1m blocks of ``coefficients``; and ``dlogA``.
    Rows are rescaled by powers of A_s / POWER_AMPLITUDE (Kaiser and counterterms 1, loop 2, kappa 3, skewness^2 4)
    so that a Taylor emulator is nearly exact in logA (the damping breaks this slightly). NumPy (pure_callback).
    """

    _is_external = True

    def __init__(self, cosmo=None, config: WSTConfig = WSTConfig(), coefficients=(), z: float = 0.5,
                 damping: str | float | None = "linear", kmax: float = 0.3, npoints: int = 2**16):
        self.cosmo = build_cosmology() if cosmo is None else cosmo

    def __post_init__(self, cosmo=None, config: WSTConfig = WSTConfig(), coefficients=(), z: float = 0.5,
                      damping: str | float | None = "linear", kmax: float = 0.3, npoints: int = 2**16):
        from .theory.rsd import RSDGrid, RSDS1Basis

        self.z = float(z)
        self.grid = RSDGrid(kmax=kmax)
        self.coefficients = list(coefficients)
        self.cumulant_model = (RSDS1Basis(config, self.coefficients, damping=damping, npoints=npoints, lattice=False)
                               if self.coefficients else None)
        self.klin = np.geomspace(1e-4, 10.0, 1024)
        self.cosmo.add_requirements({"fourier.pk": [{"of": "delta_m", "z": self.z, "k": self.klin}],
                                     "fourier.sigma8_z": [{"of": "delta_cb", "z": self.z}, {"of": "theta_cb", "z": self.z}],
                                     "params.A_s": None})

    def __call__(self):
        pk = np.asarray(self.cosmo.get("fourier.pk", of="delta_m", z=self.z, k=self.klin))
        f = float(self.cosmo.get("fourier.sigma8_z", of="theta_cb", z=self.z)
                  / self.cosmo.get("fourier.sigma8_z", of="delta_cb", z=self.z))
        ratio = float(self.cosmo.get("params.A_s")) / POWER_AMPLITUDE
        self.grid_terms = self.grid(self.klin, pk, f) / np.array([ratio, ratio**2, ratio, ratio, ratio])[:, None, None]
        if self.cumulant_model is not None:
            kappa, skewness2 = self.cumulant_model.cumulants(self.klin, pk, f)
        else:
            kappa, skewness2 = np.zeros((0, 1)), np.zeros(0)
        self.kappa, self.skewness2 = kappa / ratio**3, skewness2 / ratio**4
        self.dlogA = np.log(ratio)
        return self

    def tree_flatten(self):
        return [self.grid_terms, self.kappa, self.skewness2, self.dlogA], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.grid_terms, obj.kappa, obj.skewness2, obj.dlogA = children
        return obj


def rsd_rows(basis):
    """Grid rows, kappa and skewness^2 at the basis's amplitude, from a (possibly emulated) RSDBasis output."""
    a = jnp.exp(basis.dlogA)
    scale = jnp.stack([a, a**2, a, a, a])[:, None, None]
    return basis.grid_terms * scale, basis.kappa * a**3, basis.skewness2 * a**4


def _rsd_counterterms(prefix, latex):
    return {name: Parameter(f"{name}{prefix}", value=0.0, prior=dict(limits=[-100.0, 100.0]),
                            ref=dict(dist="norm", loc=0.0, scale=2.0), latex=rf"{label}{latex}")
            for name, label in (("c0", "c_0"), ("c2", "c_2"), ("c4", "c_4"))}


class RSDPowerTheory(Calculator):
    """Redshift-space matter P(k) multipoles, lattice-averaged: projection.matrix @ P_s grid + V/N projection.shot.

    Counterterms ``c0_pk``, ``c2_pk``, ``c4_pk`` [(Mpc/h)^2]: P_s -= 2 (c0 + c2 mu^2 + c4 mu^4) k^2 P_L.
    """

    def __init__(self, projection, basis=None, shotnoise: float = 0.0):
        self.basis = basis
        self.counterterms = _rsd_counterterms("_pk", r"^{P}")

    def __post_init__(self, projection, basis=None, shotnoise: float = 0.0):
        self.matrix = jnp.asarray(projection.matrix)
        self.noise = jnp.asarray(shotnoise * projection.shot)

    def __call__(self):
        rows, _, _ = rsd_rows(self.basis)
        c = [self.counterterms[name].value for name in ("c0", "c2", "c4")]
        grid = rows[0] + rows[1] - 2 * (c[0] * rows[2] + c[1] * rows[3] + c[2] * rows[4])
        self.flattheory = self.matrix @ grid.ravel() + self.noise
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
    """

    def __init__(self, projection, coefficients, q: float = 0.8, basis=None, shotnoise: float = 0.0,
                 cumulant_index=None):
        self.basis = basis
        self.counterterms = _rsd_counterterms("", "")

    def __post_init__(self, projection, coefficients, q: float = 0.8, basis=None, shotnoise: float = 0.0,
                      cumulant_index=None):
        self.matrix = jnp.asarray(projection.matrix)
        self.noise = jnp.asarray(shotnoise * projection.shot)
        self.ells = [c.ell for c in coefficients]
        self.q = float(q)
        #: Rows of the basis cumulants for each selected coefficient (the basis may hold more coefficients).
        self.cumulant_index = np.arange(len(coefficients)) if cumulant_index is None else np.asarray(cumulant_index)

    def __call__(self):
        from .theory.rsd import s1m_from_variances

        rows, kappa, skewness2 = rsd_rows(self.basis)
        c = [self.counterterms[name].value for name in ("c0", "c2", "c4")]
        grid = rows[0] + rows[1] - 2 * (c[0] * rows[2] + c[1] * rows[3] + c[2] * rows[4])
        variances = self.matrix @ grid.ravel() + self.noise
        linear = self.matrix @ rows[0].ravel()
        self.flattheory = s1m_from_variances(variances, linear, kappa[self.cumulant_index],
                                             skewness2[self.cumulant_index], self.ells, self.q)
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


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
        self.precision = jnp.asarray(np.linalg.inv(self._covariance))

    def __call__(self):
        self.flattheory = self.theory.flattheory
        return super().__call__()
