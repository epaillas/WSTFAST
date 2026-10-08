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
    """

    def __init__(self, coefficients, config: WSTConfig = WSTConfig(), basis=None, edgeworth: bool = True,
                 **basis_kwargs):
        self.basis = WSTBasis(config=config, **basis_kwargs) if basis is None else basis
        self.cs2 = Parameter("cs2", value=0.0, prior=dict(limits=[-20.0, 20.0]),
                             ref=dict(dist="norm", loc=0.0, scale=1.0), latex="c_s^2")
        fields = sorted({(c.j, c.ell) for c in coefficients if c.kind == "S21"})
        self.noise = {key: Parameter(f"noise_j{key[0]}_l{key[1]}", value=0.0, prior=dict(limits=[-1.0, 5.0]),
                                     ref=dict(dist="norm", loc=0.0, scale=0.1), latex=rf"a_{{N,{key[0]},{key[1]}}}")
                      for key in fields}

    def __post_init__(self, coefficients, config: WSTConfig = WSTConfig(), basis=None, edgeworth: bool = True,
                      **basis_kwargs):
        self.assembly = Assembly(config, coefficients, edgeworth=edgeworth)

    def __call__(self):
        noise = jnp.stack([self.noise[key].value for key in self.assembly.noise_keys]) if self.noise else jnp.zeros(0)
        self.flattheory = self.assembly(self.basis.s1_terms, self.basis.s21_terms, self.cs2.value, noise)
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
