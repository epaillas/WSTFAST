"""desilike calculators: cosmology, WST basis (emulated), WST theory and Gaussian likelihood.

The graph is

    CosmoprimoCosmology -> WSTBasis -> WSTTheory -> WSTLikelihood

``WSTBasis`` holds everything that depends on cosmology and is what the Taylor emulator
replaces; ``WSTTheory`` adds the nuisance parameters exactly.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from desilike.base import Calculator, GaussianLikelihood
from desilike.parameter import Parameter
from desilike.theories import CosmoprimoCosmology

from .config import QUIJOTE_COSMOLOGY, WSTConfig
from .theory import Assembly, WSTMatterBasis

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

    Outputs ``s1_terms`` (n_S1, 4) and ``s21_terms`` (n_S21, 2); see ``wstfast.theory.model``.
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
    its Gaussian-chaos value).
    """

    def __init__(self, coefficients, config: WSTConfig = WSTConfig(), basis=None, **basis_kwargs):
        self.basis = WSTBasis(config=config, **basis_kwargs) if basis is None else basis
        self.cs2 = Parameter("cs2", value=0.0, prior=dict(limits=[-20.0, 20.0]),
                             ref=dict(dist="norm", loc=0.0, scale=1.0), latex="c_s^2")
        fields = sorted({(c.j, c.ell) for c in coefficients if c.kind == "S21"})
        self.noise = {key: Parameter(f"noise_j{key[0]}_l{key[1]}", value=0.0, prior=dict(limits=[-1.0, 5.0]),
                                     ref=dict(dist="norm", loc=0.0, scale=0.1), latex=rf"a_{{N,{key[0]},{key[1]}}}")
                      for key in fields}

    def __post_init__(self, coefficients, config: WSTConfig = WSTConfig(), basis=None, **basis_kwargs):
        self.assembly = Assembly(config, coefficients)

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


class WSTLikelihood(GaussianLikelihood):
    """Gaussian likelihood of a WST data vector with a fixed covariance."""

    def __init__(self, theory, data, covariance):
        self.theory = theory
        self.flatdata = jnp.asarray(data, dtype="f8")
        self._covariance = np.asarray(covariance, dtype="f8")

    def __post_init__(self, theory, data, covariance):
        self.precision = jnp.asarray(np.linalg.inv(self._covariance))

    def __call__(self):
        self.flattheory = self.theory.flattheory
        return super().__call__()
