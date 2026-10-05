"""Likelihood construction, profiling and MH sampling with desilike."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from desilike import build, get_params

from .calculators import WSTBasis, WSTLikelihood, WSTTheory, build_cosmology
from .config import WSTConfig
from .data import WSTDataset


def covariance_for(dataset: WSTDataset, kind: str = "auto", of_mean: bool = False) -> tuple[np.ndarray, str]:
    """Covariance of the data vector; 'auto' falls back to diagonal with too few realizations."""
    nreal, ndata = dataset.vectors.shape
    if kind == "auto":
        kind = "sample" if nreal >= ndata + 3 else "diagonal"
        if kind == "diagonal":
            warnings.warn(f"only {nreal} realizations for {ndata} data points: using a DIAGONAL covariance. "
                          "Measure more realizations for a usable full covariance.", stacklevel=2)
    return dataset.covariance(kind=kind, of_mean=of_mean), kind


def load_emulated_basis(path: Path, dataset: WSTDataset, vary):
    """Trained Taylor emulator of WSTBasis, checked against the data it will be compared with."""
    from desilike.emulators import Emulator

    settings = json.loads(Path(path).with_suffix(".json").read_text())
    # The basis does not depend on q, which only enters the exact assembly.
    trained = WSTConfig(**settings["config"]).with_q(dataset.config.q)
    if trained != dataset.config:
        raise ValueError(f"emulator {path} was trained for {trained}, the data have {dataset.config}")
    for key, value in dict(z=dataset.metadata["redshift"], shotnoise=dataset.shotnoise).items():
        if not np.isclose(settings[key], value):
            raise ValueError(f"emulator {path} was trained with {key}={settings[key]}, the data have {value}")
    if sorted(settings["vary"]) != sorted(vary):
        raise ValueError(f"emulator {path} varies {settings['vary']}, requested {list(vary)}")
    return Emulator.read(str(path)).to_calculator(), settings["bounds"]


def build_likelihood(dataset: WSTDataset, covariance: np.ndarray, vary=("omega_cdm", "logA"),
                     emulator: Path | None = None) -> WSTLikelihood:
    """WST likelihood with an exact (CLASS) or Taylor-emulated cosmology-dependent basis."""
    if emulator is None:
        basis = WSTBasis(cosmo=build_cosmology(vary), config=dataset.config, z=dataset.metadata["redshift"],
                         shotnoise=dataset.shotnoise)
    else:
        basis, bounds = load_emulated_basis(emulator, dataset, vary)
        # Keep the chains inside the emulated box.
        params = get_params(basis)
        for name, (low, high) in bounds.items():
            params[name].update(prior=dict(limits=[low, high]))
    theory = WSTTheory(dataset.coefficients, config=dataset.config, basis=basis)
    return WSTLikelihood(theory, dataset.mean, covariance)


def profile(posterior, output: Path, seed: int = 42, nstarts: int = 4):
    """Minuit maximisation from several starting points; the best is refined and saved."""
    from desilike.profilers import Minuit, Profiler

    graph = build(posterior)
    candidates = Profiler(graph, kernel=Minuit(), rng=seed).maximize(niterations=nstarts)
    best = candidates.choice(index=int(np.argmax(np.asarray(candidates.logpdf))), squeeze=True)
    names = graph.params.select(varied=True, derived=False).names()
    start = np.concatenate([np.asarray(best.best[name], dtype="f8").reshape(-1) for name in names])
    profiles = Profiler(graph, kernel=Minuit(), rng=seed).maximize(niterations=1, start=start)
    profiles.write(str(output))
    return profiles


def bestfit_values(profiles) -> dict[str, float]:
    chosen = profiles.choice(squeeze=True)
    return {str(name): float(np.asarray(value).reshape(-1)[-1]) for name, value in chosen.best.items()}


def sample_mh(posterior, profiles, output_dir: Path, chains: int = 4, seed: int = 42, max_steps: int = 50000,
              gelman_rubin: float = 1.03, ess: float = 300.0, check_every: int = 500):
    """Metropolis-Hastings chains started at the best fit, with the profile covariance as proposal."""
    from desilike.conditioning import AffineConditioner
    from desilike.samplers import MH, Sampler
    from desilike.samplers.proposals import GaussianProposal

    graph = build(posterior)
    varied = graph.params.select(varied=True, derived=False)
    covariance = profiles.covariance.select(varied)
    center = {name: value for name, value in bestfit_values(profiles).items() if name in varied.names()}
    ndim = len(varied)
    # desilike's MH multiplies the proposal by 2.38^2 / sqrt(ndim); this gives 2.38^2 / ndim in whitened units.
    sampler = Sampler(graph, kernel=MH(covariance=np.eye(ndim) / np.sqrt(ndim)), nparallel=chains, rng=seed,
                      output_dir=output_dir, conditioner=AffineConditioner(covariance=covariance, rescale="full"),
                      proposal=GaussianProposal(covariance, center=center))
    return sampler.run(gelman_rubin=gelman_rubin, ess=ess, min_steps=check_every, check_every=check_every,
                       max_steps=max_steps)


def summarize(likelihood, profiles, samples, dataset: WSTDataset, covariance_kind: str) -> dict:
    best = bestfit_values(profiles)
    build(likelihood)({name: value for name, value in best.items()
                       if name in get_params(likelihood).select(varied=True, derived=False).names()})
    residual = np.asarray(likelihood.flatdata) - np.asarray(likelihood.flattheory)
    chi2 = float(residual @ np.asarray(likelihood.precision) @ residual)
    nvaried = len(get_params(likelihood).select(varied=True, derived=False))
    summary = dict(bestfit=best, chi2=chi2, ndata=int(residual.size), nvaried=nvaried,
                   nrealizations=int(dataset.vectors.shape[0]), covariance=covariance_kind,
                   coefficients=[c.label for c in dataset.coefficients])
    if samples is not None:
        names = [name for name in best if name in samples]
        summary["posterior"] = {name: dict(mean=float(np.asarray(samples.mean(name))),
                                           std=float(np.asarray(samples.std(name)))) for name in names}
    return summary


def plot_fit(path: Path, likelihood, dataset: WSTDataset, covariance: np.ndarray, config: WSTConfig):
    """Data versus best-fit theory, as residuals in units of the error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, theory = np.asarray(likelihood.flatdata), np.asarray(likelihood.flattheory)
    error = np.sqrt(np.diag(covariance))
    labels = [c.label for c in dataset.coefficients]
    fig, axes = plt.subplots(2, 1, figsize=(max(6, 0.35 * len(labels)), 6), sharex=True,
                             gridspec_kw=dict(height_ratios=(2, 1)))
    x = np.arange(len(labels))
    axes[0].errorbar(x, data, error, fmt="o", ms=3, label="data")
    axes[0].plot(x, theory, "x", label="best fit")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("coefficient")
    axes[0].legend()
    axes[1].axhspan(-1, 1, color="0.9")
    axes[1].plot(x, (data - theory) / error, "o", ms=3)
    axes[1].set_ylabel(r"$\Delta / \sigma$")
    axes[1].set_xticks(x, labels, rotation=90, fontsize=7)
    fig.suptitle(f"WST fit (q = {config.q}, cells {config.cellsize:.2f} Mpc/h)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
