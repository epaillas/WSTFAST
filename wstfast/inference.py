"""Likelihood construction, profiling and MH sampling with desilike."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from desilike import build, get_params

from .calculators import PowerBasis, PowerTheory, WSTBasis, WSTLikelihood, WSTTheory, build_cosmology
from .config import QUIJOTE_COSMOLOGY, WSTConfig
from .data import PowerDataset, WSTDataset
from .theory.power import LatticeBinning, default_knodes


def covariance_for(dataset: WSTDataset, kind: str = "auto", of_mean: bool = False) -> tuple[np.ndarray, str]:
    """Covariance of the data vector; 'auto' falls back to diagonal with too few realizations."""
    nreal, ndata = dataset.vectors.shape
    if kind == "auto":
        kind = "sample" if nreal >= ndata + 3 else "diagonal"
        if kind == "diagonal":
            warnings.warn(f"only {nreal} realizations for {ndata} data points: using a DIAGONAL covariance. "
                          "Measure more realizations for a usable full covariance.", stacklevel=2)
    return dataset.covariance(kind=kind, of_mean=of_mean), kind


def _fix_unvaried(basis, settings: dict, vary, path) -> dict:
    """Fix the emulated parameters not in ``vary`` at the Quijote fiducial; return the bounds of the others."""
    unknown = set(vary) - set(settings["vary"])
    if unknown:
        raise ValueError(f"emulator {path} varies {settings['vary']}, not {sorted(unknown)}")
    params = get_params(basis)
    for name in set(settings["vary"]) - set(vary):
        params[name].update(value=QUIJOTE_COSMOLOGY[name], fixed=True)
    return {name: bounds for name, bounds in settings["bounds"].items() if name in vary}


def load_emulated_basis(path: Path, dataset: WSTDataset, vary):
    """Trained Taylor emulator of WSTBasis, checked against the data it will be compared with.

    ``vary`` may be a subset of the emulated parameters: the others are fixed at the Quijote fiducial.
    """
    from desilike.emulators import Emulator

    settings = json.loads(Path(path).with_suffix(".json").read_text())
    if settings.get("stat", "wst") != "wst":
        raise ValueError(f"emulator {path} is for {settings['stat']}, not the WST")
    # The basis does not depend on q, which only enters the exact assembly.
    trained = WSTConfig(**settings["config"]).with_q(dataset.config.q)
    if trained != dataset.config:
        raise ValueError(f"emulator {path} was trained for {trained}, the data have {dataset.config}")
    for key, value in dict(z=dataset.metadata["redshift"], shotnoise=dataset.shotnoise).items():
        if not np.isclose(settings[key], value):
            raise ValueError(f"emulator {path} was trained with {key}={settings[key]}, the data have {value}")
    basis = Emulator.read(str(path)).to_calculator()
    return basis, _fix_unvaried(basis, settings, vary, path)


def build_likelihood(dataset: WSTDataset, covariance: np.ndarray, vary=("omega_cdm", "logA"),
                     emulator: Path | None = None) -> WSTLikelihood:
    """WST likelihood with an exact (CLASS) or Taylor-emulated cosmology-dependent basis."""
    if emulator is None:
        basis = WSTBasis(cosmo=build_cosmology(vary), config=dataset.config, z=dataset.metadata["redshift"],
                         shotnoise=dataset.shotnoise)
    else:
        basis, bounds = load_emulated_basis(emulator, dataset, vary)
        _bound_to_emulator(basis, bounds)
    theory = WSTTheory(dataset.coefficients, config=dataset.config, basis=basis)
    return WSTLikelihood(theory, dataset.mean, covariance)


def _bound_to_emulator(basis, bounds):
    """Keep the chains inside the emulated box."""
    params = get_params(basis)
    for name, (low, high) in bounds.items():
        params[name].update(prior=dict(limits=[low, high]))


def load_emulated_power_basis(path: Path, dataset: PowerDataset, vary, order: str, cutoff: float | None):
    """Trained Taylor emulator of PowerBasis, checked against the data and model it will be used with.

    Returns the emulated basis, its emulation bounds and the k nodes it was trained on. ``vary`` may be a
    subset of the emulated parameters: the others are fixed at the Quijote fiducial.
    """
    from desilike.emulators import Emulator

    settings = json.loads(Path(path).with_suffix(".json").read_text())
    if settings.get("stat") != "pk":
        raise ValueError(f"emulator {path} is not a P(k) emulator")
    knodes = np.asarray(settings["knodes"])
    checks = dict(order=(settings["order"] == order), cutoff=(settings["cutoff"] == cutoff),
                  z=np.isclose(settings["z"], dataset.metadata["redshift"]), kmax=knodes[-1] >= dataset.edges[-1])
    for key, ok in checks.items():
        if not ok:
            raise ValueError(f"emulator {path} does not match the requested {key} "
                             f"(trained: order={settings['order']}, cutoff={settings['cutoff']}, z={settings['z']}, "
                             f"vary={settings['vary']}, kmax={knodes[-1]:.3f})")
    basis = Emulator.read(str(path)).to_calculator()
    return basis, _fix_unvaried(basis, settings, vary, path), knodes


def build_power_likelihood(dataset: PowerDataset, covariance: np.ndarray, vary=("omega_cdm", "logA"),
                           order: str = "one-loop", cutoff: float | None = 0.5, emulator: Path | None = None,
                           counterterm: bool | None = None, window: str | None = "cic") -> WSTLikelihood:
    """Real-space matter P(k) likelihood with an exact (CLASS + loop) or Taylor-emulated basis."""
    if emulator is None:
        knodes = default_knodes(dataset.edges[-1], boxsize=dataset.metadata["boxsize"])
        basis = PowerBasis(cosmo=build_cosmology(vary), knodes=knodes, z=dataset.metadata["redshift"], order=order,
                           cutoff=cutoff)
    else:
        basis, bounds, knodes = load_emulated_power_basis(emulator, dataset, vary, order, cutoff)
        _bound_to_emulator(basis, bounds)
    binning = LatticeBinning(dataset.edges, dataset.metadata["boxsize"], dataset.metadata["nmesh"], knodes,
                             window=window)
    theory = PowerTheory(binning, basis=basis, shotnoise=dataset.shotnoise, order=order, counterterm=counterterm)
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
              gelman_rubin: float = 1.03, ess: float = 300.0, check_every: int = 500, proposal=None):
    """Metropolis-Hastings chains started at the best fit, with the profile covariance as proposal.

    ``proposal`` (samples of an earlier run of the same posterior) replaces the profile covariance by the
    covariance of those chains (first 30% dropped as burn-in), for posteriors where Minuit's Hessian is a poor
    proposal.
    """
    from desilike.conditioning import AffineConditioner
    from desilike.samplers import MH, Sampler
    from desilike.samplers.proposals import GaussianProposal

    graph = build(posterior)
    varied = graph.params.select(varied=True, derived=False)
    if proposal is None:
        covariance = profiles.covariance.select(varied)
    else:
        covariance = proposal.remove_burnin(0.3).covariance().select(varied)
    center = {name: value for name, value in bestfit_values(profiles).items() if name in varied.names()}
    ndim = len(varied)
    # desilike's MH multiplies the proposal by 2.38^2 / sqrt(ndim); this gives 2.38^2 / ndim in whitened units.
    sampler = Sampler(graph, kernel=MH(covariance=np.eye(ndim) / np.sqrt(ndim)), nparallel=chains, rng=seed,
                      output_dir=output_dir, conditioner=AffineConditioner(covariance=covariance, rescale="full"),
                      proposal=GaussianProposal(covariance, center=center))
    return sampler.run(gelman_rubin=gelman_rubin, ess=ess, min_steps=check_every, check_every=check_every,
                       max_steps=max_steps)


def summarize(likelihood, profiles, samples, dataset: WSTDataset | PowerDataset, covariance_kind: str) -> dict:
    best = bestfit_values(profiles)
    build(likelihood)({name: value for name, value in best.items()
                       if name in get_params(likelihood).select(varied=True, derived=False).names()})
    residual = np.asarray(likelihood.flatdata) - np.asarray(likelihood.flattheory)
    chi2 = float(residual @ np.asarray(likelihood.precision) @ residual)
    nvaried = len(get_params(likelihood).select(varied=True, derived=False))
    summary = dict(bestfit=best, chi2=chi2, ndata=int(residual.size), nvaried=nvaried,
                   nrealizations=int(dataset.vectors.shape[0]), covariance=covariance_kind)
    if isinstance(dataset, PowerDataset):
        summary["k"] = dataset.k.tolist()
    else:
        summary["coefficients"] = [c.label for c in dataset.coefficients]
    if samples is not None:
        names = [name for name in best if name in samples]
        summary["posterior"] = {name: dict(mean=float(np.asarray(samples.mean(name))),
                                           std=float(np.asarray(samples.std(name)))) for name in names}
    return summary


def plot_fit(path: Path, likelihood, dataset: WSTDataset, covariance: np.ndarray, config: WSTConfig,
             bestfit: dict | None = None, nvaried: int | None = None, title: str = ""):
    """Data versus the best-fit model, S1 and S2/S1 side by side, grouped by l, with (data - model) / sigma below.

    ``likelihood`` must have been evaluated at the best fit; sigma is the square root of the diagonal of
    ``covariance`` (the errors of the fitted volume). The chi^2 uses the full covariance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, model = np.asarray(likelihood.flatdata), np.asarray(likelihood.flattheory)
    error = np.sqrt(np.diag(covariance))
    residual = data - model
    chi2 = float(residual @ np.linalg.solve(covariance, residual))
    coefficients = dataset.coefficients
    kinds = [kind for kind in ("S1", "S21") if any(c.kind == kind for c in coefficients)]
    counts = [sum(c.kind == kind for c in coefficients) for kind in kinds]
    fig, axes = plt.subplots(2, len(kinds), figsize=(7 + 0.25 * len(coefficients), 6.5), sharex="col", squeeze=False,
                             gridspec_kw=dict(height_ratios=(2.2, 1), width_ratios=counts))
    for col, kind in enumerate(kinds):
        index = np.array([i for i, c in enumerate(coefficients) if c.kind == kind])
        ells = np.array([coefficients[i].ell for i in index])
        x = np.arange(index.size)
        ax, axr = axes[0, col], axes[1, col]
        for ell in np.unique(ells):
            mask = ells == ell
            line = ax.errorbar(x[mask], data[index[mask]], error[index[mask]], fmt="o", ms=4, label=f"l = {ell}")
            color = line[0].get_color()
            ax.plot(x[mask], model[index[mask]], "-", color=color, lw=1.2, alpha=0.8)
            axr.plot(x[mask], residual[index[mask]] / error[index[mask]], "o", ms=4, color=color)
        ax.plot([], [], "k-", lw=1.2, label="best-fit model")
        ax.set_yscale("log")
        ax.legend(fontsize=9, ncol=2)
        axr.axhspan(-1, 1, color="0.9")
        axr.axhline(0, color="k", lw=0.8)
        axr.set_ylim(-3, 3)
        if kind == "S1":
            ticks = [f"{config.sigma(coefficients[i].j):.0f}" for i in index]
            ax.set_title(r"$S_1(j, l)$")
            axr.set_xlabel(r"$\sigma_j$ [Mpc/h]  (grouped by $l$)")
        else:
            ticks = [f"{config.sigma(coefficients[i].j):.0f}→{config.sigma(coefficients[i].j2):.0f}" for i in index]
            ax.set_title(r"$S_2(j_1, j_2, l)\,/\,S_1(j_1, l)$")
            axr.set_xlabel(r"$\sigma_{j_1}\to\sigma_{j_2}$ [Mpc/h]  (grouped by $l$)")
        axr.set_xticks(x, ticks, rotation=60, fontsize=8)
    axes[0, 0].set_ylabel("coefficient")
    axes[1, 0].set_ylabel(r"(data − model) / $\sigma$")
    header = title or f"WST best fit, q = {config.q}"
    dof = f"{data.size} − {nvaried} = {data.size - nvaried} dof" if nvaried else f"{data.size} data points"
    summary = f"$\\chi^2$ = {chi2:.1f} for {dof} (full covariance)"
    if bestfit:
        latex = {"omega_cdm": r"$\omega_{\rm cdm}$", "logA": r"$\ln 10^{10}A_s$", "n_s": "$n_s$", "h": "$h$",
                 "omega_b": r"$\omega_{\rm b}$"}
        summary += ";  best fit: " + ", ".join(f"{label}={bestfit[name]:.4f}" for name, label in latex.items()
                                               if name in bestfit)
    fig.suptitle(f"{header}\n{summary}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_power_fit(path: Path, likelihood, dataset: PowerDataset, covariance: np.ndarray, title: str = ""):
    """k P(k) of the data versus the best-fit theory, and residuals in units of the error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, theory = np.asarray(likelihood.flatdata), np.asarray(likelihood.flattheory)
    error, k = np.sqrt(np.diag(covariance)), dataset.k
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw=dict(height_ratios=(2, 1)))
    axes[0].errorbar(k, k * data, k * error, fmt="o", ms=3, label="data")
    axes[0].plot(k, k * theory, "-", label="best fit")
    axes[0].set_ylabel(r"$k P(k)$ [$(\mathrm{Mpc}/h)^2$]")
    axes[0].legend()
    axes[1].axhspan(-1, 1, color="0.9")
    axes[1].plot(k, (data - theory) / error, "o", ms=3)
    axes[1].set_ylabel(r"$\Delta / \sigma$")
    axes[1].set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
