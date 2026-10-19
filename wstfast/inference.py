"""Likelihoods, profiling and sampling with desilike.

The fits of the model (the emulated ``Basis`` of a tracer in redshift or real space) are built by ``build_fit`` from a
``FitOptions`` (scripts/fit.py is its command line). The builders of the former real-space matter model
(``build_wst_theory``, ``build_power_theory``, ``build_joint_likelihood``) remain until it is retired.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
from desilike import build, get_params

from .calculators import (APGeometry, CombinedCaps, JointTheory, LegacyPowerTheory, LinearCompression,
                          NuisanceParameters, PowerBasis, PowerTheory, S1mTheory, S21mTheory, WSTBasis, WSTLikelihood,
                          WSTTheory, build_cosmology, combine_caps)
from .config import QUIJOTE_COSMOLOGY, Coefficient, ModelSpec, WSTConfig, select_coefficients
from .data import (PowerDataset, WSTDataset, chi_squared, hartlap_factor, load_band_dataset, load_measurement,
                   load_power_dataset, load_s1m_dataset, load_s21m_dataset, percival_m1, sample_covariance)
from .theory.power import LatticeBinning, default_knodes
from .theory.spectrum import (APPolynomial, BandProjection, MultipoleProjection, S1mProjection, SpectrumGrid,
                              multipole_ap_projection, s1m_ap_projection)


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


def build_wst_theory(dataset: WSTDataset, vary=("omega_cdm", "logA"), emulator: Path | None = None,
                     layer2: bool = False) -> WSTTheory:
    """WST theory with an exact (CLASS) or Taylor-emulated cosmology-dependent basis."""
    if emulator is None:
        basis = WSTBasis(cosmo=build_cosmology(vary), config=dataset.config, z=dataset.metadata["redshift"],
                         shotnoise=dataset.shotnoise)
    else:
        basis, bounds = load_emulated_basis(emulator, dataset, vary)
        _bound_to_emulator(basis, bounds)
    return WSTTheory(dataset.coefficients, config=dataset.config, basis=basis, layer2=layer2)


def build_likelihood(dataset: WSTDataset, covariance: np.ndarray, vary=("omega_cdm", "logA"),
                     emulator: Path | None = None, layer2: bool = False) -> WSTLikelihood:
    """WST likelihood with an exact (CLASS) or Taylor-emulated cosmology-dependent basis."""
    return WSTLikelihood(build_wst_theory(dataset, vary, emulator, layer2=layer2), dataset.mean, covariance)


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


def build_power_theory(dataset: PowerDataset, vary=("omega_cdm", "logA"), order: str = "one-loop",
                       cutoff: float | None = 0.5, emulator: Path | None = None, counterterm: bool | None = None,
                       window: str | None = "cic") -> LegacyPowerTheory:
    """Real-space matter P(k) theory with an exact (CLASS + loop) or Taylor-emulated basis."""
    if emulator is None:
        knodes = default_knodes(dataset.edges[-1], boxsize=dataset.metadata["boxsize"])
        basis = PowerBasis(cosmo=build_cosmology(vary), knodes=knodes, z=dataset.metadata["redshift"], order=order,
                           cutoff=cutoff)
    else:
        basis, bounds, knodes = load_emulated_power_basis(emulator, dataset, vary, order, cutoff)
        _bound_to_emulator(basis, bounds)
    binning = LatticeBinning(dataset.edges, dataset.metadata["boxsize"], dataset.metadata["nmesh"], knodes,
                             window=window)
    return LegacyPowerTheory(binning, basis=basis, shotnoise=dataset.shotnoise, order=order, counterterm=counterterm)


def build_power_likelihood(dataset: PowerDataset, covariance: np.ndarray, vary=("omega_cdm", "logA"),
                           order: str = "one-loop", cutoff: float | None = 0.5, emulator: Path | None = None,
                           counterterm: bool | None = None, window: str | None = "cic") -> WSTLikelihood:
    """Real-space matter P(k) likelihood with an exact (CLASS + loop) or Taylor-emulated basis."""
    theory = build_power_theory(dataset, vary, order=order, cutoff=cutoff, emulator=emulator,
                                counterterm=counterterm, window=window)
    return WSTLikelihood(theory, dataset.mean, covariance)


def joint_vectors(wst: WSTDataset, power: PowerDataset) -> np.ndarray:
    """Concatenated data vectors [WST, P(k)] of the same realizations."""
    if list(wst.files) != list(power.files):
        raise ValueError("the WST and P(k) datasets must hold the same realizations in the same order")
    return np.concatenate([wst.vectors, power.vectors], axis=1)


def build_joint_likelihood(wst: WSTDataset, power: PowerDataset, covariance: np.ndarray, vary=("omega_cdm", "logA"),
                           wst_emulator: Path | None = None, power_emulator: Path | None = None,
                           order: str = "one-loop", cutoff: float | None = 0.5, layer2: bool = False) -> WSTLikelihood:
    """Joint WST + P(k) likelihood with shared cosmology; ``covariance`` is that of ``joint_vectors``."""
    theory = JointTheory([build_wst_theory(wst, vary, wst_emulator, layer2=layer2),
                          build_power_theory(power, vary, order=order, cutoff=cutoff, emulator=power_emulator)])
    return WSTLikelihood(theory, joint_vectors(wst, power).mean(axis=0), covariance)


def fix_parameters(likelihood, values: dict[str, float]):
    """Fix the named parameters of ``likelihood`` at the given values (e.g. nuisances, for bounds on information)."""
    params = get_params(likelihood)
    unknown = sorted(set(values) - set(params.names()))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}")
    for name, value in values.items():
        params[name].update(value=float(value), fixed=True)


def parse_fixed(items) -> dict[str, float]:
    """['a=1', 'b=2.5'] -> {'a': 1.0, 'b': 2.5}."""
    out = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if not value:
            raise ValueError(f"--fix expects NAME=VALUE, got {item!r}")
        out[name] = float(value)
    return out


def profile(posterior, output: Path, seed: int = 42, nstarts: int = 4, start: dict | None = None):
    """Minuit maximisation from several starting points; the best is refined and saved. ``start``: values of (some
    of) the varied parameters, e.g. another fit's best fit, added as one more starting point (the others at the
    reference)."""
    from desilike.profilers import Minuit, Profiler

    graph = build(posterior)
    names = graph.params.select(varied=True, derived=False).names()
    candidates = Profiler(graph, kernel=Minuit(), rng=seed).maximize(niterations=nstarts)
    best = candidates.choice(index=int(np.argmax(np.asarray(candidates.logpdf))), squeeze=True)
    logbest = float(np.max(np.asarray(candidates.logpdf)))
    if start is not None:
        given = {name: float(start[name]) for name in names if name in start}
        point = np.array([given.get(name, float(np.asarray(best.best[name]).reshape(-1)[-1])) for name in names])
        given_run = Profiler(graph, kernel=Minuit(), rng=seed).maximize(niterations=1, start=point)
        if float(np.max(np.asarray(given_run.logpdf))) > logbest:
            best = given_run.choice(index=0, squeeze=True)
    start = np.concatenate([np.asarray(best.best[name], dtype="f8").reshape(-1) for name in names])
    profiles = Profiler(graph, kernel=Minuit(), rng=seed).maximize(niterations=1, start=start)
    profiles.write(str(output))
    return profiles


def bestfit_values(profiles) -> dict[str, float]:
    chosen = profiles.choice(squeeze=True)
    return {str(name): float(np.asarray(value).reshape(-1)[-1]) for name, value in chosen.best.items()}


def hessian_covariance(graph, center: dict, varied):
    """Inverse of minus the exact (JAX) Hessian of the compiled log-posterior ``graph`` at ``center``, as a desilike
    Covariance (``graph`` is reused: building the posterior again would invalidate it).

    Minuit's profile covariance can underestimate the posterior widths of these fits several-fold (2-19x on the
    halo WST), which makes it a poor proposal / whitening; the exact Hessian at the best fit does not.
    Non-positive curvature directions are clipped.
    """
    import jax
    import jax.numpy as jnp
    from desilike.samples.covariance import Covariance

    names = varied.names()
    point = jnp.asarray([center[name] for name in names], dtype="f8")
    hessian = np.asarray(jax.hessian(lambda x: graph(dict(zip(names, x))))(point))
    w, v = np.linalg.eigh(-hessian)
    w = np.clip(w, w.max() * 1e-12, None)
    return Covariance((v / w) @ v.T, params=list(varied))


def sample_nuts(posterior, profiles, output_dir: Path, chains: int = 4, seed: int = 42, max_steps: int = 20000,
                gelman_rubin: float = 1.03, ess: float = 300.0, check_every: int = 200, warmup: int = 1000,
                dense: bool = True, proposal=None):
    """NUTS chains (BlackJAX via desilike) started around the best fit.

    The parameters are whitened by the covariance of ``proposal`` (samples of an earlier run) or else by the exact
    Hessian at the best fit (``hessian_covariance``); the step size and a dense (or diagonal) mass matrix are then
    adapted over ``warmup`` steps. The halo WST posteriors that took hours with MH converge in a few minutes.
    """
    from desilike.conditioning import AffineConditioner
    from desilike.samplers import BlackjaxNUTS, Sampler
    from desilike.samplers.proposals import GaussianProposal

    graph = build(posterior)
    varied = graph.params.select(varied=True, derived=False)
    center = {name: value for name, value in bestfit_values(profiles).items() if name in varied.names()}
    if proposal is None:
        covariance = hessian_covariance(graph, center, varied)
    else:
        covariance = proposal.remove_burnin(0.3).covariance().select(varied)
    sampler = Sampler(graph, kernel=BlackjaxNUTS(), nparallel=chains, rng=seed, output_dir=output_dir,
                      conditioner=AffineConditioner(covariance=covariance, rescale="full"),
                      proposal=GaussianProposal(covariance, center=center))
    adaptation = dict(steps=warmup, **({"is_mass_matrix_diagonal": False} if dense else {}))
    return sampler.run(gelman_rubin=gelman_rubin, ess=ess, min_steps=check_every, check_every=check_every,
                       max_steps=max_steps, adaptation=adaptation)


def sample_mh(posterior, profiles, output_dir: Path, chains: int = 4, seed: int = 42, max_steps: int = 50000,
              gelman_rubin: float = 1.03, ess: float = 300.0, check_every: int = 500, proposal=None,
              adaptation_steps: int = 10000):
    """Metropolis-Hastings chains started at the best fit, with the profile covariance as proposal.

    ``proposal`` (samples of an earlier run of the same posterior) replaces the profile covariance by the
    covariance of those chains (first 30% dropped as burn-in), for posteriors where Minuit's Hessian is a poor
    proposal. The proposal covariance is learnt from the chains over the first ``adaptation_steps`` steps
    (0: never; desilike's MH only adapts when asked to).
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
                       max_steps=max_steps, adaptation={"steps": adaptation_steps} if adaptation_steps else None)


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
    chi2 = chi_squared(residual, covariance)
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


def plot_triangle(path: Path, runs: dict, params=("omega_cdm", "logA", "h"), truth: dict | None = None,
                  title: str = ""):
    """Triangle plot of the posteriors of ``runs`` ({legend: samples.h5 of fit.py, or a list of chain files, e.g.
    the chains/samples_<i>.h5 of a run in progress}) in ``params``, with the ``truth`` marked; the chains' first half
    is dropped as burn-in."""
    import matplotlib

    matplotlib.use("Agg")
    from desilike.samples import Samples
    from getdist import MCSamples, plots

    latex = {"omega_cdm": r"\omega_{\rm cdm}", "logA": r"\ln(10^{10}A_s)", "h": "h", "n_s": "n_s", "b1": "b_1",
             "c_v": "c_v"}
    gd = []
    for legend, file in runs.items():
        chains = []
        for f in (file if isinstance(file, (list, tuple)) else [file]):
            read = Samples.read(str(f))
            chains += read if isinstance(read, list) else [read]
        values = [np.concatenate([np.asarray(c[name])[len(np.asarray(c[name])) // 2:] for c in chains]) for name in params]
        gd.append(MCSamples(samples=np.column_stack(values), names=list(params),
                            labels=[latex.get(n, n) for n in params], label=legend))
    plotter = plots.get_subplot_plotter(width_inch=6)
    plotter.triangle_plot(gd, list(params), filled=True, markers=truth or {}, title_limit=1)
    if title:
        plotter.fig.suptitle(title, y=1.02)
    plotter.export(str(path))

# ---------------------------------------------------------------------------------------------------------------------
# Fits of the line-of-sight-resolved WST and P(k) of a tracer (the emulated Basis; scripts/fit.py)
# ---------------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class FitOptions:
    """What a fit of the WST and P(k) measurements includes (``build_fit``; the command line of scripts/fit.py).

    The model is that of the emulator (its ``ModelSpec``: tracer, real or redshift space, Fingers-of-God, AP order,
    cubic biases); these options select the data, the nuisance priors and the covariance. Data: ``stats`` among
    "pk", "s1m", "s21m" and "band" (the m-summed first-order coefficients at q = 2, the band powers of the wavelet
    filters, on their own scale cut ``band_min_scale`` and multipoles ``band_ells``; they need only the spectrum, so
    they reach smaller scales than S1m at q < 2) from the box measurements under ``data_dir`` (subdirectory ``rsd``, or ``real`` for a real-space
    emulator), with the scale cuts ``s1_min_scale``, ``s1_ells``, ``s21_min_scale``, ``s21_min_scale2``,
    ``s21_min_ratio`` and ``kmin``, ``kmax``, ``rebin`` of P(k); several exponents ``q`` concatenate the S1m vectors.
    Survey fits: ``survey`` (caps), ``window`` (one ``survey.SurveyWindow`` file per cap), ``combine_caps``.
    """

    emulator: Path
    data_dir: Path = Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los")
    stats: tuple = ("pk", "s1m")
    q: tuple = (0.8,)
    s1_min_scale: float = 25.0
    s1_ells: tuple = (0, 1, 2, 3, 4)
    s21_min_scale: float = 17.6
    s21_min_scale2: float = 70.0
    s21_min_ratio: float = 2.8
    band_min_scale: float = 12.5
    #: Largest sigma_j of the band powers: below the S1m cut, a block never carries both exponents (their ratio would
    #: measure the Edgeworth factor to a precision the model does not have).
    band_max_scale: float = 1e9
    band_ells: tuple = (0, 1, 2, 3, 4)
    #: Keep only the bands whose 84th-percentile wavenumber (of the weight w_{jl}(k) k^2, a Gamma distribution in
    #: sigma^2 k^2) lies below this [h/Mpc]: the k range of the band vector, comparable to a kmax of P(k).
    band_kmax: float | None = None
    #: Counterterm of the band powers: "shared" (the S1m c0, the same spectrum) or "separate" (its own, c0_band).
    band_counterterm: str = "shared"
    #: The band powers are smooth functionals of the spectrum, numerically low-rank on the lattice: their sample
    #: correlation has eigenvalues down to the rounding floor (1e-14). ``band_cut`` keeps the principal components of the
    #: band block with eigenvalue (of the correlation) above it (``band_compression``); the dropped directions demand a
    #: model accuracy far beyond the emulator's (sqrt of the eigenvalue, relative).
    band_cut: float = 1e-4
    kmin: float = 0.0
    kmax: float = 0.12
    rebin: int = 2
    vary: tuple = ("omega_cdm", "logA")
    #: Errors of a survey of this volume [(Gpc/h)^3]: the one-box covariance / (V / V_box).
    volume: float | None = None
    #: Errors of the realization mean.
    covariance_of_mean: bool = False
    survey: tuple | None = None
    window: tuple | None = None
    combine_caps: bool = False
    #: Survey: analytic covariance (analytic_covariance_rsd_wst.py --survey) instead of box realizations.
    covariance_file: Path | None = None
    #: npz files of modelling errors (labels, covariance in data units or fractional) added as they are.
    extra_covariance: tuple = ()
    no_percival: bool = False
    #: Estimate the covariance from the realizations of this directory (the data are still those of data_dir).
    covariance_data_dir: Path | None = None
    #: Use only the realizations of data_dir that also exist in this directory (paired seeds).
    realizations_like: Path | None = None
    #: Leave out the realizations of data_dir that exist in this directory (held-out seeds).
    exclude_like: Path | None = None
    #: Parameters to fix, ["NAME=VALUE", ...].
    fix: tuple | None = None
    #: Sobol points of the control-variate correction of the S1m cumulants (0: none).
    cumulant_points: int = 2**20
    #: Gaussian priors on emulated cosmological parameters, ["NAME=LOC,SCALE", ...] (truncated to the emulator box).
    gaussian_prior: tuple = ()
    #: Free white term W_j of the S21m modulus spectra per first-layer scale.
    s21_white: bool = False
    #: Layer-2 template (scripts/layer2/template_rsd.py): S21m x (1 + d0 + R_l d2), R_l = 1 / (1 + c_l n s_lin^2).
    s21_layer2: Path | None = None
    #: c_l calibrated on the halo pairs outside the data vector ("halo") or transferred from matter.
    s21_layer2_c: str = "halo"
    #: Marginalize analytically over the template amplitudes: one pair ("global") or one pair per l ("per_l").
    s21_layer2_marg: str | None = None
    s21_layer2_marg_width: float = 1.0
    #: Cubic biases of the S1m trispectrum: priors centred on zero ("zero") or on the halo relations ("relation").
    cubic_biases: str | None = None
    #: Alcock-Paczynski distortions of the trial cosmology (redshift space).
    ap: bool = False
    #: Real space: average the |m| blocks of S1m and S21m within their equally distributed classes (``m_compression``;
    #: the noise amplitudes of the |m| >= 1 first-layer fields are then tied, ``NuisanceParameters(tie_m=True)``),
    #: so that the data vector is as short as the isotropic field allows (the Hartlap and Percival factors shrink).
    compress_m: bool = False
    #: The configuration of the data (set by build_problem; the layer-2 template is re-keyed to it).
    layer2_config: WSTConfig | None = None


def load_basis(path: Path, vary, prepare=None):
    """Emulated ``Basis`` as a calculator and its settings; ``prepare(emulator, settings)`` acts on the emulator first
    (e.g. ``contract_s21m``), and its result is kept as settings["prepared"]. The emulated parameters not in ``vary``
    are fixed at the fiducial."""
    from desilike.emulators import Emulator

    settings = json.loads(path.with_suffix(".json").read_text())
    if settings.get("stat") != "rsd":
        raise ValueError(f"{path} is not an emulator of the Basis (train_emulator.py --stat rsd)")
    if settings.get("tracer", "matter") == "matter" and settings.get("layout") != "monomial":
        raise SystemExit(f"{path} holds the retired matter-only basis layout: retrain it (train_emulator.py --stat rsd "
                         "--tracer matter now trains the monomial basis of the matter limit)")
    emulator = Emulator.read(str(path))
    if prepare is not None:
        settings["prepared"] = prepare(emulator, settings)
    basis = emulator.to_calculator()
    _bound_to_emulator(basis, _fix_unvaried(basis, settings, vary, path))
    return basis, settings


def exact_basis(settings: dict, vary):
    """The exact (CLASS) ``Basis`` an emulator with ``settings`` was trained on, varying ``vary`` (which may include
    parameters no emulator has, e.g. ``m_ncdm``); about 20 s per cosmology."""
    from .calculators import Basis, build_cosmology

    config = WSTConfig(**settings["config"])
    coefficients = [c for c in select_coefficients(config, s1_min_scale=settings["s1_min_scale"])
                    if c.kind == "S1" and c.label in settings["coefficients"]]
    s21 = [c for c in select_coefficients(config, s21_min_scale=0.0, s21_min_ratio=0.0)
           if c.kind == "S21" and c.label in settings.get("s21_coefficients", [])]
    s21.sort(key=lambda c: settings["s21_coefficients"].index(c.label))
    return Basis(cosmo=build_cosmology(tuple(vary), fiducial=settings.get("fiducial", "quijote")), config=config,
                 coefficients=coefficients, z=settings["z"], kmax=settings["kmax"], ir=settings["ir"],
                 s21_coefficients=s21, tracer=settings.get("tracer", "matter"),
                 space=ModelSpec.from_settings(settings).space, s21_damping=settings.get("s21_damping"),
                 cubic_biases=settings.get("cubic_biases", False), fog=settings.get("fog", False),
                 s21_lattice=tuple(settings.get("lattice", (1000.0, 256))), s21_cellsize=settings.get("s21_cellsize", 0.0),
                 s21_projected=settings.get("s21_projected", True), ap_order=settings.get("ap_order", 0))


def s21m_selection(options, config, settings):
    """S21 coefficients of the fit and those of the emulator (in its order)."""
    coefficients = [c for c in select_coefficients(config, s21_min_scale=options.s21_min_scale,
                                                   s21_min_ratio=options.s21_min_ratio,
                                                   s21_min_scale2=options.s21_min_scale2) if c.kind == "S21"]
    emulated21 = [c for c in select_coefficients(config, s21_min_scale=0.0, s21_min_ratio=0.0)
                  if c.kind == "S21" and c.label in settings.get("s21_coefficients", [])]
    emulated21.sort(key=lambda c: settings["s21_coefficients"].index(c.label))
    return coefficients, emulated21


def contract_unprojected_s21m(options, config, emulator, settings):
    """For an emulator of the unprojected S21m basis and an S21m fit: fold the S21m projection (its AP polynomial with
    ``ap``) into the emulator (``contract_s21m``), so that a likelihood call costs what a projected emulator does."""
    from numpy.polynomial.legendre import leggauss

    from wstfast.calculators import contract_s21m
    from wstfast.theory.second_layer import S21mAPProjection, S21mProjection, first_layer_fields

    if settings.get("s21_projected", True) or "s21m" not in options.stats:
        return None
    coefficients, emulated21 = s21m_selection(options, config, settings)
    if any(c not in emulated21 for c in coefficients):
        return None  # build_problem reports the missing terms
    x, _ = leggauss(12)  # the (K, mu_K) grid of S21mBasis
    boxsize, nmesh = settings.get("lattice", [1000.0, 256])
    grid = dict(k=np.geomspace(4e-3, 0.12, 24), mu=x[6:], boxsize=float(boxsize), nmesh=int(nmesh))
    fields = first_layer_fields(coefficients)
    projection = (APPolynomial(S21mAPProjection(config, coefficients, fields, grid["k"], grid["mu"],
                                                boxsize=grid["boxsize"], nmesh=grid["nmesh"]))
                  if options.ap else S21mProjection(config, coefficients, fields, grid["k"], grid["mu"],
                                                 boxsize=grid["boxsize"], nmesh=grid["nmesh"]))
    return contract_s21m(emulator, coefficients, emulated21, projection)


def apply_gaussian_priors(basis, settings, items):
    """Gaussian priors ['n_s=0.9649,0.042', ...] on varied emulated parameters, within the emulator bounds."""
    params = get_params(basis)
    for item in items:
        name, _, values = item.partition("=")
        loc, scale = (float(v) for v in values.split(","))
        if name not in settings["vary"] or params[name].fixed:
            raise ValueError(f"gaussian_prior: {name} is not a varied emulated parameter")
        params[name].update(prior=dict(dist="norm", loc=loc, scale=scale, limits=list(settings["bounds"][name])))


def load_correction(options, config, coefficients, settings):
    """Control-variate correction of the emulated S1m cumulants at the Quijote cosmology (cached next to the emulator)."""
    from cosmoprimo import Cosmology

    from wstfast.theory.first_layer import cumulant_correction as cumulant_correction

    cache = options.emulator.with_suffix(f".correction_{options.cumulant_points}.npz")
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        labels = list(stored["labels"])
        index = [labels.index(c.label) for c in coefficients]
        return _tile_ap(settings, stored["kappa"][index], stored["skewness2"][index])
    every = [c for c in select_coefficients(config, s1_min_scale=settings["s1_min_scale"]) if c.kind == "S1"]
    fourier = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY).get_fourier()
    klin = np.geomspace(1e-4, 10.0, 1024)
    pklin = fourier.pk_interpolator(of="delta_m")(klin, z=settings["z"])
    f = float(fourier.sigma8_z(settings["z"], of="theta_cb") / fourier.sigma8_z(settings["z"], of="delta_cb"))
    spec = ModelSpec.from_settings(settings)
    cubic = dict(cubic=True) if spec.cubic else {}
    if spec.fog:
        cubic["fog"] = True
    kappa, skewness2 = cumulant_correction(config, every, klin, pklin, f, npoints_reference=options.cumulant_points, **cubic)
    np.savez(cache, kappa=kappa, skewness2=skewness2, labels=np.array([c.label for c in every]))
    print(f"wrote {cache}")
    labels = [c.label for c in every]
    index = [labels.index(c.label) for c in coefficients]
    return _tile_ap(settings, kappa[index], skewness2[index])


def _tile_ap(settings, kappa, skewness2):
    """The control-variate correction (computed without AP) for every AP-polynomial term of an ``ap_order`` emulator."""
    from wstfast.theory.layout import AP_TERMS

    n = AP_TERMS[ModelSpec.from_settings(settings).ap_order]
    return np.tile(kappa, n), np.tile(skewness2, n)


def load_layer2(options, projection, coefficients, first_layer, q, kwargs):
    """Template of the non-Gaussian second layer for S21mTheory(layer2=...): d0 and d2 per (j1, j2, l, |m1|, |m2|),
    the matter linear variances of the first-layer fields, the c_l (halo-calibrated or transferred from matter), and the
    fiducial linear block variances at the template's biases, from the basis evaluated at the Quijote cosmology."""
    from desilike import build

    from wstfast.calculators import S21mTheory, basis_rows
    from wstfast.theory.bias import MATTER_BETA, beta_vector, monomials
    from wstfast.theory.layout import TREE as TREE_INDEX

    template = layer2_template(options)
    rows = {tuple(r["key"]): (r["gauss"], r["tree"]) for r in template["rows"]}
    var = {(r["key"][0], r["key"][2]): r["var"] for r in template["rows"]}
    which = options.s21_layer2_c
    c = {int(k): v for k, v in template[f"{which}_c"].items()}
    c_err = {int(k): v for k, v in template.get(f"{which}_c_err", {}).items()} or {ell: 0.1 * abs(v) for ell, v in c.items()}
    b = template["biases"]
    beta = (MATTER_BETA if kwargs["nuisance"].spec.tracer == "matter"
            else np.asarray(beta_vector(b["b1"], b["b2"], b["bG2"], b["bGamma3"])))
    probe = S21mTheory(projection, coefficients, first_layer, q=q, **kwargs)
    graph = build(probe)
    fiducial = {n: QUIJOTE_COSMOLOGY[n] for n in get_params(probe).select(varied=True, derived=False).names()
                if n in QUIJOTE_COSMOLOGY}
    graph(fiducial)
    rows_fid = basis_rows(probe.basis)[0]
    tree = np.tensordot(np.asarray(monomials(beta, 2))[np.array(TREE_INDEX)], np.asarray(rows_fid[:len(TREE_INDEX)]), axes=1)
    blocks = np.asarray(probe.matrix) @ tree.ravel()
    offsets = np.cumsum([0] + [f.ell + 1 for f in first_layer])
    lin_ref = {(f.j, f.ell, m1): float(blocks[offsets[i] + m1]) for i, f in enumerate(first_layer) for m1 in range(f.ell + 1)}
    print(f"non-Gaussian second layer: {which} c_l = " + ", ".join(f"{ell}: {v:.2f}" for ell, v in sorted(c.items())),
          flush=True)
    return dict(rows=rows, var=var, c=c, c_err=c_err, beta_ref=beta.tolist(), lin_ref=lin_ref)


#: The measurement settings of the layer-2 template (scripts/layer2/template_rsd.py: the halo superset), whose (j1, j2)
#: keys are re-keyed by sigma for a fit of another configuration (a survey mesh).
TEMPLATE_CONFIG = WSTConfig(J=9, L=6, L2=4, sigma0=0.8, cellsize=1000.0 / 256, step=2**0.5, min_dj=2)


def layer2_template(options):
    """The ``s21_layer2`` template, with its rows keyed in the fit's configuration (``options.layer2_config``, set by
    build_problem): the template's scale indices mapped by sigma (rows of scales the fit lacks dropped)."""
    template = json.loads(options.s21_layer2.read_text())
    config = getattr(options, "layer2_config", None)
    if config is None or (np.isclose(config.sigma(0), TEMPLATE_CONFIG.sigma(0)) and config.step == TEMPLATE_CONFIG.step):
        return template

    def index(j):
        sigma = TEMPLATE_CONFIG.sigma(j)
        jj = int(round(np.log(sigma / config.sigma(0)) / np.log(config.step)))
        return jj if 0 <= jj <= config.J and np.isclose(config.sigma(jj), sigma, rtol=1e-6) else None

    rows = []
    for r in template["rows"]:
        j1, j2 = index(r["key"][0]), index(r["key"][1])
        if j1 is not None and j2 is not None:
            rows.append(dict(r, key=[j1, j2] + list(r["key"][2:])))
    return dict(template, rows=rows)


def layer2_templates(options, labels, mean):
    """Data-vector shapes S21m d0 and S21m R_l d2 of the template pieces (fiducial cosmology, calibrated c_l), all l
    together ('global') or one pair per l ('per_l'); zero outside S21m."""
    template = layer2_template(options)
    c = {int(k): max(v, 0.0) for k, v in template[f"{options.s21_layer2_c}_c"].items()}
    rows = {tuple(r["key"]): r for r in template["rows"]}
    groups = {}
    for i, label in enumerate(labels):
        label = label.split(":")[-1]  # without the cap / q prefixes
        if not label.startswith("S21m"):
            continue
        key = tuple(int(part[1:]) for part in label.split("_")[1:])
        r = rows[key]
        group = 0 if options.s21_layer2_marg == "global" else r["ell"]
        d0, d2 = groups.setdefault(group, (np.zeros(len(labels)), np.zeros(len(labels))))
        d0[i] = mean[i] * r["gauss"]
        d2[i] = mean[i] * r["tree"] / (1.0 + c[r["ell"]] * r["var"])
    return [v for pair in groups.values() for v in pair]


def layer2_variance(options, labels, mean):
    """(S21m dE)^2 per data point, dE^2 = dd0^2 + (R_l dd2)^2 with R_l at the calibrated c_l (zero for other statistics)."""
    template = layer2_template(options)
    c = {int(k): max(v, 0.0) for k, v in template[f"{options.s21_layer2_c}_c"].items()}
    rows = {tuple(r["key"]): r for r in template["rows"]}
    out = np.zeros(len(labels))
    for i, label in enumerate(labels):
        label = label.split(":")[-1]  # without the cap / q prefixes
        if label.startswith("S21m"):
            key = tuple(int(part[1:]) for part in label.split("_")[1:])
            r = rows[key]
            renorm = 1.0 / (1.0 + c[r["ell"]] * r["var"])
            out[i] = (mean[i] * np.hypot(r["gauss_err"], renorm * r["tree_err"])) ** 2
    return out


def fiducial_omega_b(settings) -> float:
    """omega_b of the emulator's fiducial (fixed in the AP ratios), with the massive neutrinos of AbacusSummit (matter at
    the redshifts of the fits)."""
    from wstfast.config import ABACUS_COSMOLOGY

    if settings.get("fiducial") == "abacus":
        return ABACUS_COSMOLOGY["omega_b"] + 0.0006442  # omega_ncdm of one 0.06 eV neutrino
    return QUIJOTE_COSMOLOGY["omega_b"]


def build_problem(options, basis=None, settings=None, space=None, window=None, prefix=""):
    """Theories, data vectors (nrealizations, ndata), labels and metadata of ``options.stats``, from the emulator
    ``options.emulator`` (or the given ``basis`` and ``settings``); also returns the basis, its settings, its
    ``ModelSpec`` and the tracer. ``space`` is the data subdirectory (default ``rsd``, or ``real`` for a real-space
    emulator); survey (``survey_problem``): the cap's subdirectory, ``window`` its ``survey.SurveyWindow`` and ``prefix``
    that of its labels. The projections use the emulator's lattice (``settings["lattice"]``)."""
    if settings is None:
        settings = json.loads(Path(options.emulator).with_suffix(".json").read_text())
    spec = ModelSpec.from_settings(settings)
    if space is None:
        space = "real" if spec.space == "real" else "rsd"
    first = sorted((options.data_dir / space).glob("wst_r*.npz"))[0]
    q0 = options.q[0]
    config = load_measurement(first, q=q0)["config"]
    options.layer2_config = config  # the layer-2 template is re-keyed to it (layer2_template)
    if basis is None:
        basis, settings = load_basis(options.emulator, options.vary, prepare=lambda e, st: contract_unprojected_s21m(
            options, config, e, st))
        apply_gaussian_priors(basis, settings, options.gaussian_prior)
    tracer = spec.tracer
    if options.cubic_biases is not None and not spec.cubic:
        raise SystemExit(f"cubic biases need an emulator trained with --cubic-biases ({options.emulator} is not)")
    if options.ap and spec.space == "real":
        raise SystemExit("real space has no AP distortions")
    nuisance = NuisanceParameters(spec, cubic=options.cubic_biases, tie_m=options.compress_m)
    grid = SpectrumGrid(kmax=settings["kmax"])
    emulated = settings["coefficients"]
    boxsize, nmesh = settings.get("lattice", [1000.0, 256])  # the periodic lattice the emulator describes
    lattice = dict(boxsize=float(boxsize), nmesh=int(nmesh))
    ap = APGeometry(basis, settings["z"], omega_b=fiducial_omega_b(settings)) if options.ap else None

    theories, vectors, labels = [], [], []
    files = None
    if getattr(options, "realizations_like", None) is not None:
        ids = {path.name for path in (options.realizations_like / space).glob("wst_r*.npz")}
        files = [path for path in sorted((options.data_dir / space).glob("wst_r*.npz")) if path.name in ids]
        if not files:
            raise SystemExit(f"no realizations of {options.data_dir} in {options.realizations_like}")
    if getattr(options, "exclude_like", None) is not None:
        ids = {path.name for path in (options.exclude_like / space).glob("wst_r*.npz")}
        files = [path for path in (files or sorted((options.data_dir / space).glob("wst_r*.npz"))) if path.name not in ids]
    if "s1m" in options.stats:
        coefficients = [c for c in select_coefficients(config, s1_min_scale=options.s1_min_scale, s1_ells=options.s1_ells)
                        if c.kind == "S1"]
        missing = [c.label for c in coefficients if c.label not in emulated]
        if missing:
            raise ValueError(f"the emulator has no cumulants for {missing}")
        correction = load_correction(options, config, coefficients, settings) if options.cumulant_points else None
        projection = (APPolynomial(s1m_ap_projection(config, coefficients, grid, **lattice)) if options.ap
                      else S1mProjection(config, coefficients, grid, **lattice))
        for q in options.q:
            s1m = load_s1m_dataset(options.data_dir, space, coefficients, q=q, files=files)
            files = s1m.files
            theories.append(S1mTheory(projection, coefficients, basis=basis, nuisance=nuisance, q=q,
                                      shotnoise=s1m.shotnoise,
                                      cumulant_index=[emulated.index(c.label) for c in coefficients],
                                      correction=correction, ap=ap, sigmas=[config.sigma(c.j) for c in coefficients],
                                      window=window))
            vectors.append(s1m.vectors)
            qprefix = f"q{q:g}:" if len(options.q) > 1 else ""
            labels += [f"{prefix}{qprefix}S1m_j{c.j}_l{c.ell}_m{m}" for c in coefficients for m in range(c.ell + 1)]
        meta = s1m.metadata
    if "s21m" in options.stats:
        from wstfast.theory.second_layer import s21m_entries

        unprojected = not settings.get("s21_projected", True)
        if options.ap and not unprojected and not spec.ap_order:
            raise SystemExit("ap for s21m needs an emulator with the unprojected S21m basis (--s21-unprojected) "
                             "or with AP-polynomial terms (--ap-order)")
        coefficients = [c for c in select_coefficients(config, s21_min_scale=options.s21_min_scale,
                                                       s21_min_ratio=options.s21_min_ratio,
                                                       s21_min_scale2=options.s21_min_scale2) if c.kind == "S21"]
        emulated21 = [c for c in select_coefficients(config, s21_min_scale=0.0, s21_min_ratio=0.0)
                      if c.kind == "S21" and c.label in settings.get("s21_coefficients", [])]
        emulated21.sort(key=lambda c: settings["s21_coefficients"].index(c.label))
        missing = [c.label for c in coefficients if c not in emulated21]
        if missing:
            raise ValueError(f"the emulator has no S21m terms for {missing}")
        first_layer = [Coefficient("S1", ell, j) for j, ell in sorted({(c.j, c.ell) for c in coefficients})]
        missing = [c.label for c in first_layer if c.label not in emulated]
        if missing:
            raise ValueError(f"the emulator has no cumulants for the first-layer {missing}")
        basis_entries = {(emulated21[i].label, m1, m2): n for n, (i, m1, m2) in enumerate(s21m_entries(emulated21))}
        entry_index = [basis_entries[coefficients[i].label, m1, m2] for i, m1, m2 in s21m_entries(coefficients)]
        s21m = load_s21m_dataset(options.data_dir, space, coefficients, q=q0, files=files)
        files = s21m.files
        s21m_kwargs = dict(basis=basis, nuisance=nuisance, entry_index=entry_index,
                           cumulant_index=[emulated.index(c.label) for c in first_layer], shotnoise=s21m.shotnoise,
                           white_noise=options.s21_white, config=config, window=window)
        if spec.ap_order and options.ap:  # AP-polynomial terms of the basis, at the trial AP ratios
            s21m_kwargs.update(ap=ap, first_ap=APPolynomial(s1m_ap_projection(config, first_layer, grid, **lattice)))
        if unprojected and settings.get("prepared") is not None:  # emulator contracted at load (contract_s21m)
            s21m_kwargs.update(entry_index=None, ap=ap, s21_projection=settings["prepared"])
        elif unprojected:  # lambda_2 projected at assembly from the emulated field grids, with AP if --ap
            from numpy.polynomial.legendre import leggauss

            from wstfast.theory.second_layer import S21mAPProjection, S21mProjection, first_layer_fields

            fields, basis_fields = first_layer_fields(coefficients), first_layer_fields(emulated21)
            x, _ = leggauss(12)  # the (K, mu_K) grid of S21mBasis
            s21m_kwargs.update(entry_index=None, ap=ap,
                               s21_projection=(S21mAPProjection if options.ap else S21mProjection)(
                                   config, coefficients, fields, np.geomspace(4e-3, 0.12, 24), x[6:],
                                   boxsize=float(boxsize), nmesh=int(nmesh)),
                               s21_field_rows=[basis_fields.index(field) for field in fields])
        layer2 = (load_layer2(options, S1mProjection(config, first_layer, grid, **lattice), coefficients, first_layer, q0,
                              s21m_kwargs) if getattr(options, "s21_layer2", None) is not None else None)
        theories.append(S21mTheory(S1mProjection(config, first_layer, grid, **lattice), coefficients, first_layer, q=q0,
                                   layer2=layer2, **s21m_kwargs))
        vectors.append(s21m.vectors)
        labels += [f"{prefix}S21m_j{coefficients[i].j}_j{coefficients[i].j2}_l{coefficients[i].ell}_m{m1}_m{m2}"
                   for i, m1, m2 in s21m_entries(coefficients)]
        meta = s21m.metadata
    if "band" in options.stats:
        coefficients = [c for c in select_coefficients(config, s1_min_scale=options.band_min_scale,
                                                       s1_ells=options.band_ells)
                        if c.kind == "S1" and config.sigma(c.j) <= options.band_max_scale + 1e-6]
        if options.band_kmax is not None:
            from scipy.special import gammaincinv

            coefficients = [c for c in coefficients
                            if np.sqrt(gammaincinv(c.ell + 1.5, 0.84)) / config.sigma(c.j) <= options.band_kmax]
        if not coefficients:
            raise SystemExit("no band power passes the cuts")
        if "s1m" in options.stats and config.sigma(min(c.j for c in coefficients)) < options.s1_min_scale - 1e-6 \
                and options.band_max_scale >= options.s1_min_scale:
            raise SystemExit("the band powers overlap the S1m scales: set band_max_scale below s1_min_scale (a block "
                             "with both exponents measures the Edgeworth factor beyond the model's accuracy)")
        if options.ap:
            raise SystemExit("the band powers have no AP projection yet")
        projection = BandProjection(config, coefficients, grid, **lattice)
        lost = [(c.label, s) for c, s in zip(coefficients, projection.support) if s > 0.01]
        if lost:
            raise SystemExit("bands beyond the emulator's grid (kmax = %.2f): " % grid.k[-1]
                             + ", ".join(f"{label} ({100 * s:.0f}%)" for label, s in lost))
        if options.band_counterterm not in ("shared", "separate"):
            raise SystemExit(f"unknown band_counterterm {options.band_counterterm!r}")
        band = load_band_dataset(options.data_dir, space, coefficients, files=files)
        files = band.files
        theories.append(PowerTheory(projection, basis=basis, nuisance=nuisance, shotnoise=band.shotnoise,
                                    counterterms="" if options.band_counterterm == "shared" else "_band"))
        vectors.append(band.vectors)
        labels += [f"{prefix}S1q2_j{c.j}_l{c.ell}" for c in coefficients]
        meta = band.metadata
    if "pk" in options.stats:
        ells = (0,) if spec.space == "real" else (0, 2, 4)  # real space: the monopole
        power = load_power_dataset(options.data_dir, space, kmin=options.kmin, kmax=options.kmax, rebin=options.rebin,
                                   files=files, ells=ells)
        files = power.files
        projection = (APPolynomial(multipole_ap_projection(power.edges, grid, ells=ells, **lattice)) if options.ap
                      else MultipoleProjection(power.edges, grid, ells=ells, **lattice))
        theories.append(PowerTheory(projection, basis=basis, nuisance=nuisance, shotnoise=power.shotnoise, ap=ap))
        vectors.append(power.vectors)
        labels += [f"{prefix}P{ell}_k{k:.4f}" for ell in ells for k in power.k]
        meta = power.metadata
    return dict(theories=theories, vectors=np.concatenate(vectors, axis=1), labels=labels, meta=meta, basis=basis,
                settings=settings, spec=spec, tracer=tracer)


def box_entries(box_dir: Path, labels, config: WSTConfig, q: float) -> tuple[np.ndarray, float]:
    """Box realizations (nbox, ndata) of the S1m / S21m entries ``labels`` (of a fit of ``config``, prefixes ignored),
    matched by sigma in the box measurements of ``box_dir``/rsd, and the box volume [(Gpc/h)^3]."""
    files = sorted((Path(box_dir) / "rsd").glob("wst_r*.npz"))
    first = load_measurement(files[0], q=q)
    box_config = first["config"]

    def j_box(j):
        sigma = config.sigma(j)
        matches = [jb for jb in range(box_config.J + 1) if np.isclose(box_config.sigma(jb), sigma, rtol=1e-6)]
        if not matches:
            raise SystemExit(f"no box scale for sigma = {sigma:.2f} Mpc/h in {box_dir}")
        return matches[0]

    keys = []
    for label in labels:
        parts = label.split(":")[-1].split("_")
        if parts[0] == "S1m":
            keys.append(("S1m", j_box(int(parts[1][1:])), int(parts[2][1:]), int(parts[3][1:])))
        elif parts[0] == "S21m":
            keys.append(("S21m", j_box(int(parts[1][1:])), j_box(int(parts[2][1:])), int(parts[3][1:]),
                         int(parts[4][1:]), int(parts[5][1:])))
        else:
            raise SystemExit(f"survey fits support S1m and S21m only ({label})")
    vectors = []
    for path in files:
        m = load_measurement(path, q=q)
        vectors.append([m["S1m"][k[1], k[2], k[3]] if k[0] == "S1m"
                        else m["S2m"][k[1], k[2], k[3], k[4], k[5]] / m["S1m"][k[1], k[3], k[4]] for k in keys])
    return np.array(vectors), (first["metadata"]["boxsize"] / 1000.0) ** 3


def survey_problem(options):
    """Theories, data, labels and covariance of a survey fit (``survey``): one window per cap, shared parameters."""
    from wstfast.data import sample_covariance as sample
    from wstfast.survey import SurveyWindow

    if options.window is None or len(options.window) != len(options.survey):
        raise SystemExit("a survey fit needs one window file per cap")
    if options.covariance_data_dir is None and options.covariance_file is None:
        raise SystemExit("a survey fit needs covariance_data_dir (box realizations) or covariance_file (analytic)")
    basis = settings = None
    caps = []
    for cap, path in zip(options.survey, options.window):
        window = SurveyWindow.load(path)
        if window.veff is None:
            raise SystemExit(f"{path} has no effective volume (rebuild it with dr1_window.py --test products)")
        problem = build_problem(options, basis=basis, settings=settings, space=cap, window=window, prefix=f"{cap}:")
        basis, settings = problem["basis"], problem["settings"]
        caps.append((cap, window, problem))
    nreal = {p["vectors"].shape[0] for _, _, p in caps}
    if len(nreal) != 1:
        raise SystemExit(f"the caps have different numbers of realizations: {nreal}")
    nreal = nreal.pop()
    theories = [t for _, _, p in caps for t in p["theories"]]
    vectors = np.concatenate([p["vectors"] for _, _, p in caps], axis=1)
    labels = [label for _, _, p in caps for label in p["labels"]]
    blocks, nsims = [], None
    if options.covariance_file is not None:  # analytic ln covariance of one survey, in data units through the data mean
        analytic = np.load(options.covariance_file)
        index = {str(label): i for i, label in enumerate(analytic["labels"])}
        missing = [label for label in labels if label not in index]
        if missing:
            raise SystemExit(f"{options.covariance_file} has no entries {missing[:5]}...")
        rows = np.array([index[label] for label in labels])
        mean = vectors.mean(axis=0)
        covariance = analytic["lncov"][np.ix_(rows, rows)] * np.outer(mean, mean)
        covariance /= nreal if options.covariance_of_mean else 1
        nsims = 10**9  # exact: no Hartlap / Percival factors
    else:
        for cap, window, problem in caps:
            box, vbox = box_entries(options.covariance_data_dir, problem["labels"], options.layer2_config, options.q[0])
            nsims = box.shape[0]
            blocks.append(sample(box) * vbox / window.veff / (nreal if options.covariance_of_mean else 1))
        covariance = np.zeros((len(labels), len(labels)))
        start = 0
        for block in blocks:
            covariance[start:start + len(block), start:start + len(block)] = block
            start += len(block)
    if getattr(options, "s21_layer2", None) is not None:  # the template's error is the same in every cap
        mean = vectors.mean(axis=0)
        error = np.sqrt(layer2_variance(options, labels, mean))
        base = np.array([label.split(":", 1)[-1] for label in labels])
        covariance = covariance + np.where(base[:, None] == base[None, :], np.outer(error, error), 0.0)
        if options.s21_layer2_marg is not None:
            for vector in layer2_templates(options, labels, mean):
                covariance = covariance + options.s21_layer2_marg_width**2 * np.outer(vector, vector)
    meta = dict(caps[0][2]["meta"], caps=list(options.survey), veff={cap: w.veff for cap, w, _ in caps})
    combine = None
    if getattr(options, "combine_caps", False):
        theories, vectors, labels, covariance, combine = combine_survey_caps(caps, theories, vectors, labels, covariance)
        meta["combined"] = "GCcomb"
    return dict(theories=theories, vectors=vectors, labels=labels, meta=meta, basis=basis, settings=settings,
                spec=caps[0][2]["spec"], tracer=caps[0][2]["tracer"], covariance=covariance, nsims=nsims,
                combine=combine)


def first_layer_entries(base_labels):
    """For each entry label (without cap prefix), the index of the S1m entry of its first-layer block (S21m_j1_j2_l_m1_m2
    -> S1m_j1_l_m1), or -1 for S1m entries."""
    index = {label: i for i, label in enumerate(base_labels)}
    first = []
    for label in base_labels:
        parts = label.split(":")[-1].split("_")
        if parts[0] == "S21m":
            key = f"S1m_{parts[1]}_{parts[3]}_{parts[4]}"
            if key not in index:
                raise SystemExit(f"combine_caps needs the first-layer S1m entry {key} of {label} in the data vector")
            first.append(index[key])
        else:
            first.append(-1)
    return np.array(first)


def combine_survey_caps(caps, theories, vectors, labels, covariance):
    """The caps combined as desi-clustering's GCcomb (``calculators.combine_caps``): data vectors per realization,
    one ``CombinedCaps`` theory, labels prefixed GCcomb:, and the covariance projected with the Jacobian of the
    combination at the data mean. Returns also ``combine`` = (jacobian (n, n_all), per-cap labels, per-cap data
    mean) for the extra covariances of ``build_fit``."""
    base = [label.split(":", 1)[1] for label in labels if label.startswith(f"{caps[0][0]}:")]
    positions, starts, weights = [], [], []
    for cap, window, problem in caps:
        own = {label.split(":", 1)[1]: i for i, label in enumerate(labels) if label.startswith(f"{cap}:")}
        if set(own) != set(base):
            raise SystemExit(f"combine_caps: the caps have different data vectors ({cap})")
        positions.append([own[b] for b in base])  # in the concatenated vector of all caps
        starts.append(min(own.values()))  # the cap's own vector starts here
        weights.append(float(np.sum(next(iter(window.s1m.values()))["weight"])))  # the cap's FKP weight sum
    positions, starts, weights = np.array(positions), np.array(starts), np.array(weights)
    first = first_layer_entries(base)
    per_cap = len(theories) // len(caps)
    combined = np.array([combine_caps(np.stack([v[p] for p in positions]), weights, first) for v in vectors])
    theory = CombinedCaps(theories, per_cap, positions - starts[:, None], weights, first)  # per-cap positions
    # Jacobian at the data mean (the S21m weights depend on the S1m means; a second-order effect in the covariance)
    mean = vectors.mean(axis=0)
    n, nall = len(base), len(labels)
    jac = np.zeros((n, nall))
    for i in range(n):
        if first[i] < 0:
            w = weights
        else:
            w = weights * np.array([mean[p[first[i]]] for p in positions])
        w = w / w.sum()
        for c, p in enumerate(positions):
            jac[i, p[i]] = w[c]
    new_labels = [f"GCcomb:{b}" for b in base]
    print(f"caps combined as GCcomb: weights {dict(zip([c for c, _, _ in caps], np.round(weights / weights.sum(), 3)))}, "
          f"{n} entries from {nall}", flush=True)
    return [theory], combined, new_labels, jac @ covariance @ jac.T, (jac, list(labels), mean)


def m_compression(labels):
    """Compression of a real-space data vector: the |m| blocks of S1m and the (|m1|, |m2|) entries of S21m averaged within
    the classes that an isotropic field makes equally distributed (blocks of the same number of components: m = 0, or
    |m| >= 1; for S21m the four combinations of the two layers). Returns the matrix A (ncompressed, n) and the labels
    of the compressed entries (``m1+`` for the |m| >= 1 class). Other entries pass through."""
    groups, order = {}, []
    for i, label in enumerate(labels):
        prefix, _, base = label.rpartition(":")
        parts = base.split("_")
        if parts[0] in ("S1m", "S21m"):
            key = "_".join(parts[:-1 if parts[0] == "S1m" else -2]
                           + [("m0" if int(m[1:]) == 0 else "m1+") for m in parts[-1 if parts[0] == "S1m" else -2:]])
        else:
            key = base
        key = f"{prefix}:{key}" if prefix else key
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    matrix = np.zeros((len(order), len(labels)))
    for row, key in enumerate(order):
        matrix[row, groups[key]] = 1.0 / len(groups[key])
    return matrix, order


def band_compression(vectors, labels, cut: float):
    """Compression of the band-power entries (labels ``S1q2_*``) onto the principal components of their sample
    correlation with eigenvalue above ``cut`` (standardized units); the other entries pass through. Returns the matrix
    A (ncompressed, n) and the labels (``S1q2_pc{i}``, by decreasing eigenvalue), or (None, labels) when no entry is a
    band or none is dropped."""
    band = np.array([label.rpartition(":")[-1].startswith("S1q2_") for label in labels])
    if not band.any():
        return None, list(labels)
    values = vectors[:, band]
    std = values.std(axis=0, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(np.corrcoef(values, rowvar=False))
    keep = np.flatnonzero(eigenvalues > cut)[::-1]  # decreasing eigenvalue
    if len(keep) == band.sum():
        return None, list(labels)
    prefix = labels[np.flatnonzero(band)[0]].rpartition(":")[0]
    prefix = f"{prefix}:" if prefix else ""
    rows, out = [], []
    first = True
    for i, label in enumerate(labels):
        if not band[i]:
            row = np.zeros(len(labels))
            row[i] = 1.0
            rows.append(row)
            out.append(label)
        elif first:  # the components, at the position of the first band entry
            first = False
            for n, k in enumerate(keep):
                row = np.zeros(len(labels))
                row[band] = eigenvectors[:, k] / std
                rows.append(row)
                out.append(f"{prefix}S1q2_pc{n}")
    print(f"band powers: {len(keep)} of {band.sum()} principal components kept (correlation eigenvalues > {cut:g})",
          flush=True)
    return np.array(rows), out


def covariance_terms(options: FitOptions, labels, mean, n: int) -> np.ndarray:
    """Volume-independent terms added to the covariance (n, n): the Monte Carlo error of the layer-2 template, its
    analytically marginalized amplitudes, and the ``extra_covariance`` files (modelling errors, matched by label)."""
    out = np.zeros((n, n))
    if options.s21_layer2 is not None:
        out += np.diag(layer2_variance(options, labels, mean))
        if options.s21_layer2_marg is not None:
            for vector in layer2_templates(options, labels, mean):
                out += options.s21_layer2_marg_width**2 * np.outer(vector, vector)
    for path in options.extra_covariance:
        extra = np.load(path)
        index = {str(label): i for i, label in enumerate(extra["labels"])}
        present = np.array([i for i, label in enumerate(labels) if label in index])  # e.g. one cap's entries
        if not len(present):
            raise SystemExit(f"{path} has no covariance for any of the data points")
        rows = np.array([index[labels[i]] for i in present])
        block = extra["fractional" if "fractional" in extra.files else "covariance"][np.ix_(rows, rows)]
        if "fractional" in extra.files:  # relative errors (e.g. the window calibration): in data units via the mean
            block = block * np.outer(mean[present], mean[present])
        out[np.ix_(present, present)] += block
    return out


def build_fit(options: FitOptions) -> dict:
    """The likelihood of a fit (``FitOptions``): theories and data (``build_problem``, or ``survey_problem``), the
    covariance (sample covariance of the realizations, or of ``covariance_data_dir``; scaled to ``volume``; plus
    ``covariance_terms``), the Hartlap-corrected inverse with the Percival factor m1, and the parameters ``fix`` fixed.
    With ``compress_m`` and ``band_cut`` the data, theory and covariance are compressed by ``m_compression`` and
    ``band_compression``. Returns
    dict(likelihood, theory, labels, problem, covariance, nsims, m1, ...)."""
    if options.volume is not None and options.covariance_of_mean:
        raise SystemExit("volume and covariance_of_mean are exclusive")
    if options.compress_m and options.survey:
        raise SystemExit("compress_m is for box fits in real space")
    problem = survey_problem(options) if options.survey else build_problem(options)
    theories, vectors, labels, meta, tracer = (problem[k] for k in ("theories", "vectors", "labels", "meta", "tracer"))
    theory = theories[0] if len(theories) == 1 else JointTheory(theories)
    nreal = vectors.shape[0]
    mean = vectors.mean(axis=0)
    compression = None
    if options.compress_m:
        compression, labels = m_compression(labels)
        vectors = vectors @ compression.T
    if options.band_cut > 0 and not options.survey:
        cut, labels = band_compression(vectors, labels, options.band_cut)
        if cut is not None:
            vectors = vectors @ cut.T
            compression = cut if compression is None else cut @ compression
    if compression is not None:
        theory = LinearCompression(theory, compression)
    ndata = vectors.shape[1]
    nsims = nreal  # realizations of the covariance estimate
    if options.survey:  # box covariance at each cap's effective volume, with the layer-2 terms (survey_problem)
        covariance, nsims = problem["covariance"], problem["nsims"]
    elif options.covariance_data_dir is None:
        covariance = sample_covariance(vectors, of_mean=options.covariance_of_mean)
    else:  # one-box covariance of another set; of the mean of this set's nreal boxes with --covariance-of-mean
        other = dataclasses.replace(options)
        other.data_dir, other.realizations_like, other.exclude_like = options.covariance_data_dir, None, None
        cov_vectors = build_problem(other, basis=problem["basis"], settings=problem["settings"])["vectors"]
        if compression is not None:
            cov_vectors = cov_vectors @ compression.T
        nsims = cov_vectors.shape[0]
        covariance = sample_covariance(cov_vectors)
        if options.covariance_of_mean:
            covariance = covariance / nreal
    if options.volume is not None and not options.survey:
        covariance = covariance * (meta["boxsize"] / 1000.0) ** 3 / options.volume
    if options.survey:  # the layer-2 terms are in the survey covariance; the extras on the per-cap labels, then combined
        combine = problem.get("combine")
        if combine is not None:
            full = covariance_terms(dataclasses.replace(options, s21_layer2=None), combine[1], combine[2], len(combine[1]))
            covariance = covariance + combine[0] @ full @ combine[0].T
        else:
            covariance = covariance + covariance_terms(dataclasses.replace(options, s21_layer2=None), labels, mean,
                                                       ndata)
    else:  # on the uncompressed entries (the template is keyed by them), then compressed
        terms = covariance_terms(options, problem["labels"], mean, len(problem["labels"]))
        covariance = covariance + (compression @ terms @ compression.T if compression is not None else terms)
    print(f"{ndata} data points ({', '.join(options.stats)}), {nreal} realizations")

    fixed = parse_fixed(options.fix)
    nparams = len([n for n in get_params(theory).select(varied=True, derived=False).names() if n not in fixed])
    m1 = 1.0 if options.no_percival else percival_m1(nsims, ndata, nparams)
    print(f"Hartlap factor {hartlap_factor(nsims, ndata):.4f} ({nsims} realizations, {ndata} data points); "
          f"Percival m1 {m1:.4f} for {nparams} parameters", flush=True)
    likelihood = WSTLikelihood(theory, vectors.mean(axis=0), covariance * m1)
    fix_parameters(likelihood, fixed)
    return dict(likelihood=likelihood, theory=theory, labels=list(labels), problem=problem, covariance=covariance,
                nsims=int(nsims), m1=m1, nparams=nparams, ndata=int(ndata), nreal=int(nreal))
