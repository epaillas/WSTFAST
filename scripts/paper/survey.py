#!/usr/bin/env python
"""Figure, table and numbers of the survey-geometry section (wstfast.survey).

Reads the measurements written by

    python scripts/survey/v0a_gaussian.py --nmesh 200 --erosion 0 1 2 3 4 --erosion-b 3 --realizations 16 \
        --realizations-a 4 --output outputs/survey/v0a
    python scripts/survey/v0a_window.py --output outputs/survey/v0a_window
    python scripts/survey/v0b_halos.py --mode uniform --window none --output outputs/survey/v0b   # then --reuse (CIC)
    python scripts/survey/v0b_halos.py --mode radial --window none --output outputs/survey/v0c    # then --reuse (CIC)
    python scripts/survey/v0b_mock_window.py --mode {uniform,radial} --los {z,local} \
        --output outputs/survey/mock_window/{mode}_{los}

and writes paper/figures/survey.pdf, paper/generated/table_survey.tex and paper/generated/numbers_survey.tex.

``--halo-window``: paper/figures/survey_halo_window.pdf, the DR1 NGC test of the window on Quijote halo cut-sky mocks
(``dr1_window.py --test halos``, outputs/survey/dr1_halos_v2_*): the parent-box data vector, the survey data vector,
and the box data vector times the window of the model (delta and K calibrated on pairs 0-274, at the halo signal),
all on the held-out pairs 275-299. The window factors at the Quijote truth (with the nuisances of the box baseline
fit) are cached in outputs/paper/survey_halo_window.npz.
Biases are in units of the scatter of S1m over realizations of the masked survey (``sigma``), as in the scripts.

    python scripts/paper/survey.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SURVEY = ROOT / "outputs/survey"
FIGURES, GENERATED, PNG = ROOT / "paper/figures", ROOT / "paper/generated", ROOT / "outputs/paper"
FULL = 7.0  # PRD text width [in]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#d9d8d4", "#ebeae6"
MARKERS = ["o", "s", "D", "^"]

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})


def residuals(masked, reference, ratio, ratio_error):
    """(masked - R reference) per block in units of the masked scatter, its error, and the uncorrected bias."""
    scatter = masked.std(axis=0, ddof=1)
    residual = masked - ratio * reference
    z = residual.mean(axis=0) / scatter
    error = np.sqrt((residual.std(axis=0, ddof=1) / np.sqrt(len(masked))) ** 2
                    + (ratio_error * reference.mean(axis=0)) ** 2) / scatter
    raw = (masked - reference).mean(axis=0) / scatter
    return z, error, raw


def summarize(z, error, raw):
    finite = np.isfinite(z)
    return dict(raw_min=float(np.nanmin(np.nanmax(np.abs(raw), axis=(1, 2)))),
                raw_max=float(np.nanmax(np.abs(raw))), max=float(np.nanmax(np.abs(z))),
                rms=float(np.sqrt(np.nanmean(z**2))), chi2=float(np.nansum((z / error) ** 2)),
                n=int(finite.sum()), error=float(np.nanmax(error)))


def flat(x):
    """Blocks (j, l, |m|) in the order of the data vector: l, then sigma_j, then |m|."""
    return np.array([x[j, ell, m] for ell in range(x.shape[1]) for j in range(x.shape[0]) for m in range(ell + 1)])


def main():
    numbers = {}
    rows = []

    # Gaussian fields: edges alone (test A) and the varying noise (test B).
    a = np.load(SURVEY / "v0a/test_a.npz")
    masked_a, reference_a, erosions, sigmas = a["masked"], a["reference"], a["erosion"], a["sigma"]
    edge_sigma = np.abs((masked_a - reference_a).mean(axis=0)) / masked_a.std(axis=0, ddof=1)
    edge_frac = np.abs((masked_a - reference_a).mean(axis=0) / reference_a.mean(axis=0))
    edge_by_erosion = np.nanmax(edge_sigma, axis=(2, 3, 4)) if edge_sigma.ndim == 5 else np.nanmax(edge_sigma, axis=(2, 3))
    frac_by_erosion = np.nanmax(edge_frac, axis=(2, 3))
    for i, c in enumerate(erosions):
        tag = ["None", "One", "Two", "Three", "Four"][int(c)]
        numbers[f"SurveyEdgeSigma{tag}"] = f"{np.nanmax(edge_by_erosion[i]):.2g}"
        numbers[f"SurveyEdgeFracMin{tag}"] = f"{100 * np.nanmin(frac_by_erosion[i]):.1f}"
        numbers[f"SurveyEdgeFracMax{tag}"] = f"{100 * np.nanmax(frac_by_erosion[i]):.1f}"
    b = np.load(SURVEY / "v0a/test_b.npz")
    scatter_b = b["masked"].std(axis=0, ddof=1)
    numbers["SurveyNoiseEffective"] = f"{np.nanmax(np.abs(b['exact'] - b['effective']) / scatter_b):.2f}"
    numbers["SurveyNoiseLocal"] = f"{np.nanmax(np.abs(b['exact'] - b['local']) / scatter_b):.2f}"
    error_b = scatter_b / np.sqrt(len(b["masked"]))
    numbers["SurveyNoiseChisq"] = f"{np.nansum(((b['masked'].mean(axis=0) - b['effective']) / error_b) ** 2):.0f}"
    numbers["SurveyNoiseN"] = str(int(np.isfinite(b["effective"]).sum()))
    numbers["SurveyGaussNreal"] = str(len(b["masked"]))

    # Gaussian-mock R at a fiducial spectrum, applied to other spectra (same seeds across variants).
    w = np.load(SURVEY / "v0a_window/measurements.npz")
    spread = {}
    for c in ("0", "1"):
        ratio = w[f"ratio_c{c}"][0]
        base = w[f"fiducial_c{c}_masked"] - ratio * w[f"fiducial_c{c}_reference"]
        scatter = w[f"fiducial_c{c}_masked"].std(axis=0, ddof=1)
        for name in ("bias_f", "amplitude", "omega_cdm", "n_s"):
            other = w[f"{name}_c{c}_masked"] - ratio * w[f"{name}_c{c}_reference"]
            spread[name, c] = np.nanmax(np.abs((other - base).mean(axis=0)) / scatter)
    numbers["SurveyGaussSpectrumNone"] = f"{max(spread[k] for k in spread if k[1] == '0'):.2f}"
    numbers["SurveyGaussSpectrumOne"] = f"{max(spread[k] for k in spread if k[1] == '1'):.2f}"
    numbers["SurveyGaussBiasNone"] = f"{spread['bias_f', '0']:.2f}"

    # Halos with the Gaussian-mock R (no window, and with the CIC window), and with the mock-calibrated R.
    for mode, directory, label in (("uniform", "v0b", "Uniform"), ("radial", "v0c", "Radial")):
        h = np.load(SURVEY / directory / "measurements.npz")
        for window, ratio, ratio_error in (("Gauss", h["ratio"], h["ratio_error"]),
                                           ("GaussCic", *[np.load(SURVEY / directory / "ratio_cic.npz")[k]
                                                          for k in ("ratio", "ratio_error")])):
            s = summarize(*residuals(h["masked"], h["reference"], ratio, ratio_error))
            numbers[f"Survey{window}{label}Rms"] = f"{s['rms']:.2f}"
            numbers[f"Survey{window}{label}Chisq"] = f"{s['chi2']:.0f}"
            rows.append((mode, r"$z$", "Gaussian" + ("+CIC" if window == "GaussCic" else ""), s))
        excess = (h["masked"].mean(axis=0) / h["reference"].mean(axis=0) - 1) / (h["ratio"] - 1)
        numbers[f"SurveyExcess{label}"] = f"{100 * (np.nanmedian(excess[0]) - 1):.1f}"
    numbers["SurveyHaloNcal"] = str(len(np.load(SURVEY / "v0b/measurements.npz")["masked"]))

    split = {}
    for mode, los in (("uniform", "z"), ("radial", "z"), ("radial", "local")):
        m = np.load(SURVEY / f"mock_window/{mode}_{los}/measurements.npz")
        z, error, raw = residuals(m["test_masked"], m["test_reference"], m["ratio"], m["ratio_error"])
        s = summarize(z, error, raw)
        split[mode, los] = (z, error, m)
        key = f"{mode.capitalize()}{'Local' if los == 'local' else 'Z'}"
        numbers[f"SurveySplit{key}Max"] = f"{s['max']:.2f}"
        numbers[f"SurveySplit{key}Rms"] = f"{s['rms']:.2f}"
        numbers[f"SurveySplit{key}Chisq"] = f"{s['chi2']:.0f}"
        rows.append((mode, "local" if los == "local" else r"$z$", "halos", s))
        numbers["SurveySplitNcal"] = str(len(m["calibration_masked"]))
        numbers["SurveySplitNtest"] = str(len(m["test_masked"]))
    numbers["SurveySplitMax"] = f"{max(np.nanmax(np.abs(v[0])) for v in split.values()):.2f}"
    numbers["SurveySplitErr"] = f"{max(np.nanmax(v[1]) for v in split.values()):.2f}"
    calibrated = split["radial", "local"][2]["ratio"]
    numbers["SurveyEdgeMin"] = f"{100 * np.nanmin(1 - calibrated):.0f}"
    numbers["SurveyEdgeMax"] = f"{100 * np.nanmax(1 - calibrated):.0f}"
    numbers["SurveyNblocks"] = str(int(np.isfinite(calibrated).sum()))

    # Wide angle: local- vs plane-parallel references of the same boxes and thinning.
    loc, pp = split["radial", "local"][2], split["radial", "z"][2]
    ref_l = np.concatenate([loc["calibration_reference"], loc["test_reference"]])
    ref_z = np.concatenate([pp["calibration_reference"], pp["test_reference"]])
    scatter = np.concatenate([loc["calibration_masked"], loc["test_masked"]]).std(axis=0, ddof=1)
    diff = ref_l - ref_z
    zw = diff.mean(axis=0) / scatter
    ew = diff.std(axis=0, ddof=1) / np.sqrt(len(diff)) / scatter
    numbers["SurveyWideMax"] = f"{np.nanmax(np.abs(zw)):.2f}"
    numbers["SurveyWideRms"] = f"{np.sqrt(np.nanmean(zw**2)):.2f}"
    numbers["SurveyWideChisq"] = f"{np.nansum((zw / ew) ** 2):.0f}"
    numbers["SurveyWideFrac"] = f"{100 * np.nanmax(np.abs(diff.mean(axis=0) / ref_z.mean(axis=0))):.2f}"

    write_numbers(numbers)
    write_table(rows)
    figure(sigmas, erosions, edge_by_erosion, calibrated, split, rows)


def write_numbers(numbers):
    GENERATED.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by scripts/paper/survey.py; do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in numbers.items()]
    (GENERATED / "numbers_survey.tex").write_text("\n".join(lines) + "\n")


def write_table(rows):
    lines = [r"\begin{table}", r"\caption{Window correction of the halo $\Sonem$ in the test cone, without erosion. "
             r"Biases of the \SurveyNblocks{} coefficients in units of the scatter of one masked survey: before the "
             r"correction (``Before'': range over $\sigma_j$ of the largest bias), after it (largest and rms), and the $\chi^2$ of "
             r"the corrected residuals with their errors. $R$ from Gaussian mocks is calibrated on 32 mocks and tested "
             r"on \SurveyHaloNcal{} halo boxes, without (Gaussian) and with (Gaussian+CIC) the CIC window; $R$ from "
             r"halo mocks (halos) is calibrated on boxes 0--39 and tested on boxes 40--79.}",
             r"\label{tab:survey}", r"\begin{ruledtabular}", r"\begin{tabular}{lllcccc}",
             r"$\bar n$ & LOS & $R$ from & Before & Max & rms & $\chi^2$ \\", r"\hline"]
    for mode, los, source, s in rows:
        lines.append(f"{mode} & {los} & {source} & {s['raw_min']:.1f}--{s['raw_max']:.1f} & {s['max']:.2f} & "
                     f"{s['rms']:.2f} & {s['chi2']:.0f}/{s['n']} \\\\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    (GENERATED / "table_survey.tex").write_text("\n".join(lines) + "\n")


def figure(sigmas, erosions, edge_by_erosion, calibrated, split, rows):
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw=dict(width_ratios=[1, 1, 1.6], wspace=0.38))

    # (a) Size of the edge effect: 1 - R per block (halo mocks, radial nbar, local line of sight).
    ax = axes[0]
    for ell in range(calibrated.shape[1]):
        for j, sigma in enumerate(sigmas):
            values = 100 * (1 - calibrated[j, ell, :ell + 1])
            ax.plot(np.full(len(values), sigma * (1 + 0.035 * (ell - 2))), values, ls="none", marker="o", ms=3,
                    color=SERIES[0], alpha=0.85, mec="white", mew=0.4)
    ax.set_xscale("log")
    ax.set_xticks(sigmas)
    ax.set_xticklabels([f"{s:.0f}" for s in sigmas])
    ax.minorticks_off()
    ax.set_xlabel(r"$\sigma_j\ [\mpch]$".replace(r"\mpch", r"h^{-1}\,\mathrm{Mpc}"))
    ax.set_ylabel(r"$1-R\ [\%]$")
    ax.set_title(r"(a) edge suppression of $S_1^m$", loc="left")

    # (b) Erosion: largest edge bias of the masked Gaussian fields vs the erosion, per scale.
    ax = axes[1]
    for j, sigma in enumerate(sigmas):
        values = np.maximum(edge_by_erosion[:, j], 1e-3)
        ax.plot(erosions, values, marker=MARKERS[j], ms=4, lw=1.5, color=SERIES[j], mec="white", mew=0.5,
                label=rf"$\sigma_j={sigma:.0f}$")
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"erosion $c$ [units of $\sigma_j$]")
    ax.set_ylabel(r"largest $|\mathrm{bias}|/\sigma$")
    ax.set_title("(b) uncorrected, Gaussian fields", loc="left")
    ax.legend(frameon=False, fontsize=6, handlelength=1.5, loc="lower left")

    # (c) Residuals after the correction, per block, with the error band of the mock-calibrated test.
    ax = axes[2]
    gauss = [r for r in rows if r[0] == "uniform" and r[2] == "Gaussian"][0]
    h = np.load(SURVEY / "v0b/measurements.npz")
    zg, eg, _ = residuals(h["masked"], h["reference"], h["ratio"], h["ratio_error"])
    series = [(r"$R$ from Gaussian mocks; uniform $\bar n$, LOS $z$", zg), ]
    series += [(rf"$R$ from halo mocks; {mode} $\bar n$, LOS {'local' if los == 'local' else '$z$'}", split[mode, los][0])
               for mode, los in (("uniform", "z"), ("radial", "z"), ("radial", "local"))]
    x = np.arange(len(flat(zg)))
    band = np.nanmax(np.stack([flat(split[k][1]) for k in split]), axis=0)
    ax.fill_between(x, -band, band, color=BAND, lw=0, label=r"$\pm1$ error (halo mocks)")
    for i, (label, z) in enumerate(series):
        ax.plot(x + 0.18 * (i - 1.5), flat(z), ls="none", marker=MARKERS[i], ms=2.8, color=SERIES[i], mec="white",
                mew=0.3, label=label)
    ax.axhline(0, color=MUTED, lw=0.6)
    edges = np.cumsum([len(sigmas) * (ell + 1) for ell in range(calibrated.shape[1])])
    for e in edges[:-1]:
        ax.axvline(e - 0.5, color=GRID, lw=0.6)
    for ell, (lo, hi) in enumerate(zip(np.r_[0, edges[:-1]], edges)):
        ax.text(0.5 * (lo + hi) - 0.5, 0.42, rf"${ell}$", ha="center", va="top", fontsize=6, color=MUTED)
    ax.set_xlim(-1, x[-1] + 1)
    ax.set_ylim(-0.75, 0.45)
    ax.set_xticks([])
    ax.set_xlabel(r"$S_1^m$ coefficient ($\ell$, then $\sigma_j$, then $|m|$)")
    ax.set_ylabel(r"$(S_{\rm masked}-R\,S_{\rm ref})/\sigma$")
    ax.set_title(f"(c) after the correction, halos", loc="left")
    legend = ax.legend(frameon=True, fontsize=5.5, loc="lower left", ncol=1, handletextpad=0.2, markerscale=1.3,
                       borderaxespad=0.1, labelspacing=0.25, fancybox=False)
    legend.get_frame().set(facecolor="white", edgecolor="none", alpha=1.0)
    for a in axes:
        a.tick_params(colors=INK)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "survey.pdf", bbox_inches="tight")
    fig.savefig(PNG / "survey.png", dpi=200, bbox_inches="tight")


HALO_WINDOW = SURVEY / "dr1_window_products_halos_v2_recal"
HALO_PAIRS = [SURVEY / f"dr1_halos_v2_{t}/partial_halos.npz" for t in "abcd"]
HALO_TEST = 275  # pairs >= HALO_TEST are held out of the calibration of delta and K


def halo_window_factors(cap="NGC"):
    """Window factor of the model per S1m entry, W = S_survey model / S_box model, at the Quijote truth and the halo
    nuisances of the box baseline fit (outputs/inference/rsd/fast_noap_V60), cached."""
    import importlib.util
    import json
    import shlex

    cache = PNG / f"survey_halo_window_{cap}.npz"
    if cache.exists():
        f = np.load(cache)
        return f["labels"], f["factor"]
    from desilike import build, get_params

    from wstfast.survey import SurveyWindow

    spec = importlib.util.spec_from_file_location("fit_rsd", ROOT / "scripts/fit_rsd.py")
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    truth = dict(omega_cdm=0.120925743885, logA=3.061143632659324, h=0.6711)
    best = json.loads((ROOT / "outputs/inference/rsd/fast_noap_V60/summary.json").read_text())["bestfit"]
    args = fr.parse_args(["--data-dir", str(SURVEY / "halos_v2_w4_275-299"), "--emulator",
                          str(ROOT / "outputs/emulators/survey_lrg_z0.51.h5")]
                         + shlex.split("--stats s1m --cubic-biases relation --fog shared --vary omega_cdm logA h "
                                       "--output-dir x"))
    window = SurveyWindow.load(HALO_WINDOW / f"window_z0.4-0.6_{cap}.npz")
    out = {}
    for name, w in (("box", None), ("survey", window)):
        problem = fr.build_problem(args, space=cap, window=w, prefix=f"{cap}:")
        theory = problem["theories"][0]
        names = get_params(theory).select(varied=True, derived=False).names()
        point = {n: truth.get(n, best.get(n, float(get_params(theory)[n].value))) for n in names}
        out[name] = np.asarray(build(theory)(point))
    labels = np.array([label.split(":")[1] for label in problem["labels"]])
    PNG.mkdir(parents=True, exist_ok=True)
    np.savez(cache, labels=labels, factor=out["survey"] / out["box"])
    return labels, out["survey"] / out["box"]


def fig_halo_window(cap="NGC"):
    """Box, survey and window-corrected box data vectors of the halo cut-sky test, and their residuals."""
    from wstfast.survey import SurveyWindow

    parts = [np.load(path) for path in HALO_PAIRS]
    pairs = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    test, calibration = pairs["realizations"] >= HALO_TEST, pairs["realizations"] < HALO_TEST
    labels, factor = halo_window_factors(cap)
    window = SurveyWindow.load(HALO_WINDOW / f"window_z0.4-0.6_{cap}.npz")
    w1 = {tuple(int(t[1:]) for t in label.split("_")[1:]): f for label, f in zip(labels, factor)}  # (j, l, |m|)
    # S1m entries in the order l, sigma_j, |m| (as flat); S21m entries (j1, j2) = (0, 3), l >= 1, then |m1|, |m2|
    s1_keys = [(j, ell, m) for ell in range(5) for j in range(4) for m in range(ell + 1)]
    s21_all = [(ell, m1, m2) for ell in range(5) for m1 in range(ell + 1) for m2 in range(ell + 1)]
    s21_keys = [k for k in s21_all if k[0] >= 1]
    s21_index = [s21_all.index(k) for k in s21_keys]
    survey1 = np.array([pairs[cap][test, j, ell, m] for j, ell, m in s1_keys]).T
    box1 = np.array([pairs[f"{cap}_box"][test, j, ell, m] for j, ell, m in s1_keys]).T
    corrected1 = box1 * np.array([w1[k] for k in s1_keys])
    # S21m: survey S2m / S1m_survey; box S2m / S1m_box; corrected = K S2m_box / (W S1m_box)
    s1_survey0 = np.array([pairs[cap][test, 0, ell, m1] for ell, m1, _ in s21_keys]).T
    s1_box0 = np.array([pairs[f"{cap}_box"][test, 0, ell, m1] for ell, m1, _ in s21_keys]).T
    survey21 = pairs[f"{cap}_measured"][test][:, s21_index]
    box21 = pairs[f"{cap}_none"][test][:, s21_index] * s1_survey0 / s1_box0
    keys21 = [(SurveyWindow.key(25.0), SurveyWindow.key(70.7), ell, m1, m2) for ell, m1, m2 in s21_keys]
    k21 = np.array([window.s21m[k] for k in keys21])
    corrected21 = box21 * k21 / np.array([w1[0, ell, m1] for ell, m1, _ in s21_keys])

    fig, axes = plt.subplots(3, 2, figsize=(FULL, 4.4), sharex="col", sharey="row",
                             gridspec_kw=dict(height_ratios=[1.5, 1, 1], hspace=0.08, wspace=0.05,
                                              width_ratios=[60, 54]))
    stats = {}
    for col, (name, survey, box, corrected, groups) in enumerate((
            (r"S_1^m", survey1, box1, corrected1, [k[1] for k in s1_keys]),
            (r"S_{21}^m", survey21, box21, corrected21, [k[0] for k in s21_keys]))):
        x = np.arange(survey.shape[1])
        mean_s, mean_b, mean_c = survey.mean(0), box.mean(0), corrected.mean(0)
        sigma_one = survey.std(0, ddof=1)  # one NGC survey
        residual = corrected / survey - 1  # paired, per pair
        r_mean, r_err = residual.mean(0), residual.std(0, ddof=1) / np.sqrt(len(residual))
        z, z_err = r_mean * mean_s / sigma_one, r_err * mean_s / sigma_one
        stats[name] = dict(rms_pct=100 * np.sqrt(np.mean(r_mean**2)), max_pct=100 * np.max(np.abs(r_mean)),
                           z_max=np.max(np.abs(z)), chi2=np.sum((r_mean / r_err) ** 2), n=len(x),
                           window_min=100 * np.min(1 - mean_s / mean_b), window_max=100 * np.max(1 - mean_s / mean_b))
        ax = axes[0, col]
        ax.plot(x, mean_b, color=MUTED, lw=0.9, ls="--", label=r"parent box (same cells, no mask)")
        ax.plot(x, mean_c, color=SERIES[1], lw=1.1, label=r"box $\times$ window (model)")
        ax.plot(x, mean_s, ls="none", marker="o", ms=2.3, color=SERIES[0], mec="white", mew=0.3,
                label=r"NGC footprint")
        ax.set_yscale("log")
        ax.set_title(rf"${name}$", fontsize=8)
        ax = axes[1, col]
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.plot(x, 100 * (mean_b / mean_s - 1), color=MUTED, lw=0.9, ls="--")
        ax.plot(x, 100 * (mean_c / mean_s - 1), color=SERIES[1], lw=1.1)

        ax = axes[2, col]
        ax.fill_between(x, -z_err, z_err, color=BAND, lw=0, step="mid", label=r"error of the mean (25 mocks)")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.plot(x, z, ls="none", marker="o", ms=2.3, color=SERIES[1], mec="white", mew=0.3)
        ax.set_ylim(-0.5, 0.5)

        edges = [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
        lo = [0] + edges
        hi = edges + [len(groups)]
        for row in axes[:, col]:
            for e in edges:
                row.axvline(e - 0.5, color=GRID, lw=0.6, zorder=0)
            row.set_xlim(-1, len(x))
            row.tick_params(colors=INK)
            for side in ("top", "right"):
                row.spines[side].set_visible(False)
        axes[2, col].set_xticks([0.5 * (a + b) - 0.5 for a, b in zip(lo, hi)],
                                [rf"${groups[a]}$" for a in lo])
        axes[2, col].tick_params(axis="x", length=0)
        axes[2, col].set_xlabel(r"$\ell$" + (r" (then $\sigma_j$, then $|m|$)" if col == 0 else
                                             r" (then $|m_1|$, $|m_2|$); $(\sigma_1,\sigma_2)=(25,71)\,h^{-1}\mathrm{Mpc}$"))
    axes[0, 0].set_ylabel(r"data vector $X$")
    axes[1, 0].set_ylabel(r"$X/X_{\rm NGC}-1$ [\%]")
    axes[2, 0].set_ylabel(r"$(X_{\rm win}-X_{\rm NGC})/\sigma$")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=3, fontsize=7, frameon=False,
               handlelength=2.0, columnspacing=1.5)
    axes[2, 1].legend(loc="lower right", fontsize=6, frameon=False)
    fig.align_ylabels(axes[:, 0])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "survey_halo_window.pdf", bbox_inches="tight")
    fig.savefig(PNG / "survey_halo_window.png", dpi=200, bbox_inches="tight")
    for name, st in stats.items():
        print(name, {k: round(float(v), 3) for k, v in st.items()})
    return stats


if __name__ == "__main__":
    import sys

    fig_halo_window() if "--halo-window" in sys.argv else main()
