#!/usr/bin/env python
"""Compare a redshift-space analytic covariance (scripts/analytic_covariance_rsd.py) with the sample covariance.

The data vector is [ln S1m blocks, P_0, P_2, P_4] of every box (as fit_rsd.py, with S1m in ln). Reports the ratio of
standard deviations per statistic, the rms correlation differences by block, the smallest correlation eigenvalues and
the mean chi2 of the boxes about their mean under each precision matrix (expected: the number of entries), and writes
the figure of the correlation matrices next to the analytic file. With ``--emulator``, also the Fisher errors on the
cosmological parameters (model and fiducial point of fisher_rsd.py; nuisances fixed or marginalised with their priors).

    python scripts/feasibility/covariance_vs_nbody_rsd.py outputs/covariance/halos_rsd_q0.8.npz \
        --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wstfast.config import select_coefficients
from wstfast.data import load_measurement, load_power_dataset, load_s1m_dataset


def fisher_errors(data_dir, emulator, labels, covariances, nreal, volume):
    """Errors on the cosmology with each covariance (linear data vector), nuisances fixed and marginalised."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from desilike import build, get_params
    from fisher_rsd import COSMOLOGY, load_fit_rsd, prior_sigma, step_of

    import wstfast.theory  # noqa: F401
    from wstfast.calculators import JointTheory
    from wstfast.config import QUIJOTE_COSMOLOGY

    fr = load_fit_rsd()
    args = fr.parse_args(["--data-dir", str(data_dir), "--emulator", str(emulator), "--output-dir", "unused",
                          "--vary", *COSMOLOGY, "--rebin", "2", "--stats", "pk", "s1m", "--kmax", "0.12", "--q", "0.8"])
    problem = fr.build_problem(args)
    assert list(problem["labels"]) == labels, "data vector of fit_rsd.py differs from the analytic file"
    theory = JointTheory(problem["theories"])
    params = get_params(theory).select(varied=True, derived=False)
    names = params.names()
    fiducial = {name: float(params[name].value) for name in names}
    fiducial.update({name: QUIJOTE_COSMOLOGY[name] for name in COSMOLOGY})
    for path in (Path("outputs/inference/rsd/halos_bias_only_V60_s1m/summary.json"),
                 Path("outputs/inference/rsd/halos_bias_only_V60_pk/summary.json")):
        if path.exists():
            best = json.loads(path.read_text())["bestfit"]
            fiducial.update({k: float(v) for k, v in best.items() if k in names and k not in COSMOLOGY})
    model = build(theory)
    jacobian = np.array([(np.asarray(model({**fiducial, n: fiducial[n] + step_of(n)}))
                          - np.asarray(model({**fiducial, n: fiducial[n] - step_of(n)}))) / (2 * step_of(n))
                         for n in names])
    cosmo = [names.index(n) for n in COSMOLOGY]
    priors = np.array([np.inf if n in COSMOLOGY else prior_sigma(params[n]) for n in names])
    used = [i for i in range(len(names)) if i in cosmo or np.any(jacobian[i] != 0)]
    out = {}
    for name, (c, hartlap) in covariances.items():
        precision = hartlap * np.linalg.inv(c / volume)
        fixed = jacobian[cosmo] @ precision @ jacobian[cosmo].T
        marg = jacobian[used] @ precision @ jacobian[used].T + np.diag(1.0 / priors[used] ** 2)
        index = [used.index(i) for i in cosmo]
        out[name] = dict(fixed=np.sqrt(np.diag(np.linalg.inv(fixed))).tolist(),
                         marg=np.sqrt(np.diag(np.linalg.inv(marg))[index]).tolist())
    print(f"Fisher errors at V = {volume:g} (Gpc/h)^3, " + ", ".join(COSMOLOGY) + ":")
    for treatment in ("fixed", "marg"):
        for name, errors in out.items():
            print(f"  {treatment:5s} {name:26s} " + "  ".join(f"{e:.4f}" for e in errors[treatment]))
    return out


def correlation(c):
    sd = np.sqrt(np.diag(c))
    return c / np.outer(sd, sd)


def plot(path, labels, analytic, nbody, groups):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])
    a, n = correlation(analytic), correlation(nbody)
    starts = [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.6), constrained_layout=True)
    for ax, (matrix, title, vmax) in zip(axes, [(a, "Analytic", 1.0), (n, "N-body sample", 1.0),
                                                (a - n, "Analytic − N-body", 0.3)]):
        image = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        for b in starts:
            ax.axhline(b - 0.5, color="#333", lw=1.0)
            ax.axvline(b - 0.5, color="#333", lw=1.0)
        centres = [(lo + hi - 1) / 2 for lo, hi in zip([0] + starts, starts + [len(groups)])]
        names = [groups[lo] for lo in [0] + starts]
        ax.set_xticks(centres)
        ax.set_xticklabels(names, fontsize=9, rotation=90)
        ax.set_yticks(centres)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(title, fontsize=13, loc="left")
        bar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.01)
        bar.set_label("correlation" if vmax == 1 else "Δ correlation", fontsize=9)
    fig.suptitle("Correlation matrices of [ln S1m(σ, ℓ, |m|), P_0, P_2, P_4]; S1m grouped by ℓ, σ and |m| inside",
                 fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analytic", type=Path)
    parser.add_argument("--rebin", type=int, default=2)
    parser.add_argument("--s1-min-scale", type=float, default=25.0)
    parser.add_argument("--emulator", type=Path, default=None, help="RSD emulator for the Fisher comparison")
    parser.add_argument("--volume", type=float, default=25.0, help="volume of the Fisher errors in (Gpc/h)^3")
    args = parser.parse_args()

    analytic = np.load(args.analytic)
    labels, q, edges = list(analytic["labels"]), float(analytic["q"]), analytic["pk_edges"]
    data_dir = Path(str(analytic["data_dir"]))
    files = sorted((data_dir / "rsd").glob("wst_r*.npz"))
    config = load_measurement(files[0], q=q)["config"]
    coefficients = [c for c in select_coefficients(config, s1_min_scale=args.s1_min_scale) if c.kind == "S1"]
    s1m = load_s1m_dataset(data_dir, "rsd", coefficients, q=q, files=files)
    power = load_power_dataset(data_dir, "rsd", kmin=0.0, kmax=edges[-1], rebin=args.rebin, files=files, ells=(0, 2, 4))
    assert np.allclose(power.edges, edges)
    x = np.concatenate([np.log(s1m.vectors), power.vectors], axis=1)
    assert x.shape[1] == len(labels), (x.shape, len(labels))
    nbody = np.cov(x, rowvar=False)
    covs = {"analytic": analytic["lncov"], "analytic, 2nd chaos only": analytic["gauss"]}
    groups = [label.split("_")[0] if label.startswith("P") else f"S1m ℓ{label.split('_l')[1][0]}" for label in labels]
    kinds = np.array([g if g.startswith("P") else "S1m" for g in groups])
    iu = np.triu_indices(len(labels), 1)
    n = len(labels)
    print(f"{n} entries ({(kinds == 'S1m').sum()} S1m, {n - (kinds == 'S1m').sum()} P_l), {len(x)} boxes "
          f"(sampling error of a std ~{1 / np.sqrt(2 * (len(x) - 1)):.3f}, of a correlation ~{1 / np.sqrt(len(x)):.3f})")
    summary = dict(analytic=str(args.analytic), nboxes=len(x), cases={})
    dx = x - x.mean(axis=0)
    hartlap = (len(x) - n - 2.0) / (len(x) - 1.0)
    for name, c in [("N-body", nbody)] + list(covs.items()):
        precision = np.linalg.inv(c) * (hartlap if name == "N-body" else 1.0)
        chi2 = np.einsum("ni,ij,nj->n", dx, precision, dx).mean() * len(x) / (len(x) - 1)
        eig = np.linalg.eigvalsh(correlation(c))
        line = f"{name:26s} <chi2 of boxes> {chi2:6.1f} / {n}   smallest corr eigenvalues " + " ".join(f"{e:.1e}" for e in eig[:4])
        case = dict(chi2_mean=chi2, eigenvalues=eig.tolist())
        if name != "N-body":
            ratio = np.sqrt(np.diag(c) / np.diag(nbody))
            diff = correlation(c) - correlation(nbody)
            line += "\n" + " " * 27 + "std / N-body: " + "  ".join(
                f"{kind} {np.median(ratio[kinds == kind]):.3f} [{ratio[kinds == kind].min():.2f}, {ratio[kinds == kind].max():.2f}]"
                for kind in ("S1m", "P0", "P2", "P4"))
            blocks = {}
            for a in ("S1m", "P"):
                for b in ("S1m", "P"):
                    if a == "P" and b == "S1m":
                        continue
                    ma, mb = np.char.startswith(kinds, a), np.char.startswith(kinds, b)
                    mask = np.outer(ma, mb) & ~np.eye(n, dtype=bool)
                    blocks[f"{a}-{b}"] = float(np.sqrt(np.mean(diff[mask] ** 2)))
            line += "\n" + " " * 27 + f"corr rms {np.sqrt(np.mean(diff[iu] ** 2)):.3f}: " + "  ".join(
                f"{k} {v:.3f}" for k, v in blocks.items())
            case.update(std_ratio=ratio.tolist(), corr_rms_blocks=blocks)
        print(line)
        summary["cases"][name] = case
    if args.emulator is not None:
        mean = x.mean(axis=0)
        scale = np.where(kinds == "S1m", np.exp(mean), 1.0)  # ln S1m -> S1m
        linear = {"N-body": (nbody * np.outer(scale, scale), hartlap)}
        linear.update({name: (c * np.outer(scale, scale), 1.0) for name, c in covs.items() if "only" not in name})
        summary["fisher"] = fisher_errors(data_dir, args.emulator, labels, linear, len(x), args.volume)
    figure = args.analytic.with_name(args.analytic.stem + "_correlations.png")
    plot(figure, labels, covs["analytic"], nbody, groups)
    output = args.analytic.with_name(args.analytic.stem + "_vs_nbody.json")
    output.write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {figure} and {output}")


if __name__ == "__main__":
    main()
