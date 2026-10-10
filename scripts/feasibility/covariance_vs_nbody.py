#!/usr/bin/env python
"""Compare an analytic covariance (scripts/analytic_covariance.py) with the sample covariance of the N-body boxes.

Reports, for ln(data vector): the ratio of standard deviations, the correlation differences by block, the smallest
eigenvalues of the correlation matrices, the marginalised Fisher errors on the cosmological parameters (emulated model
Jacobian, cs2 and noise amplitudes marginalised; Hartlap for the sample covariance) and the mean chi2 of the N-body
realizations about their mean under each precision matrix (expected: the number of coefficients). Optionally also
the sample covariance of Gaussian-field mocks (scripts/feasibility/gaussian_mocks_wst.py). Writes a JSON summary and
the figure of the correlation matrices next to the analytic file. Example:

    python scripts/feasibility/covariance_vs_nbody.py outputs/covariance/analytic_q0.8.npz \
        --mocks outputs/covariance/gaussian_mocks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wstfast.theory  # noqa: F401,E402  (enables JAX double precision)
from wstfast.config import QUIJOTE_COSMOLOGY, Coefficient  # noqa: E402
from wstfast.data import load_measurement  # noqa: E402

DATA = Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256")


def parse_label(label):
    parts = label.split("_")
    if parts[0] == "S1":
        return Coefficient("S1", int(parts[2][1:]), int(parts[1][1:]))
    return Coefficient("S21", int(parts[3][1:]), int(parts[1][1:]), int(parts[2][1:]))


def log_vector(s1, s2, coefficients):
    return np.array([np.log(s1[c.j, c.ell]) if c.kind == "S1" else np.log(s2[c.j, c.j2, c.ell] / s1[c.j, c.ell])
                     for c in coefficients])


def load_vectors(files, coefficients, q):
    vectors = []
    for path in files:
        m = load_measurement(path, q=q)
        vectors.append(log_vector(m["S1"], m["S2"], coefficients))
    return np.array(vectors)


def load_mocks(directory, coefficients, q, max_sigma=8.0):
    """ln data vectors of the mocks that both passes reproduce (if pass2/ exists), robust outlier cut otherwise."""
    directory = Path(directory)
    vectors = []
    for path in sorted(directory.glob("mock_*.npz")):
        with np.load(path) as a:
            twin = directory / "pass2" / path.name
            if twin.exists():
                with np.load(twin) as b:
                    if not all(np.allclose(a[k], b[k], rtol=1e-5, equal_nan=True) for k in ("S1", "S2")):
                        continue
            iq = int(np.flatnonzero(np.isclose(a["q"], q))[0])
            vectors.append(log_vector(a["S1"][iq], a["S2"][iq], coefficients))
    x = np.array(vectors)
    median = np.median(x, axis=0)
    deviation = np.abs(x - median) / (1.4826 * np.median(np.abs(x - median), axis=0))
    return x[(deviation < max_sigma).all(axis=1)]


def correlation(c):
    sd = np.sqrt(np.diag(c))
    return c / np.outer(sd, sd)


def jacobian_ln(data_dir, coefficients, q, emulator, mean):
    """d ln S / d theta for the cosmological parameters, cs2 and the noise amplitudes (emulated model)."""
    from desilike import build
    from fisher import derivatives

    from wstfast.calculators import WSTTheory
    from wstfast.data import load_dataset
    from wstfast.inference import load_emulated_basis

    dataset = load_dataset(data_dir, "real", coefficients, q=q)
    basis, _ = load_emulated_basis(emulator, dataset, ("omega_cdm", "logA"))
    graph = build(WSTTheory(coefficients, config=dataset.config, basis=basis))
    names = ["omega_cdm", "logA", "cs2"] + sorted({f"noise_j{c.j}_l{c.ell}" for c in coefficients if c.kind == "S21"})
    center = dict({n: QUIJOTE_COSMOLOGY[n] for n in names[:2]}, **{n: 0.0 for n in names[2:]})
    jac = derivatives(lambda p: np.asarray(graph(p)), center, names) / mean
    return jac


def plot(path, labels, analytic, nbody, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    coefficients = [parse_label(label) for label in labels]

    def tick(c):
        if c.kind == "S1":
            return f"{config.sigma(c.j):.0f} ℓ{c.ell}"
        return f"{config.sigma(c.j):.1f}→{config.sigma(c.j2):.0f} ℓ{c.ell}"

    ticks = [tick(c) for c in coefficients]
    ns1 = sum(c.kind == "S1" for c in coefficients)
    bounds = [i for i in range(1, len(coefficients))
              if coefficients[i].ell != coefficients[i - 1].ell or i == ns1]
    cmap = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])
    a, n = correlation(analytic), correlation(nbody)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.6), constrained_layout=True)
    for ax, (matrix, title, vmax) in zip(axes, [(a, "Analytic", 1.0), (n, "N-body sample", 1.0),
                                                (a - n, "Analytic − N-body", 0.3)]):
        image = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        for b in bounds:
            lw, color = (1.6, "#333") if b == ns1 else (0.6, "#999")
            ax.axhline(b - 0.5, color=color, lw=lw)
            ax.axvline(b - 0.5, color=color, lw=lw)
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(ticks)))
        ax.set_yticklabels(ticks, fontsize=6.5)
        ax.set_title(title, fontsize=13, loc="left", pad=22)
        for x, text in ((ns1 / 2, "S1(σ, ℓ)"), (ns1 + (len(ticks) - ns1) / 2, "S21(σ1→σ2, ℓ)")):
            ax.text(x, -1.2, text, ha="center", va="bottom", fontsize=10, color="#333")
        bar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.01)
        bar.set_label("correlation" if vmax == 1 else "Δ correlation", fontsize=9)
    fig.suptitle("Correlation matrices of ln(data vector); σ in Mpc/h, thin lines separate ℓ, thick line S1 | S21",
                 fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analytic", type=Path, help="output of scripts/analytic_covariance.py")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--mocks", type=Path, default=None, help="directory of Gaussian-field mocks")
    parser.add_argument("--emulator", type=Path, default=Path("outputs/emulators/wst_basis_taylor_superset.h5"),
                        help="emulator for the Fisher Jacobian ('none' to skip the Fisher comparison)")
    args = parser.parse_args()

    analytic = np.load(args.analytic)
    labels, q = list(analytic["labels"]), float(analytic["q"])
    coefficients = [parse_label(label) for label in labels]
    files = sorted((args.data_dir / "real").glob("wst_r*.npz"))
    config = load_measurement(files[0], q=q)["config"]
    x = load_vectors(files, coefficients, q)
    covs = {"N-body": np.cov(x, rowvar=False), "analytic": analytic["lncov"]}
    nreal = {"N-body": len(x), "analytic": np.inf}
    if args.mocks is not None:
        xg = load_mocks(args.mocks, coefficients, q)
        covs["Gaussian mocks"], nreal["Gaussian mocks"] = np.cov(xg, rowvar=False), len(xg)
    s1 = np.array([c.kind == "S1" for c in coefficients])
    iu = np.triu_indices(len(labels), 1)
    summary = dict(analytic=str(args.analytic), q=q, nreal={k: (None if np.isinf(v) else v) for k, v in nreal.items()},
                   cases={})
    print(f"{len(labels)} coefficients, q = {q}, {len(x)} N-body boxes "
          f"(sampling error of a std ~{1 / np.sqrt(2 * (len(x) - 1)):.3f}, of a correlation ~{1 / np.sqrt(len(x)):.3f})")
    reference = covs["N-body"]
    for name, c in covs.items():
        if name == "N-body":
            continue
        ratio = np.sqrt(np.diag(c) / np.diag(reference))
        diff = correlation(c) - correlation(reference)
        blocks = {}
        for block, (a, b) in {"S1-S1": (s1, s1), "S1-S21": (s1, ~s1), "S21-S21": (~s1, ~s1)}.items():
            mask = np.outer(a, b) & ~np.eye(len(labels), dtype=bool)
            if mask.any():
                blocks[block] = float(np.sqrt(np.mean(diff[mask] ** 2)))
        line = (f"{name:15s} std / N-body: S1 median {np.median(ratio[s1]):.3f}" if s1.any() else f"{name:15s}")
        if (~s1).any():
            line += f"  S21 median {np.median(ratio[~s1]):.3f} [{ratio[~s1].min():.2f}, {ratio[~s1].max():.2f}]"
        print(line + f"  corr rms {np.sqrt(np.mean(diff[iu] ** 2)):.3f} "
              + " ".join(f"{k} {v:.3f}" for k, v in blocks.items()))
        summary["cases"][name] = dict(std_ratio=ratio.tolist(), corr_rms=float(np.sqrt(np.mean(diff[iu] ** 2))),
                                      corr_rms_blocks=blocks)
    for name, c in covs.items():
        eig = np.linalg.eigvalsh(correlation(c))
        print(f"{name:15s} smallest correlation eigenvalues " + " ".join(f"{e:.1e}" for e in eig[:6]))
        summary["cases"].setdefault(name, {})["eigenvalues"] = eig.tolist()

    if str(args.emulator).lower() != "none":
        mean = np.exp(x.mean(axis=0))
        jac = jacobian_ln(args.data_dir, coefficients, q, args.emulator, mean)
        jac = jac[np.abs(jac).sum(axis=1) > 0]
        dx = x - x.mean(axis=0)
        for name, c in covs.items():
            n, N = len(labels), nreal[name]
            hartlap = 1.0 if np.isinf(N) else (N - n - 2.0) / (N - 1.0)
            precision = np.linalg.inv(c)
            inverse = np.linalg.inv(jac @ (hartlap * precision) @ jac.T)
            sigma = np.sqrt(np.diag(inverse)[:2])
            chi2 = np.einsum("ni,ij,nj->n", dx, precision, dx).mean() * len(x) / (len(x) - 1)
            print(f"{name:15s} sigma(omega_cdm) {sigma[0]:.5f}  sigma(logA) {sigma[1]:.4f}  "
                  f"r {inverse[0, 1] / sigma.prod():+.2f}  <chi2 of N-body boxes> {chi2:.1f} / {n}")
            summary["cases"].setdefault(name, {}).update(sigma=dict(omega_cdm=sigma[0], logA=sigma[1]),
                                                         chi2_mean=chi2)
    figure = args.analytic.with_name(args.analytic.stem + "_correlations.png")
    plot(figure, labels, covs["analytic"], reference, config)
    output = args.analytic.with_name(args.analytic.stem + "_vs_nbody.json")
    output.write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {figure} and {output}")


if __name__ == "__main__":
    main()
