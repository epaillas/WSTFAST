#!/usr/bin/env python
"""Bias of the halo WST baseline from the unmodelled non-Gaussianity of the second layer (linear Fisher estimate).

The baseline likelihood of the paper (fit_rsd.py --stats s1m s21m --s21-min-scale 25 --s21-white, cubic biases from the
relation, omega_cdm, logA, h free, n_s fixed, sample covariance of the 1188 halo boxes) is linearised at the best fit of
outputs/inference/rsd/paper_c_wst_nsfixed_V25. The data vector is shifted by the measured deficit of every S21m
coefficient, delta d = S21m x (S2m / S2m_G - 1), with S2m_G the chi-distribution value from the exact <|Y|^2> (stored
q = 2 moment) of the same box, and the shift of the parameters, F^-1 J^T C^-1 delta d, is quoted in units of the
marginalised error at V = 25 and 60 (Gpc/h)^3. All nuisance parameters (biases, counterterms, W, noise amplitudes) are
marginalised. Example:

    python scripts/feasibility/fisher_layer2_rsd.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import wstfast.theory  # noqa: E402,F401
from desilike import build, get_params  # noqa: E402
from fit_rsd import build_problem, parse_args  # noqa: E402
from wstfast.calculators import JointTheory, WSTLikelihood  # noqa: E402
from wstfast.data import sample_covariance  # noqa: E402

DATA = ROOT / "data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los"
RUN = ROOT / "outputs/inference/rsd/paper_c_wst_nsfixed_V25"
ARGS = ["--data-dir", str(DATA), "--emulator", "outputs/emulators/rsd_biased_basis_taylor_4p_ir_sv1_cubic.h5",
        "--stats", "s1m", "s21m", "--s21-min-scale", "25", "--s21-white", "--vary", "omega_cdm", "logA", "h",
        "--cubic-biases", "relation", "--output-dir", str(ROOT / "outputs/feasibility/fisher_layer2_rsd")]
COSMO = ("omega_cdm", "logA", "h")


def measured_deficit(labels, q=0.8):
    """Mean over the halo boxes of S2m / S2m_G - 1 for every S21m label (|m2| = 0: one component, else two)."""
    files = sorted((DATA / "rsd").glob("wst_r*.npz"))
    stored = list(np.load(files[0])["q"])
    iq, i2 = stored.index(q), stored.index(2.0)
    keys = []
    for label in labels:
        if label.startswith("S21m"):
            j1, j2, ell, m1, m2 = (int(p[1:]) for p in label.split("_")[1:])
            keys.append((j1, j2, ell, m1, m2))
    total = np.zeros(len(keys))
    for path in files:
        s2 = np.load(path)["S2m"]
        for i, (j1, j2, ell, m1, m2) in enumerate(keys):
            n = 1 if m2 == 0 else 2
            gauss = (2 * s2[i2, j1, j2, ell, m1, m2] / n) ** (q / 2) * np.exp(gammaln((n + q) / 2) - gammaln(n / 2))
            total[i] += s2[iq, j1, j2, ell, m1, m2] / gauss - 1
    return dict(zip(keys, total / len(files)))


def main():
    args = parse_args(ARGS)
    problem = build_problem(args)
    theories, vectors, labels = problem["theories"], problem["vectors"], problem["labels"]
    theory = theories[0] if len(theories) == 1 else JointTheory(theories)
    covariance = sample_covariance(vectors)  # one box (Hartlap-corrected)
    likelihood = WSTLikelihood(theory, vectors.mean(axis=0), covariance)
    graph = build(likelihood)
    best = json.loads((RUN / "summary.json").read_text())["bestfit"]
    names = get_params(likelihood).select(varied=True, derived=False).names()
    center = {n: best[n] for n in names}

    def model(point):
        graph(point)
        return np.asarray(likelihood.flattheory, dtype="f8")

    m0 = model(center)
    jac = []
    for n in names:
        h = 1e-3 * max(abs(center[n]), 1e-2)
        jac.append((model(dict(center, **{n: center[n] + h})) - model(dict(center, **{n: center[n] - h}))) / (2 * h))
    jac = np.array(jac)
    deficit = measured_deficit(labels)
    shift = np.zeros(len(labels))
    for i, label in enumerate(labels):
        if label.startswith("S21m"):
            key = tuple(int(p[1:]) for p in label.split("_")[1:])
            shift[i] = m0[i] * deficit[key]
    precision = np.linalg.inv(covariance)
    fisher = jac @ precision @ jac.T
    inverse = np.linalg.inv(fisher)
    delta = inverse @ jac @ precision @ shift
    sigma1 = np.sqrt(np.diag(inverse))
    out = {}
    print(f"{len(labels)} data, {len(names)} parameters; shift from the unmodelled second-layer deficit:")
    for n in COSMO:
        i = names.index(n)
        row = {f"V{v}": float(delta[i] / (sigma1[i] / np.sqrt(v))) for v in (1, 25, 60)}
        out[n] = dict(shift=float(delta[i]), sigma_V1=float(sigma1[i]), **row)
        print(f"  {n:10s} shift {delta[i]:+.5f}   " + "  ".join(f"{k}: {x:+.2f} sigma" for k, x in row.items()))
    print("  noise amplitudes absorbing it: " + ", ".join(f"{n}: {delta[names.index(n)]:+.3f}" for n in names
                                                        if n.startswith("noise")))
    # The same shift with the noise amplitudes and W held fixed: how much of it they absorb.
    keep = [i for i, n in enumerate(names) if not (n.startswith("noise") or n.startswith("W_"))]
    sub = jac[keep]
    delta_fixed = np.linalg.inv(sub @ precision @ sub.T) @ sub @ precision @ shift
    sig_fixed = np.sqrt(np.diag(np.linalg.inv(sub @ precision @ sub.T)))
    kept = [names[i] for i in keep]
    print("  with the noise amplitudes and W fixed: " + "  ".join(
        f"{n}: {delta_fixed[kept.index(n)] / (sig_fixed[kept.index(n)] / 5):+.2f} sigma (V25)" for n in COSMO))
    out["deficit"] = {str(k): v for k, v in deficit.items()}
    path = ROOT / "outputs/feasibility/fisher_layer2_rsd.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
