#!/usr/bin/env python
"""Cost of marginalising a second-layer non-Gaussianity amplitude: Fisher errors on (omega_cdm, logA).

The model of the paper (cs2 and one noise amplitude a_{j1,l} per first-layer field) is extended by
S21 -> S21 [1 + sum_p c_p t(j1, j2, l)] with a template t from scripts/feasibility/second_layer_gaussian.py: the
measured N-body deficit S2/S2_G - 1 minus its Gaussian (no-gravity) part, which has the shape of the tree-level
gravitational term (their ratio is constant at fixed (j1, l)) but no Monte Carlo noise. The Fisher degradation does
not depend on the normalisation of t. Cases differ in how the amplitudes c_p are shared:

    per field   one c per first-layer field (j1, l) with more than one S21 coefficient (a field with a single
                coefficient, e.g. 25 -> 70.7 Mpc/h, is exactly degenerate with its noise amplitude)
    per l       one c per multipole
    global      a single c
    layer2      the existing phenomenological --layer2 factor (one C_l per l), for comparison

each optionally with a Gaussian prior of the given fractional width around the template amplitude. Example:

    python scripts/feasibility/fisher_layer2.py           # template with the N-body shape
    python scripts/feasibility/fisher_layer2.py --tree    # template = the tree-level term d2 (realistic model shape)
    python scripts/feasibility/fisher_layer2.py --scan    # baseline extended to adjacent pairs and/or sigma_j1 = 8.8
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/paper"), str(ROOT / "scripts")]

import common  # noqa: E402
import wstfast.theory  # noqa: E402,F401
from desilike import build  # noqa: E402
from fisher import STEPS, marginalised_errors  # noqa: E402
from wstfast.calculators import WSTTheory  # noqa: E402
from wstfast.inference import load_emulated_basis  # noqa: E402


def jacobian(graph, center, names):
    rows = []
    for name in names:
        step = 0.5 if name.startswith("layer2") else STEPS["noise" if name.startswith("noise") else name]
        plus, minus = dict(center, **{name: center[name] + step}), dict(center, **{name: center[name] - step})
        rows.append((np.asarray(graph(plus)) - np.asarray(graph(minus))) / (2 * step))
    return np.array(rows)


def main():
    pairs = json.loads((ROOT / "outputs/feasibility/second_layer_gaussian.json").read_text())["pairs"]
    # --tree: the template is the measured tree-level term d2 itself (a model with the tree shape), not the N-body shape.
    key = "tree" if "--tree" in sys.argv else None
    template = {(r["j1"], r["j2"], r["ell"]): r[key] if key else r["nbody"] - r["gauss"] for r in pairs}
    truth = {(r["j1"], r["j2"], r["ell"]): r["nbody"] - r["gauss"] for r in pairs}
    gauss = {(r["j1"], r["j2"], r["ell"]): r["gauss"] for r in pairs}
    ncosmo = len(common.COSMOLOGY)
    fiducial = dict(common.TRUTH)
    results = {}
    # --scan: the baseline extended to smaller scales instead of baseline / conservative.
    vectors = ({"baseline": {}, "+ adjacent pairs": dict(s21_min_ratio=2.0), "+ sigma_j1 = 8.8": dict(s21_min_scale=8.8),
                "+ both": dict(s21_min_ratio=2.0, s21_min_scale=8.8)} if "--scan" in sys.argv
               else {"baseline": None, "conservative": None})
    for vector, overrides in vectors.items():
        coefficients = common.coefficients(vector) if overrides is None else common.coefficients("baseline", **overrides)
        dataset = common.dataset(coefficients)
        nreal = dataset.vectors.shape[0]
        covariance = np.cov(dataset.vectors, rowvar=False)
        basis, _ = load_emulated_basis(common.EMULATOR, dataset, common.COSMOLOGY)
        s21 = [c for c in coefficients if c.kind == "S21"]
        fields = sorted({(c.j, c.ell) for c in s21})
        names = list(common.COSMOLOGY) + ["cs2"] + [f"noise_j{j}_l{ell}" for j, ell in fields]
        center = dict(fiducial, **{name: 0.0 for name in names[ncosmo:]})
        graph = build(WSTTheory(coefficients, config=dataset.config, basis=basis))
        base = jacobian(graph, center, names)
        model = np.asarray(graph(center))

        def rows(group):
            """One Jacobian row per amplitude: d model / d c = model * t on the S21 entries of the group."""
            counts = {}
            for c in s21:
                counts[group(c)] = counts.get(group(c), 0) + 1
            # A group with one coefficient is exactly degenerate with that field's noise amplitude: no new freedom.
            groups = sorted(g for g, n in counts.items() if n > 1)
            out = np.zeros((len(groups), len(coefficients)))
            for i, c in enumerate(coefficients):
                if c.kind == "S21" and group(c) in groups:
                    out[groups.index(group(c)), i] = model[i] * template[c.j, c.j2, c.ell]
            return out

        cases = {"nominal": (base, 0)}
        for label, group in (("per field", lambda c: (c.j, c.ell)), ("per l", lambda c: c.ell), ("global", lambda c: 0)):
            extra = rows(group)
            cases[label] = (np.concatenate([base, extra]), len(extra))
        graph2 = build(WSTTheory(coefficients, config=dataset.config, basis=basis, layer2=True))
        names2 = names + [f"layer2_l{ell}" for ell in sorted({c.ell for c in s21})]
        center2 = dict(center, **{name: 10.0 for name in names2[len(names):]})
        cases["layer2 (C_l, existing)"] = (jacobian(graph2, center2, names2), len(names2) - len(names))

        print(f"\n{vector}: {len(coefficients)} coefficients, {len(fields)} first-layer fields, {nreal} boxes, V = 1 (Gpc/h)^3")
        print(f"  {'amplitudes':26s} {'n':>3s} {'prior':>7s}  {'sigma(ocdm)':>11s} {'ratio':>6s}  {'sigma(logA)':>11s} {'ratio':>6s}")
        results[vector] = {}
        reference = None
        for label, (jac, nextra) in cases.items():
            for prior in ([None] if nextra == 0 or label.startswith("layer2") else [None, 0.5, 0.2]):
                fisher_prior = None
                if prior is not None:  # Gaussian prior of width prior x (template amplitude = 1) on each c
                    jac_full = jac
                    precision = np.linalg.inv(covariance) * (nreal - covariance.shape[0] - 2.0) / (nreal - 1.0)
                    fisher = jac_full @ precision @ jac_full.T
                    fisher[-nextra:, -nextra:] += np.eye(nextra) / prior**2
                    errors = np.sqrt(np.diag(np.linalg.inv(fisher))[:ncosmo])
                else:
                    errors, _ = marginalised_errors(jac, covariance, nreal, ncosmo, fisher_prior)
                if reference is None:
                    reference = errors
                tag = "flat" if prior is None else f"{100 * prior:.0f}%"
                print(f"  {label:26s} {nextra:3d} {tag if nextra else '':>7s}  {errors[0]:11.5f} {errors[0] / reference[0]:6.2f}"
                      f"  {errors[1]:11.4f} {errors[1] / reference[1]:6.2f}")
                results[vector][f"{label}, {tag}"] = dict(n=nextra, sigma=dict(zip(common.COSMOLOGY, errors.tolist())),
                                                          ratio=(errors / reference).tolist())
        # Fisher bias from the unmodelled deficit: delta theta = F^-1 J P delta d, delta d = S21 x (full N-body deficit),
        # in units of sigma at V = 1 (it grows as sqrt(V)), for the nominal model and each amplitude choice.
        deficit = np.zeros(len(coefficients))
        for i, c in enumerate(coefficients):
            if c.kind == "S21":
                deficit[i] = model[i] * (truth[c.j, c.j2, c.ell] + gauss[c.j, c.j2, c.ell])
        precision = np.linalg.inv(covariance) * (nreal - covariance.shape[0] - 2.0) / (nreal - 1.0)
        print("  Fisher bias from the second-layer deficit [sigma at V=1; x sqrt(V)]:")
        for label, (jac, nextra) in cases.items():
            if label.startswith("layer2"):
                continue
            fisher = jac @ precision @ jac.T
            inverse = np.linalg.inv(fisher)
            shift = inverse @ jac @ precision @ deficit
            sigma = np.sqrt(np.diag(inverse))
            ratio = shift[:ncosmo] / sigma[:ncosmo]
            print(f"    {label:12s} " + "  ".join(f"{n}: {r:+.2f}" for n, r in zip(common.COSMOLOGY, ratio))
                  + "   V for 0.5 sigma: " + f"{(0.5 / max(abs(ratio))) ** 2:.0f}" + " (Gpc/h)^3")
            results[vector][f"bias {label}"] = ratio.tolist()
        # How degenerate are the per-field amplitudes with the noise amplitudes?
        jac = cases["per field"][0]
        precision = np.linalg.inv(covariance) * (nreal - covariance.shape[0] - 2.0) / (nreal - 1.0)
        cov = np.linalg.inv(jac @ precision @ jac.T)
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        kept = [f for f in fields if sum(1 for c in s21 if (c.j, c.ell) == f) > 1]
        a0, c0 = ncosmo + 1, ncosmo + 1 + len(fields)
        pair = [corr[a0 + fields.index(f), c0 + i] for i, f in enumerate(kept)]
        print("  corr(a, c) per field " + " ".join(f"j{j}l{ell}:{v:+.2f}" for (j, ell), v in zip(kept, pair)))
        results[vector]["corr_a_c"] = dict(zip([f"j{j}_l{ell}" for j, ell in kept], pair))
    out = ROOT / f"outputs/feasibility/fisher_layer2{'_tree' if key else ''}{'_scan' if '--scan' in sys.argv else ''}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
