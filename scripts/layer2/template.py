#!/usr/bin/env python
"""Second-layer non-Gaussianity template for the model (wstfast.theory.model.Assembly, layer2_template).

Aggregates the per-seed perturbation-theory measurements of scripts/feasibility/second_layer_uv.py (cached in
outputs/feasibility/second_layer_uv{,_j3}/seed_*.json) into, for every S21 coefficient (j1, j2, l):

    gauss    d0, S2/S2_G - 1 on the Gaussian field delta_1 (the non-Gaussianity of the modulus itself)
    tree     d2, the tree-level gravitational term at the fiducial cosmology (eps^2 coefficient), for the UV
             treatment --cut of the nonlinear SPT terms
    var_ref  n s_lin^2 of the first-layer field U_{j1,l} at the fiducial cosmology (emulated basis), by which the
             model rescales d2 with cosmology (d2 is proportional to the amplitude of P_L)

with their errors over seeds, and the N-body value for reference. Writes outputs/layer2/template.json. Example:

    python scripts/layer2/template.py --cut none
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/paper"), str(ROOT / "scripts/feasibility")]

import common  # noqa: E402
import wstfast.theory  # noqa: E402,F401
from desilike import build  # noqa: E402
from second_layer_gaussian import nbody_ratios  # noqa: E402
from wstfast.config import Coefficient  # noqa: E402
from wstfast.theory import all_coefficients  # noqa: E402

#: Per-seed caches of second_layer_uv.py: sigma_j1 >= 12.5 Mpc/h (--j1-min 4) and sigma_j1 = 8.8 Mpc/h (j1 = 3).
CACHES = (ROOT / "outputs/feasibility/second_layer_uv", ROOT / "outputs/feasibility/second_layer_uv_j3")
#: The c_l are calibrated on the first-layer fields of the data vectors (sigma_j1 >= 12.5 Mpc/h).
CALIBRATION_MIN_SIGMA1 = 12.0


def fiducial_variances(config):
    """n s_lin^2 (S1 basis column 'linear') of every S1(j, l), from the emulated basis at the truth."""
    coeffs = common.coefficients("baseline")
    ds = common.dataset(coeffs)
    basis, _ = common.emulated_basis(ds)
    out = build(basis)(dict(common.TRUTH))
    s1_terms = np.asarray(out.s1_terms if hasattr(out, "s1_terms") else basis.s1_terms)
    s1 = [c for c in all_coefficients(config) if c.kind == "S1"]
    return {(c.j, c.ell): float(s1_terms[i, 0]) for i, c in enumerate(s1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cut", default="none", help="UV treatment of the SPT terms (a cut of second_layer_uv.py)")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/layer2/template.json")
    args = parser.parse_args()

    caches = [[json.loads(p.read_text()) for p in sorted(c.glob("seed_*.json"))] for c in CACHES if c.exists()]
    caches = [[s for s in seeds if args.cut in s] for seeds in caches]
    caches = [seeds for seeds in caches if seeds]
    source = {}  # coefficient -> the seeds that measured it
    for seeds in caches:
        for k in seeds[0]["gauss"]:
            source.setdefault(eval(k), seeds)
    keys = sorted(source, key=lambda p: (p[2], p[0], p[1]))
    config = common.config()
    variances = fiducial_variances(config)
    nbody = nbody_ratios(config, keys)
    rows = {}
    for key in keys:
        j1, j2, ell = key
        seeds = source[key]
        d0 = np.array([s["gauss"][repr(key)] for s in seeds])
        d2 = np.array([s[args.cut][repr(key)] for s in seeds])
        label = Coefficient("S21", ell, j1, j2).label
        rows[label] = dict(j1=j1, j2=j2, ell=ell, sigma1=config.sigma(j1), sigma2=config.sigma(j2),
                           gauss=float(d0.mean()), gauss_err=float(d0.std(ddof=1) / np.sqrt(d0.size)),
                           tree=float(d2.mean()), tree_err=float(d2.std(ddof=1) / np.sqrt(d2.size)),
                           var_ref=variances[j1, ell], nbody=nbody[key]["mean"], nbody_std=nbody[key]["std"],
                           nseeds=len(seeds))
    payload = dict(cut=args.cut, nseeds=min(len(s) for s in caches), coefficients=rows,
                   calibration_min_sigma1=CALIBRATION_MIN_SIGMA1)
    # Calibration on every pair, and held out: on the adjacent pairs only (sigma_j2 = 2 sigma_j1, outside the baseline
    # and conservative data vectors) or on the non-adjacent ones only.
    subsets = {"": lambda r: True, "_adjacent": lambda r: r["sigma2"] / r["sigma1"] < 2.4,
               "_nonadjacent": lambda r: r["sigma2"] / r["sigma1"] >= 2.4}
    for suffix, keep in subsets.items():
        calibration = calibrate({k: r for k, r in rows.items() if keep(r) and r["sigma1"] >= CALIBRATION_MIN_SIGMA1})
        print(f"  calibration{suffix or ' (all pairs)'}: " + ", ".join(f"c_{ell} = {c:.2f} +- {e:.2f}"
                                                                     for ell, (c, e) in calibration.items()))
        payload[f"pade_c{suffix}"] = {str(ell): c for ell, (c, _) in calibration.items()}
        payload[f"pade_c_err{suffix}"] = {str(ell): e for ell, (_, e) in calibration.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}: {len(rows)} coefficients, " + ", ".join(f"{len(s)} seeds" for s in caches)
          + f", cut {args.cut}")


def calibrate(rows):
    """Per-l constant c_l of the beyond-tree suppression R = 1 / (1 + c_l n s_lin^2(sigma_j1)), fitted to the N-body
    deficit minus the Gaussian-field term over every pair of the template (weighted least squares with the
    single-box N-body scatter and the template error; errors from the curvature, scaled by sqrt(chi2/dof))."""
    from scipy.optimize import minimize_scalar

    out = {}
    for ell in sorted({r["ell"] for r in rows.values()}):
        sel = [r for r in rows.values() if r["ell"] == ell]
        target = np.array([r["nbody"] - r["gauss"] for r in sel])
        tree, var = np.array([r["tree"] for r in sel]), np.array([r["var_ref"] for r in sel])
        sigma = np.hypot(np.array([r["nbody_std"] for r in sel]) / np.sqrt(868.0),
                         np.hypot([r["tree_err"] for r in sel], [r["gauss_err"] for r in sel]))

        def chi2(c):
            return float(np.sum(((target - tree / (1 + c * var)) / sigma) ** 2))

        best = minimize_scalar(chi2, bounds=(-0.5 / var.max(), 500.0), method="bounded").x
        h = 1e-3 * max(abs(best), 1.0)
        curvature = (chi2(best + h) - 2 * chi2(best) + chi2(best - h)) / h**2
        err = np.sqrt(2.0 / max(curvature, 1e-12)) * np.sqrt(max(chi2(best) / max(len(sel) - 1, 1), 1.0))
        out[ell] = (float(best), float(err))
    return out


if __name__ == "__main__":
    main()
