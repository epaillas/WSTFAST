#!/usr/bin/env python
"""Second-layer template of the redshift-space halo WST, its calibration and the held-out prediction.

Aggregates the perturbation-theory measurements of scripts/feasibility/second_layer_rsd.py (d0: Kaiser field + shot
noise, d2: tree-level gravitational term, per (j1, j2, l, |m1|, |m2|)) and compares
    d0 + R_l d2,   R_l = 1 / (1 + c_l n s_lin^2(sigma_j1))
with the deficit S2m / S2m_G - 1 of the Quijote halo boxes (exact <|Y|^2> of each box). n s_lin^2 is the matter linear
variance of the first-layer field (as in the matter template, outputs/layer2/template.json). The c_l are
    halo-calibrated: fitted on every pair except the data-vector pair sigma_j1 = 25 -> 70.7 Mpc/h (held out)
    transferred:     the matter values of outputs/layer2/template.json, with no halo input
and both predict the held-out pair. Writes outputs/layer2/template_rsd.json.

--matter-real: the template of matter in real space (second_layer_rsd.py --tracer matter --space real) against the 868
real-space matter boxes. The c_l are calibrated on the adjacent pairs sigma_j2 = 2 sigma_j1, which no data vector uses
(as the m-summed matter template, LAYER2_CALIBRATION = "adjacent"), stored as "matter_c", and predict the held-out
data-vector pairs sigma_j2 / sigma_j1 >= 2.8. Writes outputs/layer2/template_real.json. Example:

    python scripts/layer2/template_rsd.py
    python scripts/layer2/template_rsd.py --matter-real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs/feasibility/second_layer_rsd"
DATA = ROOT / "data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los"
NBODY = ROOT / "outputs/layer2/halo_deficit_nbody.json"
HELD_OUT = (6, 9)  # sigma_j1 = 25 -> sigma_j2 = 70.7 Mpc/h, the halo data vector
MATTER_CACHE = ROOT / "outputs/feasibility/second_layer_matter_real"
MATTER_DATA = ROOT / "data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los"
MATTER_NBODY = ROOT / "outputs/layer2/matter_real_deficit_nbody.json"
Q = 0.8


def sigma(j):
    return 3.125 * 2 ** (j / 2)


def nbody_deficit(keys, data=DATA / "rsd", cache=NBODY):
    """Mean and single-box scatter of S2m / S2m_G - 1 over the N-body boxes in ``data`` (cached in ``cache``)."""
    if cache.exists():
        cached = json.loads(cache.read_text())
        if all(repr(k) in cached for k in keys):
            return {k: cached[repr(k)] for k in keys}
    files = sorted(data.glob("wst_r*.npz"))
    stored = list(np.load(files[0])["q"])
    iq, i2 = stored.index(Q), stored.index(2.0)
    values = np.zeros((len(files), len(keys)))
    for f, path in enumerate(files):
        s2 = np.load(path)["S2m"]
        for i, (j1, j2, ell, m1, m2) in enumerate(keys):
            n = 1 if m2 == 0 else 2
            gauss = (2 * s2[i2, j1, j2, ell, m1, m2] / n) ** (Q / 2) * np.exp(gammaln((n + Q) / 2) - gammaln(n / 2))
            values[f, i] = s2[iq, j1, j2, ell, m1, m2] / gauss - 1
    out = {k: dict(mean=float(values[:, i].mean()), std=float(values[:, i].std(ddof=1)), n=len(files))
           for i, k in enumerate(keys)}
    cache.write_text(json.dumps({repr(k): v for k, v in out.items()}, indent=1))
    return out


def calibrate(rows, ells):
    """c_l by weighted least squares of the N-body deficit minus d0 against d2 / (1 + c n s^2)."""
    out = {}
    for ell in ells:
        sel = [r for r in rows if r["ell"] == ell]
        target = np.array([r["nbody"] - r["gauss"] for r in sel])
        tree, var = np.array([r["tree"] for r in sel]), np.array([r["var"] for r in sel])
        err = np.hypot(np.array([r["nbody_std"] / np.sqrt(r["nbody_n"]) for r in sel]),
                       np.hypot([r["tree_err"] for r in sel], [r["gauss_err"] for r in sel]))

        def chi2(c):
            return float(np.sum(((target - tree / (1 + c * var)) / err) ** 2))

        best = minimize_scalar(chi2, bounds=(-0.5 / var.max(), 500.0), method="bounded").x
        h = 1e-3 * max(abs(best), 1.0)
        curv = (chi2(best + h) - 2 * chi2(best) + chi2(best - h)) / h**2
        out[ell] = (float(best), float(np.sqrt(2 / max(curv, 1e-12)) * np.sqrt(max(chi2(best) / max(len(sel) - 1, 1), 1))),
                    chi2(best), len(sel))
    return out


def aggregate(cache, data, nbody_cache):
    """Template rows (seed means and errors of d0, d2, N-body deficit) and the seed files."""
    seeds = [json.loads(p.read_text()) for p in sorted(cache.glob("seed_*.json"))]
    # Extra seeds of single pairs (second_layer_rsd.py --only), pooled with the all-pair ones where they overlap.
    for extra in sorted(cache.parent.glob(cache.name + "_*")):
        seeds += [json.loads(p.read_text()) for p in sorted(extra.glob("seed_*.json"))]
    keys = sorted((eval(k) for k in seeds[0]["gauss"]), key=lambda k: (k[2], k[0], k[1], k[3], k[4]))
    matter = json.loads((ROOT / "outputs/layer2/template.json").read_text())
    var = {(r["j1"], r["ell"]): r["var_ref"] for r in matter["coefficients"].values()}
    nbody = nbody_deficit(keys, data, nbody_cache)
    rows = []
    for key in keys:
        j1, j2, ell, m1, m2 = key
        d0 = np.array([s["gauss"][repr(key)] for s in seeds if repr(key) in s["gauss"]])
        d2 = np.array([s["tree"][repr(key)] for s in seeds if repr(key) in s["tree"]])
        rows.append(dict(key=key, j1=j1, j2=j2, ell=ell, m1=m1, m2=m2, var=var[j1, ell],
                         gauss=float(d0.mean()), gauss_err=float(d0.std(ddof=1) / np.sqrt(d0.size)),
                         tree=float(d2.mean()), tree_err=float(d2.std(ddof=1) / np.sqrt(d2.size)),
                         nbody=nbody[key]["mean"], nbody_std=nbody[key]["std"], nbody_n=nbody[key]["n"],
                         nseeds=int(d2.size)))
    return rows, seeds, matter


def main_matter_real(volume=64.0):
    """Matter in real space: c_l from the adjacent pairs, prediction of the data-vector pairs (ratio >= 2.8)."""
    rows, seeds, msummed = aggregate(MATTER_CACHE, MATTER_DATA / "real", MATTER_NBODY)
    adjacent = lambda r: r["j2"] - r["j1"] == 2  # noqa: E731  (sigma_j2 = 2 sigma_j1)
    c = calibrate([r for r in rows if adjacent(r)], range(1, 5))
    c_msummed = {int(k): v for k, v in msummed["pade_c"].items()}
    scale = np.sqrt(volume)  # single-box scatter / sqrt(V / (1 Gpc/h)^3)
    print(f"{len(seeds)} seed files, {rows[0]['nbody_n']} N-body boxes; c_l calibrated on the adjacent pairs:")
    for ell, (cl, e, x2, n) in c.items():
        print(f"  l = {ell}: c = {cl:7.2f} +- {e:5.2f} (chi2 {x2:7.1f} / {n} rows)   m-summed template c = {c_msummed[ell]:6.2f}")

    def report(selection, title):
        print(f"\n{title} [%]: mean over |m1|, |m2| of N-body, d0, d2, prediction; rms and max |pull| at V = {volume:g}"
              f" for the Gaussian second layer (d0 = d2 = 0), d0 only, and the prediction")
        pulls = {"gauss": [], "d0": [], "pred": []}
        for j1, j2 in sorted({(r["j1"], r["j2"]) for r in selection}):
            for ell in range(1, 5):
                sel = [r for r in selection if (r["j1"], r["j2"], r["ell"]) == (j1, j2, ell)]
                pred = np.array([r["gauss"] + r["tree"] / (1 + max(c[ell][0], 0) * r["var"]) for r in sel])
                nb = np.array([r["nbody"] for r in sel])
                err = np.array([np.hypot(r["nbody_std"] / scale, np.hypot(r["gauss_err"], r["tree_err"])) for r in sel])
                p = dict(gauss=nb / err, d0=(nb - np.array([r["gauss"] for r in sel])) / err, pred=(nb - pred) / err)
                for k in pulls:
                    pulls[k] += list(p[k])
                rms = "  ".join(f"{k} {np.sqrt(np.mean(v**2)):5.1f}/{np.abs(v).max():5.1f}" for k, v in p.items())
                print(f"  {sigma(j1):4.1f} -> {sigma(j2):4.1f}  l={ell}  {100 * nb.mean():+6.2f}  "
                      f"{100 * np.mean([r['gauss'] for r in sel]):+6.2f}  {100 * np.mean([r['tree'] for r in sel]):+6.2f}  "
                      f"{100 * pred.mean():+6.2f}   {rms}")
        print("  all: " + "  ".join(f"{k} rms {np.sqrt(np.mean(np.square(v))):.2f}" for k, v in pulls.items()))

    report([r for r in rows if adjacent(r)], "calibration pairs (sigma_j2 = 2 sigma_j1)")
    report([r for r in rows if r["j2"] - r["j1"] >= 3], "held-out data-vector pairs (sigma_j2 / sigma_j1 >= 2.8)")
    out = dict(nseeds=len(seeds), biases=seeds[0]["meta"]["biases"], calibration="adjacent",
               matter_c={str(k): v[0] for k, v in c.items()}, matter_c_err={str(k): v[1] for k, v in c.items()},
               msummed_c={str(k): v for k, v in c_msummed.items()},
               rows=[{k: (list(v) if k == "key" else v) for k, v in r.items()} for r in rows])
    (ROOT / "outputs/layer2/template_real.json").write_text(json.dumps(out, indent=1))
    print("wrote outputs/layer2/template_real.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matter-real", action="store_true", help="the real-space matter template (template_real.json)")
    parser.add_argument("--volume", type=float, default=64.0, help="volume [(Gpc/h)^3] of the --matter-real pulls")
    args = parser.parse_args()
    if args.matter_real:
        return main_matter_real(args.volume)
    rows, seeds, matter = aggregate(CACHE, DATA / "rsd", NBODY)
    c_matter = {int(k): v for k, v in matter["pade_c"].items()}
    training = [r for r in rows if (r["j1"], r["j2"]) != HELD_OUT]
    halo_c = calibrate(training, range(1, 5))
    print(f"{len(seeds)} seed files; held-out pair from {max(r['nseeds'] for r in rows)} seeds; c_l calibrated on halos (all pairs but 25 -> 70.7) vs matter:")
    for ell, (c, e, x2, n) in halo_c.items():
        print(f"  l = {ell}: halo c = {c:7.2f} +- {e:5.2f} (chi2 {x2:6.1f} / {n} pairs)   matter c = {c_matter[ell]:6.2f}")
    print("\nheld-out pair 25 -> 70.7 [%]: N-body, Gaussian d0, tree d2, prediction (halo c), prediction (matter c);"
          " pulls at V = 25 (single-box scatter / 5)")
    held = []
    for r in rows:
        if (r["j1"], r["j2"]) != HELD_OUT:
            continue
        pred_h = r["gauss"] + r["tree"] / (1 + max(halo_c[r["ell"]][0], 0) * r["var"])
        pred_m = r["gauss"] + r["tree"] / (1 + max(c_matter[r["ell"]], 0) * r["var"])
        s25 = r["nbody_std"] / 5
        held.append(dict(r, pred_halo=pred_h, pred_matter=pred_m))
        print(f"  l={r['ell']} |m1|={r['m1']} |m2|={r['m2']}  {100 * r['nbody']:+6.2f}  {100 * r['gauss']:+6.2f}  "
              f"{100 * r['tree']:+6.2f}({100 * r['tree_err']:.2f})  {100 * pred_h:+6.2f}  {100 * pred_m:+6.2f}   "
              f"pull Gauss {(r['nbody']) / s25:+5.1f}  halo-c {(r['nbody'] - pred_h) / s25:+5.1f}  "
              f"matter-c {(r['nbody'] - pred_m) / s25:+5.1f}")
    for name, field in (("Gaussian second layer", None), ("halo c", "pred_halo"), ("matter c", "pred_matter")):
        res = np.array([(h["nbody"] - (h[field] if field else 0.0)) / (h["nbody_std"] / 5) for h in held])
        print(f"  rms pull at V = 25, {name:22s}: {np.sqrt(np.mean(res**2)):.2f}")
    out = dict(nseeds=len(seeds), biases=seeds[0]["meta"]["biases"], halo_c={str(k): v[0] for k, v in halo_c.items()},
               halo_c_err={str(k): v[1] for k, v in halo_c.items()}, matter_c={str(k): v for k, v in c_matter.items()}, matter_c_err=matter["pade_c_err"],
               rows=[{k: (list(v) if k == "key" else v) for k, v in r.items()} for r in rows])
    (ROOT / "outputs/layer2/template_rsd.json").write_text(json.dumps(out, indent=1))
    print("wrote outputs/layer2/template_rsd.json")


if __name__ == "__main__":
    main()
