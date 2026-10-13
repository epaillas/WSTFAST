#!/usr/bin/env python
"""UV sensitivity of the tree-level second-layer non-Gaussianity.

The tree-level term d2 of S2 / S2_G - 1 (scripts/feasibility/second_layer_gaussian.py) is measured with the
nonlinear SPT fields built from a truncated linear field: delta = delta_1 + eps delta_2[delta_1^<] +
eps^2 delta_3[delta_1^<], with delta_1^< = delta_1 for k < cut and 0 above (delta_1 itself is kept whole). All
cuts share the realization, so their differences carry little sample variance. A strong dependence on the cut
means the tree-level term is dominated by UV modes, where SPT fails and a counterterm is needed. Per-seed results
are cached in outputs/feasibility/second_layer_uv/seed_*.json, so runs can be extended. Example:

    python scripts/feasibility/second_layer_uv.py --seeds 0-5 --workers 3 --cuts 0.15 0.25 0.4 none
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/feasibility"), str(ROOT / "scripts/paper")]

import common  # noqa: E402
from pt_modulus_noise import SPTFields, cic_window  # noqa: E402
from second_layer_gaussian import BOXSIZE, EPS, NMESH, Z, nbody_ratios, pairs, parse_seeds, ratios  # noqa: E402
from wstfast.config import QUIJOTE_COSMOLOGY  # noqa: E402
from wstfast.measure import Lattice, PowerMultipoles  # noqa: E402

CACHE = ROOT / "outputs/feasibility/second_layer_uv"
CACHE_DEFAULT = CACHE


def label(cut):
    return "none" if cut is None else f"{cut:g}"


def run_seed(seed, cuts, j1_min, j1_max=99, cache=None):
    from cosmoprimo import Cosmology

    cache = Path(cache or CACHE)
    path = cache / f"seed_{seed:04d}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    todo_cuts = [c for c in cuts if label(c) not in done]
    if not todo_cuts:
        return seed, done
    t0 = time.time()
    config = common.config()
    klin = np.geomspace(1e-4, 10.0, 1024)
    pklin = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY).get_fourier().pk_interpolator(
        of="delta_m")(klin, z=Z)
    pk = lambda x: np.exp(np.interp(np.log(x), np.log(klin), np.log(pklin)))  # noqa: E731
    lattice = Lattice(NMESH)
    spectra = PowerMultipoles(lattice, BOXSIZE, kmax=0.4, ells=(0,))
    todo = [p for p in pairs(config, j1_min) if p[0] <= j1_max]
    harmonics = {ell: lattice.harmonics(ell) for ell in {p[2] for p in todo}}
    window = cic_window(lattice)
    spt = SPTFields(lattice, pk, BOXSIZE, seed=10_000 + seed)
    kphys = lattice.kmag * (NMESH / BOXSIZE)
    d0 = done.get("gauss") or {repr(k): v for k, v in
                               ratios(window * spt.delta1, config, todo, lattice, spectra, harmonics).items()}
    done["gauss"] = d0
    for cut in todo_cuts:
        if cut is None:
            delta2, delta3 = spt.delta2, spt.delta3
        else:
            low = spt.delta1 * (kphys < cut)
            delta2, theta2 = spt._second(low)
            delta3 = spt._third(low, delta2, theta2)
        curve = [dict(d0)]
        for eps in EPS[1:]:
            values = [ratios(window * f, config, todo, lattice, spectra, harmonics) for f in
                      (spt.delta1 + eps * delta2 + eps**2 * delta3, -spt.delta1 + eps * delta2 - eps**2 * delta3)]
            curve.append({repr(k): float(np.mean([v[k] for v in values])) for k in values[0]})
        design = np.stack([np.ones(len(EPS)), np.square(EPS), np.power(EPS, 4)], axis=1)
        done[label(cut)] = {key: float(np.linalg.solve(design, [c[key] for c in curve])[1]) for key in curve[0]}
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(done))
    print(f"seed {seed}: cuts {[label(c) for c in todo_cuts]} in {time.time() - t0:.0f} s", flush=True)
    return seed, done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0-5")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cuts", nargs="+", default=["0.15", "0.25", "0.4", "none"])
    parser.add_argument("--j1-min", type=int, default=4)
    parser.add_argument("--j1-max", type=int, default=99)
    parser.add_argument("--cache", type=Path, default=CACHE, help="per-seed cache (one per range of j1)")
    args = parser.parse_args()
    cuts = [None if c == "none" else float(c) for c in args.cuts]
    seeds = parse_seeds(args.seeds)
    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(pool.map(run_seed, seeds, [cuts] * len(seeds), [args.j1_min] * len(seeds),
                                [args.j1_max] * len(seeds), [args.cache] * len(seeds)))
    config = common.config()
    keys = sorted((eval(k) for k in results[seeds[0]]["gauss"]), key=lambda p: (p[2], p[0], p[1]))
    nbody = nbody_ratios(config, keys)
    print(f"\ntree-level d2 [%] vs UV cut of the nonlinear terms (k_cut in h/Mpc), {len(seeds)} seeds; "
          "N-body - Gaussian for reference")
    print("pair              " + "".join(f"{label(c):>14s}" for c in cuts) + "   N-body - d0")
    for key in keys:
        j1, j2, ell = key
        d0 = np.mean([results[s]["gauss"][repr(key)] for s in seeds])
        row = []
        for c in cuts:
            v = np.array([results[s][label(c)][repr(key)] for s in seeds])
            row.append(f"{100 * v.mean():+7.2f}({100 * v.std(ddof=1) / np.sqrt(len(v)):.2f})")
        print(f"{config.sigma(j1):5.1f}->{config.sigma(j2):5.1f} l={ell}  " + "".join(f"{r:>14s}" for r in row)
              + f"   {100 * (nbody[key]['mean'] - d0):+6.2f}")


if __name__ == "__main__":
    main()
