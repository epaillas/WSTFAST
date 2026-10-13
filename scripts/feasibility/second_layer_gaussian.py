#!/usr/bin/env python
"""Where does the non-Gaussianity of the second WST layer come from? Gaussian vs SPT vs N-body fields.

For each first-layer field U = |delta * psi_{j1,l}| and second-layer scale j2, the ratio S2 / S2_G - 1 compares
the measured <|Y|^q>, Y = U * psi_{j2,l}, with the chi-distribution value built from the variance of Y implied by
the exact <|Y|^2> of the same box. (The binned-P_UU variance of the original second_layer.py, evaluated at the
bin k, is biased by up to a few per cent at sigma_j2 = 70.7 Mpc/h; binned=True reproduces it.) It is measured on

    delta = delta_1 + eps delta_2 + eps^2 delta_3   (EdS SPT, delta_1 Gaussian with the linear spectrum at z = 0.5)

for a few eps, each averaged over delta_1 -> -delta_1, with the CIC window of the N-body meshes. The ratio is even
in eps; the fit d(eps) = d0 + d2 eps^2 + d4 eps^4 separates the Gaussian value d0 (the chi^2-like non-Gaussianity
of the modulus itself: ring diagrams of P, no gravity) from the tree-level gravitational term d2. All eps share
the realization, so d2 carries little sample variance. The full SPT sum (eps = 1) is not used: without shell
crossing it overshoots the N-body non-Gaussianity on the mesh. Compared with the Quijote mean (same exact
estimator, from the stored q = 2 moment). Example:

    python scripts/feasibility/second_layer_gaussian.py --seeds 0-19 --workers 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/feasibility"))
sys.path.insert(0, str(ROOT / "scripts/paper"))

import common  # noqa: E402
from pt_modulus_noise import SPTFields, cic_window  # noqa: E402
from wstfast.config import QUIJOTE_COSMOLOGY  # noqa: E402
from wstfast.measure import Lattice, PowerMultipoles, _moments, _rfft, wavelet_modulus  # noqa: E402

BOXSIZE, NMESH, Z = 1000.0, 256, 0.5
EPS = (0.0, 0.3, 0.5)


def pairs(config, j1_min):
    """The (j1, j2, l) of second_layer.py (second-layer filter inside k < 0.4 h/Mpc), with j1 >= j1_min."""
    return [(j1, j2, ell) for ell in range(1, config.lmax2 + 1) for j1, j2 in config.second_layer_pairs()
            if config.sigma(j2) * 0.4 > 6.0 and j1 >= j1_min]


def ratios(fk, config, todo, lattice, spectra, harmonics, binned=False):
    """S2 / S2_G - 1 for every (j1, j2, l) of ``todo`` on one field (rfft array, window applied)."""
    out = {}
    k, nmodes, volume, q = spectra.k, spectra.nmodes, BOXSIZE**3, config.q
    cell = BOXSIZE / NMESH
    for ell in sorted({p[2] for p in todo}):
        for j1 in sorted({p[0] for p in todo if p[2] == ell}):
            u = wavelet_modulus(fk, lattice, harmonics[ell], config.sigma(j1) / cell, ell)
            mean = u.mean(dtype=np.float64)
            fu = _rfft(u - mean)
            puu = spectra(fu, fu)[0]
            fu[0, 0, 0] = mean * NMESH**3
            n = 2 * ell + 1
            for _, j2, _ in (p for p in todo if p[0] == j1 and p[2] == ell):
                s2, s2sq = _moments(wavelet_modulus(fu, lattice, harmonics[ell], config.sigma(j2) / cell, ell), [q, 2.0])
                if binned:  # variance from the binned P_UU with the filter at the bin k, as second_layer.py did
                    weight = (config.sigma(j2) * k) ** (2 * ell) * np.exp(-((config.sigma(j2) * k) ** 2))
                    var = np.sum(nmodes * puu * weight) / volume / n
                else:  # exact: <|Y|^2> = n s_Y^2 in the same box
                    var = s2sq / n
                gauss = (2 * var) ** (q / 2) * np.exp(gammaln((n + q) / 2) - gammaln(n / 2))
                out[j1, j2, ell] = s2 / gauss - 1
    return out


def run_seed(seed, j1_min):
    from cosmoprimo import Cosmology

    t0 = time.time()
    config = common.config()
    klin = np.geomspace(1e-4, 10.0, 1024)
    pklin = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY).get_fourier().pk_interpolator(
        of="delta_m")(klin, z=Z)
    pk = lambda x: np.exp(np.interp(np.log(x), np.log(klin), np.log(pklin)))  # noqa: E731
    lattice = Lattice(NMESH)
    spectra = PowerMultipoles(lattice, BOXSIZE, kmax=0.4, ells=(0,))
    todo = pairs(config, j1_min)
    harmonics = {ell: lattice.harmonics(ell) for ell in {p[2] for p in todo}}
    window = cic_window(lattice)
    spt = SPTFields(lattice, pk, BOXSIZE, seed=10_000 + seed)
    curve = []
    for eps in EPS:
        fields = [spt.delta1] if eps == 0 else [spt.delta1 + eps * spt.delta2 + eps**2 * spt.delta3,
                                                -spt.delta1 + eps * spt.delta2 - eps**2 * spt.delta3]
        values = [ratios(window * f, config, todo, lattice, spectra, harmonics) for f in fields]
        curve.append({key: float(np.mean([v[key] for v in values])) for key in values[0]})
    design = np.stack([np.ones(len(EPS)), np.square(EPS), np.power(EPS, 4)], axis=1)
    result = {}
    for key in curve[0]:
        d0, d2, _ = np.linalg.solve(design, [c[key] for c in curve])
        result[key] = dict(gauss=d0, tree=d2)
    print(f"seed {seed} done in {time.time() - t0:.0f} s", flush=True)
    return seed, result


def nbody_ratios(config, keys):
    """Quijote mean and error of S2 / S2_G - 1 with the exact variance <|Y|^2> (the stored q = 2 moment)."""
    files = sorted((common.DATA_DIR / "real").glob("wst_r*.npz"))
    stored = list(np.load(files[0])["q"])
    iq, i2 = stored.index(config.q), stored.index(2.0)
    values = {key: [] for key in keys}
    for path in files:
        s2 = np.load(path)["S2"]
        for j1, j2, ell in keys:
            n = 2 * ell + 1
            gauss = (2 * s2[i2, j1, j2, ell] / n) ** (config.q / 2) * np.exp(gammaln((n + config.q) / 2) - gammaln(n / 2))
            values[j1, j2, ell].append(s2[iq, j1, j2, ell] / gauss - 1)
    return {key: dict(mean=float(np.mean(v)), err=float(np.std(v, ddof=1) / np.sqrt(len(v))), std=float(np.std(v, ddof=1)))
            for key, v in values.items()}


def parse_seeds(text):
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--j1-min", type=int, default=2, help="smallest first-layer j (2: 6.3 Mpc/h)")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/feasibility/second_layer_gaussian.json")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(pool.map(run_seed, seeds, [args.j1_min] * len(seeds)))
    config = common.config()
    nbody = nbody_ratios(config, list(results[seeds[0]]))
    rows = []
    for key in sorted(results[seeds[0]], key=lambda p: (p[2], p[0], p[1])):
        j1, j2, ell = key
        row = dict(j1=j1, j2=j2, ell=ell, sigma1=config.sigma(j1), sigma2=config.sigma(j2))
        values = {kind: np.array([results[s][key][kind] for s in seeds]) for kind in ("gauss", "tree")}
        values["sum"] = values["gauss"] + values["tree"]
        for kind, v in values.items():
            row[kind] = float(v.mean())
            row[kind + "_err"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        row["nbody"], row["nbody_err"], row["nbody_std"] = nbody[key]["mean"], nbody[key]["err"], nbody[key]["std"]
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(seeds=seeds, pairs=rows), indent=2))
    print(f"\nS2/S2_G - 1 [%], {len(seeds)} seeds    Gaussian d0    tree d2    d0 + d2      N-body")
    for r in rows:
        print(f"{r['sigma1']:5.1f} -> {r['sigma2']:5.1f}  l={r['ell']}  "
              + "  ".join(f"{100 * r[k]:+6.2f}({100 * r[k + '_err']:.2f})" for k in ("gauss", "tree", "sum", "nbody")))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
