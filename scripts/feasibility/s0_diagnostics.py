#!/usr/bin/env python
"""What S0 = <|delta_cell|^q> measures, and why it is outside perturbation theory.

For one Quijote snapshot this paints the CDM field on the 3.9 Mpc/h CIC mesh of
2108.07821 and on 10 Mpc/h meshes (BOSS-like, CIC and TSC). It also builds a
Poisson subsample with the CMASS number density (an unbiased tracer, so shot
noise only) and a uniform random catalogue of the same size. For each field it
reports:

* the cell variance and the effective Gaussian smoothing of the assignment kernel;
* S0, S0 of the Gaussian twin with the same |delta(k)| (NG = ratio - 1) and
  E|x|^q for a Gaussian of the same variance;
* how much of S0 comes from underdense cells (delta < -0.5) and empty cells;
* the share of the cell variance that comes from k above the Nyquist frequency
  (through aliasing) or from k > 0.25 h/Mpc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln

import wst_feasibility as W


def read_positions(snapdir: Path, rsd: bool, los: int = 2):
    import h5py
    import hdf5plugin  # noqa: F401

    files = sorted(snapdir.glob("snap_*.hdf5"), key=lambda p: int(p.stem.rsplit(".", 1)[1]))
    header = W.read_header(files)
    a = 1.0 / (1.0 + header["redshift"])
    rsd_factor = np.sqrt(a) * (1.0 + header["redshift"]) / header["hubble_z"]
    for filename in files:
        with h5py.File(filename, "r") as handle:
            pos = handle["PartType1/Coordinates"][:].astype(np.float64) / 1e3
            if rsd:
                pos[:, los] += handle["PartType1/Velocities"][:, los].astype(np.float64) * rsd_factor
        yield np.remainder(pos, header["boxsize"]), header


def paint(chunks, nmesh: int, boxsize: float, scheme: str):
    grid = np.zeros(nmesh**3)
    for pos in chunks:
        x = pos * (nmesh / boxsize)
        if scheme == "cic":
            i0 = np.floor(x).astype(np.int64)
            d = x - i0
            offsets = (0, 1)
            weights = lambda o, dd: (1 - dd) if o == 0 else dd  # noqa: E731
        else:  # tsc
            i0 = np.rint(x).astype(np.int64)
            d = x - i0
            offsets = (-1, 0, 1)
            weights = lambda o, dd: {-1: 0.5 * (0.5 - dd) ** 2, 0: 0.75 - dd**2, 1: 0.5 * (0.5 + dd) ** 2}[o]  # noqa: E731
        for ox in offsets:
            ix, wx = (i0[:, 0] + ox) % nmesh, weights(ox, d[:, 0])
            for oy in offsets:
                iy, wy = (i0[:, 1] + oy) % nmesh, weights(oy, d[:, 1])
                for oz in offsets:
                    iz, wz = (i0[:, 2] + oz) % nmesh, weights(oz, d[:, 2])
                    grid += np.bincount((ix * nmesh + iy) * nmesh + iz, weights=wx * wy * wz, minlength=nmesh**3)
    grid = grid.reshape((nmesh,) * 3)
    return grid / grid.mean() - 1.0, grid


def stats(delta, counts, nmesh, boxsize, q, seed):
    lat = W.Lattice(nmesh)
    fk = W.rfft(delta.astype(np.float32))
    power = np.abs(fk) ** 2 * lat.mult
    kphys = lat.kmag * (nmesh / boxsize)
    var_lattice = float(power.sum() / nmesh**6)
    twin = W.phase_randomize(delta.astype(np.float32), seed)
    s0 = float(np.mean(np.abs(delta) ** q))
    s0_twin = float(np.mean(np.abs(twin.astype(np.float64)) ** q))
    var = float(delta.var())
    s0_gauss = (2 * var) ** (q / 2) * np.exp(gammaln((1 + q) / 2)) / np.sqrt(np.pi)
    under = delta < -0.5
    contrib = np.abs(delta) ** q
    return {
        "var": var,
        "var_lattice": var_lattice,
        "frac_var_k_gt_0.25": float(power[kphys > 0.25].sum() / power.sum()),
        "S0": s0,
        "S0_twin": s0_twin,
        "NG": s0 / s0_twin - 1,
        "S0_gauss_same_var": float(s0_gauss),
        "frac_cells_delta_lt_-0.5": float(under.mean()),
        "frac_S0_from_delta_lt_-0.5": float(contrib[under].sum() / contrib.sum()),
        "frac_cells_empty": float(np.mean(counts <= 0)),
        "skewness": float(np.mean(delta**3) / var**1.5),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realization", default="0")
    parser.add_argument("--nbar", type=float, default=3e-4, help="tracer density in (h/Mpc)^3")
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("outputs/feasibility"))
    args = parser.parse_args()
    snapdir = W.SNAPSHOT_ROOT / args.realization / "snapdir_003"
    rng = np.random.default_rng(int(args.realization) + 7)
    results = {}
    for space in ("real", "rsd"):
        positions, header = [], None
        sub = []
        for pos, header in read_positions(snapdir, rsd=(space == "rsd")):
            positions.append(pos.astype(np.float32))
            frac = args.nbar * header["boxsize"] ** 3 / 512**3
            sub.append(pos[rng.random(len(pos)) < frac])
        box = header["boxsize"]
        sub = np.concatenate(sub)
        uniform = rng.random((len(sub), 3)) * box
        configs = {
            "matter, CIC, 3.9 Mpc/h (256^3)": (positions, 256, "cic"),
            "matter, CIC, 10 Mpc/h (100^3)": (positions, 100, "cic"),
            "matter, TSC, 10 Mpc/h (100^3)": (positions, 100, "tsc"),
            f"subsample nbar={args.nbar:g}, TSC, 10 Mpc/h": ([sub], 100, "tsc"),
            f"uniform randoms nbar={args.nbar:g}, TSC, 10 Mpc/h": ([uniform], 100, "tsc"),
        }
        for name, (chunks, nmesh, scheme) in configs.items():
            delta, counts = paint((c.astype(np.float64) for c in chunks), nmesh, box, scheme)
            results[f"{space} | {name}"] = stats(delta, counts, nmesh, box, args.q, seed=11)
            print(space, name, json.dumps({k: round(v, 4) for k, v in results[f"{space} | {name}"].items()}), flush=True)
        del positions
    fn = args.output / f"s0_diagnostics_r{args.realization}.json"
    fn.write_text(json.dumps(results, indent=1))
    print(f"wrote {fn}")


if __name__ == "__main__":
    main()
