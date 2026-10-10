#!/usr/bin/env python
"""Time the 4th-chaos covariance (wstfast.theory.covariance.chaos_covariances) and check it against a reference.

The inputs (field power, linear power, modulus spectra) are read from an output of scripts/analytic_covariance.py, so
no theory is recomputed; the reference is the ``chaos4`` of another such output. The error is reported relative to
sqrt(cov4_aa cov4_bb), and relative to the full ln-covariance diagonal (what matters for the data covariance).

    python scripts/feasibility/benchmark_chaos.py outputs/covariance/inputs_q0.8.npz \
        --reference outputs/covariance/analytic_q0.8.npz --subset S1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wstfast.config import Coefficient, WSTConfig
from wstfast.theory.covariance import ModulusSpectra, chaos_covariances, last_layers

DATA = Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256")


def parse_label(label):
    parts = label.split("_")
    if parts[0] == "S1":
        return Coefficient("S1", int(parts[2][1:]), int(parts[1][1:]))
    return Coefficient("S21", int(parts[3][1:]), int(parts[1][1:]), int(parts[2][1:]))


def table(k, values):
    lk, lv = np.log(k), np.log(values)
    return lambda x: np.exp(np.interp(np.log(x), lk, lv))  # noqa: E731


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, help="output of scripts/analytic_covariance.py (input tables)")
    parser.add_argument("--reference", type=Path, default=None, help="output with the reference chaos4")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--subset", choices=("all", "S1", "S21"), default="all")
    parser.add_argument("--nmesh", type=int, default=128)
    parser.add_argument("--tol", type=float, default=1e-8, help="pair-mesh tolerance (0: nmesh for every pair)")
    args = parser.parse_args()

    inputs = np.load(args.inputs)
    with np.load(sorted((args.data_dir / "real").glob("wst_r*.npz"))[0]) as first:
        meta = json.loads(str(first["metadata"]))
    config = WSTConfig(**meta["config"])
    labels = list(inputs["labels"])
    keep = [i for i, label in enumerate(labels) if args.subset == "all" or label.split("_")[0] == args.subset]
    coefficients = [parse_label(labels[i]) for i in keep]
    pfield, plin = table(inputs["pfield_k"], inputs["pfield"]), table(inputs["pfield_k"], inputs["plin"])
    moduli = None
    if "moduli_k" in inputs:
        keys = [tuple(int(v) for v in row) for row in inputs["moduli_fields"]]
        moduli = ModulusSpectra(inputs["moduli_k"], dict(zip(keys, inputs["moduli_response"])),
                                {(a, b): inputs["moduli_cross"][i, j] for i, a in enumerate(keys)
                                 for j, b in enumerate(keys)})
    fields, spectra = last_layers(config, coefficients, pfield, plin, moduli)
    t0 = time.time()
    cov2, cov4 = chaos_covariances(fields, float(inputs["q"]), meta["boxsize"], args.nmesh, spectra, tol=args.tol)
    elapsed = time.time() - t0
    print(f"{len(fields)} fields, nmesh <= {args.nmesh}, tol {args.tol:g}: {elapsed:.1f}s")
    if args.reference is not None:
        reference = np.load(args.reference)
        assert [labels[i] for i in keep] == [list(reference["labels"])[i] for i in keep]
        ref4 = reference["chaos4"][np.ix_(keep, keep)]
        lncov = reference["lncov"][np.ix_(keep, keep)]
        scale4 = np.sqrt(np.outer(np.diag(ref4), np.diag(ref4)))
        scale = np.sqrt(np.outer(np.diag(lncov), np.diag(lncov)))
        print(f"  cov4 vs reference: max |d| / sqrt(c4_aa c4_bb) {np.max(np.abs(cov4 - ref4) / scale4):.2e}, "
              f"max |d| / sqrt(C_aa C_bb) of the ln-covariance {np.max(np.abs(cov4 - ref4) / scale):.2e}")


if __name__ == "__main__":
    main()
