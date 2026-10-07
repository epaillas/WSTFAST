#!/usr/bin/env python
"""Train a Taylor emulator of the cosmology-dependent WST basis (all coefficients, no nuisance), or of the P(k) basis.

The WST configuration, redshift and particle shot noise are read from the measurements, so the
emulator matches the data it will be compared with. With --stat pk the emulated basis is
[P_L, L_Lambda] of dsc-model's real-space matter P(k) on k nodes up to --kmax (any fit with a
smaller kmax can use it). Examples:

    python scripts/train_emulator.py --data-dir data/quijote/fiducial/z0.5/J4_L4_sigma0.8_n256 --vary omega_cdm logA

    # five parameters: --budget 2 keeps the pure cubic terms and needs 51 nodes (821 at full order 3)
    python scripts/train_emulator.py --data-dir data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256 \
        --vary omega_cdm logA n_s h omega_b --budget 2 --output outputs/emulators/wst_basis_taylor_superset_5p.h5

    # one-loop P(k) with dsc-model's regulator (Lambda = 0.5 h/Mpc); --cutoff none for unregulated SPT
    python scripts/train_emulator.py --stat pk --data-dir data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256 \
        --vary omega_cdm logA --output outputs/emulators/pk_taylor.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from desilike import build, setup_logging
from desilike.emulators import Emulator, Space
from wstfast.calculators import PowerBasis, PowerTheory, WSTBasis, build_cosmology
from wstfast.config import QUIJOTE_COSMOLOGY, select_coefficients
from wstfast.data import load_measurement, load_power_dataset
from wstfast.theory import Assembly, all_coefficients
from wstfast.theory.power import ORDERS, LatticeBinning, default_knodes

#: Default emulation box: about +-15% around the Quijote fiducial (as in dsc-model).
DEFAULT_BOUNDS = {"h": (0.6111, 0.7311), "omega_cdm": (0.0959, 0.1459), "logA": (2.761, 3.361),
                  "omega_b": (0.0200, 0.0241), "n_s": (0.90, 1.02)}


def basis_settings(data_dir: Path, space: str, q: float | None = None) -> dict:
    """WST configuration, redshift and shot noise of the first measurement found."""
    path = next(iter(sorted((data_dir / space).glob("wst_r*.npz"))), None)
    if path is None:
        raise FileNotFoundError(f"no measurements in {data_dir / space}")
    measurement = load_measurement(path, q=q)
    meta = measurement["metadata"]
    return dict(config=measurement["config"].to_dict(), z=meta["redshift"],
                shotnoise=meta["boxsize"] ** 3 / meta["nparticles"])


def predict(graph, assembly, point):
    """WST coefficients (nuisance parameters at zero) from a built exact or emulated basis."""
    basis = graph(point)
    noise = np.zeros(len(assembly.noise_keys))
    return np.asarray(assembly(basis.s1_terms, basis.s21_terms, 0.0, noise))


def train_power(args):
    """Emulate PowerBasis on nodes up to --kmax; validate the binned P(k) (cs2_pk = 0) at random points."""
    dataset = load_power_dataset(args.data_dir, args.space, kmax=args.kmax, rebin=1)
    meta = dataset.metadata
    knodes = default_knodes(args.kmax, boxsize=meta["boxsize"])
    bounds = {name: tuple(args.bounds.get(name, DEFAULT_BOUNDS[name])) for name in args.vary}
    settings = dict(stat="pk", z=meta["redshift"], order=args.pk_order, cutoff=args.cutoff, knodes=knodes.tolist(),
                    vary=list(args.vary), bounds=bounds)
    basis = PowerBasis(cosmo=build_cosmology(args.vary), knodes=knodes, z=meta["redshift"], order=args.pk_order,
                       cutoff=args.cutoff)
    emulator = Emulator(basis, Space(bounds=bounds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emulator.train(engine="taylor", order=args.order, accuracy=args.accuracy, budget=args.budget,
                   checkpoint=str(args.output.with_suffix(".checkpoint.npz")))
    emulator.write(str(args.output))
    args.output.with_suffix(".json").write_text(json.dumps(settings, indent=2))
    print(f"wrote {args.output}")

    binning = LatticeBinning(dataset.edges, meta["boxsize"], meta["nmesh"], knodes)
    theories = {name: build(PowerTheory(binning, basis=b, shotnoise=dataset.shotnoise, order=args.pk_order,
                                        counterterm=False))
                for name, b in (("exact", basis), ("emulated", emulator.to_calculator()))}
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(args.nvalidation):
        point = {name: float(rng.uniform(*bounds[name])) for name in args.vary}
        exact, approx = (np.asarray(theories[name](point)) for name in ("exact", "emulated"))
        errors.append(np.abs(approx / exact - 1))
    errors = np.array(errors)
    report = dict(npoints=args.nvalidation, order=args.order, accuracy=args.accuracy, budget=args.budget,
                  k=dataset.k.tolist(), max_relative_error=float(errors.max()),
                  median_relative_error=float(np.median(errors)),
                  max_relative_error_per_bin=errors.max(axis=0).tolist())
    args.output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ("k", "max_relative_error_per_bin")},
                     indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stat", choices=("wst", "pk"), default="wst")
    parser.add_argument("--data-dir", type=Path, default=Path("data/quijote/fiducial/z0.5/J4_L4_sigma0.8_n256"))
    parser.add_argument("--space", choices=("real",), default="real", help="the model is real-space only for now")
    parser.add_argument("--q", type=float, default=None, help="WST exponent (default: the measurement's first q)")
    parser.add_argument("--vary", nargs="+", default=["omega_cdm", "logA"], choices=sorted(QUIJOTE_COSMOLOGY))
    parser.add_argument("--bounds", type=json.loads, default={}, help='JSON overrides, e.g. \'{"logA": [2.9, 3.2]}\'')
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--accuracy", type=int, default=2)
    parser.add_argument("--budget", type=int, default=None,
                        help="cap on the total degree of mixed terms (2 keeps 5-parameter training at 51 nodes)")
    parser.add_argument("--kmax", type=float, default=0.3, help="P(k): largest k of the emulated nodes [h/Mpc]")
    parser.add_argument("--pk-order", choices=ORDERS, default="one-loop", help="P(k): perturbative order")
    parser.add_argument("--cutoff", type=lambda v: None if v.lower() == "none" else float(v), default=0.5,
                        help="P(k): loop regulator in h/Mpc, or 'none' for unregulated SPT")
    parser.add_argument("--nvalidation", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: outputs/emulators/wst_basis_taylor.h5 or outputs/emulators/pk_taylor.h5")
    args = parser.parse_args()
    setup_logging()
    if args.output is None:
        args.output = Path(f"outputs/emulators/{'wst_basis' if args.stat == 'wst' else 'pk'}_taylor.h5")
    if args.stat == "pk":
        return train_power(args)

    settings = basis_settings(args.data_dir, args.space, q=args.q)
    settings["vary"] = list(args.vary)
    bounds = {name: tuple(args.bounds.get(name, DEFAULT_BOUNDS[name])) for name in args.vary}
    settings["bounds"] = bounds
    basis = WSTBasis(cosmo=build_cosmology(args.vary), config=wstfast.WSTConfig(**settings["config"]),
                     z=settings["z"], shotnoise=settings["shotnoise"])

    emulator = Emulator(basis, Space(bounds=bounds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emulator.train(engine="taylor", order=args.order, accuracy=args.accuracy, budget=args.budget,
                   checkpoint=str(args.output.with_suffix(".checkpoint.npz")))
    emulator.write(str(args.output))
    args.output.with_suffix(".json").write_text(json.dumps(settings, indent=2))
    print(f"wrote {args.output}")

    # Validation of the predicted coefficients against the exact basis at random points of the box,
    # for all coefficients and for the default perturbative selection.
    config = wstfast.WSTConfig(**settings["config"])
    selections = {"all": all_coefficients(config), "default": select_coefficients(config)}
    assemblies = {name: Assembly(config, coefficients) for name, coefficients in selections.items()}
    # Build each graph once: rebuilding recompiles the exact basis (minutes) at every point.
    exact_graph, emulated_graph = build(basis), build(emulator.to_calculator())
    rng = np.random.default_rng(0)
    errors = {name: [] for name in selections}
    for _ in range(args.nvalidation):
        point = {name: float(rng.uniform(*bounds[name])) for name in args.vary}
        for name, assembly in assemblies.items():
            exact, approx = predict(exact_graph, assembly, point), predict(emulated_graph, assembly, point)
            errors[name].append(np.max(np.abs(approx / exact - 1)))
    report = dict(npoints=args.nvalidation, order=args.order, accuracy=args.accuracy, budget=args.budget,
                  max_relative_error={name: float(np.max(value)) for name, value in errors.items()},
                  median_relative_error={name: float(np.median(value)) for name, value in errors.items()})
    args.output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
