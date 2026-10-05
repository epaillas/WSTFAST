#!/usr/bin/env python
"""Train a Taylor emulator of the cosmology-dependent WST basis (all coefficients, no nuisance).

The WST configuration, redshift and particle shot noise are read from the measurements, so the
emulator matches the data it will be compared with. Example:

    python scripts/train_emulator.py --data-dir data/quijote/z0.5/J4_L4_sigma0.8_n256 --vary omega_cdm logA
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wstmodel.theory  # noqa: F401  (enables JAX double precision)
from desilike import build, setup_logging
from desilike.emulators import Emulator, Space
from wstmodel.calculators import WSTBasis, build_cosmology
from wstmodel.config import QUIJOTE_COSMOLOGY, select_coefficients
from wstmodel.data import load_measurement
from wstmodel.theory import Assembly, all_coefficients

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


def predict(basis, assembly, point):
    """WST coefficients (nuisance parameters at zero) from an exact or emulated basis."""
    build(basis)(point)
    noise = np.zeros(len(assembly.noise_ells))
    return np.asarray(assembly(basis.s1_terms, basis.s21_terms, 0.0, noise))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/quijote/z0.5/J4_L4_sigma0.8_n256"))
    parser.add_argument("--space", choices=("real",), default="real", help="the model is real-space only for now")
    parser.add_argument("--q", type=float, default=None, help="WST exponent (default: the measurement's first q)")
    parser.add_argument("--vary", nargs="+", default=["omega_cdm", "logA"], choices=sorted(QUIJOTE_COSMOLOGY))
    parser.add_argument("--bounds", type=json.loads, default={}, help='JSON overrides, e.g. \'{"logA": [2.9, 3.2]}\'')
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--accuracy", type=int, default=2)
    parser.add_argument("--nvalidation", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("outputs/emulators/wst_basis_taylor.h5"))
    args = parser.parse_args()
    setup_logging()

    settings = basis_settings(args.data_dir, args.space, q=args.q)
    settings["vary"] = list(args.vary)
    bounds = {name: tuple(args.bounds.get(name, DEFAULT_BOUNDS[name])) for name in args.vary}
    settings["bounds"] = bounds
    basis = WSTBasis(cosmo=build_cosmology(args.vary), config=wstmodel.WSTConfig(**settings["config"]),
                     z=settings["z"], shotnoise=settings["shotnoise"])

    emulator = Emulator(basis, Space(bounds=bounds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emulator.train(engine="taylor", order=args.order, accuracy=args.accuracy,
                   checkpoint=str(args.output.with_suffix(".checkpoint.npz")))
    emulator.write(str(args.output))
    args.output.with_suffix(".json").write_text(json.dumps(settings, indent=2))
    print(f"wrote {args.output}")

    # Validation of the predicted coefficients against the exact basis at random points of the box,
    # for all coefficients and for the default perturbative selection.
    config = wstmodel.WSTConfig(**settings["config"])
    selections = {"all": all_coefficients(config), "default": select_coefficients(config)}
    assemblies = {name: Assembly(config, coefficients) for name, coefficients in selections.items()}
    emulated = emulator.to_calculator()
    rng = np.random.default_rng(0)
    errors = {name: [] for name in selections}
    for _ in range(args.nvalidation):
        point = {name: float(rng.uniform(*bounds[name])) for name in args.vary}
        for name, assembly in assemblies.items():
            exact, approx = predict(basis, assembly, point), predict(emulated, assembly, point)
            errors[name].append(np.max(np.abs(approx / exact - 1)))
    report = dict(npoints=args.nvalidation, order=args.order, accuracy=args.accuracy,
                  max_relative_error={name: float(np.max(value)) for name, value in errors.items()},
                  median_relative_error={name: float(np.median(value)) for name, value in errors.items()})
    args.output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
