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
from wstfast.calculators import (JointTheory, PowerBasis, PowerTheory, RSDBasis, RSDPowerTheory, S1mTheory,
                                 S21mTheory, WSTBasis, build_cosmology)
from wstfast.config import QUIJOTE_COSMOLOGY, Coefficient, select_coefficients
from wstfast.data import load_measurement, load_power_dataset, load_s1m_dataset
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


def train_rsd(args):
    """Emulate RSDBasis (redshift-space P_s grid + S1m cumulants of every S1 coefficient with sigma >= --s1-min-scale +
    S21m terms of the S21 coefficients of the --s21-* cuts, unless --no-s21); validate the S1m blocks, the multipoles
    (counterterms at zero) and S21m (noise amplitudes at zero) against the exact basis at random points."""
    from wstfast.theory.rsd import MultipoleProjection, RSDGrid, S1mProjection

    first = sorted((args.data_dir / "rsd").glob("wst_r*.npz"))[0]
    measurement = load_measurement(first, q=0.8 if args.q is None else args.q)
    config, meta = measurement["config"], measurement["metadata"]
    coefficients = [c for c in select_coefficients(config, s1_min_scale=args.s1_min_scale) if c.kind == "S1"]
    s21 = [] if args.no_s21 else [c for c in select_coefficients(config, s21_min_scale=args.s21_min_scale,
                                                                   s21_min_ratio=args.s21_min_ratio,
                                                                   s21_min_scale2=args.s21_min_scale2)
                                   if c.kind == "S21"]
    first_layer = [Coefficient("S1", ell, j) for j, ell in sorted({(c.j, c.ell) for c in s21})]
    missing = [c.label for c in first_layer if c not in coefficients]
    if missing:
        raise SystemExit(f"--s1-min-scale must include the first-layer coefficients {missing}")
    bounds = {name: tuple(args.bounds.get(name, DEFAULT_BOUNDS[name])) for name in args.vary}
    settings = dict(stat="rsd", config=config.to_dict(), z=meta["redshift"], coefficients=[c.label for c in coefficients],
                    s21_coefficients=[c.label for c in s21], s1_min_scale=args.s1_min_scale, kmax=args.kmax, damping="linear", ir=args.ir, vary=list(args.vary),
                    bounds=bounds, tracer=args.tracer)
    basis = RSDBasis(cosmo=build_cosmology(args.vary), config=config, coefficients=coefficients, z=meta["redshift"],
                     kmax=args.kmax, ir=args.ir, s21_coefficients=s21, tracer=args.tracer)
    emulator = Emulator(basis, Space(bounds=bounds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emulator.train(engine="taylor", order=args.order, accuracy=args.accuracy, budget=args.budget,
                   checkpoint=str(args.output.with_suffix(".checkpoint.npz")))
    emulator.write(str(args.output))
    args.output.with_suffix(".json").write_text(json.dumps(settings, indent=2))
    print(f"wrote {args.output}")

    s1m = load_s1m_dataset(args.data_dir, "rsd", coefficients, q=config.q)
    power = load_power_dataset(args.data_dir, "rsd", kmax=min(args.kmax, 0.2), rebin=2, files=s1m.files, ells=(0, 2, 4))
    grid = RSDGrid(kmax=args.kmax)  # the same grid as the basis
    projections = dict(s1m=S1mProjection(config, coefficients, grid), pk=MultipoleProjection(power.edges, grid))
    # One graph per basis (a basis cannot be shared between graphs): S1m blocks then multipoles.
    def theories(b):
        out = [S1mTheory(projections["s1m"], coefficients, q=config.q, basis=b, shotnoise=s1m.shotnoise,
                         tracer=args.tracer),
               RSDPowerTheory(projections["pk"], basis=b, shotnoise=power.shotnoise, tracer=args.tracer)]
        if s21:
            out.append(S21mTheory(S1mProjection(config, first_layer, grid), s21, first_layer, q=config.q, basis=b,
                                  cumulant_index=[coefficients.index(c) for c in first_layer],
                                  shotnoise=s1m.shotnoise, tracer=args.tracer))
        return out

    graphs = {name: build(JointTheory(theories(b))) for name, b in (("exact", basis), ("emulated", emulator.to_calculator()))}
    nblock = projections["s1m"].matrix.shape[0]
    npk = projections["pk"].matrix.shape[0]
    slices = {"s1m": slice(0, nblock), "pk": slice(nblock, nblock + npk)}
    if s21:
        slices["s21m"] = slice(nblock + npk, None)
    rng = np.random.default_rng(0)
    errors = {stat: [] for stat in slices}
    for _ in range(args.nvalidation):
        point = {name: float(rng.uniform(*bounds[name])) for name in args.vary}
        exact, approx = (np.asarray(graphs[kind](point)) for kind in ("exact", "emulated"))
        for stat, sl in slices.items():
            errors[stat].append(np.max(np.abs(approx[sl] / exact[sl] - 1)))
    report = dict(npoints=args.nvalidation, order=args.order, accuracy=args.accuracy, budget=args.budget,
                  max_relative_error={k: float(np.max(v)) for k, v in errors.items()},
                  median_relative_error={k: float(np.median(v)) for k, v in errors.items()})
    args.output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stat", choices=("wst", "pk", "rsd"), default="wst",
                        help="rsd: redshift-space P_s grid + S1m cumulants (data from the _los measurements)")
    parser.add_argument("--s1-min-scale", type=float, default=17.6, help="rsd: smallest sigma_j of the S1m cumulants")
    parser.add_argument("--ir", action="store_true", help="rsd: BAO infrared resummation of the P_s grid")
    parser.add_argument("--no-s21", action="store_true", help="rsd: leave out the S21m terms")
    parser.add_argument("--tracer", choices=("matter", "biased"), default="matter",
                        help="rsd: biased-tracer basis (bias-monomial coefficients; wstfast.theory.rsd_bias)")
    parser.add_argument("--s21-min-scale", type=float, default=17.6, help="rsd: smallest sigma_j1 of the S21m terms")
    parser.add_argument("--s21-min-scale2", type=float, default=70.0, help="rsd: smallest sigma_j2 of the S21m terms")
    parser.add_argument("--s21-min-ratio", type=float, default=2.8, help="rsd: smallest sigma_j2 / sigma_j1 of S21m")
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
        args.output = Path(f"outputs/emulators/{dict(wst='wst_basis', pk='pk', rsd='rsd_basis' if args.tracer == 'matter' else 'rsd_biased_basis')[args.stat]}_taylor.h5")
    if args.stat == "pk":
        return train_power(args)
    if args.stat == "rsd":
        return train_rsd(args)

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
