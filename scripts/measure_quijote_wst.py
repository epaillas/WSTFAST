#!/usr/bin/env python
"""Measure the WST of Quijote CDM snapshots, one file per realization, space and scale configuration.

Each snapshot is read and painted once per space, then transformed with every requested scale
configuration (each --sigma0). All exponents --q are stored in the same file, together with the
power-spectrum multipoles of the first-layer moduli (cross with delta, auto) and of delta.
Output files are

    {output-dir}/{tag}/{space}/wst_r{realization:05d}.npz,   tag = e.g. J4_L4_sigma0.8_n256

and existing files are skipped, so interrupted jobs can simply be resubmitted. Examples:

    # the half-octave superset of docs/wst_eft_feasibility.md on the 10 local fiducial boxes
    python scripts/measure_quijote_wst.py --superset --realizations 0 1 10 100 1000 10000-10004

    # the same on a GPU
    python scripts/measure_quijote_wst.py --superset --backend torch --realizations 0-1499

    # a single dyadic configuration
    python scripts/measure_quijote_wst.py --realizations 0-1499 --J 4 --L 4 --sigma0 0.8 --q 0.8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from wstfast.config import SUPERSET, WSTConfig
from wstfast.data import save_measurement
from wstfast.measure import Lattice, PowerMultipoles, measure_wst
from wstfast.quijote import SNAPSHOT_ROOT, load_density

SUPERSET_QS = [0.5, 0.8, 1.0, 2.0]


def parse_realizations(items: list[str]) -> list[int]:
    """Accept single ids and inclusive ranges: ['0', '10-12'] -> [0, 10, 11, 12]."""
    out = []
    for item in items:
        first, _, last = item.partition("-")
        out.extend(range(int(first), int(last or first) + 1))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--realizations", nargs="+", required=True, help="ids and inclusive ranges, e.g. 0-1499")
    parser.add_argument("--spaces", nargs="+", choices=("real", "rsd"), default=["real", "rsd"])
    parser.add_argument("--redshift", type=float, default=0.5)
    parser.add_argument("--nmesh", type=int, default=256)
    parser.add_argument("--superset", action="store_true",
                        help=f"half-octave superset: J={SUPERSET.J}, L={SUPERSET.L}, L2={SUPERSET.L2}, "
                             f"min_dj={SUPERSET.min_dj}, step=sqrt(2), q={SUPERSET_QS}")
    parser.add_argument("--J", type=int, default=4)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--L2", type=int, default=None, help="largest l of the second layer (default L)")
    parser.add_argument("--min-dj", type=int, default=1, help="smallest j2 - j1 of the second layer")
    parser.add_argument("--step", type=float, default=2.0, help="ratio of consecutive scales")
    parser.add_argument("--sigma0", type=float, nargs="+", default=[0.8], help="smallest wavelet width(s) in cells")
    parser.add_argument("--q", type=float, nargs="+", default=[0.8], help="exponents, all stored in each file")
    parser.add_argument("--kmax", type=float, default=0.4, help="largest k [h/Mpc] of the stored spectra")
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy",
                        help="torch runs painting and the WST on --device (e.g. a GPU)")
    parser.add_argument("--device", default="auto", help="torch device: auto, cpu or cuda")
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("data/quijote/fiducial/z0.5"))
    args = parser.parse_args()
    if args.superset:
        args.J, args.L, args.L2, args.min_dj, args.step = SUPERSET.J, SUPERSET.L, SUPERSET.L2, SUPERSET.min_dj, SUPERSET.step
        args.sigma0, args.q = [SUPERSET.sigma0], SUPERSET_QS
    return args


def make_backend(args):
    """(device, lattice, spectra factory, estimator) for the chosen backend; set up once per run."""
    if args.backend == "numpy":
        return None, Lattice(args.nmesh), PowerMultipoles, measure_wst
    from wstfast.measure_torch import (TorchLattice, TorchPowerMultipoles, default_device,
                                        measure_wst_torch)

    device = default_device(args.device)
    print(f"torch backend on {device}" + (f" ({__import__('torch').cuda.get_device_name(device)})"
                                          if device.type == "cuda" else ""), flush=True)
    return device, TorchLattice(args.nmesh, device=device, lmax=args.L), TorchPowerMultipoles, measure_wst_torch


def main():
    args = parse_args()
    device, lattice, multipoles, estimator = make_backend(args)
    spectra = None
    configs = [WSTConfig(J=args.J, L=args.L, L2=args.L2, min_dj=args.min_dj, step=args.step, sigma0=sigma0,
                         q=args.q[0]) for sigma0 in args.sigma0]
    for realization in parse_realizations(args.realizations):
        for space in args.spaces:
            paths = {config: args.output_dir / config.tag(args.nmesh) / space / f"wst_r{realization:05d}.npz"
                     for config in configs}
            pending = {config: path for config, path in paths.items() if not path.exists()}
            for path in set(paths.values()) - set(pending.values()):
                print(f"skip {path}")
            if not pending:
                continue
            start = time.time()
            delta, header = load_density(realization, redshift=args.redshift, nmesh=args.nmesh,
                                         rsd=space == "rsd", root=args.snapshot_root, device=device)
            spectra = spectra or multipoles(lattice, header["boxsize"], kmax=args.kmax)
            metadata = dict(realization=realization, space=space, los="z" if space == "rsd" else None,
                            redshift=round(header["redshift"], 6), boxsize=header["boxsize"], nmesh=args.nmesh,
                            nparticles=header["nparticles"], mass_assignment="cic", backend=args.backend)
            for config, path in pending.items():
                config = WSTConfig(**{**config.to_dict(), "cellsize": header["boxsize"] / args.nmesh})
                result = estimator(delta, config, qs=args.q, lattice=lattice, spectra=spectra)
                save_measurement(path, result, config, metadata)
                print(f"wrote {path} ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
