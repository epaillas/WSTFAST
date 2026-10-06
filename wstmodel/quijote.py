"""Read Quijote Gadget-HDF5 snapshots and paint the CDM density contrast on a mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SNAPSHOT_ROOT = Path("/Users/epaillas/data/quijote/snapshots/fiducial")
SNAPNUM = {0.0: "004", 0.5: "003", 1.0: "002", 2.0: "001", 3.0: "000"}


def snapshot_files(snapdir: Path) -> list[Path]:
    files = sorted(Path(snapdir).glob("snap_*.hdf5"), key=lambda p: int(p.stem.rsplit(".", 1)[1]))
    if not files:
        raise FileNotFoundError(f"no snapshot files in {snapdir}")
    return files


def read_header(files) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401  (Quijote compression filter)

    with h5py.File(files[0], "r") as handle:
        attrs = handle["Header"].attrs
        header = {
            "boxsize": float(attrs["BoxSize"]) / 1e3,
            "redshift": float(attrs["Redshift"]),
            "omega_m": float(attrs["Omega0"]),
            "omega_l": float(attrs["OmegaLambda"]),
            "nparticles": int(np.asarray(attrs["NumPart_Total"], dtype=np.uint64)[1]
                              + (np.asarray(attrs.get("NumPart_Total_HighWord", np.zeros(6)), dtype=np.uint64)[1] << 32)),
        }
    header["hubble_z"] = 100.0 * np.sqrt(header["omega_m"] * (1 + header["redshift"]) ** 3 + header["omega_l"])
    return header


def iter_positions(files, header: dict, rsd: bool, los: int = 2):
    """Yield CDM positions in Mpc/h per file, optionally shifted to redshift space along `los`."""
    import h5py
    import hdf5plugin  # noqa: F401

    a = 1.0 / (1.0 + header["redshift"])
    rsd_factor = np.sqrt(a) * (1.0 + header["redshift"]) / header["hubble_z"]  # Gadget stores v / sqrt(a)
    for filename in files:
        with h5py.File(filename, "r") as handle:
            pos = handle["PartType1/Coordinates"][:].astype(np.float64) / 1e3
            if rsd:
                pos[:, los] += handle["PartType1/Velocities"][:, los].astype(np.float64) * rsd_factor
        yield np.remainder(pos, header["boxsize"])


def paint_cic(chunks, nmesh: int, boxsize: float) -> np.ndarray:
    """Cloud-in-cell density contrast on an nmesh^3 periodic mesh."""
    grid = np.zeros(nmesh**3)
    for pos in chunks:
        x = pos * (nmesh / boxsize)
        i0 = np.floor(x).astype(np.int64)
        d = x - i0
        i0 %= nmesh
        i1 = (i0 + 1) % nmesh
        for cx in (0, 1):
            ix, wx = (i1[:, 0], d[:, 0]) if cx else (i0[:, 0], 1 - d[:, 0])
            for cy in (0, 1):
                iy, wy = (i1[:, 1], d[:, 1]) if cy else (i0[:, 1], 1 - d[:, 1])
                for cz in (0, 1):
                    iz, wz = (i1[:, 2], d[:, 2]) if cz else (i0[:, 2], 1 - d[:, 2])
                    grid += np.bincount((ix * nmesh + iy) * nmesh + iz, weights=wx * wy * wz, minlength=nmesh**3)
    grid = grid.reshape((nmesh,) * 3)
    return (grid / grid.mean() - 1.0).astype(np.float32)


def load_density(realization: int | str, redshift: float = 0.5, nmesh: int = 256, rsd: bool = False,
                 los: int = 2, root: Path = SNAPSHOT_ROOT, device=None):
    """CIC density contrast of one Quijote fiducial snapshot, and its header.

    With a torch ``device`` the particles are painted there (see ``wstmodel.measure_torch``).
    """
    snapdir = Path(root) / str(realization) / f"snapdir_{SNAPNUM[redshift]}"
    files = snapshot_files(snapdir)
    header = read_header(files)
    positions = iter_positions(files, header, rsd=rsd, los=los)
    if device is None:
        delta = paint_cic(positions, nmesh, header["boxsize"])
    else:
        from .measure_torch import paint_cic_torch

        delta = paint_cic_torch(positions, nmesh, header["boxsize"], device=device)
    return delta, header
