"""Read Quijote Gadget-HDF5 snapshots and FoF halo catalogues, and paint density contrasts on a mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SNAPSHOT_ROOT = Path("/Users/epaillas/data/quijote/snapshots/fiducial")
HALO_ROOT = Path("/Users/epaillas/data/quijote/halos/FoF/fiducial")
#: Quijote fiducial (Omega_m, Omega_L), for the redshift-space mapping of halos.
QUIJOTE_OMEGA = (0.3175, 0.6825)
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
                 los: int = 2, root: Path = SNAPSHOT_ROOT, device=None, nbar: float | None = None):
    """CIC density contrast of one Quijote fiducial snapshot, and its header.

    With a torch ``device`` the particles are painted there (see ``wstfast.measure_torch``). With ``nbar``
    [(h/Mpc)^3], a random subsample of that mean density is painted (a Poisson tracer with b1 = 1 and no other bias,
    seeded by the realization); ``header["nparticles"]`` is then the number kept.
    """
    snapdir = Path(root) / str(realization) / f"snapdir_{SNAPNUM[redshift]}"
    files = snapshot_files(snapdir)
    header = read_header(files)
    positions = iter_positions(files, header, rsd=rsd, los=los)
    if nbar is not None:
        fraction = nbar * header["boxsize"] ** 3 / header["nparticles"]
        if not 0 < fraction <= 1:
            raise ValueError(f"nbar = {nbar} needs a fraction {fraction} of the particles")
        rng = np.random.default_rng([int(realization), 20261010])
        kept = []
        for chunk in positions:
            kept.append(chunk[rng.random(len(chunk)) < fraction])
        positions = kept
        header = {**header, "nparticles": int(sum(len(chunk) for chunk in kept)), "nbar": nbar}
    if device is None:
        delta = paint_cic(positions, nmesh, header["boxsize"])
    else:
        from .measure_torch import paint_cic_torch

        delta = paint_cic_torch(positions, nmesh, header["boxsize"], device=device)
    return delta, header


def read_fof(groupdir) -> dict:
    """Read a Gadget FoF catalogue (``group_tab_NNN.*`` in ``groupdir``), as Pylians' readfof.FoF_catalog.

    Returns ``mass`` [Msun/h], ``pos`` [Mpc/h], ``vel`` (as stored: peculiar km/s / (1 + z)) and ``npart``.
    """
    groupdir = Path(groupdir)
    files = sorted(groupdir.glob("group_tab_*.*"), key=lambda p: int(p.suffix[1:]))
    if not files:
        raise FileNotFoundError(f"no group_tab files in {groupdir}")
    out = {name: [] for name in ("npart", "mass", "pos", "vel")}
    nfiles = None
    for filename in files:
        with open(filename, "rb") as handle:
            ngroups = int(np.fromfile(handle, np.int32, 1)[0])
            np.fromfile(handle, np.int32, 2)  # TotNgroups, Nids
            np.fromfile(handle, np.uint64, 1)  # TotNids
            nfiles = int(np.fromfile(handle, np.uint32, 1)[0])
            out["npart"].append(np.fromfile(handle, np.int32, ngroups))
            np.fromfile(handle, np.int32, ngroups)  # offsets
            out["mass"].append(np.fromfile(handle, np.float32, ngroups).astype("f8") * 1e10)
            out["pos"].append(np.fromfile(handle, np.float32, 3 * ngroups).reshape(-1, 3).astype("f8") / 1e3)
            out["vel"].append(np.fromfile(handle, np.float32, 3 * ngroups).reshape(-1, 3).astype("f8"))
            np.fromfile(handle, np.float32, 12 * ngroups)  # GroupTLen, GroupTMass
            if handle.read(1):
                raise IOError(f"{filename}: unread bytes at the end of the file")
    if nfiles != len(files):
        raise IOError(f"{groupdir}: {len(files)} files, header says {nfiles}")
    return {name: np.concatenate(value) for name, value in out.items()}


def halo_positions(realization: int | str, redshift: float = 0.5, mmin: float = 1e13, rsd: bool = False,
                   los: int = 2, root: Path = HALO_ROOT, boxsize: float = 1000.0):
    """Positions [Mpc/h] of the FoF halos of mass >= ``mmin`` [Msun/h], optionally in redshift space along ``los``."""
    catalogue = read_fof(Path(root) / str(realization) / f"groups_{SNAPNUM[redshift]}")
    keep = catalogue["mass"] >= mmin
    pos = catalogue["pos"][keep]
    if rsd:
        omega_m, omega_l = QUIJOTE_OMEGA
        hubble = 100.0 * np.sqrt(omega_m * (1 + redshift) ** 3 + omega_l)  # km/s / (Mpc/h)
        velocity = catalogue["vel"][keep, los] * (1 + redshift)  # peculiar km/s
        pos[:, los] += velocity * (1 + redshift) / hubble
    return np.remainder(pos, boxsize)


def load_halo_density(realization: int | str, redshift: float = 0.5, nmesh: int = 256, rsd: bool = False,
                      los: int = 2, mmin: float = 1e13, root: Path = HALO_ROOT, boxsize: float = 1000.0, device=None):
    """CIC density contrast of the halos of one Quijote fiducial FoF catalogue, and a header with their number."""
    pos = halo_positions(realization, redshift=redshift, mmin=mmin, rsd=rsd, los=los, root=root, boxsize=boxsize)
    if device is None:
        delta = paint_cic([pos], nmesh, boxsize)
    else:
        from .measure_torch import paint_cic_torch

        delta = paint_cic_torch([pos], nmesh, boxsize, device=device)
    omega_m, omega_l = QUIJOTE_OMEGA
    header = dict(boxsize=boxsize, redshift=redshift, omega_m=omega_m, omega_l=omega_l, nparticles=len(pos),
                  hubble_z=100.0 * np.sqrt(omega_m * (1 + redshift) ** 3 + omega_l), mmin=mmin)
    return delta, header
