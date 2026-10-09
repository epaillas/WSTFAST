"""Storage of WST measurements, WST and P(k) data vectors, and covariance estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Coefficient, WSTConfig


def save_measurement(path: Path, result: dict, config: WSTConfig, metadata: dict) -> None:
    """Write every array of ``result`` (coefficients for all q, optional spectra) and the metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata, config=config.to_dict())
    np.savez(path, metadata=json.dumps(meta), **result)


def load_measurement(path: Path, q: float | None = None) -> dict:
    """Read a measurement; S0, S1 and S2 (S1m and S2m if stored) are returned for one exponent q (default: config's).

    Spectra (``k``, ``Pdd``, ``PUd``, ``PUU``, ...) are returned as stored. Files written before
    several q were stored at once hold a single q and no ``q`` array.
    """
    with np.load(path) as data:
        out = {name: np.asarray(data[name]) for name in data.files if name != "metadata"}
        out["metadata"] = json.loads(str(data["metadata"]))
    config = WSTConfig(**out["metadata"]["config"])
    q = config.q if q is None else float(q)
    if "q" in out:
        matches = np.flatnonzero(np.isclose(out["q"], q))
        if matches.size == 0:
            raise ValueError(f"{path} has q = {out['q'].tolist()}, not {q}")
        for name in ("S0", "S1", "S2", "S1m", "S2m"):
            if name in out:
                out[name] = out[name][matches[0]]
    elif not np.isclose(q, config.q):
        raise ValueError(f"{path} only has q = {config.q}")
    out["config"] = config.with_q(q)
    return out


def flatten(measurement: dict, coefficients) -> np.ndarray:
    """Data vector: S1(j, l) and reduced S21 = S2(j1, j2, l) / S1(j1, l)."""
    s1, s2 = measurement["S1"], measurement["S2"]
    return np.array([s1[c.j, c.ell] if c.kind == "S1" else s2[c.j, c.j2, c.ell] / s1[c.j, c.ell]
                     for c in coefficients])


def sample_covariance(vectors: np.ndarray, kind: str = "sample", of_mean: bool = False) -> np.ndarray:
    """Covariance of one realization (or of the mean) of ``vectors`` (nreal, ndata), Hartlap-corrected for 'sample'.

    'sample' needs more realizations than data points; 'diagonal' keeps only the variances and
    is meant for smoke tests with a handful of local snapshots.
    """
    nreal, ndata = vectors.shape
    if nreal < 2:
        raise ValueError("at least two realizations are needed to estimate a covariance")
    covariance = np.cov(vectors, rowvar=False, ddof=1).reshape(ndata, ndata)
    if kind == "sample":
        if nreal < ndata + 3:
            raise ValueError(f"{nreal} realizations cannot estimate a {ndata}-point covariance; "
                             "measure more realizations or use kind='diagonal'")
        covariance = covariance / ((nreal - ndata - 2.0) / (nreal - 1.0))
    elif kind == "diagonal":
        covariance = np.diag(np.diag(covariance))
    else:
        raise ValueError(f"unknown covariance kind {kind!r}")
    return covariance / nreal if of_mean else covariance


@dataclass
class WSTDataset:
    """Data vectors of all realizations found for one space (real or rsd)."""

    vectors: np.ndarray  # (nrealizations, ndata)
    coefficients: list
    config: WSTConfig
    metadata: dict
    files: list

    @property
    def mean(self) -> np.ndarray:
        return self.vectors.mean(axis=0)

    @property
    def shotnoise(self) -> float:
        """Particle shot noise V / N of the measured field, in (Mpc/h)^3."""
        return self.metadata["boxsize"] ** 3 / self.metadata["nparticles"]

    def covariance(self, kind: str = "sample", of_mean: bool = False) -> np.ndarray:
        return sample_covariance(self.vectors, kind=kind, of_mean=of_mean)


def load_dataset(data_dir: Path, space: str, coefficients: list[Coefficient], q: float | None = None) -> WSTDataset:
    files = sorted((Path(data_dir) / space).glob("wst_r*.npz"))
    if not files:
        raise FileNotFoundError(f"no measurements in {Path(data_dir) / space}")
    measurements = [load_measurement(path, q=q) for path in files]
    config = measurements[0]["config"]
    keys = ("redshift", "boxsize", "nmesh", "nparticles", "space")
    reference = {key: measurements[0]["metadata"][key] for key in keys}
    for path, measurement in zip(files, measurements):
        if measurement["config"] != config or any(measurement["metadata"][key] != reference[key] for key in keys):
            raise ValueError(f"{path} was measured with different settings")
    vectors = np.array([flatten(measurement, coefficients) for measurement in measurements])
    return WSTDataset(vectors=vectors, coefficients=list(coefficients), config=config, metadata=reference,
                      files=[str(path) for path in files])


@dataclass
class PowerDataset:
    """Real-space matter power spectra (monopole of delta, shot noise included, CIC not deconvolved)."""

    vectors: np.ndarray  # (nrealizations, nbins)
    k: np.ndarray  # mode-weighted bin centres
    nmodes: np.ndarray
    edges: np.ndarray
    metadata: dict
    files: list

    @property
    def mean(self) -> np.ndarray:
        return self.vectors.mean(axis=0)

    @property
    def shotnoise(self) -> float:
        """Particle shot noise V / N of the measured field, in (Mpc/h)^3."""
        return self.metadata["boxsize"] ** 3 / self.metadata["nparticles"]

    def covariance(self, kind: str = "sample", of_mean: bool = False) -> np.ndarray:
        return sample_covariance(self.vectors, kind=kind, of_mean=of_mean)


def load_power_dataset(data_dir: Path, space: str = "real", kmin: float = 0.0, kmax: float = 0.2, rebin: int = 1,
                       files=None, ells=(0,)) -> PowerDataset:
    """P_dd multipoles ``ells`` (concatenated, ell-major) of every realization in bins of ``rebin`` x k_f, keeping bins
    with kmin <= k <= kmax.

    Bins are merged from the first measured bin by mode-weighted averages; ``files`` fixes the
    realizations (default: every ``wst_r*.npz`` of ``data_dir / space``). The files store ell = 0, 2, 4.
    """
    if any(ell not in (0, 2, 4) for ell in ells):
        raise ValueError("stored multipoles are ell = 0, 2, 4")
    files = sorted((Path(data_dir) / space).glob("wst_r*.npz")) if files is None else [Path(f) for f in files]
    if not files:
        raise FileNotFoundError(f"no measurements in {Path(data_dir) / space}")
    keys = ("redshift", "boxsize", "nmesh", "nparticles", "space")
    with np.load(files[0]) as data:
        k, nmodes, edges = data["k"], data["nmodes"], data["k_edges"]
        reference = {key: json.loads(str(data["metadata"]))[key] for key in keys}
    nk = (k.size // rebin) * rebin
    weights = nmodes[:nk].reshape(-1, rebin)
    kbin = (k[:nk].reshape(-1, rebin) * weights).sum(axis=1) / weights.sum(axis=1)
    keep = (kbin >= kmin) & (kbin <= kmax)
    if not keep.any():
        raise ValueError(f"no bins with {kmin} <= k <= {kmax}")
    vectors = []
    for path in files:
        with np.load(path) as data:
            metadata = json.loads(str(data["metadata"]))
            if any(metadata[key] != reference[key] for key in keys) or not np.array_equal(data["k"], k):
                raise ValueError(f"{path} was measured with different settings")
            power = [(data["Pdd"][ell // 2, :nk].reshape(-1, rebin) * weights).sum(axis=1) / weights.sum(axis=1)
                     for ell in ells]
        vectors.append(np.concatenate([p[keep] for p in power]))
    bin_edges = edges[:nk + 1:rebin]
    first = np.flatnonzero(keep)[0]
    return PowerDataset(vectors=np.array(vectors), k=kbin[keep], nmodes=weights.sum(axis=1)[keep],
                        edges=bin_edges[first:first + keep.sum() + 1], metadata=reference,
                        files=[str(path) for path in files])


def load_s1m_dataset(data_dir: Path, space: str, coefficients, q: float, files=None) -> WSTDataset:
    """Line-of-sight-resolved S1m of every |m| block of ``coefficients`` (coefficient-major, |m| = 0..l)."""
    files = sorted((Path(data_dir) / space).glob("wst_r*.npz")) if files is None else [Path(f) for f in files]
    if not files:
        raise FileNotFoundError(f"no measurements in {Path(data_dir) / space}")
    measurements = [load_measurement(path, q=q) for path in files]
    if "S1m" not in measurements[0]:
        raise ValueError(f"{files[0]} has no line-of-sight-resolved coefficients (measure with --los-resolved)")
    vectors = np.array([np.concatenate([m["S1m"][c.j, c.ell, :c.ell + 1] for c in coefficients])
                        for m in measurements])
    keys = ("redshift", "boxsize", "nmesh", "nparticles", "space")
    metadata = {key: measurements[0]["metadata"][key] for key in keys}
    return WSTDataset(vectors=vectors, coefficients=list(coefficients), config=measurements[0]["config"],
                      metadata=metadata, files=[str(path) for path in files])
