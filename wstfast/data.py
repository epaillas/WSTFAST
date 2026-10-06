"""Storage of WST measurements, data-vector selection and covariance estimation."""

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
    """Read a measurement; S0, S1 and S2 are returned for one exponent q (default: the config's).

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
        for name in ("S0", "S1", "S2"):
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
        """Covariance of one realization (or of the mean), Hartlap-corrected for 'sample'.

        'sample' needs more realizations than data points; 'diagonal' keeps only the variances and
        is meant for smoke tests with a handful of local snapshots.
        """
        nreal, ndata = self.vectors.shape
        if nreal < 2:
            raise ValueError("at least two realizations are needed to estimate a covariance")
        covariance = np.cov(self.vectors, rowvar=False, ddof=1).reshape(ndata, ndata)
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
