"""Real-space matter power spectrum of dsc-model (tree level or regulated one-loop EFT), binned like the data.

The model is dsc-model's (``eft_loop.py``),

    P(k) = P_L(k) + L_Lambda(k) - 2 cs2 k^2 P_L(k),

with the loop L regulated at Lambda = 0.5 h/Mpc by default (``cutoff=None``: unregulated SPT, as in
the WST model). Tree level is P_L. The measured ``Pdd`` is the bin average over lattice modes of a CIC
mesh that is not deconvolved, with the particle shot noise included, so the binned prediction is

    P_b = < W^2(k) P(|k|) >_b + (V / N) < sum_n W^2(k + 2 k_N n) >_b,
    W^2(k) = prod_i sinc^4(k_i H / 2),  sum_n W^2(k + 2 k_N n) = prod_i [1 - 2/3 sin^2(k_i H / 2)],

with signal aliasing neglected (k <= 0.3 h/Mpc << k_N). This replaces dsc-model's jaxpower window
matrix: P is computed on a coarse k grid (``knodes``) and the bin average is a fixed matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .eft_loop import EFTLoopConvention, LinearSpectrum, Quadrature, loop_integrals

ORDERS = ("tree", "one-loop")


def default_knodes(kmax: float, boxsize: float = 1000.0) -> np.ndarray:
    """Nodes spaced by the fundamental mode, from just below k_f to just beyond kmax."""
    kfund = 2 * np.pi / boxsize
    return kfund * np.arange(0.9, kmax / kfund + 2.5, 1.0)


def loop_convention(cutoff: float | None) -> EFTLoopConvention | None:
    return None if cutoff is None else EFTLoopConvention(cutoff=float(cutoff))


def matter_power_terms(klin, pklin, knodes, order: str = "one-loop", cutoff: float | None = 0.5,
                       quadrature: Quadrature = Quadrature()) -> np.ndarray:
    """Rows P_L and L_Lambda (zero at tree level) on ``knodes``, shape (2, nnodes)."""
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}; choose among {ORDERS}")
    spectrum = LinearSpectrum(np.asarray(klin, dtype="f8"), np.asarray(pklin, dtype="f8"))
    knodes = np.asarray(knodes, dtype="f8")
    loop = np.zeros_like(knodes)
    if order == "one-loop":
        loop = loop_integrals(knodes, np.zeros(1), spectrum, 0.0, loop_convention(cutoff), quadrature,
                              rsd=False)["loop"][:, 0]
    return np.array([spectrum(knodes), loop])


class LatticeBinning:
    """Linear map from P(k) on ``knodes`` to bin averages over the modes of a periodic box.

    ``matrix`` (nbins, nnodes) averages W^2(k) times a cubic spline through the nodes; ``noise``
    (nbins,) is the bin average of the aliased CIC shot-noise factor. Bins follow
    ``measure.PowerMultipoles`` (a mode belongs to bin b if edges[b] <= |k| < edges[b + 1]);
    ``k`` and ``nmodes`` are the mode-weighted bin centres and mode counts.
    """

    def __init__(self, edges, boxsize: float, nmesh: int, knodes, window: str | None = "cic"):
        self.edges = np.asarray(edges, dtype="f8")
        self.knodes = np.asarray(knodes, dtype="f8")
        kfund = 2 * np.pi / boxsize
        nmax = int(np.ceil(self.edges[-1] / kfund))
        if nmax >= nmesh // 2:
            raise ValueError("bins beyond the Nyquist frequency")
        n = np.arange(-nmax, nmax + 1)
        nx, ny, nz = (a.ravel() for a in np.meshgrid(n, n, n, indexing="ij"))
        n2 = nx**2 + ny**2 + nz**2
        kmag = kfund * np.sqrt(n2)
        index = np.digitize(kmag, self.edges) - 1
        inside = (index >= 0) & (index < len(self.edges) - 1) & (n2 > 0)
        index, n2 = index[inside], n2[inside]
        kmodes = kfund * np.stack([nx[inside], ny[inside], nz[inside]])
        if window == "cic":
            half = 0.5 * kmodes * boxsize / nmesh
            window2 = np.prod(np.sinc(half / np.pi) ** 4, axis=0)
            noise = np.prod(1.0 - 2.0 / 3.0 * np.sin(half) ** 2, axis=0)
        elif window is None:
            window2 = noise = np.ones(index.size)
        else:
            raise ValueError(f"unknown mass-assignment window {window!r}")
        if self.knodes[0] > kmag[inside].min() or self.knodes[-1] < kmag[inside].max():
            raise ValueError(f"knodes [{self.knodes[0]:.4f}, {self.knodes[-1]:.4f}] do not cover the modes "
                             f"[{kmag[inside].min():.4f}, {kmag[inside].max():.4f}]")
        nbins = len(self.edges) - 1
        self.nmodes = np.bincount(index, minlength=nbins).astype("f8")
        self.k = np.bincount(index, weights=kmag[inside], minlength=nbins) / self.nmodes
        self.noise = np.bincount(index, weights=noise, minlength=nbins) / self.nmodes
        # Window-weighted counts per (bin, distinct |k|), then the spline from the nodes to each |k|.
        unique, inverse = np.unique(n2, return_inverse=True)
        counts = np.zeros((nbins, unique.size))
        np.add.at(counts, (index, inverse), window2)
        spline = CubicSpline(self.knodes, np.eye(self.knodes.size), axis=0)(kfund * np.sqrt(unique))
        self.matrix = counts @ spline / self.nmodes[:, None]

    def __call__(self, power_nodes, shotnoise: float = 0.0):
        return self.matrix @ power_nodes + shotnoise * self.noise
