"""GPU (torch) backend of the WST estimator; same algorithm and outputs as ``wstmodel.measure``.

The spherical harmonics of every l are computed once (on the CPU, with scipy) and kept on the
device, so that a configuration is set up once and then applied to many realizations, as with
kymatio's filter bank. Coefficients are accumulated in float64; fields and FFTs are float32.

    lattice = TorchLattice(256, device="cuda")
    result = measure_wst_torch(delta, config, qs=[0.5, 0.8], lattice=lattice,
                               spectra=TorchPowerMultipoles(lattice, boxsize=1000.0))
"""

from __future__ import annotations

import numpy as np
import torch

from .config import WSTConfig
from .measure import Lattice, PowerMultipoles


def default_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


class TorchLattice:
    """Fourier lattice on a torch device, with cached spherical harmonics."""

    def __init__(self, nmesh: int, device="auto", lmax: int | None = None):
        self.device = default_device(device) if isinstance(device, str) else device
        self.n = nmesh
        self.numpy = Lattice(nmesh)
        self.kmag = torch.from_numpy(self.numpy.kmag).to(self.device)
        self._harmonics = {}
        for ell in range(lmax + 1 if lmax is not None else 0):
            self.harmonics(ell)

    def harmonics(self, ell: int) -> torch.Tensor:
        """Real spherical harmonics of degree l, shape (2l+1, n, n, n//2+1), sum_m Y^2 = 1."""
        if ell not in self._harmonics:
            self._harmonics[ell] = torch.from_numpy(np.stack(self.numpy.harmonics(ell))).to(self.device)
        return self._harmonics[ell]

    def radial(self, sigma: float, ell: int) -> torch.Tensor:
        x = sigma * self.kmag
        return x**ell * torch.exp(-0.5 * x**2)


class TorchPowerMultipoles:
    """Binned power-spectrum multipoles (line of sight z) on a torch device."""

    def __init__(self, lattice: TorchLattice, boxsize: float, kmax: float = 0.4, ells=(0, 2, 4)):
        reference = PowerMultipoles(lattice.numpy, boxsize, kmax=kmax, ells=ells)
        device = lattice.device
        self.k, self.edges, self.nmodes, self.ells = reference.k, reference.edges, reference.nmodes, reference.ells
        self.nbins, self.norm = reference.nbins, reference.norm
        self.inside = torch.from_numpy(reference.inside).to(device)
        self.index = torch.from_numpy(reference.index[reference.inside]).to(device)
        self.weights = [torch.from_numpy(w).to(device=device, dtype=torch.float64) for w in reference.weights]
        self._nmodes = torch.from_numpy(reference.nmodes).to(device)

    def __call__(self, fa, fb) -> np.ndarray:
        cross = (fa * torch.conj(fb)).real.reshape(-1)[self.inside].to(torch.float64)
        out = torch.stack([torch.bincount(self.index, weights=w * cross, minlength=self.nbins) for w in self.weights])
        return (out / self._nmodes * self.norm).cpu().numpy()


def wavelet_modulus(fk, lattice: TorchLattice, sigma: float, ell: int) -> torch.Tensor:
    """|field * psi_{sigma,l}| given the field's rfft (sigma in cells)."""
    filtered = fk * (lattice.radial(sigma, ell) * lattice.harmonics(ell)) * (-1j) ** ell  # (2l+1, ...)
    x = torch.fft.irfftn(filtered, s=(lattice.n,) * 3, dim=(-3, -2, -1))
    return torch.sqrt((x * x).sum(dim=0))


def _moments(field, qs) -> np.ndarray:
    absolute = field.abs().to(torch.float64)
    return np.array([float((absolute**q).mean()) for q in qs])


def measure_wst_torch(delta, config: WSTConfig, qs=None, lattice: TorchLattice | None = None,
                      spectra: TorchPowerMultipoles | None = None) -> dict:
    """Same outputs as ``wstmodel.measure.measure_wst``, computed on ``lattice.device``."""
    qs = np.atleast_1d(config.q if qs is None else qs).astype(float)
    nmesh = delta.shape[0]
    lattice = lattice or TorchLattice(nmesh)
    field = torch.as_tensor(np.asarray(delta, dtype=np.float32), device=lattice.device)
    fk = torch.fft.rfftn(field)
    nq, nj = len(qs), config.J + 1
    out = {"q": qs, "S0": _moments(field, qs),
           "S1": np.full((nq, nj, config.L + 1), np.nan),
           "S2": np.full((nq, nj, nj, config.lmax2 + 1), np.nan)}
    if spectra is not None:
        nell, nk = len(spectra.ells), spectra.nbins
        out.update(k=spectra.k, k_edges=spectra.edges, nmodes=spectra.nmodes, Pdd=spectra(fk, fk),
                   PUd=np.full((nj, config.L + 1, nell, nk), np.nan), PUU=np.full((nj, config.L + 1, nell, nk), np.nan),
                   Umean=np.full((nj, config.L + 1), np.nan))
    pairs = config.second_layer_pairs()
    for ell in range(config.L + 1):
        for j1 in range(nj):
            u1 = wavelet_modulus(fk, lattice, config.sigma0 * config.step**j1, ell)
            out["S1"][:, j1, ell] = _moments(u1, qs)
            second_layer = ell <= config.lmax2 and any(pair[0] == j1 for pair in pairs)
            if spectra is None and not second_layer:
                continue
            mean = float(u1.to(torch.float64).mean())
            fu = torch.fft.rfftn(u1 - mean)
            if spectra is not None:
                out["Umean"][j1, ell] = mean
                out["PUd"][j1, ell] = spectra(fu, fk)
                out["PUU"][j1, ell] = spectra(fu, fu)
            if not second_layer:
                continue
            fu[0, 0, 0] = mean * nmesh**3  # restore the mean for the low-pass (l = 0) second layer
            for _, j2 in (pair for pair in pairs if pair[0] == j1):
                u2 = wavelet_modulus(fu, lattice, config.sigma0 * config.step**j2, ell)
                out["S2"][:, j1, j2, ell] = _moments(u2, qs)
    return out


def paint_cic_torch(chunks, nmesh: int, boxsize: float, device="auto") -> np.ndarray:
    """Cloud-in-cell density contrast on the device; same result as ``wstmodel.quijote.paint_cic``."""
    device = default_device(device) if isinstance(device, str) else device
    grid = torch.zeros(nmesh**3, dtype=torch.float64, device=device)
    for pos in chunks:
        x = torch.as_tensor(pos, dtype=torch.float64, device=device) * (nmesh / boxsize)
        i0 = torch.floor(x).long()
        d = x - i0
        i0 = i0 % nmesh
        i1 = (i0 + 1) % nmesh
        for cx in (0, 1):
            ix, wx = (i1[:, 0], d[:, 0]) if cx else (i0[:, 0], 1 - d[:, 0])
            for cy in (0, 1):
                iy, wy = (i1[:, 1], d[:, 1]) if cy else (i0[:, 1], 1 - d[:, 1])
                for cz in (0, 1):
                    iz, wz = (i1[:, 2], d[:, 2]) if cz else (i0[:, 2], 1 - d[:, 2])
                    grid.index_add_(0, (ix * nmesh + iy) * nmesh + iz, wx * wy * wz)
        del x, i0, i1, d
    grid = grid.reshape((nmesh,) * 3)
    return (grid / grid.mean() - 1.0).to(torch.float32).cpu().numpy()
