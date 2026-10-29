"""GPU (torch) backend of the WST estimator; same algorithm and outputs as ``wstfast.measure``.

The spherical harmonics of every l are computed once (on the CPU, with scipy) and kept on the
device, so that a configuration is set up once and then applied to many realizations, as with
kymatio's filter bank. Coefficients are accumulated in float64; fields and FFTs are float32.

    lattice = TorchLattice(256, device="cuda")
    result = measure_wst_torch(delta, config, qs=[0.5, 0.8], lattice=lattice,
                               spectra=TorchPowerMultipoles(lattice, boxsize=1000.0))

The survey estimator (``wstfast.survey.measure_wst_survey``: support, weights, local line of sight) has its torch
counterpart in ``measure_wst_survey_torch``; ``survey.LambdaWindow``, ``survey.noise_variance_maps`` and
``survey.signal_variances`` run on the device when given a ``TorchLattice``.
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
        self.numpy.__dict__.pop("_harmonics", None)  # the host copies (30 GB at 688^3, l <= 6), rebuilt if released

    def harmonics(self, ell: int) -> torch.Tensor:
        """Real spherical harmonics of degree l, shape (2l+1, n, n, n//2+1), sum_m Y^2 = 1."""
        if ell not in self._harmonics:
            self._harmonics[ell] = torch.from_numpy(np.stack(self.numpy.harmonics(ell))).to(self.device)
        return self._harmonics[ell]

    def release(self):
        """Drop the cached harmonics (30 GB at 688^3, l <= 6) and any tensor only kept by reference cycles from the
        device; the harmonics are rebuilt on demand."""
        import gc

        self._harmonics.clear()
        gc.collect()
        torch.cuda.empty_cache()

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


def wavelet_modulus(fk, lattice: TorchLattice, sigma: float, ell: int, los: bool = False):
    """|field * psi_{sigma,l}| given the field's rfft (sigma in cells).

    With ``los``, also return the line-of-sight-resolved moduli U_|m| (|m| = 0..l), as ``wstfast.measure``.
    """
    filtered = fk * (lattice.radial(sigma, ell) * lattice.harmonics(ell)) * (-1j) ** ell  # (2l+1, ...), m = -l..l
    x2 = torch.fft.irfftn(filtered, s=(lattice.n,) * 3, dim=(-3, -2, -1)) ** 2
    total = torch.sqrt(x2.sum(dim=0))
    if not los:
        return total
    per_m = [torch.sqrt(x2[ell])] + [torch.sqrt(x2[ell + m] + x2[ell - m]) for m in range(1, ell + 1)]
    return total, per_m


def _moments(field, qs) -> np.ndarray:
    absolute = field.abs().to(torch.float64)
    return np.array([float((absolute**q).mean()) for q in qs])


def measure_wst_torch(delta, config: WSTConfig, qs=None, lattice: TorchLattice | None = None,
                      spectra: TorchPowerMultipoles | None = None, los: bool = False) -> dict:
    """Same outputs as ``wstfast.measure.measure_wst``, computed on ``lattice.device``."""
    qs = np.atleast_1d(config.q if qs is None else qs).astype(float)
    nmesh = delta.shape[0]
    lattice = lattice or TorchLattice(nmesh)
    field = torch.as_tensor(np.asarray(delta, dtype=np.float32), device=lattice.device)
    fk = torch.fft.rfftn(field)
    nq, nj, nl, nl2 = len(qs), config.J + 1, config.L + 1, config.lmax2 + 1
    out = {"q": qs, "S0": _moments(field, qs),
           "S1": np.full((nq, nj, nl), np.nan),
           "S2": np.full((nq, nj, nj, nl2), np.nan)}
    if los:
        out.update(S1m=np.full((nq, nj, nl, nl), np.nan), S2m=np.full((nq, nj, nj, nl2, nl2, nl2), np.nan))
    if spectra is not None:
        nell, nk = len(spectra.ells), spectra.nbins
        out.update(k=spectra.k, k_edges=spectra.edges, nmodes=spectra.nmodes, Pdd=spectra(fk, fk),
                   PUd=np.full((nj, nl, nell, nk), np.nan), PUU=np.full((nj, nl, nell, nk), np.nan),
                   Umean=np.full((nj, nl), np.nan))
        if los:
            out.update(PUd_m=np.full((nj, nl, nl, nell, nk), np.nan), PUU_m=np.full((nj, nl, nl, nell, nk), np.nan),
                       Umean_m=np.full((nj, nl, nl), np.nan))
    pairs = config.second_layer_pairs()
    for ell in range(nl):
        for j1 in range(nj):
            sigma1 = config.sigma0 * config.step**j1
            second_layer = ell <= config.lmax2 and any(pair[0] == j1 for pair in pairs)
            if los:
                u1, per_m = wavelet_modulus(fk, lattice, sigma1, ell, los=True)
                fields = [(None, u1)] + list(enumerate(per_m))
            else:
                fields = [(None, wavelet_modulus(fk, lattice, sigma1, ell))]
            for m1, u in fields:
                if m1 is None:
                    out["S1"][:, j1, ell] = _moments(u, qs)
                else:
                    out["S1m"][:, j1, ell, m1] = _moments(u, qs)
                if spectra is None and not second_layer:
                    continue
                mean = float(u.to(torch.float64).mean())
                fu = torch.fft.rfftn(u - mean)
                if spectra is not None:
                    suffix, index = ("", (j1, ell)) if m1 is None else ("_m", (j1, ell, m1))
                    out["Umean" + suffix][index] = mean
                    out["PUd" + suffix][index] = spectra(fu, fk)
                    out["PUU" + suffix][index] = spectra(fu, fu)
                if not second_layer:
                    continue
                fu[0, 0, 0] = mean * nmesh**3  # restore the mean for the low-pass (l = 0) second layer
                for _, j2 in (pair for pair in pairs if pair[0] == j1):
                    sigma2 = config.sigma0 * config.step**j2
                    if m1 is None:
                        out["S2"][:, j1, j2, ell] = _moments(wavelet_modulus(fu, lattice, sigma2, ell), qs)
                    else:
                        _, per_m2 = wavelet_modulus(fu, lattice, sigma2, ell, los=True)
                        for m2, u2 in enumerate(per_m2):
                            out["S2m"][:, j1, j2, ell, m1, m2] = _moments(u2, qs)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Survey geometry (``wstfast.survey``)


def survey_components(fk, lattice: TorchLattice, sigma: float, ell: int, index) -> torch.Tensor:
    """Real components X_m (m = -l..l) of the field with rfft ``fk`` on the cells ``index``: (2l + 1, ncells).
    One component at a time, so that the peak memory is one mesh rather than 2l + 1 (688^3 at l = 6 overflows 80 GB)."""
    filtered = fk * lattice.radial(sigma, ell) * (-1j) ** ell
    return torch.stack([torch.fft.irfftn(filtered * y, s=(lattice.n,) * 3).reshape(-1)[index]
                        for y in lattice.harmonics(ell)])


def squared_kernel(lattice: TorchLattice, sigma: float, ell: int) -> np.ndarray:
    """sum_m psi_{sigma,l}^m(x)^2 in real space (n, n, n)."""
    filtered = (lattice.radial(sigma, ell) * lattice.harmonics(ell)).to(torch.complex64) * (-1j) ** ell
    return (torch.fft.irfftn(filtered, s=(lattice.n,) * 3, dim=(-3, -2, -1)) ** 2).sum(dim=0).cpu().numpy()


class TorchLOSRotation:
    """``wstfast.survey.LOSRotation`` on a torch device."""

    def __init__(self, ell: int, device):
        from .survey import LOSRotation

        reference = LOSRotation(ell)
        self.ell, self.device = ell, device
        self.terms = {axis: (torch.as_tensor(c0, dtype=torch.float32, device=device),
                             [(p, torch.as_tensor(a, dtype=torch.float32, device=device),
                               torch.as_tensor(b, dtype=torch.float32, device=device)) for p, a, b in harmonics])
                      for axis, (c0, harmonics) in reference.terms.items()}
        self._reference = reference

    def tables(self, theta, phi):
        return tuple(torch.as_tensor(t, device=self.device) for t in self._reference.tables(theta, phi))

    def rotate(self, x: torch.Tensor, tables) -> torch.Tensor:
        for axis, cos, sin in ((2, tables[0], tables[1]), (1, tables[2], tables[3])):
            c0, harmonics = self.terms[axis]
            out = c0 @ x
            for p, a, b in harmonics:
                out = out + cos[p - 1] * (a @ x) + sin[p - 1] * (b @ x)
            x = out
        return x


def subtract_mean_torch(u, labels=None):
    """``survey.subtract_mean`` for a tensor; ``labels`` a numpy array or a tensor of the same device."""
    if labels is None:
        return u - u.to(torch.float64).mean().to(u.dtype)
    labels = torch.as_tensor(labels, device=u.device)
    sums = torch.bincount(labels, weights=u.to(torch.float64))
    counts = torch.bincount(labels).to(torch.float64)
    return u - (sums / counts)[labels].to(u.dtype)


def measure_wst_survey_torch(delta, config: WSTConfig, support=None, weights=None, qs=None, los=None,
                             erosion: float = 0.0, moment_support=None, lattice: TorchLattice | None = None,
                             second_layer_bins: dict | None = None, fields=None) -> dict:
    """Same arguments and outputs as ``wstfast.survey.measure_wst_survey``, computed on ``lattice.device``."""
    from .survey import _angles, _Moments, m_powers

    qs = np.atleast_1d(config.q if qs is None else qs).astype(float)
    nmesh = delta.shape[0]
    lattice = lattice or TorchLattice(nmesh)
    device = lattice.device
    support = np.ones(delta.shape, dtype=bool) if support is None else np.asarray(support, dtype=bool)
    moment_support = support if moment_support is None else np.asarray(moment_support, dtype=bool) & support
    moments = _Moments(support, moment_support, weights, erosion, config, qs)
    index = torch.as_tensor(moments.index, device=device)
    inner = torch.as_tensor(moments.inner, device=device)
    weight = {j: torch.as_tensor(w, dtype=torch.float64, device=device) for j, w in moments.weights.items()}

    def average(values, j):
        w = weight[j] if len(values) == len(index) else weight[j][inner]
        absolute = values.abs().to(torch.float64)
        return np.array([float(w @ absolute**q) for q in qs])

    if los is None or isinstance(los, str):
        if los not in (None, "z"):
            raise ValueError(f"unknown line of sight {los!r}")
        theta = phi = None
    else:
        los = np.asarray(los)
        if los.ndim == 4:
            los = los.reshape(3, -1)[:, moments.index]
        theta, phi = _angles(los.T.astype(np.float64) if los.ndim == 2 else los.astype(np.float64))
    resolve = los is not None

    field = torch.as_tensor(np.where(support, delta, 0.0).astype(np.float32), device=device)
    fk = torch.fft.rfftn(field)
    nq, nj, nl, nl2 = len(qs), config.J + 1, config.L + 1, config.lmax2 + 1
    out = {"q": qs, "S0": average(field.reshape(-1)[index], 0),
           "S1": np.full((nq, nj, nl), np.nan), "S2": np.full((nq, nj, nj, nl2), np.nan)}
    if resolve:
        out.update(S1m=np.full((nq, nj, nl, nl), np.nan), S2m=np.full((nq, nj, nj, nl2, nl2, nl2), np.nan))
    pairs = config.second_layer_pairs()
    for ell in range(nl):
        rotation = TorchLOSRotation(ell, device) if theta is not None and ell > 0 else None
        frames = {}

        def rotate(x, everywhere):
            key = everywhere or np.ndim(theta) == 0
            if key not in frames:
                cells = slice(None) if key else moments.inner
                frames[key] = rotation.tables(theta if np.ndim(theta) == 0 else theta[cells],
                                              phi if np.ndim(phi) == 0 else phi[cells])
            return rotation.rotate(x, frames[key])

        def moduli(f, sigma, resolve_m=True, everywhere=False):
            cells = index if everywhere else index[inner]
            x = survey_components(f, lattice, sigma, ell, cells)
            total = torch.sqrt((x * x).sum(dim=0))
            if not (resolve and resolve_m):
                return total, []
            if rotation is not None:
                x = rotate(x, everywhere)
            return total, [torch.sqrt(p) for p in m_powers(x)]

        for j1 in range(nj):
            second_layer = ell <= config.lmax2 and any(pair[0] == j1 for pair in pairs)
            u1, per_m = moduli(fk, config.sigma0 * config.step**j1, everywhere=second_layer)
            out["S1"][:, j1, ell] = average(u1, j1)
            for m1, u in enumerate(per_m):
                out["S1m"][:, j1, ell, m1] = average(u, j1)
            if not second_layer:
                continue
            for m1, u in [(None, u1)] + list(enumerate(per_m)):
                second = torch.zeros(nmesh**3, dtype=torch.float32, device=device)
                second[index] = u if ell == 0 else subtract_mean_torch(u, None if second_layer_bins is None
                                                                       else second_layer_bins[j1, ell])
                if fields is not None and ell >= 1:
                    fields(j1, ell, m1, second.reshape((nmesh,) * 3).cpu().numpy(), float(u.to(torch.float64).mean()))
                fu = torch.fft.rfftn(second.reshape((nmesh,) * 3))
                for _, j2 in (pair for pair in pairs if pair[0] == j1):
                    u2, per_m2 = moduli(fu, config.sigma0 * config.step**j2, resolve_m=m1 is not None)
                    if m1 is None:
                        out["S2"][:, j1, j2, ell] = average(u2, j2)
                    else:
                        for m2, v in enumerate(per_m2):
                            out["S2m"][:, j1, j2, ell, m1, m2] = average(v, j2)
    return out


def paint_cic_torch(chunks, nmesh: int, boxsize: float, device="auto") -> np.ndarray:
    """Cloud-in-cell density contrast on the device; same result as ``wstfast.quijote.paint_cic``."""
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
