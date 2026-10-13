"""WST of a density field in survey geometry: masked, weighted moments and the local line of sight.

See docs/survey_plan.md. The field is the overdensity normalised by the randoms, delta = (n_g - alpha n_r) / (alpha n_r),
painted in a padded, non-periodic box (``SurveyMesh``), so that the signal variance of every wavelet block is the box
value inside the footprint and only the noise varies with position. ``measure_wst_survey`` then differs from
``measure.measure_wst`` in three ways:

* **Support and weights.** Every moment is a weighted average over the support (the cells inside the footprint),
  optionally eroded by ``erosion`` x sigma_j for scale j. The weights are FKP-like, w = (nbar P0 / (1 + nbar P0))^2 per
  cell (``fkp_weights``): with delta normalised by the randoms, FKP weighting acts on the volume average of a quadratic
  statistic, not on the field.
* **Local line of sight.** The |m| moduli are about the line of sight of each cell, not about z: the 2l + 1 real
  components X_m of the wavelet transform are rotated cell by cell to the frame where the line of sight is the polar
  axis (``LOSRotation``). The rotation is exact; inside one wavelet the line of sight is taken constant.
* **Second layer.** The first-layer modulus U is set to zero outside the support, after subtracting its mean over
  the support (l >= 1; the l = 0 low-pass keeps the mean, as in the box), so that the edge of the footprint is not
  a step of height <U>.

With a full support, uniform weights and line of sight z, the coefficients are those of ``measure_wst``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import WSTConfig
from .measure import Lattice, _irfft, _rfft, real_harmonics


# ---------------------------------------------------------------------------------------------------------------------
# Rotation of real harmonic components to the local line of sight


def _fibonacci_sphere(npoints: int) -> np.ndarray:
    index = np.arange(npoints) + 0.5
    cos_theta = 1.0 - 2.0 * index / npoints
    phi = np.pi * (1.0 + 5**0.5) * index
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=-1)


def _angles(vectors: np.ndarray):
    """Polar angles (theta, phi) of unit vectors of shape (..., 3)."""
    return np.arccos(np.clip(vectors[..., 2], -1.0, 1.0)), np.arctan2(vectors[..., 1], vectors[..., 0])


def _harmonic_matrix(ell: int, vectors: np.ndarray) -> np.ndarray:
    return np.array(real_harmonics(ell, *_angles(vectors)))


def axis_rotation(axis: int, angle: float) -> np.ndarray:
    """Active rotation matrix by ``angle`` about the coordinate axis ``axis`` (0, 1, 2 = x, y, z)."""
    c, s = np.cos(angle), np.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    out = np.eye(3)
    out[i, i] = out[j, j] = c
    out[i, j], out[j, i] = -s, s
    return out


def harmonic_rotation(ell: int, rotation: np.ndarray) -> np.ndarray:
    """Matrix D of the real harmonics of degree l under a rotation R: Y(R u) = D Y(u), m ordered -l..l."""
    points = _fibonacci_sphere(4 * (2 * ell + 1) ** 2)
    return _harmonic_matrix(ell, points @ rotation.T) @ np.linalg.pinv(_harmonic_matrix(ell, points))


class LOSRotation:
    """Cell-by-cell rotation of the real components X_m (m = -l..l) to the frame of a local line of sight n.

    The rotation R = R_y(-theta) R_z(-phi) takes n = (theta, phi) to z, and the components in that frame are
    X' = D(R) X, D(R) = D_y(-theta) D_z(-phi). Each matrix element of D about a fixed axis is a trigonometric polynomial
    of degree l in the angle, so 2l + 1 samples give it exactly; the expansion is applied cell by cell.
    """

    def __init__(self, ell: int):
        self.ell = ell
        n = 2 * ell + 1
        angles = 2 * np.pi * np.arange(n) / n
        self.terms = {}
        for axis in (2, 1):
            samples = np.array([harmonic_rotation(ell, axis_rotation(axis, a)) for a in angles])
            c = np.fft.fft(samples, axis=0) / n
            self.terms[axis] = (c[0].real, [(p, 2 * c[p].real, -2 * c[p].imag) for p in range(1, ell + 1)])

    def _apply(self, axis, angle, x):
        c0, harmonics = self.terms[axis]
        out = c0 @ x
        for p, a, b in harmonics:
            out += np.cos(p * angle) * (a @ x) + np.sin(p * angle) * (b @ x)
        return out

    def __call__(self, components: np.ndarray, theta, phi) -> np.ndarray:
        """Rotated components, shape (2l + 1, ncells) as ``components``; theta, phi of shape (ncells,) or scalars."""
        if self.ell == 0:
            return components
        x = self._apply(2, -np.asarray(phi), components.astype(np.float64))
        return self._apply(1, -np.asarray(theta), x).astype(components.dtype)


def m_powers(components: np.ndarray) -> list[np.ndarray]:
    """sum of X_m^2 over m = +-|m|, for |m| = 0..l, of components ordered m = -l..l."""
    ell = (len(components) - 1) // 2
    squares = components * components
    return [squares[ell]] + [squares[ell - m] + squares[ell + m] for m in range(1, ell + 1)]


# ---------------------------------------------------------------------------------------------------------------------
# Painting


def paint_cic_weighted(positions: np.ndarray, weights: np.ndarray | None, nmesh: int, boxsize: float,
                       offset=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Cloud-in-cell weighted counts on a non-periodic nmesh^3 mesh with nodes at offset + i boxsize / nmesh.

    Every position must lie at least one cell inside the far edge of the box (no wrapping).
    """
    x = (np.asarray(positions, dtype=np.float64) - np.asarray(offset)) * (nmesh / boxsize)
    if x.min() < 0 or x.max() >= nmesh - 1:
        raise ValueError("positions outside the mesh: enlarge boxsize or move the offset")
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    i0 = np.floor(x).astype(np.int64)
    d = x - i0
    grid = np.zeros(nmesh**3)
    for cx in (0, 1):
        wx = d[:, 0] if cx else 1 - d[:, 0]
        for cy in (0, 1):
            wy = d[:, 1] if cy else 1 - d[:, 1]
            for cz in (0, 1):
                wz = d[:, 2] if cz else 1 - d[:, 2]
                index = ((i0[:, 0] + cx) * nmesh + i0[:, 1] + cy) * nmesh + i0[:, 2] + cz
                grid += np.bincount(index, weights=weights * wx * wy * wz, minlength=nmesh**3)
    return grid.reshape((nmesh,) * 3)


@dataclass
class SurveyMesh:
    """Overdensity of a survey catalogue, normalised by its randoms, in a padded non-periodic box.

    ``delta`` = (n_g - alpha n_r) / (alpha n_r) on the ``support`` (cells with a random density above ``threshold``
    times its mean over the cells with randoms) and 0 elsewhere; ``nbar`` [(h/Mpc)^3] is alpha n_r per cell volume.
    Positions are comoving Cartesian [Mpc/h] with the ``observer`` at the given point.
    """

    delta: np.ndarray
    nbar: np.ndarray
    support: np.ndarray
    boxsize: float
    offset: np.ndarray
    observer: np.ndarray
    alpha: float

    @property
    def nmesh(self) -> int:
        return self.delta.shape[0]

    @property
    def cellsize(self) -> float:
        return self.boxsize / self.nmesh

    @classmethod
    def from_catalogues(cls, data, randoms, nmesh: int, data_weights=None, random_weights=None, boxsize=None,
                        padding: float = 300.0, observer=(0.0, 0.0, 0.0), threshold: float = 0.1):
        """Paint data and randoms (positions (N, 3) [Mpc/h]) in a cube enclosing the randoms with ``padding`` [Mpc/h]
        on every side (to keep the wavelets of the footprint from wrapping around the periodic FFT)."""
        randoms = np.asarray(randoms, dtype=np.float64)
        low, high = randoms.min(axis=0), randoms.max(axis=0)
        size = float(np.max(high - low)) + 2 * padding
        boxsize = size if boxsize is None else float(boxsize)
        if boxsize < size:
            raise ValueError(f"boxsize {boxsize} smaller than the extent of the randoms plus padding, {size}")
        offset = 0.5 * (low + high) - 0.5 * boxsize
        ng = paint_cic_weighted(data, data_weights, nmesh, boxsize, offset)
        nr = paint_cic_weighted(randoms, random_weights, nmesh, boxsize, offset)
        alpha = ng.sum() / nr.sum()
        support = nr > threshold * nr[nr > 0].mean()
        delta = np.zeros_like(ng)
        delta[support] = ng[support] / (alpha * nr[support]) - 1.0
        nbar = alpha * nr / (boxsize / nmesh) ** 3
        return cls(delta.astype(np.float32), nbar, support, boxsize, offset, np.asarray(observer, dtype=float), alpha)

    def fkp_weights(self, p0: float = 1e4) -> np.ndarray:
        """FKP weights of a quadratic statistic of delta, (nbar P0 / (1 + nbar P0))^2, zero outside the support."""
        x = self.nbar * p0
        return np.where(self.support, (x / (1.0 + x)) ** 2, 0.0)

    def line_of_sight(self) -> np.ndarray:
        """Unit vectors from the observer to the support cells, shape (3, ncells), in the order of the support."""
        index = np.nonzero(self.support)
        vectors = np.stack(index).astype(np.float64) * self.cellsize + self.offset[:, None] - self.observer[:, None]
        return vectors / np.linalg.norm(vectors, axis=0)


# ---------------------------------------------------------------------------------------------------------------------
# Measurement


def distance_to_edge(support: np.ndarray) -> np.ndarray:
    """Distance [cells] of every support cell to the complement of the support (infinite for a full support)."""
    if support.all():
        return np.full(support.shape, np.inf)
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(support)


def erode(support: np.ndarray, radius: float) -> np.ndarray:
    """Cells of ``support`` farther than ``radius`` cells from its complement (a full support is not eroded)."""
    return support if radius <= 0 else distance_to_edge(support) > radius


class _Moments:
    """Weighted moments over the support cells, with one eroded mask per scale."""

    def __init__(self, support, moment_support, weights, erosion, config: WSTConfig, qs):
        self.qs = qs
        self.index = np.flatnonzero(support)
        weights = np.ones(support.shape) if weights is None else np.asarray(weights, dtype=np.float64)
        self.weights = {}
        distance = distance_to_edge(moment_support).ravel()[self.index] if erosion > 0 else None
        inside = moment_support.ravel()[self.index]
        for j in range(config.J + 1):
            mask = inside if distance is None else distance > erosion * config.sigma0 * config.step**j
            w = np.where(mask, weights.ravel()[self.index], 0.0)
            if w.sum() <= 0:
                raise ValueError(f"no weight left at scale j = {j}: reduce the erosion")
            self.weights[j] = w / w.sum()

    def __call__(self, values, j):
        """sum_x w(x) |values|^q, values on the support cells."""
        w = self.weights[j]
        absolute = np.abs(values).astype(np.float64)
        return np.array([np.dot(w, absolute**q) for q in self.qs])


def _components(fk, lattice: Lattice, harmonics, sigma, ell, index):
    """Real components X_m = (field * psi_{sigma,l}^m)(x), m = -l..l, on the cells ``index``: (2l + 1, ncells)."""
    phase = (-1j) ** ell
    radial = lattice.radial(sigma, ell)
    out = np.empty((2 * ell + 1, len(index)), dtype=np.float32)
    for i, y in enumerate(harmonics):
        out[i] = _irfft(fk * (phase * radial * y), lattice.n).ravel()[index]
    return out


def measure_wst_survey(delta: np.ndarray, config: WSTConfig, support: np.ndarray | None = None,
                       weights: np.ndarray | None = None, qs=None, los=None, erosion: float = 0.0,
                       moment_support: np.ndarray | None = None, lattice: Lattice | None = None) -> dict:
    """WST coefficients of a density contrast mesh restricted to a survey support, for one or several exponents q.

    ``support``: boolean mesh of the cells inside the footprint (default: every cell, a periodic box). ``weights``:
    per-cell weights of the moments (default uniform; see ``SurveyMesh.fkp_weights``). ``erosion``: the moments of
    scale j average only over the support cells farther than ``erosion`` x sigma_j from its edge. ``los``: None
    (no |m| resolution), "z", a unit vector (3,), or unit vectors per support cell (3, ncells), in the order of
    ``np.nonzero(support)`` (``SurveyMesh.line_of_sight``). ``moment_support`` (default ``support``, and inside it):
    the cells the moments average over, before erosion; with a full ``support`` it gives the coefficients the
    unmasked field would have on the same cells, the edge-free reference of a masked measurement.

    Returns ``q``, ``S0``, ``S1`` and ``S2`` as ``measure_wst``, and with ``los`` also ``S1m`` and ``S2m``.
    """
    qs = np.atleast_1d(config.q if qs is None else qs).astype(float)
    nmesh = delta.shape[0]
    lattice = lattice or Lattice(nmesh)
    support = np.ones(delta.shape, dtype=bool) if support is None else np.asarray(support, dtype=bool)
    moment_support = support if moment_support is None else np.asarray(moment_support, dtype=bool) & support
    moments = _Moments(support, moment_support, weights, erosion, config, qs)
    index = moments.index
    if los is None or isinstance(los, str):
        if los not in (None, "z"):
            raise ValueError(f"unknown line of sight {los!r}")
        theta = phi = None
    else:
        los = np.asarray(los, dtype=np.float64)
        theta, phi = _angles(los.T if los.ndim == 2 else los)
    resolve = los is not None

    field = np.where(support, delta, 0.0).astype(np.float32)
    fk = _rfft(field)
    nq, nj, nl, nl2 = len(qs), config.J + 1, config.L + 1, config.lmax2 + 1
    out = {"q": qs, "S0": moments(field.ravel()[index], 0),
           "S1": np.full((nq, nj, nl), np.nan), "S2": np.full((nq, nj, nj, nl2), np.nan)}
    if resolve:
        out.update(S1m=np.full((nq, nj, nl, nl), np.nan), S2m=np.full((nq, nj, nj, nl2, nl2, nl2), np.nan))
    pairs = config.second_layer_pairs()
    for ell in range(nl):
        harmonics = lattice.harmonics(ell)
        rotation = LOSRotation(ell) if theta is not None else None

        def moduli(f, sigma, resolve_m=True):
            x = _components(f, lattice, harmonics, sigma, ell, index)
            total = np.sqrt((x * x).sum(axis=0))
            if not (resolve and resolve_m):
                return total, []
            if rotation is not None:
                x = rotation(x, theta, phi)
            return total, [np.sqrt(p) for p in m_powers(x)]

        for j1 in range(nj):
            u1, per_m = moduli(fk, config.sigma0 * config.step**j1)
            out["S1"][:, j1, ell] = moments(u1, j1)
            for m1, u in enumerate(per_m):
                out["S1m"][:, j1, ell, m1] = moments(u, j1)
            if ell > config.lmax2 or not any(pair[0] == j1 for pair in pairs):
                continue
            for m1, u in [(None, u1)] + list(enumerate(per_m)):
                second = np.zeros(nmesh**3, dtype=np.float32)
                second[index] = u if ell == 0 else u - u.mean(dtype=np.float64)
                fu = _rfft(second.reshape(delta.shape))
                for _, j2 in (pair for pair in pairs if pair[0] == j1):
                    u2, per_m2 = moduli(fu, config.sigma0 * config.step**j2, resolve_m=m1 is not None)
                    if m1 is None:
                        out["S2"][:, j1, j2, ell] = moments(u2, j2)
                    else:
                        for m2, v in enumerate(per_m2):
                            out["S2m"][:, j1, j2, ell, m1, m2] = moments(v, j2)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Window correction from Gaussian mocks on the survey geometry


class GaussianMocks:
    """Gaussian fields with a linear Kaiser spectrum, plus white noise of the local density, on a survey mesh.

    The signal is b delta + f (n.grad)^2 / grad^2 delta, with a line of sight n along z (``observer`` None) or from
    the ``observer`` to each cell (wide angle, through the six second derivatives of the potential). The noise
    variance per cell is 1 / (nbar V_cell) inside the ``support`` and its ``weights``-averaged value outside, so that
    the unmasked field is the reference of the box model at an effective noise level.

    ``pk``: linear power spectrum [(Mpc/h)^3] as a function of k [h/Mpc]. ``offset``: position of mesh node 0 [Mpc/h].
    ``window="cic"``: as for a CIC-painted catalogue, the signal is multiplied by the CIC window W(k) and the noise
    gets the aliased CIC shot-noise spectrum, prod_i (1 - 2/3 sin^2(k_i c / 2)) / nbar (not W^2 / nbar).
    ``discrete``: instead of Gaussian noise, Poisson counts per cell of mean nbar V_cell (1 + delta), with delta the
    Gaussian signal clipped at -1 (``"poisson"``) or its lognormal transform (``"lognormal"``, a non-Gaussian field whose
    spectrum is not exactly the input one).
    """

    def __init__(self, nmesh: int, cellsize: float, pk, bias: float, f: float, nbar: np.ndarray, support: np.ndarray,
                 weights: np.ndarray | None = None, observer=None, offset=(0.0, 0.0, 0.0), window: str | None = None,
                 discrete: str | None = None):
        self.n, self.cell, self.bias, self.f = nmesh, cellsize, bias, f
        if discrete not in (None, "poisson", "lognormal"):
            raise ValueError(f"unknown discrete {discrete!r}")
        self.discrete = discrete
        k = 2 * np.pi * np.fft.fftfreq(nmesh) / cellsize
        kz = 2 * np.pi * np.fft.rfftfreq(nmesh) / cellsize
        self.kvec = np.meshgrid(k, k, kz, indexing="ij", sparse=True)
        k2 = sum(c**2 for c in self.kvec)
        kmag = np.sqrt(k2)
        self.amplitude = np.where(k2 > 0, np.sqrt(pk(np.where(k2 > 0, kmag, 1.0)) / cellsize**3), 0.0)
        self.k2 = np.where(k2 > 0, k2, 1.0)
        sigma2 = np.where(nbar > 0, 1.0 / (np.where(nbar > 0, nbar, 1.0) * cellsize**3), 0.0)
        w = np.where(support, 1.0 if weights is None else weights, 0.0)
        effective = np.sum(w * sigma2) / np.sum(w)
        self.noise = np.sqrt(np.where(support, sigma2, effective)).astype(np.float32)
        self.mean_counts = (1.0 / (self.noise.astype(np.float64) ** 2)).astype(np.float32)  # nbar V_cell
        self.window = None
        if window == "cic":
            sx, sy, sz = (np.sinc(c * cellsize / (2 * np.pi)) ** 2 for c in self.kvec)
            self.window = (sx * sy * sz).astype(np.float32)
            ax, ay, az = (1 - 2 / 3 * np.sin(c * cellsize / 2) ** 2 for c in self.kvec)
            self.noise_window = np.sqrt(ax * ay * az).astype(np.float32)
        elif window is not None:
            raise ValueError(f"unknown window {window!r}")
        self.observer = None if observer is None else np.asarray(observer, dtype=float)
        if self.observer is not None:
            x = [np.arange(nmesh) * cellsize + offset[i] - self.observer[i] for i in range(3)]
            grid = np.meshgrid(*x, indexing="ij", sparse=True)
            norm = np.sqrt(sum(g**2 for g in grid))
            self.los = [(g / norm).astype(np.float32) for g in grid]

    def __call__(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        white = _rfft(rng.standard_normal((self.n,) * 3).astype(np.float32))
        dk = white * self.amplitude
        gaussian_window = self.window is not None and self.discrete is None
        if gaussian_window:
            dk = dk * self.window
        if self.observer is None:
            signal = _irfft(dk * (self.bias + self.f * self.kvec[2] ** 2 / self.k2), self.n)
        else:
            signal = self.bias * _irfft(dk, self.n)
            for i in range(3):
                for j in range(i, 3):
                    term = _irfft(dk * (self.kvec[i] * self.kvec[j] / self.k2), self.n)
                    signal += (1 if i == j else 2) * self.f * self.los[i] * self.los[j] * term
        if self.discrete is None:
            noise = self.noise * rng.standard_normal((self.n,) * 3).astype(np.float32)
            if gaussian_window:
                noise = _irfft(_rfft(noise) * self.noise_window, self.n)
            return (signal + noise).astype(np.float32)
        if self.discrete == "lognormal":
            density = self.mean_counts * np.exp(signal - 0.5 * np.var(signal, dtype=np.float64))
        else:
            density = self.mean_counts * np.maximum(1.0 + signal, 0.0)
        field = (rng.poisson(density) / self.mean_counts - 1.0).astype(np.float32)
        if self.window is not None:  # counts in cells smoothed by the assignment window
            field = _irfft(_rfft(field) * self.window, self.n).astype(np.float32)
        return field


def window_ratio(mocks: GaussianMocks, config: WSTConfig, support: np.ndarray, weights=None, qs=None, los=None,
                 erosion: float = 0.0, realizations: int = 4, seed: int = 0, keys=("S1m",),
                 lattice: Lattice | None = None) -> dict:
    """Window correction R = <S_masked> / <S_unmasked> of each coefficient, from paired Gaussian mocks.

    Each mock is measured masked by the ``support`` (as the data) and unmasked on the same cells with the same
    weights (the box model at the effective noise). The survey model is then R x the box model. Returns, per key,
    ``ratio`` and ``error`` (scatter of the per-mock ratios over sqrt(realizations)), and the raw ``masked`` and
    ``reference`` measurements of shape (realizations, ...).
    """
    kwargs = dict(qs=qs, los=los, erosion=erosion, lattice=lattice or Lattice(mocks.n), weights=weights)
    masked, reference = {key: [] for key in keys}, {key: [] for key in keys}
    for r in range(realizations):
        field = mocks(seed + r)
        a = measure_wst_survey(field, config, support=support, **kwargs)
        b = measure_wst_survey(field, config, moment_support=support, **kwargs)
        for key in keys:
            masked[key].append(a[key])
            reference[key].append(b[key])
    out = {}
    for key in keys:
        m, ref = np.array(masked[key]), np.array(reference[key])
        out[key] = dict(ratio=m.mean(axis=0) / ref.mean(axis=0),
                        error=(m / ref).std(axis=0, ddof=1) / np.sqrt(realizations) if realizations > 1 else None,
                        masked=m, reference=ref)
    return out
