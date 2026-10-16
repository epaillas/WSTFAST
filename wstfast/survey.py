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
from pathlib import Path

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

    def tables(self, theta, phi):
        """cos and sin of p (-phi) and p (-theta), p = 1..l, float32 (shape (l, ncells), or (l,) for scalar angles),
        to rotate many component sets of the same cells with ``rotate``."""
        p = np.arange(1, self.ell + 1).reshape((-1,) + (1,) * np.ndim(phi))
        a, b = -p * np.asarray(phi), -p * np.asarray(theta)
        return tuple(t.astype(np.float32) for t in (np.cos(a), np.sin(a), np.cos(b), np.sin(b)))

    def rotate(self, components: np.ndarray, tables) -> np.ndarray:
        """Components (2l + 1, ncells) in the local frame, with the trigonometric ``tables`` of their cells."""
        x = components.astype(np.float32)
        for axis, cos, sin in ((2, tables[0], tables[1]), (1, tables[2], tables[3])):
            c0, harmonics = self.terms[axis]
            out = c0.astype(np.float32) @ x
            for p, a, b in harmonics:
                out += cos[p - 1] * (a.astype(np.float32) @ x) + sin[p - 1] * (b.astype(np.float32) @ x)
            x = out
        return x

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
# Catalogue coordinates


def comoving_distance(z, fiducial=None) -> np.ndarray:
    """Comoving distance [Mpc/h] of redshifts ``z`` in a flat fiducial (``theory.ap.APFiducial``, default Quijote, the
    frame of the AP ratios)."""
    from scipy.integrate import cumulative_trapezoid

    from .theory.ap import QUIJOTE_FIDUCIAL

    fiducial = fiducial or QUIJOTE_FIDUCIAL
    z = np.asarray(z, dtype=np.float64)
    grid = np.linspace(0.0, max(float(z.max()), 0.0) * 1.01 + 1e-3, 4096)
    om, orad, h2 = fiducial.omega_m, fiducial.omega_r, fiducial.h**2
    e = np.sqrt((orad * (1 + grid) ** 4 + om * (1 + grid) ** 3) / h2 + 1.0 - (om + orad) / h2)
    return np.interp(z, grid, 2997.92458 * cumulative_trapezoid(1.0 / e, grid, initial=0.0))


def sky_to_cartesian(ra, dec, z, fiducial=None) -> np.ndarray:
    """Comoving Cartesian positions (N, 3) [Mpc/h] of RA, Dec [deg] and redshift, with the observer at the origin."""
    r = comoving_distance(z, fiducial)
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([r * np.cos(dec) * np.cos(ra), r * np.cos(dec) * np.sin(ra), r * np.sin(dec)], axis=1)


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
        self.inner = np.flatnonzero(inside)  # positions in ``index`` of the cells that can carry weight

    def __call__(self, values, j):
        """sum_x w(x) |values|^q, values on the support cells (``index``) or on its moment cells (``index[inner]``)."""
        w = self.weights[j] if len(values) == len(self.index) else self.weights[j][self.inner]
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
                       moment_support: np.ndarray | None = None, lattice: Lattice | None = None,
                       second_layer_bins: dict | None = None) -> dict:
    """WST coefficients of a density contrast mesh restricted to a survey support, for one or several exponents q.

    ``support``: boolean mesh of the cells inside the footprint (default: every cell, a periodic box). ``weights``:
    per-cell weights of the moments (default uniform; see ``SurveyMesh.fkp_weights``). ``erosion``: the moments of
    scale j average only over the support cells farther than ``erosion`` x sigma_j from its edge. ``los``: None
    (no |m| resolution), "z", a unit vector (3,), unit vectors per support cell (3, ncells) in the order of
    ``np.nonzero(support)`` (``SurveyMesh.line_of_sight``), or a unit vector per mesh cell (3, n, n, n)
    (``SurveyGeometry.line_of_sight(full=True)``, needed when the field support is not the footprint). ``moment_support`` (default ``support``, and inside it):
    the cells the moments average over, before erosion; with a full ``support`` it gives the coefficients the
    unmasked field would have on the same cells, the edge-free reference of a masked measurement.
    ``second_layer_bins`` (``second_layer_bins``): {(j1, l): label per support cell}; for l >= 1 the first-layer modulus
    then loses its mean in each bin (of first-layer coverage and nbar) before the second layer, instead of its mean over
    the support. Near the edges the mean of U follows the coverage, and a single mean would leave that profile in the
    second layer as a deterministic offset; in a periodic box (one bin) both are the same.

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
        los = np.asarray(los)
        if los.ndim == 4:
            los = los.reshape(3, -1)[:, index]
        theta, phi = _angles(los.T.astype(np.float64) if los.ndim == 2 else los.astype(np.float64))
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
        frames = {}  # trigonometric tables of the rotation of this l, built once for every scale

        def rotate(x, everywhere):
            if ell == 0:
                return x
            key = everywhere or np.ndim(theta) == 0
            if key not in frames:
                cells = slice(None) if key else moments.inner
                frames[key] = rotation.tables(theta if np.ndim(theta) == 0 else theta[cells],
                                              phi if np.ndim(phi) == 0 else phi[cells])
            return rotation.rotate(x, frames[key])

        def moduli(f, sigma, resolve_m=True, everywhere=False):
            """Moduli on the support cells (``everywhere``, to feed a second layer) or only on the moment cells."""
            cells = index if everywhere else index[moments.inner]
            x = _components(f, lattice, harmonics, sigma, ell, cells)
            total = np.sqrt((x * x).sum(axis=0))
            if not (resolve and resolve_m):
                return total, []
            if rotation is not None:
                x = rotate(x, everywhere)
            return total, [np.sqrt(p) for p in m_powers(x)]

        for j1 in range(nj):
            second_layer = ell <= config.lmax2 and any(pair[0] == j1 for pair in pairs)
            u1, per_m = moduli(fk, config.sigma0 * config.step**j1, everywhere=second_layer)
            out["S1"][:, j1, ell] = moments(u1, j1)
            for m1, u in enumerate(per_m):
                out["S1m"][:, j1, ell, m1] = moments(u, j1)
            if not second_layer:
                continue
            for m1, u in [(None, u1)] + list(enumerate(per_m)):
                second = np.zeros(nmesh**3, dtype=np.float32)
                second[index] = u if ell == 0 else subtract_mean(u, None if second_layer_bins is None
                                                                 else second_layer_bins[j1, ell])
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


def calibrate_window(fields, config: WSTConfig, support: np.ndarray, weights=None, qs=None, los=None,
                     erosion: float = 0.0, keys=("S1m",), lattice: Lattice | None = None) -> dict:
    """Window correction R = <S_masked> / <S_reference> of each coefficient, from paired survey-matched mocks.

    ``fields`` yields one density mesh per mock (``box_to_survey`` for catalogue mocks, ``GaussianMocks`` for Gaussian
    ones). Each is measured masked by the ``support``, as the data, and unmasked on the same cells with the same weights.
    A mock can also be a pair (masked input, reference input), for effects that act on the survey field only, such as
    the radial integral constraint (``RadialIntegralConstraint``). The unmasked field is the periodic mock at the
    effective noise, which is what the box model describes. The survey model is then R x the box model. Pairing
    cancels the cosmic variance of the mocks, so a few tens of them calibrate R well below the error of one survey.

    Returns, per key, ``ratio``, ``error`` (scatter of the per-mock ratios over sqrt(number of mocks)) and the raw
    ``masked`` and ``reference`` measurements, of shape (nmocks, ...).
    """
    kwargs = dict(qs=qs, los=los, erosion=erosion, weights=weights)
    masked, reference = {key: [] for key in keys}, {key: [] for key in keys}
    for field in fields:
        survey, field = field if isinstance(field, tuple) else (field, field)
        lattice = lattice or Lattice(field.shape[0])
        a = measure_wst_survey(survey, config, support=support, lattice=lattice, **kwargs)
        b = measure_wst_survey(field, config, moment_support=support, lattice=lattice, **kwargs)
        for key in keys:
            masked[key].append(a[key])
            reference[key].append(b[key])
    out = {}
    for key in keys:
        m, ref = np.array(masked[key]), np.array(reference[key])
        out[key] = dict(ratio=m.mean(axis=0) / ref.mean(axis=0),
                        error=(m / ref).std(axis=0, ddof=1) / np.sqrt(len(m)) if len(m) > 1 else None,
                        masked=m, reference=ref)
    return out


def _on_device(lattice) -> bool:
    """Whether ``lattice`` is a ``measure_torch.TorchLattice`` (the computation then runs on its device)."""
    return hasattr(lattice, "device")


def _squared_kernel(lattice, sigma: float, ell: int) -> np.ndarray:
    """sum_m psi_{sigma,l}^m(x)^2 in real space (n, n, n), on the lattice's device if it has one."""
    if _on_device(lattice):
        from .measure_torch import squared_kernel

        return squared_kernel(lattice, sigma, ell)
    radial = (-1j) ** ell * lattice.radial(sigma, ell)
    return sum(_irfft(radial * y, lattice.n) ** 2 for y in lattice.harmonics(ell))


def coverage(support: np.ndarray, config: WSTConfig, lattice, j: int, ell: int) -> np.ndarray:
    """Fraction of the power of the (sigma_j, l) wavelet inside the support, (|psi_l|^2 * M)(x) / sum |psi_l|^2 with
    |psi_l|^2 = sum_m |psi_l^m|^2, at every support cell (order of ``np.flatnonzero(support)``); float32 in [0, 1]."""
    n = support.shape[0]
    kernel2 = _squared_kernel(lattice, config.sigma0 * config.step**j, ell)
    kernel2 /= kernel2.sum(dtype=np.float64)
    f = _irfft(_rfft(kernel2) * _rfft(np.asarray(support, dtype=np.float32)), n).ravel()[np.flatnonzero(support)]
    return np.clip(f, 0.0, 1.0).astype(np.float32)


def second_layer_bins(support: np.ndarray, nbar: np.ndarray, config: WSTConfig, lattice=None, ncoverage: int = 10,
                      nnbar: int = 3, interior: float = 0.999) -> dict:
    """Bins of the support cells for the mean subtraction of the second layer (``measure_wst_survey``), {(j1, l):
    labels}, for every first-layer (j1, l) of the second layer: the coverage of the (sigma_j1, l) wavelet, in ``ncoverage``
    quantile bins of the edge cells (coverage < ``interior``) plus one interior bin, times ``nnbar`` quantile bins of
    nbar (the noise level of U). They depend on the geometry only."""
    lattice = lattice or Lattice(support.shape[0])
    index = np.flatnonzero(support)
    density = np.asarray(nbar).ravel()[index]
    edges = np.unique(np.quantile(density, np.linspace(0, 1, nnbar + 1)[1:-1]))
    nbin = np.digitize(density, edges)
    out = {}
    for j1 in sorted({j for j, _ in config.second_layer_pairs()}):
        for ell in range(1, config.lmax2 + 1):
            f = coverage(support, config, lattice, j1, ell)
            edge = f < interior
            cbin = np.full(len(f), ncoverage)
            if edge.any():
                qedges = np.unique(np.quantile(f[edge], np.linspace(0, 1, ncoverage + 1)[1:-1]))
                cbin[edge] = np.digitize(f[edge], qedges)
            _, out[j1, ell] = np.unique(cbin * (nnbar + 1) + nbin, return_inverse=True)
    return out


def subtract_mean(u: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """``u`` minus its mean (float64 accumulation), or minus its mean in each bin of ``labels``."""
    if labels is None:
        return u - u.mean(dtype=np.float64)
    means = np.bincount(labels, weights=u) / np.bincount(labels)
    return (u - means[labels]).astype(u.dtype)


class LambdaWindow:
    """Window of the first layer at the level of the component variances, which are linear in the spectrum.

    The survey field is the masked (and optionally constrained) field, so the variance of a rotated wavelet component
    at a support cell x is lambda_m(x) = T_m(x) lambda_m^box + lambda_N(x): the box signal variance times an edge
    transfer, plus the noise of the cell. For a white spectrum T is exactly the coverage
    f(x) = (|psi_l|^2 * M)(x) / sum |psi_l|^2, the fraction of the wavelet power inside the support M (computed by FFT,
    with |psi_l|^2 = sum_m |psi_l^m|^2). For a coloured spectrum T_m(x) = f(x) c_m(f), where the correction c_m is
    calibrated on paired mocks of the signal in bins of f: c_m = sum X'_m,masked^2 / sum f X'_m,reference^2 over the
    cells of a bin and the mocks. Pooling the cells of a bin keeps the noise of T far below that of a per-cell ratio
    (which the concave moments would turn into a bias), and c depends only weakly on the spectrum. The survey moments
    then follow from the box moment evaluated cell by cell (``s1m_gaussian``).

    ``los``: unit vectors per support cell (3, ncells) or per mesh cell (3, n, n, n), as ``measure_wst_survey``.
    ``lattice``: a ``Lattice``, or a ``measure_torch.TorchLattice`` to run on its device (as ``noise_variance_maps``
    and ``signal_variances``).
    """

    def __init__(self, config: WSTConfig, support: np.ndarray, los, lattice: Lattice | None = None, nbins: int = 40,
                 second_layer: bool = False, second_layer_bins: dict | None = None):
        self.config, self.support = config, np.asarray(support, dtype=bool)
        self.index = np.flatnonzero(self.support)
        self.lattice = lattice or Lattice(support.shape[0])
        los = np.asarray(los)
        self.second_layer = second_layer
        self.second_layer_bins = second_layer_bins
        if second_layer:
            if los.ndim != 4:
                raise ValueError("the second layer needs the line of sight of every mesh cell")
            # angles of every cell, for the first layer of the periodic reference
            self._angles_full = tuple(a.astype(np.float32) for a in _angles(los.reshape(3, -1).T.astype(np.float64)))
        if los.ndim == 4:
            los = los.reshape(3, -1)[:, self.index]
        theta, phi = _angles(los.T.astype(np.float64))
        if _on_device(self.lattice):
            from .measure_torch import TorchLOSRotation

            self.rotations = {ell: TorchLOSRotation(ell, self.lattice.device) for ell in range(1, config.L + 1)}
            self._index = __import__("torch").as_tensor(self.index, device=self.lattice.device)
        else:
            self.rotations = {ell: LOSRotation(ell) for ell in range(1, config.L + 1)}
        self.tables = {ell: r.tables(theta, phi) for ell, r in self.rotations.items()}
        self.blocks = [(j, ell) for j in range(config.J + 1) for ell in range(config.L + 1)]
        self.edges = np.linspace(0.0, 1.0 + 1e-6, nbins + 1)
        self.coverage, self.bins = {}, {}
        for ell in range(config.L + 1):
            for j in range(config.J + 1):
                f = coverage(self.support, config, self.lattice, j, ell)
                self.coverage[j, ell] = f
                self.bins[j, ell] = np.clip(np.digitize(f, self.edges) - 1, 0, nbins - 1)
        self.nbins = nbins
        self.masked = {b: np.zeros((2 * b[1] + 1, nbins)) for b in self.blocks}
        self.reference = {b: np.zeros((2 * b[1] + 1, nbins)) for b in self.blocks}
        self.pairs = config.second_layer_pairs() if second_layer else []
        self.blocks2 = [(j1, j2, ell, m1) for j1, j2 in self.pairs for ell in range(config.lmax2 + 1)
                        for m1 in range(ell + 1)]
        self.masked2 = {b: np.zeros((2 * b[2] + 1, nbins)) for b in self.blocks2}
        self.reference2 = {b: np.zeros((2 * b[2] + 1, nbins)) for b in self.blocks2}
        self.count = 0

    def squares(self, field: np.ndarray):
        """Squared rotated components X'_m(x)^2 (2l + 1, ncells) of every block (j, l) of ``field``."""
        if _on_device(self.lattice):
            import torch

            from .measure_torch import survey_components

            fk = torch.fft.rfftn(torch.as_tensor(np.asarray(field, dtype=np.float32), device=self.lattice.device))
            for ell in range(self.config.L + 1):
                for j in range(self.config.J + 1):
                    x = survey_components(fk, self.lattice, self.config.sigma0 * self.config.step**j, ell, self._index)
                    if ell > 0:
                        x = self.rotations[ell].rotate(x, self.tables[ell])
                    yield (j, ell), (x * x).cpu().numpy()
            return
        fk = _rfft(np.asarray(field, dtype=np.float32))
        for ell in range(self.config.L + 1):
            harmonics = self.lattice.harmonics(ell)
            for j in range(self.config.J + 1):
                x = _components(fk, self.lattice, harmonics, self.config.sigma0 * self.config.step**j, ell, self.index)
                if ell > 0:
                    x = self.rotations[ell].rotate(x, self.tables[ell])
                yield (j, ell), x * x

    def second_layer_squares(self, field: np.ndarray, survey: bool):
        """Squared rotated second-layer components X'_m2(x)^2 (2l + 1, ncells) on the support cells, for every
        (j1, j2, l, |m1|): as ``measure_wst_survey`` for the survey field (first-layer moduli on the support, minus
        their support mean, or their mean in each ``second_layer_bins`` bin, for l >= 1, zero elsewhere), and as the
        periodic box for the reference (moduli everywhere)."""
        if _on_device(self.lattice):
            yield from self._second_layer_squares_device(field, survey)
            return
        n = self.support.shape[0]
        fk = _rfft(np.asarray(np.where(self.support, field, 0.0) if survey else field, dtype=np.float32))
        cells = self.index if survey else np.arange(n**3)
        for ell in range(self.config.lmax2 + 1):
            harmonics = self.lattice.harmonics(ell)
            tables = None
            if ell > 0:
                tables = self.tables[ell] if survey else self.rotations[ell].tables(*self._angles_full)
            for j1 in sorted({j for j, _ in self.pairs}):
                x = _components(fk, self.lattice, harmonics, self.config.sigma0 * self.config.step**j1, ell, cells)
                if ell > 0:
                    x = self.rotations[ell].rotate(x, tables)
                for m1, power in enumerate(m_powers(x)):
                    u = np.sqrt(power)
                    second = np.zeros(n**3, dtype=np.float32)
                    bins = (self.second_layer_bins[j1, ell] if survey and ell > 0 and self.second_layer_bins is not None
                            else None)
                    second[cells] = u if ell == 0 else subtract_mean(u, bins)
                    fu = _rfft(second.reshape((n,) * 3))
                    for _, j2 in (pair for pair in self.pairs if pair[0] == j1):
                        x2 = _components(fu, self.lattice, harmonics, self.config.sigma0 * self.config.step**j2, ell,
                                         self.index)
                        if ell > 0:
                            x2 = self.rotations[ell].rotate(x2, self.tables[ell])
                        yield (j1, j2, ell, m1), x2 * x2
                del x
            del tables

    def _second_layer_squares_device(self, field, survey: bool):
        """``second_layer_squares`` on the device of a ``TorchLattice``."""
        import torch

        from .measure_torch import subtract_mean_torch, survey_components

        n, device = self.support.shape[0], self.lattice.device
        values = np.where(self.support, field, 0.0) if survey else field
        fk = torch.fft.rfftn(torch.as_tensor(np.asarray(values, dtype=np.float32), device=device))
        cells = self._index if survey else torch.arange(n**3, device=device)
        for ell in range(self.config.lmax2 + 1):
            tables = None
            if ell > 0:
                tables = self.tables[ell] if survey else self.rotations[ell].tables(*self._angles_full)
            for j1 in sorted({j for j, _ in self.pairs}):
                x = survey_components(fk, self.lattice, self.config.sigma0 * self.config.step**j1, ell, cells)
                if ell > 0:
                    x = self.rotations[ell].rotate(x, tables)
                for m1, power in enumerate(m_powers(x)):
                    u = torch.sqrt(power)
                    bins = (self.second_layer_bins[j1, ell] if survey and ell > 0 and self.second_layer_bins is not None
                            else None)
                    second = torch.zeros(n**3, dtype=torch.float32, device=device)
                    second[cells] = u if ell == 0 else subtract_mean_torch(u, bins)
                    fu = torch.fft.rfftn(second.reshape((n,) * 3))
                    for _, j2 in (pair for pair in self.pairs if pair[0] == j1):
                        x2 = survey_components(fu, self.lattice, self.config.sigma0 * self.config.step**j2, ell,
                                               self._index)
                        if ell > 0:
                            x2 = self.rotations[ell].rotate(x2, self.tables[ell])
                        yield (j1, j2, ell, m1), (x2 * x2).cpu().numpy()
                del x
            del tables

    def _binned(self, block, values):
        return np.array([np.bincount(self.bins[block], weights=v, minlength=self.nbins) for v in values])

    def add(self, masked: np.ndarray, reference: np.ndarray):
        """Accumulate one paired mock: the survey field (zeroed outside the support here) and its periodic parent."""
        for block, x2 in self.squares(np.where(self.support, masked, 0.0)):
            self.masked[block] += self._binned(block, x2)
        for block, x2 in self.squares(reference):
            self.reference[block] += self._binned(block, x2 * self.coverage[block])
        for survey, store in ((True, self.masked2), (False, self.reference2)):
            for block, x2 in (self.second_layer_squares(masked if survey else reference, survey)
                              if self.second_layer else []):
                coverage_block = (block[1], block[2])  # the sigma_j2 wavelet of degree l
                store[block] += self._binned(coverage_block, x2 if survey else x2 * self.coverage[coverage_block])
        self.count += 1

    def correction2(self) -> dict:
        """{(j1, j2, l, |m1|): c (2l + 1, nbins)} of the second layer, 1 in bins without cells."""
        return {b: np.where(self.reference2[b] > 0, self.masked2[b] / np.where(self.reference2[b] > 0,
                                                                               self.reference2[b], 1.0), 1.0)
                for b in self.blocks2}

    def s2m_ratio(self, weights: np.ndarray, q: float, correction2: dict | None = None) -> dict:
        """Survey / box S2m of every (j1, j2, l, |m1|) as an array over |m2|: the weighted mean over the support of
        T^{q/2}, with T = f c the transfer of the two components +-m2 (averaged), so that the box moment, with its
        non-Gaussianity, is rescaled cell by cell."""
        correction2 = self.correction2() if correction2 is None else correction2
        w = np.asarray(weights, dtype=np.float64) / np.sum(weights)
        out = {}
        for (j1, j2, ell, m1), c in correction2.items():
            t = self.coverage[j2, ell][None, :] * c[:, self.bins[j2, ell]]
            per_m = [t[ell]] + [0.5 * (t[ell - m] + t[ell + m]) for m in range(1, ell + 1)]
            out[j1, j2, ell, m1] = np.array([(np.clip(tm, 0.0, None) ** (q / 2)) @ w for tm in per_m])
        return out

    def correction(self) -> dict:
        """{(j, l): c (2l + 1, nbins)}, 1 in bins without cells."""
        return {b: np.where(self.reference[b] > 0, self.masked[b] / np.where(self.reference[b] > 0,
                                                                             self.reference[b], 1.0), 1.0)
                for b in self.blocks}

    def transfer(self, correction: dict | None = None) -> dict:
        """{(j, l): T (2l + 1, ncells)} on the support cells, in the order of ``np.flatnonzero(support)``;
        ``correction`` = None: from the mocks accumulated here, False: the coverage alone."""
        if correction is None:
            correction = self.correction()
        out = {}
        for b in self.blocks:
            c = np.ones((2 * b[1] + 1, self.nbins)) if correction is False else correction[b]
            out[b] = (self.coverage[b][None, :] * c[:, self.bins[b]]).astype(np.float32)
        return out


def noise_variance_maps(sigma2: np.ndarray, config: WSTConfig, support: np.ndarray, lattice: Lattice | None = None,
                        window: str | None = None):
    """Noise variance of a component at each support cell, (|psi_l|^2 * sigma_N^2 M)(x) averaged over m (white noise of
    variance ``sigma2`` per cell, zero outside the support): {(j, l): (ncells,)}.

    ``window="cic"``: the shot noise of a CIC-painted catalogue, whose nodes are correlated (2/3 and 1/6 per axis, the
    aliased spectrum prod_i (1 - 2/3 sin^2(k_i c / 2)) / nbar, as ``S1mProjection.shot`` and ``GaussianMocks(window="cic")``):
    |psi_l|^2 becomes sum_m psi_l^m (psi_l^m * C), with sigma2 slowly varying on the cell scale. White noise
    overestimates the CIC shot noise of the data by 10-20% in lambda at sigma = 25 Mpc/h on 10 Mpc/h cells."""
    index = np.flatnonzero(support)
    n = support.shape[0]
    if window == "cic":
        lattice = getattr(lattice, "numpy", None) or (lattice if isinstance(lattice, Lattice) else Lattice(n))
        k = 2 * np.pi * np.fft.fftfreq(n)
        a = 1 - 2 / 3 * np.sin(k / 2) ** 2
        aliased = (a[:, None, None] * a[None, :, None] * a[None, None, :n // 2 + 1]).astype(np.float32)
    elif window is not None:
        raise ValueError(f"unknown window {window!r}")
    lattice = lattice or Lattice(n)
    fs = _rfft(np.where(support, sigma2, 0.0).astype(np.float32))
    out = {}
    for ell in range(config.L + 1):
        for j in range(config.J + 1):
            sigma = config.sigma0 * config.step**j
            if window == "cic":
                radial = (-1j) ** ell * lattice.radial(sigma, ell)
                kernel2 = sum(_irfft(radial * y, n) * _irfft(radial * y * aliased, n) for y in lattice.harmonics(ell))
            else:
                kernel2 = _squared_kernel(lattice, sigma, ell)
            out[j, ell] = _irfft(_rfft(kernel2 / (2 * ell + 1)) * fs, n).ravel()[index]
    return out


def signal_variances(pcell: np.ndarray, config: WSTConfig, lattice: Lattice) -> dict:
    """Exact lattice variance of each component (m = -l..l, about z) of a periodic field with spectrum ``pcell`` per
    cell (rfft lattice, P / V_cell): {(j, l): (2l + 1,)}."""
    n3 = lattice.n**3
    out = {}
    if _on_device(lattice):
        import torch

        weight = torch.as_tensor(lattice.numpy.multiplicity * pcell / n3, dtype=torch.float32, device=lattice.device)
        for ell in range(config.L + 1):
            y2 = lattice.harmonics(ell) ** 2
            for j in range(config.J + 1):
                radial2 = lattice.radial(config.sigma0 * config.step**j, ell) ** 2
                out[j, ell] = (weight * radial2 * y2).to(torch.float64).sum(dim=(1, 2, 3)).cpu().numpy()
        return out
    for ell in range(config.L + 1):
        harmonics = lattice.harmonics(ell)
        for j in range(config.J + 1):
            radial2 = lattice.radial(config.sigma0 * config.step**j, ell) ** 2
            out[j, ell] = np.array([np.sum(lattice.multiplicity * pcell * radial2 * y**2) / n3 for y in harmonics])
    return out


def _chi_moment(lam_a, lam_b, q):
    """E (X_a^2 + X_b^2)^{q/2} for independent Gaussians of variances lam_a, lam_b (lam_b None: one component)."""
    from scipy.special import gammaln

    p = q / 2
    if lam_b is None:
        return (2 * lam_a) ** p * np.exp(gammaln((1 + q) / 2)) / np.sqrt(np.pi)
    phi = (np.arange(64) + 0.5) * np.pi / 128
    c2 = np.cos(phi)[:, None] ** 2
    return 2**p * np.exp(gammaln(1 + p)) * np.mean((np.asarray(lam_a) * c2 + np.asarray(lam_b) * (1 - c2)) ** p, axis=0)


def s1m_gaussian(lam_signal: dict, config: WSTConfig, q: float, weights: np.ndarray, transfer: dict | None = None,
                 lam_noise: dict | None = None) -> np.ndarray:
    """Gaussian S1m (J + 1, L + 1, L + 1) of a survey: the weighted mean over the support cells of the moment at
    lambda_m(x) = T_m(x) lambda_m^box + lambda_N(x). ``weights``: per support cell; ``transfer`` (``LambdaWindow``)
    and ``lam_noise`` (``noise_variance_maps``) default to 1 and 0."""
    w = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    out = np.full((config.J + 1, config.L + 1, config.L + 1), np.nan)
    for (j, ell), lam in lam_signal.items():
        lam = np.asarray(lam, dtype=np.float64)[:, None]
        if transfer is not None:
            lam = transfer[j, ell] * lam
        if lam_noise is not None:
            lam = lam + lam_noise[j, ell][None, :]
        lam = np.broadcast_to(lam, (2 * ell + 1, len(w)))
        out[j, ell, 0] = _chi_moment(lam[ell], None, q) @ w
        for m in range(1, ell + 1):
            out[j, ell, m] = _chi_moment(lam[ell - m], lam[ell + m], q) @ w
    return out


class CellWindow:
    """Exact first-layer window of a survey at sample cells, from the footprint alone.

    The components of the wavelet transform of the masked field at a cell x, rotated to its line of sight, are linear
    in the field, X'_m(x) = sum_y g_m(y) delta(y) with g_m = sum_m' D_mm'(R_x) psi_l^m'(x - .) W, so their covariance
    is linear in the spectrum,

        C_mm'(x) = (1/N) sum_k ghat_m(k) ghat_m'(k)^* P_cell(k, mu_x) + sum_y g_m(y) g_m'(y) sigma_N^2(y),

    with mu_x about the line of sight at x (constant over the wavelet, as in the estimator). Each g is compact, within
    ``halfwidth`` sigma_j of x, so its transform is taken on a sub-box of ``pad`` times that diameter around x. Per
    block (j, l, |m|) this gives the 2 x 2 covariance of (X_m, X_-m): the Gaussian moment E (X_m^2 + X_-m^2)^{q/2}
    then follows exactly from its eigenvalues, cell by cell, with the correlation of the two components near an edge
    and the local shot noise. It is the window of the DESI power spectrum (from the randoms, linear in the spectrum,
    independent of cosmology and of the tracer) one level down, at the variances; the moment is taken after it.

    ``cells``: flat mesh indices of the sample cells (``sample_cells``); ``sigma2``: shot-noise variance per cell
    (default 1 / (nbar V_cell) on the support); outside the support the unmasked reference carries its FKP-weighted
    mean, as ``GaussianMocks``. ``covariances`` evaluates the kernels for given spectra.
    """

    def __init__(self, geometry: SurveyGeometry, config: WSTConfig, cells, halfwidth: float = 4.0, pad: float = 2.0,
                 sigma2: np.ndarray | None = None, workers: int = -1, periodic: bool = False):
        self.g, self.config, self.workers = geometry, config, workers
        self.cells = np.asarray(cells, dtype=np.int64)
        n = geometry.nmesh
        self.sizes = {j: int(np.ceil(halfwidth * pad * config.sigma0 * config.step**j)) * 2
                      for j in range(config.J + 1)}
        self.margin = max(self.sizes.values()) // 2 + 1
        support = geometry.support
        if sigma2 is None:
            sigma2 = np.where(support, 1.0 / np.where(geometry.nbar > 0, geometry.nbar, 1.0) / geometry.cellsize**3, 0.0)
        w = np.where(support, geometry.weights, 0.0)
        effective = float(np.sum(w * sigma2) / np.sum(w))
        m = self.margin
        # beyond the mesh the field is zero (a padded survey mesh) or wraps around (``periodic``, a test box)
        mode = dict(mode="wrap") if periodic else dict(mode="constant")
        self.mask = np.pad(support.astype(np.float32), m, **mode)
        self.noise_in = np.pad(np.where(support, sigma2, 0.0).astype(np.float32), m, **mode)
        self.noise_all = np.pad(np.where(support, sigma2, effective).astype(np.float32), m,
                                **(mode if periodic else dict(mode="constant", constant_values=effective)))
        ijk = np.stack(np.unravel_index(self.cells, (n, n, n)), axis=1)
        self.ijk = ijk
        vectors = ijk * geometry.cellsize + geometry.offset - geometry.observer
        self.los = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.theta, self.phi = _angles(self.los)
        self._kernels, self._modes, self._rotations = {}, {}, {}

    def _prepare(self, j: int):
        """Centred wavelets psi_l^m (m = -l..l) on the sub-lattice of scale j, and its modes (k [h/Mpc], unit vectors,
        multiplicity of the rfft half-space)."""
        if j in self._kernels:
            return
        ns = self.sizes[j]
        lattice = Lattice(ns)
        sigma = self.config.sigma0 * self.config.step**j
        kernels = {}
        for ell in range(self.config.L + 1):
            radial = (-1j) ** ell * lattice.radial(sigma, ell)
            psi = np.array([np.roll(_irfft(radial * y, ns), ns // 2, axis=(0, 1, 2)) for y in lattice.harmonics(ell)])
            kernels[ell] = psi.astype(np.float32)  # (2l + 1, ns, ns, ns), r = 0 at index ns // 2
        self._kernels[j] = kernels
        k = 2 * np.pi * np.fft.fftfreq(ns) / self.g.cellsize
        kz = 2 * np.pi * np.fft.rfftfreq(ns) / self.g.cellsize
        kx, ky, kz = np.meshgrid(k, k, kz, indexing="ij")
        kmag = np.sqrt(kx**2 + ky**2 + kz**2)
        unit = np.stack([kx, ky, kz], axis=-1) / np.where(kmag > 0, kmag, 1.0)[..., None]
        mult = np.full(kmag.shape, 2.0, dtype=np.float32)
        mult[..., 0] = 1.0
        if ns % 2 == 0:
            mult[..., -1] = 1.0
        self._modes[j] = dict(kmag=kmag.astype(np.float32), unit=unit.astype(np.float32), mult=mult)
        for ell in range(self.config.L + 1):
            if ell not in self._rotations:
                self._rotations[ell] = LOSRotation(ell)

    def _subbox(self, array, ic: int, j: int):
        ns, h = self.sizes[j], self.sizes[j] // 2
        start = self.ijk[ic] + self.margin - h
        return array[start[0]:start[0] + ns, start[1]:start[1] + ns, start[2]:start[2] + ns]

    def covariances(self, spectra, ells=None, grid=None, processes: int | None = None):
        """``_covariances`` over all sample cells, split over ``processes`` forked workers (the arrays are shared)."""
        if not processes or processes <= 1 or len(self.cells) <= 1:
            return self._covariances(spectra, ells, grid, np.arange(len(self.cells)))
        import multiprocessing as mp

        global _CELLWINDOW_JOB
        _CELLWINDOW_JOB = (self, list(spectra) if spectra is not None else None, ells, grid)
        chunks = np.array_split(np.arange(len(self.cells)), min(processes * 4, len(self.cells)))
        workers = self.workers
        self.workers = 1
        try:
            with mp.get_context("fork").Pool(processes) as pool:
                parts = pool.map(_cellwindow_chunk, chunks)
        finally:
            self.workers = workers
            _CELLWINDOW_JOB = None
        axis = {"masked": 1, "unmasked": 1} if grid is None else {"masked": 0, "unmasked": 0}
        return {k: np.concatenate([p[k] for p in parts], axis=axis.get(k, 0)) for k in parts[0]}

    def _covariances(self, spectra, ells, grid, which):
        """Covariances of the rotated components at every sample cell and block, for ``spectra`` (callables
        P(k, mu) [(Mpc/h)^3], mu about the local line of sight; or, with ``grid`` an ``RSDGrid``, None for the linear
        rows on the grid). Returns a dict of arrays indexed [spectrum, cell, j, l, |m|, 3] with the three entries
        (C_mm, C_-m-m, C_m-m) (for m = 0: (C_00, C_00, 0)): ``masked`` and ``unmasked`` (signal, per spectrum) and
        ``noise_masked``, ``noise_unmasked`` (per unit shot noise, indexed [cell, j, l, |m|, 3]); with ``grid`` the
        signal entries are rows [cell, j, l, |m|, 3, nk * nmu].

        The sub-lattice sum (1/N) sum_k ghat_a ghat_b^* P misses the fine structure of P within a k-cell, which only
        matters at the origin, where P has a |k| cusp (equivalently: the sum sees the periodic images of xi at the
        sub-box size, whose shells converge slowly). The error is ghat_a(0) ghat_b(0) E, with E the same lattice error
        for a Gaussian envelope exp(-k^2 R^2) whose width R matches the second moment of the kernel pair (``_moments``):
        E = (1/N_full) sum_full S P - (1/N_sub) sum_sub S P, with the full sum of the mesh (orientation-independent)
        and the sub sum about the cell's line of sight. Callable spectra only (the grid rows do not include it yet)."""
        from scipy import fft as sfft

        cfg = self.config
        ells = range(cfg.L + 1) if ells is None else list(ells)
        which = np.asarray(which)
        nc, nj, nl = len(which), cfg.J + 1, cfg.L + 1
        if grid is not None:
            from .theory.rsd import grid_interpolation

            nrow = grid.k.size * grid.mu.size
            out = {k: np.zeros((nc, nj, nl, nl, 3, nrow)) for k in ("masked", "unmasked")}
            tables = None
        else:
            spectra = list(spectra)
            out = {k: np.zeros((len(spectra), nc, nj, nl, nl, 3)) for k in ("masked", "unmasked")}
            full = Lattice(self.g.nmesh)
            kfull, mufull = full.kmag / self.g.cellsize, np.abs(full.mu)
            pfull = [np.where(kfull > 0, p(np.where(kfull > 0, kfull, 1e-4), mufull), 0.0) / self.g.cellsize**3
                     * full.multiplicity / full.n**3 for p in spectra]
        out.update({k: np.zeros((nc, nj, nl, nl, 3)) for k in ("noise_masked", "noise_unmasked")})
        for j in range(nj):
            self._prepare(j)
            modes = self._modes[j]
            ns = self.sizes[j]
            cell3 = self.g.cellsize**3
            sigma = self.config.sigma0 * self.config.step**j * self.g.cellsize
            radii = sigma * np.geomspace(0.25, 4.0, 25)  # envelope widths of the cusp correction
            if grid is None:
                efull = np.array([[float(np.sum(np.exp(-kfull**2 * R**2) * pf)) for R in radii] for pf in pfull])
            offsets = (np.arange(ns) - ns // 2) * self.g.cellsize
            ox, oy, oz = np.meshgrid(offsets, offsets, offsets, indexing="ij", sparse=True)
            r2 = ox**2 + oy**2 + oz**2
            for io, ic in enumerate(which):
                W = self._subbox(self.mask, ic, j)
                s_in, s_all = self._subbox(self.noise_in, ic, j), self._subbox(self.noise_all, ic, j)
                mu = np.abs(np.tensordot(modes["unit"], self.los[ic].astype(np.float32), axes=([3], [0])))
                kmag = modes["kmag"]
                if grid is not None:
                    interp = grid_interpolation(grid, np.clip(kmag.ravel(), 1e-6, None), mu.ravel())
                    weights = [None]
                else:
                    weights, errors = [], []
                    for si, p in enumerate(spectra):
                        pcell = np.where(kmag > 0, p(np.where(kmag > 0, kmag, 1e-4), mu), 0.0) / cell3
                        weights.append((modes["mult"] * pcell / ns**3).astype(np.float32))
                        esub = np.array([float(np.sum(np.exp(-kmag**2 * R**2) * weights[-1], dtype=np.float64))
                                         for R in radii])
                        errors.append(efull[si] - esub)  # E(R): what the sub-lattice sum misses, per unit ghat(0)^2
                for ell in ells:
                    psi = self._kernels[j][ell]
                    rotation = self._rotations[ell]
                    psi_r = rotation(psi.reshape(2 * ell + 1, -1), self.theta[ic], self.phi[ic]).reshape(psi.shape)
                    g = psi_r * W
                    ghat = sfft.rfftn(g, axes=(1, 2, 3), workers=self.workers)
                    phat = sfft.rfftn(psi_r, axes=(1, 2, 3), workers=self.workers)
                    for m in range(ell + 1):
                        a, b = ell + m, ell - m
                        for name, kern, real, noise in (("masked", ghat, g, s_in), ("unmasked", phat, psi_r, s_all)):
                            prods = [np.abs(kern[a]) ** 2, np.abs(kern[b]) ** 2,
                                     (kern[a] * np.conj(kern[b])).real if m else np.zeros_like(kmag)]
                            if grid is not None:
                                base = (modes["mult"] / cell3 / ns**3).ravel()
                                for e, pr in enumerate(prods):
                                    out[name][io, j, ell, m, e] = interp.T @ (base * pr.ravel())
                            else:
                                # cusp correction per pair: dc x E(R), R from the pair's second moment (translation
                                # invariant: sum_y,y' g_a g_b |y - y'|^2 = dc_a M_b + M_a dc_b - 2 D_a . D_b)
                                pairs = [(a, a), (b, b), (a, b)] if m else [(a, a), (a, a), None]
                                scales = []
                                for pair in pairs:
                                    if pair is None:
                                        scales.append(None)
                                        continue
                                    ga, gb = real[pair[0]].astype(np.float64), real[pair[1]].astype(np.float64)
                                    dca, dcb = ga.sum(), gb.sum()
                                    dc = dca * dcb
                                    if dc <= 0:
                                        scales.append((dc, sigma))
                                        continue
                                    ma, mb = np.sum(ga * r2), np.sum(gb * r2)
                                    da = np.array([np.sum(ga * ox), np.sum(ga * oy), np.sum(ga * oz)])
                                    db = np.array([np.sum(gb * ox), np.sum(gb * oy), np.sum(gb * oz)])
                                    q = dca * mb + ma * dcb - 2 * da @ db
                                    scales.append((dc, float(np.sqrt(np.clip(q / (6 * dc), radii[0]**2, radii[-1]**2)))))
                                for si, w in enumerate(weights):
                                    for e, pr in enumerate(prods):
                                        value = float(np.sum(w * pr, dtype=np.float64))
                                        if scales[e] is not None:
                                            dc, R = scales[e]
                                            value += dc * np.interp(R, radii, errors[si])
                                        out[name][si, io, j, ell, m, e] = value
                            nm = "noise_" + name
                            out[nm][io, j, ell, m, 0] = float(np.sum(real[a] ** 2 * noise, dtype=np.float64))
                            out[nm][io, j, ell, m, 1] = float(np.sum(real[b] ** 2 * noise, dtype=np.float64))
                            out[nm][io, j, ell, m, 2] = float(np.sum(real[a] * real[b] * noise, dtype=np.float64)) if m else 0.0
        return out

    @staticmethod
    def moment(cov, q: float) -> np.ndarray:
        """E (X_m^2 + X_-m^2)^{q/2} of Gaussian components with covariance entries ``cov`` [..., 3] = (C_mm, C_-m-m,
        C_m-m), from the eigenvalues; for m = 0 (C_m-m = 0 and C_-m-m = C_mm) the one-component moment is used."""
        a, b, c = cov[..., 0], cov[..., 1], cov[..., 2]
        single = (c == 0) & (a == b)
        half, disc = 0.5 * (a + b), np.sqrt(np.maximum(0.25 * (a - b) ** 2 + c**2, 0.0))
        lam_p, lam_m = np.maximum(half + disc, 1e-30), np.maximum(half - disc, 1e-30)
        two = _chi_moment(lam_p.ravel(), lam_m.ravel(), q).reshape(a.shape)
        one = _chi_moment(np.maximum(a, 1e-30), None, q)
        return np.where(single, one, two)

    @staticmethod
    def s1m(cov, noise, weights, q: float, noise_scale: float = 1.0) -> np.ndarray:
        """Weighted mean over the sample cells of the moment at ``cov`` + noise_scale x ``noise`` ([cell, j, l, |m|, 3]),
        with the cells' ``weights``: S1m (J + 1, L + 1, L + 1) (NaN for |m| > l)."""
        total = cov + noise_scale * noise
        values = CellWindow.moment(total, q)  # (cell, j, l, m)
        w = np.asarray(weights, dtype=np.float64) / np.sum(weights)
        out = np.tensordot(w, values, axes=([0], [0]))
        nl = out.shape[-1]
        return np.where(np.arange(nl)[None, :, None] >= np.arange(nl)[None, None, :], out, np.nan)


_CELLWINDOW_JOB = None


def _cellwindow_chunk(which):
    window, spectra, ells, grid = _CELLWINDOW_JOB
    return window._covariances(spectra, ells, grid, which)


def cell_bins(geometry: SurveyGeometry, ndistance: int = 24, ndensity: int = 3) -> np.ndarray:
    """Bin label of every support cell (order of ``np.flatnonzero(support)``): quantile bins of the distance to the
    edge (``ndistance``) times quantile bins of the density (``ndensity``)."""
    index = np.flatnonzero(geometry.support)
    distance = distance_to_edge(geometry.support).ravel()[index]
    nbar = geometry.nbar.ravel()[index]
    db = np.digitize(distance, np.unique(np.quantile(distance, np.linspace(0, 1, ndistance + 1)[1:-1])))
    nb = np.digitize(nbar, np.unique(np.quantile(nbar, np.linspace(0, 1, ndensity + 1)[1:-1])))
    return np.unique(db * (ndensity + 1) + nb, return_inverse=True)[1].ravel()


def sample_cells(geometry: SurveyGeometry, per_bin: int = 64, ndistance: int = 24, ndensity: int = 3, seed: int = 0):
    """Stratified sample of support cells for ``CellWindow``: ``per_bin`` cells per bin of ``cell_bins``, drawn with
    probability proportional to the FKP weight within the bin. Returns (flat cell indices, weights, bin labels) with
    weights summing to the bins' FKP weight sums, so that a weighted mean over the sample estimates the weighted mean
    over the support."""
    rng = np.random.default_rng(seed)
    index = np.flatnonzero(geometry.support)
    w = geometry.weights.ravel()[index]
    label = cell_bins(geometry, ndistance, ndensity)
    cells, weights, labels = [], [], []
    for b in range(label.max() + 1):
        members = np.flatnonzero(label == b)
        pick = rng.choice(members, size=min(per_bin, len(members)), replace=False, p=w[members] / w[members].sum())
        cells.append(index[pick])
        weights.append(np.full(len(pick), w[members].sum() / len(pick)))
        labels.append(np.full(len(pick), b))
    return np.concatenate(cells), np.concatenate(weights), np.concatenate(labels)


class RICTransfer:
    """Effect of the radial integral constraint on the first-layer component variances, per bin of ``cell_bins``, from
    paired signal-only Gaussian mocks: T_m = sum |X'_m|^2 / sum |X_m|^2 over the cells of a bin and the mocks, with X'
    the masked components of the constrained field and X those of the unconstrained one (``LambdaWindow.squares``).
    The constraint is linear, so T is independent of the tracer; its dependence on the spectrum is second order in a
    shift of < 1%. ``apply`` scales the exact masked covariances of ``CellWindow`` (signal part) by it."""

    def __init__(self, window: "LambdaWindow", labels: np.ndarray, nbins: int | None = None):
        self.window, self.labels = window, np.asarray(labels)
        self.nbins = int(self.labels.max()) + 1 if nbins is None else nbins
        self.constrained = {b: np.zeros((2 * b[1] + 1, self.nbins)) for b in window.blocks}
        self.plain = {b: np.zeros((2 * b[1] + 1, self.nbins)) for b in window.blocks}
        self.count = 0

    def add(self, constrained: np.ndarray, field: np.ndarray):
        support = self.window.support
        for store, values in ((self.constrained, constrained), (self.plain, field)):
            for block, x2 in self.window.squares(np.where(support, values, 0.0)):
                for i in range(x2.shape[0]):
                    store[block][i] += np.bincount(self.labels, weights=x2[i], minlength=self.nbins)
        self.count += 1

    def transfer(self) -> dict:
        return {b: self.constrained[b] / np.where(self.plain[b] > 0, self.plain[b], 1.0) for b in self.window.blocks}

    @staticmethod
    def apply(transfer: dict, masked: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Masked signal covariances [..., cell, j, l, |m|, 3] of sample cells with bin ``labels``, scaled by the
        transfer of their components (C_mm by T_m, C_-m-m by T_-m, C_m-m by sqrt(T_m T_-m))."""
        out = np.array(masked, dtype=float)
        for (j, ell), t in transfer.items():
            for m in range(ell + 1):
                tp, tm = t[ell + m][labels], t[ell - m][labels]
                out[..., j, ell, m, 0] *= tp
                out[..., j, ell, m, 1] *= tm
                out[..., j, ell, m, 2] *= np.sqrt(tp * tm)
        return out

    def save(self, path):
        np.savez(path, labels=self.labels, count=self.count,
                 **{f"t_{j}_{ell}": v for (j, ell), v in self.transfer().items()})

    @staticmethod
    def load(path) -> dict:
        f = np.load(path)
        return {tuple(int(x) for x in k.split("_")[1:]): f[k] for k in f.files if k.startswith("t_")}


def window_ratio(mocks: GaussianMocks, config: WSTConfig, support: np.ndarray, realizations: int = 4, seed: int = 0,
                 **kwargs) -> dict:
    """``calibrate_window`` on Gaussian mocks ``mocks(seed)``, ..., ``mocks(seed + realizations - 1)``."""
    return calibrate_window((mocks(seed + r) for r in range(realizations)), config, support, **kwargs)


def s1m_window_factor(weight, transfer, noise, signal, box_noise, noise_scale, q: float):
    """Survey / box ratio of the Gaussian part of S1m for blocks (B,) with window histograms (B, nbin): the
    weighted mean over the histogram bins of [(T signal + noise_scale lambda_N) / (signal + box_noise)]^{q/2}.
    ``signal``, ``box_noise``: the box signal and noise variances of the blocks (the theory's, with AP);
    ``noise_scale``: the shot-noise amplitude (1 + alpha0) of the survey noise maps. NumPy or JAX."""
    import jax.numpy as jnp

    xp = jnp if any(isinstance(v, jnp.ndarray) for v in (signal, box_noise, noise_scale)) else np
    lam = transfer * signal[:, None] + noise_scale * noise
    ratio = lam / (signal + box_noise)[:, None]
    return xp.sum(weight * xp.maximum(ratio, 1e-30) ** (q / 2), axis=1) / xp.sum(weight, axis=1)


@dataclass
class SurveyWindow:
    """Window of one survey cap for the likelihood (``S1mTheory(window=)``, ``S21mTheory(window=)``).

    ``s1m``: per first-layer block, keyed (sigma [Mpc/h], l, |m|), the histogram of the support cells in (transfer,
    noise) bins: FKP ``weight``, mean ``transfer`` T (``LambdaWindow``, averaged over the +-m components) and mean
    ``noise`` lambda_N per unit shot-noise amplitude (``noise_variance_maps``), and the S-level correction ``delta``.
    The survey S1m is delta x ``s1m_window_factor`` x the box S1m. ``s21m``: per S21m entry, keyed (sigma1, sigma2, l,
    |m1|, |m2|), K = <S2m_survey / S2m_box> of paired mocks; the survey S21m is K / (delta W)_first layer x the box
    S21m. ``nbin`` bins per block (zero weight where a block has fewer).
    """

    s1m: dict
    s21m: dict
    q: float = 0.8
    veff: float | None = None  # FKP effective volume of the cap [(Gpc/h)^3], for the survey covariance

    @staticmethod
    def key(sigma: float) -> float:
        return round(float(sigma), 1)

    @classmethod
    def build(cls, window: "LambdaWindow", correction: dict, noise: dict, weights: np.ndarray, q: float = 0.8,
              ntransfer: int = 16, nnoise: int = 4) -> "SurveyWindow":
        """Histograms of the first-layer blocks from a ``LambdaWindow`` (its coverage), its ``correction``, the
        ``noise_variance_maps`` (per unit shot noise) and the FKP ``weights`` of the support cells; delta = 1 and no
        S21m entries until calibrated (``calibrate``)."""
        config = window.config
        weights = np.asarray(weights, dtype=np.float64)
        s1m = {}
        for (j, ell), transfer in window.transfer(correction).items():
            for m in range(ell + 1):
                t = transfer[ell] if m == 0 else 0.5 * (transfer[ell - m] + transfer[ell + m])
                lam_n = noise[j, ell]
                tb = np.digitize(t, np.unique(np.quantile(t, np.linspace(0, 1, ntransfer + 1)[1:-1])))
                nb = np.digitize(lam_n, np.unique(np.quantile(lam_n, np.linspace(0, 1, nnoise + 1)[1:-1])))
                _, label = np.unique(tb * (nnoise + 1) + nb, return_inverse=True)
                h = np.bincount(label, weights=weights)
                s1m[cls.key(config.sigma(j)), ell, m] = dict(
                    weight=h, weight2=np.bincount(label, weights=weights**2),
                    transfer=np.bincount(label, weights=weights * t) / h,
                    noise=np.bincount(label, weights=weights * lam_n) / h, delta=1.0)
        return cls(s1m=s1m, s21m={}, q=q)

    def factor(self, key, signal, box_noise, noise_scale=1.0):
        """``s1m_window_factor`` of one block (scalars)."""
        b = self.s1m[key]
        return float(s1m_window_factor(b["weight"][None], b["transfer"][None], b["noise"][None],
                                       np.atleast_1d(signal), np.atleast_1d(box_noise), noise_scale, self.q)[0])

    def veff_block(self, key, signal, box_noise, noise_scale=1.0) -> float:
        """Effective volume of a block [(Gpc/h)^3]: the cap's ``veff`` with the cells weighted by w s, s the block's
        local mean [(T signal + noise_scale lambda_N) / (signal + box_noise)]^{q/2} relative to the box (the weight of a
        cell in the fluctuation of ln S): veff (sum H s)^2 / sum H2 s^2 x sum H2 / (sum H)^2."""
        b = self.s1m[key]
        if self.veff is None or "weight2" not in b:
            raise ValueError("the window has no effective volume or no sum of squared weights (rebuild it)")
        local = ((b["transfer"] * signal + noise_scale * b["noise"]) / (signal + box_noise)) ** (self.q / 2)
        h, h2 = b["weight"], b["weight2"]
        return float(self.veff * (h @ local) ** 2 / (h2 @ local**2) * h2.sum() / h.sum() ** 2)

    def calibrate_s1m(self, key, survey, box, signal, box_noise):
        """delta of a block from paired mocks (``survey``, ``box`` (nmock,) measured on the support and unmasked on the
        same cells), at the box variances of the mocks' spectrum: delta = <survey / box> x F(T = 1) / F(T), the box
        reference having the survey's local noise. Returns the per-mock ratios (for the calibration covariance)."""
        b = self.s1m[key]
        unit = float(s1m_window_factor(b["weight"][None], np.ones_like(b["transfer"])[None], b["noise"][None],
                                       np.atleast_1d(signal), np.atleast_1d(box_noise), 1.0, self.q)[0])
        ratio = np.asarray(survey) / np.asarray(box) * unit / self.factor(key, signal, box_noise)
        b["delta"] = float(ratio.mean())
        return ratio

    def save(self, path):
        arrays = {"q": self.q}
        if self.veff is not None:
            arrays["veff"] = self.veff
        for (sigma, ell, m), b in self.s1m.items():
            for name, value in b.items():
                arrays[f"s1m_{sigma}_{ell}_{m}_{name}"] = value
        for key, value in self.s21m.items():
            arrays["s21m_" + "_".join(str(k) for k in key)] = value
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> "SurveyWindow":
        f = np.load(path)
        s1m, s21m = {}, {}
        for name in f.files:
            parts = name.split("_")
            if parts[0] == "s1m":
                key = (float(parts[1]), int(parts[2]), int(parts[3]))
                s1m.setdefault(key, {})["_".join(parts[4:])] = (float(f[name]) if f[name].ndim == 0 else f[name])
            elif parts[0] == "s21m":
                s21m[(float(parts[1]), float(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]))] = float(f[name])
        return cls(s1m=s1m, s21m=s21m, q=float(f["q"]), veff=float(f["veff"]) if "veff" in f.files else None)

    def s1m_arrays(self, keys):
        """(weight, transfer, noise) (B, nbin) and delta (B,) of the blocks ``keys``, zero-padded to one nbin."""
        nbin = max(len(self.s1m[k]["weight"]) for k in keys)
        pad = lambda v: np.pad(np.asarray(v, dtype=float), (0, nbin - len(v)))  # noqa: E731
        return (np.array([pad(self.s1m[k]["weight"]) for k in keys]),
                np.array([pad(self.s1m[k]["transfer"]) for k in keys]),
                np.array([pad(self.s1m[k]["noise"]) for k in keys]),
                np.array([self.s1m[k]["delta"] for k in keys]))


def block_signal(config: WSTConfig, j: int, ell: int, m: int, pfield, noise: float, boxsize: float = 1000.0,
                 nmesh: int = 128) -> float:
    """Signal variance per component of the |m| block (j, l) of a periodic box with field power ``pfield(k, mu)``
    (shot noise ``noise`` removed), by a half-lattice mode sum: the ``signal`` of ``SurveyWindow.veff_block``."""
    from .theory.covariance import HalfLattice, components

    lat = HalfLattice(nmesh, boxsize)
    keep = lat.positive & (lat.k <= np.pi * nmesh / boxsize)
    k, mu, mult = lat.k[keep], lat.mu[keep], lat.multiplicity[keep]
    w = lat.radial(config.sigma(j), ell)[keep] ** 2 * np.sum(lat.harmonics(ell)[components(ell, m)][:, keep] ** 2, axis=0)
    return float(((pfield(k, mu) - noise) * w * mult).sum() / boxsize**3 / (1 if m == 0 else 2))


def survey_covariance(box: np.ndarray, labels, config: WSTConfig, window: "SurveyWindow", pfield, noise: float,
                      boxsize: float = 1000.0) -> np.ndarray:
    """A box covariance of ln S1m / ln S21m entries (``labels``, of ``config``) on a survey cap: the box correlations,
    with the variance of every entry rescaled from the box volume to its effective volume on the cap
    (``SurveyWindow.veff_block`` of its S1m block, or of the sigma_j2 block of an S21m second layer)."""
    veff = []
    for label in labels:
        parts = label.split(":")[-1].split("_")
        if parts[0] == "S1m":
            j, ell, m = int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:])
        else:
            j, ell, m = int(parts[2][1:]), int(parts[3][1:]), int(parts[5][1:])
        key = (window.key(config.sigma(j)), ell, m)
        block = window.s1m[key]
        box_noise = float(block["weight"] @ block["noise"] / block["weight"].sum())
        veff.append(window.veff_block(key, block_signal(config, j, ell, m, pfield, noise, boxsize), box_noise))
    scale = np.sqrt((boxsize / 1000.0) ** 3 / np.array(veff))
    return box * np.outer(scale, scale)


def window_correction(survey: np.ndarray, model: np.ndarray, box: np.ndarray | None = None,
                      box_model: np.ndarray | None = None) -> dict:
    """S-level correction delta of the window model and its calibration covariance, from fiducial mocks.

    ``survey`` (nmock, ndata): the coefficients measured on the survey geometry; ``model`` (ndata,) or (nmock, ndata):
    the window model's prediction (``s1m_gaussian`` with the transfer, or R2 x the mock's own box S2m). Unpaired,
    delta = <survey> / model. Paired, with ``box`` (nmock, ndata) the same mocks measured unmasked on the same cells
    and ``box_model`` (ndata,) the model's box prediction, delta = <survey / box> x box_model / model: the same
    expectation without the mocks' cosmic variance. The window-corrected model of a survey is delta x model; the
    calibration error of delta, Cov(per-mock ratio) / nmock, enters the fit as ``covariance`` in data units (multiply
    by the data vector: ``covariance_of``).

    Returns delta (ndata,), ratio_covariance (fractional, of delta) and the number of mocks.
    """
    survey, model = np.asarray(survey, dtype=float), np.asarray(model, dtype=float)
    if box is None:
        ratio = survey / model
    else:
        ratio = survey / np.asarray(box, dtype=float) * (np.asarray(box_model, dtype=float) / model)
    delta = ratio.mean(axis=0)
    fractional = ratio / delta
    return dict(delta=delta, ratio_covariance=np.cov(fractional, rowvar=False, ddof=1) / len(ratio),
                nmock=len(ratio))


def covariance_of(correction: dict, data: np.ndarray, labels) -> dict:
    """The calibration covariance of ``window_correction`` in data units, for ``fit_rsd.py --extra-covariance``:
    dict(labels, covariance) with covariance = diag(data) ratio_covariance diag(data)."""
    data = np.asarray(data, dtype=float)
    return dict(labels=np.asarray(labels), covariance=correction["ratio_covariance"] * np.outer(data, data))


# ---------------------------------------------------------------------------------------------------------------------
# Survey geometry and catalogue mocks cut to it


@dataclass
class SurveyGeometry:
    """Footprint of a survey on a padded mesh: the support, the target density nbar(x) [(h/Mpc)^3] and the FKP weights.

    Mesh node (i, j, k) is at ``offset`` + (i, j, k) boxsize / nmesh [Mpc/h], comoving, with the observer at
    ``observer``. ``region`` (optional, int per cell): the region within which the randoms take their redshifts from the
    data, for ``RadialIntegralConstraint`` (for DESI LRGs, the photometric regions N and S, with S across both caps).
    """

    support: np.ndarray
    nbar: np.ndarray
    boxsize: float
    offset: np.ndarray
    observer: np.ndarray
    p0: float = 1e4
    region: np.ndarray | None = None

    @property
    def nmesh(self) -> int:
        return self.support.shape[0]

    @property
    def cellsize(self) -> float:
        return self.boxsize / self.nmesh

    def distance(self) -> np.ndarray:
        """Comoving distance [Mpc/h] of every mesh node to the observer."""
        x = [np.arange(self.nmesh) * self.cellsize + self.offset[i] - self.observer[i] for i in range(3)]
        gx, gy, gz = np.meshgrid(*x, indexing="ij", sparse=True)
        return np.sqrt(gx**2 + gy**2 + gz**2)

    @property
    def weights(self) -> np.ndarray:
        """FKP weights of a quadratic statistic, (nbar P0 / (1 + nbar P0))^2, zero outside the support."""
        x = self.nbar * self.p0
        return np.where(self.support, (x / (1.0 + x)) ** 2, 0.0)

    def line_of_sight(self, full: bool = False) -> np.ndarray:
        """Unit vectors from the observer to the support cells, (3, ncells) in the order of the support, or to every
        mesh cell, (3, n, n, n) float32, with ``full``."""
        if full:
            x = [np.arange(self.nmesh) * self.cellsize + self.offset[i] - self.observer[i] for i in range(3)]
            grid = np.stack(np.meshgrid(*x, indexing="ij")).astype(np.float32)
            return grid / np.linalg.norm(grid, axis=0)
        index = np.nonzero(self.support)
        vectors = np.stack(index).astype(np.float64) * self.cellsize + self.offset[:, None] - self.observer[:, None]
        return vectors / np.linalg.norm(vectors, axis=0)

    def sampling(self, nbar_tracer: float) -> np.ndarray:
        """Probability of keeping an object of a sample of density ``nbar_tracer``: nbar(x) / nbar_tracer on the support
        and its FKP-weighted mean elsewhere (the effective density of the reference field)."""
        inside = self.nbar / nbar_tracer
        if np.any(inside[self.support] > 1 + 1e-6):
            raise ValueError("target nbar above the density of the mock")
        w = self.weights
        return np.where(self.support, inside, np.sum(w * inside) / np.sum(w))

    def fkp_field(self, data, randoms, data_weights=None, random_weights=None, expected=None) -> np.ndarray:
        """FKP overdensity of a catalogue on this geometry's mesh: delta = (n_g - alpha n_r) / (nbar V_cell) on the
        support (0 elsewhere), CIC-painted, alpha = sum w_data / sum w_randoms. With randoms whose redshifts are drawn
        from the data (DESI clustering catalogues), the radial integral constraint comes with them. ``expected``
        (mesh, optional): the expected CIC count per cell to divide by instead of nbar V_cell (e.g. alpha n_r)."""
        ng = paint_cic_weighted(data, data_weights, self.nmesh, self.boxsize, self.offset)
        nr = paint_cic_weighted(randoms, random_weights, self.nmesh, self.boxsize, self.offset)
        alpha = ng.sum() / nr.sum()
        if expected is None:
            expected = self.nbar * self.cellsize**3
        expected = np.where(self.support & (expected > 0), expected, np.inf)
        return np.where(self.support, (ng - alpha * nr) / expected, 0.0).astype(np.float32)

    @property
    def veff(self) -> float:
        """FKP effective volume (sum w)^2 / sum w^2 x V_cell of the support [(Gpc/h)^3]."""
        w = self.weights[self.support]
        return float(np.sum(w) ** 2 / np.sum(w**2) * self.cellsize**3 / 1e9)

    @property
    def shotnoise(self) -> float:
        """Effective shot noise 1 / nbar of the support, averaged with the FKP weights [(Mpc/h)^3]."""
        w, nbar = self.weights[self.support], self.nbar[self.support]
        return float(np.sum(np.where(nbar > 0, w / np.where(nbar > 0, nbar, 1.0), 0.0)) / np.sum(w))

    @classmethod
    def cone(cls, nmesh: int, boxsize: float, observer, half_angle: float, rmin: float, rmax: float, nbar_of_r,
             p0: float = 1e4):
        """Cone about +z of ``half_angle`` degrees and comoving shell rmin..rmax around the ``observer`` (mesh offset 0),
        with radial density ``nbar_of_r`` [(h/Mpc)^3] as a function of distance [Mpc/h]."""
        geometry = cls(np.zeros((nmesh,) * 3, dtype=bool), np.zeros((nmesh,) * 3), float(boxsize), np.zeros(3),
                       np.asarray(observer, dtype=float), p0)
        r = geometry.distance()
        z = np.arange(nmesh)[None, None, :] * geometry.cellsize - geometry.observer[2]
        geometry.support = (r > rmin) & (r < rmax) & (z / r > np.cos(np.radians(half_angle)))
        geometry.nbar = np.where(geometry.support, nbar_of_r(r), 0.0)
        return geometry

    @classmethod
    def from_randoms(cls, randoms, nmesh: int, nbar_of_r, random_weights=None, padding: float = 300.0,
                     observer=(0.0, 0.0, 0.0), threshold: float = 0.1, p0: float = 1e4, regions=None):
        """Footprint of a random catalogue (positions (N, 3) [Mpc/h]): the cells whose random density exceeds
        ``threshold`` times its mean over the cells with randoms, with the radial density ``nbar_of_r`` (completeness
        is taken as uniform inside the footprint, as for weighted DESI clustering catalogues). ``regions`` (int per
        random): each support cell gets the region holding most of its randoms."""
        mesh = SurveyMesh.from_catalogues(randoms[:1], randoms, nmesh, random_weights=random_weights, padding=padding,
                                          observer=observer, threshold=threshold)
        geometry = cls(mesh.support, np.zeros(mesh.support.shape), mesh.boxsize, mesh.offset, mesh.observer, p0)
        geometry.nbar = np.where(geometry.support, nbar_of_r(geometry.distance()), 0.0)
        if regions is not None:
            regions = np.asarray(regions)
            labels = np.unique(regions)
            counts = [paint_cic_weighted(randoms[regions == label], None, nmesh, mesh.boxsize, mesh.offset)
                      for label in labels]
            geometry.region = np.where(mesh.support, labels[np.argmax(counts, axis=0)], -1)
        return geometry

    @classmethod
    def from_angular_mask(cls, cap: str, zrange, mask: AngularMask, nbar, cellsize: float = 10.0,
                          padding: float = 300.0, nmesh: int | None = None, p0: float = 1e4, threshold: float = 0.1):
        """Geometry of a DESI ``cap`` in the bin ``zrange``: a cube enclosing the footprint with ``padding`` [Mpc/h],
        the cells inside the bin whose direction has a completeness above ``threshold``, nbar(x) = completeness x
        nbar(r, region), and the photometric regions (``desi_region``)."""
        import healpy as hp

        rmin, rmax = comoving_distance(np.array(zrange))
        ra, dec = hp.pix2ang(mask.nside, mask.pixels, lonlat=True, nest=True)
        # the extremes of the shell are at its inner and outer radii
        edges = np.concatenate([sky_to_cartesian(ra, dec, np.full(len(ra), z)) for z in zrange])
        low, high = edges.min(axis=0) - padding, edges.max(axis=0) + padding
        if nmesh is None:
            nmesh = int(np.ceil(np.max(high - low) / cellsize / 8)) * 8
        boxsize = nmesh * cellsize
        if boxsize < np.max(high - low):
            raise ValueError(f"nmesh {nmesh} x cellsize {cellsize} smaller than the footprint,"
                             f" {np.max(high - low):.0f}")
        offset = 0.5 * (low + high) - 0.5 * boxsize
        g = cls(np.zeros((nmesh,) * 3, dtype=bool), np.zeros((nmesh,) * 3), boxsize, offset, np.zeros(3), p0)
        r = g.distance()
        shell = np.flatnonzero((r > rmin) & (r < rmax))
        xyz = np.stack(np.unravel_index(shell, r.shape)).T * cellsize + offset
        cell_ra = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360
        cell_dec = np.degrees(np.arcsin(xyz[:, 2] / r.ravel()[shell]))
        completeness = mask(cell_ra, cell_dec)
        inside = completeness > threshold
        cells = shell[inside]
        region = desi_region(cell_dec[inside], cap)
        g.support.ravel()[cells] = True
        g.nbar.ravel()[cells] = completeness[inside] * nbar(r.ravel()[cells], region)
        g.region = np.full((nmesh,) * 3, -1, dtype=np.int8)
        g.region.ravel()[cells] = region
        return g


def box_to_survey(positions, boxsize: float, geometry: SurveyGeometry, displacements=None, los: str = "local",
                  seed=0) -> np.ndarray:
    """Density contrast of a periodic mock box cut to a survey geometry, for ``calibrate_window``.

    The box (``positions`` (N, 3) in [0, boxsize) Mpc/h, real space) is tiled from the mesh offset to cover the mesh.
    With ``displacements`` (N, 3) [Mpc/h], the redshift-space shift of each object (velocity (1 + z) / H(z)), objects
    move by its projection on the line of sight: from the observer (``los="local"``) or z (``los="z"``). Objects are
    then kept with the probability ``geometry.sampling`` of their nearest cell, and the CIC counts are normalised by
    their expected value, delta = counts / (nbar V_cell (p * K)) - 1, everywhere (the support is applied by the
    measurement). K = [1/8, 3/4, 1/8] per axis is the CIC response of a node to objects uniform in the nearest cells:
    nbar(x) jumps from cell to cell with the angular completeness, and p alone would leave a fixed additive pattern
    (p * K) / p - 1 in delta (36% rms on the DR1 support, 3-19% too high S1m of halo mocks).
    """
    from scipy.ndimage import convolve1d

    positions = np.asarray(positions, dtype=np.float64)
    n, cell = geometry.nmesh, geometry.cellsize
    ntile = int(np.ceil(geometry.boxsize / boxsize))
    nbar = len(positions) / boxsize**3
    prob = geometry.sampling(nbar)
    rng = np.random.default_rng(seed)
    kept = []
    for shift in np.ndindex(ntile, ntile, ntile):
        x = positions + geometry.offset + np.array(shift) * boxsize
        if displacements is not None:
            if los == "z":
                x[:, 2] += displacements[:, 2]
            else:
                direction = x - geometry.observer
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                x += np.sum(displacements * direction, axis=1, keepdims=True) * direction
        grid = (x - geometry.offset) / cell
        inside = np.all((grid >= 0) & (grid < n - 1), axis=1)
        x, grid = x[inside], grid[inside]
        cells = np.round(grid).astype(int)
        keep = rng.random(len(x)) < prob[cells[:, 0], cells[:, 1], cells[:, 2]]
        kept.append(x[keep])
    counts = paint_cic_weighted(np.concatenate(kept), None, n, geometry.boxsize, geometry.offset)
    expected = prob
    for axis in range(3):
        expected = convolve1d(expected, [0.125, 0.75, 0.125], axis=axis, mode="nearest")
    return (counts / (expected * nbar * cell**3) - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# Radial integral constraint


class RadialIntegralConstraint:
    """Radial integral constraint of randoms whose redshifts are drawn from the data, applied to survey meshes.

    In DESI clustering catalogues each random takes the redshift of a data object of its region (for LRGs the
    photometric regions N and S, S spanning both caps), so the radial profile of the randoms is that of the data and
    the mean of delta vanishes in every thin shell of every region. For the FKP field (n_g - alpha n_r) / nbar, with
    nbar the expected density, this is

        delta'(x) = delta(x) - delta_s,   delta_s = sum_s nbar delta / sum_s nbar,

    with s the (region, distance shell) of x and the sums over the support cells of every geometry, so that a region
    split across meshes (the two caps) is constrained jointly. It also imposes the global integral constraint.
    ``width``: shell width [Mpc/h], by default the largest cell size (DESI uses ~1.3 Mpc/h for LRGs, on catalogues).
    """

    def __init__(self, *geometries: SurveyGeometry, width: float | None = None):
        width = max(g.cellsize for g in geometries) if width is None else float(width)
        self.index, self.nbar, labels = [], [], []
        for g in geometries:
            index = np.flatnonzero(g.support)
            region = np.zeros(len(index), dtype=np.int64) if g.region is None else g.region.ravel()[index]
            if np.any(region < 0):
                raise ValueError("support cells without a region")
            shell = np.floor(g.distance().ravel()[index] / width).astype(np.int64)
            self.index.append(index)
            self.nbar.append(g.nbar.ravel()[index])
            labels.append(np.stack([region, shell], axis=1))
        _, inverse = np.unique(np.concatenate(labels), axis=0, return_inverse=True)
        inverse = inverse.ravel()
        self.nkeys = int(inverse.max()) + 1
        self.keys = np.split(inverse, np.cumsum([len(i) for i in self.index])[:-1])
        self.norm = sum(np.bincount(k, weights=n, minlength=self.nkeys) for k, n in zip(self.keys, self.nbar))

    def __call__(self, *deltas: np.ndarray):
        """The constrained meshes, one per geometry, in the same order (a single mesh for a single geometry)."""
        if len(deltas) != len(self.index):
            raise ValueError(f"{len(deltas)} meshes for {len(self.index)} geometries")
        values = [d.ravel()[i].astype(np.float64) for d, i in zip(deltas, self.index)]
        total = sum(np.bincount(k, weights=n * v, minlength=self.nkeys)
                    for k, n, v in zip(self.keys, self.nbar, values))
        mean = total / np.where(self.norm > 0, self.norm, 1.0)
        out = []
        for d, i, k, v in zip(deltas, self.index, self.keys, values):
            d = d.copy()
            d.ravel()[i] = v - mean[k]
            out.append(d)
        return out[0] if len(out) == 1 else tuple(out)

# ---------------------------------------------------------------------------------------------------------------------
# DESI DR1 clustering catalogues
#
# Coordinates are comoving in the Quijote fiducial (``sky_to_cartesian``), the frame of the AP ratios. The geometry of a
# cap (``SurveyGeometry.from_angular_mask``) is the cells inside the redshift bin whose direction has a non-negligible
# angular completeness of the randoms, with nbar(x) = completeness x n_R(r), n_R the radial density of the data in the
# photometric region R of the cell. The completeness is purely angular, which holds for the complete (no fibre
# assignment) mocks.

DESI_ROOT = Path("data/desi/abacus_v4.2_complete")
DESI_CAPS = ("NGC", "SGC")
DESI_REGIONS = ("N", "S")
DESI_RANDOM_DENSITY = 2500.0  # DR1 randoms per deg^2 per random file
DESI_NORTH_DEC = 32.375  # N = NGC above this declination (BASS/MzLS); S = the rest of NGC and all of SGC (DECaLS)


def desi_catalogue_path(mock: int, cap: str, randoms: int | None = None, tracer: str = "LRG_complete",
                   root: Path = DESI_ROOT) -> Path:
    """Path of a clustering catalogue: the data, or random file number ``randoms``."""
    name = f"{tracer}_{cap}_clustering.dat.fits" if randoms is None else f"{tracer}_{cap}_{randoms}_clustering.ran.fits"
    return Path(root) / f"mock{mock}" / name


def read_desi_catalogue(path, zrange=None, columns=("RA", "DEC", "Z")) -> dict:
    """Columns of a clustering catalogue, optionally restricted to zrange[0] <= Z < zrange[1]."""
    import fitsio

    table = fitsio.read(str(path), columns=list(columns))
    keep = slice(None) if zrange is None else (table["Z"] >= zrange[0]) & (table["Z"] < zrange[1])
    return {c: np.asarray(table[c][keep], dtype=np.float64) for c in columns}


def desi_region(dec, cap: str) -> np.ndarray:
    """Region index (0 = N, 1 = S) of objects at declination ``dec`` [deg] in ``cap``."""
    dec = np.asarray(dec)
    return np.where((cap == "NGC") & (dec > DESI_NORTH_DEC), 0, 1)


@dataclass
class AngularMask:
    """HEALPix completeness of a random catalogue: randoms per pixel over their density in fully covered pixels, so that
    partly covered edge pixels and the small veto masks count in proportion."""

    nside: int
    completeness: np.ndarray  # nested ordering

    @classmethod
    def from_randoms(cls, ra, dec, density: float, nside: int = 128):
        """``density``: randoms per deg^2 before masking (DESI DR1: 2500 per random file)."""
        import healpy as hp

        counts = np.bincount(hp.ang2pix(nside, ra, dec, lonlat=True, nest=True), minlength=hp.nside2npix(nside))
        expected = density * hp.nside2pixarea(nside, degrees=True)
        return cls(nside, np.minimum(counts / expected, 1.0))

    def __call__(self, ra, dec) -> np.ndarray:
        """Completeness at the directions RA, Dec [deg]."""
        import healpy as hp

        return self.completeness[hp.ang2pix(self.nside, ra, dec, lonlat=True, nest=True)]

    @property
    def pixels(self) -> np.ndarray:
        return np.flatnonzero(self.completeness > 0)

    def area(self, region: int | None = None, cap: str | None = None) -> float:
        """Effective solid angle [sr] (sum of completeness), or that of ``region`` of ``cap``."""
        import healpy as hp

        weight = self.completeness[self.pixels]
        if region is not None:
            ra, dec = hp.pix2ang(self.nside, self.pixels, lonlat=True, nest=True)
            weight = weight * (desi_region(dec, cap) == region)
        return weight.sum() * hp.nside2pixarea(self.nside)


def desi_radial_density(data: list[dict], masks: dict, zrange, width: float = 10.0):
    """n_R(r) [(h/Mpc)^3] of each photometric region, from data catalogues ``data`` (dicts with RA, DEC, Z and
    ``cap``, possibly several mocks) and the angular ``masks`` of each cap: counts per shell / (mocks Omega_R r^2 dr).

    Returns a function (r, region) -> nbar."""
    rmin, rmax = comoving_distance(np.array(zrange))
    edges = np.arange(rmin, rmax + width, width)
    centres = 0.5 * (edges[1:] + edges[:-1])
    shell = (edges[1:] ** 3 - edges[:-1] ** 3) / 3
    nmocks = len({d["mock"] for d in data})
    profiles = []
    for region in range(len(DESI_REGIONS)):
        counts = sum(np.histogram(comoving_distance(d["Z"])[desi_region(d["DEC"], d["cap"]) == region],
                                  edges)[0] for d in data)
        solid = sum(masks[cap].area(region, cap) for cap in masks)
        profiles.append(counts / (nmocks * solid * shell) if solid > 0 else np.zeros_like(centres))

    def nbar(r, region):
        out = np.zeros(np.shape(r))
        for index, profile in enumerate(profiles):
            sel = region == index
            out[sel] = np.interp(r[sel], centres, profile)  # constant beyond the outer shell centres
        return out

    nbar.edges, nbar.profiles = edges, profiles
    return nbar
