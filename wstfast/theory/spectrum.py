"""The redshift-space power spectrum P_s(k, mu) of a tracer on a fine (k, mu) grid, and the linear maps from that grid to
observables: the |m| block variances of S1m (``S1mProjection``), the P(k) multipoles (``MultipoleProjection``), both
as sums over the modes of the measurement mesh, and their Alcock-Paczynski versions (``APPolynomial`` of an
``APProjection``). Real space is the limit f = 0.

    P_s(k, mu) = (b1 + f mu^2)^2 P_L + P_1loop,s - 2 (c0 + c2 mu^2 + c4 mu^4) k^2 P_L + V / N (stochastic terms),

resolved into the monomials of beta (``layout.GRID_TERMS``). The one-loop spectrum is the vendored dsc-model integral
(eft_loop.py; renormalized as in CLASS-PT, ``renormalize``), optionally BAO IR-resummed (ir.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import sph_harm_y

from ..config import WSTConfig
from .bias import monomial_position
from .eft_loop import LinearSpectrum, Quadrature, loop_integrals, radial_rule
from .layout import LOOP


def renormalize(parts, spectrum: LinearSpectrum, quadrature: Quadrature):
    """Absorb the b2 part of P13 into b1 and the k -> 0 limit of the b2^2 part of P22 into the shot noise."""
    out = {name: np.array(value) for name, value in parts.items()}
    for e in ([1, 0, 1, 0, 0], [0, 1, 1, 0, 0]):
        out["13"][..., monomial_position(e)] = 0.
    p, w = radial_rule(quadrature.qmin, quadrature.qmax, quadrature.nq)
    constant = 0.5 * float(np.dot(w, p**3 * spectrum(p) ** 2)) / (2 * np.pi**2)  # b2^2 / 2 int P^2
    out["22"][..., monomial_position([0, 0, 2, 0, 0])] -= constant
    out["loop"] = out["22"] + out["13"]
    return out


class SpectrumGrid:
    """Fine (k, mu) grid of the redshift-space tracer spectrum pieces (mu >= 0; P_s is even in mu), rows
    ``layout.GRID_TERMS`` (18, nk, nmu): the tree monomials TREE of (b1 + f mu^2)^2 P_L, the one-loop monomials LOOP
    (renormalized, ``renormalize``) and the counterterm shapes k^2 P_L mu^{0, 2, 4}.

    The loop is computed on a coarse grid (dsc-model's quadrature, or its cosmology-independent tables) and interpolated:
    cubic in ln k, and exactly (polynomial in mu^2 through the Gauss nodes) in mu.

    With ``ir=True`` (BAO infrared resummation, dsc-model / CLASS-PT conventions, ir.py), with the anisotropic
    damping D(k, mu) and the wiggle part P_w = P_L - P_nw of the DST split:
        tree [P_nw + e^-D (1 + D) P_w],   loop L[P_L] + (e^-D - 1) W,   counterterm shapes k^2 mu^{2n} [P_nw + e^-D P_w],
    with W = L22[P_L] - L22[P_nw] + L13[P_nw] P_w / P_nw (the loop is computed for P_L and for P_nw).
    """

    def __init__(self, kmax: float = 0.3, nk: int = 240, nmu: int = 16, nk_loop: int = 40, nmu_loop: int = 8,
                 quadrature: Quadrature = Quadrature(), ir: bool = False, tables: bool = True):
        self.ir = ir
        #: Evaluate the loop from cosmology-independent tables (``loop_tables.LoopTables``, built at the first call or read
        #: from the disk cache of ``loop_tables.cached_tables``, equal to ``loop_integrals`` to rounding) instead of the
        #: direct quadrature.
        self.tables = tables
        self._tables = None
        self.k = np.geomspace(1e-3, kmax, nk)
        x, _ = leggauss(2 * nmu)
        self.mu = x[nmu:]
        self.k_loop = np.geomspace(2e-3, kmax, nk_loop)
        x, _ = leggauss(2 * nmu_loop)
        self.mu_loop = x[nmu_loop:]
        self.quadrature = quadrature
        self.loop_to_grid = lagrange_matrix(self.mu_loop**2, self.mu**2)  # (nmu, nmu_loop)

    def _loop_parts(self, spectrum, f):
        """dsc-model's ``loop_integrals`` (unregulated) on the coarse loop grid, from the tables if ``tables``."""
        if not self.tables:
            return loop_integrals(self.k_loop, self.mu_loop, spectrum, f, None, self.quadrature, rsd=True,
                                  biased=True)
        if self._tables is None:
            from .loop_tables import cached_tables

            self._tables = cached_tables(self.k_loop, self.mu_loop, self.quadrature, biased=True)
        return self._tables(spectrum, f)

    def _to_grid(self, coarse):
        from scipy.interpolate import CubicSpline

        fine = CubicSpline(np.log(self.k_loop), coarse, axis=0)(np.log(np.clip(self.k, self.k_loop[0], None)))
        return fine @ self.loop_to_grid.T

    def _to_grid_components(self, coarse):
        return np.stack([self._to_grid(coarse[..., i]) for i in LOOP])

    def _loop(self, spectrum, f):
        return renormalize(self._loop_parts(spectrum, f), spectrum, self.quadrature)

    def __call__(self, klin, pklin, f, h=None, r_bao=None):
        """Rows (18, nk, nmu) for the linear spectrum tabulated on ``klin`` and growth rate ``f`` (0: real space);
        with ``ir``, also h and the sound horizon at drag ``r_bao`` [Mpc/h]."""
        spectrum = LinearSpectrum(np.asarray(klin, "f8"), np.asarray(pklin, "f8"))
        plin = spectrum(self.k)[:, None]
        mu2 = self.mu[None, :] ** 2 * np.ones_like(plin)
        k2 = self.k[:, None] ** 2
        shapes = np.stack([f**2 * mu2**2, 2 * f * mu2, np.ones_like(mu2)])  # TREE monomials
        parts = self._loop(spectrum, f)
        loop = self._to_grid_components(parts["loop"])
        if not self.ir:
            return np.concatenate([shapes * plin, loop, np.stack([k2 * plin * mu2**n for n in (0, 1, 2)])])
        from .ir import damping, split_linear

        nowiggle, sigma2, dsigma2, _ = split_linear(spectrum, h, r_bao / h)
        pnw = nowiggle(self.k)[:, None]
        parts_nw = self._loop(nowiggle, f)
        ratio = (spectrum(self.k_loop) / nowiggle(self.k_loop) - 1.0)[:, None, None]
        wiggle = self._to_grid_components(parts["22"] - parts_nw["22"] + parts_nw["13"] * ratio)
        d = np.asarray(damping(self.k, self.mu, f, sigma2, dsigma2))
        pw = plin - pnw
        damped = pnw + np.exp(-d) * pw
        return np.concatenate([shapes * (pnw + np.exp(-d) * (1 + d) * pw), loop + (np.exp(-d) - 1) * wiggle,
                               np.stack([k2 * damped * mu2**n for n in (0, 1, 2)])])


def lagrange_matrix(nodes, points):
    """Matrix of Lagrange interpolation weights from ``nodes`` to ``points`` (len(points), len(nodes))."""
    nodes, points = np.asarray(nodes, "f8"), np.asarray(points, "f8")
    out = np.ones((points.size, nodes.size))
    for j, xj in enumerate(nodes):
        for m, xm in enumerate(nodes):
            if m != j:
                out[:, j] *= (points - xm) / (xj - xm)
    return out


def lattice_modes(boxsize: float, nmesh: int, kmax: float, window: str | None = "cic"):
    """|k|, |mu| (line of sight z), CIC W^2 and aliased shot-noise factor, and multiplicity / V for the modes of the
    measurement mesh with 0 < |k| <= kmax (rfft half-space, as the estimators)."""
    from ..measure import Lattice

    lattice = Lattice(nmesh)
    scale = nmesh / boxsize
    kmag = lattice.kmag.ravel() * scale
    keep = (kmag > 0) & (kmag <= kmax)
    cell = boxsize / nmesh
    axes = np.meshgrid(2 * np.pi * np.fft.fftfreq(nmesh) / cell, 2 * np.pi * np.fft.fftfreq(nmesh) / cell,
                       2 * np.pi * np.fft.rfftfreq(nmesh) / cell, indexing="ij")
    half = [0.5 * a.ravel()[keep] * cell for a in axes]
    if window == "cic":
        window2 = np.prod([np.sinc(h / np.pi) ** 4 for h in half], axis=0)
        noise = np.prod([1.0 - 2.0 / 3.0 * np.sin(h) ** 2 for h in half], axis=0)
    else:
        window2 = noise = np.ones(keep.sum())
    weight = lattice.multiplicity.ravel()[keep] / boxsize**3
    return dict(k=kmag[keep].astype("f8"), mu=np.abs(lattice.mu.ravel()[keep]).astype("f8"), window2=window2,
                noise=noise, weight=weight, multiplicity=lattice.multiplicity.ravel()[keep])


def grid_interpolation(grid: SpectrumGrid, k, mu):
    """Sparse (nmodes, nk * nmu) matrix interpolating grid values to modes: linear in ln k, Lagrange in mu^2."""
    from scipy import sparse

    logk = np.log(np.clip(k, grid.k[0], grid.k[-1]))
    i = np.clip(np.searchsorted(np.log(grid.k), logk) - 1, 0, grid.k.size - 2)
    t = (logk - np.log(grid.k[i])) / (np.log(grid.k[i + 1]) - np.log(grid.k[i]))
    wmu = lagrange_matrix(grid.mu**2, mu**2)  # (nmodes, nmu)
    nmu = grid.mu.size
    cols = np.concatenate([(i[:, None] * nmu + np.arange(nmu)), ((i + 1)[:, None] * nmu + np.arange(nmu))], axis=1)
    vals = np.concatenate([(1 - t)[:, None] * wmu, t[:, None] * wmu], axis=1)
    indptr = np.arange(k.size + 1) * (2 * nmu)  # 2 nmu entries per row, built directly (no duplicate summing)
    return sparse.csr_matrix((vals.ravel(), cols.ravel().astype(np.int64), indptr), shape=(k.size, grid.k.size * nmu))


class S1mProjection:
    """Linear map from the P_s grid to the variance of every |m| block of the selected S1 coefficients.

    ``matrix`` (nblock, nk * nmu) and ``shot`` (nblock,) give lambda_M = matrix @ P_grid + shot * V / N, with the
    lattice mode sums of W^2 R^2 c_l^2 |Y_l|m||^2 per real component. ``blocks`` lists (coefficient index, |m|).
    """

    def __init__(self, config: WSTConfig, coefficients, grid: SpectrumGrid, boxsize: float = 1000.0, nmesh: int = 256,
                 window: str | None = "cic"):
        modes = lattice_modes(boxsize, nmesh, grid.k[-1], window)
        interp = grid_interpolation(grid, modes["k"], modes["mu"])
        self.k2mu2 = (grid.k[:, None] * grid.mu[None, :]) ** 2  # stochastic shapes on the grid
        self.k2 = grid.k[:, None] ** 2 * np.ones_like(grid.mu)[None, :]
        theta = np.arccos(np.clip(modes["mu"], -1, 1))
        rows, shot, self.blocks = [], [], []
        for index, c in enumerate(coefficients):
            sigma = config.sigma(c.j)
            radial = (sigma * modes["k"]) ** (2 * c.ell) * np.exp(-((sigma * modes["k"]) ** 2))
            for m in range(c.ell + 1):
                ang = 4 * np.pi / (2 * c.ell + 1) * np.abs(sph_harm_y(c.ell, m, theta, 0.0)) ** 2
                weight = modes["weight"] * radial * ang
                rows.append(interp.T @ (weight * modes["window2"]))
                shot.append(np.sum(weight * modes["noise"]))
                self.blocks.append((index, m))
        self.matrix, self.shot = np.array(rows), np.array(shot)


class MultipoleProjection:
    """Linear map from the P_s grid to the binned multipoles (as ``measure.PowerMultipoles`` estimates them):
    P_ell(b) = mean over the modes of bin b of (2 ell + 1) L_ell(mu) [W^2 P_s(k, mu) + (V/N) aliased noise]."""

    def __init__(self, edges, grid: SpectrumGrid, ells=(0, 2, 4), boxsize: float = 1000.0, nmesh: int = 256,
                 window: str | None = "cic"):
        from scipy.special import eval_legendre

        modes = lattice_modes(boxsize, nmesh, edges[-1], window)
        interp = grid_interpolation(grid, modes["k"], modes["mu"])
        self.k2mu2 = (grid.k[:, None] * grid.mu[None, :]) ** 2  # stochastic shapes on the grid
        self.k2 = grid.k[:, None] ** 2 * np.ones_like(grid.mu)[None, :]
        index = np.digitize(modes["k"], edges) - 1
        inside = (index >= 0) & (index < len(edges) - 1)
        nbins = len(edges) - 1
        nmodes = np.bincount(index[inside], weights=modes["multiplicity"][inside], minlength=nbins)
        self.k = np.bincount(index[inside], weights=modes["multiplicity"][inside] * modes["k"][inside],
                             minlength=nbins) / nmodes
        self.ells = tuple(ells)
        rows, shot = [], []
        for ell in self.ells:
            leg = (2 * ell + 1) * eval_legendre(ell, modes["mu"]) * modes["multiplicity"] * inside
            for b in range(nbins):
                sel = inside & (index == b)
                w = np.where(sel, leg, 0.0) / nmodes[b]
                rows.append(interp.T @ (w * modes["window2"]))
                shot.append(np.sum(w * modes["noise"]))
        self.matrix, self.shot = np.array(rows), np.array(shot)


def ap_matrix(weights, kmodes, mumodes, logk, nodes2, denominators, qpar, qperp):
    """(ntarget, nk * nmu) matrix of a linear observable of a (k, mu) grid with AP distortions: the observed modes
    (``kmodes``, ``mumodes``, with per-target ``weights`` (ntarget, nmodes)) are interpolated at their true-frame
    coordinates (linear in ln k on ``logk``, Lagrange in mu^2 on the ``nodes2``), and the result is divided by
    q_par q_perp^2. JAX; shared by ``APProjection`` and ``second_layer.S21mAPProjection``."""
    from .ap import observed_to_true

    k, mu, volume = observed_to_true(kmodes, mumodes, qpar, qperp)
    nk, nmu = logk.size, nodes2.size
    logkt = jnp.clip(jnp.log(k), logk[0], logk[-1])
    i = jnp.clip(jnp.searchsorted(logk, logkt) - 1, 0, nk - 2)
    t = (logkt - logk[i]) / (logk[i + 1] - logk[i])
    diff = mu[:, None] ** 2 - nodes2[None, :]
    ones = jnp.ones_like(diff[:, :1])
    left = jnp.cumprod(jnp.concatenate([ones, diff[:, :-1]], axis=1), axis=1)
    right = jnp.cumprod(jnp.concatenate([ones, diff[:, :0:-1]], axis=1), axis=1)[:, ::-1]
    lagrange = left * right / denominators  # (nmodes, nmu)
    out = jnp.zeros((nk, weights.shape[0], nmu))
    for index, factor in ((i, 1 - t), (i + 1, t)):
        contribution = weights.T[:, :, None] * (factor[:, None] * lagrange)[:, None, :]  # (nmodes, ntarget, nmu)
        out = out + jax.ops.segment_sum(contribution, index, num_segments=nk)
    return volume * jnp.transpose(out, (1, 0, 2)).reshape(weights.shape[0], nk * nmu)


class APNodes:
    """Chebyshev nodes in x = (ln q_par, ln q_perp) and the least-squares fit of a quantity known at them as
    sum_alpha x^alpha D_alpha (total degree <= ``order``), with D_0 the exact value at q = 1: the AP polynomial shared by
    every AP-dependent piece of the model (``APPolynomial``, and the observed-frame cumulants and first-layer modulus
    spectra of ``first_layer`` and ``second_layer``). ``nodes`` x ``nodes`` points of |x| <= ``xmax`` (by default the
    emulator boxes: |ln q_par| <= 0.085, |ln q_perp| <= 0.045 around the fiducial)."""

    def __init__(self, order: int = 3, xmax=(0.085, 0.045), nodes: int = 9):
        self.order = int(order)
        self.powers = [(a, b) for a in range(order + 1) for b in range(order + 1 - a)]
        cheb = np.cos(np.pi * (np.arange(nodes) + 0.5) / nodes)
        #: (ln q_par, ln q_perp) of the nodes
        self.points = [(a * xmax[0], b * xmax[1]) for a in cheb for b in cheb]
        design = np.array([[a**p * b**r for p, r in self.powers[1:]] for a, b in self.points])
        self._solve = np.linalg.pinv(design)  # (nalpha - 1, npoint): least squares of the remainder
        #: (nalpha, 1 + npoint): ``fit`` as a linear map of the values at q = 1 and at the nodes, stacked
        self.matrix = np.block([[np.ones((1, 1)), np.zeros((1, len(self.points)))],
                                [-self._solve.sum(axis=1, keepdims=True), self._solve]])

    def fit(self, center, values):
        """D_alpha (nalpha, ...) from the value at q = 1 (``center``, (...)) and at the nodes (``values``, (npoint, ...))."""
        center = np.asarray(center)
        rest = np.tensordot(self._solve, np.asarray(values) - center[None], axes=1)
        return np.concatenate([center[None], rest])

    def monomials(self, qpar, qperp):
        """(nalpha,) x^alpha at the AP ratios."""
        x, y = jnp.log(qpar), jnp.log(qperp)
        return jnp.stack([x**p * y**r for p, r in self.powers])

    def product(self, a, b):
        """Coefficients of the product of two AP polynomials of the same ``powers`` (leading axes), truncated at
        ``order``: c_gamma = sum_{alpha + beta = gamma} a_alpha b_beta. ``a`` (nalpha, ...), ``b`` (nalpha, ...)
        broadcast after the first axis."""
        out = None
        for i, (p1, r1) in enumerate(self.powers):
            for j, (p2, r2) in enumerate(self.powers):
                if p1 + p2 + r1 + r2 > self.order:
                    continue
                term = a[i] * b[j]
                if out is None:
                    out = np.zeros((len(self.powers),) + np.shape(term))
                out[self.powers.index((p1 + p2, r1 + r2))] += term
        return out


class APPolynomial:
    """An AP projection as a polynomial in x = (ln q_par, ln q_perp) with the volume factor exact:
    matrix(q) = (q_par q_perp^2)^-1 sum_alpha x^alpha D_alpha, total degree <= ``order``.

    D_0 is the exact projection at q = 1 and the other D_alpha are least-squares fits of
    ``projection.at(q) * q_par q_perp^2 - D_0`` on ``nodes`` x ``nodes`` Chebyshev points of |x| <= ``xmax`` (by default
    the emulator boxes: |ln q_par| <= 0.085, |ln q_perp| <= 0.045 around the fiducial). Without AP (q = 1) it is exact.
    On the halo baseline (S1m + S21m, V = 60) the model differs from the exact per-call projection by a whitened
    |delta| <= 0.03 sigma within the posterior (0.07 sigma at |ln q| ~ 0.05) at order 3, and 0.07 (0.12) at order 2;
    the remainder is at the level of the projection's own interpolation in ln k. The per-call cost is that of a few
    fixed matrix products instead of an interpolation at every mode. ``__call__`` is that of ``APProjection`` (grid
    rows, with the shot-noise column); ``terms`` gives the D_alpha, to contract into an emulator
    (``calculators.contract_s21m``).
    """

    def __init__(self, projection, order: int = 3, xmax=(0.085, 0.045), nodes: int = 9):
        self.nodes = APNodes(order, xmax, nodes)
        self.powers = self.nodes.powers
        # D_0 is the exact projection at q = 1 (no AP: no error); the other D_alpha fit the AP-dependent remainder
        center = np.asarray(projection.at(1.0, 1.0))
        mats = np.array([np.asarray(projection.at(np.exp(a), np.exp(b))) * np.exp(a + 2 * b) for a, b in self.nodes.points])
        self.terms = self.nodes.fit(center, mats)
        self._terms = jnp.asarray(self.terms)
        for name in ("noise", "field_index", "k2mu2", "k2", "entries"):  # passed through for the theories
            setattr(self, name, getattr(projection, name, None))

    def weights(self, qpar, qperp):
        """(nalpha,) x^alpha times the volume factor."""
        x, y = jnp.log(qpar), jnp.log(qperp)
        return jnp.stack([x**p * y**r for p, r in self.powers]) / (qpar * qperp**2)

    def at(self, qpar=1.0, qperp=1.0):
        return jnp.tensordot(self.weights(qpar, qperp), self._terms, axes=1)

    def __call__(self, rows, qpar=1.0, qperp=1.0, shotnoise=0.0):
        """(R, ntarget) for grid rows (R, nk, nmu), as ``APProjection.__call__``."""
        rows = jnp.asarray(rows)
        values = rows.reshape(rows.shape[0], -1) @ self.at(qpar, qperp).T
        return values + (jnp.asarray(shotnoise, dtype="f8")[..., None] * self.noise) / (qpar * qperp**2)


class APProjection:
    """Linear observables of the P_s grid with Alcock-Paczynski distortions applied at assembly (ap.py).

    Holds the observed-frame weights of each target on the measurement mesh modes (``weights`` (ntarget, nmodes), with
    the window, and ``noise`` (ntarget,) for the shot noise); ``__call__(rows, qpar, qperp)`` interpolates each grid row
    at the true-frame (k', mu') of every mode (linear in ln k, exact in mu^2, as ``grid_interpolation``), applies the
    weights and divides by q_par q_perp^2. At q_par = q_perp = 1 it equals the fixed-matrix projections.

    The true-frame coordinates of a mode depend only on (k_perp^2, k_par^2), which many lattice modes share (to the
    bit: k and |mu| are computed from the same integers), so the modes are grouped and their weights summed: the
    result is unchanged and the interpolation runs over ~10x fewer points. The evaluation is jitted.
    """

    def __init__(self, grid: SpectrumGrid, k, mu, weights, noise):
        self.logk = jnp.asarray(np.log(grid.k))
        self.k2mu2 = (grid.k[:, None] * grid.mu[None, :]) ** 2  # stochastic shapes on the grid
        self.k2 = grid.k[:, None] ** 2 * np.ones_like(grid.mu)[None, :]
        nodes = np.asarray(grid.mu, dtype="f8") ** 2
        self.nodes2 = jnp.asarray(nodes)
        self.denominators = jnp.asarray([np.prod([nodes[j] - nodes[m] for m in range(nodes.size) if m != j])
                                         for j in range(nodes.size)])
        coordinates, inverse = np.unique(np.column_stack([np.asarray(k, "f8"), np.asarray(mu, "f8")]), axis=0,
                                         return_inverse=True)
        grouped = np.zeros((np.shape(weights)[0], len(coordinates)))
        np.add.at(grouped.T, inverse.ravel(), np.asarray(weights, "f8").T)
        self.k, self.mu = jnp.asarray(coordinates[:, 0]), jnp.asarray(coordinates[:, 1])
        self.weights, self.noise = jnp.asarray(grouped), jnp.asarray(noise)
        self._evaluate = jax.jit(self._project)

    def _lagrange(self, x):
        """Lagrange weights (n, nnodes) at points x (n,) of the mu^2 nodes, as products of the other differences."""
        diff = x[:, None] - self.nodes2[None, :]
        ones = jnp.ones_like(diff[:, :1])
        left = jnp.cumprod(jnp.concatenate([ones, diff[:, :-1]], axis=1), axis=1)
        right = jnp.cumprod(jnp.concatenate([ones, diff[:, :0:-1]], axis=1), axis=1)[:, ::-1]
        return left * right / self.denominators

    def interpolate(self, rows, qpar, qperp):
        """Grid rows (R, nk, nmu) at the true-frame coordinates of the (grouped) modes: (R, nmodes), and the volume
        factor."""
        from .ap import observed_to_true

        k, mu, volume = observed_to_true(self.k, self.mu, qpar, qperp)
        logk = jnp.clip(jnp.log(k), self.logk[0], self.logk[-1])
        i = jnp.clip(jnp.searchsorted(self.logk, logk) - 1, 0, self.logk.size - 2)
        t = (logk - self.logk[i]) / (self.logk[i + 1] - self.logk[i])
        lagrange = self._lagrange(mu**2)
        rows = jnp.asarray(rows)
        lower = jnp.sum(rows[:, i, :] * lagrange[None], axis=-1)
        upper = jnp.sum(rows[:, i + 1, :] * lagrange[None], axis=-1)
        return (1 - t) * lower + t * upper, volume

    def at(self, qpar=1.0, qperp=1.0):
        """(ntarget, nk * nmu) matrix at the AP ratios (the grid rows enter linearly; no shot noise)."""
        return ap_matrix(self.weights, self.k, self.mu, self.logk, self.nodes2, self.denominators, qpar, qperp)

    def _project(self, rows, qpar, qperp, shotnoise):
        values, volume = self.interpolate(rows, qpar, qperp)
        return volume * (values @ self.weights.T + jnp.asarray(shotnoise)[..., None] * self.noise)

    def __call__(self, rows, qpar=1.0, qperp=1.0, shotnoise=0.0):
        """(R, ntarget) for grid rows (R, nk, nmu); ``shotnoise`` is a scalar or one value per row (R,)."""
        return self._evaluate(jnp.asarray(rows), qpar, qperp, jnp.asarray(shotnoise, dtype="f8"))


def s1m_ap_projection(config: WSTConfig, coefficients, grid: SpectrumGrid, kmax: float = 0.27, boxsize: float = 1000.0,
                      nmesh: int = 256, window: str | None = "cic") -> APProjection:
    """APProjection of the S1m block variances (observed-frame filters; modes with k <= kmax)."""
    modes = lattice_modes(boxsize, nmesh, kmax, window)
    theta = np.arccos(np.clip(modes["mu"], -1, 1))
    weights, noise = [], []
    for c in coefficients:
        sigma = config.sigma(c.j)
        radial = (sigma * modes["k"]) ** (2 * c.ell) * np.exp(-((sigma * modes["k"]) ** 2))
        for m in range(c.ell + 1):
            w = modes["weight"] * radial * 4 * np.pi / (2 * c.ell + 1) * np.abs(sph_harm_y(c.ell, m, theta, 0.0)) ** 2
            weights.append(w * modes["window2"])
            noise.append(np.sum(w * modes["noise"]))
    return APProjection(grid, modes["k"], modes["mu"], np.array(weights), np.array(noise))


def multipole_ap_projection(edges, grid: SpectrumGrid, ells=(0, 2, 4), boxsize: float = 1000.0, nmesh: int = 256,
                            window: str | None = "cic") -> APProjection:
    """APProjection of the binned multipoles (as ``MultipoleProjection``)."""
    from scipy.special import eval_legendre

    modes = lattice_modes(boxsize, nmesh, edges[-1], window)
    index = np.digitize(modes["k"], edges) - 1
    inside = (index >= 0) & (index < len(edges) - 1)
    keep = {name: value[inside] for name, value in modes.items()}
    index = index[inside]
    nbins = len(edges) - 1
    nmodes = np.bincount(index, weights=keep["multiplicity"], minlength=nbins)
    weights, noise = [], []
    for ell in ells:
        leg = (2 * ell + 1) * eval_legendre(ell, keep["mu"]) * keep["multiplicity"]
        for b in range(nbins):
            w = np.where(index == b, leg, 0.0) / nmodes[b]
            weights.append(w * keep["window2"])
            noise.append(np.sum(w * keep["noise"]))
    return APProjection(grid, keep["k"], keep["mu"], np.array(weights), np.array(noise))
