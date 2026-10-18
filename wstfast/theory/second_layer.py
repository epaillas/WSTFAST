"""Second layer of the line-of-sight-resolved WST of a biased, discrete tracer (line of sight z).

The first-layer moduli U_|m| = |X_M| of each |m| block M of X_a = (delta_s * psi_{sigma1,l}^a)(x) (n_M = 1 component
for m = 0, 2 otherwise) are filtered again with psi_{sigma2,l} and resolved in |m2|:

    S21m(j1, j2, l, |m1|, |m2|) = S2m / S1m(j1, l, |m1|),   S2m = < |U_|m1| * psi_{sigma2,l}^{M2}|^q >.

Model. The block components are i.i.d. Gaussians of variance lambda_M, so the second-chaos projection
U ~ <U> + <U> / (2 n_M lambda_M) (|X_M|^2 - n_M lambda_M) holds block by block, with the pair kernel

    F_M(p, q) = (-1)^l R(p) R(q) c_l^2 w_M Re[Y_lM(p) Y_lM(q)^*],   R = (sigma1 k)^l e^{-sigma1^2 k^2 / 2},

and gives the power spectrum of U_|m| (normalised by <U>^2) as the tree-level response term plus the
Gaussian noise term, both anisotropic in K:

    P_UU(K) / <U>^2 = r_long(K)^2 P_L(K) + (1 + a) g(K),
    r_long(K) = int_p 2 [Z2(q, -K) Z1(q) P(q) + Z2(-K, p) Z1(p) P(p)] D(p) D(q) F_M(p, q) / (2 n_M lambda_M),
    g(K) = int_p P_s(p) P_s(q) F_M(p, q)^2 / (2 (n_M lambda_M)^2),   q = K - p,

with the tracer Z kernels of the Eulerian bias expansion (bias.py), P_s = Z1^2 P_L D^2 and
D(k) = exp(-(f k mu sigma_v)^2 / 2) (or, with ``fog``, the Fingers-of-God counterterm of every short leg instead).
For a discrete tracer of shot noise N the response numerator also holds the Poisson term,
delta_d(p) delta_d(q) > delta_g(p + q) / nbar (r = r0 + N r1), the noise the Poisson pieces of P_s(p) P_s(q), and
the normalization n_M lambda_M its shot noise; everything is resolved into the monomials of beta (``layout``):
lambda_2 = [R0 + N R1 + N^2 R2 + (1 + a)(G0 + N G1 + N^2 G2)] / (n0 + N n1)^2.
The damping is effective: sigma_v = SIGMA_V_FACTOR x the linear sigma_v. With the linear sigma_v the noise of
the line-of-sight-oriented fields (small |m1|) has too much power along the line of sight; sigma_v ~ 2 sigma_v,lin
removes that anisotropy, which the one-loop EFT spectrum does not (it also stands for the missing non-Gaussian
noise terms). One free amplitude a per first-layer field absorbs what is left (~10%).

The second layer is taken Gaussian: Y = U * psi_{sigma2,l}^{M2} has n_2 = 1 or 2 components of variance

    lambda_2 = sum_K (mult / V) c_l^2 |Y_l|m2|(mu_K)|^2 (sigma2 K)^{2l} e^{-sigma2^2 K^2} P_UU(K) / <U>^2

(a sum over the modes of the measurement mesh), and with the one-point relation <U>^q / S1m of the first layer,

    S21m = Gamma((n2 + q) / 2) / Gamma(n2 / 2) (2 lambda_2)^{q/2}
           x [Gamma((n1 + 1) / 2) / Gamma(n1 / 2)]^q / [Gamma((n1 + q) / 2) / Gamma(n1 / 2)] x (1 + E_1(1))^q / (1 + E_1(q)),

where E_1(q) is the Edgeworth factor of S1m(j1, l, |m1|) (first_layer.py).

Accuracy at the true cosmology (688 Quijote boxes, q = 0.8, l = 1-4, errors of a 60 (Gpc/h)^3 survey, full
covariance with the Hartlap factor). For sigma2 = 70.7 Mpc/h and sigma1 = 17.7, 25 Mpc/h (108 values) the
parameter-free model (a = 0) is within 3% of the data and with the 28 noise amplitudes chi^2 = 73 for 80 dof. The pair
sigma1 = 17.7, sigma2 = 50 Mpc/h is not described (chi^2 = 249 for its 54 values, a residual trend in |m2|), hence the
default sigma2 >= 70 Mpc/h of the selection.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln

from ..config import WSTConfig
from .bias import galaxy_kernel, linear_components, monomial_product, product_coefficients
from .first_layer import angular_weights, spherical_harmonic, spherical_harmonics
from .layout import FOG_KNL, S21_SIZES, s21m_assemble


#: Effective velocity dispersion of the modulus spectra, in units of the linear sigma_v.
SIGMA_V_FACTOR = 2.0


def s21m_entries(coefficients):
    """(coefficient index, |m1|, |m2|) of every S21m entry of the S21 ``coefficients`` (coefficient-major)."""
    return [(index, m1, m2) for index, c in enumerate(coefficients)
            for m1 in range(c.ell + 1) for m2 in range(c.ell + 1)]


def first_layer_fields(coefficients):
    """Sorted first-layer fields (j1, l, |m1|) of the S21 ``coefficients``."""
    return sorted({(c.j, c.ell, m1) for c in coefficients for m1 in range(c.ell + 1)})


class ModulusSpectra:
    """Response and noise of the power spectra of every first-layer field (sigma1, l, |m1|) on a (K, mu_K >= 0) grid,
    resolved into the monomials of beta: ``__call__`` returns ``grids`` (nfield, nk, nmu, 296) with the columns
    [R0 | R1 | R2 | G0 | G1 | G2] and ``norms`` (nfield, 16) with [n0 | n1] (``layout``).

    With ``fog``, the short modes are not velocity-damped but carry the Fingers-of-God counterterm, one factor
    (1 - c_v k_z^2 / k_NL^2) per leg (two per P_L of a short mode, one per leg of the response), and the orders 1 and 2
    in c_v of the blocks R0, R1, G0, G1 (grids) and n0 (norms) are appended: grids
    [... | R0' | R1' | G0' | G1' | R0'' | R1'' | G0'' | G1''], norms [n0 | n1 | n0' | n0'']. R0 = r^2 is expanded to
    second order in c_v consistently: R0' = 2 r rho1, R0'' = rho1^2 + 2 r rho2 with r = r0 - c_v rho1 + c_v^2 rho2.

    The p integral uses a log-spaced radial grid, Gauss-Legendre in mu_p and the midpoint rule in phi_p over
    [0, pi] (the integrands are even in phi_p for K in the x-z plane). The geometry is shared by all fields.
    """

    def __init__(self, fields, k, mu, npts: int = 56, nx: int = 24, nphi: int = 16, pmin: float = 2e-3,
                 cellsize: float = 0.0):
        self.fields = [(float(s), int(ell), int(m)) for s, ell, m in fields]
        #: Mass-assignment window of the first-layer legs, exp(-k^2 c^2 / 12) (the Gaussian approximation of CIC used by
        #: the S1m cumulants); 0 (no window) by default, as for the box fits.
        self.cellsize = float(cellsize)
        self.k, self.mu = np.asarray(k, "f8"), np.asarray(mu, "f8")
        pmax = 7.0 / min(s for s, _, _ in self.fields)
        p = np.geomspace(pmin, pmax, npts)
        x, wx = leggauss(nx)
        phi = (np.arange(nphi) + 0.5) * np.pi / nphi
        P, X, PH = np.meshgrid(p, x, phi, indexing="ij")
        sin = np.sqrt(1 - X**2)
        self.p = P.ravel()
        self.pvec = np.stack([P * sin * np.cos(PH), P * sin * np.sin(PH), P * X], axis=-1).reshape(-1, 3)
        self.mup = X.ravel()
        self.weights = ((p**3 * np.gradient(np.log(p)))[:, None, None] * wx[None, :, None]
                        * (2 * np.pi / nphi) / (2 * np.pi) ** 3 * np.ones_like(P)).ravel()
        self.harmonics = {(ell, m): spherical_harmonic(ell, m, self.pvec) for _, ell, m in self.fields}

    def window(self, k):
        """Mass-assignment window of a first-layer leg of wavenumber ``k`` (1 without ``cellsize``)."""
        return np.exp(-((k * self.cellsize) ** 2) / 12.0) if self.cellsize else 1.0

    def __call__(self, pk, f: float, sigma_v: float, fog: bool = False, stretches=None, fit=None):
        """``grids``, ``norms`` as in the class docstring. With ``stretches`` (a list of (q_perp, q_perp, q_par)), the
        first-layer filters and harmonics of every short leg are evaluated at its observed wavevector (the true one
        times the stretch) while the physics stays in the true frame (the Jacobians cancel in true-frame variables), and
        the outputs gain a leading axis over the stretches: grids (nstretch, nfield, nk, nmu, ncol), norms (nstretch,
        nfield, nnorm). Needs ``fog`` (no velocity damping). With ``fit`` (nout, nstretch), a linear map of the stretches
        (``spectrum.APNodes.matrix``: the AP polynomial), the leading axis is that of its output instead; the terms linear in
        the squared kernels are contracted before their products with the p integrands (fewer rows)."""
        if fog:
            sigma_v = 0.0
        if stretches is not None and not fog:
            raise ValueError("stretches (AP in the first layer) need fog=True")
        single = stretches is None
        stretches = [np.ones(3)] if single else [np.asarray(x, "f8") for x in stretches]
        damp = lambda k, mu: np.exp(-0.5 * (f * k * mu * sigma_v) ** 2)  # noqa: E731
        p, w = self.p, self.weights
        dp, pkp = damp(p, self.mup), pk(p)
        sp = (p * self.mup) ** 2 / FOG_KNL**2  # k_z^2 / k_NL^2 of the leg p
        z1p = linear_components(self.pvec, f)
        psp = (pkp * dp**2)[:, None] * product_coefficients(z1p, z1p)  # (np, 15)
        norms, kernels, harmonics = [], [], []  # per stretch
        for stretch in stretches:
            pobs = self.pvec * stretch
            po = np.sqrt(np.sum(pobs**2, axis=-1))
            harm = self.harmonics if single or np.all(stretch == 1) else spherical_harmonics(self.harmonics, pobs)
            radial = {s: np.exp(-0.5 * (s * po) ** 2) * self.window(po) for s, _, _ in self.fields}
            nrm, ker = [], []
            for s, ell, m in self.fields:
                c2w = 4 * np.pi / (2 * ell + 1) * (1.0 if m == 0 else 2.0)
                rp = (s * po) ** ell * radial[s]
                a = w * rp**2 * c2w * np.abs(harm[ell, m]) ** 2
                extra = [a @ (psp * (2 * sp)[:, None]), a @ (psp * (2 * sp**2)[:, None])] if fog else []
                nrm.append(np.concatenate([a @ psp, [np.sum(a)], *extra]))
                ker.append((-1) ** ell * c2w * rp)
            norms.append(nrm)
            kernels.append(ker)
            harmonics.append(harm)
        kernels = np.array(kernels)  # (nstretch, nfield, np)
        nk, nmu = self.k.size, self.mu.size
        ncol = sum(S21_SIZES[:6]) + (2 * sum(S21_SIZES[i] for i in (0, 1, 3, 4)) if fog else 0)
        grids = np.zeros((len(stretches) if fit is None else len(fit), len(self.fields), nk, nmu, ncol))
        contract = (lambda x: x) if fit is None else (lambda x: np.tensordot(fit, x, axes=1))  # noqa: E731
        for i, kk in enumerate(self.k):
            pkk = pk(np.array([kk]))[0]
            for a, mk in enumerate(self.mu):
                kvec = np.array([kk * np.sqrt(1 - mk**2), 0.0, kk * mk])
                qvec = kvec - self.pvec
                qn = np.sqrt(np.sum(qvec**2, axis=-1))
                muq = qvec[:, 2] / np.maximum(qn, 1e-30)
                dq, pkq = damp(qn, muq), pk(qn)
                sq = qvec[:, 2] ** 2 / FOG_KNL**2
                z1q = linear_components(qvec, f)
                minus = np.broadcast_to(-kvec, qvec.shape)
                long = 2 * ((pkq * dp * dq)[:, None] * product_coefficients(galaxy_kernel(np.stack([qvec, minus], axis=-2), f), z1q)
                            + (pkp * dp * dq)[:, None] * product_coefficients(galaxy_kernel(np.stack([minus, self.pvec], axis=-2), f), z1p))
                long = w[:, None] * long
                psq = (pkq * dq**2)[:, None] * product_coefficients(z1q, z1q)
                pair = w[:, None] * monomial_product(psp, 2, psq, 2)
                z1k = monomial_product(linear_components(kvec, f), 1, np.eye(5)[0], 1)  # beta_0 Z1(K), degree 2
                shot = w[:, None] * (psp + psq)
                if fog:
                    sl, s2 = sp + sq, 2 * (sp + sq)  # legs of the response; of the pair of spectra
                    long1, long2 = long * sl[:, None], long * (0.5 * sl**2)[:, None]
                    pair1, pair2 = pair * s2[:, None], pair * (0.5 * s2**2)[:, None]
                    shot1 = w[:, None] * (psp * (2 * sp)[:, None] + psq * (2 * sq)[:, None])
                    shot2 = w[:, None] * (psp * (2 * sp**2)[:, None] + psq * (2 * sq**2)[:, None])
                # kernel of every (stretch, field): its filter at the observed q leg, times the p-leg factor
                kernel = np.empty(kernels.shape)
                for n, stretch in enumerate(stretches):
                    qobs = qvec * stretch
                    qo = np.sqrt(np.sum(qobs**2, axis=-1))
                    cache = {key: np.real(harmonics[n][key] * np.conj(y))
                             for key, y in spherical_harmonics(harmonics[n], qobs).items()}
                    gauss = {s: np.exp(-0.5 * (s * qo) ** 2) * self.window(qo) for s, _, _ in self.fields}
                    radial = {}  # (s q)^l exp(-(s q)^2 / 2) per distinct (s, l), integer powers by products
                    for s, ell, _ in self.fields:
                        if (s, ell) not in radial:
                            radial[s, ell] = gauss[s].copy()
                            for _ in range(ell):
                                radial[s, ell] *= s * qo
                    for index, (s, ell, m) in enumerate(self.fields):
                        np.multiply(kernels[n, index], radial[s, ell], out=kernel[n, index])
                        kernel[n, index] *= cache[ell, m]
                # all (stretch, field) at once: matrix products over the p points, then the beta polynomials
                longs = [long, long1, long2] if fog else [long]
                rs = np.split(kernel @ np.concatenate(longs, axis=1) / 2, len(longs), axis=-1)
                r = rs[0]
                r1 = (kernel @ w / 2)[..., None] * z1k
                quads = [pair, shot, w[:, None]] + ([pair1, shot1, pair2, shot2] if fog else [])
                quad = np.split(contract(np.square(kernel)) @ np.concatenate(quads, axis=1) / 2,
                                np.cumsum([x.shape[1] for x in quads])[:-1], axis=-1)
                bilinear = [monomial_product(r, 2, r, 2) * pkk, 2 * monomial_product(r, 2, r1, 2) * pkk,
                            monomial_product(r1, 2, r1, 2) * pkk]
                if fog:
                    rho1, rho2 = rs[1], rs[2]
                    bilinear += [2 * monomial_product(r, 2, rho1, 2) * pkk, 2 * monomial_product(rho1, 2, r1, 2) * pkk,
                                 (monomial_product(rho1, 2, rho1, 2) + 2 * monomial_product(r, 2, rho2, 2)) * pkk,
                                 2 * monomial_product(rho2, 2, r1, 2) * pkk]
                bilinear = [contract(x) for x in bilinear]
                columns = bilinear[:3] + quad[:3]
                if fog:
                    columns += bilinear[3:5] + quad[3:5] + bilinear[5:7] + quad[5:7]
                grids[:, :, i, a] = np.concatenate(columns, axis=-1)
        if single:
            return grids[0], np.array(norms[0])
        return grids, contract(np.array(norms))


class S21mProjection:
    """Lattice sums from the field grids of ``ModulusSpectra`` to lambda_2 of every S21m entry.

    ``__call__(grids)`` maps (nfield, nk, nmu) grids to (nentry,) values; ``entries`` lists (coefficient index,
    |m1|, |m2|) and ``field_index`` the first-layer field of each entry.
    """

    def __init__(self, config: WSTConfig, coefficients, fields, k, mu, boxsize: float = 1000.0, nmesh: int = 256):
        from .spectrum import grid_interpolation, lattice_modes

        modes = lattice_modes(boxsize, nmesh, float(k[-1]), window=None)
        interp = grid_interpolation(SimpleNamespace(k=np.asarray(k), mu=np.asarray(mu)), modes["k"], modes["mu"])
        self.entries = s21m_entries(coefficients)
        self.field_index = np.array([fields.index((coefficients[i].j, coefficients[i].ell, m1))
                                     for i, m1, _ in self.entries], dtype=int)
        rows = []
        for index, m1, m2 in self.entries:
            c = coefficients[index]
            sigma2 = config.sigma(c.j2)
            radial = (sigma2 * modes["k"]) ** (2 * c.ell) * np.exp(-((sigma2 * modes["k"]) ** 2))
            weight = modes["weight"] * radial * angular_weights(c.ell, modes["mu"])[m2]
            rows.append(interp.T @ weight)
        self.matrix = np.array(rows)  # (nentry, nk * nmu)

    def __call__(self, grids):
        flat = np.asarray(grids).reshape(len(grids), -1)
        return np.sum(self.matrix * flat[self.field_index], axis=1)


class S21mAPProjection:
    """``S21mProjection`` with Alcock-Paczynski distortions, for the unprojected S21m basis (field grids on (K, mu_K)).

    The second-layer wavelets act on observed-frame modes; ``at(qpar, qperp)`` (nentry, nk * nmu) interpolates the
    field grids at the true-frame (K', mu') of every observed mode (linear in ln K, Lagrange in mu^2, as
    ``spectrum.APProjection``) and divides by q_par q_perp^2, so that lambda_2 = sum_g matrix[e, g] grid[field(e), g]. At
    q = 1 it equals ``S21mProjection.matrix`` up to the grouping of the modes. The first-layer filters inside the
    modulus spectra stay in the true frame (AP enters the S21m ratio through the second layer only; the first-layer
    AP mostly cancels in P_UU / <U>^2). Modes are grouped in ``nbins_k`` log-K x ``nbins_mu`` mu^2 bins (summed weights,
    mean coordinates), far finer than the modulus grid. Jitted.
    """

    def __init__(self, config: WSTConfig, coefficients, fields, k, mu, boxsize: float = 1000.0, nmesh: int = 256,
                 nbins_k: int = 512, nbins_mu: int = 64):
        import jax
        import jax.numpy as jnp

        from .spectrum import lattice_modes

        modes = lattice_modes(boxsize, nmesh, float(k[-1]), window=None)
        logk = np.log(modes["k"])
        kbin = np.clip(((logk - logk.min()) / (np.ptp(logk) + 1e-12) * nbins_k).astype(int), 0, nbins_k - 1)
        mbin = np.clip((modes["mu"] ** 2 * nbins_mu).astype(int), 0, nbins_mu - 1)
        groups, inverse = np.unique(kbin * nbins_mu + mbin, return_inverse=True)
        count = np.bincount(inverse, minlength=len(groups))
        self.kmodes = jnp.asarray(np.bincount(inverse, weights=modes["k"]) / count)
        self.mumodes = jnp.asarray(np.sqrt(np.bincount(inverse, weights=modes["mu"] ** 2) / count))
        self.entries = s21m_entries(coefficients)
        self.field_index = np.array([fields.index((coefficients[i].j, coefficients[i].ell, m1))
                                     for i, m1, _ in self.entries], dtype=int)
        weights = []
        for index, _, m2 in self.entries:
            c = coefficients[index]
            sigma2 = config.sigma(c.j2)
            radial = (sigma2 * modes["k"]) ** (2 * c.ell) * np.exp(-((sigma2 * modes["k"]) ** 2))
            weights.append(np.bincount(inverse, weights=modes["weight"] * radial * angular_weights(c.ell, modes["mu"])[m2],
                                       minlength=len(groups)))
        self.weights = jnp.asarray(np.array(weights))  # (nentry, ngroup)
        self.nk, self.nmu = len(k), len(mu)
        self.logk = jnp.asarray(np.log(np.asarray(k, "f8")))
        nodes = np.asarray(mu, "f8") ** 2
        self.nodes2 = jnp.asarray(nodes)
        self.denominators = jnp.asarray([np.prod([nodes[j] - nodes[m] for m in range(nodes.size) if m != j])
                                         for j in range(nodes.size)])
        self._matrix = jax.jit(self._build)

    def _build(self, qpar, qperp):
        from .spectrum import ap_matrix

        return ap_matrix(self.weights, self.kmodes, self.mumodes, self.logk, self.nodes2, self.denominators, qpar,
                         qperp)

    def at(self, qpar=1.0, qperp=1.0):
        """(nentry, nk * nmu) at the AP ratios."""
        return self._matrix(qpar, qperp)


class S21mBasis:
    """S21m terms of the S21 ``coefficients``: (nentry, sum S21_SIZES), columns [R0 | R1 | R2 | G0 | G1 | G2 | n0 | n1]
    (``layout``). With ``fog`` (Fingers-of-God counterterm instead of the velocity damping, see ``ModulusSpectra``),
    followed by the orders 1 and 2 in c_v of the blocks FOG_S21_BLOCKS, [R0' | R1' | G0' | G1' | n0' | R0'' | ... | n0'']."""

    def __init__(self, config, coefficients, boxsize: float = 1000.0, nmesh: int = 256,
                 kmax: float = 0.12, nk: int = 24, nmu: int = 6, sigma_v_factor: float = SIGMA_V_FACTOR,
                 fog: bool = False, ap: int = 0, ap_nodes: int = 5, **quadrature):
        self.config, self.coefficients, self.fog = config, list(coefficients), bool(fog)
        #: Order of the AP polynomial (0: none): the first-layer filters of the modulus spectra in the observed frame
        #: (``ModulusSpectra(stretches=...)``) and the second-layer projection (``spectrum.APPolynomial`` of an
        #: ``S21mAPProjection``), multiplied as polynomials; see ``__call__``.
        self.ap, self.ap_nodes = int(ap), int(ap_nodes)
        if self.ap and not self.fog:
            raise ValueError("ap needs fog=True")
        #: Velocity damping of the modulus spectra in units of the linear sigma_v. The matter value (2) is too strong
        #: for halos: 1 brings the measured response and noise of the halo moduli into agreement (and removes the
        #: S21m amplitude systematic; scripts/feasibility/modulus_response_noise.py --sigma-v-factor).
        self.sigma_v_factor = float(sigma_v_factor)
        self.fields = first_layer_fields(self.coefficients)
        x, _ = leggauss(2 * nmu)
        k, mu = np.geomspace(4e-3, kmax, nk), x[nmu:]
        self.spectra = ModulusSpectra([(config.sigma(j), ell, m) for j, ell, m in self.fields], k, mu,
                                      **quadrature)
        self.projection = S21mProjection(config, self.coefficients, self.fields, k, mu, boxsize=boxsize, nmesh=nmesh)
        if self.ap:
            from .spectrum import APNodes, APPolynomial

            self.ap_poly = APPolynomial(S21mAPProjection(config, self.coefficients, self.fields, k, mu, boxsize=boxsize,
                                                         nmesh=nmesh), order=self.ap)
            self.nodes = APNodes(self.ap, nodes=self.ap_nodes)

    def __call__(self, klin, pklin, f):
        """(nentry, sum S21_SIZES [+ FoG orders]); with ``ap``, (nentry, nalpha * ncol), alpha-major: the AP polynomial
        of the S21m terms, sum_alpha x^alpha T_alpha, where the lambda_2 columns still need the volume factor
        1 / (q_par q_perp^2) of the second-layer sum (the norms do not; ``s21m_lambda_mask``). The lambda_2 terms are the
        product of the polynomials of the second-layer projection (D_beta) and of the first-layer grids (G_alpha),
        truncated at the order; the norms are the first-layer polynomial."""
        flat, norms = self.field_grids(klin, pklin, f)
        proj = self.projection
        if not self.ap:
            lam = np.einsum("eg,ecg->ec", proj.matrix, flat[proj.field_index])
            return s21m_assemble(lam, norms[proj.field_index], self.fog)
        nodes, index = self.nodes, proj.field_index
        projection = self.ap_poly.terms  # (nalpha, nentry, ngrid), without the volume factor
        lam = np.zeros((len(nodes.powers), len(index), flat.shape[2]))
        for i, (p1, r1) in enumerate(nodes.powers):
            for j, (p2, r2) in enumerate(nodes.powers):
                if p1 + p2 + r1 + r2 <= nodes.order:
                    lam[nodes.powers.index((p1 + p2, r1 + r2))] += np.einsum("eg,ecg->ec", projection[i], flat[j][index])
        return np.concatenate([s21m_assemble(lam[a], norms[a][index], self.fog) for a in range(len(nodes.powers))],
                              axis=1)

    def field_grids(self, klin, pklin, f):
        """The unprojected S21m pieces: modulus-spectrum grids (nfield, ncol, nk * nmu) of every first-layer field
        (``self.fields``) on the (K, mu_K) grid of ``self.spectra``, and their first-layer norms (nfield, nnorm). The
        projection to lambda_2 (fixed, or with AP) and ``s21m_assemble`` then give the S21m terms."""
        logk, logp = np.log(klin), np.log(pklin)
        pk = lambda x: np.exp(np.interp(np.log(np.maximum(x, 1e-30)), logk, logp))  # noqa: E731
        sigma_v = self.sigma_v_factor * float(np.sqrt(np.trapezoid(pklin, klin) / (6 * np.pi**2)))
        if self.ap:  # the first-layer filters at q = 1 and the AP nodes, fitted as a polynomial in x = ln q
            stretches = [np.ones(3)] + [np.exp([b, b, a]) for a, b in self.nodes.points]
            grids, norms = self.spectra(pk, f, sigma_v, fog=self.fog, stretches=stretches, fit=self.nodes.matrix)
            nalpha, nfield, nk, nmu, ncol = grids.shape
            return np.moveaxis(grids, -1, 2).reshape(nalpha, nfield, ncol, nk * nmu), norms
        grids, norms = self.spectra(pk, f, sigma_v, fog=self.fog)
        nfield, nk, nmu, ncol = grids.shape
        return np.moveaxis(grids, -1, 1).reshape(nfield, ncol, nk * nmu), norms


def one_point_ratio(n, q):
    """Gaussian <U>^q / E[U^q] of an n-component modulus."""
    return np.exp(q * (gammaln((n + 1) / 2) - gammaln(n / 2)) - (gammaln((n + q) / 2) - gammaln(n / 2)))


def s21m_from_terms(terms, amplitudes, n1, n2, q, edgeworth_q=0.0, edgeworth_1=0.0):
    """S21m from the basis terms (nentry, 2), the noise amplitude of each entry's first-layer field, the block
    sizes n1, n2 of each entry and the first-layer Edgeworth factors E_1(q), E_1(1) of each entry."""
    lam2 = terms[:, 0] + (1 + amplitudes) * terms[:, 1]
    gamma2 = np.exp(gammaln((n2 + q) / 2) - gammaln(n2 / 2))
    return (gamma2 * one_point_ratio(n1, q) * (2 * lam2) ** (q / 2)
            * (1 + edgeworth_1) ** q / (1 + edgeworth_q))
