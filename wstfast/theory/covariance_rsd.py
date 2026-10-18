"""Analytic covariance of the redshift-space WST data vector of a biased tracer: S1m blocks, S21m entries, P_l.

Extends ``covariance.RSDGaussianCovariance`` (S1m and P_l) to the line-of-sight-resolved S21m entries
S21m(j1, j2, l, |m1|, |m2|) = S2m / S1m(j1, l, |m1|), along the lines of the real-space ``covariance.ShellCovariance``.
The second layer of an S21m entry is the wavelet transform of a first-layer modulus field U_a, a = (j1, l, |m1|), with
the block weight W2 = R2^2 sum_{m in |m2|} Y_m^2 at sigma_{j2}. In units of <U_a><U_b>,

    P_{U_a U_b}(K) = r_a r_b P_L(K) + N_ab(K),        P_{U_a d}(K) = r_a Z1(K) P_L(K),

with the tree response r_a (second_layer: r0 + N r1 over n lambda) and the noise N_ab: the Gaussian-chaos cross noise
g_ab of the tracer (``ModulusCrossRSD``), with the auto noise of the fit (1 + a_a) g_aa + W and the cross noise
rescaled by the geometric mean of the auto ratios. The part of U_a correlated with the analysed field d is
c_a = P_{U_a d} / P_d; the rest, N_ab - c_a c_b P_d + r_a r_b P_L, is its own noise.

2nd chaos (Gaussian band powers, Cov(dP_d(k), dP_d(k')) = 2 P_d^2 per lattice mode, plus the noise of U):

* ``delta``: ln S21m moves with the band powers of d at the second-layer scales, q/2 c_a^2 W2 / (V Sigma_2), and,
  through its S1m(j1, l, |m1|) denominator, by -q/2 W1 / (V Sigma_1);
* ``noise_level``: the small-scale band powers that set the noise of U_a, weighted by the noise fraction f_N of the
  second-layer variance: q/2 f_N (2 P_d W1^2 / sum P_d^2 W1^2 - W1 / (V Sigma_1));
* ``u_noise``: band powers of the noise of U at the second-layer scales, q^2/2 / V^2 sum W2_e W2_f
  [2 C_ab N'_ab + N'_ab^2] / (Sigma_2e Sigma_2f), C_ab = c_a c_b P_d the correlated and N'_ab the rest.

4th chaos: every last layer (S1m blocks of d, S21m second layers of U_a) exactly on FFT lattices
(``covariance.chaos_covariances``, anisotropic), with the cross spectra above.
"""

from __future__ import annotations

import numpy as np

from .bias import galaxy_kernel
from .covariance import HalfLattice, chaos_covariances, components
from .bias import linear_components
from .first_layer import spherical_harmonic
from .second_layer import ModulusSpectra


class ModulusCrossRSD(ModulusSpectra):
    """Response r_a and Gaussian-chaos cross noise g_ab of every pair of first-layer fields of a biased, discrete
    tracer at fixed biases, on the (K, mu_K) grid: the fixed-beta, cross-field version of second_layer.ModulusSpectra
    (whose auto terms it reproduces: r_a^2 P_L(K) and g_aa)."""

    def spectra(self, pk, f: float, sigma_v: float, beta, shotnoise: float):
        """Return r (nfield, nk, nmu), g (nfield, nfield, nk, nmu) and the block normalisations n lambda (nfield,)."""
        beta = np.asarray(beta, "f8")
        N = float(shotnoise)
        damp = lambda k, mu: np.exp(-0.5 * (f * k * mu * sigma_v) ** 2)  # noqa: E731
        p, w = self.p, self.weights
        dp, pkp = damp(p, self.mup), pk(p)
        z1p = linear_components(self.pvec, f) @ beta
        pdp = pkp * z1p**2 * dp**2 + N
        radial = {s: np.exp(-0.5 * (s * p) ** 2) for s, _, _ in self.fields}
        norms, kernels = [], []
        for s, ell, m in self.fields:
            c2w = 4 * np.pi / (2 * ell + 1) * (1.0 if m == 0 else 2.0)
            rp = (s * p) ** ell * radial[s]
            norms.append(np.sum(w * rp**2 * c2w * np.abs(self.harmonics[ell, m]) ** 2 * pdp))
            kernels.append((-1) ** ell * c2w * rp)
        norms = np.array(norms)
        nf, nk, nmu = len(self.fields), self.k.size, self.mu.size
        r = np.zeros((nf, nk, nmu))
        g = np.zeros((nf, nf, nk, nmu))
        for i, kk in enumerate(self.k):
            for a, mk in enumerate(self.mu):
                kvec = np.array([kk * np.sqrt(1 - mk**2), 0.0, kk * mk])
                qvec = kvec - self.pvec
                qn = np.sqrt(np.sum(qvec**2, axis=-1))
                muq = qvec[:, 2] / np.maximum(qn, 1e-30)
                dq, pkq = damp(qn, muq), pk(qn)
                z1q = linear_components(qvec, f) @ beta
                minus = np.broadcast_to(-kvec, qvec.shape)
                long = 2 * (pkq * dp * dq * (galaxy_kernel(np.stack([qvec, minus], axis=-2), f) @ beta) * z1q
                            + pkp * dp * dq * (galaxy_kernel(np.stack([minus, self.pvec], axis=-2), f) @ beta) * z1p)
                z1k = linear_components(kvec, f) @ beta
                pair = w * pdp * (pkq * z1q**2 * dq**2 + N)
                cache, kern = {}, []
                for index, (s, ell, m) in enumerate(self.fields):
                    if (ell, m) not in cache:
                        cache[ell, m] = np.real(self.harmonics[ell, m] * np.conj(spherical_harmonic(ell, m, qvec)))
                    kern.append(kernels[index] * (s * qn) ** ell * np.exp(-0.5 * (s * qn) ** 2) * cache[ell, m])
                kern = np.array(kern)  # (nf, np)
                r[:, i, a] = (kern @ (w * long) / 2 + N * z1k * (kern @ w) / 2) / norms
                g[:, :, i, a] = (kern * pair) @ kern.T / 2 / np.outer(norms, norms)
        return r, g, norms


class RSDWSTCovariance:
    """Covariance of ln S1m, ln S21m and P_l of one box (2nd chaos by half-lattice sums, plus the 4th chaos).

    ``pfield(k, mu)``: power of the analysed field d (window and shot noise included); ``plin(k)``: linear matter
    power; ``z1(k, mu)``: the tracer's linear kernel b1 + f mu^2; ``moduli(a, b, k, mu)`` -> (r_a, r_b, N_ab) of the
    first-layer fields (callables on arbitrary k, mu; see ``ModulusTables``). ``s1m``: S1 coefficients; ``s21m``:
    list of (j1, j2, l, m1, m2, field index); the order is [S1m blocks, S21m entries, P_l bins].
    """

    TERMS = ("delta", "noise_level", "u_noise")

    def __init__(self, config, s1m, s21m, fields, q, boxsize, pfield, plin, z1, moduli, pk_edges=None,
                 pk_ells=(0, 2, 4), nmesh=128, include=TERMS):
        from scipy.special import eval_legendre

        lat = HalfLattice(nmesh, boxsize)
        V = boxsize**3
        keep = lat.positive & (lat.k <= np.pi * nmesh / boxsize)
        k, mu, mult = lat.k[keep], lat.mu[keep], lat.multiplicity[keep]
        P = pfield(k, mu)
        PL = plin(k)
        Z1 = z1(k, mu)

        def block_weight(sigma, ell, m):
            return lat.radial(sigma, ell)[keep] ** 2 * np.sum(lat.harmonics(ell)[components(ell, m)][:, keep] ** 2, axis=0)

        rows, self.labels = [], []
        for c in s1m:
            for m in range(c.ell + 1):
                w = block_weight(config.sigma(c.j), c.ell, m)
                rows.append(q / 2 * w / (V * (P * w * mult).sum() / V))
                self.labels.append(f"S1m_j{c.j}_l{c.ell}_m{m}")
        self.ns1m = len(rows)
        # first-layer fields: block weight, Sigma_1, and the normalisation of the noise level
        first = {}
        for index, (sigma, ell, m) in enumerate(fields):
            w1 = block_weight(sigma, ell, m)
            first[index] = (w1, (P * w1 * mult).sum() / V, (mult * P**2 * w1**2).sum())
        # moduli spectra on the lattice, per pair of fields used
        used = sorted({e[5] for e in s21m})
        r = {a: moduli.response(a, k, mu) for a in used}
        noise = {}
        for a in used:
            for b in used:
                if b >= a:
                    noise[a, b] = noise[b, a] = moduli.noise(a, b, k, mu)
        corr = {a: r[a] * Z1 * PL / P for a in used}  # c_a = P_{U_a d} / P_d
        second = []
        for (j1, j2, ell, m1, m2, a) in s21m:
            w2 = block_weight(config.sigma(j2), ell, m2)
            puu = r[a] ** 2 * PL + noise[a, a]
            sigma2 = (puu * w2 * mult).sum() / V
            n_own = puu - corr[a] ** 2 * P
            fraction = (n_own * w2 * mult).sum() / V / sigma2
            w1, sigma1, norm_n = first[a]
            row = np.zeros_like(P)
            if "delta" in include:
                row += q / 2 * corr[a] ** 2 * w2 / (V * sigma2) - q / 2 * w1 / (V * sigma1)
            if "noise_level" in include:
                row += q / 2 * fraction * (2 * P * w1**2 / norm_n - w1 / (V * sigma1))
            rows.append(row)
            second.append((a, w2, sigma2))
            self.labels.append(f"S21m_j{j1}_j{j2}_l{ell}_m{m1}_m{m2}")
        self.ns21m = len(second)
        if pk_edges is not None:
            index = np.digitize(k, pk_edges) - 1
            for ell in pk_ells:
                legendre = (2 * ell + 1) * eval_legendre(ell, mu)
                for i in range(len(pk_edges) - 1):
                    inside = index == i
                    nmodes = mult[inside].sum()
                    rows.append(np.where(inside, legendre / nmodes, 0.0))
                    kmean = (mult[inside] * k[inside]).sum() / nmodes
                    self.labels.append(f"P{ell}_k{kmean:.4f}")
        D = np.array(rows)
        self.delta_part = (D * (2 * P**2 * mult)) @ D.T
        self.u_part = np.zeros_like(self.delta_part)
        if "u_noise" in include:
            o = self.ns1m
            for i, (a, w2a, sa) in enumerate(second):
                for j, (b, w2b, sb) in enumerate(second):
                    if j < i:
                        continue
                    cab = corr[a] * corr[b] * P
                    nab = r[a] * r[b] * PL + noise[a, b] - cab
                    value = q**2 / 2 / V**2 * (mult * w2a * w2b * (2 * cab * nab + nab**2)).sum() / (sa * sb)
                    self.u_part[o + i, o + j] = self.u_part[o + j, o + i] = value
        self.lncov = self.delta_part + self.u_part
        self._chaos_inputs = (config, s1m, s21m, fields, q, boxsize, pfield, plin, z1, moduli)

    def chaos4(self, nmesh=128, **kwargs):
        """4th chaos of every last layer (S1m blocks of d and S21m second layers of U_a), (ns1m + ns21m) square."""
        config, s1m, s21m, fields, q, boxsize, pfield, plin, z1, moduli = self._chaos_inputs
        last, source = [], []
        for c in s1m:
            for m in range(c.ell + 1):
                last.append((config.sigma(c.j), c.ell, m))
                source.append(None)
        for (j1, j2, ell, m1, m2, a) in s21m:
            last.append((config.sigma(j2), ell, m2))
            source.append(a)
        cache = {}

        def spectra(i, j, k, mu):
            ua, ub = source[i], source[j]
            key = (ua, ub, k.shape, float(k.flat[-1]), float(np.asarray(mu).flat[-1]))
            if key not in cache:
                if ua is None and ub is None:
                    value = pfield(k, mu)
                elif ua is None or ub is None:
                    value = moduli.response(ua if ub is None else ub, k, mu) * z1(k, mu) * plin(k)
                else:
                    value = (moduli.response(ua, k, mu) * moduli.response(ub, k, mu) * plin(k)
                             + moduli.noise(ua, ub, k, mu))
                cache[key] = value
            return cache[key]

        _, cov4 = chaos_covariances(last, q, boxsize, nmesh, spectra, anisotropic=True, **kwargs)
        return cov4


class ModulusTables:
    """r_a(k, mu) and N_ab(k, mu) of the first-layer fields from ``ModulusCrossRSD`` tables: linear in ln k (constant
    outside the table), Lagrange in mu^2 through the Gauss nodes. ``amplitudes`` (nfield,) and ``white`` (nfield,) set
    the auto noise (1 + a) g_aa + W; the cross noise is g_ab rescaled by sqrt(N_aa N_bb / (g_aa g_bb))."""

    def __init__(self, k, mu, r, g, amplitudes=None, white=None):
        from .spectrum import lagrange_matrix

        self.logk, self.nodes = np.log(np.asarray(k)), np.asarray(mu) ** 2
        self.lagrange = lambda mu: lagrange_matrix(self.nodes, np.asarray(mu) ** 2)  # noqa: E731
        nf = r.shape[0]
        a = np.zeros(nf) if amplitudes is None else np.asarray(amplitudes)
        W = np.zeros(nf) if white is None else np.asarray(white)
        auto = np.array([(1 + a[i]) * g[i, i] + W[i] for i in range(nf)])
        ratio = auto / np.array([g[i, i] for i in range(nf)])
        self.r = r
        self.n = g * np.sqrt(ratio[:, None] * ratio[None, :])
        for i in range(nf):
            self.n[i, i] = auto[i]

    def _interp(self, table, k, mu):
        k, mu = np.broadcast_arrays(np.asarray(k, "f8"), np.asarray(mu, "f8"))
        lk = np.clip(np.log(np.maximum(k, 1e-30)), self.logk[0], self.logk[-1])
        i = np.clip(np.searchsorted(self.logk, lk) - 1, 0, self.logk.size - 2)
        t = (lk - self.logk[i]) / (self.logk[i + 1] - self.logk[i])
        L = self.lagrange(np.abs(mu).ravel()).reshape(mu.shape + (self.nodes.size,))
        lower, upper = table[i], table[i + 1]  # (..., nmu)
        return np.sum(((1 - t)[..., None] * lower + t[..., None] * upper) * L, axis=-1)

    def response(self, a, k, mu):
        return self._interp(self.r[a], k, mu)

    def noise(self, a, b, k, mu):
        return self._interp(self.n[a, b], k, mu)
