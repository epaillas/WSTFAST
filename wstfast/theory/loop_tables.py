"""Cosmology-independent tables of the unregulated redshift-space one-loop integrals (``eft_loop.loop_integrals``).

The loop integrands are products of Z kernels, which depend on the cosmology only through f and are exact
polynomials in f (``bias.galaxy_kernel_powers``), times linear spectra at |p| and |k - p| (P22) or at |p| and k (P13).
On the fixed quadrature of ``loop_integrals`` (convention=None), the kernel products, summed over the azimuth phi and
weighted with the quadrature weights, are tabulated once per power of f:

    P22(k, mu) = sum_a f^a sum_(p, x, sign) T22_a(k, mu; p, x, sign) P(p) P(|k - sign p|),
    P13(k, mu) = P(k) sum_a f^a sum_p T13_a(k, mu; p) P(p),

so that a cosmology costs only these contractions; the same tables serve the no-wiggle spectrum of the IR
resummation. The results equal ``loop_integrals`` to rounding.
"""

from __future__ import annotations

import numpy as np

from .bias import MATTER_BETA, galaxy_kernel_powers, multiply_first
from .eft_loop import Quadrature, _vectors, radial_rule

NPOWER = 5  # f^0 .. f^4 in the products Z2 Z2 and Z1 Z3


def _power_products(left, right=None, biased=True):
    """Coefficients per power of f of the product of two kernels in powers of f, given components first as
    (npower, 5, ...) (biased) or contracted with beta = (1, 1, 0, 0, 0) as (npower, ...) (matter): returns
    (NPOWER, 15, ...) or (NPOWER, 1, ...). ``right=None`` is the square of ``left`` (each pair of powers once)."""
    square = right is None
    right = left if square else right
    out = np.zeros((NPOWER, 15 if biased else 1) + left.shape[2 if biased else 1:])
    for i in range(left.shape[0]):
        for j in range(i if square else 0, right.shape[0]):
            if np.any(left[i]) and np.any(right[j]):
                term = multiply_first(left[i], 1, right[j], 1) if biased else (left[i] * right[j])[None]
                out[i + j] += term if not square or i == j else 2 * term
    return out


class LoopTables:
    """Tables of ``loop_integrals(k, mu, power, f, None, quadrature, rsd=True, biased=biased)``.

    ``__call__(power, f)`` returns the same dict ('22', '13', 'loop'), each (nk, nmu) for matter or (nk, nmu, 15)
    (degree-2 bias monomials) with ``biased``. Building costs about one direct evaluation.
    """

    def __init__(self, k, mu, quadrature: Quadrature = Quadrature(), biased: bool = False):
        self.k, self.mu = np.asarray(k, dtype=float), np.asarray(mu, dtype=float)
        self.quadrature, self.biased = quadrature, biased
        self.p, self.p22, self.pr, self.t22, self.t13 = [], [], [], [], []
        self.columns22 = self.columns13 = None
        for ki in self.k:
            self._build(ki)

    def _kernels(self, vectors):
        """Kernel powers of f, components first: (npower, 5, ...) or, for matter, (npower, ...)."""
        powers = galaxy_kernel_powers(vectors)
        if not self.biased:
            return np.ascontiguousarray(np.moveaxis(powers @ MATTER_BETA, -1, 0))
        return np.ascontiguousarray(np.moveaxis(powers, (-2, -1), (0, 1)))

    def _build(self, ki):
        q = self.quadrature
        gx, gw = np.polynomial.legendre.leggauss(q.nx)
        phi = 2 * np.pi * np.arange(q.nphi) / q.nphi
        p, rw = radial_rule(q.qmin, q.qmax, q.nq, breaks=(ki / 2, ki, 2 * ki))
        nmu = self.mu.size
        ncomp = 15 if self.biased else 1
        t13 = np.zeros((nmu, p.size, NPOWER, ncomp))
        t22, p_all, pr_all = [], [], []
        for start in range(0, p.size, q.chunk):
            pp, rrw = p[start:start + q.chunk], rw[start:start + q.chunk]
            boundary = np.minimum(1., ki / (2 * pp))
            for lower, upper in ((np.zeros_like(pp), boundary), (boundary, np.ones_like(pp))):
                x = lower[:, None] + .5 * (upper - lower)[:, None] * (gx + 1)
                wx = .5 * (upper - lower)[:, None] * gw
                kv, pv = _vectors(ki, self.mu, pp, x, phi)
                weight = (rrw[:, None] * pp[:, None] ** 3 * wx / (2 * np.pi**2 * q.nphi))[None, :, :, None]
                gamma = self._kernels(np.stack([kv, pv, -pv], axis=-2))
                lin = self._kernels(kv[..., None, :])
                i13 = 6 * _power_products(lin, gamma, self.biased) * weight  # (NPOWER, ncomp, nmu, np, nx, nphi)
                t13[:, start:start + pp.size] += np.moveaxis(i13.sum(axis=(4, 5)), (0, 1), (2, 3))
                for sign in (1., -1.):
                    remainder = kv - sign * pv
                    pr = np.linalg.norm(remainder[0, :, :, 0], axis=-1)  # (np, nx): independent of mu and phi
                    k2 = self._kernels(np.stack([sign * pv, remainder], axis=-2))
                    mask = 2. * (pr > pp[:, None])[None, :, :, None]
                    i22 = (_power_products(k2, biased=self.biased) * weight * mask).sum(axis=-1)  # (NPOWER, ncomp, nmu, np, nx)
                    t22.append(np.moveaxis(i22, (0, 1), (3, 4)).reshape(nmu, -1, NPOWER * ncomp))
                    pr_all.append(pr.ravel())
                    p_all.append(np.broadcast_to(pp[:, None], pr.shape).ravel())
        t22 = np.concatenate(t22, axis=1)
        t13 = t13.reshape(nmu, p.size, NPOWER * ncomp)
        if self.columns22 is None:  # structural zeros (bias monomials absent at some power of f)
            self.columns22 = np.flatnonzero(np.any(t22 != 0, axis=(0, 1)))
            self.columns13 = np.flatnonzero(np.any(t13 != 0, axis=(0, 1)))
        for table, columns in ((t22, self.columns22), (t13, self.columns13)):
            if np.any(np.delete(table, columns, axis=-1)):
                raise RuntimeError("loop table columns differ between k values")
        self.p.append(p)
        self.p22.append(np.concatenate(p_all))
        self.pr.append(np.concatenate(pr_all))
        self.t22.append(np.ascontiguousarray(t22[..., self.columns22]))
        self.t13.append(np.ascontiguousarray(t13[..., self.columns13]))

    def _expand(self, values, columns, f):
        """(nk, nmu, ncolumns) -> sum over the powers of f: (nk, nmu, ncomp)."""
        ncomp = 15 if self.biased else 1
        full = np.zeros(values.shape[:2] + (NPOWER * ncomp,))
        full[..., columns] = values
        return np.einsum("kmac,a->kmc", full.reshape(values.shape[:2] + (NPOWER, ncomp)), f ** np.arange(NPOWER))

    def __call__(self, power, f):
        f = float(f)
        i22, i13 = [], []
        for ki, p, p22, pr, t22, t13 in zip(self.k, self.p, self.p22, self.pr, self.t22, self.t13):
            # pr = 0 only where the table (the mask pr > p) is zero
            i22.append(np.einsum("mpc,p->mc", t22, power(p22) * power(np.where(pr > 0, pr, 1.))))
            i13.append(power(np.array([ki]))[0] * np.einsum("mpc,p->mc", t13, power(p)))
        out = {"22": self._expand(np.array(i22), self.columns22, f),
               "13": self._expand(np.array(i13), self.columns13, f)}
        if not self.biased:
            out = {name: value[..., 0] for name, value in out.items()}
        out["loop"] = out["22"] + out["13"]
        return out
