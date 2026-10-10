"""Redshift-space kernels of Eulerian-biased tracers, decomposed along the bias vector.

The tracer density through cubic order (Assassi et al. 2014; the CLASS-PT / dsc-model basis) is

    delta_g = b1 delta + b2 / 2 (delta^2 - <delta^2>) + bG2 G2[Phi_g] + bGamma3 (G2[Phi_g] - G2[Phi_v]),

with G2[Phi](k1, k2) = sigma^2(k1, k2) = (k1.k2)^2 / (k1^2 k2^2) - 1 and the nonlinear potentials of delta (Phi_g)
and theta (Phi_v). The number-density map to redshift space with the matter velocity u = k_par / k^2 theta,

    delta_s = delta_g + sum_j (f k_par)^j / j! [u^j (1 + delta_g)],

is affine in the biases, so every Z_n is a linear combination of the components along

    beta = (1, b1, b2, bG2, bGamma3)

(component 1: the pure velocity terms). Z1 = b1 + f mu^2 and the matter kernels are beta = (1, 1, 0, 0, 0).
Products of d kernels are homogeneous polynomials of degree d in beta: ``product_coefficients`` maps them to the
coefficients of the monomials ``monomial_exponents(d)``, which the theories contract with ``monomials(beta, d)``.
This keeps every emulated basis independent of the biases.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement
from math import factorial

import numpy as np

from .eft_loop import _divide, _dot, _Kernels, _product, _subsets

#: Free bias parameters; beta = (1, *BIAS_NAMES).
BIAS_NAMES = ("b1", "b2", "bG2", "bGamma3")
NBETA = len(BIAS_NAMES) + 1


class _BiasedKernels(_Kernels):

    def _zero_mean(self, kernel):
        """Operators are defined with their mean subtracted: zero kernel at zero total momentum."""
        return {mask: np.where(self.norms[mask] == 0., 0., value) for mask, value in kernel.items()}

    def _tidal_product(self, left, right):
        """Symmetrized kernel of G2 built from the potentials of ``left`` and ``right``: sigma^2(p, q) L(p) R(q)."""
        out = {}
        for mask in self.momenta:
            value = None
            for sub, rest, weight in _subsets(mask):
                if sub not in left or rest not in right:
                    continue
                s2 = _divide(_dot(self.momenta[sub], self.momenta[rest]) ** 2, self.norms[sub] * self.norms[rest]) - 1.
                term = weight * s2 * left[sub] * right[rest]
                value = term if value is None else value + term
            if value is not None:
                out[mask] = value
        return self._zero_mean(out)

    def operators(self):
        """Kernels of the bias operators b1, b2, bG2, bGamma3 (dicts over momentum masks)."""
        square = {mask: .5 * value for mask, value in self._zero_mean(_product(self.F, self.F)).items()}
        tidal = self._tidal_product(self.F, self.F)
        gamma3 = {mask: value - self._tidal_product(self.G, self.G).get(mask, 0.) for mask, value in tidal.items()}
        return [self.F, square, tidal, gamma3]

    def powers(self):
        """Z_n of the full mask along beta and in powers of f: (..., n + 1, NBETA), Z_n = sum_j f^j [..., j, :]."""
        velocity = {mask: _divide(self.parallel[mask], self.norms[mask]) * self.G[mask] for mask in self.momenta}
        zeros = np.zeros(self.batch_shape)
        ops = self.operators()
        out = [[zeros] + [zeros + op.get(self.full, 0.) for op in ops]]
        power = velocity
        for j in range(1, self.n + 1):
            factor = self.parallel[self.full] ** j / factorial(j)
            out.append([factor * power.get(self.full, 0.)]
                       + [factor * _product(op, power).get(self.full, 0.) for op in ops])
            power = _product(power, velocity)
        return np.stack([np.stack(row, axis=-1) for row in out], axis=-2)

    def components(self, f):
        """Z_n of the full mask along beta: (..., NBETA)."""
        return np.einsum("...jb,j->...b", self.powers(), float(f) ** np.arange(self.n + 1))


def galaxy_kernel(vectors, f=0., los=(0., 0., 1.)):
    """Components of the symmetrized redshift-space tracer Z_n along beta, shape (..., NBETA), n <= 3."""
    return _BiasedKernels(vectors, los).components(float(f))


def galaxy_kernel_powers(vectors, los=(0., 0., 1.)):
    """Components of Z_n along beta and in powers of f: (..., n + 1, NBETA), with Z_n = sum_j f^j [..., j, :]."""
    return _BiasedKernels(vectors, los).powers()


@lru_cache(None)
def monomial_exponents(degree: int, nvar: int = NBETA) -> np.ndarray:
    """Exponents (nmono, nvar) of the monomials of total degree ``degree`` in beta, in a fixed order."""
    out = []
    for combo in combinations_with_replacement(range(nvar), degree):
        out.append(np.bincount(combo, minlength=nvar))
    return np.array(out, dtype=int)


@lru_cache(None)
def _product_terms(d1: int, d2: int, nvar: int = NBETA):
    """(target, i, j) for every monomial i of degree d1 and j of degree d2: x^i x^j = x^target."""
    return tuple((int(t), i, j) for (i, j), t in np.ndenumerate(
        np.argmax(_product_map(d1, d2, nvar), axis=1).reshape(len(monomial_exponents(d1, nvar)), -1)))


def multiply_first(a, d1: int, b, d2: int):
    """Coefficients (n12, ...) of the product of polynomials of degrees d1 and d2 with coefficients first: a (n1, ...),
    b (n2, ...). Accumulates contiguous arrays (no outer product)."""
    out = np.zeros((len(monomial_exponents(d1 + d2)),) + np.broadcast_shapes(a.shape[1:], b.shape[1:]))
    for target, i, j in _product_terms(d1, d2):
        out[target] += a[i] * b[j]
    return out


def _multiply(a, d1, b, d2):
    """Coefficients-last version of ``multiply_first``: a (..., n1), b (..., n2) -> (..., n12)."""
    return np.moveaxis(multiply_first(np.moveaxis(a, -1, 0), d1, np.moveaxis(b, -1, 0), d2), 0, -1)


def product_coefficients(*factors):
    """Monomial coefficients (..., nmono) of the product of affine-in-beta factors, each (..., NBETA)."""
    first = [np.ascontiguousarray(np.moveaxis(factor, -1, 0)) for factor in factors]
    out = first[0]
    for degree, factor in enumerate(first[1:], start=1):
        out = multiply_first(out, degree, factor, 1)
    return np.moveaxis(out, 0, -1)


def monomials(beta, degree: int):
    """Values (nmono,) of the monomials of degree ``degree`` at beta (JAX- and NumPy-compatible)."""
    import jax.numpy as jnp

    exponents = jnp.asarray(monomial_exponents(degree))
    return jnp.prod(jnp.asarray(beta)[None, :] ** exponents, axis=1)


def beta_vector(b1=1., b2=0., bG2=0., bGamma3=0.):
    import jax.numpy as jnp

    return jnp.stack([jnp.ones_like(jnp.asarray(b1, dtype=float)), b1, b2, bG2, bGamma3])


MATTER_BETA = np.array([1., 1., 0., 0., 0.])


@lru_cache(None)
def _product_map(d1: int, d2: int, nvar: int = NBETA) -> np.ndarray:
    """(n1 * n2, n12) 0/1 matrix multiplying monomial coefficients of degrees d1 and d2."""
    e1, e2, e12 = monomial_exponents(d1, nvar), monomial_exponents(d2, nvar), monomial_exponents(d1 + d2, nvar)
    lookup = {tuple(e): i for i, e in enumerate(e12)}
    matrix = np.zeros((len(e1) * len(e2), len(e12)))
    for i, a in enumerate(e1):
        for j, b in enumerate(e2):
            matrix[i * len(e2) + j, lookup[tuple(a + b)]] = 1.
    return matrix


def monomial_product(a, d1: int, b, d2: int):
    """Coefficients (..., n12) of the product of polynomials with coefficients a (..., n1) and b (..., n2)."""
    return _multiply(np.asarray(a), d1, np.asarray(b), d2)


def lift(coefficients, d_from: int, d_to: int):
    """Homogenize: multiply by beta_0^(d_to - d_from) = 1 (coefficients of degree d_from -> d_to)."""
    unit = np.zeros(len(monomial_exponents(d_to - d_from)))
    unit[0] = 1.  # beta_0^n is the first monomial
    return monomial_product(np.asarray(coefficients), d_from, unit, d_to - d_from)


def monomial_position(exponents) -> int:
    """Index of the monomial with ``exponents`` in ``monomial_exponents(sum(exponents))``."""
    exponents = np.asarray(exponents)
    table = monomial_exponents(int(exponents.sum()))
    return int(np.flatnonzero(np.all(table == exponents, axis=1))[0])
