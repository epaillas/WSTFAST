"""Column layout of the cosmology-dependent basis: everything is resolved into the monomials of the bias vector
beta = (1, b1, b2, bG2, bGamma3) (``bias.py``), so the basis depends on cosmology only and serves any tracer (matter is
beta = (1, 1, 0, 0, 0) with no shot noise).

- P_s grid rows (``GRID_TERMS``): TREE (degree 2 in beta, (b1 + f mu^2)^2 P_L), LOOP (degree 2, one loop) and the
  counterterm shapes k^2 P_L mu^{0, 2, 4}.
- S1m cumulants, per |m| block: the tree trispectrum (degree 4) and the Poisson terms of a discrete tracer,
  T + B / n + P / n^2 + 1 / n^3 (``KAPPA_SIZES``); with ``cubic``, the T_3111 of the cubic operators
  (``CUBIC_KAPPA_SIZES``); and the m = 0 third cumulant B + P / n + 1 / n^2 (``SKEW_SIZES``).
- S21m, per entry (``S21_SIZES``): the response R0 + N R1 + N^2 R2, the noise G0 + N G1 + N^2 G2 and the first-layer
  normalization n0 + N n1, with N the tracer shot noise:

      lambda_2 = [R0 + N R1 + N^2 R2 + (1 + a)(G0 + N G1 + N^2 G2)] / (n0 + N n1)^2.

- ``fog``: the Fingers-of-God counterterm orders 1 and 2 appended to the blocks they act on (``fog_sizes``), combined
  at a given c_v by ``collapse_fog``.
- ``ap``: the columns repeated per term of the AP polynomial in (ln q_par, ln q_perp), alpha-major (``AP_TERMS``,
  ``ap_evaluate``).

``amplitude_scalings`` gives the powers of A_s of every column (a Taylor emulator is then nearly exact in logA) and
``basis_layout`` reads (cubic, fog, nalpha) back from the number of kappa columns.
"""

from __future__ import annotations

import numpy as np

from .bias import CUBIC_NAMES, monomial_exponents, monomial_position


#: Degree-2 monomials of the tree spectrum: beta_0^2 (f^2 mu^4), beta_0 b1 (2 f mu^2), b1^2 (1).
TREE = tuple(monomial_position(e) for e in ([2, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 2, 0, 0, 0]))


#: Degree-2 monomials of the one-loop spectrum (bGamma3 only enters P13, with beta_0 or b1).
LOOP = tuple(i for i, e in enumerate(monomial_exponents(2)) if not (e[4] and (e[2] or e[3] or e[4] == 2)))


#: Rows of the P_s grid.
GRID_TERMS = (tuple(f"tree_{i}" for i in TREE) + tuple(f"loop_{i}" for i in LOOP) + ("ct0", "ct2", "ct4"))


#: Columns of the per-block cumulants: tree (deg 4), B / n (deg 3), P / n^2 (deg 2), 1 / n^3.
KAPPA_SIZES = (len(monomial_exponents(4)), len(monomial_exponents(3)), len(monomial_exponents(2)), 1)


#: Columns of the cubic-operator trispectrum (``cubic``), appended to KAPPA_SIZES: one degree-3 block per cubic bias.
CUBIC_KAPPA_SIZES = (len(monomial_exponents(3)),) * len(CUBIC_NAMES)


#: Columns of the m = 0 third cumulant: tree (deg 3), P / n (deg 2), 1 / n^2.
SKEW_SIZES = (len(monomial_exponents(3)), len(monomial_exponents(2)), 1)


#: Columns of the S21m terms: R0, R1, R2 (deg 4), G0 (deg 4), G1 (deg 2), G2, n0 (deg 2), n1.
S21_SIZES = (70, 70, 70, 70, 15, 1, 15, 1)


#: k_NL of the Fingers-of-God counterterm [h/Mpc] (Tucci & Philcox 2026).
FOG_KNL = 0.3


#: Blocks of S21_SIZES that the Fingers-of-God counterterm acts on (R2, G2 and n1 have no short-mode spectra).
FOG_S21_BLOCKS = (0, 1, 3, 4, 6)


def fog_sizes(cubic: bool = False):
    """Columns appended by ``fog`` (orders 1 and 2 of the counterterm) to the kappa, skewness and S21m outputs."""
    kappa = KAPPA_SIZES[0] + (sum(CUBIC_KAPPA_SIZES) if cubic else 0)
    s21 = sum(S21_SIZES[i] for i in FOG_S21_BLOCKS)
    return dict(kappa=2 * kappa, skewness=2 * SKEW_SIZES[0], s21m=2 * s21)


def split(values, sizes):
    """Split the last axis of ``values`` into blocks of ``sizes``."""
    edges = np.cumsum((0,) + tuple(sizes))
    return [values[..., a:b] for a, b in zip(edges[:-1], edges[1:])]


def amplitude_scalings(ratio, cubic: bool = False, fog: bool = False, ap_terms: int = 1):
    """Powers of A_s / A_s,fid of every basis output (grid rows, kappa / skewness / S21m columns), so that a
    Taylor emulator is nearly exact in logA. ``cubic``: the kappa columns include CUBIC_KAPPA_SIZES (P^3, like the
    tree trispectrum). ``fog``: with the counterterm orders of ``fog_sizes`` (the powers of the blocks they expand)."""
    import jax.numpy as jnp

    xp = np if isinstance(ratio, (float, np.floating)) else jnp
    powers = lambda counts: xp.concatenate([xp.full((n,), 1.0) * ratio**p for p, n in counts])  # noqa: E731
    clustering = [(3, KAPPA_SIZES[0])] + ([(3, sum(CUBIC_KAPPA_SIZES))] if cubic else [])
    s21 = list(zip((3, 2, 1, 2, 1, 0, 1, 0), S21_SIZES))
    s21_fog = [s21[i] for i in FOG_S21_BLOCKS]
    # with ``ap_terms`` AP-polynomial terms the columns repeat per term (alpha-major)
    return dict(grid=powers([(1, len(TREE)), (2, len(LOOP)), (1, 3)]),
                kappa=powers(ap_terms * (list(zip((3, 2, 1, 0), KAPPA_SIZES))
                                         + ([(3, sum(CUBIC_KAPPA_SIZES))] if cubic else []) + (2 * clustering if fog else []))),
                skewness=powers(ap_terms * (list(zip((2, 1, 0), SKEW_SIZES)) + (2 * [(2, SKEW_SIZES[0])] if fog else []))),
                s21m=powers(ap_terms * (s21 + (2 * s21_fog if fog else []))))


#: Number of terms of an AP polynomial of total degree 0, 1, 2, 3, 4 in (ln q_par, ln q_perp) (``spectrum.APNodes``).
AP_TERMS = (1, 3, 6, 10, 15)


def basis_layout(kappa_columns: int):
    """(cubic, fog, nalpha) of a basis from the number of its kappa columns: with the cubic-operator blocks, with
    the Fingers-of-God orders, and the number of AP-polynomial terms (1 without AP)."""
    bases = {(c, f): sum(KAPPA_SIZES) + (sum(CUBIC_KAPPA_SIZES) if c else 0) + (fog_sizes(c)["kappa"] if f else 0)
             for c in (False, True) for f in (False, True)}
    for nalpha in AP_TERMS:
        for (cubic, fog), base in bases.items():
            if base * nalpha == kappa_columns:
                return cubic, fog, nalpha
    raise ValueError(f"{kappa_columns} kappa columns match no basis layout")


def ap_powers(nalpha: int):
    """The (p, r) powers of x = (ln q_par, ln q_perp) of an AP polynomial with ``nalpha`` terms (``spectrum.APNodes``)."""
    order = AP_TERMS.index(nalpha)
    return [(a, b) for a in range(order + 1) for b in range(order + 1 - a)]


def ap_evaluate(values, nalpha: int, qpar=1.0, qperp=1.0):
    """sum_alpha x^alpha values_alpha of alpha-major AP-polynomial columns (last axis nalpha * n) -> (..., n). NumPy or JAX."""
    import jax.numpy as jnp

    if nalpha == 1:
        return values
    x, y = jnp.log(qpar), jnp.log(qperp)
    weights = jnp.stack([x**p * y**r for p, r in ap_powers(nalpha)])
    return jnp.tensordot(values.reshape(*values.shape[:-1], nalpha, -1), weights, axes=([-2], [0]))


def is_fog(kappa=None, s21m=None) -> bool:
    """Whether basis outputs hold the Fingers-of-God counterterm orders (from their number of columns)."""
    if kappa is not None:
        return basis_layout(kappa.shape[-1])[1]
    n = s21m.shape[-1]
    base = sum(S21_SIZES) + fog_sizes()["s21m"]
    return n % base == 0 and n // base in AP_TERMS


def _collapse(values, base, blocks, c, order):
    """values[..., :base] with the orders 1 and 2 (the trailing columns, ``blocks`` the (start, size) of each block
    they expand, in order) combined as X0 - c X1 + c^2 X2 (order 2) or X0 - c X1 (order 1)."""
    import jax.numpy as jnp

    xp = np if isinstance(values, np.ndarray) and isinstance(c, (int, float, np.floating)) else jnp
    out = values[..., :base]
    n = sum(size for _, size in blocks)
    for o, factor in ((1, -c), (2, c**2)):
        if o > order:
            break
        column, pieces, last = base + (o - 1) * n, [], 0
        for start, size in blocks:
            pieces += [out[..., last:start], out[..., start:start + size] + factor * values[..., column:column + size]]
            last, column = start + size, column + size
        out = xp.concatenate(pieces + [out[..., last:]], axis=-1)
    return out


def collapse_fog(kappa=None, skewness=None, s21m=None, c_s1=0.0, c_s21=0.0, order: int = 2):
    """Standard-layout kappa, skewness and S21m columns from outputs with the Fingers-of-God counterterm orders (see
    ``fog_sizes``), at the counterterm c_s1 of the first-layer cumulants and c_s21 of the modulus spectra. Arguments
    left None are returned None."""
    out = []
    if kappa is not None:
        cubic = kappa.shape[-1] != sum(KAPPA_SIZES) + fog_sizes()["kappa"]
        base = sum(KAPPA_SIZES) + (sum(CUBIC_KAPPA_SIZES) if cubic else 0)
        blocks = [(0, KAPPA_SIZES[0])] + ([(sum(KAPPA_SIZES), sum(CUBIC_KAPPA_SIZES))] if cubic else [])
        out.append(_collapse(kappa, base, blocks, c_s1, order))
    else:
        out.append(None)
    out.append(None if skewness is None else _collapse(skewness, sum(SKEW_SIZES), [(0, SKEW_SIZES[0])], c_s1, order))
    if s21m is not None:
        edges = np.cumsum((0,) + S21_SIZES)
        out.append(_collapse(s21m, sum(S21_SIZES), [(edges[i], S21_SIZES[i]) for i in FOG_S21_BLOCKS], c_s21, order))
    else:
        out.append(None)
    return tuple(out)


def s21m_lambda_mask(fog: bool):
    """Boolean mask of the lambda_2 columns (the second-layer sums, which carry the AP volume factor) in the S21m column
    layout of ``s21m_assemble``; the others are the first-layer norms."""
    nlam = sum(S21_SIZES[:6]) + (2 * sum(S21_SIZES[i] for i in (0, 1, 3, 4)) if fog else 0)
    nnorm = sum(S21_SIZES[6:]) + (2 * S21_SIZES[6] if fog else 0)
    return s21m_assemble(np.ones((1, nlam)), np.zeros((1, nnorm)), fog)[0] > 0.5


def s21m_assemble(lam, norms, fog: bool):
    """S21m terms in the column layout of ``second_layer.S21mBasis`` from the projected grid columns ``lam`` (..., ncol) and
    the first-layer norms of each entry (..., nnorm): [lam | norms], or with ``fog`` the base blocks followed by the
    orders 1 and 2 of both. NumPy or JAX."""
    import jax.numpy as jnp

    xp = np if isinstance(lam, np.ndarray) and isinstance(norms, np.ndarray) else jnp
    if not fog:
        return xp.concatenate([lam, norms], axis=-1)
    nbase, norder = sum(S21_SIZES[:6]), sum(S21_SIZES[i] for i in (0, 1, 3, 4))
    n0 = S21_SIZES[6]
    base = [lam[..., :nbase], norms[..., :n0 + 1]]
    orders = [[lam[..., nbase + o * norder:nbase + (o + 1) * norder],
               norms[..., n0 + 1 + o * n0:n0 + 1 + (o + 1) * n0]] for o in range(2)]
    return xp.concatenate(base + orders[0] + orders[1], axis=-1)


def s21m_field_powers(fog: bool):
    """Powers of A_s / A_s,fid of the grid columns and of the norm columns of ``second_layer.S21mBasis.field_grids`` (those of
    ``amplitude_scalings``'s s21m columns, mapped back through ``s21m_assemble``)."""
    final = np.log(np.asarray(amplitude_scalings(np.e, fog=fog)["s21m"]))
    nlam, nnorm = (sum(S21_SIZES[:6]), sum(S21_SIZES[6:])) if not fog else (
        sum(S21_SIZES[:6]) + 2 * sum(S21_SIZES[i] for i in (0, 1, 3, 4)), S21_SIZES[6] + S21_SIZES[7] + 2 * S21_SIZES[6])
    source = s21m_assemble(np.arange(nlam, dtype=float), nlam + np.arange(nnorm, dtype=float), fog).astype(int)
    powers = np.empty(nlam + nnorm)
    powers[source] = final
    return powers[:nlam], powers[nlam:]
