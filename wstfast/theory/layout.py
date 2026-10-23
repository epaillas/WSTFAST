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
- S21m with one-loop short modes (``second_layer.LoopModulusSpectra``; ``S21_LOOP_PIECES``): the short-mode spectrum
  is the one-loop P_s of the grid, sum_i w_i u_i(p) over the pieces i = the degree-2 bias monomials of tree + loop
  (weights the monomials of beta), the counterterm shapes (c0, c2, c4), the constant shot noise (N (1 + alpha0)) and
  the k^2 mu^2 stochastic shape (N alpha2). The response r, the noise g and the norms n are then bilinear / linear in
  the pieces, and their columns are the symmetric products of the pieces (``product_blocks``), the response ones with
  the stochastic pieces times Z1(K) (degree 1 in beta):

      lambda_2 = [sum_{i<=j} w_i w_j (R_ij + (1 + a) G_ij)] / (sum_i w_i n_i)^2,

  with no Fingers-of-God orders (the counterterms replace c_v in the short modes); ``s21m_layout`` tells the two S21m
  layouts apart by their number of columns.
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


#: Pieces (name, degree in beta of the weight) of the one-loop short-mode spectrum of the S21m model: the degree-2
#: monomials of tree + loop (weights ``bias.monomials(beta, 2)``), the counterterm shapes (weights c0, c2, c4), the
#: constant shot noise (weight N (1 + alpha0)) and the k^2 mu^2 stochastic shape (weight N alpha2).
S21_LOOP_PIECES = (("beta", 2), ("c0", 0), ("c2", 0), ("c4", 0), ("N", 0), ("alpha2", 0))


#: Pieces of the modulus response r: those of the spectrum, the stochastic ones times Z1(K) (degree 1).
S21_LOOP_RESPONSE_PIECES = (("beta", 2), ("c0", 0), ("c2", 0), ("c4", 0), ("N", 1), ("alpha2", 1))


def product_blocks(pieces):
    """Blocks ((i, j), degree, size) of the symmetric products of ``pieces`` ((name, degree) each), i <= j in the order of
    the pieces: the columns of a quantity bilinear in the pieces (the cross terms carry the factor 2)."""
    return [((i, j), di + dj, len(monomial_exponents(di + dj)))
            for i, (_, di) in enumerate(pieces) for j, (_, dj) in enumerate(pieces) if j >= i]


def s21_loop_blocks():
    """Column blocks of the one-loop S21m terms, in order: the response products R_ij (times P_L(K)), the noise products
    G_ij and the norms n_i; each a list of (key, degree, size), key = (i, j) or i into the pieces."""
    return dict(response=product_blocks(S21_LOOP_RESPONSE_PIECES), noise=product_blocks(S21_LOOP_PIECES),
                norms=[(i, d, len(monomial_exponents(d))) for i, (_, d) in enumerate(S21_LOOP_PIECES)])


#: Number of columns of the one-loop S21m terms per entry (without AP terms): 266 + 160 + 20.
S21_LOOP_COLUMNS = sum(size for blocks in s21_loop_blocks().values() for _, _, size in blocks)


def s21m_layout(ncol: int):
    """(loop, fog, nalpha) of S21m terms from their number of columns: whether the one-loop short-mode layout, whether
    the tree layout holds the Fingers-of-God orders, and the number of AP-polynomial terms."""
    tree = sum(S21_SIZES)
    for nalpha in AP_TERMS:
        if ncol == S21_LOOP_COLUMNS * nalpha:
            return True, False, nalpha
        if ncol == tree * nalpha:
            return False, False, nalpha
        if ncol == (tree + fog_sizes()["s21m"]) * nalpha:
            return False, True, nalpha
    raise ValueError(f"{ncol} S21m columns match no layout")


def fog_sizes(cubic: bool = False):
    """Columns appended by ``fog`` (orders 1 and 2 of the counterterm) to the kappa, skewness and S21m outputs."""
    kappa = KAPPA_SIZES[0] + (sum(CUBIC_KAPPA_SIZES) if cubic else 0)
    s21 = sum(S21_SIZES[i] for i in FOG_S21_BLOCKS)
    return dict(kappa=2 * kappa, skewness=2 * SKEW_SIZES[0], s21m=2 * s21)


def split(values, sizes):
    """Split the last axis of ``values`` into blocks of ``sizes``."""
    edges = np.cumsum((0,) + tuple(sizes))
    return [values[..., a:b] for a, b in zip(edges[:-1], edges[1:])]


#: Powers of A_s of the pieces of the one-loop short-mode spectrum (the loop part of the beta piece is A_s^2; the
#: tree part sets the power).
S21_LOOP_PIECE_POWERS = (1, 1, 1, 1, 0, 0)


def s21_loop_powers():
    """Powers of A_s / A_s,fid of the one-loop S21m columns (``s21_loop_blocks``): the products of the piece powers, the
    response ones plus one for P_L(K)."""
    blocks = s21_loop_blocks()
    p = S21_LOOP_PIECE_POWERS
    return ([(p[i] + p[j] + 1, size) for (i, j), _, size in blocks["response"]]
            + [(p[i] + p[j], size) for (i, j), _, size in blocks["noise"]] + [(p[i], size) for i, _, size in blocks["norms"]])


def amplitude_scalings(ratio, cubic: bool = False, fog: bool = False, ap_terms: int = 1, s21_loop: bool = False):
    """Powers of A_s / A_s,fid of every basis output (grid rows, kappa / skewness / S21m columns), so that a
    Taylor emulator is nearly exact in logA. ``cubic``: the kappa columns include CUBIC_KAPPA_SIZES (P^3, like the
    tree trispectrum). ``fog``: with the counterterm orders of ``fog_sizes`` (the powers of the blocks they expand).
    ``s21_loop``: the S21m columns of the one-loop short-mode layout (no Fingers-of-God orders)."""
    import jax.numpy as jnp

    xp = np if isinstance(ratio, (float, np.floating)) else jnp
    powers = lambda counts: xp.concatenate([xp.full((n,), 1.0) * ratio**p for p, n in counts])  # noqa: E731
    clustering = [(3, KAPPA_SIZES[0])] + ([(3, sum(CUBIC_KAPPA_SIZES))] if cubic else [])
    s21 = list(zip((3, 2, 1, 2, 1, 0, 1, 0), S21_SIZES))
    s21_fog = [s21[i] for i in FOG_S21_BLOCKS]
    s21_columns = s21_loop_powers() if s21_loop else s21 + (2 * s21_fog if fog else [])
    # with ``ap_terms`` AP-polynomial terms the columns repeat per term (alpha-major)
    return dict(grid=powers([(1, len(TREE)), (2, len(LOOP)), (1, 3)]),
                kappa=powers(ap_terms * (list(zip((3, 2, 1, 0), KAPPA_SIZES))
                                         + ([(3, sum(CUBIC_KAPPA_SIZES))] if cubic else []) + (2 * clustering if fog else []))),
                skewness=powers(ap_terms * (list(zip((2, 1, 0), SKEW_SIZES)) + (2 * [(2, SKEW_SIZES[0])] if fog else []))),
                s21m=powers(ap_terms * s21_columns))


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
    """Whether basis outputs hold the Fingers-of-God counterterm orders (from their number of columns; S21m terms of the
    one-loop layout hold none)."""
    if kappa is not None:
        return basis_layout(kappa.shape[-1])[1]
    return s21m_layout(s21m.shape[-1])[1]


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


def s21m_lambda_mask(fog: bool, loop: bool = False):
    """Boolean mask of the lambda_2 columns (the second-layer sums, which carry the AP volume factor) in the S21m column
    layout of ``s21m_assemble``; the others are the first-layer norms."""
    if loop:
        blocks = s21_loop_blocks()
        nlam = sum(size for _, _, size in blocks["response"] + blocks["noise"])
        nnorm = sum(size for _, _, size in blocks["norms"])
    else:
        nlam = sum(S21_SIZES[:6]) + (2 * sum(S21_SIZES[i] for i in (0, 1, 3, 4)) if fog else 0)
        nnorm = sum(S21_SIZES[6:]) + (2 * S21_SIZES[6] if fog else 0)
    return s21m_assemble(np.ones((1, nlam)), np.zeros((1, nnorm)), fog)[0] > 0.5


def s21m_assemble(lam, norms, fog: bool):
    """S21m terms in the column layout of ``second_layer.S21mBasis`` from the projected grid columns ``lam`` (..., ncol) and
    the first-layer norms of each entry (..., nnorm): [lam | norms] (also the one-loop layout), or with ``fog`` the base
    blocks followed by the orders 1 and 2 of both. NumPy or JAX."""
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


def s21m_field_powers(fog: bool, loop: bool = False):
    """Powers of A_s / A_s,fid of the grid columns and of the norm columns of ``second_layer.S21mBasis.field_grids`` (those of
    ``amplitude_scalings``'s s21m columns, mapped back through ``s21m_assemble``)."""
    final = np.log(np.asarray(amplitude_scalings(np.e, fog=fog, s21_loop=loop)["s21m"]))
    if loop:
        nnorm = sum(size for _, _, size in s21_loop_blocks()["norms"])
        return final[:-nnorm], final[-nnorm:]
    nlam, nnorm = (sum(S21_SIZES[:6]), sum(S21_SIZES[6:])) if not fog else (
        sum(S21_SIZES[:6]) + 2 * sum(S21_SIZES[i] for i in (0, 1, 3, 4)), S21_SIZES[6] + S21_SIZES[7] + 2 * S21_SIZES[6])
    source = s21m_assemble(np.arange(nlam, dtype=float), nlam + np.arange(nnorm, dtype=float), fog).astype(int)
    powers = np.empty(nlam + nnorm)
    powers[source] = final
    return powers[:nlam], powers[nlam:]
