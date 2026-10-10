"""BAO infrared resummation of the redshift-space matter spectrum, vendored from dsc-model.

Copied from the desilike fork used by dsc-model (/Users/epaillas/code/desilike, branch ``codex/ds-refactor-jax``,
``theories/galaxy_clustering/density_split_eft/ir.py``, untracked upstream); only the LinearSpectrum import is
changed. Original docstring:

Shared BAO infrared resummation for the density-split EFT.

One wiggle/no-wiggle split, one anisotropic displacement and one matched
NLO prescription are used for every pair (docs/density_split_eft.md §10):

    tree   K_A K_B [P_nw + e^-D (1 + D) P_w]       (one-loop)
           K_A K_B [P_nw + e^-D P_w]               (tree+IR)
    loop   L[P_L] + (e^-D - 1) W
    ct     C[P_nw + e^-D P_w]

with the wiggle loop W = [L22[P_L] - L22[P_nw]] + L13[P_nw] P_w(k)/P_nw(k).
As in CLASS-PT, only the external leg of the 13 diagram carries the damped
wiggle; its internal loop (a smooth, non-BAO displacement integral) stays at
fixed order. Expanding e^-D recovers the unresummed NLO spectrum.

with D = k² {[1 + f mu² (2 + f)] Sigma² + f² mu² (mu² - 1) dSigma²}.
The split ports the CLASS-PT DST (Hamann et al. 2010) with its native
k range, grid size and harmonic cut; Sigma² and dSigma² use its k_S, r_BAO
(sound horizon at drag) and log-trapezoid quadrature. This is a
CLASS-PT-convention leading-displacement prescription, not a claim of
numerical identity with native resummed CLASS-PT multipoles.
Units are h/Mpc and (Mpc/h)^3 throughout the public interface.
"""
from dataclasses import asdict, dataclass

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import CubicSpline

IR_SCHEMES = ('none', 'bao')


@dataclass(frozen=True)
class IRConvention:
    scheme: str = 'bao'
    split: str = 'classpt-dst-hamann2010-v1'
    k_s: float = .2
    nir: int = 65536
    k_min_physical: float = 7.e-5
    k_max_physical: float = 7.
    harmonic_cut: tuple = (120, 240)
    nint: int = 500
    r_bao: str = 'rs_drag'
    displacement: str = 'anisotropic-leading-v1'
    version: int = 1

    def __post_init__(self):
        if self.scheme != 'bao' or self.split != 'classpt-dst-hamann2010-v1' or self.version != 1:
            raise ValueError('unsupported IR convention')
        object.__setattr__(self, 'harmonic_cut', tuple(int(x) for x in self.harmonic_cut))

    def as_dict(self):
        result = asdict(self)
        result['harmonic_cut'] = list(self.harmonic_cut)
        return result


def validate_scheme(ir_scheme):
    if ir_scheme not in IR_SCHEMES:
        raise ValueError(f'ir_scheme must be one of {IR_SCHEMES}')
    return ir_scheme


def _nowiggle_physical(spectrum_physical, convention):
    """Literal port of CLASS-PT steps 1-3; returns (k, P_nw) on the DST grid, 1/Mpc."""
    nir = convention.nir
    nleft, nright = convention.harmonic_cut
    kgrid = convention.k_min_physical + np.arange(nir) * (
        (convention.k_max_physical - convention.k_min_physical) / (nir - 1.))
    logkp = np.log(kgrid) + np.log(spectrum_physical(kgrid))
    # Forward DST through a symmetric 4N real FFT (sign-convention free).
    buffer = np.zeros(4 * nir)
    values = np.where(np.arange(nir) % 2 == 0, 1., -1.) * logkp
    index = np.arange(nir)
    buffer[2 * index + 1] = values
    buffer[4 * nir - 2 * index - 1] = values
    out = np.fft.fft(buffer).real[nir - 1 - index]
    odd, even = out[0::2], out[1::2]
    throw = nright - nleft
    keep = np.concatenate([np.arange(nleft), np.arange(nleft + throw, nir // 2)])
    abscissa = keep + 1.
    full = np.arange(1, nir // 2 + 1, dtype=float)
    cmnew = np.empty(nir)
    cmnew[0::2] = CubicSpline(abscissa, odd[keep], bc_type='natural')(full)
    cmnew[1::2] = CubicSpline(abscissa, even[keep], bc_type='natural')(full)
    # Inverse transform, again through a symmetric 4N buffer.
    buffer = np.zeros(4 * nir)
    buffer[0] = .5 * cmnew[nir - 1]
    buffer[2 * nir] = -.5 * cmnew[nir - 1]
    i = np.arange(1, nir)
    half = .5 * cmnew[nir - 1 - i]
    buffer[i] = half
    buffer[4 * nir - i] = half
    buffer[2 * nir - i] = -half
    buffer[2 * nir + i] = -half
    transformed = np.fft.fft(buffer).real
    out2 = np.where(index % 2 == 0, 1., -1.) * transformed[2 * index + 1]
    return kgrid, np.exp(out2 / (2. * nir)) / kgrid


def nowiggle_spectrum(spectrum, h, convention=IRConvention()):
    """No-wiggle LinearSpectrum on the input spectrum's own grid.

    ``spectrum`` is a LinearSpectrum in h units. Outside the DST range the
    smooth spectrum equals the input, as in CLASS-PT.
    """
    from .eft_loop import LinearSpectrum
    h = float(h)
    physical = lambda q: spectrum(np.asarray(q) / h) / h**3
    kgrid, pnw = _nowiggle_physical(physical, convention)
    smooth = CubicSpline(kgrid, pnw, bc_type='natural')
    k_physical = spectrum.k * h
    inside = (k_physical >= kgrid[0]) & (k_physical <= kgrid[-1])
    power = np.array(spectrum.power, copy=True)
    power[inside] = smooth(k_physical[inside]) * h**3
    if not np.isfinite(power).all() or np.any(power <= 0.):
        raise ValueError('no-wiggle split returned non-finite or non-positive power')
    return LinearSpectrum(spectrum.k, power, low_slope=spectrum.low_slope), dict(
        kgrid=kgrid, power=pnw, spline=smooth)


def displacement_moments(split, r_bao_physical, h, convention=IRConvention()):
    """(Sigma², dSigma²) in (Mpc/h)², with CLASS-PT's quadrature and k_S."""
    h = float(h)
    nint = convention.nint
    kmin = convention.k_min_physical
    ks = convention.k_s * h
    q = kmin * np.exp(np.arange(nint + 1) * np.log(ks / kmin) / nint)
    p = split['spline'](q)
    x = q * float(r_bao_physical)
    s, c = np.sin(x), np.cos(x)
    first = p * (1. - 3. * s / x + 6. * (s / x**3 - c / x**2))
    second = -p * (3. * c * x + (-3. + x**2) * s) / x**3
    dlog = np.diff(np.log(q))
    trapezoid = lambda y: np.sum(dlog * (q[1:] * y[1:] + q[:-1] * y[:-1]) / 2.)
    sigma2 = trapezoid(first) / (6. * np.pi**2) * h**2
    dsigma2 = trapezoid(second) / (2. * np.pi**2) * h**2
    if not np.isfinite([sigma2, dsigma2]).all() or sigma2 <= 0.:
        raise ValueError('invalid BAO displacement moments')
    return float(sigma2), float(dsigma2)


def split_linear(spectrum, h, r_bao_physical, convention=IRConvention()):
    """Return (no-wiggle LinearSpectrum, sigma2, dsigma2, provenance)."""
    nowiggle, split = nowiggle_spectrum(spectrum, h, convention)
    sigma2, dsigma2 = displacement_moments(split, r_bao_physical, h, convention)
    provenance = dict(convention.as_dict(), r_bao_physical_mpc=float(r_bao_physical),
                      r_bao_mpc_over_h=float(r_bao_physical) * float(h),
                      sigma2=sigma2, dsigma2=dsigma2)
    return nowiggle, sigma2, dsigma2, provenance


def damping(k, mu, f, sigma2, dsigma2, rsd=True):
    """D(k, mu) with shape (nk, nmu); k may be an (nk, nmu) grid, e.g. AP-mapped."""
    k, mu = jnp.asarray(k), jnp.asarray(mu)
    f = f if rsd else 0.
    mu2 = mu**2
    angular = (1. + f * mu2 * (2. + f)) * sigma2 + f**2 * mu2 * (mu2 - 1.) * dsigma2
    return (k[:, None]**2 if k.ndim == 1 else k**2) * angular[None, :]


def ir_linear(power_linear, power_nowiggle, D, nlo=False):
    """P_nw + e^-D (1 + nlo D) P_w with shape (nk, nmu); spectra may be (nk,) or (nk, nmu)."""
    pl, pnw = jnp.asarray(power_linear), jnp.asarray(power_nowiggle)
    if pl.ndim == 1:
        pl, pnw = pl[:, None], pnw[:, None]
    factor = jnp.exp(-D) * (1. + D) if nlo else jnp.exp(-D)
    return pnw + factor * (pl - pnw)


def wiggle_loop(subtracted_22, subtracted_22_nowiggle, renormalized_13_nowiggle,
                power_linear, power_nowiggle):
    """W on (nk, ..., npair) arrays: Delta L22 + L13[P_nw] P_w/P_nw(k)."""
    ratio = np.asarray(power_linear) / np.asarray(power_nowiggle) - 1.
    shape = (-1,) + (1,) * (np.ndim(renormalized_13_nowiggle) - 1)
    return (np.asarray(subtracted_22) - np.asarray(subtracted_22_nowiggle)
            + np.asarray(renormalized_13_nowiggle) * ratio.reshape(shape))


def ir_loop(loop, wiggle, D):
    """L + (e^-D - 1) W on (..., nk, nmu) angular arrays."""
    return loop + (jnp.exp(-D) - 1.) * wiggle
