"""Alcock-Paczynski (AP) distortions, applied at assembly (after emulation), as in dsc-model.

Ported from dsc-model's ``density_split_eft/ap.py`` (desilike fork, branch ``codex/ds-refactor-jax``). An observed
wavevector (k, mu), measured with the fiducial distance-redshift relation, corresponds to the true-frame

    k' = k s(mu),  mu' = mu / (q_par s(mu)),  s(mu) = sqrt(mu^2 / q_par^2 + (1 - mu^2) / q_perp^2),

with q_par = D_H(z) / D_H,fid(z) and q_perp = D_M(z) / D_M,fid(z) (both in Mpc/h). Every spectrum, the shot noise
included, is divided by q_par q_perp^2. The emulated basis stays true-frame and cosmology-only; the WST filters,
the multipole weights and the mass-assignment window act in the observed frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

_NZ = 32  # Gauss-Legendre nodes of the comoving-distance integral


@dataclass(frozen=True)
class APFiducial:
    """Flat LCDM fiducial whose distance-redshift relation defines the observed coordinates."""

    h: float
    omega_b: float
    omega_cdm: float
    omega_r: float

    @property
    def omega_m(self):
        return self.omega_b + self.omega_cdm


#: Massless Quijote fiducial with the radiation of the CLASS solve (photons and massless neutrinos), as in dsc-model.
QUIJOTE_FIDUCIAL = APFiducial(h=0.6711, omega_b=0.02206838529, omega_cdm=0.120925743885, omega_r=4.183853747134498e-05)


def _expansion(z, h, omega_m, omega_r):
    a3, a4 = (1.0 + z) ** 3, (1.0 + z) ** 4
    return jnp.sqrt((omega_r * a4 + omega_m * a3) / h**2 + 1.0 - (omega_m + omega_r) / h**2)


def ap_ratios(z, h, omega_b, omega_cdm, fiducial: APFiducial = QUIJOTE_FIDUCIAL):
    """(q_par, q_perp) of a flat trial cosmology at redshift z; JAX-traceable in (h, omega_b, omega_cdm)."""
    nodes, weights = np.polynomial.legendre.leggauss(_NZ)
    zz, ww = 0.5 * z * (nodes + 1.0), 0.5 * z * weights

    def distances(h, omega_m):  # D_H and D_M in units of c / (100 km/s/Mpc), i.e. Mpc/h up to a constant
        return 1.0 / _expansion(z, h, omega_m, fiducial.omega_r), jnp.sum(ww / _expansion(zz, h, omega_m, fiducial.omega_r))

    dh, dm = distances(h, omega_b + omega_cdm)
    dh_fid, dm_fid = distances(fiducial.h, fiducial.omega_m)
    return dh / dh_fid, dm / dm_fid


def observed_to_true(k, mu, qpar, qperp):
    """True-frame k', mu' of observed modes (any matching shapes) and the volume factor 1 / (q_par q_perp^2)."""
    scale = jnp.sqrt(mu**2 / qpar**2 + (1.0 - mu**2) / qperp**2)
    return k * scale, mu / (qpar * scale), 1.0 / (qpar * qperp**2)
