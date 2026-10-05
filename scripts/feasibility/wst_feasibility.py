#!/usr/bin/env python
"""Feasibility diagnostics for a perturbative (EFT) model of the 3D solid-harmonic WST.

For each Quijote snapshot this measures, on the N-body density contrast and on a
phase-randomized (Gaussian) copy with identical |delta(k)|:

* WST coefficients S0, S1(j, l), S2(j1, j2, l) with the kymatio/ACM conventions
  (Fourier-space solid harmonic wavelets, sigma_j = sigma0 2^j cells, modulus
  summed over m, |.|^q only at the end of each layer, l1 = l2). The per-l filter
  normalisation differs from kymatio by a constant, which rescales coefficients
  but does not affect any ratio used here.
* Covariance eigenvalues and normalised cumulant invariants (K4, K33, K3v) of
  every first- and second-layer wavelet vector, with the exact Gaussian
  expectation E_G|X|^q and its next-to-leading Edgeworth correction.
* The quadratic-chaos approximation of the second layer.
* In real space, the large-scale cross-power of each first-layer modulus field
  with delta (its linear response bias, which vanishes for a Gaussian field).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import scipy.fft as sfft
from scipy.special import gammaln, roots_hermitenorm, sph_harm_y

WORKERS = os.cpu_count()
SNAPSHOT_ROOT = Path("/Users/epaillas/data/quijote/snapshots/fiducial")


# ---------------------------------------------------------------------------
# Quijote snapshot -> density contrast
# ---------------------------------------------------------------------------


def read_header(files):
    import h5py
    import hdf5plugin  # noqa: F401  (Quijote compression filter)

    with h5py.File(files[0], "r") as handle:
        attrs = handle["Header"].attrs
        header = {
            "boxsize": float(attrs["BoxSize"]) / 1e3,
            "redshift": float(attrs["Redshift"]),
            "omega_m": float(attrs["Omega0"]),
            "omega_l": float(attrs["OmegaLambda"]),
        }
    header["hubble_z"] = 100.0 * np.sqrt(
        header["omega_m"] * (1 + header["redshift"]) ** 3 + header["omega_l"]
    )
    return header


def paint_snapshot(snapdir: Path, nmesh: int, rsd: bool, los: int = 2):
    """CIC-paint CDM particles (optionally shifted to redshift space) to a mesh."""
    import h5py
    import hdf5plugin  # noqa: F401

    files = sorted(snapdir.glob("snap_*.hdf5"), key=lambda p: int(p.stem.rsplit(".", 1)[1]))
    header = read_header(files)
    boxsize = header["boxsize"]
    a = 1.0 / (1.0 + header["redshift"])
    rsd_factor = np.sqrt(a) * (1.0 + header["redshift"]) / header["hubble_z"]
    grid = np.zeros(nmesh**3, dtype=np.float64)
    for filename in files:
        with h5py.File(filename, "r") as handle:
            pos = handle["PartType1/Coordinates"][:].astype(np.float64) / 1e3
            if rsd:
                pos[:, los] += handle["PartType1/Velocities"][:, los].astype(np.float64) * rsd_factor
        pos = np.remainder(pos, boxsize) * (nmesh / boxsize)
        i0 = np.floor(pos).astype(np.int64)
        d = pos - i0
        del pos
        i0 %= nmesh
        i1 = (i0 + 1) % nmesh
        for cx in (0, 1):
            ix = i1[:, 0] if cx else i0[:, 0]
            wx = d[:, 0] if cx else 1.0 - d[:, 0]
            for cy in (0, 1):
                iy = i1[:, 1] if cy else i0[:, 1]
                wy = d[:, 1] if cy else 1.0 - d[:, 1]
                for cz in (0, 1):
                    iz = i1[:, 2] if cz else i0[:, 2]
                    wz = d[:, 2] if cz else 1.0 - d[:, 2]
                    idx = (ix * nmesh + iy) * nmesh + iz
                    grid += np.bincount(idx, weights=wx * wy * wz, minlength=nmesh**3)
        del i0, i1, d
    grid = grid.reshape((nmesh,) * 3)
    delta = (grid / grid.mean() - 1.0).astype(np.float32)
    return delta, header


# ---------------------------------------------------------------------------
# Solid harmonic wavelets on the rfft lattice
# ---------------------------------------------------------------------------


class Lattice:
    def __init__(self, nmesh: int):
        self.n = nmesh
        k = (2 * np.pi * np.fft.fftfreq(nmesh)).astype(np.float32)
        kz = (2 * np.pi * np.fft.rfftfreq(nmesh)).astype(np.float32)
        kx, ky, kz = np.meshgrid(k, k, kz, indexing="ij")
        self.kmag = np.sqrt(kx**2 + ky**2 + kz**2)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_theta = np.where(self.kmag > 0, kz / self.kmag, 1.0)
        self.theta = np.arccos(np.clip(cos_theta, -1, 1)).astype(np.float64)
        self.phi = np.arctan2(ky, kx).astype(np.float64)
        # Hermitian multiplicity of each rfft mode (for lattice power sums).
        mult = np.full(self.kmag.shape, 2.0, dtype=np.float32)
        mult[..., 0] = 1.0
        if nmesh % 2 == 0:
            mult[..., -1] = 1.0
        self.mult = mult
        self.order = np.argsort(self.kmag, axis=None)
        self.ksorted = self.kmag.reshape(-1)[self.order]

    def real_harmonics(self, ell: int):
        """Real spherical harmonics times sqrt(4 pi/(2l+1)): sum_m Y^2 = 1."""
        norm = np.sqrt(4 * np.pi / (2 * ell + 1))
        out = []
        for m in range(-ell, ell + 1):
            ylm = sph_harm_y(ell, abs(m), self.theta, self.phi)
            if m == 0:
                y = ylm.real
            elif m > 0:
                y = np.sqrt(2) * (-1) ** m * ylm.real
            else:
                y = np.sqrt(2) * (-1) ** m * ylm.imag
            out.append((norm * y).astype(np.float32))
        return out

    def radial(self, sigma: float, ell: int):
        x = sigma * self.kmag
        return (x**ell * np.exp(-0.5 * x**2)).astype(np.float32)


def irfft(arr, n):
    return sfft.irfftn(arr, s=(n, n, n), workers=WORKERS)


def rfft(arr):
    return sfft.rfftn(arr, workers=WORKERS)


# ---------------------------------------------------------------------------
# Gaussian expectations and Edgeworth corrections
# ---------------------------------------------------------------------------

_T = np.exp(np.linspace(-40.0, 40.0, 16001))


def gaussian_norm_moment(eigenvalues, q):
    """E|X|^q for zero-mean Gaussian X with covariance eigenvalues (0<q<2)."""
    lam = np.asarray(eigenvalues, dtype=np.float64)
    a = q / 2
    laplace = np.exp(-0.5 * np.log1p(2 * _T[:, None] * lam[None, :]).sum(axis=1))
    integrand = (1.0 - laplace) * _T ** (-a)  # dt t^{-1-a} = d(log t) t^{-a}
    log_t = np.log(_T)
    return a / np.exp(gammaln(1 - a)) * np.trapezoid(integrand, log_t)


def chi_moment(n, s2, q):
    """E|X|^q for an isotropic n-dimensional Gaussian with per-component variance s2."""
    return (2 * s2) ** (q / 2) * np.exp(gammaln((n + q) / 2) - gammaln(n / 2))


_HX, _HW = roots_hermitenorm(80)
_HW = _HW / _HW.sum()


def gaussian_abs_moment_with_mean(mean, var, q):
    y = mean + np.sqrt(var) * _HX
    return float(np.sum(_HW * np.abs(y) ** q))


def edgeworth_nlo(n, k4, k33, k3v, q):
    """Relative NLO correction to E|X|^q for an O(n)-invariant function."""
    c4 = q * (q - 2) / (8 * n * (n + 2))
    c6 = q * (q - 2) * (q - 4) / (72 * n * (n + 2) * (n + 4))
    return c4 * k4, c6 * (6 * k33 + 9 * k3v)


def vector_stats(fields, third: bool):
    """Covariance and normalised cumulant invariants of a wavelet vector field."""
    n = len(fields)
    flat = [f.reshape(-1) for f in fields]
    npts = flat[0].size
    mu = np.array([f.mean(dtype=np.float64) for f in flat])
    cov = np.empty((n, n))
    for a in range(n):
        for b in range(a, n):
            cov[a, b] = cov[b, a] = np.dot(flat[a], flat[b]) / npts - mu[a] * mu[b]
    z = np.zeros_like(flat[0])
    for f in flat:
        z += f * f
    var_z = z.var(dtype=np.float64)
    s2 = np.trace(cov) / n
    k4 = (var_z - 2 * np.sum(cov**2)) / s2**2
    k33 = k3v = 0.0
    if third:
        kap = np.zeros((n, n, n))
        for a, b in combinations_with_replacement(range(n), 2):
            prod = flat[a] * flat[b]
            for c in range(b, n):
                val = np.dot(prod, flat[c]) / npts
                for (i, j, k) in {(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)}:
                    kap[i, j, k] = val
        kap /= s2**1.5
        k33 = float(np.sum(kap**2))
        k3v = float(np.sum(np.einsum("aac->c", kap) ** 2))
    return {
        "eig": np.linalg.eigvalsh(cov).tolist(),
        "s2": float(s2),
        "mean": mu.tolist(),
        "K4": float(k4),
        "K33": k33,
        "K3v": k3v,
    }


# ---------------------------------------------------------------------------
# WST with diagnostics
# ---------------------------------------------------------------------------


def lattice_power_weights(lat, fk, boxsize, sigma, ell):
    """k-percentiles (h/Mpc) of the lattice contribution to the filtered variance."""
    w = (lat.radial(sigma, ell) ** 2) * (np.abs(fk) ** 2) * lat.mult
    scale = lat.n / boxsize
    cum = np.cumsum(w.reshape(-1)[lat.order], dtype=np.float64)
    cum /= cum[-1]
    ks = lat.ksorted * scale
    k2 = float(np.sum(w * lat.kmag**2, dtype=np.float64) / np.sum(w, dtype=np.float64)) * scale**2
    return {
        "k50": float(ks[np.searchsorted(cum, 0.5)]),
        "k90": float(ks[np.searchsorted(cum, 0.9)]),
        "k2_mean": k2,
    }


def cross_bias(lat, fu, fd, boxsize, kmax=0.05):
    kphys = lat.kmag * (lat.n / boxsize)
    sel = (kphys > 0) & (kphys < kmax)
    w = lat.mult[sel]
    num = np.sum(w * (fu[sel] * np.conj(fd[sel])).real, dtype=np.float64)
    den = np.sum(w * np.abs(fd[sel]) ** 2, dtype=np.float64)
    return float(num / den)


def run_wst(delta, lat, boxsize, J=4, L=4, sigma0=0.8, q=0.8, label="", response=False,
            third_second_layer=True):
    n = delta.shape[0]
    fk = rfft(delta)
    out = {"S0": float(np.mean(np.abs(delta) ** q, dtype=np.float64)), "S1": {}, "S2": {}}
    t0 = time.time()
    for ell in range(L + 1):
        harm = lat.real_harmonics(ell)
        phase = (-1j) ** ell
        nvec = 2 * ell + 1
        even = ell % 2 == 0
        for j1 in range(J + 1):
            s1 = sigma0 * 2**j1
            rad = lat.radial(s1, ell)
            xs = [irfft(fk * (phase * rad * y), n) for y in harm]
            stats = vector_stats(xs, third=even)
            z = np.zeros_like(xs[0])
            for x in xs:
                z += x * x
            u1 = np.sqrt(z)
            entry = {
                "S1": float(np.mean(u1**q, dtype=np.float64)),
                "U1_mean": float(u1.mean(dtype=np.float64)),
                "stats": stats,
                "S1_gauss": float(gaussian_norm_moment(stats["eig"], q)),
                "S1_chi_iso": float(chi_moment(nvec, stats["s2"], q)),
                "weights": lattice_power_weights(lat, fk, boxsize, s1, ell),
            }
            c4, c6 = edgeworth_nlo(nvec, stats["K4"], stats["K33"], stats["K3v"], q)
            entry["edgeworth_c4"] = float(c4)
            entry["edgeworth_c33"] = float(c6)
            # Second-chaos projection of U1 (exact for a Gaussian field with this covariance).
            # Diagonalise once so that the projection is sum_m beta_m (x_m^2 - lam_m).
            fu = rfft(u1 - u1.mean(dtype=np.float64))
            if response:
                entry["bias_U1_delta"] = cross_bias(lat, fu, fk, boxsize)
                entry["bias_Z_delta"] = cross_bias(lat, rfft(z - z.mean(dtype=np.float64)), fk, boxsize)
            # beta from the measured joint moments (regression of U1 on z): for an
            # isotropic Gaussian this is the exact second-chaos coefficient.
            beta = float(np.cov(u1.reshape(-1)[::7], z.reshape(-1)[::7])[0, 1]
                         / z.reshape(-1)[::7].var())
            fz = rfft(z - z.mean(dtype=np.float64))
            del xs
            out["S1"][f"{j1},{ell}"] = entry
            for j2 in range(j1 + 1, J + 1):
                s2sig = sigma0 * 2**j2
                rad2 = lat.radial(s2sig, ell)
                if ell == 0:
                    # Gaussian low-pass: U1 mean is retained.
                    y = irfft(fu * rad2, n) + u1.mean(dtype=np.float64)
                    ys = [y]
                    yq = irfft(fz * rad2, n) * beta
                    s2_meas = float(np.mean(np.abs(y) ** q, dtype=np.float64))
                    var_y = float(y.var(dtype=np.float64))
                    mean_y = float(y.mean(dtype=np.float64))
                    e2 = {
                        "S2": s2_meas,
                        "Y_mean": mean_y,
                        "Y_var": var_y,
                        "S2_gauss_Y": gaussian_abs_moment_with_mean(mean_y, var_y, q),
                        "Y_var_quadchaos": float(yq.var(dtype=np.float64)),
                    }
                    st2 = vector_stats([y - mean_y], third=True)
                    e2["stats"] = st2
                    del yq, y
                else:
                    ys = [irfft(fu * (phase * rad2 * yh), n) for yh in harm]
                    yqs = [irfft(fz * (phase * rad2 * yh), n) * beta for yh in harm]
                    st2 = vector_stats(ys, third=even and third_second_layer)
                    zz = np.zeros_like(ys[0])
                    for yy in ys:
                        zz += yy * yy
                    s2_meas = float(np.mean(zz ** (q / 2), dtype=np.float64))
                    var_q = sum(float(v.var(dtype=np.float64)) for v in yqs)
                    e2 = {
                        "S2": s2_meas,
                        "Y_var": float(sum(st2["eig"])),
                        "stats": st2,
                        "S2_gauss_Y": float(gaussian_norm_moment(st2["eig"], q)),
                        "Y_var_quadchaos": var_q,
                    }
                    c4y, c6y = edgeworth_nlo(nvec, st2["K4"], st2["K33"], st2["K3v"], q)
                    e2["edgeworth_c4"] = float(c4y)
                    e2["edgeworth_c33"] = float(c6y)
                    del yqs, zz
                del ys
                out["S2"][f"{j1},{j2},{ell}"] = e2
            del u1, z, fu, fz
        print(f"  [{label}] l={ell} done ({time.time() - t0:.0f}s)", flush=True)
    return out


def phase_randomize(delta, seed):
    rng = np.random.default_rng(seed)
    fk = rfft(delta)
    noise = rfft(rng.standard_normal(delta.shape, dtype=np.float32))
    amp = np.abs(fk)
    with np.errstate(invalid="ignore", divide="ignore"):
        ph = np.where(np.abs(noise) > 0, noise / np.abs(noise), 0)
    out = irfft(amp * ph, delta.shape[0]).astype(np.float32)
    return out - out.mean(dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", nargs="+", default=["0"])
    parser.add_argument("--snapnum", default="003")
    parser.add_argument("--nmesh", type=int, default=256)
    parser.add_argument("--spaces", nargs="+", default=["real", "rsd"])
    parser.add_argument("--output", type=Path, default=Path("outputs/feasibility"))
    parser.add_argument("--J", type=int, default=4)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--sigma0", type=float, default=0.8)
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--tag", default="", help="suffix for output files (e.g. a scale configuration)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lat = Lattice(args.nmesh)
    for real in args.realizations:
        for space in args.spaces:
            fn = args.output / f"wst_diag_r{real}_{space}_n{args.nmesh}{args.tag}.json"
            if fn.exists():
                print(f"skip {fn}")
                continue
            t0 = time.time()
            snapdir = SNAPSHOT_ROOT / real / f"snapdir_{args.snapnum}"
            delta, header = paint_snapshot(snapdir, args.nmesh, rsd=(space == "rsd"))
            print(f"r{real} {space}: painted in {time.time() - t0:.0f}s, var={delta.var():.3f}", flush=True)
            kw = dict(J=args.J, L=args.L, sigma0=args.sigma0, q=args.q)
            res = {
                "realization": real,
                "space": space,
                "header": header,
                "config": dict(kw, nmesh=args.nmesh, cellsize=header["boxsize"] / args.nmesh),
                "nbody": run_wst(delta, lat, header["boxsize"], label=f"r{real} {space} nbody",
                                 response=(space == "real"), **kw),
            }
            gauss = phase_randomize(delta, seed=1000 + int(real))
            del delta
            res["gauss"] = run_wst(gauss, lat, header["boxsize"], label=f"r{real} {space} gauss",
                                   response=(space == "real"), **kw)
            del gauss
            fn.write_text(json.dumps(res))
            print(f"wrote {fn} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
