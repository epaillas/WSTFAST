#!/usr/bin/env python
"""Exact tree-level noise of the first-layer WST moduli, from perturbation-theory fields on a mesh.

The stochastic part of U = |delta * psi_{j,l}| (what is left of P_UU after the part correlated with delta)

    s(k) = [P_UU - P_Ud^2 / P_dd] / <U>^2

is measured on delta = delta_1 + eps delta_2 + eps^2 delta_3, with the EdS SPT fields computed by the standard
recursion in real space. U is homogeneous in delta, so s depends on eps only, and the fit
s(eps) = s0 + a eps^2 + b eps^4 separates the Gaussian noise s0 (all chaos orders) from its exact tree-level
non-Gaussian correction a. Small eps keep the fit polynomial (the full SPT sum is not, at the mesh scale);
since every eps uses the same realization, the differences carry no sample variance. Each seed is also run with delta_1 -> -delta_1 (delta_3 flips, delta_2 does not),
which makes s even in eps realization by realization. The fields get the CIC window of the N-body meshes.

The result is compared, in units of the second-chaos Gaussian noise g(k) of the model, with the N-body
measurements and with the analytic tree-level correction of wstfast.theory.modulus_noise (whose fourth-chaos
term G4 belongs to s0, the rest to a). Use --no-window for a like-for-like comparison. Example:

    python scripts/feasibility/pt_modulus_noise.py --seeds 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from cosmoprimo import Cosmology
from wstfast.config import QUIJOTE_COSMOLOGY
from wstfast.data import load_measurement
from wstfast.measure import Lattice, PowerMultipoles, _irfft, _rfft, wavelet_modulus
from wstfast.theory.modulus_noise import ModulusNoise
from wstfast.theory.moduli import ModulusClustering
from wstfast.theory.perturbation import OneLoopMatter, log_interpolator

K_BANDS = ((0.0, 0.03), (0.03, 0.06), (0.06, 0.1))


class SPTFields:
    """delta_1, delta_2, delta_3 of one Gaussian realization (rfft arrays), EdS recursion in real space.

    With u_m = grad lap^{-1} theta_m: alpha(k1, k2) theta_m delta_l <-> div(delta_l u_m) and
    beta(k1, k2) theta_m theta_l <-> lap(u_m . u_l) / 2.
    """

    def __init__(self, lattice: Lattice, pk, boxsize: float, seed: int):
        n = lattice.n
        k = 2 * np.pi * np.fft.fftfreq(n)
        kz = 2 * np.pi * np.fft.rfftfreq(n)
        self.kvec = np.meshgrid(k, k, kz, indexing="ij", sparse=True)
        self.k2 = lattice.kmag.astype("f8") ** 2
        self.k2[0, 0, 0] = 1.0
        self.n = n
        white = _rfft(np.random.default_rng(seed).standard_normal((n,) * 3))
        kphys = lattice.kmag * (n / boxsize)
        amplitude = np.sqrt(np.where(kphys > 0, pk(np.maximum(kphys, 1e-6)), 0.0) * n**3 / boxsize**3)
        self.delta1 = white * amplitude
        self.delta2, theta2 = self._second(self.delta1)
        self.delta3 = self._third(self.delta1, self.delta2, theta2)

    def _real(self, fk):
        return _irfft(fk, self.n)

    def _velocity(self, theta):
        return [self._real(-1j * ki * theta / self.k2) for ki in self.kvec]

    def _alpha(self, theta, delta):
        u, d = self._velocity(theta), self._real(delta)
        return sum(1j * ki * _rfft(d * ui) for ki, ui in zip(self.kvec, u))

    def _beta(self, theta_a, theta_b):
        ua, ub = self._velocity(theta_a), self._velocity(theta_b)
        return -0.5 * self.k2 * _rfft(sum(a * b for a, b in zip(ua, ub)))

    def _second(self, delta1):
        a, b = self._alpha(delta1, delta1), self._beta(delta1, delta1)
        return (5 * a + 2 * b) / 7, (3 * a + 4 * b) / 7

    def _third(self, delta1, delta2, theta2):
        a = self._alpha(delta1, delta2) + self._alpha(theta2, delta1)
        return (7 * a + 4 * self._beta(delta1, theta2)) / 18


def cic_window(lattice: Lattice):
    n = lattice.n
    k = 2 * np.pi * np.fft.fftfreq(n)
    kz = 2 * np.pi * np.fft.rfftfreq(n)
    kx, ky, kz = np.meshgrid(k, k, kz, indexing="ij", sparse=True)
    return (np.sinc(kx / (2 * np.pi)) * np.sinc(ky / (2 * np.pi)) * np.sinc(kz / (2 * np.pi))) ** 2


def stochasticity(fk, lattice, spectra, harmonics, sigma, ell):
    """(s(k), r(k)): stochastic part of P_UU / <U>^2 and response P_Ud / (<U> P_dd)."""
    u = wavelet_modulus(fk, lattice, harmonics, sigma, ell)
    mean = u.mean(dtype=np.float64)
    fu = _rfft(u - mean)
    puu, pud, pdd = spectra(fu, fu)[0], spectra(fu, fk)[0], spectra(fk, fk)[0]
    return (puu - pud**2 / pdd) / mean**2, pud / (mean * pdd)


def band_means(k, values):
    return [float(np.mean(values[(k > lo) & (k < hi)])) for lo, hi in K_BANDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256"))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--scales", type=int, nargs="+", default=[4, 6], help="first-layer j of the data config")
    parser.add_argument("--ells", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--kmax", type=float, default=0.1)
    parser.add_argument("--no-window", action="store_true", help="omit the CIC window (to compare with window-free theory)")
    parser.add_argument("--output", type=Path, default=Path("outputs/feasibility/pt_modulus_noise.json"))
    args = parser.parse_args()

    files = sorted((args.data_dir / "real").glob("wst_r*.npz"))
    first = load_measurement(files[0])
    config, meta = first["config"], first["metadata"]
    boxsize, nmesh, z = meta["boxsize"], meta["nmesh"], meta["redshift"]
    klin = np.geomspace(1e-4, 10.0, 1024)
    cosmo = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY)
    pklin = cosmo.get_fourier().pk_interpolator(of="delta_m")(klin, z=z)
    pk_np = lambda x: np.exp(np.interp(np.log(x), np.log(klin), np.log(pklin)))  # noqa: E731
    pk = log_interpolator(jnp.asarray(klin), jnp.asarray(pklin))

    lattice = Lattice(nmesh)
    spectra = PowerMultipoles(lattice, boxsize, kmax=args.kmax, ells=(0,))
    k = spectra.k
    window = 1.0 if args.no_window else cic_window(lattice)
    harmonics = {ell: lattice.harmonics(ell) for ell in args.ells}
    fields = [(j, ell) for j in args.scales for ell in args.ells]
    kloop = np.geomspace(1e-4, 3.0, 400)
    loop = jnp.asarray(OneLoopMatter(kloop)(pk))
    dpk = lambda x: jnp.interp(jnp.log(x), jnp.log(jnp.asarray(kloop)), loop)  # noqa: E731

    EPS = tuple(args.eps)
    # PT measurements: s[field][seed, sign, eps, k].
    s = {field: np.zeros((args.seeds, 2, len(EPS), k.size)) for field in fields}
    for seed in range(args.seeds):
        spt = SPTFields(lattice, pk_np, boxsize, seed=seed)
        for isign, sign in enumerate((1.0, -1.0)):
            for ieps, eps in enumerate(EPS):
                fk = window * (sign * spt.delta1 + eps * spt.delta2 + sign * eps**2 * spt.delta3)
                for j, ell in fields:
                    sigma = config.sigma(j) / config.cellsize
                    s[j, ell][seed, isign, ieps] = stochasticity(fk, lattice, spectra, harmonics[ell], sigma, ell)[0]
        print(f"seed {seed} done", flush=True)

    # N-body: same estimator on the stored spectra.
    nbody = {}
    stored = [np.load(path) for path in files]
    sel = slice(0, k.size)
    for j, ell in fields:
        umean = np.array([x["Umean"][j, ell] for x in stored])[:, None]
        puu = np.array([x["PUU"][j, ell, 0, sel] for x in stored])
        pud = np.array([x["PUd"][j, ell, 0, sel] for x in stored])
        pdd = np.array([x["Pdd"][0, sel] for x in stored])
        nbody[j, ell] = np.mean((puu - pud**2 / pdd) / umean**2, axis=0)

    design = np.stack([np.ones(len(EPS)), np.square(EPS), np.power(EPS, 4)], axis=1)
    results = dict(eps=list(EPS), k_bands=K_BANDS, window=not args.no_window, seeds=args.seeds, nbody_realizations=len(files), fields={})
    print("\nstochastic noise / second-chaos Gaussian g(k), k bands " + ", ".join(f"[{lo},{hi}]" for lo, hi in K_BANDS))
    for j, ell in fields:
        sigma = config.sigma(j)
        g = np.asarray(ModulusClustering(k, sigma, ell)(pk)[2])
        noise = ModulusNoise(sigma, ell)
        corrections = {name: np.asarray(noise.interpolate(value, k)) for name, value in noise.terms(pk, dpk).items()}
        tree = sum(value for name, value in corrections.items() if name != "G4")
        even = s[j, ell].mean(axis=1)  # antithetic average over the sign of delta_1
        coeffs = np.linalg.lstsq(design, even.mean(axis=0), rcond=None)[0]  # (3, nk)
        per_seed = np.array([np.linalg.lstsq(design, x, rcond=None)[0][1] for x in even])
        rows = {
            "N-body": band_means(k, nbody[j, ell] / g),
            "PT Gaussian s0": band_means(k, coeffs[0] / g),
            "PT tree a": band_means(k, coeffs[1] / g),
            "PT tree a, seed scatter": [float(np.std([band_means(k, x / g)[i] for x in per_seed]) / np.sqrt(args.seeds))
                                        for i in range(len(K_BANDS))],
            "PT s0 + a": band_means(k, (coeffs[0] + coeffs[1]) / g),
            "model tree": band_means(k, tree / g),
            "model G4 (vs s0 - 1)": band_means(k, corrections["G4"] / g),
        }
        results["fields"][f"j{j}_l{ell}"] = dict(sigma=sigma, k=k.tolist(), g=g.tolist(), s0=coeffs[0].tolist(),
                                                 a=coeffs[1].tolist(), a_seeds=per_seed.tolist(), **rows)
        print(f"\nsigma = {sigma:.1f} Mpc/h, l = {ell}")
        for name, values in rows.items():
            print(f"  {name:24s} " + "  ".join(f"{v:7.3f}" for v in values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
