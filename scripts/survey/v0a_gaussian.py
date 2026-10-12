"""V0a of docs/survey_plan.md: the survey S1m estimator on Gaussian fields with a known spectrum.

Geometry: a cone of half-angle ``--cone`` degrees and comoving shell ``--rmin``..``--rmax`` [Mpc/h] seen by an observer
outside a periodic box, with a DESI-LRG-like radial density nbar(r). The signal is a Gaussian field with the Kaiser
spectrum (b + f mu^2)^2 P_L(k) along z (plane parallel: the local line of sight is V1), and the noise is white with
variance 1 / (nbar V_cell) per cell. No perturbation theory enters, so every residual is the estimator's or the
survey model's.

Test A (edges). Uniform noise. The masked field (zero outside the cone) is compared with the unmasked field averaged
over the same, eroded, cells: the paired difference is the edge effect alone, as a function of the erosion
c (cells farther than c sigma_j from the edge).

Test B (varying noise). Noise with the radial nbar(r), FKP weights. The measurement is compared with three
predictions of the mean, all from the exact lattice variance of the signal lambda_s (per |m|):
  * exact: the Gaussian moment at each cell with the cell's noise variance lambda_N,m(x) = (h_m^2 * sigma_N^2)(x),
    averaged with the weights (only edge effects of the signal are missing);
  * local: lambda_N,m(x) ~ lambda_N,m^unit / nbar(x), the histogram model M1 of the plan (no gradient of nbar
    inside the wavelet);
  * effective: one noise level, the weighted mean of lambda_N,m(x) (what a box model at an effective nbar gives).
Biases are in units of the scatter of one survey.

    python scripts/survey/v0a_gaussian.py --realizations 16 --output outputs/survey/v0a
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from wstfast import WSTConfig
from wstfast.measure import Lattice, _irfft, _rfft
from wstfast.survey import erode, measure_wst_survey


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nmesh", type=int, default=256)
    parser.add_argument("--boxsize", type=float, default=2000.0)
    parser.add_argument("--cone", type=float, default=20.0, help="half-angle of the footprint [deg]")
    parser.add_argument("--rmin", type=float, default=1100.0)
    parser.add_argument("--rmax", type=float, default=2300.0)
    parser.add_argument("--observer-depth", type=float, default=900.0, help="observer at z = -depth [Mpc/h]")
    parser.add_argument("--nbar-peak", type=float, default=6e-4)
    parser.add_argument("--nbar-uniform", type=float, default=3e-4, help="nbar of test A")
    parser.add_argument("--bias", type=float, default=2.0)
    parser.add_argument("--growth-rate", type=float, default=0.75)
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--erosion", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0])
    parser.add_argument("--erosion-b", type=float, default=2.0, help="erosion of test B")
    parser.add_argument("--p0", type=float, default=1e4, help="FKP P0 [(Mpc/h)^3]")
    parser.add_argument("--realizations", type=int, default=16)
    parser.add_argument("--realizations-a", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/survey/v0a"))
    return parser.parse_args()


def linear_power():
    from cosmoprimo.fiducial import DESI

    return DESI().get_fourier().pk_interpolator().to_1d(z=0.5)


def chi_moment(lam_a, lam_b, q):
    """E (X_a^2 + X_b^2)^{q/2} for independent Gaussians of variances lam_a, lam_b (lam_b None: one component)."""
    p = q / 2
    if lam_b is None:
        return (2 * lam_a) ** p * np.exp(gammaln((1 + q) / 2)) / np.sqrt(np.pi)
    phi = (np.arange(64) + 0.5) * np.pi / 128  # (X, Y) = R (cos, sin): R^2 ~ chi^2_2, angle uniform
    c2 = np.cos(phi)[:, None] ** 2
    return 2**p * np.exp(gammaln(1 + p)) * np.mean((lam_a * c2 + lam_b * (1 - c2)) ** p, axis=0)


class Setup:
    def __init__(self, args):
        self.args = args
        n, box = args.nmesh, args.boxsize
        self.cell = box / n
        self.config = WSTConfig(J=3, L=args.L, sigma0=25.0 / self.cell, step=2**0.5, cellsize=self.cell,
                                min_dj=10)  # sigma_j = 25, 35, 50, 71 Mpc/h; no second layer
        self.lattice = Lattice(n)
        x = (np.arange(n) + 0.0) * self.cell
        observer = np.array([box / 2, box / 2, -args.observer_depth])
        gx, gy, gz = np.meshgrid(x - observer[0], x - observer[1], x - observer[2], indexing="ij")
        r = np.sqrt(gx**2 + gy**2 + gz**2)
        cos = gz / r
        self.support = (r > args.rmin) & (r < args.rmax) & (cos > np.cos(np.radians(args.cone)))
        rmid, width = 0.5 * (args.rmin + args.rmax), 0.3 * (args.rmax - args.rmin)
        self.nbar = np.where(self.support, args.nbar_peak * np.exp(-0.5 * ((r - rmid) / width) ** 2), 0.0)
        x = self.nbar * args.p0
        self.weights = np.where(self.support, (x / (1 + x)) ** 2, 0.0)
        volume = self.support.sum() * self.cell**3
        print(f"footprint {volume / 1e9:.2f} (Gpc/h)^3, nbar {self.nbar[self.support].min():.2e}"
              f"..{self.nbar[self.support].max():.2e}", flush=True)
        # Signal spectrum per cell: P_s(k, mu) / V_cell (so that var(delta) = int P d^3k / (2 pi)^3).
        kphys = self.lattice.kmag / self.cell
        pk = linear_power()(np.clip(kphys, 1e-4, None))
        kaiser = (args.bias + args.growth_rate * self.lattice.mu**2) ** 2
        self.pcell = np.where(kphys > 0, kaiser * pk / self.cell**3, 0.0)

    def scales(self):
        return [(j, self.config.sigma0 * self.config.step**j) for j in range(self.config.J + 1)]

    def filters(self, ell, sigma):
        """Fourier filters H_m (rfft lattice) of the components m = -l..l."""
        phase = (-1j) ** ell
        radial = self.lattice.radial(sigma, ell)
        return [phase * radial * y for y in self.lattice.harmonics(ell)]

    def signal_variances(self):
        """Exact lattice variance of each component of a stationary field with spectrum pcell: {(j, l): (2l+1,)}."""
        mult = self.lattice.multiplicity
        n3 = self.args.nmesh**3
        out = {}
        for j, sigma in self.scales():
            for ell in range(self.args.L + 1):
                out[j, ell] = np.array([np.sum(mult * self.pcell * np.abs(h) ** 2) / n3 for h in self.filters(ell, sigma)])
        return out

    def noise_variance_maps(self, sigma2, index):
        """lambda_N,m(x) = (h_m^2 * sigma_N^2)(x) on the cells ``index``, and the local approximation sum h_m^2."""
        n = self.args.nmesh
        fs = _rfft(sigma2.astype(np.float32))
        maps, unit = {}, {}
        for j, sigma in self.scales():
            for ell in range(self.args.L + 1):
                rows, sums = [], []
                for h in self.filters(ell, sigma):
                    kernel2 = _irfft(h, n) ** 2
                    sums.append(kernel2.sum(dtype=np.float64))
                    rows.append(_irfft(_rfft(kernel2) * fs, n).ravel()[index])
                maps[j, ell], unit[j, ell] = np.array(rows), np.array(sums)
        return maps, unit

    def gaussian_field(self, seed, nbar):
        rng = np.random.default_rng(seed)
        n = self.args.nmesh
        signal = _irfft(_rfft(rng.standard_normal((n,) * 3).astype(np.float32)) * np.sqrt(self.pcell), n)
        sigma_n = np.sqrt(np.where(nbar > 0, 1.0 / (np.where(nbar > 0, nbar, 1.0) * self.cell**3), 0.0))
        return (signal + sigma_n * rng.standard_normal((n,) * 3)).astype(np.float32)


def block_prediction(lam, ell, q):
    """S1m (|m| = 0..l) from component variances lam (2l+1, ...) ordered m = -l..l."""
    out = [chi_moment(lam[ell], None, q)]
    out += [chi_moment(lam[ell - m], lam[ell + m], q) for m in range(1, ell + 1)]
    return np.array(out)


def test_a(setup: Setup):
    args = setup.args
    nbar = np.full(setup.support.shape, args.nbar_uniform)
    masked, reference = [], []
    for r in range(args.realizations_a):
        t0 = time.time()
        field = setup.gaussian_field(1000 + r, nbar)
        masked.append([measure_wst_survey(field, setup.config, support=setup.support, qs=[args.q], los="z",
                                          erosion=c, lattice=setup.lattice)["S1m"][0] for c in args.erosion])
        reference.append([measure_wst_survey(field, setup.config, moment_support=setup.support, qs=[args.q], los="z",
                                             erosion=c, lattice=setup.lattice)["S1m"][0] for c in args.erosion])
        print(f"A {r}: {time.time() - t0:.0f} s", flush=True)
    return np.array(masked), np.array(reference)  # (nreal, nerosion, J+1, L+1, L+1)


def test_b(setup: Setup):
    args = setup.args
    masked = []
    for r in range(args.realizations):
        t0 = time.time()
        field = setup.gaussian_field(2000 + r, setup.nbar)
        masked.append(measure_wst_survey(field, setup.config, support=setup.support, weights=setup.weights,
                                         qs=[args.q], los="z", erosion=args.erosion_b, lattice=setup.lattice)["S1m"][0])
        print(f"B {r}: {time.time() - t0:.0f} s", flush=True)
    masked = np.array(masked)

    index = np.flatnonzero(setup.support)
    sigma2 = np.where(setup.nbar > 0, 1.0 / (np.where(setup.nbar > 0, setup.nbar, 1.0) * setup.cell**3), 0.0)
    maps, unit = setup.noise_variance_maps(sigma2, index)
    lam_s = setup.signal_variances()
    shape = masked.shape[1:]
    predictions = {name: np.full(shape, np.nan) for name in ("exact", "local", "effective")}
    for j, sigma in setup.scales():
        w = np.where(erode(setup.support, args.erosion_b * sigma).ravel()[index], setup.weights.ravel()[index], 0.0)
        w /= w.sum()
        for ell in range(args.L + 1):
            exact = lam_s[j, ell][:, None] + maps[j, ell]
            local = lam_s[j, ell][:, None] + unit[j, ell][:, None] * sigma2.ravel()[index][None, :]
            effective = lam_s[j, ell][:, None] + (maps[j, ell] @ w)[:, None]
            for name, lam in (("exact", exact), ("local", local), ("effective", effective)):
                values = block_prediction(lam, ell, args.q)
                predictions[name][j, ell, :ell + 1] = values[:, 0] if lam.shape[1] == 1 else values @ w
    return masked, predictions


def main():
    args = get_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup = Setup(args)
    sigmas = [s * setup.cell for _, s in setup.scales()]

    masked, reference = test_a(setup)
    np.savez(args.output / "test_a.npz", masked=masked, reference=reference, erosion=args.erosion, sigma=sigmas)
    masked_b, predictions = test_b(setup)
    np.savez(args.output / "test_b.npz", masked=masked_b, sigma=sigmas, **predictions)

    summary = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    # A: the edge bias per erosion, in units of the scatter of one survey (masked measurement), over the blocks.
    scatter_a = masked.std(axis=0, ddof=1)
    edge = (masked - reference).mean(axis=0) / scatter_a
    print("\nTest A: edge bias / sigma_survey, max over l, |m| (rows: erosion; columns: sigma_j)")
    summary["A"] = {}
    for i, c in enumerate(args.erosion):
        row = np.nanmax(np.abs(edge[i]), axis=(1, 2))
        frac = np.nanmax(np.abs((masked - reference).mean(axis=0)[i] / reference.mean(axis=0)[i]), axis=(1, 2))
        summary["A"][f"c={c:g}"] = {"max_bias_over_sigma": row.tolist(), "max_fractional": frac.tolist()}
        print(f"  c = {c:g}: " + "  ".join(f"{s:5.1f}: {b:6.2f} ({100 * f:.2f}%)" for s, b, f in zip(sigmas, row, frac)))
    # B: the prediction errors, in units of the scatter of one survey.
    mean, scatter = masked_b.mean(axis=0), masked_b.std(axis=0, ddof=1)
    error = scatter / np.sqrt(len(masked_b))
    print(f"\nTest B (erosion {args.erosion_b:g}): (mean - prediction) / sigma_survey, max over l, |m|;"
          f" error of the mean {1 / np.sqrt(len(masked_b)):.2f}")
    summary["B"] = {}
    for name, pred in predictions.items():
        z = (mean - pred) / scatter
        row = np.nanmax(np.abs(z), axis=(1, 2))
        chi2 = np.nansum(((mean - pred) / error) ** 2)
        summary["B"][name] = {"max_bias_over_sigma": row.tolist(), "chi2_diag": float(chi2),
                              "nblocks": int(np.isfinite(pred).sum())}
        print(f"  {name:9s}: " + "  ".join(f"{s:5.1f}: {b:5.2f}" for s, b in zip(sigmas, row))
              + f"   chi2_diag = {chi2:.0f} / {np.isfinite(pred).sum()}")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
