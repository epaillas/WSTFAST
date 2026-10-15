"""Window of the survey WST in the DR1 LRG geometry of one redshift bin, from Gaussian mocks on the randoms.

The geometry comes from the public complete Abacus mocks (``SurveyGeometry.from_angular_mask``: completeness of the
randoms, n(r) of the data per photometric region). In the DR1 clustering catalogues each random takes the redshift of
a data object of its photometric region (N, or S across both caps; checked through TARGETID_DATA), so the FKP field
has zero mean in every thin shell of every region (``RadialIntegralConstraint``). Two tests, on Gaussian Kaiser mocks
with the local line of sight:

``--test ric``: S1m and S21m measured with and without the RIC (shell widths ``--width``); the paired difference in
units of the scatter of one survey (per cap).

``--test lambda``: the window at the level of the component variances (``LambdaWindow``). The edge transfer
T_m(x) = f(x) c_m(f) is the coverage f of the wavelet by the support (exact for white noise) times a correction
calibrated in bins of f on paired signal-only mocks at a fiducial spectrum, with the RIC on the survey side. The
survey S1m of any spectrum is then the weighted mean over the support of the Gaussian moment at
lambda_m(x) = T_m(x) lambda_m^box + lambda_N(x), with the exact lattice lambda^box and noise maps. It is tested on
noisy, constrained mocks of several spectra; "coverage" sets c = 1 and "none" T = 1.

``--test s21m``: the second-layer window. The transfer of the second-layer variances, T2 = f2 c2 (coverage of the
sigma_j2 wavelet times a correction calibrated in coverage bins on paired noisy mocks with the RIC), gives the
survey S2m as R2 x the box S2m, R2 = <T2^{q/2}> over the support. The survey second layer subtracts the mean of the
first-layer modulus per bin of coverage and nbar (``--layer2-mean binned``), which removes the deterministic edge
profile of U; ``global`` is the support mean. Tested per mock (paired with its own box, which
cancels cosmic variance) on several spectra: S21m = S2m / S1m with the measured survey S1m, so only the
second-layer window is tested; "coverage" sets c2 = 1 and "none" R2 = 1.

``--test products``: the window files of the likelihood (``survey.SurveyWindow``, one per cap): the first-layer
histograms of the transfer (``--correction``) and noise, delta of every S1m block from paired mocks
(``--paired-mocks``), K of every S21m entry from the S21m test mocks (``--s21m-mocks``), and the per-mock ratios of
both (the calibration covariance, ``survey.window_correction``).

``--test measure``: the WST of the complete Abacus mocks (``--measure-mocks``, with ``--measure-randoms`` random files
each) on the frozen geometry and estimator (FKP field, FKP weights, local line of sight, binned second-layer mean), as
``fit_rsd.py`` survey data: ``<output>/<cap>/wst_r<mock>.npz`` (``data.save_measurement``, with the shot noise
1 / nbar_eff of the geometry).

``--test covariance``: the analytic covariance (``theory.covariance_rsd.RSDWSTCovariance``: 2nd + 4th chaos) of the
Gaussian mocks' ln S1m and ln S21m, a box covariance with the variance of every entry rescaled to its effective volume
on the cap (``SurveyWindow.veff_block``), against the sample covariance of the survey mocks (``--s1m-mocks``,
``--s21m-mocks``). For a Gaussian field the modulus has no response (r = 0), only its Gaussian-chaos noise.

``--test halos``: Quijote halo boxes cut to the geometry (``survey.box_to_survey``: tiled, RSD along the local line of
sight, thinned to nbar(x); ``--nbar-scale`` for a geometry denser than the halos), the RIC on the survey side, paired
with their periodic parent box (NGC from box 2i, SGC from box 2i + 1 of ``--halo-dir``, i = ``--seed-offset`` +
0..``--realizations``-1). Per mock and cap: the survey and box S1m and S2m, saved for ``--test products``
(``partial_halos.npz``: ``<cap>``, ``<cap>_box``; ``<cap>_measured``, ``<cap>_none`` of S21m) and the survey
measurement as ``fit_rsd.py`` survey data (``<output>/data/<cap>/wst_r<i>.npz``).

``--test vector``: the baseline data vector (S1m at sigma = 25-71 Mpc/h, l <= 4; S21m = S2m / S1m at (25, 70.7), l >= 1)
before and after the window, as the forward model defines it: the same fiducial mocks measured unmasked (the box)
and on the footprint with the RIC. Saves the means and a figure.

    python scripts/survey/dr1_window.py --test ric --realizations 10 --width 2.5 5 10 --output outputs/survey/dr1_ric
    python scripts/survey/dr1_window.py --test lambda --calibrate 12 --realizations 6 \\
        --output outputs/survey/dr1_lambda_window

With ``--device cuda`` (or ``cpu``) the WST, the window and the variances run with torch (``wstfast.measure_torch``),
on a GPU (or multithreaded on the CPU); the mocks and the RIC stay in numpy. At NERSC, with the public tree as the catalogue root, one test per GPU:

    ROOT=/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/mocks/AbacusSummit/dark/v4.2
    sbatch -A <account> -C gpu -q regular -N 1 --gpus-per-node=4 -t 02:00:00 --wrap "
        CUDA_VISIBLE_DEVICES=0 python scripts/survey/dr1_window.py --device cuda --catalogue-root $ROOT \\
            --test ric --realizations 20 --width 2.5 5 10 --output outputs/survey/dr1_ric &
        CUDA_VISIBLE_DEVICES=1 python scripts/survey/dr1_window.py --device cuda --catalogue-root $ROOT \\
            --test lambda --calibrate 32 --realizations 16 --output outputs/survey/dr1_lambda_window &
        wait"

(build the geometry cache once first, without the GPUs, so that the two runs do not both write it).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wstfast import WSTConfig
from wstfast import survey
from wstfast.measure import Lattice
from wstfast.survey import (GaussianMocks, LambdaWindow, RadialIntegralConstraint, SurveyGeometry, measure_wst_survey,
                            noise_variance_maps, s1m_gaussian, signal_variances)

VARIANTS = {  # bias, growth rate, amplitude, tilt of the linear spectrum
    "fiducial": (2.0, 0.76, 1.0, 0.0),
    "low_bf": (1.6, 0.60, 1.0, 0.0),
    "amplitude": (2.0, 0.76, 1.21, 0.0),
    "tilt": (2.0, 0.76, 1.0, -0.06),
}


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", choices=("ric", "lambda", "s21m", "vector", "products", "measure", "covariance",
                                           "halos"), required=True)
    parser.add_argument("--zrange", type=float, nargs=2, default=[0.4, 0.6])
    parser.add_argument("--cellsize", type=float, default=10.0)
    parser.add_argument("--mocks", type=int, nargs="+", default=[0, 1, 2], help="data for n(r)")
    parser.add_argument("--randoms", type=int, nargs="+", default=[0, 1], help="random files for the mask")
    parser.add_argument("--width", type=float, nargs="+", default=[5.0], help="RIC shell widths [Mpc/h]")
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--layer2-mean", choices=("binned", "global"), default="binned",
                        help="mean subtracted from the first-layer modulus before the second layer of the survey "
                             "measurement: per bin of first-layer coverage and nbar (survey.second_layer_bins), or "
                             "over the whole support")
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--calibrate", type=int, default=12, help="signal-only mocks for the lambda window")
    parser.add_argument("--realizations", type=int, default=10, help="test mocks (per variant for lambda)")
    parser.add_argument("--catalogue-root", type=Path, default=survey.DESI_ROOT,
                        help="directory with mock<i>/LRG_complete_* catalogues")
    parser.add_argument("--device", default="numpy", help="numpy, or a torch device (cpu, cuda)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                        help="spectra of the test mocks (lambda, s21m)")
    parser.add_argument("--seed-offset", type=int, default=0, help="offset of the test-mock seeds (independent runs)")
    parser.add_argument("--paired", action="store_true",
                        help="lambda: also measure each test mock unmasked on the same cells (its box), so that the "
                             "S-level correction is calibrated paired, without the mocks' cosmic variance")
    parser.add_argument("--window-files", type=Path, nargs="*", default=[],
                        help="covariance: SurveyWindow file of each cap (in the order NGC, SGC)")
    parser.add_argument("--s1m-mocks", type=Path, nargs="*", default=[],
                        help="covariance: --test lambda results.npz / partial_fiducial.npz (survey S1m of mocks)")
    parser.add_argument("--covariance-box", type=float, nargs=2, default=[1000.0, 128], metavar=("BOXSIZE", "NMESH"),
                        help="covariance: box of the analytic mode sums")
    parser.add_argument("--correction", type=Path, default=None,
                        help="products: correction.npz of a --test lambda run (first-layer correction c_m(f))")
    parser.add_argument("--paired-mocks", type=Path, nargs="*", default=[],
                        help="products: partial_fiducial.npz of --test lambda --paired runs (delta of S1m)")
    parser.add_argument("--s21m-mocks", type=Path, nargs="*", default=[],
                        help="products: results.npz / partial_fiducial.npz of --test s21m runs (K of S21m)")
    parser.add_argument("--measure-mocks", type=int, nargs="+", default=[0],
                        help="measure: the complete Abacus mocks to measure (fit_rsd.py survey data)")
    parser.add_argument("--measure-randoms", type=int, nargs="+", default=list(range(18)),
                        help="measure: random files of each mock")
    parser.add_argument("--nbar-scale", type=float, default=1.0,
                        help="multiply the geometry's nbar(x) (e.g. 0.5 for halo mocks sparser than the survey)")
    parser.add_argument("--halo-dir", type=Path, default=Path("data/quijote/halos_m13_z0.5"),
                        help="halos: r<i>.npz (positions, displacements [Mpc/h]) of Quijote halo boxes of 1 Gpc/h")
    parser.add_argument("--geometry-dir", type=Path, default=Path("outputs/survey/dr1_geometry"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_geometries(args) -> dict[str, SurveyGeometry]:
    """NGC and SGC geometries of the bin on one mesh size (so that the caps share one Lattice), cached."""
    args.geometry_dir.mkdir(parents=True, exist_ok=True)
    path = args.geometry_dir / f"geometry_z{args.zrange[0]}-{args.zrange[1]}_cell{args.cellsize:g}.npz"
    if path.exists():
        f = np.load(path)
        return {cap: SurveyGeometry(f[f"{cap}_support"], f[f"{cap}_nbar"], float(f[f"{cap}_boxsize"]),
                                    f[f"{cap}_offset"], np.zeros(3), region=f[f"{cap}_region"])
                for cap in survey.DESI_CAPS}
    masks, data = {}, []
    for cap in survey.DESI_CAPS:
        ran = [survey.read_desi_catalogue(survey.desi_catalogue_path(args.mocks[0], cap, i, root=args.catalogue_root),
                                          columns=("RA", "DEC")) for i in args.randoms]
        masks[cap] = survey.AngularMask.from_randoms(np.concatenate([r["RA"] for r in ran]),
                                                     np.concatenate([r["DEC"] for r in ran]),
                                                     density=survey.DESI_RANDOM_DENSITY * len(ran))
        for mock in args.mocks:
            data.append(survey.read_desi_catalogue(survey.desi_catalogue_path(mock, cap, root=args.catalogue_root),
                                                   args.zrange)
                        | dict(cap=cap, mock=mock))
        print(f"{cap}: mask {np.degrees(np.degrees(masks[cap].area())):.0f} deg^2", flush=True)
    nbar = survey.desi_radial_density(data, masks, args.zrange)
    geometries = {cap: SurveyGeometry.from_angular_mask(cap, args.zrange, masks[cap], nbar, cellsize=args.cellsize)
                  for cap in survey.DESI_CAPS}
    nmesh = max(g.nmesh for g in geometries.values())
    geometries = {cap: g if g.nmesh == nmesh else
                  SurveyGeometry.from_angular_mask(cap, args.zrange, masks[cap], nbar, cellsize=args.cellsize,
                                                   nmesh=nmesh)
                  for cap, g in geometries.items()}
    np.savez(path, **{f"{cap}_{key}": getattr(g, key) for cap, g in geometries.items()
                      for key in ("support", "nbar", "boxsize", "offset", "region")},
             nbar_edges=nbar.edges, nbar_profiles=np.array(nbar.profiles))
    return geometries


def linear_power(z):
    from cosmoprimo.fiducial import DESI

    return DESI().get_fourier().pk_interpolator().to_1d(z=z)


def spectrum(pk, amplitude, tilt):
    return lambda k: amplitude * pk(k) * (np.asarray(k) / 0.05) ** tilt


def mocks_of(geometries, pk, variant, signal_only=False):
    b, f, a, t = VARIANTS[variant]
    return {cap: GaussianMocks(g.nmesh, g.cellsize, spectrum(pk, a, t), b, f,
                               np.where(g.support, 1e12, 0.0) if signal_only else g.nbar, g.support,
                               weights=g.weights, observer=g.observer, offset=g.offset)
            for cap, g in geometries.items()}


def run_ric(args, geometries, setups, pk):
    constraints = {w: RadialIntegralConstraint(*geometries.values(), width=w) for w in args.width}
    mocks = mocks_of(geometries, pk, "fiducial")
    keys = ("S1m", "S2m")
    results = {cap: {name: {key: [] for key in keys} for name in ["none"] + [f"ric{w:g}" for w in args.width]}
               for cap in geometries}
    for i in range(args.realizations):
        t0 = time.time()
        fields = {cap: mocks[cap](seed=[i, k]) for k, cap in enumerate(geometries)}
        variants = {"none": fields}
        for w, ric in constraints.items():
            variants[f"ric{w:g}"] = dict(zip(geometries, ric(*fields.values())))
        for name, meshes in variants.items():
            for cap, field in meshes.items():
                s = setups[cap]
                out = args.measure(field, s["config"], support=s["g"].support, weights=s["g"].weights,
                                   qs=[args.q], los=s["los"], lattice=s["lattice"], second_layer_bins=s["bins2"])
                for key in keys:
                    results[cap][name][key].append(out[key][0])
        print(f"realization {i}: {time.time() - t0:.0f} s", flush=True)
        np.savez(args.output / "measurements.npz",
                 **{f"{cap}_{name}_{key}": np.array(v) for cap, r in results.items() for name, rr in r.items()
                    for key, v in rr.items()})

    summary = {}
    for cap in geometries:
        for key in keys:
            base = np.array(results[cap]["none"][key])
            scatter = base.std(axis=0, ddof=1)
            for name in results[cap]:
                if name == "none":
                    continue
                diff = np.array(results[cap][name][key]) - base
                shift = diff.mean(axis=0) / scatter
                error = diff.std(axis=0, ddof=1) / np.sqrt(len(diff)) / scatter
                row = dict(max_shift=float(np.nanmax(np.abs(shift))), rms_shift=float(np.sqrt(np.nanmean(shift**2))),
                           max_error=float(np.nanmax(error)),
                           max_fractional=float(np.nanmax(np.abs(diff.mean(axis=0) / base.mean(axis=0)))),
                           per_scale=[float(np.nanmax(np.abs(shift[j]))) if np.isfinite(shift[j]).any() else None
                                      for j in range(shift.shape[0])], nblocks=int(np.isfinite(shift).sum()))
                summary[f"{cap}_{name}_{key}"] = row
                print(f"{cap} {name} {key}: |shift| max {row['max_shift']:.2f} rms {row['rms_shift']:.2f} sigma_survey"
                      f" (error <= {row['max_error']:.2f}), fractional <= {row['max_fractional']:.4f};"
                      f" per sigma_j1 {row['per_scale']}", flush=True)
    return summary


def run_lambda(args, geometries, setups, pk):
    caps = list(geometries)
    ric = RadialIntegralConstraint(*geometries.values(), width=args.width[0])
    windows = {cap: LambdaWindow(s["config"], s["g"].support, s["los"], s["lattice"]) for cap, s in setups.items()}
    path = args.output / "correction.npz"
    if path.exists():
        f = np.load(path)
        correction = {cap: {b: f[f"{cap}_{b[0]}_{b[1]}"] for b in windows[cap].blocks} for cap in caps}
    else:
        signal = mocks_of(geometries, pk, "fiducial", signal_only=True)
        for r in range(args.calibrate):
            t0 = time.time()
            fields = [signal[cap](seed=[100 + r, k]) for k, cap in enumerate(caps)]
            for cap, masked, field in zip(caps, ric(*fields), fields):
                windows[cap].add(masked, field)
            print(f"calibration {r}: {time.time() - t0:.0f} s", flush=True)
        correction = {cap: w.correction() for cap, w in windows.items()}
        np.savez(path, **{f"{cap}_{j}_{l}": c for cap, cc in correction.items() for (j, l), c in cc.items()})
        del signal
    transfer = {cap: {"window": w.transfer(correction[cap]), "coverage": w.transfer(False)}
                for cap, w in windows.items()}
    del windows

    results = {}
    for name in args.variants:
        b, f, a, t = VARIANTS[name]
        mocks = mocks_of(geometries, pk, name)
        measured = {cap: [] for cap in caps}
        boxes = {cap: [] for cap in caps}  # the same mock unmasked on the same cells: pairs the S-level calibration
        for r in range(args.realizations):
            t0 = time.time()
            fields = [mocks[cap](seed=[200 + args.seed_offset + r, k]) for k, cap in enumerate(caps)]
            for cap, survey, field in zip(caps, ric(*fields), fields):
                s = setups[cap]
                common = dict(weights=s["g"].weights, qs=[args.q], los=s["los"], lattice=s["lattice"])
                out = args.measure(survey, s["config"], support=s["g"].support, second_layer_bins=s["bins2"], **common)
                measured[cap].append(out["S1m"][0])
                if args.paired:
                    boxes[cap].append(args.measure(field, s["config"], moment_support=s["g"].support,
                                                   **common)["S1m"][0])
            print(f"{name} test {r}: {time.time() - t0:.0f} s", flush=True)
            np.savez(args.output / f"partial_{name}.npz", **{cap: np.array(v) for cap, v in measured.items()},
                     **{f"{cap}_box": np.array(v) for cap, v in boxes.items() if v})
        for cap in caps:
            s, g = setups[cap], geometries[cap]
            grid = getattr(s["lattice"], "numpy", s["lattice"])  # the numpy Lattice of a TorchLattice
            kaiser = (b + f * grid.mu**2) ** 2
            kphys = grid.kmag / g.cellsize
            pcell = np.where(kphys > 0, kaiser * spectrum(pk, a, t)(np.clip(kphys, 1e-4, None)) / g.cellsize**3, 0.0)
            lam = signal_variances(pcell, s["config"], s["lattice"])
            weights = g.weights.ravel()[np.flatnonzero(g.support)]
            results[name, cap] = dict(measured=np.array(measured[cap]))
            for model in ("window", "coverage"):
                results[name, cap][model] = s1m_gaussian(lam, s["config"], args.q, weights,
                                                         transfer=transfer[cap][model], lam_noise=s["noise"])
            results[name, cap]["none"] = s1m_gaussian(lam, s["config"], args.q, weights, lam_noise=s["noise"])
        np.savez(args.output / "results.npz", **{f"{n}_{c}_{k}": v for (n, c), d in results.items()
                                                  for k, v in d.items()})

    summary = {}
    for (name, cap), d in results.items():
        m = d["measured"]
        scatter = results["fiducial", cap]["measured"].std(axis=0, ddof=1)
        error = m.std(axis=0, ddof=1) / np.sqrt(len(m)) / scatter
        row = {"error_max": float(np.nanmax(error))}
        for model in ("window", "coverage", "none"):
            z = (m.mean(axis=0) - d[model]) / scatter
            row[model] = dict(max=float(np.nanmax(np.abs(z))), rms=float(np.sqrt(np.nanmean(z**2))),
                              chi2=float(np.nansum((z / error) ** 2)), nblocks=int(np.isfinite(z).sum()),
                              per_scale=[float(np.nanmax(np.abs(z[j]))) for j in range(z.shape[0])])
        summary[f"{name}_{cap}"] = row
        print(f"{name:9s} {cap}: window max {row['window']['max']:.2f} rms {row['window']['rms']:.2f}"
              f" chi2 {row['window']['chi2']:.0f}/{row['window']['nblocks']} | coverage only max"
              f" {row['coverage']['max']:.2f} | none max {row['none']['max']:.2f} | error <= {row['error_max']:.2f}"
              f" sigma_survey; window per sigma_j {row['window']['per_scale']}", flush=True)
    return summary


def run_s21m(args, geometries, setups, pk):
    caps = list(geometries)
    ric = RadialIntegralConstraint(*geometries.values(), width=args.width[0])
    windows = {cap: LambdaWindow(s["config"], s["g"].support, s["los"], s["lattice"], second_layer=True,
                                 second_layer_bins=s["bins2"])
               for cap, s in setups.items()}
    path = args.output / "correction2.npz"
    if path.exists():
        f = np.load(path)
        correction = {cap: {b: f[f"{cap}_" + "_".join(map(str, b))] for b in windows[cap].blocks2} for cap in caps}
    else:
        mocks = mocks_of(geometries, pk, "fiducial")
        for r in range(args.calibrate):
            t0 = time.time()
            fields = [mocks[cap](seed=[400 + r, k]) for k, cap in enumerate(caps)]
            for cap, survey, field in zip(caps, ric(*fields), fields):
                windows[cap].add(survey, field)
            print(f"calibration {r}: {time.time() - t0:.0f} s", flush=True)
        correction = {cap: w.correction2() for cap, w in windows.items()}
        np.savez(path, **{f"{cap}_" + "_".join(map(str, b)): c for cap, cc in correction.items()
                          for b, c in cc.items()})
    ratios = {}
    for cap, w in windows.items():
        weights = geometries[cap].weights.ravel()[w.index]
        ones = {b: np.ones_like(c) for b, c in correction[cap].items()}
        ratios[cap] = {"window": w.s2m_ratio(weights, args.q, correction[cap]),
                       "coverage": w.s2m_ratio(weights, args.q, ones)}
    del windows

    def s21m(s1m, s2m, ratio=None):
        return np.array([s2m[j1, j2, ell, m1, m2] * (1.0 if ratio is None else ratio[j1, j2, ell, m1][m2])
                         / s1m[j1, ell, m1] for j1, j2 in setups[caps[0]]["config"].second_layer_pairs()
                         for ell in range(args.L + 1) for m1 in range(ell + 1) for m2 in range(ell + 1)])

    results = {}
    for name in args.variants:
        mocks = mocks_of(geometries, pk, name)
        rows = {cap: {k: [] for k in ("measured", "window", "coverage", "none")} for cap in caps}
        for r in range(args.realizations):
            t0 = time.time()
            fields = [mocks[cap](seed=[500 + args.seed_offset + r, k]) for k, cap in enumerate(caps)]
            for cap, survey, field in zip(caps, ric(*fields), fields):
                s = setups[cap]
                common = dict(weights=s["g"].weights, qs=[args.q], los=s["los"], lattice=s["lattice"])
                masked = args.measure(survey, s["config"], support=s["g"].support, second_layer_bins=s["bins2"], **common)
                box = args.measure(field, s["config"], moment_support=s["g"].support, **common)
                s1m = masked["S1m"][0]
                rows[cap]["measured"].append(s21m(s1m, masked["S2m"][0]))
                rows[cap]["none"].append(s21m(s1m, box["S2m"][0]))
                for model in ("window", "coverage"):
                    rows[cap][model].append(s21m(s1m, box["S2m"][0], ratios[cap][model]))
            print(f"{name} test {r}: {time.time() - t0:.0f} s", flush=True)
            np.savez(args.output / f"partial_{name}.npz", **{f"{cap}_{k}": np.array(v) for cap, d in rows.items()
                                                             for k, v in d.items()})
        for cap in caps:
            results[name, cap] = {k: np.array(v) for k, v in rows[cap].items()}
        np.savez(args.output / "results.npz", **{f"{n}_{c}_{k}": v for (n, c), d in results.items()
                                                  for k, v in d.items()})
    summary = {}
    for (name, cap), d in results.items():
        scatter = results["fiducial", cap]["measured"].std(axis=0, ddof=1)
        row = {}
        for model in ("window", "coverage", "none"):
            diff = d["measured"] - d[model]  # paired: the residual of each mock against its own box
            z = diff.mean(axis=0) / scatter
            error = diff.std(axis=0, ddof=1) / np.sqrt(len(diff)) / scatter
            row[model] = dict(max=float(np.nanmax(np.abs(z))), rms=float(np.sqrt(np.nanmean(z**2))),
                              chi2=float(np.nansum((z / error) ** 2)), error_max=float(np.nanmax(error)),
                              nentries=int(np.isfinite(z).sum()))
        summary[f"{name}_{cap}"] = row
        print(f"{name:9s} {cap} S21m: window max {row['window']['max']:.2f} rms {row['window']['rms']:.2f}"
              f" chi2 {row['window']['chi2']:.0f}/{row['window']['nentries']} (error <= {row['window']['error_max']:.2f})"
              f" | coverage only max {row['coverage']['max']:.2f} | none max {row['none']['max']:.2f} sigma_survey",
              flush=True)
    return summary


def run_products(args, geometries, setups, pk):
    from wstfast.survey import SurveyWindow

    if args.correction is None:
        raise SystemExit("--test products needs --correction (correction.npz of a --test lambda run)")
    f = np.load(args.correction)
    b, fr, a, t = VARIANTS["fiducial"]
    summary = {}
    for cap, g in geometries.items():
        s = setups[cap]
        window = LambdaWindow(s["config"], g.support, s["los"], s["lattice"])
        correction = {blk: f[f"{cap}_{blk[0]}_{blk[1]}"] for blk in window.blocks}
        weights = g.weights.ravel()[window.index]
        product = SurveyWindow.build(window, correction, s["noise"], weights, q=args.q)
        product.veff = g.veff
        # box variances of the calibration mocks' spectrum (the delta ratio needs them only through T and lambda_N)
        grid = getattr(s["lattice"], "numpy", s["lattice"])
        kphys = grid.kmag / g.cellsize
        pcell = np.where(kphys > 0, (b + fr * grid.mu**2) ** 2 * spectrum(pk, a, t)(np.clip(kphys, 1e-4, None))
                         / g.cellsize**3, 0.0)
        lam = signal_variances(pcell, s["config"], s["lattice"])
        ratios = {}
        if args.paired_mocks:
            files = [np.load(path) for path in args.paired_mocks]
            survey = np.concatenate([x[cap] for x in files])
            box = np.concatenate([x[f"{cap}_box"] for x in files])
            for j in range(s["config"].J + 1):
                for ell in range(s["config"].L + 1):
                    for m in range(ell + 1):
                        key = (SurveyWindow.key(s["config"].sigma(j)), ell, m)
                        signal = lam[j, ell][ell] if m == 0 else 0.5 * (lam[j, ell][ell - m] + lam[j, ell][ell + m])
                        ratios[f"s1m_{key[0]}_{ell}_{m}"] = product.calibrate_s1m(
                            key, survey[:, j, ell, m], box[:, j, ell, m], signal, float(s["noise"][j, ell].mean()))
        if args.s21m_mocks:
            files = [np.load(path) for path in args.s21m_mocks]
            pick = lambda x, k: x[f"{cap}_{k}"] if f"{cap}_{k}" in x else x[f"fiducial_{cap}_{k}"]  # noqa: E731
            measured = np.concatenate([pick(x, "measured") for x in files])
            box = np.concatenate([pick(x, "none") for x in files])  # S2m_box / S1m_survey: the ratio is S2m_s / S2m_b
            config = s["config"]
            entries = [(j1, j2, ell, m1, m2) for j1, j2 in config.second_layer_pairs() for ell in range(args.L + 1)
                       for m1 in range(ell + 1) for m2 in range(ell + 1)]
            for n, (j1, j2, ell, m1, m2) in enumerate(entries):
                key = (SurveyWindow.key(config.sigma(j1)), SurveyWindow.key(config.sigma(j2)), ell, m1, m2)
                ratio = measured[:, n] / box[:, n]
                product.s21m[key] = float(ratio.mean())
                ratios["s21m_" + "_".join(str(k) for k in key)] = ratio
        z = "-".join(f"{v:g}" for v in args.zrange)
        product.save(args.output / f"window_z{z}_{cap}.npz")
        np.savez(args.output / f"window_ratios_z{z}_{cap}.npz", **ratios)
        if ratios:  # calibration error of delta and K for fit_rsd --extra-covariance (one mock set per statistic)
            labels, blocks = [], []
            config = s["config"]
            for kind in ("s1m", "s21m"):
                names = [k for k in ratios if k.startswith(kind + "_")]
                if not names:
                    continue
                values = np.array([ratios[k] / ratios[k].mean() for k in names]).T
                blocks.append(np.cov(values, rowvar=False, ddof=1) / len(values))
                for k in names:
                    p = k.split("_")
                    j = [jj for jj in range(config.J + 1) if SurveyWindow.key(config.sigma(jj)) == float(p[1])][0]
                    if kind == "s1m":
                        labels.append(f"{cap}:S1m_j{j}_l{p[2]}_m{p[3]}")
                    else:
                        j2 = [jj for jj in range(config.J + 1) if SurveyWindow.key(config.sigma(jj)) == float(p[2])][0]
                        labels.append(f"{cap}:S21m_j{j}_j{j2}_l{p[3]}_m{p[4]}_m{p[5]}")
            fractional = np.zeros((len(labels), len(labels)))
            start = 0
            for block in blocks:
                fractional[start:start + len(block), start:start + len(block)] = block
                start += len(block)
            np.savez(args.output / f"window_calibration_z{z}_{cap}.npz", labels=np.array(labels), fractional=fractional)
        deltas = [blk["delta"] for blk in product.s1m.values()]
        summary[cap] = dict(nblocks=len(product.s1m), nentries=len(product.s21m),
                            delta_range=[float(min(deltas)), float(max(deltas))],
                            k_range=[float(min(product.s21m.values())), float(max(product.s21m.values()))]
                            if product.s21m else None)
        print(f"{cap}: {summary[cap]}", flush=True)
    return summary


def run_measure(args, geometries, setups, pk):
    from wstfast.data import save_measurement
    from wstfast.survey import desi_catalogue_path, read_desi_catalogue, sky_to_cartesian

    summary = {}
    for mock in args.measure_mocks:
        for cap, g in geometries.items():
            t0 = time.time()
            s = setups[cap]
            path = args.output / cap / f"wst_r{mock:04d}.npz"
            if path.exists():
                continue
            data = read_desi_catalogue(desi_catalogue_path(mock, cap, root=args.catalogue_root), args.zrange)
            randoms = [read_desi_catalogue(desi_catalogue_path(mock, cap, i, root=args.catalogue_root), args.zrange)
                       for i in args.measure_randoms]
            positions = sky_to_cartesian(data["RA"], data["DEC"], data["Z"])
            rpositions = np.concatenate([sky_to_cartesian(r["RA"], r["DEC"], r["Z"]) for r in randoms])
            delta = g.fkp_field(positions, rpositions)
            out = args.measure(delta, s["config"], support=g.support, weights=g.weights, qs=[args.q], los=s["los"],
                               lattice=s["lattice"], second_layer_bins=s["bins2"])
            metadata = dict(redshift=float(np.mean(args.zrange)), zrange=list(args.zrange), boxsize=g.boxsize,
                            nmesh=g.nmesh, space=cap, nparticles=int(len(positions)), shotnoise=g.shotnoise,
                            tracer="LRG_complete", mock=mock, nrandoms=int(len(rpositions)), cap=cap)
            save_measurement(path, out, s["config"], metadata)
            summary[f"{cap}_{mock}"] = dict(ngal=int(len(positions)), seconds=round(time.time() - t0, 1))
            print(f"mock {mock} {cap}: {len(positions)} galaxies, {len(rpositions)} randoms, {time.time() - t0:.0f} s",
                  flush=True)
    return summary


def run_covariance(args, geometries, setups, pk):
    from numpy.polynomial.legendre import leggauss

    from wstfast.config import select_coefficients
    from wstfast.survey import SurveyWindow, block_signal
    from wstfast.theory.covariance_rsd import ModulusCrossRSD, ModulusTables, RSDWSTCovariance
    from wstfast.theory.rsd_moduli import first_layer_fields

    b, f, a, t = VARIANTS["fiducial"]
    plin = spectrum(pk, a, t)
    boxsize, nmesh = args.covariance_box
    summary = {}
    for (cap, g), path in zip(geometries.items(), args.window_files):
        s, window = setups[cap], SurveyWindow.load(path)
        config = s["config"]
        noise = g.shotnoise
        s1m = [c for c in select_coefficients(config, s1_min_scale=25.0) if c.kind == "S1"]
        s21 = [c for c in select_coefficients(config, s21_min_scale=25.0, s21_min_ratio=2.8, s21_min_scale2=70.0)
               if c.kind == "S21"]
        keys = first_layer_fields(s21)
        fields = [(config.sigma(j), ell, m) for j, ell, m in keys]
        entries = [(c.j, c.j2, c.ell, m1, m2, keys.index((c.j, c.ell, m1)))
                   for c in s21 for m1 in range(c.ell + 1) for m2 in range(c.ell + 1)]
        pfield = lambda k, mu: (b + f * np.asarray(mu) ** 2) ** 2 * plin(k) + noise  # noqa: E731
        z1 = lambda k, mu: b + f * np.asarray(mu) ** 2  # noqa: E731
        x, _ = leggauss(12)
        kgrid, mugrid = np.geomspace(4e-3, 0.12, 24), x[6:]
        t0 = time.time()
        r, gnoise, _ = ModulusCrossRSD(fields, kgrid, mugrid).spectra(plin, f, 0.0, [1.0, b, 0.0, 0.0, 0.0], noise)
        moduli = ModulusTables(kgrid, mugrid, 0.0 * r, gnoise)  # Gaussian field: U does not correlate with delta
        cov = RSDWSTCovariance(config, s1m, entries, fields, args.q, boxsize, pfield, plin, z1, moduli, nmesh=int(nmesh))
        n = cov.ns1m + cov.ns21m
        box = cov.lncov + cov.chaos4(int(nmesh))
        print(f"{cap}: box covariance of {n} entries ({time.time() - t0:.0f} s)", flush=True)
        # effective volume of every entry: its S1m block, or the sigma_j2 block of the second layer for S21m
        veff = []
        for label in cov.labels:
            parts = label.split("_")
            if parts[0] == "S1m":
                j, ell, m = int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:])
            else:
                j, ell, m = int(parts[2][1:]), int(parts[3][1:]), int(parts[5][1:])
            key = (SurveyWindow.key(config.sigma(j)), ell, m)
            block = window.s1m[key]
            box_noise = float(block["weight"] @ block["noise"] / block["weight"].sum())
            sig = block_signal(config, j, ell, m, pfield, noise, boxsize)
            veff.append(window.veff_block(key, sig, box_noise))
        veff = np.array(veff)
        vbox = (boxsize / 1000.0) ** 3
        std = np.sqrt(np.diag(box))
        corr = box / np.outer(std, std)
        models = {"per-block V_eff": corr * np.outer(std * np.sqrt(vbox / veff), std * np.sqrt(vbox / veff)),
                  "single V_eff": box * vbox / window.veff}
        # the mocks, in the order of cov.labels
        measured = {}
        if args.s1m_mocks:
            files = [np.load(p) for p in args.s1m_mocks]
            arr = np.concatenate([x[cap] if cap in x.files else x[f"fiducial_{cap}_measured"] for x in files])
            idx = [tuple(int(p[1:]) for p in label.split("_")[1:]) for label in cov.labels[:cov.ns1m]]
            measured["S1m"] = (np.log(np.array([arr[:, j, ell, m] for j, ell, m in idx]).T), slice(0, cov.ns1m))
        if args.s21m_mocks:
            files = [np.load(p) for p in args.s21m_mocks]
            pick = lambda x: x[f"{cap}_measured"] if f"{cap}_measured" in x.files else x[f"fiducial_{cap}_measured"]  # noqa: E731
            arr = np.concatenate([pick(x) for x in files])
            order = [(ell, m1, m2) for ell in range(args.L + 1) for m1 in range(ell + 1) for m2 in range(ell + 1)]
            idx = [order.index(tuple(int(p[1:]) for p in label.split("_")[3:])) for label in cov.labels[cov.ns1m:n]]
            measured["S21m"] = (np.log(arr[:, idx]), slice(cov.ns1m, n))
        summary[cap] = {}
        for kind, (vectors, sl) in measured.items():
            sample = np.cov(vectors, rowvar=False)
            nm = len(vectors)
            sstd = np.sqrt(np.diag(sample))
            scorr = sample / np.outer(sstd, sstd)
            off = ~np.eye(len(sstd), dtype=bool)
            row = dict(nmocks=nm, std_error=float(1 / np.sqrt(2 * (nm - 1))))
            for name, model in models.items():
                m = model[sl, sl]
                mstd = np.sqrt(np.diag(m))
                ratio = mstd / sstd
                mcorr = m / np.outer(mstd, mstd)
                resid = vectors - vectors.mean(axis=0)
                chi2 = float(np.mean(np.einsum("ni,ij,nj->n", resid, np.linalg.inv(m), resid)))
                row[name] = dict(std_ratio_median=float(np.median(ratio)),
                                 std_ratio_16_84=np.percentile(ratio, [16, 84]).tolist(),
                                 std_ratio_min_max=[float(ratio.min()), float(ratio.max())],
                                 corr_rms_diff=float(np.sqrt(np.mean((mcorr - scorr)[off] ** 2))),
                                 corr_rms_noise=float(np.sqrt(np.mean(((1 - mcorr**2) / np.sqrt(nm))[off] ** 2))),
                                 mean_chi2=chi2, ndata=int(len(sstd)))
                print(f"{cap} {kind} ({nm} mocks, {name}): std analytic / mocks median {row[name]['std_ratio_median']:.3f}"
                      f" [16-84% {row[name]['std_ratio_16_84'][0]:.3f}-{row[name]['std_ratio_16_84'][1]:.3f},"
                      f" error {row['std_error']:.3f}]; corr rms diff {row[name]['corr_rms_diff']:.3f}"
                      f" (noise {row[name]['corr_rms_noise']:.3f}); <chi2> {chi2:.1f} / {len(sstd)}", flush=True)
            summary[cap][kind] = row
        np.savez(args.output / f"covariance_{cap}.npz", labels=cov.labels[:n], box=box, veff=veff,
                 **{name.replace(" ", "_"): model for name, model in models.items()})
    return summary


def run_halos(args, geometries, setups, pk):
    from wstfast.data import save_measurement
    from wstfast.survey import box_to_survey

    caps = list(geometries)
    ric = RadialIntegralConstraint(*geometries.values(), width=args.width[0])
    keys21 = [(j1, j2, ell, m1, m2) for j1, j2 in setups[caps[0]]["config"].second_layer_pairs()
              for ell in range(args.L + 1) for m1 in range(ell + 1) for m2 in range(ell + 1)]
    rows = {f"{cap}{suffix}": [] for cap in caps for suffix in ("", "_box", "_measured", "_none")}
    for r in range(args.realizations):
        i = args.seed_offset + r
        t0 = time.time()
        fields = []
        for k, (cap, g) in enumerate(geometries.items()):
            halos = np.load(args.halo_dir / f"r{2 * i + k:04d}.npz")
            fields.append(box_to_survey(halos["positions"], 1000.0, g, displacements=halos["displacements"],
                                        los="local", seed=[i, k]))
        for cap, survey, field in zip(caps, ric(*fields), fields):
            s, g = setups[cap], geometries[cap]
            common = dict(weights=g.weights, qs=[args.q], los=s["los"], lattice=s["lattice"])
            masked = args.measure(survey, s["config"], support=g.support, second_layer_bins=s["bins2"], **common)
            box = args.measure(field, s["config"], moment_support=g.support, **common)
            s1, s2, b1, b2 = masked["S1m"][0], masked["S2m"][0], box["S1m"][0], box["S2m"][0]
            rows[cap].append(s1)
            rows[f"{cap}_box"].append(b1)
            rows[f"{cap}_measured"].append([s2[j1, j2, ell, m1, m2] / s1[j1, ell, m1] for j1, j2, ell, m1, m2 in keys21])
            rows[f"{cap}_none"].append([b2[j1, j2, ell, m1, m2] / s1[j1, ell, m1] for j1, j2, ell, m1, m2 in keys21])
            metadata = dict(redshift=0.5, boxsize=g.boxsize, nmesh=g.nmesh, space=cap, nparticles=1,
                            shotnoise=g.shotnoise, tracer="quijote_halos_m13_cutsky", realization=i, cap=cap,
                            nbar_scale=args.nbar_scale)
            save_measurement(args.output / "data" / cap / f"wst_r{i:04d}.npz", masked, s["config"], metadata)
        print(f"halos {i}: {time.time() - t0:.0f} s", flush=True)
        np.savez(args.output / "partial_halos.npz", **{k: np.array(v) for k, v in rows.items()},
                 realizations=np.arange(args.seed_offset, args.seed_offset + r + 1))
    return dict(nmocks=args.realizations)


def data_vector(out) -> np.ndarray:
    """Baseline S1m (j, l, |m|) followed by S21m = S2m / S1m at (j1, j2) = (0, 3), (l >= 1, |m1|, |m2|)."""
    s1 = [out["S1m"][j, ell, m] for j in range(4) for ell in range(5) for m in range(ell + 1)]
    s21 = [out["S2m"][0, 3, ell, m1, m2] / out["S1m"][0, ell, m1]
           for ell in range(1, 5) for m1 in range(ell + 1) for m2 in range(ell + 1)]
    return np.array(s1 + s21)


def run_vector(args, geometries, setups, pk):
    ric = RadialIntegralConstraint(*geometries.values(), width=args.width[0])
    mocks = mocks_of(geometries, pk, "fiducial")
    caps = list(geometries)
    vectors = {(cap, kind): [] for cap in caps for kind in ("box", "window")}
    for r in range(args.realizations):
        t0 = time.time()
        fields = [mocks[cap](seed=[300 + r, k]) for k, cap in enumerate(caps)]
        for cap, survey, field in zip(caps, ric(*fields), fields):
            s = setups[cap]
            common = dict(weights=s["g"].weights, qs=[args.q], los=s["los"], lattice=s["lattice"])
            masked = args.measure(survey, s["config"], support=s["g"].support, second_layer_bins=s["bins2"], **common)
            reference = args.measure(field, s["config"], moment_support=s["g"].support, **common)
            vectors[cap, "window"].append(data_vector({k: masked[k][0] for k in ("S1m", "S2m")}))
            vectors[cap, "box"].append(data_vector({k: reference[k][0] for k in ("S1m", "S2m")}))
        print(f"vector {r}: {time.time() - t0:.0f} s", flush=True)
    vectors = {key: np.array(v) for key, v in vectors.items()}
    np.savez(args.output / "vector.npz", **{f"{cap}_{kind}": v for (cap, kind), v in vectors.items()})
    plot_vector(vectors, caps, args.output / "vector.png")
    return {f"{cap}_ratio": (vectors[cap, "window"].mean(0) / vectors[cap, "box"].mean(0)).tolist() for cap in caps}


def plot_vector(vectors, caps, path):
    import matplotlib.pyplot as plt

    colors = {"NGC": "#2a78d6", "SGC": "#eb6834"}  # categorical slots 1 and 2
    n1 = 60
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw=dict(height_ratios=(2.2, 1), hspace=0.06))
    x = np.arange(len(vectors[caps[0], "box"][0]))
    for cap in caps:
        box, window = vectors[cap, "box"].mean(0), vectors[cap, "window"].mean(0)
        axes[0].plot(x, box, ls="--", lw=1.6, color=colors[cap], label=f"{cap}, box (before)")
        axes[0].plot(x, window, ls="-", lw=1.6, color=colors[cap], label=f"{cap}, DR1 window + RIC (after)")
        axes[1].plot(x, window / box - 1, ls="-", lw=1.6, color=colors[cap], label=cap)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("coefficient")
    axes[1].set_ylabel("after / before − 1")
    axes[1].axhline(0, color="#52514e", lw=0.8)
    axes[1].set_xlabel("data-vector index")
    for ax in axes:
        ax.axvline(n1 - 0.5, color="#52514e", lw=0.8)
        for b in (15, 30, 45):
            ax.axvline(b - 0.5, color="#c3c2b7", lw=0.6, ls=":")
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    top = axes[0].get_ylim()[1]
    for b, label in zip((0, 15, 30, 45, 60), ("S1m σ=25", "σ=35", "σ=50", "σ=71", "S21m (25, 70.7)")):
        axes[0].text(b + 0.5, top, label, va="top", ha="left", fontsize=9, color="#52514e")
    axes[0].legend(frameon=False, fontsize=9, ncol=2, loc="lower left")
    axes[0].set_title("Baseline WST data vector of the DR1 LRG 0.4 < z < 0.6 geometry (Gaussian mocks, fiducial spectrum)",
                      fontsize=11, loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    args = get_args()
    args.output.mkdir(parents=True, exist_ok=True)
    geometries = build_geometries(args)
    if args.nbar_scale != 1.0:  # a sparser tracer in the same geometry (the cache keeps the survey's nbar)
        for g in geometries.values():
            g.nbar = g.nbar * args.nbar_scale
    pk = linear_power(np.mean(args.zrange))  # CLASS before torch: their two OpenMP runtimes clash the other way round
    nmesh = next(iter(geometries.values())).nmesh
    if args.device == "numpy":
        lattice, args.measure = Lattice(nmesh), measure_wst_survey
    else:
        from wstfast.measure_torch import TorchLattice, measure_wst_survey_torch

        lattice, args.measure = TorchLattice(nmesh, device=args.device, lmax=args.L), measure_wst_survey_torch
    setups = {}
    for cap, g in geometries.items():
        print(f"{cap}: mesh {g.nmesh}^3 x {g.cellsize:g} Mpc/h, {g.support.sum() * g.cellsize**3 / 1e9:.2f}"
              f" (Gpc/h)^3, regions {np.bincount(g.region[g.support])}", flush=True)
        # sigma = 25, 35, 50, 71 Mpc/h; with --test ric, S21m at (25, 71)
        config = WSTConfig(J=3, L=args.L, L2=args.L, sigma0=25.0 / g.cellsize, step=2**0.5, cellsize=g.cellsize,
                           min_dj=10 if args.test == "lambda" else 3, q=args.q)
        setups[cap] = dict(config=config, lattice=lattice, los=g.line_of_sight(full=True), g=g, bins2=None)
        if args.layer2_mean == "binned" and config.second_layer_pairs():
            setups[cap]["bins2"] = survey.second_layer_bins(g.support, g.nbar, config, lattice)
        if args.test in ("lambda", "products"):
            sigma2 = np.where(g.nbar > 0, 1.0 / (np.where(g.nbar > 0, g.nbar, 1.0) * g.cellsize**3), 0.0)
            setups[cap]["noise"] = noise_variance_maps(sigma2, config, g.support, lattice)
    runs = dict(ric=run_ric, s21m=run_s21m, vector=run_vector, products=run_products, measure=run_measure,
                covariance=run_covariance, halos=run_halos,
                **{"lambda": run_lambda})
    summary = runs[args.test](args, geometries, setups, pk)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
