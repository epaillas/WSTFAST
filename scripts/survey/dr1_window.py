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
    parser.add_argument("--test", choices=("ric", "lambda"), required=True)
    parser.add_argument("--zrange", type=float, nargs=2, default=[0.4, 0.6])
    parser.add_argument("--cellsize", type=float, default=10.0)
    parser.add_argument("--mocks", type=int, nargs="+", default=[0, 1, 2], help="data for n(r)")
    parser.add_argument("--randoms", type=int, nargs="+", default=[0, 1], help="random files for the mask")
    parser.add_argument("--width", type=float, nargs="+", default=[5.0], help="RIC shell widths [Mpc/h]")
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--calibrate", type=int, default=12, help="signal-only mocks for the lambda window")
    parser.add_argument("--realizations", type=int, default=10, help="test mocks (per variant for lambda)")
    parser.add_argument("--catalogue-root", type=Path, default=survey.DESI_ROOT,
                        help="directory with mock<i>/LRG_complete_* catalogues")
    parser.add_argument("--device", default="numpy", help="numpy, or a torch device (cpu, cuda)")
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
                                         qs=[args.q], los=s["los"], lattice=s["lattice"])
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
    for name, (b, f, a, t) in VARIANTS.items():
        mocks = mocks_of(geometries, pk, name)
        measured = {cap: [] for cap in caps}
        for r in range(args.realizations):
            t0 = time.time()
            fields = [mocks[cap](seed=[200 + r, k]) for k, cap in enumerate(caps)]
            for cap, field in zip(caps, ric(*fields)):
                s = setups[cap]
                out = args.measure(field, s["config"], support=s["g"].support, weights=s["g"].weights,
                                         qs=[args.q], los=s["los"], lattice=s["lattice"])
                measured[cap].append(out["S1m"][0])
            print(f"{name} test {r}: {time.time() - t0:.0f} s", flush=True)
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


def main():
    args = get_args()
    args.output.mkdir(parents=True, exist_ok=True)
    geometries = build_geometries(args)
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
                           min_dj=3 if args.test == "ric" else 10, q=args.q)
        setups[cap] = dict(config=config, lattice=lattice, los=g.line_of_sight(full=True), g=g)
        if args.test == "lambda":
            sigma2 = np.where(g.nbar > 0, 1.0 / (np.where(g.nbar > 0, g.nbar, 1.0) * g.cellsize**3), 0.0)
            setups[cap]["noise"] = noise_variance_maps(sigma2, config, g.support, lattice)
    summary = (run_ric if args.test == "ric" else run_lambda)(args, geometries, setups, pk)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
