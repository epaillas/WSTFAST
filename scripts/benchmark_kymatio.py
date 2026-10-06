#!/usr/bin/env python
"""Benchmark wstmodel's WST estimator against kymatio's HarmonicScattering3D on a Quijote snapshot.

In a configuration both codes support (dyadic scales, same l in both layers), they must agree up to

    S1_kymatio(j, l) = N^3 K_l^q S1_ours(j, l),   S2_kymatio(j1, j2, l) = N^3 K_l^(2q) S2_ours(j1, j2, l),

where N^3 converts kymatio's lattice sums to our means and K_l is the ratio of the per-l filter
normalisations: kymatio 0.3.0 uses N_l (2 pi)^(3/2) (with N_l from its solid_harmonic_3d), we use
sqrt(4 pi / (2l + 1)), and both use a plain Gaussian for l = 0. The report does not rely on K_l:
for each l, S1 ratios must be the same constant at every scale and S2 ratios its square, which
holds for any per-l normalisation (other kymatio versions); K_l^q is printed for comparison.

    python scripts/benchmark_kymatio.py --realization 0 --nmesh 128
    python scripts/benchmark_kymatio.py --timing --repeats 3 --threads 8   # warm timings only
    python scripts/benchmark_kymatio.py --filters                          # filter banks only, seconds

On a cluster, point --snapshot-root at the Quijote fiducial snapshots; --device auto runs kymatio on a
GPU when torch sees one (wstmodel always runs on the CPU).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.special
from scipy.special import factorial

from wstmodel import WSTConfig
from wstmodel.measure import Lattice, measure_wst
from wstmodel.quijote import SNAPSHOT_ROOT, load_density


def _install_sph_harm_shim():
    """kymatio 0.3.0 imports scipy.special.sph_harm, removed in scipy 1.17; map it to sph_harm_y."""
    if not hasattr(scipy.special, "sph_harm"):
        scipy.special.sph_harm = lambda m, n, azimuth, polar: scipy.special.sph_harm_y(n, m, polar, azimuth)


def kymatio_normalisation(ell: int) -> float:
    """Ratio K_l of kymatio's Fourier-space filter prefactor to ours."""
    if ell == 0:
        return 1.0
    if ell % 2 == 0:
        double_factorial = np.prod(np.arange(ell + 1, 0, -2), dtype=float)
        norm = 1.0 / (2 * np.pi * np.sqrt(ell + 0.5) * double_factorial)
    else:
        norm = 1.0 / (2 ** (0.5 * (ell + 3)) * np.sqrt(np.pi * (2 * ell + 1)) * factorial((ell + 1) / 2))
    return norm * (2 * np.pi) ** 1.5 / np.sqrt(4 * np.pi / (2 * ell + 1))


def torch_device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def describe_devices(device, threads=None):
    """Print where each code runs: kymatio on the torch device, wstmodel always on the CPU."""
    import torch

    import wstmodel.measure as measure

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        where = f"GPU {index} ({torch.cuda.get_device_name(index)})"
    else:
        where = f"CPU ({torch.get_num_threads()} torch threads)"
    print(f"kymatio device: {where}; torch.cuda.is_available() = {torch.cuda.is_available()}, "
          f"visible GPUs = {torch.cuda.device_count()}")
    print(f"wstmodel device: CPU ({threads or measure.WORKERS} FFT workers)", flush=True)


def run_kymatio(delta, config, qs, device="auto"):
    """kymatio coefficients reshaped to S1 (nq, J+1, L+1) and S2 (nq, J+1, J+1, L+1)."""
    _install_sph_harm_shim()
    import torch
    from kymatio.torch import HarmonicScattering3D

    device = torch_device(device)
    describe_devices(device)
    scattering = HarmonicScattering3D(J=config.J, shape=delta.shape, L=config.L, sigma_0=config.sigma0,
                                      max_order=2, integral_powers=list(qs)).to(device)
    out = scattering(torch.from_numpy(delta.astype(np.float32)).to(device)).cpu().numpy()  # (paths, L+1, nq)
    nj = config.J + 1
    s1 = np.transpose(out[:nj], (2, 0, 1))
    s2 = np.full((len(qs), nj, nj, config.L + 1), np.nan)
    for index, (j1, j2) in enumerate((j1, j2) for j1 in range(nj) for j2 in range(j1 + 1, nj)):
        s2[:, j1, j2] = out[nj + index].T
    return s1, s2


def timing(delta, config, qs, repeats: int, threads: int, device="auto"):
    """Initialisation, first-call and warm-call wall times of both codes on the same threads."""
    import os

    import torch

    _install_sph_harm_shim()
    from kymatio.torch import HarmonicScattering3D

    import wstmodel.measure as measure

    torch.set_num_threads(threads)
    measure.WORKERS = threads
    device = torch_device(device)
    describe_devices(device, threads)
    x = torch.from_numpy(delta.astype(np.float32)).to(device)

    def run_and_wait(engine):
        out = engine(x)
        if device.type == "cuda":
            torch.cuda.synchronize()  # GPU kernels are asynchronous
        return out

    rows = {}
    for name, init, call in (
        (f"kymatio ({device.type})", lambda: HarmonicScattering3D(J=config.J, shape=delta.shape, L=config.L,
                                                                   sigma_0=config.sigma0, max_order=2,
                                                                   integral_powers=list(qs)).to(device),
         run_and_wait),
        ("wstmodel", lambda: Lattice(delta.shape[0]),
         lambda lattice: measure_wst(delta, config, qs=qs, lattice=lattice)),
    ):
        start = time.perf_counter()
        engine = init()
        t_init = time.perf_counter() - start
        times = []
        for _ in range(repeats + 1):
            start = time.perf_counter()
            call(engine)
            times.append(time.perf_counter() - start)
        rows[name] = (t_init, times[0], np.mean(times[1:]), np.std(times[1:]))
        del engine
    print(f"nmesh={delta.shape[0]}, J={config.J}, L={config.L}, {len(qs)} q, {threads} threads ({os.cpu_count()} cores)")
    print(f"{'':16s} {'init [s]':>9s} {'1st call':>9s} {'warm call':>14s}")
    for name, (t_init, first, mean, std) in rows.items():
        print(f"{name:16s} {t_init:9.2f} {first:9.2f} {mean:8.2f} ± {std:4.2f}")


def compare_filters(nmesh: int, config: WSTConfig, device="auto"):
    """Compare kymatio's stored filter bank with ours: sum_m |psi_{j,l}^m(k)|^2 on the full FFT grid.

    This isolates the filters from FFTs, backends and devices. They must agree up to K_l^2.
    """
    _install_sph_harm_shim()
    from kymatio.torch import HarmonicScattering3D

    device = torch_device(device)
    scattering = HarmonicScattering3D(J=config.J, shape=(nmesh,) * 3, L=config.L, sigma_0=config.sigma0,
                                      max_order=2, integral_powers=[config.q]).to(device)
    k = 2 * np.pi * np.fft.fftfreq(nmesh)
    kmag = np.sqrt(sum(x**2 for x in np.meshgrid(k, k, k, indexing="ij")))
    print(f"filter bank on {nmesh}^3 (kymatio stores {len(scattering.filters)} l-blocks)")
    print("   l  j   max |kymatio / (K_l^2 ours) - 1| where ours > 1e-6 max")
    for ell, block in enumerate(scattering.filters):
        block = block.detach().cpu().numpy()
        power = (block[..., 0] ** 2 + block[..., 1] ** 2).sum(axis=1)  # (J+1, n, n, n), sum over m
        for j in range(config.J + 1):
            x = config.sigma0 * 2**j * kmag
            ours = kymatio_normalisation(ell) ** 2 * x ** (2 * ell) * np.exp(-(x**2))
            mask = ours > 1e-6 * ours.max()
            print(f"   {ell}  {j}   {np.max(np.abs(power[j][mask] / ours[mask] - 1)):.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--realization", type=int, default=0)
    parser.add_argument("--nmesh", type=int, default=128)
    parser.add_argument("--J", type=int, default=4)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--sigma0", type=float, default=0.8)
    parser.add_argument("--q", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--timing", action="store_true", help="only time initialisation and warm calls")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="auto", help="kymatio device: auto, cpu or cuda")
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_ROOT)
    parser.add_argument("--filters", action="store_true", help="only compare the filter banks (no snapshot needed)")
    args = parser.parse_args()
    if args.filters:
        compare_filters(32, WSTConfig(J=args.J, L=args.L, sigma0=args.sigma0), args.device)
        report_versions()
        return

    delta, header = load_density(args.realization, nmesh=args.nmesh, root=args.snapshot_root)
    config = WSTConfig(J=args.J, L=args.L, sigma0=args.sigma0, cellsize=header["boxsize"] / args.nmesh)
    if args.timing:
        timing(delta, config, args.q, args.repeats, args.threads, args.device)
        return
    start = time.time()
    ours = measure_wst(delta, config, qs=args.q, lattice=Lattice(args.nmesh))
    print(f"wstmodel: {time.time() - start:.0f} s")
    start = time.time()
    s1_k, s2_k = run_kymatio(delta, config, args.q, args.device)
    print(f"kymatio:  {time.time() - start:.0f} s")

    report_versions()
    ncells = args.nmesh**3
    kl = np.array([kymatio_normalisation(ell) for ell in range(config.L + 1)])
    for iq, q in enumerate(args.q):
        r1 = s1_k[iq] / (ncells * ours["S1"][iq])
        r2 = s2_k[iq] / (ncells * ours["S2"][iq])
        print(f"q = {q}")
        print("   l   K_l^q expected   S1 ratio (median)   S1 spread   S2 ratio / S1 ratio^2 - 1 (max)")
        for ell in range(config.L + 1):
            # Normalisation-free test: S1 ratios must be the same for every j, and S2 ratios their square.
            c1 = np.median(r1[:, ell])
            spread = np.max(np.abs(r1[:, ell] / c1 - 1))
            square = np.nanmax(np.abs(r2[..., ell] / c1**2 - 1))
            print(f"   {ell}   {kl[ell]**q:14.6g}   {c1:17.6g}   {spread:9.1e}   {square:9.1e}")


def report_versions():
    import kymatio
    import scipy
    import torch

    print(f"kymatio {kymatio.__version__}, scipy {scipy.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    shim = getattr(scipy.special.sph_harm, "__name__", "") == "<lambda>"
    print(f"kymatio from {kymatio.__file__}")
    print(f"spherical harmonics: {'compatibility shim over sph_harm_y' if shim else 'scipy.special.sph_harm (native)'}")


if __name__ == "__main__":
    main()
