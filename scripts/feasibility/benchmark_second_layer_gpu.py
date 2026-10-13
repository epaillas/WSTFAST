#!/usr/bin/env python
"""Benchmark of the second-layer template workload (scripts/feasibility/second_layer_rsd.py) on CPU and GPU.

One "field" is what the template measures five times per seed: the line-of-sight-resolved two-layer WST of a 256^3
redshift-space field, U_{j1,l,|m1|} for sigma_j1 = 12.5, 17.7, 25 Mpc/h and l = 1..4, then every second layer
Y_{j2,l,|m2|} with sigma_j2 / sigma_j1 >= 2 up to 70.7 Mpc/h, and the deficit S2m / S2m_G - 1 from the moments
q = 0.8 and 2 (486 numbers). The field here is a toy halo field (Kaiser-distorted Gaussian field with a BBKS spectrum
normalised to Quijote at z = 0.5, plus white shot noise, CIC window): the cost does not depend on its content.

Variants, all on the same field:
    cpu        NumPy / SciPy FFTs at 256^3 (wstfast.measure; the current production path)
    gpu        torch at 256^3 (wstfast.measure_torch)
    gpu-down   torch, with every second layer on the Fourier-truncated mesh of U: 128^3, or 64^3 for sigma_j2 >= 50
               (the second-layer wavelet is ~exp(-50) at those Nyquist frequencies, so this is exact to float32)

It prints the time per field and per template seed (5 fields), and the largest difference of the 486 deficits of
each variant from the reference (gpu, or cpu if --cpu-only). Needs numpy, scipy, torch and this repository.
Examples:

    python scripts/feasibility/benchmark_second_layer_gpu.py                     # gpu and gpu-down, 2 repeats
    python scripts/feasibility/benchmark_second_layer_gpu.py --cpu               # also the CPU path (slow)
    python scripts/feasibility/benchmark_second_layer_gpu.py --device cpu --variants gpu-down   # torch on CPU
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.special import gammaln

BOXSIZE, NMESH, Q = 1000.0, 256, 0.8
SIGMA0, STEP = 3.125, 2**0.5
J1S, JMAX = (4, 5, 6), 9
F, B1, SHOT = 0.76, 1.94, 3700.0  # growth rate, linear bias, shot noise (Mpc/h)^3 of the halo baseline


def sigma(j):
    return SIGMA0 * STEP**j


def pairs():
    return [(j1, j2, ell) for ell in range(1, 5) for j1 in J1S for j2 in range(j1 + 2, JMAX + 1)]


def bbks_power(k, h=0.6711, omega_m=0.3175, ns=0.9624, sigma8=0.834 * 0.77):
    """BBKS linear spectrum normalised to sigma_8(z = 0.5) of the Quijote fiducial (toy accuracy is enough)."""
    gamma = omega_m * h
    q = k / gamma
    t = np.log(1 + 2.34 * q) / (2.34 * q) * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25
    p = k**ns * t**2
    kk = np.geomspace(1e-4, 10, 4000)
    qq = kk / gamma
    tt = np.log(1 + 2.34 * qq) / (2.34 * qq) * (1 + 3.89 * qq + (16.1 * qq) ** 2 + (5.46 * qq) ** 3 + (6.71 * qq) ** 4) ** -0.25
    w = 3 * (np.sin(8 * kk) - 8 * kk * np.cos(8 * kk)) / (8 * kk) ** 3
    s8 = np.trapezoid(kk**ns * tt**2 * w**2 * kk**2, kk) / (2 * np.pi**2)
    return p * sigma8**2 / s8


def toy_field(seed=0):
    """rfft of a 256^3 toy halo field: (b1 + f mu^2) delta_L + white shot noise, times the CIC window."""
    rng = np.random.default_rng(seed)
    k1 = 2 * np.pi * np.fft.fftfreq(NMESH)
    kz = 2 * np.pi * np.fft.rfftfreq(NMESH)
    kx, ky, kzz = np.meshgrid(k1, k1, kz, indexing="ij", sparse=True)
    kcell2 = kx**2 + ky**2 + kzz**2
    kphys = np.sqrt(kcell2) * NMESH / BOXSIZE
    kcell2[0, 0, 0] = 1.0
    mu2 = kzz**2 / kcell2
    white = np.fft.rfftn(rng.standard_normal((NMESH,) * 3))
    signal = white * np.sqrt(np.where(kphys > 0, bbks_power(np.maximum(kphys, 1e-6)), 0.0) * NMESH**3 / BOXSIZE**3)
    noise = np.fft.rfftn(rng.standard_normal((NMESH,) * 3)) * np.sqrt(SHOT * NMESH**3 / BOXSIZE**3)
    window = (np.sinc(kx / (2 * np.pi)) * np.sinc(ky / (2 * np.pi)) * np.sinc(kzz / (2 * np.pi))) ** 2
    return (window * ((B1 + F * mu2) * signal) + noise).astype(np.complex64)


def deficit(s_q, s_2, m2):
    n = 1 if m2 == 0 else 2
    return s_q / ((2 * s_2 / n) ** (Q / 2) * np.exp(gammaln((n + Q) / 2) - gammaln(n / 2))) - 1


def run_cpu(fk):
    from wstfast.measure import Lattice, _moments, _rfft, wavelet_modulus

    lattice, cell, out = Lattice(NMESH), BOXSIZE / NMESH, {}
    harmonics = {ell: lattice.harmonics(ell) for ell in range(1, 5)}
    todo = pairs()
    for ell in range(1, 5):
        for j1 in J1S:
            _, per_m1 = wavelet_modulus(fk, lattice, harmonics[ell], sigma(j1) / cell, ell, los=True)
            for m1, u in enumerate(per_m1):
                mean = u.mean(dtype=np.float64)
                fu = _rfft(u - mean)
                fu[0, 0, 0] = mean * NMESH**3
                for _, j2, _ in (p for p in todo if p[0] == j1 and p[2] == ell):
                    _, per_m2 = wavelet_modulus(fu, lattice, harmonics[ell], sigma(j2) / cell, ell, los=True)
                    for m2, y in enumerate(per_m2):
                        out[j1, j2, ell, m1, m2] = deficit(*_moments(y, [Q, 2.0]), m2)
    return out


def truncate(fu, n_small):
    """Central n_small^3 Fourier modes of a 256^3 rfft array, rescaled so that irfftn gives the same field values."""
    import torch

    h = n_small // 2
    rows = torch.cat([torch.arange(0, h), torch.arange(NMESH - h, NMESH)]).to(fu.device)
    small = fu.index_select(0, rows).index_select(1, rows)[:, :, : h + 1]
    return small * (n_small / NMESH) ** 3


def run_torch(fk_np, device, downsample, sync):
    import torch

    from wstfast.measure_torch import TorchLattice, wavelet_modulus

    lattices = {n: TorchLattice(n, device=device) for n in ((NMESH, 128, 64) if downsample else (NMESH,))}
    fk = torch.as_tensor(fk_np, device=device)
    out, todo = {}, pairs()

    def moments(y):
        a = y.to(torch.float64)
        return float((a**Q).mean()), float((a * a).mean())

    for ell in range(1, 5):
        for j1 in J1S:
            _, per_m1 = wavelet_modulus(fk, lattices[NMESH], sigma(j1) / (BOXSIZE / NMESH), ell, los=True)
            for m1, u in enumerate(per_m1):
                mean = float(u.to(torch.float64).mean())
                fu = torch.fft.rfftn(u - mean)
                fu[0, 0, 0] = mean * NMESH**3
                for _, j2, _ in (p for p in todo if p[0] == j1 and p[2] == ell):
                    n2 = (64 if sigma(j2) >= 49 else 128) if downsample else NMESH
                    f2 = fu if n2 == NMESH else truncate(fu, n2)
                    _, per_m2 = wavelet_modulus(f2, lattices[n2], sigma(j2) / (BOXSIZE / n2), ell, los=True)
                    for m2, y in enumerate(per_m2):
                        out[j1, j2, ell, m1, m2] = deficit(*moments(y), m2)
    sync()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="torch device (auto: cuda if available)")
    parser.add_argument("--variants", nargs="+", default=["gpu", "gpu-down"], choices=("gpu", "gpu-down"))
    parser.add_argument("--cpu", action="store_true", help="also time the NumPy / SciPy path (minutes per field)")
    parser.add_argument("--repeats", type=int, default=2, help="timed repeats per variant (after one warm-up)")
    args = parser.parse_args()

    import torch

    from wstfast.measure_torch import default_device

    device = default_device(args.device)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    print(f"torch {torch.__version__} on {name}; {len(pairs())} (j1, j2, l) pairs, 486 deficits per field")
    t0 = time.time()
    fk = toy_field()
    print(f"toy field built in {time.time() - t0:.1f} s")

    results, timings = {}, {}
    for variant in args.variants:
        run = lambda: run_torch(fk, device, variant == "gpu-down", sync)  # noqa: E731
        run()  # warm-up: cuFFT plans, harmonics cache
        times = []
        for _ in range(args.repeats):
            t0 = time.time()
            results[variant] = run()
            times.append(time.time() - t0)
        timings[variant] = min(times)
        if device.type == "cuda":
            print(f"  {variant}: peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB")
    if args.cpu:
        t0 = time.time()
        results["cpu"] = run_cpu(fk)
        timings["cpu"] = time.time() - t0

    reference = "gpu" if "gpu" in results else ("cpu" if "cpu" in results else next(iter(results)))
    print(f"\n{'variant':10s} {'s / field':>10s} {'s / seed (5)':>13s} {'max |diff| vs ' + reference:>22s}")
    for variant, seconds in timings.items():
        diff = max(abs(results[variant][k] - results[reference][k]) for k in results[reference])
        print(f"{variant:10s} {seconds:10.1f} {5 * seconds:13.1f} {diff:22.2e}")
    print("\n(the current production path is 'cpu': ~11.6 min per seed on a laptop, 1 worker)")


if __name__ == "__main__":
    main()
