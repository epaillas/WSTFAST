#!/usr/bin/env python
"""Timing and output comparison of WSTFAST's WST against ACM's (pypower painting + kymatio).

ACM's estimator is ``acm.estimators.galaxy_clustering.wst.WaveletScatteringTransform``, used as in
``scripts/abacus_base.py:compute_wst``: CIC painting with pypower (no compensation, no interlacing), then
kymatio's ``HarmonicScattering3D`` on torch. WSTFAST is used as in ``compute_wst_fast``: its own CIC painting
and ``measure_wst`` (numpy) or ``measure_wst_torch``. Both codes see the same catalog and the same mesh.

Matching the outputs:

- ACM returns [S0, S1/S2 flattened as kymatio's (paths, L+1)], all divided by N^3 (lattice means).
  S0 uses the configuration's q, but S1 and S2 take index 0 of kymatio's default
  ``integral_powers=(0.5, 1, 2)``, so **ACM's S1 and S2 are at q = 0.5 whatever the configuration's q**.
- The filters differ only by a constant per l, so S1_acm = K_l^0.5 S1_ours and S2_acm = K_l S2_ours, with
  K_l from ``scripts/benchmark_kymatio.py:kymatio_normalisation``. S0 needs no conversion.
- ACM reads its mesh back on lattice points that sit on the mesh nodes, so both codes transform the same
  field (ACM's has x and y swapped by ``np.meshgrid``, a reflection the WST is invariant to). S0 and every
  l = 0 coefficient then agree to float32 precision (~1e-8), and so does l >= 1 once the filters are
  negligible at the Nyquist frequency. For filters only a cell or less wide they are not: the codes treat
  the unpaired Nyquist modes differently (WSTFAST: rfft with real harmonics; kymatio: complex harmonics on
  the full FFT grid), so l >= 1 at sigma0 = 0.4 cells (j5) differs by ~4e-3 at j = 0, ~1e-4 at j = 1, and S2
  inherits it from j1 <= 1; at sigma0 = 1 cell (j4_alt) the j = 0 difference is ~1e-5.

Timings are split as production uses them: a one-time setup per configuration (kymatio's filter bank, or
WSTFAST's lattice) and a per-catalog time (painting + transform), the median of --repeats.

    python benchmark/wstfast_vs_acm.py --meshsizes 32 64 --compare-meshsize 64 --repeats 1   # quick check
    python benchmark/wstfast_vs_acm.py                                                       # full sweep
    python benchmark/wstfast_vs_acm.py --device cuda --meshsizes 64 128 192 256 320          # on a GPU node

Writes timing.json, timing_vs_meshsize.png, coefficients.npz and coefficients.png to --output-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_kymatio import _install_sph_harm_shim, kymatio_normalisation, torch_device  # noqa: E402

import wstfast.measure as measure  # noqa: E402
from wstfast.config import WSTConfig  # noqa: E402
from wstfast.quijote import SNAPNUM, SNAPSHOT_ROOT, iter_positions, read_header, snapshot_files  # noqa: E402

# Same presets as scripts/abacus_base.py (meshsize is swept here instead).
WST_CONFIGS = {
    'j5': {'J': 5, 'L': 3, 'q': 0.8, 'sigma': 0.4},
    'j4': {'J': 4, 'L': 4, 'q': 1, 'sigma': 0.8},
    'j4_alt': {'J': 4, 'L': 4, 'q': 1, 'sigma': 1.0},
}
ACM_Q = 0.5  # kymatio's default integral_powers[0], what ACM keeps for S1 and S2
COLORS = {'ACM (kymatio)': '#2a78d6', 'WSTFAST (numpy)': '#eb6834', 'WSTFAST (torch)': '#1baf7a'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wst-config', choices=sorted(WST_CONFIGS), default='j5')
    parser.add_argument('--meshsizes', type=int, nargs='+', default=[32, 48, 64, 96, 128, 160, 192])
    parser.add_argument('--compare-meshsize', type=int, default=128, help='meshsize of the coefficient comparison')
    parser.add_argument('--repeats', type=int, default=3, help='per-catalog calls timed (median)')
    parser.add_argument('--threads', type=int, default=8, help='torch threads and WSTFAST FFT workers')
    parser.add_argument('--device', default='auto', help='torch device of kymatio and WSTFAST torch: auto, cpu or cuda')
    parser.add_argument('--backends', nargs='+', choices=('numpy', 'torch'), default=['numpy', 'torch'],
                        help='WSTFAST backends to run')
    parser.add_argument('--realization', default='0', help='Quijote fiducial realization')
    parser.add_argument('--redshift', type=float, default=0.5)
    parser.add_argument('--nparticles', type=float, default=1e6, help='random subsample of the CDM particles')
    parser.add_argument('--snapshot-root', type=Path, default=SNAPSHOT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/benchmark'))
    args = parser.parse_args()
    args.meshsizes = sorted(set(args.meshsizes) | {args.compare_meshsize})
    return args


def load_catalog(realization, redshift, nparticles, root, seed=42):
    """Random subsample of a Quijote snapshot's CDM particles, shifted to [-L/2, L/2) as the HOD catalogs."""
    files = snapshot_files(Path(root) / str(realization) / f'snapdir_{SNAPNUM[redshift]}')
    header = read_header(files)
    fraction = min(1.0, nparticles / header['nparticles'])
    rng = np.random.default_rng(seed)
    chunks = [pos[rng.random(len(pos)) < fraction] for pos in iter_positions(files, header, rsd=False)]
    boxsize = np.full(3, header['boxsize'])
    positions = np.concatenate(chunks) - boxsize / 2
    print(f'catalog: Quijote {realization}, z={redshift}, {len(positions)} particles in {header["boxsize"]:g} Mpc/h',
          flush=True)
    return positions, boxsize


class ACM:
    """ACM's WaveletScatteringTransform, as in compute_wst (the kymatio filter bank is the setup)."""

    name = 'ACM (kymatio)'

    def __init__(self, wst_args, device):
        self.wst_args = wst_args
        self.device = device

    def _estimator(self, positions, boxsize, meshsize, init=None):
        from acm.estimators.galaxy_clustering.wst import WaveletScatteringTransform

        estimator = WaveletScatteringTransform(data_positions=positions, boxsize=boxsize, boxcenter=0.0,
                                               meshsize=np.repeat(meshsize, 3), init_kymatio=init,
                                               backend='pypower', kymatio_backend='torch', **self.wst_args)
        if init is None and estimator.S is not None:
            estimator.S.to(self.device)  # ACM picks cuda when available; follow --device instead
        estimator.device = self.device
        return estimator

    def setup(self, positions, boxsize, meshsize):
        return self._estimator(positions, boxsize, meshsize).S

    def call(self, setup, positions, boxsize, meshsize):
        estimator = self._estimator(positions, boxsize, meshsize, init=setup)
        estimator.set_density_contrast()
        return estimator.run()


class WSTFast:
    """WSTFAST, as in compute_wst_fast (the lattice and cached harmonics are the setup)."""

    def __init__(self, backend, wst_args, device):
        self.backend, self.device = backend, device
        self.name = f'WSTFAST ({backend})'
        self.config = WSTConfig(J=wst_args['J'], L=wst_args['L'], sigma0=wst_args['sigma'], q=wst_args['q'])
        self.qs = sorted({ACM_Q, float(wst_args['q'])})

    def setup(self, positions, boxsize, meshsize):
        if self.backend == 'torch':
            from wstfast.measure_torch import TorchLattice

            return TorchLattice(meshsize, device=self.device, lmax=self.config.L)
        return measure.Lattice(meshsize)

    def call(self, lattice, positions, boxsize, meshsize):
        cell_positions = np.mod((positions + boxsize / 2) * (meshsize / boxsize), meshsize)
        if self.backend == 'torch':
            from wstfast.measure_torch import measure_wst_torch, paint_cic_torch

            delta = paint_cic_torch([cell_positions], meshsize, float(meshsize), device=lattice.device)
            return measure_wst_torch(delta, self.config, qs=self.qs, lattice=lattice)
        from wstfast.quijote import paint_cic

        delta = paint_cic([cell_positions], meshsize, float(meshsize))
        return measure.measure_wst(delta, self.config, qs=self.qs, lattice=lattice)


def synchronize(device):
    import torch

    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def time_code(code, positions, boxsize, meshsize, repeats, device):
    """(setup seconds, median per-catalog seconds, output of the last call)."""
    synchronize(device)
    start = time.perf_counter()
    setup = code.setup(positions, boxsize, meshsize)
    synchronize(device)
    t_setup = time.perf_counter() - start
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = code.call(setup, positions, boxsize, meshsize)
        synchronize(device)
        times.append(time.perf_counter() - start)
    del setup
    return t_setup, float(np.median(times)), out


def acm_to_arrays(vector, J, L):
    """Split ACM's flat output into S0, S1 (J+1, L+1) and S2 (J+1, J+1, L+1), NaN where not measured."""
    nj = J + 1
    paths = np.asarray(vector[1:]).reshape(-1, L + 1)
    s1 = paths[:nj]
    s2 = np.full((nj, nj, L + 1), np.nan)
    for index, (j1, j2) in enumerate((j1, j2) for j1 in range(nj) for j2 in range(j1 + 1, nj)):
        s2[j1, j2] = paths[nj + index]
    return float(vector[0]), s1, s2


def wstfast_to_acm(result, J, L, q):
    """WSTFAST's coefficients at ACM's exponents and normalisation, flattened in ACM's order."""
    kl = np.array([kymatio_normalisation(ell) for ell in range(L + 1)])
    iq = int(np.flatnonzero(np.isclose(result['q'], q))[0])
    ia = int(np.flatnonzero(np.isclose(result['q'], ACM_Q))[0])
    s1 = result['S1'][ia] * kl**ACM_Q
    s2 = result['S2'][ia] * kl ** (2 * ACM_Q)
    nj = J + 1
    paths = [s1[j] for j in range(nj)] + [s2[j1, j2] for j1 in range(nj) for j2 in range(j1 + 1, nj)]
    return np.concatenate([[result['S0'][iq]], np.ravel(paths)])


def plot_timing(timings, meshsizes, title, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, rows in timings.items():
        n = [m for m in meshsizes if str(m) in rows and rows[str(m)].get('call') is not None]
        if not n:
            continue
        call = [rows[str(m)]['call'] for m in n]
        setup = [rows[str(m)]['setup'] for m in n]
        color = COLORS[name]
        ax.plot(n, call, '-o', color=color, lw=2, ms=6, label=f'{name}, per catalog')
        ax.plot(n, setup, '--', color=color, lw=1.5, alpha=0.8, label=f'{name}, one-time setup')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(meshsizes)
    ax.set_xticklabels([str(m) for m in meshsizes])
    ax.set_xlabel('meshsize (cells per side)')
    ax.set_ylabel('wall time [s]')
    ax.set_title(title, fontsize=9)
    ax.grid(True, which='major', color='0.9', lw=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlim(meshsizes[0] / 1.15, meshsizes[-1] * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coefficients(acm, ours, J, L, title, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nj = J + 1
    n1 = nj * (L + 1)
    index = np.arange(len(acm))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
    for ax in (top, bottom):
        ax.axvspan(0.5, n1 + 0.5, color='0.95', lw=0)
        ax.grid(True, axis='y', color='0.9', lw=0.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    top.plot(index, ours, '-', color=COLORS['WSTFAST (numpy)'], lw=1.5, label='WSTFAST (in ACM normalisation)')
    top.plot(index, acm, 'o', color=COLORS['ACM (kymatio)'], ms=4.5, mfc='none', mew=1.2, label='ACM (kymatio)')
    top.set_yscale('log')
    top.set_ylabel('coefficient')
    top.set_title(title, fontsize=9)
    top.legend(fontsize=8, frameon=False, loc='upper right')
    for x, label in ((0, 'S0'), ((n1 + 1) / 2, 'S1'), ((n1 + len(acm)) / 2, 'S2')):
        top.text(x, 0.03, label, transform=top.get_xaxis_transform(), ha='center', fontsize=9, color='0.25')
    bottom.plot(index, ours / acm - 1, '.', color='0.3', ms=4)
    bottom.axhline(0, color='0.6', lw=0.8)
    bottom.set_ylabel('WSTFAST / ACM − 1')
    bottom.set_xlabel(f'coefficient index (ACM order: S0, then (j, l) paths with l fastest, L={L})')
    bottom.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    import torch

    _install_sph_harm_shim()
    torch.set_num_threads(args.threads)
    measure.WORKERS = args.threads
    device = torch_device(args.device)
    wst_args = WST_CONFIGS[args.wst_config]
    print(f'config {args.wst_config}: {wst_args}; device {device}; {args.threads} threads', flush=True)
    print(f'note: ACM computes S0 at q={wst_args["q"]} but S1, S2 at q={ACM_Q} (kymatio default integral_powers)')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    positions, boxsize = load_catalog(args.realization, args.redshift, args.nparticles, args.snapshot_root)
    codes = [ACM(wst_args, device)] + [WSTFast(backend, wst_args, device) for backend in args.backends]

    # Untimed warm-up on a small mesh: imports, torch/pypower initialisation and FFT plans are one-off costs.
    for code in codes:
        time_code(code, positions, boxsize, 16, 1, device)

    timings = {code.name: {} for code in codes}
    outputs = {}
    failed = set()
    for meshsize in args.meshsizes:
        for code in codes:
            if code.name in failed:
                continue
            try:
                t_setup, t_call, out = time_code(code, positions, boxsize, meshsize, args.repeats, device)
            except (RuntimeError, MemoryError) as error:
                print(f'{code.name} failed at meshsize {meshsize} ({type(error).__name__}: {error}); '
                      'skipping larger meshes', flush=True)
                failed.add(code.name)
                continue
            timings[code.name][str(meshsize)] = dict(setup=t_setup, call=t_call)
            print(f'n={meshsize:4d}  {code.name:16s} setup {t_setup:8.2f} s   per catalog {t_call:8.2f} s', flush=True)
            if meshsize == args.compare_meshsize:
                outputs[code.name] = out
            del out
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    meta = dict(config=args.wst_config, wst_args=wst_args, device=str(device), threads=args.threads,
                repeats=args.repeats, nparticles=len(positions), boxsize=boxsize.tolist(),
                realization=args.realization, redshift=args.redshift)
    with open(args.output_dir / 'timing.json', 'w') as f:
        json.dump(dict(meta, timings=timings), f, indent=2)
    title = (f'{args.wst_config}: J={wst_args["J"]}, L={wst_args["L"]}, σ0={wst_args["sigma"]} cells; '
             f'{len(positions):.2g} particles; {device.type}, {args.threads} threads')
    plot_timing(timings, args.meshsizes, title, args.output_dir / 'timing_vs_meshsize.png')

    acm_name = codes[0].name
    if acm_name not in outputs:
        print(f'no ACM output at meshsize {args.compare_meshsize}; skipping the coefficient comparison')
        return
    J, L = wst_args['J'], wst_args['L']
    acm = np.asarray(outputs[acm_name], dtype=np.float64)
    s0, s1, s2 = acm_to_arrays(acm, J, L)
    saved = dict(acm=acm, acm_S0=s0, acm_S1=s1, acm_S2=s2)
    print(f'coefficients at meshsize {args.compare_meshsize}: max |WSTFAST / ACM - 1|')
    ours_flat = None
    for code in codes[1:]:
        if code.name not in outputs:
            continue
        ours = wstfast_to_acm(outputs[code.name], J, L, wst_args['q'])
        ratio = ours / acm - 1
        n1 = (J + 1) * (L + 1)
        print(f'  {code.name:16s} S0 {abs(ratio[0]):.1e}   S1 {np.max(abs(ratio[1:1 + n1])):.1e}   '
              f'S2 {np.max(abs(ratio[1 + n1:])):.1e}')
        saved[code.backend] = ours
        ours_flat = ours if ours_flat is None else ours_flat
    np.savez(args.output_dir / 'coefficients.npz', metadata=json.dumps(meta), **saved)
    if ours_flat is not None:
        plot_coefficients(acm, ours_flat, J, L, f'{title}; meshsize {args.compare_meshsize}; '
                          f'S0 at q={wst_args["q"]}, S1/S2 at q={ACM_Q}',
                          args.output_dir / 'coefficients.png')
    print(f'wrote {args.output_dir}/timing.json, timing_vs_meshsize.png, coefficients.npz, coefficients.png')


if __name__ == '__main__':
    main()
