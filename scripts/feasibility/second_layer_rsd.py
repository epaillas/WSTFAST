#!/usr/bin/env python
"""Second-layer non-Gaussianity of the line-of-sight-resolved halo WST in redshift space, on perturbation-theory fields.

The analogue of second_layer_uv.py for biased tracers in redshift space. On the Quijote mesh (256^3, 1 Gpc/h, CIC
window) a Gaussian delta_1 with the linear spectrum at z = 0.5 gives the EdS SPT fields delta_2, theta_2, delta_3,
theta_3, the tracer field of wstfast.theory.bias through third order,

    delta_g = b1 delta + b2/2 (delta^2 - <>) + bG2 G2[Phi_g] + bGamma3 (G2[Phi_g] - G2[Phi_v])
              + b3 delta^3/6 + bG3 G3 + bdG2 delta G2         (cubic: relation of bias.cubic_priors + offsets),

and its redshift-space map delta_s = delta_g + sum_j (-i k_z)^j / j! FT[u_z^j (1 + delta_g)], u_z = -f d_z lap^-1 theta,
each truncated order by order: delta_s = s1 + eps s2 + eps^2 s3, plus a Gaussian white shot noise (1 + alpha0) V / N.
The biases and alpha0 are the best fit of the halo baseline (scripts/layer2/halo_template_biases.json). For each
first-layer field U_{j1,l,|m1|} and second-layer scale j2 the deficit S2m / S2m_G - 1 (exact <|Y_|m2||^2> of the same
box) is measured at eps = 0, +-0.3, +-0.5 (delta_1 and the noise flipped together), and fitted as d0 + d2 eps^2 + d4 eps^4:
d0 is the Gaussian (Kaiser + shot noise) value, d2 the tree-level gravitational term. Per-seed results are cached in
outputs/feasibility/second_layer_rsd/seed_*.json (with --only, in second_layer_rsd_<pairs>/). Example:

    python scripts/feasibility/second_layer_rsd.py --seeds 0-39 --workers 3
    CUDA_VISIBLE_DEVICES=0 python scripts/feasibility/second_layer_rsd.py --seeds 40-79 --device cuda   # one GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/feasibility")]

from pt_modulus_noise import SPTFields, cic_window  # noqa: E402
from wstfast.config import QUIJOTE_COSMOLOGY, WSTConfig  # noqa: E402
from wstfast.measure import Lattice, _irfft, _moments, _rfft, wavelet_modulus  # noqa: E402

DATA = ROOT / "data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los"
CACHE = ROOT / "outputs/feasibility/second_layer_rsd"
#: Best-fit biases of the halo baseline (outputs/inference/rsd/paper_c_wst_nsfixed_V25), shared with the cluster.
BIASES = ROOT / "scripts/layer2/halo_template_biases.json"
BOXSIZE, NMESH, Z, Q = 1000.0, 256, 0.5, 0.8
EPS = (0.0, 0.3, 0.5)
J1S = (4, 5, 6)  # sigma_j1 = 12.5, 17.7, 25 Mpc/h


def config():
    path = sorted((DATA / "rsd").glob("wst_r*.npz"))[0]
    meta = json.loads(str(np.load(path)["metadata"]))
    return WSTConfig(**meta["config"]), meta


def pairs(cfg, only=None):
    """(j1, j2, l): sigma_j1 in J1S, sigma_j2 / sigma_j1 >= 2, sigma_j2 <= 70.7 Mpc/h (the halo data reach); or only
    the (j1, j2) pairs of ``only``."""
    if only:
        return [(j1, j2, ell) for ell in range(1, 5) for j1, j2 in only]
    return [(j1, j2, ell) for ell in range(1, 5) for j1 in J1S for j2 in range(j1 + 2, cfg.J + 1)]


def biases(path=BIASES):
    """Biases of the template (a JSON of b1, b2, bG2, bGamma3, db3, dbG3, dbdG2, alpha0, or a fit summary.json with
    them under 'bestfit'); cubic ones from the relation plus offsets (bias.cubic_priors)."""
    from wstfast.theory.bias import cubic_priors

    best = json.loads(Path(path).read_text())
    best = best.get("bestfit", best)
    b3, bg3, bdg2 = cubic_priors(best["b1"], best["b2"]) + np.array([best["db3"], best["dbG3"], best["dbdG2"]])
    return dict(b1=best["b1"], b2=best["b2"], bG2=best["bG2"], bGamma3=best["bGamma3"], b3=b3, bG3=bg3, bdG2=bdg2,
                alpha0=best["alpha0"])


class HaloRSDFields:
    """Orders s1, s2, s3 (rfft arrays) of the redshift-space tracer field, from SPTFields."""

    def __init__(self, spt: SPTFields, f: float, b: dict):
        self.spt, self.n = spt, spt.n
        kx, ky, kz = spt.kvec
        self.kz, self.k2 = kz, spt.k2
        d1k = spt.delta1
        d2k, t2k = spt._second(d1k)
        d3k = spt.delta3
        a = spt._alpha(d1k, d2k) + spt._alpha(t2k, d1k)
        t3k = (3 * a + 12 * spt._beta(d1k, t2k)) / 18
        real = spt._real
        d1, d2, d3 = real(d1k), real(d2k), real(d3k)
        m1, m2, m2mt = self.tidal(d1k), self.tidal(d2k), self.tidal(d2k - t2k)
        g2_11 = self.g2(m1, m1, d1, d1)
        g2_12 = self.g2(m1, m2, d1, d2)
        g2_1x = self.g2(m1, m2mt, d1, real(d2k - t2k))
        zero = lambda x: x - x.mean()  # noqa: E731  (operators are defined with their mean subtracted)
        dg1 = b["b1"] * d1
        dg2 = b["b1"] * d2 + 0.5 * b["b2"] * zero(d1**2) + b["bG2"] * zero(g2_11)
        dg3 = (b["b1"] * d3 + b["b2"] * zero(d1 * d2) + 2 * b["bG2"] * zero(g2_12) + 2 * b["bGamma3"] * zero(g2_1x)
               + b["b3"] / 6 * zero(d1**3) + b["bG3"] * zero(self.g3(m1, d1)) + b["bdG2"] * zero(d1 * g2_11))
        u1, u2, u3 = (real(1j * f * kz / self.k2 * t) for t in (d1k, t2k, t3k))
        T = self.map_term
        self.s1 = _rfft(dg1) + T(1, u1)
        self.s2 = _rfft(dg2) + T(1, u2 + u1 * dg1) + T(2, u1**2)
        self.s3 = (_rfft(dg3) + T(1, u3 + u1 * dg2 + u2 * dg1) + T(2, 2 * u1 * u2 + u1**2 * dg1) + T(3, u1**3))

    def map_term(self, j, field):
        """(-i k_z)^j / j! FT[field]."""
        from math import factorial

        return (-1j * self.kz) ** j / factorial(j) * _rfft(field)

    def tidal(self, fk):
        """Components (xx, yy, zz, xy, xz, yz) of d_i d_j lap^-1 of a field (rfft)."""
        kx, ky, kz = self.spt.kvec
        comps = ((kx, kx), (ky, ky), (kz, kz), (kx, ky), (kx, kz), (ky, kz))
        return [self.spt._real(ki * kj / self.k2 * fk) for ki, kj in comps]

    @staticmethod
    def g2(ma, mb, a, b):
        """G2(A, B) = sum_ij M_ij[A] M_ij[B] - A B."""
        diag = ma[0] * mb[0] + ma[1] * mb[1] + ma[2] * mb[2]
        off = 2 * (ma[3] * mb[3] + ma[4] * mb[4] + ma[5] * mb[5])
        return diag + off - a * b

    @staticmethod
    def g3(m, trace):
        """G3 = -1/2 [2 tr M^3 - 3 tr M^2 tr M + (tr M)^3] of the tidal tensor M of delta_1 (tr M = delta_1)."""
        xx, yy, zz, xy, xz, yz = m
        tr2 = xx**2 + yy**2 + zz**2 + 2 * (xy**2 + xz**2 + yz**2)
        det = xx * (yy * zz - yz**2) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
        e2 = 0.5 * (trace**2 - tr2)
        tr3 = trace**3 - 3 * trace * e2 + 3 * det
        return -0.5 * (2 * tr3 - 3 * tr2 * trace + trace**3)


def deficits(fk, cfg, todo, lattice, harmonics):
    """S2m / S2m_G - 1 for every (j1, j2, l, |m1|, |m2|) of one field (rfft array)."""
    out = {}
    cell = BOXSIZE / NMESH
    for ell in sorted({p[2] for p in todo}):
        for j1 in sorted({p[0] for p in todo if p[2] == ell}):
            _, per_m1 = wavelet_modulus(fk, lattice, harmonics[ell], cfg.sigma(j1) / cell, ell, los=True)
            for m1, u in enumerate(per_m1):
                mean = u.mean(dtype=np.float64)
                fu = _rfft(u - mean)
                fu[0, 0, 0] = mean * NMESH**3
                for _, j2, _ in (p for p in todo if p[0] == j1 and p[2] == ell):
                    _, per_m2 = wavelet_modulus(fu, lattice, harmonics[ell], cfg.sigma(j2) / cell, ell, los=True)
                    for m2, y in enumerate(per_m2):
                        n = 1 if m2 == 0 else 2
                        sq, s2 = _moments(y, [Q, 2.0])
                        gauss = (2 * s2 / n) ** (Q / 2) * np.exp(gammaln((n + Q) / 2) - gammaln(n / 2))
                        out[repr((j1, j2, ell, m1, m2))] = float(sq / gauss - 1)
    return out


def deficits_torch(fk_np, cfg, todo, context):
    """deficits() with wstfast.measure_torch on ``context['device']`` (256^3 throughout; matches the CPU to ~1e-8)."""
    import torch

    from wstfast.measure_torch import wavelet_modulus as modulus_torch

    lattice = context["lattice"]
    fk = torch.as_tensor(np.asarray(fk_np, dtype=np.complex64), device=context["device"])
    cell = BOXSIZE / NMESH
    out = {}
    for ell in sorted({p[2] for p in todo}):
        for j1 in sorted({p[0] for p in todo if p[2] == ell}):
            _, per_m1 = modulus_torch(fk, lattice, cfg.sigma(j1) / cell, ell, los=True)
            for m1, u in enumerate(per_m1):
                mean = float(u.to(torch.float64).mean())
                fu = torch.fft.rfftn(u - mean)
                fu[0, 0, 0] = mean * NMESH**3
                for _, j2, _ in (p for p in todo if p[0] == j1 and p[2] == ell):
                    _, per_m2 = modulus_torch(fu, lattice, cfg.sigma(j2) / cell, ell, los=True)
                    for m2, y in enumerate(per_m2):
                        a = y.to(torch.float64)
                        sq, s2 = float((a**Q).mean()), float((a * a).mean())
                        n = 1 if m2 == 0 else 2
                        gauss = (2 * s2 / n) ** (Q / 2) * np.exp(gammaln((n + Q) / 2) - gammaln(n / 2))
                        out[repr((j1, j2, ell, m1, m2))] = float(sq / gauss - 1)
    return out


def torch_context(device):
    from wstfast.measure_torch import TorchLattice, default_device

    device = default_device(device)
    return dict(device=device, lattice=TorchLattice(NMESH, device=device))


def run_seed(seed, only=None, cache=CACHE, context=None, bias_path=BIASES):
    from cosmoprimo import Cosmology

    path = cache / f"seed_{seed:04d}.json"
    if path.exists():
        return seed, json.loads(path.read_text())
    t0 = time.time()
    cfg, meta = config()
    cosmo = Cosmology(engine="class", m_ncdm=0.0, **QUIJOTE_COSMOLOGY)
    klin = np.geomspace(1e-4, 10.0, 1024)
    pklin = cosmo.get_fourier().pk_interpolator(of="delta_m")(klin, z=Z)
    f = float(cosmo.growth_rate(Z))
    pk = lambda x: np.exp(np.interp(np.log(x), np.log(klin), np.log(pklin)))  # noqa: E731
    lattice = Lattice(NMESH)
    todo = pairs(cfg, only)
    harmonics = {ell: lattice.harmonics(ell) for ell in range(1, 5)} if context is None else None
    window = cic_window(lattice)
    b = biases(bias_path)
    spt = SPTFields(lattice, pk, BOXSIZE, seed=20_000 + seed)
    fields = HaloRSDFields(spt, f, b)
    shot = (1 + b["alpha0"]) * BOXSIZE**3 / meta["nparticles"]
    noise = _rfft(np.random.default_rng(30_000 + seed).standard_normal((NMESH,) * 3)) * np.sqrt(shot * NMESH**3 / BOXSIZE**3)
    s1, s2, s3 = (window * s for s in (fields.s1, fields.s2, fields.s3))
    t1 = time.time()
    if context is None:
        measure = lambda fk: deficits(fk, cfg, todo, lattice, harmonics)  # noqa: E731
    else:
        measure = lambda fk: deficits_torch(fk, cfg, todo, context)  # noqa: E731
    curve = []
    for eps in EPS:
        if eps == 0:
            curve.append(measure(s1 + noise))
            continue
        plus = measure(s1 + eps * s2 + eps**2 * s3 + noise)
        minus = measure(-s1 + eps * s2 - eps**2 * s3 - noise)
        curve.append({k: 0.5 * (plus[k] + minus[k]) for k in plus})
    design = np.stack([np.ones(len(EPS)), np.square(EPS), np.power(EPS, 4)], axis=1)
    result = {"gauss": {}, "tree": {}}
    for key in curve[0]:
        d0, d2, _ = np.linalg.solve(design, [c[key] for c in curve])
        result["gauss"][key], result["tree"][key] = float(d0), float(d2)
    result["meta"] = dict(f=f, shot=shot, biases={k: float(v) for k, v in b.items()})
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    print(f"seed {seed} done in {time.time() - t0:.0f} s (fields {t1 - t0:.0f} s, WST {time.time() - t1:.0f} s)", flush=True)
    return seed, result


def parse_seeds(text):
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0-39")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", nargs="+", default=None, metavar="J1,J2",
                        help="measure only these (j1, j2) pairs, cached separately (e.g. 6,9: the halo data vector)")
    parser.add_argument("--biases", type=Path, default=BIASES, help="JSON of the template biases (see biases())")
    parser.add_argument("--device", default=None,
                        help="measure with torch on this device (cuda, cpu, auto), one process looping over the seeds; "
                             "default: NumPy / SciPy with --workers processes")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    only = [tuple(int(x) for x in p.split(",")) for p in args.only] if args.only else None
    cache = CACHE if only is None else CACHE.with_name(CACHE.name + "_" + "_".join(f"{a}{b}" for a, b in only))
    if args.device is not None:
        context = torch_context(args.device)
        for seed in seeds:
            run_seed(seed, only, cache, context, args.biases)
    else:
        n = len(seeds)
        with ProcessPoolExecutor(args.workers) as pool:
            for seed, _ in pool.map(run_seed, seeds, [only] * n, [cache] * n, [None] * n, [args.biases] * n):
                pass
    print("done")


if __name__ == "__main__":
    main()
