# WSTFAST

Perturbative modelling of the 3D solid-harmonic wavelet scattering transform (WST)
of the matter field, following the approach of the sibling `dsc-model` project.
The motivation and the numbers behind every modelling choice are in
[docs/wst_eft_feasibility.md](docs/wst_eft_feasibility.md).

Installation (in the `desi-clustering-classpt` environment):

```bash
pip install -e . --no-deps
```

## Pipeline

1. **Measure** the WST of Quijote CDM snapshots, one `.npz` per realization, space and scale configuration:

   ```bash
   python scripts/measure_quijote_wst.py --superset --realizations 0 1 10 100 1000 10000-10004
   ```

   - **The superset** is the production setting.
     - Scales: σ_j = 3.1·2^(j/2) Mpc/h, for j = 0…9.
     - First layer: l ≤ 6. Second layer: l ≤ 4, with σ_j2/σ_j1 ≥ 2.
     - Exponents q = 0.5, 0.8, 1 and 2 are stored in the same file.
     - Spectra: the power-spectrum multipoles of every first-layer modulus with δ, their auto-spectra, and P_δδ.
     - Every dyadic configuration with σ0 = 0.8 cells (including J = L = 4) is the subset of even j.
     - It is used to choose the configuration offline, by Fisher information and model validity.
   - Single configurations: `--J --L --L2 --min-dj --step --sigma0 --q`.
   - Files go to `data/quijote/fiducial/z0.5/<tag>/{real,rsd}/` (e.g. `J9_L6_L2-4_dj2_sigma0.8_step1.414_n256`). Existing files are skipped.
   - `--backend torch` runs painting and the WST with torch, on a GPU when available (`--device`).
     It reproduces the numpy backend to about 1e-7 and caches the spherical harmonics once per run.
   - For the 1500 boxes of the DSC P(k) run (ids in `scripts/slurm/quijote_dsc_realizations.txt`), use
     `scripts/slurm/measure_quijote_wst_gpu.sh` (GPU nodes, one process per GPU) or
     `scripts/slurm/measure_quijote_wst_cpu.sh`.

2. **Train the Taylor emulator** of the cosmology-dependent part of the model:

   ```bash
   python scripts/train_emulator.py --vary omega_cdm logA
   ```

   - The WST settings, redshift and particle shot noise are read from the measurements.
   - Writes `outputs/emulators/wst_basis_taylor.h5`, plus a `.json` with the settings and a
     `.validation.json`.
   - Order 3 over ±15% about the Quijote fiducial reproduces the default data vector to better than
     1e-4.

3. **Infer** with desilike: a Minuit profile, then Metropolis–Hastings chains started from it.

   ```bash
   python scripts/inference.py --output-dir outputs/inference/fiducial
   python scripts/inference.py --emulator none --method profile   # exact model, CLASS at each step
   ```

   - Writes `profiles.h5`, `chains/`, `samples.h5`, `summary.json` and `bestfit.png`.
   - The covariance is that of one (1 Gpc/h)³ box by default; `--covariance-of-mean` gives the
     errors of the realization mean instead.
   - With fewer realizations than data points + 3, `--covariance auto` (the default) falls back to
     a diagonal covariance and warns. A usable full covariance needs the WST of many more Quijote
     boxes, measured on a cluster.

## Matter power spectrum fits

The real-space matter P(k) of the same boxes (`Pdd`, stored in the WST files) can be fitted with
the same machinery, so P(k) and WST posteriors can be compared on equal footing. The model is the
one of `dsc-model`.

```bash
python scripts/train_emulator.py --stat pk --data-dir data/quijote/fiducial/z0.5/J9_L6_L2-4_dj2_sigma0.8_step1.414_n256 \
    --vary omega_cdm logA --output outputs/emulators/pk_taylor.h5
python scripts/fit_power.py --kmax 0.15 --output-dir outputs/inference/pk/kmax0.15
python scripts/fit_power.py --kmax 0.15 --emulator none --method profile   # exact model
```

- **Model**: P = P_L + L_Λ − 2 `cs2_pk` k² P_L.
  - The one-loop L_Λ uses dsc-model's code, vendored in `wstfast/theory/eft_loop.py`: EdS kernels
    with the exp4 loop regulator at Λ = 0.5 h/Mpc.
  - `--cutoff none` gives unregulated SPT, the loop of the WST model. The two differ mainly by a
    shift of `cs2_pk` (≈ 0.5 (Mpc/h)²).
  - `--order tree` is linear theory, with no free parameter by default (`--counterterm` adds one).
  - Train a separate emulator with matching `--pk-order` / `--cutoff` for those variants.
- **Binning** (`wstfast/theory/power.py`): the prediction is averaged over the lattice modes of each
  bin. It includes the CIC window, which is not deconvolved in the data, and the known aliased particle
  shot noise. This replaces dsc-model's window matrix.
- **Emulator**: the basis is normalised to the fiducial A_s (P_L ∝ A_s, L ∝ A_s²), so the Taylor
  emulator is exact in logA. It is trained on k nodes up to `--kmax` (0.3 by default) and serves any
  fit with a smaller kmax.
- `scripts/fisher.py` uses the same P(k) model (`--pk-order`, `--pk-cutoff`). Its P(k) errors agree
  with the MH posteriors to a few per cent for kmax ≥ 0.15.

## Model

The model is in `wstfast/theory/model.py`, for real-space matter only. Its data vector has two
parts:

- **S1(j, l)** at one-loop order.
  - The Gaussian-limit expression uses the linear plus one-loop SPT variance.
  - A counterterm `cs2`, the CIC window and the particle shot noise are included.
  - The next-to-leading Edgeworth correction uses the zero-lag tree trispectrum and squared tree
    bispectrum of the wavelet vector (`wstfast/theory/cumulants.py`), with no free parameter.
- **Reduced second-order coefficients S2(j1, j2, l) / S1(j1, l)**, for l ≥ 1.
  - The first-layer modulus field is a biased tracer.
  - Its scale-dependent response is computed from the tree-level bispectrum, with no free
    parameter.
  - Its noise is the Gaussian-chaos value plus its tree-level non-Gaussian correction
    (`wstfast/theory/modulus_noise.py`), checked against the exact tree level measured on
    perturbation-theory fields (`scripts/feasibility/pt_modulus_noise.py`).
  - A free amplitude `noise_j{j1}_l{l}` per first-layer field scales the Gaussian-chaos part and
    absorbs what is beyond tree level.
  - The second layer is taken to be Gaussian.

Default scale cuts keep S1 at σ_j ≥ 25 Mpc/h and S2/S1 at σ_j1 ≥ 12.5 Mpc/h with σ_j2 / σ_j1 ≥ 2.8
(adjacent pairs, whose second-layer non-Gaussianity is not perturbative, are left out). On the
half-octave superset that gives 44 coefficients; set the cuts with `--s1-min-scale`, `--s21-min-scale`
and `--s21-min-ratio`.

The cosmology-dependent band integrals (`WSTBasis`) are what the emulator replaces. The nuisance
parameters stay exact in `WSTTheory`, as in the DSC model.

Not yet included in this real-space model: one-loop responses. Redshift space (line-of-sight-resolved
S1m, S21m and P_ℓ, `wstfast/theory/rsd.py`, `rsd_moduli.py`, `scripts/fit_rsd.py`) and biased tracers are
handled by the redshift-space model below.

## Biased tracers (Quijote FoF halos)

The redshift-space model extends to Eulerian-biased tracers (`wstfast/theory/bias.py`, `rsd_bias.py`).
- **Bias expansion:** b1, b2, bG2, bΓ3 at one loop (the CLASS-PT basis, with the b2 parts of P13 and of the
  k → 0 limit of P22 renormalized).
- **Stochastic terms:** (1 + α0) V/N and α2 k²μ² V/N.
- **Cumulants:** the Poisson terms of a discrete tracer enter the S1m cumulants, and the S21m response and
  noise.
- **Emulation:** every basis output is resolved into the coefficients of the monomials of
  β = (1, b1, b2, bG2, bΓ3), so the emulator stays cosmology-only.
- **Matter limit:** with β = (1, 1, 0, 0, 0) and no shot noise, the model is exactly the matter model
  (`tests/test_bias.py`).

```bash
scripts/download_quijote_halos.sh 0 1999                      # Globus, ~26 MB per box
python scripts/measure_quijote_wst.py --tracer halos --mmin 1e13 --superset --los-resolved --spaces rsd --backend torch
python scripts/train_emulator.py --stat rsd --tracer biased --ir --vary omega_cdm logA n_s h \
    --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los \
    --output outputs/emulators/rsd_biased_basis_taylor_4p_ir.h5
python scripts/fit_rsd.py --stats pk s1m s21m --emulator outputs/emulators/rsd_biased_basis_taylor_4p_ir.h5 \
    --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los ...
```

Quijote FoF catalogues stop at 20 particles (1.3e13 Msun/h), so `--mmin 1e13` keeps every halo
(nbar ≈ 3.1e-4 (h/Mpc)^3 at z = 0.5).

## Layout

- `wstfast/`
  - `config.py`: WST settings and coefficient selection.
  - `measure.py`: the WST estimator.
  - `quijote.py`: snapshot and FoF halo reading, CIC painting.
  - `data.py`: storage, data vectors and covariance.
  - `theory/`: SPT loops, zero-lag tree cumulants, modulus-field responses and noise, and model assembly (JAX);
    `eft_loop.py` (vendored from dsc-model) and `power.py` for the matter P(k).
  - `calculators.py`: desilike calculators and likelihood.
  - `inference.py`: profiling, MH sampling, summaries and plots.
- `scripts/`: the three pipeline steps, `fit_power.py` for P(k) fits and `fisher.py`. `scripts/feasibility/` holds the diagnostics of the
  feasibility study.
- `tests/`: run with `python -m pytest`.
