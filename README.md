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
   - Files go to `data/quijote/z0.5/<tag>/{real,rsd}/` (e.g. `J9_L6_L2-4_dj2_sigma0.8_step1.414_n256`). Existing files are skipped.
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

## Model

The model is in `wstmodel/theory/model.py`, for real-space matter only. Its data vector has two
parts:

- **S1(j, l)** at Gaussian order.
  - The variance uses the linear plus one-loop SPT power spectrum.
  - A counterterm `cs2`, the CIC window and the particle shot noise are included.
- **Reduced second-order coefficients S2(j1, j2, l) / S1(j1, l)**, for l ≥ 1.
  - The first-layer modulus field is a biased tracer.
  - Its scale-dependent response is computed from the tree-level bispectrum, with no free
    parameter.
  - Its noise is (1 + `noise_l`) times the Gaussian-chaos value.
  - The second layer is taken to be Gaussian.

Default scale cuts keep S1 at σ_j ≥ 25 Mpc/h and S2/S1 at σ_j1 ≥ 12.5 Mpc/h. That gives 22
coefficients; set the cuts with `--s1-min-scale` and `--s21-min-scale`.

The cosmology-dependent band integrals (`WSTBasis`) are what the emulator replaces. The nuisance
parameters stay exact in `WSTTheory`, as in the DSC model.

Not yet included:
- Edgeworth corrections to S1 (these need the zero-lag tree trispectrum);
- one-loop responses;
- redshift-space distortions;
- galaxies.

## Layout

- `wstmodel/`
  - `config.py`: WST settings and coefficient selection.
  - `measure.py`: the WST estimator.
  - `quijote.py`: snapshot reading and CIC painting.
  - `data.py`: storage, data vectors and covariance.
  - `theory/`: SPT loops, modulus-field responses and model assembly (JAX).
  - `calculators.py`: desilike calculators and likelihood.
  - `inference.py`: profiling, MH sampling, summaries and plots.
- `scripts/`: the three pipeline steps. `scripts/feasibility/` holds the diagnostics of the
  feasibility study.
- `tests/`: run with `python -m pytest`.
