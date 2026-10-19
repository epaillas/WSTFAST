# Refactor plan: one model, with real space and matter as limits

Goal for the code release: a single model of the WST coefficients and the power spectrum of a biased tracer in
redshift space. The other cases are parameter limits of that model, not separate code:

| case                        | limit of the full model                                        |
|-----------------------------|----------------------------------------------------------------|
| matter (instead of tracer)  | beta = (1, b1=1, b2=bG2=bGamma3=0), no shot noise, no cubic biases |
| real space                  | f = 0, no FoG (c_v = 0), mu-counterterms c2 = c4 = 0, no AP     |
| real-space matter           | both                                                           |

Decided 2026-10-18: real space becomes the f = 0 limit of the redshift-space halo stack, and small numerical
changes to the old real-space matter results are accepted.

---

## 1. Where things stand

There are two independent model stacks. The halo one already extends the matter one; the real-space one shares
almost nothing with either.

### A. Real-space matter (oldest, m-summed S1 / S21)

| layer        | code |
|--------------|------|
| one loop     | `theory/perturbation.py` `OneLoopMatter` (own P22/P13, continuum) |
| S1 cumulants | `theory/cumulants.py` `ZeroLagCumulants` (own F2/G2/F3 kernels, isotropic Edgeworth) |
| moduli       | `theory/moduli.py` `ModulusClustering`, `theory/modulus_noise.py` `ModulusNoise` (tree dN) |
| basis        | `theory/model.py` `WSTMatterBasis` (continuum band integrals) |
| assembly     | `theory/model.py` `Assembly` (cs2, noise_j_l, layer2 template / Padé) |
| calculators  | `WSTBasis`, `WSTTheory`, `PowerBasis`, `PowerTheory` (`theory/power.py` `LatticeBinning`) |
| inference    | `inference.py` `build_wst_theory` / `build_power_theory` / `build_joint_likelihood` |
| covariance   | `theory/covariance.py` (`ShellCovariance`, `analytic_covariance`) |
| scripts      | `fit_power.py`, `fit_joint.py`, `fisher.py`, `analytic_covariance.py`, `layer2/fit.py`, `layer2/template.py`, `scripts/paper/common.py` |

### B. Redshift-space matter and halos (line-of-sight resolved S1m / S21m)

| layer        | matter                                   | halos (subclass or parallel)                    |
|--------------|------------------------------------------|-------------------------------------------------|
| one loop     | `eft_loop.py`, `loop_tables.py`, `ir.py` | same, `biased=True`                             |
| P_s grid     | `rsd.py` `RSDGrid`                       | `rsd_bias.py` `BiasedRSDGrid(RSDGrid)`          |
| S1m          | `rsd.py` `RSDS1Basis`                    | `rsd_bias.py` `BiasedRSDS1Basis(RSDS1Basis)`    |
| moduli       | `rsd_moduli.py` `ModulusSpectraRSD`      | `BiasedModulusSpectraRSD(ModulusSpectraRSD)`    |
| S21m         | `rsd_moduli.py` `RSDS21mBasis`           | `BiasedRSDS21mBasis` (parallel, not a subclass) |
| connected noise | —                                     | `connected_noise_rsd.py` (only feasibility scripts and tests) |
| calculators  | `RSDBasis(tracer="matter")`              | `RSDBasis(tracer="biased")`: two code paths inside one `__call__` |
| theories     | `S1mTheory`, `S21mTheory`, `RSDPowerTheory`, each with `tracer=` branches |
| covariance   | `covariance_rsd.py` (`RSDWSTCovariance`, `ModulusCrossRSD`), plus `RSDGaussianCovariance` in `covariance.py` |
| inference    | built inline in `scripts/fit_rsd.py` (no `inference.py` builder) |

`rsd_bias.py` already says it: with beta = (1, 1, 0, 0, 0) and no shot noise, every biased piece reduces to the
matter one. So the matter redshift-space classes are already redundant. They only survive as base classes and as
the `tracer="matter"` branches.

### Problems an external user would hit

1. **Two models of the same quantity.** S1 in real space is computed by `WSTMatterBasis` (continuum integrals,
   `OneLoopMatter`, isotropic cumulants). In redshift space it comes from `RSDS1Basis` (lattice sums, `eft_loop`,
   block cumulants). Nothing ties them together at f = 0.
2. **The same perturbation-theory kernels are written five times:** `perturbation.f2_kernel`,
   `cumulants._f2/_g2/_f3`, `eft_loop._Kernels`, `bias._BiasedKernels`, and `connected_noise_rsd._Tracer.z1/z2/z3`.
   `wavelet_weight` is defined in both `model.py` and `covariance.py`. Real spherical harmonics are defined in
   `measure.py`, `covariance.py`, `rsd.py`, `rsd_moduli.py` and `survey.py`.
3. **Too many flags.** `RSDBasis` takes `tracer`, `cubic_biases`, `fog`, `ap_order`, `s21_damping` and
   `s21_projected`. Eight `raise ValueError` lines encode which combinations are allowed. `S21mTheory` takes
   about 30 keyword arguments, most of them exploratory (`k2_noise`, `connected_amplitude`, `response_amplitude`,
   `bispectrum_shot`, `noise_shot`, `full_norm`, ...). The paper runs use `--fog --ap --ap-order --cubic-biases
   --s21-white --s21-layer2*` and none of the others.
4. **Magic column layouts.** The biased basis outputs are flat monomial blocks (`KAPPA_SIZES`, `S21_SIZES =
   (70, 70, ...)`, alpha-major AP terms). They are unpacked by helpers spread over `rsd_bias.py` and
   `calculators.py` (`biased_rows`, `split`, `basis_layout`, `ap_evaluate`, `collapse_fog`, ...).
5. **Two entry points.** The real-space fits go through `inference.py`. The redshift-space fits are assembled in
   a 783-line script (`fit_rsd.py`) that reads the model options back from the emulator's `settings` dict.
6. **Compatibility code for old emulators** in `RSDBasis.tree_unflatten` (emulators trained before S21m existed,
   or before the unprojected field grids).

---

## 2. Target design

### Principle: limits are settings, not subclasses

The matter and real-space cases are *values of the parameters* of the full model. Subclassing (as
`Biased*(Matter*)` does today) keeps two implementations alive, and that is the duplication we want to remove.
There will be one implementation, plus presets that fix parameters:

```python
spec = ModelSpec(tracer="biased", space="redshift", fog=True, ap_order=3, cubic=True, ir=True)   # paper baseline
spec = ModelSpec.matter(space="real")   # = tracer="matter", space="real": fixes b1=1, b2=bG2=bGamma3=0, N=0, f=0
```

`ModelSpec` is a small frozen dataclass in `config.py`, next to `WSTConfig`. It is validated once in
`__post_init__`, which replaces the scattered `raise ValueError` lines. It is written to emulator metadata, which
replaces the free-form `settings` dict that `fit_rsd.py` reads back.

- `tracer="matter"` does not select different code. The theories fix the bias, stochastic and cubic parameters to
  their matter values, and the basis drops the shot-noise columns it does not need (an optimization only).
- `space="real"` makes the basis evaluate at f = 0 instead of the cosmology's f. The theories then fix c2, c4,
  c_v and AP. A real-space emulator is a separate training run of the same basis.

Inheritance stays only where it is a genuine extension, such as the survey-window wrapper around the box
theories.

### Modules (merged and renamed, no new physics)

```
wstfast/theory/
  eft_loop.py      one-loop integrals; THE kernels (F_n, G_n, Z_n, biased Z_n, cubic operators)
                   <- absorbs perturbation.py, cumulants._f2/_g2/_f3, connected_noise_rsd._Tracer kernels
  bias.py          monomial algebra of beta (unchanged)
  loop_tables.py, ir.py, ap.py   unchanged
  spectrum.py      <- rsd.py grid part + rsd_bias.BiasedRSDGrid: P_s(k, mu) of the tracer, lattice projections,
                      multipoles, AP projections                                   (rename of rsd.py)
  first_layer.py   <- rsd.py S1m part + rsd_bias.BiasedRSDS1Basis + cumulant corrections
  second_layer.py  <- rsd_moduli.py + rsd_bias.BiasedModulusSpectraRSD / BiasedRSDS21mBasis
                      + connected_noise_rsd.py (replaces modulus_noise.py at f = 0)
  layout.py        <- named views of the basis columns (kappa, skewness, S21m blocks, AP terms): one place that
                      knows the monomial layout. Today this is spread over rsd_bias.py and calculators.py
  covariance.py    <- covariance.py + covariance_rsd.py (one Gaussian-chaos covariance, f = 0 included)
  harmonics.py?    one real-spherical-harmonic implementation (or put it in measure.py and import it from there)
deleted: model.py, perturbation.py, cumulants.py, moduli.py, modulus_noise.py, rsd_bias.py, rsd_moduli.py,
         covariance_rsd.py, connected_noise_rsd.py, power.py (LatticeBinning -> spectrum.py)
```

The guiding rule is to extend existing files rather than add new ones. `layout.py` is the one genuinely new
module, and it can go into `bias.py` instead if you prefer.

### Calculators (`calculators.py`)

```
CosmoprimoCosmology -> Basis(spec, config, coefficients, ...)       one class, replaces WSTBasis / PowerBasis / RSDBasis
                         |-> PowerTheory   (P_l; real space: l = 0 only)   replaces PowerTheory / RSDPowerTheory
                         |-> S1Theory      (S1m blocks)                    replaces WSTTheory (S1 part) / S1mTheory
                         |-> S21Theory     (S21m)                          replaces WSTTheory (S21 part) / S21mTheory
                         -> GaussianLikelihood
```

- **Shared nuisance registry.** Bias, counterterms, stochastic terms, FoG c_v, AP and the second-layer template
  constants are declared once, with priors, in a `NuisanceParameters(spec)` object. Today each theory builds its
  own `Parameter` dicts and matches them by name. The limits above are applied by fixing parameters in this one
  place.
- **Only the options the paper uses get keyword arguments:** the baseline plus the robustness runs actually cited
  (cubic biases, white noise W_j, layer-2 template and its marginalization, AP order, FoG). The exploratory
  options either go or are folded into a single `extensions=` dict marked experimental. Candidates to remove,
  since no paper run uses them: `ng_amplitude`, `k4_counterterm`, `b_shot`, `noise_shot`, `full_norm`,
  `response_amplitude`, `k2_stochastic`, `s21_k2`, `s21_connected*`, `s21_noise_calibration`, `kappa_shot`,
  `shared_counterterms`, `s21_damping`, `s21_projected=True`.
- **One builder.** `inference.build_likelihood(spec, datasets, emulator, ...)` serves both cases. `fit_rsd.py`
  becomes a thin CLI over it (renamed `fit.py`). `fit_power.py`, `fit_joint.py`, `fisher.py` / `fisher_rsd.py`
  and `analytic_covariance*.py` collapse onto the same builder.

---

## 3. Decisions to settle before coding

**D1. The real-space statistic.** *Decided 2026-10-18: (a).* The real-space matter measurements are m-summed only:
868 boxes under `J9_..._n256/real`, and there is no `_los/real`. The m-summed S1 at f = 0 comes out of the
unified model for free (`rsd.s1_rsd`, the anisotropic moment of all blocks, is exact there). The m-summed S21 is
different: it needs the modulus spectra of the *full* modulus |X|, which the redshift-space stack does not have.
Options:

- (a) *Recommended.* Re-measure real-space matter line-of-sight resolved
  (`measure_quijote_wst.py --spaces real --los-resolved`, same boxes) and use S1m / S21m everywhere. Sec. "Matter in
  real space" is then rewritten in S1m / S21m. Its statement that S1m carries no information beyond S1 in real
  space becomes a check of the f = 0 limit. One statistic, one code path. Cost: a GPU measurement campaign, plus
  rerunning the Sec. IV fits and figures.
- (b) Keep m-summed S1 / S21 for real space and add a "summed" field type to `second_layer.py`. This keeps the
  data, but leaves two statistics in the code and adds new model code late in the project.

**D2. Emulator compatibility.** Recommended: keep the output layout of the biased basis byte-identical, so the
paper's halo emulators (`rsd_biased_basis_*`, `ap3_base`) load without retraining. Drop the loaders for
pre-S21m emulators. Retrain only the real-space matter emulator (it is new anyway under D1).

**D3. Naming.** `S1m` / `S21m`, or `S1` / `S21` with the m-resolution implied? Recommended: keep `S1m` / `S21m`,
which matches the paper's notation.

**D4. What happens to the retired and feasibility scripts** (`scripts/feasibility/`, `scripts/layer2/`, `scripts/survey/v0*`).
Recommended: keep them in a `scripts/archive/` that is excluded from the release and not maintained against the
new API, rather than porting them.

---

## 4. Steps

Each step lands as its own commit on `main`, with the test suite green. Steps 0-2 change no numbers.

0. **Freeze references.** *Done 2026-10-18: `tests/test_regression.py` (marked slow; fixtures in `tests/fixtures/`,
   regenerate with `python tests/test_regression.py [halo] [basis] [real]`).* Golden outputs of the current code:
   - the halo baseline (`fit_halo_baseline.sh` settings) with the paper emulator: P_l, S1m and S21m at the
     best fit and at the fiducial values;
   - the real-space matter `WSTTheory` prediction at the fiducial cosmology;
   - one non-emulated `RSDBasis(tracer="biased", fog=True, ap_order=3)` evaluation.

   Save them as regression fixtures in `tests/fixtures/`. Steps 1-4 must reproduce the halo ones to 1e-10.
1. **Kernels and helpers.** *Done 2026-10-18, smaller than first planned.* `connected_noise_rsd` already used the
   `bias.py` kernels. The remaining duplicate kernels (`perturbation.f2_kernel`, `cumulants._f2/_g2/_f3`) and helpers
   (`wavelet_weight`, `legendre`, `log_weights`) live only in real-space modules that step 6 deletes, so they were
   not moved. What was done: the surviving stack no longer imports from those modules (`LegSampler` moved to
   `rsd.py`), `rsd_moduli._harmonic` is replaced by `rsd.spherical_harmonic`, and `covariance.real_harmonics` by
   `measure.real_harmonics`. The fixtures are unchanged. Still depending on real-space modules:
   `covariance.py` (`cross_noise`, `ShellCovariance`: the real-space covariance, replaced in step 3/6).
2. **Collapse matter into biased in redshift space.** *Done 2026-10-18, in three commits.*
   - 2a: the calculators have one code path. `RSDBasis` always builds the monomial basis, and `tracer="matter"` is
     the matter limit taken by the theories (`BiasParameters(tracer="matter")`: b1 = 1, the rest 0, fixed). Emulators
     of the retired matter layout (`rsd_basis_taylor_4p*`, not used by the paper) are refused at load; new ones are
     tagged `layout="monomial"`.
   - 2b: `rsd.py` / `rsd_bias.py` / `rsd_moduli.py` became `spectrum.py` (`SpectrumGrid`, the projections, AP),
     `first_layer.py` (`S1mBasis`, `LegSampler`, harmonics, S1m assembly), `second_layer.py` (`ModulusSpectra`,
     `S21mBasis`, the S21m projections) and `layout.py` (the monomial column layout). Matter-only code was deleted.
     Renames: `biased_scalings` -> `amplitude_scalings`, `BIASED_GRID_TERMS` -> `GRID_TERMS`, and in the calculators
     `biased_rows` / `biased_grid` / `biased_cumulants` -> `basis_rows` / `tracer_grid` / `tracer_cumulants`.
   - 2c: `config.ModelSpec` (tracer, space, ir, fog, ap_order, cubic) validates the model options once. `RSDBasis`
     holds one as `spec`, `train_emulator.py` stores it as `settings["model"]`, and `fit_rsd.py` reads it back
     (`ModelSpec.from_settings` also reads the separate keys of older emulators).

   Gates: the halo and basis fixtures are unchanged to 1e-10. Instead of a fit of the matter boxes, the old matter
   classes were frozen as `tests/fixtures/regression_matter_rsd.npz` before deletion. The matter limit reproduces
   them: cumulants and S21m terms to 1e-10, and the P_l + S1m + S21m theories with `tracer="matter"` to 1e-8. A
   matter emulator of the new layout is trained only when a matter fit is needed.
3. **Real space as f = 0.** *Done 2026-10-18.* `ModelSpec(space="real")` (no FoG, no AP). `RSDBasis(space="real")`
   evaluates the basis at f = 0, and the theories take `space="real"` (`calculators.real_space_limit`: c2, c4 and
   alpha2 fixed at 0; AP, FoG and ck4 refused). Gates, on the 35 S1 coefficients with sigma >= 17.7 Mpc/h, l = 0-6, at
   the Quijote cosmology (matter, cs2 = 0.5):
   - **Blocks:** at f = 0 the |m| block variances of a coefficient agree to <= 7e-5 (relative; l = 1 to 1e-9). The
     residual is the cubic lattice, which the measurement shares. The block Edgeworth factors E_M agree to <= 3e-3,
     the Monte Carlo error of the 2^16-point cumulants. (The 1e-10 first planned holds only in the continuum.)
   - **Against the old real-space S1:** the m-summed S1 from the f = 0 blocks equals the old `WSTMatterBasis` S1
     (`regression_real.npz`) to <= 2e-3. That is <= 0.05 sigma of one 1 (Gpc/h)^3 box at sigma >= 25 Mpc/h (<= 0.4
     sigma at the 60 (Gpc/h)^3 of the fits), and <= 0.13 box sigma at 17.7 Mpc/h (l = 6). The largest relative
     difference, at sigma = 70.7 Mpc/h and l = 0, is the lattice sum at low k, and there the new model is closer to
     the 868-box mean (-0.017 against +0.028 box sigma). `test_real_space_is_the_f0_limit` checks this on 16
     coefficients.
   - **Connected noise:** the f = 0 limit of `connected_noise_rsd` against the old `ModulusNoise` was already a
     test (`test_real_space_matter_limit`, to 2-3e-3). Its reference module goes in step 6, so freeze it as a fixture
     there.

   Real-space fits still need a real-space emulator (`train_emulator.py --stat rsd --space real`) and the line-of-sight
   resolved real-space boxes (D1). The data subdirectories are named `real` / `rsd` while `ModelSpec.space` uses
   `real` / `redshift`; reconcile them in step 4.
4. **Calculators and inference.** *Done 2026-10-18.*
   - `Basis` (was `RSDBasis`; the alias `RSDBasis = Basis` keeps the trained emulators loading, since desilike stores
     the class path). `PowerTheory` (was `RSDPowerTheory`), `S1mTheory` and `S21mTheory` take
     `nuisance=NuisanceParameters(spec, cubic=...)`. That registry declares the biases, counterterms, c_v, S21m noise
     amplitudes and W_j with their priors, and applies the matter and real-space limits; it replaces `BiasParameters`,
     the per-theory parameter helpers and `real_space_limit`. The former real-space P(k) theory is
     `LegacyPowerTheory` until step 6.
   - Pruned (used by none of the 107 paper and survey fit summaries): `ng_amplitude`, `k4_counterterm`, `b_shot`,
     `noise_shot`, `full_norm`, `response_amplitude`, `k2_stochastic`, `s21_k2`, `s21_connected*`,
     `s21_noise_calibration`, `kappa_shot`, `shared_counterterms`, `s21_layer2_prior`, and `fog="separate"`. `--fog`
     is gone: c_v comes from the emulator's `ModelSpec`. The basis options (`s21_damping`, the unprojected S21m the
     survey emulator needs) and `tracer_grid`'s ck4 / alpha_k2 terms (used only by `sample_grids` for the survey
     spectrum basis) are kept.
   - `inference.FitOptions` + `inference.build_fit` (not `build_likelihood`, which names the old real-space builder
     until step 6) hold what `fit_rsd.py` did: `build_problem`, `survey_problem`, the layer-2 template, the
     covariance assembly. `scripts/fit.py` (was `fit_rsd.py`) is the command line, sampling and summary; the summary
     records the options and the `ModelSpec`. A real-space emulator fits `<data-dir>/real` (P(k): the monopole).
     Scripts that imported `fit_rsd` now use `fit.fit_options(fit.parse_args(...))` and `wstfast.inference`.

   Gates: halo fixtures unchanged to 1e-10 through the new command line; matter fixture unchanged; and the end-to-end
   profile of `fit_halo_baseline.sh --volume 60` reproduces `outputs/inference/rsd/fullap_ap_V60` (chi^2 19.253
   against 19.252, best fit equal to the optimizer tolerance).
5. **Real-space data and fits** (per D1). *In progress (2026-10-18).* The 868 line-of-sight-resolved real-space boxes
   are measured (`J9_..._los/real`). The real-space emulator `outputs/emulators/real_matter_basis_taylor_4p_ir.h5`
   (`--space real --tracer matter --ir`, 4 parameters, S1 >= 12.5 Mpc/h, S21 sigma_j1 >= 12.5, ratio 2.8) validates
   to 3.5e-4 (S1m), 1.4e-3 (S21m) and 6.6e-3 (P_0). First fits (`outputs/inference/real/`, no layer-2 template):
   - S1m alone is unbiased to V = 64 (<= 0.5 sigma, chi^2 13 for 57 dof);
   - S1m + S21m with sigma_j1 >= 17.6 is unbiased in 2 parameters (<= 0.8 sigma at V = 64), but chi^2 = 495 for 191 dof
     at V = 64 (329 with W_j). The residuals are positive in every S21m group, grow with l, and are largest at
     sigma_j1 = 17.7;
   - sigma_j1 = 12.5 (the old baseline) fails (+2.9 sigma at V = 8), and so does the 4-parameter conservative fit at
     V = 60 (n_s +1.3 sigma, chi^2 460);
   - the missing ingredient is the m-resolved second-layer non-Gaussianity of real-space matter (the old m-summed
     model had a template; template_rsd.json exists for halos only).

   With the real-space matter template (`outputs/layer2/template_real.json`, 6c35f33: 100 PT seeds, c_l calibrated on
   the adjacent pairs, held-out residual 0.17 single-box sigma at j6->j9, 0.4-0.5 at sigma_j1 = 17.7 and ~0.9 at 12.5)
   plus W_j, i.e. the halo-baseline recipe:
   - conservative (S1m >= 25 + S21m sigma_j1 >= 17.6, 222 points): chi^2 = 133 for 189 dof at V = 64, with
     omega_cdm +0.13 sigma and lnA +0.21 sigma (old m-summed conservative: -0.34, +0.39). With 4 parameters at V = 60:
     omega_cdm -0.7, lnA +0.7, n_s +1.05, h -0.2 sigma, chi^2 126 for 187. Errors at V = 60: sigma(omega_cdm) 0.0038,
     sigma(lnA) 0.027, sigma(n_s) 0.020, sigma(h) 0.0060, against P_0 (k <= 0.1, same model) 0.0032, 0.028, 0.018,
     0.0064;
   - baseline (sigma_j1 >= 12.5, 384 points), template only: unbiased at V = 64 (-0.03, +0.09 sigma) but chi^2 = 780
     for 339 dof, so sigma_j1 = 12.5 is not described even with the template;
   - template without W: conservative chi^2 222 at V = 64 (omega_cdm -0.5, lnA +0.56 sigma).
   Decided 2026-10-18: the Sec. IV baseline is S1m >= 25 + S21m sigma_j1 >= 17.6 with the template and W. A draft of
   Sec. IV is in paper/main_v4.tex (untracked, with paper/figures/real_*.pdf and paper/generated/numbers_real.tex from
   scripts/paper/real_space.py). In that draft the baseline stays within 0.2 sigma up to V = 64 (chi^2 133 for 189 dof),
   and the extended vector is biased by +1 sigma at V = 1. At V = 60 with 4 parameters, the WST does not beat P_0 to
   k = 0.1 on omega_cdm (0.0038 against 0.0032). For l = 1 the tree response (Eq. response_rsd, no 2 F2 P P term)
   misses the change of sign at K sigma ~ 2. The analytic-covariance section, the Delta N appendix and two discussion
   paragraphs still describe the m-summed matter model.
   Revised 2026-10-18: the Sec. IV baseline varies omega_cdm, logA and h (n_s fixed), as for halos. The posterior panel
   is at V = 60 (WST baseline against P_0 to k = 0.1, 0.2), and the sigma_j1 = 12.5 vector appears only in the shift
   panel. In 3 parameters the baseline stays within 0.22 sigma up to V = 64, and the sigma_j1 = 12.5 vector is biased by
   +0.6 sigma at V = 1 and +1.8 sigma at V = 64, with no gain in the errors.

   Original plan for this step: Measure, train the real-space emulator, rerun the Sec. IV fits and
   figures (`scripts/paper/` moves onto the new API), and update the paper text and numbers. This is the only
   step that changes published numbers.
6. **Delete the old stack.** Remove `model.py`, `perturbation.py`, `cumulants.py`, `moduli.py`,
   `modulus_noise.py`, the old calculators and `inference` builders, and the scripts that only fed them. Archive
   the feasibility scripts (D4).
7. **Release polish.** README with a 20-line quick start (load emulator, build likelihood, fit one box), module
   docstrings rewritten for the new layout, a public API in `wstfast/__init__.py`, and tests that run in a few
   minutes without the Quijote data (small synthetic config).

## 5. Risks

- **f = 0 edge cases.** Not yet tested in the redshift-space code. The velocity damping and FoG columns scale
  with f and should vanish cleanly, but IR resummation in redshift space and the AP polynomial need checking at
  f = 0.
- **Changes in Sec. IV.** The lattice and `eft_loop` changes may move the real-space S1 at the 0.1-1% level.
  Under (a) the real-space fit is a new analysis anyway, but the Sec. IV narrative (the model ladder, building
  blocks, the layer-2 template at 8.8 Mpc/h) has to be regenerated.
- **Scope creep.** Steps 0-4 do not touch the paper and can stop at any point with a cleaner code base. Steps
  5-6 are tied to the paper timeline.
