# From periodic boxes to survey geometry: plan

Status: planning draft, 2026-10-12. Nothing here is implemented yet.

Goal: fit the halo/galaxy WST baseline to mock catalogues with a realistic footprint and radial selection n(z). The
baseline is S1m (σ ≥ 25 Mpc/h, q = 0.8) + S21m (σ1 ≥ 25, σ2 ≥ 70 Mpc/h), with cubic biases and a free white term W.
After the mocks, the same pipeline should run on data. This note lists what assumes a periodic box today, how to
remove each assumption, and the validation ladder that has to pass before data.

## 0. Decisions (2026-10-12)

- **Target sample:** DESI DR1 LRG, in the standard redshift bins 0.4–0.6, 0.6–0.8 and 0.8–1.1, NGC and SGC.
- **Weights:** FKP-like (E3).
- **Mocks** are HOD galaxies. For now we assume the halo bias model (cubic biases, W, damping 1 × σ_v) carries over to
  them unchanged. A box-level HOD check is deferred, not dropped (§8).
- **Covariance:** analytic (C2) for now. The EZmocks (C1) come later.
- **Fibre assignment (V4):** deferred.
- **Priority:** recover the AbacusSummit c000 truth from the mean of the 25 DR1 LRG Abacus mocks (V3). The rungs below
  it exist to make that fit trustworthy.

### DR1 inputs (public, `https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/`)

| What | Path | Notes |
|---|---|---|
| Data | `LSS/iron/LSScats/v1.5/LRG_{NGC,SGC}_clustering.dat.fits` | 143 MB per cap. Columns RA, DEC, Z, WEIGHT (completeness × imaging), WEIGHT_FKP, NX |
| Randoms | `LSS/iron/LSScats/v1.5/LRG_{NGC,SGC}_{0..17}_clustering.ran.fits` | ~1 GB each; a few files are enough for the WST |
| Covariance mocks | `mocks/EZmock/dark/v1/mock{1..1000}/LRG_ffa_{NGC,SGC}_clustering.{dat,ran}.fits`, `LRG_ffa_nz.txt` | Fast fibre assignment (ffa), not altmtl |
| Validation mocks | `mocks/AbacusSummit/dark/v4.2/altmtl{0..24}/mock*/LSScats/LRG_{NGC,SGC}_clustering.dat.fits` | Full fibre assignment (altmtl), AbacusSummit c000 cosmology |

These catalogues are ~100 GB in total, so V2 onwards runs at NERSC (the public tree is mirrored there under
`/global/cfs/cdirs/desi/public/dr1/`). V0 and V1 run locally on the Quijote halo boxes.

**Not public, but needed for a clean V3/V4 split:** AbacusSummit LRG cutsky mocks without fibre assignment (the
"complete" version of the same 25 realisations), and the matching LRG HOD cubic boxes. Without them, the light-cone
test (V3) and the fibre-assignment test (V4) can only be run together, on altmtl.

## 1. What assumes a periodic, uniform box today

| Where | Assumption |
|---|---|
| `measure.py`: `Lattice`, `wavelet_modulus`, `measure_wst` | Periodic FFT convolution; a single fixed line of sight (z); moments are averaged over every mesh cell with equal weight; the second layer subtracts one global mean of U |
| `measure.py`: `PowerMultipoles` | Lattice binning, with no window |
| `theory/rsd.py`: `RSDS1Basis`, `lattice_modes` | Block variances λ_\|m\| are sums over the modes of the measurement mesh, with the CIC window and a uniform shot noise V/N |
| `theory/rsd.py`: Edgeworth, `rsd_bias.py`: `--kappa-shot` | Cumulants at a single n̄ |
| `theory/rsd_moduli.py`, `rsd_bias.py`: S21m | P_UU/⟨U⟩² is stationary; λ_2 is a lattice sum; the U noise uses a single n̄ |
| `theory/covariance_rsd.py` | Periodic volume V, Gaussian band powers per lattice mode |
| `theory/ap.py` | Already survey-ready: it maps fiducial-frame k, μ to the true frame. It needs no change, but it has to be applied at each effective redshift |
| Emulator / `fit_rsd.py` | One redshift (z = 0.5); data come from `.npz` box files |

## 2. Strategy: keep the box basis, add an observational layer

The emulated, cosmology-dependent basis stays the box basis. Survey effects are applied at assembly, the way AP is
already applied. Two things make this possible:

1. **Normalise the signal, not the noise.** Measure the overdensity δ(x) = (n_g − α n_r) / (α n_r), not the FKP field.
   Inside the footprint, the signal part of every block variance λ_\|m\| is then the box value. Only the noise part
   varies with position: λ_N(x) ∝ 1/n̄(x), and it can be computed exactly by FFT from the randoms, as
   (|ψ|² ⋆ 1/(α n_r))(x). FKP-type weights move from the field into the average of the moments (see E3).

2. **S1m is a weighted average of the local moment.** At leading order, the measured coefficient is

   S1m = Σ_x w(x) M_q[λ_s + λ_N(x), κ(x)] / Σ_x w(x),

   where M_q is the box moment with its Edgeworth factor, evaluated at the local noise. The noise level and the
   stochastic cumulants depend on n̄(x), and the edge transfer on the distance to the boundary. So the model needs only
   a small weighted histogram of the survey points in (n̄, z, edge class). That histogram is computed once from the
   randoms. Evaluating the model in each bin costs no more than the box model does now, since λ_N is a scalar shift of
   the `shotnoise` column that already exists.

What is not local needs explicit treatment:
- **Edges:** there the wavelet integrates over the region outside the footprint (M2).
- **The U mean in the second layer:** it varies with n̄(x) and would print the selection function into S21m (E4).
- **Redshift evolution** across the bin (M4).

## 3. Estimator features (`wstfast/survey/`)

- **E1. Catalogue → mesh.** Read data and randoms (RA, Dec, z, weights), convert to comoving coordinates with the
  fiducial cosmology (`ap.QUIJOTE_FIDUCIAL`, or a DESI fiducial), and paint with CIC/TSC into a padded box. Only
  σ ≥ 17.7 Mpc/h is needed, so cells can be 6–8 Mpc/h: a DESI LRG z-bin then fits in a mesh of ≤ 640³, which the
  torch backend can handle on one GPU. The rule is σ_min/cell ≥ 2.5. Check whether `acm` / jaxpower already does
  the painting before writing our own.
- **E2. Local line of sight.** S1m and S2m resolve |m| about z. In a survey, the X_lm at each cell are rotated to the
  local line of sight x̂ with the real Wigner-D matrix D^l(x̂), and U_\|m\| is built from the rotated components. This
  is exact pointwise: the wide-angle error inside one wavelet is O(σ/r) ≲ 3% in angle at r ≳ 1 Gpc/h. A test against
  the global-z code on a distant box is part of V1.
- **E3. Masked, FKP-weighted moments.** There are two kinds of weight, and they act at different stages.
  - **Catalogue weights:** WEIGHT multiplies galaxies and randoms at painting, so it enters n_g and n_r and hence n̄(x).
  - **FKP weight:** with δ normalised by the randoms, the FKP weight acts on the volume average of the moments, not on
    the field.
  
  The moment average runs over a per-scale support mask M_j (random density above a threshold, optionally eroded by
  c·σ_j). Each cell gets weight w(x) = [n̄P_0 / (1 + n̄P_0)]², with P_0 = 10⁴ (Mpc/h)³ as in DESI LRG. This is FKP for
  a quadratic statistic: with δ normalised by the randoms, the effective field per cell is n̄wδ and S1m is quadratic
  in it. A natural WST refinement is a per-block weight w_b(x) = [λ_s,b / (λ_s,b + λ_N,b(x))]², using the fiducial
  λ_s. It is optimal per block rather than per P(k) amplitude, and costs nothing extra. Both are switches; V0c picks
  one. The erosion factor c is set in V0b.
- **E4. Second-layer mean subtraction.** Divide U by its mean in bins of n̄(x) (and z), estimated from the data:
  U' = U / Ū(n̄, z) − 1, before the second wavelet. This removes the selection modulation without fixing a cosmology.
  It induces a small integral constraint, which is modelled in M3 and checked in V0.
- **E5. Window products from randoms.** For each (j, l, |m|), compute λ_N(x) and the edge transfer maps (M2), and
  store the weighted histogram that the model consumes.
- **E6. P_l on the cutsky,** for the P_l comparison only: use a standard window-matrix estimator (jaxpower/pypower)
  rather than `PowerMultipoles`. This is lower priority, since the baseline is the WST alone.

## 4. Model features

- **M1. Histogram-averaged S1m.** `s1m_rsd` is evaluated per histogram bin, with the bin's shot noise and Poisson /
  `kappa-shot` cumulants, and the bins are averaged with weights. For a uniform box this reduces exactly to the
  current model (unit test). Survey modes are dense, so the continuum limit replaces the lattice sum there; the box
  lattice sum stays for the box tests.
- **M2. Edge transfer.** Near the boundary, λ(x) = ∫ P(k) |ψ ⋆ (M e^{ik·y})|²(x) differs from λ_box. Three options,
  in increasing cost:
  (a) erosion only, at c ≈ 2–3. This is probably enough for S1m at σ ≤ 50, but it would cost most of the volume at
      the second-layer scale σ2 ≥ 70;
  (b) normalised convolution, dividing by (|ψ| ⋆ M). This is first-order correct;
  (c) an exact transfer kernel per (j, l, |m|): λ is linear in P, so a few band-power basis maps, computed once on
      the randoms (by FFT, or by averaging over Gaussian random fields), give λ(x) for any P.
  The plan is (a) for S1m first, and (c) for S21m.
- **M3. S21m in a survey.** P_UU/⟨U⟩² keeps its box form (response + noise + W). The changes are:
  - the U noise uses the local n̄ (by histogram);
  - λ_2 is projected through the window, with kernel (c) of M2 applied to P_UU, in the same way a P(k) window matrix
    is applied;
  - the E4 mean subtraction removes low-K power, which enters as an integral-constraint term.
  This is the main modelling risk: the σ2 wavelets are 150–200 Mpc/h across, comparable to the radial depth of a
  z-bin.
- **M4. Redshift evolution.** Evaluate the basis at a few z nodes inside the bin, through D(z) and f(z) and the AP
  ratios at each z. Weight the nodes by the z-histogram, with linear evolution of the biases or constant biases per
  bin. This needs an emulator with z as an input, or one emulator per z node. It is tested against the
  single-z_eff approximation in V3.
- **M5. Observational systematics.** Fibre-assignment incompleteness and imaging weights enter only through w and n̄.
  The radial integral constraint, from n(z) taken from the data, is handled like E4. The σ ≥ 25 cut probably makes
  fibre collisions negligible, but this is tested in V4 rather than assumed.

## 5. Covariance

- **C1. Mock covariance** (EZmocks or equivalent, ~1000 realisations, same footprint), with the Hartlap and Percival
  factors already in `wstfast.data`.
- **C2. Analytic survey covariance:** replace V by an effective volume per block, Σ w² / (Σ w)² weighted by the local
  variance. Validate it against C1 as was done for the box (`covariance_vs_nbody_rsd.py`). It is useful for Fisher
  forecasts and as a cross-check of C1.
- **C3. Pixel-level sanity check:** compare cutsky errors with box errors rescaled by volume, on V0 mocks.

## 6. Validation ladder

Each rung isolates one effect and has a known truth. Unless stated, the pass criterion is the one used for the box
baseline: shifts ≤ 0.3σ at V = 60 (Gpc/h)³ on the 1106-box mean, with acceptable χ². Each rung runs the identical fit
in box and survey modes.

| Rung | Mocks | Effect tested | Pass criterion |
|---|---|---|---|
| **V0a** | Gaussian / lognormal fields with known P, cut to a mask | Estimator + M1/M2 algebra, no PT error | Measured λ(x) and S1m match the model to ≪ statistical error |
| **V0b** | Quijote halo boxes (the existing 1106) with a non-periodic sub-box mask in a padded mesh | Edges only (n̄ uniform) | S1m/S21m cosmology unchanged vs periodic; erosion c chosen here |
| **V0c** | Same boxes, downsampled with a radial n̄(z) along one axis (factor ~3–5 variation) | Varying noise, E4 mean subtraction, M1 histogram | Same as box fit, with only larger errors |
| **V1** | Same boxes, placed at r ~ 1–2 Gpc/h from an observer, RSD along the radial direction | Local line of sight E2, wide angle | Matches the plane-parallel fit; local-LOS code reproduces global-z at r → ∞ |
| **V2** | Quijote halo boxes cut to the DR1 LRG footprint and n(z) of one z-bin. The footprint and n(z) come from the v1.5 randoms; the box is replicated if the bin needs more volume, and z is fixed | All geometry together, no evolution, known truth | Pass criterion. Covariance C2 vs these cutsky boxes within ~10% |
| **V3** | AbacusSummit DR1 LRG mocks (25 altmtl; the complete version if it can be obtained) | Redshift evolution M4, HOD galaxies, with fibre assignment if altmtl | Recovers c000 within the pass criterion, scaled to the 25-mock mean |
| **V4** | Abacus altmtl vs complete, paired; EZmock ffa vs complete if available | Fibre assignment, M5, radial integral constraint | Shifts vs the complete mock ≤ 0.2σ |
| **V5** | DR1 LRG data, blinded | End to end | Unblinding after the mock tests and the covariance are frozen |

V0 and V1 need only the Quijote halo boxes already on disk plus randoms that we generate, so they can start now.
V2 needs only the public DR1 randoms. V3 onwards needs the mocks at NERSC (`acm` already reads Abacus catalogues).
The covariance (C1) comes from the 1000 EZmock LRG ffa mocks.

## 7. Order of work

1. `wstfast/survey/`: E1 (painting + randoms), E3 (masked moments), and the V0a Gaussian-field harness. Unit test:
   with a full-box mask the survey path reproduces `measure_wst` exactly.
2. M1 + E5 (λ_N maps, histograms), then V0c with S1m only. This is the cheapest end-to-end check of the strategy in §2.
3. E2 (local line of sight) + V1.
4. V0b: choose the erosion for S1m.
5. E4 + M3 (S21m: mean subtraction, window kernel), then V0b/V0c with S21m. This step carries the most risk.
6. Covariance C2; obtain cutsky mocks; V2.
7. M4 + V3, then M5 + V4.

Steps 1–5 use only data already on disk.

### Status

**Done (2026-10-12):** step 1 and E2.
- `wstfast/survey.py` (estimator) and `tests/test_survey.py`. Exact checks: the box limit reproduces `measure_wst`, a
  constant line of sight matches the rotated box, and a mode along the line of sight is pure m = 0.

**V0a on Gaussian fields** (`scripts/survey/v0a_gaussian.py`). Run on a 200³ mesh of a 2 Gpc/h box: a 1.4 (Gpc/h)³
cone, n̄ = 1.5–6 × 10⁻⁴ (h/Mpc)³, b = 2, f = 0.75, q = 0.8, S1m at σ = 25–71 Mpc/h with l ≤ 4. The edge test (A)
used 4 realisations, the varying-noise test (B) 16. The output is in outputs/survey/v0a.

- **Edges (A).** The table gives the largest edge bias of S1m over blocks, in units of the error of one survey
  (fractional in brackets).

  | Erosion c | σ = 25 | σ = 35 | σ = 50 | σ = 71 Mpc/h |
  |---|---|---|---|---|
  | 0 | 14 (3.8%) | 15 (5.5%) | 15 (7.9%) | 7.8 (10.7%) |
  | 1 | 4.8 (1.6%) | 6.1 (2.4%) | 4.2 (3.3%) | 2.6 (4.3%) |
  | 2 | 1.2 (0.4%) | 0.9 (0.5%) | 0.7 (0.7%) | 0.4 (1.0%) |
  | 3 | 0.09 | 0.04 | 0.02 | 0.05 |

  Erosion by 3σ works, but at σ = 71 it removes 210 Mpc/h from every edge, which is too much for DR1 LRG z-bins.
  So an edge-transfer model (M2) is needed, not optional.
- **Varying noise (B, erosion 3).** All three predictions match the mean: χ² = 54, 46 and 55 for 60 blocks
  (diagonal), with a largest residual of 0.65σ against an error of the mean of 0.25σ. They differ from each other by
  ≤ 0.08σ (exact vs local) and ≤ 0.03σ (exact vs effective). At LRG densities n̄P ≫ 1 on these scales, so the noise is
  a few per cent of λ and S1m is nearly linear in it. **A single effective noise level suffices for S1m.** The
  n̄ histogram of M1 is not needed for the LRGs, though it could still matter for S21m (the noise of U) and for the
  Poisson cumulants.

**M2: window correction from Gaussian mocks on the randoms (replaces the flat-boundary plan).**
`wstfast.survey.GaussianMocks` and `window_ratio`. Gaussian Kaiser fields are drawn with noise at the local n̄
(plane-parallel or wide-angle). Each is measured masked and unmasked on the same cells, and R_b = ⟨S_masked⟩ /
⟨S_unmasked⟩. The survey model is R_b × the box model at the effective noise.

Robustness (`scripts/survey/v0a_window.py`, outputs/survey/v0a_window): R was calibrated at the fiducial spectrum
(b = 2, f = 0.75, 4 mocks), then applied to 6 paired mocks of each variant: b = 1.6 with f = 0.6; amplitude × 1.21;
ω_cdm = 0.145; n_s = 0.90; n̄ / 2. All biases below are in units of one survey's error.

| | Uncorrected edge bias | Corrected, every variant | Spread of R across variants (same seeds) |
|---|---|---|---|
| Erosion 0 | 8–12σ | max 0.65σ, rms 0.20σ | ≤ 0.12σ for the spectrum, ≤ 0.22σ for n̄/2 (at σ = 25) |
| Erosion 1σ_j | 3–5σ | max 0.31σ, rms 0.09–0.10σ | ≤ 0.09σ |

The residual common to all variants is the noise of the 4-mock calibration plus that of the test. More calibration
mocks would reduce it as 1/√N.

The density dependence doesn't matter in practice, because R is computed with the real n̄ of the randoms. What is
left is a b/f dependence of ≤ 0.17σ at erosion 0 (≤ 0.08σ at 1σ_j).

**Baseline (decided 2026-10-12):** no erosion. R comes from ≥ 16 (here 32) mocks at the fiducial spectrum and is
recomputed once at the best fit. Erosion by 1σ_j is the fallback if the halo tests (V0b/V0c) show that R from
Gaussian mocks misses non-Gaussian edge effects.

**V0b/V0c on Quijote halos** (`scripts/survey/v0b_halos.py`, no erosion, 40 boxes, 32 calibration mocks). Each
redshift-space halo box is tiled 2×2×2 into the V0a cone. V0b uses all halos; V0c thins them to the radial n̄(r)
and uses FKP weights. The table gives residuals in units of one survey's error, as max / rms over blocks, with
χ² for 60 blocks (error ≤ 0.10σ):

| R from | V0b (uniform) | V0c (radial) |
|---|---|---|
| Uncorrected | 5.5–6.7σ | 4.4–5.6σ |
| Gaussian mocks | 0.34 / 0.17σ, χ² 625 | 0.12 / 0.04σ, χ² 68 |
| Gaussian + CIC window (aliased shot noise) | 0.18 / 0.09σ, χ² 179 | 0.23 / 0.10σ, χ² 310 |

The correction removes ~97% of the edge bias for halos. The rest is a coherent ~0.1σ per block that depends on
the sample:
- Relative to Gaussian mocks, the halo edge effect is +4% (V0b) or +0.4% (V0c).
- The CIC window moves the mock R by +3% at σ = 25 and +1% at σ = 71, in both geometries.
- Sub-Poisson noise and lognormal mocks don't explain it (`scripts/survey/v0b_diagnose.py`).

**Decision (2026-10-12): R is calibrated on survey-matched mocks of the tracer, as DESI does for the P(k) window.**
- **Mocks:** HOD boxes are cut to the survey geometry (`SurveyGeometry` from the randoms with the n(z) of the sample,
  then `box_to_survey`). Each box is tiled, given RSD along the local line of sight, and thinned to n̄(x).
- **Calibration:** `calibrate_window` measures every mock masked and unmasked on the same cells.
  R = ⟨S_masked⟩/⟨S_reference⟩ and the survey model is R × the box model.
- **Why pairing:** the reference is the parent box, so pairing cancels cosmic variance and a few tens of mocks are
  enough.
- **Gaussian mocks** (`GaussianMocks`, `window_ratio`) stay as a fast cross-check.
- **Needed for DR1:** the parent AbacusSummit LRG HOD boxes (c000, as many phases as possible, at the snapshots of
  the z-bins). The public cutsky mocks alone cannot be paired with their parent boxes.

**Split test** (`scripts/survey/v0b_mock_window.py`, outputs/survey/mock_window). R is calibrated on Quijote halo
boxes 0–39 cut to the V0a cone and applied to boxes 40–79, with no erosion. Residuals are in units of one survey's
error:

| Setup | Uncorrected | Corrected, max / rms | χ² / 60 |
|---|---|---|---|
| Uniform n̄, line of sight z | 5.1–6.1σ | 0.12 / 0.05σ | 67 |
| Radial n̄(r), line of sight z | 4.5–5.2σ | 0.14 / 0.06σ | 88 |
| Radial n̄(r), local line of sight | 4.7–5.2σ | 0.14 / 0.05σ | 76 |

A mock-calibrated R transfers between independent mocks of the tracer at ≲ 0.1σ. The ~0.1σ mismatch left by
Gaussian mocks is gone.

**V1, wide angle.** For the same boxes, the local-line-of-sight reference differs from the plane-parallel one by
≤ 0.11σ (rms 0.05σ, χ² 69/60, ≤ 0.26% fractionally) at r = 1.1–2.3 Gpc/h. So the plane-parallel box model × R is
enough at DESI distances.

**Next:**
1. S21m through the same R. The second layer of the masked modulus is measured with the support-mean subtraction
   (E4).
2. The parameter-level test: fit the masked halo S1m (+S21m) with R × the box model, using the cubic halo emulator
   and analytic covariance.
3. The DR1 geometry from the randoms, once the Abacus LRG HOD boxes are available.

## 8. Open decisions

- **Mesh per z-bin:** the 0.8–1.1 LRG bin extends to r ≈ 2.4 Gpc/h, so the NGC box is ~4 Gpc/h on a side. At
  6–7 Mpc/h cells that is a 600–700³ mesh: check the torch backend's memory with L = 4 and per-|m| outputs.
- **How much volume to give up:** eroding S1m edges is cheap. For S21m, erosion at σ2 ≥ 70 would discard most of a
  z-bin, so the window kernel (M2c) is probably unavoidable, or σ2 would have to be lowered (which tree-level S21m
  does not support at σ1 = 17.7).
- **Galaxies vs halos:** this is deferred by decision (§0). If V3 fails, the first thing to check is a box-level HOD
  test: Abacus LRG HOD boxes, already measured by `abacus_base.py`. It separates HOD effects (satellites, FoG on the
  S21m damping and W) from geometry.
