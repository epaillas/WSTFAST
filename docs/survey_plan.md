# From periodic boxes to survey geometry: plan

Status: planning draft, 2026-10-12. Nothing here is implemented yet.

Goal: fit the halo/galaxy WST baseline to mock catalogues with a realistic footprint and radial selection n(z). The
baseline is S1m (σ ≥ 25 Mpc/h, q = 0.8) + S21m (σ1 ≥ 25, σ2 ≥ 70 Mpc/h), with cubic biases, the shared Fingers-of-God
counterterm c_v of both layers, the non-Gaussian second-layer template and a free white term W (now consistent with zero):
`scripts/fit_halo_baseline.sh` (since 2026-10-14).
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
| Complete mocks | `mocks/AbacusSummit/dark/v4.2/mock{0..24}/LRG_complete_{NGC,SGC}_clustering.dat.fits`, `..._{0..17}_clustering.ran.fits`, `LRG_complete_nz.txt` | No fibre assignment, same 25 realisations. 0.66 GB of data and 14 GB of randoms per mock (each random file ≈ 4.6× the data; angular positions shared across mocks, Z drawn per mock). `scripts/survey/download_abacus_lrg.sh`; mocks 0–2 with randoms 0–1 are local in `data/desi/` |

These catalogues are ~100 GB in total, so V2 onwards runs at NERSC (the public tree is mirrored there under
`/global/cfs/cdirs/desi/public/dr1/`). V0 and V1 run locally on the Quijote halo boxes.

**Not public:** the matching LRG HOD cubic boxes (parents of the cutsky mocks). The complete mocks are public
(2025-10), so V3 (light cone) and V4 (fibre assignment) can be run separately.

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

**DR1 geometry and the radial integral constraint (2026-10-14).** `SurveyGeometry.from_angular_mask` (with
`AngularMask` and `desi_radial_density` in `wstfast.survey`) builds the geometry of a DR1 LRG z-bin from the public
complete Abacus mocks:
- the angular completeness on HEALPix (nside 128) from the randoms, at 2500 randoms per deg² per file (the median
  pixel is partly masked by the LRG veto masks, so it can't set the normalisation);
- n(r) of the data per photometric region;
- nbar(x) = completeness × n_R(r);
- regions N (NGC, Dec > 32.375) and S (the rest of NGC and all of SGC).

At 10 Mpc/h cells, the 0.4–0.6 bin is 1.46 (NGC) + 0.87 (SGC) (Gpc/h)³ on 344³ meshes. The areas are 4528 and
2552 deg², as in the n(z) file.

**RIC.** In these mocks each random takes the Z of a data object of its own photometric region; matching
TARGETID_DATA shows S drawn jointly from NGC-S and SGC (55/45). This is the DESI fiducial for LRGs (desi-clustering:
`ric_regions = ["N", "S"]`, 1000 radial bins). So the FKP field has zero mean in every thin shell of every region,
and neither `GaussianMocks` nor `box_to_survey` had this. `RadialIntegralConstraint` imposes
δ' = δ − Σ_s n̄δ / Σ_s n̄ jointly over the meshes of both caps.

On Gaussian Kaiser mocks with the local line of sight in the DR1 0.4–0.6 geometry (`scripts/survey/dr1_window.py
--test ric`, 10 paired realisations, outputs/survey/dr1_ric), the RIC shifts the masked coefficients by (max, rms over
blocks, in units of one survey's error, ±0.1):

| | NGC S1m | NGC S21m | SGC S1m | SGC S21m |
|---|---|---|---|---|
| Max | 0.49 (0.85%) | 0.34 (0.7%) | 0.25 (1.0%) | 0.14 (0.7%) |
| Rms | 0.18 | 0.12 | 0.08 | 0.05 |

The shift is the same for shell widths of 2.5, 5 and 10 Mpc/h. Against the 25-mock mean it is up to ~2.5σ, so the
RIC is part of the window: the calibration mocks get it on the survey side (`calibrate_window` and `LambdaWindow`
take paired survey / reference inputs).

**λ-level window (2026-10-14).** `LambdaWindow` (`scripts/survey/dr1_window.py --test lambda`,
outputs/survey/dr1_lambda_window) puts the window on the component variances, which are linear in P:

    λ_m(x) = T_m(x) λ_m^box + λ_N(x),   T_m(x) = f(x) c_m(f).

- f is the coverage of the wavelet by the support (an FFT; exact for white noise).
- c_m is a correction calibrated in 40 coverage bins on 16 paired signal-only Gaussian mocks at the fiducial spectrum,
  with the RIC.
- A per-cell ratio estimator of T was too noisy: the concave moment turned its noise into a large bias.
- The survey S1m is the FKP-weighted mean of the Gaussian moment cell by cell, with the exact λ^box of any spectrum.

Tested on 8 noisy, constrained mocks per spectrum, in units of one survey's error (max / rms over 60 S1m blocks):

| Spectrum | NGC window | SGC window | Coverage only | No window |
|---|---|---|---|---|
| Fiducial (b 2, f 0.76) | 0.52 / 0.23, χ² 25 | 0.76 / 0.35, χ² 58 | 8–9 | 29–40 |
| b 1.6, f 0.6 | 0.46 / 0.19, χ² 25 | 0.66 / 0.30, χ² 65 | 7 | 23–32 |
| Amplitude × 1.21 | 0.56 / 0.25, χ² 25 | 0.83 / 0.37, χ² 56 | 9–10 | 31–44 |
| Tilt Δn_s = −0.06 | 0.60 / 0.28, χ² 36 | 0.86 / 0.37, χ² 65 | 8–9 | 29–39 |

- The edge effect of the DR1 footprint is large: S1m is 8–24% below the box value.
- The window brings every spectrum to agreement within the test's own precision (≤ 0.35–0.40σ per block, from
  8 mocks). That precision is not yet the 0.1σ target.
- Spectrum dependence of the correction, from residuals paired across spectra (same seeds), max / rms:

  | Variant | Paired difference |
  |---|---|
  | Amplitude | 0.07 / 0.02σ (none expected) |
  | b, f | ≤ 0.20 / 0.06σ (error 0.08) |
  | Tilt | ≤ 0.16 / 0.08σ (error 0.01, significant) |

  A Δn_s of 0.06 is ~15× the Planck error, so recalibrating c once at the best fit removes it.

**100-mock test (2026-10-15,** outputs/survey/dr1_lambda_window_100: 32 calibration mocks, 100 test mocks per
spectrum, test error 0.1σ**).** The λ window does **not** reach 0.1σ on its own.
- Residual: max 0.37–0.57σ, rms 0.14–0.25σ, χ² 158–359 / 60. The 8-mock agreement was the test's noise.
- It is a fixed geometric residual: 0.2% rms (≤ 0.5%) of S1m, correlated at 0.98–0.997 across the four spectra. The
  window still removes > 98% of the 19–32σ edge effect.
- An S-level correction per block, δ_b = ⟨S_measured⟩ / S_window from fiducial mocks 0–49, applied to mocks 50–99 of
  every spectrum, leaves:
  - NGC: max 0.43–0.60σ, rms 0.18–0.23σ, χ² 60–73 / 60 (error ≤ 0.22σ);
  - SGC: rms 0.22–0.30σ, χ² 99–141 / 60.
- So the window model is λ window × δ_b. The residual left in SGC, and the 0.1σ level, need more mocks (GPU). The
  candidate causes of the geometric residual are the coverage-binned correction (c varies within a bin) and the
  correlation of ±m components near edges, which the independent-component moment ignores.

**Survey emulator (2026-10-15):** `outputs/emulators/survey_lrg_z0.51.h5`.
- Settings: biased, cubic, FoG, IR; c000 fiducial (one 0.06 eV neutrino, cb power); 10 Mpc/h cells; lattice
  3440 Mpc/h / 344³; z = 0.509; unprojected S21m, so AP applies to every statistic; 369 nodes.
- Validation over the ±15% box, max relative error: S1m 3.2e-4, S21m 1.1e-3, P_l 0.6% (the box emulator: 3.7e-4,
  1.0e-3, 3.9%).

**S21m window (2026-10-15,** `dr1_window.py --test s21m`**).** T₂ = f₂ c₂ on the σ_j2 coverage, with survey
S2m = R₂ × box S2m. Each test mock is paired with its own box. The residuals below are over the 54 baseline entries
(l ≥ 1), in units of one survey's error.

| Estimator | Run | NGC max / rms | SGC max / rms | Fractional residual |
|---|---|---|---|---|
| Global support mean | local, 16 mocks | 0.61–1.15σ / 0.26–0.42σ | 1.11–1.12σ / 0.35–0.44σ | ≤ 4.3% |
| Binned mean (E4) | NERSC A100, 32 mocks | 0.54–0.62σ / 0.22–0.23σ | 0.50–0.54σ / 0.21–0.22σ | ≤ 1.6% (0.5–0.6% rms) |

- **Global mean.** The residual depends on b and f (pattern correlation 0.84–0.91): a deterministic offset, the
  edge profile of ⟨U⟩, which ψ₂ carries into X₂.
- **Binned mean** (`second_layer_bins`: 10 coverage quantile bins + 1 interior bin, × 3 n̄ bins). The residual is
  the same for all four spectra (correlation 0.93–0.999), so it is geometric, as for S1m.
- **With the S-level δ_b** (16 fiducial mocks to calibrate, the other 16 to test): χ² 73–83 / 54 (NGC) and
  58–68 / 54 (SGC), consistent with the test error of ≤ 0.37σ.
- **The window model for both layers is the λ window × δ_b.** The 0.1σ level still needs more mocks.
- The l = 0 S21 entry (not in the baseline) keeps U's mean (the second-layer low-pass) and carries the largest
  residual (−3.1σ NGC): another reason to keep it out.
- Measured with the binned estimator, the survey S21m moves farther from the box (up to ~280σ in NGC). This is the
  same statistic in a box, so it is all window.

**Independent fiducial mocks (2026-10-15, NERSC A100s,** `--variants fiducial --seed-offset`**).**
- New mocks: 300 for S1m (seeds 1200+, 2200+) and 102 for S21m (1500+, 2500+).
- δ_b is calibrated on the earlier runs (100 for S1m, 32 for S21m) and tested on the new mocks. The errors include
  both the test and the calibration of δ.
- Residuals (baseline entries, in units of one survey's error):

  | | S1m (60) | S21m (54, l ≥ 1) |
  |---|---|---|
  | NGC | max 0.18σ, rms 0.08σ, χ² 25 / 60 (error ≈ 0.12σ) | max 0.29σ, rms 0.13σ, χ² 45 / 54 (error ≈ 0.15σ) |
  | SGC | max 0.24σ, rms 0.12σ, χ² 67 / 60 (error ≈ 0.11σ) | max 0.36σ, rms 0.16σ, χ² 71 / 54 (error ≈ 0.15σ) |

  The reverse split (calibrate on the new mocks, test on the old) agrees.
- Without δ_b, the window alone leaves 0.12–0.17σ rms (max 0.3–0.5σ) on all the mocks.
- **The window model, λ window × fiducial δ_b, is consistent with the mocks at the ~0.1σ level of one survey for
  both layers.**

**Precision needed for the Abacus mean.** The 25-mock mean has errors 5× smaller than one survey. A per-entry
window error of 0.1σ of one survey is 0.5σ of the mean, so the criterion has to be at the parameter level.
- δ_b is spectrum-independent, so its error falls as 1/√N_cal: it can be calibrated on many cheap Gaussian mocks
  (~1 min per paired S21m mock on an A100).
- Its remaining uncertainty enters the fit as an extra covariance, Cov_δ (as DESI does for window systematics).
- Next: the parameter shift from the δ_b error, and the N_cal it needs.

**Analytic survey covariance on the Gaussian mocks (2026-10-15,** `dr1_window.py --test covariance`**).**
- The model: box covariance (`RSDWSTCovariance`, 2nd + 4th chaos, box 1 Gpc/h / 128³, Gaussian inputs: Kaiser P + 1/n̄_eff,
  zero modulus response, Gaussian-chaos noise of U), with every variance rescaled to the entry's effective volume on
  the cap (`SurveyWindow.veff_block`) and the box correlations kept.
- Against 400 survey S1m and 134 S21m mocks per cap (ln S):

  | | σ analytic / mocks | Correlations, rms diff (noise) | ⟨χ²⟩ |
  |---|---|---|---|
  | NGC S1m | 1.06 (± 0.035) | 0.044 (0.044) | 94 / 60 |
  | SGC S1m | 1.11 | 0.043 (0.044) | 97 / 60 |
  | NGC S21m | 1.13 (± 0.06) | 0.087 (0.083) | 45 / 54 |
  | SGC S21m | 1.18 | 0.094 (0.083) | 43 / 54 |

- **Variances 5–18% too large**, the direction of the modes the RIC and the binned second-layer mean remove.
- **Directions:**
  - S21m: the eigenvalues of A⁻¹S span 0.10–2.3, as sampling noise does.
  - S1m: a few nearly degenerate directions are 2.5–5× larger in the mocks (eigenvalues to 5.2, against a noise
    spread of ≤ 1.9), hence the ⟨χ²⟩ excess.
- **Parameters:** the marginalised errors on ω_cdm, logA and h from the two covariances agree within 1–5% (8% for
  ω_cdm from SGC S21m). The analytic covariance is fine for parameter errors, but not for the goodness of fit of S1m.
- **Per-block vs single V_eff:** makes almost no difference.

**Next:**
1. Propagate the δ_b calibration error to the parameters (Cov_δ in the fit) and size N_cal for the 25-mock mean.
2. S21m through the same R. The second layer of the masked modulus is measured with the support-mean subtraction
   (E4).
2. The parameter-level test: fit the masked halo S1m (+S21m) with R × the box model, using the cubic halo emulator
   and analytic covariance.
3. The DR1 geometry from the randoms, once the Abacus LRG HOD boxes are available.

## 8. End-to-end pipeline (design, 2026-10-14)

This section assumes the λ-level window (`LambdaWindow`) works for the whole data vector. It is validated for S1m on
Gaussian mocks (§7, status). The S21m form below is the same construction one layer up, and is still to be tested.

### 8.1 How DESI does it, and what we copy

In desilike / desi-clustering `full_shape`, **the window is not inside the emulator.**
- The Taylor emulator is fitted to `theory.pt` only: the bias-independent PT blocks (`desi-clustering
  full_shape/tools.py`, `TaylorEmulator(compile(theory.pt))`).
- The window enters the emulator in only two ways:
  - it sets the k-grid (and ells) on which those blocks are emulated: `Spectrum2PolesObservable` does
    `theory.update(k=window.theory k, ells=window.theory.ells)`;
  - its options are part of the emulator's cache key.
- At every likelihood call, the bias combination is analytic and the window matrix is applied in JAX:
  `flattheory = W @ theory.poles` (`desilike/observables/galaxy_clustering/stats.py:236-241`).

We keep the same split:
- **Emulator:** the cosmology-dependent box basis on a survey-matched grid.
- **Window:** a precomputed, cosmology-independent geometry product applied at assembly, after the biases and AP.

The difference is that our statistic is not linear in P, so the window acts on the block variances λ, which are, and
the moment is taken afterwards.

### 8.2 Stages

**Stage 1. Inputs** (per tracer, z-bin, cap). The data, all 18 random files and the n(z), read in place at NERSC
(`/global/cfs/cdirs/desi/public/dr1/...`).

**Stage 2. Geometry** (once per z-bin, from the randoms; `SurveyGeometry.from_angular_mask`):
- angular completeness;
- n_R(r) per photometric region;
- n̄(x), FKP weights and regions on a 10 Mpc/h mesh;
- the `RadialIntegralConstraint` over both caps;
- the local line of sight.

The two caps share one mesh size.

**Stage 3. Window products** (once per z-bin, cosmology-independent, GPU):
- **First layer.** For every block b = (j, l, |m|):
  - the coverage map f_b(x);
  - the noise map λ_N,b(x) (exact from n̄);
  - the correction c_b(f), from ~32 paired signal-only Gaussian mocks at the fiducial spectrum, with the RIC.
- **Second layer.** The same three for the σ₂ wavelet acting on the masked, mean-subtracted modulus:
  - f₂ and its correction c₂ from the same mocks, with T₂ = f₂ c₂ multiplying λ₂ (which is linear in P_UU);
  - the first-layer normalisation ⟨U⟩ of the survey, from the first-layer products.
- **Compression.** The likelihood never sees cells. Per block, the cells are binned in (coverage, noise) and
  stored as weighted histograms: FKP weight sums, mean T and mean λ_N per bin. This is ~40 × 8 bins per block, so
  the products are a few MB per z-bin.
- Products are recomputed once at the best fit (to remove the tilt dependence of c, which is 0.16σ at Δn_s = 0.06).

**Stage 4. Measurement** (GPU, `measure_wst_survey_torch`):
- The field is the FKP form δ = (n_g − α n_r)/(n̄ V_cell) on the geometry's support. This needs a small extension of
  `SurveyMesh`, which today divides by α n_r per cell.
- The randoms carry the RIC automatically.
- The coefficients are FKP-weighted moments with the local line of sight.
- The data vector per cap and z-bin is S1m (60) + S21m (55). There are 25 mocks × 2 caps × 3 bins; the mean is
  taken per cap and bin.

**Stage 5. Emulator** (retrained once per z-bin, with `train_emulator.py`, no new script):
- **Survey config:** σ₀ = 25 Mpc/h, step √2, J = 3, L = 4, S21 at (j1, j2) = (0, 3), cell 10 Mpc/h. The CIC filter
  of the cumulant legs uses the 10 Mpc/h cell, which the current emulator bakes in at 3.9 Mpc/h.
- **Unprojected S21m terms.** S21m is emulated on the (field, K, μ_K) grid, and λ₂ is projected at likelihood time,
  as S1m already is. Today the λ₂ lattice sum (box 1000, mesh 256) is inside the emulator, so neither AP nor the
  window could act on it.
- **Redshift:** one emulator per z_eff (0.51, 0.71, 0.92). The z-evolution inside a bin (M4) is a later refinement.

**Stage 6. Survey model** (likelihood time; JAX; `S1mTheory` / `S21mTheory` get an optional `window=` of stage 3
products):
1. Box λ_b from the emulated grid with AP: `S1mProjection` on a 10 Mpc/h lattice.
2. Survey S1m_b = Σ_bins H_b,i · M_q[T_i λ_b + λ_N,i ; κ_i, skew_i] / Σ H_b,i.
   - `s1m_from_variances` is vmapped over bins.
   - The noise-dependent cumulant columns use the bin's N_i.
   - The cumulants scale as κ → T²κ and skew² → T³ skew², which leaves the Edgeworth ratio invariant. This is the
     approximation to test on halos.
3. Survey S21m: λ₂ = T₂ (response + noise terms) per bin, normalised by the survey first-layer λ, then
   `s21m_from_terms`. The layer-2 template is unchanged.
4. **Caps** are separate data vectors with their own window products and shared parameters.
5. **z-bins** share cosmology and have their own biases, c_v and W.

**Stage 7. Covariance** (analytic, `RSDWSTCovariance` extended, no new file):
- The box covariance at the survey's effective n̄, with V → V_eff,b per block, where
  V_eff,b = (Σ w)² / Σ w² × V_cell on that block's support.
- Cross-blocks use √(V_eff,a V_eff,b).
- For the Abacus mean: Cov / 25.
- This is checked later against EZmock covariances (C1), and on cut-sky halo mocks (V2).

**Stage 8. Fit** (`fit_rsd.py` with `--survey <products>` and the DR1 data vectors; NUTS):
- Varied: ω_cdm, logA, h (n_s fixed, as in the baseline), plus the per-bin cubic biases, c_v and W, with the
  layer-2 amplitudes marginalised analytically.
- Truth: AbacusSummit c000 (ω_cdm 0.1200, ω_b 0.02237, h 0.6736, n_s 0.9649).
- AP is relative to the Quijote fiducial frame of the coordinates.

### 8.3 Validation before the Abacus fit

| Step | Test | Pass |
|---|---|---|
| W1 | S1m λ-window, 100 Gaussian mocks × 4 spectra (running) | ≤ 0.1σ per block, χ² ok |
| W2 | S21m λ₂-window, same mocks and spectra | ≤ 0.1σ |
| W3 | Non-Gaussian edges: Quijote halos cut to the DR1 geometry (`box_to_survey`), windowed box model vs measured, with the halo emulator | ≤ 0.1σ, or a residual template |
| W4 | End-to-end fit to the mean of cut-sky halo mocks (known cosmology), survey covariance | shifts ≤ 0.3σ |
| W5 | Analytic V_eff covariance vs the scatter of the cut-sky halo mocks | ≤ 10–20% |
| V3 | Fit to the mean of the 25 Abacus complete mocks, 0.4–0.6 first, then all bins | recovers c000 |

### 8.4 Where the code goes (extending existing files)

| Piece | File |
|---|---|
| Geometry, RIC, window products, FKP mesh from a geometry, histogram compression | `wstfast/survey.py` |
| GPU estimator and window | `wstfast/measure_torch.py` |
| `window=` products in the assembly | `wstfast/calculators.py` (`S1mTheory`, `S21mTheory`) |
| Unprojected S21m terms, survey config, per-z emulators | `wstfast/theory/rsd_moduli.py`, `rsd_bias.py`, `scripts/train_emulator.py` |
| V_eff covariance | `wstfast/theory/covariance_rsd.py`, `scripts/analytic_covariance_rsd_wst.py` |
| Survey data and window options | `scripts/fit_rsd.py` |
| Geometry / window tests / mock measurement driver (gains a measurement mode) | `scripts/survey/dr1_window.py` |

### 8.5 Main risks

1. **Non-Gaussian edges.** On halos the edge effect differed from Gaussian mocks by ~4% (V0b). If W3 fails, the
   correction c is calibrated on halo cut-sky mocks instead, with the same products and different mocks.
2. **S21m window.** The σ₂ = 70.7 wavelets are 150–200 Mpc/h across. Coverage is low over much of a bin, so T₂ is far
   from 1 and the mean subtraction of the second layer (E4) interacts with the RIC. This is the riskiest piece.
3. **Retraining.** The emulator retrain (survey cell, unprojected S21m) is the largest single cost, with three per z
   if bins are kept separate.
4. **HOD galaxies vs halos.** The Abacus mocks are HOD galaxies, whose satellites give FoG; the model is validated
   on halos. Deferred by decision (§0); check first if V3 fails.

## 9. Open decisions

- **Mesh per z-bin:** the 0.8–1.1 LRG bin extends to r ≈ 2.4 Gpc/h, so the NGC box is ~4 Gpc/h on a side. At
  6–7 Mpc/h cells that is a 600–700³ mesh: check the torch backend's memory with L = 4 and per-|m| outputs.
- **How much volume to give up:** eroding S1m edges is cheap. For S21m, erosion at σ2 ≥ 70 would discard most of a
  z-bin, so the window kernel (M2c) is probably unavoidable, or σ2 would have to be lowered (which tree-level S21m
  does not support at σ1 = 17.7).
- **Galaxies vs halos:** this is deferred by decision (§0). If V3 fails, the first thing to check is a box-level HOD
  test: Abacus LRG HOD boxes, already measured by `abacus_base.py`. It separates HOD effects (satellites, FoG on the
  S21m damping and W) from geometry.
