# From periodic boxes to survey geometry: plan

Status: planning draft, 2026-10-12. Nothing here is implemented yet.

Goal: fit the halo/galaxy WST baseline to mock catalogues with a realistic footprint and radial selection n(z). The
baseline is S1m (σ ≥ 25 Mpc/h, q = 0.8) + S21m (σ1 ≥ 25, σ2 ≥ 70 Mpc/h), with cubic biases, the shared Fingers-of-God
counterterm c_v of both layers, the non-Gaussian second-layer template, a free white term W (now consistent with zero)
and Alcock-Paczynski distortions in every piece of the data vector (AP-polynomial emulator, since 2026-10-15):
`scripts/fit_halo_baseline.sh` (`--no-ap` for the box fits without AP).
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

**Calibration error and survey fit (2026-10-15).**
- **Paired S1m calibration.** With each mock also measured unmasked on the same cells (`dr1_window --test lambda
  --paired`), δ = ⟨S_survey/S_box⟩ F(T=1)/F(T) loses the mocks' cosmic variance: the per-mock scatter is 0.5–0.65%.
- **Size of the calibration error** (`fisher_rsd.py --window-ratios`). With 827 paired S1m and 734 S21m mocks, the
  calibration error shifts the parameters of the 25-mock mean by 0.10–0.11σ rms; unpaired S1m would have needed ~2500
  mocks. It enters the fit as `--extra-covariance` (fractional, per cap): `dr1_window --test products` writes
  `window_calibration_<z>_<cap>.npz`.
- **The window in the theory:** `SurveyWindow` (per cap) and `S1mTheory(window=)`, `S21mTheory(window=)`.
- **Survey fits:** `fit_rsd.py --survey NGC SGC --window …` with `--covariance-file` (analytic,
  `analytic_covariance_rsd_wst.py --survey`) or `--covariance-data-dir` (boxes scaled to V_eff).
- **Abacus measurements:** `dr1_window --test measure`. All 25 complete mocks × 18 random files are in
  `outputs/survey/abacus_lrg_z0.4-0.6/` (NERSC A100, ~6 min per mock).
- **Survey emulator:** `outputs/emulators/survey_lrg_z0.51.h5`, unprojected S21m. It lacks the AP polynomials of the
  baseline's `ap3` emulator (cumulant AP, ~1e-4).

**Halo calibration (W3) and the end-to-end halo test (W4), 2026-10-15** (`dr1_window --test halos`).
- **Mocks:** 300 pairs of Quijote M > 10¹³ halo boxes (z = 0.5) cut to the DR1 geometry with n̄ × 0.5 (`--nbar-scale`;
  the halos have n̄ = 3.1e-4 against a 6e-4 peak), with the RIC, each paired with its parent box. Compact catalogues:
  `data/quijote/halos_m13_z0.5/` (local and NERSC).
- **Halo δ** (pairs 0–274): 0.983–1.042 (NGC), 0.981–1.084 (SGC), against ≤ 1.2% for Gaussian mocks. The non-Gaussian
  edge effect of halos, larger in SGC. Abacus windows with the halo δ / K:
  `outputs/survey/dr1_window_products/window_<z>_<cap>_halodelta.npz` (+ `window_calibration_…_halodelta.npz`).
- **W4** (mean of pairs 275–299, halo windows, halo-box covariance, baseline + AP): **fails**.
  - Profile: ω_cdm 0.0950 (the emulator's lower edge), logA 3.22, h 0.710, against the truth 0.1209, 3.061, 0.6711;
    χ² = 2523 / 197.
  - The Abacus fits fail the same way: 11 mocks, NUTS, Gaussian δ, ω_cdm 0.098 at the lower edge, logA 3.20, h 0.723,
    against 0.120, 3.036, 0.674.
- **Diagnosis** (data / model at the Quijote truth, with the halo nuisances of the box baseline fit
  `outputs/inference/rsd/fast_noap_V60`):
  - **The window works.** Survey data / windowed model tracks box data / box model block by block within ~0.5%, across
    a window suppression of 8–22%.
  - **The box model in the survey configuration does not match the box data on the survey mesh.**
    - About 5% low everywhere, including σ = 25 Mpc/h and high l.
    - Up to 11–20% low for l = 0 at σ = 50–71 Mpc/h.
    - The fit compensates by raising logA and lowering ω_cdm.
  - **Likely causes:**
    - For the l = 0 excess: the halo mocks tile a 1 Gpc/h box over a 3440 Mpc/h mesh, so their power sits on the
      1 Gpc/h lattice while the model assumes the continuum. This is an artefact of the test.
    - For the ~5% offset: open. Candidates are nuisances taken from the 3.9 Mpc/h box fit, the noise of the thinned
      halos, the survey emulator's fixed 0.06 eV neutrino and its ω_b against Quijote (≲ 1%), and CIC at 10 Mpc/h.

**Box-model diagnostics (2026-10-16): the W4 offset is an artefact of the halo mocks, not of the model.**
- **Data ladder.** The same Quijote halo boxes (NGC pairs, boxes 2i) measured step by step from the 3.9 Mpc/h periodic
  baseline to the survey-mesh parent box (`NGC_box`), S1m, paired ratios. Values are for l = 0 at σ = 25 → 71 Mpc/h, or
  the range over entries:

  | Step | Data ratio | Model ratio |
  |---|---|---|
  | 3.9 Mpc/h, survey config, re-measured | 1 exactly | |
  | 10 Mpc/h periodic (CIC) | 0.99–1.00 (0.96 at j0 l4) | 0.98–0.99 (0.95 at j0 l4): 1.1% low, constant |
  | Tiled onto the survey mesh, los z | ≤ 0.4% (≤ 1% at l2, σ 71) | (not modelled) |
  | Thinned to nbar(x), **old normalisation** | +4–7%, l0: +4 / 7 / 12 / 21% | |
  | Thinned to nbar(x), **CIC-consistent normalisation** | +1.1–3.2% | +1.1–3.9% (the larger shot noise) |
  | Local line of sight | ≤ 0.4%, but l2 m0 at σ 50 / 71: −2 / −4% (8 boxes, to check) | |

- **Cause.** `box_to_survey` kept objects with the probability p of their nearest cell, but painted them with CIC and
  normalised by p nbar V_cell. nbar(x) jumps from cell to cell with the angular completeness (nside 128). The true
  expected CIC count is nbar V (p * K), K = [1/8, 3/4, 1/8] per axis, and (p * K)/p − 1 has 36% rms on the DR1
  support (max ×9). The mocks therefore carried a fixed additive pattern in δ, following the mask and the radial shell
  edges. It gives the +3–7% everywhere and the large l = 0 excess at σ = 50–71. It is not the 1 Gpc/h tiling (≤ 0.4%).
- **Fix** (`box_to_survey`): normalise by the CIC-convolved expectation. On 8 boxes the survey-mesh parent box now
  matches the model to a mean of +0.85% and an rms of 0.85% over the 60 S1m entries (before: +5%, up to +20%).
- **The remaining +0.85%** is the survey emulator's c000 fiducial against the Quijote truth. The 0.06 eV neutrino and
  the Abacus ω_b lower P_cb by 2.5% (CLASS, k = 0.05–0.2), which is ≈ 1% in S1m. This is right for Abacus; for W4
  against Quijote it is a known ~1% offset (a Quijote-fiducial survey emulator would remove it).
- **Consequences.** The halo δ (0.98–1.08), `halos_calibration_0-274.npz`, the `*_halodelta` windows and W4 all
  used the old mocks and must be redone (300 pairs at NERSC, ~2 h on 3 A100s). The window validation on halos
  (survey/model tracks box/model to 0.5%) stands, since both sides had the pattern.
- **Abacus is not explained by this.** The FKP field (n_g − α n_r)/(nbar V) has no additive term, since α n_r is the
  true expectation. The nbar(cell) mismatch, α n_r/(nbar V), is ~20% rms after removing the random shot noise (2
  files) and only multiplies δ. Its estimated effect on S1m is ≲ 0.5% (extra white noise ~var(Q) σ²_cell V_cell); not
  measured, since a per-cell α n_r denominator needs all 18 random files.
- **Abacus 25-mock mean, profiles** (Gaussian δ, analytic covariance + calibration covariance, baseline + AP):
  - free: ω_cdm 0.1032 ± 0.0022, logA 3.213 ± 0.047, h 0.718 ± 0.006, χ² 1278 / 197 (c_v 2.3, b1 1.67);
  - c000 fixed: χ² 1406 / 200 (Δχ² = 128; c_v 2.0, b1 1.94).
  - **The caps pull in opposite directions** (diagonal residuals): NGC data below the model (−2 to −7σ of the mean,
    mostly l = 0 and σ ≥ 50), SGC above it (+3 to +5σ at σ = 25–35, l ≥ 1). Fixing c000 helps NGC and worsens SGC.
    c_v ≈ 2 against 0.96 for halos (HOD satellites).
  - **Per cap** (same setup, one window each):

    | | ω_cdm | logA | h | c_v | b1 | α₀ | χ² |
    |---|---|---|---|---|---|---|---|
    | NGC | 0.1041 ± 0.0025 | 3.161 | 0.714 | 1.70 | 1.75 | −0.52 ± 0.12 | 724 / 83 |
    | SGC | 0.1104 ± 0.0038 | 3.093 | 0.724 | 2.36 | 1.81 | −0.11 ± 0.13 | 449 / 83 |

    Each cap alone fails and is biased the same way (low ω_cdm, high h). Sharing the parameters costs only
    Δχ² = 105 (1278 against 724 + 449). So the cap split is secondary: the main misfit is within each cap. NGC wants a
    shot noise half the Poisson value (α₀ = −0.5), which is implausible: a sign that the noise or the amplitude of the
    survey model is off in that cap (the FKP n̄, the noise maps, HOD stochasticity).

**Abacus bias hunt (2026-10-16, night; in progress).**
- **CIC shot noise in the window.** The window's noise maps (`noise_variance_maps`) were white, but the data are
  CIC-painted and the box model (`S1mProjection.shot`) uses the aliased CIC noise. On pure-noise Abacus fields
  ((n_A − n_B)/(n̄V), random halves), the white maps overpredict S1m by 3.5–8% at σ = 25 Mpc/h (worst at high l). With
  `noise_variance_maps(window="cic")` the residual is flat at ~0.98, the Jensen effect of noise following the galaxy
  density. Products: `dr1_window --test products --noise-window cic` (δ calibrated against the noise of the mocks,
  `--mock-noise`): `outputs/survey/dr1_window_products_cic`. It barely moves the Abacus fit (ω_cdm 0.1045, logA 3.20,
  h 0.723), so the noise is not the cause; α₀ < 0 absorbs a signal misfit.
- **No AP:** ω_cdm 0.102, logA 3.19, h 0.703, so AP is not the lever. **Scale cuts (S1m only):** unstable and
  χ² 142 / 44 even at σ ≥ 50, so the misfit is not only small-scale (FoG). **Covariance diagonal** against the
  25-mock scatter: median 0.92 (NGC), 1.05 (SGC); 1.4–1.8 at σ 25, high l.
- **FKP denominator** (`--test denominator`, 25 mocks × 18 randoms, NERSC CPU): with α n_r instead of n̄V, S1m
  changes by +0.0% (NGC, rms 0.26%) and −0.7% (SGC, rms 0.17%), bounds since the randoms' noise raises S. The
  1/N_randoms extrapolation is unreliable. This is sub-dominant but cap-dependent, in the direction of the SGC excess.
- **Halo calibration with the fixed mocks** (300 pairs, `dr1_halos_v2_*`): δ = 0.970–0.994 against 0.988–1.003 for
  Gaussian mocks. Halo edges suppress S1m by 1–3% more. On Abacus (`*_halodelta`): ω_cdm 0.1076, logA 3.167, h 0.729.
- **W4 with the fixed mocks** (pairs 275–299, halo windows, CIC noise, halo-box covariance, baseline + AP): ω_cdm
  0.1172 ± 0.0033, logA 3.026 ± 0.065, **h 0.718 ± 0.0085 (truth 0.6711, +5.5σ)**, α₀ −0.48, χ² 424 / 197 (was
  2523). This is the same failure as Abacus, in a controlled test. S1m alone gives h 0.7125 and α₀ −0.51; the residuals
  grow with |m| at every (σ, l), i.e. the data have relatively more m ≥ 1 than m = 0.
- **Not the local line of sight.** 300 paired halo boxes (`--test halos --box-only --box-los-z`, `dr1_halos_wa_*`):
  the local-LOS parent box against the z-LOS one differs by ≤ 0.3% for S1m (mostly 0.1–0.2%, ≲ 2σ) and ≤ 0.6% for
  S21m. An 8-box estimate of −4% (l 2, m 0, σ 71) was noise.
- **The cause on halos: δ was calibrated at the wrong signal.** `SurveyWindow.calibrate_s1m` takes F(T=1)/F(T) at
  the box variances of the calibration spectrum (`--test products`: the Gaussian fiducial, linear Kaiser b = 2). On
  the 25 held-out halo pairs at the truth, (survey / window model) / (box / plain model) was up to +1% (mean +0.45%,
  growing with l and |m|; pair error 0.03–0.3%), which is the W4 residual pattern. The window factor's signal
  dependence does not carry δ from a linear Gaussian field to halos at the 1% level.
  - **Recalibration at the tracer's signal**: r = that ratio on pairs 0–274 at the truth (+ box-fit nuisances), δ → δ r
    (`dr1_window_products_halos_v2_recal`). On pairs 275–299 the residual is then −0.1% mean, 0.12–0.17% rms.
  - **W4 passes**: ω_cdm 0.1225 ± 0.0037, logA 3.113 ± 0.069, h 0.673 ± 0.011 (truth 0.1209, 3.061, 0.6711),
    α₀ +0.15, χ² 150 / 197. S1m only: 0.1229, 3.136, 0.668, χ² 119 / 104. The slightly high logA is the direction of
    the ~1% c000-fiducial offset of the survey emulator.
- **Abacus with the recalibrated halo δ** (`dr1_window_products_cic_halorecal`): joint ω_cdm 0.1267 ± 0.0029, logA
  3.110, h 0.695 ± 0.008, χ² 494 / 197 (was 1285). NGC alone: 0.1207 ± 0.0036, 3.075, 0.689 ± 0.012, χ² 168 / 83.
  SGC alone: 0.1292 ± 0.0051, 3.045, 0.716 ± 0.016, χ² 191 / 83. **SGC carries the remaining bias**, and α₀ ≈ +1 to
  +1.8 is too high.
- **Density dependence of δ** (paired Gaussian mocks at n̄/2, `dr1_lambda_paired_half_*`, 156+ pairs):
  δ_G(n̄)/δ_G(n̄/2) = 1.001–1.008 (mean +0.45%). The halo δ is at n̄/2 and Abacus at n̄, but transferring it with this
  ratio does not help (joint χ² 525, NGC χ² 218, SGC unchanged), so this stays open.
- **FKP denominator from stacked randoms** (`--test measure --expected-mocks 0..11`; the mocks share the random RA/Dec
  and differ in Z, so the stack has ~1% angular noise per cell). It lowers S1m by 1.0% (NGC) and 1.6% (SGC).
  With the recalibrated halo δ: joint **ω_cdm 0.1201 ± 0.0028, logA 3.066 ± 0.045**, h 0.708 ± 0.009 (+3.8σ),
  χ² 431; without AP, h 0.701. Data `abacus_fit_N25_expected`, windows `dr1_window_products_cic_halorecal`. This is
  the **current best setup**.
- **The stack as the whole geometry** (`--nbar-from`: nbar, FKP weights, noise maps and shot noise from the stack;
  data `abacus_lrg_z0.4-0.6_stacknbar`, halo mocks `dr1_halos_v3_*`, windows `dr1_window_products_stacknbar*`).
  - W4 passes there too: 0.1226 ± 0.0040, 3.089 ± 0.074, 0.677 ± 0.013, χ² 156.
  - But the NGC halo recalibration needs r = 0.83–0.93 (SGC 0.985–1.002), so the NGC window products in this
    geometry are inconsistent. **To find.**
  - Abacus: joint 0.1261, 2.997, 0.702, χ² 632; the errors are 3× smaller than elsewhere, so the profile is suspect.
    NGC: 0.1178, 3.031, h 0.703; SGC: 0.1251, 3.034, h 0.715. Not better than the best setup.
  - Dividing by the stacked expectation but keeping the cell-centre weights gave near-zero-expectation cells huge
    noise (NGC shot noise 2667 → 14684), which is why the geometry has to change as a whole.
- **One-survey covariance** (NUTS, mean of 25 mocks, no `--covariance-of-mean`, `abacus_N25_single_*`): best setup
  ω_cdm 0.1153 ± 0.0088 (−0.5σ), logA 2.907 ± 0.120 (−1.1σ), h 0.687 ± 0.027 (+0.5σ); original setup −1.9σ,
  −0.9σ, +1.3σ. So, at the precision of one survey, the best setup is unbiased in ω_cdm and h.
- **Open:**
  - h is +2–4σ high at 25× the volume in every variant, with α₀ ≈ +0.6–1.5;
  - the NGC window in the stack geometry;
  - the density dependence of δ (halo mocks at n̄/2; the transfer by Gaussian δ ratios did not help);
  - the analytic covariance, still at the early, biased best fit.

**Exact first-layer window (2026-10-16/17, `survey.CellWindow`, commits eea3862–16ffa2c): no calibration at the
variance level.**
- **Idea.** The rotated components (X_m, X_−m) at a cell are linear in the field, so their covariance is linear in the
  spectrum: C_mm'(x) = (1/N) Σ_k ĝ_m ĝ_m'^* P(k, μ_x) + Σ_y g_m g_m' σ²_N(y), with g = D(R_x) ψ(x − ·) W the masked,
  rotated wavelet. A sub-box FFT of g (2 × 8σ_j on a side) gives it exactly, including the correlation of ±m near an
  edge and the local shot noise (white or CIC); the Gaussian moment of each block follows from the 2 × 2 eigenvalues,
  cell by cell. It is DESI's P(k) window (from the randoms, linear in P, independent of cosmology and tracer) one
  level down, at the variances. It replaces the coverage × c(f) transfer and the Gaussian-mock δ; what it does not
  contain is the non-Gaussian edge effect (the Edgeworth part), left for a cumulant transfer or a mock test.
- **Cusp correction.** The sub-lattice sum misses the |k| cusp of P at the origin (the periodic images of ξ at the
  sub-box size: shells converging as Σ n²·n⁻⁴). Per kernel pair: + ĝ_a(0) ĝ_b(0) E(R), E the exact lattice error of a
  Gaussian envelope e^{−k²R²} (full mesh minus sub-lattice, the latter about the cell's line of sight) with R matched to
  the pair's second moment. Unmasked kernels then reproduce `signal_variances` to 5e-4 at pad 2 (l = 0 was 1.2% off).
- **Toy test** (plane edge, 96³, 100 Gaussian mocks): predicted masked / box S1m ratios within ≤ 0.07% of the mocks
  (inside the mock noise) against a 2–7% suppression, at n̄ = 4e-4 and with negligible noise. Before the cusp
  correction the masked kernels near the edge were 0.1% off.
- **DR1 NGC/SGC** (`--test exact`: 38 paired Gaussian mocks with and without the RIC, kernels at 16 cells per
  (distance, density) bin, 1152 cells): residual 0.31% rms (NGC) / 0.77% (SGC), coherent across blocks. A half-split of
  the sample shows it is the **sampling noise of the cell sample** (implied 0.34% / 0.48%), not the kernel.
  - **Control variate** (`CellWindow.control_variate`): the coverage model (f λ_box + λ_N, independent components) on
    every support cell plus the exact − model mean over the samples: 0.20% / 0.29% rms, with an implied sampling noise
    of 0.08% / 0.15% and a mock noise of 0.11% / 0.16%. A run with 80 cells per bin is in progress (NERSC CPU).
  - **RIC**: `survey.RICTransfer`, the per-bin ratio of masked component variances with and without the constraint on
    32 signal-only Gaussian mocks (`--stage ric`, GPU); it reproduces the measured RIC shift to ≲ 0.1%. Linear, so
    tracer-independent; its spectrum dependence is second order in a < 1% effect.
- **Likelihood.** Rows of every covariance entry on the emulator's (k, μ) grid (`covariances(None, grid=RSDGrid)`,
  with the cusp correction as rows and a (k/k_max)^−2.2 tail for the modes beyond k_max that masked kernels reach:
  rows match the callable kernels to 2e-4), compressed by an SVD across cells (rank ≤ 18 at 1e-4) in
  `SurveyWindow.exact`; `s1m_exact_factor` (JAX) evaluates the per-cell eigenvalue moments against the box moment at
  the theory's effective noise (4e-5 against the direct computation on the toy; AP acts on the box part only).
  `S1mTheory` uses it when the window file has an exact part (`dr1_window --test products --exact`: CIC noise, RIC
  transfer). The S21m window stays the calibrated K for now. Costs: ~12 s per cell for the kernels, 13 s for the rows,
  ~10 min once for the envelope rows; 120 processes on a Perlmutter CPU node.
- **80 cells per bin (4320 cells, NGC):** 0.157% rms / 0.44% max against the 38 mocks (mock noise 0.114%, implied
  sampling noise 0.061%); against the **827 paired RIC mocks** (noise 0.027%): **0.108% rms, 0.21% max = 0.10σ rms,
  0.25σ max of one survey**, χ² 2857 / 60. A genuine residual remains: m = 0 blocks +0.1%, m = l blocks −0.15%,
  nearly independent of scale.
- **Not the wide angle:** a toy with the observer 1.4 Gpc/h away (the line of sight tilting by ±18° across the box)
  and local-line-of-sight mocks: the plane-parallel kernels agree to 0.04% rms (inside the mock noise) and the exact
  Kaiser wide-angle kernels (`CellWindow.wide_angle`: six more transforms per component, g n_i n_j) change nothing
  (≤ 0.05% per block). A 192³ toy with J = 3 confirms the σ = 71 Mpc/h kernels to ≤ 0.07%.
- **SGC (3600 cells) against the 827 paired RIC mocks: 0.059% rms, 0.14% max = 0.054σ rms, 0.16σ max** (mock noise
  0.033%), i.e. at the target. Its small residual has the opposite sign pattern to NGC's (m = l blocks +0.1% at σ = 25),
  so the ±0.15% is geometry-dependent.
- **Not the cusp correction:** pad 3 against pad 2 on 162 NGC cells changes the predicted ratios by < 0.005%.
- **Wide angle on the DR1 sample** (exact Kaiser factors on 216 of the 4320 NGC cells): per cell ±1% for the m ≤ l − 2
  blocks of masked kernels at tilted edges, but the weighted net shift of the predicted ratios is only +0.05–0.09%
  (σ = 50–71, m ≤ l − 2) and ~0 at m = l: part of NGC's positive m = 0 residual, not its negative m = l one. Left as a
  refinement; the exact window is used plane-parallel.
- **Likelihood representation, restructured (2026-10-17, 4c62f65).** The SVD of the full rows barely compressed
  (ranks 800–1500 of 3840: the cell-to-cell variation is mostly the coverage). Now the rows minus the coverage model
  f × (unmasked row) are compressed (ranks ≤ 18) and the factor is a control variate: the coverage model summed over
  every support cell through a fine (256 × 8) coverage–noise histogram, plus the exact − model mean over the sample
  cells. The model subtracted on the samples must be the one summed over every cell (the m-averaged noise map, not
  the per-component noise), else the estimator is biased by 0.2%; with that, the JAX factor matches
  `CellWindow.control_variate` to 3e-6 on the toy for four spectra. The sampling noise is then that of the
  correction term only, so 40 cells per bin suffice (~0.085%).
- **Spectrum basis instead of the cell SVD (2026-10-17, 97332a2).** On DR1 the SVD of the rows minus the coverage
  model does not compress at all (rank = number of cells, 72 of 72 at tol 1e-4..1e-3, with or without the cusp rows:
  the masked kernels' (k, μ) shapes are all different), so the window files were 10 GB per cap. The compression is
  now along the spectrum axis: `survey.spectrum_basis` takes the principal components of 2000 biased P_s grids the
  fit's emulator can produce over the prior (`calculators.sample_biased_grids`: cosmology uniform in the emulated
  box, wide bias, counterterm and stochastic ranges; emulator `survey_lrg_z0.51`) up to a 1e-5 reconstruction of a
  held-out half: 100 components. The rows projected on it (`build_exact(basis=)`, `SurveyWindow.exact_basis`, shared
  by every block; `u` (C, 3, 100) in float32) reproduce the per-cell variances of the 72 DR1 cells to 1.3e-6 for an
  independent prior sample (the error is the part of the spectrum outside the span, nothing else). Files: 170 MB
  (NGC) / 143 MB (SGC); `dr1_window --test products --exact --exact-basis <emulator.h5|basis.npy>` builds them, and
  `--exact-from <dir>` or `SurveyWindow.load(path, basis=)` reprojects an SVD-form file in a minute (used for the
  Abacus- and stack-geometry windows, `dr1_window_products_exact_pca`, `..._exact_stack_pca`; the 10 GB files are
  deleted). Also fixed: the loader truncated field names with underscores (`hist_weight` → `hist`), which is what
  had crashed the first exact-window fits.
- **Likelihood bug found by the first Abacus fits (2026-10-17, de44acb).** With the exact window at δ = 1 the joint
  profile gave χ² 875 against 413 for the calibrated histogram window, and at the histogram fit's best point the exact
  theory vector had χ² 47000: `s1m_exact_factor` multiplied the theory's box noise by the shot-noise amplitude
  (1 + α₀) although that noise already carries it (the histogram factor never did), so at the fits' α₀ ≈ 1.2 the box
  reference was inflated and the factor 5–10% low, worst for the noisiest blocks (σ = 25, high m). Diagnosed by
  decomposing the factor at the fit's inputs (`scratchpad/decompose.py`): signal only, the window file reproduces the
  direct 4320-cell kernels of the validation. Fixed, the exact factor is 0.3–1.3% below the calibrated histogram
  factor at the same point (mean 0.7%: the halo edge correction it does not contain). The "halo" mode of the fit
  launcher multiplies the exact window by δ_halo / δ_Gauss of the histogram products (NGC 0.5%, SGC 0.9% on average,
  up to 1.7% at σ = 25) on top of the Gaussian residual: the non-Gaussian edge effect is still a calibrated factor.
  The profiles also showed that the exact factor is cheap once `_chi_moment` uses 16 nodes (22 ms with gradient per
  cap; 64 nodes were 10x slower for nothing: the integrand is periodic and analytic).
- **Abacus fits with the exact window (2026-10-17, profiles, stacked-randoms data, analytic covariance of the
  25-mock mean).** Histogram window with the halo δ (reference): ω_cdm 0.1201, logA 3.066, h 0.708, χ² 413.
  Exact window, δ = 1: 0.1111 / 3.230 / 0.688, χ² 884 (at the reference best fit its S1m vector is 0.3–1.5% above
  the reference's: the exact window is the Gaussian edge effect only, and the halo correction, 0.5–1.7% per block, is
  4–13σ of the 25-mock mean). Exact window × δ_halo/δ_Gauss ("halo" mode, the exact window's own δ, e2eeced): the
  profiles run to the h and α₀ bounds (0.6136, 2.0), χ² 982, also when started from the reference best fit, and at
  the reference point the model has χ² 3118 although its S1m entries differ from the reference's by only ±0.5%: the
  difference is an m-pattern (m = l blocks +0.2–0.7%, m < l −0.1–0.5%) that the covariance strongly penalises.
  Rescaling the cells' noise to the data's maps makes it worse (χ² 3269 at the point), so the 5% noise-level
  difference is not the cause. Reading: δ_halo/δ_Gauss does not transfer the non-Gaussian edge effect onto the exact
  window (δ_halo was calibrated at the halo mocks' spectrum and noise, δ_Gauss at the Gaussian mocks'; the histogram's
  own spectrum dependence does not cancel in the ratio). The clean step is to recalibrate the halo δ against the exact
  window itself: exact factor at the halo mocks' spectrum (halo emulator at the Quijote fiducial, box best-fit biases)
  and noise (n̄ × 0.5), δ_halo,exact = ⟨halo survey / box⟩ / exact. Pending, with the stack-geometry fits.
- **Halo δ recalibrated against the exact window (2026-10-17 evening).** δ_exact(block) = ⟨halo survey / box⟩ of the
  275 calibration pairs over the exact factor at the Quijote truth with the box-fit nuisances (`fast_noap_V60`), the
  survey-side noise doubled for the mocks' n̄ × 0.5 (scratchpad `halo_recal_exact.py`; the factor is captured from
  the W4 likelihood itself, from its evaluation at the point, not the build's). The exact window predicts the halo
  mocks to 0.1–0.9% (NGC δ_exact 0.991–1.001, SGC 0.985–1.000, the deviations at σ = 25 and high |m|), against
  0.3–1.7% for the histogram: most of what the halo δ carried was the histogram's own spectrum dependence. Products:
  `dr1_window_products_exact_pca_hrecal` (exact δ = δ_exact, histogram δ untouched for the second layer). At the
  reference best fit the S1m entries differ from the reference's by ≤ 0.17% (≤ 1.4σ). Abacus profiles (stacked-randoms
  data, started also from the reference best fit):

  | fit | ω_cdm | logA | h | χ² |
  |---|---|---|---|---|
  | reference (histogram + halo δ) joint | 0.1201 ± 0.0028 | 3.066 ± 0.045 | 0.708 ± 0.009 | 413 |
  | exact + δ_exact joint | 0.1240 ± 0.0034 | 3.089 ± 0.047 | 0.678 ± 0.012 | 468 |
  | exact + δ_exact NGC | 0.1194 ± 0.0040 | 3.051 ± 0.061 | 0.669 ± 0.015 | 240 |
  | exact + δ_exact SGC | 0.1338 ± 0.0057 | 3.013 ± 0.074 | 0.675 ± 0.019 | 185 |
  | truth | 0.1200 | 3.036 | 0.6736 | |

  **The h pull was the window**: with the exact window h is unbiased in every fit (+0.4σ, −0.3σ, +0.1σ) where the
  histogram gave +3.8σ. What remains is SGC's ω_cdm (+2.4σ; NGC is fine), which the histogram setup also showed
  (SGC 0.1267 ± 0.0051), and a joint χ² of 468 against 413. The profiles are Minuit profiles; the covariance is the
  analytic one of the 25-mock mean.
- **Baseline consolidated (2026-10-17 night, 0ffb544).** (i) The exact products had carried the *Gaussian* K of the
  second layer (±1.1–1.6% from the halo-calibrated one of `cic_halorecal`); the recalibrated products
  (`dr1_window_products_exact_pca_hrecal`, and `_halos` with the survey-side noise doubled for the halo mocks) now
  carry the halo K. (ii) `S21mTheory` takes the first-layer W of its K/(δW) from the exact window and its own δ
  when the window has one, so one first-layer window serves both layers. (iii) The paper figure (Fig. footprint,
  `scripts/paper/survey.py --halo`) is regenerated with this baseline: on the 25 held-out halo pairs the box × window
  reproduces the footprint to 0.18% rms (max 0.46%, 0.18σ, χ² 94/60) for S1m and 0.33% rms (0.28σ, χ² 50/54) for
  S21m; the residual factor of the first layer is within 0.0–0.9% of one. The paper's survey section, discussion,
  conclusions and a new appendix describe the computed window (paper/main_v2.tex, local). NUTS chains of the joint
  and NGC Abacus fits with this baseline are running (`abacus_N25_exactpca_exp_hrecal_nuts_*`); SGC alone is too
  noisy to fit separately (user's call).
- **Chain cost (2026-10-17 night).** The exact factor's gradient (22 ms per cap, four evaluations per likelihood call
  with both layers) makes the NUTS chains ~20x slower than with the histogram window: 68 min without a first 200-step
  checkpoint on the Mac. Sharing one factor between the layers is NOT an option: at the best fit the factor at the
  tree spectrum (the second layer's) differs from the one-loop one by up to 1.5% (0.8% rms). `_chi_moment` now uses
  12 nodes (exact to 1e-9 at the DR1 eigenvalue ratios); 80 cells per bin are kept (40 would add a 0.085% per-block
  sampling pattern, 0.3-0.5σ per block on the 25-mock mean). The chains run on a Perlmutter GPU node instead
  (JAX 0.9.1 on A100s in cosmodesi): fit inputs synced to outputs/ there. **Measured:** the full joint log-posterior
  gradient on the Mac is 26 ms with the exact window vs 5.7 ms with the histogram (4.6x, not 20x; 11 vs 1.9 ms for
  the value); the factor alone is 10.6 ms per cap at 12 nodes (1.0 ms on an A100), but the chain's warmup on the
  A100 took 503 s against 31 s on the Mac (the rest of the likelihood is many small ops), so the GPU is a net loss
  and the chains run on the Mac (joint and NGC side by side, `scratchpad/mac_chain.sh`). The slow checkpoints are
  the sampler's tree depth on the 25x-tightened posterior, not the window.
- **GCcomb convention (2026-10-18).** desi-clustering never fits NGC and SGC as separate observables: `combine_stats`
  (lsstypes `sum`) averages the caps' spectra with their FKP normalisations as weights, and the window matrices and
  covariances the same way, so the GCcomb spectrum is the estimator on the union of the footprints. The WST estimator
  combines identically (weighted means over cells): `fit_rsd.py --survey NGC SGC --combine-caps` now fits one
  GCcomb vector: S1m entries = the Σw-weighted mean of the caps (W_c = the cap's FKP weight sum, from the window
  file), S21m entries = the mean weighted by W_c × S1m_c of their first-layer block (the union's S2m over its S1m);
  the theory is `calculators.CombinedCaps` over the per-cap theories (each cap keeps its own window), the analytic and
  extra covariances are projected with the Jacobian at the data mean. The likelihood cost is that of the joint fit
  (both caps' windows are still evaluated); the data vector is half as long. Profile (a5a8421, same setup as the
  joint one): ω_cdm 0.1241 ± 0.0034, logA 3.050 ± 0.053, h 0.683 ± 0.013, α₀ 0.50 ± 0.22, χ² 233 / 83, against
  the joint's 0.1233 / 3.095 / 0.689, χ² 455 / 197 (truth 0.1200 / 3.036 / 0.6736): the same cosmology within
  0.3σ, logA 1σ closer to the truth. NGC NUTS chain (exact window, recalibrated δ): 0.1189 ± 0.0038, 3.029 ± 0.061,
  0.679 ± 0.014 (pulls −0.3, −0.1, +0.4σ), Gelman–Rubin 1.00. The joint chain was stopped at 600 steps per chain
  (R = 1.01, checkpoints kept) in favour of the GCcomb chain (`abacus_N25_exactpca_exp_hrecal_GCcomb_nuts`).
  **GCcomb chain (converged 06:14, 3 h):** ω_cdm 0.1240 ± 0.0032 (+1.2σ), logA 3.033 ± 0.049 (−0.1σ),
  h 0.684 ± 0.012 (+0.9σ), b₁ 1.856 ± 0.055, α₀ 0.52 ± 0.22, c_v 0.79 ± 0.18; χ² 233 / 83 at the best fit. This
  is the baseline result of the exact window on the 25-mock mean: h recovered (the histogram window's +3.8σ is
  gone); ω_cdm +1.2σ from SGC. Triangle: `abacus_N25_exactpca_exp_hrecal_GCcomb_triangle.png`.
- **Paper (2026-10-18, local):** the NGC Abacus chain is in paper/main_v2.tex as the end-to-end test of the footprint
  subsection (paragraph + Fig. `survey_abacus.pdf`, `scripts/paper/survey.py --abacus`, macros
  `numbers_survey_abacus.tex`: pulls −0.3 / −0.1 / +0.4σ at 35 (Gpc/h)³, χ² 206/114; the histogram-window NGC
  profile's h = 0.698 is +1.8σ away), with SGC and the GCcomb combination deferred to the DESI analysis; the section
  intro, Discussion and Conclusions mention it.
- **Paper restructured around the Abacus result (2026-10-18, local, v2 and v3):** a new Sec. II "Overview" right after
  the Introduction opens with `figures/overview.pdf` (`scripts/paper/survey.py --overview`: the NGC 25-mock-mean data
  vector with the best-fit model and (X − model)/σ, next to the posterior triangle with truth and MAP), says what
  the data are, what the fit is, and that three pieces (model, analytic covariance, computed window) make it
  possible, then hands over to the rest of the paper; the end-to-end paragraph of the survey section refers back to
  it (its own figure removed). `abacus_vectors()` caches data/σ/model in outputs/paper/survey_abacus_vectors.npz.
- **Residual factor:** on top of the exact window, the per-block ratio of the 827 Gaussian mocks to its prediction
  (≤ 0.2%, second order in the spectrum) can be carried as δ (`delta_res_<cap>.npz`); the Abacus fits are run with
  δ = 1 and with it.
- **Next:** the Abacus fit with the exact window and δ = 1; the cumulant (non-Gaussian edge) transfer; the second layer.

**Next:**
1. Redo the halo cut-sky run with the fixed `box_to_survey` (NERSC), then the halo δ / K and W4 (with the ~1% c000
   fiducial offset in mind, or a Quijote-fiducial survey emulator).
2. Abacus: the measured α n_r / (nbar V) at NERSC (18 random files) and an Abacus measurement with that denominator;
   the data's noise level against the model's (the NGC α₀ = −0.5); fit with the new halo δ.
3. Then the NUTS 25-mock fit, the analytic covariance at its best fit, and a survey emulator with `--ap-order 3`.
4. The S1m goodness of fit (option 3 for now: the analytic covariance for parameters only) and the EZmock covariance (C1).

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
