# Paper plan: restructure and AP-baseline updates

Status: 2026-10-15. Working draft is `paper/main_cv.tex` (fog counterterm c_v baseline); `paper/` and `scripts/paper/`
are **not tracked in git** (copy them over or commit them before continuing elsewhere).

## Current baseline (code)

`scripts/fit_halo_baseline.sh`: S1m (sigma >= 25 Mpc/h, q = 0.8) + S21m (25 -> 70.7 Mpc/h), cubic biases (relation
priors), shared fingers-of-God counterterm c_v of both layers, non-Gaussian second-layer template (per-l marginalized),
free W, n_s fixed, NUTS, and **full Alcock-Paczynski** in every piece of the data vector (since 2026-10-15, commit
8bc89b1): emulator `outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic_ap3_base.h5` (git-ignored; copies here
and on Perlmutter under the repo's `outputs/emulators/`). `--no-ap` switches to the no-AP emulator (same no-AP model,
~5x faster). The ap3 emulator covers only the baseline data vector; other S21m cuts need retraining (~45 min on one
Perlmutter node; train with MPI, then fit from the checkpoint in one process to avoid the end-of-training OOM).

AP result (V = 60, `outputs/inference/rsd/fullap_{ap,noap}_V60`): sigma(h) 0.0061 -> 0.0074 (+21%), h +0.3 sigma,
omega_cdm and logA unchanged, chi2 19.3 vs 19.4; same posterior as the earlier second-layer-only AP.

## Done in main_cv.tex

- New subsection "Alcock--Paczynski distortions" (`sec:haloap`, after "Second layer"): q ratios in Mpc/h (depend on
  Omega_m only, no new parameter), observed-frame filters/harmonics with true-frame clustering, volume factor, cubic
  polynomial in ln q (exact at q = 1, Chebyshev nodes over the emulator range), accuracy.
- Sec. V intro lists four changes (adds fiducial-cosmology coordinates); "Parameters and emulation" mentions the AP
  terms; emulator accuracy updated from an 8-point validation (S1m 4e-4 max / 6e-5 median, S21m 9e-4 / 2e-4).
  Note: that validation is at q = 1 only, and P_l had one point off by 4% (median 0.6%) -- check before P_l fits.
- `Alcock:1979mp` added to `paper/references.bib`.

## Updates by priority (AP baseline)

1. **Rerun the halo fits with AP** (`scripts/paper/run_halo_fits_cv.sh` with `--ap` / the new wrapper): WST V25/V60,
   noW, joint WST+P_l V25/V60, n_s-free NUTS, and **also the P_l-alone fits** (currently reused from older no-AP runs).
   First check the 4% P_l emulator point against the old emulator. The sigma_j1 = 17.6 robustness run is not covered
   by the ap3 emulator: retrain or keep it without AP and say so.
2. **Regenerate products**: `WSTFAST_HALO_VARIANT=cv scripts/paper/halos.py` -> `numbers_halos_cv.tex`, table, contour
   and best-fit figures; captions say AP is included.
3. **Rewrite the h interpretation** in "Baseline fit and comparison with the power spectrum": AP costs ~20% in sigma(h)
   for both WST and P_l and only h moves, since the shape fixes omega_m h^2-like combinations while AP rescales through
   Omega_m; the WST is sensitive through its anisotropic filters. Recheck the WST/P_l ratio on h.
4. **Robustness section** (n_s free, W, Delta chi2 of physical biases): new numbers; check the claims hold with AP.
5. **Fisher forecasts** (covariance section ~l.1338, discussion ~l.1572) are without AP: rerun with AP or state it.
6. **Small consistency edits**: intro roadmap (~l.144), abstract clause, conclusions bullet (one AP recipe for all
   components), a sentence in the survey-geometry section (fitted without AP).
7. **Older pending items**: rest of the second-layer non-Gaussianity updates beyond Sec. IV G; the 8.8 Mpc/h template.

## Proposed restructure (not started; user to decide)

Problem: Sec. IV (real-space matter) mixes model building, ingredient tests and cosmology fits; Sec. V rebuilds the
model for tracers, so the model appears twice and the jump feels abrupt. A separate redshift-space-matter *theory*
section is not recommended: RSD and bias share the Z kernels (beta includes the velocity term), so RSD matter is just
the b1 = 1, no-shot-noise limit and would duplicate the machinery with no fits to show. The natural middle rung is a
*validation* one: the dark-matter subsample in redshift space (Poisson tracer, b1 = 1), now a single paragraph in
"Validation of the ingredients".

Proposed outline (theory once, validation as a ladder of fields):

1. Introduction
2. The WST (as now)
3. Model, written for a tracer in redshift space with the real-space matter limit (f = 0, beta = (0,1,0,0,0), no shot
   noise) stated at each step:
   - 3.1 Field: bias expansion, P_s, shot noise
   - 3.2 First layer: Gaussian limit per |m| block (m-summed for isotropic matter), Edgeworth with tree-level
     cumulants, Poisson terms, cubic biases, FoG counterterm c_v
   - 3.3 Second layer: moduli as biased tracers (response, Gaussian noise, connected noise -- Delta N where it works,
     i.e. real-space matter, free amplitudes otherwise), non-Gaussianity E_Y, reduced coefficients
   - 3.4 Alcock-Paczynski distortions
   - 3.5 Parameters and emulation (once: polynomial in beta, c_v, ln q); notation table here, covering both cases
4. Simulations, measurements and likelihood (moved after the model, or kept before; short either way)
5. Real-space matter: first-layer checks, response/noise of moduli, fiducial ladder, parameter recovery
6. Redshift space:
   - 6.1 Dark-matter subsample (new, from the existing paragraph): discreteness, LOS blocks, one c_v for both layers,
     why matter needs ~2 sigma_v and halos ~1 sigma_v
   - 6.2 Halos: response/noise validation, baseline fit with AP vs P_l, AP and h, robustness and n_s
7. Analytic covariance (as now)
8. Survey geometry (as now)
9. Discussion; 10. Conclusions; appendices

Costs: substantial rewrite of Secs. IV-V (~400 lines moved/merged; equations mostly kept); real-space matter uses the
m-summed S1/S21 (needs a clear isotropic-limit sentence); matter-only pieces (Delta N) must be flagged; heavier model
section up front (lead each subsection with the matter limit).
Lighter alternative: keep the order, add a roadmap at the start of Sec. V, mirror the subsection headings, move AP out
of the model next to the fits (~1 h, but the model is still presented twice).
Recommendation: full restructure, drafted in a copy (`paper/main_v2.tex`) to compare with `main_cv.tex`.
