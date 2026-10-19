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

## q = 2 band powers: the small-scale Gaussian information (2026-10-19)

Question (2026-10-18): at V = 60 (Gpc/h)^3 the real-space matter P_0 to k = 0.2 gives sigma(omega_cdm) = 0.0008 against
0.0022 for the Sec. IV WST baseline (S1m at q = 0.8, sigma >= 25, + S21m). Can the WST alone extract that Gaussian
information, so that the paper does not need the explicit WST + P(k) combination? Real-space matter, (omega_cdm, ln A_s,
h), n_s fixed (no n_s-free chains: too expensive for a parameter the paper does not constrain).

**Answer: yes, through the first-order coefficients at q = 2, which the measurements already contain** (q in {0.5,
0.8, 1, 2}). S1(q = 2) = <|psi_{jl} * delta|^2> = sum_k |psi_{jl}(k)|^2 P(k) is a band power of the wavelet: exactly
linear in P(k), the Edgeworth factor carries q(q - 2) and vanishes identically, so the model is the one-loop spectrum
with the counterterm c_0 already used for P(k) -- no new theory. In u = sigma^2 k^2 the band is a Gamma(l + 3/2)
weight: k_peak = sqrt(l + 3/2)/sigma, so sigma = 17.7 with l <= 6 reaches k84 = 0.18 and sigma = 12.5 with l <= 2
k84 = 0.18. Code: `--stats band` (`spectrum.BandProjection`, `PowerTheory`, `inference.LinearCompression`), options
`--band-min-scale/--band-max-scale/--band-ells/--band-kmax/--band-cut/--band-counterterm`, and `--compress-m`
(commits 3a5c4d3, 11381f5). The fits are in `outputs/inference/real/{bands17,bands12k,full17,pk02}_*`.

Three traps, all hit on the way:

1. **Overlap (retracted).** The first fits with q = 0.8 and q = 2 on the same blocks exploded (chi^2 ~ 4e5), which
   was blamed on their ratio measuring the Edgeworth factor beyond the model's accuracy; the cause was trap 2. With
   the PCA cut the overlap is harmless: S1m(q = 0.8) >= 25 + S1(q = 2) >= 25 fits at V = 1 / 8 / 64 (chi^2 0.1 /
   0.8 / 6.7 for 42 dof, shifts <= 0.4 sigma), and S1m(q = 0.8) >= 25 on top of bands12k gives chi^2 0.1 / 0.9 /
   7.0 (46 dof) with omega_cdm 0.0 / ln A_s -0.5 / h +0.6 sigma at V = 64. It just adds nothing: Fisher at V = 60
   0.0018 / 0.0131 / 0.0044 for S1(q = 2) >= 25 alone and 0.0018 / 0.0133 / 0.0045 with q = 0.8 added (separate
   c_0: 0.0019 / 0.0133 / 0.0045); bands12k 0.0008 / 0.0077 / 0.0022 alone, 0.0008 / 0.0079 / 0.0023 with q = 0.8,
   0.0008 / 0.0081 / 0.0024 with q = 0.8 and S21m. The reduced moments of the filtered field (the Edgeworth factor)
   depend on the local slope, which the bands already fix, not on the amplitude. `build_problem` now prints a
   notice instead of refusing the overlap (2026-10-19).
2. **Numerical low rank.** The bands are smooth functionals of the same P-hat, so on the lattice their sample
   correlation has eigenvalues decaying geometrically to the float32 floor (1e-14): exact linear relations across
   boxes (checked on disjoint halves) and a garbage inverse (chi^2 = 6067/30 at V = 1). `band_compression` keeps the
   principal components of the band block above `--band-cut` (1e-4): 13 of 35 at sigma >= 17.7, l <= 6; 14 of 38 for
   the k-matched set. The Fisher errors are the same with cut 1e-3 or 1e-5 (0.0009 in omega_cdm in all three).
3. **Reach.** Bands whose weight extends past k ~ 0.2 are biased at V = 64 like P_0 would be: sigma = 12.5 with
   l >= 3 (omega_cdm -1.4 sigma, chi^2 58/9), and the sigma = 8.8, l = 0 band although its k84 is 0.18 (omega_cdm
   -2.3 sigma, chi^2 16/10). The vectors that pass V = 64: `bands17` = sigma >= 17.7, l <= 6 (chi^2 1.3/9) and
   `bands12k` = sigma >= 12.5 with k84 <= 0.2, i.e. 12.5 at l <= 2 plus 17.7-70.7 at l <= 6 (chi^2 1.4/10).

`--compress-m` averages the |m| blocks of the real-space S1m/S21m within their equally distributed classes
(m = 0 / |m| >= 1 per layer): 222 -> 84 entries, Hartlap x Percival 1.65 -> 1.17, 10% smaller errors at no cost; the
|m| >= 1 first-layer noise amplitudes are tied to one parameter per class (otherwise the posterior is flat along their
differences and NUTS cannot start; 34 -> 22 parameters).

Information budget (Fisher at V = 60 from the fit's own likelihood, central differences of the emulator; sigma of
omega_cdm / ln A_s / h; `scratchpad fisher_bands.py`):

| vector | n | sigma(omega_cdm) | sigma(ln A_s) | sigma(h) |
|---|---|---|---|---|
| Sec. IV baseline: S1m(q = 0.8) >= 25 + S21m, uncompressed | 222 | 0.0021 | 0.0169 | 0.0057 |
| same, compressed | 84 | 0.0019 | 0.0151 | 0.0051 |
| S1m(q = 0.8) >= 25 alone, compressed | 36 | 0.0019 | 0.0148 | 0.0050 |
| S1(q = 2) >= 25 alone, l <= 6 | 11 | 0.0015 | 0.0128 | 0.0043 |
| S21m alone (sigma_1 >= 17.7), compressed | 48 | 0.0105 | 0.1845 | 0.0655 |
| P_0 to 0.1 / 0.15 / 0.2 / 0.25 | 8-20 | 0.0019 / 0.0010 / 0.0008 / 0.0007 | 0.0195 / 0.0075 / 0.0070 / 0.0062 | 0.0060 / 0.0021 / 0.0019 / 0.0015 |
| bands at sigma = 17.7 only, l <= 6 | 6 | 0.0010 | 0.0136 | 0.0047 |
| bands17 (sigma >= 17.7, l <= 6) | 13 | 0.0009 | 0.0077 | 0.0022 |
| bands17 + S21m | 61 | 0.0009 | 0.0079 | 0.0023 |
| full WST: S1m(q = 0.8) >= 25 + S21m + bands at 17.7 | 90 | 0.0009 | 0.0092 | 0.0028 |
| bands12k (sigma >= 12.5, k84 <= 0.2) | 14 | 0.0008 | 0.0077 | 0.0022 |
| bands12k + S21m | 62 | 0.0008 | 0.0080 | 0.0023 |

Reading: for these three parameters in real-space matter the WST information is two-point information. The current
baseline carries what P_0 carries to k ~ 0.1; its q = 0.8 first layer is a lossy copy of the q = 2 bands at the same
scales (0.0019 vs 0.0015), S21m on its own constrains almost nothing and adds nothing on top of the bands, and the
whole gain comes from the extra octave (sigma = 17.7 alone = P_0 to 0.15). The bands cannot beat P_0 at the same reach
because they are a filtered P_0; they match it to ~10% while P_0 to 0.25 (which also passes V = 64: chi^2 6.6/16,
+0.5 sigma in omega_cdm) stays 15-25% ahead. n_s free (Fisher only): bands17 0.0020 / 0.0145 / 0.0026 / 0.0115
(n_s) against P_0 to 0.2 0.0018 / 0.0137 / 0.0024 / 0.0111.

Fits (NUTS chains at V = 60, Minuit profiles at V = 1 / 8 / 64; shifts from the truth in sigma; `collect_bands.py`):

| vector | V = 60 chain: sigma (shift) omega_cdm / ln A_s / h | chi^2/dof at V = 60 | V = 64 profile: shifts; chi^2 | V = 8 |
|---|---|---|---|---|
| P_0 to 0.2 | 0.0008 (+0.90) / 0.0073 (-0.75) / 0.0020 (+0.55) | 4.9/12 | +0.89 / -0.75 / +0.56; 5.2/12 | 0.7 |
| Sec. IV baseline, uncompressed | 0.0022 (+0.25) / 0.0172 (-0.17) / 0.0057 (+0.23) | 127/188 | -- | -- |
| bands17 | 0.0009 (+0.17) / 0.0078 (-0.43) / 0.0022 (+0.47) | 1.2/9 | +0.12 / -0.38 / +0.43; 1.3/9 | 0.2 |
| bands12k | 0.0008 (+0.00) / 0.0079 (-0.43) / 0.0023 (+0.54) | 1.3/10 | -0.04 / -0.48 / +0.59; 1.4/10 | 0.2 |
| bands17 + S21m | 0.0009 (-0.12) / 0.0081 (-0.27) / 0.0023 (+0.43) | 17.7/39 | -0.11 / -0.31 / +0.46; 18.6/39 | 3.7 |
| bands12k + S21m | 0.0008 (-0.20) / 0.0081 (-0.31) / 0.0023 (+0.52) | 17.7/40 | -0.20 / -0.32 / +0.51; 18.5/40 | -- |
| full WST (S1m q = 0.8 >= 25 + S21m + bands at 17.7) | 0.0009 (-0.26) / 0.0092 (+0.14) / 0.0028 (-0.12) | 22.5/68 | -0.25 / +0.14 / -0.13; 23.6/68 | 4.8 |
| with the sigma = 8.8, l = 0 band (`bandsk02`) | 0.0007 (-1.92) / 0.0075 (+0.51) / 0.0021 (-0.01) | 15.0/10 | -2.26 / +0.66 / -0.04; 16.0/10 | 2.0 |

The chains at V = 60 reproduce the Fisher errors; every vector but the last is unbiased through V = 64 (the P_0 shifts
of +0.9 sigma in omega_cdm at V = 60-64 are the known pull of the one-loop model, with the same sign in the bands'
best fits at the 0.1-0.5 sigma level).

Band-by-band residuals at the bands17 best fit (`outputs/paper/band_residuals_bands17_3p_prof_V64.pdf`, scratchpad
`band_residuals.py`): below 0.1% (0.05 of the single-box scatter) for every band at sigma >= 17.7; at sigma = 12.5,
l = 4 and 5 sit +0.10% and +0.35% above the model (+1.5 and +5.6 sigma at V = 64), the one-loop + c_0 model running
out at k ~ 0.2.

Consequences for the paper (user to decide):

- The real-space baseline can be the q = 2 first layer (bands12k, or bands17 + the sigma = 12.5, l <= 2 bands) plus
  S21m with the template: the WST alone then matches P_0 to k = 0.2 and the explicit WST + P(k) combination is not
  needed. The q = 0.8 first layer should go: it is a lossy copy of q = 2 and adds nothing on top of it.
- The honest message is that the second layer adds nothing measurable for (omega_cdm, ln A_s, h) in real-space
  matter; the first layer at q = 2 is the whole story there. Whatever the WST gains over P(k) has to come from the
  halo/redshift-space analysis (bias, FoG, n_s) -- which is the paper's subject anyway.
- Halos: `S1mTheory` at q = 2 is the m-resolved band power (the Edgeworth factor vanishes identically; checked
  numerically against `BandProjection`), so `--stats s1m --q 2` gives the redshift-space bands with no new code,
  except the PCA cut (`band_compression` keys on `S1q2_*` labels; extend it to the S1m block at q = 2). Not tried.
