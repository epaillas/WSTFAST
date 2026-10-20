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
| Sec. IV baseline + bands12k (q = 0.8 >= 25 + S21m + bands) | 0.0008 (-0.13) / 0.0081 (-0.36) / 0.0024 (+0.51) | 23.5/76 | -0.13 / -0.36 / +0.49; 24.7/76 | 5.0 |
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

### Halos in redshift space (2026-10-19, same night)

`S1mTheory` at q = 2 is the m-resolved band power of P_s(k, mu) (the Edgeworth factor vanishes identically), so the
halo band powers need no new theory: `--stats s1m --q 2.0` with the PCA cut, which now also acts on the S1m block at
q = 2 (`inference.band_entries`, commit 0d53d48). Data: the 1188 halo boxes (q in {0.5, 0.8, 1, 2} stored). Emulator:
the no-AP `rsd_biased_basis_taylor_4p_ir_fog_cubic.h5` (sigma >= 17.7, l <= 4; the AP emulator covers only sigma >=
25), baseline nuisances (cubic relation biases, c_v, template, W), n_s fixed, V = 60. Runs in `outputs/inference/rsd/q2/`.

Two lessons before the numbers. (1) Without the cut the q = 2 block gives chi^2 = 291/44 at V = 1 (38 of 60
components survive 1e-4), as for matter. (2) The halo posteriors with the q = 2 bands have a flat valley along
(ln A_s, c_v, alpha_0, c_0): Minuit's Hesse errors at the minimum are meaningless there (0.0086 on ln A_s with a
"+6 sigma" shift, while the q = 2 block at the baseline's best fit has chi^2 = 11/38 and every entry within 0.7
sigma). The widths below are profile likelihoods (chi^2 scans with one parameter fixed, `--fix`, parabola fits;
scratchpad `scan_summary.py`) or chains; the shifts are the scan minima.

| vector (V = 60, no AP, n_s fixed) | sigma(omega_cdm) | sigma(ln A_s) | sigma(h) | shift of the minimum [sigma] | chi^2/dof |
|---|---|---|---|---|---|
| Sec. V baseline: S1m(q = 0.8) >= 25 + S21m (chain `paper_cv_wst_nsfixed_V60`) | 0.0027 | 0.048 | 0.0063 | -0.7 / 0.0 / -0.2 | 19.4/83 |
| P_l to 0.12 (chain) / 0.15 / 0.2 (profiles) | 0.0030 / 0.0026 / 0.0021 | 0.052 / 0.045 / 0.038 | 0.0050 / 0.0044 / 0.0035 | -0.7 / +0.8 / -0.3; -0.7 / +0.9 / -0.5; -0.8 / +1.3 / -0.6 | 2.7/12; 4.7/21; 18.8/33 |
| S1m(q = 2) >= 25 (l <= 4) + S21m (scans; partial chain 0.0027 / 0.048 / 0.0052) | 0.0028 | 0.054 | 0.0051 | -0.6 / +0.7 / -0.1 | 7.1/61 |
| S1m(q = 2) >= 17.7 (l <= 4) + S21m (scans) | 0.0025 | 0.044 | 0.0049 | -0.3 / +0.8 / -0.6 | 13.1/70 |
| S1m(q = 0.8) >= 25 alone, or q = 0.8 + q = 2 on the same blocks | valley; Hesse errors 0.0013 / 0.010 / 0.0035 with +3.5 and +4.3 sigma "shifts" in ln A_s: not usable without S21m | | | | 12.8/44; 23.7/82 |

Reading: the same picture as in real space, one octave lower in reach because the model is. The q = 2 first layer
at sigma >= 25 reproduces the baseline (both are the P_s(k, mu) information to k ~ 0.11, where P_l to 0.12 sits
too); at sigma >= 17.7 with l <= 4 (k84 = 0.155) it matches P_l to 0.15 (0.0025 / 0.044 / 0.0049 against 0.0026 /
0.045 / 0.0044), 10-25% better than the baseline, unbiased. S21m is needed in redshift space: without it the first
layer alone (at either q) sits in a bias-counterterm valley. Reaching P_l to 0.2 (0.0021 / 0.038 / 0.0035) needs the
bands at sigma = 17.7 with l = 5, 6 and at sigma = 12.5 (l <= 2), which no halo emulator covers: retrain the no-AP
(or AP) emulator with S1 at j >= 4 and l <= 6 (~45 min on a Perlmutter node; user's call), then redo the scan to V =
64 since P_l itself drifts to +1.3 sigma in ln A_s at k = 0.2. NUTS chains of both q = 2 vectors are running
(`q2_25_s21_chain_V60`, then `q2_17_s21_chain_V60` from its proposal); they converge slowly along the valley.

### Does the WST add to P_l for halos in redshift space? (2026-10-19)

Same question as in real space, V = 60, no AP, n_s fixed, 1188 boxes; widths from chi^2 scans with one parameter fixed
(Minuit's Hesse errors are unreliable whenever the first layer is in the vector: they gave 0.036 on ln A_s for P_l +
baseline, the scan 0.046), shifts of the scan minima in sigma; runs in `outputs/inference/rsd/q2/pk015_*`, `pk02_*`.

| vector | sigma(omega_cdm) | sigma(ln A_s) | sigma(h) | shift of the ln A_s minimum |
|---|---|---|---|---|
| P_l to 0.15 | 0.0028 | 0.045 | 0.0044 | +0.9 |
| P_l to 0.15 + S21m | -- | 0.048 | -- | +0.4 |
| P_l to 0.15 + S1m(q = 0.8) >= 25 (no S21m) | -- | 0.049 | -- | +0.1 |
| P_l to 0.15 + Sec. V baseline (q = 0.8 >= 25 + S21m) | 0.0026 | 0.046 | 0.0043 | 0.0 |
| P_l to 0.15 + S1m(q = 2) >= 25 + S21m | -- | 0.047 | -- | +0.5 |
| P_l to 0.15 + S1m(q = 2) >= 17.7 + S21m | -- | 0.043 | -- | +0.8 |
| P_l to 0.2 | 0.0024 | 0.039 | 0.0040 | +1.4 |
| P_l to 0.2 + Sec. V baseline | -- | 0.036 | -- | +0.5 |

Reading: nothing in the WST adds to P_l at the same reach for (omega_cdm, ln A_s, h). S21m alone, the q = 0.8 first
layer alone and the full baseline leave P_l to 0.15 where it is (differences of a few percent either way are the
Hartlap/Percival cost and scan noise); the q = 2 bands at sigma >= 17.7 give 5% on ln A_s, which is their slightly
larger reach (k84 = 0.155 at l = 4) and the m-resolved angular weighting. On top of P_l to 0.2 the baseline gives
8% on ln A_s and pulls the minimum from +1.4 to +0.5 sigma, i.e. it anchors the amplitude where P_l's one-loop model
starts to drift, not new information. The joint WST + P_l(0.12) gain reported in Sec. V (omega_cdm 0.0030 -> 0.0026,
ln A_s 0.052 -> 0.047) is the first layer's reach beyond k = 0.12, not non-Gaussian information. The real-space
conclusion therefore carries over: for these three parameters the WST is a filtered two-point function, and the
second layer and the q < 2 moments bring nothing measurable at perturbative scales. The halo WST's competitiveness
with P_l comes from its first layer; the gain beyond P_l at k = 0.2 would have to come from the bands at sigma = 12.5
and l = 5, 6 (emulator in training) and from checking which of the two stays unbiased there.

### Halo band powers to k ~ 0.2: the bands12 emulator (2026-10-19, afternoon)

Follow-up of the previous subsection: the q = 2 first layer reaches P_l to 0.2 only with the bands at sigma = 17.7, l =
5, 6 and at sigma = 12.5, l <= 2, which no halo emulator covered. New emulator
`outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic_ap3_bands12.h5` (git-ignored; copies here and on Perlmutter):
the ap3_base settings (S21 of the baseline, cubic biases, fog, ir, kmax 0.3, ap_order 3, the same bounds, 369 Taylor
nodes) with S1 at j >= 4 (sigma >= 12.5) and l <= 6 (`train_emulator.py --s1-min-scale 12.5 --s1-ells 0 1 2 3 4 5 6`,
new option; commit b279657), 42 first-layer coefficients against 20. Trained in 56 min of one Perlmutter CPU node
(64 then 32 MPI ranks, single-process fit from the checkpoint; the per-node cost is that of ap3_base, the first-layer
cumulants are not the bottleneck). Validation at 2 random points (`.validation.json`): S1m max 1.8e-4 / median 1.2e-4
relative (ap3_base: 3.5e-5 / 2.7e-5; the new blocks are the hardest), P_l 5.7e-3 / 3.3e-3 and S21m 2.5e-4 / 1.4e-4
(identical to ap3_base: its P_s grid equals that of ap3_base to the bit at 6 points of the box and differs from the
no-AP fog_cubic emulator by 1.2e-4, so P_l to 0.2 fitted with it reproduces the `pk02` profiles to 0.02 sigma and the
P_l references need no rerun; `pk02_b12*`).

Data vector (`fit.py --stats s1m s21m --q 2.0 --s1-min-scale 12.5 --s1-ells 0 1 2 3 4 5 6 --s1-kmax 0.2`, the new
per-scale multipole cut on the 84th-percentile wavenumber of the band, `inference.band_k84`, shared with the band
powers): sigma = 12.5 at l <= 2 plus sigma >= 17.7 at l <= 6 = 38 S1 coefficients, 146 m-resolved entries, 75 principal
components kept by the PCA cut (1e-4), plus the 54 S21m entries of the baseline (sigma_1 = 25 -> 70.7, white term,
template per l): 129 data points, 31 parameters, the baseline nuisances, n_s fixed. Runs in `outputs/inference/rsd/q2/`:
`q2_12k_s21[_ap]_V{60,64}` (profiles) and the scans `q2_12k_s21[_ap]_{logA,omega,h}<value>_V{60,64}`; the reach
ladder `q2_12k{15,18}_s21[_ap]_*` (the same vector with `--s1-kmax` 0.15 / 0.18); the P_l references `pk02[_ap]_*`,
`pk015_*`, `pk018_*` (no-AP emulator; scans at V = 60 and 64 for both AP settings at 0.2). Scan launchers and the
parabola summary: `halo_bands12.sh`, `scan_chunk.sh`, `scan_summary.py` in the emulator session's scratchpad (the
pattern of the previous subsection).

**AP bug found on the way (fixed, commit 9e67ad0).** The first AP fit landed at chi^2 = 198 against 60 without AP, and
the AP model differed from the no-AP one at alpha = 1, where the AP polynomial is exact. Cause: `s1m_ap_projection`
summed the lattice modes to k = 0.27 (so that the stretched modes stay on the P_s grid) while `S1mProjection` sums them
to the grid's 0.3; the sigma = 12.5, l = 2 band keeps 0.17% of its weight between the two (1e-14 for the baseline's
sigma >= 25, which is why the paper's AP fits never saw it), two orders of magnitude above its error at V = 60. The AP
projection now takes the grid's kmax (the modes a stretch carries beyond the grid read its last row, a tail of 2e-4
at most): identical chi^2 at alpha = 1, 60.2 against 59.6 at the no-AP best fit. The paper's AP baseline is unchanged.

Results (widths from chi^2 scans with one parameter fixed, parabola fits; shifts of the scan minima from the truth
in sigma; n_s fixed; P_l to 0.2 rerun with AP and scanned at V = 60 and 64 for both settings):

| vector | AP | V | sigma(omega_cdm) | sigma(ln A_s) | sigma(h) | shifts omega_cdm / ln A_s / h | chi^2/dof |
|---|---|---|---|---|---|---|---|
| P_l to 0.2 | no | 60 | 0.0024 | 0.039 | 0.0040 | -0.9 / +1.4 / -0.7 | 18.8/33 |
| P_l to 0.2 | no | 64 | 0.0024 | 0.038 | 0.0039 | -1.0 / +1.4 / -0.8 | 20.0/33 |
| P_l to 0.2 | yes | 60 | 0.0024 | 0.039 | 0.0038 | -1.2 / +1.5 / -0.5 | 18.1/33 |
| P_l to 0.2 | yes | 64 | 0.0023 | 0.038 | 0.0037 | -1.3 / +1.6 / -0.5 | 19.2/33 |
| bands to k84 = 0.2 + S21m | no | 60 | 0.0020 | 0.035 | 0.0033 | +1.4 / -0.3 / +1.5 | 59.6/98 |
| bands to k84 = 0.2 + S21m | no | 64 | 0.0019 | 0.034 | 0.0032 | +1.4 / -0.3 / +1.5 | 63.4/98 |
| bands to k84 = 0.2 + S21m | yes | 60 | 0.0020 | 0.035 | 0.0034 | +0.8 / 0.0 / +1.5 | 59.7/98 |
| bands to k84 = 0.2 + S21m | yes | 64 | 0.0019 | 0.034 | 0.0033 | +0.8 / 0.0 / +1.5 | 63.5/98 |
| bands to k84 = 0.18 + S21m | no | 60 | 0.0023 | 0.042 | 0.0038 | +0.6 / +0.8 / +0.9 | 23.7/95 |
| bands to k84 = 0.18 + S21m | yes | 60 | 0.0023 | 0.042 | 0.0039 | +0.4 / +0.9 / +0.9 | 23.5/95 |

Reach ladder (no AP, V = 60; the same vector with `--s1-kmax` 0.15 / 0.18 / 0.2, against P_l to the same kmax):

| reach | bands + S21m: sigma (shift) omega_cdm / h | chi^2/dof | P_l: sigma (shift) omega_cdm / h | chi^2/dof |
|---|---|---|---|---|
| 0.15 (sigma = 12.5 at l = 0, 17.7 at l <= 3) | 0.0026 (-0.3) / 0.0047 (-0.4) | 12.5/89 | 0.0028 (-0.4) / 0.0046 (-0.4) | 4.7/21 |
| 0.18 (+ 12.5 at l = 1, 17.7 at l = 4, 5) | 0.0023 (+0.6) / 0.0038 (+0.9) | 23.7/95 | 0.0027 (-0.1) / 0.0043 (+0.1) | 9.9/27 |
| 0.2 (+ 12.5 at l = 2, 17.7 at l = 6) | 0.0020 (+1.4) / 0.0033 (+1.5) | 59.6/98 | 0.0024 (-0.9) / 0.0040 (-0.7) | 18.8/33 |

**Answer: not to k84 = 0.2; yes to k84 = 0.18.** The full vector (sigma = 12.5 at l <= 2 plus sigma >= 17.7 at l <= 6)
is 10-18% tighter than P_l to 0.2 in all three parameters, but it is biased: the minima sit at +1.4 sigma in omega_cdm
and +1.5 in h without AP (+0.8 and +1.5 with AP; ln A_s is fine), at V = 60 and 64 alike, and the last two blocks
(sigma = 12.5, l = 2, k84 = 0.184, and sigma = 17.7, l = 6, k84 = 0.180) raise chi^2 by 36 for 3 more principal
components. The best-fit residuals locate it: the m = 0 entries of the sigma = 12.5-25 bands at l >= 2 sit 0.3-0.6%
above the model (+2 to +3.8 sigma at V = 60; sigma = 12.5, l = 2, m = 0 is the worst), the large-scale blocks
(sigma >= 35) 0.4-0.6% below, so the fit raises omega_cdm and h to push power to small scales -- the one-loop + c_0
model of P_s(k, mu) running out at k ~ 0.2 along the line of sight, where the bands weight the spectrum differently
from the Legendre multipoles (P_l to 0.2 drifts the other way, -0.9 / +1.4 / -0.7). The emulator is not the cause
(S1m validation 1.8e-4, at most 0.15 of the V = 60 error of the smallest-scale entries). Cut at k84 = 0.18 the bands
stay within 1 sigma (+0.6 / +0.8 / +0.9, chi^2 23.7/95) and are as tight as P_l to 0.2: 0.0023 / 0.042 / 0.0038
against 0.0024 / 0.039 / 0.0040 (omega_cdm and h equal or 5% tighter, ln A_s 7% wider); with AP 0.0023 / 0.042 /
0.0039 and shifts +0.4 / +0.9 / +0.9 (V = 64 not run for this cut: from 60 to 64 the shifts only grow by 3%). At
k84 = 0.15 bands and P_l agree to 5% and are both unbiased.

Consequence: the halo WST, through its q = 2 first layer (m-resolved band powers) plus S21m, matches P_l to k = 0.2 when
its own reach stops at k84 = 0.18; pushing the bands to 0.2 buys 10-18% at the price of a 1.5 sigma bias in h, a model
limit, not an emulator one. With AP the picture is the same (omega_cdm shift +1.4 -> +0.8, widths unchanged). The
baseline choice (k84 = 0.18 vs P_l to 0.2, with or without the q = 0.8 layer) is the user's.

### Does modelling the halo second layer (instead of marginalising it) beat P_l to 0.2? (2026-10-19, evening)

Question: the halo WST (bands to k84 = 0.18 + S21m, `q2_12k18_s21_V60`) only ties P_l to 0.2; is the bispectrum-like
information of S21m lost to the per-l marginalisation of the layer-2 template amplitudes, and does the rest sit in
n_s and the biases, which the scans never vary? Fisher at V = 60 from the fit's own likelihood (`build_fit`: sample
covariance with Hartlap and Percival, template Monte Carlo error, marginalised amplitudes), central differences of
the bands12 emulator at the Quijote cosmology and the `pk02_V60` / `q2_12k18_s21_V60` best-fit nuisances, nuisances
marginalised with their Gaussian priors: `scripts/fisher_rsd.py --layer2-info` (new mode, `LAYER2_CASES`; output
`outputs/fisher/layer2_info_V60.json`). Template treatments: **marg** = amplitudes per l with width 1 (as fitted),
**w0.1** = width 0.1, **model** = amplitudes fixed at the template (only its Monte Carlo error in the covariance),
**exact** = also without that error (a perfectly known template). The Fisher errors reproduce the scans (row a:
0.0022 / 0.042 / 0.0038 against 0.0023 / 0.042 / 0.0038; P_l: 0.0023 / 0.040 / 0.0038 against 0.0024 / 0.039 / 0.0040).

| vector | n | omega_cdm | ln A_s | h | b1 | b2 | bG2 | n_s free: omega_cdm / ln A_s / n_s / h | b2 / bG2 |
|---|---|---|---|---|---|---|---|---|---|
| (a) bands + S21m, marg | 126 | 0.0022 | 0.0421 | 0.0038 | 0.047 | 0.25 | 0.121 | 0.0047 / 0.0687 / 0.0286 / 0.0058 | 0.25 / 0.126 |
| (b) bands + S21m, w0.1 | 126 | 0.0022 | 0.0420 | 0.0038 | 0.047 | 0.25 | 0.120 | 0.0047 / 0.0686 / 0.0285 / 0.0058 | 0.25 / 0.124 |
| (b) bands + S21m, model | 126 | 0.0022 | 0.0420 | 0.0038 | 0.047 | 0.25 | 0.120 | 0.0047 / 0.0686 / 0.0285 / 0.0058 | 0.25 / 0.124 |
| (b) bands + S21m, exact | 126 | 0.0022 | 0.0420 | 0.0038 | 0.047 | 0.24 | 0.119 | 0.0047 / 0.0686 / 0.0285 / 0.0058 | 0.25 / 0.123 |
| (c) bands alone | 72 | 0.0023 | 0.0420 | 0.0041 | 0.048 | 0.62 | 0.321 | 0.0051 / 0.0774 / 0.0360 / 0.0059 | 0.71 / 0.324 |
| (c) bands alone, b2 and bG2 known | 72 | 0.0021 | 0.0412 | 0.0037 | 0.045 | -- | -- | 0.0047 / 0.0702 / 0.0287 / 0.0058 | -- |
| (d) S21m alone, marg | 54 | 0.0352 | 0.322 | 0.149 | 0.48 | 0.63 | 0.264 | 0.079 / 0.85 / 0.54 / 0.158 | 0.63 / 0.279 |
| (d) S21m alone, w0.1 | 54 | 0.0319 | 0.289 | 0.130 | 0.48 | 0.60 | 0.252 | 0.072 / 0.75 / 0.49 / 0.138 | 0.61 / 0.274 |
| (d) S21m alone, model | 54 | 0.0319 | 0.288 | 0.130 | 0.48 | 0.60 | 0.252 | 0.071 / 0.74 / 0.49 / 0.138 | 0.61 / 0.274 |
| (d) S21m alone, exact | 54 | 0.0288 | 0.265 | 0.116 | 0.46 | 0.57 | 0.240 | 0.061 / 0.64 / 0.42 / 0.123 | 0.59 / 0.269 |
| (e) P_l to 0.2 | 48 | 0.0023 | 0.0397 | 0.0038 | 0.046 | 0.57 | 0.321 | 0.0046 / 0.0670 / 0.0321 / 0.0056 | 0.59 / 0.322 |
| (e) P_l to 0.2, b2 and bG2 known | 48 | 0.0020 | 0.0362 | 0.0033 | 0.037 | -- | -- | 0.0046 / 0.0668 / 0.0298 / 0.0056 | -- |
| P_l to 0.2 + S21m, marg | 102 | 0.0021 | 0.0384 | 0.0035 | 0.041 | 0.25 | 0.121 | 0.0046 / 0.0672 / 0.0298 / 0.0057 | 0.26 / 0.125 |
| P_l to 0.2 + S21m, model | 102 | 0.0021 | 0.0383 | 0.0035 | 0.041 | 0.25 | 0.118 | 0.0046 / 0.0671 / 0.0298 / 0.0057 | 0.25 / 0.122 |

Where S21m's information sits: the vector is a single scale pair (sigma_1 = 25, sigma_2 = 70.7) with l = 1-4. No
single l constrains anything (l = 1, 2 are fully degenerate with the noise and white amplitudes; l = 4, the best,
gives 0.39 / 0.76 / 0.42 on omega_cdm / ln A_s / h, l = 3 0.20 / 4.0 / 2.0); the information of S21m alone is in the
combination of the multipoles, ten times better than any of them.

Reading:

- **The template marginalisation is not where the information goes.** Fixing the template amplitudes (or even
  removing the template's Monte Carlo error) changes the cosmology errors of bands + S21m by less than 0.5% and those
  of b2 / bG2 by 1-3%, with or without n_s. Only for S21m alone does it matter (10% from marg to model, another 10%
  from the template error), and S21m alone is 14 / 7 / 32 times weaker than the bands on omega_cdm / ln A_s / h.
- **S21m's information is bias information.** On top of the bands it shrinks b2 from 0.62 to 0.25 and bG2 from 0.32
  to 0.12 (factor 2.5), and leaves b1 and the three cosmological parameters where they are (n_s fixed: omega_cdm
  0.0023 -> 0.0022, h 0.0041 -> 0.0038). That is the ceiling: knowing b2 and bG2 exactly would give the bands only
  0.0021 / 0.041 / 0.0037. With n_s free the bias information becomes n_s information: 0.036 -> 0.0286, exactly the
  0.0287 of the bands with b2 and bG2 known, so S21m saturates what bias-breaking can give the bands.
- **Against P_l to 0.2.** n_s fixed: no, for any treatment (0.0022 / 0.042 / 0.0038 against 0.0023 / 0.040 / 0.0038;
  ln A_s stays 6% behind). n_s free: the WST beats P_l alone on n_s by 11% (0.0286 against 0.0321) and ties
  omega_cdm, ln A_s, h (within 2-4%, in P_l's favour), but this already holds with the template marginalised, and it
  is S21m's bias information, which P_l takes just as well: P_l + S21m reaches n_s 0.0298 (= P_l with b2, bG2 known).
  The WST-only vector is then 4% ahead of P_l + S21m on n_s and 1-2% behind on the rest.

Answer: modelling the second layer instead of marginalising it would not make the WST beat P_l to 0.2 for
omega_cdm, ln A_s or h, and changes nothing for n_s either; the 11% n_s gain over P_l alone exists already and comes
from S21m pinning b2 and bG2. Caveat: "modelled" here means the calibrated template with its amplitudes fixed; its
cosmology dependence enters only through R_l(s_lin^2), so a layer-2 model whose non-Gaussian part itself scales with
the cosmology and the biases (a bispectrum prediction rather than a fitted correction) is not tested. Since what S21m
carries on top of either two-point vector is bias information and no single multipole constrains cosmology, there is
no sign that it would change the conclusion; a broader S21m vector (sigma_1 < 25, more sigma_2) is the place to look.

### Can the halo q != 2 first layer (the Edgeworth factor E(q)) be kept next to the bands? (2026-10-19, night)

Question: a Fisher follow-up found that S1m at q = 0.8 on the same blocks as the q = 2 bands (`--q 2.0 0.8`, plus
S21m) beats P_l to 0.2 by tens of percent for halos, through nuisance-breaking in the Edgeworth term (in real-space
matter the same q = 0.8 rows add nothing). Earlier notes say tree-level K4 fails at sigma = 17.7. Is the prize robust
to freeing the E nuisances (step 0), and can E itself be modelled to V = 60 accuracy (step 1)? No chains.

**Step 0, Fisher** (`scripts/fisher_rsd.py --edgeworth-info`, new mode; `outputs/fisher/edgeworth_info{,_real}_V60.json`).
The setup is that of the layer-2 Fisher above (`build_fit` likelihood, bands12 emulator, V = 60, Quijote cosmology,
`pk02_V60` / `q2_12k18_s21_V60` nuisances). The vector is the bands to k84 = 0.18 + S21m (white, layer-2 template
marginalised per l, cubic biases `relation`), plus q = 0.8 on the same 23 coefficients (136 rows). The E nuisances that
the refactor removed from the package enter as extra Jacobian columns, through a monkeypatch of `tracer_cumulants` /
`s1m_from_variances` inside the script:
- A_TN, A_TN2 (priors N(1, 1)) scale the N and N^2 Poisson pieces of the tree kappa, in S1m and in S21m's Edgeworth
  factors;
- Edgeworth amplitudes a_ng multiply E of the q = 0.8 blocks: one global, one per sigma, or one per (sigma, l), each
  with a Gaussian prior of width 0.29 (the old flat [0.5, 1.5]). S1m is linear in a_ng, so each column is S1m(a = 1) -
  S1m(a = 0).
The cubic-bias offsets and bGamma3 get priors widened x3 (6 / 3 / 3 / 3) or made flat. The q = 0.8 rows are cut at
sigma >= 12.5 (all 23 coefficients; the k84 cut leaves l <= 1 at 12.5 and l <= 5 at 17.7), 17.7 or 25.

The reference vectors reproduce the layer-2 table: P_l to 0.2 0.0023 / 0.0397 / 0.0038, n_s free 0.0046 / 0.0670 /
0.0321 / 0.0056; bands + S21m 0.0022 / 0.0420 / 0.0038, n_s free 0.0047 / 0.0686 / 0.0285 / 0.0058. Gain = 1 -
sigma / sigma(P_l to 0.2), in percent:

| q = 0.8 at sigma >= | E treatment | n_s fixed: omega_cdm / ln A_s / h | n_s free: omega_cdm / ln A_s / n_s / h |
|---|---|---|---|
| 12.5 | tree K4, priors as fitted | +40 / +24 / +15 | +39 / +40 / +48 / +30 |
| 12.5 | + A_TN, A_TN2 | +38 / +18 / +14 | +36 / +32 / +46 / +27 |
| 12.5 | + a_ng per sigma | +26 / +17 / +12 | +37 / +32 / +37 / +28 |
| 12.5 | + a_ng per (sigma, l) | +21 / +8 / +7 | +20 / +15 / +25 / +14 |
| 12.5 | cubic-bias priors flat | +40 / +18 / +15 | +38 / +39 / +47 / +29 |
| 12.5 | all: A_TN, a_ng per sigma, cubic x3 | +25 / +14 / +11 | +34 / +30 / +35 / +26 |
| 12.5 | all: A_TN, a_ng per (sigma, l), cubic x3 | +18 / +6 / +6 | +15 / +11 / +21 / +9 |
| 17.7 | tree K4, priors as fitted | +28 / +18 / +12 | +36 / +35 / +34 / +27 |
| 17.7 | + A_TN, A_TN2 | +24 / +12 / +11 | +31 / +29 / +32 / +24 |
| 17.7 | + a_ng per sigma | +25 / +12 / +12 | +35 / +30 / +32 / +27 |
| 17.7 | + a_ng per (sigma, l) | +18 / +5 / +6 | +18 / +14 / +20 / +13 |
| 17.7 | cubic-bias priors flat | +28 / +16 / +12 | +36 / +34 / +34 / +27 |
| 17.7 | all: A_TN, a_ng per sigma, cubic x3 | +21 / +10 / +10 | +29 / +26 / +29 / +23 |
| 17.7 | all: A_TN, a_ng per (sigma, l), cubic x3 | +14 / +2 / +4 | +14 / +11 / +18 / +9 |
| 25 | tree K4, priors as fitted | +15 / +0 / +5 | +23 / +15 / +24 / +18 |
| 25 | + A_TN, A_TN2 | +10 / -1 / +0 | +11 / +8 / +17 / +7 |
| 25 | + a_ng per sigma | +13 / -0 / +4 | +21 / +14 / +23 / +16 |
| 25 | + a_ng per (sigma, l) | +8 / -4 / -1 | +10 / +6 / +15 / +6 |
| 25 | cubic-bias priors flat | +14 / +0 / +4 | +20 / +14 / +21 / +15 |
| 25 | all: A_TN, a_ng per sigma, cubic x3 | +9 / -1 / -1 | +9 / +6 / +15 / +5 |
| 25 | all: A_TN, a_ng per (sigma, l), cubic x3 | +6 / -5 / -2 | +6 / +4 / +13 / +3 |

The background note's 0.0013 / 0.024 / 0.0031 (43 / 40 / 18%) came from a different nuisance treatment; recomputed with
the fit's own likelihood, the baseline is 40 / 24 / 15%.

Real-space matter control (`--real-space`): bands12k + S21m (compress-m, matter layer-2 template) + q = 0.8 at the
emulated l <= 4. On top of the bands, q = 0.8 changes the errors by +4 / +1 / 0% (sigma >= 12.5, n_s fixed) and +5%
(n_s free) at most, and by -2 to -5% once a_ng is free or the cut is 17.7 / 25. The negative values are the Hartlap /
Percival price of the extra rows. Against P_0 to 0.2 the WST stays where it was: +4 / -11 / -21% n_s fixed, about
+2% n_s free. As expected, the halo gain is nuisance-breaking (b2 / bG2 / bGamma3 / c_v through E), not new linear
information.

Step 0 reading: freeing A_TN and per-sigma Edgeworth amplitudes, with the cubic priors widened, keeps the gain above
the 15% kill line at sigma >= 12.5 and 17.7: n_s fixed 25 / 14 / 11% and 21 / 10 / 10%, so only omega_cdm stays above
15%; n_s free 34 / 30 / 35 / 26% and 29 / 26 / 29 / 23%. The cubic-bias priors barely matter (flat: -2 to -6 points
on ln A_s). What erodes the gain is block freedom in E. With one amplitude per (sigma, l) at a 29% prior, the gain
falls to 18 / 6 / 6% (n_s fixed) and 15 / 11 / 21 / 9% (n_s free). With the rows cut at sigma >= 25, it is at most 15%
(n_s on its own) with the E nuisances free. **The kill criterion is not met** with per-sigma freedom, and is met
almost everywhere with per-(sigma, l) freedom: the prize exists only if E is known to better than a per-(sigma, l)
amplitude of +-29%.

**Step 1, model-free E per block** (`outputs/drafts/nnlo_edgeworth/edgeworth_direct.py`, draft next to the NNLO code,
not in the package; output `edgeworth_direct.{json,log}`). The data E(q) = S1m(q) / Gauss(q; lambda = S1m(q = 2) / n)
- 1 is measured per box and block (j, l, m) at q = 0.5, 0.8, 1 on the 1188 halo boxes, with covariance = box scatter / 60
(Hartlap-corrected). The models are evaluated at the Quijote cosmology and the `q2_12k18_s21_V60` nuisances (b1 1.91,
b2 -0.38, bG2 -0.34, bGamma3 0.67, c_v 3.07, cubic offsets ~ 0). They are read from the emulated S1m with the Edgeworth
amplitude on and off, as alpha = K4 / (8 n (n + 2)) and beta = s32 / 72 with alpha linear in A_TN, A_TN2:
- **tree**: no cubic operators, Poisson;
- **cubic**: the relation cubic biases;
- **+ A_TN**: A_TN, A_TN2 fitted;
- **NNLO**: the draft NNLO Edgeworth, with c6 = 3.6, c35 = 1.9 fixed, or with c6, c35, A_TN, A_TN2 all fitted;
- **a_ng(sigma, l)**: cubic + A_TN with one free amplitude per (sigma, l), the calibrated-correction form.
The free parameters are fitted to the q = 0.8 entries of the blocks in the step-0 vector (52 blocks: k84 <= 0.18),
which is the exponent that vector uses. The table gives the rms over m of (data - model) / sigma_V60 per (sigma, l),
with the rms fractional residual of E in brackets:

| sigma, l | k84 | tree | cubic | cubic + A_TN | NNLO (c6 3.6) + A_TN | NNLO, c6 c35 A_TN fitted | cubic + A_TN + a_ng(sigma, l) |
|---|---|---|---|---|---|---|---|
| 12.5, 0 | 0.129 | 51.2 [58%] | 3.1 [4%] | 40.6 [46%] | 42.4 [48%] | 11.4 [13%] | 6.0 [7%] |
| 12.5, 1 | 0.159 | 24.2 [8%] | 28.6 [9%] | 33.7 [10%] | 35.5 [11%] | 16.5 [5%] | 9.6 [3%] |
| 17.7, 0 | 0.091 | 17.9 [60%] | 9.6 [32%] | 26.4 [88%] | 19.5 [65%] | 2.5 [8%] | 4.8 [16%] |
| 17.7, 1 | 0.113 | 11.2 [10%] | 26.0 [21%] | 41.4 [33%] | 38.6 [31%] | 20.5 [17%] | 10.5 [9%] |
| 17.7, 2 | 0.130 | 9.6 [6%] | 20.0 [14%] | 33.5 [23%] | 30.5 [21%] | 12.2 [8%] | 12.2 [10%] |
| 17.7, 3 | 0.145 | 7.7 [5%] | 13.7 [9%] | 23.6 [15%] | 20.3 [13%] | 7.8 [5%] | 11.5 [9%] |
| 17.7, 4 | 0.158 | 22.2 [17%] | 19.0 [15%] | 17.0 [12%] | 16.3 [12%] | 25.3 [19%] | 8.4 [6%] |
| 17.7, 5 | 0.169 | 38.3 [29%] | 33.6 [26%] | 30.3 [23%] | 31.8 [24%] | 44.4 [33%] | 8.1 [7%] |
| 25.0, 0 | 0.064 | 6.8 [83%] | 4.8 [59%] | 10.3 [126%] | 8.1 [99%] | 2.2 [27%] | 3.0 [37%] |
| 25.0, 1 | 0.080 | 3.4 [7%] | 9.3 [21%] | 15.6 [34%] | 14.4 [32%] | 7.2 [17%] | 5.7 [14%] |
| 25.0, 2 | 0.092 | 4.1 [8%] | 9.4 [19%] | 15.9 [31%] | 14.5 [28%] | 6.9 [14%] | 6.6 [14%] |
| 25.0, 3 | 0.102 | 5.4 [10%] | 9.9 [19%] | 16.2 [30%] | 14.8 [27%] | 6.9 [13%] | 7.0 [14%] |
| 25.0, 4 | 0.111 | 5.5 [10%] | 9.4 [16%] | 15.7 [27%] | 14.0 [24%] | 5.9 [10%] | 6.7 [13%] |
| 25.0, 5 | 0.120 | 5.5 [10%] | 8.9 [16%] | 14.6 [26%] | 12.9 [23%] | 5.1 [9%] | 6.3 [12%] |
| 25.0, 6 | 0.127 | 3.8 [7%] | 6.5 [11%] | 11.6 [20%] | 9.9 [17%] | 3.6 [6%] | 5.4 [11%] |
| chi2 / n | | 53435 / 52 | 37190 / 52 | 29313 / 52 | 28846 / 52 | 26439 / 52 | 1069 / 52 |

The data E is precise: sigma_V60 = 0.7-2.5% of E for l >= 1 at sigma = 17.7-25. That precision is why the q = 0.8 rows
next to the bands carry information; S1m(q) / S1m(2) cancels the cosmic variance. The chi^2 is much larger than the sum
of the z^2 because the m entries of a block are strongly correlated.

- **No model reaches V = 60 accuracy at any sigma <= 25.** Tree, cubic, or NNLO, fixed or with up to four fitted E
  parameters, they miss E by 5-60% at sigma = 12.5-17.7 (typically 8-45 sigma per (sigma, l); a single model gets an
  occasional block to 2-3 sigma, never all of them), and by 5-30% (3-16 sigma) at sigma = 25.
- **The free E nuisances do not fit consistently.** Fitting A_TN and A_TN2 to all blocks makes things worse at sigma =
  25 (A_TN2 ~ 7, an unphysical value), because the misfit is scale- and l-dependent. A sub-Poisson trispectrum does
  not explain it.
- **NNLO with fitted closures** (c6 = 4.0, c35 = 2.1, consistent with the 2026-10-12 values) brings sigma = 17.7, l <=
  3 to 5-17% but leaves l = 4-5 at 19-33%.
- **Even per-(sigma, l) amplitudes leave chi^2 = 1069 / 52** and 5-12 sigma per block. Within a block the m (line of
  sight) structure of E is wrong at the 5-15% level, and the fitted amplitudes are degenerate with A_TN (values
  unphysical).
- **At sigma = 25 alone** (fit on its 28 blocks): cubic + A_TN gives 179 / 28, NNLO with fitted closures 161 / 28,
  per-l amplitudes 136 / 28, with residuals of 2-3 sigma per l and 4-7% of E. A_TN2 is again at -8 to -18 in all of
  these fits.
- **Using all three q** (as the multi-q extension would) is far worse: the q shape of E is measured even better, and
  NLO is off in it (vector blocks: chi^2 4e4-3e5 / 156; sigma = 25: NNLO 208 / 84, NLO ~ 1000 / 84).
- **Caveat:** the biases are held at the q = 2 fit's values. Free biases could absorb part of the misfit, but that is
  the b2-c0-omega_cdm slide seen on 2026-10-11, the very degeneracy the q = 0.8 rows are supposed to break.

**Go / no-go for step 2.**
- **No-go for putting an E model into the package.** None of tree K4, cubic + A_TN or NNLO is within the V = 60 errors
  on the blocks that carry the gain. They miss by 10-40 sigma at sigma = 17.7, where step 0 puts most of the prize, and
  by 3-10 sigma per l even at sigma = 25, where the prize is at most 15% (n_s only) once A_TN is free.
- **No-go, as specified, for a fiducial-calibrated E template checked by V = 60/64 profile scans.** Such a template
  (E_data(fid) x E_model(theta) / E_model(fid), with Gaussian-prior amplitudes) is exact at the fiducial by
  construction, so profile scans on the fiducial boxes cannot fail: they test nothing. The cosmological information it
  adds is dE / d theta from a model that already misses E by 10-60% at fixed theta. Step 0 shows how much the prize
  depends on that model: it halves once each (sigma, l) gets a 29% amplitude of its own.
- **Gate before any step 2:** test the response of E, not its value. The Quijote halo derivative boxes are on disk
  (`data/quijote/{Om_m,Om_p,s8_m,s8_p}/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los`, 200 each, paired
  seeds). Measure dE / dOmega_m and dE / dsigma_8 per block and compare them with the model's derivatives at the
  biases refitted on each set (the q = 2 bands + S21m fix the biases). If the model response agrees to ~10-15% per
  (sigma, l), a calibrated template with a 10-15% prior per (sigma, l) is worth building. Its Fisher gain lies between
  the per-(sigma, l) row at a 29% prior (15-21% n_s free, 6-18% n_s fixed) and the per-sigma row; it has not been
  computed at the tighter width. If the response is off
  by as much as the level, stop: the halo WST stays at bands + S21m = P_l to 0.2, and the q != 2 first layer is not
  usable for halos at V = 60.

**Gate: the cosmology response of E** (`edgeworth_direct.py --response` / `--decompose`, the profile fits in
`outputs/drafts/nnlo_edgeworth/deriv_fits/run.sh`; output `edgeworth_response{.json,.log,_decomp.json}`).
- **Data.** E(q = 0.8) is measured per block on the 200 paired seeds of each Quijote halo derivative set, using central
  differences per seed. The steps are Delta Omega_m = +-0.01 (at fixed sigma_8 = 0.834, Omega_b and h) and Delta
  sigma_8 = +-0.015. The error of the mean is the scatter over the 200 seeds.
- **Model.** The bands12 emulator with the relation cubic biases, Poisson trispectrum and NLO E, i.e. the "cubic" model
  above. Each set gets its own nuisances: a `fit.py --method profile` fit of the `q2_12k18_s21_V60` vector (bands to
  k84 = 0.18 + S21m, V = 60 fiducial covariance), with the cosmology fixed at the set's values. For Om_pm, logA comes
  from CLASS sigma_8 ratios.
- **Package fix.** The band-power PCA compression now comes from the covariance set when `--covariance-data-dir` is
  given. Before, it came from each data set's own boxes: 69 instead of 72 components on 200 boxes, a different basis on
  each derivative set.
- **Columns.** The model response is shown in three forms:
  - *refit*: the nuisances refitted on each set;
  - *fixed*: the nuisances held at the fiducial fit, so that only the cosmology moves;
  - *template*: E_data(fid) x d ln E_model / d theta (refit), i.e. the response of the fiducial-calibrated template.

Each entry is r = <model, data> / <data, data> over m (weights 1 / err^2), with the rms over m of |model - data| / |data|
in brackets. The data column gives d ln E / d theta = (dE / d theta) / E_fid, averaged over m. The step-0 vector holds
sigma 12.5 l <= 1, 17.7 l <= 5 and all of sigma 25.

| sigma, l | k84 | dlnE/dOm data | refit | fixed | template | dlnE/ds8 data | refit | fixed | template |
|---|---|---|---|---|---|---|---|---|---|
| 12.5, 0 | 0.129 | -3.98 +- 0.08 | 1.08 [8%] | 0.66 [34%] | 1.12 [12%] | -2.00 +- 0.06 | 3.57 [257%] | -0.55 [155%] | 3.70 [270%] |
| 12.5, 1 | 0.159 | -3.45 +- 0.02 | 0.97 [11%] | 0.77 [23%] | 1.04 [18%] | -1.09 +- 0.02 | 2.74 [180%] | -1.27 [243%] | 2.94 [193%] |
| 12.5, 2 | 0.184 | -3.62 +- 0.02 | 1.13 [17%] | 0.99 [21%] | 0.93 [23%] | -0.86 +- 0.02 | 3.43 [282%] | -2.75 [478%] | 2.93 [193%] |
| 12.5, 3 | 0.204 | -3.76 +- 0.03 | 1.35 [42%] | 1.23 [43%] | 0.91 [24%] | -0.82 +- 0.02 | 4.18 [369%] | -4.50 [712%] | 2.89 [187%] |
| 12.5, 4 | 0.223 | -3.85 +- 0.03 | 1.69 [88%] | 1.59 [91%] | 0.86 [26%] | -0.80 +- 0.02 | 5.22 [501%] | -7.38 [1137%] | 2.80 [179%] |
| 12.5, 5 | 0.240 | -3.93 +- 0.03 | 2.08 [134%] | 1.95 [132%] | 0.86 [24%] | -0.81 +- 0.02 | 6.38 [632%] | -10.56 [1506%] | 2.74 [174%] |
| 12.5, 6 | 0.255 | -3.99 +- 0.03 | 2.63 [199%] | 2.47 [195%] | 0.81 [31%] | -0.81 +- 0.02 | 7.72 [804%] | -16.10 [2345%] | 2.58 [160%] |
| 17.7, 0 | 0.091 | -4.83 +- 0.22 | 1.16 [16%] | 0.64 [36%] | 1.70 [70%] | -2.38 +- 0.16 | 4.02 [302%] | -0.19 [119%] | 5.89 [489%] |
| 17.7, 1 | 0.113 | -4.42 +- 0.06 | 0.92 [14%] | 0.70 [31%] | 1.15 [16%] | -1.26 +- 0.04 | 2.46 [143%] | -0.91 [194%] | 3.03 [200%] |
| 17.7, 2 | 0.130 | -4.59 +- 0.05 | 0.96 [9%] | 0.78 [22%] | 1.12 [12%] | -0.96 +- 0.04 | 2.59 [163%] | -1.42 [253%] | 2.98 [200%] |
| 17.7, 3 | 0.145 | -4.71 +- 0.06 | 0.97 [9%] | 0.81 [19%] | 1.06 [11%] | -0.88 +- 0.04 | 2.71 [176%] | -1.72 [285%] | 2.94 [196%] |
| 17.7, 4 | 0.158 | -4.76 +- 0.06 | 1.05 [10%] | 0.91 [12%] | 1.03 [11%] | -0.84 +- 0.04 | 2.95 [205%] | -2.16 [347%] | 2.91 [193%] |
| 17.7, 5 | 0.169 | -4.84 +- 0.06 | 1.14 [19%] | 1.01 [16%] | 1.01 [11%] | -0.82 +- 0.04 | 3.26 [238%] | -2.76 [408%] | 2.92 [191%] |
| 17.7, 6 | 0.180 | -4.91 +- 0.06 | 1.29 [38%] | 1.18 [34%] | 0.97 [19%] | -0.83 +- 0.04 | 3.49 [274%] | -3.58 [539%] | 2.72 [171%] |
| 25.0, 0 | 0.064 | -5.20 +- 0.75 | 1.41 [41%] | 0.67 [33%] | 3.45 [245%] | -2.95 +- 0.55 | 4.75 [375%] | 0.09 [91%] | 11.63 [1063%] |
| 25.0, 1 | 0.080 | -5.31 +- 0.15 | 0.98 [11%] | 0.74 [28%] | 1.23 [22%] | -1.40 +- 0.10 | 2.86 [181%] | -0.87 [191%] | 3.54 [250%] |
| 25.0, 2 | 0.092 | -5.62 +- 0.14 | 0.97 [10%] | 0.77 [24%] | 1.18 [19%] | -1.14 +- 0.10 | 2.61 [172%] | -1.17 [222%] | 3.15 [226%] |
| 25.0, 3 | 0.102 | -5.99 +- 0.15 | 0.92 [13%] | 0.76 [26%] | 1.13 [13%] | -1.07 +- 0.10 | 2.55 [158%] | -1.30 [232%] | 3.03 [205%] |
| 25.0, 4 | 0.111 | -6.04 +- 0.15 | 0.95 [10%] | 0.80 [21%] | 1.13 [13%] | -0.99 +- 0.10 | 2.54 [159%] | -1.45 [249%] | 2.95 [197%] |
| 25.0, 5 | 0.120 | -6.00 +- 0.15 | 0.95 [11%] | 0.81 [20%] | 1.12 [12%] | -0.91 +- 0.11 | 2.63 [169%] | -1.57 [260%] | 3.02 [207%] |
| 25.0, 6 | 0.127 | -6.05 +- 0.16 | 0.95 [12%] | 0.82 [19%] | 1.06 [10%] | -0.90 +- 0.11 | 2.63 [167%] | -1.69 [272%] | 2.91 [191%] |

- **The data response is large and precise.** E shrinks with Omega_m (d ln E / d Omega_m = -3.5 to -6) and with sigma_8
  (-0.8 to -2). It is measured to 1-3% at sigma = 12.5, 1-5% at 17.7 (l >= 1) and 3-12% at 25 (l >= 1), so a 10-15%
  test can be resolved.
- **Omega_m passes, at about 10%, where the vector has its blocks.**
  - With the nuisances refitted, r = 0.92-1.05 at sigma = 17.7 l = 1-4, and 0.92-0.98 at sigma = 25 l >= 1 (rms 9-14%).
    l = 5 is at 1.14 [19%], and sigma 12.5 l <= 1 at 0.97-1.08. In units of the measurement error this is still
    chi^2 / n = 13-230 per block.
  - The refit matters: with fixed nuisances the model is 20-30% low.
  - The model drifts upward with l at sigma = 12.5 (r = 1.35-2.6 for l >= 3, beyond the vector) and at 17.7 l = 6.
  - The calibrated template is within 1-23% for l >= 1, at r = 1.01-1.23.
- **sigma_8 fails by a factor of 2.5-4 in every block.**
  - With the nuisances refitted, r = 2.5-3.5 at sigma = 17.7-25 (l >= 1) and 2.7-7.7 at 12.5. The template is the same
    (r = 2.6-3.5), so the calibration does not help: the fiducial E rescales the level, not the response.
  - With the nuisances held fixed, the model response has the wrong sign. The cosmology-only part is
    d ln E / d sigma_8 > 0 (r = -0.9 to -1.7 at sigma 17.7-25).
  - The whole model response therefore comes from the nuisance shifts. The linear split (`--decompose`) at sigma =
    17.7-25, as r units:

    | nuisance group | contribution to r |
    |---|---|
    | b2 | +4.1 to +4.9 |
    | b1 | -0.6 to +0.3 |
    | bG2 | -0.3 |
    | c_v, the counterterms and the other nuisances | +0.1 to +0.8 |
    | the sum, with the cosmology-only part | reproduces refit |

  - b2 is the problem. The q = 2 vector sets d b2 / d sigma_8 = -7.5, against d b1 / d sigma_8 = -3.27. The b1 shift is
    the peak-background-split value (-3.6 for b1 = 1.96), but the model's d E / d b2 turns that b2 shift into 4-5 times
    the data response. Matching the data would need d b2 / d sigma_8 of about -3.6 at this d E / d b2. The fitted shift
    rides along the b2-c0-alpha2 direction (d c0 / d sigma_8 = -44, d alpha2 / d sigma_8 = -62).
  - The shift is not fit noise. Refitting s8_pm on seeds 0-99 and 100-199 separately gives d b2 / d sigma_8 = -8.0 /
    -6.9 and d b1 / d sigma_8 = -3.28 / -3.26.
  - Either d E / d b2 of the model is about twice too large, or the bands + S21m vector maps sigma_8 into a b2 shift that
    E does not see. Both break the template the same way, because a real analysis fits the nuisances the same way.

**Go / no-go: no-go for a calibrated E template, and the q != 2 halo first layer stops here.** The response test is
the one a fiducial-calibrated template could still have passed, and it fails.
- In sigma_8 (the amplitude direction, ln A_s at fixed shape), the model and template response is 2.5-3.5
  times the data in every block of the vector. That is a 150-250% error, an order of magnitude beyond the 10-15% a
  template prior could absorb, and beyond the 29% per-(sigma, l) amplitudes that already halve the prize.
- Omega_m alone passes at about 10% for sigma = 17.7 l <= 4 and sigma 25. That is not enough: the gain needs both
  parameters, and the sigma_8 failure enters through b2, which the q = 0.8 rows were meant to constrain.
- Per the gate, the halo WST stays at bands + S21m = P_l to 0.2 at V = 60. A q != 2 halo first layer would first need
  a model of E whose bias dependence (d E / d b2 at fixed P_l) is right. That is a new perturbative ingredient (the
  b2-sourced trispectrum at the smoothing scales) and is not attempted.

### S21m at q = 2 needs no layer-2 template (2026-10-19, night)

In the halo q = 2 vector (`fit.py --q 2.0`) S21m is loaded and modelled at the first q (`build_problem`: `q0 =
options.q[0]`), i.e. at q = 2, not at the q = 0.8 of the paper's baseline; the real-space band runs (`--stats band`)
kept S21m at q = 0.8. The layer-2 template was measured at q = 0.8 (`scripts/layer2/template_rsd.py`, `Q = 0.8`) as
the deficit S2m(q) / [Gaussian prediction from S2m(q = 2)] - 1, which vanishes identically at q = 2: the q = 2
coefficient is the second-layer power itself, with no one-point (chi-distribution) assumption to correct. The halo
q = 2 runs so far therefore applied a q = 0.8 correction that does not exist at q = 2; the per-l marginalisation
(width 1) absorbed it.

Test (V = 60, no AP, n_s fixed): `q2_12k18_s21nt_*` = `q2_12k18_s21_*` without `--s21-layer2` (profile and the
16-point chi^2 scans; parabola widths, shifts of the minima from the truth):

| vector | sigma(omega_cdm) | sigma(ln A_s) | sigma(h) | shifts omega_cdm / ln A_s / h | chi^2 at the best fit |
|---|---|---|---|---|---|
| bands to k84 = 0.18 + S21m(q = 2), q = 0.8 template marginalised per l | 0.0023 | 0.042 | 0.0038 | +0.6 / +0.8 / +0.9 | 23.7/95 |
| same, no template | 0.0023 | 0.041 | 0.0038 | +0.7 / +0.8 / +0.9 | 23.9/95 |

Best fits identical in cosmology (b2 -0.38 -> -0.43, W_j6 -935 -> -877). Fisher (`fisher_rsd.py --layer2-info`, new
treatment "none", now with `--cubic-biases relation` as in the fits; the previous subsection's table omitted the cubic
offsets, which moves its entries by <= 0.5%): bands + S21m(q = 2) without template 0.0022 / 0.0419 / 0.0038, b2 / bG2
0.24 / 0.119; n_s free 0.0046 / 0.0685 / 0.0284 / 0.0058 -- the same as with the template marginalised. Caveat for the
previous subsection: its rows (d) and (f) used S21m at q = 0.8 (fit.py's default q), rows (a)/(b) at q = 2; at q = 2
the rows are S21m alone 0.0288 / 0.255 / 0.109 (none) and P_l to 0.2 + S21m 0.0021 / 0.0382 / 0.0035, n_s 0.0297 (none),
so the conclusions there hold at either q.

Consequence: the proposed halo baseline is all q = 2 -- the bands (sigma >= 12.5, l <= 6, k84 <= 0.18) and S21m (25 ->
70.7, white term W) -- with no layer-2 template, 8 marginalised amplitudes and the template's Monte Carlo error fewer, and
no dependence on the q = 0.8 Edgeworth factor (only E(1) remains in the S21m normalisation). Not run: AP and V = 64.

### The WST as a complement to P_l: the modulus cross-spectra P_Ud (2026-10-20, Phase 1)

Reframing: with P_l to 0.2 as the (well-tested) Gaussian backbone, the WST should carry only what P_l cannot. Its moduli
are even in delta, so S1 and S2 see the bispectrum only squared (S21m ~ r^2 P + noise) or through E(q). What they see
linearly is the response of the small-scale wavelet power to the large-scale modes: the cross-spectrum of a first-layer
modulus field with the density, P_{U_a d}(k, mu) = <U_a> r_a(k, mu) Z1 P_L(k) at tree level (a = sigma, l, |m|), zero
for a Gaussian field. It is already measured in every halo box (`PUd_m`, multipoles 0/2/4, also in the Omega_m and
sigma_8 derivative boxes) and r_a is the response of the S21 model (`theory.covariance_rsd.ModulusCrossRSD`).

Diagnostics before the forecast (scratchpad, V = 60):
- The second layer's complement to P_l is eaten by its own nuisances: P_l to 0.2 + S21m(q = 2, sigma_1 >= 17.7) gives
  0.0020 / 0.0358 / 0.0034 against 0.0023 / 0.0397 / 0.0038, and 0.0017 / 0.0261 / 0.0031 with its 14 noise amplitudes
  and W per sigma_1 known (n_s free: -9 to -13% -> -25 to -31%).
- Model-free, 1188 boxes: S/N^2 of a block-only amplitude, alone -> jointly with P_l(0.2): P_Ud (l = 1-4, ell = 0, 2,
  k <= 0.2) 9.6e5 -> 8.3e6 (sigma 12.5), 5.3e5 -> 3.2e6 (17.7), 2.7e5 -> 1.1e6 (25); S21m 4.5e4 -> 9.2e4. P_Ud
  correlates 0.85-0.97 with P_0 bin by bin at k <= 0.1 (sigma 12.5): sample-variance cancellation, as in multi-tracer
  analyses.
- The q = 2 bands are not fully redundant with P_l: at sigma >= 25 they add nothing, at sigma <= 17.7 (k84 <= 0.18)
  0.0018 / 0.0366 / 0.0032 and n_s 0.022 (n_s free), more than P_l to 0.25 on omega_cdm and n_s; at matched reach
  (k84 <= 0.15 with P_l to 0.15) still -19% on n_s. Open, P(k)-side question (k-tails, weighting, counterterms).

**1a, model check** (`scripts/fisher_rsd.py --response-info`): X_a = P_Ud / <U_a> (per-box mean) against the tree
response, k <= x / sigma_a, a k^2 counterterm and a constant per field fitted, V = 60 errors.
- At the P_l best-fit biases (bG2 = 0.06) and no FoG of the short modes, the data sit 10-30% above the model on the
  largest scales, with an |m| pattern (fitted per-field amplitudes 0.6-0.9 for m = 0, 1.2-1.5 for m = l).
- The pattern is the tidal bias and the FoG of the short modes: on a (b2, bG2, sigma_v) grid with no free amplitude the
  best points are bG2 = -0.3 to -0.6 (the WST fits' bG2 = -0.34), sigma_v = 6 Mpc/h (grid edge), chi^2/dof 1.4 over
  all fields at x = 0.75. With b2 = -0.5, bG2 = -0.3, sigma_v = 6: every field passes at k <= 0.75 / sigma (chi^2/dof
  <= 1.6) except four (sigma 12.5 l = 1 |m| = 0, 1 and l = 4 m = 0; sigma 17.7 l = 1 m = 0: 2.1-7.7). At k <= 1 /
  sigma most sigma <= 17.7 fields fail (chi^2/dof 2-135): the tree response holds to k ~ 0.06 (sigma 12.5), 0.04
  (17.7), 0.03 (25).

**1b, joint Fisher** (P_l with fit.py's model, binning and priors; X_a with the tree response at the shared cosmology,
b1, b2, bG2, a short-mode sigma_v and per field c_a k^2 P_L + e_a; joint sample covariance with Hartlap; V = 60;
`outputs/fisher/response_info_V60.json`):

| vector | n | omega_cdm | ln A_s | h | b1 / b2 / bG2 | n_s free: omega_cdm / ln A_s / n_s / h |
|---|---|---|---|---|---|---|
| P_l to 0.2 | 48 | 0.0023 | 0.0394 | 0.0038 | 0.044 / 0.59 / 0.33 | 0.0046 / 0.0669 / 0.0318 / 0.0057 |
| + S21m(q = 2, sigma_1 >= 17.7) (layer-2 Fisher) | 156 | 0.0020 | 0.0358 | 0.0034 | 0.038 / 0.17 / 0.088 | 0.0042 / 0.0601 / 0.0278 / 0.0052 |
| + X, sigma 17.7/25, k <= 0.75/sigma, validated fields | 182 | 0.0020 | 0.0354 | 0.0034 | 0.038 / 0.16 / 0.063 | 0.0044 / 0.0598 / 0.0282 / 0.0054 |
| + X, sigma 12.5/17.7/25, k <= 0.75/sigma, validated fields | 270 | 0.0019 | 0.0276 | 0.0034 | 0.028 / 0.10 / 0.038 | 0.0042 / 0.0509 / 0.0270 / 0.0053 |
| same, no k^2 counterterm (constant only) | 270 | 0.0017 | 0.0250 | 0.0032 | 0.025 / 0.08 / 0.032 | 0.0040 / 0.0478 / 0.0258 / 0.0051 |
| + X, sigma 12.5/17.7/25, k <= 1/sigma, all fields (beyond the tree reach) | 412 | 0.0013 | 0.0172 | 0.0030 | 0.017 / 0.06 / 0.026 | 0.0036 / 0.0350 / 0.0210 / 0.0047 |

Reading:
- **Gate 1 passes, narrowly.** Inside the tree reach, with every nuisance free, P_l + X gains -17% / -30% / -11% on
  omega_cdm / ln A_s / h (n_s fixed) and -9 / -24 / -15 / -7% with n_s free, against -13 / -10 / -11% for P_l +
  S21m. The sigma = 12.5 fields carry most of it.
- **It is the b1 lever.** sigma(b1) 0.044 -> 0.028 and ln A_s -30%: the response, measured almost free of sample
  variance, separates b1 from A_s, which neither P_l nor S21m does (S21m only reaches b2, bG2).
- **The prize is model-limited.** At k <= 1/sigma (where the tree model fails) the same vector would give -43% /
  -56% / -21% (n_s free -22 / -48 / -34 / -18%). A response model that holds to k ~ 0.08-0.1 at sigma = 12.5 (one-loop
  response, or a validated correction) is where the information is.
- Caveats: the check is at the truth cosmology with grid-fitted biases (P_l's own best fit has bG2 = 0.06, 1 sigma
  away); the derivatives are the tree model's, untested against simulations; 270 data points on 1188 boxes (Hartlap
  1.3, no Percival factor); per-field nuisances are free and flat.

Next (Phase 2, the gate that stopped the q = 0.8 template): the measured dX_a/dtheta on the Omega_m and sigma_8
derivative boxes (PUd_m is stored there) against the tree model with the nuisances refitted per set, for the validated
fields at k <= 0.75 / sigma. Only if it holds to ~10-15%: implement X in fit.py (loader, theory from the emulated basis,
AP), then profile scans and chains of P_l + X.

### P_l + modulus cross-spectra: gate, pipeline and fits (2026-10-20, Phases 2-4)

**Phase 2, derivative-box gate** (`fisher_rsd.py --response-info --response-gate`, `outputs/fisher/response_info_V60_gate.json`):
linear recovery of the Quijote steps (paired seeds, Omega_m +- 0.01 at fixed sigma_8, sigma_8 +- 0.015) by the joint
P_l + X Jacobian with every nuisance free, as a fit would see them. Shifts of the recovered step from the truth, in
units of the V = 60 errors:

| vector | Omega_m step: omega_cdm / ln A_s / h | sigma_8 step: omega_cdm / ln A_s / h |
|---|---|---|
| P_l to 0.2 (control) | +0.14 / -0.22 / +0.11 | 0.00 / +0.13 / -0.05 |
| P_l + X, sigma 17.7 / 25, k <= 0.75 / sigma | +0.09 / +0.13 / +0.14 | +0.12 / -0.23 / +0.04 |
| P_l + X, with sigma = 12.5 | -0.02 / +0.91 / +0.12 | +0.22 / -1.16 / 0.00 |

The tree response passes at sigma >= 17.7 and fails at sigma = 12.5, where it recovers only 55-73% of the amplitude
step: the b1 lever of the Phase-1 Fisher sits exactly where the tree response gets the amplitude dependence wrong (the
same failure mode as E(q)). With sigma >= 17.7 only, X adds to P_l + S21m(q = 2, sigma_1 >= 17.7) little with n_s fixed
(0.0019 / 0.0333 / 0.0033 against 0.0019 / 0.0349 / 0.0033) and 12-15% with n_s free (Fisher).

**Phase 3, pipeline.** New statistic `fit.py --stats ... pud`: X = PUd_m / Umean_m (multipoles 0, 2) of the modulus
fields with `--pud-min-scale` <= sigma <= `--pud-max-scale` (17.7, 25), l = 1-4, every |m| but `--pud-skip`
(j5_l1_m0), at k <= `--pud-x` / sigma (`data.load_response_dataset`); model `calculators.ResponseTheory`, a
second-order Taylor table of the tree response in (omega_cdm, ln A_s, h, n_s, b1, b2, bG2, sigma_v^U) about the
Quijote cosmology and (b1, b2, bG2, sigma_v) = (1.9, -0.5, -0.3, 6) (`inference.response_taylor`, cached in
`outputs/emulators/response_taylor/`, built in ~1 min), plus a counterterm cU k^2 P_L and a constant eU per field;
b1, b2, bG2 are shared with P_l and S21m. The table matches the exact tree response to 0.016 sigma rms (0.05 max) at
V = 60 at a point 2-5 sigma away. No AP or window yet.

**Phase 4, fits** (V = 60, no AP, no-AP fog_cubic emulator, profiles and chi^2 scans with one parameter fixed,
parabola widths, shifts of the minima from the truth; runs in `outputs/inference/rsd/pud/`; P_l references `q2/pk02_*`):

| vector | sigma(omega_cdm) | sigma(ln A_s) | sigma(h) | shifts omega_cdm / ln A_s / h | chi^2 / (n - p) |
|---|---|---|---|---|---|
| P_l to 0.2 | 0.0024 | 0.039 | 0.0040 | -0.9 / +1.4 / -0.7 | 18.8 / 33 |
| P_l + S21m(q = 2, sigma_1 >= 17.7, no template) | 0.0019 | 0.037 | 0.0033 | -0.9 / +1.1 / -0.4 | 42.5 / 110 |
| P_l + X, k <= 0.75 / sigma | 0.0020 | 0.034 | 0.0033 | +0.9 / -1.8 / +0.8 | 125.6 / 112 |
| P_l + S21m + X, k <= 0.75 / sigma | 0.0020 | 0.033 | 0.0033 | +0.3 / -2.1 / +0.3 | 141.8 / 189 |
| P_l + X, k <= 0.5 / sigma | 0.0021 | 0.038 | 0.0034 | -0.1 / +0.7 / 0.0 | 39.9 / 58 |
| **P_l + S21m + X, k <= 0.5 / sigma** | **0.0019** | **0.036** | **0.0033** | **-0.5 / +0.2 / -0.1** | 65.7 / 135 |

n_s free (scans of n_s and ln A_s): P_l sigma(n_s) 0.0316, sigma(ln A_s) 0.069 (shifts 0.0 / +0.8); P_l + S21m + X
(k <= 0.5 / sigma) 0.0250 (-21%), 0.058 (-16%) (shifts +0.7 / +0.7).

Splitting the k <= 0.75 / sigma pull: sigma = 25 alone recovers the truth (ln A_s 3.094, omega_cdm 0.1209, h 0.6717);
sigma = 17.7 alone pulls ln A_s to 3.039; both at k <= 0.5 / sigma recover it. The fiducial-level residual of the tree
response in the last bins (k = 0.03-0.04 at sigma = 17.7) projects onto b1-A_s, the direction X is sensitive to.

Reading / conclusion:
- The complement works, modestly. With every piece validated (tree response at sigma >= 17.7, k <= 0.5 / sigma; S21m
  at q = 2 without a template), P_l + S21m + X is 21% / 8% / 18% tighter than P_l to 0.2 on omega_cdm / ln A_s / h
  with n_s fixed and 21% on n_s with n_s free, and it removes most of P_l's own drift (ln A_s +1.4 -> +0.2 sigma). Of
  the non-Gaussian pieces, S21m carries the omega_cdm / h gain and X the unbiasing; neither reaches the b1 lever.
- The large Fisher prize (ln A_s -30 to -56%, the b1 lever) needs the response at sigma = 12.5 and k ~ 0.05-0.1, where
  the tree response fails both at the fiducial and in its amplitude dependence. The next model ingredient, if any, is
  the one-loop response of the modulus fields (or a validated, cosmology-dependent correction tested on the derivative
  boxes as above); the gate test is cheap and should come first.
- Proposed P(k) + WST vector: P_l to 0.2 + S21m(q = 2, sigma_1 >= 17.7, white W, no layer-2 template) + X(sigma =
  17.7-25, l = 1-4, k <= 0.5 / sigma). Not done: AP for X (the filters in observed coordinates and the cross-spectrum
  stretch), V = 64, chains, survey window.
