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

### Chains: P_l vs P_l + S21m(q = 2) + X (2026-10-20)

`fit.py --pud-marg` marginalises the per-field counterterm and constant of X analytically (Gaussian widths 3000 /
30000, ~10x the largest best-fit values; commit f667709): same best fit and chi^2 as the free fit, 101 -> 47
parameters. NUTS chains at V = 60, no AP, no-AP fog_cubic emulator, `outputs/inference/rsd/pud/*_chain_V60`; plots
`outputs/plots/triangle_pk_s21_pud_{nsfixed,nsfree}_V60.png`. The P_l chain with n_s fixed on this emulator did not
converge (R - 1 = 4 at the 20000-step cap, a stuck chain); its reference is the converged `q2/pk02_chain_V60`
(bands12 emulator, the same P_s grid to 1e-4).

| chain (R - 1 <= 0.01, ESS >= 366) | omega_cdm | ln A_s | n_s | h | shifts [sigma] |
|---|---|---|---|---|---|
| P_l to 0.2, n_s fixed | 0.0021 | 0.0377 | -- | 0.0035 | -0.74 / +1.16 / -- / -0.56 |
| P_l + S21m + X, n_s fixed | 0.0020 | 0.0371 | -- | 0.0034 | -0.57 / 0.00 / -- / -0.27 |
| P_l to 0.2, n_s free | 0.0044 | 0.0627 | 0.0290 | 0.0054 | -0.04 / +0.44 / -0.34 / -0.08 |
| P_l + S21m + X, n_s free | 0.0038 | 0.0590 | 0.0257 | 0.0048 | -0.55 / +0.21 / +0.30 / -0.40 |

Reading: the posteriors are smaller gains than the profile scans suggested. With n_s fixed the complement narrows the
marginals by 2-5% (the P_l marginals are themselves 10-15% narrower than its profile widths: the profile and the
marginal differ along the bias-counterterm directions), and its effect is to remove P_l's ln A_s pull (+1.16 -> 0.00
sigma) and halve the h one. With n_s free it gains 14% (omega_cdm), 6% (ln A_s), 11% (n_s) and 11% (h), with every
shift below 0.6 sigma. The validated P(k) + WST complement is therefore worth ~10% with n_s free and mainly a
robustness gain with n_s fixed; the large Fisher prize stays behind the sigma = 12.5 response model.

### Percival penalty and an analytic covariance for P_l + S21m + X (2026-10-20)

The joint vector (236 points, 47 parameters, 1188 boxes) pays Hartlap 1.25 and Percival m1 = 1.14, i.e. sigma x 1.068
against x 1.007 for P_l alone: ~6 points of the chain gains. Percival inflation sqrt(m1) of this vector with 3000 /
15000 boxes: 1.025 / 1.005.

Analytic covariance (`analytic_covariance_rsd_wst.py --pud`, `outputs/covariance/halos_pk_s21q2_pud_model.npz`): the
X block added to `RSDWSTCovariance` (``response``): with U_a = c_a d + n_a, the c_a |d|^2 part and the fluctuation of
the per-box <U_a> are rows of the 2nd-chaos matrix (so the X-P_l and X-S21m(delta) crosses come for free), the n_a d*
part gives N'_ab P per mode and 2 c_b N'_ab P with the second layer of U_b. Fixes on the way: the box branch now takes
sigma_v from c_v (fog emulators), the modulus auto noise is floored at 5% of g_aa (a negative fitted W overtook it at
high K: NaNs) and the noise tables are projected on positive semi-definite matrices per (k, mu) node. `fit.py
--covariance-file` now also works for box fits (no Hartlap / Percival).

Validation against the 1188 boxes (variance of the boxes along the eigen-directions of the analytic matrix; 1 if right):
P_l 1.04 (weakest 10%: 1.09), S21m 1.01 (1.29), **X 7.9 (58)**, P_l + X 6.0 (15). Diagonals agree (0.93-0.98) and the
correlations to 0.03-0.05 rms, but X is 85-97% correlated with P_0, and its information sits in the directions
decorrelated from P_l, where its variance is the noise of the moduli, which the Gaussian-chaos model underestimates by
up to ~50x (the connected noise that the free W absorbs in S21m). An analytic X covariance would overstate its
information: not usable. The analytic P_l + S21m blocks are fine (for parameter errors). Way forward: more
realizations (being downloaded), and/or compressing X and S21m (|m| averaging or PCA) to shrink the vector.

### What would a one-loop WST model buy on top of the proposed P(k) + WST baseline? (2026-10-20)

`fisher_rsd.py --response-info --response-oneloop` (`outputs/fisher/response_info_V60_oneloop.json`), V = 60, the
original 1188 halo boxes (pinned: the fiducial directory is growing), Hartlap and Percival as in the fits (the
per-field linear nuisances of X not counted in m1, as with `--pud-marg`); in brackets the many-simulation limit (m1 =
1). The Fisher errors match the chains to ~5% (n_s fixed, P_l 0.0022 / 0.0394 / 0.0037 against 0.0021 / 0.0377 /
0.0035; the tree baseline 0.0020 / 0.0380 / 0.0035 against 0.0020 / 0.0371 / 0.0034).

"One loop" here means: the vectors a model beyond tree level would open, with the tree Jacobian standing in for the
one-loop one. X to k <= 1 / sigma and 1.5 / sigma (k = 0.08 / 0.12 at sigma = 12.5) and with the sigma = 12.5
fields; S21m at sigma_1 = 12.5 (no emulator covers it: exact CLASS basis); the q = 0.8 first layer (sigma >= 17.7,
l <= 4) with the E(q) nuisances of the Edgeworth gate free (A_TN, A_TN2, a_ng per sigma). X nuisances per field: (c,
e) = the k^2 P_L counterterm and constant of the current fits; (c, c2, e) = the standard one-loop EFT set, with a
separate quadrupole k^2 term; (c, c2, e, k4) adds k^4 P_L.

n_s fixed, sigma(omega_cdm) / sigma(ln A_s) / sigma(h):

| vector | 1188 boxes | many simulations |
|---|---|---|
| P_l to 0.2 | 0.0022 / 0.0394 / 0.0037 | 0.0022 / 0.0390 / 0.0037 |
| tree baseline (P_l + S21m(17.7) + X(17.7-25, k <= 0.5/sigma)) | 0.0020 / 0.0380 / 0.0035 | 0.0019 / 0.0357 / 0.0033 |
| + S21m at sigma_1 = 12.5 (no X) | 0.0019 / 0.0358 / 0.0033 | 0.0018 / 0.0344 / 0.0032 |
| + S1m(q = 0.8), a_ng per sigma (no X) | 0.0018 / 0.0337 / 0.0033 | 0.0017 / 0.0320 / 0.0032 |
| + S1m(q = 0.8) + S21m(12.5) (no X) | 0.0018 / 0.0322 / 0.0034 | 0.0017 / 0.0303 / 0.0031 |
| X(12.5-25, k <= 1/sigma), (c, e) | 0.0018 / 0.0225 / 0.0039 | 0.0014 / 0.0178 / 0.0031 |
| X(12.5-25, k <= 1.5/sigma), (c, e) | 0.0012 / 0.0129 / 0.0038 | 0.0008 / 0.0089 / 0.0026 |
| X(12.5-25, k <= 1/sigma), (c, c2, e) | 0.0023 / 0.0402 / 0.0043 | 0.0018 / 0.0320 / 0.0034 |
| X(12.5-25, k <= 1.5/sigma), (c, c2, e) | 0.0016 / 0.0260 / 0.0042 | 0.0011 / 0.0180 / 0.0029 |
| X(12.5-25, k <= 1.5/sigma), (c, c2, e, k4) | 0.0022 / 0.0299 / 0.0046 | 0.0015 / 0.0207 / 0.0032 |
| all: q = 0.8 + S21m(12.5) + X(k <= 1.5/sigma), (c, c2, e) | 0.0019 / 0.0290 / 0.0050 | 0.0012 / 0.0185 / 0.0031 |

n_s free, sigma(omega_cdm) / sigma(ln A_s) / sigma(n_s) / sigma(h):

| vector | 1188 boxes | many simulations |
|---|---|---|
| P_l to 0.2 | 0.0045 / 0.0655 / 0.0313 / 0.0055 | 0.0045 / 0.0649 / 0.0311 / 0.0055 |
| tree baseline | 0.0040 / 0.0622 / 0.0270 / 0.0051 | 0.0038 / 0.0586 / 0.0254 / 0.0048 |
| + S21m at sigma_1 = 12.5 (no X) | 0.0038 / 0.0590 / 0.0260 / 0.0048 | 0.0037 / 0.0568 / 0.0250 / 0.0046 |
| + S1m(q = 0.8), a_ng per sigma (no X) | 0.0033 / 0.0503 / 0.0218 / 0.0045 | 0.0032 / 0.0480 / 0.0207 / 0.0042 |
| + S1m(q = 0.8) + S21m(12.5) (no X) | 0.0033 / 0.0489 / 0.0214 / 0.0044 | 0.0031 / 0.0461 / 0.0201 / 0.0041 |
| X(12.5-25, k <= 1/sigma), (c, e) | 0.0038 / 0.0390 / 0.0223 / 0.0052 | 0.0030 / 0.0310 / 0.0177 / 0.0041 |
| X(12.5-25, k <= 1.5/sigma), (c, c2, e) | 0.0042 / 0.0467 / 0.0225 / 0.0056 | 0.0029 / 0.0325 / 0.0156 / 0.0039 |
| all, (c, c2, e) | 0.0045 / 0.0503 / 0.0245 / 0.0063 | 0.0029 / 0.0323 / 0.0156 / 0.0040 |

Reading:
- **The counterterm budget of a one-loop X decides most of the answer.** With one k^2 term and a constant per field,
  a one-loop X at k <= 1 / sigma would cut sigma(ln A_s) by 43% (n_s fixed; -40% with n_s free) and at 1.5 / sigma
  by 67%; with the standard one-loop set (separate monopole and quadrupole k^2 terms) nothing survives at k <= 1 /
  sigma and 34% (n_s fixed) / 29% (n_s free) at 1.5 / sigma. The information of X is in the k- and mu-shape of the
  response, which per-field counterterms absorb; it is the b1 lever (sigma(b1) 0.046 -> 0.013-0.027). Nothing helps h.
- **The number of boxes matters as much as the model.** At 500-800 data points the Percival factor reaches m1 ~ 2
  with 1188 boxes (sigma x 1.4): with many simulations the realistic one-loop X at 1.5 / sigma would give -50 / -54 /
  -22% (n_s fixed) and -36 / -50 / -50 / -29% (n_s free) over P_l, against -27 / -34 / +14% and -7 / -29 / -28 / +2%
  with 1188 boxes. A compression of X (PCA, |m| averaging) or more boxes comes before any one-loop work.
- **The q = 0.8 first layer at one loop is the cheapest gain with n_s free** (-21 / -21 / -22 / -13% over P_l + S21m,
  E nuisances per sigma free, no X needed, small vector), but it halves with a_ng per (sigma, l) and needs a one-loop
  K4 whose b2 response passes the derivative-box gate (the tree one fails by 2.5-3.5x).
- **S21m at sigma_1 = 12.5 adds little** (-3% n_s fixed, -7 to -10% n_s free).
- Realistic best case with 1188 boxes, n_s free: q = 0.8 + S21m(12.5) without X, -27 / -25 / -32 / -20% over P_l
  (-18 / -21 / -21 / -14% over the tree baseline). With many boxes and a one-loop X with the standard counterterms:
  -36 / -50 / -50 / -29% over P_l.

Caveats: tree Jacobians stand in for one-loop ones; a one-loop model must still pass the derivative-box gate (the tree
response at sigma = 12.5 recovers only 55-73% of the amplitude step); a one-loop RSD bispectrum-type response for
biased tracers brings its own bias and counterterm parameters (only the per-field counterterms are modelled here).

### Chains with 2201 boxes (2026-10-20, afternoon)

The fiducial halo set on NERSC is growing to ~5000 boxes; the 1013 finished by 13:00 were added to the local copy
(`data/quijote/fiducial/z0.5/halos_m13_.../rsd`, now 2201 boxes, same settings, all with the modulus spectra; every fit
that globs the directory now uses them). Percival m1 of P_l + S21m + X: 1.141 -> 1.070 (sigma x 1.068 -> 1.034);
P_l alone 1.015 -> 1.008. The four NUTS chains rerun at V = 60, no AP (`pud/*_chain_n2201_V60`; the n_s-fixed P_l
chain stuck twice, R - 1 = 4 and 0.55, and converged with a new seed and the 1188-box P_l chain as proposal); plots
`outputs/plots/triangle_pk_s21_pud_{nsfixed,nsfree}_n2201_V60.png` (and `..._nsfree_..._centred.png`, posteriors
shifted to the truth). All R - 1 <= 0.01, ESS >= 325.

| V = 60 | 1188 boxes: P_l -> P_l + S21m + X | 2201 boxes: P_l -> P_l + S21m + X | shifts at 2201 [sigma], P_l / P_l + S21m + X |
|---|---|---|---|
| n_s fixed, omega_cdm | 0.00211 -> 0.00201 (-5%) | 0.00209 -> 0.00195 (-7%) | -0.62 / -0.08 |
| n_s fixed, ln A_s | 0.0377 -> 0.0371 (-2%) | 0.0372 -> 0.0355 (-4%) | +1.10 / -0.07 |
| n_s fixed, h | 0.00352 -> 0.00343 (-3%) | 0.00350 -> 0.00330 (-6%) | -0.51 / +0.01 |
| n_s free, omega_cdm | 0.00439 -> 0.00375 (-14%) | 0.00445 -> 0.00345 (-23%) | +0.40 / -0.40 |
| n_s free, ln A_s | 0.0627 -> 0.0590 (-6%) | 0.0620 -> 0.0548 (-12%) | +0.03 / +0.24 |
| n_s free, n_s | 0.0290 -> 0.0257 (-11%) | 0.0291 -> 0.0239 (-18%) | -0.78 / +0.43 |
| n_s free, h | 0.00535 -> 0.00477 (-11%) | 0.00547 -> 0.00446 (-18%) | +0.28 / -0.27 |

Reading: the gap widens with the covariance set, by more than the Percival factor alone (3.4% of sigma): with n_s free
it roughly doubles (-12 to -23%), with n_s fixed it goes from -2 to -5% to -4 to -7%. P_l's own marginals barely move
(as expected: its m1 was already 1.015); the combined vector, with 236 points against 48, was paying for the noise of
its 1188-box covariance beyond the Percival correction (the Hartlap-corrected precision of a 236-point vector is itself
noisy). With the full ~5000 boxes the remaining Percival cost is ~1.5%; the further gain should be smaller than this
step. In every case the complement also brings the shifts within 0.45 sigma (P_l: up to 1.1 sigma in ln A_s with n_s
fixed, 0.8 sigma in n_s with n_s free).

### Is there tree-level non-Gaussian information left in the WST? (2026-10-20, evening)

The user's question: before settling for ~10-20% over P_l (less than bispectrum analyses report), make sure every
non-Gaussian piece the WST measures that a tree-level model could handle has been tried. Method, on the halo boxes
(the local fiducial set grew to 4227 boxes during the day; Percival m1 of a 320-point vector is now 1.06): per-field
model check (chi^2/dof of X_a against the box mean, counterterm c and constant e fitted), a **joint check** of a scale
set in its joint sample covariance with c and e marginalised analytically (`fisher_rsd.py --response-info`, new:
`--response-x`, `--response-ells`, `--response-scales`, the "A" nuisance kind = a free amplitude per field), the
derivative-box gate (`--response-gate`), Fisher with (c, e) and with (A, c, e), and profile fits. The joint check
matters because the fields are 85-97% correlated with each other and with P_0: a coherent few-% pattern across fields
that every field tolerates on its own is a large chi^2 in the joint fit.

| piece | per-field check | joint check (c, e marg.) | gate | fit / Fisher at V = 60 | verdict |
|---|---|---|---|---|---|
| X, sigma 17.7-25, k <= 0.5/sigma (baseline) | ok | ok (in the fit: chi^2 66/236 with P_l + S21m) | <= 0.45 sigma | chains: -4 to -7% (n_s fixed), -12 to -23% (n_s free) with S21m | in |
| X, sigma 17.7-25, k <= 0.75/sigma | ok (chi^2/dof <= 1.6) | **156 / 84** (44 with a free amplitude per field) | 0.5 | fits biased: ln A_s -1.8 to -2.1 sigma | out |
| X, sigma 12.5, k <= 0.5/sigma (k_L <= 0.04, 3 bins) | ok (<= 1.4) | **412 / 84** (50 with free amplitudes) | 0.33 / 0.16 / 0.22 sigma | profile: chi^2 381/273, ln A_s -5 sigma, omega_cdm +2 sigma | out |
| same, Fisher | | | | P_l + S21m + X: ln A_s 0.0302 (-23% vs P_l, -17% vs P_l + S21m), n_s free -18/-26/-21/-14%; with a free amplitude per field 0.0353 (-3%), n_s free -1 to -2% | the information is the amplitude |
| X, sigma 12.5, k <= 0.6/sigma | 3 fields fail (2.3-7.2) | -- | 0.46 sigma | Fisher ln A_s 0.0280 | out |
| X, sigma 12.5-25, k <= 0.75/sigma | 4 fields fail | 1412 / 196 (336 with free amplitudes: shape errors too) | fails (55-73% of the A_s step) | -- | out |
| X, sigma 8.8, k <= 0.5/sigma | l >= 2, m <= 1 fail (chi^2/dof 3-16) | -- | -- | -- | out |
| X, sigma 35.4 (k <= 0.75-1.5/sigma) | ok | -- | -- | adds nothing (n 188 -> 216, same errors) | nothing there |
| X, hexadecapole (ell = 4) | ok | -- | -- | adds nothing (ln A_s 0.0300 vs 0.0298) | nothing there |
| X, k <= 1-1.5/sigma at sigma <= 25 | fails (chi^2/dof 4-630) | -- | -- | the large Fisher prize of the one-loop subsection | needs loops |
| q != 2 first layer, E(q) | tree K4, cubic + A_TN, NNLO: none within V = 60 errors at sigma <= 25 | | sigma_8 response 2.5-3.5x the data | | out (earlier subsections) |
| q-dependence of the cross-spectra P_{U^q d} | X_q / <U^q> = (q/2) x the same response at tree level: the q-dependence is the one-point (E-type) information | | | | nothing new at tree level |
| S21m(q = 2) | ok | ok | ok | b2, bG2 only; sigma_1 = 12.5 adds -3% (one-loop subsection) | in; its noise (connected 4-point) failed earlier |
| third layer, S2 at q != 2, cross-field second layer | 6-point and beyond, or the modulus noise again | | | | not tree level |

Where the sigma = 12.5 misfit sits (at the baseline best fit, k <= 0.5/sigma): per field, data/model - 1 of the
monopole rises from 0 at k = 0.013 to +5% at k = 0.036 for every (l, m); the quadrupole is -2 to -8% for the m = l
(transverse) fields and ~0 for m = 0. A free amplitude per field, -0.00 to -0.10 growing with m/l, brings the joint
chi^2 from 412 to 47 on 84 points (one amplitude for all fields: 321; per l: 295; per m/l: 151). The short modes of
these fields are at k ~ 0.13-0.22: this is the nonlinear, anisotropic (FoG-like) response of the halo field in
redshift space, not captured by the tree kernels with a Gaussian damping and one sigma_v.

Reading:
- **At tree level the WST's non-Gaussian information for halos in redshift space is exhausted** by the deep-squeezed
  response at sigma >= 17.7 (k_L <= 0.5/sigma) and the bias information of S21m(q = 2). Every other piece either fails
  the joint check / the gate (X at sigma = 12.5 or at k > 0.5/sigma, E(q)) or carries nothing (ell = 4, sigma >=
  35.4, the q-dependence of X).
- **The missing gain is a 3-10% amplitude per field at sigma = 12.5** (and at sigma 17.7-25 beyond k = 0.5/sigma).
  Since the information of X is its amplitude (with free amplitudes X adds ~0 to P_l + S21m), a model of those
  amplitudes to ~1% is what the b1 lever (ln A_s -23% in Fisher, -40 to -56% with the reach of the one-loop subsection)
  costs. That is one-loop (or beyond) RSD modelling of the short modes at k ~ 0.15-0.2 for a biased tracer: the same
  threshold at which bispectrum analyses move from tree level (k_max ~ 0.08, ~10-15% on sigma_8 in BOSS) to one loop
  (k_max ~ 0.2). The WST at tree level is where the tree-level bispectrum is, not below it.
- Cheapest routes beyond, in order: (i) the one-loop *response* of X rather than a one-loop bispectrum: the amplitude
  of X_a is the response of the band power of field a to a long mode (position-dependent band power), i.e. the
  separate-universe response of the one-loop P_s(k, mu) that the P_l emulator already provides (growth, dilation, PBS
  bias, Kaiser and FoG responses as derivatives of the emulated P_s), to be gated on the derivative boxes and the joint
  check; (ii) P_{|X|^2 d} (skew-spectrum form of X) in the ongoing 5000-box measurement: it removes the chaos-projection
  part of the amplitude error (O(kappa_4) ~ 1-3% at sigma = 12.5) at no cost, one extra FFT per field; (iii)
  simulation-calibrated amplitudes (the derivative boxes give the cosmology response; the bias dependence of a galaxy
  sample does not transfer).

### Chains with all 5000 boxes (2026-10-20, evening)

All 5000 fiducial halo boxes are now local (same settings, modulus spectra included). Percival m1: P_l + S21m + X
1.029 (sigma x 1.015), P_l 1.003. Chains `pud/*_chain_n5000_V60` (V = 60, no AP, proposals from the 2201-box chains;
the n_s-free P_l chain stuck once, R - 1 = 2.1, and converged with a new seed; all R - 1 <= 0.01, ESS >= 320); plots
`outputs/plots/triangle_pk_s21_pud_{nsfixed,nsfree}_n5000_V60.png` and the WST blocks at the best fit,
`outputs/plots/wst_bestfit_pk_s21_pud_n5000_V60.png`. Change of sigma, P_l to 0.2 -> P_l + S21m(q = 2) + X:

| boxes | n_s fixed: omega_cdm / ln A_s / h | n_s free: omega_cdm / ln A_s / n_s / h |
|---|---|---|
| 1188 | -5 / -2 / -3% | -14 / -6 / -11 / -11% |
| 2201 | -7 / -4 / -6% | -23 / -12 / -18 / -18% |
| 5000 | **-10 / -8 / -10%** | **-23 / -14 / -21 / -20%** |

5000 boxes, absolute: n_s fixed P_l 0.00206 / 0.0369 / 0.00347 -> 0.00185 / 0.0338 / 0.00311; n_s free 0.00445 /
0.0611 / 0.0283 / 0.00546 -> 0.00341 / 0.0528 / 0.0225 / 0.00435. Shifts from the truth: P_l up to +1.0 sigma
(ln A_s, n_s fixed) and -1.0 sigma (n_s, n_s free); P_l + S21m + X within 0.53 sigma everywhere.

Reading: the n_s-free gain saturated at ~2200 boxes; the n_s-fixed one kept growing (it was the most sensitive to
covariance noise, the gain being small against P_l's own marginal). At the full set the covariance noise is no longer
a factor (Percival 1.5% of sigma). The validated P(k) + WST complement is worth 8-10% with n_s fixed and 14-23% with
n_s free, at V = 60, and it removes P_l's ~1 sigma parameter pulls.

### AP for the modulus cross-spectra X, and AP chains of P_l and the baseline (2026-10-21)

**Model (da5f418).** `ModulusCrossRSD.spectra(ap=(q_par, q_perp))`: the p integral runs over true wavevectors, the
filters and harmonics of every leg at the observed ones (true x (q_perp, q_perp, q_par)), the spectra and the shot
noise (N q_par q_perp^2) true, the long mode at the true k' of the observed (k, mu), the multipole weights on the
observed mu, and X = r Z1(k') P_L(k') / (q_par q_perp^2) (Sec. haloap). `response_taylor(ap_omega_b=)` puts AP inside
every evaluation, so the quadratic Taylor table in (omega_cdm, h) carries it; `fit.py --stats pud --ap` works on boxes
(the survey window branch still refuses --ap). Checks: q = 1 reproduces the old response exactly; an isotropic dilation
matches X_obs(k; sigma) = X_true(k / a; sigma a) / a^3 to 1e-5; an independent observed-frame quadrature agrees to
5e-6 at q != 1 and in d X / d ln q_par, d ln q_perp to 5e-5 (d ln X0 / d ln q ~ -1.3 and -1.8 at k = 0.016); Taylor
table vs direct evaluation: dchi2 <= 0.03 to 4 sigma in omega_cdm or h (0.19 at a 2-sigma corner in five parameters),
as accurate as the no-AP table.

**Emulator.** `outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic_ap3_v5.{h5,json,validation.json}`: S1 at
sigma >= 17.6 (l <= 4), S21 at sigma1 = 17.7 and 25 (the v5 baseline; ap3_base and ap3_bands12 only have sigma1 = 25),
AP order 3, 369 nodes; Perlmutter CPU node, 48 ranks (OOM in the last round again), resumed with 32 ranks, which this
time also fitted and validated (EXIT 0; scripts in `$SCRATCH/wstfast_v5`). Validation max rel. error: S21m 2.5e-4,
P_l 0.6%. Without --ap it reproduces the no-AP baseline best fit exactly.

**Chains** (n_s free, V = 60, 5000 boxes, NUTS, R - 1 < 0.03, ESS >= 356): `pud/pns_ap_chain_n5000_V60`,
`pud/sx05mns_ap_chain_n5000_V60`; plots `outputs/plots/triangle_pk_ap_n5000_V60.png`,
`outputs/plots/triangle_pk_s21_pud_ap_n5000_V60.png`. Mean +- sigma (shift from the truth in sigma):

| run | omega_cdm | ln A_s | n_s | h | chi2 |
|---|---|---|---|---|---|
| P_l, no AP | 0.1239 +- 0.0045 (+0.66) | 3.052 +- 0.061 (-0.16) | 0.935 +- 0.028 (-0.96) | 0.6741 +- 0.0055 (+0.55) | 17.1 |
| P_l, AP | 0.1204 +- 0.0040 (-0.12) | 3.092 +- 0.060 (+0.52) | 0.956 +- 0.027 (-0.23) | 0.6713 +- 0.0040 (+0.05) | 17.0 |
| baseline, no AP | 0.1203 +- 0.0034 (-0.19) | 3.072 +- 0.053 (+0.21) | 0.974 +- 0.023 (+0.53) | 0.6709 +- 0.0043 (-0.05) | 61.5 |
| baseline, AP | 0.1164 +- 0.0032 (**-1.40**) | 3.126 +- 0.055 (**+1.19**) | 0.998 +- 0.023 (**+1.56**) | 0.6705 +- 0.0036 (-0.17) | 58.7 |

AP tightens P_l (h -27%, omega_cdm -11%) and removes its ~1 sigma pulls. With AP the baseline gains -20 / -9 / -18 /
-10% over P_l (omega_cdm / ln A_s / n_s / h), but it is biased.

**Which block (profiles, ap3_v5, V = 60).** Best fits omega_cdm / ln A_s / n_s / h:

| data | no AP | AP |
|---|---|---|
| P_l | (chain above) | 0.1194 / 3.109 / 0.962 / 0.6706 |
| P_l + X | 0.1232 / 3.066 / 0.953 / 0.6743 | 0.1192 / 3.113 / 0.976 / 0.6719 |
| P_l + S21m(17.7, 25) | 0.1182 / 3.118 / 0.981 / 0.6686 | **0.1151 / 3.163 / 1.000 / 0.6692** |
| baseline | 0.1196 / 3.090 / 0.978 / 0.6702 | (chain above) |

With AP, X sits on P_l; the bias comes from S21m with AP. S21m already pulls omega_cdm low without AP (0.1182, -0.8
sigma), and its AP pushes about one sigma further along the omega_cdm - ln A_s - n_s degeneracy (chi2 43.1 -> 40.3).
Open: whether that is the AP polynomial of the S21m layers (ap_order 3 in ln q, only tested on nodes) or the AP model
of S21m itself; next check: exact (unexpanded) S21m AP vs the ap3 polynomial at the AP chain's best-fit q, block by block
(sigma1 = 17.7 vs 25).

#### The baseline AP bias is not the S21m AP polynomial, nor the S21m AP model (2026-10-21)

Profiles (ap3_v5, V = 60, n_s free; best fit omega_cdm / ln A_s / n_s / h; P_l + S21m with AP: 0.1151 / 3.163 / 1.000
/ 0.6692, chi2 40.3):

- **Polynomial vs exact AP.** The theory at the AP chain's best fit (ln q_par = 0.0067, ln q_perp = 0.0034) with every
  AP piece exact (exact CLASS basis, APNodes forced to a single node at x*, where the least-squares fit is exact):
  per block max |Delta| / sigma <= 0.019 (S21m sigma1 = 25, l = 4), 0.012 (P_l), 0 (X); whitened |Delta|^2 = 0.007;
  chi2 58.67 vs 58.71. The AP effect itself at that point is |Delta|^2 = 228. The order-3 polynomial is exact here.
- **AP of S21m switched off piece by piece** (P_l keeps its AP): cumulants (Edgeworth) 0.1151 / 3.163 / 1.000; first-
  layer linear blocks 0.1151 / 3.163 / 1.000; second-layer S21 terms 0.1159 / 3.150 / 0.995; only the lambda_2 volume
  factor 0.1156 / 3.153 / 0.996; all of S21m's AP 0.1159 / 3.150 / 0.995. The S21m AP model moves omega_cdm by <= 0.25
  sigma: the bias is there without it.
- The question "exact AP at the truth vs no AP" is empty on these boxes: the fiducial is the truth, q = 1 there, and
  AP changes only the derivatives of the model.
- **What it is.** S21m without AP already pulls omega_cdm low (P_l + S21m no AP: 0.1182, -0.8 sigma), while P_l alone
  without AP is biased the other way (+0.66 sigma omega_cdm, -0.96 n_s): the no-AP baseline was unbiased by
  cancellation. AP removes P_l's own pull (P_l AP 0.1194 / 3.109 / 0.962 / 0.6706) and exposes S21m's. By scale:
  S21m at sigma1 = 25 only, AP: 0.1186 / 3.131 / 0.980 (-0.6 sigma; no AP 0.1220): the pull comes mostly from sigma1 =
  17.7. Delta chi2 (truth - best fit, nuisances re-profiled) = 4.3, spread: P_2 +2.8, S21m +1.7 (sigma1 17.7 l = 2 +1.1,
  l = 3 -2.4, l = 4 +1.8; sigma1 25 l = 4 +1.6). At the truth P_l alone wants bG2 = 0.07 +- 0.28, bGamma3 = -0.34 +-
  0.74 and P_l + S21m bG2 = -0.46, bGamma3 = 1.05; fixing them: P_l AP at S21m's values 0.1189 (-0.5 sigma: part of
  it), P_l + S21m AP at P_l's values 0.1123, chi2 74.8 (S21m insists). So S21m at sigma1 = 17.7 prefers low omega_cdm /
  high n_s at its own biases: a model bias of the 17.7 Mpc/h S21m, not of AP (cf. the q = 2 bands, biased from k84 ~ 0.2:
  the 17.7 first-layer fields reach those scales).
- Not changed: no fix applied, no chain rerun (the AP polynomial is not at fault). Options: cut S21m to sigma1 = 25
  (-0.6 sigma with AP; costs part of the gain), or a k84 cut of the sigma1 = 17.7 first-layer fields as for the bands,
  or one-loop S21m at 17.7.

#### A k84 cut of the S21m first layer does not remove the AP bias (2026-10-21)

New option `fit.py --s21-kmax` (`FitOptions.s21_kmax`): keep the S21m coefficients whose first-layer field (sigma_j1, l)
has `band_k84` below the cut. The sigma1 = 17.7 fields reach k84 = 0.113 / 0.130 / 0.145 / 0.158 (l = 1-4), sigma1 = 25
<= 0.111, so the band cut (0.18) would drop nothing; tested tighter ones. Baseline with AP (P_l + S21m + X, ap3_v5,
V = 60, n_s free), profile best fits and shifts from the truth (sigma = profile errors):

| S21m kept | omega_cdm | ln A_s | n_s | h | chi2 / dof |
|---|---|---|---|---|---|
| all (chain mean) | 0.1164 (-1.40) | 3.126 (+1.19) | 0.998 (+1.56) | 0.6705 (-0.17) | 58.7 / 188 |
| k84 <= 0.15 (drops 17.7 l = 4) | 0.1161 (-1.45) | 3.139 (+1.4) | 0.999 (+1.6) | 0.6700 (-0.3) | 57.4 / 168 |
| k84 <= 0.14 (drops 17.7 l = 3, 4) | 0.1172 (-1.1) | 3.131 (+1.3) | 0.993 (+1.3) | 0.6708 (-0.1) | 53.1 / 156 |
| k84 <= 0.12 (17.7 l = 1 only) | 0.1180 (-0.8) | 3.129 (+1.2) | 0.988 (+1.0) | 0.6714 (+0.1) | 48.8 / 150 |
| sigma1 = 25 only | 0.1178 (-0.85) | 3.131 (+1.25) | 0.989 (+1.1) | 0.6712 (0.0) | 48.8 / 149 |
| k84 <= 0.14, no AP | 0.1209 (0.0) | 3.084 (+0.4) | 0.971 (+0.3) | 0.6716 (+0.1) | 54.8 / 156 |

The bias shrinks with the cut but does not go away: ln A_s stays at +1.2 sigma and n_s at +1 sigma even with a single
sigma1 = 17.7 multipole, or with no sigma1 = 17.7 at all (profile shifts run ~0.3 sigma above chain means: P_l AP
ln A_s +0.8 profile, +0.52 chain). Without AP the same cut is unbiased. No AP chain run (none of the cuts is
unbiased). The pull is S21m's at every scale tested once P_l carries AP; the next suspects are the shared nuisances
(bG2 / bGamma3, which S21m pins far from P_l's values) and S21m's amplitude dependence (ln A_s), not the first-layer
reach.

#### Shared bG2 / bGamma3 are not the cause either (2026-10-21)

Baseline with AP (all of S21m, ap3_v5, V = 60, n_s free), profiles; shifts from the truth in profile sigma:

| variant | omega_cdm | ln A_s | n_s | h | chi2 / dof |
|---|---|---|---|---|---|
| shared biases (the chain's model; chain mean) | 0.1164 (-1.40) | 3.126 (+1.19) | 0.998 (+1.56) | 0.6705 (-0.17) | 58.7 / 188 |
| S21m with its own bG2_s21, bGamma3_s21 (same priors; P_l and X keep bG2, bGamma3) | 0.1163 (-1.4) | 3.137 (+1.4) | 0.998 (+1.6) | 0.6703 (-0.2) | 60.9 / 186 |
| bG2 = -0.271, bGamma3 = 0.520 fixed (coevolution at b1 = 1.95) | 0.1150 (-2.0) | 3.119 (+1.1) | 0.993 (+1.7) | 0.6701 (-0.3) | 67.8 / 190 |

Splitting the biases changes nothing: S21m alone fits bG2_s21 = -0.43 +- 0.07 and leaves bGamma3_s21 free (0.11 +- 0.99),
while P_l + X fit bG2 = -0.42 +- 0.07, bGamma3 = 1.00 +- 0.18. So the two sides agree on bG2, and bGamma3 is set by
P_l + X, not S21m (correcting the previous subsection: the P_l + S21m truth fit's bGamma3 = 1.05 came from P_l, not S21m).
Coevolution values make it worse. The cosmology pull of S21m with AP is not routed through the shared second- and
third-order biases. Remaining suspects: S21m's amplitude dependence (ln A_s, the most persistent shift) and its
coupling to h / omega_cdm through AP of P_l (the bias appears only when P_l carries AP); profile scripts in this
session's scratchpad (split_bias.py patches S21mTheory to own bG2 / bGamma3).

#### The AP bias lives on the ln A_s - n_s - omega_cdm ridge (2026-10-21)

Baseline with AP (ap3_v5, V = 60), profiles with one parameter fixed at the truth (shifts in profile sigma):

| variant | omega_cdm | ln A_s | n_s | h | chi2 / dof |
|---|---|---|---|---|---|
| all four free (chain mean; profile chi2 ~ 58.7) | 0.1164 (-1.40) | 3.126 (+1.19) | 0.998 (+1.56) | 0.6705 (-0.17) | 58.7 / 188 |
| ln A_s fixed | 0.1199 +- 0.0019 (-0.5) | - | 0.975 +- 0.015 (+0.85) | 0.6728 +- 0.0031 (+0.55) | 62.6 / 189 |
| n_s fixed | 0.1205 +- 0.0018 (-0.2) | 3.072 +- 0.035 (+0.3) | - | 0.6729 +- 0.0032 (+0.55) | 63.2 / 189 |
| ln A_s fixed, no AP | 0.1211 +- 0.0020 (+0.1) | - | 0.969 +- 0.015 (+0.45) | 0.6719 +- 0.0030 (+0.3) | 63.5 / 189 |
| P_l alone, AP, ln A_s fixed | 0.1222 +- 0.0022 (+0.6) | - | 0.945 +- 0.017 (-1.0) | 0.6725 +- 0.0033 (+0.4) | 19.8 / 33 |

Fixing either ln A_s or n_s removes most of the bias (omega_cdm within 0.5 sigma of the truth): the shift is a slide along
one direction, ln A_s up / n_s up / omega_cdm down, which the four-parameter fit takes for Delta chi2 ~ 4 (62.6 or 63.2 ->
58.7). The data, through S21m, prefer that direction at about 2 sigma: S21m's dependence on amplitude and tilt (which
the ridge trades against each other at S21m's scales) is slightly off, at the level the tree-level amplitude dependence
already failed elsewhere (sigma = 12.5 bands, halo E(q)). AP makes it visible because P_l's own pull along the ridge
(opposite sign) goes away. Next: compare d S21m / d ln A_s, d n_s, d omega_m with the Quijote derivative boxes block by
block (sigma1 = 17.7, 25), which tests the amplitude dependence directly.

#### Derivative boxes: S21m's amplitude and omega_m response is right; its offset sits on the ridge (2026-10-21)

Local Quijote derivative sets (halos, same measurement, 200 boxes each, seeds 0-199 shared by + and -): s8_p / s8_m
(sigma8 0.849 / 0.819: ln A_s 3.0963 / 3.0243) and Om_p / Om_m (Omega_m 0.3275 / 0.3075 at fixed sigma8: omega_cdm
0.12543 / 0.11642, ln A_s 3.0156 / 3.1081). Profiles of each set's mean with the fiducial covariance at V = 60
(`--covariance-data-dir`, ap3_v5, n_s free); AP only for s8 (same geometry as the fiducial). The + minus - differences
cancel the common seed noise:

| fit | s8: Delta ln A_s (true +0.0720) | s8: Delta n_s, Delta omega_cdm (true 0) | Om: Delta omega_cdm (true +0.0090) | Om: Delta ln A_s (true -0.0925) |
|---|---|---|---|---|
| P_l, no AP | +0.0598 | -0.0063, +0.0009 | +0.0102 | -0.1143 |
| P_l, AP | +0.0675 | -0.0025, +0.0003 | | |
| P_l + S21m, no AP | +0.0609 | -0.0045, +0.0006 | +0.0097 | -0.1117 |
| P_l + S21m, AP | +0.0670 | -0.0015, +0.0001 | | |
| baseline, no AP | +0.0565 | -0.0052, +0.0008 | +0.0096 | -0.1037 |
| baseline, AP | +0.0622 | -0.0020, +0.0003 | | |

Every combination recovers the amplitude and omega_m steps to 80-95% / 105-125% (about 0.1-0.3 sigma of V = 60 over
1-2 sigma steps), and adding S21m does not degrade them (P_l + S21m with AP = P_l with AP). So S21m's first derivatives
in ln A_s and omega_cdm are right at this precision: the AP bias is not a wrong amplitude response. The absolute fits
of the four sets (noisier: 200 shared seeds) repeat the fiducial pattern, and S21m pushes along the same ridge in every
set: e.g. s8_p with AP, P_l -> P_l + S21m: ln A_s +0.067 -> +0.100, n_s +0.014 -> +0.039, omega_cdm -0.0038 -> -0.0063.
Reading: S21m has an offset at the truth (a level, not a slope) that projects on the ln A_s - n_s - omega_cdm ridge; the
n_s response is untested (no n_s derivative boxes locally; Perlmutter down until 2026-10-22). Next: the n_s boxes, and
the S21m residual at the truth projected on the ridge direction, block by block.

#### Where the ridge offset sits in the data vector (2026-10-21)

Baseline with AP: residual r = data - theory at the truth (cosmology fixed, nuisances re-profiled, chi2 62.4) and the
ridge Delta t = theory(free best fit, chi2 58.7) - theory(truth). Matched-filter projection of r on the ridge,
r^T C^-1 Delta t / |Delta t|_C = 1.92 sigma (ridge length 1.76 sigma), split per entry as r_i (C^-1 Delta t)_i / |Delta t|_C:

| block | share [sigma] | rms r / err | rms Delta t / err |
|---|---|---|---|
| P_0 | +0.86 | 0.59 | 0.36 |
| P_2 | +0.62 | 0.76 | 0.17 |
| P_4 | -0.11 | 0.86 | 0.05 |
| S21m sigma1 17.7, l = 1 / 2 / 3 / 4 | +0.02 / -0.12 / -0.18 / +0.46 | 0.22 / 0.50 / 0.61 / 0.58 | ~0.1 |
| S21m sigma1 25, l = 1 / 2 / 3 / 4 | -0.00 / -0.06 / +0.14 / +0.31 | 0.17 / 0.38 / 0.51 / 0.59 | ~0.1 |
| X (all) | -0.01 | <= 0.37 | <= 0.09 |

S21m carries +0.56 of the 1.92 sigma, P_l +1.37, X 0. Within S21m no entry dominates and the m-blocks cancel: by
(sigma1, l, |m1|) the largest are 17.7 l = 3 |m1| = 3 (+0.62), l = 3 |m1| = 2 (-0.45), l = 2 |m1| = 2 (-0.35), l = 4
|m1| = 3 (+0.30); top entries S21m_j5_j9_l3_m3_m3 +0.47 (r = +0.65 err), _l2_m2_m2 -0.28, _l3_m0_m3 -0.26 (r = +1.34
err). Along the ridge the S21m prediction barely moves (|Delta t| ~ 0.1 err per entry): the slide happens in P_0 / P_2,
whose residuals at the truth (with the nuisances the joint fit sets) point along it. So no S21m entry "carries" an offset;
S21m acts through the nuisances it fixes for P_l. The shared ones that move most between P_l alone and the joint fit at
the truth are bG2 (+0.07 +- 0.28 -> -0.42; X and S21m agree on it) and, through the P_l bG2 - bGamma3 degeneracy, bGamma3
(-0.34 -> +1.0), plus b2 (-0.46 -> -0.66); splitting bG2 / bGamma3 (previous subsection) does not remove the bias, so
the question becomes which P_l nuisances, at the values the WST demands, put P_l's residual on the ridge.

### Full-WST halo vector: q = 2 bands in place of P_l (2026-10-21, stage A)

Plan: `~/.claude/plans/can-we-try-a-refactored-grove.md`. With ap3_v5 (no sigma = 12.5 or l >= 5 cumulants) the bands are
sigma >= 17.7, l <= 4 (k84 <= 0.158, near the "0.15" rung of the reach ladder; 75 m-resolved entries, 47 principal
components after the 1e-4 cut). Profiles, n_s free, V = 60, runs `outputs/inference/rsd/q2/{wstA,wstAnoX,wstA25,
wstAnoX25,pk015}_{ap,noap}_V60` (the v5 cumulant correction cache was built on the way). Profile best fits minus the truth
(the Minuit errors of the band fits are not usable, e.g. sigma(omega_cdm) = 0.0007; for scale, the v5 baseline chain
widths are 0.0034 / 0.055 / 0.023 / 0.0043):

| vector | AP | Delta omega_cdm | Delta ln A_s | Delta n_s | Delta h | chi2 / ndata |
|---|---|---|---|---|---|---|
| bands + S21m(17.7, 25) + X | yes | -0.0042 | +0.066 | +0.026 | -0.0038 | 48.4 / 235 |
| bands + S21m(17.7, 25) + X | no | -0.0021 | +0.033 | +0.010 | -0.0046 | |
| bands + S21m(17.7, 25) | yes | -0.0062 | +0.107 | +0.032 | -0.0053 | 22.4 / 155 |
| bands + S21m(17.7, 25) | no | -0.0047 | +0.083 | +0.021 | -0.0073 | 24.1 / 155 |
| bands + S21m(25) + X | yes | -0.0033 | +0.061 | +0.022 | -0.0034 | 34.5 / 181 |
| bands + S21m(25) + X | no | +0.0033 | -0.015 | -0.022 | +0.0010 | 32.3 / 181 |
| bands + S21m(25) | yes | -0.0025 | +0.069 | +0.009 | -0.0038 | 10.8 / 101 |
| bands + S21m(25) | no | -0.0009 | +0.046 | -0.001 | -0.0033 | 10.7 / 101 |
| P_l to 0.15 | yes | -0.0015 | +0.041 | 0.000 | -0.0017 | 3.2 / 36 |
| P_l to 0.15 | no | -0.0004 | +0.025 | -0.008 | -0.0012 | 3.1 / 36 |
| (v5 baseline P_l to 0.2 + S21m + X, AP, chain mean) | yes | -0.0045 | +0.065 | +0.035 | -0.0006 | 58.7 / 236 |

Replacing P_l by the bands does not remove the AP bias: with AP every WST vector slides along the same ridge (omega_cdm
-0.0025 to -0.006, ln A_s +0.06 to +0.11, n_s up), about 1 to 2 v5 sigma, as the P_l baseline did; the sigma1 = 17.7 S21m
makes it worse (bands + S21m: -0.0062 / +0.107 with it, -0.0025 / +0.069 without), and X pulls back toward the truth.
Without AP the vectors are biased too, but less, and not consistently in sign (bands + S21m(25) + X: +0.0033 / -0.015).
So the P_l / WST nuisance tension was not the cause: the bands, which share the first-layer model with S21m, show the
same AP direction. No chain run (none within ~0.5 sigma). Stage B (the union emulator, bands to k84 = 0.18 with sigma =
12.5) is pending on Perlmutter; given that the AP direction is common to every WST block, it is not expected to change
the conclusion, and is the user's call.

#### P_l alone at the joint fit's nuisance values does not slide (2026-10-21)

P_l alone with AP (ap3_v5, V = 60, n_s free), profiles with shared nuisances fixed one at a time at the values of the
truth-fixed AP baseline fit (b1 1.956, b2 -0.618, bG2 -0.451, bGamma3 1.029, alpha0 -0.017, alpha2 -12.50, cubic offsets
-0.224 / 0.034 / 0.227). Ridge direction: omega_cdm down, ln A_s up, n_s up; the joint AP baseline sits at 0.1164 / 3.126
/ 0.998 (chain mean).

| fixed | chi2 | omega_cdm | ln A_s | n_s | h |
|---|---|---|---|---|---|
| none | 17.0 | 0.1194 | 3.109 | 0.9625 | 0.6706 |
| b1 | 19.1 | 0.1224 | 3.040 | 0.9492 | 0.6726 |
| b2 | 17.0 | 0.1194 | 3.109 | 0.9622 | 0.6706 |
| bG2 | 18.5 | 0.1194 | 3.117 | 0.9679 | 0.6715 |
| bGamma3 | 18.4 | 0.1191 | 3.120 | 0.9721 | 0.6712 |
| alpha0 | 17.6 | 0.1205 | 3.096 | 0.9575 | 0.6717 |
| alpha2 | 17.0 | 0.1193 | 3.109 | 0.9631 | 0.6706 |
| cubic offsets | 17.0 | 0.1194 | 3.109 | 0.9625 | 0.6706 |
| bG2 + bGamma3 | 18.3 | 0.1190 | 3.122 | 0.9740 | 0.6710 |
| all | 21.6 | 0.1192 | 3.075 | 0.9738 | 0.6716 |

No shared nuisance moves P_l along the ridge: bG2 / bGamma3 nudge n_s up by <= 0.4 sigma (ln A_s +0.2 sigma) with
omega_cdm unchanged, b1 and alpha0 push the other way, b2 / alpha2 / cubic offsets do nothing; with all of them fixed
P_l stays at omega_cdm 0.1192 (-0.4 sigma), far from the joint 0.1164. So the slide is not P_l dragged by the nuisances
the WST fixes. It needs S21m's own cosmology dependence acting jointly: per entry the S21m prediction moves little along
the ridge (~0.1 err) but coherently over ~110 entries, and the joint fit trades that against P_l's non-shared
counterterms (c0_pk / c2_pk / c4_pk, which re-adjust along the ridge). The per-entry projection (previous subsection)
credits P_0 / P_2 because the theory change lives there, not because P_l prefers the ridge. Remaining test: the n_s
derivative boxes (Perlmutter), since S21m's amplitude and omega_m responses passed.

### What the AP baseline must give up to be unbiased (2026-10-22)

NUTS chains with AP, n_s free, V = 60, 5000 boxes, ap3_v5 (all R - 1 <= 0.03, ESS >= 335; the first P_l + X run had a
stuck chain, kept as `pud/px05mns_ap_chain_n5000_V60_stuck`, rerun with --seed 7). Mean +- sigma (shift from the truth;
change of sigma against P_l with AP); plot `outputs/plots/triangle_ap_cuts_n5000_V60.png`:

| vector | omega_cdm | ln A_s | n_s | h | chi2 / ndata |
|---|---|---|---|---|---|
| P_l (`pns_ap`) | 0.1204 +- 0.0040 (-0.12) | 3.092 +- 0.060 (+0.52) | 0.956 +- 0.027 (-0.23) | 0.6713 +- 0.0040 (+0.05) | 17.0 / 48 |
| P_l + X (`px05mns_ap`) | 0.1202 +- 0.0039 (-0.20; -4%) | 3.095 +- 0.056 (+0.60; -7%) | 0.970 +- 0.025 (+0.31; -10%) | 0.6726 +- 0.0040 (+0.37; 0%) | 37.2 / 128 |
| P_l + S21m(25) + X (`sx05mns25_ap`) | 0.1199 +- 0.0040 (-0.26; -1%) | 3.102 +- 0.057 (+0.72; -5%) | 0.976 +- 0.026 (+0.51; -6%) | 0.6727 +- 0.0040 (+0.41; +1%) | 46.9 / 182 |
| P_l + S21m(17.7, 25) + X (`sx05mns_ap`) | 0.1164 +- 0.0032 (-1.40; -20%) | 3.126 +- 0.055 (+1.19; -9%) | 0.998 +- 0.023 (+1.56; -17%) | 0.6705 +- 0.0036 (-0.17; -10%) | 58.7 / 236 |

Answer: with AP, the gain over P_l and the bias come from the same block. Dropping S21m at sigma1 = 17.7 makes the vector
unbiased (all shifts <= 0.72 sigma, at the level of P_l alone, whose ln A_s is already +0.52), but it then gains only
1-6% (with X alone 0-10%): the sigma1 = 25 S21m adds nothing once AP is on. Intermediate cuts of the 17.7 block (k84 <=
0.12-0.15, profiles above) stay biased by ~0.8-1.4 sigma in omega_cdm. There is no cheap sacrifice: the cheapest unbiased
AP vector is P_l + X (-4 / -7 / -10 / 0%), and every configuration that keeps the 17-23% gain keeps the ~1.5 sigma bias.
Without AP the full baseline is unbiased with -23 / -14 / -21 / -20% (by the cancellation of the S21m and P_l pulls).

### Diagnosis of the AP baseline bias: the short-mode spectrum inside S21m (2026-10-22)

Scratch patches of `S21mTheory` (this session's scratchpad: `s21_pieces.py`, `decouple.py`, `shortmode_check.py`),
profiles of the AP baseline (P_l + S21m(17.7, 25) + X, ap3_v5, n_s free, V = 60); best fit omega_cdm / ln A_s / n_s / h,
reference (chain mean) 0.1164 / 3.126 / 0.998 / 0.6705, truth 0.1209 / 3.061 / 0.962 / 0.6711:

| S21m variant | omega_cdm | ln A_s | n_s | h | note |
|---|---|---|---|---|---|
| free response amplitude (1 + A_j) per scale | 0.1161 | 3.138 | 0.999 | 0.6702 | A_j5 = 0.001 +- 0.031, A_j6 = 0.003 +- 0.043 |
| Edgeworth factors off (+ A_j) | 0.1164 | 3.134 | 0.998 | 0.6704 | |
| own FoG counterterm for sigma1 = 25 | 0.1162 | 3.133 | 1.000 | 0.6703 | dc_v = -0.18 +- 0.20 |
| free response amplitude per field (28) | 0.1183 | 3.124 | 0.983 | 0.6712 | all A within 1 sigma of 0; chi2 58.7 -> 43.9 |
| own b1, b2 | 0.1164 | 3.135 | 0.997 | 0.6702 | b1_s21 = 1.85 +- 0.13 |
| own bG2 | 0.1162 | 3.138 | 0.999 | 0.6703 | |
| own alpha0 | 0.1183 | 3.116 | 0.986 | 0.6718 | alpha0_s21 = 0.15 +- 0.12 |
| own alpha0 + bG2, or b1 + alpha0 | 0.1183 | 3.116 | 0.986 | 0.6718 | same as alpha0 alone |
| **own b1, b2, bG2, bGamma3, alpha0, alpha2** | **0.1190** | **3.106** | **0.980** | **0.6717** | b1 1.74 +- 0.13, b2 -0.18 +- 0.24, bG2 -0.26 +- 0.10, alpha0 0.27 +- 0.15 |
| S21m alone | 0.124 +- 0.023 | 2.99 +- 0.30 | 1.00 +- 0.18 | 0.7311 (bound) | chi2 7 / 64: no cosmology of its own |

The response amplitude of S21m is right to 3% overall and per field; the Edgeworth factors and the FoG split do nothing.
The bias is mediated by the nuisances S21m shares with P_l and X: with its own copies it drops to the P_l + X level
(-0.5 / +0.8 / +0.7 / +0.15 sigma), and the decoupled S21m asks for a lower b1 and 15-27% more shot noise; alpha0 alone
carries two thirds of it. The profile widths of the decoupled variants (0.0038 / 0.057 / 0.025 / 0.0039) are those of
P_l + X: the sigma1 = 17.7 gain over P_l with AP ran through alpha0 (and b1), i.e. S21m was pinning the stochastic
amplitude of P_l, at a biased value.

**Why.** S21m's short modes (p ~ 0.05-0.3 h/Mpc at sigma1 = 17.7) carry the tree halo spectrum P_L Z1^2 (1 - c_v k_z^2 /
k_NL^2)^2 + N (1 + alpha0), while P_l uses the one-loop P_s with c0, c2, c4 and alpha2 k^2 mu^2. At the AP chain's
best fit (b1 1.92, alpha0 0.10, alpha2 -13, c_v 1.24) the two differ by -1 to +2% at p = 0.05 but by +15% / +24% (mu =
0.85 / 1) at p = 0.1, +42% / +73% at 0.15 and +75% / +142% at 0.2 (`shortmode_check.py`): the per-leg counterterm is
(1 - 0.55)^2 = 0.2 at p = 0.2 along the line of sight, far outside its regime, and S21m has no alpha2 (-26% of N there).
Fitting the S21m form to the one-loop P_s under the sigma1 = 17.7 filter weights gives b1_eff 1.85, alpha0_eff +0.23,
c_v_eff ~ 0 (rms 1.3%; with c_v held at 1.24: b1 1.77, alpha0 +0.84, rms 11%) -- the b1 down / alpha0 up that the
decoupled S21m asked for. So the tree + c_v + Poisson short-mode model of S21m is inconsistent with the one-loop P_s of
the same fit at the sigma1 = 17.7 scales; S21m is insensitive to the amplitude of that spectrum but not to its shape
(P_L Z1^2 against N across p and mu_p), and resolves the inconsistency through b1 and alpha0. Without AP the resulting
cosmology pull cancelled against P_l's own; with AP it does not.

**Fix (not done).** Use the one-loop P_s(p, mu) of the P_l model (counterterms, alpha0, alpha2 included) as the short-mode
spectrum of the modulus response and noise in `second_layer.ModulusSpectra` (the response-function approximation of the
squeezed bispectrum, B ~ R P_L(K) P_nl(p)), in place of P_L Z1^2 with the c_v legs; the norms likewise. The P_s grid of
the basis already covers p <= 0.3 (the sigma1 = 17.7 filters are e^-28 there). It changes the S21m basis columns (the
biases enter through P_s, quadratic in beta, plus the counterterm and stochastic rows) and needs a new emulator; the X
response (`covariance_rsd.ModulusCrossRSD`) has the same short-mode model but survives through its marginalised cU / eU
(eU at sigma = 17.7 is 14-23 sigma from zero, of order N -- the same mismatch). Until then the sigma1 = 17.7 S21m cannot
be used with shared b1 / alpha0: the AP options are P_l + X (unbiased, -4 / -7 / -10 / 0%) or the decoupled S21m (no gain).

Confirmation (same day): the no-AP baseline with S21m's own alpha0 lands at P_l's own no-AP spot (0.1231 / 3.058 / 0.958 /
0.6744, widths 0.0042 / 0.057 / 0.027 / 0.0052 against P_l 0.0045 / 0.061 / 0.028 / 0.0055; alpha0_s21 = 0.11 +- 0.12):
the no-AP gain of 14-23% also ran through alpha0, and its unbiasedness was the cancellation. Chain of the fully decoupled
AP variant (`pud/sx05mns_dec_ap_chain_n5000_V60`, R - 1 < 0.03, ESS 327): omega_cdm 0.1201 +- 0.0041 (-0.19 sigma),
ln A_s 3.082 +- 0.060 (+0.35), n_s 0.974 +- 0.026 (+0.46), h 0.6726 +- 0.0041 (+0.36); widths within 2-6% of P_l with
AP, i.e. no gain; S21m's own nuisances b1 1.79 +- 0.15, b2 -0.15 +- 0.25, bG2 -0.24 +- 0.11, alpha0 +0.30 +- 0.16
against P_l + X's b1 1.945 +- 0.044, bG2 -0.44 +- 0.08, alpha0 -0.06 +- 0.20.

### One-loop short modes inside S21m: model, validation at the truth, AP chains (2026-10-23)

The fix proposed above, implemented (commit 4aa0a79): `second_layer.LoopModulusSpectra` (`S21mBasis(loop=grid)`,
`Basis(s21_loop=True)`, `ModelSpec.s21_loop`, `train_emulator.py --s21-loop`). The short-mode spectrum of every leg
of the modulus response, of the Gaussian-chaos noise and of the first-layer norms is the one-loop P_s of the P_l model,
interpolated from the basis's P_s grid (linear in ln k, Lagrange in mu^2 through its Gauss nodes; clamped at the
grid's kmax = 0.3, where the sigma >= 17.6 filters are < 1e-6 of their peak), written as a sum of pieces with
nuisance weights: the 15 degree-2 bias monomials of tree + loop, the counterterm shapes (c0, c2, c4), 1 (N (1 +
alpha0)) and k^2 mu^2 (N alpha2). The noise (products of two spectra) and the norms are exact in the pieces. The
response keeps the tree long-mode coupling: the tree part of the beta piece responds exactly (2 Z2 Z1 P_L, P_L the
IR-resummed tree row), the loop and counterterm pieces through the response-function approximation dP/d delta_L ~
R0(q, K) P(q) with R0 = 2 Z2/Z1 of a *linearly* biased tracer at a reference b1 (= 2; a ratio, weakly dependent on b1:
exact kernels would make the response columns a degree-3 polynomial divided by Z1(q), not separable; the b2 / bG2
responses of these next-to-leading pieces are dropped), and the stochastic pieces as the Poisson term, Z1(K) times the
pair's stochastic power. Columns: the symmetric products of 6 response pieces (266, times P_L(K)) + of 6 spectrum
pieces (160) + 20 norms = 446 per entry (`layout.s21_loop_blocks`), contracted by `calculators.s21m_loop_terms` with
P_l's counterterms (`c0_pk`... when P_l is fitted, else the S1m ones) and alpha0, alpha2. **c_v is dropped from S21m**
(the one-loop counterterms play its role; it stays for the S1m cumulants). The X response is mirrored
(`ModulusCrossRSD.loop_spectra`; `response_taylor(loop=)` is a table in the biases, c0, c2, c4, alpha0, alpha2, no
sigma_v; `ResponseTheory` reads them). Cost: one exact AP basis (ap_order 3) 42 s locally, about the old one.

**Validation at the true cosmology (exact basis, no emulator; S21m(17.7, 25 -> 70.7, q = 2) alone, V = 60 errors of
the 5000 boxes, 108 entries, free noise amplitudes and W; scratch `validate_truth.py`).** Control: the old model at
the AP chain's own best fit (cosmology and nuisances) reproduces its S21m fit, chi2 = 19 / 108. With the shared
nuisances fixed at P_l's profile at the truth (b1 1.948 +- 0.012, b2 -0.46 +- 0.5, bG2 0.07 +- 0.4, bGamma3 -0.3 +-
1.1, alpha0 0.02 +- 0.17, alpha2 -13.2, c0 / c2 / c4 = 1.2 / -2.3 / 4.0) neither model fits (old chi2 136, one-loop
225): S21m wants bG2 ~ -0.35 +- 0.1, which P_l leaves free (the P_l-only AP chain's bG2 = +0.07, bGamma3 = -0.4 are far
from the joint values -0.42, +1.0). The decoupling test proper, S21m alone with its own b1, b2, bG2, bGamma3, alpha0,
alpha2 (c at P_l's):

| model | b1 | b2 | bG2 | bGamma3 | alpha0 | chi2 | dchi2 of pinning b1, alpha0 at P_l's |
|---|---|---|---|---|---|---|---|
| old (tree + c_v) | 1.58 +- 0.13 (-2.8 sigma from 1.948) | -0.16 +- 0.21 | -0.31 +- 0.10 | 0.09 +- 1.0 (prior) | -0.02 +- 0.13 | 12.4 | 6.0 (bG2 -> -0.41) |
| one-loop | 2.27 +- 0.26 (+1.2 sigma) | -0.18 +- 0.24 | -0.36 +- 0.11 | 1.30 +- 0.32 | 0.04 +- 0.25 | 11.9 | 2.2 (bG2 -> -0.31, bGamma3 1.1) |

So at the truth the one-loop short modes bring S21m's own b1 from 2.8 sigma below P_l's to 1.2 sigma above (with a
twice larger error: bGamma3 now enters through P13 and is degenerate with b1), alpha0 agrees in both, and the cost of
sharing (b1, alpha0) drops from 6.0 to 2.2. The reference b1 of the response kernel does not matter (1.5 / 2 / 2.5:
b1 = 2.25 / 2.27 / 2.28, chi2 11.3 / 11.9 / 12.3). Direct check of the response on the box X = P_Ud / <U> at the AP
chain's best fit (scratch `x_check.py`, 80 points, diagonal chi2 with a free constant and k^2 P_L per field as the
fit marginalises): old (sigma_v damping) 59 -> 9, one-loop 194 -> 5; the one-loop response needs a constant eU ~
+0.2 to +0.85 N at sigma = 17.7 (+0.05 N at 25), i.e. a missing white term in P_Ud of order the Poisson three-point
pieces N [P(p) + P(q)] (neither model has them; marginalised in the fits).

**Fits of the v5 baseline with the one-loop short modes (emulator `rsd_biased_basis_taylor_4p_ir_fog_cubic_ap3_v5loop.h5`,
trained on Perlmutter, S21m validation 5e-4 max; n_s free, V = 60, 5000 boxes).** The chains were stopped once the
profiles showed the slide; profile best fits (shift in units of the P_l chain sigma, AP or no AP; truth 0.1209 / 3.061
/ 0.962 / 0.6711). X is the one-loop mirror unless noted:

| variant | omega_cdm | ln A_s | n_s | h | b1 | bG2 | bGamma3 | alpha0 | chi2 / 236 |
|---|---|---|---|---|---|---|---|---|---|
| old model, AP (chain mean, for reference) | 0.1164 (-1.1) | 3.126 (+1.1) | 0.998 (+1.3) | 0.6705 (-0.2) | 1.92 | -0.42 | 1.02 | 0.10 | 58.7 |
| one-loop S21m + X, AP | 0.1131 (-1.9) | 3.165 (+1.7) | 1.024 (+2.3) | 0.6681 (-0.8) | 1.92 | -0.25 | 0.72 | 0.21 | 55.8 |
| one-loop S21m + X, no AP | 0.1152 (-1.3) | 3.132 (+1.2) | 1.013 (+1.8) | 0.6654 (-1.0) | 1.94 | -0.26 | 0.78 | 0.15 | 59.8 |
| own bGamma3 in S21m, AP | 0.1130 | 3.165 | 1.024 | 0.6681 | 1.92 | -0.24 | 0.69 (0.73) | 0.21 | 56.1 |
| own bGamma3 in S21m and X, AP | 0.1150 | 3.138 | 1.008 | 0.6696 | 1.94 | -0.24 | 0.61 (0.92) | 0.06 | 55.0 |
| own bG2, bGamma3 in S21m and X, AP | 0.1165 | 3.131 | 0.989 | 0.6689 | 1.91 | 0.17 (-0.25) | -0.58 (0.91) | 0.17 | 49.8 |
| P_l + one-loop X only, AP (128 points) | 0.1150 | 3.163 | 1.004 | 0.6690 | 1.90 | -0.30 | 0.69 | 0.16 | 34.3 / 128 |
| one-loop S21m + **tree X**, AP | 0.1166 (-1.1) | 3.114 (+0.9) | 1.004 (+1.6) | 0.6709 (0.0) | 1.95 | -0.37 | 0.96 | -0.00 | 71.3 |
| one-loop S21m + **tree X**, no AP | 0.1196 (-0.3) | 3.072 (+0.2) | 0.988 (+0.9) | 0.6705 (-0.1) | 1.97 | -0.38 | 1.03 | -0.06 | 74.0 |

(own copies in parentheses.) **Result: the one-loop short modes do not remove the AP bias.** (1) The one-loop X mirror
is itself biased: P_l + one-loop X alone slides to 0.1150 / 3.163 / 1.004, where P_l + tree X was unbiased (0.1202 /
3.095 / 0.970, chain). Its Taylor table is accurate (quadratic terms < 8% of the linear ones over 5 steps), so this is
the model: the response needs a constant eU of +0.2 to +0.85 N at sigma = 17.7 (above), and the marginalised eU, cU
cannot absorb a response that now moves with P_l's bG2, bGamma3, counterterms and alpha0. Keep the tree X
(`response_taylor` without `loop`) with any S21m model. (2) With the tree X, the one-loop S21m makes the shared b1 and
alpha0 agree with P_l (1.95, 0.00, as the validation at the truth predicted), but the AP slide is the old one (-1.1 /
+0.9 / +1.6 sigma; chi2 +15 against the one-loop X, the two blocks now disagreeing), and S21m still pins bG2 ~ -0.37
and bGamma3 ~ 1.0, far from P_l alone at the truth (+0.07, -0.3). So the b1 / alpha0 inconsistency was real but was
not what carries the bias; the remaining channels are the S21m bG2 / bGamma3 responses and the ridge itself. Decoupling
bG2 and bGamma3 does not remove it either (0.1165 / 3.131 / 0.989 with the one-loop X). The code stays (`--s21-loop`,
off by default); the baseline emulator and fits are unchanged.

**Which shared nuisance carries the slide (one-loop S21m, tree X, AP; S21m given its own copies, suffix _s21).**
Profile best fits, shift in units of the P_l AP chain sigma:

| S21m's own copies | omega_cdm | ln A_s | n_s | h | chi2 / 236 | S21m's own values |
|---|---|---|---|---|---|---|
| none | 0.1166 (-1.1) | 3.114 (+0.9) | 1.004 (+1.6) | 0.6709 (0.0) | 71.3 | |
| bG2, bGamma3 | 0.1169 (-1.0) | 3.109 (+0.8) | 1.000 (+1.4) | 0.6712 (0.0) | 70.5 | bG2 -0.37, bGamma3 1.05 |
| bG2, bGamma3, c0, c2, c4 | 0.1179 (-0.8) | 3.103 (+0.7) | 0.995 (+1.2) | 0.6717 (+0.1) | 65.9 | c0 69 +- 28, c2 38 +- 17 |
| b1, b2, alpha0, alpha2 | 0.1153 (-1.4) | 3.148 (+1.5) | 1.010 (+1.8) | 0.6691 (-0.5) | 59.1 | b1 2.45 +- 0.23 |
| all (biases, alphas, counterterms) | 0.1191 (-0.45) | 3.102 (+0.7) | 0.980 (+0.7) | 0.6718 (+0.2) | 55.6 | b1 2.34 +- 0.24, c0 46 +- 44 |

No single channel: whichever subset stays shared transmits the slide, and only full decoupling removes it (to the
level of the old fully decoupled chain, 0.1201 / 3.082 / 0.974, profile widths 0.0040 / 0.061 / 0.026 = P_l + X: no
gain). Even fully decoupled, S21m asks for nuisances P_l rejects (b1 2.34 against 1.93, counterterms 5-10x P_l's),
as it did at the truth (b1 2.27, bGamma3 1.3). So the one-loop S21m at sigma1 = 17.7 still needs a short-mode /
response region inconsistent with P_l's at the true cosmology; its 30 noise amplitudes and W absorb the rest. The
inconsistency is not in the short-mode spectrum any more (that is P_l's own now) but in what multiplies it: the
response kernel (tree long-mode coupling, approximate loop response), the Gaussian-chaos noise (no connected
noise), or the Gaussian second layer. Every gain of S21m(17.7) over P_l + X with AP runs through that inconsistency.

### The fix: one-loop short modes + K^2 noise of the moduli; unbiased AP baseline with the gain (2026-10-23, night)

**Diagnosis from the measured modulus spectra** (P_UU_m, P_Ud_m of the 5000 boxes, scratch `puu_shape.py`,
`pud_ratio.py`). (1) At P_l's nuisances the model response is 15-30% below the measured P_Ud, flat in K (|m| = 0:
x1.15, |m| = l: x1.25-1.40), tree and one-loop alike; at the joint fit's nuisances (bG2 -0.37, shared by S21m and
X, weakly constrained by P_l) it matches to 1-3%. X hides such an amplitude in its marginalised k^2 P_L and constant
terms over its 2-4 K bins; a per-field response amplitude (S21m only, or shared with X) does not remove the bias. (2)
The **shape** of P_UU / <U>^2 is not described even at the joint fit's nuisances with a free (1 + a) g + W per field:
for |m| >= 1 the monopole of the data rises above the model with K over 0.02-0.05 h/Mpc and the quadrupole falls below
it. Across the l = 1-4 second-layer filters (peaks at K = sqrt(l) / 70.7) this acts as a tilt in K, which the fit
produces by raising n_s along the ln A_s - n_s - omega_cdm ridge. The connected noise of the moduli is analytic at low
K: adding W2 K^2 + W3 K^2 mu^2 per field absorbs a third of the shape residual (chi2 3772 -> 2538 / 756).

**Model:** `--s21-k2` (with `--s21-white`): W2_j K^2 + W3_j K^2 mu_K^2 per first-layer scale in P_UU / <U>^2, next to
W_j (`NuisanceParameters.white(k2=True)`, `S21mTheory(white_k2=True)`; lattice sums at q = 1 like W; 4 parameters).
With the one-loop S21m emulator (`ap3_v5loop`, `--s21-loop` training) and the **tree X** (`--pud-loop` off, the
default: the one-loop X is itself biased). Profiles: the old tree S21m with the same K^2 terms stays biased (-1.0 /
+1.2 / +1.1 sigma with AP), so both pieces are needed. W3_j6 is 2.7 sigma from zero.

**Chains** (NUTS, n_s free, V = 60, 5000 boxes, R - 1 = 0.01; `pud/lsx05k2ns_{ap,}_chain_n5000_V60`; plots
`outputs/plots/triangle_k2noise_{ap,noap}_n5000_V60.png`). Mean +- sigma (shift from the truth in sigma; change of
sigma against P_l with the same AP setting):

| vector | omega_cdm | ln A_s | n_s | h | chi2 / ndata |
|---|---|---|---|---|---|
| P_l (AP) | 0.1204 +- 0.0040 (-0.12) | 3.092 +- 0.060 (+0.52) | 0.956 +- 0.027 (-0.23) | 0.6713 +- 0.0040 (+0.05) | 17.0 / 48 |
| P_l + X (AP) | 0.1202 +- 0.0039 (-0.20; -4%) | 3.095 +- 0.056 (+0.60; -7%) | 0.970 +- 0.025 (+0.31; -10%) | 0.6726 +- 0.0040 (+0.37; 0%) | 37.2 / 128 |
| old baseline (AP) | 0.1164 +- 0.0032 (-1.40; -20%) | 3.126 +- 0.055 (+1.19; -9%) | 0.998 +- 0.023 (+1.56; -17%) | 0.6705 +- 0.0036 (-0.17; -10%) | 58.7 / 236 |
| **new (AP)** | **0.1195 +- 0.0032 (-0.44; -20%)** | **3.080 +- 0.051 (+0.38; -15%)** | **0.976 +- 0.021 (+0.66; -22%)** | **0.6727 +- 0.0035 (+0.46; -11%)** | 58.4 / 236 |
| P_l (no AP) | 0.1239 +- 0.0045 (+0.66) | 3.052 +- 0.061 (-0.16) | 0.935 +- 0.028 (-0.96) | 0.6741 +- 0.0055 (+0.55) | 17.1 / 48 |
| old baseline (no AP) | 0.1203 +- 0.0034 (-0.19; -23%) | 3.072 +- 0.053 (+0.21; -14%) | 0.974 +- 0.023 (+0.53; -21%) | 0.6709 +- 0.0043 (-0.05; -20%) | 61.5 / 236 |
| **new (no AP)** | **0.1232 +- 0.0034 (+0.67; -24%)** | **3.034 +- 0.050 (-0.54; -18%)** | **0.956 +- 0.022 (-0.31; -24%)** | **0.6744 +- 0.0043 (+0.77; -21%)** | 59.2 / 236 |

**Result: the AP baseline is unbiased (all shifts <= 0.77 sigma, at the level of P_l alone) and keeps the gain over P_l:
-20 / -15 / -22 / -11% with AP, -24 / -18 / -24 / -21% without** (the old baseline's 9-20% with AP came with a 1.4-1.6
sigma bias). Proposed new baseline: `fit.py --emulator ..._ap3_v5loop.h5 --stats pk s21m pud --kmax 0.2 --q 2.0
--s21-min-scale 17.6 --s21-white --s21-k2 --pud-x 0.5 --pud-marg --cubic-biases relation --vary omega_cdm logA n_s h`
(+ `--ap`). Open: the physical origin of the K^2 noise (halo connected noise of the moduli) and of the X response
amplitude; the paper text and the survey pipeline still use the old model.

#### Correction (2026-10-24): the one-loop response had a factor-2 bug; K^2 noise terms; converged chains

**Bug.** In `LoopModulusSpectra` (4aa0a79) the loop and counterterm pieces of the short-mode spectrum responded as
2 R0 P instead of R0 P (R0 = 2 Z2/Z1, the response-function approximation, as the tree part does), in S21m and in the X
mirror. Fixed in d6fa9fc (`LoopModulusSpectra.loop_response = 1`). Everything above run with `--s21-loop` (the
validation at the truth, the v5loop emulator, the profiles and the decoupling tests) has the factor 2; the old baseline,
P_l and P_l + X are unaffected. At the no-AP best fit the fix moves S21m by 11 sigma whitened (up to 3.3 sigma per
entry, V = 60), but refitting the S21m-only noise terms at fixed cosmology leaves dchi2 = 2.3.

**K^2 noise terms (scratch wrapper `k2noise.py`, not in the package yet).** The measured modulus spectra P_UU of the
5000 boxes exceed the model with (1 + a) g + W by a residual growing with K over 0.02-0.05 h/Mpc for |m| >= 1 (shape
check with free a, W per field: chi2 3772 / 756 at the joint fit's nuisances; with per-field K^2 and K^2 mu^2 terms 2538),
which mimics a tilt across the l = 1-4 second-layer filters. Per first-layer scale W2_j K^2 + W3_j K^2 mu_K^2 in
P_UU / <U>^2 (lattice sums at q = 1, like W). With the buggy one-loop S21m they gave an unbiased, 15-25% tighter
posterior; the old tree S21m with the same terms stays biased (-1.0 / +1.2 / +1.1 sigma, AP profile).

**Converged chains with the corrected emulator** (`rsd_biased_basis_taylor_4p_ir_fog_cubic_ap3_v5loopfix.h5`, S21m
validation 2.3e-4; one-loop S21m + K^2 noise, tree X; `pud/lk2{_ap,}_chain_n5000_V60`, R - 1 <= 0.01, ESS 333 / 385;
plots `outputs/plots/triangle_lk2_{ap,noap}_n5000_V60.png`). Mean +- sigma (shift; change of sigma vs P_l):

| vector | omega_cdm | ln A_s | n_s | h | chi2 / ndata |
|---|---|---|---|---|---|
| P_l, AP | 0.1204 +- 0.0040 (-0.12) | 3.092 +- 0.060 (+0.52) | 0.956 +- 0.027 (-0.23) | 0.6713 +- 0.0040 (+0.05) | 17.0 / 48 |
| P_l + X, AP | 0.1202 +- 0.0039 (-0.20; -4%) | 3.095 +- 0.056 (+0.60; -7%) | 0.970 +- 0.025 (+0.31; -10%) | 0.6726 +- 0.0040 (+0.37; 0%) | 37.2 / 128 |
| old baseline, AP | 0.1164 +- 0.0032 (-1.40; -20%) | 3.126 +- 0.055 (+1.19; -9%) | 0.998 +- 0.023 (+1.56; -17%) | 0.6705 +- 0.0036 (-0.17; -10%) | 58.7 / 236 |
| **one-loop S21m + K^2, AP** | 0.1180 +- 0.0037 (-0.79; -7%) | 3.099 +- 0.058 (+0.64; -3%) | 0.986 +- 0.025 (+0.93; -8%) | 0.6712 +- 0.0038 (+0.03; -3%) | 62.5 / 236 |
| P_l, no AP | 0.1239 +- 0.0045 (+0.66) | 3.052 +- 0.061 (-0.16) | 0.935 +- 0.028 (-0.96) | 0.6741 +- 0.0055 (+0.55) | 17.1 / 48 |
| old baseline, no AP | 0.1203 +- 0.0034 (-0.19; -23%) | 3.072 +- 0.053 (+0.21; -14%) | 0.974 +- 0.023 (+0.53; -21%) | 0.6709 +- 0.0043 (-0.05; -20%) | 61.5 / 236 |
| **one-loop S21m + K^2, no AP** | 0.1228 +- 0.0041 (+0.46; -8%) | 3.039 +- 0.058 (-0.38; -5%) | 0.957 +- 0.026 (-0.20; -9%) | 0.6737 +- 0.0051 (+0.51; -7%) | 64.4 / 236 |

**Reading.** With the corrected response the full vector is unbiased without AP and within 0.93 sigma with AP (the
residual slide along the old ridge, n_s high), but it gains only 3-9% over P_l: about what P_l + X alone gives with AP
(0-10%). The 15-25% gain of the buggy chains came from the doubled loop response, which supplied response amplitude the
data want (cf. the 15-30% response deficit at P_l's nuisances above, absorbed in the joint fits by bG2 and the
cosmology). Open: a better nonlinear response of the short modes (beyond the response-function approximation at a fixed
reference bias), which is where the S21m(17.7) gain appears to live.
