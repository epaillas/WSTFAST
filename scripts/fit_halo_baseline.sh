#!/bin/bash
# The halo WST baseline (since 2026-10-15; paper/main_cv.tex): S1m (sigma_j >= 25 Mpc/h, q = 0.8) + S21m (sigma_j1 = 25,
# sigma_j2 = 70.7 Mpc/h) of Quijote FoF halos in redshift space at z = 0.5, with
#   - the shared Fingers-of-God counterterm c_v of both layers (--fog shared; emulator trained with --fog), which replaces
#     the velocity damping of the S1m cumulants and of the S21m modulus spectra;
#   - the cubic biases around the halo relations (--cubic-biases relation);
#   - the non-Gaussian second layer: halo template, held-out c_l, per-l amplitudes marginalised analytically;
#   - one white noise term W per first-layer scale (now consistent with zero) and the per-field noise amplitudes;
#   - Alcock-Paczynski distortions of the trial cosmology in every piece of the data vector (--ap), with the same
#     recipe throughout: AP polynomials in ln q of the P_s grid projections, the S1m cumulants, and both S21m layers
#     (emulator trained with --ap-order 3; covers this data vector only);
#   - n_s fixed (omega_cdm, logA, h varied), NUTS.
# --no-ap (handled here) drops AP and switches to the no-AP emulator, which gives the same no-AP model ~5x faster.
# Other extra arguments are passed to fit_rsd.py and override these (e.g. --volume 60 --output-dir ..., --emulator ...,
# --vary omega_cdm logA n_s h). Run from the repository root:
#   scripts/fit_halo_baseline.sh --volume 25 --output-dir outputs/inference/rsd/my_run
PY=${PYTHON:-/Users/epaillas/code/miniconda3/envs/desi-clustering-classpt/bin/python}
EMULATOR=outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic_ap3_base.h5
AP=--ap
args=()
for arg in "$@"; do
    if [ "$arg" = "--no-ap" ]; then
        EMULATOR=outputs/emulators/rsd_biased_basis_taylor_4p_ir_fog_cubic.h5
        AP=
    else
        args+=("$arg")
    fi
done
exec $PY scripts/fit_rsd.py \
    --data-dir data/quijote/fiducial/z0.5/halos_m13_J9_L6_L2-4_dj2_sigma0.8_step1.414_n256_los \
    --emulator $EMULATOR $AP \
    --stats s1m s21m --s21-min-scale 25 --s21-white --cubic-biases relation \
    --s21-layer2 outputs/layer2/template_rsd.json --s21-layer2-c halo --s21-layer2-marg per_l \
    --fog shared --vary omega_cdm logA h "${args[@]}"
