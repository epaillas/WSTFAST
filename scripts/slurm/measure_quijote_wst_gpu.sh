#!/bin/bash
# Half-octave WST superset (z = 0.5, real + redshift space) of one Quijote cosmology on GPU nodes,
# one process per GPU. Each array task handles 4 GPUs x PER_GPU realizations of the list.
#
#     # fiducial: the 1500 ids of the DSC P(k) run (6 tasks x 4 GPUs x 63)
#     sbatch --array=0-5 scripts/slurm/measure_quijote_wst_gpu.sh
#     # a derivative cosmology: ids 0-499 (2 tasks x 4 GPUs x 63)
#     COSMOLOGY=Om_p REALIZATIONS=scripts/slurm/quijote_ids_0-499.txt sbatch --array=0-1 scripts/slurm/measure_quijote_wst_gpu.sh
#
# Outputs go to $DATA_ROOT/$COSMOLOGY/z0.5/<tag>/{real,rsd}/. Resubmitting is safe: finished files are
# skipped. Fill in the CHANGE_ME settings and time one realization first to set --time.
#SBATCH --job-name=wst-quijote-gpu
#SBATCH --account=CHANGE_ME
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/wst-quijote-gpu-%A_%a.log

set -euo pipefail
REPO=/CHANGE/ME/wstfast
QUIJOTE_ROOT=/CHANGE/ME/Quijote/Snapshots   # contains <cosmology>/<id>/snapdir_003/snap_003.*.hdf5
DATA_ROOT=$REPO/data/quijote
COSMOLOGY=${COSMOLOGY:-fiducial}
REALIZATIONS=${REALIZATIONS:-scripts/slurm/quijote_dsc_realizations.txt}
GPUS=4
PER_GPU=${PER_GPU:-63}

source activate CHANGE_ME_ENV   # numpy, scipy, h5py, hdf5plugin, CUDA torch; then `pip install -e $REPO --no-deps`
cd "$REPO"
mkdir -p logs
for gpu in $(seq 0 $((GPUS - 1))); do
    first=$(((SLURM_ARRAY_TASK_ID * GPUS + gpu) * PER_GPU + 1))
    ids=$(sed -n "${first},$((first + PER_GPU - 1))p" "$REALIZATIONS")
    [ -z "$ids" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=16 python scripts/measure_quijote_wst.py --superset \
        --backend torch --device cuda --snapshot-root "$QUIJOTE_ROOT/$COSMOLOGY" \
        --output-dir "$DATA_ROOT/$COSMOLOGY/z0.5" --realizations $ids --spaces real rsd --redshift 0.5 \
        --nmesh 256 > "logs/wst-gpu-${COSMOLOGY}-${SLURM_ARRAY_TASK_ID}-${gpu}.log" 2>&1 &
done
wait
