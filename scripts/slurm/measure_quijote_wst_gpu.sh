#!/bin/bash
# Half-octave WST superset of the 1500 Quijote fiducial snapshots of the DSC P(k) run (z = 0.5, real +
# redshift space) on GPU nodes: 6 array tasks x 4 GPUs x 63 realizations, one process per GPU.
# Fill in the CHANGE_ME settings, then
#     sbatch scripts/slurm/measure_quijote_wst_gpu.sh
# Resubmitting is safe: finished files are skipped. Time one realization first and adjust --time.
#SBATCH --job-name=wst-quijote-gpu
#SBATCH --account=CHANGE_ME
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/wst-quijote-gpu-%A_%a.log

set -euo pipefail
SNAPSHOT_ROOT=/CHANGE/ME/Quijote/Snapshots/fiducial   # contains <id>/snapdir_003/snap_003.*.hdf5
OUTPUT_DIR=/CHANGE/ME/wstfast/data/quijote/z0.5
REPO=/CHANGE/ME/wstfast
REALIZATIONS=$REPO/scripts/slurm/quijote_dsc_realizations.txt   # the 1500 ids of the DSC P(k) run
GPUS=4
PER_GPU=63

source activate CHANGE_ME_ENV   # numpy, scipy, h5py, hdf5plugin, CUDA torch; then `pip install -e $REPO --no-deps`
cd "$REPO"
mkdir -p logs
for gpu in $(seq 0 $((GPUS - 1))); do
    first=$(((SLURM_ARRAY_TASK_ID * GPUS + gpu) * PER_GPU + 1))
    ids=$(sed -n "${first},$((first + PER_GPU - 1))p" "$REALIZATIONS")
    [ -z "$ids" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=16 python scripts/measure_quijote_wst.py --superset \
        --backend torch --device cuda --snapshot-root "$SNAPSHOT_ROOT" --output-dir "$OUTPUT_DIR" \
        --realizations $ids --spaces real rsd --redshift 0.5 --nmesh 256 > "logs/wst-gpu-${SLURM_ARRAY_TASK_ID}-${gpu}.log" 2>&1 &
done
wait
