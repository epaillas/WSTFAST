#!/bin/bash
# Half-octave WST superset of Quijote fiducial snapshots (z = 0.5, real + redshift space) as a SLURM
# job array: 30 tasks x 50 realizations, one node per task. Fill in the CHANGE_ME settings, then
#     sbatch scripts/slurm/measure_quijote_wst.sh
# Resubmitting is safe: finished files are skipped. About 6 min per realization on 18 cores.
#SBATCH --job-name=wst-quijote
#SBATCH --account=CHANGE_ME
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/wst-quijote-%A_%a.log

set -euo pipefail
SNAPSHOT_ROOT=/CHANGE/ME/Quijote/Snapshots/fiducial   # contains <id>/snapdir_003/snap_003.*.hdf5
OUTPUT_DIR=/CHANGE/ME/wst-model/data/quijote/z0.5
REPO=/CHANGE/ME/wst-model
REALIZATIONS=$REPO/scripts/slurm/quijote_dsc_realizations.txt   # the 1500 ids of the DSC P(k) run
PER_TASK=50

source activate CHANGE_ME_ENV   # numpy, scipy, h5py, hdf5plugin; then `pip install -e $REPO --no-deps`
cd "$REPO"
mkdir -p logs
ids=$(sed -n "$((SLURM_ARRAY_TASK_ID * PER_TASK + 1)),$(((SLURM_ARRAY_TASK_ID + 1) * PER_TASK))p" "$REALIZATIONS")
python scripts/measure_quijote_wst.py --superset --snapshot-root "$SNAPSHOT_ROOT" --output-dir "$OUTPUT_DIR" \
    --realizations $ids --spaces real rsd --redshift 0.5 --nmesh 256
