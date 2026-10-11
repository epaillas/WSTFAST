#!/usr/bin/env bash
# Globus batch transfer of Quijote FoF halo catalogues (z = 0.5, groups_003/group_tab_003.0) to this Mac.
# Usage: [COSMOLOGY=Om_p] [DST=<endpoint>] scripts/download_quijote_halos.sh FIRST LAST [DEST]
#   realizations FIRST..LAST inclusive, existing files skipped; COSMOLOGY: fiducial (default) or a derivative set
#   (ns_p, s8_m, Om_p, ...: seeds matched to fiducial 0-499). Needs Globus Connect Personal on the destination.
set -euo pipefail

SRC=e0eae0aa-5bca-11ea-9683-0e56c063f437   # Quijote_simulations2 (/Halos)
DST=${DST:-6b8d27e9-86c3-11f1-9a26-0ee7ef9370d9}   # default: Silver M5 (Globus Connect Personal)
COSMOLOGY=${COSMOLOGY:-fiducial}
FIRST=${1:?first realization}
LAST=${2:?last realization}
DEST=${3:-$HOME/data/quijote/halos/FoF/$COSMOLOGY}

batch=$(mktemp)
for i in $(seq "$FIRST" "$LAST"); do
  if [ ! -s "$DEST/$i/groups_003/group_tab_003.0" ]; then
    echo "/Halos/FoF/$COSMOLOGY/$i/groups_003/group_tab_003.0 $DEST/$i/groups_003/group_tab_003.0" >> "$batch"
  fi
done
n=$(wc -l < "$batch" | tr -d ' ')
echo "$n files to transfer"
[ "$n" -gt 0 ] && globus transfer "$SRC" "$DST" --batch "$batch" --label "quijote FoF $COSMOLOGY $FIRST-$LAST" \
  --sync-level size --notify off
rm -f "$batch"
