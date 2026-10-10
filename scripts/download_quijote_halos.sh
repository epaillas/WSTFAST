#!/usr/bin/env bash
# Globus batch transfer of Quijote fiducial FoF halo catalogues (z = 0.5, groups_003/group_tab_003.0) to this Mac.
# Usage: scripts/download_quijote_halos.sh FIRST LAST [DEST]   (realizations FIRST..LAST inclusive; existing files skipped)
# Needs Globus Connect Personal running on the destination endpoint.
set -euo pipefail

SRC=e0eae0aa-5bca-11ea-9683-0e56c063f437   # Quijote_simulations2 (/Halos)
DST=6b8d27e9-86c3-11f1-9a26-0ee7ef9370d9   # Silver M5 (Globus Connect Personal)
FIRST=${1:?first realization}
LAST=${2:?last realization}
DEST=${3:-$HOME/data/quijote/halos/FoF/fiducial}

batch=$(mktemp)
for i in $(seq "$FIRST" "$LAST"); do
  if [ ! -s "$DEST/$i/groups_003/group_tab_003.0" ]; then
    echo "/Halos/FoF/fiducial/$i/groups_003/group_tab_003.0 $DEST/$i/groups_003/group_tab_003.0" >> "$batch"
  fi
done
n=$(wc -l < "$batch" | tr -d ' ')
echo "$n files to transfer"
[ "$n" -gt 0 ] && globus transfer "$SRC" "$DST" --batch "$batch" --label "quijote FoF fiducial $FIRST-$LAST" \
  --sync-level size --notify off
rm -f "$batch"
