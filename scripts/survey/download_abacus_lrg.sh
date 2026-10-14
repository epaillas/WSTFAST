#!/usr/bin/env bash
# Download the 25 DESI DR1 AbacusSummit LRG "complete" (no fibre assignment) cutsky mocks and their randoms.
#
# Order: n(z) and data catalogues for every mock first, then the randoms index by index across mocks, so the
# analysis can start on the first few random files while the rest (~14 GB per mock) arrive. Resumable: rerun to
# continue; finished files are skipped.
#
#   MOCKS="0 1 2" scripts/survey/download_abacus_lrg.sh [first_ran last_ran] [jobs]
#
# The full set is ~370 GB (randoms dominate); locally fetch a few mocks for testing and measure the rest at NERSC,
# where the same tree is under /global/cfs/cdirs/desi/public/dr1/.
set -euo pipefail

BASE=https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/mocks/AbacusSummit/dark/v4.2
OUT=${OUT:-data/desi/abacus_v4.2_complete}
RAN0=${1:-0}
RAN1=${2:-17}
JOBS=${3:-6}
MOCKS=${MOCKS:-$(seq 0 24)}

fetch() {  # fetch <mock> <file>
    local dir="$OUT/mock$1" url="$BASE/mock$1/$2"
    mkdir -p "$dir"
    local remote
    remote=$(curl -sfI "$url" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
    if [[ -f "$dir/$2" && -n "$remote" && $(stat -f%z "$dir/$2") == "$remote" ]]; then return 0; fi
    curl -sf --retry 5 --retry-delay 10 -C - -o "$dir/$2" "$url" || { echo "FAILED $1 $2" >&2; return 1; }
    echo "done mock$1/$2"
}
export -f fetch
export BASE OUT

{
    for m in $MOCKS; do
        echo "$m LRG_complete_nz.txt"
        for cap in NGC SGC; do echo "$m LRG_complete_${cap}_clustering.dat.fits"; done
    done
    for i in $(seq "$RAN0" "$RAN1"); do
        for m in $MOCKS; do
            for cap in NGC SGC; do echo "$m LRG_complete_${cap}_${i}_clustering.ran.fits"; done
        done
    done
} | xargs -P "$JOBS" -L 1 bash -c 'fetch "$0" "$1"'
