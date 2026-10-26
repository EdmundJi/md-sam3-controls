#!/bin/bash
# get_sarrarp.sh [OUT] -- download SAR-RARP50 (train + test releases, UCL RDR /
# figshare articles 24932529 and 24932499) and prepare each video as soon as it
# arrives (scripts/prepare_sarrarp.py: decode the 1-Hz annotated frames, drop the video).
OUT=${1:-/workspace/data/SAR-RARP50}
PY=${PY:-/workspace/.venvs/mdsam3-verify/bin/python}
REPO=${REPO:-/workspace/md-sam3}
mkdir -p "$OUT/zips"
for a in 24932529 24932499; do
  curl -s --retry 5 "https://api.figshare.com/v2/articles/$a/files?page_size=100"
done | $PY -c '
import json, sys
dec = json.JSONDecoder(); s = sys.stdin.read(); i = 0
while i < len(s):
    files, i = dec.raw_decode(s, i)
    while i < len(s) and s[i].isspace(): i += 1
    for f in files:
        if f["name"].endswith(".zip"):
            print(f["name"], f["size"], f["download_url"])' > "$OUT/files.txt"
echo "$(wc -l < "$OUT/files.txt") zips, $(awk '{s+=$2} END {printf "%.1f GB", s/1e9}' "$OUT/files.txt")"
get() {  # NAME SIZE URL
  local z="$OUT/zips/$1"
  [ -f "$OUT/${1%.zip}/DONE" ] && return
  for t in 1 2 3 4 5; do
    curl -sL --retry 5 -C - -o "$z" "$3"
    [ "$(stat -c %s "$z" 2>/dev/null)" = "$2" ] && break
    echo "retry $1 ($t)"; sleep 10
  done
  $PY "$REPO/scripts/prepare_sarrarp.py" "$z" "$OUT" --workers 1 --delete-zips
}
export -f get; export OUT PY REPO
sort -k2 -n -r "$OUT/files.txt" | xargs -P 6 -L 1 bash -c 'get "$0" "$1" "$2"'
echo "prepared $(ls "$OUT"/video_*/DONE | wc -l) videos, $(du -sh "$OUT" | cut -f1)"
