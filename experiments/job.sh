#!/bin/bash
# job.sh NAME SCRIPT [ARGS...] -- one inference run; skipped if already done.
name=$1; script=$2; shift 2
R=${R:-/workspace/results/ipcai}
PY=${PY:-/workspace/.venvs/mdsam3-verify/bin/python}
MODEL=${MODEL:-/workspace/models/facebook/sam3/sam3.pt}
out=$R/$name
if [ -f "$out/metrics.json" ]; then echo "[skip] $name"; exit 0; fi
mkdir -p "$out" "$R/_logs"
t0=$(date +%s)
echo "[$(date '+%F %T')] start $name (gpu ${GPU:-0})"
cd /workspace/md-sam3
for attempt in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PY src/$script --model-path "$MODEL" \
    --output-dir "$out" --seed 1 --device cuda "$@" > "$R/_logs/$name.log" 2>&1
  rc=$?
  # retry when concurrent jobs exhausted GPU memory
  [ $rc -ne 0 ] && grep -q "OutOfMemoryError" "$R/_logs/$name.log" && [ $attempt -lt 4 ] || break
  echo "[$(date '+%F %T')] retry $name after OOM (attempt $attempt)"
  sleep 120
done
echo "[$(date '+%F %T')] end $name rc=$rc $(( $(date +%s) - t0 ))s $(grep -o '"biou": [0-9.]*' "$out/metrics.json" 2>/dev/null)"
exit 0
