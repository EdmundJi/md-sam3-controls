#!/bin/bash
# dev_lists.sh P DATASET ROOT -- build the offline deletion lists of a development
# set once its no-deletion run (P_keep, which logs scores) has finished.
REPO=$(cd "$(dirname "$0")/.." && pwd)
R=${R:-$REPO/results}; PY=${PY:-python}; P=$1
while [ ! -f $R/${P}_keep/metrics.json ]; do sleep 30; done
cd "$REPO" && mkdir -p $R/oracle/$P && $PY scripts/oracle_decisions.py --dataset $2 --data-root $3 \
  --c0 $R/${P}_keep --dm $R/${P}_keep/retention_decisions.json --scores $R/${P}_keep/retention_decisions.json \
  --out $R/oracle/$P > $R/oracle/$P/diagnostics.txt 2>&1
echo "[$(date "+%F %T")] dev lists for $P" >> $R/run_oracle.log
