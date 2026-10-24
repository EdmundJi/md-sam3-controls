#!/bin/bash
# run_oracle.sh [WORKERS] -- queue oracle_jobs.txt on GPU 0; build the
# score-ranked lists once each C0_scores run finishes.
N=${1:-7}
R=${R:-/workspace/results/ipcai}
PY=${PY:-/workspace/.venvs/mdsam3-verify/bin/python}
cd "$(dirname "$0")"
bash gen_oracle_jobs.sh
cp oracle_jobs.txt queue_oracle.txt
for ds in EV17 EV18; do
  run=$([ $ds = EV17 ] && echo EV17_C0_scores || echo EV18all_C0_scores)
  root=$([ $ds = EV17 ] && echo /workspace/data/endovis_2017 || echo /workspace/data/endovis_2018)
  c0=$([ $ds = EV17 ] && echo EV17_C0_ref || echo EV18all_C0_ref)
  dm=$([ $ds = EV17 ] && echo EV17_C2_ref || echo EV18all_C2_ref)
  ( while [ ! -f $R/$run/metrics.json ]; do sleep 30; done
    cd /workspace/md-sam3 && $PY scripts/oracle_decisions.py --dataset ${ds,,} --data-root $root \
      --c0 $R/$c0 --dm $R/$dm/retention_decisions.json --scores $R/$run/retention_decisions.json \
      --out $R/oracle/$ds > $R/oracle/$ds/diagnostics.txt 2>&1
    echo "[$(date '+%F %T')] score lists for $ds" ) &
done
for i in $(seq 1 $N); do bash worker.sh queue_oracle.txt 0 & sleep 15; done
wait
echo "[$(date '+%F %T')] ALL DONE"
