#!/bin/bash
# test_fusion.sh -- the method chosen on the development data (FUSE-delall-Q50,
# per-sequence rule) and the delall+DM alternative, on the three test sets.
# Waits for the runs it needs, then writes virtual runs <prefix>_FUSE-*_{frame,seq}.
R=/workspace/results/ipcai
PY=/workspace/.venvs/mdsam3-verify/bin/python
cd /workspace/md-sam3
wait_for() { for r in "$@"; do while [ ! -f $R/$r/metrics.json ]; do sleep 30; done; done; }
fuse() { out=$1; ds=$2; root=$3; shift 3; args=""; for r in "$@"; do args="$args --run $R/$r"; done
         $PY scripts/fuse_paths.py --dataset $ds --data-root $root $args --out $R/$out >> $R/test_fusion.log 2>&1; }
(
 wait_for EV17_C2_delall_sp EV17_Q50_sp EV17_C2_sp
 fuse EV17_FUSE-delall-Q50 ev17 /workspace/data/endovis_2017 EV17_C2_delall_sp EV17_Q50_sp
 fuse EV17_FUSE-delall-DM ev17 /workspace/data/endovis_2017 EV17_C2_delall_sp EV17_C2_sp
 echo "[$(date '+%F %T')] test fusion EV17" >> $R/run_oracle.log ) &
(
 wait_for EV18all_C2_delall_sp EV18all_Q50_sp EV18all_C2_sp
 fuse EV18all_FUSE-delall-Q50 ev18 /workspace/data/endovis_2018 EV18all_C2_delall_sp EV18all_Q50_sp
 fuse EV18all_FUSE-delall-DM ev18 /workspace/data/endovis_2018 EV18all_C2_delall_sp EV18all_C2_sp
 echo "[$(date '+%F %T')] test fusion EV18" >> $R/run_oracle.log ) &
(
 wait_for TESTcholecL_delall TESTcholecL_Q50 TESTcholecL_DM
 fuse TESTcholecL_FUSE-delall-Q50 cholec /workspace/data/CholecSeg8k TESTcholecL_delall TESTcholecL_Q50
 fuse TESTcholecL_FUSE-delall-DM cholec /workspace/data/CholecSeg8k TESTcholecL_delall TESTcholecL_DM
 echo "[$(date '+%F %T')] test fusion cholec" >> $R/run_oracle.log ) &
wait
