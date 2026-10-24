#!/bin/bash
# run_controls.sh [WORKERS] -- run control_jobs.txt, fuse each finished draw with the
# memory-free path, then shut the machine down 45 min after everything is done
# unless $R/KEEP_ALIVE exists (results stay on disk).
N=${1:-7}
R=/workspace/results/ipcai
PY=/workspace/.venvs/mdsam3-verify/bin/python
cd "$(dirname "$0")"
bash gen_control_jobs.sh
cp control_jobs.txt queue_controls.txt
for i in $(seq 1 $N); do bash worker.sh queue_controls.txt 0 & sleep 10; done
wait
echo "[$(date '+%F %T')] CONTROL RUNS DONE" >> $R/run_oracle.log
cd /workspace/md-sam3
for s in $(seq 1 20); do
  $PY scripts/fuse_paths.py --dataset ev17 --data-root /workspace/data/endovis_2017 --run $R/EV17_C2_delall_sp --run $R/EV17_Q50-perm_s$s --out $R/EV17_FUSE-delall-Q50perm_s$s
  $PY scripts/fuse_paths.py --dataset ev18 --data-root /workspace/data/endovis_2018 --run $R/EV18all_C2_delall_sp --run $R/EV18all_Q50-perm_s$s --out $R/EV18all_FUSE-delall-Q50perm_s$s
  $PY scripts/fuse_paths.py --dataset cholec --data-root /workspace/data/CholecSeg8k --run $R/TESTcholecL_delall --run $R/TESTcholecL_Q50-perm_s$s --out $R/TESTcholecL_FUSE-delall-Q50perm_s$s
done >> $R/control_fusion.log 2>&1
echo "[$(date '+%F %T')] CONTROL FUSION DONE" >> $R/run_oracle.log
sleep 2700
[ -e $R/KEEP_ALIVE ] || { echo "[$(date '+%F %T')] auto shutdown" >> $R/run_oracle.log; shutdown -h now; }
