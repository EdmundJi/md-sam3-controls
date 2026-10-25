#!/bin/bash
# run_rev.sh [WORKERS] -- exclusive timing runs, then the revision runs in parallel,
# then score fusion of the new paths. Shuts the machine down 60 min after the end
# unless $R/KEEP_ALIVE exists (results stay on disk).
N=${1:-5}
R=/workspace/results/ipcai
PY=/workspace/.venvs/mdsam3-verify/bin/python
L=$R/run_rev.log
cd "$(dirname "$0")"
bash gen_rev_jobs.sh
echo "[$(date '+%F %T')] TIMING START" >> $L
while read -r line; do
  set -- $line; n=$1; s=$2; shift 2
  bash job.sh "$n" "$s" "$@" >> $L 2>&1 < /dev/null
done < rev_timing.txt
echo "[$(date '+%F %T')] TIMING DONE" >> $L
cp rev_jobs.txt queue_rev.txt
for i in $(seq 1 $N); do bash worker.sh queue_rev.txt 0 >> $L 2>&1 < /dev/null & sleep 10; done
wait
echo "[$(date '+%F %T')] REV RUNS DONE" >> $L
cd /workspace/md-sam3
fuse() { $PY scripts/fuse_paths.py --dataset $1 --data-root $2 --run $R/$3 --run $R/$4 --out $R/$5; }
{
  fuse ev17 /workspace/data/endovis_2017 EV17_delall_png EV17_Q50_png EV17_FUSE-delall-Q50_png
  fuse ev18 /workspace/data/endovis_2018 EV18all_delall_png EV18all_Q50_png EV18all_FUSE-delall-Q50_png
  fuse cholec /workspace/data/CholecSeg8k TESTcholecL_delall_png TESTcholecL_Q50_png TESTcholecL_FUSE-delall-Q50_png
  for q in 25 75; do
    fuse ev17 /workspace/data/endovis_2017 EV17_C2_delall_sp EV17_Q$q EV17_FUSE-delall-Q$q
    fuse ev18 /workspace/data/endovis_2018 EV18all_C2_delall_sp EV18all_Q$q EV18all_FUSE-delall-Q$q
    fuse cholec /workspace/data/CholecSeg8k TESTcholecL_delall TESTcholecL_Q$q TESTcholecL_FUSE-delall-Q$q
  done
} >> $R/rev_fusion.log 2>&1
echo "[$(date '+%F %T')] REV FUSION DONE" >> $L
sleep 3600
[ -e $R/KEEP_ALIVE ] || { echo "[$(date '+%F %T')] auto shutdown" >> $L; shutdown -h now; }
