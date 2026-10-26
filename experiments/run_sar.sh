#!/bin/bash
# run_sar.sh [WORKERS] -- SAR-RARP50 runs (gen_sar_jobs.sh) in parallel, then score
# fusion (two-path tracker, per video and per frame; delall+DM; each random control
# fused with the memory-free path). Shuts the machine down 60 min after the end
# unless $R/KEEP_ALIVE exists.
N=${1:-4}
R=/workspace/results/ipcai
DS=/workspace/data/SAR-RARP50
PY=/workspace/.venvs/mdsam3-verify/bin/python
L=$R/run_sar.log
cd "$(dirname "$0")"
# start only once all 54 videos are prepared (get_sarrarp.sh)
while [ "$(ls $DS/video_*/DONE 2>/dev/null | wc -l)" -lt 54 ]; do sleep 60; done
echo "[$(date '+%F %T')] SAR DATA READY: $(ls $DS/video_*/DONE | wc -l) videos" >> $L
bash gen_sar_jobs.sh
cp sar_jobs.txt queue_sar.txt
for i in $(seq 1 $N); do bash worker.sh queue_sar.txt 0 >> $L 2>&1 < /dev/null & sleep 20; done
wait
echo "[$(date '+%F %T')] SAR RUNS DONE" >> $L
cd /workspace/md-sam3
fuse() { $PY scripts/fuse_paths.py --dataset sarrarp --data-root $DS --run $R/$1 --run $R/$2 --out $R/$3; }
{
  fuse SAR_delall SAR_Q50 SAR_FUSE-delall-Q50
  fuse SAR_delall SAR_DM SAR_FUSE-delall-DM
  for s in $(seq 1 20); do fuse SAR_delall SAR_Q50-perm_s$s SAR_FUSE-delall-Q50perm_s$s; done
} >> $R/sar_fusion.log 2>&1
echo "[$(date '+%F %T')] SAR FUSION DONE" >> $L
sleep 3600
[ -e $R/KEEP_ALIVE ] || { echo "[$(date '+%F %T')] auto shutdown" >> $L; /usr/bin/shutdown -h now; }
