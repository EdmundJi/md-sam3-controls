#!/bin/bash
# rerun_dmsig_sar.sh -- the exploratory SAR_DMsig run was stopped at 13:28 on
# 2026-10-26: five concurrent processes exceeded GPU memory and SAR_keep crashed
# with an out-of-memory error (job.sh repeats it with identical settings). Rerun
# SAR_DMsig from the separate DMsig code copy once a main worker has exited (at
# most four processes on the GPU). Shuts the machine down 90 min after both the
# main runs (SAR FUSION DONE) and this run have ended, unless $R/KEEP_ALIVE2 exists.
R=/workspace/results/ipcai
Q=/workspace/md-sam3/experiments/ipcai2027/queue_sar.txt
DS=/workspace/data/SAR-RARP50
touch $R/KEEP_ALIVE        # run_sar.sh must not shut down while this run is pending
while [ -s $Q ] || [ "$(pgrep -fc '[s]rc/inference_extra.py')" -ge 4 ]; do sleep 60; done
cd /workspace/md-sam3-dmsig/experiments/ipcai2027
rm -rf $R/SAR_DMsig
echo "SAR_DMsig inference_extra.py --dataset sarrarp --split all --data-root $DS --config C2 --save-preds" > queue_dmsig.txt
echo "[$(date '+%F %T')] rerun SAR_DMsig (stopped at 13:28 to free GPU memory)" >> $R/run_dmsig.log
bash worker.sh queue_dmsig.txt 0 >> $R/run_dmsig.log 2>&1
echo "[$(date '+%F %T')] DMSIG RERUN DONE" >> $R/run_dmsig.log
while ! grep -q "SAR FUSION DONE" $R/run_sar.log; do sleep 60; done
sleep 5400
[ -e $R/KEEP_ALIVE2 ] || { echo "[$(date '+%F %T')] auto shutdown (rerun_dmsig_sar.sh)" >> $R/run_sar.log; /usr/bin/shutdown -h now; }
