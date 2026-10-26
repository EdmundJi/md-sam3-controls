#!/bin/bash
# Exploratory (preregistration Amendment 1, item 6i): DM with r = sigmoid(s) * c, the most
# lenient reading of ReMeDI-SAM3's r = s * c (the paper's Eq. 1 is SAM3's own memory score,
# max(0, 2 sigmoid(s) - 1) * c). Runs from a separate copy of the code so that the
# preregistered runs keep the unchanged code. One worker; results in the shared results dir.
set -e
SRC=/workspace/md-sam3; DST=/workspace/md-sam3-dmsig
R=/workspace/results/ipcai; PY=/workspace/.venvs/mdsam3-verify/bin/python
rm -rf $DST; cp -r $SRC $DST
$PY - <<'PYEOF'
p = "/workspace/md-sam3-dmsig/src/models/md_sam3.py"
s = open(p).read()
a = """                eff = self._get_eff_score(nco[frame_idx])
                tau_eff = self.cfg.tau_rel
"""
b = """                eff = self._get_eff_score(nco[frame_idx])
                # exploratory copy: r = sigmoid(s) * c instead of SAM3's memory score
                _e = nco[frame_idx]
                eff = float((torch.sigmoid(_e["object_score_logits"].float().flatten())
                             * _e["iou_score"].float().flatten()).mean().item())
                tau_eff = self.cfg.tau_rel
"""
assert s.count(a) == 1, s.count(a)
open(p, "w").write(s.replace(a, b))
PYEOF
sed -i 's#^cd /workspace/md-sam3$#cd /workspace/md-sam3-dmsig#' $DST/experiments/ipcai2027/job.sh
grep -q "cd /workspace/md-sam3-dmsig" $DST/experiments/ipcai2027/job.sh
cd $DST/experiments/ipcai2027
D17=/workspace/data/endovis_2017; D18=/workspace/data/endovis_2018; DCH=/workspace/data/CholecSeg8k; DS=/workspace/data/SAR-RARP50
{
  echo "EV17_DMsig inference_ev17.py --data-root $D17 --config C2 --save-preds"
  echo "EV18all_DMsig inference_ev18.py --split all --data-root $D18 --config C2 --save-preds"
  echo "TESTcholecL_DMsig inference_extra.py --dataset cholec --split test --data-root $DCH --config C2 --save-preds"
} > queue_dmsig.txt
bash worker.sh queue_dmsig.txt 0 >> $R/run_dmsig.log 2>&1
# SAR-RARP50 only once all 54 videos are prepared
while [ "$(ls $DS/video_*/DONE 2>/dev/null | wc -l)" -lt 54 ]; do sleep 60; done
echo "SAR_DMsig inference_extra.py --dataset sarrarp --split all --data-root $DS --config C2 --save-preds" > queue_dmsig.txt
bash worker.sh queue_dmsig.txt 0 >> $R/run_dmsig.log 2>&1
echo "[$(date '+%F %T')] DMSIG DONE" >> $R/run_dmsig.log
