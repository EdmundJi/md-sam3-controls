#!/bin/bash
# Fourth test set, SAR-RARP50 (preregistered in PREREGISTRATION_SAR-RARP50.md before
# any run on it): the rules of Table 1 with the settings of the CholecSeg8k test runs,
# and 20 rate-matched random controls of the rank rule (seeds 1-20), all 54 videos.
R=${R:-/workspace/results/ipcai}
DS=${DS:-/workspace/data/SAR-RARP50}
cd "$(dirname "$0")"
A="inference_extra.py --dataset sarrarp --split all --data-root $DS"
{
  echo "SAR_Q50 $A --config C2-quant --prune-rate 0.5 --save-preds"     # first: the controls need its decisions
  echo "SAR_delall $A --config C2 --tau-rel 2.0 --save-preds"
  echo "SAR_keep $A --config C2 --tau-rel -1 --save-preds"              # = SAM3 (C0), logs scores
  echo "SAR_DM $A --config C2 --save-preds"
  echo "SAR_topk $A --config C2-topk --save-preds"
  for s in $(seq 1 20); do
    echo "SAR_Q50-perm_s$s $A --config C2-perm --prune-seed $s --prune-mask-file $R/SAR_Q50/retention_decisions.json --save-preds"
  done
} > sar_jobs.txt
wc -l sar_jobs.txt
