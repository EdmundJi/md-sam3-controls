#!/bin/bash
# Rate-matched controls for the chosen method: the Q50 path's deletions replaced by
# random ones at Q50's per-sequence counts (20 draws per test set), predictions
# saved so each draw can be fused with the memory-free path by the same rule.
R=${R:-/workspace/results/ipcai}
cd "$(dirname "$0")"
A17="inference_ev17.py --data-root /workspace/data/endovis_2017"
A18="inference_ev18.py --split all --data-root /workspace/data/endovis_2018"
AC="inference_extra.py --dataset cholec --split test --data-root /workspace/data/CholecSeg8k"
{
  for s in $(seq 1 20); do
    echo "EV18all_Q50-perm_s$s $A18 --config C2-perm --prune-seed $s --prune-mask-file $R/EV18all_Q50_sp/retention_decisions.json --save-preds"
    echo "TESTcholecL_Q50-perm_s$s $AC --config C2-perm --prune-seed $s --prune-mask-file $R/TESTcholecL_Q50/retention_decisions.json --save-preds"
    echo "EV17_Q50-perm_s$s $A17 --config C2-perm --prune-seed $s --prune-mask-file $R/EV17_Q50_sp/retention_decisions.json --save-preds"
  done
} > control_jobs.txt
wc -l control_jobs.txt
