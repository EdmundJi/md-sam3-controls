#!/bin/bash
# Development runs (EV17 training split, CholecSeg8k dev videos) used to choose
# the retention rule before it is run on the test sets. Selection criterion
# (fixed before the runs): mean over the two development sets of bIoU - C0.
R=${R:-/workspace/results/ipcai}
O=$R/oracle
cd "$(dirname "$0")"
{
  for P in DEVev17tr DEVcholecL; do
    A=$([ $P = DEVev17tr ] && echo "inference_extra.py --dataset ev17train --data-root /workspace/data/endovis_2017" \
                           || echo "inference_extra.py --dataset cholec --split dev --data-root /workspace/data/CholecSeg8k")
    echo "${P}_keep $A --config C2 --tau-rel -1 --save-preds"   # = C0, logs scores
    echo "${P}_DM $A --config C2"
    echo "${P}_delall $A --config C2 --tau-rel 2.0"
    echo "${P}_SRG $A --config C2-SRG"
    for q in 25 50 75; do
      echo "${P}_Q$q $A --config C2-quant --prune-rate 0.$q"
      echo "${P}_Q${q}w16 $A --config C2-quant --prune-rate 0.$q --quantile-window 16"
    done
    for q in 25 50 75; do echo "${P}_R$q-score $A --config C2-replay --prune-mask-file $O/$P/score_r$q.json"; done
    echo "${P}_R50-oracle $A --config C2-replay --prune-mask-file $O/$P/oracle_r50.json"
  done
} > dev_jobs.txt
wc -l dev_jobs.txt
