#!/bin/bash
# Writes oracle_jobs.txt: oracle / score-ranked / random deletion at matched
# counts, and rate-matched controls at low memory capacity.
# Deletion lists come from scripts/oracle_decisions.py (into $O); jobs whose list
# does not exist yet wait in the queue (worker.sh).
R=${R:-/workspace/results/ipcai}
O=$R/oracle
D17=/workspace/data/endovis_2017
D18=/workspace/data/endovis_2018
cd "$(dirname "$0")"
A17="inference_ev17.py --data-root $D17"
A18="inference_ev18.py --split all --data-root $D18"
FREE18="--seq-tags train2,train5,train9,train11,train12"
RP="--config C2-replay --prune-mask-file"
{
  # 1. C0 trajectory with per-frame scores logged (nothing deleted; must equal C0_ref)
  echo "EV17_C0_scores $A17 $RP $O/EV17/keepall.json"
  echo "EV18all_C0_scores $A18 $RP $O/EV18/keepall.json"
  # 2. Fixed rate: delete floor(n_s/2) entries of every sequence
  for a in oracle anti score; do echo "EV18all_R50-$a $A18 $RP $O/EV18/${a}_r50.json"; done
  for a in oracle anti score; do echo "EV17_R50-$a $A17 $RP $O/EV17/${a}_r50.json"; done
  # 3. DM's per-sequence counts (EV18: only the 5 sequences where DM leaves a choice)
  for a in oracle anti; do echo "EV17_C2-$a $A17 $RP $O/EV17/${a}_dm.json"; done
  for a in oracle anti; do echo "EV18all_C2-$a $A18 $RP $O/EV18/${a}_dm.json $FREE18"; done
  for s in $(seq 1 20); do
    echo "EV18all_R50-perm_s$s $A18 --config C2-perm --prune-seed $s --prune-mask-file $O/EV18/oracle_r50.json"
    echo "EV17_R50-perm_s$s $A17 --config C2-perm --prune-seed $s --prune-mask-file $O/EV17/oracle_r50.json"
  done
  # 4. Low capacity (num_maskmem 3, 4): delete-all, oracle, and 20 random draws at DM's counts
  for nm in 3 4; do
    echo "EV17_C2_delall_nm$nm $A17 --config C2 --tau-rel 2.0 --num-maskmem $nm"
    for a in oracle anti; do echo "EV17_C2-${a}_nm$nm $A17 $RP $O/EV17nm$nm/${a}_dm.json --num-maskmem $nm"; done
  done
  for s in $(seq 1 20); do
    for nm in 3 4; do
      echo "EV17_C2-perm_nm${nm}_s$s $A17 --config C2-perm --prune-seed $s --num-maskmem $nm --prune-mask-file $R/EV17_C2_nm$nm/retention_decisions.json"
    done
  done
} > oracle_jobs.txt
wc -l oracle_jobs.txt
# 5. Rate sensitivity (added after the 50% results): score-ranked and 5 random
#    draws at 25% and 75% deletion
{
  for r in 25 75; do
    for p in EV18all EV17; do
      ds=${p%all}; A=$([ $p = EV17 ] && echo "$A17" || echo "$A18")
      echo "${p}_R$r-score $A $RP $O/$ds/score_r$r.json"
      for s in 1 2 3 4 5; do echo "${p}_R$r-perm_s$s $A --config C2-perm --prune-seed $s --prune-mask-file $O/$ds/score_r$r.json"; done
    done
  done
} > rate_jobs.txt
wc -l rate_jobs.txt
