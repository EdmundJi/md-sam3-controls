#!/bin/bash
# Writes ev17_jobs.txt and ev18_jobs.txt (one job per line: NAME SCRIPT ARGS...),
# references first, then priority order. Controls wait for their reference run.
R=${R:-/workspace/results/ipcai}
D17=/workspace/data/endovis_2017
D18=/workspace/data/endovis_2018
cd "$(dirname "$0")"
M17=$R/EV17_C2_ref/retention_decisions.json
M18=$R/EV18all_C2_ref/retention_decisions.json
A17="inference_ev17.py --data-root $D17"
A18="inference_ev18.py --split all --data-root $D18"
{
  echo "EV17_C2_ref $A17 --config C2 --save-preds"
  echo "EV17_C0_ref $A17 --config C0 --save-preds"
  for s in $(seq 1 10); do echo "EV17_C2-perm_s$s $A17 --config C2-perm --prune-seed $s --prune-mask-file $M17"; done
  echo "EV17_C2-even $A17 --config C2-even --prune-mask-file $M17"
  echo "EV17_C2_rep $A17 --config C2"
  echo "EV17_C0_rep $A17 --config C0"   # determinism check (Supplementary S6)
  for c in C1 C4 C4-GG C7 C7-store; do echo "EV17_${c}_ref $A17 --config $c --save-preds"; done
  for s in $(seq 1 10); do echo "EV17_C2-shift_s$s $A17 --config C2-shift --prune-seed $s --prune-mask-file $M17"; done
  for s in $(seq 11 20); do echo "EV17_C2-perm_s$s $A17 --config C2-perm --prune-seed $s --prune-mask-file $M17"; done
  for nm in 6 5 4 3 2; do
    echo "EV17_C0_nm$nm $A17 --config C0 --num-maskmem $nm"
    echo "EV17_C2_nm$nm $A17 --config C2 --num-maskmem $nm"
  done
  echo "EV17_C0_png $A17 --config C0 --frame-format png"
  echo "EV17_C2_png $A17 --config C2 --frame-format png"
  for s in $(seq 11 20); do echo "EV17_C2-shift_s$s $A17 --config C2-shift --prune-seed $s --prune-mask-file $M17"; done
  for t in 0.90 0.85 0.70 0.50; do echo "EV17_C2_tau$t $A17 --config C2 --tau-rel $t"; done
  for c in C2-SRG C4-SRG; do echo "EV17_${c} $A17 --config $c"; done
  # delete-all control (tau > 1: every entry deleted, SAM3 reads only the conditioning frame)
  echo "EV17_C2_delall $A17 --config C2 --tau-rel 2.0"
} > ev17_jobs.txt
# On EV18, DM deletes every entry of 13 of the 18 sequences. There any control
# with the same per-sequence count deletes the same set and runs the same program,
# so controls are run only where DM leaves a choice (FREE18, read from the
# reference run's retention_decisions.json; listed here for the released run).
# EV18val_* repeat two seeds that also ran on all 18 sequences, to check that a
# subset run reproduces the full run on those sequences.
FREE18="--seq-tags train2,train5,train9,train11,train12"
{
  echo "EV18all_C2_ref $A18 --config C2 --save-preds"
  echo "EV18all_C0_ref $A18 --config C0 --save-preds"
  for c in C1 C4 C4-GG C7-store; do echo "EV18all_${c}_ref $A18 --config $c --save-preds"; done
  for s in 1 2; do echo "EV18all_C2-shift_s$s $A18 --config C2-shift --prune-seed $s --prune-mask-file $M18"; done
  for s in 1 2; do echo "EV18val_C2-shift_s$s $A18 --config C2-shift --prune-seed $s --prune-mask-file $M18 $FREE18"; done
  echo "EV18all_C2_delall $A18 --config C2 --tau-rel 2.0"
  for s in $(seq 3 20); do echo "EV18all_C2-shift_s$s $A18 --config C2-shift --prune-seed $s --prune-mask-file $M18 $FREE18"; done
  for s in $(seq 1 20); do echo "EV18all_C2-perm_s$s $A18 --config C2-perm --prune-seed $s --prune-mask-file $M18 $FREE18"; done
  echo "EV18all_C2-even $A18 --config C2-even --prune-mask-file $M18 $FREE18"
  for c in C2-SRG C4-SRG; do echo "EV18all_${c} $A18 --config $c"; done
} > ev18_jobs.txt
wc -l ev17_jobs.txt ev18_jobs.txt
