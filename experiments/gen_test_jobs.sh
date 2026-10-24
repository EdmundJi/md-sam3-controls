#!/bin/bash
# Test-phase runs for the rule chosen on the development data (two paths:
# first frame only and the rank rule), DM and SAM3 on CholecSeg8k, with saved
# predictions and logged scores for scripts/fuse_paths.py.
cd "$(dirname "$0")"
A17="inference_ev17.py --data-root ${D17:-data/endovis_2017}"
A18="inference_ev18.py --split all --data-root ${D18:-data/endovis_2018}"
AC="inference_extra.py --dataset cholec --split test --data-root ${DCH:-data/CholecSeg8k}"
cat > test_jobs.txt <<EOT
TESTcholecL_keep $AC --config C2 --tau-rel -1 --save-preds
TESTcholecL_delall $AC --config C2 --tau-rel 2.0 --save-preds
TESTcholecL_Q50 $AC --config C2-quant --prune-rate 0.5 --save-preds
TESTcholecL_DM $AC --config C2 --save-preds
EV18all_Q50_sp $A18 --config C2-quant --prune-rate 0.5 --save-preds
EV17_Q50_sp $A17 --config C2-quant --prune-rate 0.5 --save-preds
EV18all_C2_sp $A18 --config C2 --save-preds
EV17_C2_sp $A17 --config C2 --save-preds
EV17_C2_delall_sp $A17 --config C2 --tau-rel 2.0 --save-preds
EV18all_C2_delall_sp $A18 --config C2 --tau-rel 2.0 --save-preds
TESTcholecL_SRG $AC --config C2-SRG --save-preds
EOT
wc -l test_jobs.txt
