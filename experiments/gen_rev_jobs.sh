#!/bin/bash
# Revision runs (after the IPCAI 2027 internal review), all added after method selection:
#  - timing: exclusive runs on EndoVis 2017 for ms/frame and peak GPU memory
#  - topk: fixed-capacity bank keeping the 6 highest-scoring entries (comparator)
#  - png: lossless input for SAM3, DM, first frame only and the rank rule
#  - rate: rank rule with rho = 0.25 / 0.75 on the test sets (sensitivity)
D17=${D17:-/workspace/data/endovis_2017}; D18=${D18:-/workspace/data/endovis_2018}; DCH=${DCH:-/workspace/data/CholecSeg8k}
cd "$(dirname "$0")"
E17="inference_ev17.py --data-root $D17"
E18="inference_ev18.py --split all --data-root $D18"
ECH="inference_extra.py --dataset cholec --split test --data-root $DCH"
{
  echo "EV17_C0_time $E17 --config C0"
  echo "EV17_DM_time $E17 --config C2"
  echo "EV17_delall_time $E17 --config C2 --tau-rel 2.0"
  echo "EV17_Q50_time $E17 --config C2-quant --prune-rate 0.5"
  echo "EV17_topk $E17 --config C2-topk --save-preds"
} > rev_timing.txt
{
  # longest first
  echo "TESTcholecL_topk $ECH --config C2-topk --save-preds"
  echo "TESTcholecL_keep_png $ECH --config C2 --tau-rel -1 --frame-format png --save-preds"
  echo "TESTcholecL_DM_png $ECH --config C2 --frame-format png --save-preds"
  echo "TESTcholecL_delall_png $ECH --config C2 --tau-rel 2.0 --frame-format png --save-preds"
  echo "TESTcholecL_Q50_png $ECH --config C2-quant --prune-rate 0.5 --frame-format png --save-preds"
  for q in 25 75; do echo "TESTcholecL_Q$q $ECH --config C2-quant --prune-rate 0.$q --save-preds"; done
  echo "EV18all_topk $E18 --config C2-topk --save-preds"
  echo "EV18all_C0_png $E18 --config C0 --frame-format png --save-preds"
  echo "EV18all_DM_png $E18 --config C2 --frame-format png --save-preds"
  echo "EV18all_delall_png $E18 --config C2 --tau-rel 2.0 --frame-format png --save-preds"
  echo "EV18all_Q50_png $E18 --config C2-quant --prune-rate 0.5 --frame-format png --save-preds"
  for q in 25 75; do echo "EV18all_Q$q $E18 --config C2-quant --prune-rate 0.$q --save-preds"; done
  echo "DEVev17tr_topk inference_extra.py --dataset ev17train --data-root $D17 --config C2-topk --save-preds"
  echo "DEVcholecL_topk inference_extra.py --dataset cholec --split dev --data-root $DCH --config C2-topk --save-preds"
  echo "EV17_C0_png $E17 --config C0 --frame-format png --save-preds"
  # EV17 DM with PNG input already exists as EV17_C2_png
  echo "EV17_delall_png $E17 --config C2 --tau-rel 2.0 --frame-format png --save-preds"
  echo "EV17_Q50_png $E17 --config C2-quant --prune-rate 0.5 --frame-format png --save-preds"
  for q in 25 75; do echo "EV17_Q$q $E17 --config C2-quant --prune-rate 0.$q --save-preds"; done
} > rev_jobs.txt
wc -l rev_timing.txt rev_jobs.txt
