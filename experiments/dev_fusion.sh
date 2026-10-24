#!/bin/bash
# dev_fusion.sh -- score-based fusion of development runs (no GPU).
# Writes virtual runs DEV*_FUSE-<paths>_{frame,seq} for scripts/dev_analysis.py.
R=/workspace/results/ipcai
PY=/workspace/.venvs/mdsam3-verify/bin/python
cd /workspace/md-sam3
fuse() {
  name=$1; shift
  args=""
  for r in "$@"; do args="$args --run $R/$r"; done
  $PY scripts/fuse_paths.py --dataset $D --data-root $ROOT $args --out $R/${P}_FUSE-$name
}
for P in DEVev17tr DEVcholecL; do
  if [ $P = DEVev17tr ]; then
    D=ev17train; ROOT=/workspace/data/endovis_2017; DM=${P}_DM_sp
  else
    D=cholec; ROOT=/workspace/data/CholecSeg8k; DM=${P}_DM
  fi
  fuse keep-delall ${P}_keep ${P}_delall
  fuse delall-DM ${P}_delall $DM
  fuse delall-Q50 ${P}_delall ${P}_Q50
  fuse delall-DM-Q50 ${P}_delall $DM ${P}_Q50
  fuse keep-DM ${P}_keep $DM
  fuse keep-DM-delall ${P}_keep $DM ${P}_delall
  fuse keep-R50 ${P}_keep ${P}_R50-score
  fuse keep-DM-delall-R50 ${P}_keep $DM ${P}_delall ${P}_R50-score
  fuse all ${P}_keep $DM ${P}_delall ${P}_Q25 ${P}_Q25w16 ${P}_Q50 ${P}_Q50w16 ${P}_Q75 ${P}_Q75w16 \
       ${P}_R25-score ${P}_R50-score ${P}_R75-score
done
