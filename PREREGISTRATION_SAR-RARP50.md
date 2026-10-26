# Preregistration: SAR-RARP50 as a fourth test set

Written on 2026-10-26, before any model was run on SAR-RARP50. The code of every
retention rule is unchanged from commit 0642f3c; this commit adds only the data
preparation, the data loader and the job list. One video (video_45) was
downloaded to check the file format and that masks and decoded frames align;
no prediction was made on it.

## Why a fourth test set

In the paper (IPCAI 2027 submission, "When should SAM3 forget?"), the EndoVis 2017
and 2018 test data were analysed before the rank rule and the two-path tracker
were designed; CholecSeg8k is the only test set first run after the design was
fixed. We have never used SAR-RARP50 in any way. It is run once, with the method
and all settings frozen.

## Data

- SAR-RARP50 (Psychogyios et al., arXiv:2401.00496): the suturing step of 50
  robot-assisted radical prostatectomies. Training release: 40 operations in 44
  video files (DOI 10.5522/04/24932529.v1). Test release: 10 operations, videos
  41-50 (DOI 10.5522/04/24932499.v1). Nothing is trained, so all 54 video files
  are test sequences.
- Frames: only the annotated frames (1 Hz; a mask's file name is its frame number
  in `video_left.avi`), decoded with OpenCV and stored as PNG
  (`scripts/prepare_sarrarp.py`). SAM3 receives JPEG (quality 95), as on the
  other test sets.
- Labels: tool clasper, wrist and shaft (ids 1-3) form one category, robotic
  instrument (the labels mark parts, not instruments); suction tool (6) and
  needle holder (7) are the other two categories. Needle, thread, clamps and
  catheter are background.
- One sequence per video file. It starts at the first annotated frame showing an
  instrument, and SAM3 receives one mask per category present there (as it
  receives one mask per type label on the other sets). A sequence with fewer than
  40 frames left would be dropped (the CholecSeg8k rule).
- Evaluation as on the other sets: bIoU (binary instrument IoU per frame; an empty
  prediction on an empty ground truth scores 1) over all later annotated frames,
  frame-weighted; class-aware mcIoU* and Challenge IoU over the three categories.

## Runs (`experiments/gen_sar_jobs.sh`, `experiments/run_sar.sh`)

SAM3 keeping all memory; DM (tau_rel = 0.95); first frame only; rank rule
(rho = 0.5); top-6 bank; the two-path tracker per video and per frame (score
fusion of the first-frame-only and rank-rule runs, `scripts/fuse_paths.py`);
20 rate-matched random controls of the rank rule (seeds 1-20), each also fused
with the memory-free path. Same SAM3 weights, SAM3-DMS commit 88475d9,
N_mem = 7, joint propagation and GPU type (RTX PRO 6000 Blackwell) as in the
paper. No setting is tuned. A run is repeated only after a crash, with identical
settings, and the repeat is reported.

## Hypotheses and analysis

- H1 (primary): the two-path tracker (per video) scores higher bIoU than SAM3.
  Reported as the frame-weighted gain with a 95% bootstrap CI over sequences
  (20,000 resamples) and a two-sided sign test over sequences.
- H2: the rank rule scores higher than random deletion at its per-sequence
  counts. Reported as the number of the 20 draws it beats (randomization
  p = 1/21 if it beats all) and the mean difference with the paper's nested
  bootstrap CI.
- Descriptive: every rule's bIoU and gain over SAM3; the two-path tracker's gap
  to the best single rule; deletion rates; the number of sequences on which DM
  deletes every entry; selector accuracy; class-aware metrics.
- The paper's pooled gain is recomputed as the mean of the per-set gains over the
  four test sets (stratified bootstrap). SAR-RARP50 is also reported alone, and
  the official test release (videos 41-50) as a subset.
- Secondary endpoint on all four test sets, defined here before it is computed on
  any set: the tracking failure rate, i.e. the share of frames whose ground truth
  shows an instrument and whose binary IoU is below 0.5, per rule, with its
  difference from SAM3 and the same bootstrap.

## Reporting

Whatever the outcome, the SAR-RARP50 results will be reported in the paper as a
fourth test set, citing this document. If not all 20 controls finish before the
paper deadline, the number of completed draws is reported. Any deviation from
this document will be listed in the paper.
