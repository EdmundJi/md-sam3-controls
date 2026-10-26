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

## Amendment 1 (2026-10-26, before any model was run on SAR-RARP50)

Made after an internal statistical review of the plan above; still no prediction
exists on SAR-RARP50.

1. **Status of the estimates.** The confirmatory primary result is H1 on
   SAR-RARP50 alone. The pooled gain over the four test sets is secondary,
   because three of its four sets have a history (EV17 and EV18 analysed before
   the design; CholecSeg8k first run after it). Sensitivity analyses for the
   pooled gain: each of the 107 sequences weighted equally, and leaving out one
   set at a time.
2. **Clustering.** The 54 video files come from 50 operations (operations 11, 15,
   17 and 29 have two files each). Confidence intervals on SAR-RARP50 resample
   operations (cluster bootstrap, all files of a drawn operation together), and
   the sign test counts operations (frame-weighted gain per operation). The
   sequence-level versions written above are reported as sensitivity analyses.
   The failure rate uses the same resampling (frame-weighted share).
3. **Decision rule for H1.** Supported if the lower bound of the 95% CI of the
   gain is above zero; replicated in direction if the gain is positive and the
   sign test gives p < 0.05; otherwise not supported. The abstract reports the
   SAR-RARP50 estimate with its CI, separately from the pooled estimate, worded
   for the observed case: "gained X [CI]" (supported); "gained X, with a CI that
   includes zero, and was better on W of N operations" (replicated in
   direction); "did not improve on SAM3 (X [CI])" (not supported, in which case
   title, abstract and conclusion are revised around the controls).
4. **Ties.** Sign tests drop ties and report their number. When the two paths of
   the tracker have equal mean scores, the memory-free path is kept (as in
   `scripts/fuse_paths.py`).
5. **Power.** With per-sequence gain SDs of 3-10 points on the earlier sets, the
   standard error on 50 operations is about 0.4-1.4 points: adequate for gains of
   1-2.5 points, not for sub-point effects. The paper will say so.
6. **Exploratory additions** (declared now, reported as exploratory, not part of
   H1/H2), run on all four test sets: (i) DM with r = sigmoid(s)·c, the most
   lenient reading of ReMeDI-SAM3's r = s·c (Eq. 1 uses SAM3's own memory score,
   max(0, 2·sigmoid(s) - 1)·c); run from a separate copy of the code so that the
   preregistered runs use the code of this commit; (ii) the per-video fusion of
   SAM3 (keep all) with first frame only, from saved outputs.
