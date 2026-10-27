# When should SAM3 forget? Rate-matched controls and a two-path tracker

Code, job lists, per-run outputs and analysis scripts for the IPCAI 2027
submission *"When should SAM3 forget? Rate-matched controls and a two-path
tracker for surgical instrument segmentation."*

Nothing here is trained. Every configuration is an inference-only change to the
memory of frozen SAM3 (run through [SAM3-DMS](https://github.com/FudanCVL/SAM3-DMS),
commit `88475d9`), initialized from the ground-truth masks of the first frame.

## Main results (test sets, bIoU %)

| Method | EndoVis 2017 (10 seq.) | EndoVis 2018 (18 seq.) | CholecSeg8k (25 seq.) | SAR-RARP50 (54 seq., preregistered) | Mean gain over SAM3 [95% CI] |
|---|---|---|---|---|---|
| SAM3 (keep all memory) | 90.28 | 73.71 | 83.00 | 78.21 | — |
| DM, τ = 0.95 (ReMeDI-SAM3 retention) | 91.57 | 74.96 | 84.14 | 78.23 | +0.92 [−0.31, +2.21] |
| First frame only (delete all memory) | 89.69 | 74.98 | 84.32 | 77.74 | +0.38 [−0.92, +1.58] |
| Top-6 bank (keep the 6 highest-scoring entries; comparator added after testing) | 91.57 | 75.16 | 83.62 | 79.35 | +1.13 [+0.19, +2.17] |
| Rank rule, ρ = 0.5 | 91.59 | 76.62 | 83.58 | 80.16 | +1.69 [+0.01, +3.57] |
| **Two-path tracker** (first frame only + rank rule, per video; selected) | 91.55 | 76.96 | 84.47 | 80.18 | +1.99 [+0.25, +3.93] |
| Two-path tracker, per frame (online variant) | 91.61 | 77.54 | 84.10 | 80.14 | +2.05 [+0.40, +3.80] |

Mean gain: mean of the four per-set frame-weighted gains; the interval resamples
sequences within sets (SAR-RARP50: operations). The two-path tracker is never
more than 0.04 bIoU below the best single rule on any test set; every single rule
(including the top-6 bank) trails the best one by 0.7 to 2.4 points on at least
one set.

The two-path tracker was chosen on development data (EndoVis 2017 training
split, five CholecSeg8k videos) by a criterion fixed before the runs, and the
test sets were run once. `results/dev_analysis.json` and
`results/test_analysis.json` hold every number; the paper's tables are
regenerated from them.

What the rate-matched controls show (DM = reliability-threshold retention):

* EndoVis 2017: DM beats all 20 random deletions at its own per-sequence
  deletion counts (+2.69, randomization p = 0.048), through two sequences;
  nested 95% CI [−0.18, +6.47].
* EndoVis 2018: DM deletes every memory entry of 13 of 18 sequences; deleting
  all memory matches it (−0.02 [−0.67, +0.60]).
* CholecSeg8k: DM deletes every entry of 11 of 25 sequences and is level with
  deleting all memory (84.14 vs 84.32).
* SAR-RARP50: DM deletes 91.9% of the entries and every entry of 8 of 54 files,
  and is level with SAM3 (+0.02 [−1.76, +2.00]).
* Tracking from the first frame alone beats SAM3's full memory on both
  development sets, EndoVis 2018 and CholecSeg8k, but loses on EndoVis 2017 and
  SAR-RARP50: whether to forget depends on the video.
* The rank rule, which ranks entries by the same score instead of thresholding
  it, beats random deletion at its own per-sequence counts on all four test
  sets, above all 20 draws each time (EndoVis 2017 +2.02 [+0.07, +4.88],
  EndoVis 2018 +3.30 [−2.81, +10.31], CholecSeg8k +0.57 [+0.07, +1.44],
  SAR-RARP50 +0.98 [+0.11, +2.16]). Replacing the two-path tracker's
  memory-path deletions by random ones lowers it on every test set (19, 19, 20
  and 19 of 20 draws below it).

## Checks run after the method was chosen

* **Top-6 bank** (`--config C2-topk`): keeps the six highest-scoring entries,
  as many as SAM3 reads. +1.12 [−0.01, +2.41] over SAM3; it would have ranked
  19th of 32 development candidates.
* **Lossless input** (`--frame-format png`): the two-path tracker gains +1.64
  [+0.23, +3.11] over SAM3 with the same input, more than any single rule. The
  single rules reorder on EndoVis 2018, where one sequence swings by 32 points
  for the rank rule; with PNG input the two-path tracker trails the best single
  rule by up to 0.25.
* **Deletion share ρ**: with ρ = 0.25 / 0.75 the two-path tracker gains +0.93 /
  +1.51 (rank rule +0.66 / +0.81); ρ = 0.5, chosen on development data, is the
  best of the three on test as well.
* **Class-aware metrics** (`scripts/classaware_offline.py`, from saved
  predictions): the two-path tracker raises mcIoU* over SAM3 on EndoVis 2017,
  EndoVis 2018 and CholecSeg8k (+1.6, +1.9, +1.3) and per-frame Challenge IoU on
  EndoVis 2018, CholecSeg8k and SAR-RARP50 (+1.9, +1.6, +2.1); it lowers mcIoU*
  on SAR-RARP50 (−1.6, all from the suction tool, which is in the first frame of
  only three files; `results/classaware_sar_perclass.json`) and Challenge IoU on
  EndoVis 2017 (−0.7).
* **Runtime**: 129 ms per frame for SAM3 on EndoVis 2017, peak 9.35 GiB; no
  rule adds measurable time.

## Fourth test set, preregistered: SAR-RARP50

Before any model was run on it, the protocol for a fourth test set, the 54 videos of
SAR-RARP50 (robot-assisted radical prostatectomy, masks at 1 Hz), was committed in
[`PREREGISTRATION_SAR-RARP50.md`](PREREGISTRATION_SAR-RARP50.md): data handling,
runs, hypotheses and analysis, with no setting changed from the paper
(commit `c43169a`, tag `prereg-sar-rarp50`; Amendment 1, commit `bdbcc39`, tag
`prereg-sar-rarp50-amendment1`, also before any run).
`scripts/prepare_sarrarp.py` prepares the data (`experiments/get_sarrarp.sh`
downloads it), `experiments/run_sar.sh` runs the job list
`experiments/gen_sar_jobs.sh`, and `scripts/sar_analysis.py` computes the
preregistered analysis (`results/sar_analysis.json`).

Results, decided by the preregistered rule:

* **H1 supported.** The two-path tracker gains +1.97 bIoU over SAM3, 95% CI over
  the 50 operations [+0.67, +3.62]; better on 29 and worse on 21 operations
  (sign test p = 0.32). Official test release (videos 41–50): +2.10 [+0.28, +4.27].
* **H2.** The rank rule exceeds all 20 random-deletion draws at its own counts
  (+0.98, nested CI over operations [+0.11, +2.16]); the two-path tracker
  exceeds 19 of 20 draws with a random memory path (+0.68 [−0.39, +1.76]).
* First frame only falls below SAM3 (77.74 vs 78.21; worse on 36 of 50
  operations); the selector kept the rank-rule path on 53 of 54 files.
* Secondary: over the four test sets the two-path tracker gains +1.99
  [+0.25, +3.93] (every sequence weighted equally +2.16 [+0.77, +3.76]; leaving
  out one set +1.57 to +2.23). The tracking failure rate on SAR-RARP50 falls from
  8.26% to 5.43% of the frames showing an instrument (−2.83 [−5.54, −0.66]).
* Exploratory (declared in the preregistration): DM with the lenient score
  σ(ŝ)·c behaves like DM (78.23, deletes 91.3%); the per-video choice between
  SAM3 and first frame only gains +0.52 [−0.08, +1.36].

Deviations: the first launch was stopped after 35 min because five concurrent
processes exceeded GPU memory and the evaluator kept every full-resolution mask
in host memory; the evaluator now stores masks losslessly compressed
(`EndoVis17Evaluator(compress=True)`, identical metrics) and all runs were
restarted from scratch with four processes. The exploratory SAR_DMsig run waited
for a free GPU slot (`experiments/rerun_dmsig_sar.sh`). Rule code and settings
are unchanged; no output of the first launch is used.

## Recompute the statistics (no GPU)

All per-run outputs are committed under `results/` (metrics, per-sequence
scores, every retention decision with its score, logs). With Python 3 and numpy:

```bash
python scripts/analysis.py          # DM controls, capacity       -> results/analysis.json
python scripts/oracle_analysis.py   # fixed-rate / oracle / low-capacity controls
python scripts/dev_analysis.py      # development ranking (rule selection)
python scripts/test_analysis.py     # test results of the chosen rule
python scripts/rev_analysis.py      # PNG input, rho sensitivity, top-6 bank, runtime -> results/rev_analysis.json
python scripts/sar_analysis.py      # preregistered SAR-RARP50 analysis, four-set summaries -> results/sar_analysis.json
python scripts/panel_analysis.py    # effect concentration, online path switching -> results/panel_analysis.json
python scripts/failure_rate.py results --pool-only   # failure rate, pooled CIs from the committed per-sequence counts
# from saved predictions (needs the data and the runs' preds/, which are not committed)
python scripts/classaware_offline.py results data/endovis_2017 data/endovis_2018 data/CholecSeg8k data/SAR-RARP50
python scripts/failure_rate.py results data/endovis_2017 data/endovis_2018 data/CholecSeg8k data/SAR-RARP50
python scripts/visibility_analysis.py results data/endovis_2017 data/endovis_2018 data/CholecSeg8k data/SAR-RARP50
```

## Reproduce the runs (GPU)

1. Clone SAM3-DMS (`git clone https://github.com/FudanCVL/SAM3-DMS third_party/SAM3-DMS`
   and check out `88475d9`, or set `SAM3_DMS_ROOT`), and put the SAM3 weights at
   `checkpoints/sam3.pt` (SHA-256 `9999e234…c3cf8c9e`; or set `SAM3_CHECKPOINT`).
   Vanilla `sam3` is not needed.
2. Put the data under `data/` (or set `D17`, `D18`, `DCH`):
   `data/endovis_2017` (EndoVis 2017 test and training releases),
   `data/endovis_2018` (EndoVis 2018 training releases 1–2 and test data),
   `data/CholecSeg8k` (CholecSeg8k, as released),
   `data/SAR-RARP50` (`python scripts/prepare_sarrarp.py ZIP_DIR data/SAR-RARP50`).
3. Run the job lists, e.g.

```bash
bash experiments/run_all.sh 7          # DM, controls, factorial, capacity (gen_jobs.sh)
bash experiments/run_oracle.sh 7       # fixed-rate, oracle and low-capacity controls
bash experiments/gen_dev_jobs.sh       # development runs  -> dev_jobs.txt
bash experiments/gen_test_jobs.sh      # test runs         -> test_jobs.txt
bash experiments/run_controls.sh 5     # controls for the two-path tracker
bash experiments/run_rev.sh 5          # timing, top-6 bank, PNG input, rho = 0.25 / 0.75
bash experiments/run_sar.sh 4          # SAR-RARP50 (preregistered), then fusion
```

Each line of a job list is one call of `src/inference_ev17.py`,
`src/inference_ev18.py` or `src/inference_extra.py` (CholecSeg8k and the
EndoVis 2017 training split); `experiments/job.sh NAME SCRIPT ARGS...` runs a
single line, and `experiments/worker.sh QUEUE` works through a job list. Key
options:

| Option | Meaning |
|---|---|
| `--config C0` | SAM3 unchanged |
| `--config C2` | DM, threshold `--tau-rel` (default 0.95; `2.0` deletes all memory, `-1` keeps all and logs scores) |
| `--config C2-quant --prune-rate 0.5` | rank rule (online, causal) |
| `--config C2-perm --prune-mask-file F --prune-seed s` | random deletion at the per-sequence counts in `F` |
| `--config C2-shift` / `C2-even` | shifted / evenly spaced deletion at the same counts |
| `--config C2-replay --prune-mask-file F` | delete exactly the entries listed in `F` (offline rules) |
| `--config C2-topk` | top-k bank: keep the k = 6 highest-scoring entries (ties: evict the oldest) |
| `--frame-format png` | lossless frames for SAM3 instead of JPEG quality 95 |
| `--num-maskmem k` | memory capacity (k−1 non-conditioning slots) |
| `--save-preds` | save predicted label maps (needed for fusion) |

Fusion of runs by SAM3's score (the two-path tracker and its online variant):

```bash
python scripts/fuse_paths.py --dataset ev18 --data-root data/endovis_2018 \
    --run results/EV18all_C2_delall_sp --run results/EV18all_Q50_sp --out results/EV18all_FUSE-delall-Q50
```

All reported runs used one NVIDIA RTX PRO 6000 Blackwell GPU with PyTorch 2.8.0
(CUDA 12.8), on which every configuration is deterministic. Long sequences are
sensitive to GPU model and input format (Supplementary S6). Alone on that GPU,
SAM3 tracks EndoVis 2017 at 129 ms per frame (peak 9.35 GiB); the retention
rules add no measurable time, so the two-path tracker costs two passes.

## Layout

```
src/models/md_sam3.py       retention rules, controls, replay, rank rule (CONFIG_PRESETS)
src/inference_ev17.py       EndoVis 2017 test driver (instrument types)
src/inference_ev18.py       EndoVis 2018 driver (instrument parts)
src/inference_extra.py      CholecSeg8k, SAR-RARP50 and EndoVis 2017 training split
src/data/                   dataset loaders (extra_datasets.py: CholecSeg8k clip joining)
src/evaluation/metrics.py   bIoU, mcIoU*, class-aware IoU
scripts/                    statistics, offline deletion lists, fusion
experiments/                job lists and the queue used for every run
results/                    every run's metrics, per-sequence results, decisions, logs
```

## Data sources

* EndoVis 2017 (Robotic Instrument Segmentation): https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org
* EndoVis 2018 (Robotic Scene Segmentation): https://endovissub2018-roboticscenesegmentation.grand-challenge.org
* CholecSeg8k: https://www.kaggle.com/datasets/newslab/cholecseg8k
* SAR-RARP50: training release https://doi.org/10.5522/04/24932529.v1, test release https://doi.org/10.5522/04/24932499.v1

## License and data

Code: see `LICENSE`. EndoVis 2017/2018 are available from the challenge
organizers; CholecSeg8k and SAR-RARP50 are released under CC BY-NC-SA 4.0. Neither data nor
SAM3 weights are redistributed here.
