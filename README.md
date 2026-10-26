# When should SAM3 forget? Rate-matched controls and a two-path tracker

Code, job lists, per-run outputs and analysis scripts for the IPCAI 2027
submission *"When should SAM3 forget? Rate-matched controls and a two-path
tracker for surgical instrument segmentation."*

Nothing here is trained. Every configuration is an inference-only change to the
memory of frozen SAM3 (run through [SAM3-DMS](https://github.com/FudanCVL/SAM3-DMS),
commit `88475d9`), initialized from the ground-truth masks of the first frame.

## Main results (test sets, bIoU %)

| Method | EndoVis 2017 (10 seq.) | EndoVis 2018 (18 seq.) | CholecSeg8k (25 seq.) | Mean gain over SAM3 [95% CI] |
|---|---|---|---|---|
| SAM3 (keep all memory) | 90.28 | 73.71 | 83.00 | — |
| DM, τ = 0.95 (ReMeDI-SAM3 retention) | 91.57 | 74.96 | 84.14 | +1.23 [−0.28, +2.84] |
| First frame only (delete all memory) | 89.69 | 74.98 | 84.32 | +0.66 [−0.96, +2.15] |
| Top-6 bank (keep the 6 highest-scoring entries; comparator added after testing) | 91.57 | 75.16 | 83.62 | +1.12 [−0.01, +2.41] |
| Rank rule, ρ = 0.5 | 91.59 | 76.62 | 83.58 | +1.60 [−0.57, +4.09] |
| **Two-path tracker** (first frame only + rank rule, per video; selected) | 91.55 | 76.96 | 84.47 | +2.00 [−0.25, +4.53] |
| Two-path tracker, per frame (online variant) | 91.61 | 77.54 | 84.10 | +2.09 [−0.04, +4.38] |

The two-path tracker is never more than 0.04 bIoU below the best single rule on
any test set; every single rule (including the top-6 bank) trails the best one
by 0.7 to 1.9 points on at least one set.

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
* Tracking from the first frame alone beats SAM3's full memory on four of five
  datasets (both development sets, EndoVis 2018, CholecSeg8k).
* The rank rule, which ranks entries by the same score instead of thresholding
  it, beats random deletion at its own per-sequence counts on all three test
  sets, above all 20 draws each time (EndoVis 2017 +2.02 [+0.07, +4.88],
  EndoVis 2018 +3.30 [−2.81, +10.31], CholecSeg8k +0.57 [+0.07, +1.44]).
  Replacing the two-path tracker's memory-path deletions by random ones lowers
  it on every test set (19, 19 and 20 of 20 draws below it).

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
  predictions): the two-path tracker raises mcIoU* over SAM3 on all three test
  sets (+1.6, +1.9, +1.3).
* **Runtime**: 129 ms per frame for SAM3 on EndoVis 2017, peak 9.35 GiB; no
  rule adds measurable time.

## Fourth test set, preregistered: SAR-RARP50

Before any model was run on it, the protocol for a fourth test set, the 54 videos of
SAR-RARP50 (robot-assisted radical prostatectomy, masks at 1 Hz), was committed in
[`PREREGISTRATION_SAR-RARP50.md`](PREREGISTRATION_SAR-RARP50.md): data handling,
runs, hypotheses and analysis, with no setting changed from the paper.
`experiments/get_sarrarp.sh` downloads and prepares the data,
`experiments/run_sar.sh` runs the job list `experiments/gen_sar_jobs.sh`.
Results will be added here unchanged, whatever they are.

## Recompute the statistics (no GPU)

All per-run outputs are committed under `results/` (metrics, per-sequence
scores, every retention decision with its score, logs). With Python 3 and numpy:

```bash
python scripts/analysis.py          # DM controls, capacity       -> results/analysis.json
python scripts/oracle_analysis.py   # fixed-rate / oracle / low-capacity controls
python scripts/dev_analysis.py      # development ranking (rule selection)
python scripts/test_analysis.py     # test results of the chosen rule
python scripts/rev_analysis.py      # PNG input, rho sensitivity, top-6 bank, runtime -> results/rev_analysis.json
# class-aware metrics from saved predictions (needs the data and the runs' preds/, which are not committed)
python scripts/classaware_offline.py results data/endovis_2017 data/endovis_2018 data/CholecSeg8k
```

## Reproduce the runs (GPU)

1. Clone SAM3-DMS (`git clone https://github.com/FudanCVL/SAM3-DMS third_party/SAM3-DMS`
   and check out `88475d9`, or set `SAM3_DMS_ROOT`), and put the SAM3 weights at
   `checkpoints/sam3.pt` (SHA-256 `9999e234…c3cf8c9e`; or set `SAM3_CHECKPOINT`).
   Vanilla `sam3` is not needed.
2. Put the data under `data/` (or set `D17`, `D18`, `DCH`):
   `data/endovis_2017` (EndoVis 2017 test and training releases),
   `data/endovis_2018` (EndoVis 2018 training releases 1–2 and test data),
   `data/CholecSeg8k` (CholecSeg8k, as released).
3. Run the job lists, e.g.

```bash
bash experiments/run_all.sh 7          # DM, controls, factorial, capacity (gen_jobs.sh)
bash experiments/run_oracle.sh 7       # fixed-rate, oracle and low-capacity controls
bash experiments/gen_dev_jobs.sh       # development runs  -> dev_jobs.txt
bash experiments/gen_test_jobs.sh      # test runs         -> test_jobs.txt
bash experiments/run_controls.sh 5     # controls for the two-path tracker
bash experiments/run_rev.sh 5          # timing, top-6 bank, PNG input, rho = 0.25 / 0.75
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
src/inference_extra.py      CholecSeg8k and EndoVis 2017 training split
src/data/                   dataset loaders (extra_datasets.py: CholecSeg8k clip joining)
src/evaluation/metrics.py   bIoU, mcIoU*, class-aware IoU
scripts/                    statistics, offline deletion lists, fusion
experiments/                job lists and the queue used for every run
results/                    every run's metrics, per-sequence results, decisions, logs
```

## License and data

Code: see `LICENSE`. EndoVis 2017/2018 are available from the challenge
organizers; CholecSeg8k and SAR-RARP50 are released under CC BY-NC-SA 4.0. Neither data nor
SAM3 weights are redistributed here.
