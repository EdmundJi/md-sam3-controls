# Rate-matched controls for memory retention in surgical video segmentation

Code and results for *"Does the Score Know Which Frames to Keep? Rate-Matched
Controls for Memory Retention in Surgical Video Segmentation."*

Memory-based video object segmentation has to decide which past frames to keep.
SAM3 makes that decision with a reliability score averaged over all tracked
objects, and several recent surgical systems modify exactly this machinery and
report gains. But a rule that discards most frames changes two things at once —
*which* frames survive and *how many* — and an ablation of "with rule" against
"without" cannot tell them apart. This repository contains the controls that
can: we re-run each mechanism at **its own measured deletion rate** with the
selection signal removed.

Nothing here is trained. Every configuration is a training-free wrapper around
frozen SAM3.

## Verify the paper's numbers in 30 seconds

The results of all 41 runs are committed under `results/`, so every table can be
regenerated with no GPU, no checkpoint, no dataset, and no dependencies beyond
the Python standard library:

```bash
python scripts/tables.py    # Tables 1-3, capacity sweep, the evaluation fault
python scripts/stats.py     # every confidence interval quoted in the paper
```

If a number these print disagrees with the paper, the paper is wrong.

## What the controls show

| Result | Interval |
|---|---|
| Deleting 62.6% of memory **at random** ≈ deleting nothing | +0.15 pp, CI [−0.15, +0.42] |
| Decoupled propagation contributes nothing on its own | −0.07 pp, CI [−0.25, +0.15] |
| Diversity-aware re-injection is inert, even once repaired to actually re-insert | exact identity |
| 6→2 memory slots: baseline −4.6 pp, gated retention −0.01 pp | — |

Everything else in the paper — including retention's headline +2.11 pp — is
reported as an existence proof, not an estimate, because its interval spans
zero. `scripts/stats.py` marks those rows `<- spans 0`.

## Layout

```
src/
  models/md_sam3.py     the three mechanisms + CONFIG_PRESETS (the only bespoke code)
  inference_ev17.py     EndoVis 2017 driver (instrument types)
  inference_ev18.py     EndoVis 2018 driver (instrument parts)
  evaluation/metrics.py bIoU, mcIoU, mcIoU*  (see "Metrics" below)
  data/                 dataset loaders
  paths.py              locates the two external SAM3 repositories
scripts/
  experiments.py        the complete experiment matrix — all 41 runs
  run_all.py            executes it (needs a GPU)
  tables.py             regenerates the paper's tables from results/
  stats.py              paired bootstrap + leave-one-out
results/                the committed output of all 41 runs
```

## Full reproduction

### 1. External repositories

Neither is vendored. Both are needed, and **SAM3-DMS must shadow vanilla SAM3
on `sys.path`** — it supplies the extended `build_sam3_video_model(...,
use_decoupled_selection=...)` signature that every configuration is built
through. `src/paths.py` enforces the ordering; if the vanilla package wins you
get a `TypeError` on startup.

```bash
git clone https://github.com/FudanCVL/SAM3-DMS third_party/SAM3-DMS
git -C third_party/SAM3-DMS checkout 88475d9

git clone https://github.com/facebookresearch/sam3 third_party/sam3
git -C third_party/sam3 checkout 5dd401d
```

Or point `SAM3_DMS_ROOT` / `SAM3_ROOT` at existing clones.

### 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Results were produced on a single NVIDIA A40 with `torch 2.8`. A CUDA build of
PyTorch is assumed to be present; the pins verify compatibility rather than
replace a working install.

### 3. Checkpoint and data

The SAM3 checkpoint (`sam3.pt`, from `facebook/sam3`) goes in `checkpoints/`,
or set `SAM3_CHECKPOINT`. EndoVis data is not redistributable — request it from
the challenge organisers and arrange it as below, or set `ENDOVIS17_ROOT` /
`ENDOVIS18_ROOT`.

```
data/endovis_2017/
  instrument_1_4_testing/instrument_dataset_{1..4}/left_frames/
  instrument_5_8_testing/instrument_dataset_{5..8}/left_frames/
  instrument_9_10_testing/instrument_dataset_{9,10}/left_frames/
  instrument_2017_test/instrument_dataset_{1..10}/TypeSegmentation/
data/endovis_2018/
  test_data_and_label/seq_{1..4}/{left_frames,labels}/
```

EV17 type annotations follow Shvets et al.; class IDs are in
`src/data/endovis17.py`.

### 4. Run

```bash
python scripts/run_all.py --dry-run     # inspect the 41 commands first
python scripts/run_all.py               # run everything missing
python scripts/run_all.py --paper-only  # just the runs behind the tables
python scripts/run_all.py --only EV17_C2 EV17_C2-random
```

Runs that already have a `metrics.json` are skipped, so an interrupted sweep can
be restarted. Inference is deterministic — three seeds gave byte-identical
output — so `--seed 1` alone reproduces everything.

## The experiment matrix

`scripts/experiments.py` is the authoritative list: every run in the paper
appears there exactly once, and nothing else was run. The naming is
`{EV17,EV18}_{config}[_variant]`.

| Config | Meaning |
|---|---|
| `C0` | frozen SAM3 baseline |
| `C1` | DMS — decoupled per-instrument propagation |
| `C2` | DM — relevance-based retention (the ReMeDI-SAM3 rule) |
| `C3`,`C5`,`C6`,`C7` | + DFS; identical to `C0`,`C1`,`C2`,`C4` because DFS cannot act |
| `C4` | DMS + DM |
| `C4-GG` | independent propagation but *group-mean* gating — the deconfound |
| `C2-SRG`, `C4-SRG` | self-calibrated threshold (exploratory), joint / decoupled |
| `C7-store` | DFS repaired to genuinely re-insert evicted entries |

Variants: `-random` / `-stride` are the rate-matched controls, `_nm{7,5,3}` the
memory-capacity sweep (7/5/3 total slots = 6/4/2 non-conditioning),
`_tau{070,050}` the fixed-threshold sweep, `_q{25,75}` the SRG quantile ablation.

The controls take their deletion rate from the run they are matched against;
those rates are recorded in `RATE_MATCHED_FROM` in `scripts/experiments.py` and
are re-derivable from the `RETENTION: pruned N/M` line every run logs.

## Metrics

`metrics.json` carries three numbers, and the difference between two of them is
one of the paper's findings:

- **`iou`** — frame-averaged binary foreground IoU (**bIoU**), the primary metric.
- **`mciou_instantiable`** — **mcIoU\***: mean over classes of each class's
  frame-averaged IoU, restricted to sequences where that class is instantiable
  at frame 0.
- **`mciou`** — the same mean *without* that restriction.

Under GT first-frame initialization a class absent from frame 0 can never be
tracked, so pooling its frames scores guaranteed zeros against it. This is not
uniform across classes: it deflates MCS from 86.67 to 78.43 on the baseline, 8.2
pp on exactly the class the analysis turns on. `python scripts/tables.py` prints
the per-class deflation. Both numbers are kept in every `metrics.json` so the
effect stays inspectable.

**None of these is the class-aware EndoVis Challenge IoU.** No number in this
repository or the paper is comparable to a published Challenge-IoU figure, and
we make no such comparison.

## Things that will bite you

- **Import order** — SAM3-DMS before vanilla SAM3, always. See `src/paths.py`.
- **JPEG transcoding** — frames are written to JPEG (quality 95) before
  inference (`src/models/md_sam3.py`). Every number, baseline included, is
  measured on lossily recompressed input. It is result-affecting and easy to
  drop by accident.
- **EV18 `seq_4` is excluded** — no instrument part in its first frame, so it
  cannot be initialized. Scoring it anyway adds a method-independent constant
  over 248 of 993 frames and compresses every EV18 contrast by about a quarter.
- **n=3 on EV18** — the percentile bootstrap is degenerate: its endpoints are
  just the extreme per-sequence deltas. `scripts/stats.py` prints it labelled as
  such and reports leave-one-out alongside.
- **GT-init is an oracle protocol.** It isolates memory from detection, which is
  the point, but it is not a deployment setting.
- **The random control is one draw, not a mean over draws.** Its RNG is
  `random.Random(12345)`, fixed in `src/models/md_sam3.py` and independent of
  `--seed`, so the control is exactly reproducible but is a single realization
  of "delete 62.6% arbitrarily". Its interval in `scripts/stats.py` is over
  *sequences*, not over random draws. Re-seeding that RNG and averaging would be
  the stronger version of this control, and we did not do it.

## Citation

```bibtex
@inproceedings{anon2027ratematched,
  title     = {Does the Score Know Which Frames to Keep? Rate-Matched Controls
               for Memory Retention in Surgical Video Segmentation},
  booktitle = {International Conference on Multimedia Modeling (MMM)},
  year      = {2027}
}
```

## License

MIT, see `LICENSE`. The external SAM3 and SAM3-DMS repositories carry their own
licenses; neither is redistributed here. EndoVis data is governed by the
challenge organisers' terms.
