# Committed results

The output of all 41 runs behind the paper, one directory per run, named as in
`scripts/experiments.py`. `scripts/tables.py` and `scripts/stats.py` read these,
so every published number can be checked without a GPU.

Each directory holds:

| File | Contents |
|---|---|
| `metrics.json` | aggregate + per-class metrics (see "Metrics" in the top-level README) |
| `per_video.csv` | per-sequence breakdown, the unit the bootstrap resamples over |
| `config.yaml` | configuration snapshot written by the run |

## Two caveats about `config.yaml`

**It does not distinguish the control runs.** These snapshots were written by the
version of the code that produced them, which recorded only the preset and its
core hyperparameters. So `EV17_C2-random/config.yaml` and `EV17_C2/config.yaml`
look identical even though the first deletes frames at random — the
distinguishing flags (`prune_mode`, `prune_rate`, `srg_quantile`, `group_gate`,
`num_maskmem`) were not being serialized. `scripts/experiments.py` is the
authoritative record of what each run was invoked with; the deletion rates are
also recoverable from the `RETENTION: pruned N/M` line each run logged.

The current code serializes all of those fields, so **re-running a
configuration produces a `config.yaml` with more keys than the committed one.**
That difference is expected and is not a discrepancy in the results.

**`model_path` points at the original machine's layout** (`/workspace/...`).
Left as written rather than rewritten, so the snapshot stays a faithful record
of the run.

## Provenance

Produced on a single NVIDIA A40, `torch 2.8`, SAM3-DMS @ `88475d9`, SAM3 @
`5dd401d`, seed 1.

Inference is deterministic: configurations C0–C7 were each run at seeds 1, 2 and
3 and gave identical metrics, so only seed 1 is committed. Note that this does
*not* mean `--seed` varies the random-pruning control — that control draws from
a separate RNG hardcoded to `random.Random(12345)`, so it is one fixed
realization rather than an average over draws (see the top-level README).
