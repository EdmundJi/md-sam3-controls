#!/usr/bin/env python3
"""Every confidence interval quoted in the paper, recomputed from ``results/``.

    python scripts/stats.py

Paired per-sequence bootstrap on bIoU: 20,000 resamples, frame-weighted,
``random.seed(0)`` re-applied per contrast so each interval is independent of
the others. No GPU needed.

Endpoints can differ from the paper by up to ~0.05 pp: the paper's intervals
came from a single Monte-Carlo stream shared across contrasts, so their exact
second decimal depends on evaluation order. Point estimates, win/loss/tie
records and every significance verdict are unaffected.

On EndoVis 2018 the bootstrap is **reported but not used as evidence**. With
n=3 the percentile interval's endpoints are just the smallest and largest of
the three per-sequence deltas, which is why the paper relies on leave-one-out
there; both are printed below so the degeneracy is visible.
"""

import csv
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_BOOT = 20_000

# Frames per sequence, used to weight the per-sequence means. EV17: DS1-DS8 are
# 75-frame sequences (74 scored, the init frame is excluded), DS9/DS10 are 300.
EV17_W = {i: (74 if i <= 8 else 299) for i in range(1, 11)}
EV18_W = {1: 249, 2: 248, 3: 248}


def per_seq(name, key="dataset_id"):
    path = RESULTS / f"{name}_seed1" / "per_video.csv"
    if not path.exists():
        return None
    with path.open() as f:
        return {int(r[key]): float(r["iou"]) for r in csv.DictReader(f)}


def wmean(vals, weights):
    keys = [k for k in vals if k in weights]
    return sum(vals[k] * weights[k] for k in keys) / sum(weights[k] for k in keys)


def bootstrap(a, b, w, n=N_BOOT):
    # Reseed per contrast so each interval is independent of how many other
    # contrasts ran before it. Without this, adding a row upstream shifts the
    # Monte-Carlo stream and perturbs every interval below it.
    random.seed(0)
    keys = sorted(k for k in a if k in b and k in w)
    point = sum((a[k] - b[k]) * w[k] for k in keys) / sum(w[k] for k in keys)
    deltas = []
    for _ in range(n):
        s = [random.choice(keys) for _ in keys]
        deltas.append(sum((a[k] - b[k]) * w[k] for k in s) / sum(w[k] for k in s))
    deltas.sort()
    win = sum(1 for k in keys if a[k] - b[k] > 1e-9)
    loss = sum(1 for k in keys if b[k] - a[k] > 1e-9)
    return point, deltas[int(.025 * n)], deltas[int(.975 * n)], win, loss, len(keys) - win - loss


def table(title, contrasts, w, key="dataset_id", note=None):
    print("=" * 84)
    print(title)
    if note:
        print(note)
    print("=" * 84)
    for x, y, label in contrasts:
        a, b = per_seq(x, key), per_seq(y, key)
        if a is None or b is None:
            print(f"  {label:<44} (missing)")
            continue
        pt, lo, hi, wn, ls, ti = bootstrap(a, b, w)
        flag = "" if (lo > 0 or hi < 0) else "   <- spans 0"
        print(f"  {label:<44}{pt:+6.2f} pp  [{lo:+6.2f},{hi:+6.2f}]  "
              f"W/L/T={wn}/{ls}/{ti}{flag}")
    print()


def leave_one_out(contrasts):
    print("=" * 84)
    print("EV18 leave-one-sequence-out (the statistic the paper actually uses)")
    print("=" * 84)
    for x, y, label in contrasts:
        a, b = per_seq(x, "seq_id"), per_seq(y, "seq_id")
        if not a or not b:
            continue
        full = wmean(a, EV18_W) - wmean(b, EV18_W)
        parts = []
        for drop in sorted(EV18_W):
            w = {k: v for k, v in EV18_W.items() if k != drop}
            parts.append(f"drop seq_{drop}: {wmean(a, w) - wmean(b, w):+6.2f}")
        print(f"  {label:<28}full {full:+6.2f} | " + " | ".join(parts))
    print()


def per_sequence_detail():
    print("=" * 84)
    print("EV17 per-sequence bIoU, baseline vs retention vs its rate-matched control")
    print("=" * 84)
    c0, c2, rnd = per_seq("EV17_C0"), per_seq("EV17_C2"), per_seq("EV17_C2-random")
    if not (c0 and c2 and rnd):
        return
    print(f"  {'seq':<6}{'weight':>8}{'C0':>9}{'C2':>9}{'random':>9}"
          f"{'C2-C0':>9}{'C2-random':>11}")
    print("  " + "-" * 62)
    for k in sorted(EV17_W):
        print(f"  DS{k:<4}{EV17_W[k]:>8}{c0[k]:9.2f}{c2[k]:9.2f}{rnd[k]:9.2f}"
              f"{c2[k] - c0[k]:+9.2f}{c2[k] - rnd[k]:+11.2f}")
    print()


def main():

    table("EV17 bIoU, paired bootstrap over 10 sequences", [
        # Does the gain come from WHICH frames are kept, or merely how many?
        ("EV17_C2-random", "EV17_C0", "random pruning vs no pruning"),
        ("EV17_C2", "EV17_C0", "retention vs no pruning"),
        ("EV17_C2", "EV17_C2-random", "retention vs rate-matched random"),
        ("EV17_C2", "EV17_C2-stride", "retention vs rate-matched stride"),
        ("EV17_C2-stride", "EV17_C0", "stride pruning vs no pruning"),
        ("EV17_C2-stride", "EV17_C2-random", "stride vs random pruning"),
        ("EV17_C4-SRG", "EV17_C4-SRG-random", "C4+SRG vs its rate-matched random"),
        ("EV17_C4-SRG", "EV17_C4-SRG-stride", "C4+SRG vs its rate-matched stride"),
        ("EV17_C2-SRG", "EV17_C2-SRG-random", "C2+SRG vs its rate-matched random"),
        ("EV17_C2-SRG", "EV17_C2-SRG-stride", "C2+SRG vs its rate-matched stride"),
        # Which configuration wins
        ("EV17_C2-SRG", "EV17_C2", "C2+SRG vs retention alone"),
        ("EV17_C2-SRG", "EV17_C0", "C2+SRG vs baseline"),
        ("EV17_C1", "EV17_C0", "DMS alone vs baseline"),
        # Deconfounding DMS: propagation or gating granularity?
        ("EV17_C4-GG", "EV17_C2", "independent propagation alone"),
        ("EV17_C4", "EV17_C4-GG", "gating granularity alone"),
    ], EV17_W)

    table("EV18 bIoU, paired bootstrap over 3 sequences", [
        ("EV18_C2", "EV18_C0", "retention vs baseline"),
        ("EV18_C4-SRG", "EV18_C2", "C4+SRG vs retention"),
        ("EV18_C2-SRG", "EV18_C0", "C2+SRG (EV17 winner) vs baseline"),
        ("EV18_C4-GG", "EV18_C2", "independent propagation alone"),
        ("EV18_C4", "EV18_C4-GG", "gating granularity alone"),
    ], EV18_W, key="seq_id",
        note="DEGENERATE at n=3: the endpoints below are just the extreme\n"
             "per-sequence deltas. Reported for completeness, not as evidence.")

    leave_one_out([
        ("EV18_C4-SRG", "EV18_C2", "C4+SRG vs retention"),
        ("EV18_C4", "EV18_C4-GG", "gating granularity"),
        ("EV18_C2", "EV18_C0", "retention vs baseline"),
    ])

    per_sequence_detail()


if __name__ == "__main__":
    main()
