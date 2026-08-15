#!/usr/bin/env python3
"""Regenerate every table in the paper from ``results/``.

    python scripts/tables.py

Reads only the committed ``metrics.json`` files, so it runs anywhere in a
second and needs no GPU, no checkpoint and no data. If a number here disagrees
with the paper, the paper is wrong.

Metric names, as used throughout:

  bIoU     frame-averaged binary foreground IoU  (``iou`` in metrics.json)
  mcIoU*   mean over classes of each class's frame-averaged IoU, restricted to
           sequences where that class is instantiable at frame 0
           (``mciou_instantiable``)
  mcIoU    the same mean *without* that restriction (``mciou``) -- the metric
           the paper argues is faulty, kept here so the deflation is visible.
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
EV17_CLASSES = ["BF", "PF", "LND", "VS", "MCS"]   # the five instantiable ones
EV18_CLASSES = ["shaft", "wrist", "clasper"]


def load(name):
    path = RESULTS / f"{name}_seed1" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def cell(value):
    return "  --  " if value is None else f"{value:6.2f}"


def row(label, name, classes, class_key, mciou_key, width=38):
    m = load(name)
    if m is None:
        return f"  {label:<{width}}" + "  (not in results/)"
    per = m.get(class_key, {})
    cells = "".join(f"{cell(per.get(c)):>8}" for c in classes)
    return (f"  {label:<{width}}{cell(m['iou'])} {cell(m[mciou_key])}  {cells}")


def header(title, classes, width=38):
    print("=" * 78)
    print(title)
    print("=" * 78)
    cols = "".join(f"{c:>8}" for c in classes)
    print(f"  {'Configuration':<{width}}{'bIoU':>6} {'mcIoU*':>6}  {cols}")
    print("  " + "-" * 76)


def table1():
    header("Table 1 -- EndoVis 2017, all configurations (GT-init, 10 seq, 1190 frames)",
           EV17_CLASSES)
    for label, name in [
        ("C0  SAM3 baseline", "EV17_C0"),
        ("C1  DMS", "EV17_C1"),
        ("C2  DM (retention)", "EV17_C2"),
        ("C4  DMS + DM", "EV17_C4"),
        ("    C4-GG (deconfound)", "EV17_C4-GG"),
    ]:
        print(row(label, name, EV17_CLASSES, "per_class_instantiable", "mciou_instantiable"))
    print("  " + "-" * 76)
    print("  self-calibrated threshold (exploratory)")
    for label, name in [
        ("    C4 + SRG (decoupled)", "EV17_C4-SRG"),
        ("    C2 + SRG (joint)", "EV17_C2-SRG"),
    ]:
        print(row(label, name, EV17_CLASSES, "per_class_instantiable", "mciou_instantiable"))
    print()


def table2():
    header("Table 2 -- rate-matched controls on EV17 "
           "(same deletion count, different choice)", ["VS", "MCS"])
    rates = {"EV17_C0": "0%", "EV17_C2-random": "61.9%",
             "EV17_C2-stride": "62.6%", "EV17_C2": "62.6%"}
    for label, name in [
        ("none (C0)", "EV17_C0"),
        ("random, rate-matched", "EV17_C2-random"),
        ("stride, rate-matched", "EV17_C2-stride"),
        ("reliability (C2)", "EV17_C2"),
    ]:
        line = row(f"{label}  [deleted {rates[name]}]", name, ["VS", "MCS"],
                   "per_class_instantiable", "mciou_instantiable")
        print(line)
    print()


def table3():
    header("Table 3 -- EndoVis 2018, seq_1..seq_3 (745 frames; seq_4 uninitializable)",
           EV18_CLASSES)
    for label, name in [
        ("C0 = C3 (baseline)", "EV18_C0"),
        ("C1 = C5 (DMS)", "EV18_C1"),
        ("C2 = C6 (DM)", "EV18_C2"),
        ("C4 = C7 (DMS+DM)", "EV18_C4"),
        ("C4-GG (deconfound)", "EV18_C4-GG"),
        ("C4 + SRG (decoupled)", "EV18_C4-SRG"),
        ("C2 + SRG (joint)", "EV18_C2-SRG"),
    ]:
        print(row(label, name, EV18_CLASSES, "per_class", "mciou"))
    print()


def dfs_identities():
    """Sec 4.4: DFS cannot act, so these pairs must agree exactly."""
    print("=" * 78)
    print("Sec 4.4 -- DFS equivalences (must be EXACT: same program, not a measurement)")
    print("=" * 78)
    ok = True
    for a, b in [("EV18_C0", "EV18_C3"), ("EV18_C1", "EV18_C5"),
                 ("EV18_C2", "EV18_C6"), ("EV18_C4", "EV18_C7")]:
        ma, mb = load(a), load(b)
        if ma is None or mb is None:
            print(f"  {a} vs {b}: (missing)")
            continue
        same = ma["iou"] == mb["iou"] and ma["mciou"] == mb["mciou"]
        ok &= same
        print(f"  {a:12s} {ma['iou']:6.2f} == {mb['iou']:6.2f} {b:12s}  "
              f"{'OK' if same else 'MISMATCH'}")
    print(f"  -> {'all identities hold' if ok else 'IDENTITIES BROKEN'}\n")


def capacity():
    """Sec 4.6: shrink the memory window; gated retention barely notices."""
    print("=" * 78)
    print("Sec 4.6 -- memory capacity (non-conditioning slots: 6 / 4 / 2)")
    print("=" * 78)
    print(f"  {'Configuration':<24}{'6 slots':>9}{'4 slots':>9}{'2 slots':>9}{'delta 6->2':>12}")
    print("  " + "-" * 62)
    for label, base, nm5, nm3 in [
        ("C0 baseline", "EV17_C0", "EV17_C0_nm5", "EV17_C0_nm3"),
        ("C2 retention", "EV17_C2", "EV17_C2_nm5", "EV17_C2_nm3"),
        ("C4+SRG", "EV17_C4-SRG", "EV17_C4-SRG_nm5", "EV17_C4-SRG_nm3"),
        ("C2+SRG", "EV17_C2-SRG", "EV17_C2-SRG_nm5", "EV17_C2-SRG_nm3"),
    ]:
        vals = [load(n) for n in (base, nm5, nm3)]
        if any(v is None for v in vals):
            print(f"  {label:<24}(missing)")
            continue
        a, b, c = (v["iou"] for v in vals)
        print(f"  {label:<24}{a:9.2f}{b:9.2f}{c:9.2f}{c - a:+12.2f}")
    print()


def evaluation_fault():
    """Sec 4.1: pooling a class over sequences where it cannot be instantiated."""
    print("=" * 78)
    print("Sec 4.1 -- the pooling fault, on the baseline (C0)")
    print("=" * 78)
    m = load("EV17_C0")
    if m is None:
        print("  (missing)\n")
        return
    print(f"  {'class':<8}{'pooled':>9}{'restricted':>12}{'deflation':>12}"
          f"{'frames':>9}{'scoreable':>11}")
    print("  " + "-" * 62)
    for c in EV17_CLASSES:
        pooled = m["per_class"][c]
        restricted = m["per_class_instantiable"][c]
        n_all = m["n_frames_per_class"][c]
        n_inst = m["n_frames_per_class_instantiable"][c]
        print(f"  {c:<8}{pooled:9.2f}{restricted:12.2f}{pooled - restricted:+12.2f}"
              f"{n_all:9d}{n_inst:11d}")
    print(f"\n  mcIoU (7 classes, pooled): {m['mciou']:.2f}"
          f"   mcIoU* (5 instantiable):  {m['mciou_instantiable']:.2f}\n")


def main():
    if not RESULTS.exists():
        sys.exit(f"no results/ directory at {RESULTS}")
    table1()
    table2()
    table3()
    dfs_identities()
    capacity()
    evaluation_fault()


if __name__ == "__main__":
    main()
