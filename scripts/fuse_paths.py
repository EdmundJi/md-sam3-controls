"""Fuse runs of the same sequences by SAM3's own score (no ground truth used).

Each run must have saved predictions and logged per-frame scores
(retention_decisions.json, field "scores"). Two rules:
  frame  at every frame take the prediction of the run whose entry score at
         that frame is highest (ties: the first run)
  seq    per sequence take the run with the highest mean score
Writes per_video.json / metrics.json for each rule into OUT_frame and OUT_seq,
so the result can be analysed like any other run (bIoU as in the pipeline).

usage (GPU host, repo root):
  python scripts/fuse_paths.py --dataset ev17 --data-root D \
      --run R/EV17_C0_scores:R/EV17_C0_ref --run R/EV17_C2_delall_sp --out R/EV17_FUSE-keep-delall
  (--run SCORES_DIR[:PREDS_DIR]; PREDS_DIR defaults to SCORES_DIR)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_decisions import sequences  # noqa: E402
from src.evaluation.metrics import compute_iou  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ev17", "ev18", "ev17train", "cholec", "sarrarp"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = []
    for spec in a.run:
        sdir, pdir = (spec.split(":") + [None])[:2]
        dec = json.load(open(Path(sdir) / "retention_decisions.json"))["sequences"]
        runs.append((dec, Path(pdir or sdir) / "preds"))

    rows = {"frame": [], "seq": []}
    picks = {"frame": [0] * len(runs), "seq": [0] * len(runs)}
    for tag, fids, load, _ in sequences(a.dataset, a.data_root):
        if any(tag not in dec for dec, _ in runs):
            continue
        S = [dict(zip(dec[tag]["tested"], dec[tag]["scores"])) for dec, _ in runs]
        seq_pick = int(np.argmax([np.mean(list(s.values())) for s in S]))
        picks["seq"][seq_pick] += 1
        ious = {"frame": [], "seq": []}
        for t in range(1, len(fids)):
            gt = load(fids[t])
            if gt is None:
                continue
            P = [cv2.imread(str(p / tag / f"{fids[t]:05d}.png"), cv2.IMREAD_GRAYSCALE) for _, p in runs]
            assert all(x is not None for x in P), (tag, t)
            fp = int(np.argmax([s.get(t, -1.0) for s in S]))
            picks["frame"][fp] += 1
            g = (gt > 0).astype(np.uint8)
            ious["frame"].append(compute_iou((P[fp] > 0).astype(np.uint8), g))
            ious["seq"].append(compute_iou((P[seq_pick] > 0).astype(np.uint8), g))
        for rule in rows:
            rows[rule].append({"seq_tag": tag, "challenge_iou": round(float(np.mean(ious[rule])) * 100, 2),
                               "num_frames": len(ious[rule]), "_sum": float(np.sum(ious[rule]))})
    for rule, rr in rows.items():
        out = Path(f"{a.out}_{rule}")
        out.mkdir(parents=True, exist_ok=True)
        n = sum(r["num_frames"] for r in rr)
        agg = round(100 * sum(r.pop("_sum") for r in rr) / n, 2)
        json.dump(rr, open(out / "per_video.json", "w"), indent=1)
        json.dump({"challenge_iou": agg, "biou": agg, "num_frames": n, "picks": picks[rule],
                   "runs": a.run, "rule": rule}, open(out / "metrics.json", "w"), indent=1)
        print(f"{out.name}: bIoU {agg:.2f} over {n} frames; picks per run {picks[rule]}")


if __name__ == "__main__":
    main()
