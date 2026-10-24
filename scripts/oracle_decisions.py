"""Deletion lists for --config C2-replay: oracle, anti-oracle and score-ranked.

Frame quality q_t is the mean per-object IoU of the baseline's (C0) saved
prediction at frame t against ground truth, the ground-truth analogue of the
group-mean score G_t (objects absent from both prediction and ground truth are
skipped; a frame with none left counts as perfect; an unlabeled frame takes the
previous frame's quality). Rankings break ties by frame
index.

usage (on the GPU host, from the repo root):
  python scripts/oracle_decisions.py --dataset ev17 --data-root D \
      --c0 R/EV17_C0_ref --dm R/EV17_C2_ref/retention_decisions.json \
      [--scores R/EV17_C0_scores/retention_decisions.json] --out R/oracle/EV17

writes into --out (same format as retention_decisions.json):
  keepall.json      DM's tested frames, nothing deleted (records C0's scores)
  oracle_dm.json    delete the k_s lowest-quality frames, k_s = DM's count
  anti_dm.json      delete the k_s highest-quality frames
  oracle_r50.json   delete the floor(n_s/2) lowest-quality frames
  anti_r50.json     delete the floor(n_s/2) highest-quality frames
  score_r50.json    delete the floor(n_s/2) lowest-scoring frames (needs --scores)
  *_r25.json, *_r75.json  the same at floor(n_s/4) and floor(3 n_s/4) (rate sensitivity)
  quality.json      per-sequence tested frames, quality and score
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def frame_quality(pred, gt, classes):
    ious = []
    for c in classes:
        p, g = pred == c, gt == c
        u = np.logical_or(p, g).sum()
        if u:
            ious.append(np.logical_and(p, g).sum() / u)
    return float(np.mean(ious)) if ious else 1.0


def sequences(dataset, root):
    """Yield (tag, frame_ids, gt_loader, init_classes)."""
    if dataset == "ev17":
        from src.data.endovis17 import get_test_sequences, load_gt_mask, extract_class_masks
        for s in get_test_sequences(root):
            H, W = cv2.imread(s["frame_files"][0]).shape[:2]
            load = (lambda s, H, W: lambda fid: load_gt_mask(s["gt_dir"], fid, H, W))(s, H, W)
            init = load(s["init_frame_id"])
            yield f"ds{s['dataset_id']}", s["frame_ids"], load, sorted(extract_class_masks(init))
    elif dataset in ("ev17train", "cholec"):
        from src.data.extra_datasets import cholec_sequences, ev17_train_sequences, extract_class_masks
        for s in (ev17_train_sequences(root) if dataset == "ev17train" else cholec_sequences(root, "all")):
            H, W = cv2.imread(s["frame_files"][0]).shape[:2]
            load = (lambda s, H, W: lambda fid: s["gt"](fid, H, W))(s, H, W)
            yield s["seq_tag"], s["frame_ids"], load, sorted(extract_class_masks(load(s["init_frame_id"])))
    else:
        from src.data.endovis18 import get_sequences, load_label_mask, extract_class_masks
        for s in get_sequences(root, "all"):
            H, W = cv2.imread(s["frame_files"][0]).shape[:2]

            def load(fid, s=s, H=H, W=W):
                p = s["label_map"].get(fid)
                return None if p is None else load_label_mask(p, H, W)
            init = load(s["init_frame_id"])
            if init is None:
                continue
            yield s["seq_tag"], s["frame_ids"], load, sorted(extract_class_masks(init))


def ranks(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    xs = np.asarray(x)[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def write_json(path, obj):
    # atomic: running jobs may read these files while they are rewritten
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def dump(path, mode, seqs):
    summary = {t: {"tested": len(d["tested"]), "deleted": len(d["deleted"]),
                   "rate": round(len(d["deleted"]) / len(d["tested"]), 4) if d["tested"] else 0.0}
               for t, d in seqs.items()}
    write_json(path, {"mode": mode, "summary": summary, "sequences": seqs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ev17", "ev18", "ev17train", "cholec"], required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--c0", required=True, help="C0 run directory with preds/")
    ap.add_argument("--dm", required=True, help="DM retention_decisions.json (tested frames, counts)")
    ap.add_argument("--scores", default=None, help="retention_decisions.json of the keepall run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dm = json.load(open(a.dm))["sequences"]
    sc = json.load(open(a.scores))["sequences"] if a.scores else None

    RATES = {"r25": 0.25, "r50": 0.5, "r75": 0.75}
    qual, arms = {}, {k: {} for k in ["keepall", "oracle_dm", "anti_dm"] +
                      [f"{a}_{r}" for r in RATES for a in ("oracle", "anti", "score")]}
    diag = {}
    for tag, fids, load, classes in sequences(a.dataset, a.data_root):
        if tag not in dm:
            continue
        tested = sorted(dm[tag]["tested"])
        q, prev = {}, 1.0
        for t in tested:
            pred = cv2.imread(str(Path(a.c0) / "preds" / tag / f"{fids[t]:05d}.png"), cv2.IMREAD_GRAYSCALE)
            gt = load(fids[t])
            assert pred is not None, (tag, t)
            # unlabeled frame (EV18 test3's last frame only): carry the previous quality
            q[t] = prev = frame_quality(pred, gt, classes) if gt is not None else prev
        k_dm, k50 = len(dm[tag]["deleted"]), len(tested) // 2
        worst = sorted(tested, key=lambda t: (q[t], t))
        best = sorted(tested, key=lambda t: (-q[t], t))
        arms["keepall"][tag] = {"tested": tested, "deleted": []}
        arms["oracle_dm"][tag] = {"tested": tested, "deleted": sorted(worst[:k_dm])}
        arms["anti_dm"][tag] = {"tested": tested, "deleted": sorted(best[:k_dm])}
        kr = {r: int(len(tested) * f) for r, f in RATES.items()}
        assert kr["r50"] == k50
        for r, k in kr.items():
            arms[f"oracle_{r}"][tag] = {"tested": tested, "deleted": sorted(worst[:k])}
            arms[f"anti_{r}"][tag] = {"tested": tested, "deleted": sorted(best[:k])}
        rec = {"tested": tested, "quality": [round(q[t], 6) for t in tested]}
        n = len(tested)
        dmdel = set(dm[tag]["deleted"])
        diag[tag] = {"n": n, "k_dm": k_dm, "mean_q": round(float(np.mean(list(q.values()))), 4),
                     "dm_overlap_oracle": len(dmdel & set(worst[:k_dm])),
                     "chance_overlap": round(k_dm * k_dm / n, 1) if n else 0}
        if sc is not None:
            assert sorted(sc[tag]["tested"]) == tested, tag
            s = dict(zip(sc[tag]["tested"], sc[tag]["scores"]))
            low = sorted(tested, key=lambda t: (s[t], t))
            for r, k in kr.items():
                arms[f"score_{r}"][tag] = {"tested": tested, "deleted": sorted(low[:k])}
            rec["score"] = [s[t] for t in tested]
            diag[tag]["spearman_score_quality"] = spearman([s[t] for t in tested], [q[t] for t in tested])
            diag[tag]["score_r50_overlap_oracle"] = len(set(low[:k50]) & set(worst[:k50]))
            diag[tag]["chance_overlap_r50"] = round(k50 * k50 / n, 1) if n else 0
        qual[tag] = rec

    for name, seqs in arms.items():
        if seqs:
            dump(out / f"{name}.json", name, seqs)
    write_json(out / "quality.json", {"sequences": qual, "diagnostics": diag})
    for tag, d in diag.items():
        print(tag, d)


if __name__ == "__main__":
    main()
