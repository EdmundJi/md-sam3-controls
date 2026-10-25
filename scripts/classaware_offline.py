"""Class-aware metrics (mcIoU*, Challenge IoU) recomputed offline from saved
predictions, for single runs and for runs fused by SAM3's score (two-path
tracker = per-video rule, online variant = per-frame rule; same rules as
scripts/fuse_paths.py). Uses the evaluators of the inference drivers, so a
single run reproduces the class-aware numbers in its metrics.json.

usage: python scripts/classaware_offline.py RESULTS_DIR D17 D18 DCH
writes RESULTS_DIR/classaware_offline.json
"""
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import endovis17 as e17  # noqa: E402
from src.data import endovis18 as e18  # noqa: E402
from src.data.extra_datasets import cholec_sequences, extract_class_masks as cholec_classes  # noqa: E402
from src.evaluation.metrics import ClassAwareIoU, EndoVis17Evaluator  # noqa: E402


def ev18_evaluator():
    """EndoVis18Evaluator from src/inference_ev18.py without importing torch/SAM3."""
    src = (ROOT / "src/inference_ev18.py").read_text()
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.ClassDef) and n.name == "EndoVis18Evaluator")
    ns = {"np": np, "ClassAwareIoU": ClassAwareIoU, "CLASS_IDS": e18.CLASS_IDS,
          "CLASS_ABBR": e18.CLASS_ABBR, "Dict": Dict, "List": List}
    exec(ast.get_source_segment(src, node), ns)
    return ns["EndoVis18Evaluator"]()


def seqs(dataset, root):
    """Yield (eval_id, tag, frame_ids, gt_loader, init_classes, H, W) as the drivers do."""
    if dataset == "ev17":
        for s in e17.get_test_sequences(root):
            H, W = cv2.imread(s["frame_files"][0]).shape[:2]
            load = (lambda s, H, W: lambda fid: e17.load_gt_mask(s["gt_dir"], fid, H, W))(s, H, W)
            init = load(s["init_frame_id"])
            yield s["dataset_id"], f"ds{s['dataset_id']}", s["frame_ids"], load, \
                sorted(e17.extract_class_masks(init)), H, W
    elif dataset == "ev18":
        for s in e18.get_sequences(root, "all"):
            H, W = cv2.imread(s["frame_files"][0]).shape[:2]

            def load(fid, s=s, H=H, W=W):
                p = s["label_map"].get(fid)
                return None if p is None else e18.load_label_mask(p, H, W)
            init = load(s["init_frame_id"])
            if init is None or not e18.extract_class_masks(init):
                continue      # not initializable: excluded, as in the driver
            yield s["seq_id"], s.get("seq_tag", f"test{s['seq_id']}"), s["frame_ids"], load, \
                sorted(e18.extract_class_masks(init)), H, W
    else:
        for s in cholec_sequences(root, "test"):
            H, W = cv2.imread(s["frame_files"][0]).shape[:2]
            load = (lambda s, H, W: lambda fid: s["gt"](fid, H, W))(s, H, W)
            yield s["dataset_id"], s["seq_tag"], s["frame_ids"], load, \
                sorted(cholec_classes(load(s["init_frame_id"]))), H, W


def scores(run_dir, tag):
    d = json.load(open(run_dir / "retention_decisions.json"))["sequences"][tag]
    return dict(zip(d["tested"], d["scores"]))


def evaluate(dataset, root, paths, rule=None):
    ev = ev18_evaluator() if dataset == "ev18" else EndoVis17Evaluator()
    for eid, tag, fids, load, init_cls, H, W in seqs(dataset, root):
        if not (paths[0] / "preds" / tag).is_dir():
            continue
        ev.set_init_classes(eid, init_cls)
        if rule:
            S = [scores(p, tag) for p in paths]
            seq_pick = int(np.argmax([np.mean(list(s.values())) for s in S]))
        for t in range(1, len(fids)):
            gt = load(fids[t])
            if gt is None:
                continue
            k = 0 if not rule else (seq_pick if rule == "seq" else int(np.argmax([s.get(t, -1.0) for s in S])))
            p = cv2.imread(str(paths[k] / "preds" / tag / f"{fids[t]:05d}.png"), cv2.IMREAD_GRAYSCALE)
            ev.add_frame(eid, fids[t], np.zeros((H, W), np.uint8) if p is None else p, gt)
    m = ev.compute()
    return {k: m.get(k) for k in ("biou", "mciou_instantiable", "ciou_cls_gt", "ciou_cls_union", "num_frames")}


def add_diffs(out):
    """Differences from SAM3 (C0) for every rule and metric (quoted in the paper)."""
    for ds, rr in out.items():
        base = rr["C0"]
        rr["diff_vs_C0"] = {k: {m: round(v[m] - base[m], 2) for m in ("biou", "mciou_instantiable", "ciou_cls_gt",
                                                                     "ciou_cls_union")}
                            for k, v in rr.items() if k not in ("C0", "diff_vs_C0")}
    return out


def main():
    if sys.argv[1] == "--diffs-only":       # recompute only the differences of an existing file
        f = Path(sys.argv[2]) / "classaware_offline.json"
        f.write_text(json.dumps(add_diffs(json.loads(f.read_text())), indent=1))
        return
    R = Path(sys.argv[1])
    roots = {"ev17": sys.argv[2], "ev18": sys.argv[3], "cholec": sys.argv[4]}
    runs = {
        "ev17": {"C0": ["EV17_C0_ref"], "DM": ["EV17_C2_sp"], "delall": ["EV17_C2_delall_sp"],
                 "Q50": ["EV17_Q50_sp"], "topk": ["EV17_topk"]},
        "ev18": {"C0": ["EV18all_C0_ref"], "DM": ["EV18all_C2_sp"], "delall": ["EV18all_C2_delall_sp"],
                 "Q50": ["EV18all_Q50_sp"], "topk": ["EV18all_topk"]},
        "cholec": {"C0": ["TESTcholecL_keep"], "DM": ["TESTcholecL_DM"], "delall": ["TESTcholecL_delall"],
                   "Q50": ["TESTcholecL_Q50"], "topk": ["TESTcholecL_topk"]},
    }
    out = {}
    for ds, rr in runs.items():
        out[ds] = {}
        for name, dirs in rr.items():
            if (R / dirs[0] / "preds").is_dir():     # top-6 bank was added later
                out[ds][name] = evaluate(ds, roots[ds], [R / d for d in dirs])
        two = [R / rr["delall"][0], R / rr["Q50"][0]]
        out[ds]["method"] = evaluate(ds, roots[ds], two, "seq")
        out[ds]["online"] = evaluate(ds, roots[ds], two, "frame")
        print(ds, json.dumps(out[ds]), flush=True)
    json.dump(add_diffs(out), open(R / "classaware_offline.json", "w"), indent=1)


if __name__ == "__main__":
    main()
