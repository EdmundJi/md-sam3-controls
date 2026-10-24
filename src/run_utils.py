"""Shared helpers for the EV17/EV18 inference scripts (rate-matched controls,
config snapshots, retention-decision logs, saved predictions)."""
import dataclasses
import json
import os

import cv2
import numpy as np


def add_control_args(p):
    p.add_argument("--prune-seed", type=int, default=None,
                   help="RNG seed for random/perm/shift controls")
    p.add_argument("--prune-mask-file", default=None,
                   help="reference run's retention_decisions.json (perm/shift)")
    p.add_argument("--frame-format", default=None, choices=["jpg95", "png"],
                   help="frames handed to SAM3: lossy JPEG q95 (default) or lossless")
    p.add_argument("--save-preds", action="store_true",
                   help="save every predicted label map as PNG under preds/")
    p.add_argument("--num-maskmem", type=int, default=None,
                   help="override SAM3 num_maskmem (capacity sweep; default 7)")
    p.add_argument("--quantile-window", type=int, default=None,
                   help="'quantile' mode: number of earlier scores in the reference set (0 = all)")


def apply_control_args(cfg, args):
    if args.prune_seed is not None:
        cfg.prune_seed = args.prune_seed
    if args.prune_mask_file is not None:
        cfg.prune_mask_file = args.prune_mask_file
    if args.frame_format is not None:
        cfg.frame_format = args.frame_format
    if getattr(args, "quantile_window", None) is not None:
        cfg.quantile_window = args.quantile_window
    return cfg


def full_config(cfg, args=None):
    d = dataclasses.asdict(cfg)
    d["num_maskmem_override"] = (getattr(args, "num_maskmem", None)
                                 or os.environ.get("MDSAM3_NUM_MASKMEM"))
    return d


def dump_decisions(inferencer, out_dir, logger):
    """Write per-sequence retention decisions so controls can replay them."""
    rl = getattr(inferencer, "reinject_log", None)
    if rl:
        n_inj = sum(len(r["injected"]) for r in rl)
        n_read = sum(len(r["read_within_5"]) for r in rl)
        logger.info(f"  re-injection: {len(rl)} events, {n_inj} entries re-inserted, "
                    f"{n_read} read within 5 frames")
        with open(os.path.join(out_dir, "reinjection.json"), "w") as f:
            json.dump({"events": len(rl), "reinserted": n_inj, "read_within_5": n_read,
                       "log": rl}, f)
    seqs = getattr(inferencer, "decisions", {}) or {}
    if not seqs:
        return
    summary = {}
    for tag, d in seqs.items():
        n, k = len(d["tested"]), len(d["deleted"])
        summary[tag] = {"tested": n, "deleted": k,
                        "rate": round(k / n, 4) if n else 0.0}
        logger.info(f"  retention {tag}: {k}/{n} deleted ({summary[tag]['rate']*100:.1f}%)")
    with open(os.path.join(out_dir, "retention_decisions.json"), "w") as f:
        json.dump({"mode": inferencer.cfg.prune_mode,
                   "prune_seed": inferencer.cfg.prune_seed,
                   "summary": summary, "sequences": seqs}, f)


def save_pred(out_dir, seq_tag, frame_id, pred):
    d = os.path.join(out_dir, "preds", str(seq_tag))
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, f"{int(frame_id):05d}.png"), pred.astype(np.uint8))
