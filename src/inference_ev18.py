"""
Inference script for EndoVis 2018 evaluation (instrument part segmentation).

Tracks three instrument parts (shaft/clasper/wrist) as separate objects
using the GT first-frame initialization protocol.

Usage:
    python src/inference_ev18.py \\
        --config C7 \\
        --model-path checkpoints/sam3.pt \\
        --data-root data/endovis_2018 \\
        --output-dir results/C0 \\
        --seed 1

Outputs:
    {output_dir}/metrics.json    — CIoU, IoU, mcIoU, per_class (shaft/clasper/wrist)
    {output_dir}/per_video.csv   — Per-sequence breakdown
    {output_dir}/config.yaml     — Experiment config snapshot
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import REPO_ROOT, default_checkpoint, default_data_root  # noqa: E402

from src.data.endovis18 import (
    CLASS_IDS, CLASS_ABBR,
    get_sequences, load_label_mask, extract_class_masks,
)
from src.models.md_sam3 import CONFIG_PRESETS, MDSam3Config, build_md_sam3, MDSam3Inferencer
from src.evaluation.metrics import ClassAwareIoU
from src.run_utils import add_control_args, apply_control_args, full_config, dump_decisions, save_pred

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference_ev18")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="MD-SAM3 EV18 Evaluation")
    p.add_argument("--config", default="C7")
    p.add_argument("--model-path", default=default_checkpoint())
    p.add_argument("--data-root", default=default_data_root("ev18"))
    p.add_argument("--split", default="test", choices=["test", "train", "all"],
                   help="test = seq_1-4 (all earlier runs); train/all add the 15 training sequences")
    p.add_argument("--seq-tags", default="",
                   help="comma-separated seq tags to run (default: every sequence of the split)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dms", action="store_true", default=None)
    p.add_argument("--no-dms", dest="dms", action="store_false")
    p.add_argument("--dm", action="store_true", default=None)
    p.add_argument("--no-dm", dest="dm", action="store_false")
    p.add_argument("--dfs", action="store_true", default=None)
    p.add_argument("--no-dfs", dest="dfs", action="store_false")
    p.add_argument("--tau-rel", type=float, default=None)
    p.add_argument("--adaptive-tau", action="store_true", default=None,
                   help="SRG: per-instance self-calibrated relevance threshold")
    p.add_argument("--adaptive-beta", type=float, default=None)
    p.add_argument("--prune-mode", default=None,
                   choices=["reliability", "random", "stride", "perm", "shift", "even", "replay", "quantile"])
    p.add_argument("--prune-rate", type=float, default=None)
    p.add_argument("--srg-quantile", type=float, default=None)
    add_control_args(p)
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT / "results" / args.config)
    return args


def get_config(args) -> MDSam3Config:
    cfg = CONFIG_PRESETS.get(args.config, MDSam3Config())
    if args.dms is not None:
        cfg.dms = args.dms
    if args.dm is not None:
        cfg.dm = args.dm
    if args.dfs is not None:
        cfg.dfs = args.dfs
    if args.tau_rel is not None:
        cfg.tau_rel = args.tau_rel
    if args.adaptive_tau is not None:
        cfg.adaptive_tau = args.adaptive_tau
    if args.adaptive_beta is not None:
        cfg.adaptive_beta = args.adaptive_beta
    if args.prune_mode is not None:
        cfg.prune_mode = args.prune_mode
    if args.prune_rate is not None:
        cfg.prune_rate = args.prune_rate
    if args.srg_quantile is not None:
        cfg.srg_quantile = args.srg_quantile
    apply_control_args(cfg, args)
    return cfg


class EndoVis18Evaluator:
    """Binary CIoU + per-part mcIoU for EV18."""

    def __init__(self):
        self.results = {}  # {seq_id: {frame_id: (pred_mask, gt_mask)}}
        # Parts present in each sequence's init frame. Under GT-init a part
        # absent from frame 0 is never tracked, so its frames are structural
        # zeros; per_class_instantiable excludes them (as on EV17).
        self.init_classes = {}

    def set_init_classes(self, seq_id, classes):
        self.init_classes[seq_id] = set(int(c) for c in classes)

    def add_frame(self, seq_id, frame_id, pred: np.ndarray, gt: np.ndarray):
        self.results.setdefault(seq_id, {})[frame_id] = (pred, gt)

    def compute(self):
        from src.evaluation.metrics import compute_iou
        all_ciou, all_iou = [], []
        per_class = {c: [] for c in CLASS_IDS}
        per_class_inst = {c: [] for c in CLASS_IDS}
        cls_aware = ClassAwareIoU(CLASS_IDS)

        for seq_id, frames in self.results.items():
            inst = self.init_classes.get(seq_id)
            for fid, (pred, gt) in frames.items():
                pred_bin = (pred > 0).astype(np.uint8)
                gt_bin = (gt > 0).astype(np.uint8)
                ciou = compute_iou(pred_bin, gt_bin)
                all_ciou.append(ciou)
                all_iou.append(ciou)
                cls_aware.add(pred, gt)
                for cls_id in CLASS_IDS:
                    gt_cls = (gt == cls_id).astype(np.uint8)
                    if gt_cls.sum() > 0:
                        pred_cls = (pred == cls_id).astype(np.uint8)
                        v = compute_iou(pred_cls, gt_cls)
                        per_class[cls_id].append(v)
                        if inst is None or cls_id in inst:
                            per_class_inst[cls_id].append(v)

        mean_ciou = np.mean(all_ciou) * 100 if all_ciou else 0.0
        mean_iou = np.mean(all_iou) * 100 if all_iou else 0.0
        class_means = {c: np.mean(per_class[c]) * 100 if per_class[c] else 0.0
                       for c in CLASS_IDS}
        present = [c for c in CLASS_IDS if per_class[c]]
        mciou = np.mean([class_means[c] for c in present]) if present else 0.0

        return {
            "challenge_iou": round(mean_ciou, 2),
            "iou": round(mean_iou, 2),
            "mciou": round(mciou, 2),
            "per_class": {CLASS_ABBR[c]: round(class_means[c], 2) for c in CLASS_IDS},
            "num_frames": len(all_ciou),
            "per_class_instantiable": {CLASS_ABBR[c]: round(float(np.mean(per_class_inst[c])) * 100, 2)
                                       if per_class_inst[c] else 0.0 for c in CLASS_IDS},
            "mciou_instantiable": round(float(np.mean([np.mean(per_class_inst[c]) * 100
                                       for c in CLASS_IDS if per_class_inst[c]])), 2)
                                  if any(per_class_inst[c] for c in CLASS_IDS) else 0.0,
            "n_frames_per_class": {CLASS_ABBR[c]: len(per_class[c]) for c in CLASS_IDS},
            "n_frames_per_class_instantiable": {CLASS_ABBR[c]: len(per_class_inst[c]) for c in CLASS_IDS},
            "biou": round(mean_ciou, 2),
            **cls_aware.result(),
        }


def run_sequence(inferencer: MDSam3Inferencer, seq: dict) -> dict:
    import cv2
    frame_files = seq["frame_files"]
    if not frame_files:
        return {}
    first_img = cv2.imread(frame_files[0])
    H, W = first_img.shape[:2]

    init_frame_id = seq["init_frame_id"]
    init_label_path = seq["label_map"].get(init_frame_id)
    if init_label_path is None:
        logger.warning(f"seq_{seq['seq_id']}: no label for init frame {init_frame_id}")
        return {}

    init_gt = load_label_mask(init_label_path, H, W)
    if init_gt is None:
        return {}
    init_masks = extract_class_masks(init_gt)
    if not init_masks:
        logger.warning(f"seq_{seq['seq_id']}: no instrument parts in init frame")
        return {}

    logger.info(f"seq_{seq['seq_id']}: {len(frame_files)} frames, "
                f"parts={list(init_masks.keys())} {[CLASS_ABBR[c] for c in init_masks]}")

    seq["init_parts"] = sorted(init_masks.keys())
    inferencer.seq_tag = seq.get("seq_tag", f"test{seq['seq_id']}")
    t0 = time.time()
    predictions = inferencer.run_sequence(frame_files, init_masks, 0)
    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s ({len(frame_files)/elapsed:.1f} fps)")
    return predictions


def evaluate_sequence(predictions, seq, evaluator: EndoVis18Evaluator, H, W):
    import cv2
    for local_idx, (fpath, frame_id) in enumerate(
        zip(seq["frame_files"], seq["frame_ids"])
    ):
        if local_idx == 0:
            continue
        label_path = seq["label_map"].get(frame_id)
        if label_path is None:
            continue
        gt = load_label_mask(label_path, H, W)
        if gt is None:
            continue
        pred = predictions.get(local_idx, np.zeros((H, W), dtype=np.uint8))
        evaluator.add_frame(seq["seq_id"], frame_id, pred, gt)


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = get_config(args)
    logger.info(f"Config {args.config}: dms={cfg.dms}, dm={cfg.dm}, dfs={cfg.dfs}")

    os.makedirs(args.output_dir, exist_ok=True)

    import yaml
    config_dict = {
        "experiment_id": args.config,
        "seed": args.seed,
        "model_path": args.model_path,
        "dataset": "endovis2018",
        "model": {
            "components": {"dms": cfg.dms, "dm": cfg.dm, "dfs": cfg.dfs},
            "hyperparams": {
                "tau_rel": cfg.tau_rel,
                "tau_occ": cfg.tau_occ,
                "lambda_dfs": cfg.lambda_dfs,
                "memory_size": cfg.memory_size,
            },
        },
        "split": args.split,
        "full_config": full_config(cfg, args),
    }
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logger.info(f"Loading SAM3 from {args.model_path}")
    inferencer = build_md_sam3(cfg, args.model_path, args.device)
    _nm = args.num_maskmem or os.environ.get('MDSAM3_NUM_MASKMEM')
    if _nm:
        inferencer.predictor.num_maskmem = int(_nm)
        logger.info(f'CAPACITY OVERRIDE: num_maskmem={_nm}')

    sequences = get_sequences(args.data_root, args.split)
    if args.seq_tags:
        keep = set(args.seq_tags.split(","))
        sequences = [s for s in sequences if s.get("seq_tag") in keep]
        logger.info(f"SEQ SUBSET: {sorted(keep)}")
    if not sequences:
        logger.error(f"No {args.split} sequences found in {args.data_root}")
        return
    logger.info(f"Found {len(sequences)} {args.split} sequences: "
                f"{[s.get('seq_tag', s['seq_id']) for s in sequences]}")

    evaluator = EndoVis18Evaluator()
    per_video_rows = []

    for seq in sequences:
        import cv2
        H, W = cv2.imread(seq["frame_files"][0]).shape[:2]

        preds = run_sequence(inferencer, seq)
        if not preds:
            logger.warning(
                f"seq_{seq['seq_id']}: not initializable under GT-init -> "
                f"EXCLUDED from evaluation (was previously scored as all-empty)"
            )
            continue
        evaluator.set_init_classes(seq["seq_id"], seq.get("init_parts", []))
        evaluate_sequence(preds, seq, evaluator, H, W)
        if args.save_preds:
            for local_idx, frame_id in enumerate(seq["frame_ids"]):
                if local_idx in preds:
                    save_pred(args.output_dir, seq.get("seq_tag", seq["seq_id"]),
                              frame_id, preds[local_idx])

        # Per-sequence metrics
        tmp_eval = EndoVis18Evaluator()
        tmp_eval.set_init_classes(seq["seq_id"], seq.get("init_parts", []))
        for local_idx, (fpath, frame_id) in enumerate(
            zip(seq["frame_files"], seq["frame_ids"])
        ):
            if local_idx == 0:
                continue
            label_path = seq["label_map"].get(frame_id)
            if label_path is None:
                continue
            gt = load_label_mask(label_path, H, W)
            if gt is None:
                continue
            pred = preds.get(local_idx, np.zeros((H, W), dtype=np.uint8))
            tmp_eval.add_frame(seq["seq_id"], frame_id, pred, gt)

        if tmp_eval.results:
            vid_metrics = tmp_eval.compute()
            per_video_rows.append({"seq_id": seq["seq_id"],
                                   "seq_tag": seq.get("seq_tag", ""), **vid_metrics})
            logger.info(f"  seq_{seq['seq_id']}: CIoU={vid_metrics['challenge_iou']:.2f}%")

    if getattr(inferencer, "n_prune_seen", 0):
        _r = inferencer.n_pruned / inferencer.n_prune_seen
        logger.info(f"RETENTION: pruned {inferencer.n_pruned}/"
                    f"{inferencer.n_prune_seen} = {_r*100:.2f}% "
                    f"(mode={cfg.prune_mode})")
    dump_decisions(inferencer, args.output_dir, logger)
    metrics = evaluator.compute()
    logger.info("\n=== EV18 RESULTS ===")
    logger.info(f"Challenge IoU: {metrics['challenge_iou']:.2f}%")
    logger.info(f"IoU:           {metrics['iou']:.2f}%")
    logger.info(f"mcIoU:         {metrics['mciou']:.2f}%")
    logger.info(f"Per-class: {metrics['per_class']}")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if per_video_rows:
        keys = ["seq_id", "seq_tag", "challenge_iou", "iou", "mciou", "mciou_instantiable",
                "ciou_cls_gt", "ciou_cls_union", "cum_miou"]
        part_keys = list(per_video_rows[0].get("per_class", {}).keys())
        with open(os.path.join(args.output_dir, "per_video.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys + part_keys)
            writer.writeheader()
            for row in per_video_rows:
                flat = {k: row[k] for k in keys if k in row}
                flat.update(row.get("per_class", {}))
                writer.writerow(flat)

    with open(os.path.join(args.output_dir, "per_video.json"), "w") as f:
        json.dump(per_video_rows, f, indent=1, default=float)

    logger.info(f"\nResults saved to {args.output_dir}")
    return metrics


if __name__ == "__main__":
    main()
