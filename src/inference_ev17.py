"""
Inference script for EndoVis 2017 evaluation.

Usage:
    python src/inference_ev17.py \
        --config C0 \
        --model-path checkpoints/sam3.pt \
        --data-root data/endovis_2017 \
        --output-dir results/C0 \
        --seed 1

Outputs:
    {output_dir}/metrics.json       — Challenge IoU, IoU, mcIoU, per-class
    {output_dir}/per_video.csv      — Per-dataset breakdown
    {output_dir}/config.yaml        — Experiment config snapshot
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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import REPO_ROOT, default_checkpoint, default_data_root  # noqa: E402

_INIT_CLASSES = {}

from src.data.endovis17 import get_test_sequences, load_gt_mask, extract_class_masks, load_frame_rgb
from src.evaluation.metrics import EndoVis17Evaluator
from src.models.md_sam3 import CONFIG_PRESETS, MDSam3Config, build_md_sam3, MDSam3Inferencer
from src.run_utils import add_control_args, apply_control_args, full_config, dump_decisions, save_pred

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference_ev17")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="MD-SAM3 EV17 Evaluation")
    p.add_argument("--config", default="C7",
                   help="Config name (C0-C7, M2-recency, M2-random, M2-dfs) or 'custom'")
    p.add_argument("--model-path", default=default_checkpoint(),
                   help="Path to SAM3 checkpoint")
    p.add_argument("--data-root", default=default_data_root("ev17"),
                   help="Path to EndoVis 2017 data root")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for results")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Custom config overrides
    p.add_argument("--dms", action="store_true", default=None)
    p.add_argument("--no-dms", dest="dms", action="store_false")
    p.add_argument("--dm", action="store_true", default=None)
    p.add_argument("--no-dm", dest="dm", action="store_false")
    p.add_argument("--dfs", action="store_true", default=None)
    p.add_argument("--no-dfs", dest="dfs", action="store_false")
    p.add_argument("--tau-rel", type=float, default=None)
    p.add_argument("--tau-occ", type=float, default=None)
    p.add_argument("--lambda-dfs", type=float, default=None)
    p.add_argument("--memory-size", type=int, default=None)
    p.add_argument("--dfs-mode", default=None, choices=["diversity", "recency", "random"])
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
    if args.config in CONFIG_PRESETS:
        cfg = CONFIG_PRESETS[args.config]
    else:
        cfg = MDSam3Config()

    # Apply any overrides
    if args.dms is not None:
        cfg.dms = args.dms
    if args.dm is not None:
        cfg.dm = args.dm
    if args.dfs is not None:
        cfg.dfs = args.dfs
    if args.tau_rel is not None:
        cfg.tau_rel = args.tau_rel
    if args.tau_occ is not None:
        cfg.tau_occ = args.tau_occ
    if args.lambda_dfs is not None:
        cfg.lambda_dfs = args.lambda_dfs
    if args.memory_size is not None:
        cfg.memory_size = args.memory_size
    if args.dfs_mode is not None:
        cfg.dfs_mode = args.dfs_mode
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


def run_sequence(inferencer: MDSam3Inferencer, seq: dict) -> dict:
    """Run inference on one test sequence and return per-frame type masks."""
    frame_files = seq["frame_files"]
    if not frame_files:
        return {}

    # Load initialization frame GT mask
    init_local_idx = 0
    init_frame_id = seq["init_frame_id"]
    import cv2
    first_img = cv2.imread(frame_files[0])
    H, W = first_img.shape[:2]

    init_gt = load_gt_mask(seq["gt_dir"], init_frame_id, H, W)
    if init_gt is None:
        logger.warning(f"No GT for init frame {init_frame_id} in ds{seq['dataset_id']}")
        return {}

    init_masks = extract_class_masks(init_gt)
    if not init_masks:
        logger.warning(f"No instruments in init frame for ds{seq['dataset_id']}")
        return {}

    logger.info(f"Dataset {seq['dataset_id']}: {len(frame_files)} frames, "
                f"classes={list(init_masks.keys())}")

    inferencer.seq_tag = f"ds{seq['dataset_id']}"  # for MDSAM3_SCORE_LOG diagnostics

    t0 = time.time()
    predictions = inferencer.run_sequence(frame_files, init_masks, init_local_idx)
    _INIT_CLASSES[seq['dataset_id']] = sorted(init_masks.keys())
    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s ({len(frame_files)/elapsed:.1f} fps)")

    return predictions


def evaluate_sequence(predictions: dict, seq: dict, evaluator: EndoVis17Evaluator,
                      H: int, W: int):
    """Compare predictions with GT for frames 1..N (skip init frame)."""
    import cv2
    ds_id = seq["dataset_id"]
    frame_files = seq["frame_files"]
    frame_ids = seq["frame_ids"]

    for local_idx, (fpath, frame_id) in enumerate(zip(frame_files, frame_ids)):
        if local_idx == 0:
            continue   # skip init frame
        pred = predictions.get(local_idx, np.zeros((H, W), dtype=np.uint8))
        gt = load_gt_mask(seq["gt_dir"], frame_id, H, W)
        if gt is None:
            continue
        evaluator.add_frame(ds_id, frame_id, pred, gt)


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = get_config(args)
    logger.info(f"Config {args.config}: dms={cfg.dms}, dm={cfg.dm}, dfs={cfg.dfs}")
    logger.info(f"  tau_rel={cfg.tau_rel}, tau_occ={cfg.tau_occ}, "
                f"lambda_dfs={cfg.lambda_dfs}, M={cfg.memory_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config snapshot
    import yaml
    config_dict = {
        "experiment_id": args.config,
        "seed": args.seed,
        "model_path": args.model_path,
        "dataset": "endovis2017",
        "model": {
            "components": {"dms": cfg.dms, "dm": cfg.dm, "dfs": cfg.dfs},
            "hyperparams": {
                "tau_rel": cfg.tau_rel,
                "tau_occ": cfg.tau_occ,
                "lambda_dfs": cfg.lambda_dfs,
                "memory_size": cfg.memory_size,
                "dfs_mode": cfg.dfs_mode,
                "adaptive_tau": cfg.adaptive_tau,
                "adaptive_beta": cfg.adaptive_beta,
            },
        },
        "full_config": full_config(cfg, args),
    }
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Load model
    logger.info(f"Loading SAM3 from {args.model_path}")
    inferencer = build_md_sam3(cfg, args.model_path, args.device)
    _nm = args.num_maskmem or os.environ.get('MDSAM3_NUM_MASKMEM')
    if _nm:
        inferencer.predictor.num_maskmem = int(_nm)
        logger.info(f'CAPACITY OVERRIDE: num_maskmem={_nm}')

    # Load test sequences
    sequences = get_test_sequences(args.data_root)
    if not sequences:
        logger.error(f"No test sequences found in {args.data_root}")
        return
    logger.info(f"Found {len(sequences)} test sequences")

    evaluator = EndoVis17Evaluator()
    per_video_rows = []

    for seq in sequences:
        import cv2
        H, W = cv2.imread(seq["frame_files"][0]).shape[:2]

        preds = run_sequence(inferencer, seq)
        evaluator.set_init_classes(seq['dataset_id'],
                                   _INIT_CLASSES.get(seq['dataset_id'], []))
        evaluate_sequence(preds, seq, evaluator, H, W)
        if args.save_preds:
            for local_idx, frame_id in enumerate(seq["frame_ids"]):
                if local_idx in preds:
                    save_pred(args.output_dir, f"ds{seq['dataset_id']}", frame_id, preds[local_idx])

        # Per-video metrics
        tmp_eval = EndoVis17Evaluator()
        for local_idx, (fpath, frame_id) in enumerate(
            zip(seq["frame_files"], seq["frame_ids"])
        ):
            if local_idx == 0:
                continue
            pred = preds.get(local_idx, np.zeros((H, W), dtype=np.uint8))
            gt = load_gt_mask(seq["gt_dir"], frame_id, H, W)
            if gt is not None:
                tmp_eval.set_init_classes(seq["dataset_id"],
                        _INIT_CLASSES.get(seq["dataset_id"], []))
                tmp_eval.add_frame(seq["dataset_id"], frame_id, pred, gt)

        if tmp_eval.results:
            vid_metrics = tmp_eval.compute()
            per_video_rows.append({
                "dataset_id": seq["dataset_id"],
                **vid_metrics,
            })
            logger.info(f"  DS{seq['dataset_id']}: CIoU={vid_metrics['challenge_iou']:.2f}%")

    # Final metrics
    if getattr(inferencer, "n_prune_seen", 0):
        _r = inferencer.n_pruned / inferencer.n_prune_seen
        logger.info(f"RETENTION: pruned {inferencer.n_pruned}/"
                    f"{inferencer.n_prune_seen} = {_r*100:.2f}% "
                    f"(mode={cfg.prune_mode})")
    dump_decisions(inferencer, args.output_dir, logger)
    metrics = evaluator.compute()
    logger.info("\n=== RESULTS ===")
    logger.info(f"Challenge IoU: {metrics['challenge_iou']:.2f}%")
    logger.info(f"IoU:           {metrics['iou']:.2f}%")
    logger.info(f"mcIoU:         {metrics['mciou']:.2f}%")
    logger.info(f"Per-class: {metrics['per_class']}")

    # Save results
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save per-video CSV
    if per_video_rows:
        keys = ["dataset_id", "challenge_iou", "iou", "mciou", "mciou_instantiable",
                "biou_nonempty", "ciou_cls_gt", "ciou_cls_union", "cum_miou"]
        with open(os.path.join(args.output_dir, "per_video.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys + list(per_video_rows[0].get("per_class", {}).keys()))
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
