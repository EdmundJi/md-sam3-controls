"""Inference on the development / confirmation data (src/data/extra_datasets.py).

Same configurations, controls and outputs as src/inference_ev17.py; bIoU is the
binary instrument IoU per frame (an empty/empty frame scores 1).
usage:
  python src/inference_extra.py --dataset ev17train|cholec|sarrarp [--split dev|test|train|all] \
      --data-root D --config C0 --output-dir OUT [control args as in inference_ev17.py]
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import inference_ev17 as ev17  # noqa: E402  (shared args, config, logging)
from src.data.extra_datasets import (cholec_sequences, ev17_train_sequences, extract_class_masks,  # noqa: E402
                                     sarrarp_sequences)
from src.evaluation.metrics import EndoVis17Evaluator  # noqa: E402
from src.models.md_sam3 import build_md_sam3  # noqa: E402
from src.run_utils import dump_decisions, full_config, save_pred  # noqa: E402

logger = ev17.logger


def main():
    extra = [a for a in ("--dataset", "--split") if a in sys.argv]
    ds_name, split = "cholec", "all"
    argv = list(sys.argv)
    for a in extra:
        i = argv.index(a)
        val = argv[i + 1]
        del argv[i:i + 2]
        if a == "--dataset":
            ds_name = val
        else:
            split = val
    sys.argv = argv
    args = ev17.parse_args()
    ev17.set_seed(args.seed)
    cfg = ev17.get_config(args)
    os.makedirs(args.output_dir, exist_ok=True)
    import yaml
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump({"experiment_id": args.config, "dataset": ds_name, "split": split, "seed": args.seed,
                   "full_config": full_config(cfg, args)}, f)
    inferencer = build_md_sam3(cfg, args.model_path, args.device)
    nm = args.num_maskmem or os.environ.get("MDSAM3_NUM_MASKMEM")
    if nm:
        inferencer.predictor.num_maskmem = int(nm)
    seqs = (ev17_train_sequences(args.data_root) if ds_name == "ev17train"
            else sarrarp_sequences(args.data_root, split) if ds_name == "sarrarp"
            else cholec_sequences(args.data_root, split))
    keep = os.environ.get("SEQ_TAGS")
    if keep:
        seqs = [s for s in seqs if s["seq_tag"] in set(keep.split(","))]
    logger.info(f"{ds_name}/{split}: {len(seqs)} sequences")

    evaluator = EndoVis17Evaluator(compress=True)   # whole-run metrics; masks stored compressed
    rows = []
    for s in seqs:
        H, W = cv2.imread(s["frame_files"][0]).shape[:2]
        init = s["gt"](s["init_frame_id"], H, W)
        init_masks = extract_class_masks(init)
        inferencer.seq_tag = s["seq_tag"]
        t0 = time.time()
        preds = inferencer.run_sequence(s["frame_files"], init_masks, 0)
        logger.info(f"  {s['seq_tag']}: {len(s['frame_files'])} frames, classes={sorted(init_masks)}, "
                    f"Done in {time.time() - t0:.1f}s")
        tmp = EndoVis17Evaluator()
        for e in (evaluator, tmp):
            e.set_init_classes(s["dataset_id"], sorted(init_masks))
        for li, fid in enumerate(s["frame_ids"]):
            if li == 0:
                continue
            g = s["gt"](fid, H, W)
            if g is None:
                continue
            p = preds.get(li, np.zeros((H, W), np.uint8))
            evaluator.add_frame(s["dataset_id"], fid, p, g)
            tmp.add_frame(s["dataset_id"], fid, p, g)
        if args.save_preds:
            for li, fid in enumerate(s["frame_ids"]):
                if li in preds:
                    save_pred(args.output_dir, s["seq_tag"], fid, preds[li])
        if tmp.results:
            m = tmp.compute()
            rows.append({"seq_tag": s["seq_tag"], "dataset_id": s["dataset_id"], **m})
            logger.info(f"    bIoU={m['challenge_iou']:.2f}")
    if getattr(inferencer, "n_prune_seen", 0):
        logger.info(f"RETENTION: pruned {inferencer.n_pruned}/{inferencer.n_prune_seen} "
                    f"= {100 * inferencer.n_pruned / inferencer.n_prune_seen:.2f}% (mode={cfg.prune_mode})")
    dump_decisions(inferencer, args.output_dir, logger)
    metrics = evaluator.compute()
    logger.info(f"Challenge IoU: {metrics['challenge_iou']:.2f}%")
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(os.path.join(args.output_dir, "per_video.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)


if __name__ == "__main__":
    main()
