"""Metric computation for EndoVis evaluation."""
import zlib

import numpy as np
from typing import Dict, List, Tuple


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between binary prediction and ground truth."""
    pred_bin = pred > 0
    gt_bin = gt > 0
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


class ClassAwareIoU:
    """Class-aware IoU variants used by published EndoVis results.

    Our primary metric ('challenge_iou' in older outputs) is binary foreground
    IoU, which ignores instrument identity. These variants score the label map:
      - cls_gt:    per frame, mean IoU over classes present in that frame's GT
                   (frames with no instrument in GT are skipped)
      - cls_union: per frame, mean IoU over classes in GT or prediction
                   (frames with neither are skipped)
      - cum_miou:  per class, total intersection / total union over all
                   frames, then mean over classes that occur in GT
    """

    def __init__(self, class_ids):
        self.class_ids = list(class_ids)
        self.f_gt, self.f_union = [], []
        self.inter = {c: 0 for c in self.class_ids}
        self.uni = {c: 0 for c in self.class_ids}
        self.seen = {c: False for c in self.class_ids}

    def add(self, pred: np.ndarray, gt: np.ndarray):
        ious_gt, ious_union = [], []
        for c in self.class_ids:
            p, g = pred == c, gt == c
            ps, gs = bool(p.any()), bool(g.any())
            if not (ps or gs):
                continue
            i = int(np.logical_and(p, g).sum())
            u = int(np.logical_or(p, g).sum())
            iou = i / u if u else 0.0
            ious_union.append(iou)
            if gs:
                ious_gt.append(iou)
                self.seen[c] = True
            self.inter[c] += i
            self.uni[c] += u
        if ious_gt:
            self.f_gt.append(float(np.mean(ious_gt)))
        if ious_union:
            self.f_union.append(float(np.mean(ious_union)))

    def result(self) -> Dict:
        present = [c for c in self.class_ids if self.seen[c]]
        cum = [self.inter[c] / self.uni[c] for c in present if self.uni[c]]
        return {
            "ciou_cls_gt": round(float(np.mean(self.f_gt)) * 100, 2) if self.f_gt else 0.0,
            "ciou_cls_union": round(float(np.mean(self.f_union)) * 100, 2) if self.f_union else 0.0,
            "cum_miou": round(float(np.mean(cum)) * 100, 2) if cum else 0.0,
        }


def _pack(m: np.ndarray):
    m = np.ascontiguousarray(m)
    return zlib.compress(m.tobytes(), 1), m.dtype.str, m.shape


def _unpack(packed) -> np.ndarray:
    buf, dtype, shape = packed
    return np.frombuffer(zlib.decompress(buf), dtype=np.dtype(dtype)).reshape(shape).copy()


class EndoVis17Evaluator:
    """Computes Challenge IoU, IoU, and mcIoU for EV17."""

    CLASS_IDS = [1, 2, 3, 4, 5, 6, 7]
    CLASS_ABBR = {1: "BF", 2: "PF", 3: "LND", 4: "VS", 5: "GR", 6: "MCS", 7: "UP"}

    def __init__(self, compress: bool = False):
        # Per-frame storage: {dataset_id: {frame_id: (pred_type_mask, gt_type_mask)}}.
        # compress=True stores each mask losslessly zlib-compressed (bytes, dtype, shape) and
        # restores it exactly in compute(); it keeps memory bounded on long full-HD test sets
        # (SAR-RARP50: 16k frames at 1920x1080 would otherwise hold ~67 GB of masks).
        self.compress = compress
        self.results = {}
        # {dataset_id: set(class ids present in that sequence's INIT frame)}.
        # Under GT-init a class absent from frame 0 can never be tracked, so its
        # frames are structural zeros and must not be pooled into that class's IoU.
        self.init_classes = {}

    def set_init_classes(self, dataset_id: int, classes):
        self.init_classes[int(dataset_id)] = set(int(c) for c in classes)

    def add_frame(self, dataset_id: int, frame_id: int,
                  pred_type_mask: np.ndarray, gt_type_mask: np.ndarray):
        """Register a prediction frame for evaluation."""
        if dataset_id not in self.results:
            self.results[dataset_id] = {}
        if self.compress:
            pred_type_mask, gt_type_mask = _pack(pred_type_mask), _pack(gt_type_mask)
        self.results[dataset_id][frame_id] = (pred_type_mask, gt_type_mask)

    def compute(self) -> Dict:
        """Compute all metrics over all registered frames."""
        # Aggregate across all frames and datasets
        all_challenge_iou = []
        all_iou = []
        per_class_iou = {c: [] for c in self.CLASS_IDS}
        per_class_inst = {c: [] for c in self.CLASS_IDS}
        cls_aware = ClassAwareIoU(self.CLASS_IDS)
        # EV17 challenge binary protocol: frames without any instrument in GT
        # are discounted (our default scores an empty/empty frame as 1.0)
        biou_nonempty = []

        for ds_id, frames in self.results.items():
            inst = self.init_classes.get(int(ds_id))
            for frame_id, (pred, gt) in frames.items():
                if self.compress:
                    pred, gt = _unpack(pred), _unpack(gt)
                # Challenge IoU: binary (any instrument vs background)
                pred_binary = (pred > 0).astype(np.uint8)
                gt_binary = (gt > 0).astype(np.uint8)
                challenge_iou = compute_iou(pred_binary, gt_binary)
                all_challenge_iou.append(challenge_iou)
                all_iou.append(challenge_iou)  # NB: binary IoU (bIoU), not class-aware
                cls_aware.add(pred, gt)
                if gt_binary.any():
                    biou_nonempty.append(challenge_iou)

                # Per-class IoU
                for cls_id in self.CLASS_IDS:
                    pred_cls = (pred == cls_id).astype(np.uint8)
                    gt_cls = (gt == cls_id).astype(np.uint8)
                    if gt_cls.sum() > 0:
                        cls_iou = compute_iou(pred_cls, gt_cls)
                        per_class_iou[cls_id].append(cls_iou)
                        if inst is None or cls_id in inst:
                            per_class_inst[cls_id].append(cls_iou)

        mean_challenge_iou = np.mean(all_challenge_iou) * 100 if all_challenge_iou else 0.0
        mean_iou = np.mean(all_iou) * 100 if all_iou else 0.0

        class_means = {}
        for cls_id in self.CLASS_IDS:
            vals = per_class_iou[cls_id]
            class_means[cls_id] = np.mean(vals) * 100 if vals else 0.0
        # Only average over classes that actually appear in GT (standard EV17 protocol)
        present = [cls_id for cls_id in self.CLASS_IDS if len(per_class_iou[cls_id]) > 0]
        mc_iou = np.mean([class_means[c] for c in present]) if present else 0.0

        inst_means = {c: (np.mean(per_class_inst[c]) * 100 if per_class_inst[c] else 0.0)
                      for c in self.CLASS_IDS}
        inst_present = [c for c in self.CLASS_IDS if per_class_inst[c]]
        mc_iou_inst = (np.mean([inst_means[c] for c in inst_present])
                       if inst_present else 0.0)

        return {
            "challenge_iou": round(mean_challenge_iou, 2),
            "iou": round(mean_iou, 2),
            "mciou": round(mc_iou, 2),
            "per_class": {
                self.CLASS_ABBR[c]: round(class_means[c], 2) for c in self.CLASS_IDS
            },
            # Restricted to sequences where the class was instantiable at frame 0
            "mciou_instantiable": round(mc_iou_inst, 2),
            "per_class_instantiable": {
                self.CLASS_ABBR[c]: round(inst_means[c], 2) for c in self.CLASS_IDS
            },
            "n_frames_per_class": {
                self.CLASS_ABBR[c]: len(per_class_iou[c]) for c in self.CLASS_IDS
            },
            "n_frames_per_class_instantiable": {
                self.CLASS_ABBR[c]: len(per_class_inst[c]) for c in self.CLASS_IDS
            },
            "num_frames": len(all_challenge_iou),
            "biou": round(mean_challenge_iou, 2),
            "biou_nonempty": round(float(np.mean(biou_nonempty)) * 100, 2) if biou_nonempty else 0.0,
            "n_frames_nonempty": len(biou_nonempty),
            **cls_aware.result(),
        }

    def compute_per_dataset(self) -> Dict[int, Dict]:
        """Compute metrics per dataset."""
        per_ds = {}
        for ds_id, frames in self.results.items():
            all_ciou = []
            per_class = {c: [] for c in self.CLASS_IDS}
            for frame_id, (pred, gt) in frames.items():
                pred_bin = (pred > 0).astype(np.uint8)
                gt_bin = (gt > 0).astype(np.uint8)
                all_ciou.append(compute_iou(pred_bin, gt_bin))
                for cls_id in self.CLASS_IDS:
                    pred_cls = (pred == cls_id).astype(np.uint8)
                    gt_cls = (gt == cls_id).astype(np.uint8)
                    if gt_cls.sum() > 0:
                        per_class[cls_id].append(compute_iou(pred_cls, gt_cls))
            per_ds[ds_id] = {
                "challenge_iou": np.mean(all_ciou) * 100 if all_ciou else 0.0,
                "per_class": {self.CLASS_ABBR[c]: np.mean(per_class[c]) * 100
                              if per_class[c] else 0.0 for c in self.CLASS_IDS},
            }
        return per_ds
