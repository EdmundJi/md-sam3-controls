"""EndoVis 2017 dataset loader for instrument type segmentation evaluation."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Class ID to instrument name mapping
CLASS_NAMES = {
    0: "Background",
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Vessel Sealer",
    5: "Grasping Retractor",
    6: "Monopolar Curved Scissors",
    7: "Other / Ultrasound Probe",
}

# Abbreviated names for tables (matches paper Table 1)
CLASS_ABBR = {
    1: "BF", 2: "PF", 3: "LND", 4: "VS", 5: "GR", 6: "MCS", 7: "UP",
}

# Which datasets use partial test frames (225-299) vs full (0-299)
PARTIAL_TEST_DATASETS = {1, 2, 3, 4, 5, 6, 7, 8}
FULL_TEST_DATASETS = {9, 10}


def get_test_sequences(data_root: str) -> List[Dict]:
    """
    Returns a list of test sequence descriptors.
    Each entry has: dataset_id, frame_dir, gt_dir, frame_ids, init_frame_id.
    """
    data_root = Path(data_root)
    sequences = []

    # Datasets 1-4: test frames in instrument_1_4_testing/
    for ds in [1, 2, 3, 4]:
        frame_dir = data_root / "instrument_1_4_testing" / f"instrument_dataset_{ds}" / "left_frames"
        gt_dir = data_root / "instrument_2017_test" / f"instrument_dataset_{ds}" / "TypeSegmentation"
        if frame_dir.exists() and gt_dir.exists():
            frames = sorted(
                (p for p in frame_dir.glob("*.png") if p.stem.startswith("frame")),
                key=lambda p: int(p.stem[5:])
            )
            frame_ids = [int(p.stem[5:]) for p in frames]
            sequences.append({
                "dataset_id": ds,
                "frame_dir": str(frame_dir),
                "gt_dir": str(gt_dir),
                "frame_ids": frame_ids,
                "init_frame_id": frame_ids[0],
                "frame_files": [str(f) for f in frames],
            })

    # Datasets 5-8: test frames in instrument_5_8_testing/
    for ds in [5, 6, 7, 8]:
        frame_dir = data_root / "instrument_5_8_testing" / f"instrument_dataset_{ds}" / "left_frames"
        gt_dir = data_root / "instrument_2017_test" / f"instrument_dataset_{ds}" / "TypeSegmentation"
        if frame_dir.exists() and gt_dir.exists():
            frames = sorted(
                (p for p in frame_dir.glob("*.png") if p.stem.startswith("frame")),
                key=lambda p: int(p.stem[5:])
            )
            frame_ids = [int(p.stem[5:]) for p in frames]
            sequences.append({
                "dataset_id": ds,
                "frame_dir": str(frame_dir),
                "gt_dir": str(gt_dir),
                "frame_ids": frame_ids,
                "init_frame_id": frame_ids[0],
                "frame_files": [str(f) for f in frames],
            })

    # Datasets 9-10: full sequences in instrument_9_10_testing/
    for ds in [9, 10]:
        frame_dir = data_root / "instrument_9_10_testing" / f"instrument_dataset_{ds}" / "left_frames"
        gt_dir = data_root / "instrument_2017_test" / f"instrument_dataset_{ds}" / "TypeSegmentation"
        if frame_dir.exists() and gt_dir.exists():
            frames = sorted(
                (p for p in frame_dir.glob("*.png") if p.stem.startswith("frame")),
                key=lambda p: int(p.stem[5:])
            )
            frame_ids = [int(p.stem[5:]) for p in frames]
            sequences.append({
                "dataset_id": ds,
                "frame_dir": str(frame_dir),
                "gt_dir": str(gt_dir),
                "frame_ids": frame_ids,
                "init_frame_id": frame_ids[0],
                "frame_files": [str(f) for f in frames],
            })

    sequences.sort(key=lambda s: s["dataset_id"])
    return sequences


def load_gt_mask(gt_dir: str, frame_id: int, h: int, w: int) -> Optional[np.ndarray]:
    """Load type segmentation GT mask as integer array (H x W)."""
    import cv2
    path = os.path.join(gt_dir, f"frame{frame_id:03d}.png")
    if not os.path.exists(path):
        path = os.path.join(gt_dir, f"frame{frame_id}.png")
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def extract_class_masks(type_mask: np.ndarray) -> Dict[int, np.ndarray]:
    """Extract per-class binary masks from a type segmentation mask."""
    classes = {}
    for cls_id in range(1, 8):
        binary = (type_mask == cls_id).astype(np.uint8)
        if binary.sum() > 0:
            classes[cls_id] = binary
    return classes


def load_frame_rgb(frame_path: str) -> np.ndarray:
    """Load a video frame as RGB numpy array."""
    import cv2
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load frame: {frame_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
