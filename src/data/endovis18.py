"""EndoVis 2018 dataset loader for instrument part segmentation evaluation.

Test set: seq_1..4 under test_data_and_label/.
Labels are RGBA PNGs. We track three instrument parts as separate classes:
  1 = instrument-shaft   (RGBA 0,255,0,128)
  2 = instrument-clasper (RGBA 0,255,255,128)
  3 = instrument-wrist   (RGBA 125,255,12,128)
All other classes (tissue, anatomy) map to background (0).
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CLASS_IDS = [1, 2, 3]
CLASS_ABBR = {1: "shaft", 2: "clasper", 3: "wrist"}

# RGB values (ignore alpha) → class id
_RGB_TO_CLASS = {
    (0, 255, 0): 1,
    (0, 255, 255): 2,
    (125, 255, 12): 3,
}

# Lookup table: index by (R*256 + G) or just build full 256^3 table lazily
_LUT = None


def _build_lut():
    global _LUT
    if _LUT is not None:
        return _LUT
    # 256x256x256 byte table, default 0 (background)
    lut = np.zeros((256, 256, 256), dtype=np.uint8)
    for (r, g, b), cls in _RGB_TO_CLASS.items():
        lut[r, g, b] = cls
    _LUT = lut
    return lut


def rgba_to_class_mask(rgba: np.ndarray) -> np.ndarray:
    """Convert H×W×4 RGBA label image to H×W integer class mask."""
    lut = _build_lut()
    r = rgba[:, :, 0].astype(np.int32)
    g = rgba[:, :, 1].astype(np.int32)
    b = rgba[:, :, 2].astype(np.int32)
    return lut[r, g, b]


def get_test_sequences(data_root: str) -> List[Dict]:
    """Return list of test sequence descriptors for EV18 (seq_1..4)."""
    data_root = Path(data_root)
    # locate the drive-download sub-folder
    test_dir = data_root / "test_data_and_label"
    # there may be a subdirectory like drive-download-*
    sub_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    if len(sub_dirs) == 1:
        base = sub_dirs[0]
    else:
        # fallback: test_dir itself
        base = test_dir

    sequences = []
    for seq_name in sorted(base.iterdir()):
        if not seq_name.is_dir() or not seq_name.name.startswith("seq_"):
            continue
        seq_id = int(seq_name.name.split("_")[1])
        frame_dir = seq_name / "left_frames"
        label_dir = seq_name / "labels"
        if not frame_dir.exists() or not label_dir.exists():
            continue
        frames = sorted(
            (p for p in frame_dir.glob("*.png") if p.stem.startswith("frame")),
            key=lambda p: int(p.stem[5:]),
        )
        labels = sorted(
            (p for p in label_dir.glob("*.png") if p.stem.startswith("frame")),
            key=lambda p: int(p.stem[5:]),
        )
        if not frames:
            continue
        frame_ids = [int(p.stem[5:]) for p in frames]
        label_ids = {int(p.stem[5:]): str(p) for p in labels}
        sequences.append({
            "seq_id": seq_id,
            "frame_dir": str(frame_dir),
            "label_dir": str(label_dir),
            "frame_ids": frame_ids,
            "init_frame_id": frame_ids[0],
            "frame_files": [str(f) for f in frames],
            "label_map": label_ids,
        })
    sequences.sort(key=lambda s: s["seq_id"])
    return sequences


def load_label_mask(label_path: str, h: int, w: int) -> Optional[np.ndarray]:
    """Load RGBA label PNG and return H×W integer class mask."""
    import cv2
    img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        # unexpected grayscale
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        # cv2 loads as BGRA; convert to RGBA
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alpha = np.full(rgba.shape[:2] + (1,), 128, dtype=np.uint8)
        rgba = np.concatenate([rgba, alpha], axis=2)
    if rgba.shape[:2] != (h, w):
        rgba = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_NEAREST)
    return rgba_to_class_mask(rgba)


def extract_class_masks(class_mask: np.ndarray) -> Dict[int, np.ndarray]:
    """Extract per-class binary masks for instrument parts present in frame."""
    out = {}
    for cls_id in CLASS_IDS:
        binary = (class_mask == cls_id).astype(np.uint8)
        if binary.sum() > 0:
            out[cls_id] = binary
    return out
