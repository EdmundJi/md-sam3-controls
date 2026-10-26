"""Development and confirmation data beyond the EV17/EV18 test protocol.

ev17train  EndoVis 2017 training release, datasets 1-8, frames 0-224 (development).
           Per-instrument binary masks are merged into the type labels of the
           test release (1 BF, 2 PF, 3 LND, 4 VS, 5 GR, 6 MCS, 7 other).
cholec     CholecSeg8k: 101 clips of 80 frames from 17 cholecystectomy videos;
           temporally contiguous clips are joined into one sequence, which
           starts at its first frame showing an instrument (sequences with
           fewer than 40 frames left are dropped).
           Instrument classes grasper (watershed id 31 -> 1) and L-hook
           electrocautery (32 -> 2); everything else is background. Clips
           without an instrument in their first frame cannot be initialized
           and are skipped. Split by video: CHOLEC_DEV_VIDEOS for development,
           the rest for confirmation.

sarrarp    SAR-RARP50 (robot-assisted radical prostatectomy, suturing step): 50
           operations in 54 video files, masks at 1 Hz (prepared with
           scripts/prepare_sarrarp.py). One sequence per video file, tracked on
           the annotated frames only and starting at the first one showing an
           instrument. Instrument categories: robotic instrument (tool clasper,
           wrist and shaft, ids 1-3 -> 1), suction tool (6 -> 2) and needle
           holder (7 -> 3); needle, thread, clamps and catheter are background.
           Split: official test release (videos 41-50) or train release.
           Never used for development (fourth test set, preregistered).

Each sequence dict has seq_tag, dataset_id (int), frame_files, frame_ids,
init_frame_id and gt(frame_id, H, W) -> HxW uint8 type map (or None).
"""
import re
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

EV17_TRAIN_TYPES = [("Bipolar_Forceps", 1), ("Prograsp_Forceps", 2), ("Large_Needle_Driver", 3),
                    ("Vessel_Sealer", 4), ("Grasping_Retractor", 5),
                    ("Monopolar_Curved_Scissors", 6), ("Other", 7)]
CHOLEC_DEV_VIDEOS = {"video12", "video17", "video18", "video20", "video24"}
CHOLEC_IDS = {31: 1, 32: 2}


def _resize(m, H, W):
    return m if m.shape == (H, W) else cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)


def ev17_train_sequences(root: str) -> List[Dict]:
    root = Path(root)
    out = []
    for ds in range(1, 9):
        part = "instrument_1_4_training" if ds <= 4 else "instrument_5_8_training"
        base = root / part / f"instrument_dataset_{ds}"
        frames = sorted(base.glob("left_frames/frame*.png"), key=lambda p: int(p.stem[5:]))
        folders = []
        for d in sorted((base / "ground_truth").iterdir()):
            t = [tid for key, tid in EV17_TRAIN_TYPES if key.lower() in d.name.lower()]
            if not d.is_dir() or not t:
                raise ValueError(f"unmapped label folder {d}")
            folders.append((d, t[0]))

        def gt(fid, H, W, folders=folders):
            m = np.zeros((H, W), np.uint8)
            found = False
            for d, tid in folders:
                p = d / f"frame{fid:03d}.png"
                if p.exists():
                    found = True
                    b = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    m[_resize(b, H, W) > 0] = tid
            return m if found else None
        ids = [int(p.stem[5:]) for p in frames]
        out.append({"seq_tag": f"tr{ds}", "dataset_id": 100 + ds, "frame_files": [str(p) for p in frames],
                    "frame_ids": ids, "init_frame_id": ids[0], "gt": gt})
    return out


def cholec_sequences(root: str, split: str = "all", merge: bool = True) -> List[Dict]:
    """merge=True joins temporally contiguous clips of a video (clip k+1 starts
    80 frames after clip k) into one sequence; 80-frame clips are too short for
    memory retention to matter."""
    root = Path(root)
    groups = []
    for video_dir in sorted(p for p in root.glob("video*") if p.is_dir()):
        video = video_dir.name
        if (split == "dev" and video not in CHOLEC_DEV_VIDEOS) or (split == "test" and video in CHOLEC_DEV_VIDEOS):
            continue
        clips = sorted((p for p in video_dir.glob(f"{video}_*") if p.is_dir()), key=lambda p: int(p.name.rsplit("_", 1)[1]))
        for c in clips:
            start = int(c.name.rsplit("_", 1)[1])
            if merge and groups and groups[-1][0] == video and groups[-1][2] + 80 == start:
                groups[-1][1].append(c)
                groups[-1][2] = start
            else:
                groups.append([video, [c], start])
    out = []
    for i, (video, clips, _) in enumerate(groups):
        frames, owner = [], {}
        for c in clips:
            fs = sorted(c.glob("frame_*_endo.png"), key=lambda p: int(re.search(r"frame_(\d+)_", p.name).group(1)))
            for f in fs:
                fid = int(re.search(r"frame_(\d+)_", f.name).group(1))
                frames.append((fid, f))
                owner[fid] = c
        ids = [fid for fid, _ in frames]
        assert ids == sorted(ids) and len(set(ids)) == len(ids), clips

        def gt(fid, H, W, owner=owner):
            c = owner.get(fid)
            p = None if c is None else c / f"frame_{fid}_endo_watershed_mask.png"
            if p is None or not p.exists():
                return None
            w = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            m = np.zeros(w.shape, np.uint8)
            for v, cl in CHOLEC_IDS.items():
                m[w == v] = cl
            return _resize(m, H, W)
        H, W = cv2.imread(str(frames[0][1])).shape[:2]
        # start at the first frame showing an instrument (needed to initialize)
        k = next((j for j, fid in enumerate(ids) if (lambda m: m is not None and m.any())(gt(fid, H, W))), None)
        if k is None or len(ids) - k < 40:
            continue
        frames, ids = frames[k:], ids[k:]
        out.append({"seq_tag": f"{video}_{ids[0]:05d}", "dataset_id": 1000 + i,
                    "frame_files": [str(f) for _, f in frames],
                    "frame_ids": ids, "init_frame_id": ids[0], "gt": gt, "n_clips": len(clips)})
    return out


SAR_IDS = {1: 1, 2: 1, 3: 1, 6: 2, 7: 3}
SAR_TEST_VIDEOS = {f"video_{i}" for i in range(41, 51)}


def sarrarp_sequences(root: str, split: str = "all") -> List[Dict]:
    root = Path(root)
    out = []
    videos = sorted((p for p in root.glob("video_*") if (p / "DONE").exists()),
                    key=lambda p: [int(x) for x in p.name[6:].split("_")])
    for i, vdir in enumerate(videos):
        test = vdir.name in SAR_TEST_VIDEOS
        if (split == "test" and not test) or (split == "train" and test):
            continue
        ids = sorted(int(p.stem) for p in (vdir / "segmentation").glob("*.png"))

        def gt(fid, H, W, vdir=vdir):
            p = vdir / "segmentation" / f"{fid:09d}.png"
            if not p.exists():
                return None
            w = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            m = np.zeros(w.shape, np.uint8)
            for v, cl in SAR_IDS.items():
                m[w == v] = cl
            return _resize(m, H, W)
        k = next((j for j, fid in enumerate(ids) if gt(fid, 1080, 1920).any()), None)
        if k is None or len(ids) - k < 40:
            continue
        ids = ids[k:]
        out.append({"seq_tag": vdir.name, "dataset_id": 2000 + i,
                    "frame_files": [str(vdir / "frames" / f"{fid:09d}.png") for fid in ids],
                    "frame_ids": ids, "init_frame_id": ids[0], "gt": gt})
    return out


def extract_class_masks(type_mask: np.ndarray) -> Dict[int, np.ndarray]:
    return {c: (type_mask == c).astype(np.uint8) for c in range(1, 8) if (type_mask == c).any()}
