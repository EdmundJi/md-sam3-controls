"""Prepare SAR-RARP50 for tracking: for every video zip, decode the frames that
have a segmentation mask (1 Hz; the mask file name is the frame number in
video_left.avi) and store them losslessly next to the masks.

  OUT/video_XX/frames/000000060.png   decoded frame 60
  OUT/video_XX/segmentation/000000060.png   mask (copied unchanged)

usage: python scripts/prepare_sarrarp.py ZIP_DIR_OR_ZIPS... OUT_ROOT [--workers N] [--delete-zips]
"""
import argparse
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2


def prepare(zip_path: str, out_root: str, delete_zip: bool) -> str:
    zp = Path(zip_path)
    out = Path(out_root) / zp.stem
    done = out / "DONE"
    if done.exists():
        return f"{zp.stem}: already prepared"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        segs = [n for n in names if n.startswith("segmentation/") and n.endswith(".png")]
        z.extractall(out, members=segs + ["video_left.avi"])
    wanted = sorted(int(Path(n).stem) for n in segs)
    (out / "frames").mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(out / "video_left.avi"))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(wanted)
    idx, saved = 0, 0
    last = wanted[-1]
    while idx <= last:
        if idx in want:
            ok, img = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out / "frames" / f"{idx:09d}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])  # lossless
            saved += 1
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()
    missing = len(wanted) - saved
    (out / "video_left.avi").unlink()
    if delete_zip:
        zp.unlink()
    msg = f"{zp.stem}: {saved}/{len(wanted)} frames saved, video has {n_total} frames"
    if missing:
        msg += f", {missing} MISSING"
    else:
        done.write_text(msg + "\n")
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+")
    ap.add_argument("out_root")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--delete-zips", action="store_true")
    a = ap.parse_args()
    zips = [z for x in map(Path, a.zips) for z in (sorted(x.glob("video_*.zip")) if x.is_dir() else [x])]
    zips.sort(key=lambda p: -p.stat().st_size)  # largest first
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(prepare, [str(z) for z in zips], [a.out_root] * len(zips), [a.delete_zips] * len(zips)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
