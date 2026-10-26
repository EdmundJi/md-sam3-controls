"""Tracking failure rate (secondary endpoint, defined in the SAR-RARP50
preregistration before it was computed on any set): the share of frames whose
ground truth shows an instrument and whose binary instrument IoU is below 0.5.
Computed from saved predictions for every rule of Table 1, including the two
fused rules (per-video and per-frame choice by SAM3's score, as in
scripts/fuse_paths.py). Pooled difference from SAM3 = mean over the test sets of
the frame-weighted per-set differences, CI from resampling sequences within
each set (as for the bIoU gains).

usage: python scripts/failure_rate.py RESULTS_DIR D17 D18 DCH [DSAR]
       python scripts/failure_rate.py RESULTS_DIR --pool-only   (pooled CIs from failure_rate.json)
writes RESULTS_DIR/failure_rate.json
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classaware_offline as ca  # noqa: E402
import analysis as ia  # noqa: E402

THR = 0.5
RUNS = {
    "ev17": {"C0": "EV17_C0_ref", "DM": "EV17_C2_sp", "delall": "EV17_C2_delall_sp", "Q50": "EV17_Q50_sp",
             "topk": "EV17_topk"},
    "ev18": {"C0": "EV18all_C0_ref", "DM": "EV18all_C2_sp", "delall": "EV18all_C2_delall_sp",
             "Q50": "EV18all_Q50_sp", "topk": "EV18all_topk"},
    "cholec": {"C0": "TESTcholecL_keep", "DM": "TESTcholecL_DM", "delall": "TESTcholecL_delall",
               "Q50": "TESTcholecL_Q50", "topk": "TESTcholecL_topk"},
    "sarrarp": {"C0": "SAR_keep", "DM": "SAR_DM", "delall": "SAR_delall", "Q50": "SAR_Q50", "topk": "SAR_topk"},
}
NAMES = {"ev17": "EV17", "ev18": "EV18", "cholec": "Cholec", "sarrarp": "SAR"}


def iou(p, g):
    u = np.logical_or(p, g).sum()
    return 1.0 if u == 0 else float(np.logical_and(p, g).sum() / u)


def per_set(job):
    ds, root, R = job
    runs = {k: Path(R) / v for k, v in RUNS[ds].items()}
    out = {}
    for _eid, tag, fids, load, _cls, H, W in ca.seqs(ds, root):
        if not all((d / "preds" / tag).is_dir() for d in runs.values()):
            continue
        S = {k: ca.scores(runs[k], tag) for k in ("delall", "Q50")}
        pick = max(("delall", "Q50"), key=lambda k: np.mean(list(S[k].values())))  # ties: delall (first)
        if np.mean(list(S["delall"].values())) == np.mean(list(S["Q50"].values())):
            pick = "delall"
        fail = {k: [] for k in list(runs) + ["method", "online"]}
        for t in range(1, len(fids)):
            gt = load(fids[t])
            if gt is None or not gt.any():
                continue
            g = gt > 0
            v = {}
            for k, d in runs.items():
                p = cv2.imread(str(d / "preds" / tag / f"{fids[t]:05d}.png"), cv2.IMREAD_GRAYSCALE)
                v[k] = iou(np.zeros((H, W), bool) if p is None else p > 0, g) < THR
                fail[k].append(v[k])
            fail["method"].append(v[pick])
            fp = "Q50" if S["Q50"].get(t, -1.0) > S["delall"].get(t, -1.0) else "delall"
            fail["online"].append(v[fp])
        out[tag] = {k: [int(np.sum(x)), len(x)] for k, x in fail.items()}
    return ds, out


def pool(per):
    """Per-set aggregates and the pooled difference from SAM3 (mean over sets of the
    frame-weighted per-set differences). The bootstrap resamples sequences within
    sets; on SAR-RARP50 it resamples operations, all files of one together
    (preregistration, Amendment 1)."""
    res = {"agg": {}, "pooled": {}}
    stacks = {}
    for ds, seqs in per.items():
        keys = sorted(seqs)
        n = np.array([seqs[k]["C0"][1] for k in keys], float)
        rate = {r: np.array([seqs[k][r][0] / max(seqs[k][r][1], 1) for k in keys]) * 100 for r in seqs[keys[0]]}
        res["agg"][NAMES[ds]] = {r: ia.wmean(v, n) for r, v in rate.items()}
        if ds == "sarrarp":
            ops = sorted({k.split("_")[1] for k in keys})
            units = [np.array([i for i, k in enumerate(keys) if k.split("_")[1] == o]) for o in ops]
        else:
            units = [np.array([i]) for i in range(len(keys))]
        for r, v in rate.items():
            if r != "C0":
                stacks.setdefault(r, []).append((v - rate["C0"], n, units))
    for r, items in stacks.items():
        rng = np.random.default_rng(ia.RNG_SEED)
        acc = np.zeros(ia.NBOOT)
        for d, w, units in items:
            num = np.array([(d[u] * w[u]).sum() for u in units])
            den = np.array([w[u].sum() for u in units])
            pick = rng.integers(0, len(units), size=(ia.NBOOT, len(units)))
            acc += num[pick].sum(1) / den[pick].sum(1)
        acc /= len(items)
        lo, hi = np.percentile(acc, [2.5, 97.5])
        res["pooled"][r] = {"diff": float(np.mean([ia.wmean(d, w) for d, w, _ in items])), "ci": [float(lo), float(hi)],
                            "fewer": int(sum((d < -1e-9).sum() for d, _, _ in items)),
                            "more": int(sum((d > 1e-9).sum() for d, _, _ in items)), "n_sets": len(items)}
    return res


def main():
    R = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pool-only":     # recompute pooled CIs from failure_rate.json
        per = json.load(open(Path(R, "failure_rate.json")))["per_seq"]
    else:
        roots = dict(zip(["ev17", "ev18", "cholec", "sarrarp"], sys.argv[2:]))
        with ProcessPoolExecutor(len(roots)) as ex:
            per = dict(ex.map(per_set, [(ds, root, R) for ds, root in roots.items()]))
    res = {"threshold": THR, "per_seq": per, **pool(per)}
    Path(R, "failure_rate.json").write_text(json.dumps(res, indent=1))
    for s, a in res["agg"].items():
        print(s, "  ".join(f"{k}={v:.2f}" for k, v in a.items()))
    for r, p in res["pooled"].items():
        print(f"{r:7s} {p['diff']:+.2f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] fewer/more {p['fewer']}/{p['more']}")


if __name__ == "__main__":
    main()
