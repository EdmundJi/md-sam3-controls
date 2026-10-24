"""Choose the retention rule on the development data (no GPU).

Development sets: DEVev17tr (EndoVis 2017 training split) and DEVcholec
(CholecSeg8k dev videos, contiguous clips joined). Criterion, fixed before the runs: mean over the two
sets of the frame-weighted bIoU gain over the no-deletion run (keep = C0).
'self' picks, per sequence, the candidate among keep/R25/R50/R75-score with the
highest mean score over that run's own entries.
Usage: python scripts/dev_analysis.py [RESULTS_DIR]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as ia  # noqa: E402

RES = ia.RES
SETS = ["DEVev17tr", "DEVcholecL"]
CONFIGS = ["keep", "DM", "delall", "SRG", "Q25", "Q25w16", "Q50", "Q50w16", "Q75", "Q75w16",
           "R25-score", "R50-score", "R75-score", "R50-oracle"]


def decisions(run):
    p = RES / run / "retention_decisions.json"
    return json.load(open(p))["sequences"] if p.exists() else None


def rate(run):
    d = decisions(run)
    if not d:
        return None
    return sum(len(v["deleted"]) for v in d.values()) / max(1, sum(len(v["tested"]) for v in d.values()))


def self_select(P, keys):
    cands = ["keep", "R25-score", "R50-score", "R75-score"]
    vals, picks = [], []
    info = {}
    for c in cands:
        r, d = ia.load(f"{P}_{c}"), decisions(f"{P}_{c}")
        if r is None or d is None or sorted(r) != keys:
            return None, None
        info[c] = (r, d)
    for k in keys:
        best = max(cands, key=lambda c: np.mean(info[c][1][k]["scores"]) if info[c][1][k]["scores"] else -1)
        vals.append(info[best][0][k][0])
        picks.append(best)
    return np.array(vals), picks


def main():
    out, gains = {}, {}
    for P in SETS:
        keep = ia.load(f"{P}_keep")
        if keep is None:
            print(f"{P}: keep run missing")
            continue
        keys = sorted(keep)
        w = ia.weights(keep, keys)
        base = ia.vec(keep, keys)
        print(f"\n== {P}: {len(keys)} sequences, {int(w.sum())} frames, keep (C0) {ia.wmean(base, w):.2f}")
        out[P] = {}
        fusions = sorted(d.name[len(P) + 1:] for d in RES.glob(f"{P}_FUSE-*") if (d / "per_video.json").exists())
        rows = {c: ia.load(f"{P}_{c}") for c in CONFIGS + fusions}
        sv, picks = self_select(P, keys)
        for c in CONFIGS + fusions + ["self"]:
            if c == "self":
                if sv is None:
                    continue
                v = sv
                rt = None
            else:
                r = rows[c]
                if r is None or sorted(r) != keys:
                    continue
                v = ia.vec(r, keys)
                rt = rate(f"{P}_{c}")
            d = det = ia.paired_boot(v, base, w)
            W, L = ia.wl(v, base)
            out[P][c] = {"agg": ia.wmean(v, w), "gain": det[0], "ci": [det[1], det[2]], "W": W, "L": L, "rate": rt}
            gains.setdefault(c, {})[P] = det[0]
            print(f"  {c:11s} {ia.wmean(v, w):6.2f}  gain {d[0]:+5.2f} [{d[1]:+5.2f},{d[2]:+5.2f}]  W/L {W}/{L}"
                  + (f"  deleted {100 * rt:4.1f}%" if rt is not None else ""))
        if picks:
            from collections import Counter
            print("  self picks:", dict(Counter(picks)))
    print("\n== criterion: mean gain over the development sets ==")
    # the oracle uses ground truth and is not eligible
    ranked = sorted(((np.mean(list(g.values())), c, g) for c, g in gains.items()
                     if len(g) == len(SETS) and c != "R50-oracle"), reverse=True)
    for m, c, g in ranked:
        print(f"  {c:11s} {m:+5.2f}   " + "  ".join(f"{P} {g[P]:+5.2f}" for P in SETS))
    out["ranking"] = [(c, m) for m, c, _ in ranked]
    (RES / "dev_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
