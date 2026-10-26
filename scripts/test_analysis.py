"""Test-set evaluation of the rule chosen on the development data (no GPU).

Method (chosen by scripts/dev_analysis.py): two paths per video, memory-free
(delete every non-conditioning entry) and the online rank rule (Q50); per
sequence keep the path whose entries have the higher mean score. Compared with
the baseline (C0), DM, each path alone, a random choice between the two paths
(expected value: their per-sequence mean) and the best path per sequence
(ground-truth choice, an upper bound).
Usage: python scripts/test_analysis.py [RESULTS_DIR]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as ia  # noqa: E402

RES = ia.RES
SETS = {
    "EV17": {"C0": "EV17_C0_ref", "DM": "EV17_C2_ref", "delall": "EV17_C2_delall", "Q50": "EV17_Q50_sp",
             "method": "EV17_FUSE-delall-Q50_seq", "fuse_frame": "EV17_FUSE-delall-Q50_frame",
             "alt": "EV17_FUSE-delall-DM_seq", "SRG": "EV17_C2-SRG", "topk": "EV17_topk",
            "keep_delall": "EV17_FUSE-keep-delall_seq"},
    "EV18": {"C0": "EV18all_C0_ref", "DM": "EV18all_C2_ref", "delall": "EV18all_C2_delall", "Q50": "EV18all_Q50_sp",
             "method": "EV18all_FUSE-delall-Q50_seq", "fuse_frame": "EV18all_FUSE-delall-Q50_frame",
             "alt": "EV18all_FUSE-delall-DM_seq", "SRG": "EV18all_C2-SRG", "topk": "EV18all_topk",
            "keep_delall": "EV18all_FUSE-keep-delall_seq"},
    "Cholec": {"C0": "TESTcholecL_keep", "DM": "TESTcholecL_DM", "delall": "TESTcholecL_delall", "Q50": "TESTcholecL_Q50",
               "method": "TESTcholecL_FUSE-delall-Q50_seq", "fuse_frame": "TESTcholecL_FUSE-delall-Q50_frame",
               "alt": "TESTcholecL_FUSE-delall-DM_seq", "SRG": "TESTcholecL_SRG", "topk": "TESTcholecL_topk",
            "keep_delall": "TESTcholecL_FUSE-keep-delall_seq"},
    # fourth test set, preregistered (PREREGISTRATION_SAR-RARP50.md); used once its runs exist
    "SAR": {"C0": "SAR_keep", "DM": "SAR_DM", "delall": "SAR_delall", "Q50": "SAR_Q50",
            "method": "SAR_FUSE-delall-Q50_seq", "fuse_frame": "SAR_FUSE-delall-Q50_frame",
            "alt": "SAR_FUSE-delall-DM_seq", "topk": "SAR_topk",
            "keep_delall": "SAR_FUSE-keep-delall_seq"},
}


def fmt(r):
    return f"{r['diff']:+6.2f} [{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] W/L {r['W']}/{r['L']}"


PREFIX = {"EV17": "EV17", "EV18": "EV18all", "Cholec": "TESTcholecL", "SAR": "SAR"}


def draws(pattern, keys):
    out = []
    for k in range(1, 101):
        r = ia.load(pattern.format(s=k))
        if r is not None and sorted(r) == keys:
            out.append(ia.vec(r, keys))
    return np.stack(out) if out else None


def controls(out):
    """Rate-matched controls for the chosen method: (i) the rank rule against random
    deletion at its per-sequence counts; (ii) the two-path tracker against the same
    tracker whose memory path deletes at random (same counts), fused by the same rule."""
    pooled = {}
    for name in SETS:
        if name not in out:
            continue
        keys = sorted(out[name]["per_seq"])
        w = np.array([out[name]["weights"][k] for k in keys])
        V = {n: np.array([out[name]["per_seq"][k][n] for k in keys]) for n in ("Q50", "method")}
        pre = PREFIX[name]
        res = {}
        for arm, pat in [("Q50", f"{pre}_Q50-perm_s{{s}}"), ("method", f"{pre}_FUSE-delall-Q50perm_s{{s}}_seq")]:
            C = draws(pat, keys)
            if C is None:
                continue
            aggs = np.array([ia.wmean(c, w) for c in C])
            d = ia.wmean(V[arm], w) - aggs
            lo, hi = ia.nested_boot(V[arm], C, w)
            wins = int((d > 0).sum())
            W, L = ia.wl(V[arm], C.mean(0))
            res[arm] = {"K": len(C), "control_mean": float(aggs.mean()), "control_range": [float(aggs.min()), float(aggs.max())],
                        "diff": float(d.mean()), "ci": [lo, hi], "wins": wins, "p": (1 + len(C) - wins) / (len(C) + 1),
                        "W": W, "L": L}
            pooled.setdefault(arm, []).append((V[arm], C, w))
            print(f"  {name} {arm:6s} vs random-deletion control (K={len(C)}): control {aggs.mean():.2f} "
                  f"({aggs.min():.2f}-{aggs.max():.2f}); diff {d.mean():+.2f} [{lo:+.2f},{hi:+.2f}]; "
                  f"above {wins}/{len(C)} draws (p={(1 + len(C) - wins) / (len(C) + 1):.3f}); W/L {W}/{L}")
        out[name]["controls"] = res
    rng = np.random.default_rng(ia.RNG_SEED)
    out["controls_pooled"] = {}
    n_sets = len([s for s in SETS if s in out])
    for arm, items in pooled.items():
        if len(items) != n_sets:
            continue
        acc = np.zeros(ia.NBOOT)
        pt = []
        for a, C, w in items:
            K, S = C.shape
            idx = rng.integers(0, S, size=(ia.NBOOT, S))
            kk = rng.integers(0, K, size=(ia.NBOOT, S))
            acc += ((a[idx] - C[kk, idx]) * w[idx]).sum(1) / w[idx].sum(1)
            pt.append(ia.wmean(a, w) - np.mean([ia.wmean(c, w) for c in C]))
        acc /= len(items)
        lo, hi = np.percentile(acc, [2.5, 97.5])
        out["controls_pooled"][arm] = {"diff": float(np.mean(pt)), "ci": [float(lo), float(hi)]}
        print(f"  pooled {arm} - random-deletion control: {np.mean(pt):+.2f} [{lo:+.2f},{hi:+.2f}]")


def main():
    out = {}
    for name, runs in SETS.items():
        data = {k: ia.load(v) for k, v in runs.items()}
        if data["C0"] is None or data["method"] is None:
            print(f"{name}: not ready")
            continue
        keys = sorted(data["C0"])
        w = ia.weights(data["C0"], keys)
        V = {k: ia.vec(d, keys) for k, d in data.items() if d is not None and sorted(d) == keys}
        V["random_choice"] = (V["delall"] + V["Q50"]) / 2
        V["best_path"] = np.maximum(V["delall"], V["Q50"])
        res = {"agg": {k: ia.wmean(v, w) for k, v in V.items()}, "n_seq": len(keys), "n_frames": int(w.sum())}
        print(f"\n== {name}: {len(keys)} sequences, {int(w.sum())} frames ==")
        print("  " + "  ".join(f"{k}={v:.2f}" for k, v in res["agg"].items()))
        res["contrasts"] = {}
        for other in ["C0", "DM", "delall", "Q50", "random_choice", "SRG", "best_path"]:
            if other not in V:
                continue
            pt, lo, hi = ia.paired_boot(V["method"], V[other], w)
            W, L = ia.wl(V["method"], V[other])
            res["contrasts"][other] = {"diff": pt, "ci": [lo, hi], "W": W, "L": L}
            print(f"  method - {other:13s} {fmt(res['contrasts'][other])}")
        if "alt" in V:
            for other in ["C0", "DM", "delall"]:
                pt, lo, hi = ia.paired_boot(V["alt"], V[other], w)
                W, L = ia.wl(V["alt"], V[other])
                print(f"  alt(delall+DM) - {other:6s} {pt:+6.2f} [{lo:+6.2f},{hi:+6.2f}] W/L {W}/{L}")
        res["per_seq"] = {k: {n: float(V[n][i]) for n in V} for i, k in enumerate(keys)}
        res["weights"] = {k: float(w[i]) for i, k in enumerate(keys)}
        out[name] = res
    controls(out)
    present = [s for s in SETS if s in out]
    if len(present) >= 3:
        g = {o: np.mean([out[s]["contrasts"][o]["diff"] for s in present]) for o in ["C0", "DM", "delall", "Q50"]}
        print(f"\nmean gain of the method over the {len(present)} test sets: "
              + "  ".join(f"vs {k} {v:+.2f}" for k, v in g.items()))
        out["mean_gain"] = g
    (RES / "test_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
