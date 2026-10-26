"""Analyses requested by the internal review panel, from saved results (no GPU):
effect concentration (median per-sequence gain, leave-one-sequence-out pooled gain,
per-sequence shares of the tracker's advantage), the tracker against each of its paths,
sequence-weighted means, and path switching of the online variant.
Usage: python scripts/panel_analysis.py [RESULTS_DIR]   -> RESULTS_DIR/panel_analysis.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as ia  # noqa: E402

RES = ia.RES
PATHS = {"EV17": ("EV17_C2_delall_sp", "EV17_Q50_sp"), "EV18": ("EV18all_C2_delall_sp", "EV18all_Q50_sp"),
         "Cholec": ("TESTcholecL_delall", "TESTcholecL_Q50"), "SAR": ("SAR_delall", "SAR_Q50")}


def pooled(T, sets, a, b, drop=None):
    rng = np.random.default_rng(ia.RNG_SEED)
    acc, pts = np.zeros(ia.NBOOT), []
    for s in sets:
        keys = [k for k in sorted(T[s]["per_seq"]) if (s, k) != drop]
        w = np.array([T[s]["weights"][k] for k in keys])
        d = np.array([T[s]["per_seq"][k][a] - T[s]["per_seq"][k][b] for k in keys])
        pts.append(ia.wmean(d, w))
        idx = rng.integers(0, len(d), size=(ia.NBOOT, len(d)))
        acc += (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
    acc /= len(sets)
    lo, hi = np.percentile(acc, [2.5, 97.5])
    return float(np.mean(pts)), float(lo), float(hi)


def main():
    T = json.load(open(RES / "test_analysis.json"))
    sets = [s for s in ["EV17", "EV18", "Cholec", "SAR"] if s in T]
    out = {"sets": sets}
    g = [T[s]["per_seq"][k]["method"] - T[s]["per_seq"][k]["C0"] for s in sets for k in T[s]["per_seq"]]
    out["per_seq_gain"] = {"median": float(np.median(g)), "mean": float(np.mean(g)), "n": len(g),
                           "n_pos": int(sum(x > 1e-6 for x in g)), "n_neg": int(sum(x < -1e-6 for x in g)),
                           "n_tie": int(sum(abs(x) <= 1e-6 for x in g))}
    # leave one sequence out
    loo = []
    for s in sets:
        for k in T[s]["per_seq"]:
            loo.append((s, k, pooled(T, sets, "method", "C0", drop=(s, k))))
    loo.sort(key=lambda r: r[2][0])
    out["loo_min"] = {"drop": loo[0][:2], "gain_ci": loo[0][2]}
    out["loo_max"] = {"drop": loo[-1][:2], "gain_ci": loo[-1][2]}
    # tracker vs its paths and vs SAM3, pooled and per set; sequence-weighted per set
    out["pooled"] = {f"method-{b}": pooled(T, sets, "method", b) for b in ["C0", "delall", "Q50", "DM"]}
    out["per_set"] = {}
    for s in sets:
        keys = sorted(T[s]["per_seq"])
        w = np.array([T[s]["weights"][k] for k in keys])
        V = {r: np.array([T[s]["per_seq"][k][r] for k in keys]) for r in ["C0", "delall", "Q50", "method", "DM"]}
        r = {}
        for b in ["C0", "delall", "Q50"]:
            pt, lo, hi = ia.paired_boot(V["method"], V[b], w)
            r[f"method-{b}"] = [pt, lo, hi]
        r["seq_weighted_gain"] = float(np.mean(V["method"] - V["C0"]))
        r["same_as_Q50"] = int(sum(abs(V["method"] - V["Q50"]) < 1e-6))
        r["same_as_delall"] = int(sum(abs(V["method"] - V["delall"]) < 1e-6))
        # shares of the tracker's advantage over first frame only, by sequence
        contrib = (V["method"] - V["delall"]) * w / w.sum()
        order = np.argsort(-contrib)
        r["adv_over_delall"] = float(contrib.sum())
        r["top_contrib_over_delall"] = [(keys[i], round(float(contrib[i]), 2)) for i in order[:3]]
        contrib0 = (V["method"] - V["C0"]) * w / w.sum()
        o0 = np.argsort(-np.abs(contrib0))
        r["top_contrib_over_C0"] = [(keys[i], round(float(contrib0[i]), 2)) for i in o0[:3]]
        out["per_set"][s] = r
    # online variant: path switches
    sw = {}
    for s in sets:
        A = json.load(open(RES / PATHS[s][0] / "retention_decisions.json"))["sequences"]
        B = json.load(open(RES / PATHS[s][1] / "retention_decisions.json"))["sequences"]
        n_sw, n_fr, runs = 0, 0, []
        for k in T[s]["per_seq"]:
            SA = dict(zip(A[k]["tested"], A[k]["scores"]))
            SB = dict(zip(B[k]["tested"], B[k]["scores"]))
            ts = sorted(set(SA) | set(SB))
            pick = [1 if SB.get(t, -1.0) > SA.get(t, -1.0) else 0 for t in ts]
            ch = sum(1 for i in range(1, len(pick)) if pick[i] != pick[i - 1])
            n_sw += ch
            n_fr += len(pick)
            runs.append(len(pick) / (ch + 1))
        sw[s] = {"switches": n_sw, "frames": n_fr, "switch_per_100_frames": 100 * n_sw / n_fr,
                 "median_mean_run_length": float(np.median(runs))}
    out["online_switching"] = sw
    (RES / "panel_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
