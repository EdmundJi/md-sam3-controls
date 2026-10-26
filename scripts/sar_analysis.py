"""Analysis of the preregistered fourth test set, SAR-RARP50
(PREREGISTRATION_SAR-RARP50.md), and the four-set summaries (no GPU).
Run scripts/test_analysis.py first (it includes SAR once the runs exist).

H1 (primary): two-path tracker (per video) vs SAM3 on SAR-RARP50: frame-weighted
    gain, 95% bootstrap CI over the 50 operations and two-sided sign test over
    operations (Amendment 1); the sequence-level versions are sensitivity analyses.
    Decision: supported if the CI lies above 0; replicated in direction if gain > 0
    and sign p < 0.05; otherwise not supported.
H2: rank rule vs random deletion at its per-sequence counts (20 draws).
Descriptive: every rule vs SAM3, gap to the best single rule, deletion rates,
    sequences fully emptied by DM, selector accuracy; the official test release
    (videos 41-50) as a subset; sign tests over all four test sets.
Usage: python scripts/sar_analysis.py [RESULTS_DIR]   -> RESULTS_DIR/sar_analysis.json
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as ia  # noqa: E402

RES = ia.RES
RULES = ["DM", "topk", "delall", "Q50", "method", "fuse_frame"]
SINGLE = ["DM", "topk", "delall", "Q50"]
OFFICIAL_TEST = {f"video_{i}" for i in range(41, 51)}


def op(k):
    """Operation of a video file: video_11_2 -> 11 (Amendment 1: resampling unit)."""
    return k.split("_")[1]


def cluster_contrast(ps, wts, keys, a_key, b_key, rng_seed=ia.RNG_SEED):
    """Frame-weighted mean of a-b; 95% CI from a bootstrap over operations (all files
    of a drawn operation together); sign test over operations (frame-weighted gain
    per operation, ties dropped)."""
    ops = sorted({op(k) for k in keys})
    idx_of = {o: [i for i, k in enumerate(keys) if op(k) == o] for o in ops}
    d = np.array([ps[k][a_key] - ps[k][b_key] for k in keys])
    w = np.array([wts[k] for k in keys])
    num = np.array([(d[idx_of[o]] * w[idx_of[o]]).sum() for o in ops])
    den = np.array([w[idx_of[o]].sum() for o in ops])
    rng = np.random.default_rng(rng_seed)
    pick = rng.integers(0, len(ops), size=(ia.NBOOT, len(ops)))
    stats = num[pick].sum(1) / den[pick].sum(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    g_op = num / den
    W, L = int((g_op > 1e-6).sum()), int((g_op < -1e-6).sum())
    return {"gain": float(num.sum() / den.sum()), "ci": [float(lo), float(hi)], "W_ops": W, "L_ops": L,
            "ties_ops": len(ops) - W - L, "sign_p_ops": sign_p(W, L), "n_ops": len(ops)}


def cluster_nested(a, C, w, keys, rng_seed=ia.RNG_SEED):
    """Nested bootstrap for a rule against K random-deletion draws at the operation
    level (Amendment 1): each replicate resamples operations, all files of a drawn
    operation together, and draws one control seed per resampled file."""
    K = C.shape[0]
    ops = sorted({op(k) for k in keys})
    U = [np.array([i for i, k in enumerate(keys) if op(k) == o]) for o in ops]
    rng = np.random.default_rng(rng_seed)
    pick = rng.integers(0, len(U), size=(ia.NBOOT, len(U)))
    stats = np.empty(ia.NBOOT)
    for b in range(ia.NBOOT):
        idx = np.concatenate([U[j] for j in pick[b]])
        kk = rng.integers(0, K, size=len(idx))
        stats[b] = ((a[idx] - C[kk, idx]) * w[idx]).sum() / w[idx].sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def contrasts(ps, wts, keys, ref="C0"):
    w = np.array([wts[k] for k in keys])
    out = {}
    for r in RULES:
        a = np.array([ps[k][r] for k in keys])
        b = np.array([ps[k][ref] for k in keys])
        pt, lo, hi = ia.paired_boot(a, b, w)
        W, L = ia.wl(a, b)
        out[r] = {"gain": pt, "ci": [lo, hi], "W": W, "L": L, "sign_p": sign_p(W, L)}
    return out


def main():
    T = json.load(open(RES / "test_analysis.json"))
    if "SAR" not in T:
        sys.exit("SAR-RARP50 not in test_analysis.json yet")
    S = T["SAR"]
    keys = sorted(S["per_seq"])
    out = {"n_seq": len(keys), "n_frames": S["n_frames"], "agg": S["agg"]}
    nf = np.array([S["weights"][k] for k in keys])
    out["frames_per_seq"] = {"median": float(np.median(nf)), "min": int(nf.min()), "max": int(nf.max())}

    # H1 (Amendment 1: operation-level CI and sign test), every rule vs SAM3
    out["vs_C0_cluster"] = {r: cluster_contrast(S["per_seq"], S["weights"], keys, r, "C0") for r in RULES}
    h1 = out["vs_C0_cluster"]["method"]
    out["H1"] = ("supported" if h1["ci"][0] > 0 else
                 "replicated in direction" if h1["gain"] > 0 and h1["sign_p_ops"] < 0.05 else "not supported")
    # sequence-level versions (sensitivity, as originally written)
    out["vs_C0"] = contrasts(S["per_seq"], S["weights"], keys)
    off = [k for k in keys if k in OFFICIAL_TEST]
    w = nf
    if off:
        out["official_test"] = {"n_seq": len(off), "vs_C0": contrasts(S["per_seq"], S["weights"], off)}
        out["official_test"]["agg"] = {r: ia.wmean(np.array([S["per_seq"][k][r] for k in off]),
                                                   np.array([S["weights"][k] for k in off]))
                                       for r in ["C0"] + RULES}
    # gap of the two-path tracker to the best single rule (set level) and to the rank rule, with CIs
    agg = {r: ia.wmean(np.array([S["per_seq"][k][r] for k in keys]), w) for r in ["C0"] + RULES}
    best = max(SINGLE, key=lambda r: agg[r])
    out["best_single_rule"] = best
    out["gap_to_best_single"] = agg["method"] - agg[best]
    out["method_vs_cluster"] = {r: cluster_contrast(S["per_seq"], S["weights"], keys, "method", r)
                                for r in ["DM", "delall", "Q50", "topk"]}
    for r in ["DM", "delall", "Q50", "topk"]:
        a = np.array([S["per_seq"][k]["method"] for k in keys])
        b = np.array([S["per_seq"][k][r] for k in keys])
        pt, lo, hi = ia.paired_boot(a, b, w)
        W, L = ia.wl(a, b)
        out.setdefault("method_vs", {})[r] = {"diff": pt, "ci": [lo, hi], "W": W, "L": L, "sign_p": sign_p(W, L)}

    # H2: controls (computed in test_analysis.py, sequence-level nested bootstrap)
    out["controls"] = S.get("controls", {})
    # ... and with the operation-level nested bootstrap of Amendment 1 (primary for SAR)
    import test_analysis as ta
    out["controls_cluster"] = {}
    for arm, pat in [("Q50", "SAR_Q50-perm_s{s}"), ("method", "SAR_FUSE-delall-Q50perm_s{s}_seq")]:
        C = ta.draws(pat, keys)
        if C is None:
            continue
        a = np.array([S["per_seq"][k][arm] for k in keys])
        lo, hi = cluster_nested(a, C, nf, keys)
        d = ia.wmean(a, nf) - np.array([ia.wmean(c, nf) for c in C])
        out["controls_cluster"][arm] = {"K": len(C), "diff": float(d.mean()), "ci": [lo, hi],
                                        "wins": int((d > 0).sum()), "p": (1 + len(C) - int((d > 0).sum())) / (len(C) + 1)}

    # deletion rates
    for name, run in [("DM", "SAR_DM"), ("Q50", "SAR_Q50"), ("topk", "SAR_topk")]:
        f = RES / run / "retention_decisions.json"
        if f.exists():
            d = json.load(open(f))["sequences"]
            t = sum(len(v["tested"]) for v in d.values())
            x = sum(len(v["deleted"]) for v in d.values())
            full = sum(1 for v in d.values() if v["tested"] and len(v["deleted"]) == len(v["tested"]))
            out.setdefault("deletion", {})[name] = {"rate": 100 * x / t, "full_deletion_seqs": full, "n_seqs": len(d)}

    # selector: per-video choice vs the better path
    A = json.load(open(RES / "SAR_delall" / "retention_decisions.json"))["sequences"]
    B = json.load(open(RES / "SAR_Q50" / "retention_decisions.json"))["sequences"]
    rows = []
    for k in keys:
        v = S["per_seq"][k]
        m = float(np.mean(B[k]["scores"]) - np.mean(A[k]["scores"]))
        d = v["Q50"] - v["delall"]
        regret = v["best_path"] - v["method"]
        rows.append({"seq": k, "margin": m, "diff": d, "regret": regret, "wrong": bool(regret > 1e-6),
                     "kept_Q50": bool(abs(d) > 1e-6 and abs(v["method"] - v["Q50"]) < 1e-6)})
    wrong = [r for r in rows if r["wrong"]]
    out["selector"] = {"n": len(rows), "n_wrong": len(wrong), "wrong_below_1pt": sum(r["regret"] < 1 for r in wrong),
                       "picked_Q50": sum(r["margin"] > 0 for r in rows), "kept_Q50": sum(r["kept_Q50"] for r in rows),
                       "paths_equal": sum(abs(r["diff"]) <= 1e-6 for r in rows),
                       "wrong": [(r["seq"], round(r["margin"], 4), round(r["regret"], 2)) for r in wrong]}

    # sign tests over all test sets in test_analysis.json
    sets = [s for s in ["EV17", "EV18", "Cholec", "SAR"] if s in T]

    def test(a, b):
        d = [v[a] - v[b] for s in sets for v in T[s]["per_seq"].values()]
        W, L = sum(x > 1e-6 for x in d), sum(x < -1e-6 for x in d)
        return {"W": W, "L": L, "p": sign_p(W, L)}
    out["sign_tests_all_sets"] = {"sets": sets}
    out["sign_tests_all_sets"] |= {f"{r} vs C0": test(r, "C0") for r in RULES}
    out["sign_tests_all_sets"] |= {f"method vs {r}": test("method", r) for r in SINGLE}
    n_all = sum(1 for s in sets if T[s].get("controls", {}).get("Q50", {}).get("wins") == T[s]["controls"]["Q50"]["K"])
    out["rank_rule_beats_all_draws_on_n_sets"] = n_all
    out["joint_p_if_independent"] = float(np.prod([1 / (T[s]["controls"]["Q50"]["K"] + 1) for s in sets
                                                   if T[s].get("controls", {}).get("Q50", {}).get("wins")
                                                   == T[s]["controls"]["Q50"]["K"]])) if n_all else None
    if "mean_gain" in T:
        out["mean_gain_all_sets"] = T["mean_gain"]
    # pooled-gain sensitivity (Amendment 1): every sequence weighted equally; leave one set out.
    # Stratified bootstrap; within SAR-RARP50 the resampling unit is the operation.
    def strat(use, seq_weighted=False, a_key="method", b_key="C0"):
        rng = np.random.default_rng(ia.RNG_SEED)
        acc_n, acc_d, acc_m, pts, n_all, d_all = np.zeros(ia.NBOOT), np.zeros(ia.NBOOT), np.zeros(ia.NBOOT), [], 0, 0.0
        for s_ in use:
            ks = sorted(T[s_]["per_seq"])
            w = np.ones(len(ks)) if seq_weighted else np.array([T[s_]["weights"][k] for k in ks], float)
            d = np.array([T[s_]["per_seq"][k][a_key] - T[s_]["per_seq"][k][b_key] for k in ks])
            units = ([np.array([i for i, k in enumerate(ks) if op(k) == o]) for o in sorted({op(k) for k in ks})]
                     if s_ == "SAR" else [np.array([i]) for i in range(len(ks))])
            num = np.array([(d[u] * w[u]).sum() for u in units])
            den = np.array([w[u].sum() for u in units])
            pick = rng.integers(0, len(units), size=(ia.NBOOT, len(units)))
            acc_n += num[pick].sum(1)
            acc_d += den[pick].sum(1)
            acc_m += num[pick].sum(1) / den[pick].sum(1)
            pts.append(num.sum() / den.sum())
            n_all += len(ks)
            d_all += float(d.sum())
        if seq_weighted:                 # mean over all sequences
            return d_all / n_all, np.percentile(acc_n / acc_d, [2.5, 97.5]), n_all
        return float(np.mean(pts)), np.percentile(acc_m / len(use), [2.5, 97.5]), n_all
    g, ci, n = strat(sets, seq_weighted=True)
    out["pooled_seq_weighted"] = {"gain": float(g), "ci": [float(x) for x in ci], "n": n}
    g, ci, n = strat(sets)
    out["pooled_sets_cluster"] = {"gain": float(g), "ci": [float(x) for x in ci], "n": n}
    out["pooled_leave_one_set_out"] = {}
    for drop in sets:
        g, ci, _ = strat([s_ for s_ in sets if s_ != drop])
        out["pooled_leave_one_set_out"][drop] = {"gain": float(g), "ci": [float(x) for x in ci]}
    # failure rate on SAR (preregistered secondary endpoint), operation-level CI
    fr = RES / "failure_rate.json"
    if fr.exists():
        F = json.load(open(fr))
        if "sarrarp" in F["per_seq"]:
            P = F["per_seq"]["sarrarp"]
            ks = sorted(P)
            ps = {k: {r: 100 * P[k][r][0] / max(P[k][r][1], 1) for r in P[k]} for k in ks}
            wf = {k: P[k]["C0"][1] for k in ks}
            out["failure_rate_SAR"] = {"agg": F["agg"].get("SAR"),
                                       "method_vs_C0": cluster_contrast(ps, wf, ks, "method", "C0")}
    (RES / "sar_analysis.json").write_text(json.dumps(out, indent=1))

    c = out["vs_C0_cluster"]["method"]
    print(f"H1 (two-path vs SAM3, operation bootstrap): {c['gain']:+.2f} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}] "
          f"ops W/L {c['W_ops']}/{c['L_ops']} (ties {c['ties_ops']}) sign p={c['sign_p_ops']:.3g} -> {out['H1']}")
    print(f"SAR-RARP50: {len(keys)} sequences, {S['n_frames']} frames "
          f"(per sequence median {np.median(nf):.0f}, {nf.min()}-{nf.max()})")
    print("  bIoU: " + "  ".join(f"{r}={agg[r]:.2f}" for r in ["C0"] + RULES))
    for r, c in out["vs_C0"].items():
        print(f"  {r:10s} - SAM3 {c['gain']:+.2f} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}] W/L {c['W']}/{c['L']} "
              f"sign p={c['sign_p']:.3g}")
    if off:
        print(f"  official test release ({len(off)} videos): method - SAM3 "
              f"{out['official_test']['vs_C0']['method']['gain']:+.2f} "
              f"[{out['official_test']['vs_C0']['method']['ci'][0]:+.2f},"
              f"{out['official_test']['vs_C0']['method']['ci'][1]:+.2f}]")
    print(f"  best single rule {best}; two-path gap {out['gap_to_best_single']:+.2f}")
    for r, c in out["method_vs"].items():
        print(f"  method - {r:7s} {c['diff']:+.2f} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}] W/L {c['W']}/{c['L']} "
              f"p={c['sign_p']:.3g}")
    for arm, c in out["controls"].items():
        print(f"  control {arm}: {c['diff']:+.2f} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}] above {c['wins']}/{c['K']}")
    print(f"  deletion: {out.get('deletion')}")
    print(f"  selector: {out['selector']['n_wrong']} wrong of {len(rows)} "
          f"({out['selector']['wrong_below_1pt']} cost <1 pt); picked rank-rule path {out['selector']['picked_Q50']}")
    print(f"  sign tests over {sets}: " + "; ".join(f"{k} {v['W']}/{v['L']} p={v['p']:.3g}"
                                                  for k, v in out["sign_tests_all_sets"].items() if k != "sets"))
    print(f"  rank rule above all draws on {n_all} sets; joint p {out['joint_p_if_independent']}")


if __name__ == "__main__":
    main()
