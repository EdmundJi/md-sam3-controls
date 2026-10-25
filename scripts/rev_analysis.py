"""Analyses added after the internal review (no GPU): lossless-input robustness,
rank-rule rate sensitivity on the test sets, the top-6 bank on development
data, deletion rates, and runtime / peak memory from the exclusive timing runs.
All of these were run after the method was chosen and are not candidates.
Usage: python scripts/rev_analysis.py [RESULTS_DIR]   -> RESULTS_DIR/rev_analysis.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as ia  # noqa: E402

RES = ia.RES
PRE = {"EV17": "EV17", "EV18": "EV18all", "Cholec": "TESTcholecL"}
C0 = {"EV17": "EV17_C0_ref", "EV18": "EV18all_C0_ref", "Cholec": "TESTcholecL_keep"}


def jpeg_runs(s):
    p = PRE[s]
    return {"C0": C0[s], "DM": {"EV17": "EV17_C2_ref", "EV18": "EV18all_C2_ref", "Cholec": "TESTcholecL_DM"}[s],
            "delall": {"EV17": "EV17_C2_delall", "EV18": "EV18all_C2_delall", "Cholec": "TESTcholecL_delall"}[s],
            "Q50": f"{p}_Q50_sp" if s != "Cholec" else "TESTcholecL_Q50",
            "method": f"{p}_FUSE-delall-Q50_seq", "online": f"{p}_FUSE-delall-Q50_frame"}


def png_runs(s):
    p = PRE[s]
    return {"C0": f"{p}_keep_png" if s == "Cholec" else f"{p}_C0_png", "DM": "EV17_C2_png" if s == "EV17" else f"{p}_DM_png",
            "delall": f"{p}_delall_png", "Q50": f"{p}_Q50_png",
            "method": f"{p}_FUSE-delall-Q50_png_seq", "online": f"{p}_FUSE-delall-Q50_png_frame"}


def rate_runs(s):
    p = PRE[s]
    return {f"Q{q}": f"{p}_Q{q}" for q in (25, 75)} | \
           {f"two-path Q{q}": f"{p}_FUSE-delall-Q{q}_seq" for q in (25, 75)}


def table(runs_of, ref="C0"):
    """Per set: aggregate bIoU of each run; pooled gain over `ref` with a CI that
    resamples sequences within each set (as in the paper's Table 1)."""
    out, stacks = {}, {}
    for s in PRE:
        runs = runs_of(s)
        base = ia.load(C0[s]) if ref == "C0j" else ia.load(runs[ref])
        if base is None:
            continue
        keys = sorted(base)
        w = ia.weights(base, keys)
        out[s] = {}
        for name, r in runs.items():
            d = ia.load(r)
            if d is None or sorted(d) != keys:
                continue
            out[s][name] = ia.wmean(ia.vec(d, keys), w)
            stacks.setdefault(name, []).append((ia.vec(d, keys) - ia.vec(base, keys), w))
    pooled = {}
    for name, items in stacks.items():
        if len(items) != len(PRE):
            continue
        rng = np.random.default_rng(ia.RNG_SEED)   # fresh per row, as in make_tables_v2.py
        acc = np.zeros(ia.NBOOT)
        for d, w in items:
            idx = rng.integers(0, len(d), size=(ia.NBOOT, len(d)))
            acc += (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
        acc /= len(items)
        lo, hi = np.percentile(acc, [2.5, 97.5])
        pooled[name] = {"gain": float(np.mean([ia.wmean(d, w) for d, w in items])), "ci": [float(lo), float(hi)],
                        "W": int(sum((d > 1e-6).sum() for d, _ in items)), "L": int(sum((d < -1e-6).sum() for d, _ in items))}
    return {"agg": out, "pooled": pooled}


def deletion_rates():
    out = {}
    for s in PRE:
        for name, r in [("Q50", jpeg_runs(s)["Q50"]), ("DM", jpeg_runs(s)["DM"]), ("topk", f"{PRE[s]}_topk")]:
            f = RES / r / "retention_decisions.json"
            if not f.exists():
                continue
            d = json.load(open(f))["sequences"]
            t = sum(len(v["tested"]) for v in d.values())
            x = sum(len(v["deleted"]) for v in d.values())
            full = sum(1 for v in d.values() if v["tested"] and len(v["deleted"]) == len(v["tested"]))
            out.setdefault(s, {})[name] = {"rate": 100 * x / t, "full_deletion_seqs": full, "n_seqs": len(d)}
    return out


def dev_topk():
    """Development gain of the top-6 bank, as in scripts/dev_analysis.py (mean over the
    two development sets of the frame-weighted gain over keeping all memory)."""
    gains = {}
    for p in ["DEVev17tr", "DEVcholecL"]:
        a, b = ia.load(f"{p}_topk"), ia.load(f"{p}_keep")
        if a is None or b is None:
            return None
        keys = sorted(b)
        w = ia.weights(b, keys)
        gains[p] = ia.wmean(ia.vec(a, keys) - ia.vec(b, keys), w)
    gains["mean"] = float(np.mean(list(gains.values())))
    dev = json.load(open(RES / "dev_analysis.json"))
    ranking = [r for r in dev["ranking"]]
    gains["would_rank"] = 1 + sum(1 for _, g in ranking if g > gains["mean"])
    gains["n_candidates"] = len(ranking)
    return gains


def sign_tests():
    """Sequence-level two-sided sign tests (ties dropped) on the 53 test sequences,
    each rule against SAM3 and the two-path tracker against each single rule."""
    from math import comb
    T = json.load(open(RES / "test_analysis.json"))
    sets = ["EV17", "EV18", "Cholec"]

    def test(a, b):
        d = [v[a] - v[b] for s in sets for v in T[s]["per_seq"].values()]
        w, l = sum(x > 1e-6 for x in d), sum(x < -1e-6 for x in d)
        p = min(1.0, 2 * sum(comb(w + l, i) for i in range(min(w, l) + 1)) / 2 ** (w + l))
        return {"W": w, "L": l, "p": p}
    out = {f"{r} vs C0": test(r, "C0") for r in ["DM", "delall", "Q50", "topk", "method", "fuse_frame"]}
    out |= {f"method vs {r}": test("method", r) for r in ["DM", "delall", "Q50", "topk"]}
    out["rank rule beats all 20 random draws on all three sets, joint p"] = (1 / 21) ** 3
    return out


def selector_margins():
    """Per video: confidence margin of the two-path selector (mean G of the rank-rule path
    minus mean G of the memory-free path) against the realised bIoU difference."""
    T = json.load(open(RES / "test_analysis.json"))
    pairs = {"EV17": ("EV17_C2_delall_sp", "EV17_Q50_sp"), "EV18": ("EV18all_C2_delall_sp", "EV18all_Q50_sp"),
             "Cholec": ("TESTcholecL_delall", "TESTcholecL_Q50")}
    rows = []
    for s, (a, b) in pairs.items():
        A = json.load(open(RES / a / "retention_decisions.json"))["sequences"]
        B = json.load(open(RES / b / "retention_decisions.json"))["sequences"]
        for seq, v in T[s]["per_seq"].items():
            m = float(np.mean(B[seq]["scores"]) - np.mean(A[seq]["scores"]))
            d = v["Q50"] - v["delall"]
            rows.append({"set": s, "seq": seq, "margin": m, "diff": d, "regret": v["best_path"] - v["method"],
                         "wrong": bool(abs(d) > 1e-6 and np.sign(m) != np.sign(d))})
    wrong = [r for r in rows if r["wrong"]]
    order = sorted(rows, key=lambda r: -abs(r["margin"]))
    half = len(rows) // 2
    return {"n": len(rows), "n_wrong": len(wrong),
            "wrong_below_1pt": sum(r["regret"] < 1 for r in wrong),
            "wrong": [(r["set"], r["seq"], round(r["margin"], 4), round(r["regret"], 2)) for r in wrong],
            "median_abs_margin_correct": float(np.median([abs(r["margin"]) for r in rows if not r["wrong"]])),
            "median_abs_margin_wrong": float(np.median([abs(r["margin"]) for r in wrong])),
            "correct_in_top_half_by_margin": [sum(not r["wrong"] for r in order[:half]), half],
            "correct_in_bottom_half_by_margin": [sum(not r["wrong"] for r in order[half:]), len(rows) - half],
            "rows": rows}


TIMING = re.compile(r"TIMING (\S+): (\d+) frames, prep ([\d.]+) s, track ([\d.]+) s .* peak ([\d.]+) GiB")


def timing():
    out = {}
    for name, log in [("SAM3", "EV17_C0_time"), ("DM", "EV17_DM_time"), ("first frame only", "EV17_delall_time"),
                      ("rank rule", "EV17_Q50_time"), ("top-6 bank", "EV17_topk")]:
        f = RES / "_logs" / f"{log}.log"
        if not f.exists():
            continue
        rows = TIMING.findall(f.read_text(errors="ignore"))
        if not rows:
            continue
        n = sum(int(r[1]) for r in rows)
        track = sum(float(r[3]) for r in rows)
        out[name] = {"frames": n, "ms_per_frame": 1000 * track / n, "fps": n / track,
                     "peak_gib": max(float(r[4]) for r in rows), "prep_s": sum(float(r[2]) for r in rows)}
    if "first frame only" in out and "rank rule" in out:
        a, b = out["first frame only"], out["rank rule"]
        out["two-path (sequential)"] = {"ms_per_frame": a["ms_per_frame"] + b["ms_per_frame"],
                                        "peak_gib": max(a["peak_gib"], b["peak_gib"])}
        out["two-path (parallel)"] = {"ms_per_frame": max(a["ms_per_frame"], b["ms_per_frame"]),
                                      "peak_gib": a["peak_gib"] + b["peak_gib"]}
    return out


def main():
    out = {"png": table(png_runs), "png_vs_jpeg_C0": table(png_runs, ref="C0j"),
           "jpeg": table(jpeg_runs), "rate": table(rate_runs, ref="C0j"),
           "deletion_rates": deletion_rates(), "dev_topk": dev_topk(), "timing": timing(),
           "sign_tests": sign_tests(), "selector_margins": selector_margins()}
    for k in ["png", "jpeg", "rate"]:
        print(f"\n== {k} ==")
        for s, row in out[k]["agg"].items():
            print(f"  {s:7s} " + "  ".join(f"{n}={v:.2f}" for n, v in row.items()))
        for n, r in out[k]["pooled"].items():
            print(f"  pooled {n:16s} {r['gain']:+.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] W/L {r['W']}/{r['L']}")
    print("\n== deletion rates ==", json.dumps(out["deletion_rates"], indent=None))
    print("== dev top-6 bank ==", out["dev_topk"])
    print("== timing (EV17, exclusive GPU) ==")
    for n, r in out["timing"].items():
        print(f"  {n:22s} {r['ms_per_frame']:.1f} ms/frame, peak {r['peak_gib']:.2f} GiB")
    print("== sign tests ==", {k: (v if isinstance(v, float) else f"{v['W']}/{v['L']} p={v['p']:.3f}")
                                for k, v in out["sign_tests"].items()})
    sm = out["selector_margins"]
    print("== selector ==", {k: v for k, v in sm.items() if k != "rows"})
    (RES / "rev_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
