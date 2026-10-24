"""Oracle, score-ranked and low-capacity controls (no GPU).

Arms delete the same number of memory entries per sequence and differ only in
which entries they delete:
  oracle  the lowest-quality entries (quality: true IoU of the baseline's prediction)
  anti    the highest-quality entries
  score   the lowest-scoring entries (SAM3's score on the baseline run)
  random  uniformly random entries (20 draws)
Rates: R50 = floor(n_s/2) per sequence; DM = DM's per-sequence count (nm7, nm3, nm4).
Usage: python scripts/oracle_analysis.py [RESULTS_DIR]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as ia  # noqa: E402  (reads RESULTS_DIR from argv)

RES = ia.RES


def det(a, b, w):
    pt, lo, hi = ia.paired_boot(a, b, w)
    W, L = ia.wl(a, b)
    return {"diff": pt, "ci": [lo, hi], "W": W, "L": L}


def vs_draws(a, C, w):
    """a: (S,) deterministic arm; C: (K,S) random draws."""
    aggs = np.array([ia.wmean(c, w) for c in C])
    d = ia.wmean(a, w) - aggs
    lo, hi = ia.nested_boot(a, C, w)
    K = len(C)
    wins = int((d > 0).sum())
    W, L = ia.wl(a, C.mean(0))
    return {"diff": float(d.mean()), "ci": [lo, hi], "wins": wins, "K": K,
            "p": (1 + K - wins) / (K + 1), "W": W, "L": L}


def draws(pattern, keys, dm=None, full=frozenset()):
    out = []
    for s in range(1, 101):
        r = ia.load(pattern.format(s=s))
        if dm is not None:
            r = ia.fill_control(r, dm, full)
        if r is not None and sorted(r) == keys:
            out.append(ia.vec(r, keys))
    return np.stack(out) if out else None


def fmt(name, r):
    if "wins" in r:
        return (f"  {name:28s} {r['diff']:+6.2f} [{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}]  "
                f"above {r['wins']}/{r['K']} draws (p={r['p']:.3f})  W/L={r['W']}/{r['L']}")
    return f"  {name:28s} {r['diff']:+6.2f} [{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}]  W/L={r['W']}/{r['L']}"


def block(title, keys, arms, C, base_name="C0"):
    """arms: {name: (S,) vector}; C: random draws. Prints aggregates and contrasts."""
    w = arms["_w"]
    res = {"agg": {k: ia.wmean(v, w) for k, v in arms.items() if k != "_w"}}
    if C is not None:
        aggs = np.array([ia.wmean(c, w) for c in C])
        res["agg"]["random_mean"] = float(aggs.mean())
        res["agg"]["random_range"] = [float(aggs.min()), float(aggs.max())]
    print(f"\n== {title}: {len(keys)} sequences, {int(w.sum())} frames ==")
    print("  " + "  ".join(f"{k}={v:.2f}" for k, v in res["agg"].items() if not isinstance(v, list))
          + (f"  random range {res['agg']['random_range'][0]:.2f}-{res['agg']['random_range'][1]:.2f}"
             f" (K={len(C)})" if C is not None else ""))
    res["vs_random"], res["paired"] = {}, {}
    for k, v in arms.items():
        if k.startswith("_") or C is None or k == base_name:
            continue
        res["vs_random"][k] = vs_draws(v, C, w)
        print(fmt(f"{k} - random", res["vs_random"][k]))
    if C is not None and base_name in arms:
        lo, hi = ia.nested_boot(arms[base_name], C, w)        # base - random
        aggs = np.array([ia.wmean(c, w) for c in C])
        res["random_minus_base"] = {"diff": float(aggs.mean() - ia.wmean(arms[base_name], w)), "ci": [-hi, -lo]}
        print(f"  {'random - ' + base_name:28s} {res['random_minus_base']['diff']:+6.2f} [{-hi:+6.2f},{-lo:+6.2f}]")
    names = [k for k in arms if not k.startswith("_")]
    for x in names:
        for y in names:
            if x != y and (y == base_name or (x, y) in [("oracle", "anti"), ("oracle", "score"),
                                                        ("oracle", "DM"), ("DM", "delall"),
                                                        ("score", "anti"), ("oracle", "delall"),
                                                        ("score", "delall")]):
                res["paired"][f"{x}-{y}"] = det(arms[x], arms[y], w)
                print(fmt(f"{x} - {y}", res["paired"][f"{x}-{y}"]))
    res["per_seq"] = {k: {n: float(arms[n][i]) for n in names} for i, k in enumerate(keys)}
    if C is not None:
        for i, k in enumerate(keys):
            res["per_seq"][k]["random_mean"] = float(C[:, i].mean())
    return res


def sanity(prefix, out):
    a, b = ia.load(f"{prefix}_C0_scores"), ia.load(f"{prefix}_C0_ref")
    if a and b:
        same = sorted(a) == sorted(b) and all(abs(a[k][0] - b[k][0]) < 1e-9 for k in b)
        print(f"{prefix}: C0_scores equals C0_ref on every sequence: {same}")
        out[f"{prefix}_scores_run_equals_C0"] = same


def fixed_rate(prefix, label, out, pct=50):
    c0 = ia.load(f"{prefix}_C0_ref")
    keys = sorted(c0)
    arms = {"C0": ia.vec(c0, keys), "_w": ia.weights(c0, keys)}
    for a in ["oracle", "score", "anti"]:
        r = ia.load(f"{prefix}_R{pct}-{a}")
        if r is not None and sorted(r) == keys:
            arms[a] = ia.vec(r, keys)
    if len(arms) == 2:
        return
    da = ia.load(f"{prefix}_C2_delall")
    if pct == 50 and da is not None and sorted(da) == keys:
        arms["delall"] = ia.vec(da, keys)
    C = draws(f"{prefix}_R{pct}-perm_s{{s}}", keys)
    out[f"{label}_R{pct}"] = block(f"{label}, delete {pct}% of each sequence's entries", keys, arms, C)


def dmcount(prefix, label, out, nm=None):
    sfx = f"_nm{nm}" if nm else ""
    dmrun = f"{prefix}_C2_nm{nm}" if nm else f"{prefix}_C2_ref"
    dm = ia.load(dmrun)
    c0 = ia.load(f"{prefix}_C0_nm{nm}" if nm else f"{prefix}_C0_ref")
    if dm is None or c0 is None:
        return
    keys = sorted(dm)
    p = RES / dmrun / "retention_decisions.json"
    summ = json.load(open(p))["summary"]
    full = {k for k, v in summ.items() if v["deleted"] == v["tested"]}
    arms = {"C0": ia.vec(c0, keys), "DM": ia.vec(dm, keys), "_w": ia.weights(dm, keys)}
    for a in ["oracle", "anti"]:
        r = ia.fill_control(ia.load(f"{prefix}_C2-{a}{sfx}"), dm, full)
        if r is not None and sorted(r) == keys:
            arms[a] = ia.vec(r, keys)
    da = ia.load(f"{prefix}_C2_delall{sfx}")
    if da is not None and sorted(da) == keys:
        arms["delall"] = ia.vec(da, keys)
    pat = f"{prefix}_C2-perm{sfx}_s{{s}}"
    C = draws(pat, keys, dm, full)
    title = f"{label}, DM's per-sequence counts" + (f", num_maskmem={nm}" if nm else "")
    out[f"{label}_DMcount{sfx}"] = block(title, keys, arms, C)


def diagnostics(ds, out):
    p = RES / "oracle" / ds / "quality.json"
    if not p.exists():
        return
    d = json.load(open(p))
    diag = d["diagnostics"]
    rho = {k: v.get("spearman_score_quality") for k, v in diag.items()}
    seqs = d["sequences"]
    if all("score" in v for v in seqs.values()):
        # pooled within-sequence rank correlation: average of per-sequence rho, and pooled raw
        vals = [r for r in rho.values() if r is not None]
        print(f"\n{ds}: Spearman(score, quality) per sequence: "
              + " ".join(f"{k}={'NA' if r is None else f'{r:+.2f}'}" for k, r in rho.items())
              + f"  mean {np.mean(vals):+.2f}")
        ov = {k: (v.get("score_r50_overlap_oracle"), v.get("chance_overlap_r50")) for k, v in diag.items()}
        tot = sum(o for o, _ in ov.values()), sum(c for _, c in ov.values())
        print(f"{ds}: score-ranked half vs oracle half: overlap {tot[0]} (chance {tot[1]:.0f})")
        out[f"{ds}_diag"] = {"spearman": rho, "mean_spearman": float(np.mean(vals)),
                             "r50_overlap": tot[0], "r50_chance": tot[1]}
    tot = sum(v["dm_overlap_oracle"] for v in diag.values()), sum(v["chance_overlap"] for v in diag.values())
    print(f"{ds}: DM's deletions vs oracle's (same counts): overlap {tot[0]} (chance {tot[1]:.0f})")
    out.setdefault(f"{ds}_diag", {})["dm_overlap"] = tot


if __name__ == "__main__":
    out = {}
    for pre in ["EV17", "EV18all"]:
        sanity(pre, out)
    for pct in [50, 25, 75]:
        fixed_rate("EV17", "EV17", out, pct)
        fixed_rate("EV18all", "EV18", out, pct)
    dmcount("EV17", "EV17", out)
    dmcount("EV18all", "EV18", out)
    for nm in [4, 3]:
        dmcount("EV17", "EV17", out, nm)
    for ds in ["EV17", "EV18"]:
        diagnostics(ds, out)
    (RES / "oracle_analysis.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RES / 'oracle_analysis.json'}")
