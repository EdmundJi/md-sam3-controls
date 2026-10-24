"""Statistics for the IPCAI 2027 revision (no GPU).

Reads per_video.json from each run directory under RESULTS (default:
results_ipcai/ next to this repo's root) and prints/writes:
  - aggregate bIoU per configuration (frame-weighted over sequences)
  - paired sequence bootstrap CIs between deterministic configurations
  - for multi-seed controls: the distribution of the aggregate contrast over
    draws, and a nested bootstrap CI (resample sequences, then draw one control
    seed per sequence in each replicate)
  - per-sequence wins/losses
Usage: python scripts/analysis.py [RESULTS_DIR]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
NBOOT = 20000
RNG_SEED = 0


def load(run):
    """{seq: (bIoU, n_frames, row)} for one run directory, or None."""
    p = RES / run / "per_video.json"
    if not p.exists():
        return None
    rows = json.load(open(p))
    out = {}
    for r in rows:
        key = r.get("seq_tag") or f"ds{r.get('dataset_id', r.get('seq_id'))}"
        out[key] = (float(r["challenge_iou"]), int(r["num_frames"]), r)
    return out


def vec(run_data, keys):
    return np.array([run_data[k][0] for k in keys])


def weights(run_data, keys):
    return np.array([run_data[k][1] for k in keys], dtype=float)


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def paired_boot(a, b, w, rng=None):
    """CI of the frame-weighted mean of a-b, resampling sequences. Every contrast
    uses the same resampling indices (fresh generator, fixed seed), so intervals
    do not depend on the order in which contrasts are computed."""
    n = len(a)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n, size=(NBOOT, n))
    d = a - b
    stats = (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return wmean(d, w), float(lo), float(hi)


def nested_boot(a, C, w, rng=None):
    """a: (S,) DM per sequence; C: (K,S) control draws. Each replicate
    resamples sequences and picks one control draw per picked sequence."""
    K, S = C.shape
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, S, size=(NBOOT, S))
    kk = rng.integers(0, K, size=(NBOOT, S))
    d = a[idx] - C[kk, idx]
    stats = (d * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def wl(a, b, tol=1e-6):
    return int((a - b > tol).sum()), int((b - a > tol).sum())


def dm_fullset(prefix):
    """Sequences on which the reference DM run deletes every examined entry."""
    p = RES / f"{prefix}_C2_ref" / "retention_decisions.json"
    if not p.exists():
        return set()
    summ = json.load(open(p))["summary"]
    return {k for k, v in summ.items() if v["deleted"] == v["tested"]}


def fill_control(r, dm, full):
    """Controls may be run only where DM leaves a choice. Where DM deletes every
    entry, a control with the same count deletes the same set and executes the
    same program, so its result equals DM's; fill those sequences from DM."""
    if r is None:
        return None
    missing = set(dm) - set(r)
    if not missing:
        return r
    if not missing <= full:
        return None          # incomplete run, not a subset run
    r = dict(r)
    for k in missing:
        r[k] = dm[k]
    return r


def analyze(prefix, label, out):
    rng = np.random.default_rng(RNG_SEED)
    dm = load(f"{prefix}_C2_ref")
    c0 = load(f"{prefix}_C0_ref")
    if dm is None or c0 is None:
        print(f"[{label}] reference runs missing")
        return
    keys = sorted(dm)
    full = dm_fullset(prefix)
    w = weights(dm, keys)
    a = vec(dm, keys)
    res = {"n_seq": len(keys), "n_full_deletion": len(full), "n_frames": int(w.sum()), "sequences": keys,
           "C2": wmean(a, w), "C0": wmean(vec(c0, keys), w)}
    print(f"\n===== {label}: {len(keys)} sequences, {int(w.sum())} frames =====")
    print(f"C0 {res['C0']:.2f}   C2 (DM) {res['C2']:.2f}")

    # deterministic contrasts
    det = {}
    for other in ["C0", "C1", "C4", "C4-GG", "C7-store", "C2-even", "C0_png", "C2_png", "C2_delall"]:
        run = load(f"{prefix}_{other}_ref") or load(f"{prefix}_{other}")
        if other == "C2-even":
            run = fill_control(run, dm, full)
        if run is None or sorted(run) != keys:
            continue
        b = vec(run, keys)
        pt, lo, hi = paired_boot(a, b, w, rng)
        W, L = wl(a, b)
        det[other] = {"agg": wmean(b, w), "C2_minus": pt, "ci": [lo, hi], "W": W, "L": L}
        print(f"  C2 - {other:9s} agg={wmean(b, w):6.2f}  diff={pt:+6.2f} "
              f"[{lo:+6.2f},{hi:+6.2f}]  W/L={W}/{L}")
    # other useful deterministic pairs
    pairs = [("C2", "C0"), ("C1", "C0"), ("C4", "C2"), ("C4-GG", "C2"), ("C4", "C4-GG"),
             ("C7-store", "C4")]
    runs = {"C2": dm, "C0": c0}
    for nm in ["C1", "C4", "C4-GG", "C7-store"]:
        r = load(f"{prefix}_{nm}_ref")
        if r is not None and sorted(r) == keys:
            runs[nm] = r
    res["pairs"] = {}
    for x, y in pairs:
        if x in runs and y in runs:
            pt, lo, hi = paired_boot(vec(runs[x], keys), vec(runs[y], keys), w, rng)
            W, L = wl(vec(runs[x], keys), vec(runs[y], keys))
            res["pairs"][f"{x}-{y}"] = {"diff": pt, "ci": [lo, hi], "W": W, "L": L}
            print(f"  {x:>8s} - {y:<6s} {pt:+6.2f} [{lo:+6.2f},{hi:+6.2f}] W/L={W}/{L}")
    res["deterministic"] = det
    da = load(f"{prefix}_C2_delall")
    if da is not None and sorted(da) == keys:
        pt, lo, hi = paired_boot(vec(da, keys), vec(c0, keys), w, rng)
        W, L = wl(vec(da, keys), vec(c0, keys))
        same = [k for k in keys if abs(da[k][0] - dm[k][0]) < 1e-9]
        res["delall"] = {"agg": wmean(vec(da, keys), w), "minus_C0": pt, "ci": [lo, hi], "W": W, "L": L,
                         "identical_to_DM_on": same}
        print(f"  delete-all - C0 {pt:+6.2f} [{lo:+6.2f},{hi:+6.2f}] W/L={W}/{L}; identical to DM on "
              f"{len(same)}/{len(keys)} sequences (full-deletion set: {len(full)})")

    # multi-seed controls
    res["controls"] = {}
    for mode in ["perm", "shift"]:
        draws = []
        for s in range(1, 101):
            r = fill_control(load(f"{prefix}_C2-{mode}_s{s}"), dm, full)
            if r is not None and sorted(r) == keys:
                draws.append(vec(r, keys))
        if not draws:
            continue
        C = np.stack(draws)                       # (K, S)
        aggs = np.array([wmean(c, w) for c in C])
        contrasts = res["C2"] - aggs
        lo, hi = nested_boot(a, C, w, rng)
        cmean = C.mean(0)
        W, L = wl(a, cmean)
        frac_beat = (a[None, :] > C).mean(0)       # per sequence: share of draws DM beats
        # random vs no deletion (nested over control draws, C0 deterministic)
        c0v = vec(c0, keys)
        lo0, hi0 = nested_boot(c0v, C, w, rng)     # C0 - control
        res["controls"][mode] = {
            "K": len(draws), "agg_mean": float(aggs.mean()), "agg_min": float(aggs.min()),
            "agg_max": float(aggs.max()), "C2_minus_mean": float(contrasts.mean()),
            "C2_minus_min": float(contrasts.min()), "C2_minus_max": float(contrasts.max()),
            "nested_ci": [lo, hi], "W": W, "L": L,
            "draws_where_C2_wins_agg": int((contrasts > 0).sum()),
            "ctrl_minus_C0_mean": float(aggs.mean() - res["C0"]),
            "ctrl_minus_C0_nested_ci": [-hi0, -lo0],
            "per_seq": {k: {"C2": float(a[i]), "C0": float(c0v[i]),
                            "ctrl_mean": float(cmean[i]), "ctrl_min": float(C[:, i].min()),
                            "ctrl_max": float(C[:, i].max()), "frac_C2_beats": float(frac_beat[i])}
                        for i, k in enumerate(keys)},
        }
        print(f"  {mode} x{len(draws)}: agg {aggs.mean():.2f} ({aggs.min():.2f}-{aggs.max():.2f})"
              f"  C2-ctrl {contrasts.mean():+.2f} ({contrasts.min():+.2f}..{contrasts.max():+.2f})"
              f" nested CI [{lo:+.2f},{hi:+.2f}]  W/L={W}/{L}  C2 wins agg in "
              f"{(contrasts > 0).sum()}/{len(draws)} draws;  ctrl-C0 {aggs.mean() - res['C0']:+.2f}"
              f" [{-hi0:+.2f},{-lo0:+.2f}]")
        for i, k in enumerate(keys):
            print(f"      {k:8s} C0={c0v[i]:6.2f} C2={a[i]:6.2f} ctrl={cmean[i]:6.2f} "
                  f"[{C[:, i].min():6.2f},{C[:, i].max():6.2f}] C2>ctrl in {frac_beat[i]*100:3.0f}% draws")
    out[label] = res


def capacity(out):
    rng = np.random.default_rng(RNG_SEED)
    res = {}
    for cfg in ["C0", "C2"]:
        base = load(f"EV17_{cfg}_ref")
        if base is None:
            continue
        keys = sorted(base)
        w = weights(base, keys)
        a = vec(base, keys)
        row = {"nm7": wmean(a, w)}
        for nm in [6, 5, 4, 3, 2]:
            r = load(f"EV17_{cfg}_nm{nm}")
            if r is None:
                continue
            b = vec(r, keys)
            pt, lo, hi = paired_boot(b, a, w, rng)
            row[f"nm{nm}"] = {"agg": wmean(b, w), "diff": pt, "ci": [lo, hi]}
        res[cfg] = row
        print(f"capacity {cfg}: " + "  ".join(
            f"{k}={v if isinstance(v, float) else v['agg']:.2f}" +
            ("" if isinstance(v, float) else f"({v['diff']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}])")
            for k, v in row.items()))
    # DM minus baseline at each capacity (same slots)
    c0, c2 = load("EV17_C0_ref"), load("EV17_C2_ref")
    if c0 is not None and c2 is not None:
        keys = sorted(c0)
        w = weights(c0, keys)
        gap = {"nm7": dict(zip(["diff", "lo", "hi"], paired_boot(vec(c2, keys), vec(c0, keys), w, rng)))}
        for nm in [6, 5, 4, 3, 2]:
            a, b = load(f"EV17_C2_nm{nm}"), load(f"EV17_C0_nm{nm}")
            if a is None or b is None:
                continue
            gap[f"nm{nm}"] = dict(zip(["diff", "lo", "hi"], paired_boot(vec(a, keys), vec(b, keys), w, rng)))
            gap[f"nm{nm}"]["W"], gap[f"nm{nm}"]["L"] = wl(vec(a, keys), vec(b, keys))
        res["DM_minus_C0"] = gap
        print("capacity DM-C0: " + "  ".join(f"{k}={v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]" for k, v in gap.items()))
    out["capacity"] = res


if __name__ == "__main__":
    out = {}
    analyze("EV17", "EV17", out)
    analyze("EV18all", "EV18", out)
    capacity(out)
    (RES / "analysis.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RES / 'analysis.json'}")


def metrics(run):
    p = RES / run / "metrics.json"
    return json.load(open(p)) if p.exists() else None


def class_table(prefix, label, out):
    """Aggregate class metrics (instantiable-restricted) per configuration;
    multi-seed controls are averaged over draws."""
    rows = {}
    names = ["C0_ref", "C1_ref", "C2_ref", "C4_ref", "C4-GG_ref", "C7_ref", "C7-store_ref",
             "C2-even", "C2-SRG", "C4-SRG", "C0_png", "C2_png"]
    for n in names:
        m = metrics(f"{prefix}_{n}")
        if m:
            rows[n] = m
    for mode in ["perm", "shift"]:
        ref = metrics(f"{prefix}_C2_ref")
        ms = [metrics(f"{prefix}_C2-{mode}_s{s}") for s in range(1, 101)]
        # subset runs (only sequences where DM leaves a choice) are not comparable here
        ms = [m for m in ms if m and ref and m.get("num_frames") == ref.get("num_frames")]
        if ms:
            avg = {"challenge_iou": float(np.mean([m["challenge_iou"] for m in ms])),
                   "mciou_instantiable": float(np.mean([m["mciou_instantiable"] for m in ms])),
                   "per_class_instantiable": {c: float(np.mean([m["per_class_instantiable"][c] for m in ms]))
                                              for c in ms[0]["per_class_instantiable"]},
                   "K": len(ms)}
            rows[f"C2-{mode} (mean of {len(ms)})"] = avg
    print(f"\n----- {label}: class metrics (instantiable-restricted) -----")
    for n, m in rows.items():
        pc = m.get("per_class_instantiable", {})
        extra = "  ".join(f"{c}={v:.2f}" for c, v in pc.items() if v)
        anchor = ""
        if "biou_nonempty" in m:
            anchor = f"  [bIoU_nonempty={m['biou_nonempty']:.2f} ciou_cls_gt={m.get('ciou_cls_gt', 0):.2f} " \
                     f"ciou_cls_union={m.get('ciou_cls_union', 0):.2f} cum_miou={m.get('cum_miou', 0):.2f}]"
        print(f"  {n:24s} bIoU={m['challenge_iou']:.2f} mcIoU*={m.get('mciou_instantiable', 0):.2f}  {extra}{anchor}")
    out[f"{label}_classes"] = rows


def retention_rates(prefix, label, out):
    p = RES / f"{prefix}_C2_ref" / "retention_decisions.json"
    if not p.exists():
        return
    d = json.load(open(p))["summary"]
    tot_t = sum(v["tested"] for v in d.values()); tot_d = sum(v["deleted"] for v in d.values())
    print(f"\n{label} retention: {tot_d}/{tot_t} = {100 * tot_d / tot_t:.1f}% deleted; per sequence: " +
          ", ".join(f"{k}={100 * v['rate']:.0f}%" for k, v in sorted(d.items())))
    out[f"{label}_retention"] = {"deleted": tot_d, "tested": tot_t, "per_seq": d}


if __name__ == "__main__":
    out = json.loads((RES / "analysis.json").read_text()) if (RES / "analysis.json").exists() else {}
    for pre, lab in [("EV17", "EV17"), ("EV18all", "EV18")]:
        class_table(pre, lab, out)
        retention_rates(pre, lab, out)
    (RES / "analysis.json").write_text(json.dumps(out, indent=1))


def identities(out):
    """C7 must equal C4 (DFS structurally inert): compare per-sequence values."""
    for pre in ["EV17", "EV18all"]:
        a, b = load(f"{pre}_C4_ref"), load(f"{pre}_C7_ref")
        if a and b:
            same = all(abs(a[k][0] - b[k][0]) < 1e-9 for k in a)
            print(f"{pre}: C7 == C4 on every sequence: {same}")
            out[f"{pre}_C7_eq_C4"] = same
    for pre in ["EV17", "EV18all"]:
        p = RES / f"{pre}_C2_ref" / "retention_decisions.json"
        if p.exists():
            d = json.load(open(p))["summary"]
            free = [k for k, v in d.items() if 0.05 < v["rate"] < 0.95]
            alld = [k for k, v in d.items() if v["deleted"] == v["tested"]]
            nond = [k for k, v in d.items() if v["deleted"] == 0]
            print(f"{pre}: {len(free)}/{len(d)} sequences with 5-95% deletion: {sorted(free)}; "
                  f"all entries deleted on {len(alld)}: {sorted(alld)}; none deleted on {len(nond)}")
            out[f"{pre}_free_sequences"] = sorted(free)
            out[f"{pre}_all_deleted"] = sorted(alld)


if __name__ == "__main__":
    out = json.loads((RES / "analysis.json").read_text())
    identities(out)
    (RES / "analysis.json").write_text(json.dumps(out, indent=1))
