"""How often instruments leave the view or enter after the first frame, from the
ground-truth masks, and which path wins on such sequences (review panel request).
Flags per sequence: 'enters' = instrument classes absent from the first frame
hold >= 1% of the later ground-truth instrument pixels (they cannot be tracked); 'leaves' = a class present in the first frame is absent in >= 10% of
the later annotated frames. Path win: rank-rule path vs memory-free path (bIoU).
usage: python scripts/visibility_analysis.py RESULTS_DIR D17 D18 DCH [DSAR]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classaware_offline as ca  # noqa: E402

NAMES = {"ev17": "EV17", "ev18": "EV18", "cholec": "Cholec", "sarrarp": "SAR"}


def main():
    R = Path(sys.argv[1])
    roots = dict(zip(["ev17", "ev18", "cholec", "sarrarp"], sys.argv[2:]))
    T = json.load(open(R / "test_analysis.json"))
    out = {}
    for ds, root in roots.items():
        S = NAMES[ds]
        rows = []
        for _eid, tag, fids, load, init_cls, H, W in ca.seqs(ds, root):
            if tag not in T.get(S, {}).get("per_seq", {}):
                continue
            pres, n, px, px_new = {}, 0, 0, 0
            for t in range(1, len(fids)):
                g = load(fids[t])
                if g is None:
                    continue
                n += 1
                m = g > 0
                px += int(m.sum())
                px_new += int((m & ~np.isin(g, init_cls)).sum())
                for c in np.unique(g):
                    if c:
                        pres[int(c)] = pres.get(int(c), 0) + 1
            v = T[S]["per_seq"][tag]
            rows.append({"seq": tag, "enters": px_new >= 0.01 * max(px, 1), "px": px, "px_new": px_new,
                         "leaves": any(pres.get(c, 0) <= 0.9 * n for c in init_cls),
                         "q50_minus_delall": v["Q50"] - v["delall"], "method_minus_C0": v["method"] - v["C0"],
                         "delall_minus_C0": v["delall"] - v["C0"]})
        def summ(sel):
            return {"n": len(sel), "rank_path_better": sum(r["q50_minus_delall"] > 1e-6 for r in sel),
                    "memory_free_better": sum(r["q50_minus_delall"] < -1e-6 for r in sel),
                    "mean_q50_minus_delall": float(np.mean([r["q50_minus_delall"] for r in sel])) if sel else None,
                    "mean_method_minus_C0": float(np.mean([r["method_minus_C0"] for r in sel])) if sel else None}
        out[S] = {"n_seq": len(rows), "enters": sum(r["enters"] for r in rows), "leaves": sum(r["leaves"] for r in rows),
                  "untrackable_pixel_share": 100 * sum(r["px_new"] for r in rows) / max(sum(r["px"] for r in rows), 1),
                  "leaves_summary": summ([r for r in rows if r["leaves"]]),
                  "stays_summary": summ([r for r in rows if not r["leaves"]]), "rows": rows}
        print(S, {k: v for k, v in out[S].items() if k != "rows"})
    (R / "visibility_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
