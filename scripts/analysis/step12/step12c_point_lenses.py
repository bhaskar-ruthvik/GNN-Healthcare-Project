"""STEP12c: full lens profile (incl. travel and feasibility of new picks) of the steered equal blend vs the references, from a
candidate-score dump. Setting (pre-stated for 2019 on 2026-10-14 before any 2019 output): equal blend 0.5·z(GNN) + 0.5·z(table),
30% quality weight, steering + 0.3·new − 0.4·busy. Seed pairings as in step12c_frontiers.py. Aggregates only.
Usage: python scripts/analysis/step12/step12c_point_lenses.py --dump <npz> --out <json> [--pairs default|test]"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, json, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); ap.add_argument("--out", required=True); args = ap.parse_args()
root = str(workspace()); os.chdir(root)
from referral_gnn.candidates import Dump, zs, STEERED_BLEND, pcp_bootstrap_index  # noqa: E402
DM = Dump(args.dump); Z = DM.Z
off, n_ref, tpos, refm, pcp = DM.off, DM.n_ref, DM.tpos, DM.refm, DM.pcp
PAIRS = [(P["G"], P["T"]) for P in DM.pairings("GT")]
busy, new, ins, cat, mips, none, ma, dist = (Z[k].astype(float) for k in ("busy", "new", "ins", "cat", "mips", "mips_none", "mips_asof", "dist"))
none = none.astype(bool); dist = np.where(dist < 0, np.nan, dist)
A_NEW, B_BUSY = STEERED_BLEND["a_new"], STEERED_BLEND["b_busy"]
def lens(j, i):
    nw = new[j] > 0
    return {"fit": float((tpos[i] + off[i]) in set(j.tolist())), "busy1": busy[j[0]], "busy5": busy[j].mean(), "new1": new[j[0]], "new5": new[j].mean(),
            "gain5": mips[j].mean() - refm[i], "good_new5": float((nw & (mips[j] > refm[i]) & ~none[j] & (busy[j] == 0)).mean()),
            "insurer_seen5": ins[j].mean(), "new_and_insurer5": float((nw & (ins[j] > 0)).mean()), "new_insurer_rate": float(ins[j][nw].mean()) if nw.any() else np.nan,
            "treated_category5": cat[j].mean(), "dist_mean5": float(np.nanmean(dist[j])) if np.isfinite(dist[j]).any() else np.nan, "dist_top1": dist[j[0]]}
codes, NC, IDX = pcp_bootstrap_index(pcp, 20261014); CNT = np.bincount(codes, minlength=NC)
def ci(d):
    d = np.nan_to_num(np.asarray(d, float)); s = np.bincount(codes, weights=d, minlength=NC); b = s[IDX].sum(1) / CNT[IDX].sum(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
out = {"setting": {"blend": "0.5·z(GNN) + 0.5·z(table)", "quality_weight": 0.3, "a_new": A_NEW, "b_busy": B_BUSY}, "n": int(n_ref), "per_pair": []}
for gk, tk in PAIRS:
    G, T = Z[gk].astype(float), Z[tk].astype(float); rows = {"GNN": [], "Table": [], "Steered blend": []}
    for i in range(n_ref):
        a, b = off[i], off[i + 1]; m = zs(ma[a:b])
        sg, st = 0.7 * zs(G[a:b]) + 0.3 * m, 0.7 * zs(T[a:b]) + 0.3 * m
        sb = 0.7 * zs(0.5 * zs(G[a:b]) + 0.5 * zs(T[a:b])) + 0.3 * m + A_NEW * new[a:b] - B_BUSY * busy[a:b]
        for lab, s in (("GNN", sg), ("Table", st), ("Steered blend", sb)): rows[lab].append(lens(a + np.argsort(-s, kind="stable")[:5], i))
    R = {k: pd.DataFrame(v) for k, v in rows.items()}
    res = {k: {c: float(np.nanmean(d[c])) for c in d.columns} | {"dist_median_top1": float(np.nanmedian(d.dist_top1))} for k, d in R.items()}
    res["delta_blend_minus_GNN_ci"] = {c: ci(R["Steered blend"][c] - R["GNN"][c]) for c in ("fit", "busy1", "busy5", "new5", "new1", "gain5", "good_new5", "dist_mean5")}
    res["delta_blend_minus_table_ci"] = {c: ci(R["Steered blend"][c] - R["Table"][c]) for c in ("fit", "busy1", "busy5", "new5", "new1", "gain5", "good_new5", "dist_mean5")}
    out["per_pair"].append({"pair": [gk[7:], tk[7:]], **res})
os.makedirs(os.path.dirname(args.out), exist_ok=True)
if os.path.exists(args.out): raise SystemExit(f"REFUSED: {args.out} exists")
json.dump(out, open(args.out, "w"), indent=1, default=float)
cols = ["fit", "busy1", "busy5", "new1", "new5", "gain5", "good_new5", "insurer_seen5", "new_insurer_rate", "treated_category5", "dist_mean5", "dist_median_top1"]
for p in out["per_pair"]:
    print("\n===", p["pair"]); print(pd.DataFrame({k: p[k] for k in ("GNN", "Table", "Steered blend")}).T[cols].round(3).to_string())
    print("  blend − GNN CI:", {k: [round(x, 3) for x in v] for k, v in p["delta_blend_minus_GNN_ci"].items()})
