"""STEP12e (STEP12E_TOPK_PLAN.md): the steered equal blend and the hybrids at list length K (10, 15), from a step10_evaluate
--dump-scores file made with --list-size K. Aggregates only.
  fixed     lenses at the fixed STEP12c setting for every arm, per seed pairing, with paired PCP-cluster CIs vs the GNN;
  frontier  (validation only) the same steering grid per family; Pareto sets and lowest top-pick load at fit ≥ F_K and
            newness ≥ the unsteered table's, F_K = min(unsteered GNN, table fit at K) − 0.5 pt.
Usage: python3 scripts/analysis/step12/step12e_topk.py --dump <npz> --k 10 --out <json> [--test]"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, json, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); ap.add_argument("--k", type=int, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--test", action="store_true", help="2019: GNN / table / blends only, no frontier")
args = ap.parse_args(); K = args.k
root = str(workspace()); os.chdir(root)
if os.path.exists(args.out): raise SystemExit(f"REFUSED: {args.out} exists")
from referral_gnn.candidates import Dump, STEERING_GRID, STEERED_BLEND, pareto as pareto_points, pcp_bootstrap_index  # noqa: E402
DM = Dump(args.dump); Z = DM.Z
off, n_ref, tpos, refm, pcp = DM.off, DM.n_ref, DM.tpos, DM.refm, DM.pcp
PAIRS = DM.pairings("GT" if args.test else "GTDO")
busy, new, mips, none, ma, dist = (Z[k].astype(float) for k in ("busy", "new", "mips", "mips_none", "mips_asof", "dist"))
none = none.astype(bool); dist = np.where(dist < 0, np.nan, dist)
seg = DM.seg; A_NEW, B_BUSY = STEERED_BLEND["a_new"], STEERED_BLEND["b_busy"]; GRID = STEERING_GRID
COLS = ["fit", "busy1", "busyK", "newK", "gainK", "good_newK", "distK"]
def lens(j, i):
    return (float((tpos[i] + off[i]) in set(j.tolist())), busy[j[0]], busy[j].mean(), new[j].mean(), mips[j].mean() - refm[i],
            float(((new[j] > 0) & (mips[j] > refm[i]) & ~none[j] & (busy[j] == 0)).mean()), float(np.nanmean(dist[j])) if np.isfinite(dist[j]).any() else np.nan)
def bases(P):
    """Per family, per referral: the 70/30 model–MIPS base score (B = equal blend of the GNN and table z-scores)."""
    out = {f: DM.weighted(DM.scores(k)) for f, k in P.items()}
    out["B"] = DM.blend(DM.scores(P["G"]), DM.scores(P["T"]))
    return out
def rank(base, i, a_, b_):
    a, b = seg[i]; return a + np.argsort(-(base[i] + a_ * new[a:b] - b_ * busy[a:b]), kind="stable")[:K]
codes, NC, IDX = pcp_bootstrap_index(pcp, 20261019); CNT = np.bincount(codes, minlength=NC)
def ci(d):
    d = np.asarray(d, float); ok = np.isfinite(d); s = np.bincount(codes[ok], weights=d[ok], minlength=NC); c = np.bincount(codes[ok], minlength=NC)
    den = c[IDX].sum(1); b = s[IDX].sum(1)[den > 0] / den[den > 0]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
ARMS = [("GNN", "G", 0, 0), ("Table", "T", 0, 0), ("Plain blend", "B", 0, 0), ("Steered blend", "B", A_NEW, B_BUSY),
        ("Dual encoder", "D", 0, 0), ("Dual encoder, steered", "D", A_NEW, B_BUSY), ("Table-offset GNN", "O", 0, 0), ("Table-offset GNN, steered", "O", A_NEW, B_BUSY)]
out = {"step": "STEP12e", "k": K, "split": "test" if args.test else "validation", "n": int(n_ref), "dump": args.dump,
       "setting": {"blend": "0.5·z(GNN) + 0.5·z(table)", "quality_weight": 0.3, "a_new": A_NEW, "b_busy": B_BUSY},
       "list_length_below_K": float(np.mean(np.diff(off) < K)), "per_pair": []}
for p, P in enumerate(PAIRS):
    base = bases(P); arms = [a for a in ARMS if a[1] in base]
    R = {lab: pd.DataFrame([lens(rank(base[f], i, a_, b_), i) for i in range(n_ref)], columns=COLS) for lab, f, a_, b_ in arms}
    res = {"pair": {f: k[7:] for f, k in P.items()}, "fixed": {lab: R[lab].mean().to_dict() for lab in R},
           "ci_vs_GNN": {lab: {c: ci(R[lab][c] - R["GNN"][c]) for c in COLS} for lab in R if lab != "GNN"}}
    if not args.test:
        rows = []
        for f in base:
            for a_ in GRID:
                for b_ in GRID:
                    m = np.nanmean([lens(rank(base[f], i, a_, b_), i) for i in range(n_ref)], 0)
                    rows.append({"family": f, "a_new": a_, "b_busy": b_, **dict(zip(COLS, m.tolist()))})
        Fd = pd.DataFrame(rows); Fd.to_csv(args.out.replace(".json", f"_frontier_pair{p}.csv"), index=False)
        u = Fd[(Fd.a_new == 0) & (Fd.b_busy == 0)].set_index("family")
        floor = float(min(u.loc["G", "fit"], u.loc["T", "fit"]) - 0.005); target = float(u.loc["T", "newK"])
        def pareto(d): return [r.to_dict() for r in pareto_points(d, new="newK")]
        res["frontier"] = {"fit_floor": floor, "new_target_table_unsteered": target,
                           "n_points_above_floor": {f: int(((Fd.family == f) & (Fd.fit >= floor)).sum()) for f in base},
                           "lowest_busy1_at_target": {f: (float(Fd[(Fd.family == f) & (Fd.fit >= floor) & (Fd.newK >= target)].busy1.min())
                                                          if ((Fd.family == f) & (Fd.fit >= floor) & (Fd.newK >= target)).any() else None) for f in base},
                           "pareto": {f: pareto(Fd[(Fd.family == f) & (Fd.fit >= floor)]) for f in base}}
    out["per_pair"].append(res)
    print(f"pair {p}: " + "  ".join(f"{lab}: fit {v['fit']:.3f} busy1 {v['busy1']:.3f} busyK {v['busyK']:.3f} new {v['newK']:.3f}" for lab, v in res["fixed"].items() if lab in ("GNN", "Table", "Steered blend")), flush=True)
    if "frontier" in res: print("   frontier:", res["frontier"]["fit_floor"], res["frontier"]["new_target_table_unsteered"], res["frontier"]["lowest_busy1_at_target"], flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(out, open(args.out, "w"), indent=1, default=float); print("saved", args.out)
