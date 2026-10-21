"""STEP12d judging (STEP12D_HEAVY_PLAN.md): do the heavier hybrids reach a better (load, newness) trade-off at equal fit than the
steered equal blend of the GNN and the table (STEP12c)? No pass/fail.

Input: a step10_evaluate --dump-scores file (validation) containing the GNN runs (s42, s1, s42_dup), the table runs (s42, s1, s7) and
the STEP12d runs (dual encoder, table-offset GNN, stacked LightGBM; seeds 42 / 1 / 7). Pairing k uses (GNN_k, table_k, hybrid seed k).
Every family is steered by the same knowable attributes on the same grid (a·new − b·busy, a, b ∈ GRID) at the 30% quality weight:
  B  steered equal blend 0.5·z(GNN) + 0.5·z(table)           D  dual encoder         O  table-offset GNN         S  stacked LightGBM
Per family and pairing: Pareto set in (busy1 ↓, new5 ↑) among points with fit ≥ F (F ∈ {0.445, 0.450}); the lowest busy1 reachable
at new5 targets; the unsteered lenses. Aggregates only → runs/analysis/step12d/frontier.json (+ grid caches)."""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json, os, sys
import numpy as np, pandas as pd
root = str(workspace()); os.chdir(root)
DUMP = sys.argv[1] if len(sys.argv) > 1 else "runs/analysis/step12d/dump/validation_candidate_scores_PRIVATE.npz"
from referral_gnn.candidates import Dump, STEERING_GRID as GRID, pareto as pareto_points  # noqa: E402
DM = Dump(DUMP); Z = DM.Z
off, n_ref, tpos, refm, seg = DM.off, DM.n_ref, DM.tpos, DM.refm, DM.seg
PAIRS = DM.pairings("GTDOS")
busy, new, mips, none, ma, dist = (Z[k].astype(float) for k in ("busy", "new", "mips", "mips_none", "mips_asof", "dist")); none = none.astype(bool); dist = np.where(dist < 0, np.nan, dist)
COLS = ["busy1", "busy5", "new1", "new5", "gain5", "fit", "good_new5", "dist5"]
def lens(j, i):
    return (busy[j[0]], busy[j].mean(), new[j[0]], new[j].mean(), mips[j].mean() - refm[i], float((tpos[i] + off[i]) in set(j.tolist())),
            float(((new[j] > 0) & (mips[j] > refm[i]) & ~none[j] & (busy[j] == 0)).mean()), float(np.nanmean(dist[j])) if np.isfinite(dist[j]).any() else np.nan)
def run(P):
    base = {}
    for f, k in P.items():
        if f in "GT": continue
        base[f] = DM.weighted(DM.scores(k))
    base["B"] = DM.blend(DM.scores(P["G"]), DM.scores(P["T"]))
    rows = []
    for A_ in GRID:
        for B_ in GRID:
            acc = {f: [] for f in base}
            for i in range(n_ref):
                for f in base: acc[f].append(lens(DM.top(base[f], i, A_, B_), i))
            for f in base: rows.append({"family": f, "a_new": A_, "b_busy": B_, **dict(zip(COLS, np.nanmean(acc[f], 0).tolist()))})
    return pd.DataFrame(rows)
def pareto(d):
    return d if len(d) == 0 else pd.DataFrame(pareto_points(d))
os.makedirs("runs/analysis/step12d", exist_ok=True); out = {"dump": DUMP, "grid": GRID, "pairs": PAIRS, "per_pair": []}
for k, P in enumerate(PAIRS):
    cache = f"runs/analysis/step12d/frontier_points_pair{k}.csv"
    D = pd.read_csv(cache) if os.path.exists(cache) else run(P)
    if not os.path.exists(cache): D.to_csv(cache, index=False)
    res = {"unsteered": D[(D.a_new == 0) & (D.b_busy == 0)].set_index("family")[COLS].round(4).to_dict("index"), "floors": {}}
    for F in (0.445, 0.450):
        fr = {f: pareto(D[(D.family == f) & (D.fit >= F)]) for f in "BDOS"}
        res["floors"][str(F)] = {"n_points": {f: int(((D.family == f) & (D.fit >= F)).sum()) for f in "BDOS"},
                                 "lowest_busy1_at_new5": {str(t): {f: (float(fr[f][fr[f].new5 >= t].busy1.min()) if len(fr[f]) and (fr[f].new5 >= t).any() else None) for f in "BDOS"} for t in (0.25, 0.30, 0.35, 0.40)},
                                 "pareto": {f: fr[f].round(4).to_dict("records") for f in "BDOS"}}
    out["per_pair"].append(res)
json.dump(out, open("runs/analysis/step12d/frontier.json", "w"), indent=1, default=float)
for k, r in enumerate(out["per_pair"]):
    print(f"\n=== pairing {k}: unsteered (B = steered-blend family at a = b = 0, i.e. plain equal blend)")
    print(pd.DataFrame(r["unsteered"]).T[["fit", "busy1", "busy5", "new5", "gain5", "dist5"]].round(3).to_string())
    for F, v in r["floors"].items():
        print(f"  fit ≥ {F}: lowest top-pick busiest share at new-to-PCP ≥ target   (B blend, D dual, O offset, S stacked)   points: {v['n_points']}")
        for t, d in v["lowest_busy1_at_new5"].items(): print(f"    new5 ≥ {t}: " + "  ".join(f"{f} {('%.3f' % x) if x is not None else '  —  '}" for f, x in d.items()))
