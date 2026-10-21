"""STEP12c: frontier comparison — does any TWO-model approach reach a better (load, newness) trade-off at equal fit than the best
SINGLE-model approach, when every approach may be steered by the same knowable attributes?
Families (all at the 30% quality weight; steering a·new − b·busy added to the ranking score; a, b ∈ GRID):
  G      GNN alone                          T      table alone
  B      0.5·z(GNN) + 0.5·z(table)          S      per referral, the steered GNN list or the steered table list, whichever has the
                                                   larger new5 − 2·busy5 (knowable)
For fit floors F ∈ {0.445, 0.450}: per family, the Pareto set of points with fit ≥ F in (busy1 ↓, new5 ↑); for new5 targets, the
lowest busy1 each family can reach. Three seed pairings. Aggregates only → runs/analysis/step12c/frontiers.json"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json, os
import numpy as np, pandas as pd
root = str(workspace()); os.chdir(root)
from referral_gnn.candidates import Dump, STEERING_GRID as GRID, pareto as pareto_points  # noqa: E402
DM = Dump("runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz"); Z = DM.Z
off, n_ref, tpos, refm, seg = DM.off, DM.n_ref, DM.tpos, DM.refm, DM.seg
PAIRS = [(P["G"], P["T"]) for P in DM.pairings("GT")]
busy, new, mips, none, ma = DM.busy, DM.new, DM.mips, DM.none, DM.ma
def lens(j, i):
    return (busy[j[0]], busy[j].mean(), new[j[0]], new[j].mean(), mips[j].mean() - refm[i], float((tpos[i] + off[i]) in set(j.tolist())),
            float(((new[j] > 0) & (mips[j] > refm[i]) & ~none[j] & (busy[j] == 0)).mean()))
COLS = ["busy1", "busy5", "new1", "new5", "gain5", "fit", "good_new5"]
def run(pair):
    gk, tk = pair; G, T = Z[gk].astype(float), Z[tk].astype(float)
    zG, zT, zB = DM.weighted(G), DM.weighted(T), DM.blend(G, T)
    rows = []
    for A_ in GRID:
        for B_ in GRID:
            acc = {f: [] for f in "GTBS"}
            for i in range(n_ref):
                tops = {f: DM.top(base, i, A_, B_) for f, base in (("G", zG), ("T", zT), ("B", zB))}
                for f in "GTB": acc[f].append(lens(tops[f], i))
                lg, lt = acc["G"][-1], acc["T"][-1]
                acc["S"].append(lt if (lt[3] - 2 * lt[1]) > (lg[3] - 2 * lg[1]) else lg)
            for f in "GTBS": rows.append({"family": f, "a_new": A_, "b_busy": B_, **dict(zip(COLS, np.mean(acc[f], 0).tolist()))})
    return pd.DataFrame(rows)
def pareto(d):
    if len(d) == 0: return pd.DataFrame(columns=["a_new", "b_busy", "fit", "busy1", "busy5", "new5", "new1", "gain5", "good_new5"])
    return pd.DataFrame(pareto_points(d))
out = {"pairs": [[g[7:], t[7:]] for g, t in PAIRS], "grid": GRID, "per_pair": []}
for k_, p in enumerate(PAIRS):
    cache = f"runs/analysis/step12c/frontier_points_pair{k_}.csv"                 # aggregates only (grid × family means)
    if os.path.exists(cache): D = pd.read_csv(cache)
    else: D = run(p); D.to_csv(cache, index=False)
    res = {"points": D.to_dict("records"), "floors": {}}
    for F in (0.445, 0.450):
        fr = {}
        for f in "GTBS":
            P = pareto(D[(D.family == f) & (D.fit >= F)]); fr[f] = P[["a_new", "b_busy", "fit", "busy1", "busy5", "new5", "new1", "gain5", "good_new5"]].round(4).to_dict("records")
        tgt = {}
        for t in (0.25, 0.30, 0.35, 0.40):
            tgt[str(t)] = {f: (min([q["busy1"] for q in fr[f] if q["new5"] >= t], default=None)) for f in "GTBS"}
        tgt["n_points_at_floor"] = {f: int(((D.family == f) & (D.fit >= F)).sum()) for f in "GTBS"}
        res["floors"][str(F)] = {"pareto": fr, "lowest_busy1_at_new5_target": tgt}
    out["per_pair"].append(res)
json.dump(out, open("runs/analysis/step12c/frontiers.json", "w"), indent=1, default=float)
for p, r in zip(out["pairs"], out["per_pair"]):
    print(f"\n=== {p}")
    for F, v in r["floors"].items():
        print(f"  fit ≥ {F}: lowest top-pick busiest share reachable at new-to-PCP ≥ target (G = GNN, T = table, B = blend, S = two-list selection)")
        for t, d in v["lowest_busy1_at_new5_target"].items():
            if t == "n_points_at_floor": print("    grid points at this fit floor:", d); continue
            print(f"    new5 ≥ {t}: " + "  ".join(f"{f} {('%.3f' % x) if x is not None else '  —  '}" for f, x in d.items()))
