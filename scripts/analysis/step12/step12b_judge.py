"""STEP12b judging (STEP12B_HYBRID_PLAN.md, rule frozen before the run). Paired hybrid − GNN contrasts on per-referral series
(seed-averaged; hybrids average their two seed pairings), PCP-cluster bootstrap. Setting: 30% quality weight (blend0.3) decides;
base is descriptive.

"Combines the strengths" (all must hold, vs the GNN):
  load      upper 95% CI of Δbusy1 ≤ +1.0 pt AND upper 95% CI of Δbusy5 ≤ +1.0 pt     (non-inferior on top pick and top-5 slots)
  newness   Δnew_pcp5 ≥ +5.0 pts with lower 95% CI > 0 AND lower 95% CI of Δnew_pcp1 > 0  (more new picks, also at the top pick)
  feasible  Δgood_new5 ≥ 0 (point)                                                        (new picks not lower-MIPS / busier junk)
  quality   lower 95% CI of Δgain5 ≥ −0.5 MIPS points
  fit       r5(hybrid) ≥ min(r5 GNN, r5 table) − 1.0 pt (point)
Travel (median and mean), insurer-feasible new picks (new_ins5) and the table reference are reported, not judged.
Tie-break if several arms pass: largest Δnew_pcp5, then lowest busy1. Split-k arms are reported under their own label
("top pick = GNN, slots after = table") and judged by the same rule (they cannot pass by position alone: busy5 and new_pcp1 count).
Usage: python scripts/analysis/step12/step12b_judge.py --eval runs/analysis/step12b/eval
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--eval", required=True); ap.add_argument("--bootstrap", type=int, default=2000)
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
cf = pd.read_csv(os.path.join(args.eval, "validation_configs.csv")); pr = pd.read_csv(os.path.join(args.eval, "validation_per_referral_PRIVATE.csv.gz"))
codes, uniq = pd.factorize(pr.pcp); NC = len(uniq); IDX = np.random.default_rng(20261014).integers(0, NC, size=(args.bootstrap, NC))
CNT = np.bincount(codes, minlength=NC).astype(float); DEN = CNT[IDX].sum(1)
M = ["r5", "gain5", "busy1", "busy5", "new_pcp5", "new_pcp1", "good_new5", "new_ins5", "dist5m"]
ser = lambda fam, met, v="blend0.3": np.mean([pr[f"{c}|{met}"].to_numpy(float) for c in cf[(cf.family == fam) & (cf.variant == v)].config], axis=0)
def mean(x): return float(np.nanmean(x))
def boot(d):
    s1 = np.bincount(codes, weights=np.nan_to_num(d), minlength=NC); b = s1[IDX].sum(1) / DEN
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
REF = {"GNN": "model_unmasked", "Table": "lgbm_tabular"}
cand = sorted(f for f in cf.family.unique() if (f.startswith("hyb_") or f.startswith("s12b_") or f.startswith("s12_lgbm_flat")) and "@" not in f)
out = {"setting": "30% quality weight; paired vs GNN; PCP-cluster bootstrap", "references": {}, "arms": {}}
for k, f in REF.items():
    out["references"][k] = {m: mean(ser(f, m)) for m in M} | {"median_travel_mi": float(np.nanmedian(ser(f, "dist5m")))}
g = {m: ser(REF["GNN"], m) for m in M}; fit_floor = min(out["references"]["GNN"]["r5"], out["references"]["Table"]["r5"]) - 0.01
for f in cand:
    h = {m: ser(f, m) for m in M}; d = {m: h[m] - g[m] for m in M}; ci = {m: boot(d[m]) for m in ("busy1", "busy5", "new_pcp5", "new_pcp1", "gain5", "good_new5", "new_ins5", "r5")}
    crit = {"load_top1_noninferior": ci["busy1"][1] <= 0.01, "load_top5_noninferior": ci["busy5"][1] <= 0.01,
            "new5_gain_≥5pts_CI>0": mean(d["new_pcp5"]) >= 0.05 and ci["new_pcp5"][0] > 0, "new1_CI>0": ci["new_pcp1"][0] > 0,
            "good_new5_≥_GNN": mean(d["good_new5"]) >= 0, "quality_noninferior": ci["gain5"][0] >= -0.5, "fit_≥_floor": mean(h["r5"]) >= fit_floor}
    out["arms"][f] = {"label": "top pick = GNN, slots after = table" if f.startswith("hyb_split") else None, "lenses": {m: mean(h[m]) for m in M} | {"median_travel_mi": float(np.nanmedian(h["dist5m"]))},
                      "delta_vs_GNN": {m: mean(d[m]) for m in M}, "ci_delta_vs_GNN": ci, "criteria": crit, "combines_strengths": bool(all(crit.values())),
                      "base_lenses": {m: mean(ser(f, m, "base")) for m in M}}
passing = [f for f in cand if out["arms"][f]["combines_strengths"]]
out["passing"] = sorted(passing, key=lambda f: (-out["arms"][f]["delta_vs_GNN"]["new_pcp5"], out["arms"][f]["lenses"]["busy1"]))
out["selected_for_2019"] = out["passing"][0] if passing else None
os.makedirs("runs/analysis/step12b", exist_ok=True); fn = "runs/analysis/step12b/judge_summary.json"
if os.path.exists(fn): fn = fn.replace(".json", pd.Timestamp.now().strftime("_%Y%m%dT%H%M%S.json"))
json.dump(out, open(fn, "w"), indent=1); print("wrote", fn)
pd.set_option("display.width", 250)
rows = [{"arm": "GNN", **{m: out["references"]["GNN"][m] for m in ("r5", "gain5", "busy1", "busy5", "new_pcp5", "new_pcp1", "good_new5", "new_ins5")}},
        {"arm": "Table", **{m: out["references"]["Table"][m] for m in ("r5", "gain5", "busy1", "busy5", "new_pcp5", "new_pcp1", "good_new5", "new_ins5")}}]
rows += [{"arm": f, **{m: out["arms"][f]["lenses"][m] for m in ("r5", "gain5", "busy1", "busy5", "new_pcp5", "new_pcp1", "good_new5", "new_ins5")}, "pass": out["arms"][f]["combines_strengths"]} for f in cand]
print(pd.DataFrame(rows).round(3).to_string(index=False)); print("passing:", out["passing"], "| selected for 2019:", out["selected_for_2019"])
