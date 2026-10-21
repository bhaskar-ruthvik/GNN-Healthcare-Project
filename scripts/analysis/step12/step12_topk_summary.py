"""STEP12 top-K lenses: K = 5, 10, 15 (runs/analysis/step12_topk/K*/, 2018 STEP10 primary cohort, seed-averaged arms).
Lenses over the K-list: fit = chosen specialist in the top K; quality = mean MIPS of the K minus the chosen; travel = mean
distance of the K (per referral); load = busiest-10% share of the K slots (and of the top pick, unchanged by K except where
filling adds candidates); new = share of the K new to the PCP. GNN − table contrasts with PCP-cluster 95% CIs. Aggregates only."""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json, os
import numpy as np, pandas as pd
root = str(workspace()); os.chdir(root)
ARMS = {"GNN": ["_v5_unmasked_s42", "_v5_unmasked_s1"], "No-graph GNN": ["_v5_nograph_s42", "_v5_nograph_s1"], "Table (LightGBM)": ["_lgbm_tabular_s42", "_lgbm_tabular_s1"]}
LENS = {"fit": "r5", "quality": "gain5", "travel": "dist5m", "load_slots": "busy5", "load_top1": "busy1", "new": "new_pcp5"}
out = {}
for K in (5, 10, 15):
    pr = pd.read_csv(f"runs/analysis/step12_topk/K{K}/validation_per_referral_PRIVATE.csv.gz")
    def col(sfx, m, var):
        c = [c for c in pr.columns if c.startswith("model:") and c.endswith(f"{var}|{m}") and ("+" in c) == bool(var) and c.split(":", 1)[1].split("+")[0].split("|")[0].endswith(sfx)]
        assert len(c) == 1, (sfx, m, var, c); return pr[c[0]].to_numpy(float)
    S = {}
    for v, var in (("base", ""), ("q30", "+blend0.3")):
        for a, sfx in ARMS.items(): S[(a, v)] = {k: np.mean([col(s, m, var) for s in sfx], axis=0) for k, m in LENS.items()}
        S[("PCP habit", v)] = {k: pr[f"pcp{var}|{m}"].to_numpy(float) for k, m in LENS.items()}
    codes, uniq = pd.factorize(pr.pcp); rng = np.random.default_rng(20261012); IDX = rng.integers(0, len(uniq), size=(2000, len(uniq)))
    s0 = np.bincount(codes, minlength=len(uniq)).astype(float)
    def ci(x): s1 = np.bincount(codes, weights=np.nan_to_num(x), minlength=len(uniq)); b = s1[IDX].sum(1) / s0[IDX].sum(1); return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    res = {"n": int(len(pr)), "list_shorter_than_K": float((pr.n_elig < K).mean()), "filled_beyond_25mi": float(pr.filled.mean()),
           "practice": {"load_top1": float(pr.ref_busy.mean()), "travel_mean": float(pr.ref_dist.mean()), "new": float(pr.ref_new_pcp.mean())}, "arms": {}, "gnn_minus_table": {}}
    for (a, v), d in S.items(): res["arms"][f"{a}|{v}"] = {k: float(np.nanmean(x)) for k, x in d.items()}
    for v in ("base", "q30"):
        for k in LENS:
            dd = S[("GNN", v)][k] - S[("Table (LightGBM)", v)][k]; res["gnn_minus_table"][f"{k}|{v}"] = {"est": float(np.nanmean(dd)), "ci": ci(dd)}
    out[f"K{K}"] = res
os.makedirs("runs/analysis/step12_topk", exist_ok=True); json.dump(out, open("runs/analysis/step12_topk/topk_summary.json", "w"), indent=1)
for K, r in out.items():
    print(f"\n=== {K}: n={r['n']}, list shorter than K {r['list_shorter_than_K']:.1%}, filled beyond 25 mi {r['filled_beyond_25mi']:.1%}; practice load {r['practice']['load_top1']:.3f} travel {r['practice']['travel_mean']:.1f} new {r['practice']['new']:.3f}")
    t = pd.DataFrame(r["arms"]).T; print(t.round(3).to_string())
    print("  GNN − table: " + "; ".join(f"{k} {c['est']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]" for k, c in r["gnn_minus_table"].items()))
