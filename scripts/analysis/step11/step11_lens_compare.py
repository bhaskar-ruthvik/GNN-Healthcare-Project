"""Graph model vs tabular LightGBM on the four lenses (+ new connections), 2018 and 2019 side by side (STEP11 B4/G1; MRR not used).
Reads runs/analysis/step11_1_tabular/{validation,test}_per_referral_PRIVATE.csv.gz. Seed-averaged arms; PCP-cluster 95% CIs.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os

import numpy as np
import pandas as pd

root = str(workspace()); os.chdir(root)
D = "runs/analysis/step11_1_tabular"
ARMS = {"GNN": ["_v5_unmasked_s42", "_v5_unmasked_s1"], "Table": ["_lgbm_tabular_s42", "_lgbm_tabular_s1"]}
L = {"quality": "gain5", "load": "busy1", "fit": "r5", "travel": "dist5m", "new": "new_pcp5"}
out = {}
for split in ("validation", "test"):
    f = f"{D}/{split}_per_referral_PRIVATE.csv.gz"
    if not os.path.exists(f): continue
    pr = pd.read_csv(f); codes, uniq = pd.factorize(pr["pcp"]); rng = np.random.default_rng(20261009); idx = rng.integers(0, len(uniq), size=(2000, len(uniq)))
    cnt = np.bincount(codes, minlength=len(uniq)).astype(float)
    def col(sfx, m, var):
        c = [c for c in pr.columns if c.startswith("model:") and c.endswith(f"{var}|{m}") and ("+" in c) == bool(var) and c.split(":", 1)[1].split("+")[0].split("|")[0].endswith(sfx)]
        assert len(c) == 1, (sfx, m, var, c); return pr[c[0]].to_numpy(float)
    arm = lambda a, m, var: np.mean([col(s, m, var) for s in ARMS[a]], axis=0)
    res = {"n": int(len(pr)), "actual_load": float(pr.ref_busy.mean()), "actual_new": float(pr.ref_new_pcp.mean())}
    for vname, var in (("q30", "+blend0.3"), ("base", "")):
        for k, m in L.items():
            g, t = arm("GNN", m, var), arm("Table", m, var); d = np.nan_to_num(g - t)
            b = np.bincount(codes, weights=d, minlength=len(uniq))[idx].sum(1) / cnt[idx].sum(1)
            res[f"{vname}|{k}"] = {"gnn": float(np.nanmean(g)), "table": float(np.nanmean(t)), "diff": float(np.mean(d)), "ci": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}
    out[split] = res
json.dump(out, open(f"{D}/lens_compare.json", "w"), indent=1)
for split, r in out.items():
    print(f"=== {split} (n {r['n']:,}; actual busiest share {100*r['actual_load']:.1f}%, actual new-to-PCP {100*r['actual_new']:.1f}%)")
    for vname in ("q30", "base"):
        print("  " + vname + ": " + " | ".join(f"{k}: GNN {r[f'{vname}|{k}']['gnn']:.3f} table {r[f'{vname}|{k}']['table']:.3f} diff {r[f'{vname}|{k}']['diff']:+.3f} [{r[f'{vname}|{k}']['ci'][0]:+.3f},{r[f'{vname}|{k}']['ci'][1]:+.3f}]" for k in L))
