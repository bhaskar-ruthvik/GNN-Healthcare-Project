"""PRELIMINARY data-size comparison (STEP11 §11.4, user request 2026-10-08): models trained on 2015–2016 vs on 2015–2017,
all scored on the same 2018 validation referrals (STEP10 primary cohort, base configuration). Reads per-referral results
by run-name suffix, so arm grouping in the evaluator's tables is not used. Descriptive; PCP-cluster bootstrap CIs.
    python scripts/analysis/step11/step11_prelim_datasize.py --dir runs/analysis/step11_4_prelim
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="runs/analysis/step11_4_prelim"); ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args(); root = str(workspace()); os.chdir(root)
pr = pd.read_csv(os.path.join(args.dir, "validation_per_referral_PRIVATE.csv.gz"))

def col(suffix, metric="r5"):
    c = [c for c in pr.columns if c.startswith("model:") and "+" not in c and c.endswith("|" + metric) and c.split(":", 1)[1].split("|")[0].endswith(suffix)]
    assert len(c) == 1, (suffix, c); return pr[c[0]].to_numpy(float)

ARMS = {  # period -> model -> run-name suffixes (seed-averaged where two)
    "2015-2016": {"GNN": ["_v5_unmasked_upto2017_s42"], "GNN, no message passing": ["_v5_nograph_upto2017_s42"],
                  "LightGBM pure tabular": ["_lgbm_tabular_upto2017_s42", "_lgbm_tabular_upto2017_s1"],
                  "LightGBM + network counts": ["_lgbm_graph_upto2017_s42", "_lgbm_graph_upto2017_s1"]},
    "2015-2017": {"GNN": ["_v5_unmasked_s42"], "GNN, no message passing": ["_v5_nograph_s42"],
                  "LightGBM pure tabular": ["_lgbm_tabular_s42", "_lgbm_tabular_s1"],
                  "LightGBM + network counts": ["_lgbm_graph_s42", "_lgbm_graph_s1"]}}
def arm(period, model, metric="r5"): return np.mean([col(s, metric) for s in ARMS[period][model]], axis=0)

rng = np.random.default_rng(20261008); codes, uniq = pd.factorize(pr["pcp"]); cnt = np.bincount(codes)
idx = rng.integers(0, len(uniq), size=(args.boot, len(uniq)))
def ci(x):
    s = np.bincount(codes, weights=x); b = s[idx].sum(1) / cnt[idx].sum(1); return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]

rows = []
for period in ARMS:
    for model in ARMS[period]:
        rows.append({"trained_on": period, "model": model, "R@5": arm(period, model).mean(), "R@5_ci": ci(arm(period, model)),
                     "MIPS_gain_top5": arm(period, model, "gain5").mean(), "busiest_top1": arm(period, model, "busy1").mean(),
                     "new_to_PCP_top5": arm(period, model, "new_pcp5").mean()})
tab = pd.DataFrame(rows)
gaps = []
for a, b in (("GNN", "LightGBM pure tabular"), ("LightGBM + network counts", "LightGBM pure tabular"), ("GNN", "GNN, no message passing"),
             ("GNN", "LightGBM + network counts")):
    d = {p: arm(p, a) - arm(p, b) for p in ARMS}
    gaps.append({"gap": f"{a} − {b}", "trained_2015_2016": d["2015-2016"].mean(), "ci_2015_2016": ci(d["2015-2016"]),
                 "trained_2015_2017": d["2015-2017"].mean(), "ci_2015_2017": ci(d["2015-2017"]),
                 "change_with_more_data": (d["2015-2017"] - d["2015-2016"]).mean(), "ci_change": ci(d["2015-2017"] - d["2015-2016"])})
gaps = pd.DataFrame(gaps)
json.dump({"preliminary": True, "n_referrals": int(len(pr)), "models": tab.to_dict("records"), "gaps": gaps.to_dict("records")},
          open(os.path.join(args.dir, "datasize_summary.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 250); f = lambda x: f"{x:.3f}" if isinstance(x, float) else (f"[{x[0]:+.3f}, {x[1]:+.3f}]" if isinstance(x, list) else x)
print(tab.map(f).to_string(index=False)); print(); print(gaps.map(f).to_string(index=False))
