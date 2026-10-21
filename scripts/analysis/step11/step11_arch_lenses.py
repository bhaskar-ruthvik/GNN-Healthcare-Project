"""STEP11 B5 revision 2: judge architecture changes on the four lenses (2018 STEP10 primary cohort), against the current GNN;
tabular LightGBM shown as the reference. Seed-averaged arms; PCP-cluster bootstrap; Holm-type adjustment across the arms
(alpha / 3 per arm-metric family). Noise band per lens = |GNN s42 − GNN s42 duplicate| on that lens.

Success (frozen): improves quality (MIPS gain at λ = 0.3, higher) OR load (busiest-10% top-1 share, lower) beyond the noise band
(adjusted CI excludes 0 and |point| > band), without losing more than the band on the other of the two, without losing more than
0.02 on fit (top-5 match), and without travel (per-referral mean distance of the top 5) worse beyond its band.
MRR is not used.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os

import numpy as np
import pandas as pd

root = str(workspace()); os.chdir(root)
D = "runs/analysis/step11_5"; pr = pd.read_csv(f"{D}/validation_per_referral_PRIVATE.csv.gz")
ARMS = {"GNN (current)": ["_v5_unmasked_s42", "_v5_unmasked_s1"], "C2: skip + interaction": ["_c2_skip_s42", "_c2_skip_s1"],
        "C1: history dropout 0.3": ["_c1_hd03_s42", "_c1_hd03_s1"], "C1: history dropout 0.6": ["_c1_hd06_s42", "_c1_hd06_s1"],
        "Table (LightGBM)": ["_lgbm_tabular_s42", "_lgbm_tabular_s1"]}
def col(sfx, metric, var):
    c = [c for c in pr.columns if c.startswith("model:") and c.endswith(f"{var}|{metric}") and ("+" in c) == bool(var) and c.split(":", 1)[1].split("+")[0].split("|")[0].endswith(sfx)]
    assert len(c) == 1, (sfx, metric, var, c); return pr[c[0]].to_numpy(float)
def arm(a, metric, var): return np.mean([col(s, metric, var) for s in ARMS[a]], axis=0)
L = {"quality": "gain5", "load": "busy1", "fit": "r5", "travel": "dist5m", "new": "new_pcp5"}
codes, uniq = pd.factorize(pr["pcp"]); rng = np.random.default_rng(20261009); B = 2000; idx = rng.integers(0, len(uniq), size=(B, len(uniq)))
cnt = np.bincount(codes, minlength=len(uniq)).astype(float)
def boot(x):
    x = np.nan_to_num(x); s = np.bincount(codes, weights=x, minlength=len(uniq)); return s[idx].sum(1) / cnt[idx].sum(1)
ALPHA = 0.05 / 3
out = {"cohort": "2018 STEP10 primary cohort", "n": int(len(pr)), "alpha_per_arm": ALPHA, "variants": {}}
for vname, var in (("with 30% quality weight", "+blend0.3"), ("base ranking", "")):
    band = {k: abs(float(np.nanmean(col("_v5_unmasked_s42", m, var)) - np.nanmean(col("_v5_unmasked_s42_dup", m, var)))) for k, m in L.items()}
    rows = []
    for a in ARMS:
        r = {"arm": a, **{k: float(np.nanmean(arm(a, m, var))) for k, m in L.items()}, "travel_median": float(np.nanmedian(arm(a, "dist5m", var)))}
        if a != "GNN (current)":
            for k, m in L.items():
                d = arm(a, m, var) - arm("GNN (current)", m, var); b = boot(d)
                r[f"d_{k}"] = float(np.nanmean(d)); r[f"d_{k}_ci"] = [float(np.quantile(b, ALPHA / 2)), float(np.quantile(b, 1 - ALPHA / 2))]
            if a != "Table (LightGBM)":
                q_up = r["d_quality_ci"][0] > 0 and r["d_quality"] > band["quality"]; l_dn = r["d_load_ci"][1] < 0 and -r["d_load"] > band["load"]
                q_ok = r["d_quality"] > -band["quality"]; l_ok = r["d_load"] < band["load"]
                fit_ok = r["d_fit_ci"][0] > -0.02; trav_ok = r["d_travel"] < band["travel"] or r["d_travel_ci"][0] <= 0
                r["success"] = bool((q_up and l_ok or l_dn and q_ok) and fit_ok and trav_ok)
                r["why"] = {"quality_up": q_up, "load_down": l_dn, "quality_not_worse": q_ok, "load_not_worse": l_ok, "fit_ok": fit_ok, "travel_ok": trav_ok}
        rows.append(r)
    out["variants"][vname] = {"noise_band": band, "arms": rows}
json.dump(out, open(f"{D}/arch_lenses.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
for vname, v in out["variants"].items():
    t = pd.DataFrame(v["arms"]); print(f"\n=== {vname}   noise band: " + ", ".join(f"{k} {x:.3f}" for k, x in v["noise_band"].items()))
    print(t[["arm", "fit", "quality", "travel_median", "load", "new"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for r in v["arms"]:
        if "d_quality" in r:
            print(f"  {r['arm']:<26} vs GNN: quality {r['d_quality']:+.2f} [{r['d_quality_ci'][0]:+.2f},{r['d_quality_ci'][1]:+.2f}]  load {100*r['d_load']:+.1f}pt [{100*r['d_load_ci'][0]:+.1f},{100*r['d_load_ci'][1]:+.1f}]  "
                  f"fit {100*r['d_fit']:+.1f}pt [{100*r['d_fit_ci'][0]:+.1f},{100*r['d_fit_ci'][1]:+.1f}]  travel {r['d_travel']:+.2f} mi  new {100*r['d_new']:+.1f}pt" + (f"  → {'SUCCESS' if r['success'] else 'not a success'}" if "success" in r else ""))
