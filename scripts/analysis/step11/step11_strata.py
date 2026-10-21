"""STEP11 §11.2 splits on the four lenses (quality, fit, travel, load) + new connections, 2018 STEP10 primary cohort.
Reads runs/analysis/step11_2/validation_per_referral_PRIVATE.csv.gz (step10_evaluate --strata). Seed-averaged arms.
Per split and stratum: practice, PCP habit, GNN, GNN without message passing, tabular LightGBM, at base and with the 30% quality
weight; GNN − table contrasts for MIPS gain and busiest share with PCP-cluster 95% CIs. Candidate-side: share of recommended
top-5 slots going to early-tenure (≤ 6 Medicare years), low-experience (< 10 prior referrals received) and non-metro specialists,
vs actual referrals and vs availability. Cells with < 100 patients are flagged. Aggregates only.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os

import numpy as np
import pandas as pd

root = str(workspace()); os.chdir(root)
D = "runs/analysis/step11_2"; pr = pd.read_csv(f"{D}/validation_per_referral_PRIVATE.csv.gz")
ARMS = {"GNN": ["_v5_unmasked_s42", "_v5_unmasked_s1"], "GNN, no message passing": ["_v5_nograph_s42", "_v5_nograph_s1"], "Table (LightGBM)": ["_lgbm_tabular_s42", "_lgbm_tabular_s1"]}
def col(sfx, metric, var=""):
    c = [c for c in pr.columns if c.startswith("model:") and c.endswith(f"{var}|{metric}") and ("+" in c) == bool(var) and c.split(":", 1)[1].split("+")[0].split("|")[0].endswith(sfx)]
    assert len(c) == 1, (sfx, metric, var, c); return pr[c[0]].to_numpy(float)
def arm(name, metric, var=""): return np.mean([col(s, metric, var) for s in ARMS[name]], axis=0)
LENSES = {"fit": "r5", "quality": "gain5", "travel": "dist5m", "load": "busy1", "new": "new_pcp5"}
series = {}
for v, var in (("base", ""), ("q30", "+blend0.3")):
    for a in ARMS: series[(a, v)] = {k: arm(a, m, var) for k, m in LENSES.items()}
    series[("PCP habit", v)] = {k: pr[f"pcp{var}|{m}"].to_numpy(float) for k, m in LENSES.items()}
series[("Actual referral", "base")] = {"fit": np.ones(len(pr)), "quality": np.zeros(len(pr)), "travel": pr.ref_dist.to_numpy(float),
                                       "load": pr.ref_busy.astype(float).to_numpy(), "new": pr.ref_new_pcp.astype(float).to_numpy()}

def bands(x, edges, labels): return pd.cut(x, edges, labels=labels, right=True)
S = {
    "Patient age": bands(pr.patient_age, [-1, 64, 74, 84, 200], ["<65", "65–74", "75–84", "85+"]),
    "Patient local access (distance to 5th-nearest eligible specialist)": pd.qcut(pr.access5, 3, labels=["near", "middle", "far"]),
    "PCP Medicare tenure": bands(pr.pcp_tenure, [-1, 6, 13, 99], ["early (≤6 y)", "mid (7–13 y)", "late (≥14 y)"]),
    "PCP network so far (distinct specialists used)": pd.qcut(pr.pcp_net, 3, labels=["thin", "medium", "wide"], duplicates="drop"),
    "Referred specialist Medicare tenure": bands(pr.ref_tenure, [-1, 6, 13, 99], ["early (≤6 y)", "mid (7–13 y)", "late (≥14 y)"]),
    "Referred specialist experience (referrals received before)": bands(pr.ref_prior_refs, [-1, 0, 9, 49, 10 ** 9], ["0", "1–9", "10–49", "50+"]),
    "Referred specialist location": pr.ref_nonmetro.map({True: "non-metro", False: "metro"}),
}
codes, uniq = pd.factorize(pr["pcp"]); rng = np.random.default_rng(20261009); B = 1000; idx = rng.integers(0, len(uniq), size=(B, len(uniq)))
def ci(x, m):
    w = m.astype(float); s1 = np.bincount(codes, weights=np.where(m, x, 0.0), minlength=len(uniq)); s0 = np.bincount(codes, weights=w, minlength=len(uniq))
    b = s1[idx].sum(1) / np.maximum(s0[idx].sum(1), 1e-9); return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]

out = {"cohort": "2018 first-time referrals, STEP10 primary cohort", "n": int(len(pr)), "splits": {}}
for sname, lab in S.items():
    rows = []
    for g in [x for x in (lab.cat.categories if hasattr(lab, "cat") else sorted(lab.dropna().unique()))]:
        m = (lab == g).to_numpy(); n_pat = int(pr.patient[m].nunique()); n_pcp = int(pr.pcp[m].nunique())
        rec = {"stratum": str(g), "referrals": int(m.sum()), "patients": n_pat, "pcps": n_pcp, "small_cell": n_pat < 100}
        for (a, v), sr in series.items():
            for k, x in sr.items():
                rec[f"{a}|{v}|{k}"] = float(np.nanmedian(x[m])) if k == "travel" else float(np.nanmean(x[m]))
        for v in ("base", "q30"):
            for k in ("quality", "load", "fit"):
                d = series[("GNN", v)][k] - series[("Table (LightGBM)", v)][k]
                rec[f"GNN−table|{v}|{k}"] = float(np.nanmean(d[m])); rec[f"GNN−table|{v}|{k}|ci"] = ci(np.nan_to_num(d), m)
        rec["available|early"] = float(pr.elig_early[m].mean()); rec["available|lowexp"] = float(pr.elig_lowexp[m].mean()); rec["available|nonmetro"] = float(pr.elig_nonmetro[m].mean())
        rows.append(rec)
    out["splits"][sname] = rows

# candidate-side exposure (all referrals)
exp = []
HAS_EXP = any(c.endswith("|early5") for c in pr.columns)
for k_, ref in ((("early", (pr.ref_tenure <= 6)), ("lowexp", (pr.ref_prior_refs < 10)), ("nonmetro", pr.ref_nonmetro.astype(bool))) if HAS_EXP else ()):
    r = {"group": {"early": "Early-tenure specialists (≤6 Medicare years)", "lowexp": "Low-experience specialists (<10 referrals received)", "nonmetro": "Non-metro specialists"}[k_],
         "available_share": float(pr[f"elig_{k_}"].mean()), "actual_referrals": float(ref.mean())}
    for a in ARMS:
        r[f"{a}|base|top5"] = float(arm(a, f"{k_}5").mean()); r[f"{a}|q30|top5"] = float(arm(a, f"{k_}5", "+blend0.3").mean())
        r[f"{a}|base|top1"] = float(arm(a, f"{k_}1").mean())
    r["PCP habit|base|top5"] = float(pr[f"pcp|{k_}5"].mean()); exp.append(r)
out["exposure"] = exp
json.dump(out, open(f"{D}/strata_summary.json", "w"), indent=1, default=float)

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
for sname, rows in out["splits"].items():
    t = pd.DataFrame(rows); print(f"\n=== {sname}")
    show = t[["stratum", "referrals", "patients", "pcps", "small_cell",
              "Actual referral|base|load", "GNN|q30|load", "Table (LightGBM)|q30|load", "GNN|q30|quality", "Table (LightGBM)|q30|quality",
              "GNN|q30|fit", "Table (LightGBM)|q30|fit", "GNN|q30|travel", "Table (LightGBM)|q30|travel", "GNN|q30|new", "Table (LightGBM)|q30|new"]].copy()
    show.columns = ["stratum", "n", "pat", "pcp", "small", "load:actual", "load:GNN", "load:table", "MIPS:GNN", "MIPS:table", "fit:GNN", "fit:table", "mi:GNN", "mi:table", "new:GNN", "new:table"]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("  GNN − table (30% weight):  " + " | ".join(f"{r['stratum']}: load {r['GNN−table|q30|load']:+.3f} [{r['GNN−table|q30|load|ci'][0]:+.3f},{r['GNN−table|q30|load|ci'][1]:+.3f}], MIPS {r['GNN−table|q30|quality']:+.2f} [{r['GNN−table|q30|quality|ci'][0]:+.2f},{r['GNN−table|q30|quality|ci'][1]:+.2f}]" for r in rows))
print("\n=== Candidate side: share of recommended top-5 slots")
print(pd.DataFrame(exp).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
