"""Collect AGGREGATE numbers for the week-2 deck (presentation/data_w2.json). No referral- or provider-level values.
Sources: Step 11.1 baselines, 11.4(a) network evolution, preliminary data-size runs, snapshot run summaries.
Run from anywhere (data under the workspace, see referral_gnn/paths.py):  python scripts/deliverables/presentation/build_data_w2.py
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
try:
    import lightgbm as lgb  # feature importance of the LightGBM baselines (import before anything that loads torch)
except ImportError:
    lgb = None
import glob
import json
import os

import numpy as np
import pandas as pd

os.chdir(str(workspace())); OUTD = "deliverables/presentation"; os.makedirs(OUTD, exist_ok=True)
A = "runs/analysis"; out = {}

# ---- 11.1 baselines (2018 validation, STEP10 primary cohort) ------------------------------------------------------
trig = json.load(open(f"{A}/step11_1_tabular/validation_step11_triggers.json"))   # user 2026-10-08: tabular baselines only, no hybrids
NAMES = {"gnn": "Graph model (GNN)", "nograph": "GNN, no message passing", "lgbm_tabular": "LightGBM: claim attributes only",
         "lgbm_primary": "LightGBM + history columns", "lgbm_graph": "LightGBM + network counts", "logreg": "Logistic regression",
         "itemknn": "Similar-specialist CF", "als": "Matrix factorisation", "logreg_tabular": "Logistic regression: claim attributes only", "pcp_habit": "PCP habit", "random": "Random", "mips_sorted": "Highest MIPS first"}
KIND = {"gnn": "graph", "nograph": "graph", "lgbm_tabular": "table", "lgbm_primary": "table", "lgbm_graph": "table", "logreg": "table",
        "itemknn": "cf", "als": "cf", "logreg_tabular": "table", "pcp_habit": "heur", "random": "heur", "mips_sorted": "heur"}
rows = []
for r in trig["arms"]:
    rows.append({"key": r["arm"], "name": NAMES[r["arm"]], "kind": KIND[r["arm"]], **{k: (None if r.get(k) is None or (isinstance(r.get(k), float) and np.isnan(r[k])) else round(float(r[k]), 4))
                 for k in ("R@5", "excess_R@5", "MIPS_gain_top5", "busiest_top1", "new_to_PCP_top5", "median_dist_top5_mi", "R@5_blend0.3", "MIPS_gain_blend0.3")}})
out["baselines"] = {"n": trig["n_referrals"], "arms": rows}
def tst(k):
    t = trig[k]; return {"label": t["contrast"], "point": round(t["point"], 4), "ci": [round(v, 4) for v in t["pcp_95"]],
                         "ci_adj": [round(v, 4) for v in t["pcp_holm_bound_10"]], "outcome": t["outcome"], "band": round(t["noise_band"], 4)}
out["tests"] = {"tabular": tst("T12_tabular_vs_graph"), "message_passing": tst("T8_message_passing")}
# first-time MRR over ALL specialists (the models' own tuning criterion; 2018 selection queries), seed-averaged
def sel(pattern): return float(np.mean([json.load(open(f))["best_val_selection"] for f in glob.glob(f"runs/*_{pattern}/summary.json")]))
out["fullpool"] = {"gnn": round((sel("v5_unmasked_s42") + sel("v5_unmasked_s1")) / 2, 4), "nograph": round((sel("v5_nograph_s42") + sel("v5_nograph_s1")) / 2, 4), "lgbm_tabular": round((sel("lgbm_tabular_s42") + sel("lgbm_tabular_s1")) / 2, 4),
                   "logreg_tabular": round(sel("logreg_tabular_s42"), 4)}

# ---- LightGBM gain importance (seed-averaged), grouped ---------------------------------------------------------------
GROUP = {"pair_log_prior_referral_count": "PCP → specialist referral count", "pair_log_days_since_last_referral": "Days since that PCP last referred there",
         "pair_recency_missing": "Days since that PCP last referred there", "pair_log_patient_specialist_prior_count": "Patient already saw this specialist",
         "g_cat_share": "Specialist's share of referrals for this disease category", "g_cat_refs_log": "Specialist's share of referrals for this disease category",
         "pair_patient_distance_scaled": "Distance from patient", "pair_patient_distance_missing": "Distance from patient", "pair_distance_scaled": "Distance from PCP",
         "pair_distance_missing": "Distance from PCP", "g_coref_max_log": "Co-referral with patient's past specialists", "g_coref_sum_log": "Co-referral with patient's past specialists",
         "g_twohop_share": "Used by PCPs with overlapping networks", "g_spec_total_log": "Specialist referral volume", "g_spec_refs365_log": "Specialist referral volume",
         "g_spec_pcps365_log": "Specialist referral volume", "g_pcp_distinct_log": "PCP network size", "g_pcp_total_log": "PCP network size",
         "pair_shared_affiliation": "Shared hospital", "pair_affiliation_unknown": "Shared hospital", "pair_insurer_history": "Specialist sees this insurer",
         "pair_insurer_unknown": "Specialist sees this insurer"}
def grp(n):
    if n in GROUP: return GROUP[n]
    if n.startswith("spec_"): return "Specialist attributes (specialty, MIPS, volume…)"
    if n.startswith("pcp_"): return "PCP attributes"
    if n.startswith("dx_"): return "Diagnosis categories"
    if n.startswith("enc_static_") and 7 <= int(n.rsplit("_", 1)[1]) <= 73: return "Patient's visit history"
    if n.startswith(("enc_", "gap_")): return "Patient and encounter"
    return "Other"
imp = {}
if lgb:
    for v in ("tabular",):
        s = []
        for sd in ("s42", "s1"):
            b = lgb.Booster(model_file=glob.glob(f"runs/*_lgbm_{v}_{sd}/model.txt")[0]); g = b.feature_importance("gain"); s.append(pd.Series(g / g.sum(), index=b.feature_name()))
        m = pd.concat(s, axis=1).mean(1); gg = m.groupby(grp).sum().sort_values(ascending=False)
        imp[v] = [{"group": k, "share": round(float(x), 4)} for k, x in gg.items() if x >= 0.005]
out["importance"] = imp

# ---- 11.4(a) network evolution ----------------------------------------------------------------------------------------
ne = json.load(open(f"{A}/step11_4a/summary.json"))
out["network"] = {"yearly": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in ne["yearly"]],
                  "transitions": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in ne["transitions"]]}

# ---- preliminary data-size runs -----------------------------------------------------------------------------------------
ds = json.load(open(f"{A}/step11_4_prelim/datasize_summary.json"))
out["datasize"] = {"models": [{"trained_on": r["trained_on"], "model": r["model"], "r5": round(r["R@5"], 4), "ci": [round(x, 4) for x in r["R@5_ci"]]} for r in ds["models"] if "network counts" not in r["model"]],
                   "gaps": [{"gap": g["gap"], "a": round(g["trained_2015_2016"], 4), "a_ci": [round(x, 4) for x in g["ci_2015_2016"]],
                             "b": round(g["trained_2015_2017"], 4), "b_ci": [round(x, 4) for x in g["ci_2015_2017"]]} for g in ds["gaps"] if "network counts" not in g["gap"]]}

# ---- snapshot status (run summaries only) -----------------------------------------------------------------------------------
runs = []
for line in open(code_file("runlists/step11_4c_snapshots.tsv")):
    if line.startswith("#") or not line.strip(): continue
    name = line.split("\t")[0]; done = glob.glob(f"runs/*_{name}/summary.json")
    s = json.load(open(done[0])) if done else None
    runs.append({"name": name, "snapshot": name.split("_")[1], "model": "GNN" if "_gnn_" in name else "no message passing",
                 "done": bool(done), "firstobs_mrr": round(s["best_val_firstobs_mrr_eligible"], 4) if s else None, "best_epoch": s["best_epoch"] if s else None})
out["snapshots"] = {"runs": runs, "windows": {"S16": ["2016-09-30", "Oct–Dec 2016"], "S17": ["2017-09-30", "Oct–Dec 2017"], "S18": ["2018-09-30", "Oct–Dec 2018"], "S19": ["2019-07-21", "22 Jul–21 Oct 2019"]},
                    "train_queries": {"S16": 12687, "S17": 41702, "S18": 78967, "S19": 102535}}
# ---- 11.2 splits on the lenses (aggregates) ---------------------------------------------------------------------------
st = json.load(open(f"{A}/step11_2/strata_summary.json"))
SPLIT_LABEL = {"Patient age": "Patient age", "Patient local access (distance to 5th-nearest eligible specialist)": "Patient access to specialists",
               "PCP Medicare tenure": "PCP tenure", "PCP network so far (distinct specialists used)": "PCP network size",
               "Referred specialist Medicare tenure": "Specialist tenure", "Referred specialist experience (referrals received before)": "Specialist experience",
               "Referred specialist location": "Specialist location"}
out["splits"] = [{"split": SPLIT_LABEL[k], "stratum": r["stratum"], "n": r["referrals"], "patients": r["patients"], "pcps": r["pcps"], "small": r["small_cell"],
                  "load_d": round(r["GNN−table|q30|load"], 4), "load_ci": [round(x, 4) for x in r["GNN−table|q30|load|ci"]],
                  "mips_d": round(r["GNN−table|q30|quality"], 3), "mips_ci": [round(x, 3) for x in r["GNN−table|q30|quality|ci"]],
                  "load_gnn": round(r["GNN|q30|load"], 4), "load_table": round(r["Table (LightGBM)|q30|load"], 4), "load_actual": round(r["Actual referral|base|load"], 4),
                  "mips_gnn": round(r["GNN|q30|quality"], 2), "mips_table": round(r["Table (LightGBM)|q30|quality"], 2),
                  "fit_gnn": round(r["GNN|q30|fit"], 4), "fit_table": round(r["Table (LightGBM)|q30|fit"], 4)}
                 for k, rows in st["splits"].items() for r in rows if r["referrals"] > 0]
out["exposure"] = [{"group": e["group"], "available": round(e["available_share"], 4), "actual": round(e["actual_referrals"], 4),
                    "gnn": round(e["GNN|q30|top5"], 4), "table": round(e["Table (LightGBM)|q30|top5"], 4)} for e in st.get("exposure", [])]
nc = json.load(open(f"{A}/step11_6_picks/newconn_summary.json"))["blend0.3"]
K = ["MIPS (scored)", "distance mi", "busiest 10%", "treated this disease category before", "seen this insurer before"]
out["newconn"] = {"new_share": {"graph": nc["graph"]["new share of top-5"], "table": nc["table"]["new share of top-5"]},
                  "good_new": {"graph": nc["graph"]["good new (new, higher MIPS than chosen, not busiest)"], "table": nc["table"]["good new (new, higher MIPS than chosen, not busiest)"]},
                  "table_only_new": {k: nc["swap"]["table-only new picks"][k] for k in K}, "graph_instead": {k: nc["swap"]["graph-only picks on the same referrals"][k] for k in K},
                  "graph_instead_familiar": nc["swap"]["share of graph-only picks that are familiar to the PCP"],
                  "referrals": nc["swap"]["referrals with at least one table-only new pick"], "of": nc["swap"]["of referrals"],
                  "familiar_busy": {"graph": nc["graph"]["familiar picks"]["busiest 10%"], "table": nc["table"]["familiar picks"]["busiest 10%"]}}
mi_tab = []
if lgb:
    for sd in ("s42", "s1"):
        b_ = lgb.Booster(model_file=glob.glob(f"runs/*_lgbm_tabular_{sd}/model.txt")[0]); g_ = pd.Series(b_.feature_importance("gain"), index=b_.feature_name()); g_ = g_ / g_.sum()
        mi_tab.append(float(g_[[f"spec_num_{i}" for i in (12, 13, 14, 47, 48)]].sum()))
out["mips_importance"] = {"table": float(np.mean(mi_tab)) if mi_tab else None, "gnn_shap": 0.013}
al = json.load(open(f"{A}/step11_5/arch_lenses.json"))["variants"]["with 30% quality weight"]
out["arch"] = {"band": al["noise_band"], "arms": [{k: v for k, v in r.items() if k != "why"} for r in al["arms"]]}
json.dump(out, open(f"{OUTD}/data_w2.json", "w"), indent=1)
print("wrote presentation/data_w2.json:", {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()})
