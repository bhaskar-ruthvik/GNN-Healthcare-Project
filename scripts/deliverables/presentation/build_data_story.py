"""Collect AGGREGATE numbers for the combined story deck (presentation/data_story.json). No referral-, patient- or provider-level values.
Reuses the week-3 data file (Step 12 results; rebuilt first so it is current) and adds Steps 10–11, STEP12e and context numbers,
each block with its source file ("src").
Run from anywhere (data under the workspace, see referral_gnn/paths.py):  python3 scripts/deliverables/presentation/build_data_story.py
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os
import subprocess
import sys

os.chdir(str(workspace())); OUTD = "deliverables/presentation"; os.makedirs(OUTD, exist_ok=True)
A = "runs/analysis"; J = lambda p: json.load(open(p))
subprocess.run([sys.executable, str(script_path("build_data_w3.py"))], check=True, stdout=subprocess.DEVNULL)
out = J(f"{OUTD}/data_w3.json")                       # Step 12 blocks: groups, topk, mech, hyb, comp, frontier, blend, heavy, bchk

# ---- data and cohort ------------------------------------------------------------------------------------------------------------
man = J(sorted(__import__("glob").glob("prepared_data/referral_prepared_*/manifest_PRIVATE.json"))[0])
T = man["tables"]
out["data"] = {"src": "prepared_data/referral_prepared_*/manifest_PRIVATE.json (row counts only)", "events": T["encounter_features_PRIVATE.pkl"]["rows"],
               "patients": T["patient_nodes_PRIVATE.pkl"]["rows"], "pcps": T["pcp_locations_PRIVATE.pkl"]["rows"], "specialists": T["specialist_locations_PRIVATE.pkl"]["rows"]}
sv, st = J(f"{A}/step10_3/validation_summary.json"), J(f"{A}/step10_3/test_summary.json")
out["cohort"] = {"src": [f"{A}/step10_3/validation_summary.json", f"{A}/step10_3/test_summary.json"], "2018": sv["funnel"], "2019": st["funnel"]}

# ---- Step 10: imitation, quality weight, reliance (from the 2 Oct deck's data file, built from the step10 outputs) -----------------
import pandas as pd
d1 = J(f"{OUTD}/data.json")                                  # only the step10_1 descriptive block is reused from it
COLS = ("r5", "gain5", "gain5_lo", "gain5_hi", "busy1", "busy5", "new_pcp5", "dist5_median")
def row(a, arm): r = a.loc[arm]; return {c: (None if pd.isna(r[c]) else float(r[c])) for c in COLS}
TO = {}
for sp, yr in (("validation", "2018"), ("test", "2019")):
    a = pd.read_csv(f"{A}/step10_3/{sp}_arms.csv").set_index("arm")       # full precision (the 2 Oct data file rounds to 4 decimals)
    TO[yr] = {"practice": row(a, "practice:base"), "random": row(a, "random:base"), "mips": row(a, "mips:base"),
              "model": {v: row(a, f"model_unmasked_v5:{v}") for v in ("base", "blend0.1", "blend0.2", "blend0.3", "blend0.5")},
              "pcp": {v: row(a, f"pcp:{v}") for v in ("base", "blend0.1", "blend0.2", "blend0.3", "blend0.5")}}
rel = {}
for sp, yr in (("validation", "2018"), ("test", "2019")):
    a = pd.read_csv(f"{A}/step10_4/reliance/{sp}_arms.csv"); a = a[a.family.str.startswith("model_unmasked_v5")]
    rel[yr] = {f"{r.family.split('@')[1] if '@' in r.family else 'none'}|{r.variant}": {c: float(r[c]) for c in ("r5", "gain5", "busy1", "busy5", "dist5_median", "new_pcp5")}
               for _, r in a.iterrows() if r.variant in ("base", "blend0.3")}
out["s10"] = {"src": [f"{A}/step10_3/{{validation,test}}_arms.csv", f"{A}/step10_4/reliance/{{validation,test}}_arms.csv"], "tradeoff": TO,
              "reliance": {yr: {k: rel[yr][k] for k in ("none|base", "none|blend0.3", "no_pcp_history|base", "no_mips|base")} for yr in rel}}
ds = eval(d1["descriptive"]["03"]) if isinstance(d1["descriptive"]["03"], str) else d1["descriptive"]["03"]
out["s10"]["practice_mips"] = {"src": f"{A}/step10_1/summary.json", "random_option_higher": ds["random_option_higher"], "ref_mean": ds["ref_mean"], "elig_mean": ds["elig_mean"]}

# ---- Step 11: graph vs table (both years) ------------------------------------------------------------------------------------------
lc = J(f"{A}/step11_1_tabular/lens_compare.json")
out["s11"] = {"src": f"{A}/step11_1_tabular/lens_compare.json",
              "years": {yr: {"actual_load": lc[sp].get("actual_load"), "actual_new": lc[sp].get("actual_new"),
                             **{k.split("|")[1]: v for k, v in lc[sp].items() if k.startswith("q30|")}} for sp, yr in (("validation", "2018"), ("test", "2019"))}}

# ---- STEP12e: steered blend with 10 / 15 names ---------------------------------------------------------------------------------------
E = {}
for K in (10, 15):
    for yr, tag in (("2018", "2018"), ("2019", "2019")):
        e = J(f"{A}/step12e/K{K}/step12e_K{K}_{tag}.json")
        E.setdefault(str(K), {})[yr] = [{"fixed": {a: pp["fixed"][a] for a in ("GNN", "Table", "Steered blend")}, "ci_load": pp["ci_vs_GNN"]["Steered blend"]["busy1"],
                                         "ci_fit": pp["ci_vs_GNN"]["Steered blend"]["fit"]} for pp in e["per_pair"]]
out["s12e"] = {"src": f"{A}/step12e/K{{10,15}}/step12e_K*_{{2018,2019}}.json (plan STEP12E_TOPK_PLAN.md)", "K": E}

# ---- context: network turnover (Step 11.4a) and masking probe (Step 11.3a) -------------------------------------------------------------
d2 = J(f"{OUTD}/data_w2.json")
out["network"] = {"src": f"{A}/step11_4a/summary.json", **d2["network"]}
pr = J(f"{A}/step11_3a/probe_summary.json")
out["masking"] = {"src": f"{A}/step11_3a/probe_summary.json", **{k: {"masked": v["masked embeddings"]["auc"], "unmasked": v["unmasked embeddings"]["auc"], "inputs": v["masked model inputs (LightGBM)"]["auc"]} for k, v in pr.items()}}
json.dump(out, open(f"{OUTD}/data_story.json", "w"), indent=1, default=float)
print("wrote presentation/data_story.json", {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()}, os.path.getsize(f"{OUTD}/data_story.json") // 1024, "KB")
