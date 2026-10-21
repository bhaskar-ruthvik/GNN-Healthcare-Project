"""Collect AGGREGATE numbers for the week-3 deck (presentation/data_w3.json). No referral- or provider-level values.
Every number on the slides is read from this file; each block records the result file it came from ("src").
Sources: STEP12 group deep-dive and top-K (10-12), STEP12a congestion mechanism, STEP12b hybrids, STEP12c complementarity /
frontiers / 2019 check / blend checks, STEP12d heavier hybrids.
Run from anywhere (data under the workspace, see referral_gnn/paths.py):  python3 scripts/deliverables/presentation/build_data_w3.py
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os

import numpy as np
import pandas as pd

os.chdir(str(workspace())); OUTD = "deliverables/presentation"; os.makedirs(OUTD, exist_ok=True)
A = "runs/analysis"; out = {}
J = lambda p: json.load(open(f"{A}/{p}"))
r4 = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 8)
ci4 = lambda c: [r4(c[0]), r4(c[1])]

# ---- 1. groups that broke the trend (2018, d = GNN − table busiest-10% share of the top pick, 30% weight) --------------------
g = J("step12_groups/groups_summary.json")
LBL = {"Patient age": "Patient age", "PCP Medicare tenure": "PCP tenure"}
GL = {"early": "early (≤ 6 y)", "mid": "mid (7–13 y)", "late": "late (≥ 14 y)"}
ORDER = {"<65": 0, "65–74": 1, "75–84": 2, "85+": 3, "early": 0, "mid": 1, "late": 2}
did = sorted([r for r in g["did"] if r["variant"] == "q30"], key=lambda r: (r["split"] != "Patient age", ORDER[r["group"]]))
out["groups"] = {"src": f"{A}/step12_groups/groups_summary.json", "n": g["n"],
                 "did": [{"split": LBL[r["split"]], "group": GL.get(r["group"], r["group"]), "n": r["n"], "pcps": r["pcps"], "d_group": r4(r["d_group"]),
                          "d_rest": r4(r["d_rest"]), "est": r4(r["did_est"]), "ci": ci4(r["did_ci"]), "p": r4(r["did_p"]), "p_holm": r4(r["did_p_holm"])} for r in did],
                 "std": {k: {kk: r4(vv) for kk, vv in v.items()} for k, v in g["age_specialty_standardisation"]["q30"].items() if isinstance(v, dict)},
                 "age": {a: {"n": p["n"], "available_busy": r4(p["available_busy"]), "actual_busy": r4(p["actual_busy"]), "gnn": r4(p["GNN|q30|load"]),
                             "table": r4(p["Table|q30|load"]), "nograph": r4(p["No-graph|q30|load"])} for a, p in g["age_profile"].items()},
                 "dose": [{"q": r["quintile"], "range": r["range"], "n": r["n"], "pcps": r["pcps"], "early": r["early_tenure_pcp_referrals"],
                           "est": r4(r["d|q30"]["est"]), "ci": ci4(r["d|q30"]["ci"])} for r in g["pcp_history_dose"]["pcp_prior_refs"]],
                 "early": {k: r4(v) for k, v in g["pcp_history_dose"]["early_tenure_pcps"].items()},
                 "reg": {k: {"coef": r4(v["coef"]), "ci": ci4(v["ci"])} for k, v in g["regression_d_load"]["q30"].items()}}

# 9 Oct forest plot (week-2 deck slide "across groups"), same construction as build_data_w2.py
st = J("step11_2/strata_summary.json")
SPLIT_LABEL = {"Patient age": "Patient age", "Patient local access (distance to 5th-nearest eligible specialist)": "Patient access to specialists",
               "PCP Medicare tenure": "PCP tenure", "PCP network so far (distinct specialists used)": "PCP network size",
               "Referred specialist Medicare tenure": "Specialist tenure", "Referred specialist experience (referrals received before)": "Specialist experience",
               "Referred specialist location": "Specialist location"}
out["groups"]["w2"] = {"src": f"{A}/step11_2/strata_summary.json", "rows": [{"split": SPLIT_LABEL[k], "stratum": r["stratum"], "n": r["referrals"], "patients": r["patients"], "pcps": r["pcps"], "small": r["small_cell"],
                       "d": r4(r["GNN−table|q30|load"]), "ci": ci4(r["GNN−table|q30|load|ci"]), "gnn": r4(r["GNN|q30|load"]), "table": r4(r["Table (LightGBM)|q30|load"]), "actual": r4(r["Actual referral|base|load"])}
                      for k, rows in st["splits"].items() for r in rows if r["referrals"] > 0]}

# ---- 2. top 10 / 15 (2018, 30% weight; GNN and table seed-averaged, 2 seeds each) -----------------------------------------------
tk = J("step12_topk/topk_summary.json")
out["topk"] = {"src": f"{A}/step12_topk/topk_summary.json", "K": [5, 10, 15],
               "arms": {a: {m: [r4(tk[f"K{k}"]["arms"][f"{a}|q30"][m]) for k in (5, 10, 15)] for m in ("fit", "quality", "travel", "load_slots", "load_top1", "new")}
                        for a in ("GNN", "Table (LightGBM)")},
               "practice": {k: r4(v) for k, v in tk["K5"]["practice"].items()},
               "diff": {m: [{"est": r4(tk[f"K{k}"]["gnn_minus_table"][f"{m}|q30"]["est"]), "ci": ci4(tk[f"K{k}"]["gnn_minus_table"][f"{m}|q30"]["ci"])} for k in (5, 10, 15)]
                        for m in ("fit", "quality", "travel", "load_slots", "load_top1", "new")},
               "beyond25": [r4(tk[f"K{k}"]["filled_beyond_25mi"]) for k in (5, 10, 15)], "short": [r4(tk[f"K{k}"]["list_shorter_than_K"]) for k in (5, 10, 15)]}

# ---- 3. congestion mechanism (STEP12a; 30% weight) -------------------------------------------------------------------------------
f12 = J("step12a/final_summary.json"); s12 = J("step12a/secondary_summary.json"); dec = json.load(open("runs/analysis/step12a_decisions.json")) if os.path.exists("runs/analysis/step12a_decisions.json") else {}
D_ = f12["descriptive"]
def cell(name, kind, label, src):
    l = src["lenses_q30"]; return {"name": label, "kind": kind, "fit": r4(l["fit"]), "load": r4(l["load"]), "new": r4(l["new_to_pcp"]), "mips": r4(l["mips_gain"]), "seeds": l.get("n_seeds")}
C_ = s12["cells"]
cells = [cell(None, "tree", "LightGBM, LambdaRank (the table)", D_["LightGBM LambdaRank"]), cell(None, "tree", "LightGBM, softmax objective", D_["LightGBM softmax"]),
         cell(None, "tree", "LightGBM, binary objective", D_["LightGBM binary"]),
         {"name": "LightGBM, volume-blind attributes", "kind": "tree", "fit": r4(C_["LS"]["fit"]), "load": r4(C_["LS"]["load"]), "new": r4(C_["LS"]["new_to_pcp"]), "mips": r4(C_["LS"]["mips_gain"]), "seeds": C_["LS"]["seeds"]},
         cell(None, "graph", "GNN (current)", D_["model_unmasked"]), cell(None, "graph", "GNN, no message passing", D_["model_unmasked_nograph"]),
         {"name": "Neural, no PCP history at all", "kind": "neural", "fit": r4(C_["N"]["fit"]), "load": r4(C_["N"]["load"]), "new": r4(C_["N"]["new_to_pcp"]), "mips": r4(C_["N"]["mips_gain"]), "seeds": C_["N"]["seeds"]},
         cell(None, "neural", "Neural, LambdaRank objective", D_["neural LambdaRank"]), cell(None, "neural", "Neural, binary objective", D_["neural BCE"]),
         cell(None, "neural", "Neural, log volume", D_["N, log volume"]),
         {"name": "Neural, volume-blind attributes", "kind": "neural", "fit": r4(C_["NS"]["fit"]), "load": r4(C_["NS"]["load"]), "new": r4(C_["NS"]["new_to_pcp"]), "mips": r4(C_["NS"]["mips_gain"]), "seeds": C_["NS"]["seeds"]}]
H = f12["hypotheses"]; HS = s12["hypotheses"]
HYP = [("H1", "Model type: tree vs neural, same objective", "H1", "H1"), ("H2", "Training objective (neural LambdaRank vs softmax)", "H2", "H2"),
       ("H3", "Volume attributes explain the gap", "H3", "H3′"), ("H4", "Tree capacity (bigger trees concentrate more)", "H4", None), ("H5", "Volume scaling (log volume)", "H5", "H5")]
out["mech"] = {"src": [f"{A}/step12a/final_summary.json", f"{A}/step12a/secondary_summary.json"], "R": r4(f12["R"]), "R_half": r4(f12["R_half"]), "noise_neural": r4(f12["noise_band_neural"]),
               "cells": cells, "capacity": [{"fit": r4(c["fit"]), "load": r4(c["load"]), "leaves": c["leaves"], "rounds": c["rounds"]} for c in f12["capacity_grid"]],
               "tests": [{"id": i, "label": l, "effect": r4(H[p]["effect"]), "ci": ci4(H[p]["ci"]), "verdict": H[p]["verdict"],
                          "sec_label": (s if s else None), "sec_effect": r4(HS[s]["effect"]) if s else None, "sec_ci": ci4(HS[s]["ci"]) if s else None,
                          "sec_verdict": HS[s].get("verdict_secondary") if s else None} for i, l, p, s in HYP],
               "overall": f12["overall"], "noise_matched_H1": r4(s12["noise_degradation_matched"]["H1"]), "base_H1": r4(s12["base_ranking_point"]["H1"]),
               "volpct": {"LightGBM": r4(D_["LightGBM LambdaRank"]["volpct1_ft"]), "neural": r4(D_["N (neural softmax)"]["volpct1_ft"]), "practice": r4(D_["practice_ref_volpct_ft"])},
               "b1": {"LightGBM": r4(D_["LightGBM LambdaRank"]["B1_no_volume"]["load_change_matched"]), "GNN": r4(D_["model_unmasked"]["B1_no_volume"]["load_change_matched"]),
                      "neural": r4(D_["N (neural softmax)"]["B1_no_volume"]["load_change_matched"])},
               "gate": {k: dec.get(k) for k in ("d_G", "d_NG", "branch")} if dec else {},
               "N3": cell(None, "neural", "Neural, no PCP history (3 seeds, as in the tests)", D_["N (neural softmax)"])}

# ---- 4. simple hybrids (STEP12b; 30% weight, 2 seed pairings) ------------------------------------------------------------------
j = J("step12b/judge_summary.json")
NM = {"hyb_blend0.25": "Score blend, 25% table", "hyb_blend0.5": "Score blend, 50/50", "hyb_blend0.75": "Score blend, 75% table", "hyb_gate": "Gate by PCP / patient profile",
      "hyb_rrf": "Rank fusion (RRF)", "hyb_split1": "GNN top pick, table for the rest", "hyb_split2": "GNN top 2, table for the rest",
      "s12b_emb_lr": "GNN encoding + variables → logistic regression", "s12b_vars_lr": "Same variables, no GNN encoding", "s12_lgbm_flat_lambdarank": "Flattened graph → LightGBM"}
LK = ("r5", "busy1", "busy5", "new_pcp5", "gain5", "median_travel_mi")
out["hyb"] = {"src": f"{A}/step12b/judge_summary.json", "refs": {k: {m: r4(v[m]) for m in LK} for k, v in j["references"].items()},
              "arms": [{"key": k, "name": NM[k], **{m: r4(a["lenses"][m]) for m in LK}, "pass": a["combines_strengths"],
                        "ci": {m: ci4(c) for m, c in a["ci_delta_vs_GNN"].items()}} for k, a in j["arms"].items()]}

# ---- 5. complementarity, frontiers, 2019 check (STEP12c) ---------------------------------------------------------------------
c = J("step12c/complementarity.json")
PK = ("busy", "new", "ins", "mips", "dist")
out["comp"] = {"src": f"{A}/step12c/complementarity.json",
               "agree1": [r4(p["A_top1_agreement"]) for p in c["per_pair"]], "overlap5": [r4(p["A_top5_overlap_mean"]) for p in c["per_pair"]],
               "table_only": {k: [r4(p["B_table_only_top5"][k]) for p in c["per_pair"]] for k in PK}, "gnn_only": {k: [r4(p["B_gnn_only_top5"][k]) for p in c["per_pair"]] for k in PK},
               "oracle": r4(c["per_pair"][0]["E2_hindsight_fit_oracle"]), "fit_gnn": [r4(p["GNN"]["fit"]) for p in c["per_pair"]], "fit_table": [r4(p["Table"]["fit"]) for p in c["per_pair"]]}
fr12c = J("step12c/frontiers.json"); fr12d = J("step12d/frontier.json")
FR = []
for k in range(3):
    pc = pd.read_csv(f"{A}/step12c/frontier_points_pair{k}.csv"); pd_ = pd.read_csv(f"{A}/step12d/frontier_points_pair{k}.csv")
    b1 = pc[pc.family == "B"].set_index(["a_new", "b_busy"]).sort_index(); b2 = pd_[pd_.family == "B"].set_index(["a_new", "b_busy"]).sort_index()
    assert np.allclose(b1[["fit", "busy1", "new5"]].to_numpy(), b2[["fit", "busy1", "new5"]].to_numpy(), atol=1e-9), "12c and 12d blend grids differ"
    pts = pd.concat([pc[pc.family.isin(["G", "T", "B"])], pd_[pd_.family.isin(["O", "D", "S"])].assign(family=lambda d: d.family.map({"O": "O", "D": "D", "S": "K"}))])
    FR.append([{"f": r.family, "a": r.a_new, "b": r.b_busy, "fit": r4(r.fit), "busy1": r4(r.busy1), "busy5": r4(r.busy5), "new5": r4(r.new5), "gain5": r4(r.gain5)} for r in pts.itertuples()])
out["frontier"] = {"src": [f"{A}/step12c/frontier_points_pair{{0,1,2}}.csv", f"{A}/step12d/frontier_points_pair{{0,1,2}}.csv", f"{A}/step12c/frontiers.json", f"{A}/step12d/frontier.json"],
                   "grid": fr12c["grid"], "points": FR, "families": {"G": "GNN, steered", "T": "Table, steered", "B": "Equal blend, steered", "O": "Table-offset GNN, steered",
                                                                      "D": "Dual encoder, steered", "K": "Stacked LightGBM, steered"},
                   "lowest": {F: [{t: v for t, v in fr12d["per_pair"][k]["floors"][F]["lowest_busy1_at_new5"].items()} for k in range(3)] for F in ("0.445", "0.45")},
                   "unsteered_12d": [fr12d["per_pair"][k]["unsteered"] for k in range(3)]}
pv = J("step12c/point_lenses_validation.json"); pt = J("step12c/point_lenses_test2019.json"); pre = J("step12c/preregistration_2019.json")
LM = ("fit", "busy1", "busy5", "new5", "gain5", "good_new5", "new_insurer_rate", "dist_mean5")
def lenses(p): return [{arm: {m: r4(pp[arm][m]) for m in LM} for arm in ("GNN", "Table", "Steered blend")} | {"ci_vs_gnn": {m: ci4(v) for m, v in pp["delta_blend_minus_GNN_ci"].items()}} for pp in p["per_pair"]]
out["blend"] = {"src": [f"{A}/step12c/point_lenses_validation.json", f"{A}/step12c/point_lenses_test2019.json", f"{A}/step12c/preregistration_2019.json"],
                "setting": pv["setting"], "n2018": pv["n"], "n2019": pt["n"], "y2018": lenses(pv), "y2019": lenses(pt), "prereg": pre}

# ---- 6. heavier hybrids (STEP12d) --------------------------------------------------------------------------------------------
hd = J("step12d/heavy_descriptives.json")
out["heavy"] = {"src": [f"{A}/step12d/frontier.json", f"{A}/step12d/heavy_descriptives.json"],
                "beta": [r4(v["beta"]) for v in hd["offset_beta"].values()], "beta_epochs": [v["best_epoch"] for v in hd["offset_beta"].values()],
                "stack_share": [r4(v["share"]) for v in hd["stack_gnn_gain_share"].values()], "stack_rank": [v["rank"] for v in hd["stack_gnn_gain_share"].values()],
                "stack_nfeat": list(hd["stack_gnn_gain_share"].values())[0]["n_features"]}

# ---- 7. steered blend: groups and where the load goes (2018, 3 pairings averaged) -------------------------------------------------
bc = J("step12c/blend_checks.json")
out["bchk"] = {"src": f"{A}/step12c/blend_checks.json", "groups": [{"split": s, "group": r["group"], "n": r["referrals"], "patients": r["patients"], "pcps": r["pcps"], "small": r["small"],
                                                                      **{f"{a}|{m}": r4(r[f"{a}|{m}"]) for a in ("GNN", "Table", "Blend") for m in ("busy1", "new5", "gain5", "fit")},
                                                                      **{f"d|{m}": r4(r[f"blend−GNN|{m}"]) for m in ("busy1", "new5", "gain5", "fit")},
                                                                      **{f"d|{m}|ci": ci4(r[f"blend−GNN|{m}|ci"]) for m in ("busy1", "new5", "gain5", "fit")}}
                                                                     for s, rows in bc["groups"].items() for r in rows],
               "where": bc["where_the_load_goes"]}
json.dump(out, open(f"{OUTD}/data_w3.json", "w"), indent=1, default=float)
print("wrote presentation/data_w3.json:", {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()}, os.path.getsize(f"{OUTD}/data_w3.json") // 1024, "KB")
