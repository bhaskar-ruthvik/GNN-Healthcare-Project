"""Collect AGGREGATE numbers for the HTML presentation (presentation/data.json). No referral- or provider-level values."""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json, os
import pandas as pd
os.chdir(str(workspace())); OUTD = "deliverables/presentation"; os.makedirs(OUTD, exist_ok=True)
A = "runs/analysis"; out = {}
s1 = json.load(open(f"{A}/step10_1/summary.json"))
out["descriptive"] = {k: s1[k] for k in ("01", "03", "04", "06", "07", "08", "09", "10")} | {"funnel": s1["funnel"], "n_primary": s1["n_primary"],
    "load_by_specialty": {k: round(v["top10_share"], 3) for k, v in s1["05"]["by_specialty"].items()}, "load_weighted": s1["05"]["weighted_top10_share"]}
MAP = {"model_unmasked_v5": "model", "pcp": "pcp", "dist": "nearest", "mips": "mips", "random": "random", "practice": "practice"}
COLS = ["r5", "gain5", "gain5_lo", "gain5_hi", "gain5_cons", "gain5_cons_lo", "gain5_cons_hi", "gain1", "within1_1", "busy1", "busy5", "dist5_median", "dist5_p90", "new_pcp5"]
def arms(path, split, fams=MAP):
    a = pd.read_csv(f"{path}/{split}_arms.csv"); a = a[a.family.isin(fams)]
    rows = []
    for _, r in a.iterrows():
        rows.append({"family": fams[r.family], "variant": r.variant, **{c: (None if pd.isna(r.get(c)) else round(float(r[c]), 4)) for c in COLS if c in r}})
    return rows
out["tradeoff"] = {sp: arms(f"{A}/step10_3", sp) for sp in ("validation", "test")}
caps = {"10": f"{A}/step10_4/cap10", "25": f"{A}/step10_3", "50": f"{A}/step10_4/cap50", "none": f"{A}/step10_4/capnone"}
out["caps"] = {sp: {c: [x for x in arms(p, sp) if x["family"] in ("model", "practice", "pcp", "mips") and x["variant"] in ("base", "blend0.3")] for c, p in caps.items()} for sp in ("validation", "test")}
rel = {}
for sp in ("validation", "test"):
    a = pd.read_csv(f"{A}/step10_4/reliance/{sp}_arms.csv"); a = a[a.family.str.startswith("model_unmasked_v5")]
    rel[sp] = [{"removed": (r.family.split("@")[1] if "@" in r.family else "none"), "variant": r.variant, **{c: round(float(r[c]), 4) for c in ("r5", "gain5", "busy1", "busy5", "dist5_median", "new_pcp5")}}
               for _, r in a.iterrows() if r.variant in ("base", "blend0.3")]
out["reliance"] = rel
bias = {}
for sp in ("validation", "test"):
    g = pd.read_csv(f"{A}/step10_5/{sp}_gap_changes.csv"); g = g[g.config == "final (v5, blend 0.3)"]
    t = pd.read_csv(f"{A}/step10_5/{sp}_group_table.csv")
    bias[sp] = {"gaps": g[["attribute", "group", "reference", "quantity", "practice_gap", "recommended_gap", "change_in_gap", "lo", "hi"]].round(3).to_dict("records"),
                "groups": t[["attribute", "group", "reference", "n_patients", "practice_mips", "practice_dist"]].round(2).to_dict("records")}
    m = pd.read_csv(f"{A}/step10_5/{sp}_masking_attribution.csv"); bias[sp]["masking_attributed"] = int(m.attributed_to_masking.sum()); bias[sp]["masking_rows"] = len(m)
out["bias"] = bias
nc = {}
for sp in ("validation", "test"):
    t = pd.read_csv(f"{A}/step10_5_capnone/{sp}_group_table.csv"); t = t[t.attribute == "area_zip3"]; t25 = pd.read_csv(f"{A}/step10_5/{sp}_group_table.csv"); t25 = t25[t25.attribute == "area_zip3"]
    F = "final (v5, blend 0.3)"
    nc[sp] = [{"zip3": r.group, "practice_mips": round(r.practice_mips, 1), "practice_dist": round(r.practice_dist, 1),
               "gain_25": round(float(t25[t25.group == r.group][f"{F}|gain5"].iloc[0]), 2), "gain_none": round(float(r[f"{F}|gain5"]), 2),
               "dist_25": round(float(t25[t25.group == r.group][f"{F}|dist5m"].iloc[0]), 1), "dist_none": round(float(r[f"{F}|dist5m"]), 1)} for _, r in t.iterrows()]
out["area_nocap"] = nc
sh = pd.read_csv(f"{A}/step10_6/20260929T213930_v5_unmasked_s42/summary_shares.csv"); sh = sh[sh.explained == "diff"]
out["shap"] = sh[["input", "kind", "share_mean_abs", "lo", "hi", "mean_signed"]].round(4).to_dict("records")
p7 = f"{A}/step10_7" if os.path.exists(f"{A}/step10_7/test_arms.csv") else None
LMAP = {"model_unmasked_v5": "model", "model_unmasked_v6_load": "model_load_input", "model_unmasked_v5_posw": "model_load_weighted", "practice": "practice", "pcp": "pcp"}
out["load"] = {"official": p7 is not None, "splits": {}}
for sp in ("validation", "test"):
    path = f"{A}/step10_7" if p7 else f"{A}/step10_7_preview"
    if not os.path.exists(f"{path}/{sp}_arms.csv"): continue
    out["load"]["splits"][sp] = arms(path, sp, LMAP)
if p7:
    out["load"]["decision"] = json.load(open(f"{A}/step10_7/validation_summary.json"))["a10"]
seen = {}
for sp in ("validation", "test"):
    a_ = pd.read_csv(f"{A}/step10_4/include_seen/{sp}_arms.csv").set_index("arm").loc["model_unmasked_v5:blend0.3"]
    b_ = pd.read_csv(f"{A}/step10_3/{sp}_arms.csv").set_index("arm").loc["model_unmasked_v5:blend0.3"]
    seen[sp] = {"seen1": round(float(a_.seen1), 4), "r5_incl": round(float(a_.r5), 4), "r5_excl": round(float(b_.r5), 4), "gain_incl": round(float(a_.gain5), 2),
                "gain_excl": round(float(b_.gain5), 2), "busy_incl": round(float(a_.busy1), 4), "busy_excl": round(float(b_.busy1), 4), "dist_incl": round(float(a_.dist5_median), 1)}
out["include_seen"] = seen
conc = {}
for sp in ("validation", "test"):
    c_ = pd.read_csv(f"{A}/step10_9_concordance/{sp}_concordance.csv"); c_ = c_[(c_.top == "top pick")]
    conc[sp] = [{"pcp_race": r.pcp_race, "n_pcps": int(r.n_pcps), "n_referrals": int(r.n_referrals), "config": r.config, "availability": round(float(r.availability), 4),
                 "share": round(float(r.share), 4), "excess": round(float(r.excess), 4), "change": round(float(r.change_vs_practice), 4),
                 "change_ci": [round(float(x), 4) for x in str(r.change_ci).strip("()").split(",")]} for _, r in c_.iterrows()]
out["concordance"] = conc
json.dump(out, open(f"{OUTD}/data.json", "w"), indent=0)
print("wrote presentation/data.json", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items()})
