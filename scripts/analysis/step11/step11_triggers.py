"""Step 11.1 triggers T8 (message passing) and T9 (model class), per STEP11_PLAN (frozen 2026-10-07).

Primary metric: R@5 in excess of the exact random expectation, base configuration, STEP10 primary cohort, seed-averaged
per arm. (In a paired difference the random term cancels, so the contrast equals the difference in R@5.)
CIs: PCP-cluster bootstrap (primary, G4) and patient-cluster bootstrap. Intervals are reported at 95% and at the
Bonferroni level alpha/10 (99.5%), which bounds the Holm adjustment over the ~10 Step 11 primary contrasts (Amendment B1) until all are in.
Outcome rule (margin 0.02, on the adjusted interval): better = lo > 0 and point > +0.02; equivalent = interval inside ±0.02;
worse = hi < 0 and point < −0.02; otherwise inconclusive. Noise band (G2) = GNN dup band + max of the two arms' seed gaps.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="runs/analysis/step11_1"); ap.add_argument("--split", default="validation")
ap.add_argument("--boot", type=int, default=2000); args = ap.parse_args()
root = str(workspace()); os.chdir(root)
pr = pd.read_csv(os.path.join(args.dir, f"{args.split}_per_referral_PRIVATE.csv.gz"))
M = 0.02; LEVELS = {"95": 0.05, "holm_bound_10": 0.05 / 10}       # Amendment B1: ~10 primary contrasts

def run_col(suffix, metric="r5", variant=""):
    cols = [c for c in pr.columns if c.startswith("model:") and c.split(":", 1)[1].split("+")[0].split("|")[0].endswith(suffix)
            and c.endswith(f"{variant}|{metric}") and ("+" in c) == bool(variant)]
    assert len(cols) == 1, (suffix, variant, metric, cols); return pr[cols[0]].to_numpy(float)

ARMS = {"gnn": ["_v5_unmasked_s42", "_v5_unmasked_s1"], "gnn_dup": ["_v5_unmasked_s42_dup"], "nograph": ["_v5_nograph_s42", "_v5_nograph_s1"],
        "lgbm_primary": ["_lgbm_primary_s42", "_lgbm_primary_s1"], "lgbm_graph": ["_lgbm_graph_s42", "_lgbm_graph_s1"], "lgbm_tabular": ["_lgbm_tabular_s42", "_lgbm_tabular_s1"],
        "logreg": ["_logreg_s42"], "als": ["_als_s42"], "itemknn": ["_itemknn"], "logreg_tabular": ["_logreg_tabular_s42"]}
def _present(sfx): return any(c.startswith("model:") and "+" not in c and c.endswith("|r5") and c.split(":", 1)[1].split("|")[0].endswith(sfx) for c in pr.columns)
ARMS = {k: v for k, v in ARMS.items() if all(_present(x) for x in v)}       # folders evaluated with a subset of runs (e.g. tabular-only)
print("arms present:", sorted(ARMS))
def arm(name, metric="r5", variant=""): return np.mean([run_col(s, metric, variant) for s in ARMS[name]], axis=0)
rand = pr["random|r5"].to_numpy(float)

rng = np.random.default_rng(20261007)
def cluster_boot(x, key, B):
    codes, uniq = pd.factorize(pr[key]); sums = np.bincount(codes, weights=x); cnt = np.bincount(codes)
    idx = rng.integers(0, len(uniq), size=(B, len(uniq)))
    return sums[idx].sum(1) / cnt[idx].sum(1)

def contrast(a, b, label):
    d = arm(a) - arm(b); point = float(d.mean())
    gap = lambda n: abs(float(run_col(ARMS[n][0]).mean() - run_col(ARMS[n][1]).mean())) if len(ARMS[n]) == 2 else 0.0
    dup_band = abs(float(run_col("_v5_unmasked_s42").mean() - run_col("_v5_unmasked_s42_dup").mean()))
    band = dup_band + max(gap(a), gap(b))
    res = {"contrast": label, "point": point, "n": int(len(d)), "noise_band": band, "dup_band": dup_band, "seed_gap_a": gap(a), "seed_gap_b": gap(b)}
    for key in ("pcp", "patient"):
        bs = cluster_boot(d, key, args.boot)
        for lv, al in LEVELS.items():
            res[f"{key}_{lv}"] = [float(np.quantile(bs, al / 2)), float(np.quantile(bs, 1 - al / 2))]
    lo, hi = res["pcp_holm_bound_10"]
    res["outcome"] = ("better" if lo > 0 and point > M else "worse" if hi < 0 and point < -M else
                      "equivalent" if (lo > -M and hi < M) else "inconclusive")
    res["exceeds_noise_band"] = abs(point) > band
    return res

# T9's LightGBM variant is selected on validation by the tuning criterion (selection first-time MRR, seed-averaged)
sel = {}
for v in [v for v in ("lgbm_primary", "lgbm_graph") if v in ARMS]:                       # T9 candidates as pre-registered (B1 adds T12 separately)
    vals = [json.load(open(glob.glob(f"runs/*{s}/summary.json")[0]))["best_val_selection"] for s in ARMS[v]]; sel[v] = float(np.mean(vals))
best_lgbm = max(sel, key=sel.get) if sel else None

out = {"split": args.split, "n_referrals": int(len(pr)), "lgbm_selection_firstobs_mrr": sel, "best_lgbm": best_lgbm,
       "T8_message_passing": contrast("gnn", "nograph", "GNN − no-graph"),
       **({"T9_model_class": contrast("gnn", best_lgbm, f"GNN − {best_lgbm}")} if best_lgbm else {}),
       "T12_tabular_vs_graph": contrast("gnn", "lgbm_tabular", "GNN − LightGBM pure tabular"),
       "descriptive_nograph_vs_tabular": contrast("nograph", "lgbm_tabular", "no-graph GNN − LightGBM pure tabular"),
       **({"descriptive_other_lgbm": contrast("gnn", "lgbm_primary" if best_lgbm == "lgbm_graph" else "lgbm_graph", "GNN − other LightGBM")} if best_lgbm and "lgbm_primary" in ARMS else {})}

rows = []
for a in ARMS:
    if a == "gnn_dup": continue
    rec = {"arm": a, "R@5": arm(a).mean(), "excess_R@5": (arm(a) - rand).mean(), "MIPS_gain_top5": arm(a, "gain5").mean(),
           "busiest_top1": arm(a, "busy1").mean(), "new_to_PCP_top5": arm(a, "new_pcp5").mean(), "median_dist_top5_mi": np.nanmedian(arm(a, "dist5m"))}
    try:
        rec |= {"R@5_blend0.3": arm(a, "r5", "+blend0.3").mean(), "MIPS_gain_blend0.3": arm(a, "gain5", "+blend0.3").mean()}
    except AssertionError:
        pass
    rows.append(rec)
for nm, col in (("pcp_habit", "pcp"), ("random", "random"), ("mips_sorted", "mips")):
    rows.append({"arm": nm, "R@5": pr[f"{col}|r5"].mean(), "excess_R@5": (pr[f"{col}|r5"] - rand).mean(),
                 "MIPS_gain_top5": pr[f"{col}|gain5"].mean(), "busiest_top1": pr[f"{col}|busy1"].mean(), "new_to_PCP_top5": pr[f"{col}|new_pcp5"].mean(),
                 "median_dist_top5_mi": np.nanmedian(pr[f"{col}|dist5m"]),
                 **({"R@5_blend0.3": pr["pcp+blend0.3|r5"].mean(), "MIPS_gain_blend0.3": pr["pcp+blend0.3|gain5"].mean()} if col == "pcp" else {})})
tab = pd.DataFrame(rows); out["arms"] = tab.to_dict("records")
tab.to_csv(os.path.join(args.dir, f"{args.split}_step11_arms.csv"), index=False)
json.dump(out, open(os.path.join(args.dir, f"{args.split}_step11_triggers.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
for k in ("T12_tabular_vs_graph", "T8_message_passing", "T9_model_class", "descriptive_nograph_vs_tabular", "descriptive_other_lgbm"):
    if k not in out: continue
    r = out[k]; print(f"\n{k}: {r['contrast']}: point {r['point']:+.4f}; PCP-cluster 95% [{r['pcp_95'][0]:+.4f}, {r['pcp_95'][1]:+.4f}], "
                      f"alpha/10 [{r['pcp_holm_bound_10'][0]:+.4f}, {r['pcp_holm_bound_10'][1]:+.4f}]; patient 95% [{r['patient_95'][0]:+.4f}, {r['patient_95'][1]:+.4f}]; "
                      f"band {r['noise_band']:.4f}; outcome: {r['outcome'].upper()}")
