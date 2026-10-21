"""Step 10.5 bias analysis (STEP10_PROTOCOL A7). Reads the 10.5 evaluation outputs (per-referral, local only) and writes
AGGREGATE tables and charts: groups with < 100 patients are suppressed; no identifiers leave this script.

  <split>_group_table.csv          per attribute × group: practice and configuration means
  <split>_gap_changes.csv          gap vs the reference group (largest), in practice and in recommendations, change in gap + CI
  <split>_masking_attribution.csv  v4 unmasked − masked change-in-gap per seed, band, seed gap, attribution verdict
  bias_<split>.png                 change-in-gap forest plot for the final configuration
Usage: step10_bias.py --split validation|test [--dir runs/analysis/step10_5]
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--dir", default="runs/analysis/step10_5")
ap.add_argument("--bootstrap", type=int, default=1000); args = ap.parse_args()
root = str(workspace()); os.chdir(root)
MIN_PATIENTS = 100
P = "prepared_data/referral_prepared_20260918T040320_197341Z/"

pr = pd.read_csv(os.path.join(args.dir, f"{args.split}_per_referral_PRIVATE.csv.gz"))
cf = pd.read_csv(os.path.join(args.dir, f"{args.split}_configs.csv"))[["config", "arm", "run", "family", "variant"]]
summ = json.load(open(os.path.join(args.dir, f"{args.split}_summary.json")))
mi = load_three_node_inputs(os.path.join(root, summ["inputs"][0]))

# ---- patient attributes at the referral encounter ---------------------------------------------------------------
pc = pd.read_pickle(P + "patient_context_PRIVATE.pkl").set_index("encounter_id")
enc_id = mi.encounter_ids[pr.enc.to_numpy()]; pid = mi.patient_ids[pr.patient.to_numpy()]
ctx = pc.reindex(enc_id)
race = pd.read_pickle("prepared_data/three_node_v4/patient_race_PRIVATE.pkl").set_index("mso_membership_id").race
zips = pd.read_pickle("prepared_data/three_node_v2/patient_zip_PRIVATE.pkl").set_index("mso_membership_id").zip5
age = pd.to_numeric(ctx.age, errors="coerce").to_numpy()
attrs = pd.DataFrame({
    "gender": ctx.gender.fillna("unknown").astype(str).to_numpy(),
    "race": race.reindex(pid).fillna("unknown").astype(str).to_numpy(),
    "age_band": pd.cut(age, [-1, 64.999, 74.999, 84.999, 200], labels=["<65", "65–74", "75–84", "≥85"]).astype(str),
    "insurer": ctx.insurance_company.fillna("unknown").astype(str).to_numpy(),
    "area_zip3": zips.reindex(pid).astype("string").str[:3].fillna("unknown").to_numpy()})
attrs.loc[attrs.age_band == "nan", "age_band"] = "unknown"

# ---- per-referral quantities: practice and seed-averaged arms ------------------------------------------------------
M = ["gain5", "gain5_cons", "within1_1", "dist5m", "busy1", "busy5", "new_pcp5", "r5"]
def arm_mean(arm, m):
    cols = [f"{c}|{m}" for c in cf.loc[cf.arm == arm, "config"]]; return pr[cols].mean(axis=1).to_numpy()
def run_val(run_suffix, variant, m):
    c = cf[(cf.run.fillna("").str.endswith(run_suffix)) & (cf.variant == variant)].config.iloc[0]; return pr[f"{c}|{m}"].to_numpy()
ARMS = {"final (v5, blend 0.3)": "model_unmasked_v5:blend0.3", "v5 own list": "model_unmasked_v5:base",
        "v5 without gender/race, blend 0.3": "model_masked_v5:blend0.3", "PCP habit": "pcp:base", "random": "random:base"}
ARMS = {a: arm for a, arm in ARMS.items() if arm in set(cf.arm)}            # A8: passes with only the final runs
q = {a: {m: arm_mean(arm, m) for m in M} for a, arm in ARMS.items()}
prac = {"mips": pr.ref_mips.to_numpy(), "dist": pr.ref_dist.to_numpy(), "busy": pr.ref_busy.astype(float).to_numpy(), "new_pcp": pr.ref_new_pcp.astype(float).to_numpy()}
# (label, practice quantity, recommended quantity as a function of arm dict) — change in gap = gap(recommended) − gap(practice)
PAIRS = [("MIPS (points)", prac["mips"], lambda d: prac["mips"] + d["gain5"]),
         ("MIPS, conservative (points)", prac["mips"], lambda d: prac["mips"] + d["gain5_cons"]),
         ("distance (mi)", prac["dist"], lambda d: d["dist5m"]),
         ("busiest-10% share, top-1", prac["busy"], lambda d: d["busy1"]),
         ("busiest-10% share, top-5", prac["busy"], lambda d: d["busy5"]),
         ("new-to-PCP share, top-5", prac["new_pcp"], lambda d: d["new_pcp5"])]

pat_u, pat_idx = np.unique(pr.patient.to_numpy(), return_inverse=True); NP = len(pat_u)
rng = np.random.default_rng(0); B = np.array([np.bincount(rng.integers(0, NP, NP), minlength=NP) for _ in range(args.bootstrap)])
def gmean_boot(x, mask):
    ok = mask & ~np.isnan(x); s = np.bincount(pat_idx[ok], weights=x[ok], minlength=NP); c = np.bincount(pat_idx[ok], minlength=NP)
    return s[c > 0].sum() / max(c.sum(), 1), (B @ s) / np.maximum(B @ c, 1)

tables, gaps = [], []
for att in attrs.columns:
    g_pat = pd.Series(pr.patient.to_numpy()).groupby(attrs[att].to_numpy()).nunique()
    keep = g_pat[g_pat >= MIN_PATIENTS].index.tolist(); ref = g_pat.loc[keep].idxmax()
    for g in keep:
        m = (attrs[att] == g).to_numpy()
        row = {"attribute": att, "group": g, "reference": g == ref, "n_referrals": int(m.sum()), "n_patients": int(g_pat[g]),
               "practice_mips": float(np.nanmean(prac["mips"][m])), "practice_dist": float(np.nanmean(prac["dist"][m])),
               "practice_busy": float(prac["busy"][m].mean()), "practice_new_pcp": float(prac["new_pcp"][m].mean())}
        for a, d in q.items():
            for mm in M: row[f"{a}|{mm}"] = float(np.nanmean(d[mm][m]))
        tables.append(row)
        if g == ref: continue
        mr = (attrs[att] == ref).to_numpy()
        for lab, pq, rq in PAIRS:
            p_g, pb_g = gmean_boot(pq, m); p_r, pb_r = gmean_boot(pq, mr)
            for a, d in q.items():
                rqv = rq(d); r_g, rb_g = gmean_boot(rqv, m); r_r, rb_r = gmean_boot(rqv, mr)
                chg = (r_g - r_r) - (p_g - p_r); chb = (rb_g - rb_r) - (pb_g - pb_r)
                gaps.append({"attribute": att, "group": g, "reference": ref, "quantity": lab, "config": a, "practice_gap": p_g - p_r,
                             "recommended_gap": r_g - r_r, "change_in_gap": chg, "lo": float(np.percentile(chb, 2.5)), "hi": float(np.percentile(chb, 97.5))})
tab = pd.DataFrame(tables); gp = pd.DataFrame(gaps)

# ---- masking attribution (v4 pairs, per seed; band from v5 dup) -------------------------------------------------------
def change_in_gap_point(vals, att, g, ref, pq):
    m = (attrs[att] == g).to_numpy(); mr = (attrs[att] == ref).to_numpy()
    return (np.nanmean(vals[m]) - np.nanmean(vals[mr])) - (np.nanmean(pq[m]) - np.nanmean(pq[mr]))
att_rows = []
need = ("_v4_unmasked_s42", "_v4_unmasked_s1", "_v4_masked_s42", "_v4_masked_s1", "_v5_unmasked_s42", "_v5_unmasked_s42_dup")
have_all = all(cf.run.fillna("").str.endswith(sfx).any() for sfx in need)
for variant in (("blend0.3", "base") if have_all else ()):
    for (att, g, ref), _ in gp[gp.config == "final (v5, blend 0.3)"].groupby(["attribute", "group", "reference"]):
        for lab, pq, rq in PAIRS:
            cg = {}
            for suf in ("_v4_unmasked_s42", "_v4_unmasked_s1", "_v4_masked_s42", "_v4_masked_s1", "_v5_unmasked_s42", "_v5_unmasked_s42_dup"):
                d = {mm: run_val(suf, variant, mm) for mm in ("gain5", "gain5_cons", "dist5m", "busy1", "busy5", "new_pcp5")}
                cg[suf] = change_in_gap_point(rq(d), att, g, ref, pq)
            d42 = cg["_v4_unmasked_s42"] - cg["_v4_masked_s42"]; d1 = cg["_v4_unmasked_s1"] - cg["_v4_masked_s1"]
            band = abs(cg["_v5_unmasked_s42"] - cg["_v5_unmasked_s42_dup"])
            gap = max(abs(cg["_v4_unmasked_s42"] - cg["_v4_unmasked_s1"]), abs(cg["_v4_masked_s42"] - cg["_v4_masked_s1"]))
            attributed = bool(np.sign(d42) == np.sign(d1) and min(abs(d42), abs(d1)) > max(band, gap))
            att_rows.append({"variant": variant, "attribute": att, "group": g, "reference": ref, "quantity": lab, "unmasked_minus_masked_s42": d42,
                             "unmasked_minus_masked_s1": d1, "band": band, "seed_gap": gap, "attributed_to_masking": attributed})
at = pd.DataFrame(att_rows, columns=["variant", "attribute", "group", "reference", "quantity", "unmasked_minus_masked_s42", "unmasked_minus_masked_s1", "band", "seed_gap", "attributed_to_masking"])

pfx = os.path.join(args.dir, args.split)
tab.to_csv(pfx + "_group_table.csv", index=False); gp.to_csv(pfx + "_gap_changes.csv", index=False); at.to_csv(pfx + "_masking_attribution.csv", index=False)

# ---- chart: change in gap for the final configuration ------------------------------------------------------------------
INK, INK2, GRID, BLUE = "#0b0b0b", "#52514e", "#e6e6e3", "#2a78d6"
fam = "Helvetica Neue" if any("Helvetica Neue" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "figure.facecolor": "white",
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left"})
sel = gp[gp.config == "final (v5, blend 0.3)"]
quants = ["MIPS (points)", "distance (mi)", "busiest-10% share, top-1", "new-to-PCP share, top-5"]
labels = sel.drop_duplicates(["attribute", "group"])[["attribute", "group", "reference"]].reset_index(drop=True)
fig, axes = plt.subplots(1, len(quants), figsize=(15, 0.28 * len(labels) + 1.6), sharey=True)
yy = np.arange(len(labels))[::-1]
for ax, qn in zip(axes, quants):
    d = sel[sel.quantity == qn].set_index(["attribute", "group"]).reindex(list(zip(labels.attribute, labels.group)))
    ax.axvline(0, color=INK2, linewidth=1)
    ax.hlines(yy, d.lo, d.hi, color=BLUE, linewidth=2); ax.plot(d.change_in_gap, yy, "o", color=BLUE, markersize=5, markeredgecolor="white")
    ax.set_title(qn); ax.grid(axis="y", visible=False)
axes[0].set_yticks(yy); axes[0].set_yticklabels([f"{a}: {g} (vs {r})" for a, g, r in zip(labels.attribute, labels.group, labels.reference)], fontsize=8.5)
fig.suptitle(f"Change in each group's gap vs practice — final model, {args.split}", x=0.01, ha="left", fontweight="bold", fontsize=13)
fig.canvas.draw(); y0 = min(a_.get_tightbbox(fig.canvas.get_renderer()).y0 for a_ in fig.axes) / fig.bbox.height
fig.text(0.01, y0 - 0.02, "Change in gap = (group − reference) in recommendations minus (group − reference) in actual referrals; > 0 widens a positive gap or narrows a negative one. "
         f"Groups with < {MIN_PATIENTS} patients suppressed. Race is imputed in the source. 95% patient-cluster CIs.", fontsize=8, color=INK2, va="top")
fig.savefig(os.path.join(args.dir, f"bias_{args.split}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"{args.split}: {len(labels)} groups shown; attributed-to-masking rows: {int(at.attributed_to_masking.sum())} of {len(at)}")
