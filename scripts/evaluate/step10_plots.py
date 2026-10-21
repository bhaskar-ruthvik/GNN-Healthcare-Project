"""Step 10.2 charts from <split>_arms.csv (seed-averaged configurations). Aggregate only.
  frontier_<split>.png  fidelity (R@5 of the delivered list) vs MIPS gain of the top-5 (point estimate and conservative bound)
  load_<split>.png      MIPS change of the top-1 vs share of top-1 recommendations to specialists above their specialty's training p90
Model shown: the final model (with patient gender and race, A3); the supplementary arm without them is within noise and listed in the CSV.
Usage: step10_plots.py --split validation
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="validation"); ap.add_argument("--dir", default="runs/analysis/step10_2")
ap.add_argument("--model-family", default="model_unmasked", help="arm family drawn as the final model (10.3: e.g. model_unmasked_v5)"); args = ap.parse_args()
os.chdir(str(workspace()))
MF = args.model_family
a = pd.read_csv(os.path.join(args.dir, f"{args.split}_arms.csv")); s = json.load(open(os.path.join(args.dir, f"{args.split}_summary.json")))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
COL = {MF: "#2a78d6", "pcp": "#eb6834", "dist": "#1baf7a"}
LAB = {MF: "Final model", "pcp": "PCP's referral habit", "dist": "Nearest specialist first"}
fam = "Helvetica Neue" if any("Helvetica Neue" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "white", "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
                     "legend.frameon": False})
tag = "validation (2018)" if args.split == "validation" else "test (2019; year inspected once before)"

def order(df):
    k = df.variant.map(lambda v: 0 if v == "base" else (1 + [0.1, 0.2, 0.3, 0.5].index(float(v[5:])) if v.startswith("blend") else 10 + (999 if v == "resortall" else int(v[6:]))))
    return df.assign(_k=k).sort_values("_k")

def note(fig, text):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); y0 = min(ax.get_tightbbox(r).y0 for ax in fig.axes) / fig.bbox.height
    fig.text(0.01, y0 - 0.02, text, fontsize=8.5, color=INK2, va="top")

# ---- frontier ----------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
pcp_r5 = s["pcp_r5_floor"]
for ax, metric, title in [(axes[0], "gain5", "MIPS gain of the 5 recommended (point estimate)"), (axes[1], "gain5_cons", "MIPS gain, conservative bound")]:
    ax.axvspan(pcp_r5, 0.52, color="#f2f2ef", zorder=0); ax.axhline(0, color=INK2, linewidth=1)
    for f in [MF, "pcp", "dist"]:
        for kind, ls, mk in [("resort", "-", "o"), ("blend", "--", "s")]:
            d = order(a[(a.family == f) & (a.variant.str.startswith(kind) | (a.variant == "base"))])
            if kind == "blend" and f == "dist": continue
            ax.plot(d.r5, d[metric], ls, color=COL[f], linewidth=2, marker=mk, markersize=5, markeredgecolor="white", markeredgewidth=1,
                    label=f"{LAB[f]} — {'re-sort top K by MIPS' if kind == 'resort' else 'blend with MIPS'}" if ax is axes[0] else None)
        b = a[(a.family == f) & (a.variant == "base")].iloc[0]; ax.plot(b.r5, b[metric], "o", color=COL[f], markersize=10, markeredgecolor=INK, markeredgewidth=1.2)
    for f, mk, lab in [("mips", "D", "Highest MIPS first"), ("random", "x", "Random")]:
        b = a[a.family == f].iloc[0]; ax.plot(b.r5, b[metric], mk, color=INK, markersize=8, label=lab if ax is axes[0] else None)
        ax.annotate(lab, (b.r5, b[metric]), textcoords="offset points", xytext=(-8, -4), ha="right", fontsize=9, color=INK2)
    ax.set_title(title); ax.set_xlabel("Referred specialist is in the 5 recommended (R@5)"); ax.set_ylabel("Points vs the referred specialist")
    ax.text(pcp_r5 + 0.003, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 0, "at least as faithful\nas the PCP's habit", fontsize=8.5, color=INK2, va="top")
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=2, fontsize=9)
fig.suptitle(f"Trading fidelity for MIPS — {tag}", x=0.01, ha="left", fontweight="bold", fontsize=14)
note(fig, "Large outlined dots: each scorer's own list. Lines move from the own list toward sorting everything by MIPS. Conservative bound: recommended at their lowest score and no-score = 0, referred at their highest.")
fig.savefig(os.path.join(args.dir, f"frontier_{args.split}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- MIPS change vs load (amendment A1: mean change vs busiest share; median change vs congestion) ----------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, xcol, ycol, xlab, ylab, prac_x in [
        (axes[0], "busy1", "gain1", "Top recommendation goes to a specialty's busiest 10% (training years)", "Mean MIPS change of the top recommendation (points)", s["practice"]["busy1"]),
        (axes[1], "congestion_median", "gain1_median", "Median congestion ratio (share of recommendations ÷ share of training referrals)", "Median MIPS change of the top recommendation (points)", s["practice"]["congestion_median"])]:
    ax.axvline(prac_x, color=INK2, linewidth=1, linestyle="--"); ax.axhline(0, color=INK2, linewidth=1)
    for f in [MF, "pcp", "dist"]:
        for kind, ls in [("resort", "-"), ("blend", "--")]:
            if kind == "blend" and f == "dist": continue
            d = order(a[(a.family == f) & (a.variant.str.startswith(kind) | (a.variant == "base"))])
            ax.plot(d[xcol], d[ycol], ls, color=COL[f], linewidth=2, marker="o", markersize=5, markeredgecolor="white",
                    label=f"{LAB[f]} ({'re-sort' if kind == 'resort' else 'blend'})" if ax is axes[0] else None)
        b_ = a[(a.family == f) & (a.variant == "base")].iloc[0]; ax.plot(b_[xcol], b_[ycol], "o", color=COL[f], markersize=10, markeredgecolor=INK, markeredgewidth=1.2)
    for f, mk, lab in [("mips", "D", "Highest MIPS first"), ("random", "x", "Random")]:
        b_ = a[a.family == f].iloc[0]; ax.plot(b_[xcol], b_[ycol], mk, color=INK, markersize=8); ax.annotate(lab, (b_[xcol], b_[ycol]), textcoords="offset points", xytext=(6, 4), fontsize=9, color=INK2)
    ax.text(prac_x, 1.01, "current practice", transform=ax.get_xaxis_transform(), color=INK2, fontsize=9, ha="center", va="bottom")
    ax.set_xlabel(xlab, fontsize=10); ax.set_ylabel(ylab, fontsize=10)
axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.16), ncol=3, fontsize=9)
fig.suptitle(f"MIPS change vs load — {tag}", x=0.01, ha="left", fontweight="bold", fontsize=14)
note(fig, "No scorer uses load (amendment A1); load is measured only. A top-1 list is each scorer's single best guess, so history-based lists concentrate more than realised referrals.")
fig.savefig(os.path.join(args.dir, f"load_{args.split}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- per-referral MIPS gain of the top-5 (histogram) ----------------------------------------------------------------------
h = pd.read_csv(os.path.join(args.dir, f"{args.split}_gain_hist.csv"))
cfgs = pd.read_csv(os.path.join(args.dir, f"{args.split}_configs.csv"))[["config", "arm"]]
h = h.merge(cfgs, on="config"); h = h.groupby(["arm", "bin_lo"], as_index=False)["count"].mean()
fig, ax = plt.subplots(figsize=(8, 4.4))
show = [("random:base", INK2, "Random"), ("pcp:base", COL["pcp"], LAB["pcp"]), (MF + ":base", COL[MF], LAB[MF] + ", own list"),
        (MF + ":blend0.3", "#1baf7a", LAB[MF] + ", blend 0.3")]
for arm, colr, lab in show:
    d = h[h.arm == arm].sort_values("bin_lo"); tot = d["count"].sum()
    ax.step(np.r_[d.bin_lo, d.bin_lo.iloc[-1] + 5], np.r_[d["count"] / tot, (d["count"] / tot).iloc[-1]], where="post", color=colr, linewidth=2, label=lab)
ax.axvline(0, color=INK2, linewidth=1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlim(-60, 100); ax.set_xlabel("Mean MIPS of the 5 recommended − referred specialist's MIPS (points)"); ax.set_ylabel("Share of referrals")
ax.set_title(f"How much better or worse each referral's list is — {tag}"); ax.legend(fontsize=9)
note(fig, "Point estimate (no-score specialists at their specialty mean). Model rows averaged over two seeds.")
fig.savefig(os.path.join(args.dir, f"gain_hist_{args.split}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("saved", args.dir)
