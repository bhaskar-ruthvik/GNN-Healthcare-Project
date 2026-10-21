"""PCP–specialist race concordance (STEP10_PROTOCOL A13). Reads the A13 evaluation pass (per-referral, local only) and writes
AGGREGATE tables + a chart. Groups: PCP race with ≥ 10 PCPs and ≥ 100 patients. PCP-cluster bootstrap.

  concordant share  = share of top picks (or top-5 slots) whose race equals the PCP's race
  availability      = the random-pick expectation (share of eligible specialists who share the PCP's race)
  excess            = concordant share − availability;   change = model excess − practice excess
Usage: step10_concordance.py --split validation|test [--dir runs/analysis/step10_9_concordance]
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--dir", default="runs/analysis/step10_9_concordance")
ap.add_argument("--bootstrap", type=int, default=1000); args = ap.parse_args()
os.chdir(str(workspace()))
pr = pd.read_csv(os.path.join(args.dir, f"{args.split}_per_referral_PRIVATE.csv.gz"))
cf = pd.read_csv(os.path.join(args.dir, f"{args.split}_configs.csv"))[["config", "arm"]]
ARMS = {"PCP’s habit": "pcp:base", "Model’s own ranking": "model_unmasked_v5:base", "Our model + 30% quality weight": "model_unmasked_v5:blend0.3",
        "+ load penalty 0.2": "model_unmasked_v5:pen0.2"}
def arm_vals(arm, m):
    cols = [f"{c}|{m}" for c in cf.loc[cf.arm == arm, "config"] if f"{c}|{m}" in pr]; return pr[cols].mean(axis=1).to_numpy()
avail = pr["random|conc1"].to_numpy()                                  # exact expectation of a random pick (= share of eligible concordant)
prac = pr.ref_conc.astype(float).to_numpy()
known = (pr.pcp_race != "unknown").to_numpy()
pcp_u, pcp_idx = np.unique(pr.pcp.to_numpy(), return_inverse=True); rng = np.random.default_rng(0)
B = np.array([np.bincount(rng.integers(0, len(pcp_u), len(pcp_u)), minlength=len(pcp_u)) for _ in range(args.bootstrap)])
def mean_ci(x, mask):
    s = np.bincount(pcp_idx[mask], weights=x[mask], minlength=len(pcp_u)); c = np.bincount(pcp_idx[mask], minlength=len(pcp_u))
    est = (B @ s) / np.maximum(B @ c, 1); return float(x[mask].mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))

groups = [("all PCPs", known)]
for g, sub in pr[known].groupby("pcp_race"):
    m = (pr.pcp_race == g).to_numpy()
    if pr.loc[m, "pcp"].nunique() >= 10 and pr.loc[m, "patient"].nunique() >= 100: groups.append((g, m))
rows = []
for gname, m in groups:
    av = mean_ci(avail, m); pc = mean_ci(prac, m); pe = mean_ci(prac - avail, m)
    base = {"pcp_race": gname, "n_referrals": int(m.sum()), "n_pcps": int(pr.loc[m, "pcp"].nunique()), "availability": av[0], "practice": pc[0], "practice_ci": pc[1:],
            "practice_excess": pe[0], "practice_excess_ci": pe[1:]}
    rows.append({**base, "config": "Actual referrals", "top": "top pick", "share": pc[0], "share_ci": pc[1:], "excess": pe[0], "excess_ci": pe[1:], "change_vs_practice": 0.0, "change_ci": (0.0, 0.0)})
    for lab, arm in ARMS.items():
        for k, top in (("conc1", "top pick"), ("conc5", "top 5")):
            v = arm_vals(arm, k)
            if v.size == 0 or np.isnan(v).all(): continue
            sh = mean_ci(v, m); ex = mean_ci(v - (avail if k == "conc1" else pr["random|conc5"].to_numpy()), m)
            ch = mean_ci((v - (avail if k == "conc1" else pr["random|conc5"].to_numpy())) - (prac - avail), m)
            rows.append({**base, "config": lab, "top": top, "share": sh[0], "share_ci": sh[1:], "excess": ex[0], "excess_ci": ex[1:], "change_vs_practice": ch[0], "change_ci": ch[1:]})
out = pd.DataFrame(rows); out.to_csv(os.path.join(args.dir, f"{args.split}_concordance.csv"), index=False)
show = out[out.top.isin(["top pick"]) | (out.config == "Actual referrals")]
pd.set_option("display.width", 220)
print(show[["pcp_race", "n_pcps", "n_referrals", "config", "availability", "share", "excess", "change_vs_practice", "change_ci"]].round(3).to_string(index=False))

# chart: per PCP race, availability vs actual vs our recommendations (top pick)
INK2, GRID = "#44464f", "#e3e2ea"; COL = {"availability": "#9aa0a6", "Actual referrals": "#1b1b1f", "Model’s own ranking": "#8ab4f8", "Our model + 30% quality weight": "#2a78d6", "+ load penalty 0.2": "#1baf7a"}
fam = "Helvetica Neue" if any("Helvetica Neue" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})
gl = [g for g, _ in groups]; fig, ax = plt.subplots(figsize=(9, 0.55 * len(gl) + 1.6)); yy = np.arange(len(gl))[::-1]
for i, g in enumerate(gl):
    d = out[(out.pcp_race == g)]; av = d.availability.iloc[0]
    ax.plot([av], [yy[i]], "|", color=COL["availability"], markersize=18, markeredgewidth=3, label="availability (random pick)" if i == 0 else None)
    for j, lab in enumerate(["Actual referrals", "Model’s own ranking", "Our model + 30% quality weight", "+ load penalty 0.2"]):
        r = d[(d.config == lab) & ((d.top == "top pick"))]
        if len(r): ax.plot(r.share, yy[i] + (j - 1.5) * 0.12, "o", color=COL[lab], markersize=7, markeredgecolor="white", label=lab if i == 0 else None)
ax.set_yticks(yy); ax.set_yticklabels([f"{g} PCPs" for g in gl]); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Share of top picks whose race matches the PCP’s"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False, fontsize=9)
ax.set_title(f"Same-race PCP–specialist matching — {args.split}", loc="left", fontweight="bold")
fig.savefig(os.path.join(args.dir, f"concordance_{args.split}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
