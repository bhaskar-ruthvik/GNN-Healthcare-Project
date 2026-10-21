"""Step 10.1 descriptive charts (STEP10_PROTOCOL §4 step 2). Aggregate only; training + validation years (2015–2018); the test year is not used.

Charts (PNG, runs/analysis/step10_1/): binned summaries only; specialties with < 11 specialists suppressed; no provider identifiers.
  01 MIPS distribution of specialists          06 MIPS by referral-load group within specialty
  02 MIPS by specialty (median, IQR)           07 distance from the patient: referred vs nearest eligible
  03 referred MIPS vs what was available       08 excess distance over the nearest eligible specialist
  04 headroom within 25 miles                  09 no-MIPS-score specialists: share of options vs share of referrals, by year
  05 load concentration within specialty       10 spread of multi-valued MIPS
Also writes summary.json (funnel counts and the numbers quoted on the charts).
Usage: step10_descriptive.py [--inputs prepared_data/three_node_v4/model_inputs_PRIVATE.pkl]
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
from referral_gnn.step10 import Step10Cohort, load_yardstick, specialty_names, NON_CHOICE, NON_SPECIFIC  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--inputs", default="prepared_data/three_node_v4/model_inputs_PRIVATE.pkl")
ap.add_argument("--out", default="runs/analysis/step10_1"); args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
MIN_SPEC = 11

# ---- style: white, Helvetica Neue, thin marks, recessive grid --------------------------------
INK, INK2, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e6e6e3", "#2a78d6", "#eb6834"
fam = "Helvetica Neue" if any("Helvetica Neue" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white", "axes.titleweight": "bold",
                     "axes.titlesize": 14, "axes.titlelocation": "left", "legend.frameon": False})

def save(fig, name, note=None):
    if note:                                   # footnote placed below everything the axes draw (tick labels, axis label)
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        y0 = min(a.get_tightbbox(r).y0 for a in fig.axes) / fig.bbox.height
        fig.text(0.01, y0 - 0.015, note, fontsize=8.5, color=INK2, ha="left", va="top")
    fig.savefig(os.path.join(args.out, name), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- data -------------------------------------------------------------------------------------
mi = load_three_node_inputs(args.inputs); y = load_yardstick(mi); names = specialty_names(mi)
rows, funnel, excl_days = [], {}, {}
BINS = np.arange(0, 102.5, 5); opt_hist = np.zeros(len(BINS) - 1)
for split in ("train", "validation"):
    c = Step10Cohort(mi, split, y)
    for r in c.referrals():
        ev = y.point(r.eligible); rv = y.value[r.target]
        opt_hist += np.histogram(ev, bins=BINS)[0] / len(ev)
        rows.append({"split": split, "day": r.day, "year": int(str(np.datetime64(r.day, "D"))[:4]), "patient": r.patient, "target": r.target,
                     "specialty": names[r.specialty], "ref_mips": y.value[r.target], "elig_mean": float(ev.mean()), "elig_median": float(np.median(ev)),
                     "elig_best": float(ev.max()), "share_elig_above": float((ev > rv).mean()), "share_elig_equal": float((ev == rv).mean()), "n_elig": len(r.eligible),
                     "filled": r.filled, "ref_dist": r.target_dist, "nearest_dist": float(np.nanmin(r.pre_cap_dist)),
                     "none_share_options": float(y.none[r.pre_cap].mean())})
    funnel[split] = dict(c.counts)
    excl_days[split] = {k: mi.enc_day[np.array(v, dtype=np.int64)] for k, v in c.excluded.items()}
df = pd.DataFrame(rows)
print("primary first-time referrals (train+validation):", len(df))
summary = {"scope": "training + validation years; primary cohort per STEP10_PROTOCOL §2.1", "funnel": funnel, "n_primary": int(len(df))}

# specialist-level frame over specialists that appeared in any eligible set (zero-load included)
spec_rows = []
spec_of = {}
for split in ("train", "validation"):
    c = Step10Cohort(mi, split, y)
    for r in c.referrals():
        for s in r.eligible: spec_of.setdefault(int(s), names[r.specialty])
load = df.groupby("target").size()
sp = pd.DataFrame({"specialty": pd.Series(spec_of)}); sp["load"] = load.reindex(sp.index).fillna(0).astype(int)
sp["mips"] = y.value[sp.index.to_numpy()]; sp["none"] = y.none[sp.index.to_numpy()]
big = sp.groupby("specialty").size(); big = big[big >= MIN_SPEC].index
vol = df.groupby("specialty").size()

# 01 MIPS distribution of specialists in choice specialties -------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))
v = sp.loc[~sp.none, "mips"]
ax.hist(v, bins=np.arange(0, 102.5, 5), color=BLUE, edgecolor="white", linewidth=2)
ax.set_xlabel("MIPS score (mean of the specialist's distinct positive scores)"); ax.set_ylabel("Specialists")
ax.set_title("Specialist MIPS scores are bunched at the top")
at100 = float((v >= 99.999).mean()); med = float(v.median())
ax.text(0.02, 0.95, f"Median {med:.0f}  ·  {at100:.0%} score exactly 100\nNo MIPS score (0 in source): {int(sp.none.sum())} specialists ({sp.none.mean():.1%})",
        transform=ax.transAxes, va="top", color=INK, fontsize=10)
save(fig, "01_mips_distribution.png", f"Specialists who were an eligible option for at least one first-time referral, 2015–2018 (n = {len(sp):,}).")
summary["01"] = {"n_specialists": int(len(sp)), "median": med, "share_100": at100, "none": int(sp.none.sum())}

# 02 MIPS by specialty ---------------------------------------------------------------------------------
g = sp[~sp.none & sp.specialty.isin(big)].groupby("specialty").mips
st = pd.DataFrame({"q25": g.quantile(.25), "med": g.median(), "q75": g.quantile(.75), "n": g.size()}).sort_values("med")
st = st[st.index.isin(vol.sort_values(ascending=False).index[:25])]
fig, ax = plt.subplots(figsize=(8, 0.32 * len(st) + 1.2))
yy = np.arange(len(st))
ax.hlines(yy, st.q25, st.q75, color=BLUE, linewidth=2); ax.plot(st.med, yy, "o", color=BLUE, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
ax.set_yticks(yy); ax.set_yticklabels([f"{s}  (n={n})" for s, n in zip(st.index, st.n)], fontsize=9); ax.grid(axis="y", visible=False)
ax.set_xlim(0, 101); ax.set_xlabel("MIPS score: median (dot) and middle 50% (line)")
ax.set_title("MIPS by specialty")
save(fig, "02_mips_by_specialty.png", "The 25 choice specialties with the most first-time referrals and ≥ 11 specialists; specialists with a MIPS score, 2015–2018.")

# 03 referred MIPS vs available ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2)); bins = BINS
ref_h = np.histogram(df.ref_mips, bins=bins)[0] / len(df); opt_h = opt_hist / len(df)
ax.bar(bins[:-1] + 2.5, ref_h, width=5, color=BLUE, edgecolor="white", linewidth=2, label="Specialist actually referred to")
ax.step(np.r_[bins[:-1], 100], np.r_[opt_h, opt_h[-1]], where="post", color=ORANGE, linewidth=2, label="A randomly picked eligible specialist (same referral)")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("MIPS score"); ax.set_ylabel("Share of referrals"); ax.legend(loc="upper left")
below = float((df.ref_mips < df.elig_median).mean()); gap = float((df.ref_mips - df.elig_mean).mean())
p_up, p_eq = float(df.share_elig_above.mean()), float(df.share_elig_equal.mean())
ax.set_title("Referred specialists' MIPS vs the options available")
ax.text(0.02, 0.72, f"A random eligible option scores higher than the referred specialist {p_up:.0%} of the time,\nthe same {p_eq:.0%}, lower {1 - p_up - p_eq:.0%}. Referred − average option: {gap:+.1f} points.",
        transform=ax.transAxes, color=INK, fontsize=10, va="top")
save(fig, "03_referred_vs_available.png", f"First-time referrals, choice specialties, 2015–2018 (n = {len(df):,}). Eligible = same specialty, active in the past year, new to the patient, ≤ 25 mi.")
summary["03"] = {"ref_mean": float(df.ref_mips.mean()), "elig_mean": float(df.elig_mean.mean()), "ref_below_median_option": below, "ref_minus_avg": gap,
                 "random_option_higher": p_up, "random_option_equal": p_eq}

# 04 headroom -------------------------------------------------------------------------------------------------------
h = df.elig_best - df.ref_mips
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(h.clip(lower=0), bins=np.arange(0, 102.5, 5), color=BLUE, edgecolor="white", linewidth=2)
none_room = float((h <= 0).mean())
ax.set_xlabel("Best eligible MIPS within 25 mi − referred specialist's MIPS (points)"); ax.set_ylabel("Referrals")
ax.set_title("How much higher-MIPS room each referral had")
ax.text(0.45, 0.93, f"No higher-MIPS option: {none_room:.0%} of referrals\nMedian room where there was some: {h[h > 0].median():.0f} points",
        transform=ax.transAxes, color=INK, fontsize=10, va="top")
save(fig, "04_headroom.png", "Headroom is an upper bound: the single best option, ignoring fit beyond specialty, PCP relationships and load.")
summary["04"] = {"no_headroom": none_room, "median_headroom_if_any": float(h[h > 0].median()), "share_above_options_mean": float(df.share_elig_above.mean())}

# 05 load concentration within specialty ------------------------------------------------------------------------------
def top10_share(s):
    s = np.sort(s.to_numpy())[::-1]; k = max(1, int(np.ceil(0.1 * len(s)))); return s[:k].sum() / max(s.sum(), 1)
def gini(s):
    x = np.sort(s.to_numpy().astype(float)); n = len(x); return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else np.nan
lc = sp[sp.specialty.isin(big)].groupby("specialty").load.agg([top10_share, gini, "size", "sum"])
lc = lc[lc.index.isin(vol.sort_values(ascending=False).index[:25])].sort_values("top10_share")
fig, ax = plt.subplots(figsize=(8, 0.32 * len(lc) + 1.2)); yy = np.arange(len(lc))
ax.barh(yy, lc.top10_share, color=BLUE, height=0.7, edgecolor="white", linewidth=2)
ax.axvline(0.10, color=INK2, linewidth=1, linestyle="--"); ax.text(0.105, len(lc) - 0.4, "equal spread (10%)", color=INK2, fontsize=8.5)
ax.set_yticks(yy); ax.set_yticklabels(lc.index, fontsize=9); ax.grid(axis="y", visible=False); ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Share of the specialty's first-time referrals received by its busiest 10% of specialists")
ax.set_title("Referral load is concentrated on a few specialists")
w = float(np.average(lc.top10_share, weights=lc["sum"]))
save(fig, "05_load_concentration.png", f"Specialists that were an eligible option at least once (zero-load included), 2015–2018. Referral-weighted average across these specialties: {w:.0%}.")
summary["05"] = {"weighted_top10_share": w, "weighted_gini": float(np.average(lc.gini, weights=lc["sum"])), "by_specialty": lc.round(3).to_dict("index")}

# 06 MIPS by load group within specialty -------------------------------------------------------------------------------------
s6 = sp[sp.specialty.isin(big) & ~sp.none].copy()
def grp(g):
    out = pd.Series("No referrals", index=g.index); pos = g.load > 0
    if pos.sum() >= 3: out[pos] = pd.qcut(g.load[pos].rank(method="first"), 3, labels=["Low", "Middle", "High"]).astype(str)
    return out
s6["group"] = s6.groupby("specialty", group_keys=False).apply(grp)
order = ["No referrals", "Low", "Middle", "High"]; rng = np.random.default_rng(0)
stats = []
for gname in order:
    m = s6.loc[s6.group == gname, "mips"].to_numpy(); b = [rng.choice(m, len(m)).mean() for _ in range(1000)]
    stats.append((gname, m.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(m)))
fig, ax = plt.subplots(figsize=(7, 4)); xx = np.arange(len(order))
ax.errorbar(xx, [s[1] for s in stats], yerr=[[s[1] - s[2] for s in stats], [s[3] - s[1] for s in stats]], fmt="o", color=BLUE, markersize=8,
            markeredgecolor="white", markeredgewidth=1.5, capsize=0, elinewidth=2)
for x_, s in zip(xx, stats): ax.text(x_ + 0.08, s[1], f"{s[1]:.1f}", color=INK, fontsize=10, va="center")
ax.set_xticks(xx); ax.set_xticklabels([f"{s[0]}\n(n={s[4]:,})" for s in stats]); ax.set_ylabel("Mean MIPS (95% CI)")
ax.set_xlabel("Referral load within own specialty (thirds of specialists with ≥ 1 first-time referral)")
ax.set_title("Do busier specialists have higher MIPS?")
save(fig, "06_mips_by_load.png", "Specialists with a MIPS score in specialties with ≥ 11 specialists, 2015–2018. Load groups are formed within each specialty.")
summary["06"] = {s[0]: {"mean": s[1], "ci": [s[2], s[3]], "n": s[4]} for s in stats}

# 07 distance: referred vs nearest ------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2)); bins = np.arange(0, 62, 2)
ax.hist(df.ref_dist.clip(upper=60), bins=bins, color=BLUE, edgecolor="white", linewidth=2, label="Specialist actually referred to")
ax.hist(df.nearest_dist.clip(upper=60), bins=bins, histtype="step", color=ORANGE, linewidth=2, label="Nearest eligible specialist")
ax.set_xlabel("Distance from the patient's ZIP (miles; 60 = 60 or more)"); ax.set_ylabel("Referrals"); ax.legend()
ax.set_title("How far patients travel, and how close the nearest option was")
ax.text(0.45, 0.62, f"Median: referred {df.ref_dist.median():.1f} mi, nearest {df.nearest_dist.median():.1f} mi\nReferred within 25 mi: {(df.ref_dist <= 25).mean():.0%}",
        transform=ax.transAxes, color=INK, fontsize=10, va="top")
save(fig, "07_distance.png", "Distances between ZIP centroids. Nearest eligible = same specialty, active in the past year, new to the patient (no cap).")
summary["07"] = {"ref_median": float(df.ref_dist.median()), "ref_p90": float(df.ref_dist.quantile(.9)), "nearest_median": float(df.nearest_dist.median()), "ref_within_25": float((df.ref_dist <= 25).mean())}

# 08 excess over nearest ----------------------------------------------------------------------------------------------------------------
ex = (df.ref_dist - df.nearest_dist)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(ex.clip(upper=40), bins=np.arange(0, 42, 2), color=BLUE, edgecolor="white", linewidth=2)
ax.set_xlabel("Referred specialist's distance − nearest eligible specialist's distance (miles; 40 = 40 or more)"); ax.set_ylabel("Referrals")
ax.set_title("How much farther than necessary patients were sent")
ax.text(0.45, 0.93, f"Went to the nearest option (≤ 1 mi extra): {(ex <= 1).mean():.0%}\nMedian extra: {ex.median():.1f} mi  ·  90th percentile: {ex.quantile(.9):.1f} mi",
        transform=ax.transAxes, color=INK, fontsize=10, va="top")
save(fig, "08_excess_distance.png", "Extra distance is not necessarily a bad choice: the nearest option may not fit the case. It is the reference for recommendations.")
summary["08"] = {"nearest_or_within_1mi": float((ex <= 1).mean()), "median": float(ex.median()), "p90": float(ex.quantile(.9))}

# 09 no-score specialists by year ---------------------------------------------------------------------------------------------------------
yr = lambda days: pd.Series(np.asarray(days, dtype="datetime64[D]").astype(str)).str[:4].astype(int)
n_none = pd.concat([yr(excl_days[s]["referred_mips_none"]) for s in excl_days]).value_counts()
n_pass = df.year.value_counts() + pd.concat([yr(excl_days[s][k]) for s in excl_days for k in ("eligible_lt5", "patient_zip_missing")]).value_counts().reindex(df.year.unique()).fillna(0)
years = sorted(set(df.year))
ref_share = [(n_none.get(Y, 0)) / (n_none.get(Y, 0) + n_pass.get(Y, 0)) for Y in years]
opt_share = [df.loc[df.year == Y, "none_share_options"].mean() for Y in years]
fig, ax = plt.subplots(figsize=(7, 4)); xx = np.arange(len(years)); wdt = 0.36
ax.bar(xx - wdt / 2 - 0.01, opt_share, wdt, color=BLUE, label="Share of eligible options with no MIPS score")
ax.bar(xx + wdt / 2 + 0.01, ref_share, wdt, color=ORANGE, label="Share of referrals that went to them")
ax.set_xticks(xx); ax.set_xticklabels(years); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylim(0, max(max(opt_share), max(ref_share)) * 1.45)
ax.set_title("Specialists with no MIPS score: options vs referrals"); ax.legend(loc="upper left")
save(fig, "09_no_score_by_year.png", "No MIPS score = 0 in the source (the professor's 'new doctor'; recorded tenure does not indicate new). First-time referrals, choice specialties.\nOption share: eligible sets of the primary cohort; referral share: referrals reaching exclusion step 4.")
summary["09"] = {"years": years, "option_share": opt_share, "referral_share": ref_share}

# 10 multi-valued spread ---------------------------------------------------------------------------------------------------------
m = sp[y.multi[sp.index.to_numpy()]]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(y.rng[m.index.to_numpy()], bins=np.arange(0, 102.5, 5), color=BLUE, edgecolor="white", linewidth=2)
ax.set_xlabel("Highest − lowest of the specialist's MIPS scores (points)"); ax.set_ylabel("Specialists")
ax.set_title("Specialists with several MIPS scores: how far apart they are")
r_ = y.rng[m.index.to_numpy()]
ax.text(0.45, 0.93, f"{len(m):,} specialists ({len(m) / len(sp):.0%}) have more than one score\nMedian spread {np.median(r_):.0f} points  ·  90th percentile {np.quantile(r_, .9):.0f}",
        transform=ax.transAxes, color=INK, fontsize=10, va="top")
save(fig, "10_multi_mips_spread.png", "The scores co-occur on the same days and differ by practice size: likely one score per practice the specialist bills under. Averaged in the analysis.")
summary["10"] = {"n_multi": int(len(m)), "median_range": float(np.median(r_)), "p90_range": float(np.quantile(r_, .9))}

json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in summary.items() if k != "05"}, indent=1, default=float))
