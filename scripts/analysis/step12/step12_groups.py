"""STEP12 group deep-dive: why some groups break the "graph model eases congestion" trend (patients 85+, early-tenure PCPs).

Reads the K = 5 evaluation with --strata (runs/analysis/step12_topk/K5/validation_per_referral_PRIVATE.csv.gz; 2018 STEP10 primary
cohort). Seed-averaged arms as in step11_strata.py. Load = busiest-10% share of the top pick; d = GNN − table per referral.
  1. Is the group really different? Difference-in-differences (group's d − rest's d), PCP-cluster bootstrap, Holm across groups.
  2. Patients 85+: availability of busy specialists, specialty mix (direct standardisation to the 85+ specialty mix), familiarity.
  3. PCP history dose-response: d by quintile of the PCP's referrals made before the referral and of the familiar share of the
     eligible set; where the early-tenure PCPs sit on that curve.
  4. One regression of d on age group, PCP history, familiar share, busy availability and specialty (fixed effects), PCP-cluster CIs.
Aggregates only. Output: runs/analysis/step12_groups/groups_summary.json.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os

import numpy as np
import pandas as pd

root = str(workspace()); os.chdir(root)
SRC = "runs/analysis/step12_topk/K5/validation_per_referral_PRIVATE.csv.gz"; OUT = "runs/analysis/step12_groups"; os.makedirs(OUT, exist_ok=True)
pr = pd.read_csv(SRC)
ARMS = {"GNN": ["_v5_unmasked_s42", "_v5_unmasked_s1"], "No-graph": ["_v5_nograph_s42", "_v5_nograph_s1"], "Table": ["_lgbm_tabular_s42", "_lgbm_tabular_s1"]}
def col(sfx, metric, var=""):
    c = [c for c in pr.columns if c.startswith("model:") and c.endswith(f"{var}|{metric}") and ("+" in c) == bool(var) and c.split(":", 1)[1].split("+")[0].split("|")[0].endswith(sfx)]
    assert len(c) == 1, (sfx, metric, var, c); return pr[c[0]].to_numpy(float)
def arm(name, metric, var=""): return np.mean([col(s, metric, var) for s in ARMS[name]], axis=0)
VAR = {"q30": "+blend0.3", "base": ""}
L = {(a, v): {"load": arm(a, "busy1", vv), "load5": arm(a, "busy5", vv), "new": arm(a, "new_pcp5", vv), "mips": arm(a, "gain5", vv)} for a in ARMS for v, vv in VAR.items()}
d = {v: L[("GNN", v)]["load"] - L[("Table", v)]["load"] for v in VAR}

pr["age_g"] = pd.cut(pr.patient_age, [-1, 64, 74, 84, 200], labels=["<65", "65–74", "75–84", "85+"]).astype(str)
pr["pcp_ten_g"] = pd.cut(pr.pcp_tenure, [-1, 6, 13, 99], labels=["early", "mid", "late"]).astype(str)
GROUPS = {"Patient age": "age_g", "PCP Medicare tenure": "pcp_ten_g"}

codes, uniq = pd.factorize(pr.pcp); rng = np.random.default_rng(20261012); B = 2000; IDX = rng.integers(0, len(uniq), size=(B, len(uniq)))
def boot_mean(x, m):
    """PCP-cluster bootstrap draws of mean(x | m)."""
    s1 = np.bincount(codes, weights=np.where(m, x, 0.0), minlength=len(uniq)); s0 = np.bincount(codes, weights=m.astype(float), minlength=len(uniq))
    den = s0[IDX].sum(1)                                  # draws without any of the group's PCPs carry no information → NaN (dropped)
    return np.where(den > 0, s1[IDX].sum(1) / np.where(den > 0, den, 1), np.nan)
def summ(draws, point):
    dr = draws[np.isfinite(draws)]
    return {"est": float(point), "ci": [float(np.quantile(dr, .025)), float(np.quantile(dr, .975))],
            "p": float(min(1.0, 2 * min((dr <= 0).mean(), (dr >= 0).mean()))), "draws_used": int(len(dr))}

out = {"source": SRC, "n": int(len(pr)), "note": "d = GNN − table busiest-10% share of the top pick; seed-averaged arms; PCP-cluster bootstrap"}

# 1. difference-in-differences per group, Holm across all groups tested ------------------------------------------------------
did = []
for gname, c_ in GROUPS.items():
    for g in sorted(pr[c_].unique()):
        m = (pr[c_] == g).to_numpy()
        if m.sum() == 0 or g == "nan": continue
        for v in VAR:
            x = d[v]; dr = boot_mean(x, m) - boot_mean(x, ~m)
            did.append({"split": gname, "group": g, "variant": v, "n": int(m.sum()), "pcps": int(pr.pcp[m].nunique()),
                        "d_group": float(x[m].mean()), "d_rest": float(x[~m].mean()), **{f"did_{k}": val for k, val in summ(dr, x[m].mean() - x[~m].mean()).items()}})
dd = pd.DataFrame(did)
for v in VAR:                                            # Holm within each variant
    s = dd[dd.variant == v].sort_values("did_p"); k = len(s)
    adj = np.minimum(1, np.maximum.accumulate([(k - i) * p for i, p in enumerate(s.did_p)]))
    dd.loc[s.index, "did_p_holm"] = adj
out["did"] = dd.to_dict("records")

# 2. patients 85+ ------------------------------------------------------------------------------------------------------------
age = {}
for g in ["<65", "65–74", "75–84", "85+"]:
    m = (pr.age_g == g).to_numpy(); r = {"n": int(m.sum()), "available_busy": float(pr.elig_busy[m].mean()), "available_familiar": float(pr.elig_familiar[m].mean()),
                                         "pcp_prior_refs_median": float(pr.pcp_prior_refs[m].median()), "eligible_median": float(pr.n_elig[m].median()),
                                         "actual_busy": float(pr.ref_busy[m].mean()), "actual_new_pcp": float(pr.ref_new_pcp[m].mean())}
    for a in ARMS:
        for v in VAR:
            for k in ("load", "load5", "new", "mips"): r[f"{a}|{v}|{k}"] = float(L[(a, v)][k][m].mean())
    age[g] = r
out["age_profile"] = age
# specialty mix: share of referrals by specialty in each age group (top specialties of 85+), and direct standardisation
spec85 = pr.loc[pr.age_g == "85+", "specialty"].value_counts(normalize=True)
mix = {g: pr.loc[pr.age_g == g, "specialty"].value_counts(normalize=True).reindex(spec85.index[:10]).fillna(0).round(3).to_dict() for g in ["<65", "65–74", "75–84", "85+"]}
out["specialty_mix_top10_of_85plus"] = mix
std = {}
under85 = (pr.age_g != "85+").to_numpy() & (pr.age_g != "nan").to_numpy()
for v in VAR:
    per_spec = pd.DataFrame({"spec": pr.specialty[under85], "g": L[("GNN", v)]["load"][under85], "t": L[("Table", v)]["load"][under85],
                             "a": pr.elig_busy[under85]}).groupby("spec").mean()
    w = spec85.reindex(per_spec.index).dropna(); w = w / w.sum(); ps = per_spec.loc[w.index]
    m85 = (pr.age_g == "85+").to_numpy()
    std[v] = {"under85_standardised_to_85plus_mix": {"GNN": float(ps.g @ w), "Table": float(ps.t @ w), "d": float((ps.g - ps.t) @ w), "available_busy": float(ps.a @ w)},
              "under85_own_mix": {"GNN": float(L[("GNN", v)]["load"][under85].mean()), "Table": float(L[("Table", v)]["load"][under85].mean()), "d": float(d[v][under85].mean())},
              "85plus_actual": {"GNN": float(L[("GNN", v)]["load"][m85].mean()), "Table": float(L[("Table", v)]["load"][m85].mean()), "d": float(d[v][m85].mean())},
              "85plus_specialty_coverage": float(spec85.reindex(per_spec.index).dropna().sum())}
out["age_specialty_standardisation"] = std

# 3. PCP history dose-response ---------------------------------------------------------------------------------------------
dose = {}
for key, x_ in (("pcp_prior_refs", pr.pcp_prior_refs), ("elig_familiar", pr.elig_familiar)):
    q = pd.qcut(x_.rank(method="first"), 5, labels=False); rows = []
    for b in range(5):
        m = (q == b).to_numpy(); rr = {"quintile": b + 1, "range": [float(x_[m].min()), float(x_[m].max())], "n": int(m.sum()), "pcps": int(pr.pcp[m].nunique()),
                                       "early_tenure_pcp_referrals": int((m & (pr.pcp_ten_g == "early").to_numpy()).sum())}
        for v in VAR:
            rr[f"d|{v}"] = summ(boot_mean(d[v], m), d[v][m].mean())
            for a in ARMS: rr[f"{a}|{v}|load"] = float(L[(a, v)]["load"][m].mean()); rr[f"{a}|{v}|new"] = float(L[(a, v)]["new"][m].mean())
        rows.append(rr)
    dose[key] = rows
early = (pr.pcp_ten_g == "early").to_numpy()
dose["early_tenure_pcps"] = {"referrals": int(early.sum()), "pcps": int(pr.pcp[early].nunique()),
                             "pcp_prior_refs_median": float(pr.pcp_prior_refs[early].median()), "others_median": float(pr.pcp_prior_refs[~early].median()),
                             "elig_familiar_mean": float(pr.elig_familiar[early].mean()), "others_mean": float(pr.elig_familiar[~early].mean()),
                             "available_busy": float(pr.elig_busy[early].mean()), "others_available_busy": float(pr.elig_busy[~early].mean())}
out["pcp_history_dose"] = dose

# 4. regression of d on group attributes (specialty fixed effects), PCP-cluster bootstrap -----------------------------------
ok = (pr.age_g != "nan").to_numpy()
X = pd.DataFrame({"age_65_74": (pr.age_g == "65–74"), "age_75_84": (pr.age_g == "75–84"), "age_85plus": (pr.age_g == "85+"),
                  "log1p_pcp_prior_refs": np.log1p(pr.pcp_prior_refs), "elig_familiar": pr.elig_familiar, "elig_busy": pr.elig_busy}).astype(float)
names_ = list(X.columns); X = pd.concat([X, pd.get_dummies(pr.specialty, prefix="sp", drop_first=False).astype(float)], axis=1)[ok].to_numpy()
reg = {}
for v in VAR:
    yv = d[v][ok]; beta = np.linalg.lstsq(X, yv, rcond=None)[0][: len(names_)]
    cods = codes[ok]; groups_ = [np.where(cods == k)[0] for k in range(len(uniq))]; draws = []
    rb = np.random.default_rng(7)
    for _ in range(500):
        ix = np.concatenate([groups_[k] for k in rb.integers(0, len(uniq), len(uniq))])
        draws.append(np.linalg.lstsq(X[ix], yv[ix], rcond=None)[0][: len(names_)])
    draws = np.array(draws)
    reg[v] = {nm: {"coef": float(beta[j]), "ci": [float(np.quantile(draws[:, j], .025)), float(np.quantile(draws[:, j], .975))]} for j, nm in enumerate(names_)}
out["regression_d_load"] = reg
json.dump(out, open(f"{OUT}/groups_summary.json", "w"), indent=1, default=float)

# ---- print -----------------------------------------------------------------------------------------------------------------
pd.set_option("display.width", 220)
print("1. Group vs rest (difference-in-differences of GNN − table load)")
print(dd[["split", "group", "variant", "n", "pcps", "d_group", "d_rest", "did_est", "did_ci", "did_p", "did_p_holm"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print("\n2. Age profile (q30)")
for g, r in age.items():
    print(f"  {g:6s} n={r['n']:5d} avail_busy {r['available_busy']:.3f} avail_familiar {r['available_familiar']:.3f} eligible_med {r['eligible_median']:.0f} "
          f"pcp_prior_med {r['pcp_prior_refs_median']:.0f} | actual busy {r['actual_busy']:.3f} new {r['actual_new_pcp']:.2f} | "
          + " ".join(f"{a}: load {r[f'{a}|q30|load']:.3f} new {r[f'{a}|q30|new']:.2f}" for a in ARMS))
print("  specialty mix (top 10 of 85+):"); print(pd.DataFrame(mix).to_string())
print("  standardisation:", json.dumps(std, indent=1))
print("\n3. Dose-response")
for key in ("pcp_prior_refs", "elig_familiar"):
    print(f"  by {key}:")
    for r in dose[key]:
        print(f"    Q{r['quintile']} {r['range']} n={r['n']} pcps={r['pcps']} early-PCP refs={r['early_tenure_pcp_referrals']} | d q30 {r['d|q30']['est']:+.3f} {r['d|q30']['ci']} | "
              f"GNN {r['GNN|q30|load']:.3f} table {r['Table|q30|load']:.3f} no-graph {r['No-graph|q30|load']:.3f} | new GNN {r['GNN|q30|new']:.2f} table {r['Table|q30|new']:.2f}")
print("  early-tenure PCPs:", dose["early_tenure_pcps"])
print("\n4. Regression of d (load) — coefficients with PCP-cluster CIs")
for v in VAR:
    print(f"  {v}: " + "; ".join(f"{k} {c['coef']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]" for k, c in reg[v].items()))
