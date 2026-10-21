"""STEP12a Amendment A1 + revision 1 (post hoc, secondary; frozen 2026-10-13 before any A1 result was examined).

Reads one step10_evaluate.py folder evaluated with --strata --step12 (noise variants needed), the frozen R
(runs/analysis/step12a_decisions.json) and H4's primary p (runs/analysis/step12a/final_summary.json). Load = busiest-10% share
of the top pick; matched fit by random mixture with α per draw (§2.E).

Bootstrap (per draw):
  (a) small-k corrected seed resampling: per cell, per-cluster seed deviations from the cell mean are scaled by √(k/(k−1)), then
      seeds are resampled with replacement (same k); PCP clusters are resampled.
  (b) (a) plus, for every neural cell, N(0, s²_pooled/k) added to its load and to its fit, with s² the within-cell variance of
      seed-level means pooled across all neural cells in the folder (df = Σ(k−1)).
Hypotheses (predicted sign +): H1 LightGBM softmax − G; H2 neural LambdaRank − G; H3′ (LightGBM − G) − (LightGBM-strict −
G-strict) at one common fit floor T = min of the four (per-gap version = sensitivity); H5 neural log-volume − G.
Holm over {H1, H2, H3′, H4, H5} (m = 5), H4 entering at its primary one-sided p. Secondary support: predicted direction,
effect ≥ R/2, Holm p < 0.05 under (a) and (for H1, H3′) under (b); H3′ also requires the per-gap version to pass
(Bonferroni ×5). "Contradicted": opposite-direction Holm p < 0.05 under (a).
Always reported with: noise-degradation matched H1 / H3′, H1/H3′ at base, the fit each cell reaches, effect/R CIs, lenses.
Usage: python scripts/analysis/step12/step12_secondary.py --eval runs/analysis/step12a/secondary
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--eval", required=True); ap.add_argument("--bootstrap", type=int, default=4000)
ap.add_argument("--decisions", default="runs/analysis/step12a_decisions.json")
ap.add_argument("--primary", default="runs/analysis/step12a/final_summary.json")
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
OUT = "runs/analysis/step12a/secondary_summary.json"
if os.path.exists(OUT): raise SystemExit(f"REFUSED: {OUT} exists (never overwritten)")
cf = pd.read_csv(os.path.join(args.eval, "validation_configs.csv")); pr = pd.read_csv(os.path.join(args.eval, "validation_per_referral_PRIVATE.csv.gz"))
D = json.load(open(args.decisions)); R = D["R"]["est"]; assert D["branch"] == "tabular"
P = json.load(open(args.primary)); H4p = P["hypotheses"]["H4"]
codes, uniq = pd.factorize(pr.pcp); NC = len(uniq); B = args.bootstrap; rng = np.random.default_rng(20261014)
IDX = rng.integers(0, NC, size=(B, NC)); CNT = np.bincount(codes, minlength=NC).astype(float); DEN = CNT[IDX].sum(1); TOT = CNT.sum()
csum = lambda x: np.bincount(codes, weights=np.nan_to_num(np.asarray(x, float)), minlength=NC)
RF, RL = csum(pr["random|r5"]), csum(pr["random|busy1"]); RFb, RLb = RF[IDX].sum(1) / DEN, RL[IDX].sum(1) / DEN; RF0, RL0 = RF.sum() / TOT, RL.sum() / TOT


def fams(prefix):
    f = sorted(x for x in cf.family.unique() if x.startswith(prefix) and "@" not in x); return f


class Cell:
    def __init__(self, fam, neural, variant="blend0.3"):
        cs = cf[(cf.family == fam) & (cf.variant == variant)].config.tolist(); assert cs, f"{fam}:{variant} missing"
        self.fam, self.k, self.neural, self.cfgs = fam, len(cs), neural, cs
        F = np.stack([csum(pr[f"{c}|r5"]) for c in cs]); L = np.stack([csum(pr[f"{c}|busy1"]) for c in cs])
        self.F0, self.L0 = F.mean(0).sum() / TOT, L.mean(0).sum() / TOT
        self.seed_means = (F.sum(1) / TOT, L.sum(1) / TOT)
        infl = np.sqrt(self.k / (self.k - 1)) if self.k > 1 else 1.0                        # small-k correction (rev 1, B1)
        Fc, Lc = F.mean(0) + infl * (F - F.mean(0)), L.mean(0) + infl * (L - L.mean(0))
        w = np.stack([np.bincount(rng.integers(0, self.k, self.k), minlength=self.k) for _ in range(B)]) / self.k
        self.Fa = np.take_along_axis(w @ Fc, IDX, 1).sum(1) / DEN; self.La = np.take_along_axis(w @ Lc, IDX, 1).sum(1) / DEN
        self.Fb, self.Lb = self.Fa.copy(), self.La.copy()
    def add_pooled(self, sF, sL):
        if self.neural: self.Fb = self.Fa + rng.normal(0, sF / np.sqrt(self.k), B); self.Lb = self.La + rng.normal(0, sL / np.sqrt(self.k), B)
    def lenses(self):
        m = lambda met: float(np.nanmean(np.mean([pr[f"{c}|{met}"].to_numpy(float) for c in self.cfgs], 0)))
        return {"fit": float(self.F0), "load": float(self.L0), "mips_gain": m("gain5"), "travel_mean_mi": m("dist5m"), "new_to_pcp": m("new_pcp5"), "seeds": self.k}


def mx(F, L, T, Fr, Lr):
    a = np.clip((F - T) / np.maximum(F - Fr, 1e-9), 0, 1); return (1 - a) * L + a * Lr


def pair(A, Bc, mode):
    if mode == "point": T = min(A.F0, Bc.F0); return mx(A.F0, A.L0, T, RF0, RL0) - mx(Bc.F0, Bc.L0, T, RF0, RL0)
    Fa, La, Fb_, Lb_ = (A.Fa, A.La, Bc.Fa, Bc.La) if mode == "a" else (A.Fb, A.Lb, Bc.Fb, Bc.Lb)
    T = np.minimum(Fa, Fb_); return mx(Fa, La, T, RFb, RLb) - mx(Fb_, Lb_, T, RFb, RLb)


def h3(cells, mode, common=True):
    REF, N, LS, NS = cells
    if mode == "point":
        F = {c: (c.F0, c.L0) for c in cells}; T = min(c.F0 for c in cells); fr, lr = RF0, RL0
    else:
        F = {c: ((c.Fa, c.La) if mode == "a" else (c.Fb, c.Lb)) for c in cells}; T = np.minimum.reduce([F[c][0] for c in cells]); fr, lr = RFb, RLb
    if not common: return pair(REF, N, mode) - pair(LS, NS, mode)
    g = lambda c: mx(F[c][0], F[c][1], T, fr, lr)
    return (g(REF) - g(N)) - (g(LS) - g(NS))


def summ(pt, bs):
    return {"effect": float(pt), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "effect_over_R": float(pt / R), "effect_over_R_ci": [float(np.quantile(bs, .025) / R), float(np.quantile(bs, .975) / R)],
            "p_one_sided": float(((bs <= 0).sum() + 1) / (B + 1)), "p_opposite": float(((bs >= 0).sum() + 1) / (B + 1))}


ls_fam = fams("s12_lgbm_tabular_lambdarank_strict_L"); assert len(ls_fam) == 1, ls_fam
C = {"REF": Cell("lgbm_tabular", False), "LSM": Cell("s12_lgbm_tabular_softmax", False), "LS": Cell(ls_fam[0], False),
     "N": Cell("model_s12_gate", True), "NLR": Cell("model_s12_lr", True), "NLOG": Cell("model_s12_log", True), "NS": Cell("model_s12_strict", True)}
for extra in ("model_s12_bce", "model_s12_shares"):                    # pooled variance uses every neural cell present
    if extra in set(cf.family): C[extra] = Cell(extra, True)
neural = [c for c in C.values() if c.neural]
dfs = sum(c.k - 1 for c in neural)
sF = np.sqrt(sum((c.k - 1) * np.var(c.seed_means[0], ddof=1) for c in neural if c.k > 1) / dfs)
sL = np.sqrt(sum((c.k - 1) * np.var(c.seed_means[1], ddof=1) for c in neural if c.k > 1) / dfs)
for c in C.values(): c.add_pooled(sF, sL)

res = {}
for h, (A, Bc, lab) in {"H1": (C["LSM"], C["N"], "LightGBM softmax − neural softmax (G)"), "H2": (C["NLR"], C["N"], "neural LambdaRank − neural softmax (G)"),
                        "H5": (C["NLOG"], C["N"], "neural log-volume − neural softmax (G)")}.items():
    pt = pair(A, Bc, "point"); res[h] = {"contrast": lab, **summ(pt, pair(A, Bc, "a"))}
    if h == "H1": res[h]["pooled_b"] = summ(pt, pair(A, Bc, "b"))
cells3 = (C["REF"], C["N"], C["LS"], C["NS"])
pt3 = h3(cells3, "point"); res["H3′"] = {"contrast": "(LightGBM − G) − (LightGBM-strict − G-strict), common fit floor", **summ(pt3, h3(cells3, "a"))}
res["H3′"]["pooled_b"] = summ(pt3, h3(cells3, "b"))
res["H3′"]["per_gap_sensitivity"] = summ(h3(cells3, "point", False), h3(cells3, "a", False))
res["H4"] = {"contrast": "primary (unchanged)", "p_one_sided": H4p["p_one_sided"], "p_opposite": H4p["p_opposite"], "effect": H4p["effect"], "ci": H4p["ci"]}
# Holm (m = 5) under (a), and under (b) for H1/H3′ (the others keep their (a) p in the (b) family)
for key, sub, outk in (("p_one_sided", None, "p_holm_a"), ("p_one_sided", "pooled_b", "p_holm_b"), ("p_opposite", None, "p_opposite_holm")):
    pv = {h: (res[h][sub][key] if sub and sub in res[h] else res[h][key]) for h in res}
    order = sorted(pv, key=pv.get); run = 0.0
    for i, h in enumerate(order): run = max(run, min(1.0, (len(order) - i) * pv[h])); res[h][outk] = run
for h in ("H1", "H2", "H3′", "H5"):
    r = res[h]; ok = r["effect"] > 0 and r["effect"] >= R / 2 and r["p_holm_a"] < 0.05
    if h in ("H1", "H3′"): ok = ok and r["p_holm_b"] < 0.05
    if h == "H3′": g = r["per_gap_sensitivity"]; ok = ok and g["effect"] >= R / 2 and min(1.0, 5 * g["p_one_sided"]) < 0.05
    r["verdict_secondary"] = "supported (secondary, post hoc)" if ok else ("contradicted" if r["p_opposite_holm"] < 0.05 else "not supported")

# required companions: noise-degradation matching, base variants, fits
def noise_curve(fam):
    pts = []
    for v in ("blend0.3", "noise0.25_blend0.3", "noise0.5_blend0.3", "noise1.0_blend0.3", "noise2.0_blend0.3"):
        cs = cf[(cf.family == fam) & (cf.variant == v)].config.tolist()
        if not cs: return None
        pts.append((float(np.mean([pr[f"{c}|r5"].mean() for c in cs])), float(np.mean([pr[f"{c}|busy1"].mean() for c in cs]))))
    return sorted(pts)
def at_fit(fam, T):
    cur = noise_curve(fam); xs, ys = zip(*cur); return float(np.interp(T, xs, ys))
T1 = min(C["LSM"].F0, C["N"].F0); T3 = min(c.F0 for c in cells3)
noise = {"H1": at_fit(C["LSM"].fam, T1) - at_fit(C["N"].fam, T1),
         "H3′": (at_fit(C["REF"].fam, T3) - at_fit(C["N"].fam, T3)) - (at_fit(C["LS"].fam, T3) - at_fit(C["NS"].fam, T3))}
base = {"H1": float(pair(Cell(C["LSM"].fam, False, "base"), Cell(C["N"].fam, True, "base"), "point")),
        "H3′": float(h3(tuple(Cell(c.fam, c.neural, "base") for c in cells3), "point"))}
out = {"amendment": "A1 + revision 1 (post hoc, secondary; rules frozen before A1 results were examined)", "R": R, "R_half": R / 2,
       "pooled_neural_seed_sd": {"fit": float(sF), "load": float(sL), "df": int(dfs)},
       "hypotheses": res, "noise_degradation_matched": noise, "base_ranking_point": base,
       "cells": {k: {"family": c.fam, **c.lenses()} for k, c in C.items()},
       "rule_changes_vs_primary": ["max-gap noise band → small-k corrected seed bootstrap (+ pooled-neural term for H1/H3′)", "all-pairings sign rule dropped",
                                   "H3 (shares) → H3′ (strict attribute set, common floor)", "extra seeds only for G (6) and strict, chosen after the primary verdict"],
       "primary_verdict": P.get("overall"), "primary_reference": args.primary}
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
for h in ("H1", "H2", "H3′", "H4", "H5"):
    r = res[h]; print(h, r.get("verdict_secondary", "primary"), {k: r[k] for k in ("effect", "ci", "p_holm_a", "p_holm_b") if k in r})
print("noise-matched", noise, "| base", base, "| primary:", P.get("overall"))
