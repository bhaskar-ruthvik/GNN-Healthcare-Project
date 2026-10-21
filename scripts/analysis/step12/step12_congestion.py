"""STEP12a analysis (STEP12_CONGESTION_PLAN.md, frozen 2026-10-12): gate, frozen reference gap, signed primary contrasts.

Reads one step10_evaluate.py output folder (validation; run with --strata --step12), i.e. <dir>/validation_configs.csv (config →
arm family, run, variant) and <dir>/validation_per_referral_PRIVATE.csv.gz. Everything is at the 30% quality weight unless stated.

Matched fit (§2.E): to compare cells A and B, the one with the higher fit is mixed with the random scorer down to the other's fit:
  α = (F_s − F_w) / (F_s − F_random), matched load = (1 − α)·L_s + α·L_random,
computed per referral from busy1 / r5 and the exact random-scorer columns; α is recomputed inside every PCP-cluster bootstrap draw.
Seed-averaged series per cell; seed pairings for the sign rule use single-seed series.

Stages:
  --stage gate    G (masked no-graph GNN) and the no-graph GNN against the tabular LightGBM: d_G, d_NG, branch rule (§2.0),
                  then R = LightGBM-LambdaRank (branch inputs) − N. Writes runs/analysis/step12a_decisions.json (refuses to
                  overwrite). Must run before any §2.A–H5 cell is trained.
  --stage pcpcheck  R2 PCP-attribute check: LightGBM LambdaRank without the 11 PCP columns (fixed mode, reference config) vs the
                  reference; if |matched Δload| > the LightGBM noise band, H1 uses the no-PCP softmax cell. Adds "pcp_check" to the
                  decision file (only that key; R stays frozen). Run before training the H1 LightGBM softmax cell.
  --stage final   H1–H5 signed contrasts, Holm (one-sided, predicted direction), noise bands, seed signs, interaction, B1,
                  descriptive mechanism measures, selection-overlap sensitivity, lenses per cell. Reads the decision file.
Usage: python scripts/analysis/step12/step12_congestion.py --stage gate --eval runs/analysis/step12a/gate
Aggregates only.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["gate", "pcpcheck", "final"])
ap.add_argument("--eval-extra", nargs="*", default=[], help="final stage: further evaluation folders (same cohort) whose families are merged in "
                                                            "when not already present (B1 @ablation families, GNN / no-graph lenses)")
ap.add_argument("--eval-capacity", default=None, help="final stage: separate (--lean) evaluation folder holding the capacity grid and the reference runs")
ap.add_argument("--eval", required=True, help="step10_evaluate output folder")
ap.add_argument("--bootstrap", type=int, default=2000)
ap.add_argument("--decisions", default="runs/analysis/step12a_decisions.json")
args = ap.parse_args()
root = str(workspace()); os.chdir(root)

cf = pd.read_csv(os.path.join(args.eval, "validation_configs.csv"))
pr = pd.read_csv(os.path.join(args.eval, "validation_per_referral_PRIVATE.csv.gz"))
for ex in args.eval_extra:                               # merge families from further folders (same cohort, same referral order)
    cfx = pd.read_csv(os.path.join(ex, "validation_configs.csv")); prx = pd.read_csv(os.path.join(ex, "validation_per_referral_PRIVATE.csv.gz"))
    assert np.array_equal(prx.enc.to_numpy(), pr.enc.to_numpy()), f"{ex}: different cohort or order"
    new = cfx[~cfx.family.isin(set(cf.family))]
    cols = [c for c in prx.columns if c.split("|")[0] in set(new.config) and c not in pr.columns]
    pr = pd.concat([pr, prx[cols]], axis=1); cf = pd.concat([cf, new], ignore_index=True)
    print(f"merged {new.family.nunique()} families / {len(cols)} columns from {ex}")
VAR = "blend0.3"
codes, uniq = pd.factorize(pr.pcp); NC = len(uniq)
IDX = np.random.default_rng(20261013).integers(0, NC, size=(args.bootstrap, NC))
R_FIT = pr["random|r5"].to_numpy(float); R_LOAD = pr["random|busy1"].to_numpy(float)
sel_mask = ~pr.selection_query.astype(bool).to_numpy()


def configs(family, variant=VAR):
    c = cf[(cf.family == family) & (cf.variant == variant)].config.tolist()
    assert c, f"no configurations for {family}:{variant}"
    return c


def series(cfgs, metric):
    return np.mean([pr[f"{c}|{metric}"].to_numpy(float) for c in cfgs], axis=0)


class Cell:
    """A model cell (seed-averaged). load_metric busy1 (primary: top pick) or busy5 (secondary: top-5 slots)."""
    def __init__(self, family, label=None, variant=VAR, load_metric="busy1"):
        self.family, self.label, self.cfgs, self.lm = family, label or family, configs(family, variant), load_metric
        self.fit, self.load = series(self.cfgs, "r5"), series(self.cfgs, load_metric)
    def seeds(self):
        return [(c, pr[f"{c}|r5"].to_numpy(float), pr[f"{c}|{self.lm}"].to_numpy(float)) for c in self.cfgs]


def cmean(x, mask=None, draw=None):
    """Cluster (PCP) mean of x on the full sample (draw None) or bootstrap draws (array of shape B)."""
    m = np.ones(len(x), bool) if mask is None else mask
    s1 = np.bincount(codes, weights=np.where(m, x, 0.0), minlength=NC); s0 = np.bincount(codes, weights=m.astype(float), minlength=NC)
    if draw is None: return s1.sum() / s0.sum()
    return s1[draw].sum(1) / np.maximum(s0[draw].sum(1), 1e-9)


def matched_load(fit, load, target_fit, draw=None, mask=None, rload=None):
    """Load after mixing with random down to target_fit (no mixing if fit ≤ target). target_fit may be per-draw (array).
    The random scorer's load is its busy1 (= busy5 = availability of busy specialists)."""
    F = cmean(fit, mask, draw); L = cmean(load, mask, draw); Fr = cmean(R_FIT, mask, draw); Lr = cmean(R_LOAD if rload is None else rload, mask, draw)
    a = np.clip((F - target_fit) / np.maximum(F - Fr, 1e-9), 0.0, 1.0)
    return (1 - a) * L + a * Lr


def contrast(fa, la, fb, lb, draw=None, mask=None):
    """matched load(A) − matched load(B), the higher-fit cell degraded to the lower fit."""
    Fa, Fb = cmean(fa, mask, draw), cmean(fb, mask, draw); T = np.minimum(Fa, Fb)
    return matched_load(fa, la, T, draw, mask) - matched_load(fb, lb, T, draw, mask)


def ctr(A, B, mask=None):
    pt = float(contrast(A.fit, A.load, B.fit, B.load, None, mask)); bs = contrast(A.fit, A.load, B.fit, B.load, IDX, mask)
    return pt, bs


def seed_signs(A, B):
    return [float(contrast(fa, la, fb, lb)) for (_, fa, la), (_, fb, lb) in itertools.product(A.seeds(), B.seeds())]


def band(cells):
    """Largest within-cell seed gap in load (30% weight) over the given cells."""
    g = [abs(cmean(l1) - cmean(l2)) for c in cells for (_, _, l1), (_, _, l2) in itertools.combinations(c.seeds(), 2)]
    return float(max(g)) if g else float("nan")


def lenses(C):
    cf_ = C.cfgs
    return {"fit": float(cmean(C.fit)), "load": float(cmean(C.load)), "mips_gain": float(cmean(series(cf_, "gain5"))),
            "travel_mean_mi": float(np.nanmean(series(cf_, "dist5m"))), "new_to_pcp": float(cmean(series(cf_, "new_pcp5"))), "n_seeds": len(cf_)}


def fam_of(prefix):
    f = [x for x in cf.family.unique() if x == prefix]
    assert len(f) == 1, f"arm {prefix} not found (have: {sorted(cf.family.unique())})"; return f[0]


# =================================================================================================== gate
if args.stage == "gate":
    if os.path.exists(args.decisions): raise SystemExit(f"REFUSED: {args.decisions} exists; R is frozen once")
    T, G, NG = Cell(fam_of("lgbm_tabular"), "table"), Cell(fam_of("model_s12_gate"), "G"), Cell(fam_of("model_unmasked_nograph"), "no-graph GNN")
    dG, dG_b = ctr(T, G); dNG, dNG_b = ctr(T, NG)
    half = 0.5 * dNG; signs = seed_signs(T, G)
    if dG >= half and all(x > 0 for x in signs): branch = "tabular"
    elif np.quantile(dG_b, 0.975) < half: branch = "primary"
    else: branch = "indeterminate→primary"
    if branch == "tabular": ref, N = T, G
    else: ref, N = Cell(fam_of("lgbm_primary"), "LightGBM primary"), NG
    R, R_b = ctr(ref, N)
    out = {"stage": "gate", "eval": args.eval, "setting": "30% quality weight, matched fit (random mixture), PCP-cluster bootstrap",
           "d_G": {"est": dG, "ci": [float(np.quantile(dG_b, .025)), float(np.quantile(dG_b, .975))], "seed_pairings": signs},
           "d_NG": {"est": dNG, "ci": [float(np.quantile(dNG_b, .025)), float(np.quantile(dNG_b, .975))]}, "half_d_NG": half,
           "branch": branch, "reference": ref.family, "N": N.family,
           "R": {"est": R, "ci": [float(np.quantile(R_b, .025)), float(np.quantile(R_b, .975))], "note": "point estimate is the frozen threshold scale; CI descriptive"},
           "R_half": R / 2, "noise_band_neural": band([G, NG]), "noise_band_lgbm": band([T]),
           "lenses": {c.label: lenses(c) for c in (T, G, NG)}, "frozen": pd.Timestamp.now().isoformat(timespec="seconds")}
    if branch != "tabular": out["lenses"][ref.label] = lenses(ref)
    os.makedirs(os.path.dirname(args.decisions), exist_ok=True); json.dump(out, open(args.decisions, "w"), indent=1)
    print(json.dumps(out, indent=1)); raise SystemExit(0)

# =================================================================================================== pcpcheck / final
D = json.load(open(args.decisions)); R = D["R"]["est"]; branch = D["branch"]
fs = "tabular" if branch == "tabular" else "primary"; P = "" if branch == "tabular" else "p_"
cell = lambda fam, lab, **kw: Cell(fam_of(fam), lab, **kw)
def fams(prefix):                                   # families with a prefix, never the B1 (@ablation) families
    return sorted(f for f in cf.family.unique() if f.startswith(prefix) and "@" not in f)
REF = cell(D["reference"], "LightGBM LambdaRank")

if args.stage == "pcpcheck":
    if "pcp_check" in D: raise SystemExit("REFUSED: pcp_check already recorded")
    nop = fams(f"s12_lgbm_{fs}_lambdarank_nopcp_L"); assert len(nop) == 1, nop
    NOP = cell(nop[0], "LightGBM LambdaRank, no PCP columns"); pt, bs = ctr(NOP, REF); bL0 = band([REF])
    D["pcp_check"] = {"cell": nop[0], "matched_delta_load": pt, "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "lgbm_noise_band": bL0,
                      "h1_uses_nopcp_softmax": bool(abs(pt) > bL0), "recorded": pd.Timestamp.now().isoformat(timespec="seconds")}
    json.dump(D, open(args.decisions, "w"), indent=1); print(json.dumps(D["pcp_check"], indent=1)); raise SystemExit(0)

N = cell(D["N"], "N (neural softmax)")
use_nopcp = D.get("pcp_check", {}).get("h1_uses_nopcp_softmax", False)
LSM = cell(f"s12_lgbm_{fs}_softmax" + ("_nopcp" if use_nopcp else ""), "LightGBM softmax" + (" (no PCP columns)" if use_nopcp else ""))
LBIN = cell(f"s12_lgbm_{fs}_binary", "LightGBM binary")
NLR = cell(f"model_s12_{P}lr", "neural LambdaRank"); NBCE = cell(f"model_s12_{P}bce", "neural BCE")
B2fam = fams(f"s12_lgbm_{fs}_lambdarank_shares_L"); assert len(B2fam) == 1, B2fam
LB2 = cell(B2fam[0], "LightGBM LambdaRank, volume shares"); NB2 = cell(f"model_s12_{P}shares", "N, volume shares")
NLOG = cell(f"model_s12_{P}log", "N, log volume")
NEURAL, LGBM = [N, NLR, NBCE, NB2, NLOG], [REF, LSM, LBIN, LB2]
bN, bL = band(NEURAL), band(LGBM)

def summarise(pt, bs, sign, nb, pairs, sel, label):
    effb = sign * bs
    return {"contrast": label, "predicted_sign": "+" if sign > 0 else "−", "effect": float(pt), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "effect_over_R": float(pt / R), "effect_over_R_ci": [float(np.quantile(bs, .025) / R), float(np.quantile(bs, .975) / R)],
            "p_one_sided": float(((effb <= 0).sum() + 1) / (len(bs) + 1)), "p_opposite": float(((effb >= 0).sum() + 1) / (len(bs) + 1)),
            "noise_band": nb, "seed_pairings": pairs, "sel_excluded": float(sel)}

def pair_contrast(A, B, sign, nb):
    pt, bs = ctr(A, B)
    return summarise(pt, bs, sign, nb, seed_signs(A, B), contrast(A.fit, A.load, B.fit, B.load, None, sel_mask), f"{A.label} − {B.label}")

def h3_fn(cells):
    REF_, N_, LB2_, NB2_ = cells
    def f(draw=None, mask=None):
        return contrast(REF_.fit, REF_.load, N_.fit, N_.load, draw, mask) - contrast(LB2_.fit, LB2_.load, NB2_.fit, NB2_.load, draw, mask)
    pairs = [float(contrast(f1, l1, f2, l2) - contrast(f3, l3, f4, l4)) for (_, f1, l1), (_, f2, l2), (_, f3, l3), (_, f4, l4)
             in itertools.product(REF_.seeds(), N_.seeds(), LB2_.seeds(), NB2_.seeds())]
    return f, pairs

res = {}
res["H1"] = pair_contrast(LSM, N, +1, max(bN, bL))
res["H2"] = pair_contrast(NLR, N, +1, bN)
f3, pairs3 = h3_fn((REF, N, LB2, NB2))
res["H3"] = summarise(float(f3()), f3(IDX), +1, max(bN, bL), pairs3, f3(None, sel_mask), "(LightGBM − N) − (LightGBM-B2 − N-B2)")
res["H5"] = pair_contrast(NLOG, N, +1, bN)

# H4: capacity grid (own --lean evaluation folder with the reference runs); F_ref recomputed per draw / mask
caps, h4 = [], {"status": "not testable"}
if args.eval_capacity:
    cfc = pd.read_csv(os.path.join(args.eval_capacity, "validation_configs.csv")); prc = pd.read_csv(os.path.join(args.eval_capacity, "validation_per_referral_PRIVATE.csv.gz"))
    assert np.array_equal(prc.enc.to_numpy(), pr.enc.to_numpy()), "capacity evaluation must use the same cohort and referral order"
    def ser_c(fam, metric):
        c_ = cfc[(cfc.family == fam) & (cfc.variant == VAR)].config.tolist(); assert c_, fam
        return np.mean([prc[f"{x}|{metric}"].to_numpy(float) for x in c_], axis=0), c_
    rf, _ = ser_c(D["reference"], "r5")
    Fref = lambda draw=None, mask=None: cmean(rf, mask, draw)
    for f in sorted(x for x in cfc.family.unique() if x.startswith(f"s12_lgbm_{fs}_lambdarank_cap_L") and "@" not in x):
        L_, M_, R_ = (int(t[1:]) for t in f.split("_cap_")[1].split("_")); fit_, _ = ser_c(f, "r5"); load_, _ = ser_c(f, "busy1")
        caps.append({"family": f, "leaves": L_, "minleaf": M_, "rounds": R_, "capacity": L_ * R_, "fit": float(cmean(fit_)), "load": float(cmean(load_)),
                     "matched_load_at_Fref": float(matched_load(fit_, load_, Fref())) if cmean(fit_) >= Fref() else None})
    caps = sorted(caps, key=lambda d: (d["capacity"], -d["minleaf"]))      # ties: min leaf 200 = lower capacity
    elig = [d for d in caps if d["fit"] >= Fref()]
    if len(elig) >= 2:
        lo, hi = elig[0], elig[-1]
        def endp(d):
            fam = f"s12_lgbm_{fs}_lambdarank_L{d['leaves']}_M{d['minleaf']}_R{d['rounds']}"
            fit_, c_ = ser_c(fam, "r5"); load_, _ = ser_c(fam, "busy1"); return fam, fit_, load_, c_
        (fh, HIf, HIl, hic), (fl, LOf, LOl, loc) = endp(hi), endp(lo)
        f4 = lambda draw=None, mask=None: matched_load(HIf, HIl, Fref(draw, mask), draw, mask) - matched_load(LOf, LOl, Fref(draw, mask), draw, mask)
        seedser = lambda cs: [(prc[f"{x}|r5"].to_numpy(float), prc[f"{x}|busy1"].to_numpy(float)) for x in cs]
        pairs4 = [float(matched_load(a, b, Fref()) - matched_load(c, d, Fref())) for (a, b), (c, d) in itertools.product(seedser(hic), seedser(loc))]
        gap = lambda cs: max([abs(cmean(b1) - cmean(b2)) for (_, b1), (_, b2) in itertools.combinations(seedser(cs), 2)] or [0.0])
        h4 = summarise(float(f4()), f4(IDX), +1, max(bL, gap(hic), gap(loc)), pairs4, f4(None, sel_mask), f"capacity {fh} − {fl} (matched to F_ref)")
    else:
        h4 = {"status": "not testable", "eligible_configs": len(elig)}
res["H4"] = h4

# Holm over the testable family (one-sided, predicted direction); contradictions on the opposite side
test = [h for h in ("H1", "H2", "H3", "H4", "H5") if "effect" in res[h]]
for key, out_key in (("p_one_sided", "p_holm"), ("p_opposite", "p_opposite_holm")):
    order = sorted(test, key=lambda h: res[h][key]); k = len(order); run = 0.0
    for i, h in enumerate(order):
        run = max(run, min(1.0, (k - i) * res[h][key])); res[h][out_key] = run
for h in test:
    r = res[h]; sgn = 1 if r["predicted_sign"] == "+" else -1; eff = sgn * r["effect"]; pairs_ok = all(sgn * x > 0 for x in r["seed_pairings"])
    crit = {"effect≥R/2": bool(eff >= R / 2), "holm<0.05": bool(r["p_holm"] < 0.05), "beyond_noise": bool(eff > r["noise_band"]), "all_seed_pairings": pairs_ok}
    r["criteria"] = crit
    r["verdict"] = "supported" if all(crit.values()) else ("contradicted" if r["p_opposite_holm"] < 0.05 else "not supported")
overall = "supported: " + ", ".join(h for h in test if res[h]["verdict"] == "supported") if any(res[h]["verdict"] == "supported" for h in test) \
    else "not explained by any single factor at this resolution"

# interaction {LambdaRank, softmax} × {LightGBM, neural}; common fit floor recomputed per draw
four = [REF, LSM, NLR, N]
Fmin = lambda draw=None, mask=None: np.minimum.reduce([cmean(c.fit, mask, draw) for c in four])
ml = lambda C, draw=None: matched_load(C.fit, C.load, Fmin(draw), draw)
inter = lambda draw=None: (ml(REF, draw) - ml(LSM, draw)) - (ml(NLR, draw) - ml(N, draw))
ib = inter(IDX)
res["interaction"] = {"definition": "(LightGBM LR − LightGBM softmax) − (neural LR − neural softmax), loads matched to the lowest fit of the four",
                      "fit_floor": float(Fmin()), "matched_loads": {c.label: float(ml(c)) for c in four}, "est": float(inter()),
                      "ci": [float(np.quantile(ib, .025)), float(np.quantile(ib, .975))]}

# secondary: H1/H2/H5 at base and on top-5 slots (30% weight)
sec = {}
for h, (a, b) in {"H1": (LSM, N), "H2": (NLR, N), "H5": (NLOG, N)}.items():
    A0, B0 = Cell(a.family, a.label, variant="base"), Cell(b.family, b.label, variant="base"); pt, bs = ctr(A0, B0)
    A5, B5 = Cell(a.family, a.label, load_metric="busy5"), Cell(b.family, b.label, load_metric="busy5"); pt5, bs5 = ctr(A5, B5)
    sec[h] = {"base_top1": {"est": pt, "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]},
              "q30_top5_slots": {"est": pt5, "ci": [float(np.quantile(bs5, .025)), float(np.quantile(bs5, .975))]}}
res["secondary"] = sec

# §2.E sensitivity: matching by Gaussian-noise degradation (neural / LightGBM model configs with +noise variants)
def noise_curve(C):
    pts = [(float(cmean(C.fit)), float(cmean(C.load)))]
    for sig in (0.25, 0.5, 1.0, 2.0):
        try: X = Cell(C.family, variant=f"noise{sig}_blend0.3"); pts.append((float(cmean(X.fit)), float(cmean(X.load))))
        except AssertionError: return None
    return sorted(pts)
def noise_matched(A, B):
    """load(A) − load(B) with the higher-fit cell degraded along its own noise curve (linear interpolation) to the other's fit."""
    fa, fb = float(cmean(A.fit)), float(cmean(B.fit)); S, W = (A, B) if fa >= fb else (B, A); cur = noise_curve(S)
    if cur is None: return None
    xs, ys = zip(*cur); Ls = float(np.interp(min(fa, fb), xs, ys)); Lw = float(cmean(W.load))
    return (Ls - Lw) if S is A else (Lw - Ls)
res["noise_matching_sensitivity"] = {h: noise_matched(a, b) for h, (a, b) in {"H1": (LSM, N), "H2": (NLR, N), "H5": (NLOG, N)}.items()}

# descriptive: B1, mechanism measures, rank blend, residualisation, PCP overlap of top picks, specialty decomposition, lenses
sp = pr.specialty.to_numpy(); pcp_arr = pr.pcp.to_numpy()
def pcp_overlap(C):
    cols = [f"{c}|top1_spec" for c in C.cfgs if f"{c}|top1_spec" in pr]
    if not cols: return None
    t = pr[cols[0]].to_numpy()                                         # first seed's top picks
    js = []
    for s_ in np.unique(sp):
        m = sp == s_; byp = {p_: set(t[m & (pcp_arr == p_)]) for p_ in np.unique(pcp_arr[m]) if (m & (pcp_arr == p_)).sum() >= 5}
        for a_, b_ in itertools.combinations(byp.values(), 2): js.append(len(a_ & b_) / len(a_ | b_))
    return float(np.mean(js)) if js else None
others = [f for f in ("model_unmasked", "model_unmasked_nograph") if f in set(cf.family)]
desc = {}
for C in [REF, N] + LGBM[1:] + NEURAL[1:] + [Cell(f, f) for f in others if f != N.family]:
    d = {}
    for abl in ("no_volume", "no_volume_plus"):
        f = f"{C.family}@{abl}"
        if f in set(cf.family):
            A = Cell(f); pt, bs = ctr(A, C)
            d[f"B1_{abl}"] = {"load_change_matched": pt, "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "fit_change": float(cmean(A.fit) - cmean(C.fit))}
    for mname in ("volpct1_ft", "volpct1_all", "tau_vol", "gap12"):
        cols = [f"{c}|{mname}" for c in C.cfgs if f"{c}|{mname}" in pr]
        if cols: d[mname] = float(np.nanmean(np.mean([pr[x].to_numpy(float) for x in cols], axis=0)))
    for v in ("rblend0.3", "resid_blend0.3", "resid"):
        try: A = Cell(C.family, variant=v); d[f"{v}"] = {"load": float(cmean(A.load)), "fit": float(cmean(A.fit))}
        except AssertionError: pass
    d["pcp_overlap_top1_jaccard"] = pcp_overlap(C); d["lenses_q30"] = lenses(C); d["lenses_base"] = lenses(Cell(C.family, variant="base"))
    desc[C.label] = d
desc["practice_ref_volpct_ft"] = float(np.nanmean(pr.ref_volpct_ft)) if "ref_volpct_ft" in pr else None
dd = REF.load - N.load
desc["specialty_decomposition_REF_minus_N"] = sorted([{"specialty": s_, "share": float((sp == s_).mean()), "delta_load": float(dd[sp == s_].mean()),
                                                       "contribution": float((sp == s_).mean() * dd[sp == s_].mean())} for s_ in np.unique(sp)], key=lambda r: -abs(r["contribution"]))[:15]

# B2 disclosure: how much volume information remains under "shares" (R² of log beneficiaries from the remaining specialist columns)
def b2_residual_volume():
    import sys
    sys.path.insert(0, root)
    from referral_gnn.three_node_train import load_three_node_inputs
    from referral_gnn import volume as V
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    mi = load_three_node_inputs("prepared_data/three_node_v5/model_inputs_PRIVATE.pkl"); st = V.stats(mi)
    snap = V._last_snapshot(mi); X = snap["specialist_num"]; act = np.asarray(mi.spec_first_seen) <= int(mi.train_end_day)
    y = np.log1p(V._raw(mi, X, "beneficiary_count")[0][act]); Xs = V.transform_np(X, st, "shares")[act]
    drop = sum([st["cols"][n] for n in ("beneficiary_count", "service_count")], []); keep = [j for j in range(Xs.shape[1]) if j not in drop]
    Z = np.concatenate([Xs[:, keep], snap["specialist_cat"][act]], axis=1)
    zero_cells = np.stack([(V._raw(mi, X, n)[0][act] <= 0) for n in V.PANEL], 1).astype(float)
    out = {"n_specialists": int(act.sum()), "specialists_with_a_zero_panel_cell": int(zero_cells.any(1).sum())}
    for lab, Zx in (("remaining_columns", Z), ("remaining_plus_zero_cell_flags", np.concatenate([Z, zero_cells], 1))):
        out[f"R2_linear_{lab}"] = float(np.mean(cross_val_score(LinearRegression(), Zx, y, cv=5, scoring="r2")))
        out[f"R2_gbm_{lab}"] = float(np.mean(cross_val_score(HistGradientBoostingRegressor(random_state=0), Zx, y, cv=5, scoring="r2")))
    return out
desc["B2_residual_volume_information"] = b2_residual_volume()

out = {"stage": "final", "branch": branch, "R": R, "R_half": R / 2, "noise_band_neural": bN, "noise_band_lgbm": bL, "pcp_check": D.get("pcp_check"),
       "hypotheses": res, "overall": overall, "descriptive": desc, "capacity_grid": caps}
os.makedirs("runs/analysis/step12a", exist_ok=True)
fn = "runs/analysis/step12a/final_summary.json"
if os.path.exists(fn): fn = fn.replace(".json", pd.Timestamp.now().strftime("_%Y%m%dT%H%M%S.json"))   # never overwrite
json.dump(out, open(fn, "w"), indent=1, default=float); print("wrote", fn)
for h in ("H1", "H2", "H3", "H4", "H5"):
    r = res[h]; print(h, r.get("verdict", r.get("status")), {k: r[k] for k in ("contrast", "effect", "ci", "p_holm", "noise_band", "criteria") if k in r})
print("interaction", res["interaction"]); print(overall)
