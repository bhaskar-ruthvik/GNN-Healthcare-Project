"""STEP9_PROTOCOL §4 analyses. Aggregate output only.

  --p1                    model-free: observed specialist's mid-rank MIPS percentile within its specialty-given pool,
                          by severity; Complex − Simple contrast (patient-cluster bootstrap). Train + validation first-observed events.
  --runs DIR [DIR ...]    P2/P3-style allocation tables for trained runs (validation subset, first-observed):
                          top-1 percentile vs observed, top-5 (excl. observed) mean percentile, missing share; by severity, gender, race.
  --ablate-mips           with --runs: inference-time MIPS ablation ΔMRR (specialty pool) by severity (P3).
  --heuristics            add PCP frequency and popularity scorers.

MIPS for every candidate (incl. the observed) comes from the strictly-earlier provider snapshot at the query day
(provider_snapshot), three states: value / missing / ambiguous; missing and ambiguous never become numbers here.
Percentile = (n_below + 0.5 * n_tied_excl_self) / (n - 1) among non-missing, non-ambiguous pool candidates; pools with
< 10 such candidates are excluded from percentile summaries; targets with unknown specialty are excluded.
Writes runs/analysis/mips_*.json / .csv.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd


from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.evaluate import ModelScorer, evaluate, metrics  # noqa: E402
from referral_gnn.baselines import PCPFrequency, Popularity  # noqa: E402
from referral_gnn.train import pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", default="prepared_data/three_node_v4/model_inputs_PRIVATE.pkl")
ap.add_argument("--p1", action="store_true"); ap.add_argument("--runs", nargs="*", default=[]); ap.add_argument("--ablate-mips", action="store_true"); ap.add_argument("--heuristics", action="store_true")
ap.add_argument("--bootstrap", type=int, default=1000); ap.add_argument("--min-pool", type=int, default=10)
ap.add_argument("--split", default="validation", choices=["validation", "test"])
args = ap.parse_args()
root = str(workspace())
mi = load_three_node_inputs(os.path.join(root, args.inputs) if not os.path.isabs(args.inputs) else args.inputs)
device = pick_device("auto"); out_dir = os.path.join(root, "runs", "analysis"); os.makedirs(out_dir, exist_ok=True)
V, MISS, AMB = mi.mips_cols
SEV = {v: k for k, v in mi.severity_vocab.index.items()}; RACE = {v: k for k, v in mi.race_vocab.index.items()}; GEN = {v: k for k, v in mi.patient_vocabs["gender"].index.items()}

_snap_cache = {}
def mips_state(day):
    """(value, state) per specialist as of strictly before `day`; state 0=value, 1=missing, 2=ambiguous."""
    if day not in _snap_cache:
        if len(_snap_cache) > 64: _snap_cache.clear()
        x = mi.provider_snapshot(day)["specialist_num"]
        state = np.where(x[:, AMB] == 1, 2, np.where(x[:, MISS] == 1, 1, 0))
        raw = x[:, V] * mi.spec_encoder.scalers["mips_score"].std + mi.spec_encoder.scalers["mips_score"].mean
        _snap_cache[day] = (raw, state)
    return _snap_cache[day]

def midrank_pct(values, i):
    """mid-rank percentile of values[i] among values (all non-missing)."""
    v = values[i]; n = len(values)
    if n < 2: return np.nan
    below = (values < v).sum(); tied = (values == v).sum() - 1
    return (below + 0.5 * tied) / (n - 1)

def pool_for(q):
    """specialty-given unseen pool for query q; None if target specialty unknown."""
    d = int(mi.enc_day[q]); t = int(mi.enc_target[q]); avail = np.where(mi.available_specialists(d))[0]
    seen = mi.ps_asof.lookup(mi.enc_patient[q] * mi.n_specialist + avail, d) >= 0; avail = avail[~seen]
    sp_t = mi.specialty_asof([t], d)[0]
    if sp_t == 0: return None
    return avail[mi.specialty_asof(avail, d) == sp_t]

def strata_of(q):
    return {"severity": SEV.get(int(mi.enc_severity[q]), "UNK"), "gender": GEN.get(int(mi.enc_patient_cat[q, 0]), "UNK"), "race": RACE.get(int(mi.enc_race[q]), "UNK")}

def cluster_boot(vals, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed); s = pd.Series(vals).groupby(groups).agg(["sum", "count"]); S, C = s["sum"].to_numpy(), s["count"].to_numpy(); G = len(S)
    b = np.array([S[idx].sum() / C[idx].sum() for idx in (rng.integers(0, G, G) for _ in range(n))]); return float(np.mean(vals)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def contrast_boot(a_vals, a_groups, b_vals, b_groups, n=1000, seed=0):
    """bootstrap of mean(a) - mean(b) with independent patient-cluster resampling in each stratum."""
    rng = np.random.default_rng(seed)
    def stat(vals, groups, r):
        s = pd.Series(vals).groupby(groups).agg(["sum", "count"]); S, C = s["sum"].to_numpy(), s["count"].to_numpy(); idx = r.integers(0, len(S), len(S)); return S[idx].sum() / C[idx].sum()
    b = np.array([stat(a_vals, a_groups, rng) - stat(b_vals, b_groups, rng) for _ in range(n)])
    return float(np.mean(a_vals) - np.mean(b_vals)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

# ---------------------------------------------------------------- P1: model-free
if args.p1:
    q_all = np.concatenate([mi.query_enc["train"], mi.query_enc["validation"]])
    fo = q_all[mi.ps_asof.lookup(mi.enc_patient[q_all] * mi.n_specialist + mi.enc_target[q_all], mi.enc_day[q_all]) < 0]
    rows = []; n_unknown_sp = 0; n_small = 0
    for q in fo:
        q = int(q); pool = pool_for(q)
        if pool is None: n_unknown_sp += 1; continue
        t = int(mi.enc_target[q])
        if t not in set(pool.tolist()): continue
        raw, st = mips_state(int(mi.enc_day[q]))
        ok = st[pool] == 0; vals = raw[pool][ok]; n_ok = int(ok.sum())
        rec = {"enc": q, "patient": int(mi.enc_patient[q]), "split": mi.enc_split[q], **strata_of(q), "pool": len(pool), "n_mips": n_ok, "pool_missing_share": 1 - n_ok / max(len(pool), 1),
               "obs_state": int(st[t]), "obs_raw": float(raw[t]) if st[t] == 0 else np.nan, "obs_pct": np.nan, "tied_share": np.nan}
        if st[t] == 0 and n_ok >= args.min_pool:
            i = np.where(pool[ok] == t)[0][0]; rec["obs_pct"] = midrank_pct(vals, i); rec["tied_share"] = float((vals == vals[i]).sum() - 1) / max(n_ok - 1, 1)
        elif n_ok < args.min_pool: n_small += 1
        rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out_dir, "mips_p1_per_query.csv"), index=False)
    res = {"n_firstobs_events": int(len(fo)), "excluded_unknown_target_specialty": n_unknown_sp, "excluded_small_pool": n_small,
           "observed_mips_state_share": df.obs_state.map({0: "value", 1: "missing", 2: "ambiguous"}).value_counts(normalize=True).round(3).to_dict(), "by_severity": {}}
    print(f"P1 (model-free, train+validation first-observed events n={len(fo):,}; excluded unknown-specialty {n_unknown_sp}, small pools {n_small})")
    print(f"   observed specialist MIPS state: {res['observed_mips_state_share']}")
    print(f"{'severity':10s} {'n':>6s} {'obs pct mean [95% CI]':>28s} {'median':>7s} {'missing obs':>11s} {'pool missing':>12s} {'tied share':>10s}")
    for sev in ("Simple", "Moderate", "Complex"):
        d = df[(df.severity == sev)]; dv = d.dropna(subset=["obs_pct"])
        m, lo, hi = cluster_boot(dv.obs_pct.to_numpy(), dv.patient.to_numpy(), args.bootstrap)
        res["by_severity"][sev] = {"n": int(len(d)), "n_with_pct": int(len(dv)), "obs_pct_mean": m, "ci": [lo, hi], "obs_pct_median": float(dv.obs_pct.median()), "obs_missing_share": float((d.obs_state > 0).mean()), "pool_missing_share": float(d.pool_missing_share.mean()), "tied_share": float(dv.tied_share.mean())}
        print(f"{sev:10s} {len(d):6d} {m:8.3f} [{lo:.3f}, {hi:.3f}]        {dv.obs_pct.median():7.3f} {float((d.obs_state > 0).mean()):11.1%} {float(d.pool_missing_share.mean()):12.1%} {float(dv.tied_share.mean()):10.3f}")
    a = df[(df.severity == "Complex")].dropna(subset=["obs_pct"]); b = df[(df.severity == "Simple")].dropna(subset=["obs_pct"])
    m, lo, hi = contrast_boot(a.obs_pct.to_numpy(), a.patient.to_numpy(), b.obs_pct.to_numpy(), b.patient.to_numpy(), args.bootstrap)
    res["primary_contrast_complex_minus_simple"] = {"diff": m, "ci": [lo, hi]}
    print(f"PRIMARY P1 contrast Complex − Simple (observed MIPS percentile): {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # exploratory: by disease category (top 10) and split
    cat_names = {v: k for k, v in mi.dx_vocab.index.items()}
    df["cat0"] = [cat_names.get(int(mi.enc_dx[q, 0]), "UNK") for q in df.enc]
    top = df.cat0.value_counts().head(10).index
    res["exploratory_by_category"] = {c: {"n": int((df.cat0 == c).sum()), "obs_pct_mean": float(df[df.cat0 == c].obs_pct.mean())} for c in top}
    res["by_split"] = {s: float(df[df.split == s].obs_pct.mean()) for s in ("train", "validation")}
    json.dump(res, open(os.path.join(out_dir, "mips_p1.json"), "w"), indent=1)
    print("saved runs/analysis/mips_p1.json")

# ---------------------------------------------------------------- P2 / P3: trained runs and heuristics
if args.runs or args.heuristics:
    qv = np.sort(mi.validation_sample) if args.split == "validation" else np.sort(mi.query_enc["test"]); fo = qv[mi.ps_asof.lookup(mi.enc_patient[qv] * mi.n_specialist + mi.enc_target[qv], mi.enc_day[qv]) < 0]
    scorers = {}
    for r in args.runs:
        man = json.load(open(os.path.join(r, "run_manifest.json"))); mi_r = mi if man["inputs_path"].endswith(os.path.basename(args.inputs)) and man.get("schema") == mi.schema_version else load_three_node_inputs(man["inputs_path"])
        model = load_three_node_model(r, mi_r, device); name = os.path.basename(r).split("_", 1)[1]
        scorers[name] = (ModelScorer(model, mi_r, device), mi_r, man["train_config"]["graph_bucket"], model)
        if args.ablate_mips and not man["model_config"].get("mask_demographics"):
            # shallow copy of the inputs whose INSTANCE attribute provider_snapshot shadows the class method, so that
            # graph_batch (which calls self.provider_snapshot) sees MIPS forced to missing for every specialist (P3)
            mi_abl = copy.copy(mi_r); base_snap = mi_r.provider_snapshot
            def _ablated_snapshot(day, _base=base_snap):
                s = _base(day); x = s["specialist_num"]; x[:, V] = 0.0; x[:, MISS] = 1.0; x[:, AMB] = 0.0; return s
            mi_abl.provider_snapshot = _ablated_snapshot
            scorers[name + ":mips_ablated"] = (ModelScorer(copy.deepcopy(model), mi_abl, device), mi_abl, man["train_config"]["graph_bucket"], None)
    if args.heuristics:
        scorers["pcp_frequency"] = (PCPFrequency(mi), mi, "week", None); scorers["popularity"] = (Popularity(mi), mi, "week", None)
    rows = []
    for name, (sc, mi_r, bucket, _) in scorers.items():
        for cutoff, batch_enc, q_b in mi_r.buckets(args.split, bucket, query_subset=fo):
            sc.prepare(cutoff, batch_enc)
            for q in q_b:
                q = int(q); pool = pool_for(q)
                if pool is None: continue
                t = int(mi.enc_target[q])
                if t not in set(pool.tolist()): continue            # target not yet available on the query day (counted as unavailable elsewhere)
                raw, st = mips_state(int(mi.enc_day[q]))
                s = sc.score(q, pool); order = np.argsort(-s, kind="stable"); ranked = pool[order]
                rank = int((s > s[pool == t][0]).sum() + (s == s[pool == t][0]).sum())
                ok = st[pool] == 0; vals = raw[pool][ok]; n_ok = int(ok.sum()); okset = set(pool[ok].tolist())
                def pct_of(c):
                    if c not in okset or n_ok < args.min_pool: return np.nan
                    return midrank_pct(vals, np.where(pool[ok] == c)[0][0])
                top1 = int(ranked[0]); top5x = [int(c) for c in ranked[:6] if c != t][:5]
                rec = {"scorer": name, "enc": q, "patient": int(mi.enc_patient[q]), **strata_of(q), "pool": len(pool), "n_mips": n_ok, "rank": rank,
                       "obs_pct": pct_of(t), "top1_pct": pct_of(top1), "top1_missing": int(st[top1] > 0), "top5x_pct_mean": float(np.nanmean([pct_of(c) for c in top5x])) if top5x else np.nan,
                       "top5x_missing_share": float(np.mean([st[c] > 0 for c in top5x])) if top5x else np.nan}
                rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out_dir, "mips_p2_per_query" + ("" if args.split == "validation" else "_" + args.split) + ".csv"), index=False)
    summ = []
    for name in scorers:
        d = df[df.scorer == name]
        base = {"scorer": name, "n": len(d), "MRR": float((1 / d["rank"]).mean()), "R@5": float((d["rank"] <= 5).mean()), "R@10": float((d["rank"] <= 10).mean()),
                "obs_pct": float(d.obs_pct.mean()), "top1_pct": float(d.top1_pct.mean()), "top1_missing": float(d.top1_missing.mean()), "top5x_pct": float(d.top5x_pct_mean.mean()), "top5x_missing": float(d.top5x_missing_share.mean())}
        summ.append({**base, "stratum": "all"})
        for col in ("severity", "gender", "race"):
            for val, dd in d.groupby(col):
                if dd.patient.nunique() < 100: continue
                summ.append({"scorer": name, "stratum": f"{col}={val}", "n": len(dd), "MRR": float((1 / dd["rank"]).mean()), "R@5": float((dd["rank"] <= 5).mean()), "R@10": float((dd["rank"] <= 10).mean()),
                             "obs_pct": float(dd.obs_pct.mean()), "top1_pct": float(dd.top1_pct.mean()), "top1_missing": float(dd.top1_missing.mean()), "top5x_pct": float(dd.top5x_pct_mean.mean()), "top5x_missing": float(dd.top5x_missing_share.mean())})
    S = pd.DataFrame(summ); S.to_csv(os.path.join(out_dir, "mips_p2_summary" + ("" if args.split == "validation" else "_" + args.split) + ".csv"), index=False)
    print(f"\nP2/P3 allocation on the specialty-given pool ({args.split} first-observed):")
    print(S[S.stratum == "all"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nby severity:"); print(S[S.stratum.str.startswith("severity=")].sort_values(["stratum", "scorer"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # paired contrasts between runs (same queries): MRR and top1_pct, patient-cluster bootstrap
    names = [n for n in scorers if ":" not in n and n not in ("pcp_frequency", "popularity")]
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = df[df.scorer == names[i]].set_index("enc"); b = df[df.scorer == names[j]].set_index("enc"); idx = a.index.intersection(b.index)
            for metric, fa, fb in (("MRR", 1 / a.loc[idx, "rank"], 1 / b.loc[idx, "rank"]), ("top1_pct", a.loc[idx, "top1_pct"], b.loc[idx, "top1_pct"])):
                m = (fa - fb).to_numpy(float); ok = np.isfinite(m); mean, lo, hi = cluster_boot(m[ok], a.loc[idx, "patient"].to_numpy()[ok], args.bootstrap)
                out.append({"a": names[i], "b": names[j], "metric": metric, "diff": mean, "ci_lo": lo, "ci_hi": hi, "n": int(ok.sum())})
    if out:
        C = pd.DataFrame(out); C.to_csv(os.path.join(out_dir, "mips_p2_contrasts" + ("" if args.split == "validation" else "_" + args.split) + ".csv"), index=False); print("\npaired contrasts (a − b):"); print(C.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    if args.ablate_mips:
        for name in [n for n in scorers if n.endswith(":mips_ablated")]:
            base = name.split(":")[0]; a = df[df.scorer == base].set_index("enc"); b = df[df.scorer == name].set_index("enc"); idx = a.index.intersection(b.index)
            print(f"\nP3 MIPS ablation ΔMRR ({base} − ablated) by severity:")
            for sev in ("Simple", "Moderate", "Complex"):
                ii = idx[a.loc[idx, "severity"] == sev]; d = (1 / a.loc[ii, "rank"] - 1 / b.loc[ii, "rank"]).to_numpy(float)
                m, lo, hi = cluster_boot(d, a.loc[ii, "patient"].to_numpy(), args.bootstrap); print(f"   {sev:10s} n={len(ii):5d} Δ {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            ic = idx[a.loc[idx, "severity"] == "Complex"]; isim = idx[a.loc[idx, "severity"] == "Simple"]
            m, lo, hi = contrast_boot((1 / a.loc[ic, "rank"] - 1 / b.loc[ic, "rank"]).to_numpy(float), a.loc[ic, "patient"].to_numpy(), (1 / a.loc[isim, "rank"] - 1 / b.loc[isim, "rank"]).to_numpy(float), a.loc[isim, "patient"].to_numpy(), args.bootstrap)
            print(f"   PRIMARY P3 contrast (Complex − Simple reliance): {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("saved runs/analysis/mips_p2_*.csv")
