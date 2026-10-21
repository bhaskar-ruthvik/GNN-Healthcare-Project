"""Quality-aware recommendation lists (exploratory; not in the frozen Step 9 protocol).

Question (professor, 2026-09-24): can the recommended specialists have higher MIPS than the specialist actually
referred, while staying consistent with the case? Consistency is enforced by only choosing among candidates the
model itself ranks in its top K within the chosen specialty (specialty-given, unseen-by-patient pool).

For each first-time referral and each K: the 5-specialist list = the 5 highest-MIPS specialists among the model's
top K (missing/ambiguous MIPS treated as lowest, per the professor's "missing = 0"; ties by model score).
K = 5 is the model's own list (same five, reordered). Reported per K:
  hit rate      share of referrals whose actual specialist is in the 5-list
  top-1 pct     within-specialty MIPS percentile of the list's first specialist (mid-rank among scored candidates)
  list pct      mean percentile over the list's scored members
  list raw      mean raw MIPS over scored members;  missing share of the list
compared with the actually referred specialist (percentile / raw), with patient-cluster bootstrap CIs.
MIPS read from the strictly-earlier provider snapshot (same as the model); pools with < 10 scored candidates skipped.
Usage: mips_rerank.py <run_dir> [...] [--split validation|test] [--ks 5,10,20]
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, json, os, sys
import numpy as np, pandas as pd

from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.evaluate import ModelScorer  # noqa: E402
from referral_gnn.train import pick_device  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--split", default="validation"); ap.add_argument("--ks", default="5,10,20")
ap.add_argument("--list-size", type=int, default=5); ap.add_argument("--min-pool", type=int, default=10); ap.add_argument("--bootstrap", type=int, default=1000)
args = ap.parse_args(); KS = [int(k) for k in args.ks.split(",")]; L = args.list_size
root = str(workspace()); device = pick_device("auto")

def boot(vals, groups, n):
    r = np.random.default_rng(0); s = pd.Series(vals).groupby(groups).agg(["sum", "count"]); S, C = s["sum"].to_numpy(), s["count"].to_numpy(); G = len(S)
    b = np.array([S[i].sum() / C[i].sum() for i in (r.integers(0, G, G) for _ in range(n))]); return float(np.mean(vals)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

out_rows = []
for run in args.runs:
    man = json.load(open(os.path.join(run, "run_manifest.json"))); mi = load_three_node_inputs(man["inputs_path"]); model = load_three_node_model(run, mi, device)
    sc = ModelScorer(model, mi, device); bucket = man["train_config"]["graph_bucket"]; name = os.path.basename(run).split("_", 1)[1]
    V, MISS, AMB = mi.mips_cols; mu, sd = mi.spec_encoder.scalers["mips_score"].mean, mi.spec_encoder.scalers["mips_score"].std
    q_all = np.sort(mi.validation_sample) if args.split == "validation" else np.sort(mi.query_enc[args.split])
    fo = q_all[mi.ps_asof.lookup(mi.enc_patient[q_all] * mi.n_specialist + mi.enc_target[q_all], mi.enc_day[q_all]) < 0]
    snap_cache = {}; rows = []
    for cutoff, batch_enc, q_b in mi.buckets(args.split, bucket, query_subset=fo):
        sc.prepare(cutoff, batch_enc)
        for q in q_b:
            q = int(q); t = int(mi.enc_target[q]); d = int(mi.enc_day[q])
            sp_t = mi.specialty_asof([t], d)[0]
            if sp_t == 0: continue
            pool = np.where(mi.available_specialists(d))[0]; pool = pool[mi.ps_asof.lookup(mi.enc_patient[q] * mi.n_specialist + pool, d) < 0]
            pool = pool[mi.specialty_asof(pool, d) == sp_t]
            if t not in set(pool.tolist()): continue
            if d not in snap_cache:
                if len(snap_cache) > 64: snap_cache.clear()
                x = mi.provider_snapshot(d)["specialist_num"]; snap_cache[d] = ((x[:, V] * sd + mu), (x[:, MISS] == 1) | (x[:, AMB] == 1))
            raw, bad = snap_cache[d]
            ok = ~bad[pool]; vals = np.sort(raw[pool][ok]); n_ok = len(vals)
            if n_ok < args.min_pool: continue
            def pct(c):
                if bad[c]: return np.nan
                v = raw[c]; lo = np.searchsorted(vals, v, "left"); hi = np.searchsorted(vals, v, "right"); return (lo + 0.5 * (hi - lo - 1)) / (n_ok - 1)
            s = sc.score(q, pool); order = np.argsort(-s, kind="stable"); ranked = pool[order]
            rec = {"enc": q, "patient": int(mi.enc_patient[q]), "obs_pct": pct(t), "obs_raw": np.nan if bad[t] else float(raw[t]), "obs_missing": bool(bad[t]), "pool": len(pool)}
            for K in KS:
                cand = ranked[:K]; key_mips = np.where(bad[cand], -np.inf, raw[cand])
                pick = cand[np.lexsort((np.arange(len(cand)), -key_mips))][:L]       # MIPS desc (missing last), then model order
                if K == L: pick_top1 = ranked[0]                                         # model's own list keeps its own top-1
                else: pick_top1 = pick[0]
                p = np.array([pct(c) for c in pick]); r = np.array([np.nan if bad[c] else raw[c] for c in pick])
                rec |= {f"K{K}_hit": bool(t in set(pick.tolist())), f"K{K}_top1_pct": pct(pick_top1), f"K{K}_list_pct": float(np.nanmean(p)) if np.isfinite(p).any() else np.nan,
                        f"K{K}_list_raw": float(np.nanmean(r)) if np.isfinite(r).any() else np.nan, f"K{K}_list_missing": float(np.mean([bad[c] for c in pick]))}
            rows.append(rec)
    df = pd.DataFrame(rows); os.makedirs(os.path.join(root, "runs", "analysis"), exist_ok=True)
    df.to_csv(os.path.join(root, "runs", "analysis", f"mips_rerank_{name}_{args.split}.csv"), index=False)
    print(f"\n{name} | {args.split} | first-time referrals with specialty pool of >= {args.min_pool} scored candidates: n={len(df)} | referred specialist MIPS missing {df.obs_missing.mean():.0%}")
    obs_ok = df.dropna(subset=["obs_pct"])
    print(f"   actually referred: MIPS percentile {obs_ok.obs_pct.mean():.3f}, raw mean {obs_ok.obs_raw.mean():.1f}")
    for K in KS:
        lab = "model's own top 5" if K == L else f"5 highest-MIPS among model's top {K}"
        hit = df[f"K{K}_hit"].mean(); dd = df.dropna(subset=[f"K{K}_list_pct", "obs_pct"])
        diff, lo, hi = boot((dd[f"K{K}_list_pct"] - dd.obs_pct).to_numpy(), dd.patient.to_numpy(), args.bootstrap)
        print(f"   {lab:40s} hit {hit:.3f} | top-1 pct {df[f'K{K}_top1_pct'].mean():.3f} | list pct {df[f'K{K}_list_pct'].mean():.3f} (list − referred {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]) | list raw {df[f'K{K}_list_raw'].mean():.1f} | list missing {df[f'K{K}_list_missing'].mean():.0%}")
        out_rows.append({"model": name, "split": args.split, "K": K, "n": len(df), "hit_rate": hit, "top1_pct": df[f"K{K}_top1_pct"].mean(), "list_pct": df[f"K{K}_list_pct"].mean(),
                         "list_minus_referred_pct": diff, "ci_lo": lo, "ci_hi": hi, "list_raw": df[f"K{K}_list_raw"].mean(), "list_missing": df[f"K{K}_list_missing"].mean(),
                         "referred_pct": obs_ok.obs_pct.mean(), "referred_raw": obs_ok.obs_raw.mean(), "referred_missing": df.obs_missing.mean()})
pd.DataFrame(out_rows).to_csv(os.path.join(root, "runs", "analysis", f"mips_rerank_summary_{args.split}.csv"), index=False)
print("saved runs/analysis/mips_rerank_*")
