"""Formal failure audit of a trained three-node checkpoint on the validation subset.

Sections (all aggregate; no identifiers printed):
 1. rank anatomy by cohort (first-observed / repeat)
 2. specialty vs person: specialty of top-K vs target; rank within the target's specialty
 3. familiarity dominance: PCP-familiar share of top-10 vs of the pool; target familiarity
 4. popularity concentration of top-10 recommendations vs the pool
 5. distance: PCP-specialist distance of top-10 vs target
 6. inference-time ablations (input blocks, pair-feature groups, edge types) -> first-observed MRR
 7. gradient x input attribution per input block for the target score

Writes runs/analysis/failure_audit_<run>.json and prints a summary.
Usage: failure_audit.py <run_dir> [--checkpoint best.pt] [--n-grad 400]
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
import torch


from referral_gnn.data import AsOfIndex  # noqa: E402
from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.three_node_model import EDGE_TYPES, TEMPORAL_EDGE  # noqa: E402
from referral_gnn.evaluate import ModelScorer, evaluate, metrics  # noqa: E402
from referral_gnn.baselines import Popularity  # noqa: E402
from referral_gnn.train import pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("--checkpoint", default="best.pt"); ap.add_argument("--n-grad", type=int, default=400)
args = ap.parse_args()
root = str(workspace())
man = json.load(open(os.path.join(args.run_dir, "run_manifest.json")))
mi = load_three_node_inputs(man["inputs_path"]); device = pick_device("auto")
model = load_three_node_model(args.run_dir, mi, device, args.checkpoint); model.eval()
bucket = man["train_config"]["graph_bucket"]
out = {"run": os.path.basename(args.run_dir), "checkpoint": args.checkpoint}
P = lambda *a: print(*a, flush=True)

# cutoff-safe specialty lookup
sh_rows = mi.spec_hist_cat[:, 0]; nonnull = sh_rows > 0
ent = mi.spec_hist_asof.entity[np.argsort(mi.spec_hist_asof.order)]
sp_idx = AsOfIndex(ent[nonnull], mi.spec_hist_day[nonnull]); sp_val = sh_rows[nonnull]
def specialty_asof(spec_ids, d):
    r = sp_idx.lookup(np.asarray(spec_ids), int(d) + 1); return np.where(r >= 0, sp_val[np.clip(r, 0, None)], 0)
pop = Popularity(mi)

# ---------------- per-query pass ------------------------------------------------------------------
scorer = ModelScorer(model, mi, device)
queries = np.sort(mi.validation_sample)
rec = []
for cutoff, batch_enc, q_b in mi.buckets("validation", bucket, query_subset=queries):
    scorer.prepare(cutoff, batch_enc)
    for q in q_b:
        q = int(q); t = int(mi.enc_target[q]); d = int(mi.enc_day[q])
        avail = np.where(mi.available_specialists(d))[0]
        pos = {int(s): i for i, s in enumerate(avail)}
        if t not in pos: continue
        s = scorer.score(q, avail); order = np.argsort(-s, kind="stable")
        rank = int((s > s[pos[t]]).sum() + (s == s[pos[t]]).sum())         # pessimistic
        top10 = avail[order[:10]]
        sp_t = specialty_asof([t], d)[0]; sp_all = specialty_asof(avail, d)
        same = avail[sp_all == sp_t] if sp_t > 0 else np.array([], int)
        rank_within = int((s[sp_all == sp_t] > s[pos[t]]).sum() + 1) if sp_t > 0 else None
        prior_all, _, _ = mi.raw_pair_history(np.full(len(avail), q), avail)
        fam_pool = prior_all > 0
        prior_top = prior_all[order[:10]]
        popc = pop._pop(q)
        pcp_dist = mi.distance[mi.enc_pcp[q], :]
        repeat = bool(mi.ps_asof.lookup(np.array([mi.enc_patient[q] * mi.n_specialist + t]), d)[0] >= 0)
        rec.append({
            "enc": q, "repeat": repeat, "rank": rank, "n_pool": len(avail),
            "target_specialty_known": sp_t > 0, "n_same_specialty": int(len(same)), "rank_within_specialty": rank_within,
            "top1_specialty_match": bool(sp_t > 0 and sp_all[order[0]] == sp_t),
            "top10_specialty_match_count": int((sp_all[order[:10]] == sp_t).sum()) if sp_t > 0 else 0,
            "target_familiar": bool(prior_all[pos[t]] > 0), "fam_share_pool": float(fam_pool.mean()), "fam_share_top10": float((prior_top > 0).mean()),
            "target_pop_pct": float((popc[avail] < popc[t]).mean()), "top10_pop_pct_mean": float(np.mean([(popc[avail] < popc[c]).mean() for c in top10])),
            "target_pcp_dist": float(pcp_dist[t]) if np.isfinite(pcp_dist[t]) else np.nan, "top10_pcp_dist_median": float(np.nanmedian(pcp_dist[top10])),
            "top10": top10.tolist(),
        })
df = pd.DataFrame(rec)
fo, rp = df[~df.repeat], df[df.repeat]
out["n_eligible"] = len(df); out["n_firstobs"] = len(fo); out["n_repeat"] = len(rp)

# 1. rank anatomy
def anatomy(d):
    r = d["rank"].to_numpy(float)
    return {"MRR": float((1 / r).mean()), "median_rank": float(np.median(r)), "p25": float(np.percentile(r, 25)), "p75": float(np.percentile(r, 75)),
            **{f"share_top{k}": float((r <= k).mean()) for k in (1, 5, 10, 20, 50, 100, 500)}, "median_pool": float(d["n_pool"].median())}
out["rank_anatomy"] = {"firstobs": anatomy(fo), "repeat": anatomy(rp)}
P("\n[1] rank anatomy"); P(pd.DataFrame(out["rank_anatomy"]).T.to_string(float_format=lambda x: f"{x:.3f}"))

# 2. specialty vs person
fk = fo[fo.target_specialty_known]
out["specialty_vs_person"] = {
    "firstobs_top1_specialty_match": float(fk.top1_specialty_match.mean()),
    "firstobs_mean_top10_in_target_specialty": float(fk.top10_specialty_match_count.mean()),
    "firstobs_median_same_specialty_pool": float(fk.n_same_specialty.median()),
    "firstobs_MRR_within_target_specialty(oracle)": float((1 / fk.rank_within_specialty).mean()),
    "firstobs_share_top1_within_specialty(oracle)": float((fk.rank_within_specialty == 1).mean()),
    "firstobs_share_top5_within_specialty(oracle)": float((fk.rank_within_specialty <= 5).mean()),
    "repeat_MRR_within_target_specialty(oracle)": float((1 / rp[rp.target_specialty_known].rank_within_specialty).mean()),
    "firstobs_rank_conditional_on_top1_specialty_correct": float(fk[fk.top1_specialty_match]["rank"].median()),
    "firstobs_rank_conditional_on_top1_specialty_wrong": float(fk[~fk.top1_specialty_match]["rank"].median()),
}
P("\n[2] specialty vs person"); [P(f"   {k}: {v:.3f}") for k, v in out["specialty_vs_person"].items()]

# 3. familiarity dominance
out["familiarity"] = {
    "firstobs_target_familiar_share": float(fo.target_familiar.mean()),
    "firstobs_fam_share_pool_mean": float(fo.fam_share_pool.mean()), "firstobs_fam_share_top10_mean": float(fo.fam_share_top10.mean()),
    "firstobs_MRR_target_familiar": float((1 / fo[fo.target_familiar]["rank"]).mean()), "firstobs_MRR_target_unfamiliar": float((1 / fo[~fo.target_familiar]["rank"]).mean()),
    "firstobs_median_rank_target_unfamiliar": float(fo[~fo.target_familiar]["rank"].median()),
    "repeat_fam_share_top10_mean": float(rp.fam_share_top10.mean()),
}
P("\n[3] familiarity"); [P(f"   {k}: {v:.3f}") for k, v in out["familiarity"].items()]

# 4. popularity concentration
all_top = np.concatenate([np.array(x) for x in df.top10]); uniq, cnt = np.unique(all_top, return_counts=True)
cnt_sorted = np.sort(cnt)[::-1]; share_top50 = cnt_sorted[:50].sum() / cnt.sum()
gini = lambda x: (np.abs(np.subtract.outer(x, x)).sum() / (2 * len(x) ** 2 * x.mean()))
targets = df["enc"].map(lambda e: int(mi.enc_target[e])).to_numpy(); ut, ct = np.unique(targets, return_counts=True)
out["popularity"] = {
    "firstobs_target_popularity_percentile_mean": float(fo.target_pop_pct.mean()), "firstobs_top10_popularity_percentile_mean": float(fo.top10_pop_pct_mean.mean()),
    "distinct_specialists_in_all_top10": int(len(uniq)), "share_of_top10_slots_taken_by_50_most_recommended": float(share_top50),
    "gini_top10_slots": float(gini(cnt.astype(float))), "gini_true_targets": float(gini(ct.astype(float))), "distinct_true_targets": int(len(ut)),
}
P("\n[4] popularity"); [P(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}") for k, v in out["popularity"].items()]

# 5. distance
out["distance"] = {"firstobs_target_pcp_dist_median": float(fo.target_pcp_dist.median()), "firstobs_top10_pcp_dist_median_of_medians": float(fo.top10_pcp_dist_median.median()),
                   "repeat_target_pcp_dist_median": float(rp.target_pcp_dist.median())}
P("\n[5] PCP-specialist distance (miles)"); [P(f"   {k}: {v:.1f}") for k, v in out["distance"].items()]

# 6. inference-time ablations
fo_q = fo["enc"].to_numpy(); rp_q = rp["enc"].to_numpy()
def mrr_for(m, qs):
    d = evaluate(ModelScorer(m, mi, device), mi, "validation", query_subset=qs, bucket=bucket); return metrics(d)["mrr_eligible"]
base_fo, base_rp = mrr_for(model, fo_q), mrr_for(model, rp_q)
abl = {"none": (base_fo, base_rp)}
flags = ["no_history_features", "no_patient_specialist_prior", "no_distance", "no_diagnosis", "no_patient_history", "no_time_encoding", "no_predecessor_gap", "no_referral_edges"]
for f in flags:
    m2 = copy.deepcopy(model); setattr(m2.cfg, f, True); abl[f] = (mrr_for(m2, fo_q), mrr_for(m2, rp_q))
m2 = copy.deepcopy(model); m2.cfg.temporal_edges = False; abl["no_precedes_edges"] = (mrr_for(m2, fo_q), mrr_for(m2, rp_q))
m2 = copy.deepcopy(model); m2.active_edge_types = lambda: [et for et in EDGE_TYPES if "has_encounter" not in et[1]]; abl["no_has_encounter_edges"] = (mrr_for(m2, fo_q), mrr_for(m2, rp_q))
m2 = copy.deepcopy(model); m2.cfg.use_graph = False; abl["no_message_passing_at_all"] = (mrr_for(m2, fo_q), mrr_for(m2, rp_q))
out["inference_ablation"] = {k: {"firstobs_MRR": v[0], "repeat_MRR": v[1], "firstobs_delta": v[0] - base_fo, "repeat_delta": v[1] - base_rp} for k, v in abl.items()}
P("\n[6] inference-time ablations (trained weights; input/edges removed at test time)")
P(pd.DataFrame(out["inference_ablation"]).T.to_string(float_format=lambda x: f"{x:+.4f}"))

# 7. gradient x input per block (target score), first-observed sample
blocks = dict(mi.encounter_blocks); pair_blocks = mi.pair_feature_blocks
rng = np.random.default_rng(0); sample = rng.choice(fo_q, size=min(args.n_grad, len(fo_q)), replace=False)
agg_enc = {b: [] for b in blocks}; agg_pair = {b: [] for b in pair_blocks}; agg_dx = []
for cutoff, batch_enc, q_b in mi.buckets("validation", bucket, query_subset=np.sort(sample)):
    bt = mi.graph_batch(cutoff, batch_enc, device=device)
    x_enc = bt["x"]["encounter_num"].clone().requires_grad_(True); bt["x"]["encounter_num"] = x_enc
    h = model.encode(bt)
    e = torch.as_tensor(q_b, device=device); t = torch.as_tensor(mi.enc_target[q_b], device=device)
    pf = torch.as_tensor(mi.pair_features(q_b, mi.enc_target[q_b]), device=device).requires_grad_(True)
    sc = model.score(h, e, t, pf); sc.sum().backward()
    gx = (x_enc.grad[e] * x_enc[e]).abs().detach().cpu().numpy(); gp = (pf.grad * pf).abs().detach().cpu().numpy()
    for b, (a, z) in blocks.items(): agg_enc[b].append(gx[:, a:z].sum(1))
    for b, cols in pair_blocks.items(): agg_pair[b].append(gp[:, cols].sum(1))
enc_attr = {b: float(np.concatenate(v).mean()) for b, v in agg_enc.items()}; pair_attr = {b: float(np.concatenate(v).mean()) for b, v in agg_pair.items()}
tot = sum(enc_attr.values()) + sum(pair_attr.values())
out["grad_x_input_share"] = {**{f"encounter:{k}": v / tot for k, v in enc_attr.items()}, **{f"pair:{k}": v / tot for k, v in pair_attr.items()}}
P("\n[7] |gradient x input| share of target score, first-observed sample (encounter numeric blocks + pair features; category bag not included)")
[P(f"   {k:32s} {v:6.1%}") for k, v in sorted(out["grad_x_input_share"].items(), key=lambda kv: -kv[1])]

os.makedirs(os.path.join(root, "runs", "analysis"), exist_ok=True)
json.dump(out, open(os.path.join(root, "runs", "analysis", f"failure_audit_{os.path.basename(args.run_dir)}_{args.checkpoint.replace('.pt','')}.json"), "w"), indent=1, default=float)
P("\nsaved runs/analysis/failure_audit_*.json")
