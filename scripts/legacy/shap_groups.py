"""Grouped Shapley attributions for a three-node checkpoint (permutation sampling; Castro et al. 2009).

Explains f = score(query encounter, candidate specialist) with the historical graph fixed. Only the QUERY
encounter's own inputs, the CANDIDATE specialist's own node inputs, the query's PCP node inputs and the pair
features are perturbed; every other node keeps its real inputs. "Absent" for a group = the training-fitted
missing encoding (values 0 / missing flags 1 / categorical UNK / pair flags "unknown"), never zeros that mean
something. Shapley value of a group = its average marginal contribution over random group orderings; the
values sum exactly to f(all present) − f(all absent) for every permutation.

Sample: fixed, seeded set of first-observed and repeat validation queries (eligible targets only). Candidate =
the observed specialist (and optionally the model's top-1 when it differs). Outputs mean |phi| per group with
patient-cluster bootstrap CIs, per cohort, plus a few local examples (indices only). Latent embeddings are not
attributed; groups are original input groups.

Usage: shap_groups.py <run_dir> [--n-firstobs 30] [--n-repeat 30] [--perms 25] [--seed 0]
Writes <run_dir>/shap_groups/{per_query.csv, summary.json}
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch


from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.train import pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("--n-firstobs", type=int, default=30); ap.add_argument("--n-repeat", type=int, default=30)
ap.add_argument("--perms", type=int, default=25); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--also-top1", action="store_true")
args = ap.parse_args()
man = json.load(open(os.path.join(args.run_dir, "run_manifest.json")))
mi = load_three_node_inputs(man["inputs_path"]); device = pick_device("auto"); model = load_three_node_model(args.run_dir, mi, device); model.eval()
bucket = man["train_config"]["graph_bucket"]; masked_model = bool(man["model_config"].get("mask_demographics", False))
B = mi.encounter_blocks; PB = mi.pair_feature_blocks
schema = mi.schema_version

# ---------------- group definitions (each: how to make the group ABSENT) ----------------
spec_num_cols = mi.spec_encoder.numeric_cols; V, MISS, AMB = getattr(mi, "mips_cols", (3 * spec_num_cols.index("mips_score"),) * 3) if "mips_score" in spec_num_cols else (None, None, None)
if V is not None and not hasattr(mi, "mips_cols"): V, MISS, AMB = V, V + 1, V + 2
def spec_missing_encoding(x_row, exclude_mips=False):
    """set every numeric specialist feature to missing (value 0, missing 1, ambiguous 0) and cats to UNK; optionally leave MIPS."""
    x = x_row.clone()
    for j, c in enumerate(spec_num_cols):
        if exclude_mips and c == "mips_score": continue
        x[3 * j] = 0.0; x[3 * j + 1] = 1.0; x[3 * j + 2] = 0.0
    return x
pair_groups = {"pair_history": PB["referral_history"], "patient_specialist_prior": PB["patient_specialist"], "distance": PB["distance"] + PB.get("patient_distance", []),
               "insurer_history": PB.get("insurer", []), "affiliation": PB.get("affiliation", [])}
def pair_absent(pf, cols):
    pf = pf.clone()
    for c in cols:
        pf[c] = 0.0
    # "unknown" flags carry the absence: recency/first-observed missing flags (2,4), distance missing (6,9), insurer/affiliation unknown (11,13)
    for c in cols:
        if c in (1, 3): pf[c + 1] = 1.0
        if c in (5, 8): pf[c + 1] = 1.0
        if c in (10, 12): pf[c + 1] = 1.0
    return pf
enc_groups = {"diagnosis": ("dx", None), "patient_context": ("block+cat", (B["patient_context"], [0, 1, 2])), "patient_history": ("block", B["patient_history"]),
              "severity": ("block+cat", (B.get("severity_flags", (0, 0)), [3] if mi.enc_patient_cat.shape[1] > 3 else [])),
              "demographics": ("cat+flag", ([0] + ([4] if mi.enc_patient_cat.shape[1] > 4 else []), [3])), "time_and_gap": ("block", (B["time"][0], B["predecessor_gap"][1]))}
GROUPS = list(enc_groups) + ["pcp_attributes", "specialist_attributes", "mips"] + list(pair_groups)
if masked_model: GROUPS.remove("demographics")
if schema in ("three-node-v1", "three-node-v2"): GROUPS = [g for g in GROUPS if g not in ("insurer_history", "affiliation", "severity")]
if schema in ("three-node-v1",): GROUPS.remove("distance"); GROUPS.append("distance")   # keep name; PCP distance only
G = len(GROUPS)

def query_context(q):
    """batch for the query alone at its bucket cutoff; returns tensors and indices."""
    cutoff = [c for c, be, qb in mi.buckets("validation", bucket, query_subset=np.array([q]))][0]
    bt = mi.graph_batch(cutoff, np.array([q]), device=device)
    return bt

def score_with(bt, q, cand, present):
    """present: dict group->bool. Rebuild inputs with absent groups masked, encode, score."""
    x = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bt["x"].items()}
    enc_num = x["encounter_num"]; enc_cat = x["encounter_cat"]; row = enc_num[q]
    for g, (kind, spec) in enc_groups.items():
        if g not in GROUPS or present[g]: continue
        if kind == "dx":
            x["encounter_dx_mask"][q] = False; x["encounter_dx"][q] = 0
            a, b = B["diagnosis_flags"]; row[a:b] = 0.0; row[a] = 1.0        # no-code flag on
        elif kind == "block":
            a, b = spec; row[a:b] = 0.0
        elif kind == "block+cat":
            (a, b), cats = spec; row[a:b] = 0.0
            for c in cats: enc_cat[q, c] = 0
            if g == "patient_context": row[1] = 1.0                            # age missing flag
        elif kind == "cat+flag":
            cats, flags = spec
            for c in cats: enc_cat[q, c] = 0
            for c in flags: row[c] = 0.0
    if not present.get("pcp_attributes", True):
        p = int(mi.enc_pcp[q]); xp = x["pcp_num"][p]
        for j in range(len(mi.pcp_encoder.numeric_cols)): xp[3 * j] = 0.0; xp[3 * j + 1] = 1.0; xp[3 * j + 2] = 0.0
        x["pcp_cat"][p] = 0
    if not present.get("specialist_attributes", True):
        x["specialist_num"][cand] = spec_missing_encoding(x["specialist_num"][cand], exclude_mips=True); x["specialist_cat"][cand] = 0
    if not present.get("mips", True) and V is not None:
        xs = x["specialist_num"][cand]; xs[V] = 0.0; xs[MISS] = 1.0; xs[AMB] = 0.0
    bt2 = dict(bt); bt2["x"] = x
    with torch.no_grad():
        h = model.encode(bt2)
        pf = torch.as_tensor(mi.pair_features(np.array([q]), np.array([cand])), device=device)[0]
        for g, cols in pair_groups.items():
            if g in GROUPS and not present[g]: pf = pair_absent(pf, cols)
        return float(model.score(h, torch.tensor([q], device=device), torch.tensor([cand], device=device), pf[None, :])[0])

def shapley(bt, q, cand, rng):
    phi = np.zeros(G); f_none = score_with(bt, q, cand, {g: False for g in GROUPS}); f_all = score_with(bt, q, cand, {g: True for g in GROUPS})
    for _ in range(args.perms):
        order = rng.permutation(G); present = {g: False for g in GROUPS}; prev = f_none
        for gi in order:
            present[GROUPS[gi]] = True; cur = score_with(bt, q, cand, present); phi[gi] += cur - prev; prev = cur
    phi /= args.perms
    return phi, f_none, f_all

# ---------------- sample ----------------
rng = np.random.default_rng(args.seed)
qv = np.sort(mi.validation_sample); d = mi.enc_day[qv]; t = mi.enc_target[qv]
avail = np.array([mi.available_specialists(int(dd))[int(tt)] for dd, tt in zip(d, t)])
rep = mi.ps_asof.lookup(mi.enc_patient[qv] * mi.n_specialist + t, d) >= 0
fo = qv[avail & ~rep]; rp = qv[avail & rep]
sample = np.concatenate([rng.choice(fo, min(args.n_firstobs, len(fo)), replace=False), rng.choice(rp, min(args.n_repeat, len(rp)), replace=False)])
rows = []; t0 = time.time()
for i, q in enumerate(sample):
    q = int(q); cand = int(mi.enc_target[q]); bt = query_context(q)
    phi, f0, f1 = shapley(bt, q, cand, rng)
    rows.append({"enc": q, "patient": int(mi.enc_patient[q]), "cohort": "repeat" if rep[qv == q][0] else "firstobs", "candidate": "observed", "f_none": f0, "f_all": f1, **{f"phi:{g}": v for g, v in zip(GROUPS, phi)}})
    if args.also_top1:
        # model's top-1 in the unseen pool if different from the target
        dd = int(mi.enc_day[q]); pool = np.where(mi.available_specialists(dd))[0]; pool = pool[mi.ps_asof.lookup(mi.enc_patient[q] * mi.n_specialist + pool, dd) < 0]
        with torch.no_grad():
            h = model.encode(bt); pf = torch.as_tensor(mi.pair_features(np.full(len(pool), q), pool), device=device)
            s = model.score(h, torch.full((len(pool),), q, device=device), torch.as_tensor(pool, device=device), pf).cpu().numpy()
        top1 = int(pool[np.argmax(s)])
        if top1 != cand:
            phi2, f02, f12 = shapley(bt, q, top1, rng)
            rows.append({"enc": q, "patient": int(mi.enc_patient[q]), "cohort": rows[-1]["cohort"], "candidate": "model_top1", "f_none": f02, "f_all": f12, **{f"phi:{g}": v for g, v in zip(GROUPS, phi2)}})
    if (i + 1) % 10 == 0: print(f"  {i+1}/{len(sample)} queries, {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows); out = os.path.join(args.run_dir, "shap_groups"); os.makedirs(out, exist_ok=True); df.to_csv(os.path.join(out, "per_query.csv"), index=False)
# additivity check
err = (df[[f"phi:{g}" for g in GROUPS]].sum(1) - (df.f_all - df.f_none)).abs().max()
def boot(vals, groups, n=1000):
    r = np.random.default_rng(1); s = pd.Series(vals).groupby(groups).agg(["sum", "count"]); S, C = s["sum"].to_numpy(), s["count"].to_numpy(); b = np.array([S[i].sum() / C[i].sum() for i in (r.integers(0, len(S), len(S)) for _ in range(n))]); return float(np.mean(vals)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
summary = {"run": os.path.basename(args.run_dir), "schema": schema, "groups": GROUPS, "perms": args.perms, "seed": args.seed, "n_queries": int(len(sample)), "additivity_max_abs_error": float(err), "by_cohort": {}}
for coh in ("firstobs", "repeat"):
    dd = df[(df.cohort == coh) & (df.candidate == "observed")]
    summary["by_cohort"][coh] = {g: dict(zip(("mean_abs", "ci_lo", "ci_hi"), boot(dd[f"phi:{g}"].abs().to_numpy(), dd.patient.to_numpy()))) | {"mean_signed": float(dd[f"phi:{g}"].mean())} for g in GROUPS}
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(f"\n{summary['run']} | groups {G} | perms {args.perms} | additivity max abs error {err:.2e}")
for coh in ("firstobs", "repeat"):
    print(f"\n{coh}: mean |phi| per group (share of total), signed mean")
    s = summary["by_cohort"][coh]; tot = sum(v["mean_abs"] for v in s.values())
    for g, v in sorted(s.items(), key=lambda kv: -kv[1]["mean_abs"]):
        print(f"   {g:26s} {v['mean_abs']:7.3f}  ({v['mean_abs']/tot:5.1%})  [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]   signed {v['mean_signed']:+.3f}")
print("saved", os.path.relpath(out))
