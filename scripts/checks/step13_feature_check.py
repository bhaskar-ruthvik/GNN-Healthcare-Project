"""STEP13 §6 feature check (STEP13_GRAPH_VALUE_PLAN.md, frozen 2026-10-21): the generated aggregate columns against brute-force
hand calculations, with strict pre-cutoff access asserted, and the subset chain Flat-RF ⊂ Info-table-nohist ⊂ Info-table.

For sampled weekly buckets (training and validation) and sampled queries / candidates it recomputes, directly from the encounter
arrays with enc_day < cutoff:
  A1 (candidate's past encounters), A2 (their PCPs' encoder inputs), B1 (the query PCP's past encounters), the 16 time-encoding
  means (data.time_encoding applied to each encounter's age, then averaged), the matching terms, the no-history flags, A3 and the
  insurer share, and compares them with FeatureBuilder rows (max |difference| ≤ 1e-4).
Prints and writes aggregates only (counts and maximum differences), never rows.
Usage: python scripts/checks/step13_feature_check.py [--buckets 6] [--queries 4] [--out runs/analysis/step13/feature_check.json]
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import json

import numpy as np

from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.tabular import FeatureBuilder, GraphStats  # noqa: E402
from referral_gnn.data import time_encoding  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", default="prepared_data/three_node_v5/model_inputs_PRIVATE.pkl")
ap.add_argument("--buckets", type=int, default=6, help="sampled weekly buckets per split (training, validation)")
ap.add_argument("--queries", type=int, default=4, help="sampled queries per bucket")
ap.add_argument("--cands", type=int, default=40, help="sampled candidates per query (plus specialists without history)")
ap.add_argument("--out", default="runs/analysis/step13/feature_check.json")
args = ap.parse_args()
root = str(workspace()); os.chdir(root); os.makedirs(os.path.dirname(args.out), exist_ok=True)
mi = load_three_node_inputs(os.path.join(root, args.inputs))
rng = np.random.default_rng(20261021)
report = {"inputs": args.inputs, "tolerance": 1e-4}

# ---- 1. column sets ----------------------------------------------------------------------------------------------------
fbs = {k: FeatureBuilder(mi, k) for k in ("tabular", "flat_rf", "full_nohist", "full", "insmatch")}
N = {k: set(f.names) for k, f in fbs.items()}
hist_blk = set(range(*mi.encounter_blocks["patient_history"]))
explicit = {n for n in N["full"] if (n.startswith("enc_static_") and int(n.rsplit("_", 1)[1]) in hist_blk) or n in ("gap_log", "gap_none")
            or n.startswith("g_coref_") or (n.startswith("pair_") and n not in N["tabular"])}
assert N["tabular"] <= N["flat_rf"] <= N["full_nohist"] <= N["full"], "Flat-RF ⊂ Info-table-nohist ⊂ Info-table violated"
assert N["insmatch"] == N["full_nohist"] | {"f13_ins_share"}, "Ins-match must be Info-table-nohist + one column"
assert not (explicit & N["full_nohist"]) and not (explicit & N["flat_rf"]), "explicit history leaked into a no-history set"
assert N["full"] - N["full_nohist"] == explicit, "Info-table must add exactly the excluded explicit-history columns"
assert not any(n.startswith(("f13_A3", "g_")) for n in N["flat_rf"]), "Flat-RF must not contain A3 or graph counts"
report["columns"] = {k: len(f.names) for k, f in fbs.items()} | {"explicit_history_added_by_full": len(explicit)}
print("columns:", report["columns"])

# ---- 2. brute-force aggregates -------------------------------------------------------------------------------------------
fb = fbs["insmatch"]; F = fb.flat13; col = {n: i for i, n in enumerate(fb.names)}
dims = mi.dims(); enc_cat_sizes, pcp_cat_sizes = dims["encounter_cat_sizes"], dims["pcp_cat_sizes"]
st = mi._encounter_static(); n_cat = F.n_cat


def enc_attr(e, cutoff):
    """One encounter's attributes as the GNN reads them on a historical encounter node (explicit history excluded)."""
    c = mi.enc_dx[e][mi.enc_dx_mask[e]]; c = np.unique(c[c > 1]); dist = np.zeros(n_cat)
    if len(c): dist[c] = 1.0 / len(c)
    oh = [np.eye(k)[mi.enc_patient_cat[e, j]] for j, k in enumerate(enc_cat_sizes)]
    return np.concatenate([st[e, :7], st[e, 74:76], dist] + oh + [time_encoding(np.array([cutoff - mi.enc_day[e]]))[0]])


def mean_attr(encs, cutoff):
    return np.mean([enc_attr(e, cutoff) for e in encs], axis=0) if len(encs) else np.zeros(9 + n_cat + sum(enc_cat_sizes) + 16)


def block(prefix):
    return [col[n] for n in fb.names if n.startswith(prefix)]


A1c = block("f13_A1_"); B1c = block("f13_B1_"); A2c = block("f13_A2_"); A3c = [col[n] for n in fb.names if n.startswith("f13_A3_spec")]
worst = {k: 0.0 for k in ("A1", "A2", "B1", "time", "cosine", "age_diff", "flags", "A3", "ins_share")}
n_checked = {"queries": 0, "candidates": 0, "candidates_without_history": 0, "pcps_without_history": 0, "unknown_insurer_queries": 0}
for split in ("train", "validation"):
    qs = mi.train_query_enc if split == "train" else np.sort(mi.validation_sample)
    buckets = mi.buckets(split, "week", query_subset=qs)
    picked = set(rng.choice(len(buckets), size=min(args.buckets, len(buckets)), replace=False).tolist())
    if split == "train": picked |= {0, 1}                                    # earliest buckets: PCPs and specialists without history
    unk = [i for i, b_ in enumerate(buckets) if (mi.enc_insurer[b_[2]] < 0).any()]
    if unk: picked.add(unk[0])                                               # one bucket with an unknown-insurer query
    for bi in sorted(picked):
        cutoff, batch_enc, q_b = buckets[bi]; cutoff = int(cutoff)
        fb.prepare(cutoff, batch_enc)
        seen = mi.enc_day[F.order[:F.ptr]]; unseen = mi.enc_day[F.order[F.ptr:]]
        assert (seen < cutoff).all() and (unseen >= cutoff).all(), "aggregates must use exactly the encounters strictly before the cutoff"
        hist = np.flatnonzero(mi.enc_day < cutoff)
        snap = fb.snap; pv = F.pcp_vectors(snap)
        nohist_specs = np.setdiff1d(np.arange(mi.n_specialist), mi.enc_target[hist])
        forced = [int(x) for x in q_b if mi.enc_insurer[x] < 0][:1] + [int(x) for x in q_b if not (mi.enc_pcp[hist] == mi.enc_pcp[x]).any()][:1]
        for q in np.unique(np.r_[rng.choice(q_b, size=min(args.queries, len(q_b)), replace=False), forced].astype(np.int64)):
            q = int(q); assert mi.enc_day[q] >= cutoff, "query referral must not be in its own aggregates"
            cand = np.unique(np.r_[rng.choice(mi.n_specialist, size=args.cands, replace=False), mi.enc_target[q],
                                   rng.choice(nohist_specs, size=min(3, len(nohist_specs)), replace=False)]).astype(np.int64)
            X = fb.rows(q, cand).astype(np.float64)
            p = int(mi.enc_pcp[q]); pe = hist[mi.enc_pcp[hist] == p]
            B1 = mean_attr(pe, cutoff)
            worst["B1"] = max(worst["B1"], float(np.abs(X[:, B1c] - B1[None, :]).max()))
            worst["flags"] = max(worst["flags"], float(np.abs(X[:, col["f13_pcp_nohist"]] - float(len(pe) == 0)).max()))
            sv = snap["specialist_num"][:, F.spec_vals].astype(np.float64)
            A3 = sv[mi.enc_target[pe]].mean(0) if len(pe) else np.zeros(len(F.spec_vals))
            worst["A3"] = max(worst["A3"], float(np.abs(X[:, A3c] - A3[None, :]).max()))
            ins = int(mi.enc_insurer[q]); n_checked["unknown_insurer_queries"] += int(ins < 0); n_checked["pcps_without_history"] += int(len(pe) == 0)
            qc = enc_attr(q, cutoff)[9:9 + n_cat]
            for i, s in enumerate(cand):
                se = hist[mi.enc_target[hist] == s]; A1 = mean_attr(se, cutoff)
                d = np.abs(X[i, A1c] - A1); worst["A1"] = max(worst["A1"], float(d[:-16].max())); worst["time"] = max(worst["time"], float(d[-16:].max()))
                d2 = np.abs(X[i, block("f13_B1_time")] - B1[-16:]); worst["time"] = max(worst["time"], float(d2.max()))
                A2 = pv[mi.enc_pcp[se]].mean(0) if len(se) else np.zeros(pv.shape[1])
                worst["A2"] = max(worst["A2"], float(np.abs(X[i, A2c] - A2).max()))
                ac = A1[9:9 + n_cat]
                cos = float(ac @ qc / (np.linalg.norm(ac) * (np.linalg.norm(qc) or 1.0) + 1e-9)) if len(se) else 0.0
                worst["cosine"] = max(worst["cosine"], abs(X[i, col["f13_cat_cosine"]] - cos))
                worst["age_diff"] = max(worst["age_diff"], abs(X[i, col["f13_age_diff"]] - ((st[q, 0] - A1[0]) if len(se) else 0.0)))
                worst["flags"] = max(worst["flags"], abs(X[i, col["f13_spec_nohist"]] - float(len(se) == 0)))
                share = X[i, col["f13_ins_share"]]
                if ins < 0: assert np.isnan(share), "unknown query insurer must give a missing share"
                else: worst["ins_share"] = max(worst["ins_share"], abs(share - ((mi.enc_insurer[se] == ins).mean() if len(se) else 0.0)))
                n_checked["candidates"] += 1; n_checked["candidates_without_history"] += int(len(se) == 0)
            n_checked["queries"] += 1
        print(f"  {split} bucket {bi}: cutoff {cutoff}, worst so far {max(worst.values()):.2e}", flush=True)
report["max_abs_difference"] = worst; report["checked"] = n_checked
report["passed"] = bool(max(worst.values()) <= report["tolerance"] and n_checked["candidates_without_history"] > 0)
report["note"] = "zero counts for unknown-insurer queries or PCPs without history mean no such query exists in the inputs' query sets" 
json.dump(report, open(args.out, "w"), indent=1)
print(json.dumps(report, indent=1))
assert report["passed"], "STEP13 feature check failed"
