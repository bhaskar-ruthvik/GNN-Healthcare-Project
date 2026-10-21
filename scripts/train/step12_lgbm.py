"""STEP12a LightGBM cells (STEP12_CONGESTION_PLAN.md, frozen 2026-10-12). Same rows as every model (training_matrix: the GNN's
21-row bundles), same selection criterion as baselines11.py (first-time MRR on the validation selection subset; selection only).

Objectives (§2.A):
  lambdarank  LightGBM LambdaRank (as the existing tabular baseline)
  softmax     custom group-softmax: rows are contiguous groups of 21, p = softmax(preds per group), grad = p − y,
              hess = max(p(1 − p), 1e-6) — the GNN's one-hot cross-entropy target
  binary      pointwise logistic
  xendcg      LightGBM rank_xendcg (descriptive extra only; label-smoothed, not softmax)
Feature options: --volume shares (B2), --no-pcp (H1 PCP-attribute check). Inputs: --feature-set tabular | primary (gate branch).
STEP13 (STEP13_GRAPH_VALUE_PLAN.md, frozen 2026-10-21): --feature-set flat_rf | full_nohist | full | insmatch, arm s13_lgbm_<set>_<objective>;
  tune mode with --extra-seeds 7 also trains seed 7 at the chosen configuration (run name <arm>_s7).
Modes:
  tune      12-config grid (leaves {31,127} × min leaf {50,200} × rounds {250,500,1000}); best config written as seed 42, then
            seed 1 of the same config (identical to baselines11.run_lgbm)
  fixed     --config LEAVES:MINLEAF:ROUNDS trained for each --seeds (capacity endpoints, PCP check)
  capacity  extended grid (§2.C): leaves {7,15,31,127,255} × min leaf {50,200}, one booster per pair (seed 42) to 2,000 rounds,
            one run directory per truncation {100,250,500,1000,2000} (no selection evaluation; scored by step10_evaluate)
Bagging and the other base parameters are identical across objectives (disclosed). LightGBM is imported before torch (macOS).
Usage (data paths resolve against the workspace, see referral_gnn/paths.py):
  python scripts/train/step12_lgbm.py --objective softmax --mode tune
  python scripts/train/step12_lgbm.py --objective lambdarank --volume shares --mode fixed --config 127:50:1000 --seeds 42 1
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass
import argparse
import json
import os
import sys
import time

import numpy as np


from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.evaluate import evaluate, metrics  # noqa: E402
from referral_gnn.train import sha256_file, code_hash  # noqa: E402
from referral_gnn import tabular as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", default="prepared_data/three_node_v5/model_inputs_PRIVATE.pkl")
ap.add_argument("--feature-set", default="tabular", choices=["tabular", "primary", "flat", "flat_rf", "full_nohist", "full", "insmatch"])
ap.add_argument("--objective", required=True, choices=["lambdarank", "softmax", "binary", "xendcg"])
ap.add_argument("--volume", default="keep", choices=["keep", "shares", "strict"])
ap.add_argument("--no-pcp", action="store_true")
ap.add_argument("--extra-npz", default=None, help="STEP12d B stacking: out-of-fold GNN scores (train rows) + lookup (selection pool, evaluation sets)")
ap.add_argument("--mode", required=True, choices=["tune", "fixed", "capacity"])
ap.add_argument("--config", default=None, help="fixed mode: LEAVES:MINLEAF:ROUNDS")
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--extra-seeds", type=int, nargs="*", default=[], help="tune mode: further seeds trained at the chosen configuration (STEP13: 7)")
args = ap.parse_args()
assert args.mode != "fixed" or args.config, "--mode fixed needs --config"

root = str(workspace()); os.chdir(root)
inputs_path = os.path.join(root, args.inputs)
mi = load_three_node_inputs(inputs_path)
INPUTS_SHA = sha256_file(inputs_path); CODE_SHA = code_hash(str(code_file("referral_gnn")))  # the repository library, not the data workspace
FB_OPTS = {"volume": args.volume, "no_pcp": args.no_pcp, **({"extra_npz": os.path.relpath(args.extra_npz, root)} if args.extra_npz else {})}
STEP13 = args.feature_set in T.STEP13_SETS
ARM = f"{'s13' if STEP13 else 's12'}_lgbm_{args.feature_set}_{args.objective}" + ({"shares": "_shares", "strict": "_strict"}.get(args.volume, "")) + ("_nopcp" if args.no_pcp else "") + ("_stack" if args.extra_npz else "")   # stacking: LightGBM seed s uses GNN seed s's scores (one run per seed, fixed mode)
sel = np.sort(mi.validation_sample)
fo_sel = sel[mi.ps_asof.lookup(mi.enc_patient[sel] * mi.n_specialist + mi.enc_target[sel], mi.enc_day[sel]) < 0]


def softmax_obj(preds, data):
    s = np.asarray(preds, np.float64).reshape(-1, 21); s = s - s.max(1, keepdims=True)
    p = np.exp(s); p /= p.sum(1, keepdims=True); y = data.get_label().reshape(-1, 21)
    return (p - y).ravel(), np.maximum(p * (1.0 - p), 1e-6).ravel()


BASE = dict(learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbosity=-1, num_threads=args.threads)
OBJ = {"lambdarank": dict(objective="lambdarank", lambdarank_truncation_level=21), "softmax": dict(objective=softmax_obj),
       "binary": dict(objective="binary"), "xendcg": dict(objective="rank_xendcg")}[args.objective]
NEEDS_GROUP = args.objective in ("lambdarank", "xendcg")


def select_metric(scorer):
    t0 = time.time(); m = metrics(evaluate(scorer, mi, "validation", query_subset=fo_sel, bucket="week", pool="all"))
    return {"firstobs_mrr_eligible": m["mrr_eligible"], "firstobs_recall@5_eligible": m["recall@5_eligible"], "eval_s": round(time.time() - t0, 1)}


def write_run(name, seed, params, tuning, best, booster, num_iteration, names, arm=None):
    run_dir = os.path.join(root, "runs", time.strftime("%Y%m%dT%H%M%S") + "_" + name)
    while os.path.exists(run_dir): time.sleep(1); run_dir = os.path.join(root, "runs", time.strftime("%Y%m%dT%H%M%S") + "_" + name)
    os.makedirs(run_dir); booster.save_model(os.path.join(run_dir, "model.txt"), num_iteration=int(num_iteration))
    man = {"kind": "lgbm", "arm": arm or ARM, "step": "13" if STEP13 else ("12b" if args.feature_set == "flat" else "12a"),
           "protocol": "STEP13_GRAPH_VALUE_PLAN.md (frozen 2026-10-21)" if STEP13 else ("STEP12B_HYBRID_PLAN.md" if args.feature_set == "flat" else "STEP12_CONGESTION_PLAN.md (frozen 2026-10-12)"), "seed": seed,
           "params": {**params, "objective": args.objective, "num_iteration": int(num_iteration)}, "feature_set": args.feature_set,
           "feature_opts": FB_OPTS, "feature_names": names, "inputs_path": inputs_path, "inputs_sha256": INPUTS_SHA, "code_sha256": CODE_SHA,
           "train_config": {"graph_bucket": "week"}, "selection_metric": "firstobs_mrr_eligible", "selection_queries": int(len(fo_sel)),
           "tuning": tuning, "mode": args.mode, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(man, open(os.path.join(run_dir, "run_manifest.json"), "w"), indent=1)
    json.dump({"selection_metric": "firstobs_mrr_eligible", "best_val_selection": best, "params": man["params"]}, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    print(f"  wrote {os.path.relpath(run_dir, root)}", flush=True)


import lightgbm as lgb  # noqa: E402
t0 = time.time(); X, y, qid, fb = T.training_matrix(mi, args.feature_set, log=lambda s: None, **FB_OPTS)
print(f"[{ARM}] training matrix {X.shape} ({len(fb.names)} features) in {time.time() - t0:.0f}s", flush=True)
assert len(X) % 21 == 0 and (y.reshape(-1, 21)[:, 0] == 1).all() and y.reshape(-1, 21)[:, 1:].sum() == 0, "rows must be contiguous 21-groups, target first"
group = np.full(len(X) // 21, 21)
def dataset():
    return lgb.Dataset(X, y.astype(np.float64), group=group if NEEDS_GROUP else None, categorical_feature=fb.cat_idx, feature_name=fb.names, free_raw_data=False)
def train(leaves, minleaf, rounds, seed):
    p = dict(BASE, **OBJ, num_leaves=leaves, min_data_in_leaf=minleaf, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    return lgb.train(p, dataset(), num_boost_round=rounds)
PARAMS = lambda L, M: {"num_leaves": L, "min_data_in_leaf": M, **{k: BASE[k] for k in ("learning_rate", "feature_fraction", "bagging_fraction")}}

if args.mode == "tune":
    tuning, best = [], None
    for L in (31, 127):
        for M in (50, 200):
            t1 = time.time(); b = train(L, M, 1000, 42); fit_s = round(time.time() - t1, 1)
            for k in (250, 500, 1000):
                m = select_metric(T.TabularScorer(mi, T.BoosterAt(b, k), args.feature_set, **FB_OPTS))
                row = {"num_leaves": L, "min_data_in_leaf": M, "num_iteration": k, "fit_s": fit_s, **m}; tuning.append(row); print(f"[{ARM}] {row}", flush=True)
                if best is None or m["firstobs_mrr_eligible"] > best[0]["firstobs_mrr_eligible"]: best = (m, row, b)
    m, row, b = best
    write_run(f"{ARM}_s42", 42, PARAMS(row["num_leaves"], row["min_data_in_leaf"]), tuning, m["firstobs_mrr_eligible"], b, row["num_iteration"], fb.names)
    b1 = train(row["num_leaves"], row["min_data_in_leaf"], row["num_iteration"], 1)
    m1 = select_metric(T.TabularScorer(mi, T.BoosterAt(b1, row["num_iteration"]), args.feature_set, **FB_OPTS))
    write_run(f"{ARM}_s1", 1, PARAMS(row["num_leaves"], row["min_data_in_leaf"]), [{**row, **m1}], m1["firstobs_mrr_eligible"], b1, row["num_iteration"], fb.names)
    for sd in args.extra_seeds:                                                # STEP13: seed 7 at the configuration chosen on seed 42
        bx = train(row["num_leaves"], row["min_data_in_leaf"], row["num_iteration"], sd)
        mx = select_metric(T.TabularScorer(mi, T.BoosterAt(bx, row["num_iteration"]), args.feature_set, **FB_OPTS))
        write_run(f"{ARM}_s{sd}", sd, PARAMS(row["num_leaves"], row["min_data_in_leaf"]), [{**row, **mx}], mx["firstobs_mrr_eligible"], bx, row["num_iteration"], fb.names)
elif args.mode == "fixed":
    L, M, R = (int(x) for x in args.config.split(":"))
    for sd in args.seeds:
        b = train(L, M, R, sd); m = select_metric(T.TabularScorer(mi, T.BoosterAt(b, R), args.feature_set, **FB_OPTS))
        write_run(f"{ARM}_L{L}_M{M}_R{R}_s{sd}", sd, PARAMS(L, M), [{"num_leaves": L, "min_data_in_leaf": M, "num_iteration": R, **m}], m["firstobs_mrr_eligible"], b, R, fb.names,
                  arm=f"{ARM}_L{L}_M{M}_R{R}")
else:
    assert args.objective == "lambdarank", "the capacity grid (H4) is defined for the LambdaRank reference"
    for L in (7, 15, 31, 127, 255):
        for M in (50, 200):
            t1 = time.time(); b = train(L, M, 2000, 42); print(f"[{ARM}] capacity L{L} M{M} trained in {time.time() - t1:.0f}s", flush=True)
            for R in (100, 250, 500, 1000, 2000):
                write_run(f"{ARM}_cap_L{L}_M{M}_R{R}_s42", 42, PARAMS(L, M), [], float("nan"), b, R, fb.names, arm=f"{ARM}_cap_L{L}_M{M}_R{R}")
print(f"=== {ARM} {args.mode} done {time.strftime('%H:%M:%S')}", flush=True)
