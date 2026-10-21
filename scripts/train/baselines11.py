"""Step 11.1 non-GNN baselines (STEP11_PLAN, frozen 2026-10-07). Trains, tunes on validation only, and writes one run
directory per baseline (run_manifest.json with kind/arm, model file, summary.json) that step10_evaluate.py can score like a
model run. Nothing here reads the test split.

Selection criterion = the GNN's own: first-time MRR (eligible) on the 3,000-query validation selection subset, full pool
(evaluate(..., pool="all", bucket="week")), restricted to its first-time queries (identical to the trainer's firstobs metric).

Usage (data paths resolve against the workspace, see referral_gnn/paths.py):
    python scripts/train/baselines11.py --which lgbm_primary lgbm_graph logreg als itemknn
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
# LightGBM must be imported before torch: with torch's libomp loaded first, lib_lightgbm segfaults on macOS
# (reproduced 2026-10-07: Dataset construction crashes; the reverse order works).
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd


from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.evaluate import evaluate, metrics  # noqa: E402
from referral_gnn.train import sha256_file, code_hash  # noqa: E402
from referral_gnn import tabular as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", default="prepared_data/three_node_v5/model_inputs_PRIVATE.pkl")
ap.add_argument("--which", nargs="+", default=["lgbm_primary", "lgbm_graph", "logreg", "als", "itemknn"])
ap.add_argument("--threads", type=int, default=6, help="LightGBM threads (leave cores for the GNN chain)")
ap.add_argument("--max-configs", type=int, default=20)
args = ap.parse_args()

root = str(workspace()); os.chdir(root)
inputs_path = os.path.join(root, args.inputs)
mi = load_three_node_inputs(inputs_path)
INPUTS_SHA = sha256_file(inputs_path); CODE_SHA = code_hash(os.path.join(root, "referral_gnn"))
sel = np.sort(mi.validation_sample)
fo_sel = sel[mi.ps_asof.lookup(mi.enc_patient[sel] * mi.n_specialist + mi.enc_target[sel], mi.enc_day[sel]) < 0]
print(f"selection subset {len(sel):,} queries; first-time {len(fo_sel):,}", flush=True)


def select_metric(scorer):
    t0 = time.time()
    m = metrics(evaluate(scorer, mi, "validation", query_subset=fo_sel, bucket="week", pool="all"))
    return {"firstobs_mrr_eligible": m["mrr_eligible"], "firstobs_recall@5_eligible": m["recall@5_eligible"],
            "firstobs_recall@10_eligible": m["recall@10_eligible"], "eval_s": round(time.time() - t0, 1)}


def write_run(name, kind, arm, seed, params, tuning, best, files, feature_set=None, feature_names=None):
    run_dir = os.path.join(root, "runs", time.strftime("%Y%m%dT%H%M%S") + "_" + name); os.makedirs(run_dir)
    for fn, writer in files.items(): writer(os.path.join(run_dir, fn))
    man = {"kind": kind, "arm": arm, "step": "11.1", "protocol": "STEP11_PLAN.md (frozen 2026-10-07)", "seed": seed, "params": params,
           "feature_set": feature_set, "feature_names": feature_names, "inputs_path": inputs_path, "inputs_sha256": INPUTS_SHA,
           "code_sha256": CODE_SHA, "train_config": {"graph_bucket": "week"}, "selection_metric": "firstobs_mrr_eligible",
           "selection_queries": int(len(fo_sel)), "tuning": tuning, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(man, open(os.path.join(run_dir, "run_manifest.json"), "w"), indent=1)
    json.dump({"selection_metric": "firstobs_mrr_eligible", "best_val_selection": best, "params": params}, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    print(f"  wrote {os.path.relpath(run_dir, root)}  (selection {best:.4f})", flush=True)
    return run_dir


# ---- LightGBM LambdaRank -------------------------------------------------------------------------------------------
GRID_LEAVES, GRID_MINLEAF, GRID_ROUNDS = [31, 127], [50, 200], [250, 500, 1000]       # 12 configs <= 20 (plan budget)


def run_lgbm(feature_set):
    import lightgbm as lgb
    t0 = time.time(); X, y, qid, fb = T.training_matrix(mi, feature_set, log=lambda s: print(s, flush=True))
    print(f"[lgbm_{feature_set}] training matrix {X.shape} in {time.time() - t0:.0f}s", flush=True)
    group = np.full(len(X) // 21, 21)
    base = dict(objective="lambdarank", learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambdarank_truncation_level=21, verbosity=-1, num_threads=args.threads)
    tuning, best = [], None
    for leaves in GRID_LEAVES:
        for minleaf in GRID_MINLEAF:
            p = dict(base, num_leaves=leaves, min_data_in_leaf=minleaf, seed=42, bagging_seed=42, feature_fraction_seed=42)
            t1 = time.time()
            ds = lgb.Dataset(X, y, group=group, categorical_feature=fb.cat_idx, feature_name=fb.names, free_raw_data=False)
            booster = lgb.train(p, ds, num_boost_round=max(GRID_ROUNDS))
            fit_s = time.time() - t1
            for k in GRID_ROUNDS:
                m = select_metric(T.TabularScorer(mi, T.BoosterAt(booster, k), feature_set))
                row = {"num_leaves": leaves, "min_data_in_leaf": minleaf, "num_iteration": k, "fit_s": round(fit_s, 1), **m}
                tuning.append(row); print(f"[lgbm_{feature_set}] {row}", flush=True)
                if best is None or m["firstobs_mrr_eligible"] > best[0]["firstobs_mrr_eligible"]:
                    best = (m, row, booster)
            if len(tuning) >= args.max_configs: break
    m, row, booster = best
    params = {k: row[k] for k in ("num_leaves", "min_data_in_leaf", "num_iteration")} | {k: base[k] for k in ("learning_rate", "feature_fraction", "bagging_fraction")}
    write_run(f"lgbm_{feature_set}_s42", "lgbm", f"lgbm_{feature_set}", 42, params, tuning, m["firstobs_mrr_eligible"],
              {"model.txt": lambda path: booster.save_model(path, num_iteration=row["num_iteration"])}, feature_set, fb.names)
    # second seed of the selected configuration (G2 seed gap); no re-tuning
    p = dict(base, num_leaves=row["num_leaves"], min_data_in_leaf=row["min_data_in_leaf"], seed=1, bagging_seed=1, feature_fraction_seed=1)
    b1 = lgb.train(p, lgb.Dataset(X, y, group=group, categorical_feature=fb.cat_idx, feature_name=fb.names), num_boost_round=row["num_iteration"])
    m1 = select_metric(T.TabularScorer(mi, T.BoosterAt(b1, row["num_iteration"]), feature_set))
    write_run(f"lgbm_{feature_set}_s1", "lgbm", f"lgbm_{feature_set}", 1, params, [{**row, **m1}], m1["firstobs_mrr_eligible"],
              {"model.txt": lambda path: b1.save_model(path)}, feature_set, fb.names)
    del X, y, qid


# ---- logistic regression (pointwise, primary features) -------------------------------------------------------------
def run_logreg(feature_set="primary"):
    t0 = time.time(); X, y, qid, fb = T.training_matrix(mi, feature_set, log=lambda s: print(s, flush=True))
    print(f"[logreg] training matrix {X.shape} in {time.time() - t0:.0f}s", flush=True)
    tuning, best = [], None
    for C in (0.1, 1.0):
        t1 = time.time(); lm = T.LogisticModel(fb.cat_idx, C=C).fit(X, y); fit_s = time.time() - t1
        m = select_metric(T.TabularScorer(mi, lm, feature_set)); row = {"C": C, "fit_s": round(fit_s, 1), **m}
        tuning.append(row); print(f"[logreg] {row}", flush=True)
        if best is None or m["firstobs_mrr_eligible"] > best[0]["firstobs_mrr_eligible"]: best = (m, row, lm)
    m, row, lm = best
    nm = "logreg" if feature_set == "primary" else f"logreg_{feature_set}"
    write_run(f"{nm}_s42", "logreg", nm, 42, {"C": row["C"]}, tuning, m["firstobs_mrr_eligible"],
              {"model.pkl": lambda path: pickle.dump(lm, open(path, "wb"))}, feature_set, fb.names)


# ---- collaborative filtering ---------------------------------------------------------------------------------------
def run_als():
    tuning, best = [], None
    for f in (16, 32, 64):
        params = {"factors": f, "reg": 0.1, "alpha": 10.0, "iters": 15}
        m = select_metric(T.ALSScorer(mi, seed=42, **params)); row = {**params, **m}
        tuning.append(row); print(f"[als] {row}", flush=True)
        if best is None or m["firstobs_mrr_eligible"] > best[0]["firstobs_mrr_eligible"]: best = (m, params)
    write_run("als_s42", "als", "als", 42, best[1], tuning, best[0]["firstobs_mrr_eligible"], {})


def run_itemknn():
    m = select_metric(T.ItemKNNScorer(mi)); print(f"[itemknn] {m}", flush=True)
    write_run("itemknn", "itemknn", "itemknn", None, {"refit_days": T._PCPMatrix.REFIT}, [m], m["firstobs_mrr_eligible"], {})


for w in args.which:
    t = time.time(); print(f"=== {w}  {time.strftime('%H:%M:%S')}", flush=True)
    {"lgbm_primary": lambda: run_lgbm("primary"), "lgbm_graph": lambda: run_lgbm("graph"), "lgbm_tabular": lambda: run_lgbm("tabular"), "logreg_tabular": lambda: run_logreg("tabular"), "logreg": run_logreg,
     "als": run_als, "itemknn": run_itemknn}[w]()
    print(f"=== {w} done in {(time.time() - t) / 60:.1f} min", flush=True)
print("=== BASELINES DONE", time.strftime("%H:%M:%S"))
