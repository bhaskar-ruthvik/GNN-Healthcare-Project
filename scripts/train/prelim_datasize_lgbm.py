"""PRELIMINARY data-size runs for STEP11 §11.4 (user request 2026-10-08, before the proper snapshots): LightGBM trained only on
training queries strictly before --train-end, with the hyperparameters already selected on the full training period (no
re-tuning). Inputs are the current three_node_v5 (encoders/vocabularies fitted on 2015–2017: a disclosed limitation that the
proper snapshots remove). Writes run directories scored by step10_evaluate.py like any 11.1 baseline.

    python scripts/train/prelim_datasize_lgbm.py --train-end 2017-01-01 --variants tabular graph --seeds 42
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
try:
    import lightgbm as lgb  # must precede torch on macOS (see 2026-10-07 log)
except ImportError:
    raise
import argparse
import glob
import json
import os
import sys
import time

import numpy as np


from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.data import to_day  # noqa: E402
from referral_gnn.train import sha256_file, code_hash  # noqa: E402
from referral_gnn import tabular as T  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--train-end", required=True); ap.add_argument("--variants", nargs="+", default=["tabular", "graph"])
ap.add_argument("--seeds", nargs="+", type=int, default=[42]); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--inputs", default="prepared_data/three_node_v5/model_inputs_PRIVATE.pkl"); args = ap.parse_args()
root = str(workspace()); os.chdir(root)
inputs_path = os.path.join(root, args.inputs); mi = load_three_node_inputs(inputs_path)
end_day = int(np.asarray(to_day(np.array([args.train_end])))[0]); tag = "upto" + args.train_end[:4] if args.train_end.endswith("-01-01") else "upto" + args.train_end
for v in args.variants:
    src = json.load(open(glob.glob(f"runs/*_lgbm_{v}_s42/run_manifest.json")[0]))         # hyperparameters selected on the full period
    p0 = src["params"]
    X, y, qid, fb = T.training_matrix(mi, v, log=lambda s: None, end_day=end_day)
    print(f"[{v}] {args.train_end}: {len(X) // 21:,} training queries; params from full-period selection {p0}", flush=True)
    for seed in args.seeds:
        t0 = time.time()
        p = dict(objective="lambdarank", learning_rate=p0["learning_rate"], feature_fraction=p0["feature_fraction"], bagging_fraction=p0["bagging_fraction"],
                 bagging_freq=1, lambdarank_truncation_level=21, verbosity=-1, num_threads=args.threads, num_leaves=p0["num_leaves"],
                 min_data_in_leaf=p0["min_data_in_leaf"], seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        b = lgb.train(p, lgb.Dataset(X, y, group=np.full(len(X) // 21, 21), categorical_feature=fb.cat_idx, feature_name=fb.names), num_boost_round=p0["num_iteration"])
        name = f"lgbm_{v}_{tag}_s{seed}"; run_dir = os.path.join(root, "runs", time.strftime("%Y%m%dT%H%M%S") + "_" + name); os.makedirs(run_dir)
        b.save_model(os.path.join(run_dir, "model.txt"))
        man = {"kind": "lgbm", "arm": f"lgbm_{v}_{tag}", "step": "11.4-prelim", "preliminary": True, "train_end": args.train_end,
               "n_train_queries": int(len(X) // 21), "seed": seed, "params": p0, "params_source": "full-period selection (no re-tuning)",
               "feature_set": v, "feature_names": fb.names, "inputs_path": inputs_path, "inputs_sha256": sha256_file(inputs_path),
               "code_sha256": code_hash(os.path.join(root, "referral_gnn")), "train_config": {"graph_bucket": "week"}, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(man, open(os.path.join(run_dir, "run_manifest.json"), "w"), indent=1)
        json.dump({"preliminary": True}, open(os.path.join(run_dir, "summary.json"), "w"))
        print(f"  wrote {os.path.relpath(run_dir, root)} ({time.time() - t0:.0f}s)", flush=True)
print("PRELIM LGBM DONE", flush=True)
