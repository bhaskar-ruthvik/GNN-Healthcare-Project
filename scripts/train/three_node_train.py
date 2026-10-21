"""Train a three-node model. Inputs path is explicit; nothing here touches the four-node bundle.
Example:
    python scripts/train/three_node_train.py --name tn_summaries --no-temporal
    python scripts/train/three_node_train.py --name tn_temporal --temporal
Test-split evaluation is not run here.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import os
import sys
import time
import numpy as np


from referral_gnn.three_node_model import ThreeNodeConfig  # noqa: E402
from referral_gnn.three_node_train import ThreeNodeTrainer, load_three_node_inputs  # noqa: E402
from referral_gnn.train import TrainConfig  # noqa: E402

ABL = ["no_diagnosis", "no_distance", "no_history_features", "no_patient_specialist_prior", "no_patient_history", "no_predecessor_gap", "no_referral_edges", "no_time_encoding",
       "no_insurer_history"]
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--inputs", default="prepared_data/three_node_v1/model_inputs_PRIVATE.pkl")
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--temporal", action="store_true", help="arm 2: PRECEDES edges on")
g.add_argument("--no-temporal", action="store_true", help="arm 1: summaries only, PRECEDES edges off (identical inputs)")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--accumulate", type=int, default=1)
ap.add_argument("--device", default="auto")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--graph-bucket", default="week", choices=["day", "week"])
ap.add_argument("--select", default="mrr_eligible", choices=["mrr_eligible", "firstobs_mrr_eligible"], help="checkpoint selection metric (set by the revision protocol)")
ap.add_argument("--ablation", action="append", default=[], choices=ABL)
ap.add_argument("--no-graph", action="store_true")
ap.add_argument("--mask-demographics", action="store_true", help="STEP9 masked arm: patient gender/race -> UNK, ambiguity flags zeroed")
ap.add_argument("--max-train-days", type=int, default=None)
ap.add_argument("--lr-factor", type=float, default=1.0, help="<1 enables ReduceLROnPlateau(factor) on the selection metric")
ap.add_argument("--lr-patience", type=int, default=2)
ap.add_argument("--negatives", default="uniform", choices=["uniform", "hard", "pool"], help="uniform = bundled alternatives; hard = PCP-familiar + same-specialty + uniform; pool = target-specialty ∩ unseen-by-patient")
ap.add_argument("--cohort", default="all", choices=["all", "firstobs"], help="training queries: all events or first-observed patient-specialist pairs only")
ap.add_argument("--val-pool", default="all", choices=["all", "unseen", "specialty_unseen"], help="validation candidate pool used for selection/early stopping")
ap.add_argument("--neg-k", default="8,8,4", help="hard policy counts: familiar,specialty,uniform")
ap.add_argument("--pos-weight-alpha", type=float, default=0.0, help="A10 Option B variant 2: weight each training referral by (recent load of the referred specialist + 1)^-alpha")
ap.add_argument("--train-end", default=None, metavar="YYYY-MM-DD", help="STEP11 preliminary data-size runs: train only on queries strictly before this date (inputs unchanged; validation unchanged)")
ap.add_argument("--history-dropout", type=float, default=0.0, help="STEP11 B5 C1: training-time PCP-history dropout probability per query")
ap.add_argument("--scorer-skip", action="store_true", help="STEP11 B5 C2: scorer gets pre-HGT encodings and their element-wise product")
ap.add_argument("--loss", default="softmax", choices=["softmax", "lambdarank", "bce"], help="STEP12a §2.A training objective")
ap.add_argument("--volume-mode", default="keep", choices=["keep", "log1p", "shares", "strict"], help="STEP12a: specialist volume representation (H5 log1p, B2 shares)")
ap.add_argument("--table-offset", default=None, metavar="NPZ", help="STEP12d A: out-of-fold table scores (train rows) + selection-pool lookup; score += β·table")
ap.add_argument("--offset-table-run", default=None, help="STEP12d A: the full table run whose scores the evaluator uses as the offset")
ap.add_argument("--tab-tower", action="store_true", help="STEP12d C: dual encoder (GNN + tabular tower over the pure-tabular row)")
ap.add_argument("--exclude-fold", type=int, default=None, help="STEP12d B: train without training-query time fold k (of --n-folds)")
ap.add_argument("--n-folds", type=int, default=3)
ap.add_argument("--pcp-in-scorer", action="store_true", help="STEP13 NG+P: scorer also reads the query PCP's encoding (use with --no-graph)")
ap.add_argument("--arm", default=None, help="STEP13: evaluator arm name written to the run manifest (one arm per cell; checked for collisions)")
ap.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue an interrupted run from its last completed epoch (same flags required; checked against its manifest)")
args = ap.parse_args()

root = str(workspace())
from referral_gnn.paths import resolve_run  # noqa: E402
if args.offset_table_run: args.offset_table_run = resolve_run(args.offset_table_run) if not os.path.isabs(args.offset_table_run) else args.offset_table_run
inputs_path = os.path.join(root, args.inputs) if not os.path.isabs(args.inputs) else args.inputs
mi = load_three_node_inputs(inputs_path)
mcfg = ThreeNodeConfig(temporal_edges=bool(args.temporal), use_graph=not args.no_graph, mask_demographics=args.mask_demographics, scorer_skip=args.scorer_skip, volume_mode=args.volume_mode, table_offset=bool(args.table_offset),
                       tab_dim=(117 if args.tab_tower else 0), pcp_in_scorer=args.pcp_in_scorer, **{a: True for a in args.ablation})
tcfg = TrainConfig(lr=args.lr, max_epochs=args.epochs, patience=args.patience, accumulate_days=args.accumulate, seed=args.seed, device=args.device, max_train_days=args.max_train_days, graph_bucket=args.graph_bucket)
import json
if args.resume:
    run_dir = os.path.abspath(args.resume)
    if not os.path.basename(run_dir).endswith("_" + args.name): raise SystemExit(f"--resume {run_dir} does not belong to run name {args.name}")
    if os.path.exists(os.path.join(run_dir, "summary.json")): raise SystemExit(f"{run_dir} already finished (summary.json exists)")
    if not os.path.exists(os.path.join(run_dir, "resume_state.pt")): raise SystemExit(f"{run_dir} has no resume_state.pt (started before resume support); rerun from scratch")
    old_manifest = json.load(open(os.path.join(run_dir, "run_manifest.json")))
else:
    run_dir = os.path.join(root, "runs", time.strftime("%Y%m%dT%H%M%S") + "_" + args.name); old_manifest = None
tr = ThreeNodeTrainer(mi, mcfg, tcfg, run_dir, inputs_path)
tr.selection_metric = args.select; tr.lr_factor = args.lr_factor; tr.lr_patience = args.lr_patience
if args.train_end:
    from referral_gnn.data import to_day
    import numpy as _np
    end_day = int(_np.asarray(to_day(_np.array([args.train_end])))[0])
    keep_q = mi.train_query_enc[mi.enc_day[mi.train_query_enc] < end_day]
    tr.train_buckets = mi.buckets("train", tcfg.graph_bucket, query_subset=keep_q)
    print(f"--train-end {args.train_end}: {len(keep_q):,} of {len(mi.train_query_enc):,} training queries, {len(tr.train_buckets)} buckets")
tr.history_dropout = args.history_dropout; tr.loss = args.loss
if args.exclude_fold is not None:                                          # STEP12d B: GNN fold models for out-of-fold scores
    tr.manifest["train_queries_used"] = None
    from referral_gnn.heavy_hybrid import time_folds
    f_ = time_folds(mi, args.n_folds); keep_q = mi.train_query_enc[f_ != args.exclude_fold]
    tr.train_buckets = mi.buckets("train", tcfg.graph_bucket, query_subset=keep_q)
    print(f"--exclude-fold {args.exclude_fold}/{args.n_folds}: {len(keep_q):,} of {len(mi.train_query_enc):,} training queries"); tr.manifest["train_queries_used"] = int(len(keep_q))
if args.tab_tower:                                                         # STEP12d C: pure-tabular rows aligned with the 21-row bundles
    from referral_gnn.tabular import training_matrix
    X_, _, qid_, fb_ = training_matrix(mi, "tabular", log=lambda s: None); assert len(fb_.names) == 117
    al = np.empty_like(X_); r_ = np.searchsorted(mi.tc_enc[::21], qid_[::21]) * 21
    for j_, r0 in enumerate(r_): al[r0:r0 + 21] = X_[21 * j_: 21 * j_ + 21]
    tr.extra_train_tab = al
    import torch as _t
    sd_ = al.std(0); sd_ = np.where(sd_ < 1e-3, np.inf, sd_)            # columns constant in training → z = 0 everywhere (reviewer S1)
    tr.model.tab_mu.copy_(_t.as_tensor(al.mean(0))); tr.model.tab_sd.copy_(_t.as_tensor(sd_.astype(np.float32)))
if args.table_offset:                                                      # STEP12d A
    from referral_gnn.score_lookup import ScoreLookup
    import hashlib as _h
    meta_ = json.load(open(args.table_offset.replace(".npz", ".json")))
    assert int(meta_["seed"]) == args.seed and os.path.basename(meta_["table_run"].rstrip("/")) == os.path.basename(args.offset_table_run.rstrip("/")), "table OOF file does not match --seed / --offset-table-run"
    z_ = np.load(args.table_offset); tr.extra_train_off = z_["train_score"].astype(np.float32)
    tr.manifest["table_offset_npz_sha256"] = _h.sha256(open(args.table_offset, "rb").read()).hexdigest()
    tr.val_lookup = ScoreLookup(z_["look_enc"], z_["look_spec"], z_["look_score"])
if args.table_offset or args.tab_tower: tr._probe_device()               # STEP12d: device probe once the extras exist
tr.pos_weight_alpha = args.pos_weight_alpha; tr.neg_policy = args.negatives; tr.neg_k = tuple(int(x) for x in args.neg_k.split(",")); tr.train_cohort = args.cohort; tr.val_pool = args.val_pool
tr.manifest.update(selection_metric=args.select, lr_factor=args.lr_factor, lr_patience=args.lr_patience, improve_rel=tr.improve_rel, neg_policy=args.negatives, neg_k=list(tr.neg_k), train_cohort=args.cohort, val_pool=args.val_pool, pos_weight_alpha=args.pos_weight_alpha, train_end=args.train_end, history_dropout=args.history_dropout, loss=args.loss,
                   exclude_fold=args.exclude_fold, n_folds=args.n_folds, table_offset_npz=args.table_offset, offset_table_run=args.offset_table_run, tab_tower=args.tab_tower,
                   **({"arm": args.arm} if args.arm else {}))
if old_manifest is not None:
    keys = ("inputs_sha256", "model_config", "arm", "train_config", "seed", "selection_metric", "lr_factor", "lr_patience", "improve_rel", "neg_policy", "neg_k", "train_cohort", "val_pool", "n_parameters", "pos_weight_alpha", "train_end", "history_dropout", "loss", "exclude_fold", "n_folds", "table_offset_npz", "table_offset_npz_sha256", "offset_table_run", "tab_tower")
    old_get = lambda k: old_manifest.get(k, 0.0 if k in ("pos_weight_alpha", "history_dropout") else ("softmax" if k == "loss" else None))        # train_end: None for older runs        # runs before A10 had no weighting
    diff = [k for k in keys if json.loads(json.dumps(old_get(k))) != json.loads(json.dumps(tr.manifest.get(k)))]
    if diff:
        json.dump(old_manifest, open(os.path.join(run_dir, "run_manifest.json"), "w"), indent=1)   # restore the original manifest
        raise SystemExit(f"--resume refused: settings differ from the original run: {diff}")
    tr.manifest = {**old_manifest, "resumes": old_manifest.get("resumes", []) + [{"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": tr.device, "code_sha256": tr.manifest["code_sha256"]}]}
    tr.resume = True
json.dump(tr.manifest, open(os.path.join(run_dir, "run_manifest.json"), "w"), indent=1)
print("run dir:", os.path.relpath(run_dir, root), "| device:", tr.device, "| parameters:", f"{tr.model.n_parameters():,}", "| selection:", args.select)
print("model config:", mcfg.to_dict()); print("train config:", tcfg)
print("summary:", tr.fit())
