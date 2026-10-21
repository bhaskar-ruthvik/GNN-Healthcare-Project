"""STEP12b H2: GNN embeddings + exploratory variables → logistic regression (referral_gnn/hybrid.py). torch process only.

Training rows = the GNN's 21-row bundles (weekly buckets, identical to every model). C ∈ {0.1, 1} selected on the validation
selection subset with the shared selection criterion (first-time MRR; selection only, never a judging metric). Writes one run
directory (kind "emb_lr") per GNN seed; the GNN checkpoint is referenced, not copied. Standardised coefficients of the
exploratory variables are saved in the manifest (interpretability).
Usage (data paths resolve against the workspace, see referral_gnn/paths.py):
  python scripts/train/step12b_emb_lr.py --gnn-run runs/20260929T213930_v5_unmasked_s42
  python scripts/train/step12b_emb_lr.py --gnn-run runs/20260929T213930_v5_unmasked_s42 --no-emb
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, json, os, pickle, sys, time
import numpy as np


from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.train import pick_device, sha256_file, code_hash  # noqa: E402
from referral_gnn.evaluate import evaluate, metrics  # noqa: E402
from referral_gnn.hybrid import EmbFeatures, EmbLR, EmbLRScorer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gnn-run", required=True); ap.add_argument("--no-emb", action="store_true"); ap.add_argument("--emb-mode", default="pair", choices=["pair", "product"]); ap.add_argument("--max-queries", type=int, default=None, help="smoke tests only")
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
from referral_gnn.paths import resolve_run  # noqa: E402
args.gnn_run = resolve_run(args.gnn_run)
gman = json.load(open(os.path.join(args.gnn_run, "run_manifest.json"))); inputs_path = gman["inputs_path"]
mi = load_three_node_inputs(inputs_path); device = pick_device("auto"); model = load_three_node_model(args.gnn_run, mi, device)
feats = EmbFeatures(model, mi, device, use_emb=not args.no_emb, emb_mode=args.emb_mode); seed = int(gman["seed"])
arm = "s12b_vars_lr" if args.no_emb else ("s12b_emb_lr" if args.emb_mode == "pair" else "s12b_embprod_lr")

# training matrix: the GNN's own bundles, weekly buckets
tq = mi.train_query_enc if args.max_queries is None else mi.train_query_enc[: args.max_queries]
rows_of = lambda encs: np.searchsorted(mi.tc_enc[::21], encs) * 21
pair_all = mi.pair_features(mi.tc_enc, mi.tc_spec)
Xs, ys = [], []; t0 = time.time()
for cutoff, batch_enc, q_b in mi.buckets("train", "week", query_subset=tq):
    feats.prepare(cutoff, batch_enc)
    r = (rows_of(np.asarray(q_b))[:, None] + np.arange(21)[None, :]).reshape(-1)
    assert (mi.tc_label[r].reshape(-1, 21)[:, 0] == 1).all()
    Xs.append(feats.rows(mi.tc_enc[r], mi.tc_spec[r], pair_all[r])); ys.append(mi.tc_label[r])
X = np.concatenate(Xs).astype(np.float32); y = np.concatenate(ys).astype(np.int8); del Xs, ys
print(f"[{arm}] training matrix {X.shape} in {time.time() - t0:.0f}s (GNN {os.path.basename(args.gnn_run)})", flush=True)

sel = np.sort(mi.validation_sample)
fo_sel = sel[mi.ps_asof.lookup(mi.enc_patient[sel] * mi.n_specialist + mi.enc_target[sel], mi.enc_day[sel]) < 0]
tuning, best = [], None
for C in (0.1, 1.0):
    t1 = time.time(); lr = EmbLR(C=C, seed=seed).fit(X, y)
    m = metrics(evaluate(EmbLRScorer(feats, lr), mi, "validation", query_subset=fo_sel, bucket="week", pool="all"))
    row = {"C": C, "fit_s": round(time.time() - t1, 1), "firstobs_mrr_eligible": m["mrr_eligible"], "firstobs_recall@5_eligible": m["recall@5_eligible"]}
    tuning.append(row); print(f"[{arm}] {row}", flush=True)
    if best is None or row["firstobs_mrr_eligible"] > best[0]["firstobs_mrr_eligible"]: best = (row, lr)
row, lr = best
coef = dict(zip(feats.names, lr.lr.coef_[0].tolist()))
run_dir = os.path.join(root, "runs", time.strftime("%Y%m%dT%H%M%S") + f"_{arm}_s{seed}"); os.makedirs(run_dir)
pickle.dump(lr, open(os.path.join(run_dir, "model.pkl"), "wb"))
man = {"kind": "emb_lr", "arm": arm, "step": "12b", "protocol": "STEP12B_HYBRID_PLAN.md", "seed": seed, "gnn_run": os.path.relpath(args.gnn_run, root),
       "use_emb": not args.no_emb, "emb_mode": args.emb_mode, "params": {"C": row["C"]}, "feature_names": feats.names, "inputs_path": inputs_path, "inputs_sha256": sha256_file(inputs_path),
       "code_sha256": code_hash(str(code_file("referral_gnn"))), "train_config": {"graph_bucket": "week"}, "selection_metric": "firstobs_mrr_eligible",
       "tuning": tuning, "standardised_coefficients_exploratory": {k: v for k, v in coef.items() if not k.startswith("emb_")},
       "smoke_max_queries": args.max_queries, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(man, open(os.path.join(run_dir, "run_manifest.json"), "w"), indent=1)
json.dump({"selection_metric": "firstobs_mrr_eligible", "best_val_selection": row["firstobs_mrr_eligible"], "params": man["params"]}, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
print(f"wrote {os.path.relpath(run_dir, root)}", flush=True)
