"""STEP13 MLP cells (STEP13_GRAPH_VALUE_PLAN.md §1 and §6, frozen 2026-10-21): MLP-RF (--feature-set flat_rf) and MLP-info
(--feature-set full). The model and its training rules are in referral_gnn/mlp_ranker.py.

Tune mode (the plan's protocol): hidden {128, 256} × dropout {0.1, 0.3} at seed 42; the configuration with the best first-time
MRR on the 2018 selection sample is then trained at seeds 1 and 7. Every run gets its own directory
runs/<time>_s13_mlp_<set>_h<H>_d<D>_s<seed> with arm s13_mlp_<set>_h<H>_d<D>, so the chosen configuration's three seeds form one
arm and the other tuning configurations stay separate. The family's tuning ledger is written to
runs/analysis/step13/mlp_<set>_tuning.json.
Fixed mode: --hidden H --dropout D --seeds ... (smoke tests, resumed families).
Usage: python scripts/train/step13_mlp.py --feature-set flat_rf --mode tune
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.evaluate import evaluate, metrics  # noqa: E402
from referral_gnn.train import sha256_file, code_hash, set_seed  # noqa: E402
from referral_gnn import tabular as T  # noqa: E402
from referral_gnn.mlp_ranker import MLPRanker, MLPPredict, split_columns  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", default="prepared_data/three_node_v5/model_inputs_PRIVATE.pkl")
ap.add_argument("--feature-set", required=True, choices=["flat_rf", "full"])
ap.add_argument("--mode", required=True, choices=["tune", "fixed"])
ap.add_argument("--hidden", type=int, default=None); ap.add_argument("--dropout", type=float, default=None)
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--batch-queries", type=int, default=512, help="21-row bundles per optimiser step (design manifest)")
ap.add_argument("--max-epochs", type=int, default=100)
ap.add_argument("--max-train-queries", type=int, default=None, help="smoke tests only: first N training queries")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=6)
args = ap.parse_args()
assert args.mode == "tune" or (args.hidden and args.dropout is not None), "--mode fixed needs --hidden and --dropout"
torch.set_num_threads(args.threads)

GRID = [(h, d) for h in (128, 256) for d in (0.1, 0.3)]
LR, WD, LR_FACTOR, LR_PATIENCE, PATIENCE, IMPROVE_REL = 1e-3, 1e-5, 0.5, 2, 5, 1e-4     # LR rules as the GNN's final-model settings
root = str(workspace()); os.chdir(root)
inputs_path = os.path.join(root, args.inputs)
mi = load_three_node_inputs(inputs_path)
INPUTS_SHA = sha256_file(inputs_path); CODE_SHA = code_hash(str(code_file("referral_gnn")))
SCRIPT_SHA = sha256_file(__file__)
sel = np.sort(mi.validation_sample)
fo_sel = sel[mi.ps_asof.lookup(mi.enc_patient[sel] * mi.n_specialist + mi.enc_target[sel], mi.enc_day[sel]) < 0]
end_day = None
if args.max_train_queries:
    end_day = int(mi.enc_day[np.sort(mi.train_query_enc)[args.max_train_queries]])

t0 = time.time(); X, y, qid, fb = T.training_matrix(mi, args.feature_set, log=lambda s: None, end_day=end_day)
assert len(X) % 21 == 0 and (y.reshape(-1, 21)[:, 0] == 1).all() and y.reshape(-1, 21)[:, 1:].sum() == 0, "rows must be contiguous 21-groups, target first"
assert np.isfinite(X).all(), "non-finite feature values: the MLP has no missing-value handling beyond the missing flags"
Xn, Xc = split_columns(X, fb.cat_idx); del X
SC_MU = Xn.mean(0); SC_SD = Xn.std(0); SC_SD = np.where(SC_SD < 1e-6, np.inf, SC_SD)                   # training-only scaler; constant columns → 0
n_q = len(Xn) // 21
print(f"[s13_mlp_{args.feature_set}] training matrix {n_q:,} bundles × {Xn.shape[1]} numeric + {Xc.shape[1]} categorical in {time.time() - t0:.0f}s", flush=True)
Xn_t, Xc_t = torch.as_tensor(Xn).view(n_q, 21, -1), torch.as_tensor(Xc).view(n_q, 21, -1)


SCORER = T.TabularScorer(mi, None, args.feature_set)                   # one feature builder for every selection pass


def select_metric(model):
    SCORER.model = MLPPredict(model, fb.cat_idx, args.device)
    t1 = time.time(); m = metrics(evaluate(SCORER, mi, "validation", query_subset=fo_sel, bucket="week", pool="all"))
    return {"firstobs_mrr_eligible": m["mrr_eligible"], "firstobs_recall@5_eligible": m["recall@5_eligible"], "eval_s": round(time.time() - t1, 1)}


def train_one(hidden, dropout, seed):
    arm = f"s13_mlp_{args.feature_set}_h{hidden}_d{dropout:g}"; name = f"{arm}_s{seed}"
    run_dir = os.path.join(root, "runs", time.strftime("%Y%m%dT%H%M%S") + "_" + name); os.makedirs(run_dir)
    set_seed(seed); rng = np.random.default_rng(seed)
    model = MLPRanker(Xn.shape[1], fb.cat_sizes, hidden, dropout).to(args.device)
    model.mu.copy_(torch.as_tensor(SC_MU)); model.sd.copy_(torch.as_tensor(SC_SD.astype(np.float32)))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, threshold=IMPROVE_REL, threshold_mode="rel")
    man = {"kind": "mlp", "arm": arm, "step": "13", "protocol": "STEP13_GRAPH_VALUE_PLAN.md (frozen 2026-10-21)", "seed": seed,
           "feature_set": args.feature_set, "feature_names": fb.names, "cat_sizes": fb.cat_sizes, "hidden": hidden, "dropout": dropout,
           "params": {"lr": LR, "weight_decay": WD, "lr_factor": LR_FACTOR, "lr_patience": LR_PATIENCE, "patience": PATIENCE, "improve_rel": IMPROVE_REL,
                      "batch_queries": args.batch_queries, "max_epochs": args.max_epochs, "bundle": 21, "loss": "softmax over each 21-row bundle"},
           "inputs_path": inputs_path, "inputs_sha256": INPUTS_SHA, "code_sha256": CODE_SHA, "script_sha256": SCRIPT_SHA, "device": args.device,
           "train_config": {"graph_bucket": "week"}, "n_train_bundles": int(n_q), "max_train_queries": args.max_train_queries,
           "selection_metric": "firstobs_mrr_eligible", "selection_queries": int(len(fo_sel)), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(man, open(os.path.join(run_dir, "run_manifest.json"), "w"), indent=1)
    best, best_epoch, bad = -np.inf, -1, 0
    improved = lambda new, old: new > old * (1.0 + IMPROVE_REL) if old > -np.inf else True
    for epoch in range(1, args.max_epochs + 1):
        model.train(); t1 = time.time(); perm = rng.permutation(n_q); tot = 0.0
        for a in range(0, n_q, args.batch_queries):
            b = torch.as_tensor(perm[a:a + args.batch_queries]); xn = Xn_t[b].reshape(-1, Xn.shape[1]).to(args.device); xc = Xc_t[b].reshape(-1, Xc.shape[1]).to(args.device)
            s = model(xn, xc).view(len(b), 21)
            loss = F.cross_entropy(s, torch.zeros(len(b), dtype=torch.long, device=args.device))      # target in column 0
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        train_s = round(time.time() - t1, 1); va = select_metric(model); sched.step(va["firstobs_mrr_eligible"])
        rec = {"epoch": epoch, "train_loss": tot / n_q, "epoch_time_s": train_s, "lr": opt.param_groups[0]["lr"], **{f"val_{k}": v for k, v in va.items()}}
        open(os.path.join(run_dir, "metrics.jsonl"), "a").write(json.dumps(rec) + "\n")
        print(f"  {name} epoch {epoch:3d} | loss {rec['train_loss']:.4f} | first-time MRR {va['firstobs_mrr_eligible']:.4f} | {rec['epoch_time_s']:.0f}s + val {va['eval_s']:.0f}s", flush=True)
        if improved(va["firstobs_mrr_eligible"], best):
            best, best_epoch, bad = va["firstobs_mrr_eligible"], epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_metrics": va, "inputs_sha256": INPUTS_SHA}, os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
        if bad >= PATIENCE: break
    summary = {"selection_metric": "firstobs_mrr_eligible", "best_val_selection": best, "best_epoch": best_epoch, "epochs_run": epoch}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    print(f"  wrote {os.path.relpath(run_dir, root)} (best epoch {best_epoch}, first-time MRR {best:.4f})", flush=True)
    return {"run": os.path.basename(run_dir), "hidden": hidden, "dropout": dropout, "seed": seed, **summary}


if args.mode == "tune":
    ledger = [train_one(h, d, 42) for h, d in GRID]
    choice = max(ledger, key=lambda r: r["best_val_selection"])                 # ties: first in grid order
    finals = [train_one(choice["hidden"], choice["dropout"], s_) for s_ in (1, 7)]
    out = os.path.join(root, "runs", "analysis", "step13", f"mlp_{args.feature_set}_tuning.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"feature_set": args.feature_set, "grid": GRID, "selection_seed": 42, "ledger": ledger, "chosen": choice, "final_seed_runs": finals,
               "cell_runs": [choice["run"]] + [f["run"] for f in finals], "code_sha256": CODE_SHA, "script_sha256": SCRIPT_SHA,
               "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(out, "w"), indent=1)
    print("tuning ledger:", os.path.relpath(out, root))
else:
    for s_ in args.seeds: train_one(args.hidden, args.dropout, s_)
print(f"=== s13_mlp_{args.feature_set} {args.mode} done {time.strftime('%H:%M:%S')}", flush=True)
