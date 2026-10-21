"""Train a model. Example:
    python scripts/legacy/train.py --name full --epochs 30
    python scripts/legacy/train.py --name pilot --epochs 3 --pilot
    python scripts/legacy/train.py --name no_distance --ablation no_distance
Writes runs/<timestamp>_<name>/{run_manifest.json, metrics.jsonl, best.pt, summary.json}.
Test-split evaluation is NOT run here; see scripts/evaluate.py (step 6).
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import os
import sys
import time


from referral_gnn.data import ModelInputs, default_bundle_path  # noqa: E402
from referral_gnn.model import ModelConfig  # noqa: E402
from referral_gnn.train import TrainConfig, Trainer  # noqa: E402

ABLATIONS = ["no_diagnosis", "no_distance", "no_history_features", "no_patient_specialist_prior", "no_patient_history", "no_referral_edges", "no_time_encoding"]

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--accumulate", type=int, default=1)
ap.add_argument("--device", default="auto")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--ablation", action="append", default=[], choices=ABLATIONS)
ap.add_argument("--scorer-context", default="enc_only", choices=["enc_only", "enc_patient_pcp"])
ap.add_argument("--no-graph", action="store_true", help="PairMLP: same encoders and scorer, no message passing")
ap.add_argument("--pilot", action="store_true", help="pilot: short run, validation only")
ap.add_argument("--max-train-days", type=int, default=None)
ap.add_argument("--graph-bucket", default="day", choices=["day", "week"], help="graph history cutoff granularity; pair features stay exact")
args = ap.parse_args()

root = str(workspace())
inputs_path = os.path.join(default_bundle_path(root), "model_inputs_PRIVATE.pkl")
mi = ModelInputs.load(inputs_path)
mcfg = ModelConfig(scorer_context=args.scorer_context, use_graph=not args.no_graph, **{a: True for a in args.ablation})
tcfg = TrainConfig(lr=args.lr, max_epochs=args.epochs, patience=args.patience, accumulate_days=args.accumulate, seed=args.seed, device=args.device, max_train_days=args.max_train_days, graph_bucket=args.graph_bucket)
run_dir = os.path.join(root, "runs", time.strftime("%Y%m%dT%H%M%S") + "_" + args.name)
tr = Trainer(mi, mcfg, tcfg, run_dir, inputs_path)
print("run dir:", os.path.relpath(run_dir, root), "| device:", tr.device, "| parameters:", f"{tr.model.n_parameters():,}")
print("model config:", mcfg.to_dict())
print("train config:", tcfg)
summary = tr.fit()
print("summary:", summary)
