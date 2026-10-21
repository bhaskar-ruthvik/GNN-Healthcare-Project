"""Evaluate a trained run's best checkpoint with full candidate pools.

Default: validation subset. --full for all queries of the split. --split test
is reserved for step 6 (final evaluation) and must be requested explicitly.
Writes <run_dir>/eval_<split>[_subset][_<bucket>]/{per_query.csv, metrics.json, breakdowns.csv}.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import json
import os
import sys
import time


from referral_gnn.data import ModelInputs, default_bundle_path  # noqa: E402
from referral_gnn.evaluate import ModelScorer, evaluate, metrics, breakdowns  # noqa: E402
from referral_gnn.train import load_model, pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--split", default="validation", choices=["validation", "test"])
ap.add_argument("--full", action="store_true")
ap.add_argument("--bucket", default=None, help="graph cutoff policy; default = the run's training policy")
ap.add_argument("--isolate", default=None, choices=[None, "query", "patient"])
ap.add_argument("--checkpoint", default="best.pt", help="best.pt (overall-selected) or best_firstobs.pt")
ap.add_argument("--pool", default="all", choices=["all", "unseen", "specialty_unseen"])
ap.add_argument("--radius", type=float, default=None, help="patient-origin radius in miles (v2 inputs)")
args = ap.parse_args()

root = str(workspace())
with open(os.path.join(args.run_dir, "run_manifest.json")) as f:
    manifest = json.load(f)
device = pick_device("auto")
if manifest.get("schema") == "three-node-v1":
    from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model
    mi = load_three_node_inputs(manifest["inputs_path"])
    model = load_three_node_model(args.run_dir, mi, device, args.checkpoint)
else:
    mi = ModelInputs.load(os.path.join(default_bundle_path(root), "model_inputs_PRIVATE.pkl"))
    model = load_model(args.run_dir, mi, device, args.checkpoint)
bucket = args.bucket or manifest["train_config"]["graph_bucket"]
subset = None if (args.full or args.split == "test") else mi.validation_sample
tag = f"eval_{args.split}" + ("" if subset is None else "_subset") + f"_{bucket}" + (f"_iso{args.isolate}" if args.isolate else "") + ("" if args.checkpoint == "best.pt" else "_" + args.checkpoint.replace(".pt", "")) + ("" if args.pool == "all" else "_" + args.pool) + ("" if args.radius is None else f"_r{int(args.radius)}")
out = os.path.join(args.run_dir, tag); os.makedirs(out, exist_ok=True)

t0 = time.time()
df = evaluate(ModelScorer(model, mi, device), mi, args.split, query_subset=subset, bucket=bucket, isolate=args.isolate, log_every=50, pool=args.pool, radius_miles=args.radius)
m = metrics(df); m["time_s"] = time.time() - t0; m["bucket"] = bucket; m["split"] = args.split; m["n_queries"] = int(len(df)); m["checkpoint"] = args.checkpoint; m["pool"] = args.pool; m["radius_miles"] = args.radius
bd = breakdowns(df)
df.to_csv(os.path.join(out, "per_query.csv"), index=False)
bd.to_csv(os.path.join(out, "breakdowns.csv"), index=False)
with open(os.path.join(out, "metrics.json"), "w") as f:
    json.dump(m, f, indent=1)
print(f"{os.path.basename(args.run_dir)} | {args.split}{'' if subset is None else ' subset'} | bucket {bucket} | MRR {m['mrr_eligible']:.4f} R@1 {m['recall@1_eligible']:.3f} R@5 {m['recall@5_eligible']:.3f} R@10 {m['recall@10_eligible']:.3f} R@20 {m['recall@20_eligible']:.3f} | all-query MRR {m['mrr_all']:.4f} | availability {m['target_availability']:.3f} | {m['time_s']:.0f}s")
print(bd.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("saved to", os.path.relpath(out, root))
