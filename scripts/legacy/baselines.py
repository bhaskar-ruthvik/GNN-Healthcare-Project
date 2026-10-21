"""Step 5a: non-learned baselines with the same full-pool evaluation as the model.
Default: validation subset (3,000 fixed queries). Test only with --split test (step 6).
Writes runs/baselines_<split>[_subset]/{per_query_<name>.csv, metrics.json, breakdowns.csv}.
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


from referral_gnn.data import ModelInputs, default_bundle_path  # noqa: E402
from referral_gnn.baselines import BASELINES  # noqa: E402
from referral_gnn.evaluate import evaluate, metrics, breakdowns  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="validation", choices=["validation", "test"])
ap.add_argument("--full", action="store_true", help="all queries of the split instead of the fixed validation subset")
ap.add_argument("--only", nargs="*", default=None)
args = ap.parse_args()

root = str(workspace())
mi = ModelInputs.load(os.path.join(default_bundle_path(root), "model_inputs_PRIVATE.pkl"))
subset = None if (args.full or args.split == "test") else mi.validation_sample
out_dir = os.path.join(root, "runs", f"baselines_{args.split}" + ("" if subset is None else "_subset"))
os.makedirs(out_dir, exist_ok=True)

all_metrics, all_bd = {}, []
for name, cls in BASELINES.items():
    if args.only and name not in args.only:
        continue
    t0 = time.time()
    df = evaluate(cls(mi), mi, args.split, query_subset=subset)
    m = metrics(df); m["time_s"] = time.time() - t0
    all_metrics[name] = m
    bd = breakdowns(df); bd.insert(0, "baseline", name); all_bd.append(bd)
    df.to_csv(os.path.join(out_dir, f"per_query_{name}.csv"), index=False)
    print(f"{name:17s} MRR {m['mrr_eligible']:.4f} | R@1 {m['recall@1_eligible']:.3f} R@5 {m['recall@5_eligible']:.3f} R@10 {m['recall@10_eligible']:.3f} R@20 {m['recall@20_eligible']:.3f} | all-query R@10 {m['recall@10_all']:.3f} | median rank {m['median_rank_eligible']:.0f} | {m['time_s']:.0f}s", flush=True)
with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=1)
pd.concat(all_bd).to_csv(os.path.join(out_dir, "breakdowns.csv"), index=False)
print("saved to", os.path.relpath(out_dir, root))
