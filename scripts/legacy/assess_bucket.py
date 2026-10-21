"""Step 5c: week-vs-day cutoff assessment on the validation subset.

Evaluates one or more trained checkpoints under BOTH graph cutoffs (the
week policy delays graph history and provider snapshots by up to six days),
and reports how far the two disagree. Validation subset only; test untouched.
Usage: assess_bucket.py <run_dir> [<run_dir> ...]
Writes <run_dir>/bucket_assessment.json for each run.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os
import sys

import numpy as np


from referral_gnn.data import ModelInputs, default_bundle_path  # noqa: E402
from referral_gnn.evaluate import ModelScorer, evaluate, metrics  # noqa: E402
from referral_gnn.train import load_model, pick_device  # noqa: E402

root = str(workspace())
mi4 = None
device = pick_device("auto")
for run_dir in sys.argv[1:]:
    with open(os.path.join(run_dir, "run_manifest.json")) as f:
        man = json.load(f)
    trained_with = man["train_config"]["graph_bucket"]
    if man.get("schema") == "three-node-v1":
        from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model
        mi = load_three_node_inputs(man["inputs_path"]); model = load_three_node_model(run_dir, mi, device)
    else:
        if mi4 is None: mi4 = ModelInputs.load(os.path.join(default_bundle_path(root), "model_inputs_PRIVATE.pkl"))
        mi = mi4; model = load_model(run_dir, mi, device)
    res = {"trained_with": trained_with}
    dfs = {}
    for pol in ("week", "day"):
        df = evaluate(ModelScorer(model, mi, device), mi, "validation", query_subset=mi.validation_sample, bucket=pol)
        dfs[pol] = df.set_index("enc"); res[pol] = metrics(df)
        m = res[pol]
        print(f"{os.path.basename(run_dir)} | trained {trained_with:4s} | eval {pol:4s} | MRR {m['mrr_eligible']:.4f} R@1 {m['recall@1_eligible']:.3f} R@5 {m['recall@5_eligible']:.3f} R@10 {m['recall@10_eligible']:.3f} R@20 {m['recall@20_eligible']:.3f}", flush=True)
    a, b = dfs["week"]["rank"], dfs["day"].loc[dfs["week"].index, "rank"]
    ok = a.notna() & b.notna()
    res["rank_disagreement_share"] = float((a[ok] != b[ok]).mean())
    res["median_abs_rank_diff"] = float(np.median(np.abs(a[ok] - b[ok])))
    res["mrr_gap_day_minus_week"] = res["day"]["mrr_eligible"] - res["week"]["mrr_eligible"]
    print(f"   ranks differ for {res['rank_disagreement_share']:.1%} of eligible queries; median |Δrank| {res['median_abs_rank_diff']:.0f}; MRR(day) − MRR(week) = {res['mrr_gap_day_minus_week']:+.4f}")
    with open(os.path.join(run_dir, "bucket_assessment.json"), "w") as f:
        json.dump(res, f, indent=1)
