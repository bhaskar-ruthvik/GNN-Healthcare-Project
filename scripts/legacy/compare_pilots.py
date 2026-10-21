"""Score several three-node runs (best.pt) and the history heuristics on the validation first-observed queries
under three candidate pools: all, unseen (patient has not seen), specialty_unseen (oracle specialty).
Usage: compare_pilots.py <run_dir> [<run_dir> ...]   (each run's own inputs are used; pools are identical across runs)
Writes runs/analysis/pilot_comparison.csv.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os
import sys

import numpy as np
import pandas as pd


from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.evaluate import ModelScorer, evaluate, metrics  # noqa: E402
from referral_gnn.baselines import PCPFrequency  # noqa: E402
from referral_gnn.train import pick_device  # noqa: E402

root = str(workspace())
device = pick_device("auto")
SPLIT = "validation"
if "--split" in sys.argv:
    i = sys.argv.index("--split"); SPLIT = sys.argv[i + 1]; del sys.argv[i:i + 2]
rows = []
cache = {}
def inputs_for(path):
    if path not in cache: cache[path] = load_three_node_inputs(path)
    return cache[path]

def first_observed(mi):
    q = np.sort(mi.validation_sample) if SPLIT == "validation" else np.sort(mi.query_enc[SPLIT])
    return q[mi.ps_asof.lookup(mi.enc_patient[q] * mi.n_specialist + mi.enc_target[q], mi.enc_day[q]) < 0]

def score(name, scorer, mi, bucket):
    fo = first_observed(mi)
    rec = {"model": name}
    for pool in ("all", "unseen", "specialty_unseen"):
        df = evaluate(scorer, mi, SPLIT, query_subset=fo, bucket=bucket, pool=pool); m = metrics(df)
        rec[f"{pool}:MRR"] = m["mrr_eligible"]; rec[f"{pool}:R@5"] = m["recall@5_eligible"]; rec[f"{pool}:R@10"] = m["recall@10_eligible"]; rec[f"{pool}:avail"] = m["target_availability"]
    rows.append(rec); print(pd.DataFrame([rec]).to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)

runs = sys.argv[1:]
first_mi = None
for run in runs:
    man = json.load(open(os.path.join(run, "run_manifest.json")))
    mi = inputs_for(man["inputs_path"]); first_mi = first_mi or mi
    model = load_three_node_model(run, mi, device)
    label = os.path.basename(run).split("_", 1)[1] + f" [{man.get('schema','?').split('-')[-1]}, negs={man.get('neg_policy','uniform')}, cohort={man.get('train_cohort','all')}, ep={json.load(open(os.path.join(run,'summary.json')))['best_epoch']}]"
    score(label, ModelScorer(model, mi, device), mi, man["train_config"]["graph_bucket"])
score("PCP frequency (heuristic)", PCPFrequency(first_mi), first_mi, "week")
out = pd.DataFrame(rows)
os.makedirs(os.path.join(root, "runs", "analysis"), exist_ok=True)
out.to_csv(os.path.join(root, "runs", "analysis", "pilot_comparison.csv" if SPLIT == "validation" else f"pilot_comparison_{SPLIT}.csv"), index=False)
print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
