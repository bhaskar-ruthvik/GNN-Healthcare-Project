"""Frozen secondary metric (STEP7_PROTOCOL §6): specialty match.

For each eligible query (target in pool): does the top-1 candidate share the chosen
specialist's specialty; does any of the top-5? Specialty = latest non-null value
observed on or before the query day (cutoff-safe as-of lookup). A candidate with
missing/UNK specialty never matches. Queries whose chosen specialist has missing
specialty are excluded from the denominator and counted.

Works for three-node runs (best.pt / best_overall.pt) and the four-node main run,
and for the PatientRepeat / PCPFrequency heuristics. Validation subset by default.
Writes <run_dir>/specialty_match_<split>[_subset]_<checkpoint>.json
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


from referral_gnn.data import ModelInputs, AsOfIndex, default_bundle_path  # noqa: E402
from referral_gnn.evaluate import ModelScorer  # noqa: E402
from referral_gnn.train import pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("target", help="run dir, or baseline name: patient_repeat | pcp_frequency")
ap.add_argument("--inputs", default=None, help="inputs pickle (three-node runs); default = four-node bundle inputs")
ap.add_argument("--checkpoint", default="best.pt")
ap.add_argument("--split", default="validation", choices=["validation", "test"])
ap.add_argument("--full", action="store_true")
ap.add_argument("--bucket", default=None)
args = ap.parse_args()
root = str(workspace())

is_baseline = args.target in ("patient_repeat", "pcp_frequency")
if args.inputs:
    from referral_gnn.three_node_train import load_three_node_inputs
    mi = load_three_node_inputs(os.path.join(root, args.inputs) if not os.path.isabs(args.inputs) else args.inputs)
else:
    mi = ModelInputs.load(os.path.join(default_bundle_path(root), "model_inputs_PRIVATE.pkl"))
device = pick_device("auto")

# cutoff-safe specialty index per (specialist, day): latest non-null on or before day
sh_rows = mi.spec_hist_cat[:, 0]                      # specialty vocab index per history row (0 = UNK/missing)
nonnull = sh_rows > 0
ent = mi.spec_hist_asof.entity[np.argsort(mi.spec_hist_asof.order)]  # entity per original row
day = mi.spec_hist_day
sp_idx = AsOfIndex(ent[nonnull], day[nonnull]); sp_val = sh_rows[nonnull]
def specialty_asof(spec_ids, d):
    rows = sp_idx.lookup(np.asarray(spec_ids), int(d) + 1)      # <= d
    return np.where(rows >= 0, sp_val[np.clip(rows, 0, None)], 0)

if is_baseline:
    from referral_gnn.baselines import BASELINES
    scorer = BASELINES[args.target](mi); bucket = args.bucket or "day"; label = args.target
else:
    with open(os.path.join(args.target, "run_manifest.json")) as f:
        man = json.load(f)
    bucket = args.bucket or man["train_config"]["graph_bucket"]
    if str(man.get("schema", "")).startswith("three-node"):
        from referral_gnn.three_node_train import load_three_node_model
        model = load_three_node_model(args.target, mi, device, args.checkpoint)
    else:
        from referral_gnn.train import load_model
        model = load_model(args.target, mi, device, args.checkpoint)
    scorer = ModelScorer(model, mi, device); label = os.path.basename(args.target)

subset = None if (args.full or args.split == "test") else mi.validation_sample
queries = np.sort(mi.query_enc[args.split] if subset is None else subset)
rows = []
for cutoff, batch_enc, q_b in mi.buckets(args.split, bucket, query_subset=queries):
    scorer.prepare(cutoff, batch_enc)
    for q in q_b:
        q = int(q); t = int(mi.enc_target[q]); d = int(mi.enc_day[q])
        avail = np.where(mi.available_specialists(d))[0]
        if t not in set(avail.tolist()):
            continue
        s = scorer.score(q, avail)
        order = np.argsort(-s, kind="stable")
        top5 = avail[order[:5]]
        sp_t = specialty_asof([t], d)[0]
        sp_top = specialty_asof(top5, d)
        repeat = bool(mi.ps_asof.lookup(np.array([mi.enc_patient[q] * mi.n_specialist + t]), d)[0] >= 0)
        rows.append({"enc": q, "patient_repeat": repeat, "target_specialty_known": sp_t > 0,
                     "top1_match": bool(sp_t > 0 and sp_top[0] == sp_t), "top5_match": bool(sp_t > 0 and (sp_top == sp_t).any())})
df = pd.DataFrame(rows)
known = df[df.target_specialty_known]
def summ(d):
    return {"n": int(len(d)), "top1_specialty_match": float(d.top1_match.mean()) if len(d) else None, "top5_specialty_match": float(d.top5_match.mean()) if len(d) else None}
res = {"target": label, "checkpoint": None if is_baseline else args.checkpoint, "split": args.split, "subset": subset is not None, "bucket": bucket,
       "n_eligible": int(len(df)), "n_excluded_missing_target_specialty": int((~df.target_specialty_known).sum()),
       "all": summ(known), "firstobs": summ(known[~known.patient_repeat]), "repeat": summ(known[known.patient_repeat])}
print(json.dumps(res, indent=1))
out_dir = args.target if not is_baseline else os.path.join(root, "runs", f"baselines_{args.split}" + ("" if subset is None else "_subset"))
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, f"specialty_match_{args.split}{'' if subset is None else '_subset'}_{(args.checkpoint if not is_baseline else args.target).replace('.pt','')}.json"), "w") as f:
    json.dump(res, f, indent=1)
