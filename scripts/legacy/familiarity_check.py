"""Slide-3 numbers for any three-node run: first-time referrals on the full pool — share of the model's top-10 slots
given to specialists this PCP referred to before (vs their share of the pool), and MRR / median rank of the
actual specialist split by whether the PCP had used them before. Usage: familiarity_check.py <run_dir> [--split validation|test]"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json, os, sys
import numpy as np, pandas as pd

from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.evaluate import ModelScorer  # noqa: E402
from referral_gnn.train import pick_device  # noqa: E402
split = "validation"
if "--split" in sys.argv: i = sys.argv.index("--split"); split = sys.argv[i + 1]; del sys.argv[i:i + 2]
run = sys.argv[1]; man = json.load(open(os.path.join(run, "run_manifest.json"))); mi = load_three_node_inputs(man["inputs_path"]); dev = pick_device("auto")
sc = ModelScorer(load_three_node_model(run, mi, dev), mi, dev); bucket = man["train_config"]["graph_bucket"]
q_all = np.sort(mi.validation_sample) if split == "validation" else np.sort(mi.query_enc[split])
fo = q_all[mi.ps_asof.lookup(mi.enc_patient[q_all] * mi.n_specialist + mi.enc_target[q_all], mi.enc_day[q_all]) < 0]
rows = []
for cutoff, be, qb in mi.buckets(split, bucket, query_subset=fo):
    sc.prepare(cutoff, be)
    for q in qb:
        q = int(q); t = int(mi.enc_target[q]); d = int(mi.enc_day[q]); pool = np.where(mi.available_specialists(d))[0]
        if t not in set(pool.tolist()): continue
        s = sc.score(q, pool); st = s[pool == t][0]; rank = int((s > st).sum() + (s == st).sum())
        prior, _, _ = mi.raw_pair_history(np.full(len(pool), q), pool); fam = prior > 0
        top10 = np.argsort(-s, kind="stable")[:10]
        rows.append({"rank": rank, "target_familiar": bool(fam[pool == t][0]), "fam_pool": fam.mean(), "fam_top10": fam[top10].mean()})
df = pd.DataFrame(rows); name = os.path.basename(run).split("_", 1)[1]
out = {"run": name, "split": split, "n": len(df), "fam_share_pool": float(df.fam_pool.mean()), "fam_share_top10": float(df.fam_top10.mean())}
for fam, lab in ((True, "familiar"), (False, "new_to_pcp")):
    d = df[df.target_familiar == fam]; out |= {f"{lab}_share": len(d) / len(df), f"{lab}_mrr": float((1 / d["rank"]).mean()), f"{lab}_median_rank": float(d["rank"].median())}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(str(workspace()), "runs", "analysis", f"familiarity_{name}_{split}.json"), "w"), indent=1)
