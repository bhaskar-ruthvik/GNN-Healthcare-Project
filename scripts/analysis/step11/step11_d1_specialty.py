"""STEP11 Amendment B5, diagnostic D1: is the tabular model's full-pool advantage on first-time referrals a matter of choosing
the right SPECIALTY, or of ranking within it? Inference only, on the 2018 first-time selection queries, full pool (every
specialist first observed before the query day — the evaluator's pool "all"), weekly graph cutoff.

Per query: the target's specialty (strictly before the day); top-1 specialty match; share of the top 5 in the target's
specialty; the target's rank among all candidates and within its own specialty. Run once per model type (GNN and LightGBM
cannot share a process on this Mac), then --compare.
    python scripts/analysis/step11/step11_d1_specialty.py --kind gnn  runs/<..>_v5_unmasked_s42 runs/<..>_v5_unmasked_s1
    python scripts/analysis/step11/step11_d1_specialty.py --kind lgbm runs/<..>_lgbm_tabular_s42 runs/<..>_lgbm_tabular_s1
    python scripts/analysis/step11/step11_d1_specialty.py --compare
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import sys
if "--kind" in sys.argv and sys.argv[sys.argv.index("--kind") + 1] == "lgbm":
    import lightgbm  # noqa: F401  (before torch)
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.step10 import specialty_strict  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="*"); ap.add_argument("--kind", choices=["gnn", "lgbm"]); ap.add_argument("--compare", action="store_true")
args = ap.parse_args(); root = str(workspace()); os.chdir(root)
OUT = "runs/analysis/step11_D1"; os.makedirs(OUT, exist_ok=True)

if args.compare:
    rows = []
    for f in sorted(glob.glob(f"{OUT}/*_PRIVATE.csv")):
        d = pd.read_csv(f); n = os.path.basename(f).replace("_PRIVATE.csv", "")
        k = d.target_specialty_known
        rows.append({"run": n, "queries": len(d), "MRR": d.rr.mean(), "top1_specialty_match": d.top1_spec[k].mean(), "top5_share_target_specialty": d.top5_spec_share[k].mean(),
                     "MRR_within_specialty": (1 / d.rank_within[k]).mean(), "median_rank_within": d.rank_within[k].median(), "median_specialty_pool": d.n_within[k].median(),
                     "target_specialty_known": k.mean()})
    t = pd.DataFrame(rows); t.to_csv(f"{OUT}/summary.csv", index=False); pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}")); sys.exit()

man = json.load(open(os.path.join(args.runs[0], "run_manifest.json"))); mi = load_three_node_inputs(man["inputs_path"])
sel = np.sort(mi.validation_sample)
fo = sel[mi.ps_asof.lookup(mi.enc_patient[sel] * mi.n_specialist + mi.enc_target[sel], mi.enc_day[sel]) < 0]
for run in args.runs:
    if args.kind == "gnn":
        from referral_gnn.three_node_train import load_three_node_model
        from referral_gnn.evaluate import ModelScorer
        from referral_gnn.train import pick_device
        dev = pick_device("auto"); sc = ModelScorer(load_three_node_model(run, mi, dev), mi, dev)
    else:
        from referral_gnn.tabular import load_baseline_scorer
        sc = load_baseline_scorer(run, mi)
    rec = []
    for cutoff, batch_enc, q_b in mi.buckets("validation", "week", query_subset=fo):
        sc.prepare(cutoff, batch_enc)
        for q in q_b:
            q = int(q); day = int(mi.enc_day[q]); avail = np.where(mi.available_specialists(day))[0]; t = int(mi.enc_target[q])
            if t not in set(avail.tolist()): continue
            s = np.asarray(sc.score(q, avail), float); tpos = int(np.where(avail == t)[0][0])
            rank = int((s > s[tpos]).sum() + (s == s[tpos]).sum())                 # pessimistic
            sp = specialty_strict(mi, avail, day); tsp = int(sp[tpos]); known = tsp > 0
            order = np.argsort(-s, kind="stable"); top5 = order[:5]
            same = sp == tsp
            rec.append({"enc": q, "rr": 1.0 / rank, "target_specialty_known": known, "top1_spec": bool(known and sp[order[0]] == tsp),
                        "top5_spec_share": float(same[top5].mean()) if known else np.nan,
                        "rank_within": int((s[same] > s[tpos]).sum() + (s[same] == s[tpos]).sum()) if known else np.nan, "n_within": int(same.sum()) if known else np.nan})
    name = os.path.basename(run.rstrip("/")).split("_", 1)[1]
    pd.DataFrame(rec).to_csv(f"{OUT}/{name}_PRIVATE.csv", index=False); print(f"{name}: {len(rec)} queries scored", flush=True)
