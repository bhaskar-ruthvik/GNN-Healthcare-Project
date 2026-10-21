"""Step 6 analysis: assemble validation tables, paired patient-cluster bootstrap
comparisons, seed spread. Reads per-query CSVs written by eval_checkpoint.py
and baselines.py. Writes runs/analysis/{validation_table.csv, bootstrap.csv,
seed_spread.csv} and prints them.

Usage: analyze.py [--split validation|test] [--bootstrap 1000]
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


from referral_gnn.evaluate import metrics  # noqa: E402
from referral_gnn.data import ModelInputs, default_bundle_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="validation")
ap.add_argument("--bootstrap", type=int, default=1000)
ap.add_argument("--ref", default="main_s42:best", help="model key used as 'a' in the paired bootstrap")
ap.add_argument("--only", default=None, help="regex; keep only result sets whose key matches (baselines always kept)")
args = ap.parse_args()
root = str(workspace())
mi = ModelInputs.load(os.path.join(default_bundle_path(root), "model_inputs_PRIVATE.pkl"))
out_dir = os.path.join(root, "runs", "analysis"); os.makedirs(out_dir, exist_ok=True)
sub = "_subset" if args.split == "validation" else ""

# ---- collect per-query results ---------------------------------------------------------------
results = {}
for f in glob.glob(os.path.join(root, "runs", f"baselines_{args.split}{sub}", "per_query_*.csv")):
    results["baseline:" + os.path.basename(f)[len("per_query_"):-4]] = pd.read_csv(f)
for run in sorted(glob.glob(os.path.join(root, "runs", "2*_*"))):
    name = os.path.basename(run).split("_", 1)[1]
    for tag, ck in ((f"eval_{args.split}{sub}_week", "best"), (f"eval_{args.split}{sub}_week_best_firstobs", "best_firstobs"), (f"eval_{args.split}{sub}_week_best_overall", "best_overall")):
        f = os.path.join(run, tag, "per_query.csv")
        if os.path.exists(f):
            results[f"{name}:{ck}"] = pd.read_csv(f)
if args.only:
    import re
    results = {k: v for k, v in results.items() if k.startswith("baseline:") or re.search(args.only, k)}
if not results:
    sys.exit("no per-query results found")
ref_idx = None
for k, df in results.items():
    df = df.set_index("enc").sort_index(); results[k] = df
    ref_idx = df.index if ref_idx is None else ref_idx
    assert df.index.equals(ref_idx), f"{k}: query set differs"
print(f"{len(results)} result sets on {len(ref_idx)} identical {args.split} queries\n")

# ---- table: overall + subgroups ------------------------------------------------------------------
def row(name, df):
    m = metrics(df); fo = metrics(df[~df.patient_repeat]); rp = metrics(df[df.patient_repeat]); uf = metrics(df[~df.familiar_pair])
    return {"model": name, "MRR": m["mrr_eligible"], "R@1": m["recall@1_eligible"], "R@10": m["recall@10_eligible"], "R@20": m["recall@20_eligible"],
            "MRR_all": m["mrr_all"], "firstobs_MRR": fo["mrr_eligible"], "firstobs_R@10": fo["recall@10_eligible"], "firstobs_R@20": fo["recall@20_eligible"],
            "repeat_MRR": rp["mrr_eligible"], "unfamiliar_MRR": uf["mrr_eligible"], "unfamiliar_R@20": uf["recall@20_eligible"], "n": m["n_queries"], "n_eligible": m["n_eligible"]}
table = pd.DataFrame([row(k, v) for k, v in results.items()]).sort_values("MRR", ascending=False)
table.to_csv(os.path.join(out_dir, f"{args.split}_table{'' if not args.only else '_' + args.only.replace('|','_').replace('*','')}.csv"), index=False)
print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---- paired patient-cluster bootstrap ---------------------------------------------------------------
def rr(df):
    r = df["rank"].to_numpy(float); return np.where(np.isfinite(r), 1.0 / np.where(np.isfinite(r), r, 1.0), 0.0)   # all-query reciprocal rank (miss = 0)

def paired_bootstrap(a, b, mask=None, n=1000, seed=0):
    """Difference in mean reciprocal rank (a − b) over eligible queries, resampling patients as clusters."""
    rng = np.random.default_rng(seed)
    elig = np.isfinite(a["rank"].to_numpy(float)) & np.isfinite(b["rank"].to_numpy(float))
    if mask is not None: elig &= mask
    d = (rr(a) - rr(b))[elig]
    pat = mi.enc_patient[a.index.to_numpy()][elig]
    groups = pd.Series(d).groupby(pat).agg(["sum", "count"])
    sums, counts = groups["sum"].to_numpy(), groups["count"].to_numpy()
    G = len(sums)
    boots = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, G, G)
        boots[i] = sums[idx].sum() / counts[idx].sum()
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), int(elig.sum()), G

comparisons = []
main = args.ref if args.ref in results else None
if main is None: print(f"reference {args.ref} not found; skipping bootstrap")
if main:
    pairs = [(main, k) for k in results if k != main]
    for a_k, b_k in pairs:
        for sub_name, mask in (("all_eligible", None), ("firstobs", ~results[a_k]["patient_repeat"].to_numpy()), ("repeat", results[a_k]["patient_repeat"].to_numpy())):
            mean, lo, hi, n_q, G = paired_bootstrap(results[a_k], results[b_k], mask, n=args.bootstrap)
            comparisons.append({"a": a_k, "b": b_k, "subset": sub_name, "MRR_diff(a-b)": mean, "ci95_lo": lo, "ci95_hi": hi, "n_queries": n_q, "n_patient_clusters": G, "significant": (lo > 0) or (hi < 0)})
    comp = pd.DataFrame(comparisons)
    comp.to_csv(os.path.join(out_dir, f"{args.split}_bootstrap{'' if main == 'main_s42:best' else '_' + main.replace(':', '_')}.csv"), index=False)
    print(f"\nPaired patient-cluster bootstrap ({args.bootstrap} resamples), MRR difference {main} − other:")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

# ---- seed spread ------------------------------------------------------------------------------------
seeds = {k: v for k, v in results.items() if k.endswith(":best") and (k.startswith("main_s") or k.startswith("tn_A_s") or k.startswith("tn_B_s"))}
if len(seeds) > 1:
    ss = table[table.model.isin(seeds)].set_index("model")[["MRR", "R@10", "firstobs_MRR", "firstobs_R@10", "repeat_MRR"]]
    spread = pd.DataFrame({"mean": ss.mean(), "std": ss.std(ddof=1), "min": ss.min(), "max": ss.max()})
    spread.to_csv(os.path.join(out_dir, f"{args.split}_seed_spread.csv"))
    print("\nSeed spread (main_s42/s1/s2):"); print(spread.to_string(float_format=lambda x: f"{x:.4f}"))
print("\nsaved to", os.path.relpath(out_dir, root))
