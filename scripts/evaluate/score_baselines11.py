"""Score Step 11.1 baseline runs on the STEP10 cohort's eligible sets, in a process that runs no torch computation.

Why a separate process: on this Mac (2026-10-07), torch (MPS or CPU) and lib_lightgbm cannot both execute in one process —
with lightgbm loaded first the GNN forward segfaults; with torch first LightGBM prediction segfaults (both reproduced).
step10_evaluate.py calls this script for baseline runs and loads the saved scores; it checks that the referral ids and every
eligible set match its own cohort exactly before using them.

Usage (normally invoked by step10_evaluate.py):
    python scripts/evaluate/score_baselines11.py --inputs <pkl> --split validation --out <file.npz> [cohort flags] runs/<baseline> ...
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
try:
    import lightgbm  # noqa: F401  (must precede torch, which referral_gnn imports but never runs here)
except ImportError:
    pass
import argparse
import hashlib
import os
import sys
import time

import numpy as np


from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.step10 import Step10Cohort, load_yardstick  # noqa: E402
from referral_gnn.tabular import load_baseline_scorer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+"); ap.add_argument("--inputs", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cap", default="25"); ap.add_argument("--active", default="365"); ap.add_argument("--specialties", default="primary")
ap.add_argument("--insurer-set", action="store_true"); ap.add_argument("--include-seen", action="store_true")
ap.add_argument("--ablate", default="", help="STEP12a B1: comma list of no_volume,no_volume_plus; each baseline is also scored with it (key name@abl)")
ap.add_argument("--list-size", type=int, default=5, help="STEP12 top-K: capped sets filled to K (must match step10_evaluate)")
args = ap.parse_args()

mi = load_three_node_inputs(args.inputs); y = load_yardstick(mi)
CAP = None if args.cap == "none" else float(args.cap); ACTIVE = None if args.active == "ever" else 365
R = list(Step10Cohort(mi, args.split, y, cap=CAP, active_days=ACTIVE, keep_specialties=args.specialties,
                      insurer_set=args.insurer_set, include_seen=args.include_seen, fill_to=args.list_size).referrals())
enc_ids = np.array([r.enc for r in R], np.int64); pos_of = {int(e): i for i, e in enumerate(enc_ids)}
lengths = np.array([len(r.eligible) for r in R], np.int64)
elig_hash = hashlib.sha256(np.concatenate([r.eligible for r in R]).astype(np.int64).tobytes()).hexdigest()
out = {"enc_ids": enc_ids, "lengths": lengths, "eligible_sha256": np.array(elig_hash)}
ABL = [a for a in args.ablate.split(",") if a]
assert set(ABL) <= {"no_volume", "no_volume_plus"}, ABL
for run in args.runs:
    for abl in [None] + ABL:
        name = os.path.basename(run.rstrip("/")); t0 = time.time(); sc = load_baseline_scorer(run, mi, ablate=abl); scores = [None] * len(R)
        for cutoff, batch_enc, q_b in mi.buckets(args.split, "week", query_subset=np.sort(enc_ids)):
            sc.prepare(cutoff, batch_enc)
            for q in q_b:
                i = pos_of[int(q)]; scores[i] = np.asarray(sc.score(int(q), R[i].eligible), float)
        key = name if abl is None else f"{name}@{abl}"
        out["scores::" + key] = np.concatenate(scores)
        print(f"  scored {key} ({time.time() - t0:.0f}s)", flush=True)
np.savez(args.out, **out)
print("saved", args.out, flush=True)
