"""Snapshot builder checks (STEP11 §11.4(c)).

(a) regression: compare every array and fitted object of two input pickles (e.g. snapshots/REF vs prepared_data/three_node_v5).
(b) leakage:    the same comparison between a snapshot built normally and one built with post-cutoff data shuffled.
Paths, timestamps and hashes of files are reported but not compared. Exit code 1 on any difference.

    python scripts/checks/snapshot_compare.py snapshots/REF/prepared_data/three_node_v5 prepared_data/three_node_v5
    python scripts/checks/snapshot_compare.py --only-fitted snapshots/S16/... snapshots/S16_perturbed/...
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import os
import sys

import numpy as np
import pandas as pd


from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--only-fitted", action="store_true", help="(b): compare only objects fitted on the training split + training arrays")
args = ap.parse_args()
SKIP = ("path", "bundle", "sha", "created", "manifest", "seed_info")
FITTED = ("encoder", "scaler", "vocab", "dx_vocab", "mips_imputed", "imputed", "age_scaler", "distance_scaler", "patient_distance_scaler",
          "train_", "tc_", "insurer_index", "race_vocab", "severity_vocab", "dims")

def load(p):
    p = p if p.endswith(".pkl") else os.path.join(p, "model_inputs_PRIVATE.pkl"); return load_three_node_inputs(p)

diffs, n_cmp = [], 0
def cmp(x, y, where, depth=0):
    global n_cmp
    if depth > 6: return
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        n_cmp += 1; xa, ya = np.asarray(x), np.asarray(y)
        if xa.shape != ya.shape: diffs.append(f"{where}: shape {xa.shape} vs {ya.shape}"); return
        eq = np.array_equal(xa, ya, equal_nan=True) if xa.dtype.kind in "fc" else np.array_equal(xa, ya)
        if not eq: diffs.append(f"{where}: values differ ({int((xa != ya).sum()) if xa.dtype.kind not in 'O' else '?'} cells)")
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        n_cmp += 1
        try: pd.testing.assert_frame_equal(x, y) if isinstance(x, pd.DataFrame) else pd.testing.assert_series_equal(x, y)
        except AssertionError as e: diffs.append(f"{where}: {str(e).splitlines()[0][:120]}")
    elif isinstance(x, dict):
        if set(x) != set(y): diffs.append(f"{where}: keys differ"); return
        for k in x: cmp(x[k], y[k], f"{where}[{k!r}]", depth + 1)
    elif isinstance(x, (list, tuple)):
        if len(x) != len(y): diffs.append(f"{where}: length {len(x)} vs {len(y)}"); return
        for i, (u, v) in enumerate(zip(x, y)): cmp(u, v, f"{where}[{i}]", depth + 1)
    elif hasattr(x, "__dict__") and not callable(x):
        for k in vars(x):
            if any(s in k.lower() for s in SKIP): continue
            cmp(getattr(x, k), getattr(y, k, None), f"{where}.{k}", depth + 1)
    elif isinstance(x, (int, float, str, bool, np.generic)) or x is None:
        n_cmp += 1
        same = (x == y) or (isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y))
        if not same: diffs.append(f"{where}: {x!r} vs {y!r}")

A, B = load(args.a), load(args.b)
for k in sorted(vars(A)):
    if any(s in k.lower() for s in SKIP): continue
    if args.only_fitted and not any(k.startswith(f) or f in k for f in FITTED): continue
    if callable(getattr(A, k)): continue
    cmp(getattr(A, k), getattr(B, k, None), k)
print(f"compared {n_cmp:,} leaf objects; differences: {len(diffs)}")
for d in diffs[:40]: print("  DIFF", d)
sys.exit(1 if diffs else 0)
