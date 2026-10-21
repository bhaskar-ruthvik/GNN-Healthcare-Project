"""Leakage check (b), pre-cutoff rows (STEP11 §11.4(c); reviewer round 5). Compares a snapshot built normally (A) with the same
snapshot built with post-cutoff source rows shuffled (B): every array row dated before the training cutoff, the training
rows and selection sample, and the fitted objects must be identical. Prints identity flags and shapes only.
    python scripts/checks/snapshot_leakcheck.py snapshots/S16 snapshots/S16P2 --cutoff 2016-10-01 --version v5
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import os
import subprocess
import sys

import numpy as np


from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.data import to_day  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--cutoff", required=True)
ap.add_argument("--version", default="v5"); args = ap.parse_args()
p = lambda ws: os.path.join(ws, "prepared_data", f"three_node_{args.version}", "model_inputs_PRIVATE.pkl")
A, B = load_three_node_inputs(p(args.a)), load_three_node_inputs(p(args.b))
cut = int(np.asarray(to_day(np.array([args.cutoff])))[0]); res = []

def same(name, x, y):
    x, y = np.asarray(x), np.asarray(y)
    ok = x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y))
    res.append((name, ok, x.shape)); return ok

# selection and training rows
for k in ("validation_sample", "train_query_enc", "tc_enc", "tc_spec", "tc_label"): same(k, getattr(A, k), getattr(B, k))
same("pair features of all training rows", A.pair_features(A.tc_enc, A.tc_spec), B.pair_features(B.tc_enc, B.tc_spec))
# ids and availability
for k in ("specialist_ids", "pcp_ids", "patient_ids", "spec_first_seen"): same(k, getattr(A, k), getattr(B, k))
# encounter arrays, pre-cutoff rows
ea, eb = np.flatnonzero(A.enc_day < cut), np.flatnonzero(B.enc_day < cut); same("pre-cutoff encounter index", ea, eb)
for k in ("enc_day", "enc_patient", "enc_pcp", "enc_target", "enc_patient_num", "enc_patient_cat"):
    same(f"{k} (pre-cutoff rows)", getattr(A, k)[ea], getattr(B, k)[eb])
same("encounter static block (pre-cutoff rows; incl. history, severity)", A._encounter_static()[ea], B._encounter_static()[eb])
for k in ("enc_severity", "enc_race"):
    if hasattr(A, k): same(f"{k} (pre-cutoff rows)", getattr(A, k)[ea], getattr(B, k)[eb])
w = min(A.enc_dx.shape[1], B.enc_dx.shape[1])
same("enc_dx content (pre-cutoff rows; common width)", np.where(A.enc_dx_mask[ea][:, :w], A.enc_dx[ea][:, :w], 0), np.where(B.enc_dx_mask[eb][:, :w], B.enc_dx[eb][:, :w], 0))
# provider histories and pair history, rows dated before the cutoff
for pre, num, cat in (("spec", "spec_hist_num", "spec_hist_cat"), ("pcp", "pcp_hist_num", "pcp_hist_cat")):
    da, db = getattr(A, f"{pre}_hist_day"), getattr(B, f"{pre}_hist_day"); ra, rb = np.flatnonzero(da < cut), np.flatnonzero(db < cut)
    same(f"{pre} history days (pre-cutoff)", da[ra], db[rb]); same(f"{num} (pre-cutoff rows; incl. MIPS columns)", getattr(A, num)[ra], getattr(B, num)[rb])
    same(f"{cat} (pre-cutoff rows)", getattr(A, cat)[ra], getattr(B, cat)[rb])
if hasattr(A, "spec_hist_mips_extra"):
    ra, rb = np.flatnonzero(A.spec_hist_day < cut), np.flatnonzero(B.spec_hist_day < cut)
    same("spec_hist_mips_extra (pre-cutoff rows)", A.spec_hist_mips_extra[ra], B.spec_hist_mips_extra[rb])
pa, pb = np.flatnonzero(A.pair_day < cut), np.flatnonzero(B.pair_day < cut)
for k in ("pair_ent", "pair_day", "pair_cum", "pair_first"): same(f"{k} (pre-cutoff)", getattr(A, k)[pa], getattr(B, k)[pb])
same("distance matrix", A.distance, B.distance)
# fitted objects specific to v4/v5
for k in ("severity_vocab", "race_vocab"):
    if hasattr(A, k): res.append((k, list(getattr(A, k).index) == list(getattr(B, k).index), (len(getattr(A, k).index),)))
if hasattr(A, "mips_imputed_raw"):
    act = A.spec_first_seen <= int(A.train_end_day); same("v5 MIPS imputation, training-active specialists", A.mips_imputed_raw[act], B.mips_imputed_raw[act])
    res.append(("v5 MIPS scaler", A.spec_encoder.scalers["mips_score"].__dict__ == B.spec_encoder.scalers["mips_score"].__dict__, ()))
bad = [r for r in res if not r[1]]
for n, ok, sh in res: print(f"  {'OK  ' if ok else 'DIFF'} {n} {tuple(sh)}")
print(f"{len(res)} comparisons; differences: {len(bad)}")
fit = subprocess.run([sys.executable, "-W", "ignore", __file__.replace("snapshot_leakcheck.py", "snapshot_compare.py"), "--only-fitted",
                      os.path.dirname(p(args.a)), os.path.dirname(p(args.b))], capture_output=True, text=True)
print("fitted objects (snapshot_compare --only-fitted):", fit.stdout.strip().splitlines()[-1] if fit.stdout.strip() else fit.stderr[-300:])
sys.exit(1 if bad or fit.returncode else 0)
