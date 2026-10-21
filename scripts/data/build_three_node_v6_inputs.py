"""Build and check three-node-v6 inputs (STEP10_PROTOCOL A10, Option B variant 1). Aggregate diagnostics only.
v5 + two specialist columns of recent referral load (log1p of events in the previous 365 / 90 days, strictly before the cutoff).
Writes prepared_data/three_node_v6/ (refuses if it exists).
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import hashlib, json, os, sys
import numpy as np

from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.three_node_v6_data import ThreeNodeInputsV6, N_LOAD  # noqa: E402

root = str(workspace()); out_dir = os.path.join(root, "prepared_data", "three_node_v6")
if os.path.exists(out_dir): sys.exit(f"refusing to overwrite {out_dir}")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
v5_path = os.path.join(root, "prepared_data", "three_node_v5", "model_inputs_PRIVATE.pkl"); v5 = load_three_node_inputs(v5_path)
mi = ThreeNodeInputsV6(v5)
fails = []
def check(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + (f"  {d}" if d else "")); (not c) and fails.append(n)
va = int(mi.split_days["validation"][0]); bt = mi.graph_batch(va, np.where(mi.enc_day == va)[0]); bt5 = v5.graph_batch(va, np.where(v5.enc_day == va)[0])
x, x5 = bt["x"]["specialist_num"].numpy(), bt5["x"]["specialist_num"].numpy()
check("snapshot width = v5 + 2; dims consistent", x.shape[1] == x5.shape[1] + N_LOAD == mi.dims()["specialist_num"])
check("all v5 columns unchanged", np.allclose(x[:, :x5.shape[1]], x5))
check("load columns finite", np.isfinite(x[:, -N_LOAD:]).all())
# brute force: counts strictly before the cutoff on random specialists / days
rng = np.random.default_rng(0); bad = 0
for _ in range(2000):
    s = int(rng.integers(mi.n_specialist)); d = int(rng.integers(mi.split_days["train"][0] + 400, mi.split_days["test"][1]))
    ev = mi.enc_day[mi.enc_target == s]; want365 = int(((ev >= d - 365) & (ev < d)).sum()); want90 = int(((ev >= d - 90) & (ev < d)).sum())
    if mi.recent_load.count([s], d, 365)[0] != want365 or mi.recent_load.count([s], d, 90)[0] != want90: bad += 1
check("recent-load counts = brute force on 2,000 random (specialist, day) pairs; strictly before the day", bad == 0, f"mismatches {bad}")
check("no event on the cutoff day itself is counted", all(mi.recent_load.count([int(mi.enc_target[e])], int(mi.enc_day[e]), 1)[0] == ((mi.enc_target == mi.enc_target[e]) & (mi.enc_day == mi.enc_day[e] - 1)).sum() for e in rng.choice(mi.n_encounter, 300)))
check("encounter, PCP inputs, edges and pair features unchanged", np.allclose(bt["x"]["encounter_num"].numpy(), bt5["x"]["encounter_num"].numpy()) and set(bt["edge_index"]) == set(bt5["edge_index"])
      and np.allclose(mi.pair_features(mi.tc_enc[:2000], mi.tc_spec[:2000]), v5.pair_features(mi.tc_enc[:2000], mi.tc_spec[:2000])))
print(f"   load scaler (training cutoffs): mean {mi.load_mean.round(3).tolist()} sd {mi.load_std.round(3).tolist()}")
if fails: sys.exit(f"NOT SAVED: {fails}")
os.makedirs(out_dir); out = os.path.join(out_dir, "model_inputs_PRIVATE.pkl"); mi.save(out)
man = {"schema_version": mi.schema_version, "protocol": "STEP10_PROTOCOL A10 (Option B variant 1)", "base_inputs": "prepared_data/three_node_v5/model_inputs_PRIVATE.pkl",
       "base_inputs_sha256": sha(v5_path), "inputs_sha256": sha(out), "dims": mi.dims(), "appended_specialist_columns": ["log1p_load_365d", "log1p_load_90d"],
       "load_mean": mi.load_mean.tolist(), "load_std": mi.load_std.tolist(), "checks_passed": True,
       "code_sha256": {f: sha(str(code_file(p))) for f, p in (("three_node_v6_data.py", "referral_gnn/three_node_v6_data.py"), ("load.py", "referral_gnn/load.py"), ("build_three_node_v6_inputs.py", "scripts/data/build_three_node_v6_inputs.py"))}}
json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1); json.dump({"inputs_sha256": man["inputs_sha256"]}, open(os.path.join(out_dir, "COMPLETE.json"), "w"))
print("saved", os.path.relpath(out, root))
