import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import os
"""Build and check three-node-v5 inputs (STEP10_PROTOCOL §2.3 model side, §4 step 10.3). Aggregate diagnostics only.
Writes prepared_data/three_node_v5/ (refuses if it exists). Source: three_node_v4 inputs + mips_yardstick_v1 (read-only).
"""
import hashlib, json, os, sys
import numpy as np, pandas as pd

from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.three_node_v5_data import ThreeNodeInputsV5, N_EXTRA  # noqa: E402

root = str(workspace())
out_dir = os.path.join(root, "prepared_data", "three_node_v5")
if os.path.exists(out_dir): sys.exit(f"refusing to overwrite {out_dir}")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
v4_path = os.path.join(root, "prepared_data", "three_node_v4", "model_inputs_PRIVATE.pkl"); v4 = load_three_node_inputs(v4_path)
yd = os.path.join(root, "prepared_data", "mips_yardstick_v1")
mv = pd.read_pickle(os.path.join(yd, "mips_values_PRIVATE.pkl")); yt = pd.read_pickle(os.path.join(yd, "mips_yardstick_PRIVATE.pkl"))
mi = ThreeNodeInputsV5(v4, mv, yt)

fails = []
def check(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + (f"  {d}" if d else "")); (not c) and fails.append(n)
v, m, a = mi.mips_cols; x = mi.spec_hist_num; ex = mi.spec_hist_mips_extra
ent = mi.spec_hist_asof.entity[np.argsort(mi.spec_hist_asof.order)]
check("history rows: values finite; none rows carry the imputed value; flags binary",
      np.isfinite(x[:, v]).all() and np.allclose(x[x[:, m] == 1, v], mi.mips_imputed_scaled[ent[x[:, m] == 1]]) and set(np.unique(x[:, [m, a]])) <= {0.0, 1.0})
check("multi rows have range > 0; single rows range 0; none rows range 0",
      (ex[x[:, a] == 1, 1] > 0).all() and (ex[(x[:, a] == 0) & (x[:, m] == 0), 1] == 0).all() and (ex[x[:, m] == 1, 1] == 0).all())
# brute-force as-of check on a random sample of rows: only values first seen on/before the row day
rng = np.random.default_rng(0); idx = rng.choice(len(x), 3000, replace=False)
mv2 = mv.copy(); mv2["s"] = mv2.provider_id.astype(str).map(mi.specialist_index); mv2 = mv2.dropna(subset=["s"]); mv2["s"] = mv2.s.astype(int)
mv2["d"] = pd.to_datetime(mv2.first_day).values.astype("datetime64[D]").astype(np.int64)
grp = {s: g for s, g in mv2.groupby("s")}
bad = 0
for i in idx:
    g = grp.get(int(ent[i])); d = int(mi.spec_hist_day[i])
    pv = g[(g.d <= d) & (g.value > 0)].value.unique() if g is not None else np.array([])
    want_none = len(pv) == 0
    want_v = mi.mips_imputed_scaled[ent[i]] if want_none else mi.spec_encoder.scalers["mips_score"].transform([pv.mean()])[0][0]
    if (x[i, m] == 1) != want_none or abs(x[i, v] - want_v) > 1e-4 or (x[i, a] == 1) != (len(pv) > 1): bad += 1
check("as-of brute force on 3,000 random history rows (values first seen on/before the row day only)", bad == 0, f"mismatches {bad}")
# final state equals the yardstick for specialists whose last history row is after all their values appeared
last = pd.Series(np.arange(len(x))).groupby(ent).max()
ys = yt.set_index("provider_id").reindex(mi.specialist_ids)
raw_last = mi.spec_encoder.scalers["mips_score"].mean + x[last.to_numpy(), v] * mi.spec_encoder.scalers["mips_score"].std
yv = ys.mips_value.to_numpy()[last.index.to_numpy()]
okm = ~np.isnan(yv) & (x[last.to_numpy(), m] == 0)
agree = np.isclose(raw_last[okm], yv[okm], atol=1e-3).mean()
check("last history row agrees with the fixed yardstick for most specialists (differences = values appearing after the last row)", agree > 0.95, f"agree {agree:.3f}")
va = int(mi.split_days["validation"][0])
bt = mi.graph_batch(va, np.where(mi.enc_day == va)[0]); xs = bt["x"]["specialist_num"].numpy()
check("snapshot width = v4 + 2 and dims consistent", xs.shape[1] == v4.dims()["specialist_num"] + N_EXTRA == mi.dims()["specialist_num"])
print(f"   at validation start: none {xs[:, m].mean():.1%}, multi {xs[:, a].mean():.1%}, partial-zero {xs[:, -2].mean():.1%} of specialists")
bt4 = v4.graph_batch(va, np.where(v4.enc_day == va)[0])
check("encounter inputs, PCP inputs and graph edges unchanged from v4",
      np.allclose(bt["x"]["encounter_num"].numpy(), bt4["x"]["encounter_num"].numpy()) and np.allclose(bt["x"]["pcp_num"].numpy(), bt4["x"]["pcp_num"].numpy())
      and set(bt["edge_index"]) == set(bt4["edge_index"]))
non_mips = [c for c in range(v4.dims()["specialist_num"]) if c not in (v, m, a)]
check("non-MIPS specialist columns unchanged from v4", np.allclose(xs[:, non_mips], bt4["x"]["specialist_num"].numpy()[:, non_mips]))
check("pair features unchanged from v4", np.allclose(mi.pair_features(mi.tc_enc[:2000], mi.tc_spec[:2000]), v4.pair_features(mi.tc_enc[:2000], mi.tc_spec[:2000])))
if fails: sys.exit(f"NOT SAVED: {fails}")
os.makedirs(out_dir); out = os.path.join(out_dir, "model_inputs_PRIVATE.pkl"); mi.save(out)
man = {"schema_version": mi.schema_version, "protocol": "STEP10_PROTOCOL §2.3 (+A3)", "base_inputs": "prepared_data/three_node_v4/model_inputs_PRIVATE.pkl",
       "base_inputs_sha256": sha(v4_path), "yardstick_manifest": json.load(open(os.path.join(yd, "manifest.json"))), "inputs_sha256": sha(out), "dims": mi.dims(),
       "mips_rule": mi.mips_rule, "mips_cols": list(mi.mips_cols), "appended_specialist_columns": ["mips_partial_zero", "mips_range_div100"], "checks_passed": True,
       "code_sha256": {f: sha(str(code_file(p))) for f, p in (("three_node_v5_data.py", "referral_gnn/three_node_v5_data.py"), ("build_three_node_v5_inputs.py", "scripts/data/build_three_node_v5_inputs.py"))}}
json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1, default=str); json.dump({"inputs_sha256": man["inputs_sha256"]}, open(os.path.join(out_dir, "COMPLETE.json"), "w"))
print("saved", os.path.relpath(out, root))
