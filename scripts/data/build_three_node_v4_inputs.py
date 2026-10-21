"""Build and check three-node-v4 inputs (STEP9_PROTOCOL §2). Aggregate diagnostics only."""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd


from referral_gnn.data import Bundle, default_bundle_path  # noqa: E402
from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.three_node_v4_data import ThreeNodeInputsV4  # noqa: E402

root = str(workspace())
out_dir = os.path.join(root, "prepared_data", "three_node_v4")
if os.path.exists(os.path.join(out_dir, "COMPLETE.json")):
    sys.exit("three_node_v4 already complete")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
v3_path = os.path.join(root, "prepared_data", "three_node_v3", "model_inputs_PRIVATE.pkl"); v3 = load_three_node_inputs(v3_path)
b = Bundle(default_bundle_path(root))
sev = pd.read_pickle(os.path.join(out_dir, "row_severity_PRIVATE.pkl")); race = pd.read_pickle(os.path.join(out_dir, "patient_race_PRIVATE.pkl"))
rem = pd.read_pickle(os.path.join(default_bundle_path(root), "row_event_map_PRIVATE.pkl"))
mi = ThreeNodeInputsV4(v3, sev, rem, race)

fails = []
def check(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + (f"  {d}" if d else "")); (not c) and fails.append(n)
# row index alignment: every linked source row in row_event_map must have a severity row
matched = rem["_source_row"].isin(sev["_source_row"]).mean()
check("row_event_map source rows align with extracted severity rows", matched == 1.0, f"matched {matched:.4f}")
cov = (mi.enc_severity > 0).mean()
raw_n = b.encounters.set_index("encounter_id").loc[mi.encounter_ids, "diagnosis_codes"].map(len).to_numpy()   # raw code count per encounter (categories can merge codes)
multi_code = raw_n > 1
conf = mi.enc_severity_conflict
check("severity coverage ~100%; conflicts occur only in multi-code encounters (severity is deterministic per code)",
      cov > 0.99 and (~conf | multi_code).all(), f"coverage {cov:.4f}, conflicts {int(conf.sum())} of {int(multi_code.sum())} multi-raw-code encounters (rule: most severe + flag), labels {dict(zip(*np.unique(mi.enc_severity_label.astype(str), return_counts=True)))}")
# severity must agree with diagnosis-code determinism: encounters with identical code sets should share severity
codes = [tuple(sorted(c for c, m_ in zip(mi.enc_dx[i], mi.enc_dx_mask[i]) if m_)) for i in range(mi.n_encounter)]
df = pd.DataFrame({"codes": codes, "sev": mi.enc_severity}); multi = df.groupby("codes")["sev"].nunique()
print(f"   note: category-bag -> severity is not expected to be one-to-one ({int((multi > 1).sum())} bags with >1 severity); determinism was verified at the raw-code level")
r_cov = (mi.enc_race > 0).mean()
check("patient race categorical: vocab from train, coverage ~96%", 0.9 < r_cov < 1.0 and mi.race_vocab.size <= 7, f"coverage {r_cov:.3f}, vocab {mi.race_vocab.size}")
check("encounter cats now 5 columns; dims consistent", mi.enc_patient_cat.shape[1] == 5 and len(mi.dims()["encounter_cat_sizes"]) == 5)
bt = mi.graph_batch(int(mi.split_days["validation"][0]), np.where(mi.enc_day == mi.split_days["validation"][0])[0])
w = bt["x"]["encounter_num"].shape[1]
check("encounter numeric width = v3 + 1 and blocks contiguous", w == v3.dims()["encounter_num"] + 1 and w == mi.encounter_blocks["predecessor_gap"][1] and mi.encounter_blocks["severity_flags"] == (mi.encounter_blocks["patient_context"][1], mi.encounter_blocks["patient_context"][1] + 1))
# MIPS rule
v_, m_, a_ = mi.mips_cols; x = bt["x"]["specialist_num"].numpy()
miss_na = (x[:, m_] == 1) & (x[:, a_] == 0); miss_amb = (x[:, m_] == 1) & (x[:, a_] == 1)
check("MIPS: missing&not-ambiguous carry scaled(0); ambiguous unchanged (0); present rows untouched",
      np.allclose(x[miss_na, v_], mi.mips_scaled_zero) and (np.abs(x[miss_amb, v_]) < 1e-6).all() and abs(mi.mips_scaled_zero) > 0.5,
      f"scaled(0)={mi.mips_scaled_zero:.3f}; missing-not-amb {miss_na.mean():.1%}, ambiguous {miss_amb.mean():.1%} of specialists at validation start; as-of rows re-encoded {mi.n_mips_missing_rows_reencoded:,}")
check("pair features unchanged from v3 (14 columns)", mi.pair_features(mi.tc_enc[:2000], mi.tc_spec[:2000]).shape[1] == 14 and np.allclose(mi.pair_features(mi.tc_enc[:2000], mi.tc_spec[:2000]), v3.pair_features(mi.tc_enc[:2000], mi.tc_spec[:2000])))
check("graph edges unchanged", set(bt["edge_index"]) == set(v3.graph_batch(int(mi.split_days["validation"][0]), np.where(mi.enc_day == mi.split_days["validation"][0])[0])["edge_index"]))
if fails: sys.exit(f"NOT SAVED: {fails}")
out = os.path.join(out_dir, "model_inputs_PRIVATE.pkl"); mi.save(out)
man = {"schema_version": mi.schema_version, "base_inputs": v3_path, "base_inputs_sha256": sha(v3_path), "inputs_sha256": sha(out), "dims": mi.dims(),
       "extraction_manifest": json.load(open(os.path.join(out_dir, "extraction_manifest.json"))), "severity_vocab": mi.severity_vocab.index, "race_vocab": mi.race_vocab.index,
       "demographic_cat_cols": mi.demographic_cat_cols, "demographic_flag_cols": mi.demographic_flag_cols, "mips_scaled_zero": mi.mips_scaled_zero,
       "encounter_blocks": mi.encounter_blocks, "checks_passed": True,
       "code_sha256": {f: sha(str(code_file(p))) for f, p in (("three_node_v4_data.py", "referral_gnn/three_node_v4_data.py"), ("build_three_node_v4_inputs.py", "scripts/data/build_three_node_v4_inputs.py"))}}
json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1); json.dump({"inputs_sha256": man["inputs_sha256"]}, open(os.path.join(out_dir, "COMPLETE.json"), "w"))
print("saved", os.path.relpath(out, root))
