"""Build and check three-node-v2 inputs (v1 + patient-origin distance). Refuses to overwrite a completed v2.
Aggregate diagnostics only."""
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
from referral_gnn.three_node_v2_data import ThreeNodeInputsV2, load_zcta_points, V2_PAIR_NAMES  # noqa: E402

root = str(workspace())
out_dir = os.path.join(root, "prepared_data", "three_node_v2")
if os.path.exists(os.path.join(out_dir, "COMPLETE.json")):
    sys.exit("three_node_v2 already complete; build into a new version directory instead")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

v1_path = os.path.join(root, "prepared_data", "three_node_v1", "model_inputs_PRIVATE.pkl")
v1 = load_three_node_inputs(v1_path)
pz = pd.read_pickle(os.path.join(out_dir, "patient_zip_PRIVATE.pkl"))
gaz_path = os.path.join(root, "reference_data", "2020_Gaz_zcta_national.zip")
zcta = load_zcta_points(gaz_path)
b = Bundle(default_bundle_path(root))
mi = ThreeNodeInputsV2(v1, pz, zcta, b.specialist_locations)

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond: fails.append(name)

check("one ZIP per patient in mapping; all bundle patients present", pz.mso_membership_id.is_unique and set(pz.mso_membership_id) >= set(mi.patient_ids))
mapped = ~np.isnan(mi.patient_lat)
check("patient coordinates mapped for ~97% of patients", 0.95 < mapped.mean() <= 1.0, f"mapped {mapped.mean():.1%}, unmapped ZIPs {mi.patient_zip_unmapped.sum()}, missing ZIP {mi.patient_zip_missing.sum()}")
# pair feature shape and content
tc_enc, tc_spec = mi.tc_enc[:210000], mi.tc_spec[:210000]
pf = mi.pair_features(tc_enc, tc_spec)
check("pair features have 10 columns and the first 8 equal v1", pf.shape[1] == 10 and np.allclose(pf[:, :8], v1.pair_features(tc_enc, tc_spec)))
d = mi.patient_specialist_distance(mi.enc_patient[tc_enc], tc_spec)
check("distances finite non-negative or NaN; NaN <-> missing flag", ((d[np.isfinite(d)] >= 0).all()) and (np.isnan(d) == (pf[:, 9] == 1)).all())
lab = mi.tc_label[:210000] == 1
print(f"   training pairs: patient-distance coverage targets {np.isfinite(d[lab]).mean():.1%} vs alternatives {np.isfinite(d[~lab]).mean():.1%} (should be symmetric)")
print(f"   median patient-specialist miles: targets {np.nanmedian(d[lab]):.1f} vs sampled alternatives {np.nanmedian(d[~lab]):.1f}")
check("targets are closer to the patient than random alternatives (sanity)", np.nanmedian(d[lab]) < np.nanmedian(d[~lab]))
# radius coverage on validation first-observed / all
val = mi.query_enc["validation"]; dv = mi.patient_specialist_distance(mi.enc_patient[val], mi.enc_target[val])
rep = mi.ps_asof.lookup(mi.enc_patient[val] * mi.n_specialist + mi.enc_target[val], mi.enc_day[val]) >= 0
for r in (25, 50, 75):
    print(f"   validation targets within {r} mi of the patient: all {np.mean(dv <= r):.1%} | first-observed {np.mean(dv[~rep] <= r):.1%} | repeat {np.mean(dv[rep] <= r):.1%} (NaN counts as outside)")
check("scaler fit on training only", abs(mi.patient_distance_scaler.mean - np.nanmean(mi.patient_specialist_distance(mi.enc_patient[mi.query_enc['train']], mi.enc_target[mi.query_enc['train']]))) < 1e-6)
check("dims report pair=10 and unchanged encounter width", mi.dims()["pair"] == 10 and mi.dims()["encounter_num"] == v1.dims()["encounter_num"])
# graph batch unchanged
bt = mi.graph_batch(int(mi.split_days["validation"][0]), np.where(mi.enc_day == mi.split_days["validation"][0])[0])
check("graph batch identical in structure to v1 (patient distance is a pair feature only)", set(bt["edge_index"]) == set(v1.graph_batch(int(mi.split_days["validation"][0]), np.where(mi.enc_day == mi.split_days["validation"][0])[0])["edge_index"]))

if fails:
    sys.exit(f"NOT SAVED: {fails}")
out = os.path.join(out_dir, "model_inputs_PRIVATE.pkl"); mi.save(out)
man = {"schema_version": mi.schema_version, "base_inputs": v1_path, "base_inputs_sha256": sha(v1_path), "patient_zip_manifest": json.load(open(os.path.join(out_dir, "patient_zip_manifest.json"))),
       "zcta_reference_sha256": sha(gaz_path), "inputs_sha256": sha(out), "dims": mi.dims(), "pair_feature_names": V2_PAIR_NAMES,
       "patient_coverage": {"mapped": float(mapped.mean()), "unmapped_zip": int(mi.patient_zip_unmapped.sum()), "missing_zip": int(mi.patient_zip_missing.sum())},
       "patient_distance_scaler": {"mean": mi.patient_distance_scaler.mean, "std": mi.patient_distance_scaler.std}, "checks_passed": True,
       "code_sha256": {f: sha(str(code_file(p))) for f, p in (("three_node_v2_data.py", "referral_gnn/three_node_v2_data.py"), ("build_three_node_v2_inputs.py", "scripts/data/build_three_node_v2_inputs.py"))}}
json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
json.dump({"inputs_sha256": man["inputs_sha256"]}, open(os.path.join(out_dir, "COMPLETE.json"), "w"))
print("saved", os.path.relpath(out, root))
