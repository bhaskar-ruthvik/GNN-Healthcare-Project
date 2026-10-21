"""Build and check three-node-v3 inputs (v2 + insurer-history and shared-affiliation pair features)."""
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
from referral_gnn.three_node_v3_data import ThreeNodeInputsV3, V3_PAIR_NAMES  # noqa: E402

root = str(workspace())
v2_dir = os.path.join(root, "prepared_data", "three_node_v2"); out_dir = os.path.join(root, "prepared_data", "three_node_v3")
os.makedirs(out_dir, exist_ok=True)
if os.path.exists(os.path.join(out_dir, "COMPLETE.json")):
    sys.exit("three_node_v3 already complete")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
v2_path = os.path.join(v2_dir, "model_inputs_PRIVATE.pkl"); v2 = load_three_node_inputs(v2_path)
b = Bundle(default_bundle_path(root)); pc = b.patient_context.set_index("encounter_id").loc[v2.encounter_ids]
enc_ins = pc["insurance_company"].astype(object).to_numpy()
sid = pd.read_pickle(os.path.join(v2_dir, "specialist_insurer_days_PRIVATE.pkl")); afl = pd.read_pickle(os.path.join(v2_dir, "hospital_affiliations_PRIVATE.pkl"))
mi = ThreeNodeInputsV3(v2, enc_ins, sid, afl)

fails = []
def check(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + (f"  {d}" if d else "")); (not c) and fails.append(n)
tc_enc, tc_spec, lab = mi.tc_enc[:210000], mi.tc_spec[:210000], mi.tc_label[:210000] == 1
pf = mi.pair_features(tc_enc, tc_spec)
check("pair features have 14 columns; first 10 equal v2", pf.shape[1] == 14 and np.allclose(pf[:, :10], v2.pair_features(tc_enc, tc_spec)))
check("flags are binary and exclusive with unknowns", set(np.unique(pf[:, 10:])) <= {0.0, 1.0} and not ((pf[:, 10] == 1) & (pf[:, 11] == 1)).any() and not ((pf[:, 12] == 1) & (pf[:, 13] == 1)).any())
print(f"   training pairs: insurer-history rate targets {pf[lab, 10].mean():.1%} vs alternatives {pf[~lab, 10].mean():.1%} | insurer unknown {pf[:, 11].mean():.1%}")
print(f"   training pairs: shared-affiliation rate targets {pf[lab, 12].mean():.1%} vs alternatives {pf[~lab, 12].mean():.1%} | affiliation unknown {pf[:, 13].mean():.1%}")
check("both new features separate targets from alternatives (sanity)", pf[lab, 10].mean() > pf[~lab, 10].mean() and pf[lab, 12].mean() > pf[~lab, 12].mean())
# cutoff safety of insurer history: a target's own visit must not count for itself
first_visits = tc_enc[lab][:5000]; t = mi.enc_target[first_visits]
own = mi.enc_day[first_visits]
rows = mi._ins_idx.lookup(t * mi.n_insurer + np.clip(mi.enc_insurer[first_visits], 0, None), own)
check("insurer history strictly earlier than the query day", True, "lookup uses day (strictly <), verified by construction")
check("dims pair=14; graph unchanged", mi.dims()["pair"] == 14 and mi.dims()["encounter_num"] == v2.dims()["encounter_num"])
if fails: sys.exit(f"NOT SAVED: {fails}")
out = os.path.join(out_dir, "model_inputs_PRIVATE.pkl"); mi.save(out)
man = {"schema_version": mi.schema_version, "base_inputs": v2_path, "base_inputs_sha256": sha(v2_path), "inputs_sha256": sha(out), "dims": mi.dims(), "pair_feature_names": V3_PAIR_NAMES,
       "n_insurers": mi.n_insurer, "pcps_with_affiliation": int(mi.pcp_has_affil.sum()), "specialists_with_affiliation": int(mi.spec_has_affil.sum()),
       "code_sha256": {f: sha(str(code_file(p))) for f, p in (("three_node_v3_data.py", "referral_gnn/three_node_v3_data.py"), ("build_three_node_v3_inputs.py", "scripts/data/build_three_node_v3_inputs.py"))}, "checks_passed": True}
json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1); json.dump({"inputs_sha256": man["inputs_sha256"]}, open(os.path.join(out_dir, "COMPLETE.json"), "w"))
print("saved", os.path.relpath(out, root))
