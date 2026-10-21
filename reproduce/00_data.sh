#!/usr/bin/env bash
# Raw workbooks (data/) → prepared inputs (prepared_data/). Needs the private source workbooks in <workspace>/data/ and the
# public Census 2020 ZCTA gazetteer in <workspace>/reference_data/. Every builder refuses to overwrite a completed output and
# writes a manifest (source and code SHA-256, checks). ~1 h.
source "$(dirname "$0")/_common.sh"
step "1. cleaning: run notebooks/01_data_cleaning.ipynb top to bottom with the workspace as working directory"
echo "   (jupyter nbconvert --to notebook --execute --ExecutePreprocessor.cwd=\"\$REFERRAL_GNN_WORKSPACE\" $REPO/notebooks/01_data_cleaning.ipynb)"
echo "   → prepared_data/referral_prepared_<timestamp>/ (16 tables + manifest). Alternatively steps 1–2 in one go, into a sandbox:"
echo "     python $S/data/snapshot_build.py --name REF --train-end 2018-01-01 --holdout-end 2019-01-01   (regression check (a): identical arrays)"
step "2. model inputs"
py "$S/data/build_inputs.py"                     # four-node inputs (Steps 3–6)
py "$S/data/build_mips_yardstick.py"             # fixed per-specialist MIPS yardstick (STEP10 §2.3)
py "$S/data/extract_disease_mapping.py"          # disease-code → category map from the source's own category column
py "$S/data/build_three_node_inputs.py"          # v1: three-node graph, 33 disease categories
py "$S/data/build_three_node_v2_inputs.py"       # v2: + patient-origin distance
py "$S/data/build_three_node_v3_inputs.py"       # v3: + insurer history, shared hospital affiliation
py "$S/data/build_three_node_v4_inputs.py"       # v4: + severity, patient race
py "$S/data/build_three_node_v5_inputs.py"       # v5: as-of MIPS (final model inputs)
py "$S/data/build_three_node_v6_inputs.py"       # v6: + recent load (Step 10.7 option B only)
py "$S/data/extract_provider_ruca.py"            # provider rural/urban codes (Step 11.2 splits)
step "3. checks"
py "$S/checks/three_node_smoke.py" prepared_data/three_node_v5/model_inputs_PRIVATE.pkl     # 22 model-level checks
