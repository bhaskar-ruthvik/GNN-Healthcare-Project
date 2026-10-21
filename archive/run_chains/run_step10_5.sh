#!/bin/zsh
# STEP10_PROTOCOL A7: 10.5 evaluation pass (validation -> descriptive decision entry -> test). Run from Implementation/.
set -u
PY=env/bin/python; OUT=runs/analysis/step10_5
RUNS=($(ls -d runs/*_v5_unmasked_s42 runs/*_v5_unmasked_s1 runs/*_v5_unmasked_s42_dup runs/*_v5_masked_s42 runs/*_v4_unmasked_s42 runs/*_v4_unmasked_s1 runs/*_v4_masked_s42 runs/*_v4_masked_s1))
$PY scripts/step10_evaluate.py --split validation --step 10.5 --tag-version --out $OUT $RUNS 2>&1 | grep -E "primary cohort|scored|Traceback|Error" || exit 1
$PY scripts/step10_decide.py --step 10.5 --summary $OUT/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 \
    --amendment "STEP10_PROTOCOL A7 (10.5 descriptive)" --note "Bias analysis pass; no selection; final configuration unchanged (A5)." > /dev/null || exit 1
$PY scripts/step10_evaluate.py --split test --step 10.5 --tag-version --out $OUT $RUNS 2>&1 | grep -E "primary cohort|REFUSED|Traceback|Error"
echo "=== STEP10.5 EVAL DONE $(date '+%H:%M:%S')"
