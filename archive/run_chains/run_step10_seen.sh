#!/bin/zsh
# STEP10_PROTOCOL A12: include previously seen specialists (descriptive). Run from Implementation/.
set -u
PY=env/bin/python; OUT=runs/analysis/step10_4/include_seen
RUNS=($(ls -d runs/*_v5_unmasked_s42 runs/*_v5_unmasked_s1))
$PY scripts/step10_evaluate.py --split validation --step 10.4:include_seen --tag-version --include-seen --out $OUT $RUNS 2>&1 | grep -E "primary cohort|Traceback|Error" || exit 1
$PY scripts/step10_decide.py --step 10.4:include_seen --summary $OUT/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 \
    --amendment "STEP10_PROTOCOL A12 (descriptive)" --note "Previously seen specialists eligible; no selection." > /dev/null || exit 1
$PY scripts/step10_evaluate.py --split test --step 10.4:include_seen --tag-version --include-seen --out $OUT $RUNS 2>&1 | grep -E "primary cohort|REFUSED|Traceback|Error"
echo "=== A12 DONE $(date '+%H:%M:%S')"
