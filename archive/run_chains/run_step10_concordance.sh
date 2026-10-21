#!/bin/zsh
# STEP10_PROTOCOL A13: race concordance pass (validation -> descriptive decision entry -> test), then the concordance analysis. Run from Implementation/.
set -u
PY=env/bin/python; OUT=runs/analysis/step10_9_concordance
RUNS=($(ls -d runs/*_v5_unmasked_s42 runs/*_v5_unmasked_s1))
$PY scripts/step10_evaluate.py --split validation --step 10.9 --tag-version --load-variants --provider-race --out $OUT $RUNS 2>&1 | grep -E "primary cohort|Traceback|Error" || exit 1
$PY scripts/step10_decide.py --step 10.9 --summary $OUT/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 \
    --amendment "STEP10_PROTOCOL A13 (descriptive)" --note "Race concordance; no selection." > /dev/null || exit 1
$PY scripts/step10_evaluate.py --split test --step 10.9 --tag-version --load-variants --provider-race --out $OUT $RUNS 2>&1 | grep -E "primary cohort|REFUSED|Traceback|Error"
for S in validation test; do echo "--- $S"; $PY scripts/step10_concordance.py --split $S; done
echo "=== A13 DONE $(date '+%H:%M:%S')"
