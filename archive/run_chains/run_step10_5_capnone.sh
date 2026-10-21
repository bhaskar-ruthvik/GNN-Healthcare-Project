#!/bin/zsh
# STEP10_PROTOCOL A8: no-cap geographic follow-up. Run from Implementation/.
set -u
PY=env/bin/python; OUT=runs/analysis/step10_5_capnone
RUNS=($(ls -d runs/*_v5_unmasked_s42 runs/*_v5_unmasked_s1))
$PY scripts/step10_evaluate.py --split validation --step 10.5:capnone --tag-version --cap none --out $OUT $RUNS 2>&1 | grep -E "primary cohort|Traceback|Error" || exit 1
$PY scripts/step10_decide.py --step 10.5:capnone --summary $OUT/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 \
    --amendment "STEP10_PROTOCOL A8 (descriptive)" --note "No-cap geographic follow-up; no selection." > /dev/null || exit 1
$PY scripts/step10_evaluate.py --split test --step 10.5:capnone --tag-version --cap none --out $OUT $RUNS 2>&1 | grep -E "primary cohort|REFUSED|Traceback|Error"
for S in validation test; do $PY scripts/step10_bias.py --split $S --dir $OUT 2>&1 | grep -v -i warn | tail -1; done
echo "=== A8 DONE $(date '+%H:%M:%S')"
