#!/bin/zsh
# STEP10_PROTOCOL A10/A11: official 10.7 pass over Option A (final v5 + load variants) and Option B (v6_load, v5_posw05; two seeds each).
# validation -> decision entry (A10/A11 rule output = primary = final for 10.7) -> one test pass. Run from Implementation/.
set -u
PY=env/bin/python; OUT=runs/analysis/step10_7
for n in v6_load_s42 v6_load_s1 v5_posw05_s42 v5_posw05_s1; do
  ls runs/*_"$n"/summary.json >/dev/null 2>&1 || { echo "Option B run $n not finished; not starting the official pass"; exit 1; }
done
RUNS=($(ls -d runs/*_v5_unmasked_s42 runs/*_v5_unmasked_s1 runs/*_v6_load_s42 runs/*_v6_load_s1 runs/*_v5_posw05_s42 runs/*_v5_posw05_s1))
$PY scripts/step10_evaluate.py --split validation --step 10.7 --tag-version --load-variants --out $OUT $RUNS 2>&1 | grep -E "primary cohort|scored|Traceback|Error" || exit 1
CHOSEN=$($PY -c "import json; print(json.load(open('$OUT/validation_summary.json'))['a10']['chosen'] or '')")
[[ -n "$CHOSEN" ]] || { echo "no admissible configuration under A10/A11; stopping before test (report the frontier)"; exit 0; }
echo "A10/A11 chosen: $CHOSEN"
$PY scripts/step10_decide.py --step 10.7 --summary $OUT/validation_summary.json --final $CHOSEN \
    --amendment "STEP10_PROTOCOL A10 + A11" --note "10.7 load-aware choice by the A10 rule with the A11 fidelity budget." > /dev/null || exit 1
$PY scripts/step10_evaluate.py --split test --step 10.7 --tag-version --load-variants --out $OUT $RUNS 2>&1 | grep -E "primary cohort|REFUSED|Traceback|Error"
echo "=== STEP10.7 EVAL DONE $(date '+%H:%M:%S')"
