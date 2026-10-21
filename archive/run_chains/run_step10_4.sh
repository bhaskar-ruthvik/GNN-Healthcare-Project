#!/bin/zsh
# STEP10_PROTOCOL A6: 10.4 descriptive settings. Each setting: validation pass -> decision entry ("descriptive, no selection;
# final configuration unchanged") -> one test pass, in runs/analysis/step10_4/<setting>/. Run from Implementation/.
# Resumable (settings with test_summary.json are skipped) and stoppable between settings: on battery (unless ALLOW_BATTERY=1), or `touch runs/STOP_AFTER_CURRENT`.
set -u
PY=env/bin/python
RUNS=($(ls -d runs/*_v5_unmasked_s42 runs/*_v5_unmasked_s1))
STOP=runs/STOP_AFTER_CURRENT
typeset -A FLAGS
FLAGS=(cap10 "--cap 10" cap50 "--cap 50" capnone "--cap none" ever_active "--active ever" insurer_set "--insurer-set"
       non_choice "--specialties non_choice" non_specific "--specialties non_specific"
       reliance "--ablate no_mips,no_pcp_history,no_affiliation,no_insurer,no_distance")
for S in reliance cap10 cap50 capnone ever_active insurer_set non_choice non_specific; do
  OUT=runs/analysis/step10_4/$S
  if [[ -f $OUT/test_summary.json ]]; then echo "=== $S already complete, skipped"; continue; fi
  if [[ -e $STOP ]]; then echo "=== stop requested; stopping before $S  $(date '+%H:%M:%S')"; rm -f $STOP; break; fi
  if [[ "${ALLOW_BATTERY:-0}" != "1" ]] && pmset -g batt | grep -q "Battery Power"; then echo "=== on battery; stopping before $S  $(date "+%H:%M:%S")"; break; fi
  echo "=== $S  $(date '+%H:%M:%S')"
  rm -rf $OUT                                      # an unfinished setting restarts from scratch (no decision recorded without test output)
  $PY scripts/step10_evaluate.py --split validation --step 10.4:$S --tag-version --out $OUT ${=FLAGS[$S]} $RUNS 2>&1 | grep -E "primary cohort|scored|Traceback|Error" || break
  $PY scripts/step10_decide.py --step 10.4:$S --summary $OUT/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 \
      --amendment "STEP10_PROTOCOL A6 (10.4 descriptive)" --note "Descriptive setting; no selection; final configuration unchanged (A5)." > /dev/null || break
  $PY scripts/step10_evaluate.py --split test --step 10.4:$S --tag-version --out $OUT ${=FLAGS[$S]} $RUNS 2>&1 | grep -E "primary cohort|REFUSED|Traceback|Error"
done
echo "=== STEP10.4 CHAIN EXITED $(date '+%H:%M:%S')"
