#!/bin/zsh
# Preliminary (time-boxed) step 6: wait for main_s42, then two capped comparators, then evaluation.
set -u
PY=env/bin/python
MAIN=$(ls -d runs/2*_main_s42 | head -1)
while [ ! -f $MAIN/summary.json ]; do sleep 20; done
echo "=== main_s42 finished $(date '+%H:%M:%S'): $(tr -d '\n ' < $MAIN/summary.json)"
run() { name=$1; shift; echo "=== $name  $(date '+%H:%M:%S')"; $PY scripts/train.py --name $name --graph-bucket week --epochs 10 --patience 3 "$@" 2>&1 | grep --line-buffered -v -E "FutureWarning|warnings.warn" | grep --line-buffered -E "epoch|early|summary|WARNING|Error|Traceback"; }
run pairmlp --no-graph
run abl_no_history_no_ps --ablation no_history_features --ablation no_patient_specialist_prior
echo "=== training done $(date '+%H:%M:%S'); starting evaluation"
./scripts/run_step6_eval.sh
echo "=== PRELIM DONE $(date '+%H:%M:%S')"
