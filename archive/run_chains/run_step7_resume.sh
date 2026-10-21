#!/bin/zsh
# Resume of the frozen Step 7 queue after the 2026-09-22 overnight stop: the three seed repeats not yet completed.
# Completed and kept: tn_pilot_A, tn_pilot_B, tn_A_s42, tn_B_s42, tn_A_s1. Run from Implementation/.
set -u
PY=env/bin/python
COMMON=(--select firstobs_mrr_eligible --lr-factor 0.5 --lr-patience 2 --patience 5 --graph-bucket week)
run() { name=$1; shift; echo "=== $name  $(date '+%H:%M:%S')"; $PY scripts/three_node_train.py --name $name "${COMMON[@]}" "$@" 2>&1 | grep --line-buffered -v -E "FutureWarning|warnings.warn" | grep --line-buffered -E "epoch|early|summary|WARNING|Error|Traceback|run dir"; }
run tn_B_s1 --temporal --epochs 100 --seed 1
run tn_A_s2 --no-temporal --epochs 100 --seed 2
run tn_B_s2 --temporal --epochs 100 --seed 2
echo "=== STEP7 QUEUE DONE $(date '+%H:%M:%S')"
