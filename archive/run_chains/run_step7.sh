#!/bin/zsh
# Step 7 (frozen protocol): pilot both arms, then seed-42 main runs, then seed repeats. Run from Implementation/.
set -u
PY=env/bin/python
COMMON=(--select firstobs_mrr_eligible --lr-factor 0.5 --lr-patience 2 --patience 5 --graph-bucket week)
run() { name=$1; shift; echo "=== $name  $(date '+%H:%M:%S')"; $PY scripts/three_node_train.py --name $name "${COMMON[@]}" "$@" 2>&1 | grep --line-buffered -v -E "FutureWarning|warnings.warn" | grep --line-buffered -E "epoch|early|summary|WARNING|Error|Traceback|run dir"; }
run tn_pilot_A --no-temporal --epochs 3
run tn_pilot_B --temporal --epochs 3
echo "=== PILOT DONE $(date '+%H:%M:%S')"
run tn_A_s42 --no-temporal --epochs 100 --seed 42
run tn_B_s42 --temporal --epochs 100 --seed 42
echo "=== MAIN DONE $(date '+%H:%M:%S')"
run tn_A_s1 --no-temporal --epochs 100 --seed 1
run tn_B_s1 --temporal --epochs 100 --seed 1
run tn_A_s2 --no-temporal --epochs 100 --seed 2
run tn_B_s2 --temporal --epochs 100 --seed 2
echo "=== STEP7 QUEUE DONE $(date '+%H:%M:%S')"
