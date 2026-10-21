#!/bin/zsh
# STEP9_PROTOCOL §3: four runs, sequential. Run from Implementation/.
set -u
PY=env/bin/python
run() { name=$1; shift; echo "=== $name  $(date '+%H:%M:%S')"; $PY scripts/three_node_train.py --name $name --inputs prepared_data/three_node_v4/model_inputs_PRIVATE.pkl --no-temporal --negatives uniform --epochs 100 --select firstobs_mrr_eligible --lr-factor 0.5 --lr-patience 2 --patience 5 --graph-bucket week "$@" 2>&1 | grep --line-buffered -v -E "FutureWarning|warnings.warn" | grep --line-buffered -E "epoch|early|summary|WARNING|Error|Traceback"; }
run v4_unmasked_s42 --seed 42
run v4_masked_s42 --seed 42 --mask-demographics
run v4_unmasked_s1 --seed 1
run v4_masked_s1 --seed 1 --mask-demographics
echo "=== STEP9 RUNS DONE $(date '+%H:%M:%S')"
