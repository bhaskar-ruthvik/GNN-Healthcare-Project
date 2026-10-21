#!/bin/zsh
# After v3_pilot_hardneg finishes: converged v3 run (uniform negatives, all events), max 100 epochs, patience 5.
set -u
PY=env/bin/python
P=$(ls -d runs/2*_v3_pilot_hardneg | head -1); while [ ! -f $P/summary.json ]; do sleep 20; done
echo "=== v3_pilot_hardneg finished $(date '+%H:%M:%S')"
echo "=== v3_A_full  $(date '+%H:%M:%S')"
$PY scripts/three_node_train.py --name v3_A_full --inputs prepared_data/three_node_v3/model_inputs_PRIVATE.pkl --no-temporal --negatives uniform --epochs 100 --select firstobs_mrr_eligible --lr-factor 0.5 --lr-patience 2 --patience 5 --graph-bucket week 2>&1 | grep --line-buffered -v -E "FutureWarning|warnings.warn" | grep --line-buffered -E "epoch|early|summary|WARNING|Error|Traceback"
echo "=== V3 FULL DONE $(date '+%H:%M:%S')"
