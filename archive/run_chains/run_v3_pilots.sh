#!/bin/zsh
# Wait for ws_pilot, then two 10-epoch v3 pilots (general training on all events): uniform negatives (vs v2_pilot_A) and hard negatives (vs v2_pilot_hardneg).
set -u
PY=env/bin/python
WS=$(ls -d runs/2*_ws_pilot | head -1); while [ ! -f $WS/summary.json ]; do sleep 20; done
echo "=== ws_pilot finished $(date '+%H:%M:%S')"
run() { name=$1; shift; echo "=== $name  $(date '+%H:%M:%S')"; $PY scripts/three_node_train.py --name $name --inputs prepared_data/three_node_v3/model_inputs_PRIVATE.pkl --no-temporal --epochs 10 --select firstobs_mrr_eligible --lr-factor 0.5 --lr-patience 2 --patience 5 --graph-bucket week "$@" 2>&1 | grep --line-buffered -v -E "FutureWarning|warnings.warn" | grep --line-buffered -E "epoch|early|summary|WARNING|Error|Traceback"; }
run v3_pilot_A --negatives uniform
run v3_pilot_hardneg --negatives hard --neg-k 8,8,4
echo "=== V3 PILOTS DONE $(date '+%H:%M:%S')"
