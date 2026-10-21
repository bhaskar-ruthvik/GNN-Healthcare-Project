#!/bin/zsh
# STEP12a part 3 (frozen-plan cells): waits for the part-2 neural chain to exit, then trains the neural volume cells (B2 shares,
# H5 log1p; 3 seeds each) and the LightGBM B2 cell at the reference configuration (127 leaves, min leaf 50, 1000 rounds; seeds 42, 1).
# From Implementation/. Usage: zsh scripts/run_step12a_part3.sh <PID of the part-2 neural chain>
WAIT=${1:?pid}
while kill -0 $WAIT 2>/dev/null; do sleep 30; done
echo "=== part 2 neural chain exited; LightGBM B2 $(date '+%H:%M:%S')"
env/bin/python -W ignore scripts/step12_lgbm.py --objective lambdarank --volume shares --mode fixed --config 127:50:1000 --seeds 42 1 --threads 6 2>&1 | grep -v -i warn | grep -E "wrote|done|Error|Trace"
echo "=== neural volume cells $(date '+%H:%M:%S')"
zsh scripts/run_runlist.sh grace/runlists/step12a_volume.tsv
echo "=== PART3 DONE $(date '+%H:%M:%S')"
