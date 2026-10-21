#!/bin/zsh
# Train any run list sequentially on the laptop (MPS); generalised from run_step10_3.sh.
# Same run list and flags as grace/train_array.sbatch; all four runs share one device. Run from Implementation/.
# Resumable at epoch level and stoppable:
#   - runs with a summary.json are skipped; an unfinished run with a resume_state.pt is continued (--resume) from its last epoch;
#   - stopping mid-run (kill) costs at most the epoch in progress;
#   - before each run the chain stops if the Mac is on battery (unless ALLOW_BATTERY=1), or if runs/STOP_AFTER_CURRENT exists
#     (touch it at any time: the current run finishes, then the chain exits; the file is removed on exit).
set -u
PY=env/bin/python
RUNLIST=${1:?usage: run_runlist.sh grace/runlists/<file>.tsv}
STOP=runs/STOP_AFTER_CURRENT
grep -v '^\s*#' $RUNLIST | grep -v '^\s*$' | while IFS=$'\t' read -r NAME INPUTS EXTRA; do
  done_files=(runs/*_"$NAME"/summary.json(N))
  if (( ${#done_files} )); then echo "=== $NAME already complete, skipped"; continue; fi
  if [[ -e $STOP ]]; then echo "=== stop requested ($STOP); stopping before $NAME  $(date '+%H:%M:%S')"; rm -f $STOP; break; fi
  if [[ -z ${ALLOW_BATTERY:-} ]] && pmset -g batt | grep -q "Battery Power"; then echo "=== on battery; stopping before $NAME  $(date '+%H:%M:%S')"; break; fi
  partial=(runs/*_"$NAME"/resume_state.pt(N))
  RESUME=(); (( ${#partial} )) && RESUME=(--resume ${partial[-1]:h})
  echo "=== $NAME ${RESUME:+(resuming ${partial[-1]:h})}  $(date '+%H:%M:%S')"
  $PY scripts/three_node_train.py --name $NAME $RESUME --inputs $INPUTS --no-temporal --negatives uniform --epochs 100 \
    --select firstobs_mrr_eligible --lr-factor 0.5 --lr-patience 2 --patience 5 --graph-bucket week ${=EXTRA} 2>&1 \
    | grep --line-buffered -v -E "FutureWarning|warnings.warn" | grep --line-buffered -E "run dir|resumed|epoch|early|summary|refused|WARNING|Error|Traceback"
done
echo "=== RUNLIST CHAIN EXITED $(date '+%H:%M:%S')"
