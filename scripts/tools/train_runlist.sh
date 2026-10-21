#!/usr/bin/env bash
# Train every run in a run list (runlists/*.tsv: name <TAB> inputs <TAB> extra flags) one after another with the final-model
# training settings (no PRECEDES edges, 20 uniform negatives, first-time MRR checkpoint selection, LR halving, early stopping).
#   - finished runs (runs/*_<name>/summary.json) are skipped; an unfinished run with resume_state.pt continues from its last epoch;
#   - `touch runs/STOP_AFTER_CURRENT` stops the chain after the current run; on macOS it also stops on battery unless ALLOW_BATTERY=1.
# Usage: scripts/tools/train_runlist.sh runlists/step10_3_v5.tsv        (PYTHON=..., REFERRAL_GNN_WORKSPACE=... optional)
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd); PY=${PYTHON:-python}
LIST=${1:?usage: train_runlist.sh runlists/<file>.tsv}; [[ -f $LIST ]] || LIST=$REPO/$LIST
cd "${REFERRAL_GNN_WORKSPACE:-$REPO}"
STOP=runs/STOP_AFTER_CURRENT
grep -v '^[[:space:]]*#' "$LIST" | grep -v '^[[:space:]]*$' | while IFS=$'\t' read -r NAME INPUTS EXTRA; do
  shopt -s nullglob; done_files=(runs/*_"$NAME"/summary.json); partial=(runs/*_"$NAME"/resume_state.pt); shopt -u nullglob
  if (( ${#done_files[@]} )); then echo "=== $NAME already complete, skipped"; continue; fi
  if [[ -e $STOP ]]; then echo "=== stop requested ($STOP); stopping before $NAME"; rm -f "$STOP"; break; fi
  if [[ -z ${ALLOW_BATTERY:-} ]] && command -v pmset >/dev/null && pmset -g batt | grep -q "Battery Power"; then echo "=== on battery; stopping before $NAME"; break; fi
  RESUME=(); if (( ${#partial[@]} )); then RESUME=(--resume "$(dirname "${partial[${#partial[@]}-1]}")"); fi
  echo "=== $NAME ${RESUME[*]+${RESUME[*]}}  $(date '+%H:%M:%S')"
  # shellcheck disable=SC2086
  "$PY" "$REPO/scripts/train/three_node_train.py" --name "$NAME" ${RESUME[@]+"${RESUME[@]}"} --inputs "$INPUTS" --no-temporal --negatives uniform \
    --epochs 100 --select firstobs_mrr_eligible --lr-factor 0.5 --lr-patience 2 --patience 5 --graph-bucket week $EXTRA
done
echo "=== run list finished $(date '+%H:%M:%S')"
