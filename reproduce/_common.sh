# Sourced by every reproduce/*.sh. Runs are named without timestamps (e.g. v5_unmasked_s42); the scripts resolve them in runs/.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REFERRAL_GNN_WORKSPACE=${REFERRAL_GNN_WORKSPACE:-$REPO}
PY=${PYTHON:-python}; S=$REPO/scripts
cd "$REFERRAL_GNN_WORKSPACE"
py()    { "$PY" -W ignore "$@"; }
EVAL()  { py "$S/evaluate/step10_evaluate.py" "$@"; }                 # one evaluation pass (validation, or test after a decision)
DECIDE(){ py "$S/evaluate/step10_decide.py" "$@"; }                   # record the 2018 decision that unlocks one 2019 pass
TRAIN() { "$S/tools/train_runlist.sh" "$REPO/runlists/$1"; }          # train a run list (resumable)
step()  { echo; echo "######## $*  $(date '+%H:%M:%S')"; }
