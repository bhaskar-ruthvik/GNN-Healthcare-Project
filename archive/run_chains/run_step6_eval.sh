#!/bin/zsh
# Post-queue evaluation for step 6 (protocol: STEP6_PROTOCOL.md). Run from Implementation/ after run_step6.sh finishes.
#  1. every run: validation subset with best.pt and best_firstobs.pt
#  2. validation analysis (tables, bootstrap, seed spread)
#  3. test split, once: baselines, main_s42 (best + best_firstobs), pairmlp (best)
#  4. test analysis
set -u
PY=env/bin/python
F="grep -v -E FutureWarning|warnings.warn|eval\ (validation|test):"
echo "=== validation evals $(date '+%H:%M:%S')"
for run in runs/2*_*; do
  case $run in *pilot*) continue;; esac
  [ -f $run/best.pt ] || continue
  $PY scripts/eval_checkpoint.py $run 2>&1 | grep -E "MRR" | head -1
  $PY scripts/eval_checkpoint.py $run --checkpoint best_firstobs.pt 2>&1 | grep -E "MRR" | head -1
done
echo "=== validation analysis $(date '+%H:%M:%S')"
$PY scripts/analyze.py --split validation 2>&1 | grep -v -E "FutureWarning|warnings.warn"
echo "=== TEST evaluation (once) $(date '+%H:%M:%S')"
$PY scripts/baselines.py --split test 2>&1 | grep -v -E "FutureWarning|warnings.warn"
MAIN=$(ls -d runs/2*_main_s42 | head -1); PM=$(ls -d runs/2*_pairmlp_e30 2>/dev/null | head -1); [ -z "$PM" ] && PM=$(ls -d runs/2*_pairmlp | head -1)
$PY scripts/eval_checkpoint.py $MAIN --split test 2>&1 | grep -E "MRR" | head -1
$PY scripts/eval_checkpoint.py $MAIN --split test --checkpoint best_firstobs.pt 2>&1 | grep -E "MRR" | head -1
$PY scripts/eval_checkpoint.py $PM --split test 2>&1 | grep -E "MRR" | head -1
echo "=== test analysis $(date '+%H:%M:%S')"
$PY scripts/analyze.py --split test 2>&1 | grep -v -E "FutureWarning|warnings.warn"
echo "=== EVAL DONE $(date '+%H:%M:%S')"
