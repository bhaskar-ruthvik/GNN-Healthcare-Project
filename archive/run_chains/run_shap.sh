#!/bin/zsh
# Grouped Shapley attributions for the best runs; waits for the last Step 9 training run. Run from Implementation/.
set -u
PY=env/bin/python
M=$(ls -d runs/2*_v4_masked_s1 | head -1); while [ ! -f $M/summary.json ]; do sleep 30; done
echo "=== training finished $(date '+%H:%M:%S'); starting SHAP"
for r in v4_unmasked_s42 v3_A_full v4_masked_s42; do
  d=$(ls -d runs/2*_$r | head -1); extra=""; [ "$r" = "v4_unmasked_s42" ] && extra="--also-top1"
  if [ -f $d/shap_groups/summary.json ]; then echo "=== shap $r already done, skipping"; continue; fi
  echo "=== shap $r  $(date '+%H:%M:%S')"
  $PY scripts/shap_groups.py $d --n-firstobs 20 --n-repeat 20 --perms 12 --seed 0 $extra 2>&1 | grep -v -E "FutureWarning|warnings.warn|Pandas4Warning|keyword-only" | grep -E "queries,|additivity|firstobs:|repeat:|^   |saved|Error|Traceback"
done
echo "=== SHAP DONE $(date '+%H:%M:%S')"
