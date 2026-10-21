#!/bin/zsh
# STEP12d evaluation: one validation pass with candidate-score dump (GNN ×3, table ×3, dual ×3, offset ×3, stacked ×3), then the
# frontier comparison against the steered equal blend. From Implementation/.
set -e
RUNS=(runs/20260929T213930_v5_unmasked_s42 runs/20260930T094407_v5_unmasked_s1 runs/20260930T140429_v5_unmasked_s42_dup
      runs/20261007T195708_lgbm_tabular_s42 runs/20261007T195853_lgbm_tabular_s1 runs/20261014T174500_s12_lgbm_tabular_lambdarank_L127_M50_R1000_s7
      $(ls -d runs/*_s12d_dual_s* runs/*_s12d_offset_s* runs/*_stack_L127_M50_R1000_s*))
for r in $RUNS; do [[ -f $r/summary.json ]] || { echo "MISSING $r"; exit 1; }; done
echo "=== evaluation of ${#RUNS} runs $(date '+%H:%M:%S')"
env/bin/python -W ignore scripts/step10_evaluate.py --split validation --step step12d --strata --lean --dump-scores --out runs/analysis/step12d/dump $RUNS > runs/step12d_eval.log 2>&1
echo "=== frontier $(date '+%H:%M:%S')"
env/bin/python scripts/step12d_frontier.py runs/analysis/step12d/dump/validation_candidate_scores_PRIVATE.npz 2>&1 | grep -v -i warn
echo "=== STEP12D EVAL DONE $(date '+%H:%M:%S')"
