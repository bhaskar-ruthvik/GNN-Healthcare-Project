#!/bin/zsh
# STEP12e (STEP12E_TOPK_PLAN.md), 2018 stage: for K = 10 and 15, one validation evaluation with candidate dump (eligible sets filled
# to K) of GNN ×3, table ×3, dual ×3, offset ×3, then the fixed-setting lenses and the frontier. From Implementation/.
# The 2019 pass is run separately, after the decision entry is recorded.
set -e
RUNS=(runs/20260929T213930_v5_unmasked_s42 runs/20260930T094407_v5_unmasked_s1 runs/20260930T140429_v5_unmasked_s42_dup
      runs/20261007T195708_lgbm_tabular_s42 runs/20261007T195853_lgbm_tabular_s1 runs/20261014T174500_s12_lgbm_tabular_lambdarank_L127_M50_R1000_s7
      $(ls -d runs/*_s12d_dual_s* runs/*_s12d_offset_s*))
for r in $RUNS; do [[ -f $r/summary.json ]] || { echo "MISSING $r"; exit 1; }; done
for K in 10 15; do
  OUT=runs/analysis/step12e/K$K
  echo "=== K=$K evaluation of ${#RUNS} runs $(date '+%H:%M:%S')"
  env/bin/python -W ignore scripts/step10_evaluate.py --split validation --step step12e_topk_K$K --list-size $K --strata --lean --dump-scores --out $OUT $RUNS > runs/step12e_K${K}_eval.log 2>&1
  echo "=== K=$K lenses + frontier $(date '+%H:%M:%S')"
  env/bin/python scripts/step12e_topk.py --dump $OUT/validation_candidate_scores_PRIVATE.npz --k $K --out $OUT/step12e_K${K}_2018.json 2>&1 | grep -v -i warn
done
echo "=== STEP12E 2018 DONE $(date '+%H:%M:%S')"
