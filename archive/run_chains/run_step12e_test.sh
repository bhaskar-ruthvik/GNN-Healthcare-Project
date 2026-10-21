#!/bin/zsh
# STEP12e 2019 stage (after the decision entries step12e_topk_K10 / _K15 were recorded). One test pass per K, then the fixed-setting lenses.
set -e
RUNS=(runs/20260929T213930_v5_unmasked_s42 runs/20260930T094407_v5_unmasked_s1 runs/20260930T140429_v5_unmasked_s42_dup
      runs/20261007T195708_lgbm_tabular_s42 runs/20261007T195853_lgbm_tabular_s1 runs/20261014T174500_s12_lgbm_tabular_lambdarank_L127_M50_R1000_s7
      $(ls -d runs/*_s12d_dual_s* runs/*_s12d_offset_s*))
for K in 10 15; do
  OUT=runs/analysis/step12e/K$K
  echo "=== K=$K test pass $(date '+%H:%M:%S')"
  env/bin/python -W ignore scripts/step10_evaluate.py --split test --step step12e_topk_K$K --list-size $K --strata --lean --dump-scores --out $OUT $RUNS > runs/step12e_K${K}_test.log 2>&1
  env/bin/python scripts/step12e_topk.py --dump $OUT/test_candidate_scores_PRIVATE.npz --k $K --out $OUT/step12e_K${K}_2019.json --test 2>&1 | grep -v -i warn
done
echo "=== STEP12E 2019 DONE $(date '+%H:%M:%S')"
