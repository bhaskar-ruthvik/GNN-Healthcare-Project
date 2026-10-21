#!/usr/bin/env bash
# STEP12e (STEP12E_TOPK_PLAN.md, frozen 2026-10-19): the steered blend and hybrids with 10 and 15 names. 2018, decision, 2019 once.
source "$(dirname "$0")/_common.sh"
R=(v5_unmasked_s42 v5_unmasked_s1 v5_unmasked_s42_dup lgbm_tabular_s42 lgbm_tabular_s1 s12_lgbm_tabular_lambdarank_L127_M50_R1000_s7
   s12d_dual_s42 s12d_dual_s1 s12d_dual_s7 s12d_offset_s42 s12d_offset_s1 s12d_offset_s7)
for K in 10 15; do O=runs/analysis/step12e/K$K; step "K = $K"
  EVAL --split validation --step step12e_topk_K$K --list-size $K --strata --lean --dump-scores --out $O "${R[@]}"
  py "$S/analysis/step12/step12e_topk.py" --dump $O/validation_candidate_scores_PRIVATE.npz --k $K --out $O/step12e_K${K}_2018.json
  DECIDE --step step12e_topk_K$K --summary $O/validation_summary.json --final model_unmasked:blend0.3 --amendment "STEP12E_TOPK_PLAN.md (frozen 2026-10-19)"
  EVAL --split test --step step12e_topk_K$K --list-size $K --strata --lean --dump-scores --out $O "${R[@]}"
  py "$S/analysis/step12/step12e_topk.py" --dump $O/test_candidate_scores_PRIVATE.npz --k $K --out $O/step12e_K${K}_2019.json --test
done
