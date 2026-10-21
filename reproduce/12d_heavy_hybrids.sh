#!/usr/bin/env bash
# STEP12d (STEP12D_HEAVY_PLAN.md): dual encoder, table-offset GNN, GNN → LightGBM stacking with out-of-fold GNN scores.
source "$(dirname "$0")/_common.sh"
step "table out-of-fold scores (LightGBM process), then dual encoders, fold GNNs and offset GNNs (≈ 13 h on Apple MPS)"
for p in "42 lgbm_tabular_s42" "1 lgbm_tabular_s1" "7 s12_lgbm_tabular_lambdarank_L127_M50_R1000_s7"; do set -- $p; py "$S/train/step12d_table_oof.py" --seed $1 --table-run $2; done
TRAIN step12d_local.tsv
step "GNN out-of-fold scores and the stacked LightGBM"
D="runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz runs/analysis/step12c/dump/test_candidate_scores_PRIVATE.npz"
for p in "42 v5_unmasked_s42" "1 v5_unmasked_s1" "7 v5_unmasked_s42_dup"; do set -- $p
  # shellcheck disable=SC2086
  py "$S/train/step12d_gnn_oof.py" --seed $1 --gnn-run $2 --fold-runs s12d_fold0_s$1 s12d_fold1_s$1 s12d_fold2_s$1 --dumps $D
  py "$S/train/step12_lgbm.py" --objective lambdarank --mode fixed --config 127:50:1000 --seeds $1 --extra-npz runs/analysis/step12d/gnn_oof_s$1.npz; done
step "evaluation with candidate dump, frontier comparison, descriptives (β, stacking gain share)"
EVAL --split validation --step step12d --strata --lean --dump-scores --out runs/analysis/step12d/dump v5_unmasked_s42 v5_unmasked_s1 v5_unmasked_s42_dup \
     lgbm_tabular_s42 lgbm_tabular_s1 s12_lgbm_tabular_lambdarank_L127_M50_R1000_s7 s12d_dual_s42 s12d_dual_s1 s12d_dual_s7 s12d_offset_s42 s12d_offset_s1 s12d_offset_s7 \
     s12_lgbm_tabular_lambdarank_stack_L127_M50_R1000_s42 s12_lgbm_tabular_lambdarank_stack_L127_M50_R1000_s1 s12_lgbm_tabular_lambdarank_stack_L127_M50_R1000_s7
py "$S/analysis/step12/step12d_frontier.py" runs/analysis/step12d/dump/validation_candidate_scores_PRIVATE.npz
py "$S/analysis/step12/step12d_descriptives.py" beta; py "$S/analysis/step12/step12d_descriptives.py" gain
