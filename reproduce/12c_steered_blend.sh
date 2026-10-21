#!/usr/bin/env bash
# STEP12c: complementarity, steering frontiers, the steered equal blend, its 2019 check and the group / load checks.
source "$(dirname "$0")/_common.sh"
step "table seed 7 (reference configuration) for the third seed pairing"
py "$S/train/step12_lgbm.py" --objective lambdarank --mode fixed --config 127:50:1000 --seeds 7
R=(v5_unmasked_s42 v5_unmasked_s1 v5_unmasked_s42_dup lgbm_tabular_s42 lgbm_tabular_s1 s12_lgbm_tabular_lambdarank_L127_M50_R1000_s7)
step "candidate-score dump (2018) and the analyses on it"
EVAL --split validation --step step12c_dump --strata --lean --dump-scores --out runs/analysis/step12c/dump "${R[@]}"
py "$S/analysis/step12/step12c_complementarity.py"
py "$S/analysis/step12/step12c_steering_control.py"
py "$S/analysis/step12/step12c_frontiers.py"
py "$S/analysis/step12/step12c_point_lenses.py" --dump runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz --out runs/analysis/step12c/point_lenses_validation.json
step "2019, once: decision (setting pre-registered in runs/analysis/step12c/preregistration_2019.json) → test dump → lenses"
DECIDE --step step12c_dump --summary runs/analysis/step12c/dump/validation_summary.json --final model_unmasked:blend0.3 --amendment "STEP12c held-out check of the steered equal blend found on 2018"
EVAL --split test --step step12c_dump --strata --lean --dump-scores --out runs/analysis/step12c/dump "${R[@]}"
py "$S/analysis/step12/step12c_point_lenses.py" --dump runs/analysis/step12c/dump/test_candidate_scores_PRIVATE.npz --out runs/analysis/step12c/point_lenses_test2019.json
step "checks on the steered blend (needs reproduce/12_groups_topk.sh for the K = 5 per-referral attributes)"
py "$S/analysis/step12/step12c_blend_checks.py"
