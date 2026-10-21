#!/usr/bin/env bash
# STEP12b (STEP12B_HYBRID_PLAN.md): score blends, gate, rank fusion, split lists, GNN encoding + logistic regression, flattened graph.
source "$(dirname "$0")/_common.sh"
step "train: flattened-graph LightGBM; GNN encoding + variables → logistic regression (and variables only)"
py "$S/train/step12_lgbm.py" --feature-set flat --objective lambdarank --mode tune
for g in v5_unmasked_s42 v5_unmasked_s1; do py "$S/train/step12b_emb_lr.py" --gnn-run $g; py "$S/train/step12b_emb_lr.py" --gnn-run $g --no-emb; done
step "one evaluation pass with the score-level hybrids for both seed pairings, then the pre-stated judge"
G42=$(py -c "import sys; sys.path.insert(0,'$REPO'); from referral_gnn.paths import resolve_run as r; print(r('v5_unmasked_s42'))"); G1=$(py -c "import sys; sys.path.insert(0,'$REPO'); from referral_gnn.paths import resolve_run as r; print(r('v5_unmasked_s1'))")
T42=$(py -c "import sys; sys.path.insert(0,'$REPO'); from referral_gnn.paths import resolve_run as r; print(r('lgbm_tabular_s42'))"); T1=$(py -c "import sys; sys.path.insert(0,'$REPO'); from referral_gnn.paths import resolve_run as r; print(r('lgbm_tabular_s1'))")
EVAL --split validation --step step12b --strata --step12 --lean --out runs/analysis/step12b/eval --hybrid "$(basename $G42):$(basename $T42)" --hybrid "$(basename $G1):$(basename $T1)" \
     v5_unmasked_s42 v5_unmasked_s1 lgbm_tabular_s42 lgbm_tabular_s1 s12b_emb_lr_s42 s12b_emb_lr_s1 s12b_vars_lr_s42 s12b_vars_lr_s1 s12_lgbm_flat_lambdarank_s42 s12_lgbm_flat_lambdarank_s1
py "$S/analysis/step12/step12b_judge.py" --eval runs/analysis/step12b/eval
