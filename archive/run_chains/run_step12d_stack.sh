#!/bin/zsh
# STEP12d B after the fold models: out-of-fold GNN scores per seed (centred; scale check), then the stacked LightGBM per seed
# (reference configuration, fixed mode; LightGBM seed s uses GNN seed s's scores; seed 7 pairs with full GNN s42_dup).
set -e
PY=env/bin/python
D="runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz runs/analysis/step12c/dump/test_candidate_scores_PRIVATE.npz"
for p in "42:runs/20260929T213930_v5_unmasked_s42" "1:runs/20260930T094407_v5_unmasked_s1" "7:runs/20260930T140429_v5_unmasked_s42_dup"; do
  sd=${p%%:*}; g=${p#*:}
  F=($(ls -d runs/*_s12d_fold0_s$sd runs/*_s12d_fold1_s$sd runs/*_s12d_fold2_s$sd))
  echo "=== GNN OOF seed $sd $(date '+%H:%M:%S')"
  $PY -W ignore scripts/step12d_gnn_oof.py --seed $sd --gnn-run $g --fold-runs $F --dumps ${=D} 2>&1 | grep -v -i warn
done
for sd in 42 1 7; do
  echo "=== stacked LightGBM seed $sd $(date '+%H:%M:%S')"
  $PY -W ignore scripts/step12_lgbm.py --objective lambdarank --mode fixed --config 127:50:1000 --seeds $sd --extra-npz runs/analysis/step12d/gnn_oof_s$sd.npz --threads 6 2>&1 | grep -E "matrix|wrote|Error|Trace"
done
echo "=== STACK DONE $(date '+%H:%M:%S')"
