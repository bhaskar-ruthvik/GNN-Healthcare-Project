#!/bin/zsh
# STEP12b: sitting 1 (H2 emb-LR + no-emb on GNN s42/s1 on MPS; flat LightGBM tune on CPU, in parallel), then sitting 2
# (one lean evaluation incl. H0 hybrids for both seed pairings, then the pre-stated judge). From Implementation/.
set -e
PY=env/bin/python
G42=runs/20260929T213930_v5_unmasked_s42; G1=runs/20260930T094407_v5_unmasked_s1
T42=runs/20261007T195708_lgbm_tabular_s42; T1=runs/20261007T195853_lgbm_tabular_s1
echo "=== sitting 1 start $(date '+%H:%M:%S')"
$PY -W ignore scripts/step12_lgbm.py --feature-set flat --objective lambdarank --mode tune --threads 6 > runs/step12b_flat.log 2>&1 &
FLAT=$!
for g in $G42 $G1; do
  $PY -W ignore scripts/step12b_emb_lr.py --gnn-run $g 2>&1 | grep -v -i warn | grep -E "matrix|C'|wrote|Error|Trace"
  $PY -W ignore scripts/step12b_emb_lr.py --gnn-run $g --no-emb 2>&1 | grep -v -i warn | grep -E "matrix|C'|wrote|Error|Trace"
done
wait $FLAT; grep -E "wrote|done|Error|Trace" runs/step12b_flat.log
echo "=== sitting 1 done $(date '+%H:%M:%S')"
RUNS=($G42 $G1 $T42 $T1 $(ls -d runs/*_s12b_emb_lr_s42 runs/*_s12b_emb_lr_s1 runs/*_s12b_vars_lr_s42 runs/*_s12b_vars_lr_s1 runs/*_s12_lgbm_flat_lambdarank_s42 runs/*_s12_lgbm_flat_lambdarank_s1))
echo "=== sitting 2: evaluation of ${#RUNS} runs $(date '+%H:%M:%S')"
$PY -W ignore scripts/step10_evaluate.py --split validation --step step12b --strata --step12 --lean --out runs/analysis/step12b/eval \
    --hybrid "${G42:t}:${T42:t}" --hybrid "${G1:t}:${T1:t}" $RUNS > runs/step12b_eval.log 2>&1
echo "=== judge $(date '+%H:%M:%S')"
$PY scripts/step12b_judge.py --eval runs/analysis/step12b/eval
echo "=== STEP12B DONE $(date '+%H:%M:%S')"
