#!/usr/bin/env bash
# STEP12a (STEP12_CONGESTION_PLAN.md, frozen; amendment A1): why does the graph model concentrate load less than the table?
source "$(dirname "$0")/_common.sh"
step "part 1: gate cell (neural model with no PCP history) + a third no-graph seed; gate evaluation; frozen decisions"
TRAIN step12a_gate.tsv
EVAL --split validation --step step12a_gate --strata --step12 --out runs/analysis/step12a/gate v5_nograph_s42 v5_nograph_s1 v5_nograph_s7 \
     lgbm_primary_s42 lgbm_primary_s1 lgbm_tabular_s42 lgbm_tabular_s1 s12_gate_s42 s12_gate_s1 s12_gate_s7
py "$S/analysis/step12/step12_congestion.py" --stage gate --eval runs/analysis/step12a/gate
step "part 2: objectives (neural LambdaRank / BCE; LightGBM softmax / binary / xendcg) and the PCP-attribute check"
TRAIN step12a_objective.tsv
py "$S/train/step12_lgbm.py" --objective lambdarank --no-pcp --mode fixed --config 127:50:1000 --seeds 42
EVAL --split validation --step step12a_pcpcheck --strata --step12 --lean --out runs/analysis/step12a/pcpcheck lgbm_tabular_s42 lgbm_tabular_s1 s12_lgbm_tabular_lambdarank_nopcp_L127_M50_R1000_s42
py "$S/analysis/step12/step12_congestion.py" --stage pcpcheck --eval runs/analysis/step12a/pcpcheck
for O in softmax binary xendcg; do py "$S/train/step12_lgbm.py" --objective $O --mode tune; done
step "part 3: volume cells (B2 shares, H5 log volume); part 4: capacity grid"
py "$S/train/step12_lgbm.py" --objective lambdarank --volume shares --mode fixed --config 127:50:1000 --seeds 42 1
TRAIN step12a_volume.tsv
py "$S/train/step12_lgbm.py" --objective lambdarank --mode capacity
py "$S/train/step12_lgbm.py" --objective lambdarank --mode fixed --config 127:200:1000 --seeds 42 1
py "$S/train/step12_lgbm.py" --objective lambdarank --mode fixed --config 255:50:2000 --seeds 42 1
step "part 5: final evaluation (two passes to bound memory) and the pre-registered verdict"
# run lists of the two passes: see docs/STEP12_CONGESTION_PLAN.md §5 (pass A = all STEP12a cells + reference; pass B = the same + GNN / no-graph, volume neutralised)
echo "   EVAL --split validation --step step12a_final    --strata --step12          --out runs/analysis/step12a/final    <pass A runs>"
echo "   EVAL --split validation --step step12a_final_b1 --lean --ablate no_volume,no_volume_plus --out runs/analysis/step12a/final_b1 <pass B runs>"
echo "   EVAL --split validation --step step12a_capacity --lean --out runs/analysis/step12a/capacity <reference + capacity grid + endpoint runs>"
echo "   python $S/analysis/step12/step12_congestion.py --stage final --eval runs/analysis/step12a/final --eval-extra runs/analysis/step12a/final_b1 --eval-capacity runs/analysis/step12a/capacity"
step "amendment A1 (post hoc): three more gate seeds, volume-blind 'strict' cells, secondary analysis"
TRAIN step12a_A1_gate_seeds.tsv
py "$S/train/step12_lgbm.py" --objective lambdarank --volume strict --mode fixed --config 127:50:1000 --seeds 42 1
TRAIN step12a_A1_strict.tsv
echo "   EVAL --split validation --step step12a_secondary --strata --step12 --out runs/analysis/step12a/secondary <30 runs: gate ×6, lr, bce, shares, log, strict ×6, tables>"
echo "   python $S/analysis/step12/step12_secondary.py --eval runs/analysis/step12a/secondary"
