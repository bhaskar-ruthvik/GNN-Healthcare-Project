#!/bin/zsh
# STEP12a part 2, LightGBM side (tabular branch). CPU only; runs alongside the neural chain. From Implementation/.
#  1. R2 PCP-attribute check: LambdaRank without PCP columns at the reference configuration (127 leaves, min leaf 50, 1000 rounds), seed 42;
#     evaluated with the reference runs; step12_congestion.py --stage pcpcheck records the outcome in the decision file.
#  2. H1/H2 LightGBM cells: group-softmax (no-PCP variant if the check fires) and binary, each tuned on the 12-config grid + seed 1;
#     rank_xendcg as a descriptive extra.
set -e
PY=env/bin/python
echo "=== pcp check train $(date '+%H:%M:%S')"
$PY -W ignore scripts/step12_lgbm.py --objective lambdarank --no-pcp --mode fixed --config 127:50:1000 --seeds 42 --threads 6 2>&1 | grep -v -i warn
NOP=$(ls -d runs/*_s12_lgbm_tabular_lambdarank_nopcp_L127_M50_R1000_s42 | tail -1)
echo "=== pcp check eval $(date '+%H:%M:%S')"
$PY -W ignore scripts/step10_evaluate.py --split validation --step step12a_pcpcheck --strata --step12 --lean --out runs/analysis/step12a/pcpcheck \
    runs/20261007T195708_lgbm_tabular_s42 runs/20261007T195853_lgbm_tabular_s1 $NOP > runs/step12a_pcpcheck_eval.log 2>&1
$PY scripts/step12_congestion.py --stage pcpcheck --eval runs/analysis/step12a/pcpcheck
USE=$($PY -c "import json;print(int(json.load(open('runs/analysis/step12a_decisions.json'))['pcp_check']['h1_uses_nopcp_softmax']))")
echo "=== softmax (no-PCP=$USE) $(date '+%H:%M:%S')"
if [[ $USE == 1 ]]; then $PY -W ignore scripts/step12_lgbm.py --objective softmax --no-pcp --mode tune --threads 6 2>&1 | grep -v -i warn | grep -E "wrote|done|Error|Trace"
else $PY -W ignore scripts/step12_lgbm.py --objective softmax --mode tune --threads 6 2>&1 | grep -v -i warn | grep -E "wrote|done|Error|Trace"; fi
echo "=== binary $(date '+%H:%M:%S')"
$PY -W ignore scripts/step12_lgbm.py --objective binary --mode tune --threads 6 2>&1 | grep -v -i warn | grep -E "wrote|done|Error|Trace"
echo "=== xendcg (descriptive) $(date '+%H:%M:%S')"
$PY -W ignore scripts/step12_lgbm.py --objective xendcg --mode tune --threads 6 2>&1 | grep -v -i warn | grep -E "wrote|done|Error|Trace"
echo "=== LGBM PART2 DONE $(date '+%H:%M:%S')"
