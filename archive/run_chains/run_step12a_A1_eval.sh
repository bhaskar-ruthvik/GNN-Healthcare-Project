#!/bin/zsh
# STEP12a A1: after the strict training launcher exits, evaluate the secondary folder (--strata --step12, noise variants included)
# and run the frozen rev-1 secondary analysis. From Implementation/. Usage: zsh scripts/run_step12a_A1_eval.sh <launcher PID>
WAIT=${1:?pid}
while kill -0 $WAIT 2>/dev/null; do sleep 30; done
RUNS=($(ls -d runs/*_s12_gate_s* runs/*_s12_lr_s* runs/*_s12_bce_s* runs/*_s12_shares_s* runs/*_s12_log_s* runs/*_s12_strict_s* \
        runs/20261007T195708_lgbm_tabular_s42 runs/20261007T195853_lgbm_tabular_s1 runs/*_s12_lgbm_tabular_softmax_s* runs/*_s12_lgbm_tabular_lambdarank_strict_L127_M50_R1000_s*))
for r in $RUNS; do [[ -f $r/summary.json ]] || { echo "MISSING summary: $r"; exit 1; }; done
echo "=== secondary evaluation: ${#RUNS} runs $(date '+%H:%M:%S')"
env/bin/python -W ignore scripts/step10_evaluate.py --split validation --step step12a_secondary --strata --step12 --out runs/analysis/step12a/secondary $RUNS > runs/step12a_secondary_eval.log 2>&1 || { echo "EVAL FAILED"; exit 1; }
echo "=== secondary analysis $(date '+%H:%M:%S')"
env/bin/python scripts/step12_secondary.py --eval runs/analysis/step12a/secondary 2>&1 | grep -v -i warn
echo "=== A1 EVAL DONE $(date '+%H:%M:%S')"
