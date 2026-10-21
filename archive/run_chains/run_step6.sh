#!/bin/zsh
# Sequential experiment queue for step 6 (protocol: STEP6_PROTOCOL.md). Run from Implementation/.
set -u
PY=env/bin/python
run() { name=$1; shift; echo "=== $name  $(date '+%H:%M:%S')"; $PY scripts/train.py --name $name --graph-bucket week --epochs 30 --patience 3 "$@" 2>&1 | grep -v -E "FutureWarning|warnings.warn" | grep -E "epoch|early|summary|WARNING|Error|Traceback"; }
run main_s42
run pairmlp --no-graph
run abl_no_history --ablation no_history_features
run abl_no_ps_prior --ablation no_patient_specialist_prior
run abl_no_history_no_ps --ablation no_history_features --ablation no_patient_specialist_prior
run abl_no_diagnosis --ablation no_diagnosis
run abl_no_distance --ablation no_distance
run abl_no_ref_edges --ablation no_referral_edges
run ctx_patient_pcp --scorer-context enc_patient_pcp
run main_s1 --seed 1
run main_s2 --seed 2
echo "=== QUEUE DONE $(date '+%H:%M:%S')"
