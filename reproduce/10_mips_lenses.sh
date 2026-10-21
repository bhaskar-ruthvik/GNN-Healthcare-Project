#!/usr/bin/env bash
# Step 10 (STEP10_PROTOCOL.md): the four lenses, the 30% quality weight, filters, bias, SHAP, load-aware variants.
# Training ≈ 1 h per GNN run on Apple MPS; each evaluation pass a few minutes.
source "$(dirname "$0")/_common.sh"
FINAL=(v5_unmasked_s42 v5_unmasked_s1)
step "train: Step 9 v4 runs and Step 10 v5 runs"
TRAIN step9_v4.tsv; TRAIN step10_3_v5.tsv
step "10.1 descriptive charts (training + validation years only)"
py "$S/analysis/step10/step10_descriptive.py"
step "10.2 / 10.3 evaluation: validation → recorded decision → one 2019 pass"
EVAL --split validation --step 10.2 --out runs/analysis/step10_2 v4_unmasked_s42 v4_masked_s42 v4_unmasked_s1 v4_masked_s1
DECIDE --step 10.2 --summary runs/analysis/step10_2/validation_summary.json --final model_unmasked:blend0.3 --amendment "STEP10_PROTOCOL A3"
EVAL --split test --step 10.2 --out runs/analysis/step10_2 v4_unmasked_s42 v4_masked_s42 v4_unmasked_s1 v4_masked_s1
R103=(v4_unmasked_s42 v4_unmasked_s1 v5_unmasked_s42 v5_unmasked_s1 v5_unmasked_s42_dup v5_masked_s42)
EVAL --split validation --step 10.3 --tag-version --a4 --out runs/analysis/step10_3 "${R103[@]}"
DECIDE --step 10.3 --summary runs/analysis/step10_3/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 --amendment "STEP10_PROTOCOL A5"
EVAL --split test --step 10.3 --tag-version --a4 --out runs/analysis/step10_3 "${R103[@]}"
for s in validation test; do py "$S/evaluate/step10_plots.py" --split $s --dir runs/analysis/step10_3 --model-family model_unmasked_v5; done
step "10.4 filters and reliance (descriptive; one decision + 2019 pass per setting)"
declare -a NAMES=(reliance cap10 cap50 capnone ever_active insurer_set non_choice non_specific include_seen)
declare -a FLAGS=("--ablate no_mips,no_pcp_history,no_affiliation,no_insurer,no_distance" "--cap 10" "--cap 50" "--cap none" "--active ever"
                  "--insurer-set" "--specialties non_choice" "--specialties non_specific" "--include-seen")
for i in "${!NAMES[@]}"; do n=${NAMES[$i]}; f=${FLAGS[$i]}; O=runs/analysis/step10_4/$n
  # shellcheck disable=SC2086
  EVAL --split validation --step 10.4:$n --tag-version --out $O $f "${FINAL[@]}"
  DECIDE --step 10.4:$n --summary $O/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 --amendment "STEP10_PROTOCOL A6/A12 (descriptive)"
  # shellcheck disable=SC2086
  EVAL --split test --step 10.4:$n --tag-version --out $O $f "${FINAL[@]}"
done
step "10.5 bias (+ no-cap follow-up A8)"
R105=(v5_unmasked_s42 v5_unmasked_s1 v5_unmasked_s42_dup v5_masked_s42 v4_unmasked_s42 v4_unmasked_s1 v4_masked_s42 v4_masked_s1)
EVAL --split validation --step 10.5 --tag-version --out runs/analysis/step10_5 "${R105[@]}"
DECIDE --step 10.5 --summary runs/analysis/step10_5/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 --amendment "STEP10_PROTOCOL A7"
EVAL --split test --step 10.5 --tag-version --out runs/analysis/step10_5 "${R105[@]}"
for s in validation test; do py "$S/analysis/step10/step10_bias.py" --split $s; done
EVAL --split validation --step 10.5:capnone --tag-version --cap none --out runs/analysis/step10_5_capnone "${FINAL[@]}"
DECIDE --step 10.5:capnone --summary runs/analysis/step10_5_capnone/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 --amendment "STEP10_PROTOCOL A8"
EVAL --split test --step 10.5:capnone --tag-version --cap none --out runs/analysis/step10_5_capnone "${FINAL[@]}"
for s in validation test; do py "$S/analysis/step10/step10_bias.py" --split $s --dir runs/analysis/step10_5_capnone; done
step "10.6 per-input SHAP on the final model (≈ 70 min)"
SH=$(py -c "import sys; sys.path.insert(0,'$REPO'); from referral_gnn.paths import resolve_run; print(resolve_run('v5_unmasked_s42'))")
py "$S/analysis/step10/shap_inputs.py" "$SH" --n 60 --perms 12
py "$S/analysis/step10/shap_inputs.py" "$SH" --summarize
step "10.7 load-aware: option B runs, then option A variants + B on one pass"
TRAIN step10_7_B.tsv
R107=(v5_unmasked_s42 v5_unmasked_s1 v6_load_s42 v6_load_s1 v5_posw05_s42 v5_posw05_s1)
EVAL --split validation --step 10.7 --tag-version --load-variants --out runs/analysis/step10_7 "${R107[@]}"
DECIDE --step 10.7 --summary runs/analysis/step10_7/validation_summary.json --final model_unmasked_v5:pen0.2 --amendment "STEP10_PROTOCOL A10 + A11"
EVAL --split test --step 10.7 --tag-version --load-variants --out runs/analysis/step10_7 "${R107[@]}"
step "10.9 PCP–specialist race concordance (A13)"
EVAL --split validation --step 10.9 --tag-version --load-variants --provider-race --out runs/analysis/step10_9_concordance "${FINAL[@]}"
DECIDE --step 10.9 --summary runs/analysis/step10_9_concordance/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 --amendment "STEP10_PROTOCOL A13"
EVAL --split test --step 10.9 --tag-version --load-variants --provider-race --out runs/analysis/step10_9_concordance "${FINAL[@]}"
for s in validation test; do py "$S/analysis/step10/step10_concordance.py" --split $s; done
