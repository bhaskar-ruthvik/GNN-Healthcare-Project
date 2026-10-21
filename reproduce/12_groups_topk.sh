#!/usr/bin/env bash
# After the 9 Oct meeting: the group exceptions (85+, early-tenure PCPs) and longer lists (top 10 / 15). 2018 only.
source "$(dirname "$0")/_common.sh"
R=(v5_unmasked_s42 v5_unmasked_s1 v5_unmasked_s42_dup v5_nograph_s42 v5_nograph_s1 lgbm_tabular_s42 lgbm_tabular_s1)
for K in 5 10 15; do step "evaluation with $K names"; EVAL --split validation --step step12_topk --list-size $K --strata --out runs/analysis/step12_topk/K$K "${R[@]}"; done
py "$S/analysis/step12/step12_topk_summary.py"
py "$S/analysis/step12/step12_groups.py"
