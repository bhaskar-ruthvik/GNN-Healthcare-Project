#!/usr/bin/env bash
# Step 11 (STEP11_PLAN.md): is a plain table enough? Baselines, groups, masking probe, network over time, data size, snapshots.
source "$(dirname "$0")/_common.sh"
step "train: GNN without message passing; pure-tabular and other baselines (LightGBM / logistic regression / ALS / item-kNN)"
TRAIN step11_1_nograph.tsv
py "$S/train/baselines11.py" --which lgbm_primary lgbm_graph logreg als itemknn
py "$S/train/baselines11.py" --which lgbm_tabular logreg_tabular                # STEP11 B1: pure tabular (claim attributes only)
step "11.1 baselines on the four lenses (2018), the pure-tabular comparison (2018 → decision → 2019 once)"
EVAL --split validation --step 11.1 --out runs/analysis/step11_1 v5_unmasked_s42 v5_unmasked_s1 v5_unmasked_s42_dup v5_nograph_s42 v5_nograph_s1 \
     itemknn als_s42 logreg_s42 lgbm_primary_s42 lgbm_primary_s1 lgbm_graph_s42 lgbm_graph_s1 lgbm_tabular_s42 lgbm_tabular_s1
R111=(v5_unmasked_s42 v5_unmasked_s1 v5_unmasked_s42_dup v5_nograph_s42 v5_nograph_s1 lgbm_tabular_s42 lgbm_tabular_s1 logreg_tabular_s42)
EVAL --split validation --step 11.1-tabular --out runs/analysis/step11_1_tabular "${R111[@]}"
DECIDE --step 11.1-tabular --summary runs/analysis/step11_1_tabular/validation_summary.json --primary model_unmasked_v5:blend0.3 --final model_unmasked_v5:blend0.3 --amendment "STEP11_PLAN G1 + B4"
EVAL --split test --step 11.1-tabular --out runs/analysis/step11_1_tabular "${R111[@]}"
py "$S/analysis/step11/step11_triggers.py" --dir runs/analysis/step11_1_tabular
py "$S/analysis/step11/step11_lens_compare.py"
step "11.2 groups on the lenses (2018)"
EVAL --split validation --step 11.2 --strata --out runs/analysis/step11_2 v5_unmasked_s42 v5_unmasked_s1 v5_unmasked_s42_dup v5_nograph_s42 v5_nograph_s1 lgbm_tabular_s42 lgbm_tabular_s1
py "$S/analysis/step11/step11_strata.py"
step "11.3a masking probe (embeddings → gender / race)"
py "$S/analysis/step11/step11_probe.py" --extract && py "$S/analysis/step11/step11_probe.py" --probe
step "11.4a network over time (no model)"
py "$S/analysis/step11/step11_network_evolution.py"; py "$S/analysis/step11/step11_network_viz.py"
step "11.4 preliminary data size and 11.4c yearly snapshots"
TRAIN step11_4_prelim.tsv
py "$S/train/prelim_datasize_lgbm.py" --train-end 2017-01-01 --variants tabular graph --seeds 42 1
py "$S/analysis/step11/step11_prelim_datasize.py"
for SN in "S16 2016-10-01 2017-01-01" "S17 2017-10-01 2018-01-01" "S18 2018-10-01 2019-01-01" "S19 2019-07-22 2019-10-22"; do set -- $SN
  py "$S/data/snapshot_build.py" --name "$1" --train-end "$2" --holdout-end "$3"; done
TRAIN step11_4c_snapshots.tsv
step "11.5 architecture tries (C2 scorer skip, C1 history dropout) and 11.6 the table's extra new connections"
TRAIN step11_5_arch.tsv
EVAL --split validation --step 11.5 --out runs/analysis/step11_5 v5_unmasked_s42 v5_unmasked_s1 v5_unmasked_s42_dup lgbm_tabular_s42 lgbm_tabular_s1 \
     c2_skip_s42 c2_skip_s1 c1_hd03_s42 c1_hd03_s1 c1_hd06_s42 c1_hd06_s1
py "$S/analysis/step11/step11_arch_lenses.py"
EVAL --split validation --step 11.6-picks --dump-picks v5_unmasked_s42,lgbm_tabular_s42 --out runs/analysis/step11_6_picks v5_unmasked_s42 lgbm_tabular_s42
py "$S/analysis/step11/step11_newconn.py"
