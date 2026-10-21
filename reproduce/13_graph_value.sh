#!/usr/bin/env bash
# STEP13 phase 1 (Implementation/STEP13_GRAPH_VALUE_PLAN.md, frozen 2026-10-21): where standard models fall short and what the
# graph adds. One stage per call, in order; each stage refuses to run out of order (manifests, decision entry, single 2019 pass).
#   check      hand-calculated feature check (aggregates only)
#   design     design manifest (needs a clean, committed worktree; before any STEP13 run)
#   train      10 GNN runs (resumable), 4 LightGBM families, 2 MLP families; stops before a job when on battery (ALLOW_BATTERY=1 overrides)
#   compat     re-score the existing checkpoints on 2018 and compare with STEP12
#   resolve    resolved manifest (runs, checkpoints, tuning ledgers, compat result, evaluation commit)
#   eval2018   one 2018 evaluation of every cell's runs with candidate-score dumps, then the 2018 verdicts
#   decide     decision entry hashing the 2018 outputs (unlocks the single 2019 pass)
#   eval2019   one 2019 evaluation, then the directional-replication verdicts
# Usage: reproduce/13_graph_value.sh <stage>
source "$(dirname "$0")/_common.sh"
O=runs/analysis/step13; E=$O/eval
battery() { [[ -z ${ALLOW_BATTERY:-} ]] && command -v pmset >/dev/null && pmset -g batt | grep -q "Battery Power"; }
cell_runs() { py -c "import json; m = json.load(open('$O/resolved_manifest.json')); print(' '.join([f for c in m['cells'].values() for f in c.values()] + list(m['sensitivity'].values())))"; }
partial() { ls -d runs/*_"$1"* 2>/dev/null | grep -v '_INCOMPLETE$' | grep -q .; }
case ${1:?stage} in
  check)    step "STEP13 feature check"; py "$S/checks/step13_feature_check.py" ;;
  design)   step "STEP13 design manifest"; py "$S/analysis/step13/step13_manifest.py" design ;;
  train)    [[ -f $O/design_manifest.json ]] || { echo "REFUSED: write the design manifest first (stage design)"; exit 1; }
            step "STEP13 GNN cells"; TRAIN step13_gnn.tsv
            for FS in flat_rf full_nohist full insmatch; do
              if battery; then echo "=== on battery; stopping before LightGBM $FS"; exit 1; fi
              if ls -d runs/*_s13_lgbm_${FS}_lambdarank_s7 >/dev/null 2>&1; then echo "=== LightGBM $FS complete, skipped"; continue; fi
              if partial s13_lgbm_${FS}_lambdarank_s; then echo "REFUSED: a part-finished LightGBM $FS tuning exists; rename it *_INCOMPLETE first"; exit 1; fi
              step "STEP13 LightGBM $FS"; py "$S/train/step12_lgbm.py" --feature-set $FS --objective lambdarank --mode tune --extra-seeds 7
            done
            for FS in flat_rf full; do
              if battery; then echo "=== on battery; stopping before MLP $FS"; exit 1; fi
              if [[ -f $O/mlp_${FS}_tuning.json ]]; then echo "=== MLP $FS complete, skipped"; continue; fi
              if partial s13_mlp_${FS}_h; then echo "REFUSED: a part-finished MLP $FS tuning exists; rename its runs *_INCOMPLETE first"; exit 1; fi
              step "STEP13 MLP $FS"; py "$S/train/step13_mlp.py" --feature-set $FS --mode tune
            done ;;
  compat)   step "STEP13 backwards-compatibility check"; py "$S/analysis/step13/step13_compat.py" ;;
  resolve)  step "STEP13 resolved manifest"; py "$S/analysis/step13/step13_manifest.py" resolve ;;
  eval2018) step "STEP13 2018 evaluation"; EVAL --split validation --step 13 --dump-scores --out $E $(cell_runs)
            py "$S/analysis/step13/step13_verdicts.py" --split validation ;;
  decide)   step "STEP13 decision entry"; DECIDE --step 13 --summary $E/validation_summary.json --final model_s13_gnn_nohist:base \
              --amendment "STEP13_GRAPH_VALUE_PLAN.md (frozen 2026-10-21) + STEP13_AMENDMENTS.md A1, A2" --note "2018 verdicts: $E/validation_step13_verdicts.json" ;;
  eval2019) step "STEP13 2019 evaluation (once)"; EVAL --split test --step 13 --dump-scores --out $E $(cell_runs)
            py "$S/analysis/step13/step13_verdicts.py" --split test ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
