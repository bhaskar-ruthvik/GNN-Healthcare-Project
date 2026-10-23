#!/usr/bin/env bash
# Post hoc matched-fit comparison on the STEP13 score dumps (descriptive), then STEP14 (docs/STEP14_GRAPH_FEATURE_BLEND_PLAN.md,
# frozen 2026-10-23): the steered recommender with the graph-feature table. 2018 choice, decision record, 2019 once.
# Needs reproduce/13_graph_value.sh through eval2019 (the STEP13 dumps for both years).
source "$(dirname "$0")/_common.sh"
step "Post hoc matched fit (both years)"
for SP in validation test; do py "$S/analysis/step13/step13_matched_load.py" --split $SP; done
step "STEP14 2018 choice"; py "$S/analysis/step14/step14_blend.py" --split validation
step "STEP14 decision";    py "$S/analysis/step14/step14_blend.py" --split decide
step "STEP14 2019 (once)"; py "$S/analysis/step14/step14_blend.py" --split test
