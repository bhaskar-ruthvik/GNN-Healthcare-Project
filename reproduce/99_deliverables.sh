#!/usr/bin/env bash
# Decks (self-contained HTML) and the verification workbook, from runs/analysis/ only (aggregates) → deliverables/.
source "$(dirname "$0")/_common.sh"
P=$S/deliverables/presentation
step "2 Oct deck"; py "$P/build_data.py"; py "$P/render.py"
step "9 Oct deck (original and revised)"; py "$S/analysis/step11/step11_network_viz.py"; py "$P/build_data_w2.py"; py "$P/render_w2.py"; py "$P/render_w2_revised.py"
step "16 Oct deck (original and revised)"; py "$P/build_data_w3.py"; py "$P/render_w3.py"; py "$P/render_w3_revised.py"
step "combined story deck (needs the 2 Oct and 9 Oct deck data above)"; py "$P/build_data_story.py"; py "$P/compose_story_template.py"; py "$P/render_story.py"
step "verification workbook (recalculate in Excel or LibreOffice before reading cached values)"
py "$S/deliverables/build_verification_workbook.py" --out deliverables/referral_gnn_results_verification.xlsx
