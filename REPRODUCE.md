# Reproducing the results

Each step has one script in `reproduce/` with the exact commands, in order. Set `REFERRAL_GNN_WORKSPACE` (see README) and run
from anywhere, for example `bash reproduce/12c_steered_blend.sh`. Training is long:
- about 1 hour per GNN run on Apple MPS, about 15–30 minutes on a modern GPU;
- minutes per LightGBM model.

Evaluation passes take minutes. The scripts only append: they never overwrite a completed output.

2018 is the validation year, where every choice is made. 2019 is used once per pre-stated check, after
`scripts/evaluate/step10_decide.py` records the decision.

| Result (where shown) | Script | Output (in the workspace) |
|---|---|---|
| Prepared inputs v1–v6, MIPS yardstick | `reproduce/00_data.sh` | `prepared_data/` |
| Imitation concentrates load; 30% quality weight (+6.6 MIPS on 2019 at PCP-habit fit); filters, reliance, bias, SHAP; load penalty (2 Oct deck, story slides 4–5) | `reproduce/10_mips_lenses.sh` | `runs/analysis/step10_*` |
| Graph model vs pure-tabular LightGBM on the lenses, 2018 and 2019 (9 Oct deck, story slide 6) | `reproduce/11_graph_vs_table.sh` | `runs/analysis/step11_1_tabular/lens_compare.json` |
| Groups, masking probe, network over time, data size, snapshots (9 Oct deck, story slide 7, appendix) | `reproduce/11_graph_vs_table.sh` | `runs/analysis/step11_2`, `step11_3a`, `step11_4a`, `step11_4_prelim` |
| Group exceptions not real (Holm); top 10 / 15 (16 Oct deck, story slides 7–8) | `reproduce/12_groups_topk.sh` | `runs/analysis/step12_groups`, `step12_topk` |
| Why the graph model concentrates less: pre-registered tests H1–H5 (story slide 9) | `reproduce/12a_congestion.sh` | `runs/analysis/step12a/*_summary.json`, `step12a_decisions.json` |
| Simple hybrids (story slide 10) | `reproduce/12b_simple_hybrids.sh` | `runs/analysis/step12b/judge_summary.json` |
| Complementarity, steering frontiers, steered blend on 2018 and 2019, group checks (story slides 11–13, 15) | `reproduce/12c_steered_blend.sh` | `runs/analysis/step12c/` |
| Heavier hybrids (story slide 14) | `reproduce/12d_heavy_hybrids.sh` | `runs/analysis/step12d/` |
| Steered blend with 10 / 15 names (story slide 13) | `reproduce/12e_topk_hybrids.sh` | `runs/analysis/step12e/` |
| What the graph adds: pre-registered tests against information-matched models (H1a, H1b, H2, H3), 2018 and 2019 once (story v2 slide 16). Stages: `check`, `design`, `train`, `amend A3`, `compat`, `resolve`, `eval2018`, `decide`, `eval2019` | `reproduce/13_graph_value.sh <stage>` | `runs/analysis/step13/` (manifests, `eval/*_step13_verdicts.json`) |
| Graph information at matched fit, post hoc (story v2 slide 17); steered recommender with the graph-feature table (story v2 slide 18) | `reproduce/14_graph_feature_blend.sh` | `runs/analysis/step13/posthoc_matched_load_*.json`, `runs/analysis/step14/` |
| Decks and verification workbook | `reproduce/99_deliverables.sh` | `deliverables/` |
| Steps 5–9 (earlier MRR-era results, `docs/RESULTS.md`) | `scripts/legacy/` (commands in `archive/run_chains/`) | `runs/analysis/` |

## What was re-run to check this repository

The code was reorganised into this repository after the results were produced. A check that re-runs the cheap stages
from the repository against the original inputs and checkpoints, and compares the outputs with the saved results, is
**pending**. Its outcome will be recorded in `docs/reproduction_check.md`. Partial evidence so far: the Step 13 compatibility
stage re-scored the existing Graph (seeds 42, 1), G and Table checkpoints with this repository's code; every top-5 list and lens
was identical to the Step 12 outputs on all 5,550 referrals (`runs/analysis/step13/compat.json`). Training is not repeated: it is long, and MPS is
not bit-deterministic. The run manifests record the inputs and code hashes of every model.

## Shared definitions worth knowing

- **Lenses and eligible sets:** `referral_gnn/step10.py` (cohort, eligibility, yardstick) and `scripts/evaluate/step10_evaluate.py` (all per-referral metrics).
- **Ranking arithmetic for hybrids:** `referral_gnn/candidates.py`. This covers the 30% quality weight, the equal blend, steering, seed pairings, Pareto fronts and the PCP-cluster bootstrap.
- **Paths and run names:** `referral_gnn/paths.py`.
