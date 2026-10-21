# Scripts

Every script resolves data paths against the workspace (`referral_gnn/paths.py`) and names runs without timestamps.
Run any of them with `python scripts/<folder>/<name>.py --help`; `reproduce/` lists the exact commands per step.

## data/ — Raw workbooks → prepared inputs

| Script | What it does |
|---|---|
| `audit_new_source.py` | Aggregate-only audit of a new source workbook, in the style of the cleaning notebook's header audit. |
| `build_inputs.py` | Step 3: build model inputs from the prepared bundle and validate them. |
| `build_mips_yardstick.py` | Step 10.1 — fixed per-specialist MIPS yardstick (STEP10_PROTOCOL §2.3). |
| `build_three_node_inputs.py` | Build new private inputs and verify temporal construction without training. |
| `build_three_node_v2_inputs.py` | Build and check three-node-v2 inputs (v1 + patient-origin distance). |
| `build_three_node_v3_inputs.py` | Build and check three-node-v3 inputs (v2 + insurer-history and shared-affiliation pair features). |
| `build_three_node_v4_inputs.py` | Build and check three-node-v4 inputs (STEP9_PROTOCOL §2). |
| `build_three_node_v5_inputs.py` | Build and check three-node-v5 inputs (STEP10_PROTOCOL §2.3 model side, §4 step 10.3). |
| `build_three_node_v6_inputs.py` | Build and check three-node-v6 inputs (STEP10_PROTOCOL A10, Option B variant 1). |
| `extract_disease_mapping.py` | Read-only extraction of the source's existing diagnosis/category assignments. |
| `extract_provider_ruca.py` | STEP11 §11.2: provider practice RUCA codes (source MUP columns), one value per provider NPI, for the rural-specialist split. |
| `snapshot_build.py` | STEP11 §11.4(c): build one cumulative snapshot's inputs, end to end, on the laptop (raw data never leaves it). |

## train/ — Training (GNN, tabular baselines, LightGBM cells, hybrids)

| Script | What it does |
|---|---|
| `baselines11.py` | Step 11.1 non-GNN baselines (STEP11_PLAN, frozen 2026-10-07). |
| `prelim_datasize_lgbm.py` | PRELIMINARY data-size runs for STEP11 §11.4 (user request 2026-10-08, before the proper snapshots): LightGBM trained only on training queries strictly before --train-end, with the hyperparameters already selected on the  |
| `step12_lgbm.py` | STEP12a LightGBM cells (STEP12_CONGESTION_PLAN.md, frozen 2026-10-12). |
| `step12b_emb_lr.py` | STEP12b H2: GNN embeddings + exploratory variables → logistic regression (referral_gnn/hybrid.py). |
| `step12d_gnn_oof.py` | STEP12d B inputs (torch process; no LightGBM): out-of-fold GNN scores for the training rows, and full-GNN scores for the stacked LightGBM's selection pool and evaluation sets. |
| `step12d_table_oof.py` | STEP12d A inputs (LightGBM process; no torch computation): out-of-fold TABLE scores for the GNN's training rows and full-pool table scores for the GNN's selection queries. |
| `three_node_train.py` | Train a three-node model. |

## evaluate/ — Evaluation on the lenses; decisions that unlock 2019

| Script | What it does |
|---|---|
| `score_baselines11.py` | Score Step 11.1 baseline runs on the STEP10 cohort's eligible sets, in a process that runs no torch computation. |
| `step10_decide.py` | Record a Step 10 validation decision (STEP10_PROTOCOL §3 two-pass procedure; amendment A2). |
| `step10_evaluate.py` | Step 10.2 / 10.3 evaluation on the MIPS yardstick (STEP10_PROTOCOL.md, frozen 2026-09-28; amendments A1, A2). |
| `step10_plots.py` | Step 10.2 charts from <split>_arms.csv (seed-averaged configurations). |

## analysis/step10/ — Step 10 analyses (MIPS, filters, bias, SHAP)

| Script | What it does |
|---|---|
| `shap_inputs.py` | Step 10.6 per-input SHAP (STEP10_PROTOCOL A9). |
| `step10_bias.py` | Step 10.5 bias analysis (STEP10_PROTOCOL A7). |
| `step10_concordance.py` | PCP–specialist race concordance (STEP10_PROTOCOL A13). |
| `step10_descriptive.py` | Step 10.1 descriptive charts (STEP10_PROTOCOL §4 step 2). |

## analysis/step11/ — Step 11 analyses (graph vs table, groups, network)

| Script | What it does |
|---|---|
| `step11_arch_lenses.py` | STEP11 B5 revision 2: judge architecture changes on the four lenses (2018 STEP10 primary cohort), against the current GNN; tabular LightGBM shown as the reference. |
| `step11_d1_specialty.py` | STEP11 Amendment B5, diagnostic D1: is the tabular model's full-pool advantage on first-time referrals a matter of choosing the right SPECIALTY, or of ranking within it? Inference only, on the 2018 first-time selection q |
| `step11_judge.py` | STEP11 B5 (revision 1): score runs on the judging set J = 2018 first-time queries OUTSIDE the 3,000-query selection sample, FULL pool (every specialist first observed before the day; evaluator pool "all"), weekly cutoff; |
| `step11_lens_compare.py` | Graph model vs tabular LightGBM on the four lenses (+ new connections), 2018 and 2019 side by side (STEP11 B4/G1; MRR not used). |
| `step11_network_evolution.py` | STEP11 §11.4(a): descriptive evolution of the PCP–specialist referral network, year by year (no model). |
| `step11_network_viz.py` | Data for the deck's network-over-time slide (presentation/netviz_w2.json). |
| `step11_newconn.py` | Why does the tabular LightGBM recommend ~10 points more specialists new to the PCP than the graph model, and are they good? Reads runs/analysis/step11_6_picks/validation_top5_picks_PRIVATE.csv.gz (2018 STEP10 cohort, see |
| `step11_prelim_datasize.py` | PRELIMINARY data-size comparison (STEP11 §11.4, user request 2026-10-08): models trained on 2015–2016 vs on 2015–2017, all scored on the same 2018 validation referrals (STEP10 primary cohort, base configuration). |
| `step11_probe.py` | STEP11 §11.3a masking probe: do the masked model's encounter embeddings still encode patient gender / race (through stand-ins)? Step 1 (torch): --extract : encounter embeddings (post-HGT) |
| `step11_strata.py` | STEP11 §11.2 splits on the four lenses (quality, fit, travel, load) |
| `step11_triggers.py` | Step 11.1 triggers T8 (message passing) |

## analysis/step12/ — Step 12 analyses (groups, top-K, congestion mechanism, hybrids)

| Script | What it does |
|---|---|
| `step12_congestion.py` | STEP12a analysis (STEP12_CONGESTION_PLAN.md, frozen 2026-10-12): gate, frozen reference gap, signed primary contrasts. |
| `step12_groups.py` | STEP12 group deep-dive: why some groups break the "graph model eases congestion" trend (patients 85+, early-tenure PCPs). |
| `step12_secondary.py` | STEP12a Amendment A1 + revision 1 (post hoc, secondary; frozen 2026-10-13 before any A1 result was examined). |
| `step12_topk_summary.py` | STEP12 top-K lenses: K = 5, 10, 15 (runs/analysis/step12_topk/K*/, 2018 STEP10 primary cohort, seed-averaged arms). |
| `step12b_judge.py` | STEP12b judging (STEP12B_HYBRID_PLAN.md, rule frozen before the run). |
| `step12c_blend_checks.py` | STEP12c pre-meeting checks on the steered equal blend (setting fixed 2026-10-14: 0.5·z(GNN) |
| `step12c_complementarity.py` | STEP12c: are the GNN and the table complementary? Diagnostics on candidate-level scores (no training). |
| `step12c_frontiers.py` | STEP12c: frontier comparison — does any TWO-model approach reach a better (load, newness) |
| `step12c_point_lenses.py` | STEP12c: full lens profile (incl. |
| `step12c_steering_control.py` | STEP12c control: is the two-list selection gain due to COMBINING the models, or just to steering by knowable attributes? Single-model steering: re-rank each model's eligible set by 0.7·z(score) |
| `step12d_descriptives.py` | STEP12d descriptives for the deck (aggregates only): learned offset β per table-offset run (torch, checkpoint only) |
| `step12d_frontier.py` | STEP12d judging (STEP12D_HEAVY_PLAN.md): do the heavier hybrids reach a better (load, newness) |
| `step12e_topk.py` | STEP12e (STEP12E_TOPK_PLAN.md): the steered equal blend and the hybrids at list length K (10, 15), from a step10_evaluate --dump-scores file made with --list-size K. |

## checks/ — Model-level and snapshot checks

| Script | What it does |
|---|---|
| `snapshot_compare.py` | Snapshot builder checks (STEP11 §11.4(c)). |
| `snapshot_leakcheck.py` | Leakage check (b), pre-cutoff rows (STEP11 §11.4(c); reviewer round 5). |
| `three_node_smoke.py` | Model-level checks for the three-node model (ARCHITECTURE.md "Required verification before training"). |

## deliverables/ — Verification workbook

| Script | What it does |
|---|---|
| `build_verification_workbook.py` | Verification workbook for the professors: every reported number with its source file, and formula checks that recompute the headline claims inside the workbook. |

## deliverables/presentation/ — HTML decks (data builders, templates, renderers)

| Script | What it does |
|---|---|
| `build_data.py` | Collect AGGREGATE numbers for the HTML presentation (presentation/data.json). |
| `build_data_story.py` | Collect AGGREGATE numbers for the combined story deck (presentation/data_story.json). |
| `build_data_w2.py` | Collect AGGREGATE numbers for the week-2 deck (presentation/data_w2.json). |
| `build_data_w3.py` | Collect AGGREGATE numbers for the week-3 deck (presentation/data_w3.json). |
| `compose_story_template.py` | Compose presentation/template_story.html: the week-3 template's style and chart helpers (already checked) |
| `render.py` | Build presentation/referral_recommender_this_week.html: template + aggregate data + descriptive charts (base64). |
| `render_story.py` | Build presentation/referral_graph_story.html: story template + the first deck's base style + aggregate data. |
| `render_w2.py` | Build presentation/referral_graph_week2.html: week-2 template + the first deck's base style + aggregate data. |
| `render_w2_revised.py` | Build presentation/referral_graph_week2_revised.html: week-2 template + the first deck's base style + aggregate data. |
| `render_w3.py` | Build presentation/referral_graph_week3.html: week-3 template + the first deck's base style + aggregate data. |
| `render_w3_revised.py` | Build presentation/referral_graph_week3_revised.html: week-3 template + the first deck's base style + aggregate data. |

## tools/ — Run-list trainer, detached launcher

| Script | What it does |
|---|---|
| `detach.py` | Run a command fully detached from the launching app: double fork + setsid (new session and process group, reparented to launchd), stdin from /dev/null, stdout/stderr appended to a log. |
| `train_runlist.sh` | Train every run in a run list (runlists/*.tsv: name <TAB> inputs <TAB> extra flags) |

## legacy/ — Steps 3–9 (earlier MRR-era pipeline; results in docs/RESULTS.md)

| Script | What it does |
|---|---|
| `analyze.py` | Step 6 analysis: assemble validation tables, paired patient-cluster bootstrap comparisons, seed spread. |
| `assess_bucket.py` | Step 5c: week-vs-day cutoff assessment on the validation subset. |
| `baselines.py` | Step 5a: non-learned baselines with the same full-pool evaluation as the model. |
| `compare_pilots.py` | Score several three-node runs (best.pt) |
| `eval_checkpoint.py` | Evaluate a trained run's best checkpoint with full candidate pools. |
| `failure_audit.py` | Formal failure audit of a trained three-node checkpoint on the validation subset. |
| `familiarity_check.py` | Slide-3 numbers for any three-node run: first-time referrals on the full pool — share of the model's top-10 slots given to specialists this PCP referred to before (vs their share of the pool), and MRR / median rank of th |
| `mips_analysis.py` | STEP9_PROTOCOL §4 analyses. |
| `mips_rerank.py` | Quality-aware recommendation lists (exploratory; not in the frozen Step 9 protocol). |
| `shap_groups.py` | Grouped Shapley attributions for a three-node checkpoint (permutation sampling; Castro et al. |
| `smoke_test.py` | Step 4 pre-training checks (see STEP4_PLAN.md section 6, revised after review). |
| `specialty_match.py` | Frozen secondary metric (STEP7_PROTOCOL §6): specialty match. |
| `train.py` | Train a model. |
