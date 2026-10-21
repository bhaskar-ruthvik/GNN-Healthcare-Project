# STEP 12d — Heavier hybrids (DRAFT 1, 2026-10-14; code written and smoke-tested, nothing run for real)

## Question

The steered equal blend (STEP12c) showed that the GNN's and the table's *errors* are complementary: the blend is more accurate, and the surplus can be spent on lower load and more new connections. That combination happens at score level, after training.

This step asks whether combining the two **inside training** gives a better trade-off than combining afterwards, and what each direction reveals about what the graph adds. Three ways of letting one model learn from the other:

| | Hybrid | How | What it tells us |
|---|---|---|---|
| A | **Table → GNN** | GNN score + β·table score, with β learned; the GNN learns what the table misses | Does the graph add information beyond the table? A learned β ≈ 1 with a non-trivial GNN part means yes |
| B | **GNN → table (stacking)** | LightGBM on the 117 pure-tabular columns + the GNN's score | Can a tree use the GNN's judgement as one input among the claim attributes? |
| C | **Dual encoder** | One model: GNN encoder + a tabular tower over the 117 pure-tabular columns, end to end | Does joint training find a representation neither model finds alone? |

## Leakage controls

**Out-of-fold scores.** For A and B, the score a model *trains on* never comes from a model that saw that row.
- The training queries are cut into **K = 3 contiguous time folds**.
- **A:** the table scores on training rows come from LightGBM fitted on the other two folds (`scripts/step12d_table_oof.py`).
- **B:** the GNN scores on training rows come from GNNs retrained without that fold (`--exclude-fold k`; `scripts/step12d_gnn_oof.py`).
- Folds include later time: this is standard stacking. A row's own label is never seen by the model that scores it (disclosed).
- At evaluation the **full** models' scores are used: the existing table runs and the final GNNs.

**Process separation.** torch and LightGBM never run in one process. Scores cross between them as `.npz` files with lookups that are asserted complete.

## Seeds

Three, matching STEP12c's pairings: (GNN 42, table 42), (1, 1), (dup / 7, 7).
- **B's third pairing:** the full GNN at evaluation is `s42_dup`; the fold models are seed 7 (disclosed).
- **C:** the categorical codes in the tabular row enter the tower as standardised numbers. This is crude, but the GNN encoder already embeds specialty and credentials (disclosed).

## Judging (no pass/fail; the same frontier framework as STEP12c)

1. Each hybrid's (fit, busiest-10% top pick and top-5, new-to-PCP top pick and top-5, MIPS gain, good-new, travel) at the 30% weight, against the GNN and the table, with paired PCP-cluster CIs.
2. **Main comparison:** with the same knowable steering (a·new − b·busy) on the same grid, does the hybrid's (load, newness) frontier at fit ≥ 0.45 lie beyond the **steered equal blend's**?
   - If yes, combining inside training adds something.
   - If no, score-level combination is enough. That is an equally useful answer.
3. **Descriptive:**
   - A's learned β and the share of score variance from the GNN part;
   - B's LightGBM gain importance of `gnn_score`;
   - C's ablation (tab tower zeroed at scoring).
4. Validation 2018 only. Any 2019 pass needs a pre-registered setting, as in STEP12c.

## Code (all smoke-tested 2026-10-14)

- **`three_node_model.py`:** `table_offset` (β init 1) and `tab_dim` (tabular tower; standardisation buffers saved with the model).
- **`three_node_train.py`:** extras gathered per 21-row bundle. Validation uses an extras-aware scorer; the offset model validates on the first-time selection queries, whose full-pool table scores are precomputed. The device probe is deferred until the extras exist.
- **`scripts/three_node_train.py`:** `--table-offset NPZ --offset-table-run RUN`, `--tab-tower`, `--exclude-fold k --n-folds K` (all in the manifest; resume-checked).
- **`referral_gnn/heavy_hybrid.py`** (`ExtraModelScorer`) and **`referral_gnn/score_lookup.py`** (torch-free `ScoreLookup`, `time_folds`, `fold_of_tc_rows`, `dump_to_arrays`).
- **`tabular.FeatureBuilder(extra_npz=…)`:** a `gnn_score` column (out-of-fold for training rows, lookup elsewhere). `step12_lgbm.py --extra-npz`; arm `…_stack`.
- **`step10_evaluate.py`:** extras-aware scoring for A/C, with the offset taken from the full table run in the same evaluation. Arms `model_s12d_table_offset`, `model_s12d_dual_encoder`, `model_s12d_fold{k}`.
- **Smoke tests:**
  - table out-of-fold scores: 3 folds + full selection lookup in 94 s;
  - 1-epoch offset, dual and 3 fold GNNs on MPS;
  - GNN out-of-fold scores on 200 queries per fold + lookup, 21 s;
  - tiny stacked LightGBM;
  - evaluator over all of them with separate arms.

## Runs

| Item | Count | Local | VISION |
|---|---|---|---|
| Table out-of-fold scores (CPU) | 3 seeds | ≈ 15 min | — (runs locally anyway) |
| A: offset GNN | 3 | ≈ 3 h (≈ 1 h each) | parallel |
| C: dual-encoder GNN | 3 | ≈ 3 h | parallel |
| B: GNN fold models (2/3 of the data each) | 9 | ≈ 6 h (≈ 40 min each) | parallel |
| B: GNN out-of-fold scores + stacked LightGBM | 3 + 3 | ≈ 30 min | — |
| Evaluation + candidate dumps + frontier | — | ≈ 45 min | — |
| **Total** | **15 GNN runs** | **≈ 13 h** in ≈ 1-h resumable sittings | **≈ 1–1.5 h wall** for the 15 GNN runs, plus ≈ 1.5 h local for the rest |

Run lists: `grace/runlists/step12d_folds.tsv` (B) and `grace/runlists/step12d_offset_dual.tsv` (A, C).
