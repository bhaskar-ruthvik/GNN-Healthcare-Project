# STEP13 — Where standard models fall short, where graph models add value, and what combining them buys

Status: **FROZEN (plan level), DRAFT 5** (2026-10-21; frozen at the user's instruction). Any later change is a dated amendment below, never an edit. Resolves the draft-3 audit, the internal final check of
draft 4 (B1, B2, m1–m5) and the external draft-4 audit (`reviews/STEP13_GRAPH_VALUE_AUDIT_DRAFT4_20261021.md`, items 1–3 and the
execution clarifications). The two draft-4 reviews converged on the same three gaps.
Earlier drafts and audits are in `reviews/`.

Phase 1 (claim units H1a, H1b, H2-low-history, H2-thin-network, H2-rare-condition, H3) is frozen in two steps:
1. **Design manifest:** this document plus §6, before any tuning.
2. **Resolved manifest:** configurations and checkpoints appended after tuning (selection evaluations on the 2018 selection
   sample belong to tuning), before the phase-1 outcome evaluation.

Phase 2 (H4–H8) is a set of sketches; each needs its own amendment and review.

## 0. Framing, scope and wording

The study asks *whether and where* graph models help a referral recommender, and what combining them with a table buys.
"No advantage" and "advantage only under a defined condition" are valid answers.

Wording rules:
- **Load** is offline concentration of recommendations among the historically busiest specialists. It is not congestion,
  waiting time or capacity.
- **MIPS** is a quality proxy.
- **2018** is development data; **2019** is a previously examined temporal replication set.
- **Compatibility** (insurer seen, category treated, shared hospital) is historical compatibility.

Prior evidence is as in draft 3 §0 (unchanged; sources cited there):
- the pure table is level with the GNN on fit;
- the no-graph vs GNN contrast mixed in PCP access;
- the concentration mechanism was "no single factor";
- graph-feature tables are strong;
- new picks' insurer compatibility is 76.5% vs 71–72%;
- the steered blend's complementarity replicates.

## 1. Analysis contract

**Code and provenance.**
- The reviewed STEP13 implementation is *committed* to `referral-gnn/` before any run. The manifest records the commit, the
  tree hash, and the SHA-256 of every CLI script and the library. Data workspace: `Implementation/`.
- **Existing checkpoints** (Graph, G, Table) keep their original training provenance (run folder, `code_hash`, inputs hash).
  The manifest records that provenance *and* the commit used to load and evaluate them. A backwards-compatibility check
  (re-scoring reproduces the STEP12 dumps to within 1e-6) runs before evaluation. A new commit does not make old checkpoints
  products of that commit.
- **Design manifest** (`runs/analysis/step13/design_manifest.json`), frozen before tuning: cells, seeds, feature allowlists
  (§6), tuning grids, slice boundaries (§3), matching grid (§4), test definitions (§2).
- **Resolved manifest:** per cell, the run folders, checkpoints, chosen configurations and inputs hashes, plus the complete
  tuning ledger. Appended after tuning, before the phase-1 outcome evaluation. The design record is never edited.
- **Also in the design manifest:**
  - the numeric B₂ and the generated column lists;
  - the candidate-ranking tie rule (stable sort on the eligible-set order, as in `candidates.Dump.top`);
  - unknown categories (own category, as in training);
  - the zero-history baseline (§6).
- **Cost estimates** (§7) are re-checked after the first complete training run and the actuals logged. A cost overrun does not
  change the design.

**Seeds and the bootstrap.**
- Distinct seeds 42 / 1 / 7, *paired by seed ID across models*. Graph `s42_dup` and the G A1 seeds 2 / 3 / 4 are sensitivities.
- Two-level paired bootstrap, 2,000 draws per draw:
  1. resample the seed IDs (3, with replacement), applied to every model;
  2. resample PCPs (with replacement), with the *same* PCP draw for all models and all contrasts;
  3. preserve slice membership as observed;
  4. recompute every contrast, ratio, joint statistic and matching (§4) within the draw.
- Percentile 95% CIs. One-sided p = (1 + #draws with the contrast ≤ 0) / (1 + draws), oriented to the predicted direction.
- **Undefined draws** (H3 tolerance missed, a zero compatibility denominator, or an empty bootstrap slice) are never dropped,
  redrawn or set to zero:
  - **for the p-value,** each such draw counts as "contrast ≤ 0", so it can only weaken the claim;
  - **for the CI,** an H3 draw uses the nearest-grid contrast, tagged with its achieved newness, so the interval is for the
    nearest-grid policy within tolerance. A draw with no defined contrast (zero denominator, empty slice) is left out of the
    interval, and the share left out is reported;
  - **gates:** if more than 5% of draws are undefined for any component, the unit is inconclusive.
- Also reported: seed-level effects, and PCP-only intervals conditional on the trained models. No new variance rule after seeing results.

**Results first, then tests.**
- Every cell's native four lenses plus newness, on 2018, before any matching.
- No fit-matching where fit is the outcome.

**Verdicts.** Two separate axes.
- **Directional (confirmatory).** Each claim unit has a set of components. The unit's p-value is the **maximum** of its
  components' one-sided p-values (intersection–union). Holm is applied across the 6 claim units at α = 0.05.
  **Supported** = Holm-adjusted p < 0.05.
- **Practical equivalence (descriptive, unadjusted).** Per component, TOST: the 90% CI lies within ±δ, with δ = 2 points on
  the native scale (on the difference-in-differences scale for H2 interactions). It is reported beside the directional verdict.
  A component can be "directionally positive" and "practically equivalent" at once; both are reported, and the directional
  verdict decides the claim.
- **Opposite-direction findings** are descriptive (not multiplicity-controlled) and reported in full.
- **Untestable units.** A unit below the size floor on 2018 enters Holm with p = 1, so m stays 6.
- **Seed direction (m2).** With 3 seed IDs the seed bootstrap is anti-conservative (only 10 distinct resamples). So
  "supported" also requires each unit's seed-level effects (per seed, PCP-only) to have the predicted sign for all 3 seeds and
  every component. This is fixed now, not after results.
- **Seed pairing across families** (GNN s42 with LightGBM s42) is a convention, not shared randomness (disclosed).
- Many units may be not supported and practically equivalent or inconclusive. No rescue runs follow.

**Tuning budgets (disclosed; results are conditional on them).**
- GNN-family cells use the final-model settings tuned in Steps 7–10. Checkpoints are selected on the 2018 selection sample by
  first-time MRR (training-time selection only).
- Each LightGBM family is tuned on the 12-configuration grid by the same selection rule. Tune mode trains seeds 42 / 1;
  seed 7 is trained at the chosen configuration.
- Each MLP family is tuned on hidden {128, 256} × dropout {0.1, 0.3}, selected on seed 42. The chosen configuration is then
  trained at seeds 1 and 7. Specification in §6.
- The search histories are unequal. A weak MLP result alone will not be read as a general GNN advantage.

**2019.** One pass for phase 1, right after the 2018 verdicts are recorded, with every claim unit and component in one
decision entry. Replication for a unit supported on 2018 means every component's 2019 point estimate has the predicted sign.
For H2 that is the absolute and interaction effects against all three comparators. An H2 slice that falls below the minimum
size in 2019 is "not replicable (insufficient size)". Because the low-history and thin-network splits were already seen on
2018, those two units count as headline claims only if they replicate. Otherwise they are reported as "supported on 2018, not
replicated". This sign criterion is called **directional replication**. Matching coefficients are frozen from 2018 (§4), so
2019 matching can drift. The achieved 2019 newness gap (H3) and the 2019 CIs are reported beside the sign.

## 2. Claim units and components

### Cells
| Cell | Description | Status |
|---|---|---|
| Graph | v5 GNN, message passing | s42, s1 exist; s7 new |
| NG+P | no message passing; scorer also reads the PCP encoding `enc_pcp[enc_idx]` | new ×3 |
| G | no message passing, no PCP encoding, explicit history removed | exists s42 / s1 / s7 |
| GNN-nohist | Graph with explicit history removed (STEP12a gate ablations) | new ×3 |
| NG+P-nohist | NG+P with explicit history removed | new ×3 |
| Table | pure-tabular LightGBM | s42 / s1 / s7 exist |
| Flat-RF | receptive-field-matched fixed aggregates (§6), LightGBM | new ×3 |
| MLP-RF | MLP on the Flat-RF row | new ×3 |
| Info-table-nohist | practical superset without explicit pair or patient history (§6), LightGBM | new ×3 |
| Info-table | practical superset with all GNN inputs + aggregates incl. A3, LightGBM | new ×3 |
| MLP-info | MLP on the Info-table row | new ×3 |
| Ins-match control | Info-table-nohist + one column: the query insurer's share in the specialist's historical insurer profile (§4) | new ×3 |

"Explicit history removed" means no explicit pair- or patient-history features. Graph edges, and in the tables the graph
counts, still carry graph-derived history.

### H1a — Recovery through relational inputs
Primary: native fit at q30. Components, each one-sided, predicted > 0:
1. Prerequisite: F_Graph − F_NG+P-nohist > 0. If it fails, the unit fails, reported as "recovery fraction not established".
2. F_GNN-nohist − F_NG+P-nohist.
3. Joint half-gap: (F_GNN-nohist − F_NG+P-nohist) − 0.5·(F_Graph − F_NG+P-nohist).

G is reported beside (what PCP access adds).

Limitation, stated in advance: GNN vs NG+P also changes depth, parameter count and nonlinearity. A pass supports "the GNN
recovers relational signal that the stated non-graph controls do not", not a causal effect of connectivity alone. An
architecture-preserving edge intervention is moved to phase 2 (it needs its own rewiring rule).

### H1b — Learned propagation vs fixed aggregation
Primary: native fit at q30. Components: GNN-nohist − X > 0 for each X ∈ {Flat-RF, MLP-RF, Table, Info-table-nohist}.
- Flat-RF and MLP-RF are the *architectural* comparators: the same receptive field and attributes (§6).
- Info-table-nohist is the *practical* comparator, a superset with extra paths and counts.
- Full Graph and full Info-table / MLP-info are references, not components.
- Conclusion if supported: learned propagation beats these frozen summaries and a neural model on them, given the stated budgets.

### H2 — Weak-history slices (three claim units)
Population: the STEP10 2018 cohort; eligibility excludes never-seen specialists, so this is weak history, not cold start.
Slice boundaries are fixed in §3. Minimum size: ≥ 100 referrals and ≥ 10 PCPs, else the unit is not testable.

For each slice S and comparator C ∈ {NG+P, Info-table, MLP-info} (all with full history), two components:
- (i) **absolute:** F_Graph − F_C within S > 0;
- (ii) **interaction:** (F_Graph − F_C)_S − (F_Graph − F_C)_rest > 0.

Six components per unit, at native fit (q30).

Descriptive only:
- the label-defined low-experience-specialist slice (MUP volume caveat; not counted toward any headline);
- candidate-side exposure, with availability beside it;
- load, newness and MIPS per slice;
- the no-prior-encounter patients, as a heterogeneity contrast (not a strict negative control: PCP and specialist context still reach them).

Already seen on 2018: the PCP-history and thin-network splits. The graph's load advantage was smaller for low-history PCPs.

### H3 — Conditional historical insurer compatibility of new picks (no-explicit-history condition)
**Claim, narrowed per the audit:** "among models trained without explicit pairwise insurer-match features, the GNN's new picks
are more often historically insurer-compatible at matched newness". It is a *conditional compatibility* endpoint. Fit
differences are allowed and reported. It does not claim better recommendations overall, and does not claim that fixed
aggregation cannot encode the relation.

- **Primary:** the slot-weighted rate = Σ new slots with the insurer seen ÷ Σ new slots. Slots whose query insurer is unknown
  are excluded from both. The numerator, the denominator, and referrals with zero new picks are reported.
- **Components:** GNN-nohist − X > 0 for X ∈ {NG+P-nohist, Flat-RF, MLP-RF, Info-table-nohist}, at matched newness (§4).
- **Strong reference (descriptive, pre-stated reading):** the Ins-match control at the same matching. If the GNN-nohist also
  beats it, that is reported as compatibility beyond the accessible indicator. If not, the unit's claim stays the narrowed one.
- **Composition sensitivity (descriptive):** the rate and newness by insurer group.

## 3. Slice boundaries (written to the design manifest before tuning)

1. **Low-history PCPs:** PCP prior referrals (strictly before the referral day) ≤ 951. The cut is the 2018 cohort's lowest-
   quintile bound (STEP12 groups), disclosed, and reused unchanged for 2019. Ties at 951 are in the slice; missing → excluded
   from both slice and rest.
2. **Thin-network PCPs:** distinct specialists referred to before the referral day ≤ B₂, where B₂ is the 2018 cohort's lower-
   tertile bound computed now from that attribute (not an outcome) and written to the manifest. Reused for 2019; ties and
   missing values handled as in slice 1.
3. **Rare conditions:** the 33 source disease categories are ranked by training-year (< 2018-01-01) encounter count. The 11
   least frequent are "rare"; ties at the boundary go by category code. An encounter is in the slice if any of its categories
   is rare. Encounters with an unknown category, or none, are excluded from both slice and rest.

## 4. H3 matching (executable)

- **Ranking:** score = 0.7·z(model score) + 0.3·z(as-of MIPS) + a·[new to PCP]. z is taken within each referral's eligible
  set (`referral_gnn/candidates.py` arithmetic with b = 0).
- **Grid:** a ∈ {−0.6, −0.4, −0.3, −0.2, −0.15, −0.1, −0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6}.
- **Target:** 2018 new-to-PCP share = 35.3%.
- **Choice:** per model and seed, the a whose newness is closest to the target. Ties go to the smaller |a|, then the smaller a.
  Re-matched within every bootstrap draw.
- **Tolerance:** ±1 point.
  - **Point estimate:** each model's per-seed 2018 point estimate must itself attain the tolerance. If any does not, the unit is
    inconclusive, whatever the bootstrap shows.
  - **Draws:** a draw in which any model misses the tolerance is undefined and handled as in §1 (≤ 0 for the p-value; nearest-grid
    contrast, tagged, for the CI).
  - **Attainment gate:** if any model attains the tolerance in < 95% of draws, the unit is inconclusive.
- **2019:** the a values are frozen from 2018 (a temporal-transfer estimand). The achieved 2019 newness is reported.
- **Ins-match control** is matched the same way. Its insurer-share column uses the same weekly graph cutoff as the GNN and Flat
  profiles (not the strictly-before-the-day lookup), so it gets no extra days. Unknown query insurer → share missing, and the
  slot is excluded as above.

## 5. Phase 2 sketches (not frozen)

- **H4, data efficiency.** Choose the estimand first: label efficiency, or history availability. Keep the evaluation
  definitions fixed (assert the STEP10 cohort hash). Run a perturbation check. Snapshot holdouts are not interchangeable with
  a fixed 2018 evaluation.
- **H5, future co-referral association.** The training end precedes the label window. Representations: pre- vs post-HGT,
  NG+P, Flat-RF, and an information-matched pair predictor. Uncertainty clustered by specialist and PCP. Replacements are
  constrained to eligibility and compatibility.
- **H6, explanations.** Feature-perturbation attributions only: Integrated Gradients over embedding vectors for categoricals,
  with a training-background baseline; TreeSHAP with a training background and grouped dependent features. Any edge-mask
  method is a separately specified add-on. Time a 20-referral benchmark first.
- **H7, load-aware training.** Pending a defined objective (the "echelon" clarification), load horizon, weight scope and a
  common budget.
- **H8, combination.** Triggered by measurable referral-level complementarity, with the target fixed in advance. Rolling-origin
  base predictions for a deployment-faithful gate, or disclosed transductive cross-fitting (the STEP12d out-of-fold scores
  are the latter).

## 6. Feature allowlists and the MLP specification (frozen with the design manifest)

**Flat-RF**, for the architectural comparison: only what a 2-layer pass around the query encounter and the candidate can
reach, with the same attributes.
- **A1 (specialist ← past encounters):** means of the encounter numerics (context, severity, diagnosis flags), the
  disease-category distribution, the encounter-categorical one-hot means (insurer, ACO, gender, race), and the means of the
  16 sinusoidal time-encoding components of (cutoff − encounter day), as in `referral_gnn/data.py::time_encoding` (the GNN's own
  encounter-time input). No log-age or recency-weighted summaries: they are not what the GNN sees.
- **B1 (query ← PCP ← the PCP's past encounters):** means of the same encounter attributes as A1 (numerics, categories,
  categorical one-hots, time-encoding components). This is the two-hop path the GNN's query encounter receives through its PCP
  (`rev_has_encounter` edges into the PCP, then `has_encounter` into the query).
- **Zero history (A1, A2, B1):** denominator = the number of qualifying encounters. With none, the means are 0 and a
  `*_nohist` flag is 1.
- **A2 (specialist ← encounters ← PCPs):** means of PCP numerics plus PCP categorical one-hots and missing flags.
- The query's own PCP snapshot is already among the 117 tabular columns (not duplicated).
- **Matching terms** as in FlatGraph (category cosine, age difference).
- **No A3,** and no candidate−A3 terms.
- Plus the 117 tabular columns. Aggregation weights: unweighted means over encounters strictly before the weekly cutoff.

**Info-table-nohist** (practical superset):
- tabular 117;
- graph counts (specialist, PCP and co-referral counts, *minus* the patient-derived co-referral columns `g_coref_*`);
- Flat (incl. A3 and candidate−A3) with the A1 / A2 extensions and B1 (so it remains a superset of Flat-RF);
- distance features and shared affiliation (+ unknown flags) are already among the 117 tabular columns (not duplicated);
- *no* PCP–specialist referral count, recency or recency-missing columns, patient–specialist prior, insurer history or
  insurer-unknown flag, patient-history block, or predecessor gap.

**Info-table:** Info-table-nohist + all the excluded explicit history features.

**Ins-match control:** Info-table-nohist + the insurer-share column (§4).

The exact column names and counts are generated by code and written to the design manifest. They are checked against
hand-calculated means and counts on sampled specialist-days and query-PCP days (B1 included), with strict pre-cutoff access
asserted. The check also asserts that Flat-RF ⊂ Info-table-nohist ⊂ Info-table.

**MLP specification.**
- **Inputs:** the FeatureBuilder row. Categoricals one-hot; numerics standardised with training-only scalers; missing flags kept.
- **Training:** the same 21-row bundles (chosen specialist + 20 uniform alternatives) and softmax loss as the GNN.
- **Optimiser:** Adam, learning rate 1e-3, weight decay 1e-5, LR halving after 2 non-improving epochs, early stopping after 5,
  at most 100 epochs.
- **Checkpoint:** best first-time MRR on the 2018 selection sample (as for the GNN).
- **Architecture:** 2 hidden layers of the chosen width, ReLU, the chosen dropout.

## 7. Phase-1 work and cost
- **New code, reviewed and committed before running:**
  - the NG+P flag (`pcp_in_scorer`). The PCP index reaches `score()` identically in training and evaluation: an `enc_pcp`
    model buffer, or the trainer passing `pc` as the evaluators already do. Smoke test: G's scores are unchanged when the flag
    is off;
  - evaluator arm names derived from the manifest flags (`pcp_in_scorer`, the ablation set), with an assert that no two cells
    share an arm (GNN-nohist must not merge into the GNN arm, as happened once in STEP12d);
  - FeatureBuilder sets `flat_rf`, `full_nohist`, `full`, `insmatch`;
  - `step12_lgbm.py` accepting them;
  - the MLP trainer;
  - slice attributes;
  - the H3 matching script;
  - the bootstrap and verdict script;
  - the manifest writers.
- **Runs:**
  - 10 GNN-trainer runs, ≈ 10 h on MPS;
  - LightGBM: 4 families (Flat-RF, Info-table-nohist, Info-table, Ins-match) × (12-configuration tune + seed 7), ≈ 3 h CPU;
  - MLPs: 2 families × (4 configurations + 2 seeds) = 12 runs, ≈ 2–3 h;
  - evaluation and verdicts, ≈ 1 h.
