# STEP 12a — Why do the LightGBM models put more top picks on the busiest specialists than the GNN? (FROZEN 2026-10-12)

Status: **FROZEN** after two review rounds.
- Round 1 → draft 2: blocking items B-1 to B-7 and should-fix items S-1 to S-5 adopted.
- Round 2 → "freeze after the listed changes": items R2-1 to R2-5 and all should-change items adopted verbatim. They are marked "R2" below; in a few places they supersede draft-2 text.
- Drafts 1 and 2 are kept in `reviews/STEP12_CONGESTION_PLAN_draft{1,2}_20261012.md`.

Nothing is trained yet.

Scope: validation year (2018) only, no test pass. This is a mechanism study. It is not a model choice and adds no baseline. The pure-tabular LightGBM stays the reference. Every cell reports the four lenses and new-to-PCP. MRR is used only to select configurations.

## 0. Facts this plan rests on (verified by the reviewer)

### Existing models

2018 STEP10 primary cohort, n = 5,550. Load = busiest-10% share of the top pick.

| Model | Inputs | Objective | Fit | Load, base | Load, 30% weight |
|---|---|---|---|---|---|
| LightGBM (2 seeds) | pure tabular (117) | LambdaRank | 0.481 | 50.8% | 46.1% |
| LightGBM (2 seeds) | + history ("primary", 194) | LambdaRank | 0.484 | 51.7% | 45.7% |
| LightGBM (2 seeds) | + graph counts | LambdaRank | 0.511 | 48.6% | 43.2% |
| Logistic regression (1 seed) | primary | pointwise | 0.463 | 45.4% | 40.8% |
| Logistic regression (1 seed) | pure tabular | pointwise | 0.358 | 31.8% | 32.0% |
| No-graph GNN (2 seeds) | history; **no PCP attributes** | softmax | 0.476 | 46.2% | 40.3% |
| GNN (2 seeds) | history | softmax | 0.489 | 46.0% | 40.3% |
| PCP habit | PCP history | — | 0.456 | 47.7% | 45.2% |
| Practice | | | 1 | 29.9% | |

At base, the GNN beats the table on fit (0.489 vs 0.481) and also has lower load. At the 30% weight the table's fit is marginally higher (0.4548 vs 0.4536); matching costs α ≈ 0.004, about 0.1 pt. All contrasts are matched anyway (§2.E).

**Matched-fit check from existing outputs.** The stronger model is mixed with the random scorer down to the weaker one's fit (§2.E):

| Comparison | Setting | Weaker model's load | Stronger model's load at matched fit |
|---|---|---|---|
| Logistic regression (primary) vs LightGBM (primary) | base | 45.4% | 49.3% |
| Logistic regression (primary) vs LightGBM (primary) | 30% weight | 40.8% | 42.9% |
| Logistic regression (tabular) vs LightGBM (tabular) | base | 31.8% | 36.1% |

Part of the smoother models' lower load survives fit matching.

### Inputs and training

- **Specialist attributes are static per specialist.** One value per specialist across all rows; only MIPS is as-of in v5.
- **The six panel counts sum exactly to `beneficiary_count`.** This holds for both the age bands and female + male.
- **Scaling is linear z-scoring without a log**, so the volume columns are heavy-tailed: `beneficiary_count` reaches a z of 35.5, and 86% of specialists sit below 0.
- **The two models see volume differently.** Trees are unaffected by monotone transforms. The GNN passes specialist attributes through Linear → LayerNorm.
- **Training rows** are identical 21-row bundles: target plus 20 uniform alternatives, 1,072,512 rows / 51,072 queries.
  - 55.5% of training queries are repeat referrals; evaluation is first-time only.
  - The uniform alternatives include on average 0.84 same-specialty specialists, and 50% of bundles have none. Evaluation ranks within one specialty, which is what decides load, so that ordering gets little direct training signal under any objective. This is a shared limitation.

### Seed noise in load (pts, base / 30% weight)

| Comparison | Base | 30% weight |
|---|---|---|
| GNN, seed 42 vs seed 1 | 1.24 | 0.45 |
| GNN, seed 42 vs duplicate | 0.95 | 0.10 |
| No-graph GNN, seed 42 vs seed 1 | 1.85 | 1.96 |
| LightGBM tabular, seed 42 vs seed 1 | 0.22 | 0.14 |

**Neural noise is about 2 pts**, so neural cells get 3 seeds.

### History in neural models is NOT settled

Step 11 C1 (training-time PCP-history dropout in the GNN) *raised* load by +2.8 / +1.8 pts. History was tested only in LightGBM. Hence the gate in §2.0.

## 1. Hypotheses

| | Hypothesis |
|---|---|
| **H0 (gate)** | Neural models need history inputs to keep load low. |
| **H1 (model class)** | At a fixed objective and inputs, trees concentrate more than the neural architecture. |
| **H2 (objective)** | LambdaRank concentrates more than group-softmax or pointwise training. |
| **H3 (volume attributes)** | The LightGBM table's *extra* concentration over the neural model comes from the volume block: removing it shrinks the gap. |
| **H4 (capacity)** | Concentration rises with LightGBM capacity at matched fit. |
| **H5 (representation)** | The neural model's lower load comes from z-scored heavy tails plus LayerNorm hiding volume. A log1p volume representation would raise it. |

## 2. Design

### 2.0 Gate cell G (run first)

The GNN architecture with `use_graph=False`, trained with the GNN's exact trainer and loss (sampled softmax), and all history inputs masked:
- `no_patient_history`
- `no_predecessor_gap`
- `no_history_features`
- `no_patient_specialist_prior`
- **new** `no_insurer_history` (pair columns 10–11 in `_mask_pair`)

Inputs then equal pure tabular minus the 11 PCP attribute columns (the no-graph scorer never sees PCP attributes; disclosed). 3 seeds.

**Pre-stated branch (R2-1; matched, at the 30% weight, against the like-for-like no-graph GNN):**

| Quantity | Definition |
|---|---|
| d_G | table@F_G − G (table mixed down to G's fit by §2.E) |
| d_NG | table@F_NG − no-graph GNN |

- **Tabular branch:** d_G ≥ ½·d_NG, and every table/G seed pairing has d > 0.
- **Primary branch:** the PCP-cluster 95% CI of d_G lies entirely below ½·d_NG.
- **Indeterminate** (neither holds): run the primary branch. If time allows, also run the tabular branch, reported as secondary.

On the tabular branch, §2.A runs on tabular inputs. On the primary branch, the mechanism is neural × history, reported as the main finding, and §2.A moves to primary inputs:
- existing `lgbm_primary` and no-graph GNN, plus a 3rd no-graph seed;
- the new objective cells on primary inputs.

All primary-branch cells are labelled **mechanism probes, not baselines**.

**The branch's neural softmax cell** (written "N" below) is G on the tabular branch and the no-graph GNN on the primary branch. All later cells (§2.A, B2, H5) use the branch's inputs.

**Reference gap R (R2-3):**
- R = LightGBM-LambdaRank (branch inputs) − N, matched at the 30% weight.
- It is computed once, with its CI, right after the gate. It is written to `runs/analysis/step12a_decisions.json` before any §2.A–H5 cell is trained.
- Its point estimate is only the scale for the threshold. Its uncertainty is not carried into the tests, because the effects and R share the reference cell.
- Expected on the primary branch: R ≈ 3.3 pts (base ≈ 4.7), so R/2 ≈ 1.7 pts, below the neural noise band of ≈ 2 pts. **The noise-band criterion will then be the binding one** (stated in advance).

### 2.A Model × objective

Same inputs as the branch, same rows, same cohort.

| | LambdaRank | Group-softmax (one-hot over the 21) | Pointwise BCE |
|---|---|---|---|
| **LightGBM** (2 seeds; each objective tuned on the 12-config grid) | existing | new: **custom objective**. Rows are contiguous groups of 21: p = softmax(preds per group), grad = p − y, hess = max(p(1−p), 1e-6). | new: `objective=binary` |
| **Neural: G's architecture and trainer** (3 seeds) | new: Σ_j w_j·softplus(−(s₊ − s_j)), with w_j = \|ΔNDCG₊ⱼ\| at the current (detached) ranks × detached norm log₂(1+Σλ)/Σλ (LightGBM defaults: σ = 1, gain 2^l−1, truncation 21, lambdarank_norm) | = N | new: BCE on the 21 rows |

**PCP-attribute check (R2).** LightGBM sees the 11 PCP attribute columns; the neural cell does not. One descriptive LightGBM LambdaRank run without the PCP columns (1 seed, the reference configuration) is compared with the reference. If it differs by more than the LightGBM noise band, the H1 LightGBM-softmax cell is also trained without PCP columns, and H1 uses that cell.

- LightGBM takes Newton steps and the neural model takes first-order steps (disclosed).
- `rank_xendcg` is run as a descriptive extra only. It is heavily label-smoothed, not softmax.
- A generic one-hot MLP is optional and secondary.

### 2.B Volume attributes (H3)

Columns are mapped through `spec_encoder.numeric_cols`: value at 3i, missing / ambiguous flags at 3i+1 and 3i+2. Medians are computed once (attributes are static), over training-active specialists.

- **B1, inference-time diagnostic (no training).**
  - **Primary:** a share-preserving rescale of beneficiary_count, service_count and the six panel counts to the specialty-median beneficiary level, keeping each specialist's panel shares.
  - **B1+:** also sets rating_count and practice_size to their specialty medians, with flags set consistently.
  - Applied to the GNN, the no-graph GNN, G / N, the LightGBM table and the new cells. This needs `--ablate no_volume`, in `step10_evaluate.py` for neural runs and in `score_baselines11.py` for LightGBM runs.
- **B2, retrain.** Drop beneficiary_count and service_count, and replace the six panel counts with shares.
  - LightGBM LambdaRank: 2 seeds.
  - Neural (N's architecture and inputs): **3 seeds**.
  - The H3 primary contrast is the gap reduction (R2-5, option b): (LightGBM − N) − (LightGBM-B2 − N-B2).
  - Also report the R² of predicting log beneficiaries from the remaining columns, to show how much volume information B2 leaves. Proxies include rating_count, practice_size and MIPS-none.

### 2.C Capacity (H4)

Extended LightGBM LambdaRank grid on the branch's inputs:
- leaves {7, 15, 31, 127, 255}
- rounds {100, 250, 500, 1000, 2000}
- minimum leaf {50, 200}

All configurations are scored on load and fit (one seed per configuration).

**H4, made operational (R2-4):**
- Capacity order = leaves × rounds; ties go to min-leaf 200 as the lower capacity.
- F_ref = the reference LightGBM-LambdaRank cell's fit at the 30% weight.
- Eligible configurations: fit ≥ F_ref.
- Contrast = matched load (mixed down to F_ref) at the highest-capacity eligible configuration minus that at the lowest-capacity eligible configuration.
- Those two endpoint configurations are retrained with 2 seeds each.
- The slope across all configurations is descriptive only.
- **If fewer than 2 configurations are eligible, H4 is "not testable" and drops out of the Holm family.**

### 2.H5 Representation

N's architecture with beneficiary_count, service_count and the six panel counts entering as log1p(raw value), re-standardised with training-active statistics, at batch time (the same pattern as the existing `no_mips` path).
- log1p is the only variant (R2: rank-Gauss dropped).
- 3 seeds, so the neural noise and sign rules apply.

### 2.E Matched fit (primary rule)

Always degrade the stronger model to the weaker one's fit by an exact mixture with the random scorer:
- α = (F_s − F_w) / (F_s − F_random)
- per referral: matched busy1 = (1 − α)·busy1 + α·elig_busy

Fit and busy1 are both measured at the same setting (the 30% weight for primary contrasts); the random scorer is the same at every weight.

PCP-cluster CIs come directly from this, with **α recomputed inside each bootstrap draw (R2-3)** so the CIs include the uncertainty of the matching itself. As a sensitivity, use a softmax-temperature curve over each eligible set, and state that the matched result depends on the degradation family.

### 2.D Measures

| Role | Measure |
|---|---|
| **Primary** | Busiest-10% share of the top pick at the 30% weight |
| Secondary | The same at base; busiest share of top-5 slots; MIPS gain; fit; travel; new-to-PCP |
| **Mechanism (main descriptive)** | Mean within-set percentile of the top pick's training volume, counted two ways: first-time training referrals (matching the busy definition) and all training positives (what the models fitted) |
| Descriptive | Kendall τ-b slope of scores on log volume; distinct specialists recommended as top pick; within-specialty Gini; PCP overlap of top picks; top-1 minus top-2 score gap |

Evaluation-only checks:
1. **Rank-blend sensitivity:** the 30% weight on normalised ranks instead of z-scores. Does the GNN's extra ~1 pt drop from the weight disappear? GNN drops are 5.3–6.1 across runs, the table's 4.6–4.7.
2. **Specialty decomposition of the gap:** referral share × within-specialty Δ.
3. **Output-side residualisation:** remove each model's within-set slope on log volume and recompute load.

## 3. Pre-stated reading rules

**Primary measure:** busiest-10% top-pick share at the 30% weight, matched by §2.E. Direction of interest: lower than the LightGBM-LambdaRank reference.

**Reference gap R:** frozen as a plug-in number in `step12a_decisions.json` after the gate (§2.0). Effect/R is also reported as a descriptive ratio, with a CI from the same bootstrap draws.

**One signed primary contrast per hypothesis (R2-2).** Holm runs over this family of five, using PCP-cluster bootstrap one-sided p-values in the predicted direction.

| | Primary contrast | Predicted sign |
|---|---|---|
| H1 | LightGBM group-softmax − N | > 0 |
| H2 | neural LambdaRank − N | > 0 |
| H3 | (LightGBM − N) − (LightGBM-B2 − N-B2) (gap reduction) | > 0 |
| H4 | matched load, highest − lowest eligible capacity (§2.C) | > 0 (drops out if not testable) |
| H5 | neural log-volume − N | > 0 |

**A hypothesis is supported** when all four hold:
- the effect is in the predicted direction and ≥ R/2;
- Holm-adjusted p < 0.05;
- it is beyond the noise band. For contrasts within one family, the band is that family's largest within-family seed gap. For contrasts spanning LightGBM and neural (H1, H3), it is the larger of the two families' bands (conservative).
- it has the same sign in every seed pairing, i.e. all cross pairs: 2 × 3 = 6 for H1; 3 × 3 = 9 within the neural family (conservative).

**Other outcomes:**
- An effect significant in the opposite direction is reported as **"contradicted"**.
- If nothing is supported: **"not explained by any single factor at this resolution"**. This does not establish an interaction.

**Always reported:**
- The 2 × 2 interaction {LambdaRank, group-softmax} × {LightGBM, neural} as a difference-in-differences with PCP-cluster CI.
- B1 (primary = share-preserving rescale), the PCP-attribute check and the descriptive checks (§2.D), as supporting evidence only.
- For every primary contrast:
  - the selection-overlap sensitivity (excluding the 3,000 selection queries);
  - the four lenses and new-to-PCP.

**Gate outcome:** reported as found, whichever branch is taken (§2.0).

## 4. Information-horizon caveat (to resolve before reading H3)

The volume block is one static value per specialist from one CMS reporting year. If that year is 2018 or later, the "volume" includes validation-period activity, possibly the scored referrals themselves. H3 would then partly be about the information horizon.

**To confirm with the user / professor:** the source year of the specialist Medicare utilisation columns (Tot_Benes / Tot_Srvcs and the panel counts).

## 5. Cost (all local, AC power, no VISION)

| Work | Time |
|---|---|
| Code and tests | 2–3 h |
| LightGBM: softmax, binary, B2, PCP-attribute check, extended capacity grid (≈ 10 boosters, ≈ 50 cohort scorings) + 2 endpoint seeds | ≈ 1.5–2 h |
| Neural: G × 3, LambdaRank × 3, BCE × 3, B2 × 3, H5 × 3 (+ a 3rd no-graph seed on the primary branch; ≈ 15–16 runs × 15–25 min) | ≈ 4–5.5 h |
| Evaluation passes, incl. B1 | 30–45 min |
| **Total** | **≈ 8–10 h** (overnight-friendly; the gate runs first, alone) |

LightGBM and torch jobs run as separate processes, possibly in parallel, monitored by PID liveness.

## 6. Code (backups before edits)

- `referral_gnn/three_node_model.py`:
  - `no_insurer_history` flag;
  - `volume_transform` option (log1p) applied at batch time for H5.
- `referral_gnn/three_node_train.py`: `--loss {softmax, lambdarank, bce}` in `bucket_loss`, recorded in the manifest and checked on resume.
- `scripts/baselines11.py`:
  - `--objective {lambdarank, softmax_custom, binary, rank_xendcg}`;
  - `--volume {keep, shares}` for B2;
  - extended `--grid`.
- `referral_gnn/tabular.py`: FeatureBuilder `volume="shares"` option, with columns mapped by name.
- `scripts/step10_evaluate.py` + `score_baselines11.py`:
  - `--ablate no_volume` / `no_volume_plus` (B1);
  - per-referral top-pick volume percentile (both counts), Kendall slope, score gap;
  - `--blend-rank` sensitivity.
  - Neural cells are scored in `step10_evaluate.py` like GNN runs, never in the LightGBM process.
- New `scripts/step12_congestion.py`: matched-fit mixture, contrasts, Holm, seed-sign check, interaction, decomposition, residualisation, tables.

---

## Amendment A1 (2026-10-13, after the primary verdict; secondary analysis, disclosed as post hoc)

**Context.** The primary verdict (`runs/analysis/step12a/final_summary.json`) is **unchanged and remains the primary result**: not explained by any single factor at this resolution, and H4 contradicted. Two limits motivated this amendment:
- **Noise band.** H1 (+3.9 pts, CI [1.8, 6.5], all seed pairings positive) failed only on the max-seed-gap noise band (5.3 pts). A maximum over seed pairs cannot shrink as seeds are added.
- **B2 was too weak a test.** It left most of the volume information in place: R² of log beneficiaries from the remaining specialist columns was 0.82 overall and 0.68 within specialty.

The user approved running the secondary analysis.

**A1.1 Seed-resampling test (replaces the max-gap noise band in the secondary analysis only).**
- Each bootstrap draw resamples the seeds within every cell (with replacement, same count) and the PCP clusters. α is recomputed per draw.
- A hypothesis is supported (secondary) when all three hold:
  - the effect is in the predicted direction;
  - the effect is ≥ R/2, with R frozen at 4.3 pts;
  - the Holm-adjusted one-sided p < 0.05 over the family {H1, H2, H3′, H5}.
- H4 keeps its primary result.
- More seeds: G gets seeds 2 / 3 / 4, so 6 in total. The other neural cells keep 3 seeds; the bootstrap carries their seed variance. The LightGBM cells keep 2.

**A1.2 Strict volume test, H3′.**
- Volume mode `strict` sets the following to constants (value 0, flags 0): the volume block, the panel block (panel age/sex columns, average panel age, average panel risk), practice_size, rating_count, mips_score and the MIPS extras. It keeps specialty, credentials, years_in_medicare and rating.
- Within-specialty R² of log beneficiaries from what remains: −0.05 (gradient boosting, 5-fold CV over training-active specialists). Overall R² 0.41 comes from specialty alone, and specialty is required because load is defined within specialty.
- The rule for choosing these columns: drop the columns that carry within-specialty volume information, found by greedy CV-R² elimination (log in PROJECT_LOG).
- The MIPS 30% quality weight still uses the yardstick, which is separate from the model inputs.
- New cells:
  - LightGBM LambdaRank strict at the reference configuration, seeds 42 and 1;
  - neural strict (G's inputs and trainer), seeds 42, 1, 7, 2, 3, 4.
- H3′ = (LightGBM − N) − (LightGBM-strict − N-strict). Predicted sign: +.

**Reporting.** The primary and secondary verdicts are shown side by side. The secondary verdict is never presented without the primary one and its post-hoc status.

### A1 revision 1 (reviewer, 2026-10-13 ≈ 22:30, frozen BEFORE any result of the new G seeds or strict cells was examined)

**Secondary decision rule, H1 and H3′.** Both of the following must pass:
- (a) Seed + PCP bootstrap with a small-k correction: each cell's per-cluster seed deviations from the cell mean are scaled by √(k/(k−1)) before resampling.
- (b) The same bootstrap plus a pooled-neural seed term. In every draw, each neural cell gets an extra N(0, s²_pooled/k) on load and on fit. s²_pooled is the within-cell variance of the seed-level means, pooled across all neural cells: G×6, LambdaRank, BCE, shares, log, strict×6; df = Σ(k−1).

H2 and H5 use (a) only; they fail the R/2 bar regardless.

**Commitment.** All 6 G seeds are used whatever their values. No further seeds are added on the basis of results.

**H3′, worded and computed as follows.**
- Wording: "removing every specialist attribute that carries within-specialty volume information (counts, panel composition, average panel age and risk, practice size, rating count, MIPS and its extras; specialty, credentials, tenure and rating kept; within-specialty R² ≈ 0) narrows the LightGBM–neural gap". A positive H3′ locates the gap in this attribute set. It does not separate volume from MIPS or case mix.
- Computation: **one common fit floor per draw**, T = min(F_REF, F_N, F_LS, F_NS), applied to all four cells. The per-gap version is reported as a sensitivity, and both must pass.
- Also reported:
  - at base as well as at the 30% weight;
  - the fit each strict cell reaches;
  - the volume-only B1 (LightGBM −8.3 vs neural −2 to −5) as the descriptive evidence that points at volume specifically.

**Degradation-family sensitivity.** The secondary evaluation uses `--step12` (noise variants). Noise-matched H1 and H3′ are reported. A secondary "supported" never appears without them, nor without:
- the base vs top-5 split;
- the fit gap, framed as "at matched fit by random dilution".

**Holm family.** {H1, H2, H3′, H4, H5}, m = 5, with H4 entering at its primary p. "Contradicted" applies as in the primary. Effect/R CIs and the four lenses plus new-to-PCP are reported for every cell. The summary is never overwritten.

**Rule changes vs the primary, disclosed.**
- The max-gap band is replaced by the seed bootstrap.
- The all-pairings sign rule is dropped.
- H3 is replaced by H3′.
- Extra seeds go only to G and strict, chosen after the primary verdict.

**Strict patch fixes.**
- Use `num_dim` for the extras offset and assert the v5 width.
- Docstring.
- Neural runs named `s12_strict_s{seed}`.
- Disclosure: the rating missing flag, kept by strict, is a weak volume proxy; it was inside the R² ≈ −0.05 check.
