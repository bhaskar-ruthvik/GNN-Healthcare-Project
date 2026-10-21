# STEP 12a — Why do the LightGBM models put more top picks on the busiest specialists than the GNN? (DRAFT 2, 2026-10-12)

Status: DRAFT 2, after reviewer round 1 (verdict "another round, short"). All blocking items B-1 to B-7 and should-fix items S-1 to S-5 are adopted. Draft 1 is in `reviews/STEP12_CONGESTION_PLAN_draft1_20261012.md`. Nothing is trained yet.

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

The GNN beats the table on fit (0.489 vs 0.481) and also has lower load, so the headline gap needs no fit matching.

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
| **H3 (volume attributes)** | The LightGBM table's extra concentration comes from the volume block. |
| **H4 (capacity)** | Concentration rises with LightGBM capacity at matched fit. |
| **H5 (representation)** | The neural model's lower load comes from z-scored heavy tails plus LayerNorm hiding volume. A log1p / rank-Gauss volume representation would raise it. |

## 2. Design

### 2.0 Gate cell G (run first)

The GNN architecture with `use_graph=False`, trained with the GNN's exact trainer and loss (sampled softmax), and all history inputs masked:
- `no_patient_history`
- `no_predecessor_gap`
- `no_history_features`
- `no_patient_specialist_prior`
- **new** `no_insurer_history` (pair columns 10–11 in `_mask_pair`)

Inputs then equal pure tabular minus the 11 PCP attribute columns (the no-graph scorer never sees PCP attributes; disclosed). 3 seeds.

**Reference gap** = LightGBM-LambdaRank-tabular − G, at the 30% weight, matched fit.

Pre-stated branch:
- **G's load is closer to the GNN's (40.3%) than to the table's (46.1%):** tabular inputs suffice. §2.A runs on tabular inputs with G as the neural softmax cell.
- **Otherwise:** the mechanism is neural × history, which is reported as the main finding. §2.A moves to primary inputs (existing `lgbm_primary` and no-graph GNN, plus the new objective cells on primary inputs). These are labelled **mechanism probes, not baselines**.

### 2.A Model × objective

Same inputs as the branch, same rows, same cohort.

| | LambdaRank | Group-softmax (one-hot over the 21) | Pointwise BCE |
|---|---|---|---|
| **LightGBM** (2 seeds; each objective tuned on the 12-config grid) | existing | new: **custom objective**. Rows are contiguous groups of 21: p = softmax(preds per group), grad = p − y, hess = max(p(1−p), 1e-6). | new: `objective=binary` |
| **Neural: G's architecture and trainer** (3 seeds) | new: Σ_j w_j·softplus(−(s₊ − s_j)), with w_j = \|ΔNDCG₊ⱼ\| at the current (detached) ranks × detached norm log₂(1+Σλ)/Σλ (LightGBM defaults: σ = 1, gain 2^l−1, truncation 21, lambdarank_norm) | = G | new: BCE on the 21 rows |

- LightGBM takes Newton steps and the neural model takes first-order steps (disclosed).
- `rank_xendcg` is run as a descriptive extra only. It is heavily label-smoothed, not softmax.
- A generic one-hot MLP is optional and secondary.

### 2.B Volume attributes (H3)

Columns are mapped through `spec_encoder.numeric_cols`: value at 3i, missing / ambiguous flags at 3i+1 and 3i+2. Medians are computed once (attributes are static), over training-active specialists.

- **B1, inference-time diagnostic (no training).**
  - **Primary:** a share-preserving rescale of beneficiary_count, service_count and the six panel counts to the specialty-median beneficiary level, keeping each specialist's panel shares.
  - **B1+:** also sets rating_count and practice_size to their specialty medians, with flags set consistently.
  - Applied to the GNN, the no-graph GNN, G, the LightGBM table and the new cells. This needs `--ablate no_volume`, in `step10_evaluate.py` for neural runs and in `score_baselines11.py` for LightGBM runs.
- **B2, retrain.** Drop beneficiary_count and service_count, and replace the six panel counts with shares.
  - LightGBM LambdaRank: 2 seeds; this is the H3 primary contrast.
  - Neural (G's architecture): 2 seeds, secondary.
  - Also report the R² of predicting log beneficiaries from the remaining columns, to show how much volume information B2 leaves. Proxies include rating_count, practice_size and MIPS-none.

### 2.C Capacity (H4)

Extended LightGBM LambdaRank grid on the branch's inputs:
- leaves {7, 15, 31, 127, 255}
- rounds {100, 250, 500, 1000, 2000}
- minimum leaf {50, 200}

All configurations are scored on load and fit (one seed per configuration). H4 is the slope of matched-fit load on capacity.

### 2.H5 Representation

G's architecture with the volume block log1p-transformed at batch time (the same pattern as the existing `no_mips` path). 2 seeds.

### 2.E Matched fit (primary rule)

Always degrade the stronger model to the weaker one's fit by an exact mixture with the random scorer:
- α = (F_s − F_w) / (F_s − F_random)
- per referral: matched busy1 = (1 − α)·busy1 + α·elig_busy

PCP-cluster CIs come directly from this. As a sensitivity, use a softmax-temperature curve over each eligible set, and state that the matched result depends on the degradation family.

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

**Primary measure:** busiest-10% top-pick share at the 30% weight. Direction of interest: lower than the LightGBM-LambdaRank-tabular reference.

**Reference gap R:** LightGBM-LambdaRank (branch inputs) − G (or − no-graph GNN on the primary branch), at matched fit.

**One primary contrast per hypothesis.** Holm runs over this family of five (H1–H5), using PCP-cluster bootstrap p-values:

| | Primary contrast |
|---|---|
| H1 | LightGBM group-softmax − neural group-softmax (G) |
| H2 | neural LambdaRank − neural group-softmax (G) |
| H3 | LightGBM LambdaRank without volume (B2) − LightGBM LambdaRank |
| H4 | matched-fit load at the largest vs the smallest capacity bracketing the reference fit |
| H5 | neural log-volume − G |

**A hypothesis is supported** when all four hold:
- |effect| ≥ R / 2;
- Holm-adjusted p < 0.05;
- the effect is beyond the family's noise band (the largest within-family seed gap);
- it has the same sign in every seed pairing. The cluster bootstrap ignores training variance, which is why the seed check is needed.

Fit differences are always handled by §2.E matching.

**Always reported:**
- the 2 × 2 interaction (difference-in-differences, with CI);
- B1 (primary = share-preserving rescale) and the descriptive checks, as supporting evidence only;
- the selection-overlap sensitivity (excluding the 3,000 selection queries) for every primary contrast.

**If no hypothesis passes:** "not explained by any single factor at this resolution". This does not establish an interaction.

**Gate outcome:** reported as found, whichever branch is taken.

## 4. Information-horizon caveat (to resolve before reading H3)

The volume block is one static value per specialist from one CMS reporting year. If that year is 2018 or later, the "volume" includes validation-period activity, possibly the scored referrals themselves. H3 would then partly be about the information horizon.

**To confirm with the user / professor:** the source year of the specialist Medicare utilisation columns (Tot_Benes / Tot_Srvcs and the panel counts).

## 5. Cost (all local, AC power, no VISION)

| Work | Time |
|---|---|
| Code and tests | 2–3 h |
| LightGBM: softmax, binary, B2, extended capacity grid | ≈ 60–75 min |
| Neural: G × 3, LambdaRank × 3, BCE × 3, B2 × 2, H5 × 2 (13 runs × 15–25 min) | ≈ 3.5–4.5 h |
| Evaluation passes, incl. B1 | 30–45 min |
| **Total** | **≈ 6–8 h** (overnight-friendly) |

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
