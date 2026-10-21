# STEP 12a — Why does the GNN put fewer top picks on the busiest specialists? (DRAFT 1, 2026-10-12)

Status: draft for review. Nothing is trained yet. Validation year (2018) only. No test pass: this is a mechanism study, not a model choice.

## 0. What existing runs already show

All rows: 2018 STEP10 primary cohort (n = 5,550), seed-averaged. Load = busiest-10% share of the top pick. Sources: `runs/analysis/step11_1_tabular`, `step11_1_pre_B1`, `step11_2`.

| Model | Inputs | Objective | Fit (top-5) | Load, base | Load, 30% quality weight |
|---|---|---|---|---|---|
| LightGBM | pure tabular (117) | LambdaRank | 0.481 | 50.8% | 46.1% |
| LightGBM | + history (primary) | LambdaRank | 0.484 | 51.7% | 45.7% |
| LightGBM | + graph counts | LambdaRank | 0.511 | 48.6% | 43.2% |
| Logistic regression | + history (primary) | pointwise | 0.463 | 45.4% | 40.8% |
| Logistic regression | pure tabular | pointwise | 0.358 | 31.8% | 32.0% |
| GNN without message passing | + history | sampled softmax | 0.476 | 46.2% | 40.3% |
| GNN | + history | sampled softmax | 0.489 | 46.0% | 40.3% |
| PCP habit | PCP history | — | 0.456 | 47.7% | 45.2% |
| Practice (actual referrals) | | | 1 | 29.9% | |

Facts that follow:
1. **Message passing does not cause the gap.** The GNN and the no-graph GNN have equal load.
2. **Referral-history inputs do not remove it in a GBDT.** LightGBM with history is as concentrated as pure tabular.
3. **The high-load models are the LightGBM LambdaRank models.** Logistic regression on the same "primary" inputs sits at the GNN's level, at slightly lower fit.

So the question is better phrased as: **why does LightGBM concentrate more than the neural and linear models?**

Group deep-dive (2026-10-12): a weak PCP-history gradient exists (−1.6 pts per log-unit of PCP history, with the quality weight only). Specialty mix and busy availability do not explain the group differences.

## 1. Hypotheses

- **H1, model class.** Trees carve sharp regions on specialist volume attributes, so the top-scored specialist within a specialty is more often the high-volume one. Smooth models (MLP, linear, GNN) spread scores more.
- **H2, objective.** LambdaRank (pairwise, NDCG-weighted, truncation 21) rewards getting the single target to the top. Softmax cross-entropy (GNN) and pointwise logistic penalise differently. Either may favour popular specialists.
- **H3, volume attributes.** The specialist snapshot carries volume counts. Both the table and the GNN receive them, but the trees may use them more:
  - Medicare beneficiary_count and service_count;
  - six panel counts: age bands and female/male;
  - rating_count and practice_size.

  LightGBM gain importance is led by beneficiary count.
- **H4, capacity.** The selected LightGBM (127 leaves, 1,000 rounds) fits training popularity more closely. Concentration may scale with capacity at equal fit.
- **Not tested here (already answered):** message passing, PCP-history inputs.

## 2. Design

All new models use **identical training rows**: the GNN's 21-row bundles from `training_matrix`, with the target plus 20 uniform alternatives and weekly cutoffs. They are scored on the identical STEP10 cohort and eligible sets, with 2 seeds each. Checkpoint and hyperparameter selection use the same criterion as every existing model (validation selection subset, first-time, full pool). It is used only to pick a configuration, never to judge.

### A. Model × objective on the pure tabular features (117 columns)

The core 2 × 2, plus one pointwise cell:

| | LambdaRank | Softmax cross-entropy over the 21 | Pointwise logistic |
|---|---|---|---|
| **LightGBM** | existing (`lgbm_tabular`) | new: `objective=rank_xendcg` (listwise softmax-style) | new: `objective=binary` |
| **MLP** (new; same 117 columns: numerics standardised, categoricals one-hot; 2 × 256 ReLU, dropout 0.1) | new: pairwise logistic loss weighted by \|ΔNDCG\| (the LambdaRank gradient) | new: the GNN's exact loss | (existing logistic regression is the linear analogue) |

- Each new LightGBM objective gets the same 12-config grid as the existing table (leaves {31,127} × min leaf {50,200} × rounds {250,500,1000}).
- MLP: early stopping as for the GNN; learning rate {1e-3, 3e-4}.

Reading:
- Row difference (MLP − LightGBM, same objective) → H1.
- Column difference (same model) → H2.

### B. Volume attributes (H3)

- **B1, inference-time neutralisation (diagnostic, no training).** For every candidate, set the volume block (beneficiary_count, service_count, the six panel counts) to its specialty median among training-active specialists. Re-score the GNN, the no-graph GNN, the LightGBM table and the new cells, and record the change in load and fit. Variant B1+ also neutralises rating_count and practice_size.
- **B2, retrain without the volume block.** LightGBM LambdaRank and MLP softmax on the 117 columns minus the volume block, 2 seeds each.

Reading: if removing volume brings LightGBM to the MLP/GNN load level at a fit loss under 1.5 pts, H3 explains the gap.

### C. Capacity (H4)

Score all 12 grid configurations of the existing LightGBM tabular tuning, plus the best of each new objective, on load and fit. This traces a load–fit curve per model family. Comparing the MLP and GNN points against that curve answers: is the GNN below LightGBM at *matched fit*, or just less faithful?

### D. Measures (every cell)

- **Lenses:**
  - load: busiest-10% share of the top pick, and of the top-5 slots;
  - fit: top-5 match;
  - quality: MIPS gain at the 30% weight;
  - travel;
  - new-to-PCP share.

  Each at base and at the 30% weight.
- **Popularity slope:** per referral, the Spearman correlation between a model's scores over the eligible set and log(1 + the specialist's first-time training referrals), averaged over referrals. Reference: the chosen specialist's within-set percentile of training volume, i.e. how popularity-driven practice itself is.
- **Spread:** distinct specialists recommended as top pick, within-specialty Gini of top picks, and overlap of top picks between PCPs.
- **Score shape:** top-1 minus top-2 z-score gap. This bears on why the 30% quality weight moves the GNN's picks more (its load falls 5.7 pts with the weight vs 4.7 for the table).

## 3. Pre-stated reading rules

- **Noise band** per measure: the larger of the seed gap and the GNN s42 − duplicate gap. For load, ≈ 1 pt.
- **Gap to explain:** LightGBM LambdaRank vs GNN, 4.8 pts (base) and 5.8 pts (30% weight).
- **A factor explains the gap** if switching it alone (A row/column, B2, or the matched-fit comparison in C) moves load by ≥ half the gap, beyond the noise band, with fit within ±1.5 pts. PCP-cluster 95% CIs, Holm across the factors tested (H1–H4).
- **More than one factor** passing is a valid result (they may add up). **None** passing is a valid result: the gap is then an interaction, reported as such.
- Results are descriptive about mechanism. Nothing here changes the frozen comparisons or adds a baseline. The pure-tabular LightGBM remains the reference.

## 4. Cost and where it runs

- Local, AC power. LightGBM grids ≈ 10 min per objective. MLP ≈ 10–20 min per run (6 runs).
- Each evaluation pass ≈ 3–5 min.
- B1 needs a `no_volume` ablation in the GNN scorer (same mechanism as the existing `no_mips` ablation) and in `score_baselines11.py`.
- **Total ≈ 3 h. No VISION needed.**

## 5. Code to write (backups before edits)

- `referral_gnn/tabular.py`:
  - `TabMLP` scorer and its loader;
  - a `drop_volume` option for FeatureBuilder (removes the volume columns by name);
  - objective passthrough.
- `scripts/baselines11.py`: `--objective {lambdarank, rank_xendcg, binary}` and `--drop-volume`.
- New `scripts/train_tabmlp.py`, with the `torch` / LightGBM process separation kept: the MLP never imports lightgbm.
- `scripts/step10_evaluate.py` + `score_baselines11.py`:
  - `--ablate no_volume` for model and baseline runs;
  - save per-referral popularity slope and score-gap columns.
- New `scripts/step12_congestion.py`: the tables, the C curves and the reading rules above.

## 6. Open points for review

1. Is `rank_xendcg` an acceptable stand-in for softmax cross-entropy in LightGBM? The alternative is a custom softmax objective via the LightGBM Python API (gradient/hessian per group).
2. Should the MLP use the GNN's exact encoders (embedding categoricals) rather than one-hot, to be closer to the no-graph GNN? Either way it differs from the no-graph GNN in having no history inputs.
3. Is matched fit via the LightGBM capacity curve sound, or should matching use a fit-reducing perturbation applied equally to every model (e.g. a temperature on a softmax over the eligible set, sampled lists)?
