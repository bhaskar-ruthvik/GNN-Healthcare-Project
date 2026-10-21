# STEP 12b — Hybrid GNN + table models (DRAFT 1, 2026-10-14)

## Goal

The meeting on 2026-10-09 asked for hybrids that combine each model's upside:
- **GNN:** fewer top picks on the busiest specialists, at equal quality.
- **Table:** about 10 pts more new-to-PCP picks and slightly shorter trips.

Scope: validation 2018 only (the STEP10 primary cohort, n = 5,550). Judging uses the four lenses, at base and at the 30% quality weight. MRR is used only for selection. The pure-tabular LightGBM stays the baseline; hybrids are new models, not baselines.

## Reference models (existing, not retrained)

| Model | Seeds | Runs |
|---|---|---|
| GNN (final v5 unmasked) | 42, 1 | `20260929T213930_…_s42`, `20260930T094407_…_s1` |
| Table (LightGBM, pure tabular) | 42, 1 | `20261007T195708_…`, `20261007T195853_…` |

## Hybrids

### H0 — score-level combinations (evaluation only, no training)

`step10_evaluate.py --hybrid GNN:TABLE`, one per seed pairing: (42, 42) and (1, 1).

- **Blend:** (1 − w)·z(GNN) + w·z(table), with w ∈ {0.25, 0.5, 0.75}.
- **Reciprocal rank fusion:** Σ 1/(60 + rank).
- **Gate (FWLS):** w(q) = sigmoid(θ·meta). meta = PCP history depth, familiar share of the eligible set, patient age, eligible-set size.
  - θ is fitted by listwise likelihood of the chosen specialist and cross-fitted over two fixed PCP halves of the validation referrals (fit on one half, applied to the other).
  - Disclosed: the gate uses validation labels, but never on the referrals it scores.
- **Split k:** the graph's top k from its own ranking, then the table's ranking for the remaining slots (k = 1, 2).
  - At the 30% weight, each component's 30%-weighted ranking is used.
  - This mirrors ContextGNN's idea of the graph for familiar items and the second model for exploratory ones.

### H2 — professor's model: GNN encoding + exploratory variables → logistic regression

`scripts/step12b_emb_lr.py`, `referral_gnn/hybrid.py`. The GNN is a frozen encoder.

**Features:**
- **GNN encoding of the pair:** the scorer's last hidden layer (128). A linear model on the encounter × specialist embedding product recovered little in smoke tests (selection 0.032 vs GNN 0.089); the scorer layer gives 0.087 on a 40% training subset.
- **14 pair features:** referral history, distance, patient–specialist prior, patient distance, insurer history, affiliation.
- **4 specialist values:** MIPS, years in Medicare, beneficiaries, rating.

**Model:**
- Pointwise logistic regression on the GNN's 21-row training bundles, standardised, C ∈ {0.1, 1}.
- One run per GNN seed.
- Reported for interpretability: standardised coefficients of the exploratory variables.
- Comparison: `--no-emb`, the same variables without the GNN encoding.
- Disclosed: training-row embeddings come from the GNN trained on those rows (in-sample encoder).

### H4 — flattened graph → LightGBM

`tabular.FlatGraph`, feature set `flat`. These are SIGN / SeHGNN-style metapath **means** of node attributes, built from encounters strictly before each weekly cutoff. The query's own referral is never included; brute-force check passed.

**Aggregates:**
- **A1 (Spec ← Enc):** the specialist's mean encounter profile (patient context, severity, disease-category distribution).
- **A2 (Spec ← Enc ← PCP):** mean attributes of the PCPs who referred to the specialist.
- **A3 (PCP → Enc → Spec):** mean attributes of the query PCP's past specialists, plus candidate − A3.

**Matches:** disease-category cosine and age difference.

**Model:**
- 88 features plus the 117 pure-tabular columns.
- Degree counts are deliberately excluded; those were the earlier "graph counts" set.
- LightGBM LambdaRank, tuned on the 12-config grid, plus seed 1.

## Judging (pre-stated; revision 1 after review, 2026-10-14, before any run)

Implemented in `scripts/step12b_judge.py`.

**Setting:** the 30% quality weight (blend0.3) decides; base is descriptive.

**Method:** paired hybrid − GNN differences on per-referral series, seed-averaged (hybrids average their two seed pairings), with a PCP-cluster bootstrap (2,000 draws).

**"Combines the strengths"** requires all of the following against the GNN:

| Lens | Criterion |
|---|---|
| Load | upper CI of Δbusiest share ≤ +1.0 pt, for the **top pick** (busy1) **and** the **top-5 slots** (busy5) |
| Newness | Δnew-to-PCP top-5 (new_pcp5) ≥ +5 pts with lower CI > 0, **and** lower CI of Δnew-to-PCP top pick (new_pcp1) > 0 |
| Feasibility | Δ"good new" picks (good_new5: new, higher MIPS than chosen, not busiest) ≥ 0 |
| Quality | lower CI of ΔMIPS gain (gain5) ≥ −0.5 points |
| Fit | top-5 match (r5) ≥ min(GNN, table) − 1 pt |

**Reported, not judged:**
- travel, median and mean;
- insurer-feasible new picks (new_ins5);
- the table reference.

**Split-k arms** are labelled "top pick = GNN, slots after = table" and judged by the same rule. Because busy5 and new_pcp1 count, they cannot pass by position alone.

**Tie-break** if several arms pass: largest Δnew_pcp5, then lowest busy1. The winner is the only arm taken to 2019, in a later step with a decision entry. For 2019 the gate θ is refit on all of 2018 (no cross-fitting), and the same rule is applied.

**Gate:**
- θ includes a free log-temperature.
- The per-fold mean weights are reported.
- It is fitted to imitate the chosen specialist, not to the lenses.

**Disclosures:**
- **FlatGraph** reintroduces some PCP–specialist history indirectly. With 119 PCPs, A3 nearly identifies the PCP. It is a hybrid, not a pure-tabular model.
- **H2:** the in-sample encoder means the LR leans on the embedding. Interpretability is read from `--no-emb`, which includes history pair features and is not a baseline.
- **Multiplicity:** 10+ arms; exploratory.
- **The legacy Step-10 "chosen" field** in the evaluator summary is ignored.

**Run mode:** `--lean --strata --step12` (only base and blend0.3 are needed; bounds memory). PID-liveness monitoring.

## Cost (local, AC power)

| Item | Time | Hardware |
|---|---|---|
| H2 training: 2 GNN seeds × (encoding + 2 C values) + 2 no-emb runs | ≈ 15 min | MPS |
| H4 training: 12-config grid + seed 1 | ≈ 12 min | CPU, parallel with H2 |
| One evaluation pass: 10 runs + 2 hybrid pairings, `--strata --step12` | ≈ 15–25 min | — |
| **Total** | **≈ 45–60 min** | two sittings of ≈ 25 min each if needed |
