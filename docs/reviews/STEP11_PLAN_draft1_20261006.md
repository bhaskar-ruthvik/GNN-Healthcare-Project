# Step 11 plan: diagnostic experiments that justify architecture changes (DRAFT 1, 2026-10-06)

Status: draft for independent review. Nothing has been run. The frozen STEP10 yardstick and the final model (v5 unmasked, 30% quality weight) stay unchanged.

## 0. Why this step exists

Professor (meeting Friday 2026-10-02, via user): the reasoning behind needing better recommendations, and behind a GNN recommender, landed. The next stage is a set of exploratory experiments whose results give **numeric grounds for specific architecture changes**. Each experiment below is written as:

**question → measurement → pre-stated trigger → architecture change it would justify.**

A trigger that fires is a reason to build that change. A trigger that does not fire is also reported, because it rules a change out. Thresholds are fixed before any output. Validation and test are reported side by side (standing rule). Choices are made on validation only.

Metrics reused from STEP10 for every stratum:
- fit: R@5 and MRR within the eligible set
- quality: MIPS gain, standard estimate; the cautious estimate is kept in the tables only
- travel: median and p90 distance from the patient
- load: busiest-10% top-1 share
- new connections: new-to-PCP share of the top 5

Two cohorts are used, and the cohort is named in every table:
- STEP10 primary cohort: first-time referrals in choice specialties, 25-mile cap.
- First-time cohort with the specialty-given pool: the older yardstick, kept for continuity with the 09-25 results.

Confidence intervals come from a patient-cluster bootstrap, or a PCP-cluster bootstrap for PCP-level strata.

## 11.1 Stratified evaluation (existing checkpoints, no training)

Runs: final v5 unmasked s42/s1 (seed-averaged), plus PCP habit and practice as references.

| Split | Definition (fixed now) | Data check done | Caveat |
|---|---|---|---|
| Urban vs rural (patient) | RUCA code of the patient ZIP: metropolitan (1–3) vs non-metropolitan (4–10). Fallback if RUCA is unavailable: tertiles of distance to the nearest eligible specialist. | Patient ZIP present for 100% of the 10,619 patients. The 2020 Gazetteer (lat/lon only) is in `reference_data/`. **RUCA is not present.** | All patients are in DFW, so the rural group may be small. Report n first; collapse to 2 levels if a cell has < 100 patients. |
| Patient age | < 65 (disability-entitled), 65–74, 75–84, 85+ (as-of encounter age) | Age is available. | Overlaps A7 age bands (bias only); here the full metric set is used. |
| Specialist career stage | Medicare tenure (years_in_medicare as of the referral): early ≤ 6, mid 7–13, late ≥ 14 | Present for 100% of specialist rows. Range 2–16, median 11. **Top-coded at ~16** (enrolment-record horizon). | This is Medicare tenure, not years since training. "Late" holds everyone ≥ 14, including top-coded values. Option: graduation year from the public CMS Doctors & Clinicians file, only if specialist NPIs are available (to check). |
| PCP career stage | Same bands on PCP tenure | Present. Range 3–16, median 14. Only 119 PCPs. | Few PCPs per band (≈ 15–60), so CIs are wide. PCP-cluster bootstrap only. |

Specialist career is a **candidate-side** split, so two views are reported:
- **Exposure:** the share of top-5 slots going to each band, compared with the band's share of eligible candidates and its share of actual referrals. Exposure ratio = recommended share ÷ actual share.
- **Ranking quality when the referred specialist is in the band:** MRR and R@5, conditioned on the target's band.

Triggers (fixed now):
- **T1, cold start.** Early-career exposure ratio < 0.8 with CI upper < 1, **and** early-band target MRR below mid-band (paired CI excludes 0). This would justify a cold-start path: a feature-only specialist tower blended by history volume, and/or history dropout during training.
- **T2, rural.** Non-metro fit (R@5) lower than metro by ≥ 0.05 with CI excluding 0, **or** non-metro median travel exceeds practice. This would justify explicit geography in the graph: ZIP/area nodes or specialist–specialist proximity edges, so that message passing carries local alternatives.
- **T3, age.** Fit or MIPS gain differs across age bands by more than the s42–s1 seed gap, with CI excluding 0. This would justify age-conditioned scoring (age as a FiLM/gating input to the scorer instead of one encounter feature).
- **T4, PCP career.** Early-career PCPs show lower fit and a larger PCP-history SHAP share than late-career PCPs. This would justify a PCP-level adaptive weight on history (a gate), since newer PCPs have thin networks.

Cost: one evaluation pass (≈ 15 min) plus a per-stratum SHAP summary from the existing 10.6 per-referral file (n = 60 is too small per stratum; reported only as descriptive).

## 11.2 Gender and race masking: what was done and what remains

**What was done (checked in code, `three_node_model.py:131`).** Masking is applied at the model input, inside `encode()`, before the encoders and before HGT message passing, in **both training and scoring**. The masked model never sees patient gender or race. Its encounter embeddings, and everything propagated from them, are learned without them. It is not a filter on the recommender's output. Result (STEP9, 10.5): no attributable effect.

**What remains: proxies.** The masked model still sees:
- patient ZIP (through distance),
- insurer,
- specialist panel composition (`panel_female` / `panel_male` counts).

Each of these can stand in for gender or race. "Masked" therefore means "not given directly", not "not encoded".

Proposed experiments:
- **11.2a, probe (no training).** Fit a logistic probe on frozen encounter embeddings from the masked model (s42) to predict patient race and gender. Training-year encounters, 5-fold; report AUC on validation encounters. Repeat on the unmasked model as the upper reference, and on input features alone as the proxy reference.
  - **T5:** masked-model probe AUC ≥ 0.60 with CI above 0.5. Embeddings then encode demographics through proxies.
- **11.2b, adversarial masking (training, only if T5 fires).** Add a gradient-reversal head that predicts race/gender from the encounter embedding (λ ∈ {0.1, 0.3}, chosen on validation). Report:
  - probe AUC drop,
  - change in fit / MIPS gain,
  - change in 10.5 group gaps.

  This would justify an "embedding-level debiasing" component, with its cost stated in fit.

## 11.3 How the network evolves year to year

Two complementary views, both recommended. Option (b) is the expensive one.

**(a) Descriptive evolution (no training).** On the PCP–specialist referral graph per service year (2015–2019; 2014 is reported if non-trivial):
- active PCPs/specialists, edges, and new-edge share,
- edge persistence (Jaccard of edge sets in year t vs t+1; share of year-t edges used again in t+1),
- specialist entry/exit,
- degree concentration (Gini) and busiest-10% share,
- share of first-time referrals to new-to-PCP specialists,
- median PCP network size,
- MIPS of new vs retained edges,
- community stability (Louvain on the year graph; adjusted Rand index between consecutive years).

**(b) Predictive drift (training).** Expanding (cumulative) window. Train on 2015..t and evaluate on t+1:
- 2015 → 2016 (new run)
- 2015–16 → 2017 (new run)
- 2015–17 → 2018 (existing final model; 2018 was its validation year)
- 2015–18 → 2019 (new run, optional)

Checkpoint selection uses the last ~3 months of each training window, never the evaluated year. The existing 2015–17 model selected its checkpoint on 2018. Either disclose this or rerun it under the same rule (user/reviewer decision). Optional contrast: a sliding window that trains on the last year only, to show whether old referral patterns help or add noise. The two parts answer different questions:
- The retraining asks whether a model trained on older data degrades on later years.
- The frozen-graph test asks whether keeping the graph's connections current matters, with no retraining.

Compare:
- (i) the graph updated weekly, as now,
- (ii) the graph frozen at the end of training. This is inference only, so no retraining is needed.

Triggers:
- **T6.** Year-over-year edge persistence < 0.5 **and** the frozen-graph drop in first-time R@5 ≥ 0.03 (CI excluding 0). This would justify temporal mechanisms: time-decayed edge weights, or a temporal attention / time-encoding in message passing (TGN-style memory as the stronger variant).
- **T7.** Persistence ≥ 0.7 and frozen ≈ updated. This would justify keeping a static graph and spending complexity elsewhere.

Cost:
- (a) ≈ 10 min.
- (b)(ii) ≈ 10 min.
- (b) retraining: 2 extra training runs ≈ 2.5 h locally. They are optional and can go on Grace if VPN access works.

## 11.4 Non-GNN baselines

All baselines use the **same candidate sets, cohorts, splits and yardsticks** as the GNN, and the same strictly-earlier feature rules. Features per (encounter, candidate) row are:
- the 14 pair features,
- the specialist snapshot,
- PCP snapshot,
- encounter context and diagnosis category flags,
- specialty-match indicators.

These are exactly the GNN's explicit inputs, flattened, with no message passing.

| Baseline | Why |
|---|---|
| **LightGBM LambdaRank** (professor's suggestion) | Strong tabular ranker on identical features. This is the key test of whether message passing earns its place. |
| Logistic regression (same features) | Linear floor. |
| Implicit matrix factorisation (ALS on the PCP × specialist referral matrix, strictly earlier) | Classic collaborative-filtering recommender with no features. |
| Item-kNN (specialist co-referral similarity) | Simple neighbourhood CF. It is a "graph without learning" baseline. |
| Existing: PCP habit, popularity, MIPS-sorted, random | Already in STEP10. |

Training data:
- the same training queries,
- negatives drawn from the eligible set (as the GNN's pool policy).

Hyperparameters are tuned on validation (small grid, ≤ 20 configs for LightGBM). The scores are also run through the STEP10 quality-weight blend, so the frontier comparison is like for like. LightGBM TreeSHAP gives a feature-importance comparison with the GNN's 10.6 SHAP.

Triggers:
- **T8.** LightGBM ≥ GNN on first-time R@5 (CI includes or exceeds 0). The graph layers are then not adding value. Report this plainly. Architecture changes justified by T1/T2/T6 become the case for what the graph *should* carry. LightGBM becomes the bar any new GNN must beat.
- **T9.** GNN > LightGBM with CI excluding 0. The relational structure is earning its place, and the result goes in the pitch.

Dependencies: `lightgbm` and `implicit` are not installed. Both are PyPI installs into `Implementation/env`.

## 11.5 Synthesis

One table: trigger → fired? (numbers, CI) → architecture change → expected effect → cost. Changes are built only for fired triggers. Each built change is evaluated against the 11.4 LightGBM bar and the final model, with two seeds, under the STEP10 yardstick. This table is the "why we changed the architecture" argument for the pitch.

## Order and cost

1. 11.1 (≈ 30 min incl. setup)
2. 11.2a probe (≈ 20 min)
3. 11.3a + 11.3b(ii) (≈ 30 min)
4. 11.4 baselines (≈ 2–3 h, mostly LightGBM tuning)
5. 11.3b retraining (optional, ≈ 2.5 h)
6. 11.2b (only if T5 fires; ≈ 2.5 h for 2 seeds)
7. 11.5

Each sub-step: validation → user check → test.

## Open items for the user and the reviewer

1. RUCA codes: a public USDA ERS ZIP-to-RUCA file is needed (download). Is that acceptable, or should the distance-based fallback be used?
2. Career stage: Medicare tenure is top-coded at ~16 years. Are NPIs available to fetch graduation year? If not, tenure bands with the caveat.
3. Network evolution: is (a) + (b)(ii) enough, or are the rolling retrains (b) wanted?
4. Package installs: `lightgbm`, `implicit` (PyPI).
5. Thresholds T1–T9: reviewer to challenge.
