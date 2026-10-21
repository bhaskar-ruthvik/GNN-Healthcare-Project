# Step 11 plan: diagnostic experiments that justify architecture changes (DRAFT 4, 2026-10-07)

Status: DRAFT 4 (11.4 redesigned at the user's request; draft 3 archived at `reviews/STEP11_PLAN_draft3_20261007.md`). Earlier: All reviewer round 1 and round 2 items are adopted; the reviewer agreed to freeze once the round 2 items are adopted as stated (the review summaries are in PROJECT_LOG 2026-10-07; draft 1 is kept in `reviews/`; draft 2 differs only by the round 2 items listed in the log). It is frozen once the user confirms the open items at the end. Nothing has been run. The frozen STEP10 yardstick and the final model (v5 unmasked, 30% quality weight, A3/A5) are unchanged.

## 0. Purpose and global rules

Professor (meeting Friday 2026-10-02, via the user): the next stage is a set of exploratory experiments that give **numeric grounds for specific architecture changes**. Each experiment is written as question → measurement → pre-stated trigger → the change it would justify. A trigger that does not fire is reported too, because it rules that change out.

**Global rules (apply to every trigger):**

**G1. Where triggers fire.** Triggers fire on **validation (2018) only**. Test (2019) is reported beside it. A fired trigger counts as *replicated* only if test agrees in sign. The trigger definitions and the decision file (`step11_decisions.json`, same mechanism as `step10_decide.py`) are recorded before any Step 11 output.

**G2. Effect threshold.** The effect threshold is max(nondeterminism band, seed gap), as in STEP10 §3, computed on **the trigger's own metric**: exposure ratio for T1, MIPS gain for 11.5, and so on.
- For a single model, it comes from v5 unmasked s42 / s42_dup / s1.
- For a contrast between two learned models (GNN − no-graph, GNN − LightGBM), use the GNN's dup band plus the larger of the two models' seed gaps. LightGBM's seed gap comes from 2 seeds of its bagging and feature-fraction randomness.

**G3. One primary contrast per trigger.**
- Each trigger has exactly one primary metric and one contrast. Everything else is descriptive.
- Intervals for the primary contrasts are Holm-adjusted across all primary contrasts in Step 11 (about 9; the first step is α/9, roughly a 99.4% interval). Equivalence outcomes are tested by TOST at the same Holm-adjusted level.

**G4. Bootstrap.** The **PCP-cluster bootstrap is primary** for triggers, with the patient-cluster bootstrap reported beside it. Candidate-side exposure ratios also get a specialist-cluster sensitivity.

**G5. Strata measure interactions, not difficulty.** A stratum gap in raw fit partly reflects case mix and eligible-set size: fill-to-5 referrals give every scorer the same R@5. Every stratum trigger is therefore an **interaction**: (GNN − reference scorer) in stratum A versus stratum B. The references are PCP habit and the no-graph model (11.1). Fit is R@5 **in excess of the exact random expectation** for that referral. Fill-to-5 referrals are reported separately.

**G6. Configurations named.** Every stratum and every baseline is evaluated at three configurations:
- (i) the base ranking;
- (ii) the 30% quality blend (λ = 0.3);
- (iii) the configuration selected by STEP10 §1 / A11.

Baselines get the same λ grid {0.1, 0.2, 0.3, 0.5}, so MIPS gain can be compared at matched fidelity. All three configurations are reported, but **each trigger names exactly one** (G3).

**G7. Cohorts.** The STEP10 primary cohort is the main one. The first-time cohort with the specialty-given pool is a continuity table only.

**G8. Pre-stated cell rule.** A stratum with < 100 patients (or < 20 PCPs for PCP strata) is collapsed with its neighbour, or reported as descriptive only.

## 11.1 Baselines (run first: the stratum triggers need their contrasts)

**Two different questions:**
- *Does message passing help?* The three-node no-graph model answers this (`--no-graph`, `ThreeNodeConfig.use_graph=False`): same inputs, same trainer, same negatives, same selection. It is not a pure message-passing ablation. With `use_graph=False` the scorer is encounter + specialist + pair (`three_node_model.py:149`), so the PCP snapshot features drop out entirely (they are about 3% of the 10.6 SHAP). The primary LightGBM includes them, which partly brackets that effect.
- *Does a different model class do as well?* LightGBM answers this.

Step 5 already found PairMLP ≈ GNN (+0.004 MRR, not significant; better on first-observed). The three-node version has not been run.

| Baseline | Training rows / negatives | Notes |
|---|---|---|
| **No-graph three-node** (2 seeds: 42, 1) | Identical to final v5: `neg_policy uniform`, 21 bundled alternatives, `train_cohort all`, first-time MRR selection on the 3,000-query validation subset | ≈ 1.2 h per seed locally |
| **LightGBM LambdaRank, primary** | The exact GNN training rows (`tc_enc` / `tc_spec`, 21 per query) | Features listed below |
| **LightGBM + as-of graph features** | Same rows | Adds hand-made relational features (below). This is the comparison the pitch needs: can explicit graph features do what message passing is meant to do? |
| Logistic regression | Same rows, same features as primary LightGBM | Linear floor |
| ALS matrix factorisation (numpy implementation, no `implicit` package) | PCP × specialist referral counts strictly before the refit date | Refit monthly from strictly earlier data, so it is like-for-like with live PCP-habit counts |
| Item-kNN (specialist co-referral cosine) | Same cadence as ALS | "Graph without learning" |
| Existing: PCP habit, popularity, MIPS-sorted, random | n/a | Already in STEP10 |

**Secondary training variant.** The GNN and LightGBM are both trained on eligible-set negatives (`neg_policy pool`). This is reported as a sensitivity and never used for selection.

**LightGBM features (all strictly earlier, the same as-of rules as the GNN).** The draft 1 claim that the GNN's explicit inputs were simply flattened was wrong: in the GNN, PCP features reach the score only through message passing, and the patient's prior specialists arrive through PRECEDES edges. The primary feature set is therefore stated explicitly:
- the 14 pair features;
- the specialist snapshot, including as-of MIPS (v5);
- the encounter block (patient context, severity, patient history, diagnosis flags, time);
- the PCP snapshot;
- specialty indicators.

The **graph-feature** set adds, all as-of:
- specialist in-degree (distinct PCPs and referrals, last 365 days);
- specialist referrals by the encounter's diagnosis category;
- co-referral count between the candidate and the patient's prior specialists;
- the PCP's prior distinct-specialist count;
- 2-hop PCP → specialist → PCP overlap: the share of PCPs who share referral partners with this PCP and have referred to the candidate.

**Information horizon of the graph features.** Graph features match the GNN's information horizon: they are computed at the GNN's weekly bucket cutoff (the Monday bucket start) for that row, never at the query day and never from training-period totals. Every training row is recomputed as of its own cutoff. Definitions:
- **Co-referral:** distinct patients referred to both specialists before the cutoff, across all patients.
- **2-hop overlap:** the share of PCPs with at least 1 referral partner in common with this PCP (before the cutoff) who have referred to the candidate.

**Tuning.** At most 20 configurations per LightGBM variant. Selection uses first-time MRR on the 3,000-query selection subset, the GNN's own criterion. Validation is reported both with and without those queries.

**Explanations.** LightGBM TreeSHAP is reported, but it explains a different quantity than the 10.6 SHAP, so the comparison is qualitative only.

**Triggers.** Primary metric: first-time R@5 in excess of random, on the STEP10 primary cohort, base configuration. Equivalence margin: ±0.02 R@5. The four outcomes, using Holm-adjusted CIs:
- **better:** CI lower bound > 0 and point estimate > +0.02;
- **equivalent:** CI entirely inside ±0.02 (TOST);
- **worse:** the mirror of better;
- **inconclusive:** anything else.
- **T8. Message passing.** GNN − no-graph. If the result is *equivalent or worse*, the current message passing is not adding value; architecture changes must show they beat the no-graph model. If *better*, the result is part of the pitch.
- **T9. Model class.** GNN − best LightGBM variant (selected on validation), with the same four outcomes. The best of {no-graph, LightGBM + graph features} becomes **the bar every new architecture must beat**.
- Matched-fidelity MIPS gain (λ grid) is reported beside T8 and T9 as descriptive.
- **Stopping rule.** If LightGBM + graph features is *better* than the GNN, the Step 11.5 changes are built on that baseline's feature insight first: the GNN must reproduce what the features capture. A new GNN architecture is pursued only if it beats that bar on validation.

## 11.2 Stratified evaluation (existing and 11.1 checkpoints, no training)

| Split | Definition (fixed now) | Data facts (checked 2026-10-07, aggregate) |
|---|---|---|
| Urban vs rural (patient); the fallback is labelled "local specialist access", not rurality, because it depends on specialty and is measured per referral | RUCA code of the patient ZIP: metropolitan 1–3 vs non-metropolitan 4–10. This needs the public USDA ZIP-RUCA file; the dataset has no patient RUCA. Fallback (pre-specified) if fewer than 100 non-metropolitan patients or RUCA is unavailable: tertiles of the distance from the patient to the 5th-nearest eligible specialist ("access"). | Patient ZIP is present for all 10,619 patients. The source has RUCA only for **provider practice locations** (`Rndrng_Prvdr_RUCA_pcp`, `Rndrng_Prvdr_RUCA`, from MUP), plus the QPP `rural clinician` flags. PCPs: 116 metropolitan, 3 non-metropolitan (4.5% of rows), so a PCP-based proxy for patient rurality is not usable. |
| Rural specialists (candidate side; new, from the source RUCA) | Specialist practice RUCA: metropolitan 1–3 vs non-metropolitan 4–10; codes 0 and 99 count as unknown. Sensitivity: the QPP `rural clinician` flag (ever true). Exposure view as for the career strata. Descriptive (no trigger). | Specialists: 5,996 metropolitan, 261 non-metropolitan, 5 unknown (1.5% of rows). 684 are ever flagged as rural clinicians (206 inconsistently), so the QPP definition is broader than RUCA. Each provider has one RUCA value. |
| Patient age | < 65, 65–74, 75–84, 85+ at the encounter | Repeats the A7 bands. 10.5 found no age gap that replicates, so this split is **low prior**. |
| Specialist career, as the professor asked | **Snapshot** Medicare tenure: early ≤ 6, mid 7–13, late ≥ 14 years. Descriptive only. | Constant within every provider (0 of 6,262 vary), so it is a one-time snapshot of unknown year. Median 10. Values pile up at 15 (971 specialists; 16: 108). Medicare tenure, not career length. NPIs are in the source and all are valid-format, so graduation year could come from the public CMS Doctors & Clinicians file (needs a download). |
| Specialist experience in the data (trigger split) | As-of prior referral count received (0, 1–9, 10–49, 50+) and days since first seen | Measures cold start directly. Report how many eligible candidates have 0 prior referrals but recent claims, to show the 0–9 band is populated. |
| PCP career, as the professor asked | Snapshot tenure bands | 10 / 70 / 39 PCPs (early / mid / late). The early band is too small for inference, so this split is descriptive. |
| PCP network (trigger split) | As-of distinct specialists previously referred to: tertiles over PCPs | Measures thin networks directly. |

**Candidate-side views for specialist strata:**
- Exposure ratio = the band's share of top-5 slots ÷ its share of actual referrals, with its share of eligible candidates reported too.
- Ranking quality conditioned on the band the referred specialist falls in. This compares unpaired groups, so it is reported as a difference in means with a cluster-bootstrap CI, not as a paired contrast.

**Triggers (G1–G5 apply):**
- **T1. Cold start.**
  - Primary: the exposure ratio of the 0–9 prior-referral band, final model vs no-graph and vs PCP habit (interaction). It fires if the ratio for the final model at the base configuration is < 0.8, its CI upper bound is < 1, and it is lower than the no-graph model's ratio by more than G2.
  - Justifies: a cold-start path, i.e. a feature-only specialist tower gated by history volume, and/or history dropout in training.
- **T2. Rural / low access.**
  - Primary: the interaction (GNN − primary LightGBM, which includes the distance features) in excess R@5, base configuration, low-access vs high-access tertile. The draft 1 "or travel" clause is dropped.
  - Fires if the GNN's disadvantage in the low-access tertile exceeds G2.
  - Justifies: explicit geography in the graph (area nodes or specialist–specialist proximity edges), because message passing currently fails to use local alternatives that distance features alone can use.
- **T3. Age (low prior).** Interaction (GNN − PCP habit) in excess R@5, base configuration, 85+ vs 65–74. Fires above G2. Justifies age-conditioned scoring (FiLM/gating).
- **T4. Thin PCP network.**
  - Primary: the interaction (GNN − no-graph) in excess R@5, base configuration, thin vs thick PCP-network tertile. It fires if the GNN's advantage in the thin tertile is smaller than in the thick tertile by more than G2.
  - Descriptive only: the 10.4 `no_pcp_history` R@5 drop per tertile. It is mechanical, because less history means less to remove.
  - Justifies: a PCP-level adaptive gate on history (fall back to content for thin networks).

## 11.3 Gender and race masking: what was done and what remains

**What was done** (verified by the reviewer in `three_node_model.py:114-121`). Masking runs inside `encode()`, before the encoders and HGT, in training and scoring. It covers the gender and race categorical columns and the gender-ambiguity flag on every encounter row. It is already at the GNN/embedding stage, not an output filter.

**What remains: proxies.**
- Strongest for a patient's gender: diagnosis categories, the patient-history block (`patient_history`, columns 7:74), and the patient's prior specialists via PRECEDES edges.
- For race: ZIP/distance and insurer.
- Draft 1 also listed `panel_female/panel_male`. They are specialist-level and **cannot** leak a patient's gender into the encounter embedding, so they are removed from the list.

**11.3a Probe.**
- Model: the masked v5 model, s42. It has one seed and is the supplementary model, not the final model (A3).
- Embeddings: frozen encounter embeddings.
- Task: logistic probe for gender (AUC) and race (macro one-vs-rest AUC, unknown excluded).
- Data: fit on training-year encounters with patient-grouped 5-fold CV for regularisation; report on validation-year encounters. Given 10.5, T5 is expected **not** to fire.
- References:
  - a **nonlinear GBDT probe** on the masked model's input features only (proxy content; a linear probe on raw inputs would mix capacity with information);
  - the unmasked v5 model's embeddings (upper reference).
- **T5.** (masked-embedding AUC − input-only AUC) > 0.02 with a patient-cluster CI excluding 0, **and** a 10.5 group gap that changes, meaning it passes the A7 attribution rule. That is, message passing *amplifies* demographic signal beyond what the inputs already carry, and it matters for recommendations.

**11.3b Adversarial masking** (only if T5 fires): a gradient-reversal head on the encounter embedding, λ ∈ {0.1, 0.3} chosen on validation, two seeds. Report the probe AUC, fit, MIPS gain and the 10.5 gaps.

## 11.4 How the GNN and the network evolve as data accumulates

**Question (user, 2026-10-07).** What does the GNN learn at each stage as more data arrives, and how do those insights change? This is **not** a forecasting test: no snapshot is judged on predicting the following year. Draft 3's staleness tests are kept as a supporting part (d).

Data growth (encounters by service year): 2015 2.9k, 2016 18.7k, 2017 34.4k, 2018 37.4k, 2019 17.6k (ending 2019-10-29). 2014 has 1 encounter.

**(a) Descriptive network evolution (no model).** The rules are:
- Use years 2016–2019; 2015 is reported as context.
- Compare **January–October windows** whenever 2019 is involved.
- Compute persistence on a **stable panel** (PCPs active in both years), per PCP: the median share of a PCP's year-t specialists referred to again in t+1, and the share of the PCP's t+1 referrals that go to specialists new to that PCP.

Also reported, all descriptive:
- specialist entry and exit;
- within-specialty Gini and the busiest-10% share;
- median PCP network size;
- MIPS of new vs retained connections;
- Louvain community stability (ARI) on the stable panel.

This is the backdrop against which the model changes in (b) and (c) are read.

**(b) Cheap first look: one fixed model on a growing graph (no retraining).** The final v5 model (s42; s1 as a seed check) encodes the graph as of 2015-12-31, 2016-12-31, 2017-12-31, 2018-12-31 and 2019-10-29. Measures M2 (below) are computed per date. Disclosed limits:
- It shows one fixed model's view of a growing network, not what a model learns from more data.
- The dates up to 2017 are inside that model's training period.
- Specialists not yet seen at a date have no embedding there; only specialists present at both dates are compared.

**(c) Cumulative snapshots: the same GNN retrained on growing data.**

*Snapshots.* Each snapshot trains on everything up to its end date:

| Snapshot | Data through | Encounters (approx.) |
|---|---|---|
| S16 | 2016-12-31 | 21.6k |
| S17 | 2017-12-31 | 56k |
| S18 | 2018-12-31 | 93k |
| S19 | 2019-10-29 | 111k |

2015 alone is excluded (2.9k encounters, 1,799 training queries).

*Training rule, identical across snapshots:*
- the final-model configuration (v5 schema, unmasked, uniform negatives, same epochs cap, patience and LR schedule);
- the **last 3 months of each window** form that snapshot's selection holdout (first-time MRR); training uses the window before it;
- the holdout's encounters still enter the graph as history for later use.

S17 is **retrained under this rule**, not reused. The existing final model selected its checkpoint on 2018, so it is reported beside S17 for reference only.

*Inputs rebuilt per snapshot.* The following are all fitted on the snapshot's own training part:
- encoders, scalers, vocabularies (incl. diagnosis), v5 as-of MIPS and its specialty-mean imputation,
- STEP10 "training-period" references (load busiest-10%, MIPS yardstick imputation).

Code change: splits are fixed in the prepared bundle (`data.py:244`, `enc["split"]`), so the loader gains an explicit split-override by date. The override is applied before any fitting and recorded in the manifest and in the inputs' SHA-256. Regression check: with the current split dates, the rebuilt v5 inputs must be identical to `three_node_v5`.

*Seeds and the noise band.* Each snapshot runs with 1 seed (42). S18 gets a second seed (1) and a duplicate of seed 42. Snapshot-to-snapshot changes in any measure are read against this S18 band (G2 logic). Changes inside the band are "no change". If Grace is available, every snapshot gets 2 seeds instead.

*No-graph companion per snapshot* (1 seed). It answers the pitch question: **does message passing contribute more as the network densifies?**

*Measures* (all descriptive except T10 and T11):
- **M1. What drives recommendations.** Per-input SHAP shares (A9 method, 36 input groups, 12 permutations) on a **common probe set**: 60 first-time referrals in choice specialties from 2016 Jan–Sep, drawn once with a fixed seed. These referrals are in-sample for every snapshot. The question is "what does each model rely on for the same cases", not accuracy. Each snapshot also gets SHAP on its own holdout (out-of-sample, different cases), as a secondary view.
- **M2. What it learns about specialists.** Specialist embeddings: post-HGT outputs, graph as of the snapshot end date. Measured by:
  - specialty purity (share of the 10 nearest neighbours with the same specialty) and silhouette by specialty;
  - linear CKA between consecutive snapshots on shared specialists;
  - per-specialist Procrustes displacement, against the change in that specialist's referral volume;
  - top central specialists (PageRank on the referral graph vs embedding-norm/attention-based importance), with overlap between snapshots.
- **M3. How recommendations behave.** On the common probe set and on the STEP10 primary cohort of each snapshot's holdout:
  - new-to-PCP share of the top 5;
  - busiest-10% top-1 share;
  - MIPS gain at λ = 0 and 0.3;
  - median distance;
  - R@5 (holdout only, descriptive).
- **M4. Relation usage.** Mean HGT attention mass per edge type (has_encounter, referred_to, precedes and their reverses), if extractable from PyG HGTConv without modifying the model. Otherwise dropped and disclosed.

*Pre-stated signals (descriptive, but each linked to an architecture choice; threshold = the S18 seed band):*
- **T10. Graph value grows with data.**
  - Measure: (GNN − no-graph) holdout first-time R@5 (excess over random) increases monotonically from S16 to S19, with the S19 − S16 difference above the band; or the SHAP share of graph-carried inputs (PCP block plus relational context) rises and the PCP-history pair-feature share falls.
  - Justifies: keeping and investing in message passing. The pitch argument becomes "the graph pays off at scale".
  - Not firing argues for feature-based models at this data size.
- **T11. Embeddings do not organise by specialty even with more data.**
  - Measure: specialty purity at S19 below 0.5, with no rise above the band from S16.
  - Justifies: making specialty explicit structure (specialty nodes, or a specialty-conditioned scorer), instead of hoping message passing recovers it.

**(d) Staleness checks (supporting; from draft 3, unchanged).**
- **Drift curve:** the final model and PCP habit by quarter, 2018 Q1–2019 Q3, with the 3,000 selection queries excluded.
- **Frozen-graph factorial**, freeze at 2017-12-31, on validation (T6 primary) and 2019:
  - (0) live;
  - (1) post-freeze historical encounters dropped from the referral/PCP subgraph and kept only as PRECEDES predecessors (disclosed artefact);
  - (2) also snapshots, PRECEDES, gap and time encoding;
  - (3) also pair features;
  - plus PCP habit with frozen counts.
- **Recency window:** last 6, 12 or 24 months of edges vs full history.

Triggers:
- **T6. Staleness matters.** Level (1) − level (0) first-time R@5 on validation exceeds G2 **and** a recent-only window is at least as good as full history. Justifies edge-age weighting in message passing.
- **T7. Static suffices.** Level (1) ≈ level (0) within G2 and the drift curve flat relative to PCP habit.

The draft 3 "optional retrain" is superseded by (c).

## 11.5 Synthesis and success criteria

One table: trigger → fired on validation? (number, Holm-adjusted CI) → replicated on test? → architecture change → cost.

A change is built only if its trigger fired. **Success criterion for a built change** (fixed now). On validation, with two seeds, it must:
- close ≥ 50% of its stratum gap (the interaction defined by its trigger);
- lose no more than G2 in overall first-time R@5;
- reduce MIPS gain at λ = 0.3 by no more than G2.

On test it must agree in sign (G1); this guards against regression to the mean, since the trigger's stratum came from the same validation data. It must also **beat the 11.1 bar** (the best of no-graph and LightGBM + graph features): overall R@5 no worse than the bar by more than the 0.02 margin, and the stratum interaction better than the bar's. Each new architecture gets the same tuning budget as LightGBM (at most 20 configurations, selected on the 3,000-query subset).

## Order and cost

1. 11.1 baselines. No-graph: 2 seeds, ≈ 2.5 h. LightGBM, LR, ALS, kNN: ≈ 2 h.
2. 11.2 strata, ≈ 30 min.
3. 11.3a probe, ≈ 20 min.
4. 11.4 (a), (b) and (d), ≈ 1 h.
5. 11.4 (c), cumulative snapshots:
   - split-override code and the v5 regression check, ≈ 1 h;
   - 4 input rebuilds, ≈ 30 min;
   - GNN runs: S16, S17, S18 ×3 (s42, s1, dup), S19 = 6 runs, roughly 1–2.5 h each, growing with window size;
   - no-graph runs: 4, faster;
   - SHAP: 5 × ≈ 70 min.

   Total ≈ 15–20 h locally; parallel on Grace if the VPN works.
6. 11.3b, only if T5 fires, ≈ 2.5 h.
7. 11.5.

Each step goes validation → decision entry → test.

## Open items for the user

1. Patient rurality: download the public USDA ZIP-RUCA file, or use the access-tertile fallback. Provider RUCA is in the source; patient RUCA is not.
2. Graduation year: download the public CMS Doctors & Clinicians file and join it on NPI, or keep snapshot Medicare tenure.
3. Package install: `lightgbm` (PyPI). `implicit` is no longer needed.
4. 11.4 redesigned per the user (2026-10-07): cumulative snapshots for insight, not forecasting, plus the fixed-model first look; the staleness checks are kept as supporting evidence.
