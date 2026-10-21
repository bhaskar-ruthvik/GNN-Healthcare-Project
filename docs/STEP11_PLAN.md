# Step 11 plan: diagnostic experiments that justify architecture changes (FROZEN 2026-10-07)

Status: **FROZEN 2026-10-07** after reviewer round 4 ("freeze once adopted as written"; all round-4 items adopted; drafts 1, 3, 4 and 5 archived in `reviews/`). Two user choices remain open with pre-specified fallbacks (patient RUCA download → else access tertiles; graduation-year download → else snapshot tenure); either answer is a disclosed data-source choice, not a design change. Round 3: items adopted in 11.4(c). Draft 4 had redesigned 11.4 at the user's request. Earlier: All reviewer round 1 and round 2 items are adopted; the reviewer agreed to freeze once the round 2 items are adopted as stated (the review summaries are in PROJECT_LOG 2026-10-07; draft 1 is kept in `reviews/`; draft 2 differs only by the round 2 items listed in the log). Nothing has been run. The frozen STEP10 yardstick and the final model (v5 unmasked, 30% quality weight, A3/A5) are unchanged.

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

S17 is **retrained under this rule**, not reused. The existing final model selected its checkpoint on 2018, so it is reported beside S17 for reference only. S17 trains on less data than the final model; this is disclosed.

**Disclosure: holdouts that use test-year data.** S18's holdout is 2018 Q4 (validation-year data). S19's holdout is Aug–Oct 2019 (test-year data), so its checkpoint is chosen on 2019. S19 is **never used as a 2019 performance comparison**; it is an insight snapshot only.

*Inputs regenerated per snapshot (the full chain, not a split relabel).* A date relabel of `enc["split"]` is not enough:
- training rows come from the bundle's `train_candidates` (21 bundled candidates per query, original training split only; `data.py:179, 362-382`);
- `data.py:236-237` asserts that the encounter split equals `b.splits`;
- `validation_sample` is drawn from the old validation split (`data.py:387`).

So, per snapshot:
1. Regenerate the splits table, the candidate bundle and the 3,000-query selection sample (drawn from that snapshot's holdout, or all of it if smaller).
   - The bundle and splits come from `01_data_cleaning.ipynb`, not a script. Its code cells hard-code the split years: zero-based cell 15 (`np.select` on years 2017/2018/2019), cell 18 (uniform sampler `default_rng(CANDIDATE_SEED)` over training rows in order) and cell 27 (chronological-boundary check).
   - Cells 15–29 are therefore **extracted into a parameterised script** taking the split dates as arguments. With the original dates it must reproduce the current bundle exactly (array-level, part of check (a)).
   - Disclosed: the RNG stream moves with the query set, so overlapping 2016 queries get different sampled negatives in S16 than in S17 or the final model.
2. Rebuild v1 → v5 in order. Every stage refits on the snapshot's training part:
   - v1: diagnosis vocabulary, encoders, scalers;
   - v2: patient-distance scaler;
   - v3: insurer and affiliation (as-of, unchanged logic);
   - v4: severity and race vocabularies;
   - v5: MIPS imputation and scaler.

   The STEP10 "training-period" references (busiest-10% load, yardstick imputation) are also refitted.
3. **Regression check (a):** with the current split dates, every array and fitted object equals `three_node_v5`. Arrays and objects are compared, not pickle hashes, because pickle bytes need not be deterministic.
4. **Leakage check (b):** in a scratch copy, perturb data after S16's cutoff; S16's fitted objects and training arrays must not change. The perturbations cover everything that feeds the scalers and vocabularies (`data.py:269-325`):
   - diagnosis codes and MIPS values;
   - patient-context values (age, insurer);
   - specialist and PCP history rows;
   - pair counts.

*Seeds and noise bands.*
- GNN: 2 seeds (42, 1) for S16, S17 and S19. For S18, 3 runs: s42, s1 and a duplicate of s42.
- No-graph: 2 seeds for every snapshot.
- Every snapshot comparison is read against **that snapshot pair's own seed spread**, using the larger of the two snapshots' seed gaps (G2 logic), plus a **holdout bootstrap CI** per snapshot, since holdout sizes differ (Oct–Dec 2016 is small).
- If Grace becomes available, runs go there in parallel; nothing else changes.

*Measures* (all descriptive except T10):
- **M1. What drives recommendations.**
  - Value function, model-independent: f = score(referred) − mean score over that referral's eligible set ("how strongly the model favours what the PCP actually chose, relative to the alternatives"). The A9 per-input groups (36) and 12 permutations are kept.
  - When a specialist-input group is absent, it is set absent on **every eligible candidate together**, not only on the two A9 rows.
  - Sample: only referrals whose referred specialist is inside the eligible set (about 12% are beyond 25 mi).
  - Only |φ| shares are reported. Logit scales differ between snapshots, so signed φ are not compared.
  - **Primary:** 60 first-time referrals in choice specialties from **each snapshot's own holdout** (out-of-sample, drawn with a fixed seed).
  - **Secondary:** a shared set of 60 first-time referrals from 2016 Jan–Sep, scored by every snapshot. These are in-sample for all four. Two confounds are disclosed: S16 fits these labels most tightly (memorisation), and the cases sit on a thin 2016 graph, which works against detecting greater graph reliance.
  - The **PCP-block share** (the only input group that reaches the score solely through message passing, `three_node_model.py:149`) is reported with a patient-cluster CI. Patient-history and predecessor-gap groups are explicit encounter inputs and are reported separately, not as "graph-carried".
- **M2. What it learns about specialists.**
  - Specialist embeddings are computed from a query-independent batch: cutoff = snapshot end date, no query encounters, eval mode.
  - **Within-specialty structure (replaces the draft 4 specialty purity; see T11).**
  - Linear CKA between consecutive snapshots on shared specialists, **calibrated** against CKA between S18's seeds (s42 vs s1, s42 vs dup).
  - Per-specialist Procrustes displacement, read against that specialist's own displacement across S18 seeds (low-volume embeddings are noisier regardless of growth) and against the change in its referral volume.
  - Central specialists by PageRank on the referral graph, with overlap between snapshots. Embedding norm is dropped: LayerNorm before ReLU makes norms nearly constant.
- **M3. How recommendations behave.** On each snapshot's holdout (STEP10 primary cohort) and on the shared set:
  - new-to-PCP share of the top 5;
  - busiest-10% top-1 share;
  - MIPS gain at λ = 0 and 0.3;
  - median distance;
  - R@5 in excess of random (holdout only).
- **M4. How much each node type takes from the graph (descriptive).** Primary: HGTConv's learned skip gate `sigmoid(skip[node_type])` per layer, plus the ratio of message norm to residual norm per node type (cheap, no model change). Secondary: per-edge-type attention mass through an inference-time wrapper on `HGTConv.message`. PyG 2.8 has no `return_attention_weights`. Alpha is a softmax over all incoming edges of each destination node, pooled across edge types, so edge order is mapped back to type. Attention mass is labelled descriptive, not importance.

*Pre-stated signals (descriptive, each linked to an architecture choice):*
- **T10. Graph value grows with data.**
  - Primary: the paired difference (GNN − no-graph, both seed-averaged) in holdout first-time R@5 in excess of random. It fires if the **lower bound of the S19 − S16 bootstrap CI** exceeds the larger of the S16 and S19 seed spreads of the paired difference, **and** no adjacent step falls by more than that seed spread. The bootstrap resamples the two holdouts independently, since they contain different referrals.
  - The SHAP clause is dropped from the trigger; the graph-carried share is reported descriptively.
  - Disclosed confounds: holdout season (S19 covers Aug–Oct 29, the others Oct–Dec) and eligible-set size. Excess over random and the pairing reduce them but do not remove them.
  - Justifies: keeping and investing in message passing ("the graph pays off at scale"). Not firing argues for feature-based models at this data size.
- **T11. Message passing does not add useful structure within a specialty** (replaces the draft 4 purity signal, which was decided in advance: specialty is an encoder input and HGT keeps a residual).
  - Measure: for specialist pairs in the same specialty, the AUC with which cosine similarity predicts **co-referral substitution** The labels come from a signal the embedding did not consume. Using shared referring PCPs before the cutoff would be circular: HGT passes exactly that information into the specialist embeddings.
    - **Candidate pairs:** same specialty, **no shared referring PCP before the cutoff**.
    - **Positive pair:** the two specialists first receive referrals from the same PCP for the same diagnosis category in the **6–12 months after the cutoff** (S19 has no after-window, so T11 is read at S18; S19 is reported where data allow).
    - **Negative pairs:** candidate pairs without that. They are matched to positives on the decile of distance between the two practices **and** on the decile of referral volume (degree), since high-volume specialists share PCPs more often and embeddings carry degree.
    - This measures structure, not forecasting accuracy. It is computed on (i) the encoder output before HGT, (ii) after HGT, and (iii) the no-graph specialist encoder.
  - It fires at S18 if (ii) − (i) ≤ the band (the S18 seed spread of the same AUC, across s42, s1 and dup).
  - Justifies: explicit within-specialty substitution structure (specialist–specialist edges from co-referral, or a substitution-aware scorer) instead of relying on HGT to recover it.
  - Specialty purity relative to chance (specialty frequencies) is reported descriptively only.

*Rural specialists (11.2 note).* Report how many of the 261 non-metropolitan specialists are ever eligible under the 25-mile cap before reading exposure.

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
   - snapshot regeneration code: notebook cells extracted into a parameterised script, plus splits, candidate bundle, selection sample and the v1→v5 chain with regression and leakage checks. The raw processing runs once per snapshot (×4). ≈ 3–4 h;
   - GNN runs: S16 ×2, S17 ×2, S18 ×3, S19 ×2 = 9 runs. S18 is about 2.5 h each and S19 about 3.5 h; GNN runs alone total ≈ 15–17 h;
   - no-graph runs: 8, faster, ≈ 5 h;
   - SHAP: 4 holdout sets + 4 shared-set runs (one seed each), ≈ 12 h (larger graphs and more candidates per evaluation in later snapshots).

   **Realistic total ≈ 33–35 h locally.** Run on Grace in parallel if the VPN works; locally it is a multi-night queue (resumable, stops between runs on battery).
6. 11.3b, only if T5 fires, ≈ 2.5 h.
7. 11.5.

Each step goes validation → decision entry → test, except 11.4(b)/(c), which are insight-only and have no test pass. S19's holdout and the T11 after-window are disclosed where they touch 2019.

## Open items for the user

1. Patient rurality: download the public USDA ZIP-RUCA file, or use the access-tertile fallback. Provider RUCA is in the source; patient RUCA is not.
2. Graduation year: download the public CMS Doctors & Clinicians file and join it on NPI, or keep snapshot Medicare tenure.
3. Package install: `lightgbm` (PyPI). `implicit` is no longer needed.
4. 11.4 redesigned per the user (2026-10-07): cumulative snapshots for insight, not forecasting, plus the fixed-model first look; the staleness checks are kept as supporting evidence.

## Amendment B1 (2026-10-07, user): pure-tabular baseline

**Recorded after** the 11.1 validation pass of the other baselines, and **before** any pure-tabular output and any 11.1 test output.

**Purpose (user).** Show whether handling the claims data strictly as a table falls short, so a graph approach is needed. Relationship information (referral histories, shared patients) is exactly what a graph structure contributes. A baseline that is handed those relationships as columns therefore cannot answer the question. The 11.1 "primary" LightGBM was such a baseline: the GNN's explicit inputs already include PCP–specialist referral counts and recency, the patient–specialist prior, insurer history and the patient-history block.

**New baseline: LightGBM, pure tabular** (`feature_set = "tabular"`, `referral_gnn/tabular.py`). Two seeds, same rows, objective, grid and selection as the other LightGBM variants.

Included, each claim row's own attributes only:
- encounter patient context (age and its flags), severity and diagnosis flags;
- encounter categoricals (gender, race, insurer, ACO status);
- disease-category multi-hot;
- specialist and PCP snapshots (provider attributes, including MIPS, tenure, volume, ratings and panel mix);
- PCP–specialist and patient–specialist distance;
- shared hospital affiliation. This one is borderline: it compares two providers' listed affiliations rather than referral history, and it is disclosed.

Excluded: the patient-history block, predecessor gap, PCP–specialist referral history, patient–specialist prior and insurer history.

**New primary contrast.**
- **T12 (tabular vs graph): GNN − LightGBM pure tabular.** Same primary metric (excess R@5, base configuration), same four-outcome rule, margin 0.02, PCP-cluster CI. The Holm family grows to ~10 primary contrasts; the conservative bound becomes α/10.
- **Ladder (descriptive), reported beside it:**
  - pure tabular → + history columns (the 11.1 "primary" LightGBM) → + 10 graph counts (the 11.1 "graph" LightGBM);
  - GNN without message passing → GNN.

  The ladder separates how much of any lift comes from relationship information at all, and how much from how it is modelled.

T8 and T9 keep their definitions. T9's "bar for new architectures" is unchanged.

## Amendment B3 (2026-10-08, reviewer round 5): no PRECEDES edges in the final configuration

The final model and every Step 11 run are trained with `--no-temporal` (`temporal_edges: False`). The patient's previous visit reaches the model only through the predecessor-gap and patient-history features. Corrections:
- **M4 edge types:** has_encounter and referred_to plus their reverses only.
- **11.4(d) frozen-graph level (2):** "PRECEDES edges" is replaced by the predecessor-gap features.
- **T5 proxy list:** "prior specialists via PRECEDES" becomes "the patient-history block".

No trigger definition changes.

## Amendment B4 (2026-10-08, user): tabular baselines only; the bar is the pure-tabular model

The user's purpose: compare the graph model against tabular handling of the data only. Hybrids (LightGBM with history columns, LightGBM with network counts) and the logistic regression trained on those columns are dropped from the comparison and the deck. They remain on disk and in the log as internal diagnostics only.

Tabular baselines are now:
- LightGBM on claim attributes only (B1);
- logistic regression on the same 117 attributes (`logreg_tabular`).

**T9's "bar every new architecture must beat" is redefined as the pure-tabular LightGBM.** It is judged on both:
- top-5 match within the eligible set (STEP10 cohort);
- first-time MRR over the full pool, the models' own tuning criterion, where the table currently leads (0.112 vs 0.088).

T8 and T12 are unchanged.

## Amendment B5 (2026-10-08, user): architecture improvement starts now (11.5 brought forward)

**Why now.** On 2018 (B4), the pure-tabular LightGBM:
- is level with the graph model on top-5 match within the eligible set (T12: +0.008, CI −0.005 to +0.022);
- beats it on first-time MRR over the full pool (0.112 vs 0.088).

The user prioritises architecture work over the remaining 11.4(c) runs, which have partial results; the snapshot queue is paused after the current run and resumed later. The other diagnostic triggers (11.2 strata, 11.3 probe) have not been run; changes proposed here are justified by 11.1 and 11.4(a) evidence only, and say so.

**D1 diagnostic first** (inference only, ≈ 20 min). On the 1,274 first-time selection queries (full pool), compare the GNN and the tabular LightGBM on:
- the specialty of the top-1 and top-5 (specialty match with the referred specialist);
- the target's rank within its own specialty.

This tells whether the table's full-pool advantage is choosing the right specialty or ranking within it. C2/C3 address the former; C1 the latter. D1 is reported before any C-run result is read; it does not gate the runs.

**Changes** (each is one configuration change against the final model: v5 inputs, unmasked, uniform negatives, same training/selection settings; two seeds, 42 and 1; trained on the main inputs):
- **C1 — less reliance on PCP history (training-time history dropout).**
  - Change: for each training query, with probability p, the PCP–specialist referral-history pair features (block `referral_history`, 5 columns) are set to their absent encoding for all 21 candidates. Inference is unchanged.
  - Grid: p ∈ {0.3, 0.6}, chosen on validation (4 runs).
  - Evidence: history columns add nothing to the tabular model (0.484 vs 0.481); only 23–25% of PCP–specialist connections recur the next year; 10.4 reliance showed PCP history drives the model's pull to familiar and busy specialists.
- **C2 — stronger condition × specialist matching in the scorer (skip + interaction).**
  - Change: the scorer receives, besides the post-HGT encounter and specialist embeddings and the pair features, the pre-HGT encoder outputs of both (skip connection) and their element-wise product (an explicit interaction term). Encoders and HGT are unchanged (2 runs).
  - Evidence: the tabular model's signal is specialist attributes (48% of gain), patient distance (15%) and diagnosis categories (11%), and it leads across all specialists; in the GNN these reach the scorer only after two message-passing layers mix them with neighbours.
- **C3 — disease-category nodes.**
  - Change: a fourth node type, category (the 33 categories + UNK).
    - Encounter → category edges from each encounter's diagnosis categories, query encounters included (their own diagnoses are known at the query).
    - Specialist → category edges built from historical encounters before the cutoff, so a specialist's case mix is carried by an explicit aggregated link instead of being averaged over hundreds of encounter neighbours.
    - The query encounter can then reach specialists through its categories in two hops.
  - Implementation: new graph builder and model config; the same leakage checks as the existing graph (strictly-before edges; the query's own label never used). 2 runs.
  - Note: the current graph already lets specialists aggregate their past encounters' diagnosis features (rev_referred_to). C3 makes the category structure explicit and lets the query use it directly.
- **C4 — recency** (lower priority; run only if 11.4(c)/(d) show staleness).
  - Change: training and scoring with referral-history edges restricted to the last 24 months.
  - Note: encounter nodes already carry a time encoding (cutoff − day), so the model can in principle down-weight old edges. C4 tests whether removing old edges helps.

**Success criterion** (pre-stated; adapts 11.5 to B4). Decided on 2018 validation, seed-averaged, PCP-cluster CIs. A change is a success if:
- (i) first-time MRR over the full pool exceeds the current GNN by more than the noise band (G2) **and** reaches at least the tabular LightGBM's 0.112, or its CI includes it;
- (ii) top-5 match within the eligible set is not below the current GNN by more than 0.02;
- (iii) MIPS gain at λ = 0.3, median travel and busiest-10% share are not worse than the current GNN by more than their noise bands.

Successful changes are then combined (C1 + C2 first). The best combination is confirmed once on 2019 against the tabular LightGBM, using the T12 rule and margin.

**Order and cost** (laptop, ≈ 45–60 min per run on the main inputs):
1. D1.
2. C1 (4 runs) and C2 (2 runs), queued as soon as the snapshot run in progress finishes.
3. C3 implementation and checks, then 2 runs.
4. Combination runs.
5. Resume the paused 11.4(c) snapshot runs (S19 GNN s1, S19 no-graph × 2) when the laptop is free.

### B5 revision 1 (2026-10-08, reviewer round 6; replaces the B5 success criterion and D1 definition; B5 then FROZEN)

**Judging set, held out from selection (blocking 1).** Every GNN run selects its checkpoint on the first-time part of the 3,000-query validation selection sample; the tabular LightGBM's 0.112 is the best of 12 configurations on the same queries. Those 1,274 queries are used for **selection only**.
- **Judging set J:** all 2018 first-time queries **outside** the 3,000-query sample. Cohort: first-time, **full pool** (every specialist first observed before the day), all specialties, including the non-choice ones that STEP10 excludes.
- The current GNN (s42, s1, dup), the no-graph model and the tabular LightGBM are re-scored on J, so the bar is not inflated by selection either.
- The noise band for full-pool MRR comes from J: GNN s42 vs dup, plus seed gaps.

**Success criterion (replaces B5's; blocking 2).** Paired contrasts, PCP-cluster bootstrap, four-outcome rule as T12. Margins: 0.005 MRR for full-pool MRR (≈ 5% of the bar) and 0.02 for top-5 match. Holm across the arms' primary contrasts (should-fix 6).

A change C is a **success** if:
- (i) full-pool first-time MRR on J: C − tabular LightGBM is "better" or "equivalent" (adjusted CI lower bound > −0.005), **and** C − current GNN exceeds the J noise band;
- (ii) STEP10-cohort top-5 match: C − tabular LightGBM not worse than −0.02 (CI lower bound > −0.02), **and** C − current GNN not worse than −0.02;
- (iii) MIPS gain at λ = 0.3, median travel and busiest-10% share: not worse than the current GNN by more than their noise bands.

**Combination rule (pre-stated).** Combine only arms that pass (i)–(iii). If none passes, combine nothing and report. 2019 confirmation covers both bar metrics; full-pool MRR on 2019 uses all first-time queries.

**D1, redefined (should-fix 1).** On J, for the current GNN and the tabular LightGBM:
- Exact decomposition of the target's pessimistic rank: rank = 1 + (other-specialty candidates scored ≥ the target) + (same-specialty candidates scored ≥ the target, excluding the target). Each term's mean is reported, together with its contribution to the MRR gap: the gap with the other-specialty term replaced by the other model's, and vice versa.
- Target specialty read strictly before the day (`specialty_strict`); unknown excluded.
- Stratified by choice, non-choice and non-specific specialties (STEP10 groups).
- **Reading rule:** if more than about 70% of the MRR gap sits in the other-specialty term, specialty choice is the problem and C2/C5 lead.
- A first pass on the selection queries (superseded by J; recorded only as a hint): top-1 specialty match is 0.25 for the table vs 0.21 for the GNN; MRR within the target's specialty is 0.278 vs 0.269. The gap looks like specialty choice.

**C1 (should-fix 2).** Kept, but with its own expected effect: more new-to-PCP picks and a lower busiest share, little change in full-pool MRR. It is not combined by default. Graph edges still carry referral history, so the dropout is partial. The dropout RNG is seeded from the run seed and recorded in the manifest and the resume state.

**C3 (should-fix 3/4).** Moved after C2 and C5, and run only if they do not close the gap. When built:
- query encounters receive **category → query** edges only (they stay receive-only);
- category nodes aggregate only historical encounters and specialists (day < cutoff);
- specialist ↔ category edges and their counts come from referrals before the cutoff;
- the isolate-mode check (`evaluate.py`, `isolate="query"`) must reproduce full-batch ranks exactly.

**C5 — new (should-fix 5): an explicit specialty step.**
- **C5a, inference-only upper bound, before any training.** Combine the tabular LightGBM's implied specialty distribution (softmax of its scores, summed per specialty) with the GNN's within-specialty ranking: score = log P_table(specialty) + GNN within-specialty log-softmax. Evaluated on J. If it reaches the table's full-pool MRR or beyond, C5b is justified.
- **C5b, training.** An auxiliary specialty-classification head on the encounter embedding (softmax over training-period specialties; cross-entropy on the referred specialist's specialty, as-of), trained jointly. Score = log P(specialty | encounter) + within-specialty score. Two runs.
- C5a is a diagnostic only. It uses the tabular model's outputs, so it is not a GNN result and never counts as the GNN beating the bar.

**Order:**
1. D1 and C5a on J (inference).
2. C2 and C1 runs (2 + 4).
3. C5b if C5a is promising.
4. C3 only if the gap remains.
5. Combinations per the rule.
6. 2019 confirmation.

**B5 is FROZEN with this revision.**

### B5 revision 2 (2026-10-08 23:45, user): judge on the four lenses, not MRR (restores the 09-28 reframing)

The user restated the 09-28 decision: MRR does not drive model decisions. Revision 1 made full-pool first-time MRR the primary success criterion. That was a drift back to the old framing; the reviewer tightened it but did not question it, and neither did I. **Revision 1's criterion (i) and the D1/C5a MRR decompositions are withdrawn as decision tools** (kept in the log as descriptive only).

**Criterion (replaces revision 1's (i)–(iii)).** STEP10 primary cohort (2018 first-time referrals, eligible set), seed-averaged, PCP-cluster CIs, Holm across arms. Both the base ranking and the 30% quality weight (λ = 0.3) are reported. A change C succeeds if, against the current GNN:
- it improves **quality** (MIPS gain at λ = 0.3) or **load** (busiest-10% top-1 share, lower) beyond the noise band,
- without losing more than the noise band on the other two, and
- without losing more than 0.02 on **fit** (top-5 match of the chosen specialist) or worsening **travel** (median distance) beyond its noise band.

The same lenses (plus the new-to-PCP share of the top 5) are reported for the tabular LightGBM as the reference. MRR is shown as a sanity line only. Checkpoint selection inside training is unchanged; it is a stopping rule, not a judging criterion.
