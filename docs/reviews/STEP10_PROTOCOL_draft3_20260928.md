# Step 10 protocol — MIPS-based evaluation of recommended connections. DRAFT 3 2026-09-28 (reviewer round 1 adopted; user decisions recorded; awaiting reviewer concurrence)

Follows the professor meeting of 2026-09-25 (PROJECT_LOG 2026-09-28). Replaces MRR as the criterion for model and analysis decisions. Draft 1 and its audit: `reviews/STEP10_PROTOCOL_draft1_20260928.md`, PROJECT_LOG 2026-09-28. Edited after freezing only by dated amendment.

Step numbering used everywhere (log, protocol, results): **10.0** protocol · **10.1** yardstick table + descriptive charts · **10.2** baseline on the yardstick (v4 checkpoints) · **10.3** v5 inputs + retraining · **10.4** filters and reliance ablations · **10.5** bias · **10.6** per-input SHAP · **10.7** load-aware recommendation (later).

## 1. Question and decision rule
Does the recommender propose specialists (connections) with higher MIPS than the specialist actually referred to, while staying consistent with the case? Where do those connections come from, what do they cost in travel, and how do they spread load? The observed referral is the comparison point, not assumed optimal.

**Primary result = a frontier, not a single number.** For each scorer family and setting: x-axis **fidelity** = within-specialty R@10 of the referred specialist; y-axis **MIPS gain** (top-5, conservative bound, §2.4). Distance from the patient and load are co-reported for every point.

**Admissibility rule (pre-registered; applied on validation only).** Every recommendation already satisfies the 25-mile cap (§2.2 rule 4). A configuration is admissible if (a) validation fidelity (R@10) ≥ PCP habit's validation R@10, and (b) its recommendations are no farther from patients than current practice: median and p90 of recommended distance (top-5 slots) ≤ the median and p90 of the referred specialist's distance on the same referrals (user decision 2026-09-28: distance is judged from the patient, never relative to the referred specialist). Among admissible configurations, the chosen one maximises validation MIPS gain (conservative bound). Checkpoints are never selected on MIPS.

**Primary contrast (one).** The chosen quality-aware model configuration vs current practice on MIPS gain; secondary: vs PCP-habit + MIPS re-sort at matched fidelity (the nearest admissible K). Everything else is exploratory and reported in full.

## 2. Definitions

### 2.1 Cohort and exclusion order (counts reported at each step; identical query set for every scorer)
First-time referrals (no earlier visit of the patient to that specialist). Exclusions, in order:
1. Referred specialist's specialty unknown (`missing`).
2. **Non-choice specialties** (user decision 2026-09-28: excluded): facility-based services the PCP does not choose a named doctor for — diagnostic radiology, interventional radiology, nuclear medicine, pathology, emergency medicine, anesthesiology, CRNA, hospitalist, critical care, ambulatory surgical center, radiation therapy centers (24.5% of all referral events). **Non-specific labels** — internal medicine, family practice, general practice, nurse practitioner, physician assistant, clinical nurse specialist, single/multispecialty group (13.4%). Both are reported as separate secondary tables, not in the primary analysis.
3. Referred specialist inactive under §2.2 rule 2 (share reported; these referrals stay in a sensitivity table with ever-active).
4. Referred specialist's MIPS is zero/imputed (no reference value; reported separately with the share).
5. Eligible set (before the distance cap) smaller than 5.
6. Missing patient ZIP: excluded from travel metrics only.
`mips_rerank.py`'s rules (drop referrals whose target is outside the pool; ≥ 10 scored candidates) are replaced by this list.

### 2.2 "Fits the case" — the eligible set for each referral
A specialist is eligible if **all** hold:
1. **Same specialty** as the referred specialist, both read **strictly before** the referral date (`specialty_asof` currently reads on-or-before; changed for this step and checked).
2. **Active**: at least one claim in the **365 days** before the referral date (sensitivity: ever active).
3. **New to the patient.**
4. **Distance from the patient ≤ 25 miles** (patient ZIP centroid to specialist location; 86% of actual first-time referrals fall within it). If fewer than 5 eligible specialists lie within 25 mi (≈ 15% of referrals), the list is filled with the nearest eligible specialists beyond 25 mi; these referrals are flagged and reported separately. Missing patient ZIP → no cap applied, flagged, excluded from distance metrics. Caps of 10 and 50 mi and no cap are the 10.4 sweep. The referred specialist need not be inside the cap. MIPS-sorted breaks ties by distance.
**Secondary eligible set:** the above ∩ specialists with prior claims under the patient's insurer (as-of) — the feasibility check.
**Reported plausibility shares for every list:** prior insurer history; prior encounters in the query's disease category (as-of); new to the PCP.

### 2.3 MIPS
Raw points (0–100); no percentiles in reported results. Zeros are "no MIPS score (0 in source)"; the professor treats them as new doctors, footnoted with the tenure contradiction (§7).
**Model input (v5, as-of):** for each specialist on each day, the mean of the distinct **positive** values seen strictly before that day; flags `mips_none` (only zeros seen so far; value imputed with the specialty mean of positive values over specialists active in the training period), `mips_multi` (> 1 positive distinct value), `mips_partial_zero` (a zero alongside positive values); input `mips_range` (max − min of positive values, 0 if single).
**Yardstick (evaluation):** one fixed value per specialist = mean of all distinct positive values, applied identically to the referred specialist and every candidate. Scorers never rank on the yardstick; MIPS-sorted ranks on the as-of input value.
**Bounds:** *conservative* = recommended at their minimum distinct value, referred at their maximum, imputed recommended = 0; *generous* = the reverse, imputed at the specialty mean. A gain is claimed only if it holds under the conservative bound. The point estimate uses means with imputed recommended at the specialty mean.

### 2.4 Metrics (per scorer, per split)
**Quality.** MIPS gain top-5 (mean of top-5 − referred) and top-1, in points: mean, median and a histogram of per-referral gain. Share of top-1 better / equal / worse than referred (imputed slots count as not better). Stratified by referred MIPS band (< 50, 50–< 90, 90–< 100, = 100) and by `mips_multi`. **Low-MIPS avoidance**: share of top-1 below the training-period p10 of its specialty. **Headroom captured**: gain ÷ (best eligible MIPS − referred MIPS), over referrals with headroom > 0.
**Connections.** New-to-PCP share of top-5 slots (reference: the referred specialist's new-to-PCP share, ≈ 36%). Good new connections: slots new to the PCP and better than referred. Share of slots on `mips_none` specialists.
**Distance (from the patient).** (i) Distance to each recommended specialist (top-1; top-5 slots): median, p90, histogram, shares beyond 10 / 25 mi; the same for the referred specialist on the same referrals. (ii) **Excess over nearest** = recommended distance − distance to the nearest eligible specialist (median, p90); the referred specialist's excess is the reference (training sample: median 4.5, p90 20.7 mi). No metric is defined relative to the referred specialist's distance. Missing ZIP → excluded from distance metrics.
**Load (within specialty).** Top-1 recommendations compared with actual referrals, one per referral; top-5 as slot shares (1/5 per slot). Gini **within each specialty** over its eligible specialists (zeros included), referral-weighted average across specialties. Share of recommendations to specialists above their specialty's training-period p90 first-time volume. **Congestion ratio** per specialist = recommended load ÷ training-period first-time load (median, p90 across specialists; the same definition is reused for 10.7). Number of distinct specialists recommended.
**Fidelity.** Within-specialty R@10 (frontier axis); MRR and R@5 reported for continuity.

### 2.5 Scorers
1. **Current practice** (reference row).
2. **PCP habit** (ties by distance, then fixed-seed random).
3. **Random**: exact expectation (top-5 mean = eligible-set mean; share better = share of the eligible set above the referred specialist).
4. **MIPS-sorted** on the as-of input value, ties by distance.
5. **Distance-sorted** (nearest first; the minimum-distance reference).
6. **Final model** — v4 masked (seeds 42, 1) and v4 unmasked (seeds 42, 1) in 10.2; v5 masked (seeds 42, 1) in 10.3; v5 unmasked (seed 42) for 10.5 only.
7. **Quality-aware re-sorts** for scorers 2, 5 and 6: take the scorer's top K within the eligible set, re-sort by as-of MIPS (ties by the scorer's own order), K ∈ {5, 10, 20, 50, all}. K = 5 reproduces the scorer's own list reordered; K = all equals scorer 4.
The **final configuration** for 10.4–10.6 is named in the decision record (§3) before any test output.

## 3. Splits, test procedure, uncertainty
- **Two passes.** Pass 1 produces **validation-only** tables. The choice (input version, scorer, K, cap) is written to `runs/analysis/step10_decisions.json` with a timestamp and logged. Pass 2 produces the test column; the analysis script refuses `--split test` unless the decision file exists and its hash is recorded in the output. Any later change is a dated amendment. Every final table shows validation and test side by side.
- Validation = all 2018 first-time referrals after §2.1 exclusions. The 3,000 queries used for checkpoint selection are included; results excluding them are reported as a sensitivity. The 2019 test year was inspected once on 2026-09-25; disclosed on every test table.
- Uncertainty: patient-cluster bootstrap, 1,000 resamples, paired between scorers; PCP-cluster bootstrap as a sensitivity (119 PCPs). CIs only, no p-values. Two seeds per learned model plus a same-seed nondeterminism band for the MIPS metrics (one duplicated run); differences below max(band, seed gap) are not attributed. Bias groups: ≥ 100 patients.

## 4. Order of work
1. **10.0** Reviewer concurrence → FREEZE.
2. **10.1** Fixed per-specialist yardstick table (aggregate checks only) and descriptive charts. Charts are aggregate: binned/hexbin plots, specialties with < 11 specialists suppressed, no provider identifiers. Charts: MIPS histogram overall and by specialty; referred vs eligible-set MIPS; per-referral headroom; Lorenz curve of first-time referral load within specialty; MIPS vs load (binned); referral distance; zero-MIPS referral share; spread of multi-valued MIPS.
3. **10.2** Scorers 1–7 on v4 checkpoints with the yardstick; pass 1 (validation), decision record, pass 2 (test). `mips_rerank.py` updated (exclusions, yardstick source, bounds) or replaced.
4. **10.3** v5 inputs (§2.3 as-of rules; other attributes keep the existing ambiguity rule) with checks; retrain v5 masked (s42, s1) and unmasked (s42); checkpoint selection unchanged (first-time MRR); re-run 10.2 scorers.
5. **10.4** Candidate filters (distance caps 10 / 50 mi / none vs the primary 25 mi; insurer set) and **reliance diagnostics** (inference-time removal of MIPS, PCP history, affiliation, insurer history, distance). Reliance diagnostics describe the model; they are not configurations to keep. Keeping or dropping an input requires retraining and a decision-record entry.
6. **10.5** Bias: §2.4 metrics by patient gender, race, age band, insurer, area; group gap in practice vs in recommendations; masked vs unmasked.
7. **10.6** Per-input SHAP on the final configuration. One input = the attribute's value, missing and ambiguity columns together (~60 groups, ≈ 4× the 14-group cost). Explained quantity: model score of the top-1 minus the score of the referred specialist. CIs reported; correlated inputs (volume, practice size, panel counts) flagged as unstable.
8. **10.7** Load-aware recommendation, only after 10.2 reports the baseline load profile.

## 5. Expected result stated in advance
The base model imitates observed referrals; its top-5 was slightly below the referred specialist's MIPS on 09-25. A MIPS gain is not expected from the base model; it is expected from quality-aware re-sorting among plausible candidates, with a cost in fidelity and possibly travel. The frontier is the result.

## 6. Caveats stated in every table
Higher MIPS is an allocation property, not proven clinical superiority. MIPS is a single snapshot of unknown year applied across 2015–2019; gaps near the ceiling (95 vs 100) are not equivalent to gaps in the tail — interpret by band. Multi-valued MIPS is averaged over practices whose scores differ by a median of 29 points; the referral's own practice score is not used (symmetry with candidates). Distances use patient ZIP centroids, not addresses. Zero-MIPS specialists are labelled "no MIPS score"; the professor's "new doctor" reading is contradicted by tenure. Race is computed/imputed. Patient ZIP is fixed per patient; 5% of rows disagree with the recorded distance by > 20 mi. Test year previously inspected.

## 7. Resolved points (reviewer round 1)
- **Zero MIPS**: impute + flag as directed; conservative bound uses 0; label "no MIPS score (0 in source)"; ask the professor whether the source fills unscored clinicians with 0.
- **Multiple MIPS**: mean of positive distinct values, as-of for inputs; range and flags as inputs; paired min/max bounds; stratify by `mips_multi`.
- **Checkpoint selection**: first-time MRR (the base model is the fidelity model); quality enters by an explicit re-sort chosen by the §1 admissibility rule.

## 8. Freeze record
Round 1 (Opus, 2026-09-28): 8 blocking, 14 should-fix, 6 notes — all adopted. User decisions 2026-09-28: exclude facility-based and non-specific specialties; distance judged from the patient (25-mile cap with nearest-beyond fill, plus no-worse-than-practice median/p90 admissibility). Reviewer concurrence: pending.
