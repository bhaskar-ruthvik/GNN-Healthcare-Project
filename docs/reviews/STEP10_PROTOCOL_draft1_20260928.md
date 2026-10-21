# Step 10 protocol — MIPS-based evaluation of recommended connections. DRAFT 2026-09-28 (awaiting reviewer audit)

Follows the professor meeting of 2026-09-25 (see PROJECT_LOG 2026-09-28). Replaces MRR as the criterion for model and analysis decisions. Edited after freezing only by dated amendment.

## 1. Question
Does the recommender propose specialists (connections) with higher MIPS than the specialist actually referred to, while staying consistent with the case? Where do those connections come from (new to the PCP or not), what do they cost in travel, and how do they spread referral load? The observed referral is **not** assumed to be the best choice; it is the comparison point.

## 2. Definitions

### 2.1 Cohort
First-time referrals: the patient has no earlier visit to that specialist. Queries whose referred specialist has unknown specialty are excluded (count reported). Repeat visits are out of scope for the MIPS yardstick (the patient is returning to a known doctor).

### 2.2 "Fits the case" — the candidate set for each referral
A specialist is an eligible recommendation for a referral if **all** of the following hold:
1. **Same specialty** as the specialist the PCP actually referred to (specialty as recorded strictly before the referral date). This stands in for "the PCP has decided the kind of doctor needed". Limitation: subspecialty is not recorded (e.g. interventional vs electrophysiology cardiology).
2. **Active**: has at least one claim in the data strictly before the referral date.
3. **New to the patient**: the patient has not seen them before (consistent with the cohort).
4. **Distance**: primary analysis has **no distance cap** (the dataset is one DFW network). Distance from the patient's ZIP is reported for every recommendation. Caps of 10 / 25 / 50 miles are the Step 4 sweep. The referred specialist does **not** need to be inside the cap; the comparison is on MIPS.

The **recommendation** is the scorer's top-K within this set (K = 1 and K = 5). For the learned model, being in its top-K is the "consistent with the case" signal beyond rules 1–4: it combines diagnosis, PCP history, insurer history, affiliation and distance.

### 2.3 MIPS value (inputs version v5; see §4)
Raw MIPS points (0–100). No percentiles in reported results.
- Specialist with a single MIPS value: that value.
- Specialist with several values (14.8%; values differ by practice/group, same ZIP, on the same days — not a time series): the **mean of the distinct values**, plus a `mips_multi` flag and the `mips_range` (max − min). Sensitivity bounds: the minimum and the maximum of the distinct values.
- MIPS = 0 (3.9% of specialists all-zero): treated as **missing / new doctor** per the professor; value imputed with the specialty mean of specialists with a positive score, computed from specialists active in the training period only; flag `mips_new`.
- MIPS is a single snapshot per specialist-practice across 2015–2019 in this source; the as-of rule is kept (value only for specialists already active) but no year-specific score exists.

### 2.4 Metrics (per scorer, per split)
**Quality (primary)**
- **MIPS gain, top-5**: mean MIPS of the top-5 minus the referred specialist's MIPS, in points. Computed over referrals where the referred specialist has a real (non-imputed) score; imputed recommendations enter the mean with their imputed value and are also reported separately.
- **MIPS gain, top-1** and **share better**: share of referrals whose top-1 has higher MIPS than the referred specialist (and share equal, given the bunching at 100).
**Connection type**
- **New-connection share**: share of top-5 slots on specialists the PCP has never referred to before the referral date.
- **Good new connections**: share of top-5 slots that are both new to the PCP and higher-MIPS than the referred specialist.
- **New-doctor share**: share of top-5 slots on `mips_new` specialists.
**Cost**
- **Extra travel**: recommended distance minus referred distance (miles, median and p90), from the patient's ZIP.
**Load (Step 3 onward)**
- Distribution of recommendations per specialist vs actual first-time referrals per specialist, over a split: Gini coefficient, share of recommendations going to the 10% most-referred specialists (defined on training), number of distinct specialists ever recommended, maximum per-specialist load.
**Sanity only (not a decision criterion)**: within-specialty MRR / R@5 / R@10, reported so that a model that stops resembling any plausible referral is visible.

### 2.5 Scorers compared
1. **Current practice** (the referred specialist; the reference row).
2. **PCP habit**: rank eligible specialists by how often this PCP referred to them before.
3. **Random** within the eligible set (averaged over 20 draws).
4. **MIPS-sorted**: rank eligible specialists by MIPS alone (ties by distance). The ceiling on MIPS that ignores the case; used to show what the graph trades off, not as a recommender.
5. **Final model**, without and with patient gender and race (existing v4 runs, seeds 42 and 1), then v5 runs.

## 3. Splits, test indicators and uncertainty
- **Every table reports validation (2018) and test (2019) side by side.** Validation uses all first-time referrals in 2018 (not the 3,000-query subset).
- **All choices are made on validation** (which input version, which distance cap, which ablation to keep). Test is reported next to it and never used to choose. The 2019 test year was already inspected on 2026-09-25; every test number carries that disclosure.
- Uncertainty: patient-cluster bootstrap (1,000 resamples), paired between scorers on the same referrals; two seeds for learned models, with differences smaller than the seed gap reported as not attributable.

## 4. Order of work
1. **Freeze** this protocol after reviewer audit.
2. **v5 inputs**: §2.3 MIPS rules in the preparation step (MIPS only; other attributes keep the existing ambiguity rule). Checks: all-zero specialists → flag 1 and imputed value; multi-valued → mean of distinct values, flag 1, range > 0; single-valued unchanged; no validation/test information in the imputation means.
3. **Descriptive charts** (aggregate only): MIPS distribution overall and by specialty; referred MIPS vs eligible-set MIPS; referral load per specialist (Lorenz curve); MIPS vs load; referral distance; new-doctor referral share; spread of multi-valued MIPS.
4. **Baseline on the yardstick**: scorers 1–5 with existing v4 checkpoints (MIPS for evaluation from v5 rules), validation + test. Then retrain v4 configuration on v5 inputs (two seeds) and repeat.
5. **Ablations** judged on MIPS gain, new-connection share, extra travel and load: distance caps 10/25/50/none; inference-time removal of MIPS, PCP history, affiliation, insurer history, distance.
6. **Bias**: §2.4 metrics by patient gender, race, age band, insurer and area; gap between groups in practice vs in recommendations; with vs without gender and race.
7. **Per-input SHAP** on the final model: every specialist attribute, PCP attribute and patient-context input as its own group.
8. **Load-aware recommendation** (later): only after step 4 shows the baseline's load profile.

## 5. Expected result stated in advance
The model is trained to imitate observed referrals; its top-5 was slightly below the referred specialist's MIPS on 09-25. Without MIPS entering the ranking, a MIPS gain is not expected from the baseline. A gain is expected only from quality-aware ranking among the model's plausible candidates (or from a model given a quality objective), with a cost in resemblance to observed referrals and possibly in travel. Reporting this trade-off is the result, not a failure.

## 6. Caveats stated in every table
Higher MIPS is an allocation property, not proven clinical superiority. MIPS is a single snapshot of unknown year applied across 2015–2019. Multi-valued MIPS is averaged over practices whose scores differ by a median of 29 points. Zero-MIPS specialists are treated as new per the professor, although their recorded years in Medicare do not indicate they are new (§7). Race is computed/imputed in the source. Test year previously inspected.

## 7. Open points for the reviewer and professor
- All-zero MIPS specialists have a median of 11 years in Medicare (others 10) and none have ≤ 2 years: zero may mean "no MIPS score" (exempt, non-reporting, APM) rather than a new doctor. The handling (impute + flag) is the same either way; only the label changes.
- The multiple MIPS values co-occur on the same days and map one-to-one to different practice sizes (98.9%), consistent with one score per practice (TIN). Averaging hides a median 29-point spread; the range is kept as an input and min/max are run as sensitivity bounds for the MIPS gain.
- Checkpoint selection for v5 training: keep the existing first-time MRR selection as a fit criterion (the model must still learn what plausible referrals look like), or select on a MIPS criterion (risks a degenerate "sort by MIPS" model)?
