# Step 10.1 — MIPS yardstick and descriptive charts (2026-09-29)

Protocol: `STEP10_PROTOCOL.md` (frozen 2026-09-28), §2.1–2.3 and §4 step 2. Scope: training + validation years (2015–2018); **the 2019 test year was not used**. Aggregate only; specialties with < 11 specialists suppressed; no provider identifiers.

Code: `scripts/build_mips_yardstick.py` (raw workbook read-only → `prepared_data/mips_yardstick_v1/`), `referral_gnn/step10.py` (shared cohort/eligibility rules), `scripts/step10_descriptive.py` (charts → `runs/analysis/step10_1/`, numbers in `summary.json`).

## Yardstick (per specialist, 6,262 specialists)
| | Count |
|---|---|
| No MIPS score (only zeros in source) | 247 (3.9%) |
| One positive score | 5,127 |
| More than one positive score | 888 (spread median 27.5, p90 75.1 points) |
| A zero alongside positive scores | 48 |
Imputation value for "no score": specialty mean over specialists active by 2017-12-31 (63 specialties; overall 81.8).

## Cohort funnel (first-time referrals)
| Step | Training | Validation |
|---|---|---|
| First-time referrals | 27,699 | 16,209 |
| Unknown specialty (strictly before the day) | 4,959 | 962 |
| Facility-based specialty | 9,828 | 6,675 |
| Non-specific label | 2,223 | 1,933 |
| Referred specialist inactive (no claim in the past 365 days) | 151 | 211 |
| Referred specialist has no MIPS score | 951 | 563 |
| Fewer than 5 eligible specialists | 115 | 26 |
| Patient ZIP missing/unmapped | 489 | 289 |
| **Primary cohort** | **8,983** | **5,550** |
Notes: facility-based services are 41% of first-time referrals in validation (a new radiologist or pathologist is usual). The training year's large "unknown specialty" count is a start-of-data effect of the strictly-before rule (a specialist's first claim day has no earlier label). In validation, 8.4% of referrals needed the fill beyond 25 miles; median eligible set 49; the referred specialist is within 25 mi for 87%.

## Findings (primary cohort, 2015–2018, n = 14,533)
1. **MIPS is bunched at the top** (chart 01): median 97, 38% of eligible specialists score exactly 100; 7.2% have no score. In several specialties almost everyone has the same score (e.g. medical oncology and hematology/oncology show no spread, chart 02) — likely group-reported scores, so MIPS cannot separate specialists there. Psychiatry, podiatry and PM&R have the widest spread.
2. **Current referrals are MIPS-neutral** (chart 03): a randomly picked eligible specialist scores higher than the referred one 40% of the time, the same 24%, lower 36%; referred mean 83.7 vs average option 85.4 (−1.7 points). Consistent with the percentile result of 09-24, now in raw points.
3. **Room for improvement is limited but real** (chart 04): 37% of referrals had no higher-MIPS option within 25 mi (mostly referred specialists already at 100); where there was room, the median was 16 points. This is an upper bound (best single option, ignoring fit and load).
4. **Load is concentrated** (chart 05): the busiest 10% of specialists receive 46% of a specialty's first-time referrals on average (range 28% cardiac electrophysiology to 63% optometry; referral-weighted Gini 0.65).
5. **The busiest specialists do not have higher MIPS** (chart 06): mean MIPS by within-specialty load third — no referrals 87.3, low 88.5, middle 86.1, high 84.2 (95% CIs overlap between neighbours; high vs low separated). Load does not follow quality; spreading load need not cost MIPS.
6. **Patients are sent farther than necessary** (charts 07–08): referred specialist median 9.6 mi (p90 28.1) vs nearest eligible 3.8 mi; the referred specialist was the nearest option (≤ 1 mi extra) for 28%; median extra 4.8 mi, p90 20.2 mi.
7. **Specialists with no MIPS score are not avoided** (chart 09): they are 6–8% of options and receive 6–9% of referrals, slightly more than their share since 2016.
8. **Multi-valued MIPS is mostly outside the choice specialties**: 183 of the 888 multi-valued specialists appear as eligible options (chart 10; median spread 17, p90 68 points).

## Implications for 10.2
- The MIPS gain available to any recommender is bounded: about a third of referrals already went to the top score available nearby, and ceiling bunching compresses gains; results must be read by referred-MIPS band (§2.4).
- Findings 4–5 suggest a quality-aware recommender could also spread load without sacrificing MIPS — to be tested with the congestion ratio in 10.2.
- Distance matters on its own: current practice is ~5 mi farther than the nearest option at the median; the admissibility rule compares recommendations to that practice distance.
