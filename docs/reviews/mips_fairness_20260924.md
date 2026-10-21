# MIPS allocation and demographic masking — Step 9 results (2026-09-24)

Protocol: `STEP9_PROTOCOL.md` (frozen after independent audit). Inputs v4 = v3 + severity + patient race + MIPS missing-as-low. Models: summaries-only three-node GNN, uniform negatives, selection on first-observed validation MRR. All numbers on the fixed validation subset (3,000 queries; first-observed cohort 1,274, of which 1,192 have the chosen specialist in the specialty-given pool) unless stated. Percentile = mid-rank MIPS percentile among candidates of the chosen specialist's specialty (unseen by the patient) with a usable MIPS; missing/ambiguous never converted to numbers; pools with < 10 scored candidates excluded.

## Definitions the reader needs
- **Specialty-given pool**: candidates in the chosen specialist's specialty that the patient has not seen. Uses the true specialty (upper-bound setting; realistic when the PCP has decided the specialty).
- **MIPS state at referral time**: read from the same strictly-earlier provider snapshot the model sees. Three states: value / missing (no earlier score) / ambiguous (conflicting values). Among chosen specialists: 71% value, 13% missing, 16% ambiguous.
- **Caveats stated once**: MIPS is an input to every learned scorer; higher recommended MIPS is an allocation property, not clinical superiority. MIPS source years are unvalidated; a performance-year score is not public when that year's referrals occur. Race is a computed (imputed) field. ~31% of pool candidates tie with the observed score (bunching), handled by the mid-rank rule.

## P1 — Current practice is MIPS-neutral at every complexity level (pre-registered, model-free)
43,908 training + validation first-observed events.

| Severity | Events | Chosen specialist's MIPS percentile, mean [95% CI] | Median | Chosen specialist MIPS missing/ambiguous |
|---|---|---|---|---|
| Simple | 19,162 | 0.480 [0.476, 0.485] | 0.58 | 28% |
| Moderate | 16,844 | 0.474 [0.468, 0.479] | 0.49 | 30% |
| Complex | 7,867 | 0.487 [0.479, 0.494] | 0.52 | 29% |

**Primary contrast Complex − Simple: +0.006 [−0.003, +0.015].** A random pick would sit at 0.50. Current referrals do not favour higher-MIPS specialists for complex disease, nor for any severity. The hypothesis "quality is considered for high-complexity referrals but not for medium/low" is not supported: quality, as measured by MIPS, is not visibly considered at any level.

## P3 — The model barely uses MIPS (pre-registered reliance diagnostic)
Inference-time ablation (MIPS forced to missing for every candidate) on `v4_unmasked_s42`, ΔMRR on the specialty pool:

| Severity | n | Δ MRR (model − ablated) |
|---|---|---|
| Simple | 524 | −0.001 [−0.005, +0.002] |
| Moderate | 466 | +0.001 [−0.004, +0.007] |
| Complex | 201 | +0.006 [−0.004, +0.017] |

Complex − Simple reliance contrast: +0.007 [−0.005, +0.019]. Consistent with P1: a model trained to reproduce referrals learns almost no MIPS dependence, with a directional but non-significant lean in complex cases.

## P2 — Allocation: what each scorer recommends (validation first-observed, specialty-given pool, n = 1,192)
| Scorer | MRR | R@5 | R@10 | Observed pct | Top-1 pct | Top-1 MIPS missing | Top-5 (excl. observed) pct | Top-5 missing |
|---|---|---|---|---|---|---|---|---|
| v4 unmasked (seed 42) | 0.298 | 0.409 | 0.551 | 0.487 | 0.445 | 18% | 0.456 | 18% |
| v3 | 0.292 | 0.416 | 0.564 | 0.487 | 0.442 | 19% | 0.458 | 19% |
| PCP frequency | 0.248 | 0.357 | 0.466 | 0.487 | 0.449 | 15% | 0.454 | 16% |
| Popularity | 0.120 | 0.155 | 0.242 | 0.487 | 0.445 | 37% | 0.442 | 24% |
| v4 masked (seed 42) | 0.296 | 0.420 | 0.564 | 0.487 | 0.440 | 19% | 0.458 | 18% |

No scorer allocates higher-MIPS specialists than current practice; all recommend slightly below it (≈0.45 vs 0.49). By severity, the model's top-1 matches practice for Complex (0.500 vs 0.501) and Simple (0.466 vs 0.486) and falls below it for Moderate (0.397 vs 0.480). Popularity-based recommendation also recommends the most specialists with no usable MIPS (37% at top-1).

### P2 primary contrast — masking patient gender and race changes nothing measurable (seed 42)
Unmasked − masked, paired patient-cluster bootstrap on the specialty-given pool: MRR +0.002 [−0.008, +0.012]; top-1 MIPS percentile +0.002 [−0.010, +0.014]. On the full pool the masked run is marginally higher (first-observed MRR 0.085 vs 0.083), within run-to-run noise. Under the pre-registered rule (same sign across seeds and magnitude above the noise band) no effect is attributed to masking.

**Seed-1 repeats (both arms) and the attribution rule, applied.**
| Contrast (specialty-given pool, first-observed) | MRR | Top-1 MIPS percentile |
|---|---|---|
| unmasked − masked, seed 42 | +0.002 [−0.008, +0.012] | +0.002 [−0.010, +0.014] |
| unmasked − masked, seed 1 | −0.000 | +0.004 |
| within-arm seed gap, unmasked (42 − 1) | +0.001 | −0.000 |
| within-arm seed gap, masked (42 − 1) | −0.001 | +0.001 |
Rule (i) same sign across seeds: fails for MRR (+0.002 vs −0.000), holds for the percentile (+0.002, +0.004). Rule (ii) magnitude above max(nondeterminism band 0.006, within-arm seed gap ≤ 0.001): fails for both. **Conclusion: no effect of masking patient gender and race is attributable, on ranking or on allocated MIPS, with two seeds per arm.** All four runs land within 0.002 MRR of each other on the specialty pool (0.296–0.298) and within 0.008 on the top-1 percentile (0.437–0.445).


### Fairness slices (exploratory; specialty-given pool; groups with ≥ 100 patients)
| Patient group | n | Current practice: observed MIPS pct | PCP freq MRR | Unmasked MRR | Masked MRR | PCP freq top-1 pct | Unmasked top-1 pct | Masked top-1 pct |
|---|---|---|---|---|---|---|---|---|
| Female | 641 | 0.493 | 0.273 | 0.324 | 0.335 | 0.459 | 0.440 | 0.444 |
| Male | 551 | 0.478 | 0.219 | 0.267 | 0.250 | 0.438 | 0.452 | 0.436 |
| Hispanic | 134 | 0.465 | 0.260 | 0.284 | 0.300 | 0.439 | 0.415 | 0.428 |
| Non-Hispanic Black | 421 | 0.483 | 0.254 | 0.308 | 0.308 | 0.454 | 0.450 | 0.442 |
| Non-Hispanic White | 472 | 0.487 | 0.247 | 0.301 | 0.287 | 0.440 | 0.454 | 0.446 |
(Asian and "other" groups below the 100-patient threshold; UNK race not shown.)

Readings: (i) Referrals for female patients are easier to predict for every scorer, including the history heuristic, so this is a property of referral patterns, not of the model. (ii) The quality gap present in current practice (observed MIPS percentile 0.465 for Hispanic vs 0.487 for White patients, −0.022) is reproduced and slightly widened by the unmasked model (−0.039) and roughly preserved by the masked model (−0.018); with 134 Hispanic patients these differences are within noise and are reported, not claimed. (iii) Masking does not narrow group differences, because the inputs that drive them (diagnosis, history, geography) are unchanged.


## Implications for the professor's question
1. Where might the MIPS distribution of recommendations be "better"? Not from any scorer that reproduces referrals: current practice is MIPS-neutral, so a faithful recommender inherits that. Raising allocated quality requires an explicit objective term (re-rank the specialty-pool top-K by a stated blend of model score and MIPS, reporting the hit rate traded for the MIPS gained), which is a design decision rather than a finding.
2. Missingness is common (29% of chosen specialists) and unevenly distributed across scorers; any quality-aware re-ranking must decide how to treat it, since "missing = 0" would systematically exclude small and non-reporting practices.
3. Severity does not moderate MIPS use in practice or in the model.

## SHAP — grouped Shapley attributions (v4 unmasked, seed 42; 20 first-observed + 20 repeat validation queries, 12 permutations; additivity error 5e-7)
Shares are mean |φ| over the 14 input groups; signed means show direction for the observed specialist's score. Historical graph fixed; "absent" = the model's missing encoding.

| Input group | First-observed: share | signed | Repeat: share | signed | Model's own top-1 (first-observed): signed |
|---|---|---|---|---|---|
| Diagnosis categories | 21.5% | +0.80 | 7.6% | +0.67 | +1.11 |
| PCP–specialist referral history | 21.3% | +1.41 | 15.7% | +1.81 | **+2.93** |
| Specialist attributes (excl. MIPS) | 9.6% | −0.23 | 3.8% | −0.35 | +0.19 |
| Distance (PCP and patient origin) | 9.0% | +0.40 | 4.0% | +0.37 | +0.55 |
| Shared hospital affiliation | 8.5% | +0.53 | 1.8% | +0.16 | +0.25 |
| Severity | 6.7% | +0.18 | 4.0% | +0.29 | +0.42 |
| Patient history summaries | 6.4% | +0.06 | 4.5% | +0.13 | +0.19 |
| Demographics (gender, race) | 4.5% | +0.03 | 2.5% | +0.18 | +0.22 |
| Patient context (age, insurer, ACO) | 4.1% | +0.04 | 2.0% | +0.16 | +0.04 |
| Insurer history | 4.1% | +0.27 | 3.9% | +0.45 | +0.35 |
| Time encoding + predecessor gap | 2.5% | +0.02 | 1.3% | −0.14 | −0.07 |
| PCP attributes | 1.3% | +0.05 | 0.8% | +0.06 | +0.07 |
| **MIPS** | **0.3%** | +0.01 | 0.3% | +0.02 | +0.03 |
| Patient–specialist prior | 0.0% | 0 | **47.8%** | +5.52 | 0 |

Readings. (1) For first-time referrals the score is driven about equally by the diagnosis and by the PCP's referral history with that specialist, then by distance and hospital affiliation; MIPS is 0.3%, consistent with P3. (2) For repeat visits one feature, the patient's own prior visits, is half of the attribution. (3) The model's own top picks for first-time referrals are reached mainly through PCP familiarity (+2.93 vs +1.41 for the observed specialist): the audit's familiarity dominance seen in attribution terms. (4) Demographics carry 4.5% of attribution in the unmasked model yet masking them changed nothing measurable, so their contribution is redistributed to correlated inputs. (5) Specialist attributes have a negative signed mean: on average, the observed specialist's own attributes pull its score slightly down relative to the missing encoding.

### SHAP across runs (share of attribution, first-observed / repeat)
| Group | v4 unmasked | v4 masked | v3 |
|---|---|---|---|
| Diagnosis | 21.5 / 7.6 | 16.9 / 6.4 | 23.2 / 9.8 |
| PCP–specialist history | 21.3 / 15.7 | 24.4 / 19.5 | 24.3 / 16.6 |
| Specialist attributes | 9.6 / 3.8 | 10.3 / 2.2 | 7.6 / 2.5 |
| Distance | 9.0 / 4.0 | 12.2 / 5.1 | 11.8 / 5.2 |
| Affiliation | 8.5 / 1.8 | 7.2 / 1.5 | 6.8 / 1.2 |
| Insurer history | 4.1 / 3.9 | 5.8 / 4.4 | 5.8 / 4.6 |
| Demographics | 4.5 / 2.5 | 0 / 0 (masked) | 1.6 / 0.7 (gender only) |
| MIPS | 0.3 / 0.3 | 0.5 / 0.3 | 0.7 / 0.2 |
| Patient–specialist prior | 0 / 47.8 | 0 / 49.6 | 0 / 51.0 |
The structure is stable across runs. Masking removes the 4.5% carried by demographics and redistributes it to PCP history, distance and patient context, which is why masked and unmasked rank identically. MIPS is below 1% of attribution in every model.
