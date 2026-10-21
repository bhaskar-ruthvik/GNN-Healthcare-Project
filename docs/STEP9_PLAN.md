# Step 9 plan — quality (MIPS) and fairness evaluation on the v4 model (DRAFT for review, 2026-09-24)

Deadline: presentable results by Friday 2026-09-25. Everything below runs locally on the M4 (Grace pending data permission).

## 0. What the professor asked (message of 2026-09-24, paraphrased)
- Keep MIPS as an input; treat missing MIPS as a signal: `quality_missing` flag and MIPS = 0 when missing (distinguish low from missing).
- Hypothesis: quality is considered in referrals for high-complexity disease but not for medium/low.
- Compare two models, demographics (gender, race) unmasked vs masked; both may use MIPS. Look for scenarios where the MIPS distribution of recommendations is better.

This is a second evaluation lens on the existing ranker (observed-referral ranking stays the primary objective), not a change of objective. It is not circular: models are compared with each other and with current practice on MIPS, not graded against MIPS as ground truth.

## 1. Inputs v4 (= v3 + four additions; one artifact, both models read it)
| Addition | Definition | Where |
|---|---|---|
| MIPS missing-as-low | For specialist `mips_score` only: missing/ambiguous → scaled value of raw 0 (not the training mean); existing `missing` and `ambiguous` flags kept (the `missing` flag is the professor's `quality_missing`). Same rule for the PCP? PCP MIPS is excluded by earlier decision; unchanged. | specialist numeric block |
| Severity | `D_Disease_Code_Severity` per linked source row → per encounter via `row_event_map`. Categorical {Simple, Moderate, Complex, missing}. Conflicting rows within an encounter → most severe + `severity_conflict` flag. Vocabulary from training. | new encounter categorical + 1 flag |
| Race | Patient `mso_race`, PCP `pcp_race`, specialist `ren_race` as categoricals (training vocab; UNK for missing/unseen). Provider gender (`Rndrng_Prvdr_Gndr`, `_pcp`) also added so "demographics" = {patient gender, patient race, PCP race, PCP gender, specialist race, specialist gender}. | encounter cats (patient gender already there), pcp cats, specialist cats |
| Specialty prior (soft filter) | Two pair features: log P(specialty(candidate) \| encounter categories) and log P(specialty(candidate) \| PCP, categories), estimated from TRAINING events with add-one smoothing, averaged over the encounter's categories; candidate specialty via cutoff-safe as-of lookup. Missing specialty → log of the smoothed uniform prior + no extra flag (specialty-unknown already implied). | pair features 14–15 |

Masking is a model flag, not a second artifact: `mask_demographics=True` maps all six demographic categorical indices to UNK (0) at input time, at train and eval. This keeps the two arms on identical data and lets an unmasked checkpoint be re-evaluated masked as a diagnostic.

Checks before training: pair feature width 16 and first 14 equal v3; severity coverage and conflict counts; race/gender vocab sizes and missing shares; specialty-prior sanity (targets' prior > alternatives'); MIPS: missing rows now carry the scaled-0 value and flag = 1; graph unchanged; masked batch has zero demographic indices.

## 2. Runs (seed 42, week cutoff, summaries-only arm, uniform negatives, all events, selection = first-observed validation MRR, max 100 epochs, patience 5, LR halving 0.5/2 — identical to `v3_A_full`)
1. `v4_unmasked` — full v4.
2. `v4_masked` — `mask_demographics=True`.
Sequential on the M4, ≈ 1 h each. Both checkpoints (`best.pt`, `best_overall.pt`) kept.

## 3. Evaluation (validation subset, 3,000 fixed queries; first-observed cohort 1,274; three pools: all / unseen / specialty-given)
### 3a. Ranking (unchanged protocol)
`compare_pilots.py` for v4_unmasked, v4_masked, v3_A_full, tn_A_s42, PCP frequency. Paired patient-cluster bootstrap for v4_unmasked − v3 and v4_unmasked − v4_masked on the full and specialty pools.

### 3b. MIPS allocation analysis (new script `mips_analysis.py`)
For each eligible query and each scorer S ∈ {v4_unmasked, v4_masked, v3, PCP frequency, current practice}:
- MIPS of the observed specialist as of the query day (value, missing flag).
- MIPS of S's top-5 on the **specialty-given pool** (primary; like-for-like) and on the unseen pool (secondary): mean MIPS over scored candidates, share with MIPS missing, share with MIPS ≥ observed.
- "Current practice" = the observed specialist itself (top-1 only).
Report per stratum: severity (Simple/Moderate/Complex), disease category (top 10 by count), patient gender, patient race; and per cohort (first-observed / repeat). Tables of mean MIPS and missing share; paired difference (S top-5 mean − observed) with patient-cluster bootstrap CI per severity. Also the distribution (quantiles) of recommended MIPS vs observed per severity.
Interpretation rule stated up front: MIPS is an input to every scorer, so higher recommended MIPS is a property of the scorer's allocation, not evidence of clinical superiority; missingness is reported, never treated as zero in the analysis (only in the model input).

### 3c. Fairness slices
Ranking metrics (MRR, R@10, specialty pool) and recommended-MIPS by patient gender and race, unmasked vs masked; plus the diagnostic of the unmasked checkpoint evaluated with masking on. Report group sizes; no claims where n < 100.

## 4. Test split (once, disclosed)
Amendment to STEP7_PROTOCOL: inputs changed (v4), selection unchanged. One pass on the 2019 holdout for v4_unmasked, v4_masked, v3_A_full and heuristics: ranking on the three pools + the MIPS/fairness tables. Disclosed as a re-evaluation of a previously inspected holdout.

## 5. Deliverables
- `RESULTS.md` Step 9 section; `reviews/mips_fairness_20260924.md` write-up.
- New deck file `Graph Modeling/Referral GNN - progress 2026-09-25.pptx` (never over the user-edited deck): audit → candidate space → v3/v4 gains → MIPS by severity → masked vs unmasked.

## 6. Timeline (today)
Inputs + checks 1.5 h → runs 2 h (unattended) → evaluation scripts written during runs, executed 1 h → test pass 0.5 h → write-up + deck 1.5 h.

## 7. Risks / open points for the reviewer
- Race fields are provider- and patient-level snapshots of unknown provenance ("computed race" was excluded earlier as derived); using them only in the unmasked arm and as strata is the intended use.
- Severity rows may conflict within an encounter; rule above.
- Specialty prior is estimated on training events including repeats; alternative: first-observed events only.
- MIPS-as-0 for missing changes the specialist numeric input distribution vs v3; the v3 → v4 ranking comparison therefore mixes four changes; a v4-minus-prior ablation is optional if time allows.
- Masking at input time leaves message passing able to carry demographic information indirectly (e.g., via referral edges to specialists whose other features correlate with race); the masked arm is "demographics not observed as features", not "demographics unobservable".
