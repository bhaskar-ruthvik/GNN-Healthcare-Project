# Step 9 protocol — MIPS allocation and demographic masking (v4). FROZEN 2026-09-24 (reviewer concurrence received)

Supersedes STEP9_PLAN.md (draft). Amends STEP7_PROTOCOL only in inputs (v4) and the added analyses; selection, settings and cohorts unchanged. Edited afterwards only by dated amendment.

## 1. Decisions taken after review (user sign-off 2026-09-24)
- Demographics in the UNMASKED arm: **patient gender and patient race only** (`Gender`, `mso_race`; race is a *computed/imputed* field per source metadata, used at the professor's request). Provider gender and race stay excluded as decided on 2026-09-15. The unmasked arm is an analysis instrument, not a deployable system.
- **Specialty prior dropped** from v4 (moved to a later v5 with rolling strictly-before counts). v4 = v3 + MIPS rule + severity + patient race. The v3 → v4 ranking comparison therefore mixes three changes; reported as such.
- **Seeds: 42 and 1 for each arm** (4 runs). **Decision rule for P2:** a masked − unmasked difference is attributed to masking only if (i) it has the same sign for seed 42 and seed 1, and (ii) its magnitude exceeds the larger of the same-seed nondeterminism band (≈0.006 specialty-pool MRR from the duplicated `ws_pilot`) and the measured within-arm gap between seeds 42 and 1, each measured separately for each metric (MRR; top-1 MIPS percentile). If seed 1 has not finished, P2 is reported as descriptive only.
- **Test pass deferred**: no test evaluation in Step 9 until this protocol has been applied unchanged on validation and the write-up exists; if run, it is last and disclosed as a re-evaluation of the inspected 2019 holdout.

## 2. Inputs v4
| Item | Rule |
|---|---|
| Severity | `D_Disease_Code_Severity` is a deterministic function of the diagnosis code (0 conflicts over 8,651 codes on linked rows) → safe as an input. Per encounter via `row_event_map`: most severe of its rows + `severity_conflict` flag (expected 0 given determinism); {Simple, Moderate, Complex, missing}. New encounter categorical. Also the stratifier for §4. |
| Patient race | `mso_race` per patient (constant within patient expected; conflicts → UNK + ambiguity flag under the existing policy). New encounter categorical. |
| MIPS missing-as-low | Specialist `mips_score` **missing only** (no earlier history row, or the as-of value is null) → scaled value of raw 0 instead of 0-after-scaling; the existing `missing` flag is the professor's `quality_missing`. **Ambiguous** rows keep their current treatment (value as before, ambiguity flag). Applied identically in the encoder and in the no-earlier-row snapshot path. Note: raw MIPS is never blank on linked rows (6.6% exact zeros, rest positive); our "missing" arises from the strict as-of/ambiguity rules. Documented as a reparameterization of the first linear layer (missing and low were already distinguishable via the flag); expected effect on ranking ≈ 0. |
| Masking | `mask_demographics=True` sets patient gender and patient race categorical indices to UNK **and zeroes their ambiguity flags** at input time, train and eval. One artifact, two model configs. Claim scope: "no explicit demographic inputs"; correlated inputs (disease categories, panel sex counts, history) remain — not demographics-blind, no causal claim. |
Pre-training checks: pair width 14 = v3; encounter cats [gender, insurance, aco, severity, race]; severity/race coverage and conflict counts; MIPS: missing rows carry scaled(0) and flag 1, ambiguous unchanged, zero-MIPS rows unchanged; masked batch: demographic indices all 0 and flags 0; graph unchanged; 21 model checks rerun on v4.

## 3. Runs
`v4_unmasked_s42`, `v4_masked_s42`, `v4_unmasked_s1`, `v4_masked_s1`. Settings identical to `v3_A_full`: summaries-only arm, uniform negatives, all events, week cutoff, selection = first-observed validation MRR, max 100 epochs, patience 5, LR 0.5/2. Sequential on the M4 (~1 h each).

## 4. Pre-registered primary analyses (validation subset; first-observed cohort unless stated)
Definitions used throughout:
- **MIPS for any candidate, including the observed specialist, is read from the same strictly-earlier provider snapshot the model sees** (`provider_snapshot` at the query day). Three states: value / missing / ambiguous. Missing and ambiguous are never converted to a number in the analysis.
- **Pool-relative MIPS percentile**: mid-rank percentile among the non-missing, non-ambiguous candidates of that query's specialty-given pool: (number strictly below + 0.5 × number tied excluding itself) / (n − 1), in [0,1]; under this rule the expected percentile of a random candidate is 0.5 regardless of ties. Missing share and the share of tied observations (per severity) reported alongside every percentile.
- Specialty-given pool = unseen pool ∩ chosen specialist's specialty; **queries whose chosen specialist has unknown specialty are excluded** (count reported); pools with < 10 non-missing-MIPS candidates flagged and excluded from percentile summaries.
- Uncertainty: paired patient-cluster bootstrap (1,000 resamples). Contrasts are bootstrapped directly; overlapping CIs are not used as tests. Fairness group thresholds count **patients**, not queries (n ≥ 100 patients).

**P1 — Current practice (model-free).** Observed specialist's MIPS percentile within its specialty-given pool, by severity (Simple/Moderate/Complex). Primary contrast: mean percentile(Complex) − mean percentile(Simple), bootstrap CI. Tests the professor's hypothesis directly. Uses training + validation first-observed events (no model involved). Secondary: same by disease category (top 10), exploratory.

**P2 — Masked vs unmasked allocation.** For each arm and seed: (a) specialty-pool first-observed MRR/R@5/R@10; (b) recommended MIPS: top-1 mid-rank percentile vs observed percentile, and mean percentile of top-5 excluding the observed specialist vs 0.5, with missing share. Primary contrast: unmasked − masked on (a) MRR and (b) top-1 percentile, bootstrap CI, attributed only under the §1 decision rule. Also: observed-vs-recommended by severity for each arm (exploratory unless P1 shows a severity effect).

**P3 — Does the model use MIPS, and where.** Inference-time MIPS ablation (MIPS set to missing for all candidates) on `v4_unmasked_s42`: Δ first-observed MRR on the specialty pool by severity; Complex − Simple contrast bootstrapped. Reliance diagnostic only.

Exploratory (reported in full, all strata, no selective reporting): the same tables for `v3_A_full`, PCP frequency, popularity; MIPS–popularity and MIPS–practice-size correlations in training; sensitivity with missing MIPS set to pool minimum and to pool median; unseen-pool percentiles within specialty.

**Fairness slices (exploratory, framed as change in gap).** By patient gender and patient race (UNK/ambiguous reported as separate groups): (i) specialty-pool MRR/R@10 per group for each arm; (ii) recommended top-1 MIPS percentile per group vs observed MIPS percentile per group — i.e., does each arm widen or narrow the between-group gap that exists in current practice; (iii) unmasked checkpoint evaluated with masking on = reliance diagnostic only (out-of-distribution).

## 5. Caveats stated in every table
MIPS is an input to every learned scorer; higher recommended MIPS is an allocation property, not clinical superiority. MIPS source years are unvalidated and a performance-year score is not public when that year's referrals occur. Race is computed/imputed. Recommendations concentrate on popular specialists (audit §4); MIPS may correlate with volume. Selection checkpoint and analysis share the 3,000 validation queries (selection is on MRR; disclosed).

## 6. Order of work and fallbacks
1. v4 inputs + checks (severity, race, MIPS rule) → 2. launch the four runs (unattended) → 3. while training: P1 (model-free) and analysis script → 4. P2/P3 on seed 42 when `v4_*_s42` finish → 5. write-up + new deck file → 6. seed-1 spread appended when available. Test pass not in Step 9 unless steps 1–5 are complete with no changes.

## 7. Freeze record
Reviewer (Opus) audit 2026-09-24: 6 blocking, 12 should-fix, 4 notes; all adopted except where §1 records a user decision. Reviewer concurrence received 2026-09-24 after two definitional fixes (mid-rank percentile; P2 attribution rule), both adopted above. FROZEN.
