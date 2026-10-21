# STEP12a results: why does LightGBM concentrate more than the neural model? (2026-10-13)

## Setup

Frozen plan: `STEP12_CONGESTION_PLAN.md`.

- Validation 2018, STEP10 primary cohort, n = 5,550.
- 30% quality weight.
- Load = busiest-10% share of the top pick, compared at matched fit (random-mixture degradation, α per bootstrap draw).
- PCP-cluster bootstrap with 2,000 draws.

Sources:
- `runs/analysis/step12a/final_summary.json` (verdict);
- `step12a_decisions.json` (gate, R, PCP check);
- evaluation folders `final`, `final_b1`, `capacity`, `gate`.

## Gate (part 1)

- G is the neural model with every history input masked.
- At matched fit it is 4.3 pts less concentrated than the tabular LightGBM [1.8, 7.6], with all 6 seed pairings positive.
- → **Tabular branch.** History inputs are not needed for the neural model's lower load.
- The frozen reference gap is R = 4.3 pts (R/2 = 2.2).

## Pre-registered verdict (Holm over H1–H5, one-sided)

| | Contrast | Effect (pts) | 95% CI | Holm p | Noise band | Seed pairings | Verdict |
|---|---|---|---|---|---|---|---|
| H1 model class | LightGBM softmax − neural softmax | **+3.9** | [1.8, 6.5] | 0.002 | 5.3 | 6/6 > 0 | not supported (fails only "beyond noise") |
| H2 objective | neural LambdaRank − neural softmax | +0.9 | [−0.2, 2.3] | 0.26 | 5.3 | mixed | not supported |
| H3 volume | (LGBM − N) − (LGBM-B2 − N-B2) | −0.1 | [−2.5, 2.6] | 1.0 | 5.3 | mixed | not supported |
| H4 capacity | high − low capacity, matched to F_ref | **−3.0** | [−4.2, −1.5] | opposite side: 0.002 | 1.7 | 4/4 < 0 | **contradicted** |
| H5 representation | neural log-volume − neural softmax | +0.7 | [−0.8, 2.7] | 0.63 | 5.3 | mixed | not supported |

**Overall: not explained by any single factor at this resolution.**

### Notes on each result

- **Neural noise band (5.3 pts).** It is set by the neural LambdaRank cell's seeds (38.7 / 35.2 / 33.4%), about the size of R itself.
- **H4.** Higher-capacity LightGBM is *less* concentrated at equal fit.
- **Interaction** {LambdaRank, softmax} × {LightGBM, neural}: −0.5 [−1.9, +0.9].
- **Selection-overlap sensitivity.** Excluding the 3,000 selection queries does not change any estimate materially (H1 +4.0, H4 −3.1).

## Secondary and sensitivity

**H1 secondary readings:**

| Reading | Effect (pts) | 95% CI | Interpretation |
|---|---|---|---|
| Base ranking (top pick) | +8.0 | [4.2, 11.8] | — |
| Top-5 slots (30% weight) | +0.6 | [−0.5, 2.0] | The difference lives in the top pick, not in the list |

**Matching-method sensitivity (§2.E).**
- Degrading the stronger model with Gaussian score noise instead of random mixture: H1 becomes **+0.8 pts**, H2 +0.9, H5 +0.7.
- G is much weaker than LightGBM (fit 39.4% vs 44.2%), so H1 depends heavily on how the fit gap is bridged.
- Part of "the neural model spreads load" is "a less certain model spreads load".
- The original headline needs no matching. The full GNN and the table have equal fit at the 30% weight (45.4% vs 45.5%), with load 40.3% vs 46.1%.

## Descriptive mechanism evidence

| Cell | Fit | Load | New-to-PCP | MIPS gain | Top-pick volume percentile | τ(score, log volume) | B1 Δload (matched) |
|---|---|---|---|---|---|---|---|
| LightGBM LambdaRank | 45.5 | 46.1 | 35.3 | +6.9 | 0.76 | 0.20 | **−8.3** [−10.5, −6.4] |
| LightGBM softmax | 44.2 | 44.2 | 38.7 | +6.9 | 0.74 | 0.18 | −8.1 |
| LightGBM binary | 47.4 | 45.5 | 34.9 | +5.1 | 0.76 | 0.17 | −7.6 |
| LightGBM B2 (shares) | 45.0 | 45.5 | 36.0 | +7.0 | 0.76 | 0.19 | +0.1 (B1+ −1.7) |
| Neural softmax (G) | 39.4 | 34.9 | 44.7 | +7.4 | 0.68 | 0.15 | −3.1 [−4.6, −2.0] |
| Neural LambdaRank | 39.4 | 35.7 | 44.9 | +7.2 | 0.68 | 0.16 | −3.4 |
| Neural BCE | 38.1 | 32.6 | 46.6 | +6.7 | 0.66 | 0.15 | −4.7 |
| Neural B2 / log | 39.5 / 39.5 | 34.8 / 35.7 | 45 | +7.1 / +7.0 | 0.67 / 0.68 | 0.15 / 0.16 | 0.0 / −3.9 |
| GNN (final) | 45.4 | 40.3 | 24.9 | +7.3 | — | — | −2.1 [−2.9, −1.4] |
| No-graph GNN (3 seeds) | 44.4 | 41.4 | 27.4 | +7.1 | — | — | −2.7 |

The chosen specialist's own top-pick volume percentile is 0.66.

### Readings

1. **LightGBM leans on specialist volume information far more than any neural model at scoring time.**
   - Neutralising the volume block at inference (B1, share-preserving) cuts LightGBM's load by 8 pts at matched fit, against 2–5 pts for the neural models.
   - The differential (≈ 5 pts) is about R.
   - Retraining without the explicit counts (B2) changes nothing (H3 null), because the remaining specialist columns still predict log beneficiaries with R² 0.82 (gradient boosting; 0.56 linear).
   - Consistent story: trees exploit volume, explicit or reconstructed from proxies. B2 could not isolate it.
   - B1 is an out-of-distribution reliance diagnostic, not a retrain.
2. **LightGBM's top picks sit higher in each set's volume ranking** (0.76 vs 0.68 neural vs 0.66 chosen specialist). The neural models are closer to what PCPs actually do.
3. **Output-side residualisation** (removing each model's within-set slope on log training volume) cuts load sharply for every model (LightGBM 46 → 19%, G 35 → 12%), at a large fit cost (−6 pts).
4. **The gap is concentrated by specialty.** Dermatology is 13% of referrals but ≈ 39% of the raw LightGBM − G gap (Δ 35 pts within dermatology); then orthopaedics, ophthalmology, cardiology, neurology.
5. **Objective and capacity are not the driver.** Larger LightGBM models are slightly less concentrated at equal fit.
6. **PCP overlap of top picks** (Jaccard within specialty): LightGBM 0.05–0.07, neural 0.07–0.08. No clear difference.

## Limitations

- **Neural seed noise ≈ 5 pts:**
  - 3 seeds are too few for a 4-pt effect under the frozen beyond-noise rule;
  - G-family models stop early (best epochs 4–22).
- **The matching method changes H1 materially** (random mixture +3.9 vs noise degradation +0.8).
- **B2 leaves most volume information in** (R² 0.82). The 25% of specialists with zero panel cells (likely CMS suppression) also carry a volume signal.
- **The volume block is one static MUP release**, probably ≈ 2018 (to confirm), and so overlaps the validation year.
- **G keeps a constant 16-dimensional time encoding** that the table lacks. BCE loss scale differs (clip / weight decay).
- **Validation 2018 only**; no test pass (mechanism study).

## Addendum 2026-10-14 — Amendment A1 (post hoc, secondary; rules frozen before results)

| | Effect (pts) | 95% CI | Holm p (seed bootstrap / + pooled seed term) | Secondary verdict |
|---|---|---|---|---|
| H1 model class | +3.9 | [1.3, 7.1] | 0.006 / 0.029 | supported (post hoc) |
| H3′ strict attribute set | +0.6 | [−2.5, 3.9] | 1.0 / 1.0 | not supported |
| H2 / H5 | +0.8 / +0.7 | n.s. | — | not supported |

**H1 must be read together with:**
- noise-degradation matching: +0.9;
- base ranking: +6.1;
- top-5 slots: +0.6;
- the fit gap: G 39.8% vs LightGBM 44.2%.

**H3′.** Removing all volume-informative specialist attributes lowers concentration in both model classes (LightGBM 46.1 → 36.6%, neural 35.3 → 28.8%), but leaves the gap between them. The primary verdict is unchanged.
