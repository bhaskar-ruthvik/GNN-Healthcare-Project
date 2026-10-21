# Failure audit — three-node model (arm B, seed 42, `best.pt`), validation subset, 2026-09-23

Source numbers: `runs/analysis/failure_audit_20260922T202902_tn_B_s42_best.json` (script `scripts/failure_audit.py`). 2,918 eligible queries: 1,192 first-observed patient–specialist pairs, 1,726 repeat observations. Pools ≈ 5,400 candidates.

## 1. Where the true specialist lands (first-observed)
| | MRR | median rank | p25 | p75 | top-10 | top-50 | top-100 | top-500 |
|---|---|---|---|---|---|---|---|---|
| First-observed | 0.081 | 79 | 17 | 428 | 19% | 42% | 54% | 78% |
| Repeat | 0.742 | 1 | 1 | 2 | 97% | 100% | | |

The first-observed distribution is bimodal: a quarter of targets sit within rank 17, a quarter beyond rank 428. Section 3 explains the split.

## 2. Wrong specialty or wrong person?
- Top-1 candidate is in the target's specialty for 21% of first-observed queries; on average 2.3 of the top 10 are.
- Median size of the target's own specialty in the pool: 235 specialists.
- If the pool were restricted to the target's specialty (oracle), within-specialty MRR would be 0.249, top-1 13%, top-5 37%. So even given the right specialty, the model picks the right person only about one time in eight.
- When the top-1 specialty is right, median target rank is 21; when wrong, 106.

Both failure modes are real. Getting the specialty right roughly quarters the rank, but the within-specialty ranking is itself weak.

## 3. Familiarity dominance — the central finding
| | value |
|---|---|
| Share of the pool that this PCP has referred to before | 12.7% |
| Share of the model's top-10 slots given to those familiar specialists | **97.0%** |
| First-observed targets that are PCP-familiar | 64% → MRR 0.123 |
| First-observed targets that are PCP-unfamiliar | 36% → **MRR 0.007, median rank 568** |

The trained model is, in effect, a re-ranker of the PCP's existing referral list. It almost never surfaces a specialist the PCP has not used. For the third of first-time referrals that go to a specialist new to the PCP, it is no better than random. Nothing in training penalises this: the 20 random alternatives are unfamiliar 87% of the time, so "familiar beats unfamiliar" solves most of the training task.

## 4. Popularity concentration (the original motivation)
- Top-10 recommendations sit at the 86th percentile of specialist popularity; true first-observed targets at the 74th.
- Gini of top-10 recommendation slots across specialists 0.66 vs 0.37 for the true targets; the 50 most-recommended specialists take 22% of all top-10 slots.
- Recommendations are more concentrated than actual referrals. As trained, the model would worsen, not relieve, the load imbalance the project set out to address.

## 5. Distance
Median PCP–specialist distance of first-observed targets: 11.6 miles (repeat: 7.2). Distance is informative and used (§6), but only PCP-origin distance exists in these inputs; patient-origin distance is now available (new source) and untested.

## 6. What the trained model relies on (inference-time ablations, first-observed MRR change)
| removed at test time | Δ first-observed MRR | Δ repeat MRR |
|---|---|---|
| diagnosis categories | **−0.047** | −0.078 |
| pair referral history (count, recency, first-observed age) | −0.034 | −0.025 |
| all message passing | −0.018 | −0.029 |
| REFERRED_TO edges | −0.014 | −0.017 |
| PCP–specialist distance | −0.011 | +0.003 |
| HAS_ENCOUNTER edges | −0.007 | −0.003 |
| time encoding | −0.003 | +0.001 |
| patient history block | −0.001 | 0.000 |
| PRECEDES edges | −0.001 | −0.006 |
| predecessor gap | 0.000 | +0.001 |
| **patient–specialist prior count** | **+0.029** | −0.474 |

Readings:
- Diagnosis is the single most important input for first-observed ranking. The model does use clinical context; it is not the bottleneck it looked like.
- Message passing contributes a measurable −0.018 here, mostly through REFERRED_TO edges; PRECEDES and the predecessor gap contribute nothing, consistent with the A/B result.
- **The patient–specialist prior count actively hurts first-observed ranking.** For a first-observed query the target has zero by definition, while the patient's previously seen specialists have positive values and get pushed above the target. Removing it at test time raises first-observed MRR from 0.081 to 0.110 (+36%) while collapsing repeat MRR. The feature is doing two jobs that conflict.

Gradient×input attribution (first-observed sample, numeric blocks only): pair referral history 43%, encounter patient-history summaries 23%, distance 22%, time encoding 9%; the category bag is not covered by this decomposition but §6 shows its effect.

## 7. Implications — what to fix, in order of expected impact
1. **Separate the two tasks at inference and in training.** A recommender for *new* specialists should rank only candidates the patient has not seen; the patient's own prior specialists are a different list. Evaluate first-observed queries on the unseen pool (no label leakage: "seen before" is history). This removes the conflicting prior-count signal without retraining, and is the operational framing the project needs.
2. **Hard negatives.** Draw training alternatives from (a) the PCP's familiar specialists and (b) the target's specialty, so familiarity and specialty stop being sufficient. This is the only way to teach ranking among unfamiliar specialists (§3) and within a specialty (§2).
3. **Patient-origin distance and a plausible-candidate radius** (new ZIP column, 96.9% mappable). Expected to help both the unfamiliar third (§3) and concentration (§4) by bringing in nearby specialists the PCP has not used.
4. **A diversity/popularity term or post-hoc re-ranking** if §4 concentration persists after 1–3; measure Gini of recommendations alongside MRR.
5. Not worth more time now: PRECEDES edges, predecessor gap, time encoding (§6).

## 8. Caveats
Inference-time ablations shift the input distribution away from training and understate what retraining without a feature would recover; they show reliance, not optimal contribution. Oracle-specialty numbers use the target's specialty and are diagnostic only. All numbers are validation-subset, seed 42, one checkpoint.
