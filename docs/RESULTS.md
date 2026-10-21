# Results log

All numbers are on the fixed 3,000-query validation subset with full candidate pools (every specialist first observed strictly before the query day; ~5–6k candidates per query) unless stated otherwise. Ranks are pessimistic (ties count against the target). "Eligible" = target in the pool (97.3% of the subset). The test split has not been evaluated.

## Step 5a — baselines (2026-09-18)

| Baseline | MRR | R@1 | R@5 | R@10 | R@20 | Median rank |
|---|---|---|---|---|---|---|
| Random | 0.002 | 0.000 | 0.000 | 0.002 | 0.004 | 2,645 |
| Popularity (global prior count) | 0.023 | 0.010 | 0.026 | 0.041 | 0.062 | 582 |
| PCP prior frequency (count → recency → popularity) | 0.095 | 0.043 | 0.133 | 0.195 | 0.284 | 70 |
| Patient repeat (patient's own prior specialists first, then PCP frequency) | 0.396 | 0.292 | 0.522 | 0.584 | 0.629 | 4 |
| Nearest distance | 0.005 | 0.000 | 0.000 | 0.001 | 0.002 | 679 |

Source: `runs/baselines_validation_subset/`.

## Step 5b — pilot: 3 epochs, week cutoff (`runs/20260918T102955_pilot_week`)

| Epoch | Train loss | MRR | R@1 | R@10 | R@20 | Train time | Val time |
|---|---|---|---|---|---|---|---|
| 1 | 1.434 | 0.392 | 0.290 | 0.577 | 0.630 | 113 s | 17 s |
| 2 | 0.999 | 0.403 | 0.300 | 0.592 | 0.640 | 108 s | 17 s |
| 3 | 0.962 | 0.406 | 0.303 | 0.595 | 0.641 | 107 s | 17 s |

All-query MRR (unavailable targets = miss): 0.395. Loss still falling at epoch 3; the pilot was not run to convergence by design.

### Breakdown of the pilot checkpoint vs baselines

| Subset | n | Model MRR | Model R@10 | PCP-freq MRR | PCP-freq R@10 | Patient-repeat MRR | Patient-repeat R@10 |
|---|---|---|---|---|---|---|---|
| Target is a specialist the patient already saw (repeat) | 1,726 | 0.663 | 0.951 | 0.133 | 0.268 | 0.651 | 0.946 |
| Target is new to the patient (first-time referral) | 1,274 | **0.033** | **0.081** | 0.041 | 0.089 | 0.027 | 0.060 |
| PCP–specialist pair familiar | 2,488 | 0.476 | 0.698 | | | | |
| PCP–specialist pair unseen | 512 | 0.001 | 0.000 | 0 by construction | | | |
| Patient has prior history | 2,744 | 0.435 | 0.634 | | | | |
| Patient has no prior history | 256 | 0.071 | 0.153 | | | | |
| Target specialist unseen in training | 230 | 0.430 | 0.615 | | | | (availability 0.64) |

### Reading
- **57.5% of validation targets are repeat visits** (patient already saw that specialist, strictly earlier). Under the event grouping, a follow-up visit to the same specialist is a new "referral" event. Almost all of the headline lift over PCP frequency comes from this subset, and a two-line heuristic (patient repeat) captures nearly all of it: 0.396 vs 0.406 overall.
- **On first-time referrals the pilot model is no better than PCP frequency** (MRR 0.033 vs 0.041) and gets essentially nothing right for unseen PCP–specialist pairs. After 3 epochs it has learned the repeat/familiarity heuristics that dominate the sampled-softmax loss, not clinical matching (diagnosis → specialty) for new referrals.
- Consequence for step 6: the first-time-referral subset must be a primary reported metric, not just a breakdown; the main run needs many more epochs; and the `no_patient_specialist_prior` / `no_history_features` ablations are the ones that show whether the model can learn beyond memorised history.
- Cold start: unseen patients (0.364 MRR) and unseen target specialists (0.430 MRR, but only 64% available) are handled by features, as designed.

## Step 5c — week vs day cutoff assessment (same checkpoint)

| Eval cutoff | MRR | R@1 | R@10 | R@20 |
|---|---|---|---|---|
| week (trained) | 0.4061 | 0.303 | 0.595 | 0.641 |
| day | 0.4059 | 0.303 | 0.596 | 0.641 |

Ranks differ for 26% of eligible queries, median |Δrank| = 0, MRR gap −0.0002. The up-to-six-day history delay is immaterial for this model.

### Trained-with-day comparison (`runs/20260918T103927_pilot_day`, 1 epoch)

| Run | Optimizer steps | Wall time | MRR | R@1 | R@10 | R@20 | First-time-referral MRR / R@10 |
|---|---|---|---|---|---|---|---|
| week, epoch 1 | 156 | 2.2 min | 0.392 | 0.290 | 0.577 | 0.630 | |
| week, epoch 3 | 468 | 6.5 min | 0.406 | 0.303 | 0.595 | 0.641 | 0.033 / 0.081 |
| day, epoch 1 | 1,009 | 13.2 min | 0.409 | 0.304 | 0.600 | 0.645 | 0.033 / 0.084 |

Per optimizer step the two policies are comparable; per minute the week policy is roughly 2–3× more efficient. Neither policy changes the first-time-referral picture. Decision for step 6: train the main run and the ablations with the week policy (converged runs are cheap); optionally repeat the final configuration with the day policy as a robustness check.

## Step 6 (preliminary, time-boxed) — validation analysis (2026-09-18, 3,000 fixed queries, full pools)

Runs: `main_s42` (full GNN, 30-epoch cap, best epoch 29), `pairmlp_e30` (no message passing, early-stopped at 23), `pairmlp` (10-epoch cap), `abl_no_history_no_ps` (pair referral history and patient–specialist prior removed from the scorer, 10-epoch cap). Deferred under the frozen protocol: seeds 1–2, single-feature ablations, scorer variant.

| Model | MRR | R@1 | R@10 | R@20 | First-observed MRR | First-observed R@10 | Repeat MRR |
|---|---|---|---|---|---|---|---|
| GNN `main_s42` | 0.464 | 0.361 | 0.640 | 0.691 | 0.070 | 0.164 | 0.737 |
| No-graph `pairmlp_e30` | 0.460 | 0.353 | 0.645 | 0.691 | 0.081 | 0.187 | 0.722 |
| No-graph `pairmlp` (10 ep) | 0.443 | 0.333 | 0.634 | 0.684 | 0.070 | 0.161 | 0.700 |
| GNN pilot (3 ep) | 0.406 | 0.303 | 0.595 | 0.641 | 0.033 | 0.081 | 0.664 |
| Patient repeat heuristic | 0.396 | 0.292 | 0.584 | 0.629 | 0.027 | 0.060 | 0.651 |
| GNN, no memorised history `abl_no_history_no_ps` (10 ep) | 0.139 | 0.073 | 0.271 | 0.354 | 0.062 | 0.122 | 0.192 |
| PCP prior frequency | 0.095 | 0.043 | 0.195 | 0.284 | 0.041 | 0.089 | 0.133 |

Paired patient-cluster bootstrap (1,000 resamples, 1,871 patient clusters), MRR difference GNN − other, 95% CI:

| Comparison | All eligible | First-observed | Repeat |
|---|---|---|---|
| GNN − patient repeat | +0.068 [+0.060, +0.076] | +0.043 [+0.034, +0.051] | +0.086 [+0.073, +0.098] |
| GNN − PCP frequency | +0.369 [+0.353, +0.386] | +0.029 [+0.021, +0.038] | +0.603 |
| GNN − no-graph (30 ep) | +0.004 [−0.003, +0.011] n.s. | **−0.011 [−0.018, −0.003]** | +0.014 [+0.003, +0.025] |
| GNN − GNN without memorised history | +0.326 | +0.008 [−0.001, +0.017] n.s. | +0.545 |
| GNN − pilot (3 ep) | +0.058 | +0.037 | +0.073 |

### Reading
1. **The trained GNN beats every heuristic, including on first-observed pairs, with tight intervals.** +0.043 MRR over the patient-repeat heuristic and +0.029 over PCP frequency on first-observed pairs; both CIs exclude zero. The pilot's "no better than PCP frequency" was an under-training artefact: first-observed MRR doubled between epoch 3 and epoch 29 and had not plateaued.
2. **Message passing adds nothing measurable overall and slightly hurts first-observed ranking.** At matched budget the no-graph model is within noise overall (+0.004, CI spans zero), better on first-observed pairs (GNN − MLP = −0.011, CI excludes zero), and worse on repeats (+0.014). The explicit inputs (diagnosis, patient context and history, provider attributes, pair history, distance) carry the signal; the two HGT layers mainly sharpen repeat/familiar cases. Caveat from STEP4_PLAN: PairMLP is not information-matched, so this is a lower bound on what the graph could contribute with a different design.
3. **First-observed signal comes from non-history inputs.** Removing all memorised pair/patient history costs 0.33 MRR overall (repeats collapse) but only a non-significant 0.008 on first-observed pairs. Diagnosis codes, provider attributes and structure account for essentially all of the model's first-time-referral ability.
4. **Checkpoint selection barely matters.** Overall-selected vs first-observed-selected checkpoints differ by ≤0.005 MRR.
5. Unfamiliar PCP–specialist pairs remain nearly unranked by every model (MRR ≤ 0.011).

## Step 6 (preliminary) — TEST split, evaluated once (2026-09-18, 17,641 queries, full pools, week cutoff)

Target available for 97.7% of queries (17,238 eligible). Repeat observations: 9,886 (57.3%); first-observed pairs: 7,352 (42.7%).

| Model | MRR | R@1 | R@10 | R@20 | All-query MRR | First-observed MRR | First-observed R@10 | Repeat MRR |
|---|---|---|---|---|---|---|---|---|
| GNN `main_s42` (primary checkpoint) | **0.448** | 0.346 | 0.630 | 0.673 | 0.438 | 0.075 | 0.172 | 0.726 |
| No-graph `pairmlp_e30` | 0.444 | 0.340 | 0.628 | 0.674 | 0.434 | 0.080 | 0.180 | 0.715 |
| GNN, first-observed-selected checkpoint | 0.444 | 0.341 | 0.627 | 0.672 | 0.434 | 0.072 | 0.165 | 0.720 |
| Patient repeat heuristic | 0.387 | 0.285 | 0.576 | 0.618 | 0.378 | 0.030 | 0.066 | 0.652 |
| PCP prior frequency | 0.081 | 0.034 | 0.171 | 0.249 | 0.079 | 0.044 | 0.086 | 0.109 |
| Popularity | 0.021 | 0.009 | 0.036 | 0.053 | 0.021 | 0.019 | 0.032 | 0.023 |
| Nearest distance | 0.004 | 0.000 | 0.000 | 0.001 | 0.004 | 0.003 | 0.000 | 0.005 |
| Random | 0.001 | 0.000 | 0.002 | 0.003 | 0.001 | 0.001 | 0.001 | 0.001 |

Paired patient-cluster bootstrap on test (1,000 resamples, 4,515 patient clusters), MRR difference GNN − other, 95% CI:

| Comparison | All eligible | First-observed | Repeat |
|---|---|---|---|
| GNN − patient repeat | +0.062 [+0.057, +0.066] | +0.045 [+0.041, +0.048] | +0.074 [+0.067, +0.082] |
| GNN − PCP frequency | +0.367 [+0.355, +0.378] | +0.030 [+0.026, +0.034] | +0.618 |
| GNN − no-graph (30 ep) | +0.004 [+0.001, +0.007] | **−0.006 [−0.009, −0.003]** | +0.011 [+0.006, +0.017] |
| GNN primary − GNN first-observed-selected | +0.004 | +0.002 | +0.006 |

### Reading (test)
- Validation-to-test drop is small (MRR 0.464 → 0.448; first-observed 0.070 → 0.075), so the model generalises to the later year.
- The GNN beats every heuristic on every subset with tight intervals. On first-observed pairs it reaches 1.7× the PCP-frequency MRR and 2.5× the patient-repeat MRR.
- Graph vs no graph: +0.004 overall (barely significant on test), −0.006 on first-observed (no-graph better), +0.011 on repeats. Message passing, as designed here, contributes little; the explicit inputs do the work.
- Headline for slides: a learned model on these inputs adds real but modest value beyond memorised history, concentrated where it matters (first-time referrals), and the graph layers are not yet the reason.

Preliminary caveats: `pairmlp_e30` is protocol-compliant; `abl_no_history_no_ps` and `pairmlp` were capped at 10 epochs; seed repeats and single-feature ablations are deferred. Test was evaluated once, after the protocol freeze, with no configuration changes.

## Step 7 (frozen protocol) — three-node arms, seed 42, validation subset (2026-09-22)

Inputs `three_node_v1` (33 disease categories, patient context in encounters, no Patient node). Selection: first-observed validation MRR (`best.pt`); `best_overall.pt` reported as reference. Both arms early-stopped at epoch 28 with best first-observed at epoch 23 and best overall at epoch 27; LR halved at ~17 and ~25. Runs: `runs/20260922T193819_tn_A_s42`, `runs/20260922T202902_tn_B_s42`.

| Model (checkpoint) | MRR | R@1 | R@10 | R@20 | First-obs MRR | First-obs R@10 | Repeat MRR | Spec. match top-1 / top-5 (first-obs) |
|---|---|---|---|---|---|---|---|---|
| **Arm B, PRECEDES (best)** | 0.472 | 0.370 | 0.649 | 0.700 | **0.081** | 0.190 | 0.742 | 0.211 / 0.492 |
| Arm B (best_overall) | 0.476 | 0.374 | 0.650 | 0.699 | 0.081 | 0.191 | 0.748 | |
| **Arm A, summaries only (best)** | 0.468 | 0.368 | 0.645 | 0.699 | **0.081** | 0.183 | 0.735 | 0.197 / 0.492 |
| Arm A (best_overall) | 0.471 | 0.373 | 0.648 | 0.700 | 0.081 | 0.186 | 0.741 | |
| Four-node `main_s42` (best) | 0.464 | 0.361 | 0.640 | 0.691 | 0.070 | 0.164 | 0.737 | 0.189 / 0.472 |
| Patient repeat heuristic | 0.396 | 0.292 | 0.584 | 0.629 | 0.027 | 0.060 | 0.651 | 0.100 / 0.347 |
| PCP prior frequency | 0.095 | 0.043 | 0.195 | 0.284 | 0.041 | 0.089 | 0.133 | 0.071 / 0.268 |

Paired patient-cluster bootstrap (1,000 resamples, 1,871 clusters), MRR difference, 95% CI:

| Comparison | All eligible | First-observed | Repeat |
|---|---|---|---|
| **B − A (the protocol question)** | +0.004 [+0.000, +0.009] | **+0.001 [−0.004, +0.005] n.s.** | +0.007 [+0.001, +0.013] |
| B − A (best_overall) | +0.001 n.s. | +0.001 n.s. | +0.001 n.s. |
| B − four-node main | +0.008 [+0.000, +0.015] | **+0.011 [+0.004, +0.019]** | +0.005 n.s. |
| B − patient repeat | +0.076 | +0.054 [+0.045, +0.064] | +0.091 |
| B − PCP frequency | +0.377 | +0.041 [+0.031, +0.052] | +0.608 |

Same-checkpoint day-vs-week cutoff: A +0.0010, B +0.0007 MRR (day − week); ranks differ for 25–27% of queries with median |Δ| 0. The temporal-edge result does not depend on the cutoff policy.

### Reading
1. **PRECEDES edges add nothing measurable on first-observed pairs** (+0.001, CI spans zero) and a small, borderline-significant amount on repeats (+0.007). With the overall-selected checkpoints even that vanishes. The per-encounter history summaries already carry what two hops of same-patient edges can deliver.
2. **The revised inputs are the real gain over the four-node baseline**: +0.011 first-observed MRR (CI excludes zero), first-observed R@10 from 0.164 to 0.190, specialty match top-1 from 0.189 to 0.211. Both arms reach this; neither owes it to the edges. Attribution between disease categories vs raw codes and the removed Patient node needs its own ablation (not in this protocol).
3. Convergence is faster and stopping is well behaved: best epochs 23/27 with two LR halvings, vs the four-node run still improving at its 30-epoch cap.
4. Unfamiliar PCP–specialist pairs remain unranked by every model (MRR ≤ 0.010).
5. Seed repeats (1, 2) are running; the spread table above currently mixes four-node and three-node runs and is not yet a seed-spread estimate.

## Step 8 — failure audit and candidate-space work (2026-09-23; validation subset, first-observed queries n = 1,274 unless noted)

Context: seed repeats of the Step 7 arms were stopped as uninformative (B − A ≈ 0). Work moved to understanding the first-observed failure and changing the candidate space and inputs. Full audit: `reviews/failure_audit_20260923.md`.

### Audit findings (arm B checkpoint)
- 97% of top-10 slots go to specialists the PCP has referred to before (12.7% of the pool). First-observed targets new to the PCP (36%) get MRR 0.007.
- Patient–specialist prior count hurts first-observed ranking (+0.029 MRR when removed at inference). Diagnosis is the most-relied-on input (−0.047 when removed). PRECEDES / gap / time encoding ≈ 0.
- Oracle within-specialty MRR 0.25: even with the specialty known the person is picked right 1 in 8.
- Recommendations concentrate on popular specialists more than reality (Gini 0.66 vs 0.37).

### Candidate-space reducers (pool = specialists the patient has not seen, ≈ 5,400)
| Filter | Candidates left | Target kept |
|---|---|---|
| radius 50 mi (patient origin) | 4,652 | 87% |
| radius 25 mi | 2,997 | 71% |
| specialty (oracle) | 232 | 100% |
| specialty ∩ insurer history | 143 | 87% |
| specialty ∩ (PCP network ∪ 25 mi) | 106 | 91% |
Distance cannot shrink the space without discarding real referrals (whole dataset is DFW). Specialty is the reducer. Specialty is predictable from the disease category alone: top-1 31%, top-3 57%, top-5 70% (training counts); the GNN's implicit 21% is below that.

### Pilots (10 epochs each, summaries-only arm, seed 42; scores on three pools)
| Model | Full pool MRR / R@10 | Unseen pool MRR / R@10 | Specialty-given pool MRR / R@5 / R@10 |
|---|---|---|---|
| PCP frequency (heuristic) | 0.041 / 0.089 | 0.043 / 0.093 | 0.248 / 0.357 / 0.466 |
| v2 (+patient distance), uniform negatives | 0.071 / 0.161 | 0.104 / 0.210 | 0.281 / 0.388 / 0.535 |
| v2, hard negatives (familiar + specialty) | 0.073 / 0.180 | 0.101 / 0.216 | 0.262 / 0.369 / 0.492 |
| v2, within-specialty ranker (pool negatives, first-observed cohort) | 0.060 / 0.122 | 0.064 / 0.122 | 0.280 / 0.395 / 0.528 |
| **v3 (+insurer history, +shared affiliation), uniform negatives** | **0.078 / 0.184** | **0.114 / 0.227** | **0.292 / 0.412 / 0.562** |
| Four-node baseline (Step 6, 30 epochs) | 0.070 / 0.164 | — | — |
| Three-node arm A (Step 7, 28 epochs, v1) | 0.081 / 0.183 | — | 0.277 / 0.393 / 0.512 |

Readings: patient distance as a feature adds nothing over PCP distance (v2 ≈ v1 epoch-for-epoch). Hard negatives help the full-pool metrics slightly and hurt within specialty. The dedicated within-specialty ranker equals, not beats, the general model within specialty and is useless across specialties. Insurer history and shared hospital affiliation are the first inputs to lift every pool at once (+0.007 full, +0.010 unseen, +0.011 specialty MRR over v2 at 10 epochs, still rising). Within the specialty-given pool the v3 model places the chosen specialist in the top 5 for 41% and top 10 for 56% of first-time referrals.

### Converged v3 run (`v3_A_full`: v3 inputs, summaries-only arm, uniform negatives, all events; early-stopped at epoch 28, best first-observed epoch 23)
| First-observed, validation | Full pool MRR / R@5 / R@10 | Unseen pool MRR / R@5 / R@10 | Specialty-given pool MRR / R@5 / R@10 | Specialty match top-1 / top-5 |
|---|---|---|---|---|
| **v3 converged** | **0.086 / 0.120 / 0.196** | **0.122 / 0.173 / 0.243** | **0.292 / 0.416 / 0.564** | 0.195 / 0.492 |
| Step 7 arm A (v1, converged) | 0.081 / 0.120 / 0.183 | 0.114 / 0.158 / 0.233 | 0.277 / 0.393 / 0.512 | 0.197 / 0.492 |
| Four-node baseline (Step 6) | 0.070 / — / 0.164 | — | — | 0.189 / 0.472 |
| PCP frequency | 0.041 / 0.048 / 0.089 | 0.043 / 0.052 / 0.093 | 0.248 / 0.357 / 0.466 | 0.071 / 0.268 |
Overall validation MRR 0.471 (best-overall checkpoint, epoch 28). Progression of first-observed MRR on the full pool across the project: 0.033 (pilot) → 0.070 (four-node, 30 ep) → 0.081 (three-node v1) → 0.086 (v3). On the specialty-given pool the v3 model places the chosen specialist in the top 5 for 42% and top 10 for 56% of first-time referrals; within-specialty person ranking is where the remaining headroom is (specialty match itself is unchanged at ~0.20 top-1 / 0.49 top-5, which a diagnosis-category count table beats at 0.31 / 0.70).

## Step 9 (frozen protocol) — MIPS allocation and demographic masking on v4 (2026-09-24)

Write-up: `reviews/mips_fairness_20260924.md`. Inputs v4 = v3 + severity + patient race + MIPS missing-as-low (the last is a no-op for scored candidates; see log). Runs `v4_unmasked_s42`, `v4_masked_s42` (seed-1 repeats pending).

| First-observed, validation | Full pool MRR / R@10 | Unseen pool MRR / R@10 | Specialty-given MRR / R@5 / R@10 |
|---|---|---|---|
| v4 unmasked (seed 42) | 0.083 / 0.194 | 0.121 / 0.233 | 0.298 / 0.409 / 0.551 |
| v4 masked (seed 42) | 0.085 / 0.201 | 0.120 / 0.245 | 0.296 / 0.420 / 0.564 |
| v3 (converged) | 0.086 / 0.196 | 0.122 / 0.243 | 0.292 / 0.416 / 0.564 |
| PCP frequency | 0.041 / 0.089 | 0.043 / 0.093 | 0.248 / 0.357 / 0.466 |

Pre-registered results:
- **P1 (model-free)**: chosen specialist's within-specialty MIPS percentile 0.480 / 0.474 / 0.487 for Simple / Moderate / Complex; Complex − Simple +0.006 [−0.003, +0.015]. Current practice is MIPS-neutral at all severities.
- **P2**: unmasked − masked MRR +0.002 [−0.008, +0.012], top-1 MIPS percentile +0.002 [−0.010, +0.014] → no attributable effect of masking. No scorer allocates above-practice MIPS (top-1 percentile 0.44–0.45 vs observed 0.487).
- **P3**: MIPS ablation ΔMRR −0.001 / +0.001 / +0.006 by severity; Complex − Simple +0.007 [−0.005, +0.019]. The model barely uses MIPS.
- Fairness (exploratory): group gaps in current practice are reproduced, not narrowed, by either arm; details in the write-up.
- **Seed repeats (Step 9)**: `v4_unmasked_s1` best first-observed 0.087, `v4_masked_s1` 0.086 (full pool). On the specialty pool all four v4 runs are within 0.296–0.298 MRR; unmasked − masked is +0.002 (seed 42) and −0.000 (seed 1) for MRR, +0.002 and +0.004 for top-1 MIPS percentile; within-arm seed gaps ≤ 0.001. Under the frozen attribution rule, no masking effect is claimed.

## Final evaluation — 2019 TEST year (2026-09-25; disclosed re-evaluation of a previously inspected holdout, configuration frozen beforehand)
First-time referrals, n = 7,352 (target available 94.8%). "Within specialty" = unseen-by-patient specialists in the chosen specialist's specialty.

| Model | Within specialty MRR / R@5 / R@10 | All specialists MRR / R@10 |
|---|---|---|
| PCP's own referral habit | 0.250 / 0.362 / 0.478 | 0.044 / 0.086 |
| Last week's four-node model (Step 6) | — | 0.075 / 0.172 |
| Final model, with patient gender and race | 0.301 / 0.426 / 0.547 | 0.079 / 0.188 |
| Final model, without patient gender and race | 0.302 / 0.429 / 0.555 | 0.079 / 0.186 |
Validation → test is stable (within-specialty MRR 0.296–0.298 → 0.301–0.302).

### Recommended vs referred MIPS, and quality-aware lists (test; model without gender/race; n = 7,282 with ≥10 scored candidates)
| 5-specialist list | Referred specialist in list | Top-1 MIPS pct | List mean MIPS pct | List − referred (95% CI) | List raw MIPS | List MIPS missing |
|---|---|---|---|---|---|---|
| Actually referred specialist | — | 0.479 | — | — | 78.9 | 17% |
| Model's own top 5 | 42.4% | 0.444 | 0.459 | −0.021 [−0.028, −0.014] | 77.0 | 18% |
| 5 highest-MIPS of model's top 10 | 29.0% | 0.714 | 0.593 | +0.121 [+0.114, +0.128] | 89.3 | 3% |
| 5 highest-MIPS of model's top 20 | 22.2% | 0.753 | 0.702 | +0.227 [+0.219, +0.234] | 95.9 | 1% |
Same pattern on validation (+0.115 at top 10, +0.217 at top 20) and for the model with gender/race. As trained, recommendations are slightly below the referred specialist's MIPS; choosing within the model's own plausible top 10–20 raises it well above, at the cost of how often the actually referred specialist appears in the list (42% → 29% → 22%).

### Slide-3 before/after (validation, first-time referrals, all specialists)
| Model | Top-10 slots on PCP-familiar specialists (13% of pool) | Familiar target MRR / median rank | New-to-PCP target MRR / median rank |
|---|---|---|---|
| Tuesday's three-node model | 97.0% | 0.123 / 29 | 0.007 / 568 |
| Final model (without gender/race) | 95.8% | 0.127 / 27 | 0.011 / 416 |
The familiarity bias is only slightly reduced by this week's inputs.
