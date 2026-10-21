# Referral GNN — condensed overview for slides (as of 2026-09-18)

## 1. The problem
A patient sees their primary care physician (PCP). After the diagnosis is recorded but before a specialist is chosen, recommend suitable, accessible specialists. Observed referrals are the training signal; they show who was chosen, not who was clinically best. Frictions we care about: newer specialists struggle to find patients, popular ones are overloaded, there is no central system.

## 2. The data
- One ACO claims dataset: 608k rows, 119 columns, 2014–2019. After keeping linked referral rows and grouping claim lines into events: **111,104 referral events**, **10,619 patients**, **119 PCPs**, **6,262 specialists**.
- An event = patient + PCP + service date + rendering specialist. Multiple claim lines on one day collapse to one event. The specialist appears in the key only to reconstruct history; at prediction time it is unknown.
- No referral date exists. The specialist service date is the referral-time proxy (professor's convention). Same-day events are simultaneous; history is strictly earlier.
- Legacy regression negatives set aside (professor's guidance). First observed PCP–specialist interaction treated as first referral; frequency as a temporal edge attribute.

## 3. The graph (what we kept, what we cut)
**Kept:** four node types — Patient, PCP, Encounter, Specialist — and three relations: Patient –SEEN_BY→ PCP, PCP –HAS_ENCOUNTER→ Encounter, Encounter –REFERRED_TO→ Specialist (plus reverse edges). PCP and Specialist are role-specific even when the same NPI appears in both.
**Cut:** Specialty, Facility, Region, Disease-category and Procedure-category nodes; the Patient–Encounter edge (patient context is injected directly into the encounter instead); any learned ID embeddings (so unseen patients and specialists are handled by features and structure).

## 4. Features (what we kept, what we excluded, and why)
| Entity | Kept | Excluded |
|---|---|---|
| Patient | age, gender, insurance company, ACO status | computed race, derived age bands, redundant ACO flag, all legacy debiasing-derived fields |
| PCP | practice size, years in Medicare, beneficiary count, credentials | specialization, panel demographics/risk, ratings, graduation year, MIPS, gender |
| Specialist | specialty, credentials, MIPS, rating + rating count, panel risk, panel age/sex counts, practice size, tenure, beneficiaries, services | inferred gender, second rating source, medical-school geography |
| Encounter | principal diagnosis codes (kept as strings; availability before referral is an assumption) | procedures, derived disease/severity/CCI, total time (describes the completed specialist visit) |
| Pair (PCP–specialist) | prior referral count, days since last, first-observed age, great-circle distance from office ZIPs | old proximity flags; patient–specialist distance (no patient location in the data) |
| Pair (patient–specialist) | prior count (added during modeling, provisional) | |

Source-choice rules: MUP-first credentials with WebMD fallback; `clinician specialty` as primary specialty; five raw insurance labels mapped to ACO / NON-ACO; conflicting provider values marked missing + ambiguous rather than picking one.

## 5. Cleaning outcomes that shaped the model
- 57% of events are **repeat observations**: the patient had already seen that specialist. Under the grouping, follow-ups look like referrals. This turned out to dominate results.
- Provider attributes vary within a year for many providers; resolved conservatively as missing + flag.
- Diagnosis: 2,875 events with multiple codes (kept all), 45 with none. Training vocabulary 1,915 codes; ~10% of held-out encounters contain an unseen code.
- Geography: Census 2020 ZCTA centroids; 118/119 PCPs and 6,235/6,262 specialists located; distance coverage ~90% for both targets and alternatives (historical-lookup distance had been badly biased toward targets, so we recomputed from locations).

## 6. Splits and candidates
- Chronological: train 2014–2017 (56,020 events), validation 2018 (37,443), test observed 2019 (17,641).
- Candidates for a query on day d: every specialist first observed strictly before d (~5–6k). Target unavailable for ~2.5% of held-out queries; those count as misses in all-query metrics.
- Training: each query's target plus 20 uniformly sampled alternatives (seed 42), softmax cross-entropy over the 21.

## 7. Model (slide 8 design, made concrete)
- Inputs projected to 64 dims per node type. Encounter input = diagnosis-code embedding bag + its own patient's context + strictly-earlier patient history (prior encounters, days since last, specialties previously referred to) + a sinusoidal encoding of encounter age relative to the cutoff.
- Two HGT layers (4 heads), residual + LayerNorm + ReLU + dropout.
- Scorer: MLP over [encounter embedding ‖ candidate specialist embedding ‖ 8 pair features], exactly slide 8's form plus the pair features (HGTConv cannot consume edge attributes). ~297k parameters.
- Baselines: random, popularity, PCP prior frequency, patient repeat, nearest distance. Non-graph PairMLP ablation.

## 8. Leakage control (the part that took the most care)
- Every query day d: graph contains only encounters before d with their referral edges; provider features are as-of snapshots before d; pair features strictly earlier; candidates first seen before d.
- Query encounters are **receive-only** nodes (PCP → encounter edge only), so no query can influence any other node. Verified by tests: permuting all on/after-day targets, perturbing future encounters, removing other same-day/same-week encounters, or scoring a query alone all leave its scores unchanged.
- Weekly graph batches (cutoff at week start) are 6× faster; validated against daily cutoffs on the same checkpoint (MRR 0.4061 vs 0.4059) and against daily training (0.409 after one epoch vs 0.406 after three weekly epochs).
- All encoders, vocabularies and scalers fit on training data only. Reviewer-found leak in an early weekly design was fixed before any result was used.

## 9. Results so far (validation, 3,000 fixed queries, full candidate pools)
| Model | MRR | R@1 | R@10 | R@20 |
|---|---|---|---|---|
| Random | 0.002 | 0.000 | 0.002 | 0.004 |
| Popularity | 0.023 | 0.010 | 0.041 | 0.062 |
| PCP prior frequency | 0.095 | 0.043 | 0.195 | 0.284 |
| Patient repeat heuristic | 0.396 | 0.292 | 0.584 | 0.629 |
| GNN, 3-epoch pilot | 0.406 | 0.303 | 0.595 | 0.641 |
| GNN, main run (epoch 18, still training) | 0.440 | — | — | — |

Split by whether the patient had already seen the target specialist:
| Subset (share) | GNN pilot MRR | Patient repeat | PCP frequency | GNN main @ epoch 18 |
|---|---|---|---|---|
| Repeat observation (57%) | 0.663 | 0.651 | 0.133 | 0.702 |
| First-observed pair (43%) | 0.033 | 0.027 | 0.041 | 0.059 |

Reading: the headline lift over PCP frequency is mostly the repeat heuristic, which the model learns in one epoch. Value beyond memorised history appears only with longer training: first-observed MRR moves from 0.033 (pilot) to 0.059 (epoch 18), now above every baseline, and is still rising. That is the number to watch.

## 10. Honest limitations to state on a slide
Diagnosis availability before referral is assumed; service date is a proxy for referral time; provider attributes are snapshots, not historical versions; same-day visits may merge; patient–specialist distance unavailable; ZIP-centroid distance is not travel distance; the task measures observed selection, not clinical suitability; repeat observations are not proven follow-ups and first-observed pairs are not proven first-ever referrals.

## 11. What is next
Finish the main run and the two preliminary comparators (PairMLP; no memorised history), one test-split pass, then the deferred protocol runs: seed repeats, single-feature ablations (diagnosis, distance, referral edges), scorer variant. Possible follow-ups: loss reweighting toward first-observed referrals; fairness/load-balancing analysis (new vs popular specialists), which was the original motivation.
