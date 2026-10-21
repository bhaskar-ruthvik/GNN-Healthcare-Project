# Referral GNN — architecture specification (initial baseline)

Confirmed 2026-09-17. Basis: slide 8 of `Graph Modeling/GNN Healthcare.pptx`, aligned with the confirmed graph topology.

## Task
Given a query (patient, PCP, current encounter, day d), rank candidate specialists. Candidates are specialists first observed strictly before d. Observed referral is the training target; it does not establish clinical suitability.

## Graph
- Node types: Patient, PCP, Encounter, Specialist. PCP and Specialist are role-specific even for the same NPI.
- Edge types: Patient –SEEN_BY→ PCP, PCP –HAS_ENCOUNTER→ Encounter, Encounter –REFERRED_TO→ Specialist, plus a reverse copy of each.
- No Patient–Encounter edge (confirmed topology). Slide 8's Patient↔Encounter arrows are realized as explicit feature injection (below), not as a graph edge.

## Temporal visibility (leakage control)
- Queries are processed in graph batches with a cutoff day c. Policy `graph_bucket`:
  - "day": c = the query day; batch = all of that day's encounters.
  - "week": c = Monday-aligned week start, clipped to the split start; batch = all of the split's encounters in [c, c+7).
- Visible graph for cutoff c:
  - Historical encounters (day < c) with HAS_ENCOUNTER and REFERRED_TO edges in both directions.
  - Batch encounters (day ≥ c) are receive-only: they get the forward PCP → encounter edge and nothing else. No referral edge, no reverse edge. Hence no batch encounter can influence any other node, and a query's embedding depends only on history before c and its own features. This holds regardless of what else is in the batch, so same-day or same-week context sharing cannot occur (verified by invariance tests in `scripts/smoke_test.py`).
  - SEEN_BY edges only for patient–PCP pairs first seen before c.
  - Everything else is masked out.
- Candidate pools and pair features always use the exact query day, under both policies. Provider node features are as-of snapshots strictly before c; ambiguity flags are inputs. With "week", graph history and provider snapshots can be up to six days stale relative to the query; this is staleness, not leakage.
- Temporal encoding: 16-dimensional sinusoidal encoding of encounter age (c − service day, in days), concatenated to the encounter numeric input before the linear input projection. Since the projection is linear, this equals adding a learned linear map of the encoding to the projected features (the HGT relative-temporal-encoding form). Batch encounters have age 0. Applied at the encounter input because PyG's HGTConv has no edge-time input.
- Measured cost (M4, MPS, 2026-09-18): train step ≈ 0.66 s under either policy; epoch ≈ 11 min ("day", 1,009 steps) vs ≈ 1.7 min ("week", 156 steps). Evaluation uses the same policy as training. Pilot and ablations use "week"; the main run's policy is decided at the step 5 checkpoint.
- Isolation verification: `evaluate(..., isolate="query" | "patient")` forwards one query (or one patient's encounters) at a time at the same cutoff as the batch policy and must reproduce the full-batch ranks exactly (verified for both policies). Week vs day cutoffs are different approximations (up to six days of history delay under "week") and are compared in step 5 before the main experiment.

## Node inputs (each projected to d = 64)
No learned identity embeddings for any node type. Cold-start entities are handled by features and structure.
- Patient: age (standardized, missing/ambiguous masks), gender, insurance company, ACO status. Taken from the patient's first observed encounter so patient node features are fixed within a forward pass.
- PCP: practice size, years in Medicare, beneficiary count (standardized + masks), credentials embedding. PCP specialization excluded per approved decision.
- Specialist: 15 numeric features (standardized + missing/ambiguous masks), specialty embedding (65 categories + UNK), credentials embedding.
- Encounter = concat of
  1. Diagnosis: masked mean of 32-dim code embeddings; vocabulary from training encounters; index 0 = PAD (empty slot, excluded from the mean), 1 = UNK (unseen or < 5 training occurrences). Two explicit flags: no-diagnosis-code (45 encounters) and share of UNK codes.
  2. Explicit patient context for this encounter (age, gender, insurance, ACO, with masks) — replaces the Patient→Encounter edge.
  3. Patient history summary, strictly earlier than d: prior encounter count (log1p), days since last encounter (log1p + missing flag), multi-hot of specialties previously referred to. Each prior specialty is the specialist's specialty as observed on or before that prior encounter's day (cutoff-safe).
  4. Temporal encoding of encounter age.

## Message passing
Two HGTConv layers, hidden 64, 4 heads, ReLU, dropout 0.2, residual connection and LayerNorm per node type. Full-graph forward per batch (graph is ~130k nodes; sampling unnecessary). 296,841 parameters.
Structural note: with two layers and the encounter-only scorer, patient information reaches a query encounter only via raw patient input → PCP (layer 1) → encounter (layer 2). The patient node's own HGT outputs, the last layer's PCP outputs, and relation priors of single-relation destinations therefore receive no gradient (23 tensors). This is inherent to the confirmed topology, and is exactly why patient context is injected into the encounter input. The `enc_patient_pcp` scorer variant uses these parameters.

## Scoring (slide 8 form + pair features)
s(E_t, S) = MLP([ z_Et ‖ z_S ‖ e_pair ])
- e_pair: log1p prior PCP–specialist referral count, log1p days since last referral + missing flag, log1p first-observed age + missing flag, PCP–specialist great-circle distance (standardized) + missing flag, log1p patient–specialist prior referral count.
- MLP: 128 → 128 → 1, ReLU, dropout 0.2.
- Patient and PCP information reach the scorer through z_Et (explicit injection and HAS_ENCOUNTER messages). Concatenating z_patient and z_pcp into the scorer is an ablation variant, not the default.

## Training
- Loss: softmax cross-entropy over each query's 21 candidates (target + 20 bundled seed-42 alternatives).
- Optimizer Adam, lr 1e-3, weight decay 1e-5. Early stopping on validation MRR (patience 3). Checkpoint best model.
- Device: Apple MPS if available, else CPU.
- All encoders/scalers/vocabularies fit on training split only.

## Evaluation
- Full candidate pool ranking; Recall@1/5/10/20 and MRR.
- Reported (a) conditional on target availability and (b) over all queries with unavailable targets counted as misses.
- Breakdowns: familiar vs unseen PCP–specialist pair (prior count > 0), seen vs unseen specialist, seen vs unseen patient.
- Validation during training: fixed 3,000-query subset with full pools. Final test: all test queries.

## Baselines and ablations
- Baselines: global popularity (prior referral count as-of d), PCP prior referral frequency (ties → recency → popularity), nearest specialist by distance.
- Ablations: no distance; no referral history; no diagnosis; non-graph MLP on the same inputs; scorer with z_patient and z_pcp concatenated.

## Known limitations carried from the bundle manifest
Diagnosis assumed available before referral; service date is a referral-time proxy; provider snapshots not historically verified; same-day real visits may merge; patient–specialist distance unavailable; ZCTA points are not driving distance; conflicting provider values remain ambiguous/missing.
