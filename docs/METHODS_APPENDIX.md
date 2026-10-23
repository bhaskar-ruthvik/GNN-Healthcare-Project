# Methods appendix

*Referral GNN, as of 23 October 2026 (sections F and G added: Steps 13 and 14).*

This file gathers in one place the definitions used throughout the write-up, the decks and the verification workbook. Authoritative sources, which this summarises:
- `docs/ARCHITECTURE.md`
- `docs/STEP10_PROTOCOL.md`
- `docs/STEP11_PLAN.md`
- `docs/STEP12*_PLAN.md`
- `docs/STEP13_GRAPH_VALUE_PLAN.md`, `docs/STEP13_AMENDMENTS.md`, `docs/STEP14_GRAPH_FEATURE_BLEND_PLAN.md`

(all in the code repository), and the project's running log.

---

## A. Data

**Source.** One ACO Medicare claims workbook from Dallas–Fort Worth, service years 2014–2019: 608,181 rows and 119 columns. It contains PCP and rendering-specialist NPIs and attributes (MUP, QPP/MIPS, ratings), patient demographics, insurer, diagnosis codes and disease categories, and ZIP codes.

**Referral events.** One event is a patient + PCP + service date + rendering specialist; claim lines on the same day collapse into one event. That gives:
- 111,104 events;
- 10,619 patients;
- 119 PCPs;
- 6,262 specialists.

**Referral time.** The data has no referral date, so the specialist's service date is used as the referral time (the professor's convention). Same-day events count as simultaneous, and "history" always means strictly earlier.

**First-time vs repeat.**
- A *first-time referral* is one where the patient had no earlier visit to that specialist in the data.
- *Repeat* events (57%) are follow-ups and dominate naive accuracy.
- All reported results from Step 10 onward are on first-time referrals.

**Split (chronological).**
- Training: events up to 31 Dec 2017.
- Validation: 2018 (the "choice year").
- Test: 2019 (the held-out year).

**Geography.**
- Distances are great-circle distances between ZIP centroids (Census 2020 ZCTA).
- Patient → specialist distance uses the patient's ZIP.
- PCP → specialist distance uses the office ZIPs.

**MIPS.** Raw points, 0–100.
- *Model input (as-of):* for each specialist and day, the mean of the distinct positive values seen strictly before that day. Where only zeros were seen, the value is imputed with the specialty's training mean and flagged.
- *Evaluation yardstick:* one fixed value per specialist (the mean of its distinct positive values), applied identically to the chosen specialist and every candidate.
- 247 specialists (3.9%) have no positive score.

## B. Graph and model

**Graph: three node types.**
- Node types: PCP, Encounter, Specialist.
- Edges: PCP → HAS_ENCOUNTER → Encounter, and Encounter → REFERRED_TO → Specialist, each with reverse edges.
- No Patient node: the patient's demographics, insurer and strictly-earlier history summaries are attached to each encounter.
- No same-patient temporal (PRECEDES) edges in the final model; they added nothing measurable in Step 7.

**Leakage control, applied to each referral (query) on day d:**
- The graph contains only encounters before the cutoff, together with their referral edges.
- Provider attributes are as-of snapshots.
- Pair features are strictly earlier.
- Query encounters are receive-only nodes.
- Graph cutoffs are weekly (week start). Daily cutoffs give the same results (MRR difference 0.0002).
- Perturbation tests pass: permuting future targets, perturbing future encounters, or scoring a query alone leaves its scores unchanged.

**Inputs.**
- *Encounter:* disease-category bag (33 categories), patient context (age, gender, race, insurer/ACO), visit history summaries, time encoding.
- *PCP:* practice size, Medicare tenure, beneficiaries, credentials.
- *Specialist:* specialty, credentials, MIPS (as-of), ratings, panel composition, practice size, tenure, volume.
- *14 pair features:* PCP→specialist referral count and recency, patient–specialist prior, PCP and patient distance, insurer history, shared hospital affiliation, plus missing flags.
- Versions v1 → v5 are recorded in `prepared_data/three_node_v*/` with manifests. The final model uses **v5**.

**Model.**
- Per-type input projections to 64 dimensions.
- 2 HGT layers, 4 heads, residual + LayerNorm, dropout 0.2.
- Scorer: an MLP (hidden 128) over [encounter ‖ specialist ‖ pair features].
- 194,255 parameters.

**Training.**
- Sampled softmax over the chosen specialist plus 20 uniformly sampled alternatives.
- Adam, learning rate 1e-3, weight decay 1e-5, gradient clip 1.0.
- Learning rate halved after 2 epochs without improvement; early stop after 5.
- Checkpoint chosen on first-time validation MRR. This is a training-time selection only; no result is judged on it.
- Seeds 42 and 1, plus a same-seed duplicate (42_dup) to measure non-determinism on Apple GPU (MPS).

**"GNN, no message passing"** has the same inputs and scorer with the HGT layers removed.

**Masking variant** ("without gender and race"). Patient gender and race are zeroed inside the model before any encoder, in both training and scoring. Probe AUCs on the embeddings: 0.59 and 0.61.

## C. Baselines and hybrids

**Table (LightGBM).**
- LambdaRank on 117 claim-attribute columns: encounter, specialist and PCP snapshot, distance. No referral history and no network counts, by rule ("pure tabular").
- Same training rows as the GNN: the chosen specialist plus the same 20 alternatives.
- Tuned on a 12-configuration grid (leaves × minimum leaf size × rounds) with the GNN's selection rule.
- Reference configuration: 127 leaves, minimum leaf 50, 1,000 rounds, learning rate 0.05, feature and bagging fraction 0.8.
- Scored in a separate process from torch (macOS libomp clash).

**Other heuristics.**
- *PCP habit:* rank by the PCP's earlier referral counts, ties broken by distance.
- *Random:* exact expectation over the eligible set.
- *MIPS-sorted* and *distance-sorted.*

**Simple hybrids (Step 12b).** All score combinations use z-scores taken within each referral's candidate list.
- *Score blends:* (1 − w)·z(GNN) + w·z(table), with w ∈ {0.25, 0.5, 0.75}.
- *Gate:* w(q) = sigmoid(θ·[PCP history, familiar share, age, eligible-set size]). Fitted by listwise likelihood and cross-fitted over two fixed halves of the PCPs.
- *Reciprocal rank fusion:* 1/(60 + rank).
- *Split lists:* the GNN's top 1 or top 2, then the table.
- *GNN encoding + variables → logistic regression:* the GNN scorer's last hidden layer (128 values) plus 18 readable variables, with the GNN frozen.
- *Flattened graph → LightGBM:* 117 table columns plus 88 metapath-mean features (A1 specialist ← encounters, A2 specialist ← encounters ← PCPs, A3 PCP → encounters → specialists), using referrals before the cutoff and no counts.

**Heavier hybrids (Step 12d).**
- *Table-offset GNN:* score = GNN + β·table score, with β learned.
- *Stacking:* LightGBM with an out-of-fold GNN-score column. The GNN score comes from 3 time-fold models, centred per query, with a scale check against the full model.
- *Dual encoder:* a GNN plus a tabular tower, concatenated into the scorer.

## D. Evaluation

**Primary cohort (STEP10 §2.1).** First-time referrals, after excluding:
- unknown specialty;
- facility-based "non-choice" specialties;
- non-specific specialties;
- an inactive chosen specialist;
- a chosen specialist without a MIPS score;
- fewer than 5 eligible specialists;
- a missing patient ZIP.

That leaves **5,550 referrals in 2018 and 2,668 in 2019**.

**Eligible set ("fits the case").** A specialist is eligible if all of these hold:
- same specialty as the chosen specialist, read strictly before the referral date;
- at least one claim in the previous 365 days;
- not already seen by the patient;
- within 25 miles of the patient. When fewer qualify, the list is filled with the nearest beyond 25 miles: to 5 by default, or to K for top-K lists.

The chosen specialist is in the set for about 87% of referrals.

**Lenses** (per referral, averaged over referrals):

| Lens | Definition |
|---|---|
| Fit | the chosen specialist is in the top 5 (top K) |
| Load (top pick) | the top pick is among the busiest 10%: above the specialty's 90th percentile of first-time referrals in the training years (468 specialists). Actual choices: 29.9% (2018), 26.4% (2019) |
| Load (top 5 / list slots) | share of the list on busiest-10% specialists |
| New to the PCP | share of the list the PCP had never referred to before |
| MIPS gain | mean yardstick MIPS of the list − MIPS of the chosen specialist (points). A cautious bound (imputed at 0, ranges at their worst) is reported in Step 10 |
| Travel | patient → recommended specialist distance (mean or median as labelled) |
| Also reported | "good" new picks (new, higher MIPS than chosen, not busiest); insurer-feasibility of new picks |

**30% quality weight.**
- Rank by 0.7·z(model score) + 0.3·z(as-of MIPS), z-scored within the candidate list.
- Chosen in Step 10 by a pre-registered rule: maximise MIPS gain subject to fit ≥ PCP habit and distance no worse than practice.

**Steering.**
- Final score = (score at the 30% weight) + a·[new to the PCP] − b·[busiest 10%].
- a and b take values in {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6}.
- Frontier: for each family, the settings with fit ≥ 45% (fit ≥ min(GNN, table) − 0.5 pt for top-K). Report the lowest top-pick load at given new-to-PCP targets.
- The steered blend's setting (equal blend, a = 0.3, b = 0.4) was chosen on 2018 by the rule "lowest top-pick load with fit ≥ 0.45 and new-to-PCP ≥ 0.35 in all three pairings".

**Uncertainty.**
- 95% intervals from a PCP-cluster bootstrap (2,000 draws; 119 PCPs), paired when comparing models.
- Draws that contain no PCP from a small group are dropped (fixed 16 Oct; results changed by at most 0.001).
- Step 6–9 analyses used patient-cluster bootstraps.

**Matched fit.**
- When two models differ in fit, the more accurate one is diluted with random picks (a random mixture) until fits match, and load is then compared.
- Sensitivity check: degrading by score noise instead.

**Multiple groups.** Holm adjustment across the groups tested (7 in the group deep-dive).

**Seeds and pairings.**
- Graph models: seeds 42, 1 and 42_dup.
- Tables: seeds 42, 1 and 7 (reference configuration).
- Results are averaged over pairings and also shown per pairing.
- Neural run-to-run noise on load can reach ±5 points.

**Held-out year discipline.**
- Each analysis is frozen in a plan before running and reviewed independently.
- Every 2019 pass needs a decision entry with the SHA-256 of each 2018 output (`runs/analysis/step10_decisions.json`). The evaluator refuses otherwise, and a 2019 pass runs once.
- 2019 was first inspected in Step 6 and re-evaluated for later steps. Each re-evaluation is disclosed and was made with the configuration frozen beforehand.

## E. Pre-registered experiments (Step 12)

- **STEP12a (congestion mechanism).**
  - Gate: a neural model with all PCP-history inputs removed vs the table, at matched fit.
  - Hypotheses H1 (model class), H2 (objective), H3 (volume attributes), H4 (tree capacity), H5 (volume scaling).
  - Signed one-sided contrasts with Holm adjustment. "Supported" requires an effect ≥ R/2, Holm p < 0.05, beyond the noise band and positive in every seed pairing.
  - Amendment A1 (post hoc, disclosed): a seed bootstrap and a volume-blind ("strict") attribute set.
- **STEP12b (simple hybrids).** Pre-stated "combines the strengths" rule versus the GNN:
  - load (top pick and top 5) not worse;
  - new-to-PCP higher by ≥ 5 points with CI > 0;
  - MIPS not worse;
  - fit ≥ min − 1 point.
- **STEP12c / 12d (complementarity, steering, heavier hybrids).** Frontier comparison with no pass/fail. For 2019, the setting and the criteria were recorded first (`runs/analysis/step12c/preregistration_2019.json`).
- **STEP12e (top-10/15).** The fixed 12c setting, not re-tuned (`Implementation/STEP12E_TOPK_PLAN.md`).

## F. Step 13: what the graph adds (pre-registered)

**Question.** Where does a graph model add value over information-matched non-graph models?

**Cells** (3 seeds each, 42 / 1 / 7, paired by seed ID; the Graph cell reuses the Step 10 seeds 42 and 1 and adds seed 7):

| Cell | Definition |
|---|---|
| Graph | the final GNN |
| NG+P | the GNN without message passing; the scorer also reads the query PCP's own encoding |
| G | no message passing, no PCP encoding, explicit history removed (the Step 12a gate runs) |
| GNN-nohist, NG+P-nohist | Graph and NG+P without explicit history: patient-history block, predecessor gap, PCP→specialist referral history, patient–specialist prior, insurer history |
| Table | pure-tabular LightGBM (117 columns) |
| Flat-RF | LightGBM on the 117 columns + unweighted means over what a 2-layer pass reaches, with the attributes the GNN reads |
| MLP-RF | an MLP on the Flat-RF row |
| Info-table-nohist | Flat-RF aggregates + 3-hop PCP→specialist means + graph counts, no explicit history |
| Info-table | Info-table-nohist + all explicit history |
| MLP-info | an MLP on the Info-table row |
| Ins-match | Info-table-nohist + the query insurer's share of the candidate's past encounters |

**The Flat-RF aggregates** (304 columns in all):
- *A1:* specialist ← past encounters: context, severity, diagnosis flags, disease-category distribution, encounter-categorical one-hots, and the means of the 16 sinusoidal time-encoding components;
- *A2:* specialist ← encounters ← PCPs: the PCP encoder inputs;
- *B1:* query ← PCP ← the PCP's past encounters: the same attributes as A1;
- the category cosine and age difference, plus no-history flags.

A hand-calculated check matched every aggregate to within 1.2e-7.

**MLP.**
- Two hidden layers, ReLU, dropout. Width and dropout tuned on {128, 256} × {0.1, 0.3} at seed 42; the chosen configuration is then trained at seeds 1 and 7.
- Same 21-row bundles, softmax loss and Adam settings as the GNN.
- Checkpoint selected on first-time validation MRR (selection only).

**Claims and components** (each one-sided, predicted > 0; native fit at the 30% weight unless stated):
- **H1a.** Graph − NG+P-nohist (prerequisite); GNN-nohist − NG+P-nohist; joint half-gap.
- **H1b.** GNN-nohist − X for X ∈ {Flat-RF, MLP-RF, Table, Info-table-nohist}.
- **H2.** Graph − C, with C ∈ {NG+P, Info-table, MLP-info}, in two slices, absolute and interaction:
  - low-history PCPs: ≤ 951 prior referrals;
  - thin-network PCPs: ≤ 519 distinct specialists so far, the 2018 lower tertile.
- **H3.** GNN-nohist − X for X ∈ {NG+P-nohist, Flat-RF, MLP-RF, Info-table-nohist}, on the share of new-to-PCP top-5 slots whose specialist had seen the patient's insurer.
  - Each model is steered to the same newness, 35.3%, by a ∈ {−0.60, …, 1.00} (step 0.01).
  - Re-matched in every bootstrap draw; ±1 point tolerance; the 2018 setting is frozen for 2019.

**Statistics.**
- Paired two-level bootstrap, 2,000 draws: seed IDs and PCPs are resampled, the same draws for every model.
- One-sided p = (1 + #draws ≤ 0 or undefined) / (1 + draws). A claim's p is the maximum over its components; Holm across 5 claims.
- "Supported" also requires every component's per-seed effect to be > 0 for all 3 seeds.
- Gates: inconclusive if more than 5% of draws are undefined, or if a model reaches the newness tolerance in fewer than 95% of draws.
- 2019 is evaluated once, after a decision entry hashes the 2018 outputs; it is reported as directional replication.

**Amendments** (all written before any outcome was seen):
- *A1:* finer matching grid.
- *A2:* rare-condition slice made descriptive (9 referrals in 2018), so Holm runs over 5 claims, not 6.
- *A3:* the reproduction check of the existing models accepts float32 rounding (scores within 1e-5) when every top-5 list and lens is identical.

The design manifest, the resolved manifest and the amendment records hash the code, the inputs and every run.

**Post hoc matched-fit comparison** (descriptive; after the 2019 pass). Top-pick load and newness between pairs of cells, matched on fit by the Step 12a random-mixture method, with a PCP bootstrap (`step13_matched_load.py`).

## G. Step 14: the steered recommender with the graph-feature table (pre-registered setting)

- **Families:**
  - single models: GNN, Table, Flat-RF;
  - equal blends: GNN + Table (the Step 12c blend) and GNN + Flat-RF;
  - a three-way blend 0.7·z((z(GNN) + z(Table) + z(Flat-RF)) / 3) + 0.3·z(MIPS).
- **Steering:** + a·[new to the PCP] − b·[busiest 10%] over the Step 12c grid (64 settings). Seed pairings by ID.
- **2018 rule:** per family, the setting with the lowest top-pick load among those with fit ≥ 45% and newness ≥ 35% in all three pairings. The proposed recommender is the family with the lowest such load.
- **2019** is checked once, at the 2018 settings. "Improves on the Step 12c blend" means lower top-pick load in all three pairings, with fit and newness within 1 point. A paired PCP bootstrap is reported for the differences.
- **Disclosure:** the family was motivated by Step 13 results that had seen 2019.

## H. Software and reproducibility

- Python 3.12.12, torch 2.14.0 (MPS), torch_geometric 2.8.0.post1, LightGBM 4.7.0.
- Every run writes `run_manifest.json`: input SHA-256, code SHA-256, configuration and seed. Builders refuse to overwrite completed artefacts.
- MPS is not bit-deterministic: same-seed reruns differ slightly (the 42_dup runs measure this).
- **Code:** https://github.com/bhaskar-ruthvik/GNN-Healthcare-Project (private; access on request). `REPRODUCE.md` maps each result to its command; `README.md` covers set-up.
