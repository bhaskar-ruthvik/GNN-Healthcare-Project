# Referral GNN — revised three-node architecture

Design agreed 2026-09-21. Status: versioned three-node inputs built and input-level checks passed on 2026-09-21; three-node model, trainer and CLI implemented and 21 model-level checks passed later on 2026-09-21 (`referral_gnn/three_node_model.py`, `referral_gnn/three_node_train.py`, `scripts/three_node_train.py`, `scripts/three_node_smoke.py`). No three-node training run yet; existing checkpoints are the four-node baseline. Historical specification: `reviews/ARCHITECTURE_baseline_20260918.md`. Existing runs and the frozen Step 6 protocol remain historical records.

## Objective
Rank candidate specialists given a patient's current encounter, PCP, clinical context and permitted history. The focus is first-observed patient–specialist pairs, not patients absent from training. Observed visits are referral-label proxies, not proof of clinical suitability or first-ever referral.

## Node types and patient linkage
Exactly three node types: PCP, Encounter, Specialist. PCP and Specialist remain separate roles even for the same NPI.

Remove Patient nodes and SEEN_BY edges. Each encounter carries that encounter's patient demographics, insurance/ACO context, current clinical inputs and strictly prior patient-history summaries. Retain patient identifiers privately for linking encounters, constructing history, subgroup evaluation and patient-cluster uncertainty; do not encode identifiers as predictive numeric features.

## Relationships
- PCP → HAS_ENCOUNTER → Encounter.
- Historical Encounter → REFERRED_TO → Specialist.
- Earlier Encounter → PRECEDES → Later Encounter, for the same patient only.

Historical HAS_ENCOUNTER and REFERRED_TO retain reverse copies. PRECEDES has no reverse copy. Current query encounters are receive-only: PCP and earlier encounters can send to them, but queries cannot send to PCPs, specialists, other queries or historical encounters. Current target referral edges and reverse copies are absent.

## Choosing preceding encounters
Connect from encounters on the most recent strictly earlier visible date for the same patient. If that date contains multiple encounters, include all of them; do not select an arbitrary specialist-containing event or invent an intraday order. No same-day PRECEDES edges. No predecessor means an empty temporal neighborhood plus an explicit missing-history flag.

For a historical destination at date t, sources must be on the latest patient date strictly before t. For a query at date d with graph cutoff c, temporal sources must be strictly before c. Under daily batching c=d; under weekly batching c is the established week-start cutoff. Never connect from another within-week query, even if its date precedes the destination. This preserves batched/single-query equivalence at the same cutoff and sacrifices some recent history under weekly batching.

Pair features and own-encounter history summaries continue to use the actual query day d. Provider snapshots and graph-visible history use c. Historical provider/specialty provenance limitations are unchanged. New-to-training entities may gain history during rolling evaluation; that is not a feature-only cold-start test.

## Clinical and temporal features
Keep existing demographics, provider attributes, pair features, missing/ambiguity masks and history summaries initially. Keep PCP–specialist distance as a feature. Patient-origin distance and any 50-mile candidate restriction are pending clarification.

User approved the existing 33 D_Disease_Code_Category labels on 2026-09-21. Use distinct multi-category bags per encounter instead of raw code bags; all 33 labels occur in training. Vocabulary indices: PAD=0, UNK=1, labels from 2. Source categories remain unchanged, including overlapping names; no six-group crosswalk is invented. Severity and CCI are not automatically added. Do not alter encounter grouping from the established retrospective key without a separate decision.

Record elapsed days on PRECEDES connections as destination date minus source date, strictly positive. Initial implementation proposal: supply log1p elapsed days and a no-predecessor flag to the destination encounter encoder; supply the same inputs to the no-temporal-edge control. This gives timing information without pretending stock HGTConv consumes edge attributes. All sources on a predecessor date share this interval. True edge-time-conditioned attention would require a separate implementation choice. Three duration bands remain undefined and are not introduced.

Existing encounter-age sinusoidal encoding can remain a feature, but is an input-level approximation, not a claim of implementing the original HGT edge-relative temporal mechanism.

## Model and controlled comparison
Start with the existing two HGT layers, hidden 64, four heads, per-node-type residual/normalization/dropout, and encounter–specialist MLP scorer with existing pair features. Derive all dimensions from the revised inputs; do not reuse baseline parameter-count or unused-gradient assertions. Remove patient-node scorer variants or redesign them explicitly.

Compare two newly trained three-node configurations:
1. Patient history summaries, with no PRECEDES edges.
2. Identical features and training settings, with PRECEDES edges.

Keep candidate policy, clinical representation, splits, samples, seeds and optimization budgets matched. Retain the four-node results as historical reference. Two layers cover only limited encounter history; summaries preserve longer-range information.

## Required verification before training
- Exactly three node types; query targets and reverse target edges absent.
- Every PRECEDES edge stays within a patient and has strictly increasing dates; no reversed temporal edge.
- Latest-earlier-date ties handled without arbitrary ordering.
- No temporal source for a query is on or after the graph cutoff.
- Scores in evaluation mode invariant to other batch queries, future targets/features and future-record presence, at a fixed cutoff with frozen training-fitted preprocessing.
- Query-alone and batched predictions agree at the same cutoff.
- Changing permitted patient history can affect predictions through the temporal path; gradients reach intended parameters.
- Train-only encoders, missing masks, candidate eligibility and pair-feature correctness remain intact.

## Evaluation and explanation
Use validation to develop the revision. Report Recall@1/5/10/20 and MRR for first-observed and repeat pairs, overall, and familiar/unfamiliar PCP–specialist pairs; retain both eligible-target and all-query denominators and candidate availability.

Before training, freeze a new revision protocol specifying checkpoint selection, stopping, budgets, seeds and comparisons. Proposed primary selection: first-observed validation MRR, with overall-selected checkpoint retained as a reference. This differs from the old frozen protocol and is not yet activated.

The 2019 test has already been examined. Any eventual new test result is a revised evaluation on that existing holdout, not a new untouched test. No new test run at this checkpoint.

SHAP is a confirmed requirement. Plan grouped input-feature attributions for a precisely defined candidate score, with historical graph fixed, train-only background, repeat/first-observed summaries and saved checkpoint/background/sample/explainer settings. These explain conditional feature contributions, not graph-edge importance or causal effects. Implementation and validation remain outstanding.

## Pending meeting details
- Disease categories are now inputs; any change to event grouping remains pending and has not been applied.
- Radius origin and treatment of unmapped/out-of-radius candidates.
- Provider averaging unit/time window, and MIPS as feature versus explicit ranking preference.
- Meaning and thresholds of three duration bands; total_time is not assumed to be a pre-referral input.

## Input build checkpoint (2026-09-21)
New inputs: `prepared_data/three_node_v1/model_inputs_PRIVATE.pkl`; source category mapping, manifest and completion marker in that directory. Builder: `scripts/build_three_node_inputs.py`; implementation: `referral_gnn/three_node_data.py`. Existing four-node pipeline remains untouched. The old trainer is not compatible with these inputs; adapt it before use.

13 input checks passed, covering synthetic boundary cases and 18 real daily/weekly graph contexts. These are input-construction checks, not a claim of completed model-level score invariance or GPU forward/backward verification. Continuous predecessor gap is implemented in destination inputs (same in both arms), with raw edge gaps retained as metadata. PRECEDES can be disabled while keeping all features identical.
