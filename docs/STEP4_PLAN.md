# Step 4 — model and training pipeline (current specification)

Status: IMPLEMENTED and verified 2026-09-18 (18/18 smoke-test checks). Supersedes the proposal reviewed the same day; the change record is in §11. Implements ARCHITECTURE.md on top of the validated `ModelInputs` object from step 3.

## 1. Files

| File | Purpose |
|---|---|
| `referral_gnn/data.py` | Bundle loader, train-only encoders, as-of lookups, `buckets()`, `graph_batch()` (receive-only batch encounters), `day_batch()` wrapper. |
| `referral_gnn/model.py` | `NodeEncoder`, `ReferralGNN` (encoders → optional 2×HGTConv → pair scorer), `ModelConfig` with ablation flags; PairMLP = `use_graph=False`. |
| `referral_gnn/train.py` | `Trainer`: bucketed training, sampled softmax, validation, early stopping, best checkpoint, run manifest, device probe with CPU fallback. |
| `referral_gnn/evaluate.py` | `evaluate()` full-pool ranking, `metrics()`, `breakdowns()`, `ModelScorer`; `strict_patient_day` kept as a verification mode. |
| `referral_gnn/baselines.py` | `Random`, `Popularity`, `PCPFrequency` (exact lexicographic), `NearestDistance`. |
| `scripts/train.py` | CLI: `--name --epochs --patience --lr --accumulate --device --seed --ablation … --scorer-context --no-graph --graph-bucket --max-train-days`. Never evaluates on test. |
| `scripts/smoke_test.py` | Pre-training checks (§6). |
| `scripts/build_inputs.py` | Step 3 build + validation. |

Runs: `Implementation/runs/<timestamp>_<name>/` with `run_manifest.json` (inputs SHA256, code SHA256, torch/pyg/python versions, device and any fallback, seed, configs, bucket counts), `metrics.jsonl` (per epoch), `best.pt` (weights + model config only), `summary.json`.

## 2. Model

Hidden d = 64. 296,841 parameters.

- **Node encoders** (`NodeEncoder`): numeric → Linear(·, 64); each categorical → Embedding(size, 16) with 0 = UNK/missing; concat (+ extra) → Linear → LayerNorm → ReLU.
  - Patient: num 6, cats [3, 7, 3]. PCP: num 10, cats [7]. Specialist: num 47, cats [64, 154].
  - Encounter: num 91 = patient context 6 + patient history 3 + prior-specialty multi-hot 64 + diagnosis flags 2 (no-code, UNK share) + time encoding 16; cats [3, 7, 3]; extra = diagnosis vector (32) = masked mean of code embeddings over the (N, 6) code matrix. Diagnosis vocabulary: 0 = PAD (excluded from mean), 1 = UNK (unseen or < 5 training occurrences), 1,915 codes from 2.
- **Message passing**: 2 × `HGTConv(64, 64, heads=4)` over {patient, pcp, encounter, specialist} and six edge types (three forward + `rev_*`). Per layer: residual, LayerNorm per type, ReLU, Dropout 0.2. Full-graph forward; encounter tensor always all 111,104 rows; invisible encounters are isolated.
- **Scorer**: MLP([z_enc ‖ z_spec ‖ pair(8)]) 136 → 128 → 128 → 1, ReLU, Dropout 0.2. `scorer_context="enc_patient_pcp"` appends z_patient and z_pcp (input 264).
- **Structural note**: with the confirmed configuration, 23 parameter tensors get no gradient (patient-side HGT outputs, last-layer PCP outputs, single-relation relation priors). Inherent to two layers + encounter-only scorer + no Patient–Encounter edge; the smoke test asserts the unused set equals exactly this set.

### Ablation flags (`ModelConfig`), applied identically at train and eval
| Flag | Effect |
|---|---|
| `no_diagnosis` | zero the diagnosis vector and both diagnosis flags |
| `no_distance` | zero pair columns 5–6 (distance, distance missing) |
| `no_history_features` | zero pair columns 0–4 (prior count; recency + flag; first-observed age + flag) |
| `no_patient_specialist_prior` | zero pair column 7 (provisional feature) |
| `no_patient_history` | zero encounter block: prior count, days since, prior-specialty multi-hot |
| `no_referral_edges` | drop REFERRED_TO and its reverse from message passing; pair features kept |
| `no_time_encoding` | zero the time block |
| `use_graph=False` | PairMLP: same encoders and scorer, no HGT. Not an information-matched control: it loses PCP referral patterns and patient–PCP structure that only reach the encounter through message passing. Report as "same explicit inputs, no graph". |

## 3. Training

- Graph batches from `ModelInputs.buckets("train", policy, query_subset=train_query_enc)`; shuffled each epoch. Policy "day" → 1,009 steps/epoch; "week" → 156.
- Per step: `graph_batch(cutoff, batch_enc)` → one HGT forward → scores for the batch's eligible queries × 21 bundled candidates (positive in column 0, seed-42 alternatives) → cross-entropy with target 0. Pair features precomputed once for all 1,072,512 training pairs.
- Adam lr 1e-3, weight decay 1e-5, grad-clip 1.0. One optimizer step per bucket (`--accumulate k` to accumulate).
- Each epoch: validation on the fixed 3,000-query subset with full pools, same bucket policy as training. Selection metric: eligible-query MRR. Patience 3, max 30 epochs, best checkpoint saved.
- Seeds set for torch/numpy/python; MPS is not bit-deterministic.
- Device: `auto` → MPS if available. `_probe_device()` runs one real forward/backward at startup; `_fallback_to_cpu()` moves model and optimizer state to CPU and is also invoked on any device error during a training step or validation, which is then retried on CPU (the gradients of the interrupted accumulation window are dropped). Fallbacks are recorded in the run manifest.
- Measured: step ≈ 0.66 s (either policy); epoch ≈ 11 min (day) / 1.7 min (week); validation subset ≈ 1.2 min; MPS driver memory ≈ 3.5 GB.

## 4. Evaluation

- Per query: pool = specialists first seen strictly before the query day (~5–6k); pair features exact per query day; scores in chunks ≤ 200k pairs; pessimistic rank (ties count against) and optimistic rank recorded.
- Recorded flags: `target_available, familiar_pair (prior count > 0), target_seen_in_train, patient_seen_in_train, pcp_seen_in_train, patient_has_history, n_candidates`.
- Metrics: Recall@{1,5,10,20}, MRR, median rank; two denominators: eligible queries, and all queries with unavailable targets as misses. Breakdowns by the flags above. Target availability reported as the candidate-policy ceiling.
- Cost: 3,000 validation queries ≈ 1.2 min; full test (17,641) ≈ 7 min; full validation (37,443) ≈ 15 min.
- `isolate="patient" | "query"` (verification mode): keeps the same cutoff as `bucket` and forwards one patient's encounters, or one query alone, per batch. Because batch encounters are receive-only, both must reproduce the full-batch ranks at that cutoff exactly (checked in the smoke test for both policies). Comparisons across different cutoffs (week vs day) are not isolation checks; they measure the history-delay approximation.

## 5. Baselines (step 5), same evaluation code
`Random`; `Popularity` (global prior referral count as-of the exact query day); `PCPFrequency` (exact lexicographic: prior pair count desc, recency asc with missing last, popularity desc; equal keys keep equal scores); `NearestDistance` (missing distance ranks last).

## 6. Pre-training checks (`scripts/smoke_test.py`, all passing)
1. Forward shapes, no NaN, parameter count.
2. Gradient reaches every non-structural parameter; unused set equals the documented structural set.
3. Initial loss ≈ ln 21.
4. Withheld-label invariance (day): permuting all on/after-day referral targets, and perturbing future encounters' features, leave scores unchanged; dropping historical referral edges or zeroing pair features changes them.
5. Bucket coverage (day, week): every query exactly once, cutoff ≤ batch days, within window, cutoff ≥ split start; week batch has no referral edge on/after cutoff and seen_by edges all before cutoff.
6. Same-cutoff isolation equivalence: full-batch ranks equal query-isolated and patient-isolated ranks under the week cutoff and under the day cutoff (40 validation queries). Runtime CPU fallback unit test: optimizer state moved, weights preserved, next step succeeds.
7. Future-context invariance (week): first-day queries unchanged when other batch encounters' targets are permuted, features perturbed, or they are removed; query-alone scores equal in-batch scores. Same query-isolated equivalence on a day batch. Batch encounters have no reverse has_encounter edge.
8. Overfit on two days: loss → ≈ 0.
9. CPU vs MPS forward agreement (3.6e-6).
10. Timing and memory.

## 7. Pilot and week-vs-day assessment (step 5)
1. Baselines on the validation subset (Random, Popularity, PCPFrequency, NearestDistance).
2. Pilot: 3 epochs, `--graph-bucket week`, validation only. Confirms the pipeline trains, ranks, beats Random and ideally PCPFrequency, and that timings hold. Not the main result. Test split untouched until step 6.
3. Week-vs-day assessment: weekly batching delays graph history and provider snapshots by up to six days (provider observations can be older still). Treat it as an approximation to be measured, not assumed: evaluate the pilot checkpoint on the validation subset under both cutoffs, and train a short daily-cutoff run for comparison. If the gap is material, the main experiment uses `--graph-bucket day` (~11 min/epoch).

## 8. Risks
Small per-step batches (median ~33 queries under "day"; ~330 under "week") → `--accumulate`; cold-start entities handled by features only and made visible in breakdowns; ~10% of held-out encounters contain an unseen diagnosis code → `no_diagnosis` bounds the effect; candidate ceiling reported via all-query metrics; provider snapshot staleness under "week" ≤ 6 days.

## 9. Out of scope for step 4
Hyperparameter search, new features, patient–specialist distance, clinical-suitability evaluation, slide updates.

## 10. Reviewer questions and answers
Eligible-query MRR for selection (reviewer: equivalent choice on a fixed subset); one optimizer step per bucket by default; PairMLP kept with the documented limitation.

## 11. Change record

### After first review (2026-09-18)
Diagnosis PAD separated from UNK with explicit empty/UNK-share flags; keyword `device`; controlled invariance test instead of feature-perturbation probe; test split reserved for step 6; history ablations defined precisely; PairMLP limitation documented; runtime measured, `graph_bucket` option added.

### After second review (2026-09-18) — blocking finding fixed
- **Week-bucket leak**: batch encounters were connected to their PCP in both directions, so a later-in-week encounter (whose patient-history block can contain an earlier query's target specialty) could reach the earlier query via encounter → PCP → encounter in two layers. Fix: batch encounters are receive-only (forward PCP → encounter edge only; no reverse edge, no referral edge) and SEEN_BY edges require first seen before the cutoff. Consequence: a query's embedding depends only on history before the cutoff and its own features under both policies; same-day/same-week context sharing is impossible by construction. New checks in §6 item 6 verify this, including query-alone equivalence.
- `--graph-bucket` exposed in the CLI; `TrainConfig` default is "day" (safe default); pilot will pass `--graph-bucket week`.
- Patient-day mode re-described as patient-day isolation and demoted to a verification.
- `PCPFrequency` tie-break replaced by exact lexicographic ordering.
- Device auto-selection now probes with a real forward/backward and falls back to CPU at runtime.
- This document consolidated to the current specification.

### Clarifications (2026-09-18, third pass)
- Isolation comparisons now use the same cutoff as the batch: `evaluate(..., isolate=...)` no longer switches to daily history. Verified: week full-batch == week isolated; day full-batch == day isolated.
- Week vs day is recorded as an approximation to assess in step 5 (§7 item 3), not as equivalent. For the untrained model the two cutoffs give different target ranks on 75% of 40 sampled queries, so the effect is not negligible a priori.
- CPU fallback now also recovers mid-training and mid-validation, not only at the startup probe.
