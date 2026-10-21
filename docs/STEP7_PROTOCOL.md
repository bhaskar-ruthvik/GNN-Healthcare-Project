# Step 7 protocol — three-node revision comparison (FROZEN 2026-09-22)

Status: FROZEN 2026-09-22 after reviewer answers (see §11) and user go-ahead. Edited only by dated amendment. No three-node training had been run at freeze time. The old `STEP6_PROTOCOL.md` remains the historical four-node record and is not modified.

## 1. Question
Do same-patient PRECEDES edges (earlier encounter → later encounter, message passing) improve specialist ranking beyond the same information supplied as per-encounter history summaries? Secondary: how does the revised three-node model (disease categories, no Patient node) compare with the four-node baseline and the history heuristics?

## 2. Arms (identical inputs, seeds, budgets, candidate policy)
| Arm | Command flag | Difference |
|---|---|---|
| A summaries-only | `--no-temporal` | PRECEDES edges dropped inside the model |
| B temporal | `--temporal` | PRECEDES edges active |
Both read `prepared_data/three_node_v1/model_inputs_PRIVATE.pkl` (hash 64927bdd…, verified against its manifest at run start). Both carry the predecessor-gap features. Parameter count identical (191,087).

## 3. Training settings (fixed)
Model: 2 HGTConv layers, hidden 64, 4 heads, dropout 0.2, scorer MLP 128–128–1. Loss: softmax cross-entropy over target + the 20 bundled seed-42 alternatives. Adam lr 1e-3, weight decay 1e-5, grad-clip 1.0, one optimizer step per weekly bucket (156/epoch). ReduceLROnPlateau on the selection metric, factor 0.5, patience 2. Max 100 epochs; early stopping patience 5 on the selection metric. Improvement is defined once for both the scheduler and early stopping: new > best × (1 + 1e-4). With scheduler patience 2, a plateau halves the LR after three non-improving validations and leaves two further epochs at the reduced LR before stopping. Optimizer steps per epoch and the LR after each epoch are logged. Graph bucket: week for both arms. Validation each epoch on the fixed 3,000-query subset with full candidate pools.

## 4. Checkpoint selection (decided here)
- **Primary selection metric: first-observed-pair validation MRR** (`--select firstobs_mrr_eligible`). Rationale: the project goal is recommending specialists the patient has not seen; overall MRR is dominated by repeat observations that a heuristic already handles.
- `best_overall.pt` (overall eligible MRR) and `best_firstobs.pt` are always saved; `best.pt` is the primary. All headline numbers use `best.pt`; `best_overall.pt` numbers are reported alongside as the reference.
- The first-observed cohort of the validation sample has 1,274 queries, of which 1,192 are eligible (target in pool) under the saved candidate policy; selection uses the eligible denominator and is noisier than overall; patience 5 (not 3) compensates.

## 5. Seeds and order
1. Pilot: 3 epochs per arm, seed 42, to confirm curves and timings (~15 min total). Not a result.
2. Main: arm A and arm B, seed 42, to early stopping (~2 min/epoch each; expect 1–3 h per arm).
3. Repeats: seeds 1 and 2 for both arms, in that order, as time allows before Friday. Seed spread reported for whatever completed.

## 6. Evaluation (validation subset, both arms, both checkpoints)
- Recall@1/5/10/20, MRR, median rank; eligible and all-query denominators; target availability.
- Cohorts: first-observed vs repeat patient–specialist pairs (primary); familiar vs unfamiliar PCP–specialist pairs; patient with vs without prior history. Same 3,000 queries and pools as all prior results.
- Secondary metric (new): **specialty match** — share of eligible queries whose top-1 (and any of top-5) candidates share the chosen specialist's specialty. Frozen definition: each specialist's specialty is the latest non-null value observed on or before the query day (cutoff-safe as-of lookup, same as the patient-history construction); a candidate with missing/UNK specialty never matches; queries whose chosen specialist has missing specialty are excluded from this metric's denominator and their count reported. Computed in analysis from per-query top-K; not used in training or selection.
- Comparisons with paired patient-cluster bootstrap (1,000 resamples): B − A (primary); each arm − four-node `main_s42` (reference; note the inputs differ: categories vs codes, no Patient node); each arm − patient-repeat and PCP-frequency heuristics.
- Week-vs-day cutoff assessment on the same trained checkpoint for BOTH arms (validation), to test whether any temporal-edge gain depends on the cutoff policy. A daily-trained arm B is optional after seed repeats and, on its own, cannot isolate the edge effect under daily training (that would need a daily-trained A too); compare optimizer-step budgets, not epochs only.

## 7. Test split
The 2019 test was already examined in Step 6. One revised test pass at the end for both arms' `best.pt` plus heuristics, disclosed as a revised evaluation on a previously inspected holdout. No configuration changes after it.

## 8. Not in this protocol (separate, later)
Hard-negative sampling; loss reweighting toward first-observed events; 50-mile radius; MIPS averaging; severity/CCI; duration bands; SHAP (own protocol after the comparison); no-graph control (may be run internally, not for the presentation).

## 9. Reporting
Append to `RESULTS.md` under a new dated section with run paths, selection metric, both checkpoints, bootstrap tables and the specialty-match metric. Preserve all prior results. Deck: new file, never over the user-edited deck.

## 10. Reviewer questions — resolved
a. Patience 5. b. ReduceLROnPlateau kept, identical for both arms; realized LR trajectories and stopping epochs may differ and are logged. Four-node comparisons are historical references with changed inputs, selection and training policy, not isolated architecture effects. c. Same-checkpoint day-vs-week for both arms (§6); daily-trained B optional later.

## 11. Freeze record
Reviewed 2026-09-22 (log entry "Reviewer answers to Step 7 protocol questions"); all recommendations adopted verbatim above. Frozen on user go-ahead the same day.
