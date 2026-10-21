# Step 6 protocol — frozen 2026-09-18 before any test-split evaluation

## Runs (all: week graph cutoff, Adam 1e-3, patience 3 on overall eligible MRR, max 30 epochs, validation = fixed 3,000-query subset with full pools)
| Name | Configuration | Purpose |
|---|---|---|
| main_s42 | full model, seed 42 | primary |
| main_s1, main_s2 | full model, seeds 1, 2 | training-seed uncertainty |
| pairmlp | `--no-graph` | same explicit inputs, no message passing (not information-matched; see STEP4_PLAN) |
| abl_no_history | `--ablation no_history_features` | pair referral-history columns zeroed |
| abl_no_ps_prior | `--ablation no_patient_specialist_prior` | provisional patient–specialist prior zeroed |
| abl_no_history_no_ps | both of the above | no memorised pair/patient history in the scorer |
| abl_no_diagnosis | `--ablation no_diagnosis` | diagnosis contribution |
| abl_no_distance | `--ablation no_distance` | distance contribution |
| abl_no_ref_edges | `--ablation no_referral_edges` | graph referral structure contribution |
| ctx_patient_pcp | `--scorer-context enc_patient_pcp` | scorer variant from slide-8 discussion |

## Checkpoint selection (defined before running)
- Primary: `best.pt`, selected on overall eligible MRR (validation subset). Used for all headline and test numbers.
- Secondary: `best_firstobs.pt`, selected on first-observed-pair MRR. Reported alongside as a sensitivity check, never as the headline.
- Per-epoch logging includes overall, first-observed, repeat and unfamiliar-pair subgroup metrics.

## Metrics and reporting
- Recall@{1,5,10,20}, MRR, median rank; eligible and all-query denominators; target availability.
- Primary subgroups reported with every headline: first-observed patient–specialist pairs vs repeat observations (terminology per reviewer: "repeat observation" and "first-observed pair" — these do not prove clinical follow-up or first-ever referral).
- Additional subgroups: familiar vs unfamiliar PCP–specialist pair; entity seen/unseen in training (caveat: rolling history means unseen-in-training entities may have graph history at query time, so this is not a feature-only cold-start test).
- Uncertainty: paired bootstrap over patient clusters (1,000 resamples) on the same per-query results for model vs PatientRepeat, model vs PCPFrequency, and each ablation vs main; plus seed spread across main_s42/s1/s2.

## Test-split evaluation (once, after all validation runs finish)
- Full test (17,641 queries, full pools, week cutoff) for: main_s42 `best.pt`, main_s42 `best_firstobs.pt` (sensitivity), pairmlp `best.pt`, and all baselines. Ablation comparisons stay on validation.
- No configuration change after test evaluation.

## Not in scope
Loss reweighting or first-observed-only training (possible follow-up, would be a new protocol); hyperparameter search; patient–specialist distance.
