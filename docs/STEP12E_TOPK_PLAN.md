# STEP12e — the steered blend and the hybrids with longer lists (top 10, top 15)

Status: FROZEN 2026-10-19, before any K = 10 / 15 hybrid output exists. Descriptive; no pass/fail, no re-tuning.

## Question
At K = 5 the steered equal blend gave much lower top-pick load and the table's newness at about the same fit (STEP12c; held on
2019), and the trained-together hybrids did not beat it (STEP12d). The 10-12 top-K check showed the graph model and table converge
on list-slot load and MIPS with 10–15 names. Does the steered blend's advantage persist when the list has 10 or 15 names?

## Fixed setting (not re-tuned)
- Equal blend 0.5·z(GNN) + 0.5·z(table), 30% quality weight, steering + 0.3·new − 0.4·busy, exactly as chosen on 2018 at K = 5.
- Same seed pairings as STEP12c/12d: (GNN s42, table s42), (GNN s1, table s1), (GNN s42_dup, table L127/M50/R1000 s7);
  dual encoder and table-offset GNN seeds 42 / 1 / 7.
- Eligible sets filled to K (nearest beyond 25 mi) as in the 10-12 top-K check; cohort unchanged (5,550 / 2,668).

## Arms
GNN, table, plain equal blend, steered blend, dual encoder (steered and unsteered), table-offset GNN (steered and unsteered).
**Stacked LightGBM is excluded:** its GNN-score lookup was built for the top-5 candidate sets only, and rebuilding it would change
an input file pinned by hash in its run manifest.

## Measures (top K)
Fit (chosen specialist in the top K), top-pick load, list-slot load (share of the K slots on the busiest 10%), new to the PCP
(share of the K), MIPS gain over the list, "good" new picks, mean travel. Paired PCP-cluster 95% intervals vs the GNN.

## Frontier at K (2018 only)
Same steering grid (a, b ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6}) per family. Fit floor F_K = min(unsteered GNN fit,
unsteered table fit at K) − 0.5 pt, per pairing (mirrors the 0.45 floor at K = 5). Report, per family: the Pareto set and the
lowest top-pick load among settings with fit ≥ F_K and newness ≥ the unsteered table's newness at K.

## 2019
One test pass per K, through the two-pass guard (decision entry `step12e_topk_K10` / `_K15` recorded after the 2018 outputs).
Reported on 2019: GNN, table and the steered blend at the fixed setting only (as in STEP12c). Dual / offset are scored in the
same pass (same run list) but not reported for 2019.

## Reading
Descriptive. Whatever the K = 10 / 15 numbers show is reported against the question above; no extra seeds, settings or
re-tests to change a result.
