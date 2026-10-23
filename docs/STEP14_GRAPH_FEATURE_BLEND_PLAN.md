# STEP14 — Does adding the graph-feature table improve the steered recommender?

Status: **FROZEN** 2026-10-23, before any STEP14 number was computed. Later changes only as dated amendments at the end.

## 0. Question and honest framing

STEP12c showed that an equal blend of the GNN and the pure table, steered by a·[new to PCP] − b·[busiest 10%], roughly
halves top-pick load at table-level newness, and this held on 2019. STEP13 (post hoc, both years seen) showed that two-hop graph
summaries added to the table (Flat-RF) cut its load at matched fit with no newness cost.

**Question:** does a steered recommender that uses the graph-feature table reach a better load / newness trade-off than the
STEP12c steered blend, at the same fit floor?

**Disclosure.** The idea comes from STEP13 results on both 2018 and 2019. 2019 is therefore not a clean hold-out for *whether* to
try Flat-RF. It remains a clean check of the **settings and the family chosen on 2018 by the rule below**, the same discipline
as STEP12c/12e. No new models are trained; all scores come from the STEP13 candidate-score dumps.

## 1. Families (all at the 30% quality weight; candidates.py arithmetic)

| Family | Score before steering |
|---|---|
| G | GNN alone: 0.7·z(GNN) + 0.3·z(as-of MIPS) |
| T | pure table alone |
| F | graph-feature table (Flat-RF) alone |
| B_GT | STEP12c blend: 0.7·z(0.5·z(GNN) + 0.5·z(T)) + 0.3·z(MIPS) |
| B_GF | 0.7·z(0.5·z(GNN) + 0.5·z(F)) + 0.3·z(MIPS) |
| B_GTF | 0.7·z((z(GNN) + z(T) + z(F)) / 3) + 0.3·z(MIPS) |

Steering: + a·[new to the PCP] − b·[busiest 10%], with a, b ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6} (the STEP12c grid, 64
settings). z within each referral's eligible set; stable ties; top 5.

**Seed pairings by seed ID** (42, 1, 7): Graph (v5 s42, s1, STEP13 s7), Table (s42, s1, s7), Flat-RF (s42, s1, s7), as resolved in
`runs/analysis/step13/resolved_manifest.json`. B_GT therefore uses Graph s7 where STEP12c used s42_dup (disclosed; B_GT is
recomputed here, not copied).

**Lenses per setting:** fit (chosen in top 5), top-pick load (busiest 10%), top-5 load, newness (top 5), MIPS gain (top 5),
"good" new picks (new, higher MIPS than chosen, not busiest).

## 2. Choice on 2018 (validation dump), fixed before 2019

1. **Setting per family and pairing set** (the STEP12c rule): among the 64 settings, those with fit ≥ 0.45 **and** newness ≥ 0.35
   in **all three** pairings; choose the one with the lowest top-pick load averaged over pairings. Ties: smaller a + b, then
   smaller b. A family with no qualifying setting is "cannot reach the floors".
2. **Proposed recommender:** among families that reach the floors, the one whose chosen setting has the lowest average top-pick
   load. Ties: fewer component models.
3. Also reported on 2018: per-family frontiers (lowest top-pick load at newness ≥ 0.30 / 0.35 / 0.40 with fit ≥ 0.45, per pairing).

The choices are written to `runs/analysis/step14/choice_2018.json` and hashed in a decision entry before the 2019 analysis.

## 3. 2019 check (once)

At each family's 2018 setting (no re-tuning), on the 2019 dump: all lenses per pairing and seed-averaged. Pre-stated comparison:

- **"Improves on the STEP12c blend"** if the proposed recommender (if it is not B_GT) has a lower top-pick load than B_GT in all
  three pairings, with fit and newness each no more than 1 point below B_GT's in every pairing.
- Paired PCP-cluster bootstrap (2,000 draws, seed-averaged) for the proposed − B_GT differences in top-pick load, newness, fit and
  MIPS gain: reported, not a gate.
- If the proposed recommender is B_GT itself, the answer is "no improvement from the graph-feature table" and B_GT's 2019
  lenses are reported.

Possible outcomes are all reportable: improvement, no improvement, or a family that cannot reach the floors. No rescue settings.

## 4. Outputs

`scripts/analysis/step14/step14_blend.py --split validation | test`; `runs/analysis/step14/{choice_2018.json, results_2019.json}`;
aggregates only.
