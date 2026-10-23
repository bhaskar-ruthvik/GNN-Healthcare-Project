# STEP13 amendments

Amendments to `STEP13_GRAPH_VALUE_PLAN.md` (frozen 2026-10-21, SHA-256 42c11662195dbba811115e704f53c4f15181c343dbbaa2d11c3b64468af36809).
The frozen plan is never edited; each amendment is dated, states its basis, and is recorded in the design manifest by hash.

## A1 — H3 matching grid (2026-10-21, user decision; before any STEP13 model was trained)
**Basis.** A pre-run smoke test and the STEP12c frontier aggregates showed that the frozen grid moves top-5 newness by 3–8 points
between neighbouring values (GNN seed 1: 31.6% at a = 0.4, 38.8% at a = 0.6). The ±1-point tolerance around 35.3% was therefore
often unattainable, which would make H3 inconclusive by construction, independent of the models.
**Change.** §4 grid: a ∈ {−0.60, −0.59, …, 0.99, 1.00} (step 0.01, 161 values). Target (35.3%), tolerance (±1 pt), tie rule
(closest; then smaller |a|; then smaller a), re-matching in every draw, gates and the 2019 freeze are unchanged.

## A2 — Rare-condition slice becomes descriptive (2026-10-21, user decision; before any STEP13 model was trained)
**Basis.** Under the frozen §3 rule the 11 least frequent categories cover 9 of the 5,550 referrals in the 2018 cohort (attribute
counts only), so H2-rare could never meet the size floor (≥ 100 referrals, ≥ 10 PCPs): untestable whatever the results. Widening
"rare" until the floor is met would require categories with up to 583 training encounters, which no longer tests rarity.
**Change.** H2-rare-condition is removed from the confirmatory family: Holm runs over **5** claim units (H1a, H1b,
H2-low-history, H2-thin-network, H3). The rare slice (frozen definition) is reported descriptively (size, lenses, Graph − comparator
fit differences without inference). That genuinely rare conditions almost never reach first-time specialist referrals here is
reported as a finding.

## A3 — Backwards-compatibility criterion for GNN scores (2026-10-21, user decision; before any STEP13 outcome evaluation)
**Basis.** The compatibility check (§1) re-scored the existing checkpoints on 2018. Table scores and G's per-referral lenses
reproduced exactly. Graph s42 / s1 scores differed from the STEP12c dump by at most 1.9e-6 / 1.3e-6 (scores up to ≈ 12): float32
rounding from non-deterministic GPU (MPS) reduction order. All 5,550 top-5 lists were identical in order and every per-referral
native lens (fit, load, newness, MIPS gain, distance) was identical, for both runs. The frozen 1e-6 absolute tolerance is below
float32 resolution at these magnitudes.
**Change.** A re-scored model passes when its scores agree with the STEP12 record within 1e-5 absolute **and** every referral's
top-5 list (in order) and native lenses are identical. G (never dumped) is unchanged: per-referral lenses within 1e-6. Recorded
post-design as `runs/analysis/step13/amendments/A3.json` (the design manifest is never edited).
