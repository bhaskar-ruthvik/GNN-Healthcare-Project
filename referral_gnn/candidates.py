"""Candidate-level score dumps and the ranking arithmetic shared by the STEP12c–e analyses (complementarity, frontiers, steered
blend, heavier hybrids, top-K).

A dump is written by `scripts/evaluate/step10_evaluate.py --dump-scores`: for every evaluated referral, its eligible specialists
(flattened, with `offsets`) carrying each model's raw score (`score::<run>`) and the knowable lens attributes (busiest-10% flag,
new to the PCP, MIPS, as-of MIPS, distance, ...). The functions below reproduce, operation for operation, the arithmetic the
analyses used when they were run, keeping each script's order of operations (the reproduction check is pending; see REPRODUCE.md):
  base score at the 30% quality weight    0.7·z(model score) + 0.3·z(as-of MIPS), z within the referral's candidates
  equal blend                             0.7·z(0.5·z(GNN) + 0.5·z(table)) + 0.3·z(as-of MIPS)
  steering                                + a·[new to the PCP] − b·[busiest 10%]
"""
import numpy as np
import pandas as pd

QUALITY_WEIGHT = 0.3
STEERING_GRID = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
STEERED_BLEND = {"a_new": 0.3, "b_busy": 0.4}            # chosen on 2018 (STEP12c), frozen before 2019
# Seed pairings: one GNN run with one table run (and the STEP12d hybrids of the same seed), by run-name suffix.
SEED_PAIRINGS = [
    dict(G="_v5_unmasked_s42", T="_lgbm_tabular_s42", D="_s12d_dual_s42", O="_s12d_offset_s42", S="_stack_L127_M50_R1000_s42"),
    dict(G="_v5_unmasked_s1", T="_lgbm_tabular_s1", D="_s12d_dual_s1", O="_s12d_offset_s1", S="_stack_L127_M50_R1000_s1"),
    dict(G="_v5_unmasked_s42_dup", T="_lambdarank_L127_M50_R1000_s7", D="_s12d_dual_s7", O="_s12d_offset_s7", S="_stack_L127_M50_R1000_s7"),
]


def zs(x):
    """Standardise within one referral's candidates (all-equal scores → zeros)."""
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


class Dump:
    """A candidate-score dump with the lens attributes as typed arrays."""

    def __init__(self, path: str):
        Z = np.load(path, allow_pickle=False)
        self.path, self.Z = path, Z
        self.off = Z["offsets"]; self.n_ref = len(self.off) - 1
        self.seg = [(self.off[i], self.off[i + 1]) for i in range(self.n_ref)]
        self.tpos, self.refm, self.pcp = Z["target_pos"], Z["ref_mips"], Z["pcp"]
        self.busy, self.new = Z["busy"].astype(float), Z["new"].astype(float)
        self.mips, self.none, self.ma = Z["mips"], Z["mips_none"].astype(bool), Z["mips_asof"].astype(float)
        self.keys = [k for k in Z.files if k.startswith("score::")]

    @property
    def dist(self):
        d = self.Z["dist"].astype(float); return np.where(d < 0, np.nan, d)

    def key(self, suffix: str) -> str:
        k = [x for x in self.keys if x.endswith(suffix)]
        assert len(k) == 1, (suffix, k)
        return k[0]

    def pairings(self, families: str = "GT"):
        """Per seed pairing, {family: score key} for the families present in this dump."""
        return [{f: self.key(P[f]) for f in families} for P in SEED_PAIRINGS]

    def scores(self, key: str):
        return self.Z[key].astype(float)

    def weighted(self, s):
        """Per referral: the model score at the 30% quality weight."""
        return [0.7 * zs(s[a:b]) + 0.3 * zs(self.ma[a:b]) for a, b in self.seg]

    def blend(self, g, t):
        """Per referral: the equal blend of two models at the 30% quality weight."""
        return [0.7 * zs(0.5 * zs(g[a:b]) + 0.5 * zs(t[a:b])) + 0.3 * zs(self.ma[a:b]) for a, b in self.seg]

    def steer(self, i: int, a_new: float, b_busy: float):
        a, b = self.seg[i]
        return a_new * self.new[a:b] - b_busy * self.busy[a:b]

    def top(self, base, i: int, a_new: float = 0.0, b_busy: float = 0.0, k: int = 5):
        """Flat indices of the top-k candidates of referral i under base[i] + steering (stable ties)."""
        return self.seg[i][0] + np.argsort(-(base[i] + self.steer(i, a_new, b_busy)), kind="stable")[:k]


def pareto(d: pd.DataFrame, load: str = "busy1", new: str = "new5") -> list:
    """Points not beaten on both lower load and higher newness (sorted by load)."""
    d = d.sort_values([load, new], ascending=[True, False]); keep, best = [], -1
    for _, r in d.iterrows():
        if r[new] > best: keep.append(r); best = r[new]
    return keep


def pcp_bootstrap_index(pcp, seed: int, draws: int = 2000):
    """PCP-cluster bootstrap: (codes per referral, number of PCPs, draws × PCPs resampling index)."""
    codes, uniq = pd.factorize(pcp); nc = len(uniq)
    return codes, nc, np.random.default_rng(seed).integers(0, nc, size=(draws, nc))
