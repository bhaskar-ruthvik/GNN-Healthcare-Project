"""Non-learned ranking baselines using the same evaluation protocol as the GNN.

All history used is strictly earlier than the query day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AsOfIndex, ModelInputs
from .evaluate import Scorer


class Popularity(Scorer):
    """Global count of prior observed referrals to each specialist (as-of day)."""

    def __init__(self, mi: ModelInputs):
        self.mi = mi
        df = pd.DataFrame({"s": mi.enc_target, "d": mi.enc_day}).groupby(["s", "d"]).size().rename("n").reset_index().sort_values(["s", "d"])
        df["cum"] = df.groupby("s")["n"].cumsum()
        self.idx = AsOfIndex(df["s"].to_numpy(), df["d"].to_numpy())
        self.cum = df["cum"].to_numpy()
        self.day = None

    def _pop(self, q):
        rows = self.idx.lookup(np.arange(self.mi.n_specialist), int(self.mi.enc_day[q]))     # exact query day
        return np.where(rows >= 0, self.cum[np.clip(rows, 0, None)], 0).astype(float)

    def score(self, q, cand):
        return self._pop(q)[cand]


class PCPFrequency(Popularity):
    """Prior PCP-specialist referral count; ties broken by recency, then popularity."""

    def score(self, q, cand):
        """Exact lexicographic order: prior count desc, then recency asc (missing last), then popularity desc.
        Returned score is the rank position, so equal keys give equal scores (ties preserved)."""
        prior, recency, _ = self.mi.raw_pair_history(np.full(len(cand), q), cand)
        rec = np.where(recency < 0, np.iinfo(np.int64).max, recency)
        pop = self._pop(q)[cand]
        keys = np.stack([prior, -rec, pop], axis=1)                       # all "higher is better"
        _, inv = np.unique(keys, axis=0, return_inverse=True)            # unique rows sorted lexicographically ascending
        return inv.reshape(-1).astype(float)                              # higher index = better under all three keys


class PatientRepeat(PCPFrequency):
    """Specialists the patient already saw rank first (by patient-specialist prior count, then recency of the
    PCP pair), then everything else in PCPFrequency order. Exact lexicographic; ties preserved."""

    def score(self, q, cand):
        mi = self.mi
        d = int(mi.enc_day[q])
        ps_rows = mi.ps_asof.lookup(mi.enc_patient[q] * mi.n_specialist + cand, d)
        ps_prior = np.where(ps_rows >= 0, mi.ps_cum[np.clip(ps_rows, 0, None)], 0)
        prior, recency, _ = mi.raw_pair_history(np.full(len(cand), q), cand)
        rec = np.where(recency < 0, np.iinfo(np.int64).max, recency)
        pop = self._pop(q)[cand]
        keys = np.stack([ps_prior, prior, -rec, pop], axis=1)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        return inv.reshape(-1).astype(float)


class NearestDistance(Scorer):
    def __init__(self, mi: ModelInputs):
        self.mi = mi

    def score(self, q, cand):
        d = self.mi.distance[self.mi.enc_pcp[q], cand]
        return np.where(np.isnan(d), -np.inf, -d)


class Random(Scorer):
    def __init__(self, mi: ModelInputs, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def score(self, q, cand):
        return self.rng.random(len(cand))


BASELINES = {"random": Random, "popularity": Popularity, "pcp_frequency": PCPFrequency, "patient_repeat": PatientRepeat, "nearest_distance": NearestDistance}
