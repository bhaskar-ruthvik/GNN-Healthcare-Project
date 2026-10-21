"""three-node-v6 inputs (STEP10_PROTOCOL A10, Option B variant 1): v5 plus recent referral load as specialist inputs.

At every graph cutoff (bucket start), each specialist gets two appended columns computed strictly before the cutoff:
  log1p(referral events in the previous 365 days), log1p(referral events in the previous 90 days),
standardised with means/SDs fitted on the training-period cutoffs. Everything else is v5.
"""
from __future__ import annotations

import numpy as np

from .load import RecentLoad
from .three_node_v5_data import ThreeNodeInputsV5

N_LOAD = 2


class ThreeNodeInputsV6(ThreeNodeInputsV5):
    schema_version = "three-node-v6"

    def __init__(self, v5: ThreeNodeInputsV5):
        self.__dict__.update(v5.__dict__)
        self.encounter_blocks = dict(v5.encounter_blocks)
        self.pair_feature_blocks = dict(v5.pair_feature_blocks)
        self._enc_static_cache = None
        self.recent_load = RecentLoad(self)
        cut = np.unique([c for c, _, _ in self.buckets("train", "week", query_subset=self.train_query_enc)])
        vals = np.stack([self._raw_load(int(c)) for c in cut[:: max(1, len(cut) // 60)]])          # ~60 training cutoffs
        self.load_mean = vals.mean(axis=(0, 1)); self.load_std = np.maximum(vals.std(axis=(0, 1)), 1e-6)

    def _raw_load(self, day):
        s = np.arange(self.n_specialist)
        return np.stack([np.log1p(self.recent_load.count(s, day, 365)), np.log1p(self.recent_load.count(s, day, 90))], axis=1)

    def provider_snapshot(self, day: int):
        snap = super().provider_snapshot(day)
        z = ((self._raw_load(int(day)) - self.load_mean) / self.load_std).astype(np.float32)
        snap["specialist_num"] = np.concatenate([snap["specialist_num"], z], axis=1)
        return snap

    def dims(self):
        d = super().dims(); d["specialist_num"] = d["specialist_num"] + N_LOAD; return d
