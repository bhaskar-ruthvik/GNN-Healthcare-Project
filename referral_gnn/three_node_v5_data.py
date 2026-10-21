"""three-node-v5 inputs (STEP10_PROTOCOL §2.3, model side): v4 with the MIPS input re-defined.

For every specialist history row (provider, day t) — used only by queries on later days — MIPS is computed from the raw
distinct values first seen on or before t (mips_yardstick_v1/mips_values):
  value column      mean of the distinct POSITIVE values seen so far, scaled with the existing training scaler;
                    if none seen yet ("no MIPS score", 0 in source) → the specialty's training-period mean (imputed)
  missing column    → mips_none   (1 = only zeros seen so far; the value is imputed)
  ambiguous column  → mips_multi  (1 = more than one positive value seen so far)
  appended          mips_partial_zero (a zero alongside positive values), mips_range (max − min of positive values, /100)
Specialists with no earlier history row get the imputed value with mips_none = 1 (no value can have been seen).
v4's "missing-as-low" rule is dropped: v4.provider_snapshot is bypassed. Everything else (encounters, pairs, graph) is v4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import to_day
from .three_node_v3_data import ThreeNodeInputsV3
from .three_node_v4_data import ThreeNodeInputsV4

N_EXTRA = 2          # mips_partial_zero, mips_range


class ThreeNodeInputsV5(ThreeNodeInputsV4):
    schema_version = "three-node-v5"

    def __init__(self, v4: ThreeNodeInputsV4, mips_values: pd.DataFrame, yardstick: pd.DataFrame):
        self.__dict__.update(v4.__dict__)
        self.encounter_blocks = dict(v4.encounter_blocks)
        self.pair_feature_blocks = dict(v4.pair_feature_blocks)
        self._enc_static_cache = None
        y = yardstick.set_index("provider_id"); y.index = y.index.astype(str); y = y.reindex(self.specialist_ids)
        assert y.mips_imputed.notna().all()
        self.mips_imputed_raw = y.mips_imputed.to_numpy(float)
        scaler = self.spec_encoder.scalers["mips_score"]
        self.mips_imputed_scaled = scaler.transform(self.mips_imputed_raw)[0]

        # distinct values per specialist with first-seen day; cumulative stats over positive values in first-seen order
        mv = mips_values.copy(); mv["s"] = mv.provider_id.astype(str).map(self.specialist_index); mv = mv.dropna(subset=["s"])
        mv["s"] = mv.s.astype(np.int64); mv["d"] = to_day(mv.first_day)
        pos = mv[mv.value > 0].sort_values(["s", "d", "value"]).reset_index(drop=True)
        g = pos.groupby("s").value
        pos["csum"] = g.cumsum(); pos["cmax"] = g.cummax(); pos["cmin"] = g.cummin()
        big = 1 << 20
        self._p_key = pos.s.to_numpy() * big + pos.d.to_numpy(); self._p_start = np.searchsorted(self._p_key, np.arange(self.n_specialist) * big)
        self._p_csum, self._p_cmax, self._p_cmin = pos.csum.to_numpy(), pos.cmax.to_numpy(), pos.cmin.to_numpy()
        self._zero_first = mv[mv.value == 0].groupby("s").d.min().reindex(np.arange(self.n_specialist)).fillna(np.inf).to_numpy()
        self._big = big

        # re-encode the as-of history rows
        ent = self.spec_hist_asof.entity[np.argsort(self.spec_hist_asof.order)]
        val, none, multi, pz, rng = self.mips_asof(ent, self.spec_hist_day + 1)          # values first seen on or before the row day
        j = 3 * self.spec_encoder.numeric_cols.index("mips_score"); self.mips_cols = (j, j + 1, j + 2)
        x = self.spec_hist_num.copy()
        x[:, j] = val; x[:, j + 1] = none; x[:, j + 2] = multi
        self.spec_hist_num = x
        self.spec_hist_mips_extra = np.stack([pz, rng], axis=1).astype(np.float32)
        self.mips_rule = "v5: as-of mean of distinct positive values; none -> imputed specialty mean (flag); multi flag; partial-zero flag; range/100"

    def mips_asof(self, spec, day_exclusive):
        """MIPS state from values first seen strictly before `day_exclusive`. Returns scaled value, none, multi, partial_zero, range/100."""
        spec = np.asarray(spec, np.int64); dq = np.broadcast_to(np.asarray(day_exclusive, np.int64), spec.shape)
        end = np.searchsorted(self._p_key, spec * self._big + dq, "left"); st = self._p_start[spec]; n = end - st
        has = n > 0; last = np.clip(end - 1, 0, None)
        # csum/cmax/cmin restart per specialist (groupby cumulative), so the mean is csum[last] / n
        mean_raw = np.where(has, self._p_csum[last] / np.maximum(n, 1), np.nan)
        rng = np.where(has, (self._p_cmax[last] - self._p_cmin[last]) / 100.0, 0.0)
        scaled = np.where(has, self.spec_encoder.scalers["mips_score"].transform(np.where(has, mean_raw, 0.0))[0], self.mips_imputed_scaled[spec])
        zero_seen = self._zero_first[spec] < dq
        return scaled.astype(np.float32), (~has).astype(np.float32), (n > 1).astype(np.float32), (zero_seen & has).astype(np.float32), rng.astype(np.float32)

    def provider_snapshot(self, day: int):
        snap = ThreeNodeInputsV3.provider_snapshot(self, day)                        # bypass v4's missing-as-low rule
        x = snap["specialist_num"]; ok = snap["specialist_has_history"]
        v, m, a = self.mips_cols
        x[~ok, v] = self.mips_imputed_scaled[~ok]; x[~ok, m] = 1.0; x[~ok, a] = 0.0   # no earlier row: nothing can have been seen
        extra = np.zeros((self.n_specialist, N_EXTRA), np.float32)
        rows = self.spec_hist_asof.lookup(np.arange(self.n_specialist), day)
        extra[ok] = self.spec_hist_mips_extra[rows[ok]]
        snap["specialist_num"] = np.concatenate([x, extra], axis=1)
        return snap

    def dims(self):
        d = super().dims(); d["specialist_num"] = self.spec_encoder.num_dim + N_EXTRA; return d
