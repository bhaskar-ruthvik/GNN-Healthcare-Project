"""Recent referral load per specialist (Step 10.7, STEP10_PROTOCOL A10). Strictly-before counts only.

load(spec, day, window) = number of observed referral events to the specialist on days in [day − window, day), i.e. strictly
before `day`. Uses the referral events of every split, as the PCP-history features do (only events before the query day).
Shared by the v6 inputs (load as a model input), load-aware training weights, and the load-aware re-ranking variants.
"""
import numpy as np

_BIG = 1 << 20


class RecentLoad:
    def __init__(self, mi):
        key = np.sort(mi.enc_target.astype(np.int64) * _BIG + mi.enc_day.astype(np.int64))
        self.key = key

    def count(self, spec, day, window: int = 365) -> np.ndarray:
        spec = np.asarray(spec, np.int64); d = np.broadcast_to(np.asarray(day, np.int64), spec.shape)
        hi = np.searchsorted(self.key, spec * _BIG + d, "left")
        lo = np.searchsorted(self.key, spec * _BIG + np.maximum(d - window, 0), "left")
        return (hi - lo).astype(np.float64)
