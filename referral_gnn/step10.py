"""Step 10 shared definitions (STEP10_PROTOCOL.md, frozen 2026-09-28): cohort exclusions, eligible sets, MIPS yardstick.

Used by the 10.1 descriptive script and the 10.2+ scorer evaluation so every analysis applies identical rules.
  §2.1 exclusion order: unknown specialty → non-choice/non-specific specialty → referred inactive (365 d) → referred MIPS none
                        → eligible set (before cap) < 5 → missing/unmapped patient ZIP
  §2.2 eligible set: same specialty (strictly before the day) ∧ active (claim in the 365 days before) ∧ unseen by the patient,
                     then ≤ 25 mi from the patient, filled to exactly 5 with the nearest beyond the cap when fewer are inside.
"""
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

NON_CHOICE = {"diagnostic radiology", "interventional radiology", "nuclear medicine", "pathology", "emergency medicine", "anesthesiology",
              "certified registered nurse anesthetist (crna)", "hospitalist", "critical care (intensivists)", "ambulatory surgical center",
              "radiation therapy centers"}
NON_SPECIFIC = {"internal medicine", "family practice", "general practice", "nurse practitioner", "physician assistant",
                "certified clinical nurse specialist", "single or multispecialty clinic or group practice"}
CAP_MILES = 25.0
ACTIVE_DAYS = 365
LIST_SIZE = 5
EXCLUSION_ORDER = ["unknown_specialty", "non_choice_specialty", "non_specific_specialty", "not_in_specialty_group", "referred_inactive",
                   "referred_mips_none", "insurer_unknown", "eligible_lt5", "patient_zip_missing"]
from .paths import workspace
DEFAULT_YARDSTICK = os.path.join(str(workspace()), "prepared_data", "mips_yardstick_v1")


@dataclass
class Yardstick:
    value: np.ndarray      # mean of distinct positive values; NaN = no MIPS score
    lo: np.ndarray         # min distinct positive value
    hi: np.ndarray         # max distinct positive value
    imputed: np.ndarray    # specialty mean (training information) used for "none"
    none: np.ndarray
    multi: np.ndarray
    rng: np.ndarray

    def point(self, idx):          # point estimate: imputed at the specialty mean
        idx = np.asarray(idx); v = self.value[idx]; return np.where(np.isnan(v), self.imputed[idx], v)


def yardstick_path_for(mi) -> str:
    """The MIPS yardstick built beside the inputs' own prepared bundle (STEP11 snapshots have their own; reviewer round 5).
    For the main inputs this is DEFAULT_YARDSTICK. Refuses if the sibling yardstick does not exist."""
    bp = getattr(mi, "bundle_path", None)
    if not bp: return DEFAULT_YARDSTICK
    p = os.path.join(os.path.dirname(bp), "mips_yardstick_v1")
    if not os.path.exists(os.path.join(p, "mips_yardstick_PRIVATE.pkl")):
        raise FileNotFoundError(f"no mips_yardstick_v1 beside the inputs' bundle ({p})")
    return p


def load_yardstick(mi, path: str = None) -> Yardstick:
    if path is None: path = yardstick_path_for(mi)                # the inputs' own yardstick (main → DEFAULT_YARDSTICK)
    y = pd.read_pickle(os.path.join(path, "mips_yardstick_PRIVATE.pkl")).set_index("provider_id")
    y.index = y.index.astype(str)
    y = y.reindex(mi.specialist_ids)
    assert y.mips_imputed.notna().all(), "yardstick does not cover every specialist"
    f = lambda c: y[c].to_numpy(float)
    return Yardstick(f("mips_value"), f("mips_min"), f("mips_max"), f("mips_imputed"), y.mips_none.to_numpy(bool), y.mips_multi.to_numpy(bool), f("mips_range"))


def specialty_strict(mi, spec_idx, day) -> np.ndarray:
    """Specialty vocab index read STRICTLY before `day` (0 = unknown). §2.2 rule 1."""
    mi.specialty_asof(np.array([0]), 0)                        # builds the cached as-of index
    spec_idx = np.asarray(spec_idx, np.int64)
    r = mi._sp_asof_idx.lookup(spec_idx, np.broadcast_to(np.asarray(day, np.int64), spec_idx.shape))
    return np.where(r >= 0, mi._sp_asof_val[np.clip(r, 0, None)], 0)


def last_claim_before(mi, spec_idx, day) -> np.ndarray:
    """Day of the latest specialist history row strictly before `day` (−10**9 if none)."""
    spec_idx = np.asarray(spec_idx, np.int64)
    r = mi.spec_hist_asof.lookup(spec_idx, np.broadcast_to(np.asarray(day, np.int64), spec_idx.shape))
    return np.where(r >= 0, mi.spec_hist_day[np.clip(r, 0, None)], -10 ** 9)


def specialty_names(mi) -> Dict[int, str]:
    d = {i: s for s, i in mi.spec_encoder.vocabs["specialty"].index.items()}; d[0] = "missing"; return d


def first_time_queries(mi, split: str) -> np.ndarray:
    q = np.sort(np.asarray(mi.query_enc[split]))
    return q[mi.ps_asof.lookup(mi.enc_patient[q] * mi.n_specialist + mi.enc_target[q], mi.enc_day[q]) < 0]


@dataclass
class Referral:
    enc: int; patient: int; pcp: int; day: int; target: int; specialty: int
    eligible: np.ndarray          # final eligible set (capped + fill), specialist indices
    dist: np.ndarray              # patient distance of each eligible specialist
    n_within_cap: int; filled: bool
    pre_cap: np.ndarray           # eligible before the distance cap
    pre_cap_dist: np.ndarray
    target_dist: float


class Step10Cohort:
    """Applies §2.1 and §2.2 for one split. `referrals()` yields Referral objects for the primary cohort; `counts` holds the
    exclusion funnel; `excluded` maps exclusion reason → list of encounter ids (for secondary tables)."""

    def __init__(self, mi, split: str, yardstick: Yardstick, cap: Optional[float] = CAP_MILES, active_days: Optional[int] = ACTIVE_DAYS,
                 keep_specialties: str = "primary", insurer_set: bool = False, include_seen: bool = False, fill_to: int = LIST_SIZE):
        """keep_specialties: "primary" (exclude non-choice and non-specific), "non_choice" or "non_specific" (that group only; A6).
        insurer_set: restrict the eligible set to specialists with prior claims under the patient's insurer (A6)."""
        assert keep_specialties in ("primary", "non_choice", "non_specific")
        self.mi, self.split, self.y, self.cap, self.active_days, self.keep = mi, split, yardstick, cap, active_days, keep_specialties
        self.insurer_set = insurer_set; self.include_seen = include_seen    # A12: keep specialists the patient has seen
        self.fill_to = max(int(fill_to), LIST_SIZE)                         # STEP12 top-K: fill the capped set to K (cohort rule stays < 5)
        self.names = specialty_names(mi)
        self.counts = OrderedDict(first_time=0, **{k: 0 for k in EXCLUSION_ORDER}, primary=0)
        self.excluded: Dict[str, List[int]] = {k: [] for k in EXCLUSION_ORDER}
        self._day_cache = {}
        self.zip_bad = np.asarray(mi.patient_zip_missing) | np.asarray(mi.patient_zip_unmapped)

    def _day_pool(self, d):
        if d not in self._day_cache:
            if len(self._day_cache) > 64: self._day_cache.clear()
            pool = np.where(self.mi.available_specialists(d))[0]
            sp = specialty_strict(self.mi, pool, d)
            act = np.ones(len(pool), bool) if self.active_days is None else (d - last_claim_before(self.mi, pool, d)) <= self.active_days
            self._day_cache[d] = (pool, sp, act)
        return self._day_cache[d]

    def referrals(self):
        mi = self.mi
        for q in first_time_queries(mi, self.split):
            q = int(q); self.counts["first_time"] += 1
            t, d, p = int(mi.enc_target[q]), int(mi.enc_day[q]), int(mi.enc_patient[q])
            sp = int(specialty_strict(mi, [t], d)[0]); name = self.names.get(sp, "missing")
            reason = None
            if sp == 0 or name == "missing": reason = "unknown_specialty"
            elif name in NON_CHOICE and self.keep == "primary": reason = "non_choice_specialty"
            elif name in NON_SPECIFIC and self.keep == "primary": reason = "non_specific_specialty"
            elif self.keep == "non_choice" and name not in NON_CHOICE: reason = "not_in_specialty_group"
            elif self.keep == "non_specific" and name not in NON_SPECIFIC: reason = "not_in_specialty_group"
            elif self.active_days is not None and d - int(last_claim_before(mi, [t], d)[0]) > self.active_days: reason = "referred_inactive"
            elif self.y.none[t]: reason = "referred_mips_none"
            pool = None
            if reason is None:
                pool_all, sp_all, act = self._day_pool(d)
                pool = pool_all[(sp_all == sp) & act]
                if not self.include_seen: pool = pool[mi.ps_asof.lookup(p * mi.n_specialist + pool, d) < 0]
                if self.insurer_set:
                    has, unknown = mi.insurer_history(np.full(len(pool), q), pool)
                    if len(pool) and unknown[0] == 1: reason = "insurer_unknown"
                    else: pool = pool[has.astype(bool)]
                if reason is None and len(pool) < LIST_SIZE: reason = "eligible_lt5"
                elif reason is None and self.zip_bad[p]: reason = "patient_zip_missing"
            if reason is not None:
                self.counts[reason] += 1; self.excluded[reason].append(q); continue
            dist = mi.patient_specialist_distance(np.full(len(pool), p), pool)
            td = float(mi.patient_specialist_distance(np.array([p]), np.array([t]))[0])
            if self.cap is None:
                elig, ed, n_in, filled = pool, dist, len(pool), False
            else:
                inside = dist <= self.cap; n_in = int(inside.sum())
                if n_in >= self.fill_to: elig, ed, filled = pool[inside], dist[inside], False
                else:
                    out = np.where(~inside)[0]; out = out[np.argsort(np.nan_to_num(dist[out], nan=np.inf), kind="stable")][: self.fill_to - n_in]
                    keep = np.concatenate([np.where(inside)[0], out]); elig, ed, filled = pool[keep], dist[keep], True
            self.counts["primary"] += 1
            yield Referral(q, p, int(mi.enc_pcp[q]), d, t, sp, elig, ed, n_in, filled, pool, dist, td)

