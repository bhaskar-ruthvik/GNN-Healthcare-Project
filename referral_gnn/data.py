"""Model-input construction from the prepared bundle.

Everything here is derived from the private bundle saved by the cleaning
notebook (Section 15).  Encoders, vocabularies and scalers are fit on the
training split only.  Temporal visibility follows the strict "earlier day"
rule documented in ARCHITECTURE.md.

Nothing in this module prints or stores patient- or provider-level values;
aggregate diagnostics only.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SPEC_NUMERIC = [
    "practice_size", "years_in_medicare", "beneficiary_count", "service_count",
    "mips_score", "rating", "rating_count", "average_panel_age",
    "average_panel_risk", "panel_age_lt65", "panel_age_65_74",
    "panel_age_75_84", "panel_age_gt84", "panel_female", "panel_male",
]
SPEC_CAT = ["specialty", "credentials"]
PCP_NUMERIC = ["practice_size", "years_in_medicare", "beneficiary_count"]
PCP_CAT = ["credentials"]
PATIENT_CAT = ["gender", "insurance_company", "aco_status"]

MAX_DX = 6            # max diagnosis codes per encounter in the bundle
DX_MIN_COUNT = 5      # training occurrences below this -> UNK
TIME_DIM = 16         # sinusoidal temporal-encoding width
PAIR_FEATURE_NAMES = [
    "log_prior_referral_count",
    "log_days_since_last_referral", "recency_missing",
    "log_first_observed_age", "first_observed_missing",
    "distance_scaled", "distance_missing",
    "log_patient_specialist_prior_count",
]
_BIG = 10 ** 6        # day ordinals are ~1.8e4, so entity*_BIG + day is unique


def to_day(values) -> np.ndarray:
    """datetime-like -> int64 days since epoch."""
    return pd.to_datetime(pd.Series(values)).values.astype("datetime64[D]").astype("int64")


def day_to_str(d: int) -> str:
    return str(np.datetime64(int(d), "D"))


# --------------------------------------------------------------------------
# Encoders (fit on training data only)
# --------------------------------------------------------------------------

@dataclass
class NumericScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, values) -> "NumericScaler":
        v = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).to_numpy()
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return cls(0.0, 1.0)
        std = float(v.std())
        return cls(float(v.mean()), std if std > 0 else 1.0)

    def transform(self, values) -> Tuple[np.ndarray, np.ndarray]:
        v = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).to_numpy()
        missing = ~np.isfinite(v)
        out = np.where(missing, 0.0, (v - self.mean) / self.std)
        return out.astype(np.float32), missing


@dataclass
class Vocab:
    """String -> int with 0 reserved for unknown/missing."""
    index: Dict[str, int]

    @classmethod
    def fit(cls, values, min_count: int = 1) -> "Vocab":
        s = pd.Series(values).astype(object)
        s = s[s.map(lambda x: isinstance(x, str))]
        counts = s.value_counts()
        keep = sorted(counts[counts >= min_count].index)
        return cls({v: i + 1 for i, v in enumerate(keep)})

    @property
    def size(self) -> int:
        return len(self.index) + 1

    def transform(self, values) -> np.ndarray:
        s = pd.Series(values).astype(object)
        return s.map(lambda x: self.index.get(x, 0) if isinstance(x, str) else 0).to_numpy(dtype=np.int64)


class DxVocab(Vocab):
    """Diagnosis vocabulary: 0 = PAD (no code in slot), 1 = UNK (code unseen or
    rare in training), real codes from 2. Padding is distinct from unknown."""
    PAD, UNK = 0, 1

    @classmethod
    def fit(cls, values, min_count: int = 1) -> "DxVocab":
        base = Vocab.fit(values, min_count)
        return cls({k: v + 1 for k, v in base.index.items()})

    @property
    def size(self) -> int:
        return len(self.index) + 2

    def transform(self, values) -> np.ndarray:
        s = pd.Series(values).astype(object)
        return s.map(lambda x: self.index.get(x, self.UNK) if isinstance(x, str) else self.UNK).to_numpy(dtype=np.int64)


def time_encoding(age_days: np.ndarray, dim: int = TIME_DIM) -> np.ndarray:
    """Sinusoidal encoding of a non-negative age in days (HGT-style relative encoding)."""
    age = np.clip(age_days.astype(np.float64), 0, None)[:, None]
    half = dim // 2
    freqs = 1.0 / (10000.0 ** (np.arange(half) / half))
    ang = age * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# As-of lookup: last row with day < d for each entity
# --------------------------------------------------------------------------

class AsOfIndex:
    """Rows sorted by (entity, day). lookup() returns, for each query entity and
    query day, the original row index of the latest row with day strictly
    before the query day, or -1 when none exists."""

    def __init__(self, entity: np.ndarray, day: np.ndarray):
        entity = np.asarray(entity, dtype=np.int64)
        day = np.asarray(day, dtype=np.int64)
        assert (day < _BIG).all() and (day >= 0).all()
        self.order = np.lexsort((day, entity))
        self.entity = entity[self.order]
        self.key = self.entity * _BIG + day[self.order]

    def lookup(self, entity_q: np.ndarray, day_q) -> np.ndarray:
        entity_q = np.asarray(entity_q, dtype=np.int64)
        day_q = np.broadcast_to(np.asarray(day_q, dtype=np.int64), entity_q.shape)
        pos = np.searchsorted(self.key, entity_q * _BIG + day_q, side="left") - 1
        ok = pos >= 0
        pos_c = np.clip(pos, 0, None)
        ok &= self.entity[pos_c] == entity_q
        return np.where(ok, self.order[pos_c], -1)


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------

class Bundle:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "manifest_PRIVATE.json")) as f:
            self.manifest = json.load(f)
        ld = lambda n: pd.read_pickle(os.path.join(path, f"{n}_PRIVATE.pkl"))
        self.encounters = ld("encounter_features")
        self.patient_context = ld("patient_context")
        self.pcp_query = ld("pcp_query_features")
        self.targets = ld("referral_targets")
        self.splits = ld("split_assignments")
        self.pair_daily = ld("pair_daily")
        self.train_candidates = ld("train_candidates")
        self.candidate_audit = ld("candidate_audit")
        self.eval_queries = ld("evaluation_queries")
        self.pcp_history = ld("pcp_history")
        self.specialist_history = ld("specialist_history")
        self.pcp_locations = ld("pcp_locations")
        self.specialist_locations = ld("specialist_locations")
        self.specialist_first_seen = ld("specialist_first_seen")


# --------------------------------------------------------------------------
# Model inputs
# --------------------------------------------------------------------------

@dataclass
class ProviderEncoder:
    numeric_cols: List[str]
    cat_cols: List[str]
    scalers: Dict[str, NumericScaler]
    vocabs: Dict[str, Vocab]

    @property
    def num_dim(self) -> int:
        # value, missing, ambiguous per numeric col; ambiguous per cat col
        return 3 * len(self.numeric_cols) + len(self.cat_cols)

    def encode(self, hist: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(hist)
        num = np.zeros((n, self.num_dim), dtype=np.float32)
        j = 0
        for c in self.numeric_cols:
            v, miss = self.scalers[c].transform(hist[c])
            amb = hist[f"{c}_ambiguous"].fillna(False).astype(bool).to_numpy()
            num[:, j] = v
            num[:, j + 1] = miss
            num[:, j + 2] = amb
            j += 3
        for c in self.cat_cols:
            num[:, j] = hist[f"{c}_ambiguous"].fillna(False).astype(bool).to_numpy()
            j += 1
        cat = np.stack([self.vocabs[c].transform(hist[c]) for c in self.cat_cols], axis=1) if self.cat_cols else np.zeros((n, 0), np.int64)
        return num, cat


class ModelInputs:
    """All tensors and lookups needed to build day batches.

    Attributes are numpy arrays; `day_batch()` converts to torch tensors.
    """

    def __init__(self, bundle: Bundle, validation_query_sample: int = 3000, seed: int = 42):
        self.bundle_path = bundle.path
        self.seed = seed
        rng = np.random.default_rng(seed)
        b = bundle

        # ---- entity indexers -------------------------------------------------
        enc = b.encounters.merge(b.targets, on="encounter_id").merge(b.splits[["encounter_id", "split"]].rename(columns={"split": "_split2"}), on="encounter_id")
        assert (enc["split"] == enc["_split2"]).all()
        enc = enc.drop(columns="_split2")
        enc["_day"] = to_day(enc["_service_day"])
        enc = enc.sort_values(["_day", "encounter_id"], kind="stable").reset_index(drop=True)
        self.encounter_ids = enc["encounter_id"].astype(str).to_numpy()
        self.enc_index = {e: i for i, e in enumerate(self.encounter_ids)}
        self.enc_day = enc["_day"].to_numpy(np.int64)
        self.enc_split = enc["split"].astype(str).to_numpy()

        self.patient_ids = np.array(sorted(enc["mso_membership_id"].astype(str).unique()))
        self.pcp_ids = np.array(sorted(enc["primary_care_provider_npi"].astype(str).unique()))
        self.specialist_ids = np.array(sorted(b.specialist_first_seen.index.astype(str)))
        pat_ix = {p: i for i, p in enumerate(self.patient_ids)}
        pcp_ix = {p: i for i, p in enumerate(self.pcp_ids)}
        spec_ix = {s: i for i, s in enumerate(self.specialist_ids)}
        self.patient_index, self.pcp_index, self.specialist_index = pat_ix, pcp_ix, spec_ix
        self.n_patient, self.n_pcp, self.n_specialist, self.n_encounter = len(pat_ix), len(pcp_ix), len(spec_ix), len(enc)

        self.enc_patient = enc["mso_membership_id"].astype(str).map(pat_ix).to_numpy(np.int64)
        self.enc_pcp = enc["primary_care_provider_npi"].astype(str).map(pcp_ix).to_numpy(np.int64)
        self.enc_target = enc["rendering_provider_npi"].astype(str).map(spec_ix).to_numpy(np.int64)
        assert not np.isnan(self.enc_target.astype(float)).any()

        train_mask = self.enc_split == "train"
        self.train_end_day = int(self.enc_day[train_mask].max())      # last training day
        self.split_days = {s: (int(self.enc_day[self.enc_split == s].min()), int(self.enc_day[self.enc_split == s].max())) for s in ("train", "validation", "test")}

        # ---- specialist availability ---------------------------------------
        fs = b.specialist_first_seen.copy()
        fs.index = fs.index.astype(str)
        self.spec_first_seen = to_day(fs.loc[self.specialist_ids, "first_seen"])

        # ---- patient context per encounter --------------------------------------
        pc = b.patient_context.set_index("encounter_id").loc[self.encounter_ids]
        train_pc = pc[train_mask]
        self.age_scaler = NumericScaler.fit(train_pc["age"])
        self.patient_vocabs = {c: Vocab.fit(train_pc[c]) for c in PATIENT_CAT}
        age, age_miss = self.age_scaler.transform(pc["age"])
        age_amb = pc["age_ambiguous"].fillna(False).astype(bool).to_numpy()
        cat_amb = np.stack([pc[f"{c}_ambiguous"].fillna(False).astype(bool).to_numpy() for c in PATIENT_CAT], axis=1)
        self.enc_patient_num = np.concatenate([age[:, None], age_miss[:, None], age_amb[:, None], cat_amb], axis=1).astype(np.float32)
        self.enc_patient_cat = np.stack([self.patient_vocabs[c].transform(pc[c]) for c in PATIENT_CAT], axis=1)
        self.patient_num_dim = self.enc_patient_num.shape[1]

        # patient node features = context at the patient's first observed encounter
        first_enc = pd.Series(np.arange(self.n_encounter)).groupby(self.enc_patient).min()
        first_enc = first_enc.reindex(np.arange(self.n_patient)).to_numpy(np.int64)
        self.patient_num = self.enc_patient_num[first_enc]
        self.patient_cat = self.enc_patient_cat[first_enc]

        # ---- diagnosis codes ----------------------------------------------------
        train_codes = [c for codes in enc.loc[train_mask, "diagnosis_codes"] for c in codes]
        self.dx_vocab = DxVocab.fit(train_codes, min_count=DX_MIN_COUNT)
        dx = np.full((self.n_encounter, MAX_DX), DxVocab.PAD, dtype=np.int64)
        dx_mask = np.zeros((self.n_encounter, MAX_DX), dtype=bool)
        for i, codes in enumerate(enc["diagnosis_codes"]):
            codes = list(codes)[:MAX_DX]
            if codes:
                ids = self.dx_vocab.transform(codes)
                dx[i, : len(ids)] = ids
                dx_mask[i, : len(ids)] = True
        self.enc_dx, self.enc_dx_mask = dx, dx_mask
        self.enc_dx_empty = ~dx_mask.any(axis=1)                 # explicit "no diagnosis code" flag
        self.enc_dx_unk_share = np.where(self.enc_dx_empty, 0.0, (dx == DxVocab.UNK).sum(1) / np.maximum(dx_mask.sum(1), 1)).astype(np.float32)

        # ---- provider histories & encoders ----------------------------------------
        sh = b.specialist_history.copy()
        sh["_day"] = to_day(sh["_service_day"])
        sh["_ent"] = sh["provider_id"].astype(str).map(spec_ix)
        sh = sh.dropna(subset=["_ent"]).reset_index(drop=True)
        sh["_ent"] = sh["_ent"].astype(np.int64)
        ph = b.pcp_history.copy()
        ph["_day"] = to_day(ph["_service_day"])
        ph["_ent"] = ph["provider_id"].astype(str).map(pcp_ix)
        ph = ph.dropna(subset=["_ent"]).reset_index(drop=True)
        ph["_ent"] = ph["_ent"].astype(np.int64)

        sh_train, ph_train = sh[sh["_day"] <= self.train_end_day], ph[ph["_day"] <= self.train_end_day]
        self.spec_encoder = ProviderEncoder(SPEC_NUMERIC, SPEC_CAT,
                                            {c: NumericScaler.fit(sh_train[c]) for c in SPEC_NUMERIC},
                                            {c: Vocab.fit(sh_train[c]) for c in SPEC_CAT})
        self.pcp_encoder = ProviderEncoder(PCP_NUMERIC, PCP_CAT,
                                           {c: NumericScaler.fit(ph_train[c]) for c in PCP_NUMERIC},
                                           {c: Vocab.fit(ph_train[c]) for c in PCP_CAT})
        self.spec_hist_num, self.spec_hist_cat = self.spec_encoder.encode(sh)
        self.pcp_hist_num, self.pcp_hist_cat = self.pcp_encoder.encode(ph)
        self.spec_hist_asof = AsOfIndex(sh["_ent"].to_numpy(), sh["_day"].to_numpy())
        self.pcp_hist_asof = AsOfIndex(ph["_ent"].to_numpy(), ph["_day"].to_numpy())
        self.spec_hist_day = sh["_day"].to_numpy()
        self.pcp_hist_day = ph["_day"].to_numpy()

        # cutoff-safe specialty of each encounter's target specialist: latest non-null
        # specialty observed on or before the encounter's own day (never later).
        # Used only to build patient prior-referred-specialty histories.
        self.n_specialty = self.spec_encoder.vocabs["specialty"].size
        nn = sh.dropna(subset=["specialty"])
        sp_idx = AsOfIndex(nn["_ent"].to_numpy(), nn["_day"].to_numpy())
        sp_rows = sp_idx.lookup(self.enc_target, self.enc_day + 1)          # day < d+1  <=>  day <= d
        sp_vals = self.spec_encoder.vocabs["specialty"].transform(nn["specialty"])
        self.enc_target_specialty_asof = np.where(sp_rows >= 0, sp_vals[np.clip(sp_rows, 0, None)], 0)

        # ---- patient encounter history (strictly earlier than own day) ----------
        self._build_patient_history()

        # ---- pair history (PCP-specialist) --------------------------------------
        pdl = b.pair_daily.copy()
        pdl["_day"] = to_day(pdl["_service_day"])
        pdl["_pcp"] = pdl["primary_care_provider_npi"].astype(str).map(pcp_ix)
        pdl["_spec"] = pdl["rendering_provider_npi"].astype(str).map(spec_ix)
        pdl = pdl.dropna(subset=["_pcp", "_spec"]).reset_index(drop=True)
        self.pair_ent = (pdl["_pcp"].astype(np.int64) * self.n_specialist + pdl["_spec"].astype(np.int64)).to_numpy()
        self.pair_day = pdl["_day"].to_numpy(np.int64)
        self.pair_cum = pdl["cumulative_events"].to_numpy(np.int64)
        self.pair_first = to_day(pdl["first_pair_date"])
        self.pair_asof = AsOfIndex(self.pair_ent, self.pair_day)

        # ---- patient-specialist pair history ------------------------------------
        ps = pd.DataFrame({"ent": self.enc_patient.astype(np.int64) * self.n_specialist + self.enc_target, "day": self.enc_day})
        ps = ps.groupby(["ent", "day"]).size().rename("n").reset_index().sort_values(["ent", "day"])
        ps["cum"] = ps.groupby("ent")["n"].cumsum()
        self.ps_asof = AsOfIndex(ps["ent"].to_numpy(), ps["day"].to_numpy())
        self.ps_cum = ps["cum"].to_numpy(np.int64)

        # ---- distances (PCP x specialist, great-circle miles) ---------------------
        self.distance = self._distance_matrix(b)
        tc = b.train_candidates
        self.distance_scaler = NumericScaler.fit(tc["pcp_specialist_geo_distance_miles"])

        # ---- edges ----------------------------------------------------------------
        pairs = pd.DataFrame({"p": self.enc_patient, "c": self.enc_pcp, "d": self.enc_day})
        sb = pairs.groupby(["p", "c"])["d"].min().reset_index()
        self.seen_by_src = sb["p"].to_numpy(np.int64)
        self.seen_by_dst = sb["c"].to_numpy(np.int64)
        self.seen_by_first_day = sb["d"].to_numpy(np.int64)

        # ---- training candidates ----------------------------------------------------
        tc = tc.copy()
        self.tc_enc = tc["encounter_id"].astype(str).map(self.enc_index).to_numpy(np.int64)
        self.tc_spec = tc["rendering_provider_npi"].astype(str).map(spec_ix).to_numpy(np.int64)
        self.tc_label = tc["label"].to_numpy(np.int64)
        order = np.lexsort((-self.tc_label, self.tc_enc))          # group by query, positive first
        self.tc_enc, self.tc_spec, self.tc_label = self.tc_enc[order], self.tc_spec[order], self.tc_label[order]
        self.tc_bundle_prior = tc["prior_referral_count"].to_numpy(np.int64)[order]
        self.tc_bundle_recency = tc["days_since_last_referral"].astype("float").to_numpy()[order]
        self.tc_bundle_distance = tc["pcp_specialist_geo_distance_miles"].to_numpy()[order]
        self.train_query_enc = np.unique(self.tc_enc)

        # ---- evaluation query sets -------------------------------------------------
        self.query_enc = {s: np.where(self.enc_split == s)[0] for s in ("train", "validation", "test")}
        val = self.query_enc["validation"]
        self.validation_sample = np.sort(rng.choice(val, size=min(validation_query_sample, len(val)), replace=False))

        self.pair_feature_dim = len(PAIR_FEATURE_NAMES)
        # column ranges of the encounter numeric input, used by ablations
        p = self.patient_num_dim
        self.encounter_blocks = {
            "patient_context": (0, p),
            "patient_history": (p, p + 3 + self.n_specialty),
            "diagnosis_flags": (p + 3 + self.n_specialty, p + 5 + self.n_specialty),
            "time": (p + 5 + self.n_specialty, p + 5 + self.n_specialty + TIME_DIM),
        }
        self.pair_feature_blocks = {"referral_history": [0, 1, 2, 3, 4], "distance": [5, 6], "patient_specialist": [7]}

    # ----------------------------------------------------------------------
    def _build_patient_history(self):
        n = self.n_encounter
        prior_count = np.zeros(n, np.int64)
        days_since = np.full(n, -1, np.int64)
        prior_specs = np.zeros((n, self.n_specialty), np.uint8)
        order = np.lexsort((self.enc_day, self.enc_patient))
        cur_p = -1
        hist_days: List[int] = []
        hist_specs: List[int] = []
        i = 0
        while i < len(order):
            e = order[i]
            p, d = self.enc_patient[e], self.enc_day[e]
            if p != cur_p:
                cur_p, hist_days, hist_specs = p, [], []
            # gather all encounters of this patient on day d (simultaneous)
            j = i
            same = []
            while j < len(order) and self.enc_patient[order[j]] == p and self.enc_day[order[j]] == d:
                same.append(order[j]); j += 1
            prior = len(hist_days)
            spec_vec = np.zeros(self.n_specialty, np.uint8)
            if hist_specs:
                spec_vec[np.array(hist_specs)] = 1
            for e2 in same:
                prior_count[e2] = prior
                days_since[e2] = d - hist_days[-1] if hist_days else -1
                prior_specs[e2] = spec_vec
            for e2 in same:
                hist_days.append(d); hist_specs.append(self.enc_target_specialty_asof[e2])
            i = j
        self.enc_prior_encounters = prior_count
        self.enc_days_since_prev = days_since
        self.enc_prior_specialties = prior_specs

    def _distance_matrix(self, b: Bundle) -> np.ndarray:
        pl = b.pcp_locations.copy(); pl.index = pl.index.astype(str)
        sl = b.specialist_locations.copy(); sl.index = sl.index.astype(str)
        pl = pl.reindex(self.pcp_ids); sl = sl.reindex(self.specialist_ids)
        lat1 = np.radians(pl["lat"].astype(float).to_numpy())[:, None]
        lon1 = np.radians(pl["lon"].astype(float).to_numpy())[:, None]
        lat2 = np.radians(sl["lat"].astype(float).to_numpy())[None, :]
        lon2 = np.radians(sl["lon"].astype(float).to_numpy())[None, :]
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        return (2 * 3958.7613 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float64)   # NaN where a location is missing

    # ----------------------------------------------------------------------
    # As-of features
    # ----------------------------------------------------------------------
    def available_specialists(self, day: int) -> np.ndarray:
        """Boolean mask over specialists first observed strictly before `day`."""
        return self.spec_first_seen < day

    def provider_snapshot(self, day: int) -> Dict[str, np.ndarray]:
        """As-of (strictly earlier) provider features for all PCPs and specialists."""
        out = {}
        for name, n, asof, num, cat, enc in (
            ("pcp", self.n_pcp, self.pcp_hist_asof, self.pcp_hist_num, self.pcp_hist_cat, self.pcp_encoder),
            ("specialist", self.n_specialist, self.spec_hist_asof, self.spec_hist_num, self.spec_hist_cat, self.spec_encoder),
        ):
            rows = asof.lookup(np.arange(n), day)
            ok = rows >= 0
            x_num = np.zeros((n, enc.num_dim), np.float32)
            x_cat = np.zeros((n, len(enc.cat_cols)), np.int64)
            x_num[ok] = num[rows[ok]]
            x_cat[ok] = cat[rows[ok]]
            # providers with no earlier row: all numeric values missing
            j = 0
            for _ in enc.numeric_cols:
                x_num[~ok, j + 1] = 1.0
                j += 3
            out[f"{name}_num"] = x_num
            out[f"{name}_cat"] = x_cat
            out[f"{name}_has_history"] = ok
        return out

    def specialty_asof(self, spec_idx, day) -> np.ndarray:
        """Specialty vocab index of each specialist as observed on or before `day` (0 = unknown). Cutoff-safe."""
        if getattr(self, "_sp_asof_idx", None) is None:
            rows = self.spec_hist_cat[:, 0]; nn = rows > 0
            ent = self.spec_hist_asof.entity[np.argsort(self.spec_hist_asof.order)]
            self._sp_asof_idx = AsOfIndex(ent[nn], self.spec_hist_day[nn]); self._sp_asof_val = rows[nn]
        spec_idx = np.asarray(spec_idx, np.int64)
        day_q = np.broadcast_to(np.asarray(day, np.int64), spec_idx.shape) + 1     # < day+1  <=>  <= day
        r = self._sp_asof_idx.lookup(spec_idx, day_q)
        return np.where(r >= 0, self._sp_asof_val[np.clip(r, 0, None)], 0)

    def pair_features(self, enc_idx: np.ndarray, spec_idx: np.ndarray, day: Optional[np.ndarray] = None) -> np.ndarray:
        """Pair-level features for (query encounter, candidate specialist) at the
        encounter's own day (strictly earlier history)."""
        enc_idx = np.asarray(enc_idx, np.int64); spec_idx = np.asarray(spec_idx, np.int64)
        d = self.enc_day[enc_idx] if day is None else np.asarray(day, np.int64)
        pcp = self.enc_pcp[enc_idx]
        rows = self.pair_asof.lookup(pcp * self.n_specialist + spec_idx, d)
        ok = rows >= 0
        rc = np.clip(rows, 0, None)
        prior = np.where(ok, self.pair_cum[rc], 0)
        recency = np.where(ok, d - self.pair_day[rc], 0)
        first_age = np.where(ok, d - self.pair_first[rc], 0)
        dist = self.distance[pcp, spec_idx]
        dist_s, dist_miss = self.distance_scaler.transform(dist)
        ps_rows = self.ps_asof.lookup(self.enc_patient[enc_idx] * self.n_specialist + spec_idx, d)
        ps_prior = np.where(ps_rows >= 0, self.ps_cum[np.clip(ps_rows, 0, None)], 0)
        f = np.stack([
            np.log1p(prior),
            np.log1p(recency), (~ok),
            np.log1p(first_age), (~ok),
            dist_s, dist_miss,
            np.log1p(ps_prior),
        ], axis=1).astype(np.float32)
        return f

    def raw_pair_history(self, enc_idx, spec_idx):
        """Unscaled prior count / recency / distance (for baselines and checks)."""
        enc_idx = np.asarray(enc_idx, np.int64); spec_idx = np.asarray(spec_idx, np.int64)
        d = self.enc_day[enc_idx]; pcp = self.enc_pcp[enc_idx]
        rows = self.pair_asof.lookup(pcp * self.n_specialist + spec_idx, d)
        ok = rows >= 0; rc = np.clip(rows, 0, None)
        prior = np.where(ok, self.pair_cum[rc], 0)
        recency = np.where(ok, d - self.pair_day[rc], -1)
        return prior, recency, self.distance[pcp, spec_idx]

    # ----------------------------------------------------------------------
    # Day batches
    # ----------------------------------------------------------------------
    def days_with_queries(self, split: str) -> np.ndarray:
        return np.unique(self.enc_day[self.query_enc[split]])

    def _encounter_static(self) -> np.ndarray:
        """Cutoff-independent part of the encounter numeric input (all blocks but time)."""
        if getattr(self, "_enc_static_cache", None) is None:
            self._enc_static_cache = np.concatenate([
                self.enc_patient_num,                                                        # block patient_context
                np.log1p(self.enc_prior_encounters)[:, None].astype(np.float32),             # block patient_history ...
                np.log1p(np.clip(self.enc_days_since_prev, 0, None))[:, None].astype(np.float32),
                (self.enc_days_since_prev < 0)[:, None].astype(np.float32),
                self.enc_prior_specialties.astype(np.float32),                               # ... end patient_history
                self.enc_dx_empty[:, None].astype(np.float32),                               # block diagnosis_flags
                self.enc_dx_unk_share[:, None],
            ], axis=1)
        return self._enc_static_cache

    def buckets(self, split: str, policy: str = "day", query_subset: Optional[np.ndarray] = None) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Group a split's queries into graph batches.

        Returns a list of (cutoff_day, batch_enc, query_enc):
          - policy "day":  cutoff = the day; batch_enc = all encounters that day.
          - policy "week": cutoff = Monday-aligned week start, clipped to the
            split start; batch_enc = all encounters of that split in
            [cutoff, cutoff+7).  Graph history is < cutoff; pair features stay
            exact per query (< query day).
        query_enc is the subset of batch_enc that is actually evaluated/trained.
        """
        queries = self.query_enc[split] if query_subset is None else np.asarray(query_subset, np.int64)
        split_start = self.split_days[split][0]
        if policy == "day":
            key = self.enc_day
        elif policy == "week":
            key = np.maximum(self.enc_day - ((self.enc_day - 4) % 7), split_start)   # day 0 = Thursday; 4 = Monday
        else:
            raise ValueError(policy)
        split_enc = self.query_enc[split]
        out = []
        for cutoff in np.unique(key[queries]):
            batch_enc = split_enc[key[split_enc] == cutoff]
            out.append((int(cutoff), batch_enc, queries[key[queries] == cutoff]))
        return out

    def day_batch(self, day: int, query_enc: Optional[np.ndarray] = None, device="cpu") -> dict:
        """Graph for queries on `day` (cutoff = day). See graph_batch."""
        day = int(day)
        if query_enc is None:
            query_enc = np.where(self.enc_day == day)[0]
        query_enc = np.asarray(query_enc, np.int64)
        assert (self.enc_day[query_enc] == day).all(), "all batch queries must share the day"
        return self.graph_batch(day, query_enc, device=device)

    def graph_batch(self, cutoff: int, batch_enc: np.ndarray, device="cpu") -> dict:
        """Graph tensors with history strictly before `cutoff`.

        Historical encounters (day < cutoff) carry HAS_ENCOUNTER and REFERRED_TO
        edges in both directions. Batch encounters (`batch_enc`, day >= cutoff)
        are *receive-only*: they get the forward PCP -> encounter edge and
        nothing else, so no batch encounter can influence any other node. A
        query's embedding therefore depends only on history before the cutoff
        and on its own features, regardless of what else is in the batch.
        SEEN_BY edges exist only for patient-PCP pairs first seen before the
        cutoff. Provider features are as-of the cutoff; encounter age is
        relative to the cutoff (0 for batch encounters).
        """
        cutoff = int(cutoff)
        query_enc = np.asarray(batch_enc, np.int64)
        assert (self.enc_day[query_enc] >= cutoff).all(), "batch encounters must not precede the cutoff"
        hist = self.enc_day < cutoff
        in_batch = np.zeros(self.n_encounter, bool); in_batch[query_enc] = True
        visible = hist | in_batch

        snap = self.provider_snapshot(cutoff)
        age = np.where(hist, cutoff - self.enc_day, 0)
        enc_num = np.concatenate([self._encounter_static(), time_encoding(age)], axis=1)   # block time last
        assert enc_num.shape[1] == self.encounter_blocks["time"][1]

        t = lambda a, dt=None: torch.as_tensor(a, dtype=dt, device=device)
        sb_vis = self.seen_by_first_day < cutoff
        day = cutoff
        hist_idx = np.where(hist)[0]
        edges = {   # forward edges: historical + batch (receive-only) for has_encounter
            ("patient", "seen_by", "pcp"): np.stack([self.seen_by_src[sb_vis], self.seen_by_dst[sb_vis]]),
            ("pcp", "has_encounter", "encounter"): np.stack([self.enc_pcp[visible], np.where(visible)[0]]),
            ("encounter", "referred_to", "specialist"): np.stack([hist_idx, self.enc_target[hist]]),
        }
        reverse = {  # reverse edges: historical only, so batch encounters never send messages
            ("patient", "seen_by", "pcp"): edges[("patient", "seen_by", "pcp")],
            ("pcp", "has_encounter", "encounter"): np.stack([self.enc_pcp[hist], hist_idx]),
            ("encounter", "referred_to", "specialist"): edges[("encounter", "referred_to", "specialist")],
        }
        edge_index = {}
        for (s, r, d_), ei in edges.items():
            edge_index[(s, r, d_)] = t(ei, torch.long)
            edge_index[(d_, f"rev_{r}", s)] = t(reverse[(s, r, d_)][::-1].copy(), torch.long)

        return {
            "day": day, "cutoff": cutoff,
            "query_enc": t(query_enc, torch.long),
            "x": {
                "patient_num": t(self.patient_num), "patient_cat": t(self.patient_cat, torch.long),
                "pcp_num": t(snap["pcp_num"]), "pcp_cat": t(snap["pcp_cat"], torch.long),
                "specialist_num": t(snap["specialist_num"]), "specialist_cat": t(snap["specialist_cat"], torch.long),
                "encounter_num": t(enc_num), "encounter_cat": t(self.enc_patient_cat, torch.long),
                "encounter_dx": t(self.enc_dx, torch.long), "encounter_dx_mask": t(self.enc_dx_mask, torch.bool),
            },
            "edge_index": edge_index,
            "available": t(self.available_specialists(cutoff), torch.bool),
            "n_visible_encounters": int(visible.sum()),
        }

    # ----------------------------------------------------------------------
    def dims(self) -> dict:
        return {
            "patient_num": self.patient_num_dim, "patient_cat_sizes": [self.patient_vocabs[c].size for c in PATIENT_CAT],
            "pcp_num": self.pcp_encoder.num_dim, "pcp_cat_sizes": [self.pcp_encoder.vocabs[c].size for c in PCP_CAT],
            "specialist_num": self.spec_encoder.num_dim, "specialist_cat_sizes": [self.spec_encoder.vocabs[c].size for c in SPEC_CAT],
            "encounter_num": self.encounter_blocks["time"][1],
            "encounter_cat_sizes": [self.patient_vocabs[c].size for c in PATIENT_CAT],
            "dx_vocab": self.dx_vocab.size, "pair": self.pair_feature_dim,
            "n_patient": self.n_patient, "n_pcp": self.n_pcp, "n_specialist": self.n_specialist, "n_encounter": self.n_encounter,
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "ModelInputs":
        with open(path, "rb") as f:
            return pickle.load(f)


def default_bundle_path(root: str = ".") -> str:
    base = os.path.join(root, "prepared_data")
    runs = sorted(d for d in os.listdir(base) if d.startswith("referral_prepared_") and os.path.exists(os.path.join(base, d, "COMPLETE.json")))
    if not runs:
        raise FileNotFoundError("no completed prepared bundle found")
    return os.path.join(base, runs[-1])
