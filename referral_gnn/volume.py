"""STEP12a (STEP12_CONGESTION_PLAN.md, frozen 2026-10-12): specialist volume attributes.

The specialist snapshot (`provider_snapshot(...)["specialist_num"]`) holds, for numeric column i of `spec_encoder.numeric_cols`,
the z-scored value at 3i, a missing flag at 3i+1 and an ambiguity flag at 3i+2 (v5 appends MIPS extras at the end).
Specialist attributes are static (one value per specialist; only MIPS is as-of), so every statistic here is computed once from
the last snapshot, over training-active specialists (first seen on/before the last training day).

Volume block (§2.B): beneficiary_count, service_count and the six panel counts (age bands, female/male), which sum to
beneficiary_count. "Plus" adds rating_count and practice_size.

  neutralise(...)  B1 inference diagnostic: share-preserving rescale of the volume block to the specialty-median beneficiary
                   level (panel shares and services per beneficiary kept); B1+ also sets rating_count / practice_size to their
                   specialty medians (flags cleared).
  stats(...)       statistics for the retrained representations used by the model and the tabular features:
                   "log1p" (H5): volume block as standardised log1p(raw);
                   "shares" (B2): beneficiary/service counts removed (constant 0, flags 0), panel counts as standardised shares;
                   "strict" (Amendment A1, H3′): every specialist column carrying within-specialty volume information set to
                   constant 0 (counts, panel block, average panel age/risk, practice size, rating count, MIPS and its extras);
                   specialty, credentials, years in Medicare and rating kept.
  transform_np(...) the "log1p"/"shares" transform on a numpy snapshot matrix (tabular features); the model applies the same
                   arithmetic in torch from the same statistics (three_node_model.ThreeNodeGNN._volume).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

VOL = ["beneficiary_count", "service_count", "panel_age_lt65", "panel_age_65_74", "panel_age_75_84", "panel_age_gt84", "panel_female", "panel_male"]
PANEL = VOL[2:]
PLUS = ["rating_count", "practice_size"]
# STEP12a Amendment A1 (H3′): "strict" also removes every specialist column that carries within-specialty volume information
# (greedy CV-R² elimination, 2026-10-13): within-specialty R² of log beneficiaries from what remains ≈ −0.05.
STRICT = VOL + ["average_panel_age", "average_panel_risk", "practice_size", "rating_count", "mips_score"]


def col(mi, name):
    i = list(mi.spec_encoder.numeric_cols).index(name)
    return 3 * i, 3 * i + 1, 3 * i + 2


def _scaler(mi, name):
    s = mi.spec_encoder.scalers[name]; return float(s.mean), float(s.std)


def _last_snapshot(mi):
    return mi.provider_snapshot(int(np.max(mi.enc_day)) + 1)


def _raw(mi, X, name):
    v, m, _ = col(mi, name); mu, sd = _scaler(mi, name)
    return X[:, v].astype(np.float64) * sd + mu, X[:, m] > 0.5


def stats(mi) -> Dict:
    """All statistics, from the last snapshot and training-active specialists. Deterministic given the inputs."""
    snap = _last_snapshot(mi); X = snap["specialist_num"]; spec = snap["specialist_cat"][:, 0].astype(np.int64)
    active = np.asarray(mi.spec_first_seen) <= int(mi.train_end_day)
    out = {"cols": {n: list(col(mi, n)) for n in VOL + PLUS}, "scaler": {n: list(_scaler(mi, n)) for n in VOL + PLUS},
           "cols_all": {n: list(col(mi, n)) for n in STRICT}, "extras_start": int(mi.spec_encoder.num_dim),
           "log1p": {}, "share": {}, "spec_median": {}, "global_median": {}}
    bene, bmiss = _raw(mi, X, "beneficiary_count")
    for n in VOL:
        r, miss = _raw(mi, X, n); ok = active & ~miss
        lg = np.log1p(np.clip(r[ok], 0, None)); out["log1p"][n] = [float(lg.mean()), float(lg.std() or 1.0)]
    for n in PANEL:
        r, miss = _raw(mi, X, n); ok = active & ~miss & ~bmiss & (bene > 0)
        sh = r[ok] / bene[ok]; out["share"][n] = [float(sh.mean()), float(sh.std() or 1.0)]
    for n in ["beneficiary_count"] + PLUS:                  # specialty medians for B1 / B1+
        r, miss = _raw(mi, X, n); ok = active & ~miss
        med = {int(s): float(np.median(r[ok & (spec == s)])) for s in np.unique(spec[ok]) if s > 0}
        out["spec_median"][n] = med; out["global_median"][n] = float(np.median(r[ok]))
    out["specialty"] = spec.tolist()                         # specialty index per specialist (last snapshot; 0 = unknown)
    return out


def neutralise(mi, X: np.ndarray, spec_idx: np.ndarray, st: Dict, plus: bool = False) -> np.ndarray:
    """B1 / B1+ on snapshot rows X (rows = specialists spec_idx). Returns a modified copy."""
    X = np.array(X, dtype=np.float32, copy=True); spec_idx = np.asarray(spec_idx, np.int64)
    sp = np.asarray(st["specialty"], np.int64)[spec_idx]
    def med(n):
        g = st["global_median"][n]; d = st["spec_median"][n]
        return np.array([d.get(int(s), g) for s in sp], np.float64)
    bene, bmiss = _raw(mi, X, "beneficiary_count")
    target = med("beneficiary_count")
    factor = np.where(~bmiss & (bene > 0), target / np.maximum(bene, 1e-9), np.nan)
    for n in VOL:
        v, m, a = st["cols"][n]; mu, sd = st["scaler"][n]
        r = X[:, v].astype(np.float64) * sd + mu
        new = np.where(np.isnan(factor), np.nan, r * np.nan_to_num(factor))   # rows without a usable beneficiary count stay unchanged
        keep = ~np.isnan(new)
        X[keep, v] = ((new[keep] - mu) / sd).astype(np.float32)
    if plus:
        for n in PLUS:
            v, m, a = st["cols"][n]; mu, sd = st["scaler"][n]
            X[:, v] = ((med(n) - mu) / sd).astype(np.float32); X[:, m] = 0.0; X[:, a] = 0.0
    return X


def transform_np(X: np.ndarray, st: Dict, mode: str) -> np.ndarray:
    """"log1p" (H5) or "shares" (B2) representation of the volume block on snapshot rows X. Returns a modified copy."""
    assert mode in ("log1p", "shares", "strict")
    X = np.array(X, dtype=np.float32, copy=True)
    def raw(n):
        v, m, _ = st["cols"][n]; mu, sd = st["scaler"][n]; return X[:, v].astype(np.float64) * sd + mu, X[:, m] > 0.5
    if mode == "strict":                                   # constant 0 (value and flags) for STRICT columns and the MIPS extras (v5 layout)
        assert X.shape[1] == st["extras_start"] + 2, "strict mode is defined for the v5 specialist snapshot (num_dim + 2 MIPS extras)"
        for n in STRICT:
            for j in st["cols_all"][n]: X[:, j] = 0.0
        X[:, st["extras_start"]:] = 0.0
        return X
    if mode == "log1p":
        for n in VOL:
            v, m, _ = st["cols"][n]; r, miss = raw(n); lm, ls = st["log1p"][n]
            X[:, v] = np.where(miss, 0.0, (np.log1p(np.clip(r, 0, None)) - lm) / ls).astype(np.float32)
        return X
    bene, bmiss = raw("beneficiary_count")
    for n in PANEL:
        v, m, _ = st["cols"][n]; r, miss = raw(n); sm, ss = st["share"][n]
        bad = miss | bmiss | (bene <= 0)
        X[:, v] = np.where(bad, 0.0, (r / np.where(bene > 0, bene, 1.0) - sm) / ss).astype(np.float32); X[:, m] = np.where(bad, 1.0, X[:, m])
    for n in ("beneficiary_count", "service_count"):
        v, m, a = st["cols"][n]; X[:, v] = 0.0; X[:, m] = 0.0; X[:, a] = 0.0
    return X
