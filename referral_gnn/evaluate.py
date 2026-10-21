"""Full-pool ranking evaluation shared by the GNN, the PairMLP and baselines.

For each query the candidate pool is every specialist first observed strictly
before the query day. The target's rank is computed pessimistically (ties count
against the target); an optimistic rank is stored too. Queries whose target is
not in the pool have rank NaN and count as misses in all-query metrics.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

from .data import ModelInputs

KS = (1, 5, 10, 20)


class Scorer:
    """Interface: prepare(day, query_enc) once per forward context, then score(q, cand)."""

    def prepare(self, day: int, query_enc: np.ndarray):
        pass

    def score(self, q: int, cand: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ModelScorer(Scorer):
    def __init__(self, model, mi: ModelInputs, device: str, chunk: int = 200_000):
        self.model, self.mi, self.device, self.chunk = model, mi, device, chunk
        self.h = None

    @torch.no_grad()
    def prepare(self, cutoff: int, batch_enc: np.ndarray):
        self.model.eval()
        batch = self.mi.graph_batch(cutoff, batch_enc, device=self.device)
        self.h = self.model.encode(batch)

    @torch.no_grad()
    def score(self, q: int, cand: np.ndarray) -> np.ndarray:
        mi = self.mi
        pf = torch.as_tensor(mi.pair_features(np.full(len(cand), q), cand), device=self.device)
        out = []
        for s in range(0, len(cand), self.chunk):
            c = torch.as_tensor(cand[s:s + self.chunk], device=self.device)
            e = torch.full_like(c, int(q))
            p = torch.full_like(c, int(mi.enc_patient[q])); pc = torch.full_like(c, int(mi.enc_pcp[q]))
            out.append(self.model.score(self.h, e, c, pf[s:s + self.chunk], p, pc).float().cpu().numpy())
        return np.concatenate(out)


def _ranks(scores: np.ndarray, target_pos: int):
    st = scores[target_pos]
    greater = int((scores > st).sum())
    equal = int((scores == st).sum()) - 1
    return greater + equal + 1, greater + 1          # pessimistic, optimistic


def evaluate(scorer: Scorer, mi: ModelInputs, split: str, query_subset: Optional[np.ndarray] = None,
             bucket: str = "day", isolate: Optional[str] = None, log_every: int = 0,
             pool: str = "all", radius_miles: Optional[float] = None) -> pd.DataFrame:
    """Per-query results as a DataFrame (no identifiers, only indices/flags/ranks).

    `bucket` sets the graph cutoff policy ("day" or "week"); candidate pools and
    pair features always use the exact query day.

    `pool`: "all" = every specialist first observed before the query day; "unseen" = additionally exclude
    specialists this patient has already seen (history-only, no label use); "specialty_unseen" = unseen pool
    restricted to the chosen specialist's specialty (ORACLE specialty; diagnostic / "specialty given" setting). `radius_miles`: additionally keep
    only candidates within this patient-origin distance OR with missing distance (requires v2 inputs).
    Targets outside the pool are counted as unavailable (misses in all-query metrics).

    `isolate` is a verification mode that keeps the SAME cutoff as `bucket` but
    forwards smaller batches: "patient" = one forward per patient within the
    bucket (only that patient's encounters in the batch), "query" = one forward
    per query alone. Because batch encounters are receive-only, both must
    reproduce the ranks of the full-batch evaluation at the same cutoff exactly.
    """
    queries = np.sort(mi.query_enc[split] if query_subset is None else np.asarray(query_subset))
    train_targets = set(mi.enc_target[mi.query_enc["train"]].tolist())
    train_patients = set(mi.enc_patient[mi.query_enc["train"]].tolist())
    train_pcps = set(mi.enc_pcp[mi.query_enc["train"]].tolist())
    rows = []
    t0 = time.time()
    groups = []                                                   # (cutoff, batch_enc, queries) — cutoff identical across modes
    for cutoff, batch_enc, q_b in mi.buckets(split, bucket, query_subset=queries):
        if isolate == "patient":
            for p in np.unique(mi.enc_patient[q_b]):
                grp = q_b[mi.enc_patient[q_b] == p]
                groups.append((cutoff, batch_enc[mi.enc_patient[batch_enc] == p], grp))
        elif isolate == "query":
            for q in q_b:
                groups.append((cutoff, np.array([q]), np.array([q])))
        elif isolate is None:
            groups.append((cutoff, batch_enc, q_b))
        else:
            raise ValueError(isolate)
    for gi, (cutoff, batch_enc, grp) in enumerate(groups):
        scorer.prepare(cutoff, batch_enc)
        for q in grp:
            q = int(q); t = int(mi.enc_target[q]); day = int(mi.enc_day[q])
            avail = np.where(mi.available_specialists(day))[0]
            if pool in ("unseen", "specialty_unseen"):
                seen = mi.ps_asof.lookup(mi.enc_patient[q] * mi.n_specialist + avail, day) >= 0
                avail = avail[~seen]
            if pool == "specialty_unseen":                     # oracle: the chosen specialist's specialty as of the query day
                sp_t = mi.specialty_asof([t], day)[0]
                if sp_t > 0:
                    avail = avail[mi.specialty_asof(avail, day) == sp_t]
            if radius_miles is not None:
                dpat = mi.patient_specialist_distance(np.full(len(avail), mi.enc_patient[q]), avail)
                avail = avail[np.isnan(dpat) | (dpat <= radius_miles)]
            pos_of = {int(s): i for i, s in enumerate(avail)}
            if True:
                prior, recency, dist = mi.raw_pair_history(np.array([q]), np.array([t]))
                repeat = bool(mi.ps_asof.lookup(np.array([mi.enc_patient[q] * mi.n_specialist + t]), day)[0] >= 0)
                rec = {
                    "enc": q, "split": split, "day": day, "n_candidates": len(avail),
                    "target_available": t in pos_of,
                    "familiar_pair": bool(prior[0] > 0),
                    "patient_repeat": repeat,                 # patient already saw this specialist strictly earlier
                    "target_seen_in_train": t in train_targets,
                    "patient_seen_in_train": int(mi.enc_patient[q]) in train_patients,
                    "pcp_seen_in_train": int(mi.enc_pcp[q]) in train_pcps,
                    "patient_has_history": bool(mi.enc_prior_encounters[q] > 0),
                    "rank": np.nan, "rank_optimistic": np.nan,
                }
                if rec["target_available"]:
                    s = scorer.score(q, avail)
                    rec["rank"], rec["rank_optimistic"] = _ranks(s, pos_of[t])
                rows.append(rec)
        if log_every and (gi + 1) % log_every == 0:
            print(f"  eval {split}: {gi+1}/{len(groups)} batches, {len(rows)} queries, {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def metrics(df: pd.DataFrame, rank_col: str = "rank") -> Dict[str, float]:
    """Recall@K and MRR over eligible queries and over all queries (unavailable = miss)."""
    r = df[rank_col].to_numpy(dtype=float)
    avail = np.isfinite(r)
    out = {"n_queries": int(len(df)), "n_eligible": int(avail.sum()), "target_availability": float(avail.mean()) if len(df) else np.nan}
    for k in KS:
        hit = (r <= k) & avail
        out[f"recall@{k}_eligible"] = float(hit[avail].mean()) if avail.any() else np.nan
        out[f"recall@{k}_all"] = float(hit.mean()) if len(df) else np.nan
    rr = np.where(avail, 1.0 / np.where(avail, r, 1.0), 0.0)
    out["mrr_eligible"] = float(rr[avail].mean()) if avail.any() else np.nan
    out["mrr_all"] = float(rr.mean()) if len(df) else np.nan
    out["median_rank_eligible"] = float(np.median(r[avail])) if avail.any() else np.nan
    return out


def breakdowns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("patient_repeat", "familiar_pair", "target_seen_in_train", "patient_seen_in_train", "patient_has_history"):
        for val in (True, False):
            sub = df[df[col] == val]
            if len(sub):
                m = metrics(sub)
                rows.append({"group": col, "value": val, "n": len(sub), "recall@1": m["recall@1_eligible"], "recall@5": m["recall@5_eligible"],
                             "recall@10": m["recall@10_eligible"], "recall@20": m["recall@20_eligible"], "mrr": m["mrr_eligible"], "availability": m["target_availability"]})
    return pd.DataFrame(rows)
