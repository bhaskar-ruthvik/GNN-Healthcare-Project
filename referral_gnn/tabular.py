"""Step 11.1 non-GNN baselines (STEP11_PLAN, frozen 2026-10-07).

Every scorer here follows the evaluate.Scorer interface (prepare(cutoff, batch_enc) once per weekly bucket, then
score(q, cand)), so the existing evaluators rank exactly the same candidate sets with exactly the same rules.

Information horizon (identical to the GNN):
  - node-level inputs (specialist/PCP snapshots, encounter context, predecessor gap) as of the weekly bucket cutoff;
  - pair features exact per query (strictly before the query day), via mi.pair_features — the same call the GNN uses;
  - hand-made graph features (GraphStats) from encounters with enc_day < cutoff only.

Feature sets
  primary : encounter static block (patient context, severity, patient history, diagnosis flags) + predecessor gap
            + encounter categoricals + disease-category multi-hot + specialist snapshot + PCP snapshot + 14 pair features.
            The GNN's time encoding is constant (0) for query encounters and is omitted.
  graph   : primary + as-of relational features (GraphStats.features).
  tabular : (Amendment B1, user 2026-10-07) PURE TABULAR, nothing derived from the referral network. Each claim row's own
            attributes only: encounter patient context (age and its flags), severity, diagnosis flags, encounter
            categoricals (gender, race, insurer, ACO status), disease-category multi-hot, specialist and PCP snapshots
            (provider attributes incl. MIPS, tenure, volume, ratings, panel mix), PCP–specialist and patient–specialist
            distance, and shared hospital affiliation (a comparison of two providers' listed affiliations, not history).
            Excluded: the patient-history block, predecessor gap, PCP–specialist referral history, patient–specialist
            prior, insurer history.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Dict, List, Optional  # noqa: F401

import numpy as np

from .evaluate import Scorer


# ------------------------------------------------------------------------------------------------------------------
class GraphStats:
    """Incremental as-of relational statistics. advance(cutoff) adds every encounter with enc_day < cutoff;
    cutoffs must be non-decreasing (call reset() to go back)."""

    WINDOW = 365

    def __init__(self, mi):
        self.mi = mi
        self.order = np.argsort(mi.enc_day, kind="stable")
        self.days = mi.enc_day[self.order]
        self.n_cat = int(mi.enc_dx.max()) + 1
        self.reset()

    def reset(self):
        mi = self.mi
        self.ptr = 0; self.cutoff = -10 ** 9
        self.B = np.zeros((mi.n_pcp, mi.n_specialist), np.float32)          # cumulative PCP -> specialist referrals
        self.W = np.zeros((mi.n_pcp, mi.n_specialist), np.float32)          # same, last WINDOW days
        self.win = deque()
        self.C = np.zeros((self.n_cat, mi.n_specialist), np.float32)        # cumulative referrals by disease category
        self.P: Dict[int, set] = {}                                           # patient -> specialists referred to so far
        self.co: Dict[int, Counter] = {}                                      # specialist -> Counter(specialist -> distinct patients)
        self._dirty = True

    def advance(self, cutoff: int):
        cutoff = int(cutoff)
        if cutoff < self.cutoff:
            self.reset()
        mi = self.mi
        end = int(np.searchsorted(self.days, cutoff, "left"))
        for e in self.order[self.ptr:end]:
            e = int(e); s = int(mi.enc_target[e]); p = int(mi.enc_pcp[e]); pat = int(mi.enc_patient[e]); d = int(mi.enc_day[e])
            self.B[p, s] += 1; self.W[p, s] += 1; self.win.append((d, p, s))
            cats = mi.enc_dx[e][mi.enc_dx_mask[e]]
            for c in np.unique(cats[cats > 1]): self.C[int(c), s] += 1
            seen = self.P.setdefault(pat, set())
            if s not in seen:
                cs = self.co.setdefault(s, Counter())
                for o in seen:
                    cs[o] += 1; self.co.setdefault(o, Counter())[s] += 1
                seen.add(s)
        self.ptr = end
        while self.win and self.win[0][0] < cutoff - self.WINDOW:
            _, p, s = self.win.popleft(); self.W[p, s] -= 1
        self.cutoff = cutoff; self._dirty = True

    def _refresh(self):
        if not self._dirty: return
        Bb = (self.B > 0).astype(np.float32)
        self.spec_total = self.B.sum(0); self.spec_win_refs = self.W.sum(0); self.spec_win_pcps = (self.W > 0).sum(0).astype(np.float32)
        self.pcp_distinct = Bb.sum(1); self.pcp_total = self.B.sum(1)
        share = (Bb @ Bb.T) > 0                                               # PCPs sharing >= 1 referral partner
        np.fill_diagonal(share, False)
        self.partners = share
        self.Bb = Bb
        self._dirty = False

    N_FEATURES = 10
    NAMES = ["g_spec_total_log", "g_spec_refs365_log", "g_spec_pcps365_log", "g_cat_refs_log", "g_cat_share",
             "g_coref_max_log", "g_coref_sum_log", "g_twohop_share", "g_pcp_distinct_log", "g_pcp_total_log"]

    def features(self, q: int, cand: np.ndarray) -> np.ndarray:
        self._refresh(); mi = self.mi; cand = np.asarray(cand, np.int64)
        out = np.zeros((len(cand), self.N_FEATURES), np.float32)
        tot = self.spec_total[cand]
        out[:, 0] = np.log1p(tot); out[:, 1] = np.log1p(self.spec_win_refs[cand]); out[:, 2] = np.log1p(self.spec_win_pcps[cand])
        cats = mi.enc_dx[q][mi.enc_dx_mask[q]]; cats = np.unique(cats[cats > 1])
        cr = self.C[cats][:, cand].sum(0) if len(cats) else np.zeros(len(cand), np.float32)
        out[:, 3] = np.log1p(cr); out[:, 4] = cr / (tot + 1.0)
        prior = self.P.get(int(mi.enc_patient[q]), ())
        if prior:
            pos = {int(s): i for i, s in enumerate(cand)}
            mx = np.zeros(len(cand), np.float32); sm = np.zeros(len(cand), np.float32)
            for o in prior:
                for s, n in self.co.get(o, {}).items():
                    i = pos.get(s)
                    if i is not None: sm[i] += n; mx[i] = max(mx[i], n)
            out[:, 5] = np.log1p(mx); out[:, 6] = np.log1p(sm)
        p = int(mi.enc_pcp[q]); partners = self.partners[p]
        out[:, 7] = self.Bb[partners][:, cand].mean(0) if partners.any() else 0.0
        out[:, 8] = np.log1p(self.pcp_distinct[p]); out[:, 9] = np.log1p(self.pcp_total[p])
        return out


# ------------------------------------------------------------------------------------------------------------------
class FlatGraph:
    """STEP12b H4 "flattened graph" (SIGN / SeHGNN-style precomputed mean aggregation over metapaths), as-of features.

    advance(cutoff, snap) adds every encounter with enc_day < cutoff (the query's own referral is never included: query days are
    ≥ the weekly cutoff). Aggregates are MEANS of node attributes along metapaths (not counts):
      A1  Spec ← Enc        mean encounter profile of the specialist's past referrals: patient context (6), severity (1),
                            diagnosis flags (2), disease-category distribution (n_cat)
      A2  Spec ← Enc ← PCP  mean attribute vector of the PCPs that referred to the specialist (pcp_num)
      A3  PCP → Enc → Spec  mean attribute vector (specialist numeric values) of the query PCP's past specialists
    Pair features: A1 and A2 of the candidate; A3 of the query PCP and candidate − A3 (how the candidate differs from the PCP's
    typical specialist); cosine(query disease categories, candidate A1 categories); query age − A1 mean age; flags for "no
    history" (specialist / PCP). Degree counts are deliberately NOT included (that was the "graph counts" set).
    A2/A3 are computed from referral counts with the provider snapshot of the CURRENT cutoff (reviewer 2026-10-14: no train/serve
    skew in the as-of MIPS columns). Note: with 119 PCPs, A3 nearly identifies the PCP and A2 can equal the query PCP's own vector,
    so this set reintroduces some PCP–specialist history indirectly (fine for a hybrid; it is not pure tabular)."""

    def __init__(self, mi):
        self.mi = mi
        self.order = np.argsort(mi.enc_day, kind="stable"); self.days = mi.enc_day[self.order]
        self.n_cat = int(mi.enc_dx.max()) + 1
        st = mi._encounter_static(); self.u_static = np.concatenate([st[:, :7], st[:, 74:76]], axis=1).astype(np.float32)   # context, severity, dx flags
        self.spec_vals = [3 * i for i in range(len(mi.spec_encoder.numeric_cols))]                                          # value columns only
        self.reset()

    def reset(self):
        mi = self.mi; du = self.u_static.shape[1] + self.n_cat
        self.ptr = 0; self.cutoff = -10 ** 9
        self.SE = np.zeros((mi.n_specialist, du), np.float64); self.nS = np.zeros(mi.n_specialist, np.float64)
        self.Bc = np.zeros((mi.n_pcp, mi.n_specialist), np.float64); self.nP = np.zeros(mi.n_pcp, np.float64)   # PCP→specialist referral counts

    def _u(self, e):
        mi = self.mi; d = np.zeros(self.n_cat, np.float32); c = mi.enc_dx[e][mi.enc_dx_mask[e]]; c = c[c > 1]
        if len(c): d[np.unique(c)] = 1.0 / len(np.unique(c))
        return np.concatenate([self.u_static[e], d])

    def advance(self, cutoff: int, snap):
        cutoff = int(cutoff)
        if cutoff < self.cutoff: self.reset()
        mi = self.mi
        end = int(np.searchsorted(self.days, cutoff, "left"))
        for e in self.order[self.ptr:end]:
            e = int(e); s = int(mi.enc_target[e]); p = int(mi.enc_pcp[e])
            self.SE[s] += self._u(e); self.nS[s] += 1; self.Bc[p, s] += 1; self.nP[p] += 1
        self.ptr = end; self.cutoff = cutoff

    def names(self, n_pcp_num):
        a1 = [f"f_A1_ctx{i}" for i in range(7)] + ["f_A1_dxempty", "f_A1_dxunk"] + [f"f_A1_cat{c}" for c in range(self.n_cat)]
        return a1 + [f"f_A2_pcp{i}" for i in range(n_pcp_num)] + [f"f_A3_spec{i}" for i in range(len(self.spec_vals))] + \
               [f"f_A3diff_spec{i}" for i in range(len(self.spec_vals))] + ["f_cat_cosine", "f_age_diff", "f_spec_nohist", "f_pcp_nohist"]

    def features(self, q: int, cand: np.ndarray, snap) -> np.ndarray:
        mi = self.mi; cand = np.asarray(cand, np.int64); n = len(cand)
        ns = self.nS[cand]; has = ns > 0; den = np.maximum(ns, 1.0)[:, None]
        A1 = self.SE[cand] / den
        pv = snap["pcp_num"].astype(np.float64); sv = snap["specialist_num"][:, self.spec_vals].astype(np.float64)
        A2 = (self.Bc[:, cand].T @ pv) / den                                   # node attributes from the CURRENT snapshot (no skew)
        p = int(mi.enc_pcp[q]); A3 = (self.Bc[p] @ sv) / max(self.nP[p], 1.0)
        diff = snap["specialist_num"][cand][:, self.spec_vals] - A3[None, :]
        uq = self._u(q); qc = uq[9:]; ac = A1[:, 9:]
        cos = (ac @ qc) / (np.linalg.norm(ac, axis=1) * (np.linalg.norm(qc) or 1.0) + 1e-9)
        age = self.u_static[q, 0] - A1[:, 0]
        out = np.concatenate([A1, A2, np.broadcast_to(A3, (n, len(A3))), diff, cos[:, None], age[:, None], (~has)[:, None].astype(np.float64),
                              np.full((n, 1), float(self.nP[p] == 0))], axis=1).astype(np.float32)
        out[~has, :A1.shape[1] + A2.shape[1]] = 0.0; out[~has, -4:-2] = 0.0              # undefined means → 0 (flagged)
        return out


# ------------------------------------------------------------------------------------------------------------------
class FeatureBuilder:
    """Rows for (query encounter, candidate specialist) pairs; call prepare(cutoff, batch_enc) per bucket first."""

    def __init__(self, mi, feature_set: str = "primary", volume: str = "keep", no_pcp: bool = False, ablate: str = None, extra_npz: str = None):
        """STEP12a options (defaults = unchanged behaviour): volume "shares" (B2: beneficiary/service counts constant, panel counts
        as shares; referral_gnn/volume.py); no_pcp drops the PCP snapshot columns (H1 PCP-attribute check); ablate "no_volume" /
        "no_volume_plus" applies the B1 / B1+ inference neutralisation to the specialist snapshot (scoring only)."""
        assert feature_set in ("primary", "graph", "tabular", "flat") and volume in ("keep", "shares", "strict") and ablate in (None, "no_volume", "no_volume_plus")
        self.mi, self.feature_set = mi, feature_set
        self.volume, self.no_pcp, self.ablate = volume, bool(no_pcp), ablate
        self.vstats = None
        self.extra_npz, self.extra_lookup = extra_npz, None     # STEP12d B (stacking): one extra column "gnn_score" from an .npz
        if extra_npz:
            from .score_lookup import ScoreLookup
            z_ = np.load(extra_npz)
            self.extra_train = z_["train_score"]                                   # out-of-fold, aligned with mi.tc_enc / tc_spec rows
            self.extra_lookup = ScoreLookup(z_["look_enc"], z_["look_spec"], z_["look_score"])   # selection pool + evaluation eligible sets
        if volume != "keep" or ablate:
            from .volume import stats
            self.vstats = stats(mi)
        self.static = mi._encounter_static()                                  # (n_enc, 76): context, severity, history, dx flags
        self.n_cat = int(mi.enc_dx.max()) + 1
        self.stats = GraphStats(mi) if feature_set == "graph" else None
        self.flat = FlatGraph(mi) if feature_set == "flat" else None          # STEP12b H4: pure tabular + flattened-graph means
        d = mi.dims()
        self.enc_cat_sizes, self.spec_cat_sizes, self.pcp_cat_sizes = d["encounter_cat_sizes"], d["specialist_cat_sizes"], d["pcp_cat_sizes"]
        self.names: List[str] = []
        self.names += [f"enc_static_{i}" for i in range(self.static.shape[1])] + ["gap_log", "gap_none"]
        self.names += [f"enc_cat_{i}" for i in range(len(self.enc_cat_sizes))]
        self.names += [f"dx_{c}" for c in range(self.n_cat)]
        snap = mi.provider_snapshot(int(mi.split_days["train"][0]))
        self.names += [f"spec_num_{i}" for i in range(snap["specialist_num"].shape[1])] + [f"spec_cat_{i}" for i in range(snap["specialist_cat"].shape[1])]
        self.names += [f"pcp_num_{i}" for i in range(snap["pcp_num"].shape[1])] + [f"pcp_cat_{i}" for i in range(snap["pcp_cat"].shape[1])]
        self.names += [f"pair_{n}" for n in mi.pair_feature_names]
        if self.stats is not None: self.names += GraphStats.NAMES
        if self.flat is not None: self.names += self.flat.names(snap["pcp_num"].shape[1])
        if self.extra_npz: self.names += ["gnn_score"]
        self.keep = None
        drops = []
        if feature_set in ("tabular", "flat"):
            hist = set(range(*mi.encounter_blocks["patient_history"]))
            pair_keep = {"pair_distance_scaled", "pair_distance_missing", "pair_patient_distance_scaled", "pair_patient_distance_missing",
                         "pair_shared_affiliation", "pair_affiliation_unknown"}
            drops.append(lambda nm: ((nm.startswith("enc_static_") and int(nm.rsplit("_", 1)[1]) in hist) or nm in ("gap_log", "gap_none")
                                     or (nm.startswith("pair_") and nm not in pair_keep)))
        if self.no_pcp:
            drops.append(lambda nm: nm.startswith(("pcp_num_", "pcp_cat_")))
        if drops:
            self.keep = np.array([not any(d(nm) for d in drops) for nm in self.names])
            if feature_set in ("tabular", "flat"): assert sum(nm.startswith("pair_") for nm in np.array(self.names)[self.keep]) == len(pair_keep)
            self.names = [nm for nm, k in zip(self.names, self.keep) if k]
        self.cat_idx = []
        for nm_i, nm in enumerate(self.names):
            if nm.startswith(("enc_cat_", "spec_cat_", "pcp_cat_")): self.cat_idx.append(nm_i)
        self.cutoff = None

    def prepare(self, cutoff: int, batch_enc: np.ndarray):
        mi = self.mi; self.cutoff = int(cutoff)
        self.snap = mi.provider_snapshot(self.cutoff)
        if self.vstats is not None:                       # STEP12a: B1 neutralisation (scoring) and/or B2 representation
            from .volume import neutralise, transform_np
            X = self.snap["specialist_num"]
            if self.ablate: X = neutralise(mi, X, np.arange(len(X)), self.vstats, plus=(self.ablate == "no_volume_plus"))
            if self.volume != "keep": X = transform_np(X, self.vstats, self.volume)
            self.snap = dict(self.snap, specialist_num=X)
        _, gaps = mi.temporal_context(self.cutoff, np.unique(np.asarray(batch_enc, np.int64)))
        self.gaps = gaps
        if self.stats is not None: self.stats.advance(self.cutoff)
        if self.flat is not None: self.flat.advance(self.cutoff, self.snap)

    def rows(self, q: int, cand: np.ndarray, pair: Optional[np.ndarray] = None, extra: Optional[np.ndarray] = None) -> np.ndarray:
        mi = self.mi; cand = np.asarray(cand, np.int64); n = len(cand); q = int(q)
        g = int(self.gaps[q])
        enc = np.concatenate([self.static[q], [np.log1p(max(g, 0)), float(g < 0)], mi.enc_patient_cat[q].astype(np.float32)])
        dx = np.zeros(self.n_cat, np.float32); cats = mi.enc_dx[q][mi.enc_dx_mask[q]]; dx[cats] = 1.0
        p = int(mi.enc_pcp[q])
        pcp = np.concatenate([self.snap["pcp_num"][p], self.snap["pcp_cat"][p].astype(np.float32)])
        spec = np.concatenate([self.snap["specialist_num"][cand], self.snap["specialist_cat"][cand].astype(np.float32)], axis=1)
        if pair is None: pair = mi.pair_features(np.full(n, q), cand)
        parts = [np.broadcast_to(enc, (n, len(enc))), np.broadcast_to(dx, (n, len(dx))), spec, np.broadcast_to(pcp, (n, len(pcp))), pair]
        if self.stats is not None: parts.append(self.stats.features(q, cand))
        if self.flat is not None: parts.append(self.flat.features(q, cand, self.snap))
        if self.extra_npz:
            parts.append(np.asarray(extra if extra is not None else self.extra_lookup.get(q, cand), np.float32)[:, None])
        X = np.concatenate(parts, axis=1).astype(np.float32)
        if self.keep is not None: X = X[:, self.keep]
        assert X.shape[1] == len(self.names)
        return X


def training_matrix(mi, feature_set: str, log=print, end_day=None, **fb_opts):
    """The GNN's exact training rows: tc_enc/tc_spec, 21 per query (target first), weekly buckets in time order.
    end_day (STEP11 preliminary data-size runs): only queries strictly before that day. fb_opts: STEP12a FeatureBuilder options."""
    fb = FeatureBuilder(mi, feature_set, **fb_opts)
    tq = mi.train_query_enc if end_day is None else mi.train_query_enc[mi.enc_day[mi.train_query_enc] < end_day]
    buckets = mi.buckets("train", "week", query_subset=tq)
    rows_of = lambda encs: (np.searchsorted(mi.tc_enc[::21], encs) * 21)
    pair_all = mi.pair_features(mi.tc_enc, mi.tc_spec)                       # same precomputation as the trainer
    n_rows = 21 * len(tq)
    X = np.empty((n_rows, len(fb.names)), np.float32); y = np.empty(n_rows, np.int8); qid = np.empty(n_rows, np.int64)
    k = 0
    for bi, (cutoff, batch_enc, query_enc) in enumerate(buckets):
        fb.prepare(cutoff, batch_enc)
        for q in query_enc:
            r = rows_of(np.array([q]))[0]; sl = slice(r, r + 21)
            assert mi.tc_enc[r] == q and mi.tc_label[r] == 1
            X[k:k + 21] = fb.rows(q, mi.tc_spec[sl], pair_all[sl], fb.extra_train[sl] if fb.extra_npz else None); y[k:k + 21] = mi.tc_label[sl]; qid[k:k + 21] = q; k += 21
        if (bi + 1) % 25 == 0: log(f"  training rows: bucket {bi + 1}/{len(buckets)}, {k:,} rows")
    assert k == len(X)
    return X, y, qid, fb


# ------------------------------------------------------------------------------------------------------------------
class TabularScorer(Scorer):
    """Wraps a fitted model with predict(X) -> score (LightGBM Booster, or the logistic pipeline below)."""

    def __init__(self, mi, model, feature_set: str, **fb_opts):
        self.mi, self.model = mi, model; self.fb = FeatureBuilder(mi, feature_set, **fb_opts)

    def prepare(self, cutoff, batch_enc):
        self.fb.prepare(cutoff, batch_enc)

    def score(self, q, cand):
        return np.asarray(self.model.predict(self.fb.rows(q, cand)), float)


class LogisticModel:
    """Pointwise logistic regression on the primary features: standardised numerics, one-hot categoricals."""

    def __init__(self, cat_idx, C=1.0, seed=42):
        self.cat_idx, self.C, self.seed = list(cat_idx), C, seed

    def _design(self, X, fit=False):
        num = np.delete(X, self.cat_idx, axis=1).astype(np.float64)
        if fit:
            self.mu = num.mean(0); self.sd = num.std(0) + 1e-6
            from sklearn.preprocessing import OneHotEncoder
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(X[:, self.cat_idx].astype(np.int64))
        import scipy.sparse as sp
        return sp.hstack([sp.csr_matrix((num - self.mu) / self.sd), self.ohe.transform(X[:, self.cat_idx].astype(np.int64))]).tocsr()

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=self.C, max_iter=300, solver="lbfgs", random_state=self.seed).fit(self._design(X, fit=True), y)
        return self

    def predict(self, X):
        return self.lr.decision_function(self._design(X))


# ------------------------------------------------------------------------------------------------------------------
class _PCPMatrix:
    """PCP x specialist referral counts from encounters with enc_day < refit day; refit every REFIT days (as-of)."""

    REFIT = 28

    def __init__(self, mi):
        self.mi = mi; self.fit_day = None

    def _counts(self, cutoff):
        mi = self.mi; h = mi.enc_day < cutoff
        M = np.zeros((mi.n_pcp, mi.n_specialist), np.float32)
        np.add.at(M, (mi.enc_pcp[h], mi.enc_target[h]), 1.0)
        return M

    def maybe_refit(self, cutoff):
        if self.fit_day is None or cutoff < self.fit_day or cutoff >= self.fit_day + self.REFIT:
            self.fit_day = int(cutoff); self.refit(self._counts(cutoff))


class ALSScorer(_PCPMatrix, Scorer):
    """Implicit-feedback ALS (Hu, Koren & Volinsky 2008) on PCP x specialist counts; confidence 1 + alpha*log1p(count)."""

    def __init__(self, mi, factors=32, reg=0.1, alpha=10.0, iters=15, seed=42):
        _PCPMatrix.__init__(self, mi); self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def refit(self, M):
        rng = np.random.default_rng(self.seed); nu, ni = M.shape; f = self.f
        P = (M > 0).astype(np.float64); Cm = 1.0 + self.alpha * np.log1p(M.astype(np.float64))
        U = rng.normal(0, 0.01, (nu, f)); V = rng.normal(0, 0.01, (ni, f)); I = np.eye(f) * self.reg
        for _ in range(self.iters):
            for X, Y, Cx, Px in ((U, V, Cm, P), (V, U, Cm.T, P.T)):
                YtY = Y.T @ Y
                for u in range(X.shape[0]):
                    c = Cx[u]; nz = c > 1.0
                    A = YtY + (Y[nz].T * (c[nz] - 1.0)) @ Y[nz] + I
                    X[u] = np.linalg.solve(A, (Y.T * c) @ Px[u])
        self.U, self.V = U, V

    def prepare(self, cutoff, batch_enc):
        self.maybe_refit(int(cutoff))

    def score(self, q, cand):
        return self.V[np.asarray(cand, np.int64)] @ self.U[int(self.mi.enc_pcp[int(q)])]


class ItemKNNScorer(_PCPMatrix, Scorer):
    """Specialist-specialist cosine over PCP co-referral (log counts); score = sum_j w_pj * cos(cand, j)."""

    def refit(self, M):
        X = np.log1p(M.astype(np.float64)); n = np.linalg.norm(X, axis=0)
        self.Xn = X / np.where(n > 0, n, 1.0)                                 # columns (specialists) unit-normalised
        self.Z = self.Xn @ X.T                                                # (pcp x pcp): Z[:, p] = Xn @ x_p

    def prepare(self, cutoff, batch_enc):
        self.maybe_refit(int(cutoff))

    def score(self, q, cand):
        # sum_j x_pj cos(cand, j) = Xn[:, cand]^T (Xn x_p)
        return self.Xn[:, np.asarray(cand, np.int64)].T @ self.Z[:, int(self.mi.enc_pcp[int(q)])]


# ------------------------------------------------------------------------------------------------------------------
class BoosterAt:
    """LightGBM booster truncated to a tuned number of rounds."""

    def __init__(self, booster, num_iteration):
        self.booster, self.k = booster, int(num_iteration)

    def predict(self, X):
        return self.booster.predict(X, num_iteration=self.k)


BASELINE_KINDS = ("lgbm", "logreg", "als", "itemknn")


def is_baseline_run(man: dict) -> bool:
    return man.get("kind") in BASELINE_KINDS


def load_baseline_scorer(run_dir: str, mi, ablate: str = None) -> Scorer:
    """Rebuild the scorer of a Step 11.1 / STEP12a baseline run directory (run_manifest.json + model file).
    ablate (STEP12a B1, LightGBM/logreg only): "no_volume" / "no_volume_plus" inference neutralisation."""
    import json
    import os
    import pickle
    man = json.load(open(os.path.join(run_dir, "run_manifest.json"))); kind = man["kind"]
    if kind == "lgbm":
        import lightgbm as lgb
        return TabularScorer(mi, BoosterAt(lgb.Booster(model_file=os.path.join(run_dir, "model.txt")), man["params"]["num_iteration"]), man["feature_set"],
                             ablate=ablate, **man.get("feature_opts", {}))
    if kind == "logreg":
        return TabularScorer(mi, pickle.load(open(os.path.join(run_dir, "model.pkl"), "rb")), man["feature_set"], ablate=ablate, **man.get("feature_opts", {}))
    assert ablate is None, f"ablation {ablate} is defined for LightGBM / logistic runs only"
    if kind == "als":
        p = man["params"]; return ALSScorer(mi, factors=p["factors"], reg=p["reg"], alpha=p["alpha"], iters=p["iters"], seed=man["seed"])
    if kind == "itemknn":
        return ItemKNNScorer(mi)
    raise ValueError(kind)
