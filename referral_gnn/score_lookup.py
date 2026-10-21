"""(query encounter, candidate specialist) → score lookup, numpy only (safe in the LightGBM process). STEP12d."""
import numpy as np


class ScoreLookup:
    def __init__(self, enc, spec, score):
        enc, spec, score = np.asarray(enc, np.int64), np.asarray(spec, np.int64), np.asarray(score, np.float32)
        o = np.lexsort((spec, enc)); self.enc, self.spec, self.score = enc[o], spec[o], score[o]
        u, first = np.unique(self.enc, return_index=True); last = np.r_[first[1:], len(self.enc)]
        self.starts = {int(e): (int(a), int(b)) for e, a, b in zip(u, first, last)}

    def get(self, q, cand) -> np.ndarray:
        a, b = self.starts[int(q)]; sp = self.spec[a:b]; cand = np.asarray(cand, np.int64)
        i = np.searchsorted(sp, cand); ok = (i < len(sp)) & (sp[np.minimum(i, len(sp) - 1)] == cand)
        assert ok.all(), f"missing score for query {q} ({int((~ok).sum())} candidates)"
        return self.score[a:b][i]

    @staticmethod
    def concat(*lookups):
        return ScoreLookup(np.concatenate([l.enc for l in lookups]), np.concatenate([l.spec for l in lookups]), np.concatenate([l.score for l in lookups]))


def dump_to_arrays(npz, key_suffix):
    """Flat (enc, spec, score) from a step10_evaluate --dump-scores file for the score key ending in key_suffix."""
    k = [x for x in npz.files if x.startswith("score::") and x.endswith(key_suffix)]; assert len(k) == 1, (key_suffix, k)
    n = np.diff(npz["offsets"]); return np.repeat(npz["enc"], n), npz["spec"], npz[k[0]]


def time_folds(mi, K: int = 3) -> np.ndarray:
    """Fold id (0..K-1) per training query, aligned with mi.train_query_enc: training queries sorted by day, cut into K contiguous
    blocks of equal size."""
    q = np.asarray(mi.train_query_enc); order = np.argsort(mi.enc_day[q], kind="stable")
    f = np.empty(len(q), np.int64); f[order] = np.minimum((np.arange(len(q)) * K) // len(q), K - 1)
    return f


def fold_of_tc_rows(mi, K: int = 3) -> np.ndarray:
    """Fold id per training-candidate row (mi.tc_enc order, 21 rows per query)."""
    f = time_folds(mi, K); pos = {int(e): i for i, e in enumerate(mi.train_query_enc)}
    return np.array([f[pos[int(e)]] for e in mi.tc_enc[::21]]).repeat(21)
