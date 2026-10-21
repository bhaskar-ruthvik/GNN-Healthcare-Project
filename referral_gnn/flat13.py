"""STEP13 receptive-field-matched fixed aggregates (Implementation/STEP13_GRAPH_VALUE_PLAN.md §6, frozen 2026-10-21).

What a 2-layer pass of the three-node GNN (no PRECEDES edges) can reach around the query encounter q and a candidate s, with the
attributes the GNN reads on those nodes, as unweighted MEANS over encounters with enc_day < the weekly cutoff:
  A1  s ← past encounters             encounter numerics (patient context, severity, diagnosis flags), disease-category
                                      distribution, one-hots of the 5 encounter categoricals, and the 16 sinusoidal time-encoding
                                      components of (cutoff − encounter day) (data.time_encoding, the GNN's own encounter-time input)
  A2  s ← encounters ← PCPs           the referring PCPs' encoder inputs: all pcp_num columns + one-hots of pcp_cat
  B1  q ← PCP ← the PCP's encounters  the same attributes as A1 over the query PCP's past encounters
                                      (rev_has_encounter into the PCP, then has_encounter into the query)
  matching terms as in FlatGraph (cosine of query vs A1 categories, query age − A1 mean age) and no-history flags.
Optional (practical Info-table supersets only, outside the 2-layer reach):
  A3  q → PCP → encounters → specialists   mean specialist numeric values of the query PCP's past specialists, and candidate − A3
  ins_share                                the query insurer's share of the candidate's past encounters (Ins-match control);
                                           NaN when the query insurer is unknown
Zero history: the denominator is the number of qualifying encounters; with none the means are 0 and the flag is 1.
Explicit pair / patient history (patient-history block, predecessor gap) is not aggregated: the GNN-nohist cell zeroes it.
"""
from __future__ import annotations

import numpy as np

from .data import TIME_DIM


class Flat13:
    def __init__(self, mi, a3: bool = False, insmatch: bool = False):
        self.mi, self.a3, self.insmatch = mi, bool(a3), bool(insmatch)
        self.order = np.argsort(mi.enc_day, kind="stable"); self.days = mi.enc_day[self.order]
        self.n_cat = int(mi.enc_dx.max()) + 1
        d = mi.dims(); self.enc_cat_sizes, self.pcp_cat_sizes = list(d["encounter_cat_sizes"]), list(d["pcp_cat_sizes"])
        st = mi._encounter_static(); B = mi.encounter_blocks
        assert B["patient_context"] == (0, 6) and B["severity_flags"] == (6, 7) and B["diagnosis_flags"] == (74, 76), "encounter block layout changed"
        self.u_static = np.concatenate([st[:, :7], st[:, 74:76]], axis=1).astype(np.float64)        # context, severity, dx flags
        dist = np.zeros((mi.n_encounter, self.n_cat))                                                  # disease-category distribution (as FlatGraph._u)
        for e in range(mi.n_encounter):
            c = mi.enc_dx[e][mi.enc_dx_mask[e]]; c = np.unique(c[c > 1])
            if len(c): dist[e, c] = 1.0 / len(c)
        onehot = []
        for j, k in enumerate(self.enc_cat_sizes):
            o = np.zeros((mi.n_encounter, k)); o[np.arange(mi.n_encounter), mi.enc_patient_cat[:, j]] = 1.0; onehot.append(o)
        self.U = np.concatenate([self.u_static, dist] + onehot, axis=1)                              # per-encounter attributes, no time
        half = TIME_DIM // 2
        self.freqs = 1.0 / (10000.0 ** (np.arange(half) / half))                                       # data.time_encoding frequencies
        ang = mi.enc_day.astype(np.float64)[:, None] * self.freqs[None, :]
        self.Ts, self.Tc = np.sin(ang), np.cos(ang)                                                   # sin/cos of (day · f): sums give exact means
        self.spec_vals = [3 * i for i in range(len(mi.spec_encoder.numeric_cols))]                    # specialist value columns (as FlatGraph A3)
        self.reset()

    def reset(self):
        mi = self.mi; du = self.U.shape[1]; h = TIME_DIM // 2
        self.ptr = 0; self.cutoff = -10 ** 9
        self.SE = np.zeros((mi.n_specialist, du)); self.SS = np.zeros((mi.n_specialist, h)); self.SC = np.zeros((mi.n_specialist, h)); self.nS = np.zeros(mi.n_specialist)
        self.PE = np.zeros((mi.n_pcp, du)); self.PS = np.zeros((mi.n_pcp, h)); self.PC = np.zeros((mi.n_pcp, h)); self.nP = np.zeros(mi.n_pcp)
        self.Bc = np.zeros((mi.n_pcp, mi.n_specialist))                                               # PCP → specialist referral counts
        self.IC = np.zeros((mi.n_specialist, mi.n_insurer))                                           # specialist × insurer encounter counts

    def advance(self, cutoff: int):
        """Add every encounter with enc_day < cutoff (cutoffs non-decreasing; going back resets)."""
        cutoff = int(cutoff)
        if cutoff < self.cutoff: self.reset()
        mi = self.mi
        end = int(np.searchsorted(self.days, cutoff, "left")); e = self.order[self.ptr:end]
        if len(e):
            s, p = mi.enc_target[e], mi.enc_pcp[e]
            for A, X in ((self.SE, self.U), (self.SS, self.Ts), (self.SC, self.Tc)): np.add.at(A, s, X[e])
            for A, X in ((self.PE, self.U), (self.PS, self.Ts), (self.PC, self.Tc)): np.add.at(A, p, X[e])
            np.add.at(self.nS, s, 1.0); np.add.at(self.nP, p, 1.0); np.add.at(self.Bc, (p, s), 1.0)
            ins = mi.enc_insurer[e]; k = ins >= 0; np.add.at(self.IC, (s[k], ins[k]), 1.0)
        self.ptr = end; self.cutoff = cutoff

    def _time_mean(self, S, C, n):
        """Mean of data.time_encoding(cutoff − day) over a set, from Σ sin(day·f), Σ cos(day·f): [sin block, cos block]."""
        a = self.cutoff * self.freqs; sa, ca = np.sin(a), np.cos(a); den = np.maximum(n, 1.0)[:, None]
        return np.concatenate([(sa * C - ca * S) / den, (ca * C + sa * S) / den], axis=1)

    def _enc_names(self, tag):
        return ([f"f13_{tag}_ctx{i}" for i in range(7)] + [f"f13_{tag}_dxempty", f"f13_{tag}_dxunk"] + [f"f13_{tag}_cat{c}" for c in range(self.n_cat)]
                + [f"f13_{tag}_enccat{j}_{k}" for j, n in enumerate(self.enc_cat_sizes) for k in range(n)] + [f"f13_{tag}_time{i}" for i in range(TIME_DIM)])

    def names(self, n_pcp_num):
        out = self._enc_names("A1") + [f"f13_A2_pcpnum{i}" for i in range(n_pcp_num)] + [f"f13_A2_pcpcat{j}_{k}" for j, n in enumerate(self.pcp_cat_sizes) for k in range(n)]
        out += self._enc_names("B1") + ["f13_cat_cosine", "f13_age_diff", "f13_spec_nohist", "f13_pcp_nohist"]
        if self.a3: out += [f"f13_A3_spec{i}" for i in range(len(self.spec_vals))] + [f"f13_A3diff_spec{i}" for i in range(len(self.spec_vals))]
        if self.insmatch: out += ["f13_ins_share"]
        return out

    def pcp_vectors(self, snap):
        """PCP encoder inputs from the current snapshot: pcp_num + one-hots of pcp_cat (cached per snapshot)."""
        if getattr(self, "_pv_for", None) is snap: return self._pv
        pv = [snap["pcp_num"].astype(np.float64)]
        for j, k in enumerate(self.pcp_cat_sizes):
            o = np.zeros((len(snap["pcp_cat"]), k)); o[np.arange(len(o)), snap["pcp_cat"][:, j]] = 1.0; pv.append(o)
        self._pv_for, self._pv = snap, np.concatenate(pv, axis=1)
        return self._pv

    def features(self, q: int, cand: np.ndarray, snap) -> np.ndarray:
        mi = self.mi; cand = np.asarray(cand, np.int64); n = len(cand); q = int(q); p = int(mi.enc_pcp[q])
        ns = self.nS[cand]; has = ns > 0; den = np.maximum(ns, 1.0)[:, None]
        A1 = np.concatenate([self.SE[cand] / den, self._time_mean(self.SS[cand], self.SC[cand], ns)], axis=1)
        A2 = (self.Bc[:, cand].T @ self.pcp_vectors(snap)) / den                    # node attributes from the CURRENT snapshot (as FlatGraph)
        npp = self.nP[p:p + 1]
        B1 = np.concatenate([self.PE[p:p + 1] / max(npp[0], 1.0), self._time_mean(self.PS[p:p + 1], self.PC[p:p + 1], npp)], axis=1)
        c0 = self.u_static.shape[1]; qc = self.U[q, c0:c0 + self.n_cat]; ac = A1[:, c0:c0 + self.n_cat]
        cos = (ac @ qc) / (np.linalg.norm(ac, axis=1) * (np.linalg.norm(qc) or 1.0) + 1e-9)
        age = self.u_static[q, 0] - A1[:, 0]
        cos[~has] = 0.0; age[~has] = 0.0                                             # undefined without history → 0 (flagged)
        parts = [A1, A2, np.broadcast_to(B1, (n, B1.shape[1])), cos[:, None], age[:, None], (~has)[:, None].astype(np.float64), np.full((n, 1), float(npp[0] == 0))]
        if self.a3:
            sv = snap["specialist_num"][:, self.spec_vals].astype(np.float64); A3 = (self.Bc[p] @ sv) / max(npp[0], 1.0)
            parts += [np.broadcast_to(A3, (n, len(A3))), sv[cand] - A3[None, :]]
        if self.insmatch:
            ins = int(mi.enc_insurer[q])
            parts.append((self.IC[cand, ins] / den[:, 0] if ins >= 0 else np.full(n, np.nan))[:, None])
        return np.concatenate(parts, axis=1).astype(np.float32)
