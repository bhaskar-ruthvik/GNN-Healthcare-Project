"""STEP12b hybrids (STEP12B_HYBRID_PLAN.md).

H2 "professor's model": the trained GNN as a frozen encoder; for each (query encounter, candidate specialist) pair the
features are the GNN's pair encoding plus a small set of explicit, interpretable "exploratory" variables; a pointwise logistic
regression on top predicts the referral probability.
  - pair encoding (emb_mode "pair", primary): the GNN scorer's last hidden layer (128) for the pair, i.e. the representation its
    final linear layer reads; (emb_mode "product": element-wise product of the post-HGT encounter and specialist embeddings, 64;
    smoke tests showed a linear model on it recovers little of the GNN's ranking — kept as an option);
  - exploratory variables: the 14 pair features (PCP–specialist referral history, distance, patient–specialist prior, patient
    distance, insurer history, shared affiliation) + specialist MIPS (as of the cutoff), years in Medicare, beneficiary count,
    rating (snapshot values);
  - --no-emb variant: the exploratory variables alone (what the embeddings add);
  - training rows = the GNN's own 21-row bundles; embeddings for training rows come from the GNN that was trained on them
    (in-sample encoder; disclosed). torch only: no LightGBM in these processes (macOS libomp clash).
"""
from __future__ import annotations

import numpy as np
import torch

from .evaluate import Scorer

SPEC_VARS = ["mips_score", "years_in_medicare", "beneficiary_count", "rating"]


class EmbFeatures:
    def __init__(self, model, mi, device, use_emb: bool = True, emb_mode: str = "pair"):
        assert emb_mode in ("pair", "product")
        self.model, self.mi, self.device, self.use_emb, self.emb_mode = model, mi, device, use_emb, emb_mode
        self.sv = [3 * list(mi.spec_encoder.numeric_cols).index(n) for n in SPEC_VARS]
        d = model.cfg.scorer_hidden if emb_mode == "pair" else model.cfg.hidden
        self.names = ([f"emb_{emb_mode}{i}" for i in range(d)] if use_emb else []) + [f"pair_{n}" for n in mi.pair_feature_names] + [f"spec_{n}" for n in SPEC_VARS]

    @torch.no_grad()
    def prepare(self, cutoff, batch_enc):
        self.model.eval()
        self.snap_vals = self.mi.provider_snapshot(int(cutoff))["specialist_num"][:, self.sv].astype(np.float32)
        if self.use_emb:
            self.h = self.model.encode(self.mi.graph_batch(cutoff, batch_enc, device=self.device))

    @torch.no_grad()
    def rows(self, enc: np.ndarray, spec: np.ndarray, pair: np.ndarray = None) -> np.ndarray:
        """Aligned (encounter, specialist) pairs → feature matrix."""
        enc = np.asarray(enc, np.int64); spec = np.asarray(spec, np.int64)
        if pair is None: pair = self.mi.pair_features(enc, spec)
        parts = []
        if self.use_emb:
            e = torch.as_tensor(enc, device=self.device); s = torch.as_tensor(spec, device=self.device)
            if self.emb_mode == "product":
                parts.append((self.h["encounter"][e] * self.h["specialist"][s]).float().cpu().numpy())
            else:                                          # the scorer's input exactly as in ThreeNodeGNN.score, then all layers but the last
                m = self.model; pf = torch.as_tensor(np.asarray(pair, np.float32), device=self.device)
                x = [self.h["encounter"][e], self.h["specialist"][s]]
                if m.cfg.scorer_skip: e0, s0 = self.h["encounter0"][e], self.h["specialist0"][s]; x += [e0, s0, e0 * s0]
                x = torch.cat(x + [m._mask_pair(pf)], dim=-1)
                parts.append(m.scorer.net[:-1](x).float().cpu().numpy())
        parts += [np.asarray(pair, np.float32), self.snap_vals[spec]]
        return np.concatenate(parts, axis=1)


class EmbLR:
    """Standardised features → sklearn LogisticRegression (pointwise)."""

    def __init__(self, C=1.0, seed=42):
        self.C, self.seed = C, seed

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        self.lr = LogisticRegression(C=self.C, max_iter=500, solver="lbfgs", random_state=self.seed).fit((X - self.mu) / self.sd, y)
        return self

    def decision(self, X):
        return self.lr.decision_function((X - self.mu) / self.sd)


class EmbLRScorer(Scorer):
    def __init__(self, feats: EmbFeatures, lr: EmbLR):
        self.f, self.lr = feats, lr

    def prepare(self, cutoff, batch_enc):
        self.f.prepare(cutoff, batch_enc)

    def score(self, q, cand):
        cand = np.asarray(cand, np.int64)
        return np.asarray(self.lr.decision(self.f.rows(np.full(len(cand), int(q)), cand)), float)


def load_emb_lr_scorer(run_dir: str, mi, device: str) -> EmbLRScorer:
    import json, os, pickle
    from .three_node_train import load_three_node_model
    man = json.load(open(os.path.join(run_dir, "run_manifest.json"))); assert man["kind"] == "emb_lr"
    model = load_three_node_model(man["gnn_run"], mi, device)
    lr = pickle.load(open(os.path.join(run_dir, "model.pkl"), "rb"))
    return EmbLRScorer(EmbFeatures(model, mi, device, use_emb=man["use_emb"], emb_mode=man.get("emb_mode", "product")), lr)
