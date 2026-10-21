"""STEP12d heavy hybrids (STEP12D_HEAVY_PLAN.md).

  A  table → GNN   ("offset"): GNN score + β·(table score). Training rows carry OUT-OF-FOLD table scores (LightGBM fitted on the
                   other time folds of the training queries), so the GNN never sees in-sample table scores; at evaluation the full
                   table model's score is used (the referenced table run, scored by the evaluator).
  B  GNN → table   (stacking): LightGBM on the pure-tabular features + the GNN's score; training rows carry OUT-OF-FOLD GNN
                   scores (GNNs retrained on the other time folds); at evaluation the full GNN's score is used.
  C  dual encoder  ("tab"): one model, GNN encoder + a tabular tower over the 117 pure-tabular columns, trained end to end.

Time folds: the training queries sorted by day, cut into K contiguous blocks of equal size (K = 3). Out-of-fold scores for block k
come from a model trained on the other blocks (including later ones: standard stacking; disclosed — the scored rows' own labels are
never seen). torch and LightGBM never run in one process (macOS libomp clash): scores cross between them through .npz files.
"""
from __future__ import annotations

import numpy as np
import torch

from .evaluate import ModelScorer
from .score_lookup import ScoreLookup, time_folds, fold_of_tc_rows  # noqa: F401  (re-exported)


class ExtraModelScorer(ModelScorer):
    """ModelScorer for STEP12d models: supplies tab rows (dual encoder) and/or table-score offsets (offset model)."""

    def __init__(self, model, mi, device, lookup: ScoreLookup = None, offset_fn=None):
        super().__init__(model, mi, device)
        self.lookup, self.offset_fn = lookup, offset_fn            # offset_fn(q, cand) overrides lookup (evaluator: table run scores)
        self.fb = None
        if model.cfg.tab_dim:
            from .tabular import FeatureBuilder
            self.fb = FeatureBuilder(mi, "tabular"); assert len(self.fb.names) == model.cfg.tab_dim

    @torch.no_grad()
    def prepare(self, cutoff, batch_enc):
        super().prepare(cutoff, batch_enc)
        if self.fb is not None: self.fb.prepare(cutoff, batch_enc)

    @torch.no_grad()
    def score(self, q, cand):
        mi, dev = self.mi, self.device; cand = np.asarray(cand, np.int64)
        pf_np = mi.pair_features(np.full(len(cand), q), cand); pf = torch.as_tensor(pf_np, device=dev)
        c = torch.as_tensor(cand, device=dev); e = torch.full_like(c, int(q))
        tab = torch.as_tensor(self.fb.rows(int(q), cand, pf_np), device=dev) if self.fb is not None else None
        off = None
        if self.model.cfg.table_offset:
            o = self.offset_fn(q, cand) if self.offset_fn is not None else self.lookup.get(q, cand)
            off = torch.as_tensor(np.asarray(o, np.float32), device=dev)
        return self.model.score(self.h, e, c, pf, tab=tab, offset=off).float().cpu().numpy()
