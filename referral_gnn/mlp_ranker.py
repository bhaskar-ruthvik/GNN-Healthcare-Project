"""STEP13 MLP cells (MLP-RF, MLP-info; STEP13_GRAPH_VALUE_PLAN.md §6, frozen 2026-10-21).

A plain MLP scorer on a FeatureBuilder row, trained like the GNN: the same 21-row bundles (chosen specialist + 20 uniform
alternatives, `tabular.training_matrix`) with a softmax over each bundle, and checkpoints selected by first-time MRR on the 2018
selection sample (training-time selection only; never an outcome).
  inputs        categoricals one-hot (levels from FeatureBuilder.cat_sizes); numerics standardised with training-only means and
                standard deviations (columns constant in training → 0); missing flags kept as numerics
  architecture  2 hidden layers of width `hidden`, ReLU, dropout `dropout`, linear output
  optimiser     Adam, lr 1e-3, weight decay 1e-5; ReduceLROnPlateau(0.5, patience 2) — halved on the 3rd consecutive non-improving
                epoch, as the GNN; early stop after 5 non-improving epochs; ≤ 100 epochs
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRanker(nn.Module):
    def __init__(self, n_num: int, cat_sizes, hidden: int, dropout: float):
        super().__init__()
        self.cat_sizes = [int(k) for k in cat_sizes]
        d_in = n_num + sum(self.cat_sizes)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(n_num)); self.register_buffer("sd", torch.ones(n_num))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        assert all(int(x_cat[:, j].max()) < k and int(x_cat[:, j].min()) >= 0 for j, k in enumerate(self.cat_sizes)), "category code outside its levels"
        parts = [(x_num - self.mu) / self.sd] + [F.one_hot(x_cat[:, j], k).to(x_num.dtype) for j, k in enumerate(self.cat_sizes)]
        return self.net(torch.cat(parts, dim=1)).squeeze(-1)


def split_columns(X: np.ndarray, cat_idx):
    """(numeric float32 block, categorical int64 block) of FeatureBuilder rows."""
    num = np.delete(X, cat_idx, axis=1).astype(np.float32)
    return num, X[:, cat_idx].astype(np.int64)


class MLPPredict:
    """predict(X) on FeatureBuilder rows, for tabular.TabularScorer."""

    def __init__(self, model: MLPRanker, cat_idx, device: str = "cpu", chunk: int = 65536):
        self.model, self.cat_idx, self.device, self.chunk = model, list(cat_idx), device, chunk

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval(); num, cat = split_columns(X, self.cat_idx); out = []
        for a in range(0, len(X), self.chunk):
            out.append(self.model(torch.as_tensor(num[a:a + self.chunk], device=self.device), torch.as_tensor(cat[a:a + self.chunk], device=self.device)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def load_mlp_scorer(run_dir: str, mi, device: str = "cpu"):
    """Rebuild a STEP13 MLP run's scorer (run_manifest.json + best.pt)."""
    from .tabular import FeatureBuilder, TabularScorer
    man = json.load(open(os.path.join(run_dir, "run_manifest.json"))); assert man["kind"] == "mlp", run_dir
    fb = FeatureBuilder(mi, man["feature_set"]); assert fb.names == man["feature_names"], "feature columns differ from the trained model's"
    model = MLPRanker(len(fb.names) - len(fb.cat_idx), fb.cat_sizes, man["hidden"], man["dropout"]).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device)["model_state"]); model.eval()
    return TabularScorer(mi, MLPPredict(model, fb.cat_idx, device), man["feature_set"])
