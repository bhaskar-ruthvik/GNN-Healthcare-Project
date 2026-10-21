"""Model: per-type input encoders, HGT message passing, pair scorer.

Implements ARCHITECTURE.md. Ablations are controlled through `ModelConfig`
and are applied consistently at training and evaluation time.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

NODE_TYPES = ["patient", "pcp", "encounter", "specialist"]
EDGE_TYPES = [
    ("patient", "seen_by", "pcp"), ("pcp", "rev_seen_by", "patient"),
    ("pcp", "has_encounter", "encounter"), ("encounter", "rev_has_encounter", "pcp"),
    ("encounter", "referred_to", "specialist"), ("specialist", "rev_referred_to", "encounter"),
]
REFERRAL_EDGES = {("encounter", "referred_to", "specialist"), ("specialist", "rev_referred_to", "encounter")}


@dataclass
class ModelConfig:
    hidden: int = 64
    heads: int = 4
    layers: int = 2
    cat_dim: int = 16
    dx_dim: int = 32
    dropout: float = 0.2
    scorer_hidden: int = 128
    scorer_context: str = "enc_only"          # or "enc_patient_pcp"
    use_graph: bool = True                     # False -> PairMLP (no message passing)
    # ablations (all False = full model)
    no_diagnosis: bool = False                 # zero diagnosis embedding and diagnosis flags
    no_distance: bool = False                  # zero pair distance columns
    no_history_features: bool = False          # zero pair referral-history columns (count, recency, first-observed)
    no_patient_specialist_prior: bool = False  # zero the provisional patient-specialist prior count
    no_patient_history: bool = False           # zero encounter patient-history block
    no_referral_edges: bool = False            # drop REFERRED_TO edges (both directions) from message passing
    no_time_encoding: bool = False             # zero the temporal encoding block

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return ModelConfig(**d)


class NodeEncoder(nn.Module):
    """numeric + categorical (+ optional extra vector) -> hidden."""

    def __init__(self, num_dim: int, cat_sizes: List[int], hidden: int, cat_dim: int, extra_dim: int = 0):
        super().__init__()
        self.num = nn.Linear(num_dim, hidden)
        self.embs = nn.ModuleList([nn.Embedding(s, cat_dim) for s in cat_sizes])
        self.out = nn.Sequential(nn.Linear(hidden + cat_dim * len(cat_sizes) + extra_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())

    def forward(self, x_num, x_cat, extra=None):
        parts = [self.num(x_num)] + [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        if extra is not None:
            parts.append(extra)
        return self.out(torch.cat(parts, dim=-1))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ReferralGNN(nn.Module):
    """Encoders -> (optional) HGT -> pair scorer.

    `dims` is `ModelInputs.dims()`; `blocks` is `ModelInputs.encounter_blocks`;
    `pair_blocks` is `ModelInputs.pair_feature_blocks`.
    """

    def __init__(self, dims: dict, blocks: dict, pair_blocks: dict, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = blocks
        self.pair_blocks = pair_blocks
        d = cfg.hidden
        self.enc = nn.ModuleDict({
            "patient": NodeEncoder(dims["patient_num"], dims["patient_cat_sizes"], d, cfg.cat_dim),
            "pcp": NodeEncoder(dims["pcp_num"], dims["pcp_cat_sizes"], d, cfg.cat_dim),
            "specialist": NodeEncoder(dims["specialist_num"], dims["specialist_cat_sizes"], d, cfg.cat_dim),
            "encounter": NodeEncoder(dims["encounter_num"], dims["encounter_cat_sizes"], d, cfg.cat_dim, extra_dim=cfg.dx_dim),
        })
        self.dx_emb = nn.Embedding(dims["dx_vocab"], cfg.dx_dim, padding_idx=0)
        metadata = (NODE_TYPES, EDGE_TYPES)
        if cfg.use_graph:
            self.convs = nn.ModuleList([HGTConv(d, d, metadata, heads=cfg.heads) for _ in range(cfg.layers)])
            self.norms = nn.ModuleList([nn.ModuleDict({nt: nn.LayerNorm(d) for nt in NODE_TYPES}) for _ in range(cfg.layers)])
        self.drop = nn.Dropout(cfg.dropout)
        ctx = 2 * d if cfg.scorer_context == "enc_patient_pcp" else 0
        self.scorer = MLP(2 * d + dims["pair"] + ctx, cfg.scorer_hidden, cfg.dropout)

    # ------------------------------------------------------------------
    def _zero_block(self, x: torch.Tensor, name: str) -> torch.Tensor:
        a, b = self.blocks[name]
        mask = torch.ones(x.shape[1], device=x.device)
        mask[a:b] = 0
        return x * mask

    def _dx(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean of code embeddings; PAD slots excluded; empty rows -> zeros."""
        e = self.dx_emb(ids) * mask.unsqueeze(-1)
        n = mask.sum(1, keepdim=True).clamp(min=1)
        out = e.sum(1) / n
        if self.cfg.no_diagnosis:
            out = torch.zeros_like(out)
        return out

    def encode(self, batch: dict) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        enc_num = x["encounter_num"]
        if self.cfg.no_patient_history:
            enc_num = self._zero_block(enc_num, "patient_history")
        if self.cfg.no_diagnosis:
            enc_num = self._zero_block(enc_num, "diagnosis_flags")
        if self.cfg.no_time_encoding:
            enc_num = self._zero_block(enc_num, "time")
        h = {
            "patient": self.enc["patient"](x["patient_num"], x["patient_cat"]),
            "pcp": self.enc["pcp"](x["pcp_num"], x["pcp_cat"]),
            "specialist": self.enc["specialist"](x["specialist_num"], x["specialist_cat"]),
            "encounter": self.enc["encounter"](enc_num, x["encounter_cat"], self._dx(x["encounter_dx"], x["encounter_dx_mask"].float())),
        }
        if not self.cfg.use_graph:
            return h
        edge_index = {k: v for k, v in batch["edge_index"].items() if not (self.cfg.no_referral_edges and k in REFERRAL_EDGES)}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index)
            h = {nt: self.drop(F.relu(norm[nt](h[nt] + out[nt]))) for nt in NODE_TYPES}
        return h

    def _mask_pair(self, pf: torch.Tensor) -> torch.Tensor:
        cols = []
        if self.cfg.no_history_features:
            cols += self.pair_blocks["referral_history"]
        if self.cfg.no_distance:
            cols += self.pair_blocks["distance"]
        if self.cfg.no_patient_specialist_prior:
            cols += self.pair_blocks["patient_specialist"]
        if cols:
            mask = torch.ones(pf.shape[1], device=pf.device)
            mask[cols] = 0
            pf = pf * mask
        return pf

    def score(self, h: Dict[str, torch.Tensor], enc_idx: torch.Tensor, spec_idx: torch.Tensor, pair_feat: torch.Tensor,
              patient_idx: Optional[torch.Tensor] = None, pcp_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scores for aligned (query encounter, candidate specialist) pairs."""
        parts = [h["encounter"][enc_idx], h["specialist"][spec_idx], self._mask_pair(pair_feat)]
        if self.cfg.scorer_context == "enc_patient_pcp":
            parts += [h["patient"][patient_idx], h["pcp"][pcp_idx]]
        return self.scorer(torch.cat(parts, dim=-1))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
