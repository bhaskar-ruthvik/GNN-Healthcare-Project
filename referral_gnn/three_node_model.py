"""Three-node model (PCP, Encounter, Specialist) with optional PRECEDES edges.

Implements the revised ARCHITECTURE.md (2026-09-21). Dimensions are derived
from `ThreeNodeInputs.dims()`. The two comparison arms share identical inputs;
`temporal_edges=False` drops the PRECEDES edge type inside the model so the
graph batch itself never differs between arms.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

from .model import NodeEncoder, MLP

NODE_TYPES = ["pcp", "encounter", "specialist"]
TEMPORAL_EDGE = ("encounter", "precedes", "encounter")
EDGE_TYPES = [
    ("pcp", "has_encounter", "encounter"), ("encounter", "rev_has_encounter", "pcp"),
    ("encounter", "referred_to", "specialist"), ("specialist", "rev_referred_to", "encounter"),
    TEMPORAL_EDGE,
]
REFERRAL_EDGES = {("encounter", "referred_to", "specialist"), ("specialist", "rev_referred_to", "encounter")}


@dataclass
class ThreeNodeConfig:
    schema: str = "three-node-v1"
    hidden: int = 64
    heads: int = 4
    layers: int = 2
    cat_dim: int = 16
    dx_dim: int = 32
    dropout: float = 0.2
    scorer_hidden: int = 128
    temporal_edges: bool = True          # arm 2 = True, arm 1 (summaries only) = False
    use_graph: bool = True               # False -> no message passing (internal control only)
    # ablations (all False = full model); applied identically at train and eval
    no_diagnosis: bool = False           # zero category bag + diagnosis flags
    no_distance: bool = False            # zero pair distance columns
    no_history_features: bool = False    # zero pair referral-history columns
    no_patient_specialist_prior: bool = False
    no_patient_history: bool = False     # zero encounter patient-history block
    no_predecessor_gap: bool = False     # zero the predecessor-gap block (both arms carry it)
    no_referral_edges: bool = False
    no_time_encoding: bool = False
    mask_demographics: bool = False       # STEP9: set patient gender/race categorical indices to UNK and zero their ambiguity flags
    scorer_skip: bool = False             # STEP11 B5 C2: scorer also gets the pre-HGT encodings and their element-wise product
    no_insurer_history: bool = False      # STEP12a gate: zero the pair insurer-history columns (insurer_history, insurer_unknown)
    volume_mode: str = "keep"             # STEP12a: "keep" | "log1p" (H5) | "shares" (B2); see referral_gnn/volume.py
    table_offset: bool = False            # STEP12d A: score += β·(table score), β learned (init 1); the GNN learns on top of the table
    tab_dim: int = 0                      # STEP12d C: >0 = dual encoder; width of the pure-tabular feature row fed to a tabular tower
    tab_hidden: int = 64
    pcp_in_scorer: bool = False           # STEP13 NG+P: the scorer also reads the query PCP's encoding h["pcp"][enc_pcp[enc_idx]]

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return ThreeNodeConfig(**d)


class ThreeNodeGNN(nn.Module):
    def __init__(self, dims: dict, blocks: dict, pair_blocks: dict, cfg: ThreeNodeConfig):
        super().__init__()
        assert set(dims) >= {"pcp_num", "pcp_cat_sizes", "specialist_num", "specialist_cat_sizes", "encounter_num", "encounter_cat_sizes", "dx_vocab", "pair"}
        assert "patient_num" not in dims, "three-node model received four-node dims"
        assert "predecessor_gap" in blocks, "inputs lack the predecessor-gap block"
        self.cfg, self.blocks, self.pair_blocks, self.dims = cfg, blocks, pair_blocks, dims
        assert cfg.volume_mode in ("keep", "log1p", "shares", "strict")
        self.volume_stats = blocks.get("volume_stats")              # deterministic from the inputs; recomputed on load
        assert cfg.volume_mode == "keep" or self.volume_stats is not None, "volume_mode needs blocks['volume_stats']"
        d = cfg.hidden
        self.enc = nn.ModuleDict({
            "pcp": NodeEncoder(dims["pcp_num"], dims["pcp_cat_sizes"], d, cfg.cat_dim),
            "specialist": NodeEncoder(dims["specialist_num"], dims["specialist_cat_sizes"], d, cfg.cat_dim),
            "encounter": NodeEncoder(dims["encounter_num"], dims["encounter_cat_sizes"], d, cfg.cat_dim, extra_dim=cfg.dx_dim),
        })
        self.dx_emb = nn.Embedding(dims["dx_vocab"], cfg.dx_dim, padding_idx=0)
        if cfg.use_graph:
            metadata = (NODE_TYPES, EDGE_TYPES)
            self.convs = nn.ModuleList([HGTConv(d, d, metadata, heads=cfg.heads) for _ in range(cfg.layers)])
            self.norms = nn.ModuleList([nn.ModuleDict({nt: nn.LayerNorm(d) for nt in NODE_TYPES}) for _ in range(cfg.layers)])
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.tab_dim:                                     # STEP12d C: tabular tower (standardised pure-tabular row → tab_hidden)
            self.tab_tower = nn.Sequential(nn.Linear(cfg.tab_dim, cfg.tab_hidden), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.tab_hidden, cfg.tab_hidden), nn.ReLU())
            self.register_buffer("tab_mu", torch.zeros(cfg.tab_dim)); self.register_buffer("tab_sd", torch.ones(cfg.tab_dim))
        if cfg.table_offset: self.offset_beta = nn.Parameter(torch.ones(()))
        if cfg.pcp_in_scorer:                               # STEP13: encounter → PCP map, rebuilt from the inputs on load (not saved)
            assert "enc_pcp" in blocks, "pcp_in_scorer needs blocks['enc_pcp']"
            self.register_buffer("enc_pcp", torch.as_tensor(blocks["enc_pcp"], dtype=torch.long), persistent=False)
        self.scorer = MLP(((5 if cfg.scorer_skip else 2) + int(cfg.pcp_in_scorer)) * d + dims["pair"] + (cfg.tab_hidden if cfg.tab_dim else 0), cfg.scorer_hidden, cfg.dropout)

    # ------------------------------------------------------------------
    def _zero_block(self, x, name):
        a, b = self.blocks[name]
        m = torch.ones(x.shape[1], device=x.device); m[a:b] = 0
        return x * m

    def _dx(self, ids, mask):
        e = self.dx_emb(ids) * mask.unsqueeze(-1)
        out = e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return torch.zeros_like(out) if self.cfg.no_diagnosis else out

    def active_edge_types(self):
        out = []
        for et in EDGE_TYPES:
            if et == TEMPORAL_EDGE and not self.cfg.temporal_edges:
                continue
            if et in REFERRAL_EDGES and self.cfg.no_referral_edges:
                continue
            out.append(et)
        return out

    def encode(self, batch: dict) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        assert tuple(batch.get("node_types", NODE_TYPES)) == tuple(NODE_TYPES), "batch is not a three-node batch"
        enc_num = x["encounter_num"]
        assert enc_num.shape[1] == self.dims["encounter_num"], "encounter feature width mismatch: wrong inputs for this model"
        c = self.cfg
        if c.no_patient_history: enc_num = self._zero_block(enc_num, "patient_history")
        if c.no_diagnosis: enc_num = self._zero_block(enc_num, "diagnosis_flags")
        if c.no_time_encoding: enc_num = self._zero_block(enc_num, "time")
        if c.no_predecessor_gap: enc_num = self._zero_block(enc_num, "predecessor_gap")
        enc_cat = x["encounter_cat"]
        if c.mask_demographics:
            cat_cols = self.blocks.get("demographic_cat_cols"); flag_cols = self.blocks.get("demographic_flag_cols")
            assert cat_cols is not None, "inputs do not define demographic columns; masking unavailable"
            keep = torch.ones(enc_cat.shape[1], dtype=enc_cat.dtype, device=enc_cat.device); keep[cat_cols] = 0
            enc_cat = enc_cat * keep
            if flag_cols:
                m = torch.ones(enc_num.shape[1], device=enc_num.device); m[flag_cols] = 0
                enc_num = enc_num * m
        spec_num = x["specialist_num"]
        if c.volume_mode != "keep":                       # STEP12a: same arithmetic as the tabular features (volume.transform_np)
            from .volume import transform_np
            spec_num = torch.as_tensor(transform_np(spec_num.detach().cpu().numpy(), self.volume_stats, c.volume_mode), device=spec_num.device, dtype=spec_num.dtype)
        h = {
            "pcp": self.enc["pcp"](x["pcp_num"], x["pcp_cat"]),
            "specialist": self.enc["specialist"](spec_num, x["specialist_cat"]),
            "encounter": self.enc["encounter"](enc_num, enc_cat, self._dx(x["encounter_dx"], x["encounter_dx_mask"].float())),
        }
        h0 = {"encounter": h["encounter"], "specialist": h["specialist"]} if c.scorer_skip else None
        if not c.use_graph:
            return dict(h, **({"encounter0": h0["encounter"], "specialist0": h0["specialist"]} if h0 else {}))
        active = self.active_edge_types()
        edge_index = {et: batch["edge_index"][et] for et in active}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index)
            h = {nt: self.drop(F.relu(norm[nt](h[nt] + out[nt]))) for nt in NODE_TYPES}
        if h0: h = dict(h, encounter0=h0["encounter"], specialist0=h0["specialist"])
        return h

    def _mask_pair(self, pf):
        cols = []
        if self.cfg.no_history_features: cols += self.pair_blocks["referral_history"]
        if self.cfg.no_distance: cols += self.pair_blocks["distance"]
        if self.cfg.no_patient_specialist_prior: cols += self.pair_blocks["patient_specialist"]
        if self.cfg.no_insurer_history: cols += self.pair_blocks["insurer"]
        if cols:
            m = torch.ones(pf.shape[1], device=pf.device); m[cols] = 0
            pf = pf * m
        return pf

    def score(self, h, enc_idx, spec_idx, pair_feat, *_ignored, tab=None, offset=None) -> torch.Tensor:
        """Scores for aligned (query encounter, candidate specialist) pairs.
        Extra positional args (patient/pcp indices from the four-node scorer API) are ignored, except that with pcp_in_scorer
        a passed PCP index must agree with the model's own encounter → PCP map (so training and evaluation read the same PCP).
        STEP12d: tab = pure-tabular rows (dual encoder); offset = table scores (table-offset model)."""
        parts = [h["encounter"][enc_idx], h["specialist"][spec_idx]]
        if self.cfg.pcp_in_scorer:
            pcp_idx = self.enc_pcp[enc_idx]
            if len(_ignored) >= 2: assert torch.equal(pcp_idx, _ignored[1].to(pcp_idx.device)), "PCP index passed by the caller disagrees with enc_pcp"
            parts.append(h["pcp"][pcp_idx])
        if self.cfg.scorer_skip:                                # C2: skip (pre-HGT encodings) + explicit condition × specialist interaction
            e0, s0 = h["encounter0"][enc_idx], h["specialist0"][spec_idx]; parts += [e0, s0, e0 * s0]
        parts.append(self._mask_pair(pair_feat))
        if self.cfg.tab_dim:
            assert tab is not None, "dual-encoder model needs tab rows"
            parts.append(self.tab_tower((tab - self.tab_mu) / self.tab_sd))
        out = self.scorer(torch.cat(parts, dim=-1))
        if self.cfg.table_offset:
            assert offset is not None, "table-offset model needs table scores"
            out = out + self.offset_beta * offset
        return out

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())
