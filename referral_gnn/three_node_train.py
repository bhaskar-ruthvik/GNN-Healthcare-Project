"""Trainer for the three-node model. Inputs must be passed explicitly
(`ThreeNodeInputs` pickle path); nothing here selects the old bundle.
Reuses the sampled-softmax loss, bucketed batching, validation subset,
early stopping and checkpointing logic of the four-node trainer.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric

from .data import ModelInputs
from .three_node_data import ThreeNodeInputs
from .three_node_model import ThreeNodeConfig, ThreeNodeGNN
from .train import TrainConfig, Trainer, pick_device, set_seed, sha256_file, code_hash
from .evaluate import ModelScorer, evaluate, metrics


def load_three_node_inputs(path: str) -> ThreeNodeInputs:
    mi = ModelInputs.load(path)
    assert isinstance(mi, ThreeNodeInputs), f"{path} is not a ThreeNodeInputs pickle"
    assert getattr(mi, "schema_version", None) in ("three-node-v1", "three-node-v2", "three-node-v3", "three-node-v4", "three-node-v5", "three-node-v6"), "unexpected input schema"
    return mi


def _blocks_with_demographics(mi, mcfg=None):
    b = dict(mi.encounter_blocks)
    if hasattr(mi, "demographic_cat_cols"):
        b["demographic_cat_cols"] = list(mi.demographic_cat_cols); b["demographic_flag_cols"] = list(mi.demographic_flag_cols)
    if mcfg is not None and getattr(mcfg, "volume_mode", "keep") != "keep":       # STEP12a H5 / B2
        from .volume import stats
        b["volume_stats"] = stats(mi)
    return b


class ThreeNodeTrainer(Trainer):
    def __init__(self, mi: ThreeNodeInputs, mcfg: ThreeNodeConfig, tcfg: TrainConfig, run_dir: str, inputs_path: Optional[str] = None):
        assert isinstance(mi, ThreeNodeInputs), "ThreeNodeTrainer requires ThreeNodeInputs"
        # --- replicate Trainer.__init__ with the three-node model ---------------------------
        self.mi, self.mcfg, self.tcfg, self.run_dir = mi, mcfg, tcfg, run_dir
        os.makedirs(run_dir, exist_ok=True)
        set_seed(tcfg.seed)
        self.device = pick_device(tcfg.device)
        self.model = ThreeNodeGNN(mi.dims(), _blocks_with_demographics(mi, mcfg), mi.pair_feature_blocks, mcfg).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        self.device_fallback = None
        self.tc_day = mi.enc_day[mi.tc_enc]
        assert (np.diff(self.tc_day) >= 0).all()
        self.train_days = np.unique(self.tc_day)
        self.day_slices = {int(d): (int(a), int(b)) for d, a, b in zip(self.train_days, np.searchsorted(self.tc_day, self.train_days, "left"), np.searchsorted(self.tc_day, self.train_days, "right"))}
        self.train_buckets = mi.buckets("train", tcfg.graph_bucket, query_subset=mi.train_query_enc)
        self.tc_rows_of = lambda encs: (np.searchsorted(mi.tc_enc[::21], encs) * 21)
        t0 = time.time()
        self.train_pair_feat = mi.pair_features(mi.tc_enc, mi.tc_spec)
        self.prep_time = time.time() - t0
        manifest_path = os.path.join(os.path.dirname(inputs_path), "manifest.json") if inputs_path else None
        inputs_manifest = json.load(open(manifest_path)) if manifest_path and os.path.exists(manifest_path) else None
        self.manifest = {
            "schema": mi.schema_version,
            "inputs_path": inputs_path, "inputs_sha256": sha256_file(inputs_path) if inputs_path else None,
            "inputs_manifest_inputs_sha256": inputs_manifest.get("inputs_sha256") if inputs_manifest else None,
            "code_sha256": code_hash(os.path.dirname(os.path.abspath(__file__))),
            "torch": torch.__version__, "torch_geometric": torch_geometric.__version__, "python": platform.python_version(),
            "device": self.device, "device_fallback": None, "seed": tcfg.seed, "n_parameters": self.model.n_parameters(),
            "model_config": mcfg.to_dict(), "train_config": asdict(tcfg),
            "n_train_days": int(len(self.train_days)), "n_train_buckets": int(len(self.train_buckets)), "n_train_queries": int(len(mi.train_query_enc)),
        }
        if inputs_manifest and self.manifest["inputs_sha256"] and inputs_manifest.get("inputs_sha256") not in (None, self.manifest["inputs_sha256"]):
            raise RuntimeError("inputs pickle hash does not match its manifest; refusing to train on a modified artifact")
        with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=1)
        if not (mcfg.table_offset or mcfg.tab_dim):        # STEP12d models: probed by the caller after their extras are attached
            self._probe_device()

    # ---- negative sampling ------------------------------------------------------------------
    neg_policy = "uniform"        # "uniform" = bundled alternatives; "hard" = familiar + same-specialty + uniform; "pool" = target-specialty ∩ unseen-by-patient
    train_cohort = "all"          # "all" or "firstobs" (training queries whose target the patient had not seen before)
    neg_k = (8, 8, 4)             # hard policy: (PCP-familiar, target-specialty, uniform) per query; total 20
    neg_seed = 42

    def _prep_hard_negatives(self):
        mi = self.mi
        # per-PCP sorted (day, specialist) history for "familiar as of day d"
        order = np.lexsort((mi.pair_day, mi.pair_ent // mi.n_specialist))
        pcp_of = (mi.pair_ent // mi.n_specialist)[order]; self._fam_day = mi.pair_day[order]; self._fam_spec = (mi.pair_ent % mi.n_specialist)[order]
        self._fam_start = np.searchsorted(pcp_of, np.arange(mi.n_pcp + 1))
        # cutoff-safe specialty per specialist-day
        from .data import AsOfIndex
        rows = mi.spec_hist_cat[:, 0]; nn = rows > 0
        ent = mi.spec_hist_asof.entity[np.argsort(mi.spec_hist_asof.order)]
        self._sp_idx = AsOfIndex(ent[nn], mi.spec_hist_day[nn]); self._sp_val = rows[nn]
        self._neg_rng = np.random.default_rng(self.neg_seed)

    def _specialty_asof(self, spec_ids, d):
        r = self._sp_idx.lookup(np.asarray(spec_ids), int(d) + 1); return np.where(r >= 0, self._sp_val[np.clip(r, 0, None)], 0)

    def sample_hard_negatives(self, query_enc: np.ndarray, cutoff: int):
        """(Q, 21) specialist matrix: column 0 = target; then up to k1 PCP-familiar (as of the query day),
        k2 same-specialty-as-target (as of the query day), rest uniform from the available pool. No duplicates, target excluded."""
        mi = self.mi; k1, k2, k3 = self.neg_k; K = k1 + k2 + k3
        all_s = np.arange(mi.n_specialist)
        sp_all = self._specialty_asof(all_s, cutoff)            # specialty as of the bucket cutoff (per-bucket cost)
        out = np.empty((len(query_enc), K + 1), np.int64)
        self._neg_short = getattr(self, "_neg_short", 0)
        for i, q in enumerate(query_enc):
            q = int(q); t = int(mi.enc_target[q]); d = int(mi.enc_day[q]); p = int(mi.enc_pcp[q])
            elig = (mi.spec_first_seen < d); elig[t] = False        # exact-day availability, target excluded
            a, b = self._fam_start[p], self._fam_start[p + 1]
            hi = a + np.searchsorted(self._fam_day[a:b], d, side="left")
            fam = np.unique(self._fam_spec[a:hi]); fam = fam[elig[fam]]
            sp_t = self._specialty_asof([t], d)[0]
            same = all_s[elig & (sp_all == sp_t)] if sp_t > 0 else np.array([], np.int64)
            pool = all_s[elig]
            chosen = []
            f = self._neg_rng.choice(fam, size=min(k1, len(fam)), replace=False) if len(fam) else np.array([], np.int64); chosen.append(f)
            s2 = np.setdiff1d(same, f); s2 = self._neg_rng.choice(s2, size=min(k2, len(s2)), replace=False) if len(s2) else np.array([], np.int64); chosen.append(s2)
            got = np.concatenate(chosen); rest = np.setdiff1d(pool, got); need = K - len(got)
            if len(rest) >= need:
                u = self._neg_rng.choice(rest, size=need, replace=False)
            else:                                                   # tiny early pools: allow repeats, count it
                self._neg_short += 1; u = self._neg_rng.choice(rest, size=need, replace=True) if len(rest) else np.repeat(got[:1], need)
            chosen.append(u)
            out[i, 0] = t; out[i, 1:] = np.concatenate(chosen)
        return out

    def sample_pool_negatives(self, query_enc: np.ndarray, cutoff: int, K: int = 20):
        """(Q, K+1): target then K alternatives uniformly from {same specialty as target (as of query day)} ∩
        {not seen by this patient before the query day} ∩ {available before the query day}. Small pools sample with replacement."""
        mi = self.mi
        if not hasattr(self, "_neg_rng"): self._neg_rng = np.random.default_rng(self.neg_seed)
        all_s = np.arange(mi.n_specialist); sp_all = mi.specialty_asof(all_s, cutoff)
        out = np.empty((len(query_enc), K + 1), np.int64); self._pool_sizes = getattr(self, "_pool_sizes", [])
        for i, q in enumerate(query_enc):
            q = int(q); t = int(mi.enc_target[q]); d = int(mi.enc_day[q]); pat = int(mi.enc_patient[q])
            sp_t = mi.specialty_asof([t], d)[0]
            elig = (mi.spec_first_seen < d) & (sp_all == sp_t) if sp_t > 0 else (mi.spec_first_seen < d)
            elig[t] = False
            cand = all_s[elig]
            if len(cand):
                cand = cand[mi.ps_asof.lookup(pat * mi.n_specialist + cand, d) < 0]        # unseen by patient
            if len(cand) == 0:                                                              # no same-specialty unseen alternative: fall back to unseen pool
                self._pool_fallbacks = getattr(self, "_pool_fallbacks", 0) + 1
                cand = all_s[(mi.spec_first_seen < d) & (all_s != t)]
                cand = cand[mi.ps_asof.lookup(pat * mi.n_specialist + cand, d) < 0]
            self._pool_sizes.append(len(cand))
            out[i, 0] = t; out[i, 1:] = self._neg_rng.choice(cand, size=K, replace=len(cand) < K)
        return out

    def bucket_loss(self, cutoff: int, batch_enc: np.ndarray, query_enc: np.ndarray):
        mi, dev = self.mi, self.device
        if self.train_cohort == "firstobs":
            keep = mi.ps_asof.lookup(mi.enc_patient[query_enc] * mi.n_specialist + mi.enc_target[query_enc], mi.enc_day[query_enc]) < 0
            query_enc = query_enc[keep]
            if len(query_enc) == 0:
                return torch.zeros((), device=dev, requires_grad=True), 0
        if self.neg_policy == "pool":
            spec = self.sample_pool_negatives(query_enc, cutoff); enc = np.repeat(query_enc[:, None], spec.shape[1], axis=1)
            pf_np = mi.pair_features(enc.reshape(-1), spec.reshape(-1))
        elif self.neg_policy == "hard":
            if not hasattr(self, "_fam_start"): self._prep_hard_negatives()
            spec = self.sample_hard_negatives(query_enc, cutoff); enc = np.repeat(query_enc[:, None], spec.shape[1], axis=1)
            pf_np = mi.pair_features(enc.reshape(-1), spec.reshape(-1))
        else:
            rows = (self.tc_rows_of(query_enc)[:, None] + np.arange(21)[None, :]).reshape(-1)
            enc = mi.tc_enc[rows].reshape(-1, 21); spec = mi.tc_spec[rows].reshape(-1, 21)
            assert (enc[:, 0] == query_enc).all() and (mi.tc_label[rows].reshape(-1, 21)[:, 0] == 1).all()
            pf_np = self.train_pair_feat[rows]
        if getattr(self, "history_dropout", 0.0) > 0 and self.model.training:   # STEP11 B5 C1: PCP-history dropout (training only)
            pf_np = self._drop_history(pf_np, len(query_enc), cutoff)
        batch = mi.graph_batch(cutoff, batch_enc, device=dev)          # identical inputs for both arms
        h = self.model.encode(batch)
        e = torch.as_tensor(enc.reshape(-1), device=dev); s = torch.as_tensor(spec.reshape(-1), device=dev)
        pf = torch.as_tensor(pf_np, device=dev)
        kw = {}
        if self.model.cfg.tab_dim or self.model.cfg.table_offset:          # STEP12d: rows of the GNN's own 21-row bundles only
            assert self.neg_policy == "uniform" and self.train_cohort == "all", "STEP12d extras are aligned with the uniform 21-row bundles"
            if self.model.cfg.tab_dim: kw["tab"] = torch.as_tensor(self.extra_train_tab[rows], device=dev)
            if self.model.cfg.table_offset: kw["offset"] = torch.as_tensor(self.extra_train_off[rows], device=dev)
        scores = self.model.score(h, e, s, pf, **kw).view(len(query_enc), -1)
        target = torch.zeros(scores.shape[0], dtype=torch.long, device=dev)
        loss_kind = getattr(self, "loss", "softmax")
        if loss_kind != "softmax":                          # STEP12a §2.A objectives (column 0 = target)
            assert getattr(self, "pos_weight_alpha", 0.0) == 0, "pos-weighting is defined for the softmax loss only"
            return step12_loss(scores, loss_kind), scores.shape[0]
        if getattr(self, "pos_weight_alpha", 0.0) > 0:      # A10 Option B variant 2: down-weight referrals to busy specialists
            w = torch.as_tensor(self._pos_weights(query_enc), device=dev, dtype=scores.dtype)
            return (F.cross_entropy(scores, target, reduction="none") * w).sum() / w.sum(), scores.shape[0]
        return F.cross_entropy(scores, target), scores.shape[0]

    def train_epoch(self, epoch, rng):
        self._epoch = int(epoch)                                        # used by the deterministic C1 dropout mask
        return super().train_epoch(epoch, rng)

    def _drop_history(self, pf_np, n_query, cutoff):
        """C1: for each training query, with probability history_dropout, set the PCP–specialist referral-history pair features
        of all its candidates to the absent encoding (values 0, missing flags 1; same encoding as the 10.4 reliance diagnostic).
        The mask is a deterministic function of (seed, epoch, bucket cutoff), so resumed runs reproduce it without extra state."""
        cols = self.mi.pair_feature_blocks["referral_history"]
        rng = np.random.default_rng([int(self.tcfg.seed), int(getattr(self, "_epoch", 0)), int(cutoff), 7919])
        drop = rng.random(n_query) < self.history_dropout
        if not drop.any(): return pf_np
        pf = np.array(pf_np, copy=True).reshape(n_query, -1, pf_np.shape[-1])
        for c in cols: pf[drop, :, c] = 0.0
        for c in cols:
            if c in (1, 3, 5, 8, 10, 12) and c + 1 in cols: pf[drop, :, c + 1] = 1.0
        return pf.reshape(pf_np.shape)

    def _pos_weights(self, query_enc):
        """(recent 365-day load of the referred specialist, strictly before the referral day, + 1) ** −alpha, normalised so the
        mean weight over all training queries is 1."""
        if getattr(self, "_pw_norm", None) is None:
            from .load import RecentLoad
            self._rl = RecentLoad(self.mi); q = self.mi.train_query_enc
            self._pw_norm = float(np.mean((self._rl.count(self.mi.enc_target[q], self.mi.enc_day[q]) + 1.0) ** -self.pos_weight_alpha))
        q = np.asarray(query_enc)
        return (self._rl.count(self.mi.enc_target[q], self.mi.enc_day[q]) + 1.0) ** -self.pos_weight_alpha / self._pw_norm

    val_pool = "all"

    def validate(self) -> dict:
        subset = self.mi.validation_sample if self.tcfg.validation_subset else None
        t0 = time.time()
        scorer = ModelScorer(self.model, self.mi, self.device)
        if self.model.cfg.tab_dim or self.model.cfg.table_offset:          # STEP12d: extras-aware scorer; offset model on the first-time
            from .heavy_hybrid import ExtraModelScorer                    # selection queries only (their full-pool table scores are precomputed)
            scorer = ExtraModelScorer(self.model, self.mi, self.device, lookup=getattr(self, "val_lookup", None))
            if self.model.cfg.table_offset:
                s_ = np.sort(subset); subset = s_[self.mi.ps_asof.lookup(self.mi.enc_patient[s_] * self.mi.n_specialist + self.mi.enc_target[s_], self.mi.enc_day[s_]) < 0]
        df = evaluate(scorer, self.mi, "validation", query_subset=subset, bucket=self.tcfg.graph_bucket, pool=self.val_pool)
        m = metrics(df); m["val_time_s"] = time.time() - t0; m["val_pool"] = self.val_pool
        for name, mask in (("firstobs", ~df["patient_repeat"]), ("repeat", df["patient_repeat"]), ("unfamiliar", ~df["familiar_pair"])):
            if not mask.any():                                             # STEP12d offset model validates on first-time queries only
                for k_ in ("n", "mrr_eligible", "recall@10_eligible", "recall@20_eligible", "recall@5_eligible", "recall@1_eligible"): m[f"{name}_{k_}"] = float("nan") if k_ != "n" else 0
                continue
            sub = metrics(df[mask]); m[f"{name}_n"] = sub["n_queries"]; m[f"{name}_mrr_eligible"] = sub["mrr_eligible"]
            m[f"{name}_recall@10_eligible"] = sub["recall@10_eligible"]; m[f"{name}_recall@20_eligible"] = sub["recall@20_eligible"]
            m[f"{name}_recall@5_eligible"] = sub["recall@5_eligible"]; m[f"{name}_recall@1_eligible"] = sub["recall@1_eligible"]
        return m

    def save_checkpoint(self, path, epoch, va):
        torch.save({"schema": self.mi.schema_version, "model_state": self.model.state_dict(), "model_config": self.mcfg.to_dict(),
                    "inputs_sha256": self.manifest["inputs_sha256"], "epoch": epoch, "val_metrics": va}, path)


def step12_loss(scores: torch.Tensor, kind: str) -> torch.Tensor:
    """STEP12a §2.A losses on (Q, n) scores with the target in column 0.
    bce: pointwise binary cross-entropy over all n rows (target 1, alternatives 0), mean over rows.
    lambdarank: LightGBM's LambdaRank gradient with one positive (σ = 1, gain 2^l − 1, truncation = n, lambdarank_norm):
        loss = Σ_j w_j · softplus(−(s₊ − s_j)), w_j = |1/log2(1 + r₊) − 1/log2(1 + r_j)| / (0.01 + |s₊ − s_j|) at the current ranks/scores (detached; ideal DCG = 1),
        times the detached per-query norm log2(1 + S)/S with S = 2·Σ_j λ_j (LightGBM's sum_lambdas) and λ_j = w_j·sigmoid(−(s₊ − s_j)); mean over queries.
        d loss / d s₊ = −Σ_j w_j·sigmoid(−(s₊ − s_j)), i.e. LightGBM's λ for the positive."""
    if kind == "bce":
        y = torch.zeros_like(scores); y[:, 0] = 1.0
        return F.binary_cross_entropy_with_logits(scores, y)
    assert kind == "lambdarank", kind
    with torch.no_grad():
        order = torch.argsort(scores, dim=1, descending=True)
        rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(1, scores.shape[1] + 1, device=scores.device).expand_as(order))
        disc = 1.0 / torch.log2(1.0 + rank.float())
        w = (disc[:, :1] - disc[:, 1:]).abs()
        # lambdarank_norm also scales each pair by 1/(0.01 + |s_i − s_j|) unless the query's best and worst scores are equal
        # (LightGBM 4.x RankXENDCG/LambdarankNDCG; reviewer 2026-10-13, B-1)
        spread = scores.max(1, keepdim=True).values != scores.min(1, keepdim=True).values
        w = torch.where(spread, w / (0.01 + (scores[:, :1] - scores[:, 1:]).abs()), w)
        lam = w * torch.sigmoid(-(scores[:, :1] - scores[:, 1:]))
        tot = 2.0 * lam.sum(1, keepdim=True)              # LightGBM accumulates sum_lambdas -= 2·λ_pair over pairs
        norm = torch.where(tot > 0, torch.log2(1.0 + tot) / tot.clamp(min=1e-12), torch.ones_like(tot))
    diff = scores[:, :1] - scores[:, 1:]
    return (norm * w * F.softplus(-diff)).sum(1).mean()


# Trainer.fit saves checkpoints with torch.save(ck, ...) using ModelConfig dict; override fit's checkpoint dict via monkeypatch-free approach:
RESUME_FILE = "resume_state.pt"


def _device_rng_state(device):
    if device == "cuda" and torch.cuda.is_available(): return torch.cuda.get_rng_state()
    if device == "mps" and torch.backends.mps.is_available(): return torch.mps.get_rng_state()
    return None


def _set_device_rng_state(device, st):
    if st is None: return
    if device == "cuda" and torch.cuda.is_available(): torch.cuda.set_rng_state(st)
    elif device == "mps" and torch.backends.mps.is_available(): torch.mps.set_rng_state(st)


def _fit(self) -> dict:
    """Same loop as Trainer.fit, but checkpoints carry the three-node schema and inputs hash.
    Resumable: after every epoch the full training state (model, optimizer, LR scheduler, RNG states, best/patience counters)
    is written atomically to <run_dir>/resume_state.pt; with self.resume=True the loop continues from the last completed epoch.
    A resumed run is statistically equivalent, not bit-identical (MPS/CUDA kernels are nondeterministic anyway)."""
    rng = np.random.default_rng(self.tcfg.seed)
    best, best_epoch, bad = -np.inf, -1, 0
    best_fo, best_fo_epoch = -np.inf, -1
    best_ov, best_ov_epoch = -np.inf, -1
    # One improvement definition for both scheduler and early stopping: metric > best * (1 + rel_threshold)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="max", factor=self.lr_factor, patience=self.lr_patience,
                                                       threshold=self.improve_rel, threshold_mode="rel") if self.lr_factor < 1.0 else None
    improved = lambda new, old: new > old * (1.0 + self.improve_rel) if old > -np.inf else True
    log_path = os.path.join(self.run_dir, "metrics.jsonl")
    state_path = os.path.join(self.run_dir, RESUME_FILE)
    history, start, resumed_at = [], 1, []
    if getattr(self, "resume", False):
        st = torch.load(state_path, map_location="cpu", weights_only=False)   # RNG states must stay CPU ByteTensors; model/optimizer load onto the device
        self.model.load_state_dict(st["model_state"]); self.opt.load_state_dict(st["opt_state"])
        if sched is not None and st["sched_state"] is not None: sched.load_state_dict(st["sched_state"])
        rng.bit_generator.state = st["np_rng"]; torch.set_rng_state(st["torch_rng"]); _set_device_rng_state(self.device, st["device_rng"])
        best, best_epoch, bad, best_fo, best_fo_epoch, best_ov, best_ov_epoch = (st[k] for k in ("best", "best_epoch", "bad", "best_fo", "best_fo_epoch", "best_ov", "best_ov_epoch"))
        start = st["epoch"] + 1; resumed_at = st.get("resumed_at", []) + [st["epoch"]]
        # keep only log lines of completed (saved) epochs; an epoch interrupted after logging is re-run
        lines = [l for l in open(log_path) if l.strip() and json.loads(l)["epoch"] <= st["epoch"]] if os.path.exists(log_path) else []
        open(log_path, "w").writelines(lines); history = [json.loads(l) for l in lines]
        print(f"resumed after epoch {st['epoch']} (best {best_epoch}, patience used {bad})", flush=True)
        if st.get("stopped"): start = self.tcfg.max_epochs + 1
    for epoch in range(start, self.tcfg.max_epochs + 1):
        tr = self.train_epoch(epoch, rng)
        va = self.validate()
        if sched is not None:
            sched.step(va[self.selection_metric])
        tr["lr"] = self.opt.param_groups[0]["lr"]
        rec = {**tr, **{f"val_{k}": v for k, v in va.items()}}
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch:3d} | loss {tr['train_loss']:.4f} | val MRR {va['mrr_eligible']:.4f} R@1 {va['recall@1_eligible']:.3f} R@10 {va['recall@10_eligible']:.3f} R@20 {va['recall@20_eligible']:.3f} | first-obs MRR {va['firstobs_mrr_eligible']:.4f} R@10 {va['firstobs_recall@10_eligible']:.3f} | repeat MRR {va['repeat_mrr_eligible']:.4f} | {tr['epoch_time_s']:.0f}s + val {va['val_time_s']:.0f}s", flush=True)
        # both reference checkpoints are always kept; best.pt follows the protocol's selection metric
        if va["firstobs_mrr_eligible"] > best_fo:
            best_fo, best_fo_epoch = va["firstobs_mrr_eligible"], epoch
            self.save_checkpoint(os.path.join(self.run_dir, "best_firstobs.pt"), epoch, va)
        if va["mrr_eligible"] > best_ov:
            best_ov, best_ov_epoch = va["mrr_eligible"], epoch
            self.save_checkpoint(os.path.join(self.run_dir, "best_overall.pt"), epoch, va)
        primary = va[self.selection_metric]
        if improved(primary, best):
            best, best_epoch, bad = primary, epoch, 0
            self.save_checkpoint(os.path.join(self.run_dir, "best.pt"), epoch, va)
        else:
            bad += 1
        stop = bad >= self.tcfg.patience
        st = {"epoch": epoch, "stopped": stop, "model_state": self.model.state_dict(), "opt_state": self.opt.state_dict(),
              "sched_state": sched.state_dict() if sched is not None else None, "np_rng": rng.bit_generator.state,
              "torch_rng": torch.get_rng_state(), "device_rng": _device_rng_state(self.device), "resumed_at": resumed_at,
              "best": best, "best_epoch": best_epoch, "bad": bad, "best_fo": best_fo, "best_fo_epoch": best_fo_epoch, "best_ov": best_ov, "best_ov_epoch": best_ov_epoch}
        torch.save(st, state_path + ".tmp"); os.replace(state_path + ".tmp", state_path)     # atomic: a kill never leaves a half-written state
        if stop:
            print(f"early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break
    summary = {"selection_metric": self.selection_metric, "best_epoch": best_epoch, "best_val_selection": best,
               "best_firstobs_epoch": best_fo_epoch, "best_val_firstobs_mrr_eligible": best_fo,
               "best_overall_epoch": best_ov_epoch, "best_val_mrr_eligible": best_ov, "epochs_run": len(history),
               "final_lr": self.opt.param_groups[0]["lr"], "improve_rel": self.improve_rel, "lr_factor": self.lr_factor, "lr_patience": self.lr_patience,
               "resumed_after_epochs": resumed_at}
    with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


ThreeNodeTrainer.fit = _fit
ThreeNodeTrainer.selection_metric = "mrr_eligible"     # protocol may set "firstobs_mrr_eligible"; decided in the revision protocol
ThreeNodeTrainer.lr_factor = 1.0                        # <1.0 enables ReduceLROnPlateau on the selection metric
ThreeNodeTrainer.lr_patience = 2
ThreeNodeTrainer.improve_rel = 1e-4                     # relative improvement threshold shared by scheduler and early stopping


def load_three_node_model(run_dir: str, mi: ThreeNodeInputs, device: str, checkpoint: str = "best.pt") -> ThreeNodeGNN:
    ck = torch.load(os.path.join(run_dir, checkpoint), map_location=device)
    assert ck.get("schema") in ("three-node-v1", "three-node-v2", "three-node-v3", "three-node-v4", "three-node-v5", "three-node-v6"), "checkpoint is not a three-node checkpoint"
    assert ck.get("schema") == mi.schema_version, f"checkpoint schema {ck.get('schema')} does not match inputs {mi.schema_version}"
    mcfg = ThreeNodeConfig.from_dict(ck["model_config"])
    model = ThreeNodeGNN(mi.dims(), _blocks_with_demographics(mi, mcfg), mi.pair_feature_blocks, mcfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model
