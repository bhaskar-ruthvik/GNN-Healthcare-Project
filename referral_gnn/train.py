"""Training loop: one HGT forward per graph batch (day or week bucket), sampled softmax over the
21 bundled candidates per eligible query, validation on the fixed subset,
early stopping on eligible-query MRR, best-checkpoint saving.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric

from .data import ModelInputs
from .evaluate import ModelScorer, evaluate, metrics
from .model import ModelConfig, ReferralGNN


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 30
    patience: int = 3
    accumulate_days: int = 1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    validation_subset: bool = True
    max_train_days: Optional[int] = None    # pilot/smoke: limit training batches per epoch
    graph_bucket: str = "day"               # "day" (exact) or "week" (graph cutoff at week start; pair features stay exact)
    eval_log_every: int = 0


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():          # Grace (A100); the laptop falls through to MPS
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash(pkg_dir: str) -> str:
    h = hashlib.sha256()
    for name in sorted(os.listdir(pkg_dir)):
        if name.endswith(".py"):
            with open(os.path.join(pkg_dir, name), "rb") as f:
                h.update(name.encode()); h.update(f.read())
    return h.hexdigest()


class Trainer:
    def __init__(self, mi: ModelInputs, mcfg: ModelConfig, tcfg: TrainConfig, run_dir: str, inputs_path: Optional[str] = None):
        self.mi, self.mcfg, self.tcfg, self.run_dir = mi, mcfg, tcfg, run_dir
        os.makedirs(run_dir, exist_ok=True)
        set_seed(tcfg.seed)
        self.device = pick_device(tcfg.device)
        self.model = ReferralGNN(mi.dims(), mi.encounter_blocks, mi.pair_feature_blocks, mcfg).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        self.device_fallback = None
        # training candidates grouped into graph batches (day or week buckets)
        self.tc_day = mi.enc_day[mi.tc_enc]
        assert (np.diff(self.tc_day) >= 0).all()
        self.train_days = np.unique(self.tc_day)
        self.day_slices = {int(d): (int(a), int(b)) for d, a, b in zip(self.train_days, np.searchsorted(self.tc_day, self.train_days, "left"), np.searchsorted(self.tc_day, self.train_days, "right"))}
        self.train_buckets = mi.buckets("train", tcfg.graph_bucket, query_subset=mi.train_query_enc)
        tc_pos = np.searchsorted(mi.tc_enc[::21], np.arange(mi.n_encounter))        # first tc row-group per encounter
        self.tc_rows_of = lambda encs: (np.searchsorted(mi.tc_enc[::21], encs) * 21)
        t0 = time.time()
        self.train_pair_feat = mi.pair_features(mi.tc_enc, mi.tc_spec)      # (N_train_pairs, 8), computed once
        self.prep_time = time.time() - t0
        self._probe_device()
        self.manifest = {
            "inputs_sha256": sha256_file(inputs_path) if inputs_path else None,
            "code_sha256": code_hash(os.path.dirname(os.path.abspath(__file__))),
            "torch": torch.__version__, "torch_geometric": torch_geometric.__version__, "python": platform.python_version(),
            "device": self.device, "device_fallback": self.device_fallback, "seed": tcfg.seed, "n_parameters": self.model.n_parameters(),
            "model_config": mcfg.to_dict(), "train_config": asdict(tcfg),
            "n_train_days": int(len(self.train_days)), "n_train_buckets": int(len(self.train_buckets)), "n_train_queries": int(len(mi.train_query_enc)),
        }
        with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=1)

    def _fallback_to_cpu(self, where: str, err: Exception):
        """Move model and optimizer state to CPU after a device failure; training continues from the same weights."""
        print(f"WARNING: device {self.device} failed during {where} ({type(err).__name__}: {str(err)[:120]}); falling back to CPU", flush=True)
        self.device_fallback = (self.device_fallback or "") + f"[{where}] {self.device} -> cpu: {type(err).__name__}; "
        self.device = "cpu"
        self.model = self.model.to("cpu")
        state = self.opt.state_dict()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.tcfg.lr, weight_decay=self.tcfg.weight_decay)
        self.opt.load_state_dict(state)                    # load_state_dict casts state tensors to parameter device
        self.opt.zero_grad()
        with open(os.path.join(self.run_dir, "run_manifest.json"), "r+") as f:
            m = json.load(f); m["device"] = "cpu"; m["device_fallback"] = self.device_fallback
            f.seek(0); json.dump(m, f, indent=1); f.truncate()

    def _probe_device(self):
        """Run one real forward/backward on the chosen device; fall back to CPU on any runtime error."""
        if self.device == "cpu":
            return
        try:
            cutoff, batch_enc, query_enc = self.train_buckets[0]
            loss, _ = self.bucket_loss(cutoff, batch_enc, query_enc)
            loss.backward()
            if self.device == "mps":
                torch.mps.synchronize()
            self.opt.zero_grad()
        except Exception as e:  # noqa: BLE001
            self._fallback_to_cpu("startup probe", e)

    # ------------------------------------------------------------------
    def day_loss(self, day: int):
        """Loss for one exact day (cutoff = day)."""
        a, b = self.day_slices[day]
        return self.bucket_loss(day, np.where(self.mi.enc_day == day)[0], self.mi.tc_enc[a:b:21])

    def bucket_loss(self, cutoff: int, batch_enc: np.ndarray, query_enc: np.ndarray):
        """Sampled-softmax loss for the eligible training queries `query_enc`,
        with graph history strictly before `cutoff`."""
        mi, dev = self.mi, self.device
        rows = (self.tc_rows_of(query_enc)[:, None] + np.arange(21)[None, :]).reshape(-1)
        enc = mi.tc_enc[rows].reshape(-1, 21); spec = mi.tc_spec[rows].reshape(-1, 21)
        assert (enc[:, 0] == query_enc).all() and (mi.tc_label[rows].reshape(-1, 21)[:, 0] == 1).all()
        batch = mi.graph_batch(cutoff, batch_enc, device=dev)
        h = self.model.encode(batch)
        e = torch.as_tensor(enc.reshape(-1), device=dev); s = torch.as_tensor(spec.reshape(-1), device=dev)
        pf = torch.as_tensor(self.train_pair_feat[rows], device=dev)
        p = torch.as_tensor(mi.enc_patient[enc.reshape(-1)], device=dev); c = torch.as_tensor(mi.enc_pcp[enc.reshape(-1)], device=dev)
        scores = self.model.score(h, e, s, pf, p, c).view(-1, 21)
        target = torch.zeros(scores.shape[0], dtype=torch.long, device=dev)     # positive is column 0
        return F.cross_entropy(scores, target), scores.shape[0]

    def train_epoch(self, epoch: int, rng: np.random.Generator) -> dict:
        self.model.train()
        order = rng.permutation(len(self.train_buckets))
        if self.tcfg.max_train_days:
            order = order[: self.tcfg.max_train_days]
        tot_loss, tot_q, steps = 0.0, 0, 0
        t0 = time.time()
        self.opt.zero_grad()
        for i, bi in enumerate(order):
            cutoff, batch_enc, query_enc = self.train_buckets[bi]
            try:
                loss, nq = self.bucket_loss(cutoff, batch_enc, query_enc)
                (loss / self.tcfg.accumulate_days).backward()
            except Exception as e:  # noqa: BLE001
                if self.device == "cpu":
                    raise
                self._fallback_to_cpu(f"training epoch {epoch}, step {i}", e)     # gradients of this accumulation window are dropped
                loss, nq = self.bucket_loss(cutoff, batch_enc, query_enc)
                (loss / self.tcfg.accumulate_days).backward()
            tot_loss += loss.item() * nq; tot_q += nq
            if (i + 1) % self.tcfg.accumulate_days == 0 or i == len(order) - 1:
                if self.tcfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
                self.opt.step(); self.opt.zero_grad(); steps += 1
        return {"epoch": epoch, "train_loss": tot_loss / max(tot_q, 1), "train_queries": tot_q, "optimizer_steps": steps, "epoch_time_s": time.time() - t0}

    def validate(self) -> dict:
        subset = self.mi.validation_sample if self.tcfg.validation_subset else None
        t0 = time.time()
        try:
            df = evaluate(ModelScorer(self.model, self.mi, self.device), self.mi, "validation", query_subset=subset, bucket=self.tcfg.graph_bucket, log_every=self.tcfg.eval_log_every)
        except Exception as e:  # noqa: BLE001
            if self.device == "cpu":
                raise
            self._fallback_to_cpu("validation", e)
            df = evaluate(ModelScorer(self.model, self.mi, self.device), self.mi, "validation", query_subset=subset, bucket=self.tcfg.graph_bucket, log_every=self.tcfg.eval_log_every)
        m = metrics(df); m["val_time_s"] = time.time() - t0
        # per-epoch subgroup metrics (protocol: first-observed patient-specialist pairs vs repeat observations)
        for name, mask in (("firstobs", ~df["patient_repeat"]), ("repeat", df["patient_repeat"]), ("unfamiliar", ~df["familiar_pair"])):
            sub = metrics(df[mask])
            m[f"{name}_n"] = sub["n_queries"]; m[f"{name}_mrr_eligible"] = sub["mrr_eligible"]
            m[f"{name}_recall@10_eligible"] = sub["recall@10_eligible"]; m[f"{name}_recall@20_eligible"] = sub["recall@20_eligible"]
        return m

    def fit(self) -> dict:
        rng = np.random.default_rng(self.tcfg.seed)
        best, best_epoch, bad = -np.inf, -1, 0
        best_fo, best_fo_epoch = -np.inf, -1              # secondary checkpoint: first-observed subset MRR
        log_path = os.path.join(self.run_dir, "metrics.jsonl")
        history = []
        for epoch in range(1, self.tcfg.max_epochs + 1):
            tr = self.train_epoch(epoch, rng)
            va = self.validate()
            rec = {**tr, **{f"val_{k}": v for k, v in va.items()}}
            history.append(rec)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"epoch {epoch:3d} | loss {tr['train_loss']:.4f} | val MRR {va['mrr_eligible']:.4f} R@1 {va['recall@1_eligible']:.3f} R@10 {va['recall@10_eligible']:.3f} R@20 {va['recall@20_eligible']:.3f} | first-obs MRR {va['firstobs_mrr_eligible']:.4f} R@10 {va['firstobs_recall@10_eligible']:.3f} | repeat MRR {va['repeat_mrr_eligible']:.4f} | {tr['epoch_time_s']:.0f}s + val {va['val_time_s']:.0f}s", flush=True)
            ck = {"model_state": self.model.state_dict(), "model_config": self.mcfg.to_dict(), "epoch": epoch, "val_metrics": va}
            if va["firstobs_mrr_eligible"] > best_fo:
                best_fo, best_fo_epoch = va["firstobs_mrr_eligible"], epoch
                torch.save(ck, os.path.join(self.run_dir, "best_firstobs.pt"))
            if va["mrr_eligible"] > best:                       # primary selection: overall eligible MRR
                best, best_epoch, bad = va["mrr_eligible"], epoch, 0
                torch.save(ck, os.path.join(self.run_dir, "best.pt"))
            else:
                bad += 1
                if bad >= self.tcfg.patience:
                    print(f"early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                    break
        summary = {"best_epoch": best_epoch, "best_val_mrr_eligible": best, "best_firstobs_epoch": best_fo_epoch, "best_val_firstobs_mrr_eligible": best_fo, "epochs_run": len(history)}
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        return summary

    def load_best(self):
        ck = torch.load(os.path.join(self.run_dir, "best.pt"), map_location=self.device)
        self.model.load_state_dict(ck["model_state"])
        return ck


def load_model(run_dir: str, mi: ModelInputs, device: str, checkpoint: str = "best.pt") -> ReferralGNN:
    ck = torch.load(os.path.join(run_dir, checkpoint), map_location=device)
    model = ReferralGNN(mi.dims(), mi.encounter_blocks, mi.pair_feature_blocks, ModelConfig.from_dict(ck["model_config"])).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model
