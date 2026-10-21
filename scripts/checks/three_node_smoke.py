"""Model-level checks for the three-node model (ARCHITECTURE.md "Required verification before training").

Runs on the versioned inputs in prepared_data/three_node_v1. No training beyond a two-bucket overfit check.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import copy
import math
import os
import re
import sys
import time

import numpy as np
import torch


from referral_gnn.three_node_data import ThreeNodeInputs, TEMPORAL_EDGE  # noqa: E402
from referral_gnn.three_node_model import ThreeNodeConfig, ThreeNodeGNN, NODE_TYPES  # noqa: E402
from referral_gnn.three_node_train import ThreeNodeTrainer, load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.train import TrainConfig, pick_device, set_seed  # noqa: E402
from referral_gnn.evaluate import ModelScorer, evaluate, metrics  # noqa: E402

root = str(workspace())
inputs_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "prepared_data", "three_node_v1", "model_inputs_PRIVATE.pkl")
mi = load_three_node_inputs(inputs_path)
device = pick_device("auto")
failures = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not cond: failures.append(name)

print("device:", device, "| dims:", mi.dims())
run_dir = os.path.join(root, "runs", "_three_node_smoke")
set_seed(0)
tr = ThreeNodeTrainer(mi, ThreeNodeConfig(temporal_edges=True), TrainConfig(device=device, seed=0, graph_bucket="week"), run_dir, inputs_path)
model = tr.model
print("parameters:", f"{model.n_parameters():,}", "| device after probe:", tr.device, tr.device_fallback or "")

# ---- 1. forward: exactly three node types, shapes, finite ------------------------------------
wk = tr.train_buckets
cut, benc, qenc = wk[len(wk) // 2]
bt = mi.graph_batch(cut, benc, device=device)
check("batch exposes exactly three node types", tuple(bt["node_types"]) == tuple(NODE_TYPES) and not any(k.startswith("patient") for k in bt["x"]))
if hasattr(mi, "demographic_cat_cols"):
    m_mask = ThreeNodeGNN(mi.dims(), {**mi.encounter_blocks, "demographic_cat_cols": mi.demographic_cat_cols, "demographic_flag_cols": mi.demographic_flag_cols}, mi.pair_feature_blocks, ThreeNodeConfig(mask_demographics=True)).to(device)
    m_mask.load_state_dict(model.state_dict()); m_mask.eval()
    with torch.no_grad():
        h_m = m_mask.encode(bt)
    # masking must change encounter embeddings (demographics were in use) and must not touch pcp/specialist encoders
    check("mask_demographics: alters encounter encodings, leaves provider encoders unchanged", not torch.allclose(h_m["encounter"], model.encode(bt)["encounter"], atol=1e-6))
check("no query target / reverse-target edges in batch",
      not np.isin(benc, bt["edge_index"][("encounter", "referred_to", "specialist")][0].cpu().numpy()).any()
      and not np.isin(benc, bt["edge_index"][("specialist", "rev_referred_to", "encounter")][1].cpu().numpy()).any()
      and not np.isin(benc, bt["edge_index"][("encounter", "rev_has_encounter", "pcp")][0].cpu().numpy()).any())
te = bt["edge_index"][TEMPORAL_EDGE].cpu().numpy()
check("PRECEDES edges: same patient, strictly earlier source, query sources before cutoff",
      (mi.enc_patient[te[0]] == mi.enc_patient[te[1]]).all() and (mi.enc_day[te[0]] < mi.enc_day[te[1]]).all()
      and (mi.enc_day[te[0][np.isin(te[1], benc)]] < cut).all())
model.eval()
with torch.no_grad():
    h = model.encode(bt)
check("forward shapes / finite", set(h) == set(NODE_TYPES) and all(v.shape == (n, 64) for v, n in ((h["pcp"], mi.n_pcp), (h["encounter"], mi.n_encounter), (h["specialist"], mi.n_specialist))) and all(torch.isfinite(v).all().item() for v in h.values()))

# ---- 2. gradients: structural set --------------------------------------------------------------
model.train(); tr.opt.zero_grad()
loss, nq = tr.bucket_loss(cut, benc, qenc); loss.backward()
no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum().item() == 0]
L = model.cfg.layers - 1
# Structural: last-layer PCP outputs (pcp_L2 feeds nothing scored); relation priors p_rel for destinations
# receiving a single relation (pcp <- rev_has_encounter only; specialist <- referred_to only); and in the LAST
# layer the rev_referred_to prior into encounters, because only query encounters are scored and queries have
# no reverse referral edge (historical encounters' last-layer outputs feed nothing).
structural = re.compile(rf"^(convs\.{L}\.(out_lin\.lins|skip)\.pcp|norms\.{L}\.pcp|convs\.\d+\.p_rel\.(encounter__rev_has_encounter__pcp|encounter__referred_to__specialist)|convs\.{L}\.p_rel\.specialist__rev_referred_to__encounter)")
unexpected = [n for n in no_grad if not structural.match(n)]
check("gradient reaches every non-structural parameter", len(unexpected) == 0, f"unexpected={unexpected[:6]}; structurally unused={len(no_grad)}")
check("gradient reaches PRECEDES relation parameters when temporal edges on", any("precedes" in n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum().item() > 0))
check("initial loss near ln(21)", abs(loss.item() - math.log(21)) < 0.7, f"loss={loss.item():.3f}")
tr.opt.zero_grad()
# temporal-off arm: precedes params must be unused, everything else as above
m_off = ThreeNodeGNN(mi.dims(), mi.encounter_blocks, mi.pair_feature_blocks, ThreeNodeConfig(temporal_edges=False)).to(device)
m_off.load_state_dict(model.state_dict()); m_off.train()
tr_off = copy.copy(tr); tr_off.model = m_off
l2, _ = tr_off.bucket_loss(cut, benc, qenc); l2.backward()
prec_used = [n for n, p in m_off.named_parameters() if "precedes" in n and p.grad is not None and p.grad.abs().sum().item() > 0]
check("temporal-off arm: PRECEDES parameters receive no gradient", len(prec_used) == 0)
check("both arms have identical parameter count and input tensors", m_off.n_parameters() == model.n_parameters())

# ---- 3. invariance at a fixed cutoff (eval mode) -------------------------------------------------
model.eval()
def scores_for(m, mi_, cutoff, batch_enc, qs):
    with torch.no_grad():
        b = mi_.graph_batch(cutoff, batch_enc, device=device); hh = m.encode(b)
        rows = (tr.tc_rows_of(qs)[:, None] + np.arange(21)[None, :]).reshape(-1)
        e = torch.as_tensor(mi.tc_enc[rows], device=device); sp = torch.as_tensor(mi.tc_spec[rows], device=device)
        pf = torch.as_tensor(mi.pair_features(mi.tc_enc[rows], mi.tc_spec[rows]), device=device)
        return m.score(hh, e, sp, pf).cpu().numpy()
rng = np.random.default_rng(1)
inv_ok = True; iso_ok = True; ties_seen = 0
for cut_, benc_, qenc_ in (wk[len(wk) // 3], wk[2 * len(wk) // 3], wk[-2]):
    q0 = qenc_[:8]
    base = scores_for(model, mi, cut_, benc_, q0)
    fut = mi.enc_day >= cut_
    # (a) permute targets of all on/after-cutoff encounters
    mi2 = copy.copy(mi); tg = mi.enc_target.copy(); tg[fut] = rng.permutation(tg[fut]); mi2.enc_target = tg
    inv_ok &= np.allclose(base, scores_for(model, mi2, cut_, benc_, q0), atol=1e-5)
    # (b) perturb features of all on/after-cutoff encounters other than q0 (incl. other batch queries)
    others = np.setdiff1d(np.where(fut)[0], q0)
    mi3 = copy.copy(mi); mi3._enc_static_cache = mi._encounter_static().copy(); mi3._enc_static_cache[others] += 3.0
    mi3.enc_dx = mi.enc_dx.copy(); mi3.enc_dx[others] = 1; mi3.enc_dx_mask = mi.enc_dx_mask.copy(); mi3.enc_dx_mask[others, 0] = True
    inv_ok &= np.allclose(base, scores_for(model, mi3, cut_, benc_, q0), atol=1e-5)
    # (c) remove all other future records from the batch; (d) query alone
    inv_ok &= np.allclose(base, scores_for(model, mi, cut_, q0, q0), atol=1e-5)
    solo = np.concatenate([scores_for(model, mi, cut_, np.array([q]), np.array([q])) for q in q0])
    iso_ok &= np.allclose(base, solo, atol=1e-5)
    # same-day ties present among temporal sources?
    _, gaps = mi.temporal_context(cut_, q0)
    for q in q0:
        days, groups = mi.patient_date_groups[int(mi.enc_patient[q])]
        prev = [d for d in days if d < cut_]
        if prev and len(groups[prev[-1]]) > 1: ties_seen += 1
check("scores invariant to future targets, future/other-batch features and their presence (fixed cutoff)", bool(inv_ok))
check("query-alone == batched at the same cutoff", bool(iso_ok), f"tie cases among probed queries: {ties_seen}")
# day cutoff too
d0 = int(tr.train_days[len(tr.train_days) // 2]); qd = mi.tc_enc[tr.day_slices[d0][0]:tr.day_slices[d0][1]:21][:5]
base_d = scores_for(model, mi, d0, np.where(mi.enc_day == d0)[0], qd)
check("day cutoff: query-alone == batched", np.allclose(base_d, np.concatenate([scores_for(model, mi, d0, np.array([q]), np.array([q])) for q in qd]), atol=1e-5))

# ---- 4. temporal path is active ------------------------------------------------------------------------------
# pick queries WITH predecessors and perturb ONLY their predecessor encounters' features
cut_, benc_, qenc_ = wk[2 * len(wk) // 3]
te_all, _ = mi.temporal_context(cut_, qenc_)
has_pred = np.unique(te_all[1][np.isin(te_all[1], qenc_)])[:8]
no_pred = np.setdiff1d(qenc_, te_all[1])[:8]
preds = np.unique(te_all[0][np.isin(te_all[1], has_pred)])
base_p = scores_for(model, mi, cut_, benc_, has_pred)
mi4 = copy.copy(mi); mi4._enc_static_cache = mi._encounter_static().copy(); mi4._enc_static_cache[preds] += 3.0
pert_on = scores_for(model, mi4, cut_, benc_, has_pred)
m_off.eval()
base_off = scores_for(m_off, mi, cut_, benc_, has_pred); pert_off = scores_for(m_off, mi4, cut_, benc_, has_pred)
delta_on = np.abs(pert_on - base_p).mean(); delta_off = np.abs(pert_off - base_off).mean()
check("perturbing a query's predecessors changes its scores (temporal path active)", delta_on > 1e-4, f"mean |Δ| on={delta_on:.4f}, off={delta_off:.4f} (off arm: only via PCP/specialist paths)")
check("temporal-on vs temporal-off arms give different scores", not np.allclose(base_p, base_off, atol=1e-4))
if len(no_pred):
    _, g_np = mi.temporal_context(cut_, no_pred)
    check("no-history queries: missing-gap flag set and no incoming PRECEDES edge", (g_np[no_pred] < 0).all() and not np.isin(no_pred, te_all[1]).any(), f"n={len(no_pred)}")

# ---- 5. new provider case ---------------------------------------------------------------------------------------
# Available specialists always have >=1 earlier row by construction (first seen before cutoff), so the
# realistic "new provider" case is a candidate with missing/ambiguous attribute fields and no pair history.
snap = mi.provider_snapshot(cut_)
avail = np.where(mi.available_specialists(cut_))[0]
miss_cols = [1 + 3 * i for i in range(len(mi.spec_encoder.numeric_cols))]          # missing-flag columns
n_missing = snap["specialist_num"][avail][:, miss_cols].sum(1)
sparse = avail[n_missing >= 3][:50]
q = int(has_pred[0])
with torch.no_grad():
    hh = model.encode(mi.graph_batch(cut_, benc_, device=device))
    cand = np.concatenate([sparse, avail[:50]])
    pf = torch.as_tensor(mi.pair_features(np.full(len(cand), q), cand), device=device)
    sc = model.score(hh, torch.full((len(cand),), q, device=device), torch.as_tensor(cand, device=device), pf)
    unfam = (pf[:, 2] == 1).sum().item()
check("sparse-attribute / unfamiliar candidates get finite scores", torch.isfinite(sc).all().item() and len(sparse) > 0, f"n_sparse={len(sparse)}, n_unfamiliar_pairs={unfam}, all available have >=1 prior row: {bool(snap['specialist_has_history'][avail].all())}")

# ---- 6. overfit two buckets --------------------------------------------------------------------------------------
set_seed(0)
tr2 = ThreeNodeTrainer(mi, ThreeNodeConfig(dropout=0.0), TrainConfig(device=device, seed=0, lr=3e-3, graph_bucket="week"), run_dir, None)
b2 = [wk[40], wk[100]]
losses = []
for step in range(60):
    tr2.model.train(); tr2.opt.zero_grad()
    l = sum(tr2.bucket_loss(*b)[0] for b in b2) / 2
    l.backward(); tr2.opt.step(); losses.append(l.item())
check("overfit: loss falls well below ln(21)", losses[-1] < 0.6 * math.log(21), f"first={losses[0]:.3f} last={losses[-1]:.3f}")

# ---- 7. CPU vs MPS; save/load equivalence -------------------------------------------------------------------------
if device != "cpu":
    m_cpu = copy.deepcopy(model).to("cpu"); m_cpu.eval()
    with torch.no_grad():
        h_cpu = m_cpu.encode(mi.graph_batch(cut, benc, device="cpu"))
    diff = max((h_cpu[k] - h[k].cpu()).abs().max().item() for k in h)
    check("CPU vs MPS forward agreement", diff < 1e-3, f"max abs diff={diff:.2e}")
tr.save_checkpoint(os.path.join(run_dir, "best.pt"), 0, {})
m_re = load_three_node_model(run_dir, mi, device)
check("save/load equivalence", np.allclose(scores_for(model, mi, cut, benc, qenc[:8]), scores_for(m_re, mi, cut, benc, qenc[:8]), atol=1e-6))
try:
    from referral_gnn.train import load_model as load_four_node
    load_four_node(run_dir, mi, device); ok = False
except Exception:
    ok = True
check("old four-node loader rejects a three-node checkpoint/inputs", ok)

# ---- 8. evaluation path + timing ----------------------------------------------------------------------------------
t0 = time.time(); df = evaluate(ModelScorer(model, mi, device), mi, "validation", query_subset=mi.validation_sample[:150], bucket="week"); t_eval = time.time() - t0
check("evaluation runs on three-node inputs; per-query flags present", {"patient_repeat", "familiar_pair"} <= set(df.columns) and len(df) == 150)
def step_time(b):
    ts = []
    for _ in range(3):
        t0 = time.time(); l, _ = tr.bucket_loss(*b); l.backward(); tr.opt.zero_grad()
        if device == "mps": torch.mps.synchronize()
        ts.append(time.time() - t0)
    return min(ts)
tr.model.train()
t_week = step_time(wk[len(wk) // 2])
print(f"timing: week step {t_week:.2f}s -> epoch ≈ {len(wk) * t_week / 60:.1f} min | 150-query eval {t_eval:.1f}s -> 3,000-query validation ≈ {t_eval * 20 / 60:.1f} min")
print("untrained sanity on 150 validation queries:", {k: round(v, 4) for k, v in metrics(df).items() if k in ("mrr_eligible", "recall@10_eligible", "target_availability")})
print("\nALL PASSED" if not failures else f"\nFAILED: {failures}")
sys.exit(1 if failures else 0)
