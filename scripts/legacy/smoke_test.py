"""Step 4 pre-training checks (see STEP4_PLAN.md section 6, revised after review).

1. Forward on one training day: shapes, no NaN, parameter count.
2. Gradient reaches every parameter after one backward, except a documented
   structurally unreachable set (patient-side HGT outputs, last-layer PCP
   outputs, single-relation relation priors).
3. Withheld-label / future invariance: permuting the referral targets of all
   encounters on or after the query day, and perturbing hidden encounters'
   features, must leave query scores unchanged. Removing a visible historical
   REFERRED_TO edge set must change them (graph is used).
4. Overfit on 2 days for 60 steps: loss must fall well below ln(21).
5. CPU vs MPS agreement on one forward.
6. Timing and memory: batch build, forward, backward, validation subset estimate.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import copy
import math
import os
import sys
import time

import numpy as np
import torch


from referral_gnn.data import ModelInputs, default_bundle_path  # noqa: E402
from referral_gnn.model import ModelConfig, ReferralGNN, REFERRAL_EDGES  # noqa: E402
from referral_gnn.train import TrainConfig, Trainer, pick_device, set_seed  # noqa: E402
from referral_gnn.evaluate import ModelScorer, evaluate, metrics  # noqa: E402

root = str(workspace())
inputs_path = os.path.join(default_bundle_path(root), "model_inputs_PRIVATE.pkl")
mi = ModelInputs.load(inputs_path)
failures = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not cond: failures.append(name)

device = pick_device("auto")
print("device:", device)
set_seed(0)
run_dir = os.path.join(root, "runs", "_smoke")
tr = Trainer(mi, ModelConfig(), TrainConfig(device=device, seed=0), run_dir, inputs_path)
model = tr.model
print("parameters:", f"{model.n_parameters():,}", "| train pair-feature precompute", f"{tr.prep_time:.1f}s", "| device after probe:", tr.device, tr.device_fallback or "")

# ---- 1. forward on a training day ------------------------------------------------
day = int(tr.train_days[len(tr.train_days) // 2])
t0 = time.time(); batch = mi.day_batch(day, device=device); t_build = time.time() - t0
model.eval()
with torch.no_grad():
    t0 = time.time(); h = model.encode(batch); (h["encounter"].sum().item()); t_fwd = time.time() - t0
check("forward shapes", all(h[k].shape == (n, 64) for k, n in (("patient", mi.n_patient), ("pcp", mi.n_pcp), ("encounter", mi.n_encounter), ("specialist", mi.n_specialist))))
check("no NaN in embeddings", all(torch.isfinite(v).all().item() for v in h.values()))

# ---- 2. gradients reach every parameter -----------------------------------------------
model.train(); tr.opt.zero_grad()
loss, nq = tr.day_loss(day)
t0 = time.time(); loss.backward(); t_bwd = time.time() - t0
no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum().item() == 0]
# Structurally unreachable parameters with the confirmed design (2 layers, encounter-only scorer):
#  - patient node outputs of every layer (patient info reaches encounters only via raw input -> pcp (L1) -> encounter (L2));
#  - PCP outputs of the last layer (pcp_L2 feeds nothing scored);
#  - relation priors p_rel: a per-relation constant cancels in the per-destination softmax whenever a destination
#    receives a single relation (specialist, patient, query encounters); only historical encounters in L1 mix two relations.
import re
L = model.cfg.layers - 1
structural = re.compile(rf"^(convs\.\d+\.(out_lin\.lins|skip)\.patient|norms\.\d+\.patient|convs\.{L}\.kqv_lin\.lins\.patient|convs\.{L}\.(out_lin\.lins|skip)\.pcp|norms\.{L}\.pcp|convs\.\d+\.p_rel\.)")
unexpected = [n for n in no_grad if not structural.match(n)]
check("gradient reaches every non-structural parameter", len(unexpected) == 0, f"unexpected={unexpected[:5]}; structurally unused={len(no_grad)}")
check("initial loss near ln(21)", abs(loss.item() - math.log(21)) < 0.7, f"loss={loss.item():.3f}, ln21={math.log(21):.3f}")
tr.opt.zero_grad()

# ---- 3. withheld-label / future invariance -------------------------------------------
model.eval()
a, b = tr.day_slices[day]
enc = mi.tc_enc[a:b].reshape(-1, 21); spec = mi.tc_spec[a:b].reshape(-1, 21)
def day_scores(m, mi_):
    with torch.no_grad():
        bt = mi_.day_batch(day, device=device)
        hh = m.encode(bt)
        e = torch.as_tensor(enc.reshape(-1), device=device); s = torch.as_tensor(spec.reshape(-1), device=device)
        pf = torch.as_tensor(mi_.pair_features(enc.reshape(-1), spec.reshape(-1)), device=device)
        p = torch.as_tensor(mi_.enc_patient[enc.reshape(-1)], device=device); c = torch.as_tensor(mi_.enc_pcp[enc.reshape(-1)], device=device)
        return m.score(hh, e, s, pf, p, c).cpu().numpy()
base = day_scores(model, mi)
# (a) permute targets of every encounter on/after the query day
mi2 = copy.copy(mi)
fut = mi.enc_day >= day
rng = np.random.default_rng(1)
tgt = mi.enc_target.copy(); tgt[fut] = rng.permutation(tgt[fut]); mi2.enc_target = tgt
check("scores invariant to same-day/future referral targets", np.allclose(base, day_scores(model, mi2), atol=1e-5))
# (b) perturb node features of hidden encounters (day > query day)
mi3 = copy.copy(mi)
mi3.enc_patient_num = mi.enc_patient_num.copy(); mi3.enc_patient_num[mi.enc_day > day] += 5.0
mi3.enc_dx = mi.enc_dx.copy(); mi3.enc_dx[mi.enc_day > day] = 1
check("scores invariant to future encounters' features", np.allclose(base, day_scores(model, mi3), atol=1e-5))
# (c) removing visible referral edges must change scores
cfg_no_edges = copy.copy(model.cfg); cfg_no_edges.no_referral_edges = True
m2 = copy.deepcopy(model); m2.cfg = cfg_no_edges
check("scores depend on historical referral edges", not np.allclose(base, day_scores(m2, mi), atol=1e-4))
# (d) scores depend on pair features
mi4 = copy.copy(mi); mi4.pair_features = lambda e_, s_, day=None: np.zeros((len(np.asarray(e_)), 8), np.float32)
check("scores depend on pair features", not np.allclose(base, day_scores(model, mi4), atol=1e-4))

# ---- 3b. bucket policies ------------------------------------------------------------------
for pol in ("day", "week"):
    ok = True; covered = []
    for split in ("train", "validation", "test"):
        bks = mi.buckets(split, pol)
        for cutoff, batch_enc, q in bks:
            ok &= (mi.enc_day[batch_enc] >= cutoff).all() and (mi.enc_day[q] >= cutoff).all() and np.isin(q, batch_enc).all()
            ok &= (mi.enc_day[batch_enc] < cutoff + (7 if pol == "week" else 1)).all()
            ok &= cutoff >= mi.split_days[split][0]
            covered.append(q)
        allq = np.concatenate([q for _, _, q in bks]); ok &= len(allq) == len(mi.query_enc[split]) and len(np.unique(allq)) == len(allq)
    n_train = len(mi.buckets("train", pol, query_subset=mi.train_query_enc))
    check(f"{pol} buckets: cover every query once, cutoff <= all batch days, within window", bool(ok), f"train buckets={n_train}")
wk = mi.buckets("train", "week", query_subset=mi.train_query_enc)
cut, benc, qenc = wk[len(wk) // 2]
bt_w = mi.graph_batch(cut, benc, device=device)
ref = bt_w["edge_index"][("encounter", "referred_to", "specialist")].cpu().numpy()
check("week batch: no referral edge on/after cutoff; batch encounters have none", (mi.enc_day[ref[0]] < cut).all() and not np.isin(benc, ref[0]).any())
sb_first = mi.seen_by_first_day[np.isin(mi.seen_by_src * mi.n_pcp + mi.seen_by_dst, (bt_w["edge_index"][("patient", "seen_by", "pcp")][0] * mi.n_pcp + bt_w["edge_index"][("patient", "seen_by", "pcp")][1]).cpu().numpy())]
check("week batch: seen_by edges all first seen before cutoff", (sb_first < cut).all())

# ---- 3c. future-context invariance (reviewer-requested) ----------------------------------
# Within a week bucket, queries on the bucket's first day must be unaffected by every other batch encounter
# (later days AND same day): their targets, their features, and their very presence.
def bucket_scores(m, mi_, cutoff, batch_enc, qs):
    with torch.no_grad():
        bt = mi_.graph_batch(cutoff, batch_enc, device=device); hh = m.encode(bt)
        rows = (tr.tc_rows_of(qs)[:, None] + np.arange(21)[None, :]).reshape(-1)
        e = torch.as_tensor(mi.tc_enc[rows], device=device); sp = torch.as_tensor(mi.tc_spec[rows], device=device)
        pf = torch.as_tensor(mi.pair_features(mi.tc_enc[rows], mi.tc_spec[rows]), device=device)
        p = torch.as_tensor(mi.enc_patient[mi.tc_enc[rows]], device=device); c = torch.as_tensor(mi.enc_pcp[mi.tc_enc[rows]], device=device)
        return m.score(hh, e, sp, pf, p, c).cpu().numpy()
wk_ok = True
for cut_, benc_, qenc_ in (wk[len(wk) // 3], wk[2 * len(wk) // 3]):
    first_day_q = qenc_[mi.enc_day[qenc_] == mi.enc_day[qenc_].min()]
    others = benc_[~np.isin(benc_, first_day_q)]
    base_w = bucket_scores(model, mi, cut_, benc_, first_day_q)
    # (a) permute targets of all other batch encounters
    mi5 = copy.copy(mi); tg = mi.enc_target.copy(); tg[others] = rng.permutation(tg[others]); mi5.enc_target = tg
    wk_ok &= np.allclose(base_w, bucket_scores(model, mi5, cut_, benc_, first_day_q), atol=1e-5)
    # (b) perturb features of all other batch encounters (incl. patient-history block)
    mi6 = copy.copy(mi); mi6._enc_static_cache = mi._encounter_static().copy(); mi6._enc_static_cache[others] += 3.0
    mi6.enc_dx = mi.enc_dx.copy(); mi6.enc_dx[others] = 1
    wk_ok &= np.allclose(base_w, bucket_scores(model, mi6, cut_, benc_, first_day_q), atol=1e-5)
    # (c) drop all other batch encounters from the batch entirely
    wk_ok &= np.allclose(base_w, bucket_scores(model, mi, cut_, first_day_q, first_day_q), atol=1e-5)
    # (d) query-isolated: each query alone in the batch
    solo = np.concatenate([bucket_scores(model, mi, cut_, np.array([q]), np.array([q])) for q in first_day_q[:5]])
    wk_ok &= np.allclose(base_w[: 21 * min(5, len(first_day_q))], solo, atol=1e-5)
check("week bucket: first-day queries invariant to other batch encounters (targets, features, presence, isolation)", bool(wk_ok))
# same property on a day batch: other same-day encounters have no effect
q_day_all = np.where(mi.enc_day == day)[0]
qs = mi.tc_enc[a:b:21][:5]
base_d = bucket_scores(model, mi, day, q_day_all, qs)
solo_d = np.concatenate([bucket_scores(model, mi, day, np.array([q]), np.array([q])) for q in qs])
check("day batch: queries invariant to other same-day encounters (query-isolated equivalence)", np.allclose(base_d, solo_d, atol=1e-5))
check("batch encounters are receive-only (no reverse has_encounter edge)",
      not np.isin(benc, bt_w["edge_index"][("encounter", "rev_has_encounter", "pcp")][0].cpu().numpy()).any())

# ---- 3d. evaluation-level isolation equivalence at the SAME cutoff (week and day) --------------
sub = mi.validation_sample[:40]
for pol in ("week", "day"):
    full = evaluate(ModelScorer(model, mi, device), mi, "validation", query_subset=sub, bucket=pol).set_index("enc")
    iso_q = evaluate(ModelScorer(model, mi, device), mi, "validation", query_subset=sub, bucket=pol, isolate="query").set_index("enc")
    iso_p = evaluate(ModelScorer(model, mi, device), mi, "validation", query_subset=sub, bucket=pol, isolate="patient").set_index("enc")
    same = (full["rank"].fillna(-1) == iso_q.loc[full.index, "rank"].fillna(-1)).all() and (full["rank"].fillna(-1) == iso_p.loc[full.index, "rank"].fillna(-1)).all()
    check(f"{pol} cutoff: full-batch ranks == query-isolated == patient-isolated ranks (same cutoff)", bool(same))
# week vs day cutoffs are different approximations and need not agree; report how often ranks differ for the untrained model
full_w = evaluate(ModelScorer(model, mi, device), mi, "validation", query_subset=sub, bucket="week").set_index("enc")
full_d = evaluate(ModelScorer(model, mi, device), mi, "validation", query_subset=sub, bucket="day").set_index("enc")
print(f"   info: week vs day cutoff rank differs for {(full_w['rank'] != full_d.loc[full_w.index, 'rank']).mean():.0%} of 40 queries (untrained model; approximation to assess in step 5)")

# ---- 3e. runtime CPU fallback unit test ----------------------------------------------------------
if device != "cpu":
    tr_fb = Trainer(mi, ModelConfig(), TrainConfig(device=device, seed=0), run_dir, None)
    cutoff0, benc0, qenc0 = tr_fb.train_buckets[0]
    l0, _ = tr_fb.bucket_loss(cutoff0, benc0, qenc0); l0.backward(); tr_fb.opt.step()      # populate optimizer state on device
    w_before = {k: v.detach().cpu().clone() for k, v in tr_fb.model.state_dict().items()}
    tr_fb._fallback_to_cpu("unit test", RuntimeError("simulated device failure"))
    l1, _ = tr_fb.bucket_loss(cutoff0, benc0, qenc0); l1.backward(); tr_fb.opt.step()
    state_dev = {str(v.device) for st in tr_fb.opt.state.values() for v in st.values() if torch.is_tensor(v)}
    check("runtime fallback: model+optimizer state moved to CPU, weights preserved, step succeeds",
          tr_fb.device == "cpu" and state_dev == {"cpu"} and all(torch.equal(w_before[k], v) for k, v in tr_fb.model.state_dict().items() if k in w_before) is False
          and all(torch.allclose(w_before[k], v, atol=1e-2) for k, v in tr_fb.model.state_dict().items()),
          f"optimizer state devices={state_dev}")

# ---- 4. overfit on 2 days ----------------------------------------------------------------
set_seed(0)
tr2 = Trainer(mi, ModelConfig(dropout=0.0), TrainConfig(device=device, seed=0, lr=3e-3), run_dir, None)
days2 = [int(tr2.train_days[100]), int(tr2.train_days[700])]
losses = []
for step in range(60):
    tr2.model.train(); tr2.opt.zero_grad()
    l = sum(tr2.day_loss(d)[0] for d in days2) / 2
    l.backward(); tr2.opt.step(); losses.append(l.item())
check("overfit: loss falls well below ln(21)", losses[-1] < 0.6 * math.log(21), f"first={losses[0]:.3f} last={losses[-1]:.3f}")

# ---- 5. CPU vs MPS agreement ---------------------------------------------------------------
if device != "cpu":
    m_cpu = copy.deepcopy(model).to("cpu"); m_cpu.eval()
    with torch.no_grad():
        b_cpu = mi.day_batch(day, device="cpu"); h_cpu = m_cpu.encode(b_cpu)
    diff = max((h_cpu[k] - h[k].cpu()).abs().max().item() for k in h)
    check("CPU vs MPS forward agreement", diff < 1e-3, f"max abs diff={diff:.2e}")
else:
    print("skip CPU vs MPS agreement: running on CPU")

# ---- 6. timing / memory -------------------------------------------------------------------------
t0 = time.time()
df = evaluate(ModelScorer(model, mi, device), mi, "validation", query_subset=mi.validation_sample[:200])
t_eval = time.time() - t0
m = metrics(df)
# steady-state step time for both bucket policies (3 reps each, MPS-synchronized)
def step_time(cutoff, benc, qenc):
    ts = []
    for _ in range(3):
        t0 = time.time(); l, _ = tr.bucket_loss(cutoff, benc, qenc); l.backward(); tr.opt.zero_grad()
        if device == "mps": torch.mps.synchronize()
        ts.append(time.time() - t0)
    return min(ts)
tr.model.train()
t_day = step_time(day, np.where(mi.enc_day == day)[0], mi.tc_enc[tr.day_slices[day][0]:tr.day_slices[day][1]:21])
t_week = step_time(cut, benc, qenc)
n_day, n_week = len(tr.train_days), len(wk)
print(f"timing: first-call forward {t_fwd*1000:.0f} ms / backward {t_bwd*1000:.0f} ms (includes warm-up) | steady-state train step: day {t_day:.2f}s, week {t_week:.2f}s")
print(f"        est. epoch: day policy {n_day} steps ≈ {n_day*t_day/60:.0f} min | week policy {n_week} steps ≈ {n_week*t_week/60:.1f} min")
print(f"        200-query full-pool eval {t_eval:.1f}s -> est. 3000-query validation {t_eval*15/60:.1f} min, full test (17,641) {t_eval*17641/200/60:.0f} min")
if device == "mps":
    print(f"mps memory: current {torch.mps.current_allocated_memory()/1e9:.2f} GB, driver {torch.mps.driver_allocated_memory()/1e9:.2f} GB")
print("untrained model on 200 validation queries (sanity, should be near random):", {k: round(v, 4) for k, v in m.items() if k in ("mrr_eligible", "recall@10_eligible", "median_rank_eligible", "target_availability")})

print("\nALL PASSED" if not failures else f"\nFAILED: {failures}")
sys.exit(1 if failures else 0)
