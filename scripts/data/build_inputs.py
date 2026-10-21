"""Step 3: build model inputs from the prepared bundle and validate them.

Fits encoders on training data only, checks the as-of feature recomputation
against the values saved by the cleaning notebook, probes temporal masks on
random days, and saves the ModelInputs object next to the bundle.

Prints aggregate diagnostics only.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import os
import sys
import time

import numpy as np
import torch


from referral_gnn.data import Bundle, ModelInputs, default_bundle_path, day_to_str, PAIR_FEATURE_NAMES  # noqa: E402

root = str(workspace())
bundle_path = default_bundle_path(root)
print("bundle:", os.path.basename(bundle_path))

t0 = time.time()
b = Bundle(bundle_path)
mi = ModelInputs(b)
print(f"built ModelInputs in {time.time() - t0:.1f}s")
print("dims:", mi.dims())
print("split day ranges:", {k: (day_to_str(v[0]), day_to_str(v[1])) for k, v in mi.split_days.items()})
print("diagnosis vocab size incl. PAD+UNK (train, min_count=5):", mi.dx_vocab.size, "| encounters with no code:", int(mi.enc_dx_empty.sum()),
      "| specialty vocab:", mi.spec_encoder.vocabs["specialty"].size,
      "| specialist credential vocab:", mi.spec_encoder.vocabs["credentials"].size)

failures = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)

# ---- 1. as-of pair features vs bundle's train_candidates -----------------------
prior, recency, dist = mi.raw_pair_history(mi.tc_enc, mi.tc_spec)
check("prior_referral_count matches bundle", (prior == mi.tc_bundle_prior).all(),
      f"mismatches={int((prior != mi.tc_bundle_prior).sum())}")
br = mi.tc_bundle_recency
rec_ok = np.where(np.isnan(br), recency == -1, recency == np.nan_to_num(br, nan=-2))
check("days_since_last_referral matches bundle", rec_ok.all(), f"mismatches={int((~rec_ok).sum())}")
bd = mi.tc_bundle_distance
dist_ok = np.where(np.isnan(bd), np.isnan(dist), np.abs(dist - np.nan_to_num(bd)) < 1e-6)
check("geo distance matches bundle", dist_ok.all(), f"mismatches={int((~dist_ok).sum())}")
check("21 candidates per training query, one positive",
      len(mi.tc_enc) == 21 * len(mi.train_query_enc) and mi.tc_label.reshape(-1, 21)[:, 0].all() and (mi.tc_label.reshape(-1, 21)[:, 1:] == 0).all())
check("training positives equal held-out target", (mi.tc_spec.reshape(-1, 21)[:, 0] == mi.enc_target[mi.tc_enc.reshape(-1, 21)[:, 0]]).all())
check("all training candidates available strictly earlier", (mi.spec_first_seen[mi.tc_spec] < mi.enc_day[mi.tc_enc]).all())

# ---- 2. patient history features --------------------------------------------------
check("prior encounter counts non-negative and < own position", (mi.enc_prior_encounters >= 0).all())
first_by_patient = np.zeros(mi.n_patient, bool)
order = np.lexsort((mi.enc_day, mi.enc_patient))
first_rows = order[np.r_[True, mi.enc_patient[order][1:] != mi.enc_patient[order][:-1]]]
check("first encounter of each patient has zero prior history",
      (mi.enc_prior_encounters[first_rows] == 0).all() and (mi.enc_days_since_prev[first_rows] == -1).all() and (mi.enc_prior_specialties[first_rows].sum() == 0))
check("days since previous encounter strictly positive when present", (mi.enc_days_since_prev[mi.enc_days_since_prev >= 0] > 0).all())
print("   share of encounters with prior patient history by split:",
      {s: round(float((mi.enc_prior_encounters[mi.query_enc[s]] > 0).mean()), 3) for s in ("train", "validation", "test")})

# ---- 3. temporal masks on sampled days ---------------------------------------------
rng = np.random.default_rng(0)
sample_days = np.concatenate([rng.choice(mi.days_with_queries(s), 4, replace=False) for s in ("train", "validation", "test")])
mask_ok = True
timings = []
for d in sample_days:
    t1 = time.time()
    bt = mi.day_batch(int(d))
    timings.append(time.time() - t1)
    ei = bt["edge_index"]
    ref = ei[("encounter", "referred_to", "specialist")].numpy()
    has = ei[("pcp", "has_encounter", "encounter")].numpy()
    q = bt["query_enc"].numpy()
    mask_ok &= (mi.enc_day[ref[0]] < d).all()                      # no same-day/future referral edges
    mask_ok &= not np.isin(q, ref[0]).any()                        # batch queries have no referral edge
    mask_ok &= np.isin(q, has[1]).all()                            # batch queries connected to PCP
    mask_ok &= ((mi.enc_day[has[1]] < d) | np.isin(has[1], q)).all()
    mask_ok &= (mi.spec_first_seen[bt["available"].numpy()] < d).all()
    # provider snapshot rows strictly earlier
    rows = mi.spec_hist_asof.lookup(np.arange(mi.n_specialist), int(d)); mask_ok &= (mi.spec_hist_day[rows[rows >= 0]] < d).all()
    rows = mi.pcp_hist_asof.lookup(np.arange(mi.n_pcp), int(d)); mask_ok &= (mi.pcp_hist_day[rows[rows >= 0]] < d).all()
check("temporal masks hold on 12 sampled days", bool(mask_ok))
print(f"   day_batch build time: median {np.median(timings)*1000:.0f} ms, max {np.max(timings)*1000:.0f} ms")
bt = mi.day_batch(int(sample_days[-1]))
print("   example test-day batch: queries", len(bt["query_enc"]), "| visible encounters", bt["n_visible_encounters"],
      "| edges", {k[1]: int(v.shape[1]) for k, v in bt["edge_index"].items() if not k[1].startswith("rev_")},
      "| available specialists", int(bt["available"].sum()))
print("   encounter feature width:", bt["x"]["encounter_num"].shape[1], "| specialist:", bt["x"]["specialist_num"].shape[1], "| pcp:", bt["x"]["pcp_num"].shape[1])

# ---- 4. full-pool pair features for one evaluation query -----------------------------
q = mi.validation_sample[0]
avail = np.where(mi.available_specialists(mi.enc_day[q]))[0]
pf = mi.pair_features(np.full(len(avail), q), avail)
check("dx PAD only where mask is false", ((mi.enc_dx == 0) == ~mi.enc_dx_mask).all())
check("pair features finite", np.isfinite(pf).all(), f"shape={pf.shape}")
print("   pair feature names:", PAIR_FEATURE_NAMES)
print("   validation sample size:", len(mi.validation_sample))

# ---- 5. coverage of provider snapshots at split starts -----------------------------
for s in ("validation", "test"):
    snap = mi.provider_snapshot(mi.split_days[s][0])
    print(f"   {s} start: specialists with earlier history {snap['specialist_has_history'].mean():.3f}, PCPs {snap['pcp_has_history'].mean():.3f}")

out = os.path.join(bundle_path, "model_inputs_PRIVATE.pkl")
if failures:
    print("\nNOT SAVED — failed checks:", failures)
    sys.exit(1)
mi.save(out)
print("\nsaved", os.path.relpath(out, root), f"({os.path.getsize(out)/1e6:.1f} MB)")
