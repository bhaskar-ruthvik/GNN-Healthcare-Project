"""STEP12d B inputs (torch process; no LightGBM): out-of-fold GNN scores for the training rows, and full-GNN scores for the stacked
LightGBM's selection pool and evaluation sets.
  train_score  every training row (mi.tc_enc order) scored by the fold model that did NOT train on that row's time fold
               (--fold-runs: K runs trained with --exclude-fold 0..K-1, same seed);
  look_*       the full GNN (--gnn-run) on (i) the first-time validation selection queries × every available specialist and (ii) the
               evaluation eligible sets taken from step10_evaluate --dump-scores files (--dumps; validation and test), so the
               stacked LightGBM can be tuned and scored without torch.
Usage: python scripts/train/step12d_gnn_oof.py --seed 42 --gnn-run runs/<v5_unmasked_s42> --fold-runs runs/<f0> runs/<f1> runs/<f2>
         --dumps runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz runs/analysis/step12c/dump/test_candidate_scores_PRIVATE.npz"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, json, os, sys, time
import numpy as np

from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.evaluate import ModelScorer  # noqa: E402
from referral_gnn.train import pick_device  # noqa: E402
from referral_gnn.score_lookup import time_folds, fold_of_tc_rows  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--gnn-run", required=True)
ap.add_argument("--fold-runs", nargs="+", required=True); ap.add_argument("--dumps", nargs="*", default=[]); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
from referral_gnn.paths import resolve_run  # noqa: E402
args.gnn_run = resolve_run(args.gnn_run)
args.fold_runs = [resolve_run(r) for r in args.fold_runs]
mi = load_three_node_inputs("prepared_data/three_node_v5/model_inputs_PRIVATE.pkl"); dev = pick_device("auto"); t0 = time.time()
K = len(args.fold_runs); f = time_folds(mi, K)
for k, r in enumerate(args.fold_runs):                       # each fold run must have excluded exactly fold k
    m = json.load(open(os.path.join(r, "run_manifest.json"))); assert m.get("exclude_fold") == k and m.get("n_folds") == K and int(m["seed"]) == args.seed, (r, m.get("exclude_fold"))
train_score = np.full(len(mi.tc_enc), np.nan, np.float32); rows_of = lambda q: int(np.searchsorted(mi.tc_enc[::21], q)) * 21
for k, r in enumerate(args.fold_runs):
    sc = ModelScorer(load_three_node_model(r, mi, dev), mi, dev); qk = mi.train_query_enc[f == k]
    if args.smoke: qk = qk[:200]
    for cutoff, batch_enc, q_b in mi.buckets("train", "week", query_subset=qk):
        sc.prepare(cutoff, batch_enc)
        for q in q_b:
            r0 = rows_of(q); train_score[r0:r0 + 21] = sc.score(int(q), mi.tc_spec[r0:r0 + 21])
    print(f"fold {k}: scored {len(qk):,} training queries with {os.path.basename(r)} ({time.time() - t0:.0f}s)", flush=True)
assert args.smoke or np.isfinite(train_score).all()
# Reviewer B1: a softmax-trained logit's per-query level is arbitrary (differs by seed and between fold / full models), so the stacked
# trees get the score CENTRED per query: training rows by the mean of the 20 uniform alternatives; selection and evaluation queries by
# the mean over the full available pool (never over the filtered eligible set). Scale check: fold models vs the full model on the
# selection pool (within-query SD ratio, Spearman); outside [0.9, 1.1] the fold's training scores are divided by the ratio.
T21 = train_score.reshape(-1, 21); train_c = (T21 - np.nan_to_num(T21[:, 1:]).mean(1, keepdims=True)).reshape(-1)
full = ModelScorer(load_three_node_model(args.gnn_run, mi, dev), mi, dev)
folds_sc = [ModelScorer(load_three_node_model(r, mi, dev), mi, dev) for r in args.fold_runs]
sel = np.sort(mi.validation_sample); fo = sel[mi.ps_asof.lookup(mi.enc_patient[sel] * mi.n_specialist + mi.enc_target[sel], mi.enc_day[sel]) < 0]
E, S, V = [], [], []; ratios = [[] for _ in folds_sc]; rhos = [[] for _ in folds_sc]
from scipy.stats import spearmanr
for cutoff, batch_enc, q_b in mi.buckets("validation", "week", query_subset=fo):
    full.prepare(cutoff, batch_enc); [f_.prepare(cutoff, batch_enc) for f_ in folds_sc]
    for q in q_b:
        avail = np.where(mi.available_specialists(int(mi.enc_day[q])))[0]; s_ = full.score(int(q), avail)
        E.append(np.full(len(avail), q)); S.append(avail); V.append(s_ - s_.mean())
        for k_, f_ in enumerate(folds_sc):
            fk = f_.score(int(q), avail); ratios[k_].append(np.std(fk) / max(np.std(s_), 1e-9)); rhos[k_].append(spearmanr(fk, s_).statistic)
scale = {}
qfold = fold_of_tc_rows(mi, K)                               # fold per training row (tc order)
for k in range(K):
    r_ = float(np.median(ratios[k])); scale[k] = {"median_sd_ratio_fold_over_full": r_, "median_spearman": float(np.median(rhos[k])), "rescaled": bool(not 0.9 <= r_ <= 1.1)}
    if scale[k]["rescaled"]: train_c[qfold == k] /= r_
print("fold vs full scale check:", scale, flush=True)
for d in args.dumps:                                         # evaluation eligible sets: full GNN scores centred by the full-pool mean
    z_ = np.load(d); split = "test" if "test_" in os.path.basename(d) else "validation"
    n_ = np.diff(z_["offsets"]); encs = z_["enc"]; elig = {int(e): z_["spec"][z_["offsets"][i]:z_["offsets"][i + 1]] for i, e in enumerate(encs)}
    for cutoff, batch_enc, q_b in mi.buckets(split, "week", query_subset=np.sort(encs)):
        full.prepare(cutoff, batch_enc)
        for q in q_b:
            avail = np.where(mi.available_specialists(int(mi.enc_day[q])))[0]; s_ = full.score(int(q), avail); pos = np.searchsorted(avail, elig[int(q)])
            assert (avail[pos] == elig[int(q)]).all(); E.append(np.full(len(pos), q)); S.append(elig[int(q)]); V.append(s_[pos] - s_.mean())
    print(f"evaluation sets from {os.path.basename(d)} scored ({time.time() - t0:.0f}s)", flush=True)
E, S, V = np.concatenate(E), np.concatenate(S), np.concatenate(V).astype(np.float32)
key = E.astype(np.int64) * 100000 + S; _, first = np.unique(key, return_index=True); E, S, V = E[first], S[first], V[first]   # selection ∩ evaluation overlap: keep one
train_score = train_c
out = args.out or f"runs/analysis/step12d/gnn_oof_s{args.seed}{'_SMOKE' if args.smoke else ''}.npz"; os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(out, train_score=np.nan_to_num(train_score), look_enc=E, look_spec=S, look_score=V)
json.dump({"transform": "per-query centring (training: mean of the 20 uniform alternatives; selection/evaluation: mean over the full available pool)", "scale_check": scale, "seed": args.seed, "gnn_run": args.gnn_run, "fold_runs": args.fold_runs, "dumps": args.dumps, "n_lookup": int(len(E)), "smoke": args.smoke,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(out.replace(".npz", ".json"), "w"), indent=1)
print("saved", out, f"({time.time() - t0:.0f}s)")
