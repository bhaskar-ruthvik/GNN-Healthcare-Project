"""STEP12d A inputs (LightGBM process; no torch computation): out-of-fold TABLE scores for the GNN's training rows and full-pool
table scores for the GNN's selection queries.
  train_score  for every training row (mi.tc_enc order): LightGBM pure-tabular LambdaRank (reference configuration) trained on the
               OTHER time folds (K = 3), seed --seed;
  look_*       the full table run (--table-run) on the first-time validation selection queries × every available specialist
               (exactly the candidate pool of the GNN's validation, evaluate(pool="all")).
Usage: python scripts/train/step12d_table_oof.py --seed 42 --table-run runs/20261007T195708_lgbm_tabular_s42"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
try:
    import lightgbm as lgb
except ImportError:
    raise
import argparse, json, os, sys, time
import numpy as np

from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn import tabular as T  # noqa: E402
from referral_gnn.score_lookup import fold_of_tc_rows  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--table-run", required=True)
ap.add_argument("--k", type=int, default=3); ap.add_argument("--config", default="127:50:1000"); ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--smoke", action="store_true", help="tiny boosters and 50 selection queries (tests only)"); ap.add_argument("--out", default=None)
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
from referral_gnn.paths import resolve_run  # noqa: E402
args.table_run = resolve_run(args.table_run)
mi = load_three_node_inputs("prepared_data/three_node_v5/model_inputs_PRIVATE.pkl")
L, M, R = (int(x) for x in args.config.split(":")); R = 20 if args.smoke else R
t0 = time.time(); X, y, qid, fb = T.training_matrix(mi, "tabular", log=lambda s: None)
r0 = np.searchsorted(mi.tc_enc[::21], qid[::21]) * 21; idx = (r0[:, None] + np.arange(21)[None, :]).reshape(-1)    # matrix row → tc row
fold = fold_of_tc_rows(mi, args.k)[idx]
train_score = np.full(len(mi.tc_enc), np.nan, np.float32); boosters = []
base = dict(objective="lambdarank", lambdarank_truncation_level=21, learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            verbosity=-1, num_threads=args.threads, num_leaves=L, min_data_in_leaf=M, seed=args.seed, bagging_seed=args.seed, feature_fraction_seed=args.seed)
for k in range(args.k):
    tr_, te_ = fold != k, fold == k
    b = lgb.train(base, lgb.Dataset(X[tr_], y[tr_].astype(float), group=np.full(tr_.sum() // 21, 21), categorical_feature=fb.cat_idx, feature_name=fb.names), num_boost_round=R)
    boosters.append(b); train_score[idx[te_]] = b.predict(X[te_]); print(f"fold {k}: trained on {tr_.sum() // 21:,} queries, scored {te_.sum() // 21:,} ({time.time() - t0:.0f}s)", flush=True)
assert np.isfinite(train_score).all()
sc = T.load_baseline_scorer(args.table_run, mi)
sel = np.sort(mi.validation_sample); fo = sel[mi.ps_asof.lookup(mi.enc_patient[sel] * mi.n_specialist + mi.enc_target[sel], mi.enc_day[sel]) < 0]
# (smoke mode keeps the full selection pool: the lookup must cover every validation query)
E, S, V = [], [], []; ratios = [[] for _ in boosters]; rhos = [[] for _ in boosters]
from scipy.stats import spearmanr
for cutoff, batch_enc, q_b in mi.buckets("validation", "week", query_subset=fo):
    sc.prepare(cutoff, batch_enc)
    for q in q_b:
        avail = np.where(mi.available_specialists(int(mi.enc_day[q])))[0]; Xq = sc.fb.rows(int(q), avail); full = np.asarray(sc.model.predict(Xq), float)
        E.append(np.full(len(avail), q)); S.append(avail); V.append(full)
        for k_, b_ in enumerate(boosters):                      # reviewer S2: is the fold boosters' within-query scale the full booster's?
            fk = b_.predict(Xq); ratios[k_].append(np.std(fk) / max(np.std(full), 1e-9)); rhos[k_].append(spearmanr(fk, full).statistic)
scale = {}
for k in range(args.k):
    r_ = float(np.median(ratios[k])); scale[k] = {"median_sd_ratio_fold_over_full": r_, "median_spearman": float(np.median(rhos[k])), "rescaled": bool(not 0.9 <= r_ <= 1.1)}
    if scale[k]["rescaled"]: train_score[idx[fold == k]] /= r_          # pre-stated rule: outside [0.9, 1.1] → divide the fold's scores by the ratio
print("fold vs full scale check:", scale, flush=True)
out = args.out or f"runs/analysis/step12d/table_oof_s{args.seed}{'_SMOKE' if args.smoke else ''}.npz"; os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(out, train_score=train_score, look_enc=np.concatenate(E), look_spec=np.concatenate(S), look_score=np.concatenate(V).astype(np.float32))
json.dump({"scale_check": scale, "seed": args.seed, "k": args.k, "config": [L, M, R], "table_run": args.table_run, "n_train_rows": int(len(train_score)), "n_lookup": int(sum(map(len, E))),
           "selection_queries": int(len(fo)), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(out.replace(".npz", ".json"), "w"), indent=1)
print("saved", out, f"({time.time() - t0:.0f}s)")
