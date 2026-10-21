"""STEP11 §11.3a masking probe: do the masked model's encounter embeddings still encode patient gender / race (through stand-ins)?
Step 1 (torch):   --extract  : encounter embeddings (post-HGT) for every training- and validation-year encounter, from the masked
                               and the unmasked v5 model (seed 42), each encounter embedded as a query at its weekly cutoff.
Step 2 (no torch): --probe   : 5-fold patient-grouped out-of-fold probes:
                               (a) logistic probe on the masked model's embeddings; (b) logistic probe on the unmasked embeddings
                               (upper reference); (c) nonlinear LightGBM probe on the masked model's INPUTS (encounter numeric with
                               the demographic flag zeroed, encounter categoricals without gender/race, disease-category multi-hot):
                               what the stand-ins alone carry.
AUC: gender (F vs M); race: macro one-vs-rest over known races (unknown excluded). T5 measure = AUC(a) − AUC(c).
Embeddings are written to a private scratch file under runs/analysis/step11_3a/ (deleted with --cleanup).
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import sys
if "--probe" in sys.argv:
    import lightgbm as lgb  # noqa: F401  (no torch in this process)
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true"); ap.add_argument("--probe", action="store_true"); ap.add_argument("--cleanup", action="store_true")
args = ap.parse_args(); root = str(workspace()); os.chdir(root)
OUT = "runs/analysis/step11_3a"; os.makedirs(OUT, exist_ok=True); EMB = f"{OUT}/_embeddings_PRIVATE.npz"
sys.path.insert(0, root)

if args.extract:
    import torch
    from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model
    from referral_gnn.train import pick_device
    mi = load_three_node_inputs("prepared_data/three_node_v5/model_inputs_PRIVATE.pkl"); dev = pick_device("auto")
    runs = {"masked": [r for r in glob.glob("runs/*_v5_masked_s42") if "INCOMPLETE" not in r][0], "unmasked": [r for r in glob.glob("runs/*_v5_unmasked_s42") if "INCOMPLETE" not in r][0]}
    out = {}
    for name, run in runs.items():
        m = load_three_node_model(run, mi, dev); embs, encs = [], []
        for split in ("train", "validation"):
            for cutoff, batch_enc, q_b in mi.buckets(split, "week"):
                with torch.no_grad():
                    h = m.encode(mi.graph_batch(cutoff, batch_enc, device=dev))["encounter"]
                embs.append(h[torch.as_tensor(q_b, device=dev)].float().cpu().numpy()); encs.append(q_b)
        out[f"{name}_emb"] = np.concatenate(embs); out[f"{name}_enc"] = np.concatenate(encs); print(name, out[f"{name}_emb"].shape, flush=True)
    assert np.array_equal(out["masked_enc"], out["unmasked_enc"])
    cat = mi.enc_patient_cat[out["masked_enc"]]; blocks = mi.blocks if hasattr(mi, "blocks") else None
    np.savez(EMB, masked=out["masked_emb"], unmasked=out["unmasked_emb"], enc=out["masked_enc"], patient=mi.enc_patient[out["masked_enc"]],
             gender=cat[:, 0], race=cat[:, 4], enc_num=mi._encounter_static()[out["masked_enc"]], enc_cat=cat,
             dx=mi.enc_dx[out["masked_enc"]], dx_mask=mi.enc_dx_mask[out["masked_enc"]], n_cat=int(mi.enc_dx.max()) + 1,
             demo_cat_cols=np.array(mi.demographic_cat_cols), demo_flag_cols=np.array(mi.demographic_flag_cols))
    print("saved", EMB, flush=True)

if args.probe:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    z = np.load(EMB); pat = z["patient"]
    num = z["enc_num"].copy(); num[:, z["demo_flag_cols"]] = 0.0
    keep_cat = [i for i in range(z["enc_cat"].shape[1]) if i not in set(z["demo_cat_cols"].tolist())]
    dx = np.zeros((len(num), int(z["n_cat"])), np.float32); r_, c_ = np.nonzero(z["dx_mask"]); dx[r_, z["dx"][r_, c_]] = 1.0
    X_inputs = np.hstack([num, z["enc_cat"][:, keep_cat].astype(np.float32), dx])
    res = {}
    for target in ("gender", "race"):
        y = z[target].astype(int); known = y > 0
        classes = sorted(set(y[known].tolist()))
        if target == "gender": classes = [c for c in classes if (y == c).sum() > 100][:2]
        known &= np.isin(y, classes)
        res[target] = {"n": int(known.sum()), "classes": len(classes)}
        for name, X, kind in (("masked embeddings", z["masked"], "lr"), ("unmasked embeddings", z["unmasked"], "lr"), ("masked model inputs (LightGBM)", X_inputs, "gbm")):
            Xk, yk, gk = X[known], y[known], pat[known]; oof = np.zeros((len(yk), len(classes)))
            for tr, te in GroupKFold(5).split(Xk, yk, gk):
                if kind == "lr":
                    sc = StandardScaler().fit(Xk[tr]); clf = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(Xk[tr]), yk[tr]); p = clf.predict_proba(sc.transform(Xk[te]))
                else:
                    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=4).fit(Xk[tr], yk[tr]); p = clf.predict_proba(Xk[te])
                oof[te] = p
            if len(classes) == 2: auc = roc_auc_score(yk == classes[1], oof[:, 1])
            else: auc = roc_auc_score(yk, oof, multi_class="ovr", average="macro", labels=classes)
            rng = np.random.default_rng(20261009); up = np.unique(gk); bs = []
            for _ in range(200):
                take = np.isin(gk, rng.choice(up, len(up), replace=True))
                try:
                    bs.append(roc_auc_score(yk[take] == classes[1], oof[take, 1]) if len(classes) == 2 else roc_auc_score(yk[take], oof[take], multi_class="ovr", average="macro", labels=classes))
                except ValueError: pass
            res[target][name] = {"auc": round(float(auc), 4), "ci": [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)]}
            print(target, name, res[target][name], flush=True)
        res[target]["T5_masked_minus_inputs"] = round(res[target]["masked embeddings"]["auc"] - res[target]["masked model inputs (LightGBM)"]["auc"], 4)
    json.dump(res, open(f"{OUT}/probe_summary.json", "w"), indent=1); print(json.dumps(res, indent=1))

if args.cleanup and os.path.exists(EMB): os.remove(EMB); print("removed", EMB)
