"""STEP11 B5 (revision 1): score runs on the judging set J = 2018 first-time queries OUTSIDE the 3,000-query selection sample,
FULL pool (every specialist first observed before the day; evaluator pool "all"), weekly cutoff; D1 decomposition; C5a.

Per query and run: pessimistic rank = 1 + other-specialty candidates scored >= target + same-specialty (excl. target) >= target;
target specialty strictly before the day; specialty group (choice / non_choice / non_specific / unknown).
GNN and LightGBM cannot share a process on this Mac, so:
    python scripts/analysis/step11/step11_judge.py --kind gnn  [--dump-scores] runs/<gnn runs...>     (dump: save the first run's full score vectors for C5a)
    python scripts/analysis/step11/step11_judge.py --kind lgbm [--c5a]         runs/<lgbm runs...>    (c5a: combine with the dumped GNN scores)
    python scripts/analysis/step11/step11_judge.py --summary
Outputs (private per-query files): runs/analysis/step11_judge/<run>_PRIVATE.csv.gz; summary.json.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import sys
if "--kind" in sys.argv and sys.argv[sys.argv.index("--kind") + 1] == "lgbm":
    import lightgbm  # noqa: F401  (before torch)
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd


from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.step10 import specialty_strict, specialty_names, NON_CHOICE, NON_SPECIFIC  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="*"); ap.add_argument("--kind", choices=["gnn", "lgbm"])
ap.add_argument("--dump-scores", action="store_true"); ap.add_argument("--c5a", action="store_true"); ap.add_argument("--summary", action="store_true")
ap.add_argument("--inputs", default="prepared_data/three_node_v5/model_inputs_PRIVATE.pkl"); ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args(); root = str(workspace()); os.chdir(root)
OUT = "runs/analysis/step11_judge"; os.makedirs(OUT, exist_ok=True)
name_of = lambda r: os.path.basename(r.rstrip("/")).split("_", 1)[1]

if args.summary:
    files = {os.path.basename(f).replace("_PRIVATE.csv.gz", ""): pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/*_PRIVATE.csv.gz"))}
    rows = []
    for n, d in files.items():
        r = {"run": n, "queries": len(d), "MRR": float(d.rr.mean())}
        for g in ("choice", "non_choice", "non_specific"): r[f"MRR_{g}"] = float(d.rr[d.group == g].mean()); r[f"n_{g}"] = int((d.group == g).sum())
        k = ~d.group.isin(["unknown", "target_unavailable"]); r["mean_other_above"] = float(d.other_above[k].mean()); r["mean_same_above"] = float(d.same_above[k].mean())
        r["top1_specialty_match"] = float(d.top1_spec[k].mean()); rows.append(r)
    t = pd.DataFrame(rows); pd.set_option("display.width", 250)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # D1 counterfactual decomposition: table vs GNN (seed 42 pairs, then seed 1)
    dec = []
    for s in ("s42", "s1"):
        a, b = files.get(f"v5_unmasked_{s}"), files.get(f"lgbm_tabular_{s}")
        if a is None or b is None: continue
        assert (a.enc.to_numpy() == b.enc.to_numpy()).all()
        k = (~a.group.isin(["unknown", "target_unavailable"])).to_numpy()
        mrr = lambda o, sm: float(np.mean(1.0 / (1 + o[k] + sm[k])))
        g_o, g_s, t_o, t_s = a.other_above.to_numpy(), a.same_above.to_numpy(), b.other_above.to_numpy(), b.same_above.to_numpy()
        base, tab = mrr(g_o, g_s), mrr(t_o, t_s)
        dec.append({"seed": s, "GNN_MRR": base, "table_MRR": tab, "gap": tab - base,
                    "gap_closed_by_table_other_specialty_term": mrr(t_o, g_s) - base, "gap_closed_by_table_same_specialty_term": mrr(g_o, t_s) - base})
    c5 = files.get("c5a_table_specialty_x_gnn_within")
    if c5 is not None:                                          # C5a on its own query set (target available, specialty known/unknown)
        sub = lambda n: files[n].set_index("enc").loc[c5.enc].rr.mean() if n in files else np.nan
        print(f"\nC5a on its {len(c5):,} queries: C5a {c5.rr.mean():.4f} | table s42 {sub('lgbm_tabular_s42'):.4f} | GNN s42 {sub('v5_unmasked_s42'):.4f}")
    if dec:
        print("\nD1 decomposition (known-specialty queries):"); print(pd.DataFrame(dec).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    json.dump({"set": "J = 2018 first-time queries outside the 3,000-query selection sample; full pool", "runs": t.to_dict("records"), "d1": dec},
              open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    sys.exit()

mi = load_three_node_inputs(args.inputs); names = specialty_names(mi)
q = np.sort(mi.query_enc["validation"]); fo = q[mi.ps_asof.lookup(mi.enc_patient[q] * mi.n_specialist + mi.enc_target[q], mi.enc_day[q]) < 0]
J = np.setdiff1d(fo, mi.validation_sample); print(f"J: {len(J):,} queries", flush=True)
def group(sp_idx):
    if sp_idx <= 0: return "unknown"
    n = names.get(int(sp_idx), ""); return "non_choice" if n in NON_CHOICE else "non_specific" if n in NON_SPECIFIC else "choice"

dump = None
if args.c5a:
    gz = np.load(f"{OUT}/_gnn_scores_PRIVATE.npz"); g_enc, g_off, g_val = gz["enc"], gz["off"], gz["val"]; g_pos = {int(e): i for i, e in enumerate(g_enc)}
    c5a_rec = []
for ri, run in enumerate(args.runs):
    if args.kind == "gnn":
        from referral_gnn.three_node_train import load_three_node_model
        from referral_gnn.evaluate import ModelScorer
        from referral_gnn.train import pick_device
        dev = pick_device("auto"); sc = ModelScorer(load_three_node_model(run, mi, dev), mi, dev)
    else:
        from referral_gnn.tabular import load_baseline_scorer
        sc = load_baseline_scorer(run, mi)
    want_dump = args.dump_scores and ri == 0; d_enc, d_val = [], []
    rec = []; t0 = time.time()
    for cutoff, batch_enc, q_b in mi.buckets("validation", "week", query_subset=J):
        sc.prepare(cutoff, batch_enc)
        for qq in q_b:
            qq = int(qq); day = int(mi.enc_day[qq]); avail = np.where(mi.available_specialists(day))[0]; t = int(mi.enc_target[qq])
            hit = np.where(avail == t)[0]
            if not len(hit): rec.append({"enc": qq, "rr": 0.0, "group": "target_unavailable", "other_above": np.nan, "same_above": np.nan, "top1_spec": False}); continue
            tp = int(hit[0]); s = np.asarray(sc.score(qq, avail), float); sp = specialty_strict(mi, avail, day); tsp = int(sp[tp])
            ge = s >= s[tp]; same = sp == tsp; other_above = int((ge & ~same).sum()); same_above = int((ge & same).sum()) - 1
            rec.append({"enc": qq, "rr": 1.0 / (1 + other_above + same_above), "group": group(tsp), "other_above": other_above, "same_above": same_above,
                        "top1_spec": bool(tsp > 0 and sp[int(np.argmax(s))] == tsp)})
            if want_dump: d_enc.append(qq); d_val.append(s.astype(np.float32))
            if args.c5a and ri == 0 and qq in g_pos:                      # C5a: log P_table(specialty) + GNN within-specialty log-softmax
                i = g_pos[qq]; gs = g_val[g_off[i]:g_off[i + 1]].astype(float); assert len(gs) == len(s)
                pt = np.exp(s - s.max()); pt /= pt.sum(); spec_ids, inv = np.unique(sp, return_inverse=True)
                p_sp = np.bincount(inv, weights=pt); logp = np.log(np.maximum(p_sp[inv], 1e-300))
                gmax = pd.Series(gs).groupby(inv).transform("max").to_numpy(); lse = np.log(pd.Series(np.exp(gs - gmax)).groupby(inv).transform("sum").to_numpy()) + gmax
                comb = logp + (gs - lse); ge2 = comb >= comb[tp]
                c5a_rec.append({"enc": qq, "rr": 1.0 / (1 + int((ge2 & ~same).sum()) + int((ge2 & same).sum()) - 1), "group": group(tsp),
                                "other_above": int((ge2 & ~same).sum()), "same_above": int((ge2 & same).sum()) - 1, "top1_spec": bool(tsp > 0 and sp[int(np.argmax(comb))] == tsp)})
    pd.DataFrame(rec).to_csv(f"{OUT}/{name_of(run)}_PRIVATE.csv.gz", index=False)
    print(f"{name_of(run)}: {len(rec):,} queries, MRR {np.mean([r['rr'] for r in rec]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    if want_dump:
        off = np.r_[0, np.cumsum([len(v) for v in d_val])]
        np.savez(f"{OUT}/_gnn_scores_PRIVATE.npz", enc=np.array(d_enc), off=off, val=np.concatenate(d_val)); print("  dumped GNN score vectors for C5a", flush=True)
if args.c5a:
    pd.DataFrame(c5a_rec).to_csv(f"{OUT}/c5a_table_specialty_x_gnn_within_PRIVATE.csv.gz", index=False)
    print(f"C5a: {len(c5a_rec):,} queries, MRR {np.mean([r['rr'] for r in c5a_rec]):.4f}", flush=True)
