"""STEP12c pre-meeting checks on the steered equal blend (setting fixed 2026-10-14: 0.5·z(GNN) + 0.5·z(table), 30% MIPS weight,
+0.3·new − 0.4·busy), 2018 validation, seed-averaged over the 3 pairings, against the GNN and the table at the 30% weight.
  1. groups: lenses by the same splits as Step 11.2 (patient age, local access, PCP history, PCP network, referred specialist's
     tenure / experience / location); blend − GNN and blend − table with PCP-cluster 95% CIs; cells < 100 patients flagged.
  2. where the load goes: for each model's top pick and top-5 slots — share to early-career specialists (≤ 6 Medicare years),
     low-experience (< 10 referrals received before the day), non-metro; the within-set training-volume tier of the top pick
     (busiest 10% / next 15% / middle / bottom half); distinct specialists recommended as top pick; within-specialty Gini of top picks.
Aggregates only → runs/analysis/step12c/blend_checks.json"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json, os, sys
import numpy as np, pandas as pd

root = str(workspace()); os.chdir(root)
from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.step10 import first_time_queries  # noqa: E402
DUMP_PATH = "runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz"
Z = np.load(DUMP_PATH); off = Z["offsets"]; n_ref = len(off) - 1
from referral_gnn.candidates import Dump, zs, pcp_bootstrap_index  # noqa: E402
PAIRS = [(P["G"], P["T"]) for P in Dump(DUMP_PATH).pairings("GT")]
busy, new, mips, none, ma, dist, spec = (Z[k] for k in ("busy", "new", "mips", "mips_none", "mips_asof", "dist", "spec"))
busy = busy.astype(float); new = new.astype(float); none = none.astype(bool); ma = ma.astype(float); dist = np.where(dist < 0, np.nan, dist)
tpos, refm, encs, pcp_of = Z["target_pos"], Z["ref_mips"], Z["enc"], Z["pcp"]
# specialist attributes for "where the load goes"
mi = load_three_node_inputs("prepared_data/three_node_v5/model_inputs_PRIVATE.pkl")
snap = mi.provider_snapshot(int(np.max(mi.enc_day)) + 1); i_t = list(mi.spec_encoder.numeric_cols).index("years_in_medicare")
sc = mi.spec_encoder.scalers["years_in_medicare"]; ten = snap["specialist_num"][:, 3 * i_t] * sc.std + sc.mean; ten_missing = snap["specialist_num"][:, 3 * i_t + 1] > 0.5
early = (~ten_missing) & (ten <= 6)
ru = pd.read_pickle("prepared_data/provider_ruca_v1/specialist_ruca_PRIVATE.pkl").set_index("npi").group
nonmetro = (ru.reindex(pd.Index(mi.specialist_ids).astype(str)) == "nonmetro").to_numpy()
BIG = 1 << 20; tk = np.sort(mi.enc_target.astype(np.int64) * BIG + mi.enc_day.astype(np.int64))
day_of = dict(zip(mi.query_enc["validation"].tolist(), mi.enc_day[mi.query_enc["validation"]].tolist()))
q_tr = first_time_queries(mi, "train"); train_load = np.bincount(mi.enc_target[q_tr], minlength=mi.n_specialist).astype(float)
rows = {m: [] for m in ("GNN", "Table", "Blend")}; tops = {m: [] for m in rows}
for i in range(n_ref):
    a, b = off[i], off[i + 1]; m = 0.3 * zs(ma[a:b]); d = day_of[int(encs[i])]
    sp = spec[a:b].astype(np.int64); prior = np.searchsorted(tk, sp * BIG + d) - np.searchsorted(tk, sp * BIG); lowexp = prior < 10
    vol = train_load[sp]; pct = (pd.Series(vol).rank(method="average").to_numpy() - 1) / max(len(vol) - 1, 1)
    for name in rows: rows[name].append([])
    for gk, tk_ in PAIRS:
        G, T = Z[gk][a:b].astype(float), Z[tk_][a:b].astype(float)
        S = {"GNN": 0.7 * zs(G) + m, "Table": 0.7 * zs(T) + m, "Blend": 0.7 * zs(0.5 * zs(G) + 0.5 * zs(T)) + m + 0.3 * new[a:b] - 0.4 * busy[a:b]}
        for name, s in S.items():
            j = np.argsort(-s, kind="stable")[:5]; g = a + j
            rows[name][-1].append([busy[g[0]], busy[g].mean(), new[g[0]], new[g].mean(), mips[g].mean() - refm[i], float(tpos[i] in set(j.tolist())),
                                   float(np.nanmean(dist[g])) if np.isfinite(dist[g]).any() else np.nan,
                                   float(early[sp[j[0]]]), float(early[sp[j]].mean()), float(lowexp[j[0]]), float(lowexp[j].mean()), float(nonmetro[sp[j[0]]]), float(nonmetro[sp[j]].mean()),
                                   float(pct[j[0]])])
            tops[name].append((int(Z["specialty"][i]), int(sp[j[0]])))
COLS = ["busy1", "busy5", "new1", "new5", "gain5", "fit", "dist5", "early1", "early5", "lowexp1", "lowexp5", "nonmetro1", "nonmetro5", "volpct1"]
R = {k: pd.DataFrame(np.nanmean(np.array(v, float), axis=1), columns=COLS) for k, v in rows.items()}       # seed-averaged per referral
attrs = pd.read_csv("runs/analysis/step12_topk/K5/validation_per_referral_PRIVATE.csv.gz", usecols=lambda c: "|" not in c).set_index("enc").reindex(encs)
assert attrs.index.equals(pd.Index(encs))
codes, NC, IDX = pcp_bootstrap_index(pcp_of, 20261015)
def ci(x, msk):
    s1 = np.bincount(codes, weights=np.where(msk, np.nan_to_num(x), 0), minlength=NC); s0 = np.bincount(codes, weights=msk.astype(float), minlength=NC)
    den = s0[IDX].sum(1); ok = den > 0                                    # draws without any of the group's PCPs carry no information
    bb = s1[IDX].sum(1)[ok] / den[ok]; return [float(np.quantile(bb, .025)), float(np.quantile(bb, .975))]
SPL = {"Patient age": pd.cut(attrs.patient_age, [-1, 64, 74, 84, 200], labels=["<65", "65–74", "75–84", "85+"]),
       "Patient local access": pd.qcut(attrs.access5, 3, labels=["near", "middle", "far"]),
       "PCP history (referrals made before)": pd.qcut(attrs.pcp_prior_refs.rank(method="first"), 5, labels=["Q1 fewest", "Q2", "Q3", "Q4", "Q5 most"]),
       "PCP network (specialists used)": pd.qcut(attrs.pcp_net, 3, labels=["thin", "medium", "wide"], duplicates="drop"),
       "Referred specialist tenure": pd.cut(attrs.ref_tenure, [-1, 6, 13, 99], labels=["early ≤6y", "mid 7–13y", "late ≥14y"]),
       "Referred specialist experience": pd.cut(attrs.ref_prior_refs, [-1, 9, 49, 1e9], labels=["1–9", "10–49", "50+"]),
       "Referred specialist location": attrs.ref_nonmetro.map({True: "non-metro", False: "metro"})}
groups = {}
for name, lab in SPL.items():
    out = []
    for g in lab.cat.categories if hasattr(lab, "cat") else sorted(lab.dropna().unique()):
        msk = (lab == g).to_numpy()
        if not msk.any(): continue
        rec = {"group": str(g), "referrals": int(msk.sum()), "patients": int(attrs.patient[msk].nunique()), "pcps": int(attrs.pcp[msk].nunique()),
               "small": bool(attrs.patient[msk].nunique() < 100 or attrs.pcp[msk].nunique() < 5)}
        for k in ("busy1", "new5", "gain5", "fit", "dist5"):
            for mname in R: rec[f"{mname}|{k}"] = float(np.nanmean(R[mname][k][msk]))
            rec[f"blend−GNN|{k}"] = float(np.nanmean((R["Blend"][k] - R["GNN"][k])[msk])); rec[f"blend−GNN|{k}|ci"] = ci((R["Blend"][k] - R["GNN"][k]).to_numpy(), msk)
            rec[f"blend−table|{k}"] = float(np.nanmean((R["Blend"][k] - R["Table"][k])[msk])); rec[f"blend−table|{k}|ci"] = ci((R["Blend"][k] - R["Table"][k]).to_numpy(), msk)
        out.append(rec)
    groups[name] = out
where = {}
for mname, D in R.items():
    tp = pd.DataFrame(tops[mname], columns=["specialty", "spec"])
    gini = []
    for _, gdf in tp.groupby("specialty"):
        x = np.sort(gdf.spec.value_counts().to_numpy()); n = len(x)
        if n > 1: gini.append(((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()), x.sum()))
    tier = pd.cut(D.volpct1, [-0.01, 0.5, 0.75, 0.9, 1.0], labels=["bottom half", "50–75th", "75–90th", "top 10% of set"])
    where[mname] = {"early1": float(D.early1.mean()), "early5": float(D.early5.mean()), "lowexp1": float(D.lowexp1.mean()), "lowexp5": float(D.lowexp5.mean()),
                    "nonmetro1": float(D.nonmetro1.mean()), "nonmetro5": float(D.nonmetro5.mean()), "busy1": float(D.busy1.mean()),
                    "top_pick_volume_tier": {str(k): float(v) for k, v in tier.value_counts(normalize=True).sort_index().items()},
                    "distinct_specialists_as_top_pick_any_pairing": int(tp.spec.nunique()), "within_specialty_gini_top_picks": float(np.average([g for g, _ in gini], weights=[w for _, w in gini]))}
# reference: what practice did (the chosen specialist)
tgt = np.array([spec[off[i] + tpos[i]] if tpos[i] >= 0 else -1 for i in range(n_ref)])
okt = tgt >= 0
where["Practice (chosen)"] = {"early1": float(early[tgt[okt]].mean()), "nonmetro1": float(nonmetro[tgt[okt]].mean()), "busy1": float(attrs.ref_busy.mean()),
                              "lowexp1": float((attrs.ref_prior_refs < 10).mean())}
where["Available (eligible sets)"] = {"early": float(attrs.elig_early.mean()), "lowexp": float(attrs.elig_lowexp.mean()), "nonmetro": float(attrs.elig_nonmetro.mean()), "busy": float(attrs.elig_busy.mean())}
out = {"setting": "steered equal blend vs GNN vs table, 30% MIPS weight, 2018 validation, seed-averaged over 3 pairings", "groups": groups, "where_the_load_goes": where}
json.dump(out, open("runs/analysis/step12c/blend_checks.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
for name, recs in groups.items():
    print(f"\n== {name}")
    for r in recs:
        print(f"  {r['group']:12s} n={r['referrals']:5d} pat={r['patients']:5d} pcp={r['pcps']:3d}{' (small)' if r['small'] else ''} | load GNN {r['GNN|busy1']:.3f} table {r['Table|busy1']:.3f} blend {r['Blend|busy1']:.3f}"
              f" Δ {r['blend−GNN|busy1']:+.3f} {[round(x, 3) for x in r['blend−GNN|busy1|ci']]} | new GNN {r['GNN|new5']:.2f} table {r['Table|new5']:.2f} blend {r['Blend|new5']:.2f}"
              f" | MIPS Δ {r['blend−GNN|gain5']:+.2f} | fit GNN {r['GNN|fit']:.3f} blend {r['Blend|fit']:.3f} Δ {r['blend−GNN|fit']:+.3f} {[round(x, 3) for x in r['blend−GNN|fit|ci']]}")
print("\n== where the load goes"); print(json.dumps(where, indent=1))
