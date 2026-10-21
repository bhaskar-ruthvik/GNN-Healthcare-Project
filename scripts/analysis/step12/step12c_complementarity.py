"""STEP12c: are the GNN and the table complementary? Diagnostics on candidate-level scores (no training).

Input: runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz (step10_evaluate.py --dump-scores), 2018 STEP10 cohort.
Seed pairings (GNN, table): (s42, s42), (s1, s1), (s42_dup, s7). Lists at the 30% quality weight: 0.7·z(score) + 0.3·z(as-of MIPS).
Lens per referral: busy1 / busy5 (busiest-10% share of top pick / top-5 slots), new1 / new5 (new to the PCP), good_new5, gain5,
fit (chosen specialist in the top 5), mean distance of the top 5.

  A  agreement: top-1 agreement, top-5 overlap.
  B  disagreement: attributes of table-only vs GNN-only picks (top 5 and top 1).
  C  top-pick cross-tab: (busy, new) category of the GNN's top pick vs the table's.
  D  segments: GNN − table lens differences by PCP history depth, familiar share, patient age, eligible-set size, specialty (top 8):
     do the GNN's load advantage and the table's newness advantage sit in different segments (combinable by a gate) or together
     (a trade-off within segments)?
  E  ceilings: per referral choose the GNN list or the table list
       E1 by knowable attributes only (newness and load are known at referral time): maximise new5 − λ·busy5 → a (load, newness)
          frontier, compared with the straight line between the two models (= random per-referral mixing);
       E2 hindsight oracle for fit (list that contains the chosen specialist) — what fit could at most be with perfect selection.
Aggregates only. Output: runs/analysis/step12c/complementarity.json
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os

import numpy as np
import pandas as pd

root = str(workspace()); os.chdir(root)
from referral_gnn.candidates import Dump, zs  # noqa: E402
DM = Dump("runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz"); Z = DM.Z
off, n_ref, tpos, refm = DM.off, DM.n_ref, DM.tpos, DM.refm
PAIRS = [(P["G"], P["T"]) for P in DM.pairings("GT")]
A = {k: Z[k] for k in ("busy", "new", "ins", "cat", "mips", "mips_asof", "mips_none", "dist", "spec")}

def lists(key):
    """top-5 index arrays (within each referral) at the 30% weight."""
    out = []
    for i in range(n_ref):
        a, b = off[i], off[i + 1]; sc = 0.7 * zs(Z[key][a:b].astype(float)) + 0.3 * zs(A["mips_asof"][a:b].astype(float))
        out.append(np.argsort(-sc, kind="stable")[:5])
    return out

def lens(top, i):
    a = off[i]; g = lambda k: A[k][a + top]
    none = g("mips_none").astype(bool); mips = g("mips")
    return {"busy1": float(g("busy")[0]), "busy5": float(g("busy").mean()), "new1": float(g("new")[0]), "new5": float(g("new").mean()),
            "good_new5": float((g("new").astype(bool) & (mips > refm[i]) & ~none & ~g("busy").astype(bool)).mean()), "gain5": float(mips.mean() - refm[i]),
            "fit": float(tpos[i] in set(top.tolist())), "dist5": float(np.mean(np.where(g("dist") < 0, np.nan, g("dist"))))}

pcp = Z["pcp"]; spec_of = Z["specialty"]
res = {"pairs": [[p[0][7:], p[1][7:]] for p in PAIRS], "per_pair": []}
for gk, tk in PAIRS:
    LG, LT = lists(gk), lists(tk)
    G = pd.DataFrame([lens(LG[i], i) for i in range(n_ref)]); T = pd.DataFrame([lens(LT[i], i) for i in range(n_ref)])
    r = {"GNN": G.mean().to_dict(), "Table": T.mean().to_dict()}
    # A agreement
    top1 = np.array([LG[i][0] == LT[i][0] for i in range(n_ref)]); ov = np.array([len(set(LG[i]) & set(LT[i])) / 5 for i in range(n_ref)])
    r["A_top1_agreement"] = float(top1.mean()); r["A_top5_overlap_mean"] = float(ov.mean()); r["A_top5_overlap_dist"] = {str(k): float(v) for k, v in pd.Series(ov * 5).value_counts(normalize=True).sort_index().items()}
    # B disagreement: table-only vs GNN-only picks
    def attrs(idx_lists):
        rows = []
        for i, ids in idx_lists:
            a = off[i]
            for j in ids: rows.append({k: float(A[k][a + j]) for k in ("busy", "new", "ins", "cat", "mips", "dist")} | {"mips_none": float(A["mips_none"][a + j])})
        d = pd.DataFrame(rows); d.loc[d.dist < 0, "dist"] = np.nan; d.loc[d.mips_none == 1, "mips"] = np.nan
        return {"n": int(len(d)), **{k: float(d[k].mean()) for k in ("busy", "new", "ins", "cat", "mips", "dist")}}
    r["B_table_only_top5"] = attrs([(i, sorted(set(LT[i]) - set(LG[i]))) for i in range(n_ref)])
    r["B_gnn_only_top5"] = attrs([(i, sorted(set(LG[i]) - set(LT[i]))) for i in range(n_ref)])
    dis = [i for i in range(n_ref) if not top1[i]]
    r["B_table_top1_when_disagree"] = attrs([(i, [LT[i][0]]) for i in dis]); r["B_gnn_top1_when_disagree"] = attrs([(i, [LG[i][0]]) for i in dis])
    # C top-pick cross-tab
    cg = (G.busy1.astype(int).astype(str) + "b/" + G.new1.astype(int).astype(str) + "n"); ct = (T.busy1.astype(int).astype(str) + "b/" + T.new1.astype(int).astype(str) + "n")
    r["C_top1_crosstab_GNN_rows_table_cols"] = pd.crosstab(cg, ct, normalize=True).round(4).to_dict()
    # D segments
    fam = np.array([A["new"][off[i]:off[i + 1]].mean() for i in range(n_ref)])            # share of the eligible set new to the PCP
    nel = np.diff(off)
    segs = {"familiar share of eligible set": pd.qcut(1 - fam, 5, labels=False, duplicates="drop"), "eligible-set size": pd.qcut(nel, 5, labels=False, duplicates="drop"),
            "PCP": None, "specialty": None}
    D = {}
    for name, lab in segs.items():
        if name == "PCP": continue
        if name == "specialty":
            top = pd.Series(spec_of).value_counts().index[:8]; lab = np.where(np.isin(spec_of, top), spec_of, -1)
        dd = pd.DataFrame({"seg": lab, "dbusy1": G.busy1 - T.busy1, "dbusy5": G.busy5 - T.busy5, "dnew5": G.new5 - T.new5, "dfit": G.fit - T.fit, "dgain5": G.gain5 - T.gain5})
        D[name] = dd.groupby("seg").agg(n=("dbusy1", "size"), dbusy1=("dbusy1", "mean"), dbusy5=("dbusy5", "mean"), dnew5=("dnew5", "mean"), dfit=("dfit", "mean"), dgain5=("dgain5", "mean")).round(4).reset_index().to_dict("records")
    # per-PCP: does any PCP group show GNN better on BOTH load and newness?
    pp = pd.DataFrame({"pcp": pcp, "dbusy1": G.busy1 - T.busy1, "dnew5": G.new5 - T.new5}).groupby("pcp").agg(n=("dbusy1", "size"), dbusy1=("dbusy1", "mean"), dnew5=("dnew5", "mean"))
    pp = pp[pp.n >= 20]
    D["PCPs with ≥ 20 referrals: share where GNN has lower load AND ≥ newness"] = float(((pp.dbusy1 < 0) & (pp.dnew5 >= 0)).mean()); D["PCPs (n)"] = int(len(pp))
    D["referral-level corr(dbusy5, dnew5)"] = float(np.corrcoef(G.busy5 - T.busy5, G.new5 - T.new5)[0, 1])
    r["D_segments"] = D
    # E ceilings
    front = []
    for lam in (0, 0.25, 0.5, 1, 2, 4, 8):
        useT = (T.new5 - lam * T.busy5) > (G.new5 - lam * G.busy5)
        M = G.copy(); M.loc[useT.to_numpy()] = T.loc[useT.to_numpy()]
        front.append({"lambda": lam, "share_table_lists": float(useT.mean()), **{k: float(M[k].mean()) for k in ("busy1", "busy5", "new5", "new1", "fit", "gain5", "good_new5")}})
    r["E1_knowable_list_selection_frontier"] = front
    # straight line (random mixing) at the same busy5 as each frontier point → newness on the line
    for f in front:
        a = (f["busy5"] - r["GNN"]["busy5"]) / (r["Table"]["busy5"] - r["GNN"]["busy5"]) if r["Table"]["busy5"] != r["GNN"]["busy5"] else 0
        f["line_new5_at_same_busy5"] = float(r["GNN"]["new5"] + a * (r["Table"]["new5"] - r["GNN"]["new5"]))
    r["E2_hindsight_fit_oracle"] = float(np.maximum(G.fit, T.fit).mean())
    res["per_pair"].append(r)
os.makedirs("runs/analysis/step12c", exist_ok=True); json.dump(res, open("runs/analysis/step12c/complementarity.json", "w"), indent=1, default=float)
for p, r in zip(res["pairs"], res["per_pair"]):
    print(f"\n=== {p}\n GNN {pd.Series(r['GNN']).round(3).to_dict()}\n Table {pd.Series(r['Table']).round(3).to_dict()}")
    print(f" A top-1 agreement {r['A_top1_agreement']:.3f}; top-5 overlap {r['A_top5_overlap_mean']:.3f}")
    print(" B table-only", {k: round(v, 3) for k, v in r["B_table_only_top5"].items()}, "\n   GNN-only", {k: round(v, 3) for k, v in r["B_gnn_only_top5"].items()})
    print(" D corr(dbusy5,dnew5)", round(r["D_segments"]["referral-level corr(dbusy5, dnew5)"], 3), "| PCPs GNN better on both:", round(r["D_segments"]["PCPs with ≥ 20 referrals: share where GNN has lower load AND ≥ newness"], 3))
    for f in r["E1_knowable_list_selection_frontier"]: print("  E1", {k: round(v, 3) for k, v in f.items()})
    print(" E2 hindsight fit oracle", round(r["E2_hindsight_fit_oracle"], 3))
