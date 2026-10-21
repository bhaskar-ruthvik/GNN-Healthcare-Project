"""STEP12c control: is the two-list selection gain due to COMBINING the models, or just to steering by knowable attributes?
Single-model steering: re-rank each model's eligible set by 0.7·z(score) + 0.3·z(as-of MIPS) + a·new − b·busy (a, b grid), and
compare its (fit, busy1, busy5, new5, new1, gain5) frontier with the two-list selection (step12c_complementarity.py E1) at equal fit.
Aggregates only → runs/analysis/step12c/steering_control.json"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json, os
import numpy as np, pandas as pd
root = str(workspace()); os.chdir(root)
from referral_gnn.candidates import Dump, zs  # noqa: E402
DM = Dump("runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz"); Z = DM.Z
off, n_ref, tpos, refm = DM.off, DM.n_ref, DM.tpos, DM.refm
PAIRS = [(P["G"], P["T"]) for P in DM.pairings("GT")]
busy, new, mips, none, ma = DM.busy, DM.new, DM.mips, DM.none, DM.ma
def lens_of(top, i):
    a = off[i]; j = a + top
    return (busy[j[0]], busy[j].mean(), new[j[0]], new[j].mean(), mips[j].mean() - refm[i], float(tpos[i] in set(top.tolist())),
            float(((new[j] > 0) & (mips[j] > refm[i]) & ~none[j] & (busy[j] == 0)).mean()))
COLS = ["busy1", "busy5", "new1", "new5", "gain5", "fit", "good_new5"]
def evaluate(scorefn):
    rows = []
    for i in range(n_ref):
        a, b = off[i], off[i + 1]; top = np.argsort(-scorefn(a, b), kind="stable")[:5]; rows.append(lens_of(top, i))
    return dict(zip(COLS, np.mean(rows, 0).tolist()))
GRID = [(a, b) for a in (0, 0.25, 0.5, 1.0, 1.5) for b in (0, 0.25, 0.5, 1.0, 1.5)]
out = {"pairs": [[g[7:], t[7:]] for g, t in PAIRS], "per_pair": []}
for gk, tk in PAIRS:
    r = {}
    for lab, k in (("GNN", gk), ("Table", tk)):
        S = Z[k].astype(float)
        r[lab] = [{"a_new": a, "b_busy": b, **evaluate(lambda x, y, S=S, a=a, b=b: 0.7 * zs(S[x:y]) + 0.3 * zs(ma[x:y]) + a * new[x:y] - b * busy[x:y])} for a, b in GRID]
    out["per_pair"].append(r)
json.dump(out, open("runs/analysis/step12c/steering_control.json", "w"), indent=1)
# compare at fit ≥ 0.445 (the two-list selection's fit at λ = 2 was 0.450–0.453): best single-model points
sel = json.load(open("runs/analysis/step12c/complementarity.json"))
for p, r, c in zip(out["pairs"], out["per_pair"], sel["per_pair"]):
    two = next(f for f in c["E1_knowable_list_selection_frontier"] if f["lambda"] == 2)
    print(f"\n=== {p}\n two-list selection (λ=2): fit {two['fit']:.3f} busy1 {two['busy1']:.3f} busy5 {two['busy5']:.3f} new5 {two['new5']:.3f} new1 {two['new1']:.3f} gain5 {two['gain5']:.2f}")
    for lab in ("GNN", "Table"):
        d = pd.DataFrame(r[lab]); ok = d[(d.fit >= two["fit"] - 0.005)]
        dom = ok[(ok.busy1 <= two["busy1"] + 0.005) & (ok.busy5 <= two["busy5"] + 0.005) & (ok.new5 >= two["new5"] - 0.005)]
        print(f"  {lab}-only steering: {len(ok)} grid points at ≥ the selection's fit; points matching or beating it on busy1, busy5 AND new5: {len(dom)}")
        best = ok.assign(score=ok.new5 - ok.busy5).sort_values("score", ascending=False).head(3)
        print(best[["a_new", "b_busy", "fit", "busy1", "busy5", "new5", "new1", "gain5"]].round(3).to_string(index=False))
