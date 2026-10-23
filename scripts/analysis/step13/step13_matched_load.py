"""STEP13 post hoc (descriptive; both years already seen, 2026-10-23): load and newness at MATCHED FIT for the pairs that carry the
"graph information" story, from the STEP13 candidate-score dumps (no new scoring).

Matched fit as in STEP12a §2.E (scripts/analysis/step12/step12_congestion.py): the cell with the higher fit is mixed with the exact
random scorer down to the other's fit, α = (F_s − F_w) / (F_s − F_random), matched X = (1 − α)·X_s + α·X_random, for X = top-pick
load and newness (top 5); α is recomputed inside every PCP-cluster bootstrap draw. Cells are seed-averaged per referral (3 seeds);
per-seed-pairing points (seed IDs paired as in STEP13) are reported beside. Native lenses at the 30% quality weight.
Random scorer (exact expectation over the eligible set): fit = min(5, n)/n when the chosen specialist is eligible, top-pick load =
eligible busy share, newness = eligible new share.
Usage: python scripts/analysis/step13/step13_matched_load.py --split validation   (writes runs/analysis/step13/posthoc_matched_load_<split>.json)
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import json

import numpy as np

from referral_gnn import step13 as S  # noqa: E402
from referral_gnn.candidates import Dump, pcp_bootstrap_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["validation", "test"])
ap.add_argument("--draws", type=int, default=2000)
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
PAIRS = [("flat_rf", "table", "graph summaries added to the pure table (LightGBM)"),
         ("info_nohist", "table", "graph summaries + counts added to the pure table (LightGBM)"),
         ("mlp_rf", "flat_rf", "same graph summaries: neural MLP vs LightGBM"),
         ("gnn_nohist", "mlp_rf", "same 2-hop reach: learned propagation vs fixed summaries (both neural)"),
         ("flat_rf", "graph", "graph summaries in LightGBM vs the full GNN"),
         ("graph", "table", "reference: full GNN vs pure table (STEP11 headline)"),
         ("ngp", "table", "reference: neural without message passing vs pure table")]
cells = sorted({c for p in PAIRS for c in p[:2]})
runs = json.load(open("runs/analysis/step13/resolved_manifest.json"))["cells"]
D = Dump(f"runs/analysis/step13/eval/{args.split}_candidate_scores_PRIVATE.npz")
nat = {c: np.stack([S.native(D, "score::" + runs[c][str(s)]) for s in S.SEEDS]) for c in cells}       # (3, n_ref, 6)
FIT, BUSY1, NEW5 = 0, 1, 3
n = np.diff(D.off).astype(float)
r_fit = np.where(D.tpos >= 0, np.minimum(5, n) / n, 0.0)
r_load = np.array([D.busy[a:b].mean() for a, b in D.seg]); r_new = np.array([D.new[a:b].mean() for a, b in D.seg])
codes, nc, idx = pcp_bootstrap_index(D.pcp, seed=S.RNG_PCP, draws=args.draws)
W = np.stack([np.bincount(r, minlength=nc) for r in idx]).astype(float)
cnt = np.bincount(codes, minlength=nc).astype(float)
pp = lambda x: np.bincount(codes, weights=x, minlength=nc)


def means(x):
    """(point, draws) of the cluster-bootstrapped mean of a per-referral series."""
    s = pp(x); return s.sum() / cnt.sum(), (W @ s) / (W @ cnt)


def matched(fa, xa, fb, xb, rx):
    """X(A) − X(B) after degrading the higher-fit cell to the lower fit (point, draws)."""
    (Fa, Fa_d), (Fb, Fb_d), (Fr, Fr_d) = means(fa), means(fb), means(r_fit)
    (Xa, Xa_d), (Xb, Xb_d), (Xr, Xr_d) = means(xa), means(xb), means(rx)
    def one(Fa, Fb, Fr, Xa, Xb, Xr):
        T = np.minimum(Fa, Fb)
        aa = np.clip((Fa - T) / np.maximum(Fa - Fr, 1e-9), 0, 1); ab = np.clip((Fb - T) / np.maximum(Fb - Fr, 1e-9), 0, 1)
        return ((1 - aa) * Xa + aa * Xr) - ((1 - ab) * Xb + ab * Xr)
    return float(one(Fa, Fb, Fr, Xa, Xb, Xr)), one(Fa_d, Fb_d, Fr_d, Xa_d, Xb_d, Xr_d)


out = {"split": args.split, "status": "POST HOC, descriptive (both years already seen); not a pre-registered test", "draws": args.draws,
       "n_referrals": int(D.n_ref), "method": "STEP12a §2.E random-mixture matched fit; seed-averaged cells; PCP-cluster bootstrap",
       "random_scorer": {"fit": float(r_fit.mean()), "top_pick_load": float(r_load.mean()), "newness": float(r_new.mean())}, "pairs": []}
for a, b, what in PAIRS:
    A, Bc = nat[a].mean(0), nat[b].mean(0)
    row = {"A": a, "B": b, "what": what, "fit_A": float(A[:, FIT].mean()), "fit_B": float(Bc[:, FIT].mean()),
           "native_load_A": float(A[:, BUSY1].mean()), "native_load_B": float(Bc[:, BUSY1].mean()),
           "native_new_A": float(A[:, NEW5].mean()), "native_new_B": float(Bc[:, NEW5].mean())}
    for lens, col, rx in (("top_pick_load", BUSY1, r_load), ("newness", NEW5, r_new)):
        pt, dr = matched(A[:, FIT], A[:, col], Bc[:, FIT], Bc[:, col], rx)
        seeds = [matched(nat[a][k][:, FIT], nat[a][k][:, col], nat[b][k][:, FIT], nat[b][k][:, col], rx)[0] for k in range(len(S.SEEDS))]
        row[f"matched_{lens}_A_minus_B"] = {"point": pt, "ci95": np.percentile(dr, [2.5, 97.5]).tolist(),
                                            "per_seed_pairing": dict(zip(map(str, S.SEEDS), np.round(seeds, 4).tolist()))}
    out["pairs"].append(row)
path = f"runs/analysis/step13/posthoc_matched_load_{args.split}.json"
json.dump(out, open(path, "w"), indent=1)
print(f"wrote {path} ({out['status']})")
for r in out["pairs"]:
    L, N = r["matched_top_pick_load_A_minus_B"], r["matched_newness_A_minus_B"]
    print(f"{r['A']:11s} − {r['B']:8s} fit {r['fit_A']:.3f}/{r['fit_B']:.3f} | matched load {L['point']:+.4f} [{L['ci95'][0]:+.4f},{L['ci95'][1]:+.4f}] seeds {L['per_seed_pairing']}"
          f" | matched new {N['point']:+.4f} [{N['ci95'][0]:+.4f},{N['ci95'][1]:+.4f}]")
