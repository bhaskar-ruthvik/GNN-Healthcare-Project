"""STEP14 (Implementation/STEP14_GRAPH_FEATURE_BLEND_PLAN.md, frozen 2026-10-23): does a steered recommender that uses the
graph-feature table (Flat-RF) beat the STEP12c steered blend? Scores come from the STEP13 candidate-score dumps (no new models).

  --split validation   2018: lenses for 6 families × 64 steering settings × 3 seed pairings; per-family setting by the STEP12c
                       rule (fit ≥ 0.45 and newness ≥ 0.35 in all pairings, lowest mean top-pick load; ties smaller a + b, then
                       smaller b); the proposed recommender (lowest chosen load among families reaching the floors; ties fewer
                       components); frontiers. Writes runs/analysis/step14/choice_2018.json (once).
  --split decide       records runs/analysis/step14/decision.json with the SHA-256 of the 2018 choice (once).
  --split test         2019, once, at the 2018 settings: lenses per pairing, the pre-stated comparison with B_GT, paired PCP
                       bootstrap of the differences. Writes runs/analysis/step14/results_2019.json.
Steering arithmetic as candidates.Dump.top (base + a·new − b·busy, stable ties), vectorised over the 64 settings.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import hashlib
import json
import time

import numpy as np

from referral_gnn.candidates import Dump, zs, STEERING_GRID, pcp_bootstrap_index  # noqa: E402

PLAN_SHA256 = "e273502bb46b3194c68ea4e5f4de24842c9437598d6b4d90bc2fccd1d7102aa0"
SEEDS = (42, 1, 7)
FAMILIES = {"G": ("graph",), "T": ("table",), "F": ("flat_rf",), "B_GT": ("graph", "table"), "B_GF": ("graph", "flat_rf"),
            "B_GTF": ("graph", "table", "flat_rf")}
FIT_FLOOR, NEW_FLOOR = 0.45, 0.35
LENSES = ("fit", "busy1", "busy5", "new5", "gain5", "good5")
SETTINGS = [(a, b) for a in STEERING_GRID for b in STEERING_GRID]

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["validation", "decide", "test"])
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
OUT = "runs/analysis/step14"; os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
assert sha(os.path.join(root, "STEP14_GRAPH_FEATURE_BLEND_PLAN.md")) == PLAN_SHA256, "the plan differs from the frozen version"
CHOICE, DECISION, RESULTS = (os.path.join(OUT, f) for f in ("choice_2018.json", "decision.json", "results_2019.json"))

if args.split == "decide":
    assert os.path.exists(CHOICE) and not os.path.exists(DECISION), "decide once, after the 2018 choice"
    json.dump({"step": "14", "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"), "choice_2018_sha256": sha(CHOICE), "plan_sha256": PLAN_SHA256,
               "script_sha256": sha(__file__)},
              open(DECISION, "w"), indent=1)
    print("recorded", DECISION); sys.exit(0)
if args.split == "validation": assert not os.path.exists(CHOICE), "REFUSED: the 2018 choice exists (written once)"
else:
    dec = json.load(open(DECISION)) if os.path.exists(DECISION) else {}
    assert dec.get("choice_2018_sha256") == sha(CHOICE), "REFUSED: no decision entry for the current 2018 choice"
    assert dec.get("script_sha256") == sha(__file__), "REFUSED: the script changed after the decision"
    assert not os.path.exists(RESULTS), "REFUSED: the 2019 check runs once"

runs = json.load(open("runs/analysis/step13/resolved_manifest.json"))["cells"]
D = Dump(f"runs/analysis/step13/eval/{args.split}_candidate_scores_PRIVATE.npz")
A = np.array([s[0] for s in SETTINGS])[:, None]; Bb = np.array([s[1] for s in SETTINGS])[:, None]


def base_scores(cells, k):
    """Per referral: the family's score at the 30% quality weight for seed pairing k (candidates.py arithmetic)."""
    S_ = [D.scores("score::" + runs[c][str(SEEDS[k])]) for c in cells]
    if len(cells) == 1: return D.weighted(S_[0])
    if len(cells) == 2: return D.blend(S_[0], S_[1])
    return [0.7 * zs(sum(zs(s[a:b]) for s in S_) / len(S_)) + 0.3 * zs(D.ma[a:b]) for a, b in D.seg]


def lenses(base):
    """(n_ref, 64 settings, 6 lenses) per referral."""
    out = np.zeros((D.n_ref, len(SETTINGS), len(LENSES)))
    for i, (a, b) in enumerate(D.seg):
        nw, bz, mp, no = D.new[a:b], D.busy[a:b], D.mips[a:b], D.none[a:b]
        top = np.argsort(-(base[i][None, :] + (A * nw[None, :] - Bb * bz[None, :])), axis=1, kind="stable")[:, :5]
        good = (nw > 0) & (mp > D.refm[i]) & ~no & (bz == 0)
        out[i, :, 0] = (top == D.tpos[i]).any(1) if D.tpos[i] >= 0 else 0.0
        out[i, :, 1] = bz[top[:, 0]]; out[i, :, 2] = bz[top].mean(1); out[i, :, 3] = nw[top].mean(1)
        out[i, :, 4] = mp[top].mean(1) - D.refm[i]; out[i, :, 5] = good[top].mean(1)
    return out


t0 = time.time()
PER = {f: [lenses(base_scores(cells, k)) for k in range(len(SEEDS))] for f, cells in FAMILIES.items()}   # family → [pairing] → array
MEAN = {f: np.stack([p.mean(0) for p in PER[f]]) for f in FAMILIES}                                      # (3, 64, 6)
print(f"lenses for {len(FAMILIES)} families × {len(SETTINGS)} settings × {len(SEEDS)} pairings in {time.time() - t0:.0f}s", flush=True)
li = {k: j for j, k in enumerate(LENSES)}

if args.split == "validation":
    choice, frontiers = {}, {}
    for f in FAMILIES:
        M = MEAN[f]; ok = (M[:, :, li["fit"]] >= FIT_FLOOR).all(0) & (M[:, :, li["new5"]] >= NEW_FLOOR).all(0)
        if ok.any():
            cand = [s for s in range(len(SETTINGS)) if ok[s]]
            s = min(cand, key=lambda s: (M[:, s, li["busy1"]].mean(), sum(SETTINGS[s]), SETTINGS[s][1]))
            choice[f] = {"a_new": SETTINGS[s][0], "b_busy": SETTINGS[s][1], "settings_reaching_floors": int(ok.sum()),
                         "lenses_mean": dict(zip(LENSES, M[:, s].mean(0).round(5).tolist())),
                         "lenses_per_pairing": {str(SEEDS[k]): dict(zip(LENSES, M[k, s].round(5).tolist())) for k in range(len(SEEDS))}}
        else:
            choice[f] = {"cannot_reach_floors": True, "best_fit_unsteered": float(M[:, SETTINGS.index((0, 0)), li["fit"]].mean())}
        frontiers[f] = {str(SEEDS[k]): {str(t): (float(min((MEAN[f][k, s, li["busy1"]] for s in range(len(SETTINGS))
                                                             if MEAN[f][k, s, li["fit"]] >= FIT_FLOOR and MEAN[f][k, s, li["new5"]] >= t), default=np.nan)))
                                         for t in (0.30, 0.35, 0.40)} for k in range(len(SEEDS))}
    reach = [f for f in FAMILIES if not choice[f].get("cannot_reach_floors")]
    proposed = min(reach, key=lambda f: (choice[f]["lenses_mean"]["busy1"], len(FAMILIES[f]))) if reach else None
    out = {"step": "14", "split": "validation", "plan_sha256": PLAN_SHA256, "n_referrals": int(D.n_ref), "dump_sha256": sha(D.path),
           "pairings": {str(s): {c: runs[c][str(s)] for c in ("graph", "table", "flat_rf")} for s in SEEDS},
           "floors": {"fit": FIT_FLOOR, "newness": NEW_FLOOR}, "choice": choice, "proposed": proposed,
           "frontier_lowest_top_pick_load": frontiers, "unsteered": {f: dict(zip(LENSES, MEAN[f][:, SETTINGS.index((0, 0))].mean(0).round(5).tolist())) for f in FAMILIES}}
    json.dump(out, open(CHOICE, "w"), indent=1, default=float)
    print("wrote", CHOICE, "| proposed:", proposed)
    for f, c in choice.items():
        print(f"  {f:6s}", "cannot reach the floors (unsteered fit %.3f)" % c["best_fit_unsteered"] if c.get("cannot_reach_floors")
              else f"a={c['a_new']} b={c['b_busy']} " + " ".join(f"{k} {v:.3f}" for k, v in c["lenses_mean"].items()))
else:
    C = json.load(open(CHOICE)); prop = C["proposed"]; idx = {}
    res = {"step": "14", "split": "test", "plan_sha256": PLAN_SHA256, "n_referrals": int(D.n_ref), "dump_sha256": sha(D.path),
           "choice_2018_sha256": sha(CHOICE), "proposed_2018": prop, "families": {}}
    for f, c in C["choice"].items():
        if c.get("cannot_reach_floors"): res["families"][f] = {"cannot_reach_floors_2018": True}; continue
        s = SETTINGS.index((c["a_new"], c["b_busy"])); idx[f] = s
        res["families"][f] = {"a_new": c["a_new"], "b_busy": c["b_busy"], "lenses_mean": dict(zip(LENSES, MEAN[f][:, s].mean(0).round(5).tolist())),
                              "lenses_per_pairing": {str(SEEDS[k]): dict(zip(LENSES, MEAN[f][k, s].round(5).tolist())) for k in range(len(SEEDS))}}
    if prop and prop != "B_GT" and "B_GT" in idx:
        P, R = MEAN[prop][:, idx[prop]], MEAN["B_GT"][:, idx["B_GT"]]
        improves = bool(((P[:, li["busy1"]] < R[:, li["busy1"]]) & (P[:, li["fit"]] >= R[:, li["fit"]] - 0.01) & (P[:, li["new5"]] >= R[:, li["new5"]] - 0.01)).all())
        codes, nc, ix = pcp_bootstrap_index(D.pcp, seed=20261023, draws=2000)
        W = np.stack([np.bincount(r, minlength=nc) for r in ix]).astype(float); cnt = np.bincount(codes, minlength=nc).astype(float)
        diff = np.mean([PER[prop][k][:, idx[prop]] - PER["B_GT"][k][:, idx["B_GT"]] for k in range(len(SEEDS))], axis=0)   # (n_ref, 6)
        boot = {}
        for l in ("busy1", "new5", "fit", "gain5"):
            s_ = np.bincount(codes, weights=diff[:, li[l]], minlength=nc)
            boot[l] = {"point": float(s_.sum() / cnt.sum()), "ci95": np.percentile((W @ s_) / (W @ cnt), [2.5, 97.5]).tolist()}
        res["comparison_with_B_GT"] = {"improves_on_step12c_blend": improves, "rule": "lower top-pick load in all 3 pairings; fit and newness ≥ B_GT − 1 pt in each",
                                       "proposed_minus_B_GT_paired_bootstrap": boot}
    else:
        res["comparison_with_B_GT"] = {"improves_on_step12c_blend": False, "note": "the 2018 rule proposed B_GT itself (or no family reached the floors)"}
    json.dump(res, open(RESULTS, "w"), indent=1, default=float)
    print("wrote", RESULTS, "| proposed (2018):", prop, "| improves on B_GT:", res["comparison_with_B_GT"]["improves_on_step12c_blend"])
    for f, c in res["families"].items():
        print(f"  {f:6s}", "—" if c.get("cannot_reach_floors_2018") else " ".join(f"{k} {v:.3f}" for k, v in c["lenses_mean"].items()))
    if "proposed_minus_B_GT_paired_bootstrap" in res["comparison_with_B_GT"]: print(json.dumps(res["comparison_with_B_GT"]["proposed_minus_B_GT_paired_bootstrap"]))
