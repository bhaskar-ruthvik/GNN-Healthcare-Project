"""STEP13 backwards-compatibility check (STEP13_GRAPH_VALUE_PLAN.md §1, frozen 2026-10-21): before the phase-1 outcome
evaluation, the existing checkpoints (Graph s42 / s1, G, Table) are re-scored on 2018 with the current code and compared with
what STEP12 recorded (criterion as amended by A3, STEP13_AMENDMENTS.md):
  - Graph and Table: on identical referrals and candidates, candidate scores within 1e-5 of the STEP12c dump AND every referral's
    top-5 list (in order) and native lenses identical (GPU float32 reduction order moves GNN scores by ~1e-6; rankings must not move);
  - G (never dumped): its per-referral lens values (fit, top-pick load, newness, MIPS gain) against STEP12a/final (≤ 1e-6, same
    missing-value pattern).
The re-scoring pass writes to runs/analysis/step13/compat (2018 only; no 2019). Result: runs/analysis/step13/compat.json.
Usage: python scripts/analysis/step13/step13_compat.py
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import subprocess

import numpy as np
import pandas as pd

from referral_gnn import step13 as S  # noqa: E402
from referral_gnn.candidates import Dump  # noqa: E402

root = str(workspace()); os.chdir(root)
OUT = "runs/analysis/step13/compat"
OLD_DUMP = "runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz"
OLD_PER = "runs/analysis/step12a/final/validation_per_referral_PRIVATE.csv.gz"
runs = [S.CELLS[c]["runs"][s] for c in ("graph", "table") for s in (42, 1, 7) if "_s13_" not in "_" + S.CELLS[c]["runs"][s]] + \
       [S.CELLS["g"]["runs"][s] for s in S.SEEDS]
if not os.path.exists(os.path.join(OUT, "validation_candidate_scores_PRIVATE.npz")):
    subprocess.run([sys.executable, "-W", "ignore", str(script_path("step10_evaluate.py")), "--split", "validation", "--step", "13_compat",
                    "--dump-scores", "--out", OUT, *runs], check=True)
new, old = np.load(os.path.join(OUT, "validation_candidate_scores_PRIVATE.npz")), np.load(OLD_DUMP)
res = {"runs": runs, "scores": {}, "per_referral": {}}
for k in ("enc", "offsets", "spec", "target_pos"):
    assert np.array_equal(new[k], old[k]), f"re-scored dump differs from STEP12c in {k}"
Dn, Do = Dump(os.path.join(OUT, "validation_candidate_scores_PRIVATE.npz")), Dump(OLD_DUMP)
res["rankings"] = {}
for k in old.files:
    if k.startswith("score::") and k in new.files:
        res["scores"][k[7:]] = float(np.abs(new[k].astype(np.float64) - old[k].astype(np.float64)).max())
        bn, bo = Dn.weighted(Dn.scores(k)), Do.weighted(Do.scores(k))
        same_top = int(sum(np.array_equal(Dn.top(bn, i), Do.top(bo, i)) for i in range(Dn.n_ref)))
        ln, lo = S.native(Dn, k), S.native(Do, k)
        same_lens = int(np.all((ln == lo) | (np.isnan(ln) & np.isnan(lo)), axis=1).sum())
        res["rankings"][k[7:]] = {"referrals": int(Dn.n_ref), "top5_identical": same_top, "native_lenses_identical": same_lens}
need = {S.CELLS[c]["runs"][s] for c in ("graph", "table") for s in (42, 1, 7)} - {S.CELLS["graph"]["runs"][7]}
assert need <= set(res["scores"]), f"STEP12c dump lacks {need - set(res['scores'])}"
pn, po = pd.read_csv(os.path.join(OUT, "validation_per_referral_PRIVATE.csv.gz")), pd.read_csv(OLD_PER)
assert np.array_equal(pn.enc.to_numpy(), po.enc.to_numpy()), "per-referral outputs: referral order differs"
for r in S.CELLS["g"]["runs"].values():
    diffs = {}
    for m in ("r5", "busy1", "new_pcp5", "gain5"):
        col = f"model:{r}|{m}"
        assert col in pn.columns and col in po.columns, f"missing {col}"
        a_, b_ = pn[col].to_numpy(float), po[col].to_numpy(float)
        assert np.array_equal(np.isnan(a_), np.isnan(b_)), f"{col}: missing values differ"
        diffs[m] = float(np.nanmax(np.abs(a_ - b_))) if (~np.isnan(a_)).any() else 0.0
    res["per_referral"][r] = diffs
res["criterion"] = "A3: scores within 1e-5 and identical top-5 lists and native lenses (Graph, Table); G per-referral lenses within 1e-6"
res["tolerance_scores"] = 1e-5
res["passed"] = bool(max(res["scores"].values()) <= 1e-5
                     and all(r["top5_identical"] == r["referrals"] == r["native_lenses_identical"] for r in res["rankings"].values())
                     and all(v <= 1e-6 for d in res["per_referral"].values() for v in d.values()))
json.dump(res, open("runs/analysis/step13/compat.json", "w"), indent=1)
print(json.dumps(res, indent=1))
assert res["passed"], "backwards-compatibility check failed"
