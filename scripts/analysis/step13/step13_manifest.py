"""STEP13 manifests (STEP13_GRAPH_VALUE_PLAN.md §1 "Code and provenance", frozen 2026-10-21).

  design   before any STEP13 training or tuning: writes runs/analysis/step13/design_manifest.json — the plan hash, the committed
           code (commit, tree hash, SHA-256 of the library and every STEP13 script), the inputs hash, cells and their training
           commands, feature column lists (and the passed feature check), tuning grids, the 2018 cohort hash, slice boundaries
           (B₂ and the rare categories computed here from attributes, not outcomes), the matching grid and the test definitions.
           Refuses a dirty worktree (--draft writes design_manifest_DRAFT.json for review instead) and refuses once any s13 run
           exists.
  resolve  after tuning and training, before the phase-1 outcome evaluation: writes runs/analysis/step13/resolved_manifest.json —
           per cell and seed the run folder, configuration, checkpoint hash, training provenance (code and inputs hashes; for
           existing checkpoints their original provenance), the tuning ledgers, the passed backwards-compatibility check and the
           evaluation commit. The design manifest is never edited.
Usage: python scripts/analysis/step13/step13_manifest.py design | resolve
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file, resolve_run  # noqa: E402,F401
import argparse
import glob
import hashlib
import json
import subprocess
import time

import numpy as np

from referral_gnn import step13 as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["design", "resolve"])
ap.add_argument("--draft", action="store_true", help="design only: allow a dirty worktree and write design_manifest_DRAFT.json")
ap.add_argument("--inputs", default="prepared_data/three_node_v5/model_inputs_PRIVATE.pkl")
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
REPO = code_file("referral_gnn").parent
OUT = "runs/analysis/step13"; os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
SCRIPTS = ["scripts/train/three_node_train.py", "scripts/train/step12_lgbm.py", "scripts/train/step13_mlp.py", "scripts/evaluate/step10_evaluate.py",
           "scripts/evaluate/score_baselines11.py", "scripts/evaluate/step10_decide.py", "scripts/analysis/step13/step13_manifest.py",
           "scripts/analysis/step13/step13_compat.py", "scripts/analysis/step13/step13_verdicts.py", "scripts/checks/step13_feature_check.py",
           "scripts/tools/train_runlist.sh", "runlists/step13_gnn.tsv", "reproduce/13_graph_value.sh"]


def code_record(allow_dirty=False):
    dirty = git("status", "--porcelain")
    if dirty and not allow_dirty: sys.exit("REFUSED: the referral-gnn worktree has uncommitted changes; commit the reviewed STEP13 code first.\n" + dirty)
    from referral_gnn.train import code_hash
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "dirty": bool(dirty),
            "library_code_sha256": code_hash(str(code_file("referral_gnn"))),
            "library_files_sha256": {os.path.basename(p): sha(p) for p in sorted(glob.glob(str(REPO / "referral_gnn" / "*.py")))},
            "scripts_sha256": {s: sha(REPO / s) for s in SCRIPTS}}


if args.stage == "design":
    path = os.path.join(OUT, "design_manifest_DRAFT.json" if args.draft else "design_manifest.json")
    if os.path.exists(os.path.join(OUT, "design_manifest.json")): sys.exit("REFUSED: the design manifest is frozen and exists already.")
    started = [h for h in glob.glob("runs/*_s13_*") if os.path.isdir(h)]
    if started and not args.draft: sys.exit(f"REFUSED: STEP13 runs already exist ({started[:3]}); the design manifest precedes any tuning.")
    plan = os.path.join(root, "STEP13_GRAPH_VALUE_PLAN.md"); assert sha(plan) == S.PLAN_SHA256, "the plan differs from the frozen version"
    amend = os.path.join(root, "STEP13_AMENDMENTS.md"); assert sha(amend) == S.AMENDMENTS_SHA256, "the amendments differ from step13.AMENDMENTS_SHA256"
    fc = json.load(open(os.path.join(OUT, "feature_check.json"))); assert fc["passed"], "run scripts/checks/step13_feature_check.py first"
    from referral_gnn.three_node_train import load_three_node_inputs
    from referral_gnn.tabular import FeatureBuilder
    mi = load_three_node_inputs(os.path.join(root, args.inputs))
    Z = np.load("runs/analysis/step12c/dump/validation_candidate_scores_PRIVATE.npz")                   # the STEP10 2018 cohort (referral ids only)
    enc, pcp, day = Z["enc"], Z["pcp"], Z["day"]; H = S.History(mi)
    net = H.pcp_net(pcp, day); b2 = S.thin_network_bound(net)
    rare, counts = S.rare_categories(mi); inv = {v: k for k, v in mi.dx_vocab.index.items()}
    fsets = {k: FeatureBuilder(mi, k).names for k in ("tabular", "flat_rf", "full_nohist", "full", "insmatch")}
    man = {
        "step": "13", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "plan": S.PLAN, "plan_sha256": S.PLAN_SHA256,
        "amendments": S.AMENDMENTS, "amendments_sha256": S.AMENDMENTS_SHA256, "tests_record": S.tests_record(),
        "sensitivity_runs": S.SENSITIVITY_RUNS,
        "code": code_record(args.draft), "inputs": {"path": args.inputs, "sha256": sha(os.path.join(root, args.inputs))},
        "feature_check": {"path": os.path.join(OUT, "feature_check.json"), "sha256": sha(os.path.join(OUT, "feature_check.json")), "summary": fc},
        "cohort_2018": {"source": "STEP10 primary cohort (25 mi cap, 365-day activity, primary specialties), as in the STEP12c dump",
                        "n": int(len(enc)), "enc_sha256": hashlib.sha256(np.asarray(enc, np.int64).tobytes()).hexdigest()},
        "seeds": list(S.SEEDS), "seed_pairing": "by seed ID across models (a convention, not shared randomness)",
        "cells": {c: {**{k: v for k, v in d.items() if k != "train"},
                      "train_flags": d.get("train"), "runs": d["runs"] if d["runs"] else "chosen configuration's three seeds (runs/analysis/step13/mlp_<set>_tuning.json)"}
                  for c, d in S.CELLS.items()},
        "training": {
            "gnn": {"runlist": "runlists/step13_gnn.tsv", "final_model_flags": "--no-temporal --negatives uniform --epochs 100 --select firstobs_mrr_eligible "
                    "--lr-factor 0.5 --lr-patience 2 --patience 5 --graph-bucket week (scripts/tools/train_runlist.sh)",
                    "inputs": args.inputs, "explicit_history_ablations": S.HIST_ABL},
            "lgbm": {"command": "scripts/train/step12_lgbm.py --feature-set <set> --objective lambdarank --mode tune --extra-seeds 7",
                     "grid": {"num_leaves": [31, 127], "min_data_in_leaf": [50, 200], "rounds": [250, 500, 1000]}, "selection": "first-time MRR, 2018 selection sample, seed 42",
                     "seeds": "42 (tuning) and 1, 7 at the chosen configuration"},
            "mlp": {"command": "scripts/train/step13_mlp.py --feature-set <flat_rf|full> --mode tune", "grid": {"hidden": [128, 256], "dropout": [0.1, 0.3]},
                    "selection": "first-time MRR, 2018 selection sample, seed 42; ties: first in grid order", "fixed": {"layers": 2, "activation": "ReLU", "optimiser": "Adam",
                    "lr": 1e-3, "weight_decay": 1e-5, "lr_halving": "ReduceLROnPlateau(factor 0.5, patience 2): halved on the 3rd consecutive non-improving epoch, as the GNN",
                    "early_stop_after": 5, "max_epochs": 100, "batch_queries": 512,
                    "bundle": "21 rows (chosen + 20 uniform), softmax", "categoricals": "one-hot", "numerics": "standardised with training-only scalers; constant columns → 0",
                    "device": "cpu"}},
            "budgets_note": "GNN cells use the Steps 7–10 final settings; LightGBM 12 configurations; MLP 4 configurations — unequal search histories, disclosed"},
        "features": {k: {"n": len(v), "names": v} for k, v in fsets.items()},
        "slices": {
            "low_history": {"rule": "PCP prior referrals strictly before the referral day ≤ 951 (ties in; missing excluded from both)", "bound": S.LOW_HISTORY_MAX},
            "thin_network": {"rule": "distinct specialists referred to by the PCP before the referral day ≤ B2; B2 = 2018 cohort lower-tertile bound "
                             "(numpy quantile 1/3, linear interpolation)", "B2": b2},
            "rare": {"status": "descriptive only (amendment A2)", "rule": "the 11 least frequent of the 33 source categories by training-year (< 2018-01-01) encounter count; ties by dx id; "
                     "an encounter with any unknown category or none is excluded from both slice and rest",
                     "dx_ids": rare, "names": [inv.get(c) for c in rare], "training_year_counts": {str(c): counts[c] for c in sorted(counts)}},
            "min_size": {"referrals": S.MIN_SLICE_N, "pcps": S.MIN_SLICE_PCPS}},
        "matching": {"ranking": "0.7·z(model) + 0.3·z(as-of MIPS) + a·[new to PCP], z within the eligible set (candidates.py, b = 0)",
                     "grid": S.MATCH_GRID, "grid_note": "amendment A1: −0.60 … 1.00 step 0.01", "target_newness": S.TARGET_NEWNESS, "tolerance": S.TOLERANCE,
                     "tie_rule": "closest newness; ties to the smaller |a|, then the smaller a", "attainment_gate": S.ATTAIN_GATE,
                     "point_estimate_rule": "each model's per-seed 2018 point estimate must attain the tolerance", "test_2019": "a frozen from the 2018 full-data choice"},
        "tests": {"units": {u: [{k: v for k, v in c.items()} for c in comps] for u, comps in S.UNITS.items()}, "h3_reference": S.H3_REFERENCE,
                  "draws": S.DRAWS, "rng": {"pcp": S.RNG_PCP, "seed": S.RNG_SEED}, "alpha_holm": S.ALPHA, "m": len(S.UNITS),
                  "p_value": "(1 + #draws with contrast ≤ 0 or undefined) / (1 + draws); unit p = max over components",
                  "ci": "percentile 95% over defined draws (H3: nearest-grid contrast when the tolerance is missed); share left out reported",
                  "tost": {"delta": S.DELTA, "ci": "90% percentile", "status": "descriptive, unadjusted"},
                  "gates": {"undefined_share_max": S.UNDEFINED_GATE}, "seed_direction": "supported also requires every component's per-seed effect > 0 for all 3 seeds",
                  "untestable_unit": "p = 1 in Holm", "replication_2019": "directional: every component's 2019 point estimate > 0",
                  "headline_only_if_replicated": list(S.HEADLINE_ONLY_IF_REPLICATED)},
        "conventions": {"candidate_ranking_ties": "stable sort on the eligible-set order (candidates.Dump.top)", "unknown_categories": "their own category, as in training",
                        "zero_history": "aggregate means 0 with a *_nohist flag", "list_length": S.K,
                        "ins_share_denominator": "all of the candidate's past encounters (the A1 insurer one-hot mean), incl. the few with an unknown insurer"},
        "cost_estimate": {"gnn_runs": 10, "gnn_hours_mps": 10, "lgbm_families": 4, "lgbm_hours_cpu": 3, "mlp_runs": 12, "mlp_hours": "2-3", "evaluation_hours": 1,
                          "note": "re-checked after the first complete training run; actuals logged; overruns do not change the design"},
    }
    json.dump(man, open(path, "w"), indent=1)
    print(f"wrote {path}: B2 = {b2}, rare dx ids {rare}, feature columns " + ", ".join(f"{k} {len(v)}" for k, v in fsets.items()))

else:
    dpath = os.path.join(OUT, "design_manifest.json"); assert os.path.exists(dpath), "design manifest first"
    rpath = os.path.join(OUT, "resolved_manifest.json")
    if os.path.exists(rpath): sys.exit("REFUSED: the resolved manifest exists already.")
    design = json.load(open(dpath))
    compat = json.load(open(os.path.join(OUT, "compat.json"))); assert compat["passed"], "the backwards-compatibility check must pass first"
    assert design["tests_record"] == S.tests_record(), "test definitions in step13.py differ from the design manifest"
    drift = S.code_drift(design, str(REPO)); assert not drift, f"code changed since the design manifest: {drift}"
    from referral_gnn.three_node_model import ThreeNodeConfig
    expect_cfg = {"graph": {}, "ngp": {"use_graph": False, "pcp_in_scorer": True}, "gnn_nohist": {a: True for a in S.HIST_ABL},
                  "ngp_nohist": {"use_graph": False, "pcp_in_scorer": True, **{a: True for a in S.HIST_ABL}}}
    cells, runs = {}, {}
    for c, d in S.CELLS.items():
        if d["runs"] is None:
            t = json.load(open(os.path.join(OUT, f"mlp_{d['feature_set']}_tuning.json"))); names = dict(zip(map(str, S.SEEDS), t["cell_runs"]))
        else:
            names = {str(s): n for s, n in d["runs"].items()}
        cells[c] = {}
        for s, n in names.items():
            folder = os.path.basename(resolve_run(n).rstrip("/")); rd = os.path.join("runs", folder); cells[c][s] = folder
            m = json.load(open(os.path.join(rd, "run_manifest.json"))); summ = json.load(open(os.path.join(rd, "summary.json")))
            ck = next(p for p in ("best.pt", "model.txt") if os.path.exists(os.path.join(rd, p)))
            assert int(m["seed"] if "seed" in m else m["train_config"]["seed"]) == int(s), f"{folder}: seed differs from its seed ID {s}"
            assert m["inputs_sha256"] == design["inputs"]["sha256"], f"{folder}: trained on different inputs"
            new = "_s13_" in folder
            if new:
                assert m.get("code_sha256") == design["code"]["library_code_sha256"], f"{folder}: trained with code other than the design commit"
                if d["kind"] == "gnn":                                            # the cell's flags (final-model settings + the cell's own)
                    want = ThreeNodeConfig(temporal_edges=False, **expect_cfg[c]).to_dict()
                    assert ThreeNodeConfig.from_dict(m["model_config"]).to_dict() == want, f"{folder}: model configuration is not cell {c}'s"
                    assert m.get("arm") == f"model_s13_{c}" and m.get("neg_policy") == "uniform" and m.get("selection_metric") == "firstobs_mrr_eligible", folder
                else:
                    assert m.get("feature_set") == d["feature_set"], f"{folder}: feature set is not cell {c}'s"
                    if d["kind"] == "mlp": assert m.get("script_sha256") == design["code"]["scripts_sha256"]["scripts/train/step13_mlp.py"], f"{folder}: MLP script differs"
            runs[folder] = {"cell": c, "seed": int(s), "new_in_step13": new, "checkpoint": ck, "checkpoint_sha256": sha(os.path.join(rd, ck)),
                            "training_code_sha256": m.get("code_sha256"), "inputs_sha256": m["inputs_sha256"], "model_config": m.get("model_config"),
                            "params": m.get("params"), "feature_set": m.get("feature_set"), "hidden": m.get("hidden"), "dropout": m.get("dropout"),
                            "arm": m.get("arm"), "selection": summ.get("best_val_selection"), "tuning": m.get("tuning")}
        if c != "table":                                                   # the existing table seeds predate STEP13 (different trainers)
            if d["kind"] == "gnn":
                from referral_gnn.three_node_model import ThreeNodeConfig
                cfg = {json.dumps(ThreeNodeConfig.from_dict(runs[f]["model_config"]).to_dict(), sort_keys=True) for f in cells[c].values()}
            elif d["kind"] == "lgbm":
                cfg = {json.dumps(runs[f]["params"], sort_keys=True) for f in cells[c].values()}
            else:
                cfg = {(runs[f]["hidden"], runs[f]["dropout"]) for f in cells[c].values()}
            assert len(cfg) == 1, f"cell {c}: its seeds use different configurations"
    sensitivity = {nm: os.path.basename(resolve_run(f).rstrip("/")) for nm, f in S.SENSITIVITY_RUNS.items()}
    ledgers = {f"mlp_{fs}": json.load(open(os.path.join(OUT, f"mlp_{fs}_tuning.json"))) for fs in ("flat_rf", "full")}
    man = {"step": "13", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "design_manifest_sha256": sha(dpath), "cells": cells, "runs": runs,
           "sensitivity": sensitivity,
           "tuning_ledgers": {"lgbm": {c: runs[cells[c]["42"]]["tuning"] for c in ("flat_rf", "info_nohist", "info", "insmatch")}, **ledgers},
           "compat": {"path": os.path.join(OUT, "compat.json"), "sha256": sha(os.path.join(OUT, "compat.json"))},
           "evaluation_code": code_record(False),
           "note": "existing checkpoints (Graph s42/s1, G, Table) keep their original training provenance; they are loaded and evaluated with "
                   "the evaluation commit above, after the backwards-compatibility check"}
    json.dump(man, open(rpath, "w"), indent=1, default=str)
    print(f"wrote {rpath}: " + "; ".join(f"{c} {list(v.values())}" for c, v in cells.items()))
