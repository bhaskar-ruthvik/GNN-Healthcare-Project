"""Record a Step 10 validation decision (STEP10_PROTOCOL §3 two-pass procedure; amendment A2). Appends one timestamped entry
to runs/analysis/step10_decisions.json; never edits or removes earlier entries.

Structured fields read from the validation summary (never typed by hand): primary_contrast_config (the §1 rule output, verbatim),
best_admissible_model, best_admissible_heuristic, pcp_r5_floor, matched_pairs, and the SHA-256 of every validation_* output.
--final names the final configuration for 10.4–10.6 (must be an arm evaluated in the summary); --amendment cites its basis.
The test pass of step10_evaluate.py refuses to run without an entry for the same step and runs, or if validation outputs changed.
Usage: step10_decide.py --step 10.2 --summary runs/analysis/step10_2/validation_summary.json --final model_masked:blend0.3 \
         --amendment "STEP10_PROTOCOL A2 (2026-09-29)" [--note "..."]
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, hashlib, json, os
from datetime import datetime, timezone
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--step", required=True); ap.add_argument("--summary", required=True)
ap.add_argument("--final", required=True); ap.add_argument("--amendment", required=True); ap.add_argument("--note", default="")
ap.add_argument("--primary", default=None, help="override the rule-selected primary contrast (must cite an amendment; the rule output is kept in the entry)")
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
PATH = "runs/analysis/step10_decisions.json"
s = json.load(open(args.summary)); d = os.path.dirname(args.summary)
assert s["split"] == "validation", "decisions are recorded from validation outputs only"
assert s["step"] == args.step, f"summary is for step {s['step']}, not {args.step}"
assert not any(f.startswith("test_") for f in os.listdir(d)), "test outputs already exist; a decision cannot follow test output"
arms = pd.read_csv(os.path.join(d, "validation_arms.csv"))
assert args.final in set(arms.arm), f"--final {args.final} is not an evaluated arm"
assert args.primary is None or args.primary in set(arms.arm), f"--primary {args.primary} is not an evaluated arm"
files = {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() for f in sorted(os.listdir(d)) if f.startswith("validation_")}
entry = {"step": args.step, "runs": sorted(s["runs"]), "recorded_utc": datetime.now(timezone.utc).isoformat(),
         "primary_contrast_config": args.primary or s["chosen"], "rule_output_primary": s["chosen"], "final_config_10_4_to_10_6": args.final,
         "best_admissible_model": s["best_admissible_model"], "best_admissible_heuristic": s["best_admissible_heuristic"],
         "pcp_r5_floor": s["pcp_r5_floor"], "matched_pairs": s["matched_pairs"], "n_admissible": s["n_admissible"],
         "rule": "STEP10_PROTOCOL §1 admissibility on validation; max validation gain5_cons", "amendment": args.amendment,
         "validation_files_sha256": files, "note": args.note, "a4_v4_vs_v5": s.get("a4_v4_vs_v5"), "a10": s.get("a10")}
dec = json.load(open(PATH)) if os.path.exists(PATH) else {"entries": []}
dec["entries"].append(entry); json.dump(dec, open(PATH, "w"), indent=2)
print(json.dumps({k: v for k, v in entry.items() if k not in ("validation_files_sha256", "matched_pairs")}, indent=2))
