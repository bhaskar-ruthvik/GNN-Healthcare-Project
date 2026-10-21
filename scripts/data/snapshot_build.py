"""STEP11 §11.4(c): build one cumulative snapshot's inputs, end to end, on the laptop (raw data never leaves it).

Sandbox: <workspace>/snapshots/<NAME>/ with symlinks to the workspace data/ and reference_data/ (the notebook is read from notebooks/), and its own
prepared_data/. Steps, in order (each refuses to overwrite):
  1. notebook   — the notebook's own code cells executed verbatim, in order, in one namespace, cwd = workspace. Exactly three
                  textual substitutions, each asserted to match once:
                    cell 15: split = train if day < TRAIN_END, validation if day < HOLDOUT_END, else test   (was 2017/2018/2019)
                    cell 27: the "chronological boundaries" check uses the same dates
                    cell 29: the manifest's split label records the dates
                  `display` is a no-op (aggregate reports are not printed). Optional --perturb-after: in memory only, every
                  source row on/after that date has the fields that feed fitted objects shuffled among those rows (leakage check b).
  2. build_inputs.py → build_mips_yardstick.py → build_three_node_inputs.py → v2 → v3 → v4 → v5, each run as a subprocess
     with REFERRAL_ROOT=<workspace> (builders read data/ and write prepared_data/ under the workspace; code is the project's).

Usage:
  python scripts/data/snapshot_build.py --name S16 --train-end 2016-10-01 --holdout-end 2017-01-01
  python scripts/data/snapshot_build.py --name REF --train-end 2018-01-01 --holdout-end 2019-01-01      # regression check (a)
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--train-end", required=True); ap.add_argument("--holdout-end", required=True)
ap.add_argument("--perturb-after", default=None, help="leakage check (b): shuffle post-date fields in memory before preparation")
ap.add_argument("--perturb-keep", default=None, help="regex of extra source columns left unshuffled in the leakage check (e.g. diagnosis/severity so v4's per-code sanity check holds)")
ap.add_argument("--stages", default="notebook,base,yardstick,v1,v2,v3,v4,v5")
args = ap.parse_args()

ROOT = workspace()                                    # data root; code comes from the repository
WS = ROOT / "snapshots" / args.name
STAGES = args.stages.split(",")
os.umask(0o077)
WS.mkdir(parents=True, exist_ok=True); (WS / "prepared_data").mkdir(exist_ok=True)
for link in ("data", "reference_data"):
    if not (WS / link).exists(): (WS / link).symlink_to(ROOT / link)
log = open(WS / "build_log.txt", "a")
def say(*a):
    msg = " ".join(str(x) for x in a); print(msg, flush=True); log.write(msg + "\n"); log.flush()
say(f"=== {args.name} train<{args.train_end} holdout<{args.holdout_end} perturb_after={args.perturb_after} stages={STAGES} {time.strftime('%Y-%m-%d %H:%M:%S')}")
json.dump(vars(args), open(WS / f"snapshot_spec_{time.strftime('%Y%m%dT%H%M%S')}.json", "w"), indent=1)


def run_notebook():
    nb = json.load(open(code_file("notebooks/01_data_cleaning.ipynb")))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 30, len(cells)
    TE, HE = args.train_end, args.holdout_end
    subs = [
        (15, '''history_events["split"] = np.select([years.le(2017),years.eq(2018),years.eq(2019)],''',
             f'''_d = history_events._service_day.dt.normalize()\nhistory_events["split"] = np.select([_d.lt(pd.Timestamp("{TE}")),_d.lt(pd.Timestamp("{HE}")),_d.ge(pd.Timestamp("{HE}"))],'''),
        (27, '''check("chronological boundaries",((encounter_features.split.eq("train") & encounter_features._service_day.dt.year.le(2017)) |
      (encounter_features.split.eq("validation") & encounter_features._service_day.dt.year.eq(2018)) |
      (encounter_features.split.eq("test") & encounter_features._service_day.dt.year.eq(2019))).all())''', None),
        (29, '''"split":{"train":"2014–2017","validation":"2018","test":"observed 2019"}''',
             f'''"split":{{"train":"before {TE}","validation":"{TE} to before {HE}","test":"from {HE}"}},"snapshot":{json.dumps(args.name)}'''),
    ]
    for i, old, new in subs:
        if new is None:                                   # cell 27: match whatever whitespace the notebook uses
            import re
            pat = re.compile(r'check\("chronological boundaries",.*?eq\(2019\)\)\)\.all\(\)\)', re.S)
            m = pat.findall(cells[i]); assert len(m) == 1, f"cell {i}: boundary check not found once"
            _d = "encounter_features._service_day.dt.normalize()"
            repl = (f'check("chronological boundaries",((encounter_features.split.eq("train") & {_d}.lt(pd.Timestamp("{TE}"))) | '
                    f'(encounter_features.split.eq("validation") & {_d}.ge(pd.Timestamp("{TE}")) & {_d}.lt(pd.Timestamp("{HE}"))) | '
                    f'(encounter_features.split.eq("test") & {_d}.ge(pd.Timestamp("{HE}")))).all())')
            cells[i] = pat.sub(lambda _m: repl, cells[i])
        else:
            assert cells[i].count(old) == 1, f"cell {i}: substitution target not found exactly once"
            cells[i] = cells[i].replace(old, new)
    ns = {"__name__": "__snapshot__", "display": lambda *a, **k: None}
    cwd = os.getcwd(); os.chdir(WS)
    try:
        for i, src in enumerate(cells):
            t0 = time.time()
            exec(compile(src, f"<notebook cell {i}>", "exec"), ns)
            if i == 1 and args.perturb_after:                # leakage check (b): in memory only
                import numpy as np, pandas as pd
                df = ns["df"]; d = pd.to_datetime(df["service_from_date"], errors="coerce")
                post = (d >= pd.Timestamp(args.perturb_after)).to_numpy(); rng = np.random.default_rng(20261008)
                skip = {"mso_membership_id", "primary_care_provider_npi", "rendering_provider_npi", "service_from_date", "DOC-DOC-LINK"}
                import re as _re
                loc = [c for c in df.columns if _re.search(r"zip|city|state|lat|lon|addr|office|county|ruca|location|cbsa", c, _re.I)
                       or (args.perturb_keep and _re.search(args.perturb_keep, c, _re.I))]
                for c in df.columns:                         # location fields are static provider attributes the notebook
                    if c in skip or c in loc: continue       # requires to be consistent; they are left unshuffled (disclosed)
                    v = df.loc[post, c].to_numpy(); df.loc[post, c] = v[rng.permutation(len(v))]
                say(f"  perturbed {int(post.sum()):,} source rows on/after {args.perturb_after} (all non-key, non-location fields shuffled among them; {len(loc)} fields kept (location{" + " + args.perturb_keep if args.perturb_keep else ""}); in memory)")
            say(f"  notebook cell {i:2d} done ({time.time() - t0:.0f}s)")
    finally:
        os.chdir(cwd)


def stage_mapping():
    """The disease-code → category mapping (from the source's own category column; not fitted on any split) is copied in."""
    import shutil
    dst = WS / "prepared_data" / "three_node_v1"; dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "disease_mapping_PRIVATE.json").exists():
        shutil.copy2(ROOT / "prepared_data" / "three_node_v1" / "disease_mapping_PRIVATE.json", dst / "disease_mapping_PRIVATE.json")
    say("  disease mapping staged")


SIDE_FILES = {   # row-level extractions from the raw sources, keyed by patient/provider/day/source row; not fitted on any split
    "v2": ("three_node_v2", ["patient_zip_PRIVATE.pkl", "patient_zip_manifest.json", "specialist_insurer_days_PRIVATE.pkl", "hospital_affiliations_PRIVATE.pkl"]),
    "v4": ("three_node_v4", ["row_severity_PRIVATE.pkl", "patient_race_PRIVATE.pkl", "extraction_manifest.json"]),
}


def stage_side_files(st):
    import shutil
    d, names = SIDE_FILES[st]; dst = WS / "prepared_data" / d; dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        if not (dst / n).exists(): shutil.copy2(ROOT / "prepared_data" / d / n, dst / n)
    say(f"  side files staged for {st}: {names}")


def run_builder(script):
    t0 = time.time(); env = dict(os.environ, REFERRAL_ROOT=str(WS), PYTHONWARNINGS="ignore")
    r = subprocess.run([sys.executable, "-W", "ignore", str(script_path(script))], env=env, cwd=str(code_file(".")), capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
    say(f"  {script}: exit {r.returncode} ({time.time() - t0:.0f}s)\n    " + tail.replace("\n", "\n    "))
    if r.returncode != 0: raise SystemExit(f"{script} failed")


if "notebook" in STAGES: run_notebook()
for st, script in (("base", "build_inputs.py"), ("yardstick", "build_mips_yardstick.py"), ("v1", "build_three_node_inputs.py"),
                   ("v2", "build_three_node_v2_inputs.py"), ("v3", "build_three_node_v3_inputs.py"), ("v4", "build_three_node_v4_inputs.py"),
                   ("v5", "build_three_node_v5_inputs.py")):
    if st in STAGES:
        if st == "v1": stage_mapping()
        if st in SIDE_FILES: stage_side_files(st)
        run_builder(script)
say(f"=== {args.name} done {time.strftime('%H:%M:%S')}")
