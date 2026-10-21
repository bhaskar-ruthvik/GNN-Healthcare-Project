"""Where code and data live.

Code is this repository (REPO). Data live in a *workspace* folder with this layout (all git-ignored):
    data/             raw source workbooks (read-only; never committed)
    reference_data/   public reference files (ZCTA centroids)
    prepared_data/    model inputs built from the raw data (*_PRIVATE.pkl)
    runs/             one folder per training run (checkpoints, manifests) and runs/analysis/ (evaluation outputs)
    snapshots/        yearly snapshot sandboxes (STEP11 §11.4(c))
    deliverables/     workbook and decks built from runs/analysis/
Resolution order for the workspace:
    1. REFERRAL_ROOT           set by scripts/data/snapshot_build.py so the input builders read and write a snapshot's sandbox;
    2. REFERRAL_GNN_WORKSPACE  where you keep the data;
    3. the repository root     (default).
Every script resolves its relative paths against workspace(); code files (scripts, templates, notebook) come from REPO.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def workspace() -> Path:
    return Path(os.environ.get("REFERRAL_ROOT") or os.environ.get("REFERRAL_GNN_WORKSPACE") or REPO).resolve()


def enter_workspace() -> str:
    ws = workspace(); os.chdir(ws); return str(ws)


def code_file(rel: str) -> Path:
    """A file or folder of the repository. Falls back to a unique match by file name (scripts moved into role folders)."""
    p = REPO / rel
    if p.exists(): return p
    hits = [q for q in REPO.rglob(Path(rel).name) if "archive" not in q.parts]
    if len(hits) != 1: raise FileNotFoundError(f"{rel}: {len(hits)} matches in the repository")
    return hits[0]


def script_path(name: str) -> Path:
    """A script by file name, wherever it sits under scripts/."""
    hits = list((REPO / "scripts").rglob(name))
    if len(hits) != 1: raise FileNotFoundError(f"scripts/**/{name}: {len(hits)} matches")
    return hits[0]


def resolve_run(ref: str) -> str:
    """A run folder given as a path or by its run name, e.g. 'lgbm_tabular_s42' → 'runs/<timestamp>_lgbm_tabular_s42' in the
    workspace. Lets run lists and commands name runs without their timestamps. The returned path is valid from the current
    directory (relative when inside the workspace). An exact folder name under runs/ is also accepted. Incomplete runs (suffix _INCOMPLETE) are never matched."""
    import glob
    if ref and os.path.isdir(ref): return ref
    ws = workspace()
    if ref and (ws / ref).is_dir(): return os.path.relpath(ws / ref)
    if ref and (ws / "runs" / ref).is_dir(): return os.path.relpath(ws / "runs" / ref)      # an exact run folder name (with its timestamp)
    hits = [h for h in sorted(glob.glob(str(ws / "runs" / f"*_{ref}"))) if os.path.isdir(h) and not h.endswith("_INCOMPLETE")]
    if len(hits) != 1: raise FileNotFoundError(f"run '{ref}': {len(hits)} matching folders under {ws / 'runs'}")
    return os.path.relpath(hits[0])
