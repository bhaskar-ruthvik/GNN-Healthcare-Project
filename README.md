# Referral GNN

Specialist recommendations for primary-care referrals, built on a heterogeneous graph of PCPs, encounters and specialists
(HGT), and compared with a strong tabular model. Every model is judged on four lenses rather than on an accuracy score:
- **quality:** MIPS gain over the specialist actually chosen;
- **load:** the share of top picks going to the busiest 10% of specialists;
- **fit:** whether the chosen specialist is in the list;
- **travel.**

The main results:
- A graph model concentrates load less than a tree model at equal fit, although the graph structure itself is not the reason.
- The two model types make different mistakes.
- An equal blend of the two, steered towards specialists who are new to the PCP and less busy, roughly halves top-pick load. It does this at the table's newness, higher MIPS and a fit within 1–2 points, and the result holds on a held-out year.

`docs/METHODS_APPENDIX.md` has every definition. `REPRODUCE.md` maps each result to the command that produces it.

> **Data are not in this repository.** The work uses a private ACO Medicare claims dataset (Dallas–Fort Worth, 2014–2019).
> Raw data, prepared inputs, run outputs and checkpoints are git-ignored and must never be committed. The code runs against a
> separate *workspace* folder that holds them (see below).

## Layout

```
referral_gnn/          library (data inputs, model, trainer, evaluation, shared definitions)
  paths.py             where code and data live: the repository vs the data workspace
  step10.py            evaluation cohort, eligible sets, MIPS yardstick (STEP10 protocol)
  three_node_*.py      three-node graph inputs (v1–v6), model and trainer
  tabular.py           tabular baselines (LightGBM, logistic regression, ALS, item-kNN) and the flattened graph
  candidates.py        candidate-score dumps and the ranking arithmetic of the hybrid analyses (blend, steering, frontiers)
  data.py, model.py, train.py, evaluate.py, baselines.py   four-node pipeline (Steps 3–6; the prepared bundle pickles these)
scripts/
  data/                raw workbooks → prepared inputs (cleaning notebook, input builders v1–v6, yearly snapshots)
  train/               GNN trainer, tabular baselines, LightGBM cells, hybrid models
  evaluate/            the evaluator (all lenses; 2018, then 2019 through a recorded decision), decision recorder, plots
  analysis/step10..12  one script per analysis, each writing aggregates to runs/analysis/
  checks/              model-level and snapshot checks (leakage, exact reproduction)
  deliverables/        verification workbook and the HTML decks (templates under presentation/templates/)
  tools/               run-list trainer, detached launcher
  legacy/              Steps 3–9 (the earlier MRR-era pipeline behind docs/RESULTS.md)
runlists/              training run lists (name, inputs, flags) for every trained model
reproduce/             one script per step with the exact commands (see REPRODUCE.md)
notebooks/             01_data_cleaning.ipynb (outputs stripped)
docs/                  frozen plans and protocols, architecture, methods appendix, reviews and results write-ups
archive/               superseded one-off code and the original session run chains (kept for the record)
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export REFERRAL_GNN_WORKSPACE=/path/to/workspace     # folder with data/, reference_data/, prepared_data/, runs/
```

Without `pip install -e .`, scripts still run, because each one finds the repository root itself. Without
`REFERRAL_GNN_WORKSPACE`, the workspace is the repository root (its data folders are git-ignored).

**Workspace layout:**
- `data/`: raw workbooks (private, read-only).
- `reference_data/`: Census 2020 ZCTA gazetteer.
- `prepared_data/`: model inputs.
- `runs/<timestamp>_<name>/`: one folder per trained model (manifest, summary, checkpoints).
- `runs/analysis/`: evaluation and analysis outputs.
- `snapshots/`: yearly sandboxes.
- `deliverables/`: workbook and decks.

**Conventions every script follows:**
- **Paths:** relative paths resolve against the workspace (`referral_gnn/paths.py`).
- **Run names:** runs are named without timestamps. Anywhere a run is expected, `v5_unmasked_s42` resolves to `runs/<timestamp>_v5_unmasked_s42`.
- **Manifests:** each output is written with a manifest (input SHA-256, code SHA-256, configuration, seed).
- **No overwriting:** builders refuse to overwrite completed outputs.
- **Two-pass guard:** the evaluator runs on 2019 only after `step10_decide.py` records the 2018 decision with the SHA-256 of every 2018 output, and only once.
- **Aggregates only:** files named `*_PRIVATE*` hold referral- or provider-level values and stay in the workspace.

**Platform notes:**
- Results were produced on Apple silicon (MPS). MPS is not bit-deterministic, so same-seed reruns differ slightly; the `*_dup` runs measure this.
- torch and LightGBM are kept in separate processes (macOS libomp clash). The evaluator scores LightGBM models in a subprocess.

## Where to start

- **Results to check:** `REPRODUCE.md`, then the verification workbook (`scripts/deliverables/build_verification_workbook.py`).
- **The method:** `docs/METHODS_APPENDIX.md`, `docs/ARCHITECTURE.md`, `docs/STEP10_PROTOCOL.md`.
- **What was decided before seeing results:** the frozen plans in `docs/` (`STEP*_PLAN.md`, `STEP*_PROTOCOL.md`), the reviews in `docs/reviews/`, and the decision log written to `runs/analysis/step10_decisions.json`.
