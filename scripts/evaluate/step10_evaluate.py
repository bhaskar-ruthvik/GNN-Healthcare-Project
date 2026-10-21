"""Step 10.2 / 10.3 evaluation on the MIPS yardstick (STEP10_PROTOCOL.md, frozen 2026-09-28; amendments A1, A2).

Every scorer is turned into a ranking over each referral's eligible set (§2.2), and every ranking into two weight vectors over
that set: top-1 weights (sum 1) and top-5 slot weights (sum 5). Deterministic scorers give 0/1 weights; the random scorer gives
its exact expectation (1/n and 5/n). All metrics are weighted sums, so the random row is exact rather than simulated.

Scorers (§2.5): practice (reference) · pcp (prior PCP referrals; ties by distance, then fixed-seed random) · random ·
mips (as-of MIPS, ties by distance) · dist (nearest first) · model:<run> · <base>+resortK (top K re-sorted by as-of MIPS,
ties by the base order, rest in base order) · <base>+blendλ ((1−λ)·z(base score) + λ·z(as-of MIPS); for pcp the base score is
log1p(prior count), A2). No scorer uses load (A1).

Two passes (§3):
  --split validation  computes admissibility and the rule-selected configuration; record with scripts/evaluate/step10_decide.py.
  --split test        refuses unless a decision entry exists for this step and exactly these runs, the current validation
                      outputs match the recorded hashes, and no test outputs exist yet. Contrasts are computed for the
                      configurations named in the entry; admissibility on test is descriptive only (never re-selected).
Outputs (runs/analysis/step10_2/ by default): <split>_{configs,arms,bands,sensitivity,gain_hist}.csv, <split>_summary.json,
<split>_per_referral_PRIVATE.csv.gz.
Usage: step10_evaluate.py --split validation runs/<run> [...] [--step 10.2] [--bootstrap 1000]
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, hashlib, json, os, sys, time
from collections import defaultdict
import numpy as np, pandas as pd, torch

from referral_gnn.data import AsOfIndex, to_day  # noqa: E402
from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.evaluate import ModelScorer  # noqa: E402
from referral_gnn.train import pick_device, sha256_file  # noqa: E402
from referral_gnn.step10 import (Step10Cohort, load_yardstick, specialty_names, specialty_strict, first_time_queries,  # noqa: E402
                                 DEFAULT_YARDSTICK, LIST_SIZE, yardstick_path_for)

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--split", required=True, choices=["validation", "test"])
ap.add_argument("--step", default="10.2"); ap.add_argument("--bootstrap", type=int, default=1000); ap.add_argument("--out", default="runs/analysis/step10_2")
ap.add_argument("--tag-version", action="store_true", help="10.3 (A4): tag model arms by input version (v4/v5) and put *_dup runs in their own arm")
ap.add_argument("--a4", action="store_true", help="apply the A4 v4-vs-v5 choice rule (10.3 only)")
# 10.4 settings (A6)
ap.add_argument("--cap", default="25", help="distance cap in miles from the patient, or 'none' (A6; primary 25)")
ap.add_argument("--active", default="365", choices=["365", "ever"], help="§2.2 rule 2: 365-day activity (primary) or any earlier claim (A6)")
ap.add_argument("--specialties", default="primary", choices=["primary", "non_choice", "non_specific"], help="specialty group analysed (A6)")
ap.add_argument("--provider-race", action="store_true", help="A13: PCP–specialist race concordance metrics (prepared_data/provider_race_v1)")
ap.add_argument("--include-seen", action="store_true", help="A12: keep same-specialty specialists the patient has already seen as candidates")
ap.add_argument("--insurer-set", action="store_true", help="restrict eligible sets to specialists with prior claims under the patient's insurer (A6)")
ap.add_argument("--fidelity-floor", type=float, default=0.416, help="A10/A11 admissibility floor on R@5 for the 10.7 decision (A11: 0.456 − 0.04)")
ap.add_argument("--load-variants", action="store_true", help="10.7 Option A (A10): tie-break bands, load penalty and weekly capacity variants for model runs")
ap.add_argument("--dump-picks", default="", help="comma list of run-name suffixes; saves each referral's top-5 picks (base and +blend0.3) with candidate attributes (analysis only)")
ap.add_argument("--strata", action="store_true", help="STEP11 §11.2: per-referral split attributes + candidate-side shares (early tenure, low experience, non-metro)")
ap.add_argument("--list-size", type=int, default=5, help="STEP12 top-K: recommended list length K (≥ 5); capped sets are filled to K; the cohort is unchanged")
ap.add_argument("--ablate", default="", help="comma list of reliance diagnostics for model runs: no_mips,no_pcp_history,no_affiliation,no_insurer,no_distance (A6); "
                                             "STEP12a B1: no_volume,no_volume_plus (also applied to LightGBM/logistic runs)")
ap.add_argument("--hybrid", action="append", default=[], help="STEP12b H0: 'GNN_RUN_SUFFIX:TABLE_RUN_SUFFIX' (repeatable, one per seed pairing): "
                "score blends (0.25/0.5/0.75 table weight), reciprocal rank fusion, a cross-fitted gate (needs --strata) and "
                "'graph picks the top k, table fills the rest' (k = 1, 2)")
ap.add_argument("--dump-scores", action="store_true", help="STEP12c: save every candidate's model scores (all non-ablated model runs) and lens attributes "
                "per referral to <split>_candidate_scores_PRIVATE.npz (complementarity diagnostics)")
ap.add_argument("--lean", action="store_true", help="STEP12a capacity grid: only base and blend0.3 variants (no re-sorts, other blends or step12 extras) to bound memory")
ap.add_argument("--step12", action="store_true", help="STEP12a mechanism metrics: top-pick training-volume percentile (first-time / all positives), "
                                                      "Kendall tau-b of scores on log volume, top-1 − top-2 z-score gap; adds the rank-blend sensitivity (+rblend0.3)")
args = ap.parse_args()
root = str(workspace()); os.chdir(root); os.makedirs(args.out, exist_ok=True)
from referral_gnn.paths import resolve_run  # noqa: E402
args.runs = [resolve_run(r) for r in args.runs]          # runs by path or by run name (e.g. v5_unmasked_s42)
KS = [5, 6, 7, 8, 10, 20, 50, "all"]; LAMBDAS = [0.1, 0.2, 0.3, 0.5]; TS = [0, 1, 2, 5]; L = args.list_size; assert L >= LIST_SIZE
DECISIONS = "runs/analysis/step10_decisions.json"
run_names = sorted(os.path.basename(r.rstrip("/")) for r in args.runs)
pfx = os.path.join(args.out, args.split)

def file_hashes(d, prefix):
    return {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() for f in sorted(os.listdir(d)) if f.startswith(prefix)}

# ---- two-pass guard ----------------------------------------------------------------------------------
decision_entry = None
if args.split == "test":
    dec = json.load(open(DECISIONS)) if os.path.exists(DECISIONS) else {"entries": []}
    match = [e for e in dec["entries"] if e["step"] == args.step and sorted(e["runs"]) == run_names]
    if not match: sys.exit(f"REFUSED: no decision entry for step {args.step} and runs {run_names} in {DECISIONS}.")
    decision_entry = match[-1]
    now = file_hashes(args.out, "validation_")
    if now != decision_entry["validation_files_sha256"]: sys.exit("REFUSED: validation outputs changed since the decision was recorded.")
    existing = [f for f in os.listdir(args.out) if f.startswith("test_")]
    if existing: sys.exit(f"REFUSED: test outputs already exist in {args.out} ({existing[:3]} ...); the test pass runs once.")
    decision_hash = hashlib.sha256(json.dumps(decision_entry, sort_keys=True).encode()).hexdigest()

# ---- inputs (portable path: Grace-trained runs record their Grace path), yardstick, as-of MIPS ---------------
def resolve_inputs(man):
    p = man["inputs_path"]
    if not os.path.exists(p) and "/snapshots/" in p:            # STEP11 snapshots: never fall back to the main inputs
        sys.exit(f"REFUSED: snapshot inputs not found: {p}")
    if not os.path.exists(p) and "prepared_data/" in p:
        p = os.path.join(root, p[p.index("prepared_data/"):])
    want = man.get("inputs_sha256")
    if want and sha256_file(p) != want: sys.exit(f"inputs hash mismatch for {p}")
    return p
mans = [json.load(open(os.path.join(r, "run_manifest.json"))) for r in args.runs]
# Runs may use different input versions (10.3: v4 and v5). The cohort, eligible sets, histories and distances come from the first
# run's inputs; each model is scored with its own inputs. Allowed only if the versions agree on everything the cohort uses.
run_paths = [resolve_inputs(m) for m in mans]; mi_of = {p_: load_three_node_inputs(p_) for p_ in dict.fromkeys(run_paths)}
mi = mi_of[run_paths[0]]
for other in mi_of.values():
    if other is mi: continue
    for attr in ("enc_day", "enc_target", "enc_patient", "enc_pcp", "specialist_ids", "spec_first_seen", "spec_hist_day", "spec_hist_cat", "pair_ent", "pair_day", "pair_cum", "enc_dx"):
        assert np.array_equal(np.asarray(getattr(mi, attr)), np.asarray(getattr(other, attr))), f"input versions disagree on {attr}"
YARD = yardstick_path_for(mi); print("yardstick:", os.path.relpath(YARD, root))
y = load_yardstick(mi, YARD); names = specialty_names(mi); device = pick_device("auto")
ytab = pd.read_pickle(os.path.join(YARD, "mips_yardstick_PRIVATE.pkl")).set_index("provider_id"); ytab.index = ytab.index.astype(str); ytab = ytab.reindex(mi.specialist_ids)
vals = pd.read_pickle(os.path.join(YARD, "mips_values_PRIVATE.pkl"))
vals = vals[vals.value > 0].copy(); vals["s"] = vals.provider_id.astype(str).map(mi.specialist_index); vals = vals.dropna(subset=["s"])
vals["s"] = vals.s.astype(np.int64); vals["d"] = to_day(vals.first_day); vals = vals.sort_values(["s", "d"])
BIG = 1 << 20
v_key = (vals.s.to_numpy() * BIG + vals.d.to_numpy()); v_cum = np.r_[0, np.cumsum(vals.value.to_numpy())]
v_start = np.searchsorted(v_key, np.arange(mi.n_specialist) * BIG)

def asof_mips(spec, d):
    """§2.3 model-side value: mean of distinct positive values first seen strictly before day d; else the imputed specialty mean."""
    spec = np.asarray(spec, np.int64); end = np.searchsorted(v_key, spec * BIG + d, "left"); st = v_start[spec]; n = end - st
    return np.where(n > 0, (v_cum[end] - v_cum[st]) / np.maximum(n, 1), y.imputed[spec])

# ---- training-derived references (training information only) ----------------------------------------------------
active_train = mi.spec_first_seen <= mi.train_end_day
spec_train = ytab.specialty_train.to_numpy()
p10_by_spec = pd.Series(ytab.mips_value.to_numpy()[active_train]).groupby(spec_train[active_train]).quantile(0.10)
# training first-time load: ALL first-time training referrals (A2; not only the primary cohort), per specialist
q_tr = first_time_queries(mi, "train")
train_load = pd.Series(mi.enc_target[q_tr]).value_counts().reindex(np.arange(mi.n_specialist)).fillna(0).to_numpy()
# busiest = above the p90 of training load among training-active specialists of the same (training-period) specialty
tl = pd.DataFrame({"sp": spec_train, "load": train_load, "act": active_train})
p90_of = tl[tl.act].groupby("sp").load.quantile(0.90)
busy = train_load > tl.sp.map(p90_of).fillna(np.inf).to_numpy()
train_pos_all = np.bincount(mi.enc_target[mi.train_query_enc], minlength=mi.n_specialist).astype(float)   # STEP12a: all training positives
if args.step12:
    from scipy.stats import rankdata as _rankdata, kendalltau as _kendalltau
    def _pct(x):                                          # within-set percentile, ties averaged (0 = lowest, 1 = highest)
        return (_rankdata(x) - 1.0) / max(len(x) - 1, 1)
print(f"training first-time referrals for load: {len(q_tr):,}; busiest-10% specialists: {int(busy.sum()):,}")

# disease-category history per specialist (plausibility share). In three-node inputs enc_dx holds the 33 source disease
# categories (vocab 35 incl. PAD/UNK), not raw codes; checked 2026-09-29.
cat_n = int(mi.enc_dx.max()) + 1
e_idx, c_pos = np.nonzero(mi.enc_dx_mask)
cats = mi.enc_dx[e_idx, c_pos]; keep = cats > 1
cat_idx = AsOfIndex(mi.enc_target[e_idx[keep]].astype(np.int64) * cat_n + cats[keep], mi.enc_day[e_idx[keep]])

# ---- A13 provider race (analysis only; never a model input) -------------------------------------------------------
if args.provider_race:
    _pr = pd.read_pickle("prepared_data/provider_race_v1/pcp_race_PRIVATE.pkl").set_index("npi").race
    _sr = pd.read_pickle("prepared_data/provider_race_v1/specialist_race_PRIVATE.pkl").set_index("npi").race
    _clean = lambda ser: np.array([x if isinstance(x, str) and x else None for x in ser.tolist()], dtype=object)      # blanks / NA -> None
    pcp_race_arr = _clean(_pr.reindex(mi.pcp_ids)); spec_race_arr = _clean(_sr.reindex(mi.specialist_ids))

# ---- STEP11 §11.2 split attributes (analysis only; never model inputs) -------------------------------------------
if args.strata:
    def _tenure(asof, hist, enc):                            # years_in_medicare: one constant value per provider (snapshot)
        i = enc.numeric_cols.index("years_in_medicare"); sc_ = enc.scalers["years_in_medicare"]
        ent = asof.entity[np.argsort(asof.order)] if len(asof.entity) == len(hist) else asof.entity
        t = pd.Series(hist[:, 3 * i] * sc_.std + sc_.mean).where(hist[:, 3 * i + 1] < 0.5).groupby(ent).first()
        return t
    spec_ten = _tenure(mi.spec_hist_asof, mi.spec_hist_num, mi.spec_encoder).reindex(np.arange(mi.n_specialist)).to_numpy()
    pcp_ten = _tenure(mi.pcp_hist_asof, mi.pcp_hist_num, mi.pcp_encoder).reindex(np.arange(mi.n_pcp)).to_numpy()
    _BIG = 1 << 20; _tk = np.sort(mi.enc_target.astype(np.int64) * _BIG + mi.enc_day.astype(np.int64))
    def prior_refs(spec, d):                                 # referrals received strictly before day d
        spec = np.asarray(spec, np.int64); return np.searchsorted(_tk, spec * _BIG + d) - np.searchsorted(_tk, spec * _BIG)
    _pf = pd.DataFrame({"p": mi.enc_pcp, "s": mi.enc_target, "d": mi.enc_day}).groupby(["p", "s"]).d.min().reset_index()
    _pfirst = {p_: np.sort(g.d.to_numpy()) for p_, g in _pf.groupby("p")}
    def pcp_net(p_, d): a_ = _pfirst.get(int(p_)); return int(np.searchsorted(a_, d)) if a_ is not None else 0
    _pk = np.sort(mi.enc_pcp.astype(np.int64) * _BIG + mi.enc_day.astype(np.int64))
    def pcp_prior(p_, d): return int(np.searchsorted(_pk, int(p_) * _BIG + d) - np.searchsorted(_pk, int(p_) * _BIG))   # STEP12: referrals made strictly before d
    _ru = pd.read_pickle("prepared_data/provider_ruca_v1/specialist_ruca_PRIVATE.pkl").set_index("npi").group
    spec_nonmetro = (_ru.reindex(pd.Index(mi.specialist_ids).astype(str)) == "nonmetro").to_numpy()
    _age = mi.enc_patient_num[:, 0] * mi.age_scaler.std + mi.age_scaler.mean; _age_miss = mi.enc_patient_num[:, 1] > 0.5

# ---- cohort ----------------------------------------------------------------------------------------------
CAP = None if args.cap == "none" else float(args.cap); ACTIVE = None if args.active == "ever" else 365
cohort = Step10Cohort(mi, args.split, y, cap=CAP, active_days=ACTIVE, keep_specialties=args.specialties, insurer_set=args.insurer_set, include_seen=args.include_seen, fill_to=L)
R = list(cohort.referrals()); n_ref = len(R)
setting = {"list_size": L, "include_seen": args.include_seen, "cap": args.cap, "active": args.active, "specialties": args.specialties, "insurer_set": args.insurer_set, "ablate": args.ablate}
print(f"{args.split}: primary cohort {n_ref:,}; funnel {dict(cohort.counts)}")
sel = set(mi.validation_sample.tolist()) if args.split == "validation" else set()

# ---- model scores on each eligible set ----------------------------------------------------------------------
model_scores = {}
enc_ids = np.array([r.enc for r in R], np.int64); pos_of = {int(e): i for i, e in enumerate(enc_ids)}
ABLATIONS = [a_ for a_ in args.ablate.split(",") if a_]
class AblatedScorer(ModelScorer):
    """Reliance diagnostic (A6): one input group set to its 'absent' encoding for every specialist / candidate at inference time."""
    def __init__(self, model, mi_, device_, abl):
        super().__init__(model, mi_, device_); self.abl = abl; PB = mi_.pair_feature_blocks
        self.pair_cols = {"no_pcp_history": PB["referral_history"], "no_affiliation": PB.get("affiliation", []), "no_insurer": PB.get("insurer", []),
                          "no_distance": PB["distance"] + PB.get("patient_distance", [])}.get(abl, [])
        assert abl in ("no_mips", "no_volume", "no_volume_plus") or self.pair_cols, f"unknown or unavailable ablation {abl}"
        if abl in ("no_volume", "no_volume_plus"):
            from referral_gnn.volume import stats as _vstats
            self.vst = _vstats(mi_)
    @torch.no_grad()
    def prepare(self, cutoff, batch_enc):
        self.model.eval(); batch = self.mi.graph_batch(cutoff, batch_enc, device=self.device)
        if self.abl == "no_mips":
            assert self.mi.schema_version == "three-node-v5", "no_mips absent encoding is defined for v5 inputs"
            x = batch["x"]["specialist_num"].clone(); v, m, a = self.mi.mips_cols
            x[:, v] = torch.as_tensor(self.mi.mips_imputed_scaled, device=x.device, dtype=x.dtype); x[:, m] = 1.0; x[:, a] = 0.0; x[:, -2:] = 0.0
            batch = dict(batch); batch["x"] = dict(batch["x"]); batch["x"]["specialist_num"] = x
        if self.abl in ("no_volume", "no_volume_plus"):                 # STEP12a B1: rows of specialist_num = all specialists in index order
            from referral_gnn.volume import neutralise
            x = batch["x"]["specialist_num"]; assert x.shape[0] == self.mi.n_specialist
            xn = neutralise(self.mi, x.detach().cpu().numpy(), np.arange(x.shape[0]), self.vst, plus=(self.abl == "no_volume_plus"))
            batch = dict(batch); batch["x"] = dict(batch["x"]); batch["x"]["specialist_num"] = torch.as_tensor(xn, device=x.device, dtype=x.dtype)
        self.h = self.model.encode(batch)
    @torch.no_grad()
    def score(self, q, cand):
        mi_ = self.mi; pf = mi_.pair_features(np.full(len(cand), q), cand)
        for c_ in self.pair_cols:
            pf[:, c_] = 0.0
        for c_ in self.pair_cols:                         # absence carried by the missing/unknown flags (same encoding as shap_groups.pair_absent)
            if c_ in (1, 3, 5, 8, 10, 12): pf[:, c_ + 1] = 1.0
        pf = torch.as_tensor(pf, device=self.device)
        c = torch.as_tensor(cand, device=self.device); e = torch.full_like(c, int(q))
        p_ = torch.full_like(c, int(mi_.enc_patient[q])); pc = torch.full_like(c, int(mi_.enc_pcp[q]))
        return self.model.score(self.h, e, c, pf, p_, pc).float().cpu().numpy()

# Step 11.1 non-GNN baselines (STEP11_PLAN): scored in a separate process (scripts/evaluate/score_baselines11.py), because torch and
# lib_lightgbm cannot both execute in one process on this Mac (2026-10-07). The saved scores are checked against this cohort.
from referral_gnn.tabular import is_baseline_run
import subprocess, hashlib as _hl
base_runs = [(r_, m_, p_) for r_, m_, p_ in zip(args.runs, mans, run_paths) if is_baseline_run(m_)]
if base_runs:
    assert len({p_ for _, _, p_ in base_runs} | {run_paths[0]}) == 1, "baseline runs must use the cohort's inputs"
    npz = os.path.join(args.out, f"_{args.split}_baseline_scores_PRIVATE.npz")
    cmd = [sys.executable, "-W", "ignore", str(script_path("score_baselines11.py")), "--inputs", run_paths[0], "--split", args.split, "--out", npz,
           "--cap", args.cap, "--active", args.active, "--specialties", args.specialties, "--list-size", str(L)] + (["--insurer-set"] if args.insurer_set else []) \
          + (["--include-seen"] if args.include_seen else []) \
          + (["--ablate", ",".join(a_ for a_ in ABLATIONS if a_ in ("no_volume", "no_volume_plus"))] if any(a_ in ("no_volume", "no_volume_plus") for a_ in ABLATIONS) else []) \
          + [r_ for r_, _, _ in base_runs]
    print("scoring baselines in a separate process:", [os.path.basename(r_.rstrip("/")) for r_, _, _ in base_runs], flush=True)
    subprocess.run(cmd, check=True)
    z_ = np.load(npz)
    assert np.array_equal(z_["enc_ids"], enc_ids), "baseline scores: referral ids differ from this cohort"
    assert np.array_equal(z_["lengths"], np.array([len(r.eligible) for r in R])), "baseline scores: eligible-set sizes differ"
    assert str(z_["eligible_sha256"]) == _hl.sha256(np.concatenate([r.eligible for r in R]).astype(np.int64).tobytes()).hexdigest(), "baseline scores: eligible sets differ"
    bounds = np.r_[0, np.cumsum(z_["lengths"])]
    for r_, _, _ in base_runs:
        for abl_ in [None] + [a_ for a_ in ABLATIONS if a_ in ("no_volume", "no_volume_plus")]:
            nm_ = os.path.basename(r_.rstrip("/")) + (f"@{abl_}" if abl_ else ""); flat = z_["scores::" + nm_]
            model_scores[nm_] = [flat[bounds[i]:bounds[i + 1]] for i in range(n_ref)]; print(f"  loaded baseline scores {nm_}")
for run, man, rp in zip(args.runs, mans, run_paths):
    name = os.path.basename(run.rstrip("/")); mr = mi_of[rp]
    if is_baseline_run(man): continue
    if man.get("kind") == "emb_lr":                        # STEP12b H2: GNN encoder + logistic regression (torch only)
        from referral_gnn.hybrid import load_emb_lr_scorer
        t0 = time.time(); sc = load_emb_lr_scorer(run, mr, device); out = [None] * n_ref
        for cutoff, batch_enc, q_b in mr.buckets(args.split, "week", query_subset=np.sort(enc_ids)):
            sc.prepare(cutoff, batch_enc)
            for q in q_b:
                i = pos_of[int(q)]; out[i] = np.asarray(sc.score(int(q), R[i].eligible), float)
        model_scores[name] = out; print(f"  scored {name} ({time.time() - t0:.0f}s)", flush=True); continue
    model = load_three_node_model(run, mr, device)
    for abl in [None] + ABLATIONS:
        t0 = time.time()
        sc = ModelScorer(model, mr, device) if abl is None else AblatedScorer(model, mr, device, abl)
        if model.cfg.table_offset or model.cfg.tab_dim:    # STEP12d: dual encoder / table-offset GNN (offset = the full table run's scores)
            if abl is not None: continue                     # ablations are not defined for STEP12d models (reviewer S4)
            from referral_gnn.heavy_hybrid import ExtraModelScorer
            off_fn = None
            if model.cfg.table_offset:
                tname = os.path.basename(man["offset_table_run"].rstrip("/")); assert tname in model_scores, f"offset model needs its table run {tname} in the run list"
                off_fn = lambda q_, cand_, tname=tname: model_scores[tname][pos_of[int(q_)]]
            sc = ExtraModelScorer(model, mr, device, offset_fn=off_fn)
        out = [None] * n_ref
        for cutoff, batch_enc, q_b in mr.buckets(args.split, man["train_config"]["graph_bucket"], query_subset=np.sort(enc_ids)):
            sc.prepare(cutoff, batch_enc)
            for q in q_b:
                i = pos_of[int(q)]; out[i] = np.asarray(sc.score(int(q), R[i].eligible), float)
        key = name if abl is None else f"{name}@{abl}"; model_scores[key] = out; print(f"  scored {key} ({time.time() - t0:.0f}s)")

# ---- rankings -----------------------------------------------------------------------------------------------------
rng_fixed = np.random.default_rng(20260928)
def order_by(*keys):                   # lexicographic, first key most significant, all "smaller is better"
    return np.lexsort(tuple(reversed(keys)))
def resort(order, m, K):
    k = len(order) if K == "all" else min(K, len(order)); top = order[:k]
    top = top[np.lexsort((np.arange(k), -m[top]))]; return np.r_[top, order[k:]]
def z(x):
    s = x.std(); return (x - x.mean()) / s if s > 0 else np.zeros_like(x)
def weights_from_order(order, n):
    w1 = np.zeros(n); w5 = np.zeros(n); w1[order[0]] = 1; w5[order[:L]] = 1; return w1, w5

# ---- STEP12b H0 hybrids of a GNN run and a table run (scores already computed above) ---------------------------------
HYB_SPLIT, HYB_INFO = [], {}
if args.hybrid:
    assert args.strata, "--hybrid needs --strata (the gate arm uses the split attributes)"
    def _key(sfx):
        k_ = [k for k in model_scores if k.endswith(sfx) and "@" not in k]; assert len(k_) == 1, (sfx, k_); return k_[0]
    _rank = lambda x: np.argsort(np.argsort(-np.asarray(x, float), kind="stable"), kind="stable")
    meta = None
    for h_ in args.hybrid:
        gk, tk = (_key(x) for x in h_.split(":")); tag = f"__{gk}__{tk}"; G_, T_ = model_scores[gk], model_scores[tk]
        for w in (0.25, 0.5, 0.75):
            model_scores[f"hyb_blend{w}{tag}"] = [(1 - w) * z(np.asarray(g, float)) + w * z(np.asarray(t, float)) for g, t in zip(G_, T_)]
        model_scores[f"hyb_rrf{tag}"] = [1.0 / (60 + _rank(g)) + 1.0 / (60 + _rank(t)) for g, t in zip(G_, T_)]
        for k in (1, 2): HYB_SPLIT.append((f"hyb_split{k}{tag}", gk, tk, k))
        if args.strata:                                    # FWLS gate w(q) = sigmoid(θ·meta), cross-fitted over PCP-grouped halves
            from scipy.optimize import minimize
            if meta is None:
                mrow = []
                for r in R:
                    pri, _, _ = mi.raw_pair_history(np.full(len(r.eligible), r.enc), r.eligible)
                    mrow.append([1.0, np.log1p(pcp_prior(r.pcp, r.day)), float((pri > 0).mean()), 0.0 if _age_miss[r.enc] else float(_age[r.enc]), np.log(len(r.eligible))])
                meta = np.array(mrow); meta[:, 1:] = (meta[:, 1:] - meta[:, 1:].mean(0)) / (meta[:, 1:].std(0) + 1e-9)
                _up = np.unique([r.pcp for r in R]); _perm = np.random.default_rng(20261014).permutation(len(_up))
                _fo = {int(p_): int(_perm[j_] % 2) for j_, p_ in enumerate(_up)}; fold = np.array([_fo[int(r.pcp)] for r in R])   # fixed PCP halves
                tpos = [int(np.where(r.eligible == r.target)[0][0]) if r.target in set(r.eligible.tolist()) else -1 for r in R]
            zg = [z(np.asarray(g, float)) for g in G_]; zt = [z(np.asarray(t, float)) for t in T_]
            def nll(th, idx):                              # th = [meta weights..., log-temperature τ] (free sharpness; reviewer 2026-10-14)
                tot = 0.0
                for i_ in idx:
                    if tpos[i_] < 0: continue
                    w_ = 1 / (1 + np.exp(-meta[i_] @ th[:-1])); sc_ = np.exp(th[-1]) * ((1 - w_) * zg[i_] + w_ * zt[i_]); sc_ = sc_ - sc_.max()
                    tot -= sc_[tpos[i_]] - np.log(np.exp(sc_).sum())
                return tot / max(len(idx), 1)
            th_of = {}
            for f_ in (0, 1):
                tr_idx = np.where(fold != f_)[0]; th_of[f_] = minimize(nll, np.zeros(meta.shape[1] + 1), args=(tr_idx,), method="L-BFGS-B").x
            gate = []
            for i_ in range(n_ref):
                w_ = 1 / (1 + np.exp(-meta[i_] @ th_of[int(fold[i_])][:-1])); gate.append((1 - w_) * zg[i_] + w_ * zt[i_])
            model_scores[f"hyb_gate{tag}"] = gate
            HYB_INFO[tag] = {"gate_theta_fold0": th_of[0].tolist(), "gate_theta_fold1": th_of[1].tolist(),
                             "meta": ["intercept", "log PCP prior referrals", "familiar share of eligible set", "patient age", "log eligible-set size"],
                             "mean_table_weight": float(np.mean([1 / (1 + np.exp(-meta[i_] @ th_of[int(fold[i_])][:-1])) for i_ in range(n_ref)])),
                             "mean_table_weight_by_fold": {f_: float(np.mean([1 / (1 + np.exp(-meta[i_] @ th_of[f_][:-1])) for i_ in np.where(fold == f_)[0]])) for f_ in (0, 1)},
                             "objective_note": "gate fitted to imitate the chosen specialist (listwise likelihood), not to the lenses"}
        print(f"  hybrids built for {h_}", flush=True)

# ---- per-referral metric accumulation -----------------------------------------------------------------------------
per = defaultdict(lambda: defaultdict(list))            # config -> metric -> list over referrals
wdist = defaultdict(lambda: defaultdict(lambda: ([], [])))   # config -> {"d5","d1","x5","x1"} -> (values, weights)
load1 = defaultdict(lambda: defaultdict(float)); load5 = defaultdict(lambda: defaultdict(float))
frame_spec = {}                                          # specialists in the split's load frame -> referral specialty
ref_meta = []
H = lambda n: np.sum(1.0 / np.arange(1, n + 1))
GAIN_BINS = np.arange(-100, 105, 5)

def add_w(cfg, key, vals_, w):
    ok = ~np.isnan(vals_); a, b = wdist[cfg][key]; a.append(vals_[ok]); b.append(w[ok])

DUMP_SFX = [x for x in args.dump_picks.split(",") if x]; DUMP_ROWS = []
def _dump(cfg, r, c, order):
    run_ = cfg.split(":", 1)[1].split("+")[0]; var_ = cfg.split("+", 1)[1] if "+" in cfg else "base"
    if var_ not in ("base", "blend0.3") or not any(run_.endswith(x) for x in DUMP_SFX): return
    for rank_, j in enumerate(order[:5]):
        DUMP_ROWS.append({"enc": r.enc, "pcp": r.pcp, "run": run_, "variant": var_, "rank": rank_ + 1, "is_target": bool(j == c["t_pos"]),
                          "new_pcp": bool(c["new"][j]), "mips": float(c["v"][j]), "mips_none": bool(c["none"][j]), "busy": bool(c["busy"][j]),
                          "dist": float(c["dist"][j]), "cat": bool(c["cat"][j]), "ins": bool(c["ins"][j]), "spec": int(r.eligible[j]),
                          "prior_refs": int(prior_refs([r.eligible[j]], r.day)[0]) if args.strata else -1, "ref_mips": float(c["rv"]), "n_elig": int(len(r.eligible))})

def record(cfg, r, w1, w5, c, order=None):
    if DUMP_SFX and order is not None and cfg.startswith("model:"): _dump(cfg, r, c, order)
    v, none, rv, n = c["v"], c["none"], c["rv"], len(c["v"]); P = per[cfg]
    Lr = float(w5.sum())                                 # list length: min(K, eligible); = L for K = 5 (every set has ≥ 5)
    P["gain5"].append(w5 @ v / Lr - rv); P["gain1"].append(w1 @ v - rv)
    P["gain5_cons"].append(w5 @ c["lo0"] / Lr - c["rhi"]); P["gain5_gen"].append(w5 @ c["hi_imp"] / Lr - c["rlo"])
    P["gain5_spread_only"].append(w5 @ c["lo_imp"] / Lr - c["rhi"]); P["gain5_noscore0_only"].append(w5 @ c["v0"] / Lr - rv)
    b1 = w1 @ ((v > rv) & ~none); e1 = w1 @ ((v == rv) & ~none); P["better1"].append(b1); P["equal1"].append(e1); P["worse1"].append(1 - b1 - e1)
    for t in TS:
        ok_t = (v >= rv - t) & ~none                     # no-score slots never count as within
        P[f"within{t}_1"].append(w1 @ ok_t); P[f"within{t}_5"].append(w5 @ ok_t / Lr)
    P["low_mips1"].append(w1 @ (v < c["p10"])); P["headroom"].append(c["headroom"])
    P["new_pcp1"].append(w1 @ c["new"]); P["new_ins5"].append(w5 @ (c["new"] & c["ins"]) / Lr)          # STEP12b: top-pick newness; insurer-feasible new picks
    P["new_pcp5"].append(w5 @ c["new"] / Lr); P["good_new5"].append(w5 @ (c["new"] & (v > rv) & ~none) / Lr); P["none5"].append(w5 @ none / Lr)
    P["insurer5"].append(w5 @ c["ins"] / Lr); P["category5"].append(w5 @ c["cat"] / Lr)
    P["busy1"].append(w1 @ c["busy"]); P["busy5"].append(w5 @ c["busy"] / Lr)
    P["seen1"].append(w1 @ c["seen"]); P["seen5"].append(w5 @ c["seen"] / Lr)                 # A12: previously seen by the patient
    if "conc" in c: P["conc1"].append(w1 @ c["conc"]); P["conc5"].append(w5 @ c["conc"] / Lr)    # A13: same race as the PCP
    if args.step12 and order is not None:                                                       # STEP12a: training volume of the top pick
        P["volpct1_ft"].append(c["pft"][order[0]]); P["volpct1_all"].append(c["pall"][order[0]]); P["top1_spec"].append(float(r.eligible[order[0]]))
    if "early" in c:                                                                            # STEP11 §11.2 candidate-side shares
        for k_ in ("early", "lowexp", "nonmetro"): P[f"{k_}1"].append(w1 @ c[k_]); P[f"{k_}5"].append(w5 @ c[k_] / Lr)
    t_in = c["t_pos"]; P["r5"].append(w5[t_in] if t_in >= 0 else 0.0)
    if order is None:                                    # random: exact expectations
        P["r10"].append(min(10, n) / n if t_in >= 0 else 0.0); P["rr"].append(H(n) / n if t_in >= 0 else 0.0)
    else:
        rank = int(np.where(order == t_in)[0][0]) + 1 if t_in >= 0 else None
        P["r10"].append(float(rank is not None and rank <= 10)); P["rr"].append(1.0 / rank if rank else 0.0)
    add_w(cfg, "d5", c["dist"], w5); add_w(cfg, "d1", c["dist"], w1); add_w(cfg, "x5", c["excess"], w5); add_w(cfg, "x1", c["excess"], w1)
    okd = ~np.isnan(c["dist"]); P["dist5m"].append(float((w5[okd] @ c["dist"][okd]) / w5[okd].sum()) if w5[okd].sum() > 0 else np.nan)   # A7 per-referral
    for s, w in zip(r.eligible[w1 > 0], w1[w1 > 0]): load1[cfg][int(s)] += w
    for s, w in zip(r.eligible[w5 > 0], w5[w5 > 0]): load5[cfg][int(s)] += w / Lr

# ---- 10.7 Option A setup (A10) ---------------------------------------------------------------------------------------------
TB = [(10, 1), (10, 2), (10, 5), (20, 1), (20, 2), (20, 5)]; MUS = [0.1, 0.2, 0.3, 0.5]; KAPPAS = [1.0, 1.5, 2.0]
if args.load_variants:
    from referral_gnn.load import RecentLoad
    rl = RecentLoad(mi); cap_store = defaultdict(list); ctx_store = {}
    split_start = mi.split_days[args.split][0]

DS = defaultdict(list)
t0 = time.time()
for i, r in enumerate(R):
    elig, n = r.eligible, len(r.eligible)
    for s in elig: frame_spec.setdefault(int(s), r.specialty)
    frame_spec.setdefault(int(r.target), r.specialty)    # A2: every referred specialist is in the load frame
    v = y.point(elig); none = y.none[elig]
    lo0 = np.where(none, 0.0, y.lo[elig]); lo_imp = np.where(none, y.imputed[elig], y.lo[elig])
    hi_imp = np.where(none, y.imputed[elig], y.hi[elig]); v0 = np.where(none, 0.0, v)
    prior, _, _ = mi.raw_pair_history(np.full(n, r.enc), elig)
    ins, _ = mi.insurer_history(np.full(n, r.enc), elig)
    qc = mi.enc_dx[r.enc][mi.enc_dx_mask[r.enc]]; qc = qc[qc > 1]
    cat = np.zeros(n, bool)
    for cc in qc: cat |= cat_idx.lookup(elig.astype(np.int64) * cat_n + int(cc), r.day) >= 0
    dist = r.dist; nearest = np.nanmin(r.pre_cap_dist); excess = dist - nearest
    m_asof = asof_mips(elig, r.day)
    t_pos = int(np.where(elig == r.target)[0][0]) if r.target in set(elig.tolist()) else -1
    spname = names[r.specialty]; rv = float(y.value[r.target])
    scored = v[~none]
    c = dict(v=v, none=none, lo0=lo0, lo_imp=lo_imp, hi_imp=hi_imp, v0=v0, new=(prior == 0), ins=ins.astype(bool), cat=cat, dist=dist, excess=excess,
             busy=busy[elig], seen=(mi.ps_asof.lookup(r.patient * mi.n_specialist + elig.astype(np.int64), r.day) >= 0), rv=rv, rhi=float(y.hi[r.target]), rlo=float(y.lo[r.target]), t_pos=t_pos,
             p10=float(p10_by_spec.get(spname, -np.inf)), headroom=float(max((scored.max() if len(scored) else rv) - rv, 0.0)))   # A2: scored only
    if args.dump_scores:                                  # STEP12c: candidate-level dump (flattened with offsets)
        DS["n"].append(n); DS["spec"].append(elig.astype(np.int64)); DS["busy"].append(busy[elig].astype(np.int8)); DS["new"].append((prior == 0).astype(np.int8))
        DS["ins"].append(ins.astype(np.int8)); DS["cat"].append(cat.astype(np.int8)); DS["mips"].append(v.astype(np.float32)); DS["mips_asof"].append(m_asof.astype(np.float32))
        DS["mips_none"].append(none.astype(np.int8)); DS["dist"].append(np.nan_to_num(dist, nan=-1).astype(np.float32)); DS["target_pos"].append(t_pos); DS["ref_mips"].append(rv)
        for k_, sc_ in model_scores.items():
            if "@" not in k_: DS.setdefault("score::" + k_, []).append(np.asarray(sc_[i], np.float32))
    if args.step12:
        c["pft"] = _pct(train_load[elig]); c["pall"] = _pct(train_pos_all[elig]); c["lvol"] = np.log1p(train_load[elig])
    if args.strata:
        c["early"] = np.nan_to_num(spec_ten[elig], nan=99) <= 6; c["lowexp"] = prior_refs(elig, r.day) < 10; c["nonmetro"] = spec_nonmetro[elig]
    if args.provider_race:
        pr_ = pcp_race_arr[r.pcp]; sr_ = spec_race_arr[elig]
        c["conc"] = np.array([(isinstance(pr_, str) and isinstance(x, str) and x == pr_) for x in sr_], dtype=bool)
    ref_meta.append({"enc": r.enc, "patient": r.patient, "pcp": r.pcp, "specialty": spname, "ref_mips": rv, "ref_multi": bool(y.multi[r.target]),
                     "filled": r.filled, "selection_query": r.enc in sel, "ref_dist": r.target_dist, "ref_excess": r.target_dist - nearest,
                     "ref_new_pcp": bool(mi.raw_pair_history(np.array([r.enc]), np.array([r.target]))[0][0] == 0), "ref_busy": bool(busy[r.target]),
                     "target_in_set": t_pos >= 0,
                     **({"ref_volpct_ft": float(c["pft"][t_pos]) if t_pos >= 0 else np.nan, "ref_volpct_all": float(c["pall"][t_pos]) if t_pos >= 0 else np.nan} if args.step12 else {}),
                     **({"patient_age": (np.nan if _age_miss[r.enc] else float(_age[r.enc])), "pcp_tenure": float(pcp_ten[r.pcp]), "pcp_net": pcp_net(r.pcp, r.day),
                         "ref_tenure": float(spec_ten[r.target]), "ref_prior_refs": int(prior_refs([r.target], r.day)[0]), "ref_nonmetro": bool(spec_nonmetro[r.target]),
                         "access5": float(np.sort(r.pre_cap_dist[~np.isnan(r.pre_cap_dist)])[min(4, int((~np.isnan(r.pre_cap_dist)).sum()) - 1)]) if (~np.isnan(r.pre_cap_dist)).any() else np.nan,
                         "elig_early": float(c["early"].mean()), "elig_lowexp": float(c["lowexp"].mean()), "elig_nonmetro": float(c["nonmetro"].mean()),
                         "elig_busy": float(c["busy"].mean()), "elig_familiar": float((prior > 0).mean()), "n_elig": int(n), "pcp_prior_refs": pcp_prior(r.pcp, r.day)} if args.strata else {}),
                     **({"pcp_race": pcp_race_arr[r.pcp] if isinstance(pcp_race_arr[r.pcp], str) else "unknown",
                         "ref_conc": bool(isinstance(pcp_race_arr[r.pcp], str) and spec_race_arr[r.target] == pcp_race_arr[r.pcp]),
                         "ref_spec_race_known": isinstance(spec_race_arr[r.target], str)} if args.provider_race else {})})
    load1["practice"][int(r.target)] += 1; load5["practice"][int(r.target)] += 1
    record("random", r, np.full(n, 1.0 / n), np.full(n, min(L, n) / n), c)
    tie = rng_fixed.random(n); d_key = np.nan_to_num(dist, nan=1e9)
    bases = {"pcp": (order_by(-prior, d_key, tie), np.log1p(prior).astype(float)), "dist": (order_by(d_key, tie), -d_key)}
    for name, sc_list in model_scores.items():
        s = sc_list[i]; bases[f"model:{name}"] = (order_by(-s, tie), s)
    o = order_by(-m_asof, d_key, tie); record("mips", r, *weights_from_order(o, n), c, o)
    def _shape(cfg_, sc_):                                # STEP12a score-shape metrics (aligned with record order)
        if not (args.step12 and cfg_.startswith("model:")): return
        tau = _kendalltau(sc_, c["lvol"]).statistic if n > 2 and np.ptp(c["lvol"]) > 0 and np.ptp(sc_) > 0 else np.nan
        zs = np.sort(z(np.asarray(sc_, float)))[::-1]; per[cfg_]["tau_vol"].append(float(tau)); per[cfg_]["gap12"].append(float(zs[0] - zs[1]) if n > 1 else np.nan)
    for b, (o, s) in bases.items():
        record(b, r, *weights_from_order(o, n), c, o); _shape(b, s)
        for K in ([] if args.lean else KS):
            if b == "dist" and K == "all": continue      # identical to "mips" (ties by distance)
            o2 = resort(o, m_asof, K); record(f"{b}+resort{K}", r, *weights_from_order(o2, n), c, o2)
        if b != "dist":
            rank_in_base = np.empty(n); rank_in_base[o] = np.arange(n)
            for lam in ([0.3] if args.lean else LAMBDAS):
                bl = (1 - lam) * z(s) + lam * z(m_asof); o3 = order_by(-bl, rank_in_base); record(f"{b}+blend{lam}", r, *weights_from_order(o3, n), c, o3)
                if lam == 0.3: _shape(f"{b}+blend0.3", bl)
            if args.step12 and not args.lean and b.startswith("model:"):   # STEP12a §2.E sensitivity: degrade by Gaussian noise on the 30%-weight score
                bl03_ = 0.7 * z(s) + 0.3 * z(m_asof); nrng = np.random.default_rng([20261013, int(r.enc)])
                for sig in (0.25, 0.5, 1.0, 2.0):
                    o3 = order_by(-(bl03_ + sig * nrng.standard_normal(n)), rank_in_base); record(f"{b}+noise{sig}_blend0.3", r, *weights_from_order(o3, n), c, o3)
            if args.step12 and not args.lean:                  # STEP12a S-4: the 30% weight on within-set rank percentiles instead of z-scores
                blr = 0.7 * _pct(np.asarray(s, float)) + 0.3 * _pct(m_asof); o3 = order_by(-blr, rank_in_base)
                record(f"{b}+rblend0.3", r, *weights_from_order(o3, n), c, o3)
            if args.step12 and not args.lean and b.startswith("model:"):   # STEP12a S-4: output-side residualisation on log training volume (within set)
                xv = c["lvol"] - c["lvol"].mean(); sv = np.asarray(s, float)
                beta = float(xv @ (sv - sv.mean()) / (xv @ xv)) if (xv @ xv) > 0 else 0.0
                sr = sv - beta * xv; o3 = order_by(-sr, rank_in_base); record(f"{b}+resid", r, *weights_from_order(o3, n), c, o3)
                blq = 0.7 * z(sr) + 0.3 * z(m_asof); o3 = order_by(-blq, rank_in_base); record(f"{b}+resid_blend0.3", r, *weights_from_order(o3, n), c, o3)
        if args.load_variants and b.startswith("model:") and not b.startswith("model:hyb_") and "_v5_unmasked_" in b and not b.endswith("_dup") and "@" not in b:   # A10 Option A: final model only
            load_r = rl.count(elig, r.day, 365); zl = z(np.log1p(load_r))
            for K, t in TB:                                                        # A1: MIPS bands of t points tied -> lower load -> model order
                k = min(K, n); top = o[:k]
                top = top[np.lexsort((rank_in_base[top], load_r[top], -np.floor(m_asof[top] / t)))]
                o4 = np.r_[top, o[k:]]; record(f"{b}+tb{K}_{t}", r, *weights_from_order(o4, n), c, o4)
            bl03 = 0.7 * z(s) + 0.3 * z(m_asof)
            for mu in MUS:                                                         # A2: load penalty on the final blend
                o5 = order_by(-(bl03 - mu * zl), rank_in_base); record(f"{b}+pen{mu}", r, *weights_from_order(o5, n), c, o5)
            ob = order_by(-bl03, rank_in_base); srt = np.sort(bl03)[::-1]             # A3: stored for the weekly capacity pass
            cap_store[b].append((i, (r.day - split_start) // 7, r.specialty, ob, float(srt[0] - srt[1]) if n > 1 else 0.0))
            ctx_store[i] = (r, c)
    for key_, gk_, tk_, k_ in HYB_SPLIT:                    # STEP12b: graph's top k (its own ranking), then the table's ranking for the rest
        g_ = np.asarray(model_scores[gk_][i], float); t_ = np.asarray(model_scores[tk_][i], float)
        for var_, (sg_, st_) in {"": (g_, t_), "+blend0.3": (0.7 * z(g_) + 0.3 * z(m_asof), 0.7 * z(t_) + 0.3 * z(m_asof))}.items():
            top_ = list(order_by(-sg_, tie)[:k_]); o_ = np.array(top_ + [j for j in order_by(-st_, tie) if j not in top_])
            record(f"model:{key_}{var_}", r, *weights_from_order(o_, n), c, o_)
    if (i + 1) % 1000 == 0: print(f"  {i + 1:,}/{n_ref:,} referrals ({time.time() - t0:.0f}s)")

if args.load_variants:                                                             # A3 weekly capacity (second pass)
    train_share_w = train_load + 1.0
    for b, items in cap_store.items():
        groups = defaultdict(list)
        for it in items: groups[(it[1], it[2])].append(it)
        for kappa in KAPPAS:
            assigned = {}
            for (wk, sp), its in groups.items():
                specs = np.unique(np.concatenate([ctx_store[it[0]][0].eligible for it in its]))
                share = dict(zip(specs.tolist(), (train_share_w[specs] / train_share_w[specs].sum()).tolist()))
                cap = {s_: max(1, int(np.ceil(kappa * share[s_] * len(its)))) for s_ in share}; used = defaultdict(int)
                for it in sorted(its, key=lambda x: -x[4]):                           # most confident referrals choose first
                    ii, ob = it[0], it[3]; elig = ctx_store[ii][0].eligible
                    pick = next((j for j in ob if used[int(elig[j])] < cap[int(elig[j])]), ob[0])
                    used[int(elig[pick])] += 1; assigned[ii] = (pick, ob)
            for ii in sorted(assigned):
                pick, ob = assigned[ii]; r_, c_ = ctx_store[ii]; o6 = np.r_[pick, ob[ob != pick]]
                record(f"{b}+cap{kappa}", r_, *weights_from_order(o6, len(ob)), c_, o6)
    # every capacity config must cover every referral, in referral order
    for b in cap_store:
        for kappa in KAPPAS: assert len(per[f"{b}+cap{kappa}"]["gain5"]) == n_ref

meta = pd.DataFrame(ref_meta)
assert abs(sum(load1["practice"].values()) - n_ref) < 1e-9 and set(load1["practice"]) <= set(frame_spec)

# ---- aggregation ------------------------------------------------------------------------------------------------------
def wq(pair, q):
    d, w = (np.concatenate(pair[0]), np.concatenate(pair[1])) if pair[0] else (np.array([]), np.array([]))
    if w.sum() == 0: return np.nan
    o = np.argsort(d); cw = np.cumsum(w[o]) / w.sum(); return float(d[o][min(np.searchsorted(cw, q), len(d) - 1)])
def wshare(pair, thr):
    d, w = np.concatenate(pair[0]), np.concatenate(pair[1]); return float(w[d > thr].sum() / w.sum())

uniq, cl_idx = np.unique(meta.patient.to_numpy(), return_inverse=True)
brng = np.random.default_rng(0); B = np.array([np.bincount(brng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(args.bootstrap)])
def boot_ci(x, mask=None):
    x = np.asarray(x, float); m = np.ones(len(x), bool) if mask is None else np.asarray(mask)
    sums = np.bincount(cl_idx[m], weights=x[m], minlength=len(uniq)); cnts = np.bincount(cl_idx[m], minlength=len(uniq))
    est = (B @ sums) / np.maximum(B @ cnts, 1); return float(x[m].mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))

def load_metrics(lt, sfx):
    df = pd.DataFrame({"sp": pd.Series(frame_spec)}); df["rec"] = pd.Series(lt).reindex(df.index).fillna(0); df["train"] = train_load[df.index.to_numpy()]
    ginis, wts = [], []
    for _, g in df.groupby("sp"):
        x = np.sort(g.rec.to_numpy()); m = len(x)
        if x.sum() > 0 and m > 1: ginis.append(float((2 * np.arange(1, m + 1) - m - 1) @ x / (m * x.sum()))); wts.append(x.sum())
    share_rec = (df.rec + 1) / df.groupby("sp").rec.transform(lambda s_: (s_ + 1).sum())
    share_tr = (df.train + 1) / df.groupby("sp").train.transform(lambda s_: (s_ + 1).sum())
    cong = share_rec / share_tr; has_tr = df.train > 0
    return {f"gini_within{sfx}": float(np.average(ginis, weights=wts)) if ginis else np.nan,
            f"congestion_median{sfx}": float(cong[has_tr].median()), f"congestion_p90{sfx}": float(cong[has_tr].quantile(0.9)),
            f"congestion_median_no_train_load{sfx}": float(cong[~has_tr].median()) if (~has_tr).any() else np.nan,
            f"distinct_recommended{sfx}": int((df.rec > 0).sum()), f"max_load{sfx}": float(df.rec.max())}

metrics_mean = ["gain5", "gain5_cons", "gain5_gen", "gain5_spread_only", "gain5_noscore0_only", "gain1", "better1", "equal1", "worse1"] + \
               [f"within{t}_{k}" for t in TS for k in (1, 5)] + ["low_mips1", "new_pcp5", "new_pcp1", "new_ins5", "good_new5", "none5", "insurer5", "category5",
               "busy1", "busy5", "seen1", "seen5", "r5", "r10", "rr"]
rows = []
for cfg, P in per.items():
    row = {"config": cfg, "n": len(P["gain5"])}
    for mname in metrics_mean:
        est, lo, hi = boot_ci(P[mname]); row[mname] = est; row[mname + "_lo"] = lo; row[mname + "_hi"] = hi
    row["gain1_median"] = float(np.median(P["gain1"])); row["gain5_median"] = float(np.median(P["gain5"]))
    hr = np.array(P["headroom"]); g1 = np.array(P["gain1"]); m_ = hr > 0
    row["headroom_captured"] = float(g1[m_].mean() / hr[m_].mean()) if m_.any() else np.nan
    W = wdist[cfg]
    row |= {"dist5_median": wq(W["d5"], .5), "dist5_p75": wq(W["d5"], .75), "dist5_p90": wq(W["d5"], .9), "dist5_gt10": wshare(W["d5"], 10), "dist5_gt25": wshare(W["d5"], 25),
            "dist1_median": wq(W["d1"], .5), "dist1_p90": wq(W["d1"], .9), "dist1_gt10": wshare(W["d1"], 10), "dist1_gt25": wshare(W["d1"], 25),
            "excess5_median": wq(W["x5"], .5), "excess5_p90": wq(W["x5"], .9), "excess1_median": wq(W["x1"], .5), "excess1_p90": wq(W["x1"], .9)}
    row |= load_metrics(load1[cfg], "") | load_metrics(load5[cfg], "_5"); rows.append(row)
cf = pd.DataFrame(rows)
rd, rx = meta.ref_dist.dropna(), meta.ref_excess.dropna()
bb = boot_ci(meta.ref_busy.astype(float))
practice = {"config": "practice", "n": n_ref, "r5": 1.0, "r10": 1.0, "rr": 1.0, "gain5": 0.0, "gain1": 0.0, "gain1_median": 0.0, "gain5_median": 0.0,
            "dist1_median": float(rd.median()), "dist1_p90": float(rd.quantile(.9)), "dist1_gt10": float((rd > 10).mean()), "dist1_gt25": float((rd > 25).mean()),
            "dist5_median": float(rd.median()), "dist5_p75": float(rd.quantile(.75)), "dist5_p90": float(rd.quantile(.9)),
            "dist5_gt10": float((rd > 10).mean()), "dist5_gt25": float((rd > 25).mean()),
            "excess1_median": float(rx.median()), "excess1_p90": float(rx.quantile(.9)), "excess5_median": float(rx.median()), "excess5_p90": float(rx.quantile(.9)),
            "new_pcp5": float(meta.ref_new_pcp.mean()), "busy1": bb[0], "busy1_lo": bb[1], "busy1_hi": bb[2], "busy5": bb[0]} | \
           load_metrics(load1["practice"], "") | load_metrics(load5["practice"], "_5")
cf = pd.concat([pd.DataFrame([practice]), cf], ignore_index=True)

# Step 11.1: explicit arms so baselines and no-graph runs are never averaged with the GNN arms
ARM_OF = {}
for k_ in list(model_scores) + [x[0] for x in HYB_SPLIT]:          # STEP12b hybrids: one arm per hybrid kind, seed pairings averaged
    if k_.startswith("hyb_"): ARM_OF[k_] = k_.split("__")[0]
for r_, m_ in zip(args.runs, mans):
    n_ = os.path.basename(r_.rstrip("/"))
    if m_.get("arm"): ARM_OF[n_] = m_["arm"]
    elif m_.get("model_config", {}).get("table_offset"): ARM_OF[n_] = "model_s12d_table_offset"     # STEP12d A
    elif m_.get("model_config", {}).get("tab_dim"): ARM_OF[n_] = "model_s12d_dual_encoder"           # STEP12d C
    elif m_.get("exclude_fold") is not None: ARM_OF[n_] = f"model_s12d_fold{m_['exclude_fold']}"    # STEP12d B fold models (diagnostic only)
    elif "_s12_" in n_: ARM_OF[n_] = "model_" + __import__("re").sub(r"_s\d+$", "", n_.split("_", 1)[1])     # STEP12a neural cells: one arm per cell
    elif m_.get("model_config", {}).get("use_graph") is False: ARM_OF[n_] = None       # tagged below as *_nograph
    elif m_.get("model_config", {}).get("scorer_skip"): ARM_OF[n_] = "model_c2_skip"   # STEP11 B5 architecture arms
    elif float(m_.get("history_dropout") or 0) > 0: ARM_OF[n_] = f"model_c1_hd{float(m_['history_dropout']):g}"
def fam(c_):
    if c_.startswith("model:") and ARM_OF.get(c_.split(":", 1)[1].split("+")[0].split("@")[0]):
        run = c_.split(":", 1)[1].split("+")[0]
        return ARM_OF[run.split("@")[0]] + (f"@{run.split('@', 1)[1]}" if "@" in run else ""), run, c_.split("+", 1)[1] if "+" in c_ else "base"
    if c_.startswith("model:"):
        run = c_.split(":", 1)[1].split("+")[0]; arm = "model_masked" if "masked" in run and "unmasked" not in run else "model_unmasked"
        abl = run.split("@", 1)[1] if "@" in run else None; run_base = run.split("@", 1)[0]
        if args.tag_version:
            ver = next((v for v in ("v6", "v5", "v4", "v3") if f"_{v}_" in run_base), "vX")
            tag = "_load" if "_v6_load_" in run_base else ("_posw" if "_posw" in run_base else "")         # 10.7 Option B variants (A10)
            arm += f"_{ver}{tag}" + ("_dup" if run_base.endswith("_dup") else "")
        if run_base in ARM_OF: arm += "_nograph"                          # ARM_OF[run_base] is None: GNN with use_graph=False
        if abl: arm += f"@{abl}"
        return arm, run, c_.split("+", 1)[1] if "+" in c_ else "base"
    return c_.split("+")[0], "", c_.split("+", 1)[1] if "+" in c_ else "base"
cf[["family", "run", "variant"]] = pd.DataFrame([fam(c_) for c_ in cf.config], index=cf.index)
cf["arm"] = cf.family + ":" + cf.variant

# seed-averaged arm table; CIs recomputed on the seed-averaged per-referral series (A2; not averages of per-seed bounds)
def arm_series(arm, metric):
    cfgs = cf.loc[cf.arm == arm, "config"]; return np.mean([per[c_][metric] for c_ in cfgs], axis=0)
num = [c_ for c_ in cf.select_dtypes("number").columns if not (c_.endswith("_lo") or c_.endswith("_hi"))]
arms = cf.groupby("arm", sort=False)[num].mean().reset_index()
arms[["family", "variant"]] = arms.arm.str.split(":", n=1, expand=True)
arms["n_seeds"] = cf.groupby("arm", sort=False).size().values
for mname in ["gain5", "gain5_cons", "gain1", "r5", "within1_1", "busy1"]:
    lo, hi = [], []
    for a_ in arms.arm:
        if a_ == "practice:base": lo.append(practice.get(mname + "_lo", np.nan)); hi.append(practice.get(mname + "_hi", np.nan)); continue
        _, l_, h_ = boot_ci(arm_series(a_, mname)); lo.append(l_); hi.append(h_)
    arms[mname + "_lo"] = lo; arms[mname + "_hi"] = hi

# ---- admissibility (§1): selected on validation; on test descriptive only ----------------------------------------------
HEUR = ["pcp", "dist", "mips"]
pcp_r5 = float(arms.loc[arms.arm == "pcp:base", "r5"].iloc[0])
if args.split == "test": pcp_r5_floor = decision_entry["pcp_r5_floor"]
else: pcp_r5_floor = pcp_r5
arms["adm_fidelity"] = arms.r5 >= pcp_r5_floor
arms["adm_distance"] = (arms.dist5_median <= practice["dist5_median"]) & (arms.dist5_p90 <= practice["dist5_p90"])
arms["admissible"] = arms.adm_fidelity & arms.adm_distance & ~arms.family.isin(["practice", "random"]) & ~arms.family.str.endswith("_dup") & ~arms.family.str.contains("@")
if args.split == "test": arms = arms.rename(columns={"adm_fidelity": "adm_fidelity_descriptive", "adm_distance": "adm_distance_descriptive", "admissible": "admissible_descriptive"})
adm_col = "admissible" if args.split == "validation" else "admissible_descriptive"
adm = arms[arms[adm_col]].sort_values("gain5_cons", ascending=False)

def nearest_pairs():
    """Matched-fidelity pairs (§1): each model re-sort/blend point vs the PCP-habit grid points on both sides of its R@5."""
    pairs = []; pc = arms[arms.family == "pcp"].sort_values("r5")
    for _, row in arms[arms.family.str.startswith("model") & (arms.variant != "base")].iterrows():
        below = pc[pc.r5 <= row.r5].tail(1); above = pc[pc.r5 > row.r5].head(1)
        for _, p in pd.concat([below, above]).iterrows(): pairs.append([row.arm, p.arm])
    return pairs

if args.split == "validation":
    chosen = adm.iloc[0].arm if len(adm) else None
    best_model = adm[adm.family.str.startswith("model")].arm.head(1).tolist(); best_heur = adm[adm.family.isin(HEUR)].arm.head(1).tolist()
    best_model, best_heur = (best_model[0] if best_model else None), (best_heur[0] if best_heur else None)
    final_cfg = None; pairs = nearest_pairs()
    a4 = None
    if args.a4:                          # Amendment A4: v4-vs-v5 choice
        ref = "model_unmasked_v4:blend0.3"
        v5_adm = adm[adm.family == "model_unmasked_v5"]
        sel5 = v5_adm.arm.iloc[0] if len(v5_adm) else None
        if sel5:
            var = sel5.split(":", 1)[1]
            def run_cfg(suffix): return next(c_ for c_ in cf.config if c_.startswith("model:") and c_.split(":", 1)[1].split("+")[0].endswith(suffix) and (c_.endswith("+" + var) if var != "base" else "+" not in c_))
            m_ = lambda c_: float(np.mean(per[c_]["gain5_cons"]))
            band = abs(m_(run_cfg("_v5_unmasked_s42")) - m_(run_cfg("_v5_unmasked_s42_dup"))); gap = abs(m_(run_cfg("_v5_unmasked_s42")) - m_(run_cfg("_v5_unmasked_s1")))
            tol = max(band, gap); con = boot_ci(arm_series(sel5, "gain5_cons") - arm_series(ref, "gain5_cons"))
            adopt = sel5 if con[0] >= -tol else ref
            a4 = {"v5_selected": sel5, "v4_reference": ref, "contrast_gain5_cons": con, "band": band, "seed_gap": gap, "tolerance": tol, "adopted": adopt}
        else:
            a4 = {"v5_selected": None, "v4_reference": ref, "adopted": ref, "note": "no admissible v5 unmasked configuration"}
        chosen = a4["adopted"]; best_model = chosen
else:
    chosen = decision_entry["primary_contrast_config"]; final_cfg = decision_entry["final_config_10_4_to_10_6"]
    best_model, best_heur = decision_entry["best_admissible_model"], decision_entry["best_admissible_heuristic"]; pairs = decision_entry["matched_pairs"]

a10 = None
if args.load_variants and args.split == "validation":          # A10 decision rule (validation only)
    fin = arms.loc[arms.arm == "model_unmasked_v5:blend0.3"].iloc[0]
    cand = arms[arms.family.str.startswith("model") & ~arms.family.str.endswith("_dup") & ~arms.family.str.contains("@")].copy()
    cand["adm10"] = (cand.r5 >= args.fidelity_floor) & (cand.dist5_median <= practice["dist5_median"]) & (cand.dist5_p90 <= practice["dist5_p90"]) & (cand.gain5 >= fin.gain5 - 1.0)
    pick = cand[cand.adm10].sort_values(["busy1", "gain5_cons"], ascending=[True, False])
    a10 = {"fidelity_floor": args.fidelity_floor, "final_gain5_validation": float(fin.gain5), "n_admissible": int(cand.adm10.sum()), "chosen": pick.arm.iloc[0] if len(pick) else None,
           "admissible": pick[["arm", "r5", "gain5", "gain5_cons", "busy1", "busy5", "dist5_median"]].head(15).to_dict("records")}
    chosen = a10["chosen"] or chosen; best_model = chosen
contrasts = {}
for lab, a_ in [("primary", chosen), ("final", final_cfg)]:
    if a_:
        for mname in ("gain5_cons", "gain5", "gain1", "within1_1", "r5", "busy1"): contrasts[f"{lab}:{a_}|{mname}"] = boot_ci(arm_series(a_, mname))
if best_model and best_heur:
    contrasts[f"model_minus_heuristic:{best_model}−{best_heur}|gain5_cons"] = boot_ci(arm_series(best_model, "gain5_cons") - arm_series(best_heur, "gain5_cons"))
matched = []
for a_, b_ in pairs:
    for mname in ("gain5_cons", "gain5", "r5"):
        est, lo, hi = boot_ci(arm_series(a_, mname) - arm_series(b_, mname)); matched.append({"model": a_, "pcp": b_, "metric": mname, "diff": est, "lo": lo, "hi": hi})
pcp_ci = None
if chosen:
    pc_u, pc_idx = np.unique(meta.pcp.to_numpy(), return_inverse=True); x = arm_series(chosen, "gain5_cons")
    sums = np.bincount(pc_idx, weights=x, minlength=len(pc_u)); cnts = np.bincount(pc_idx, minlength=len(pc_u)); r2 = np.random.default_rng(1)
    Bp = np.array([np.bincount(r2.integers(0, len(pc_u), len(pc_u)), minlength=len(pc_u)) for _ in range(args.bootstrap)])
    est = (Bp @ sums) / (Bp @ cnts); pcp_ci = (float(x.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))

# ---- strata, sensitivities, histograms ---------------------------------------------------------------------------------------
bands = pd.cut(meta.ref_mips, [-1, 49.999, 89.999, 99.999, 101], labels=["<50", "50–<90", "90–<100", "100"]).astype(str)
BAND_M = ["gain5", "gain1", "better1", "equal1", "worse1", "within1_1", "r5"]
band_rows = []
for cfg, P in per.items():
    g = pd.DataFrame({"band": bands, "multi": meta.ref_multi.astype(str), **{m_: P[m_] for m_ in BAND_M}})
    for col, pre in (("band", "band "), ("multi", "multi=")):
        for key, gg in g.groupby(col): band_rows.append({"config": cfg, "stratum": pre + key, "n": len(gg), **gg[BAND_M].mean().to_dict()})
bt = pd.DataFrame(band_rows)
rnd = bt[bt.config == "random"].set_index("stratum")[["gain5", "gain1"]].add_suffix("_random")
bt = bt.join(rnd, on="stratum"); bt["gain5_minus_random"] = bt.gain5 - bt.gain5_random; bt["gain1_minus_random"] = bt.gain1 - bt.gain1_random
sens_rows = []
for cfg, P in per.items():
    for lab, mask in [("excluding_selection_queries", ~meta.selection_query.to_numpy()), ("filled_only", meta.filled.to_numpy()), ("not_filled", ~meta.filled.to_numpy())]:
        if mask.sum() == 0 or (lab == "excluding_selection_queries" and mask.all()): continue
        sens_rows.append({"config": cfg, "subset": lab, "n": int(mask.sum()), **{m_: float(np.mean(np.array(P[m_])[mask])) for m_ in ("gain5_cons", "gain5", "r5", "busy1")}})
hist_rows = [{"config": cfg, "bin_lo": int(lo_), "count": int(ct)} for cfg, P in per.items() for lo_, ct in zip(GAIN_BINS[:-1], np.histogram(np.clip(P["gain5"], -100, 99.99), GAIN_BINS)[0])]

# ---- write -----------------------------------------------------------------------------------------------------------------
cf.to_csv(pfx + "_configs.csv", index=False); arms.to_csv(pfx + "_arms.csv", index=False)
bt.to_csv(pfx + "_bands.csv", index=False); pd.DataFrame(sens_rows).to_csv(pfx + "_sensitivity.csv", index=False)
pd.DataFrame(hist_rows).to_csv(pfx + "_gain_hist.csv", index=False)
pd.DataFrame(matched).to_csv(pfx + "_matched_fidelity.csv", index=False)
pr = meta.copy()
for cfg in per:
    for mname in ("gain5", "gain5_cons", "gain1", "r5", "within1_1", "busy1", "busy5", "new_pcp5", "new_pcp1", "new_ins5", "good_new5", "none5", "dist5m", "conc1", "conc5", "early1", "early5", "lowexp1", "lowexp5", "nonmetro1", "nonmetro5",
                  "volpct1_ft", "volpct1_all", "tau_vol", "gap12", "top1_spec"):
        if mname in per[cfg] and len(per[cfg][mname]) == len(pr): pr[f"{cfg}|{mname}"] = per[cfg][mname]
pr.to_csv(pfx + "_per_referral_PRIVATE.csv.gz", index=False)
if DUMP_ROWS: pd.DataFrame(DUMP_ROWS).to_csv(pfx + "_top5_picks_PRIVATE.csv.gz", index=False); print("saved top-5 picks:", len(DUMP_ROWS))
if args.dump_scores:
    np.savez_compressed(pfx + "_candidate_scores_PRIVATE.npz", enc=np.array([r.enc for r in R]), pcp=np.array([r.pcp for r in R]), day=np.array([r.day for r in R]),
                        specialty=np.array([r.specialty for r in R]), offsets=np.r_[0, np.cumsum(DS["n"])], target_pos=np.array(DS["target_pos"]), ref_mips=np.array(DS["ref_mips"]),
                        **{k_: np.concatenate(v_) for k_, v_ in DS.items() if k_ not in ("n", "target_pos", "ref_mips")})
    print("saved candidate scores:", sum(DS["n"]), "candidates", flush=True)
summary = {"step": args.step, "split": args.split, "runs": run_names, "inputs": sorted(set(os.path.relpath(p_, root) for p_ in run_paths)), "funnel": dict(cohort.counts), "n": n_ref, "practice": practice,
           "pcp_habit_r5_this_split": pcp_r5, "pcp_r5_floor": pcp_r5_floor, "chosen": chosen, "final_config": final_cfg,
           "best_admissible_model": best_model, "best_admissible_heuristic": best_heur, "matched_pairs": pairs,
           "contrasts": contrasts, "pcp_cluster_ci_primary_gain5_cons": pcp_ci,
           "n_admissible": int(len(adm)), "admissible": adm[["arm", "r5", "gain5", "gain5_cons", "dist5_median", "dist5_p90"]].to_dict("records"),
           "training_load_reference": f"all first-time training referrals ({len(q_tr)}); busiest = above specialty p90 among training-active specialists",
           "a10": (a10 if args.split == "validation" else decision_entry.get("a10")) if args.load_variants else None,
           "a4_v4_vs_v5": (a4 if args.split == "validation" else decision_entry.get("a4_v4_vs_v5")) if args.a4 else None, "setting": setting,
           "protocol": "STEP10_PROTOCOL.md (frozen 2026-09-28) + A1–A4", "hybrid_info": HYB_INFO or None,
           "test_note": "2019 test year inspected once on 2026-09-25; admissibility columns descriptive only" if args.split == "test" else None}
if decision_entry: summary["decision_entry"] = decision_entry; summary["decision_entry_sha256"] = decision_hash
json.dump(summary, open(pfx + "_summary.json", "w"), indent=2, default=float)

show = ["arm", "n_seeds", "r5", "gain5", "gain5_cons", "gain1", "better1", "within1_1", "new_pcp5", "dist5_median", "busy1", "busy5", "gini_within", "gini_within_5", adm_col]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(arms[show].round(3).to_string(index=False))
print(f"\npractice: dist median {practice['dist5_median']:.1f} p90 {practice['dist5_p90']:.1f}; busy1 {practice['busy1']:.3f}; gini {practice['gini_within']:.3f}; PCP R@5 {pcp_r5:.3f} (floor {pcp_r5_floor:.3f})")
print("primary:", chosen, "| final:", final_cfg, "| best model:", best_model, "| best heuristic:", best_heur)
print("contrasts:", json.dumps(contrasts, default=float, indent=0)); print("PCP-cluster CI (primary, gain5_cons):", pcp_ci)
