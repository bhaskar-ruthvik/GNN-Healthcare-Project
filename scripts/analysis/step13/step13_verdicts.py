"""STEP13 phase-1 verdicts (Implementation/STEP13_GRAPH_VALUE_PLAN.md, frozen 2026-10-21; definitions in referral_gnn/step13.py).

Reads the candidate-score dump that scripts/evaluate/step10_evaluate.py --dump-scores wrote for every cell's runs, and the design
and resolved manifests (runs/analysis/step13/). Writes <eval-dir>/<split>_step13_verdicts.json (aggregates only).
  validation (2018)  native lenses of every cell first; then the six claim units: component statistics (two-level paired
                     bootstrap, PCP-only intervals, seed-level effects), unit p = max component p, Holm across the 6 units,
                     the seed-direction rule, the H3 matching gates, and practical equivalence (descriptive); the H3 matching
                     coefficients chosen on the full 2018 data are recorded for 2019.
  test (2019)        once, after a step10_decide.py entry for step 13 that hashes the 2018 verdict file: directional replication
                     (every component's 2019 point estimate in the predicted direction), CIs, and the achieved 2019 newness at the
                     frozen coefficients.
Usage: python scripts/analysis/step13/step13_verdicts.py --split validation
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from referral_gnn import step13 as S  # noqa: E402
from referral_gnn.candidates import Dump  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["validation", "test"])
ap.add_argument("--eval-dir", default="runs/analysis/step13/eval")
ap.add_argument("--manifests", default="runs/analysis/step13")
ap.add_argument("--draws", type=int, default=S.DRAWS, help="smoke tests only; the frozen value is step13.DRAWS")
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
design = json.load(open(os.path.join(args.manifests, "design_manifest.json")))
resolved = json.load(open(os.path.join(args.manifests, "resolved_manifest.json")))
assert design["plan_sha256"] == S.PLAN_SHA256 and resolved["design_manifest_sha256"] == sha(os.path.join(args.manifests, "design_manifest.json")), \
    "manifests do not match the frozen plan / each other"
AMEND = S.amendment_records(args.manifests)
assert S.expected_amendments_sha256(design, AMEND) == S.AMENDMENTS_SHA256 == sha(os.path.join(root, "STEP13_AMENDMENTS.md")), \
    "amendments differ from the design manifest and its recorded amendments"
assert design["tests_record"] == S.tests_record(), "test definitions in referral_gnn/step13.py differ from the design manifest"
drift = S.code_drift(design, str(code_file("referral_gnn").parent), AMEND); assert not drift, f"code changed since the design manifest: {drift}"
SMOKE = args.draws != S.DRAWS                                       # smoke runs never touch the single-use verdict file
out_path = os.path.join(args.eval_dir, f"{args.split}_step13_verdicts" + (f"_SMOKE{args.draws}" if SMOKE else "") + ".json")
assert SMOKE or not os.path.exists(out_path), f"{out_path} exists; verdicts are written once per split"
frozen_a = None
if args.split == "test":
    vpath = os.path.join(args.eval_dir, "validation_step13_verdicts.json"); assert os.path.exists(vpath), "2018 verdicts first"
    dec = json.load(open("runs/analysis/step10_decisions.json")) if os.path.exists("runs/analysis/step10_decisions.json") else {"entries": []}
    ok = [e for e in dec["entries"] if e["step"] == "13" and e["validation_files_sha256"].get("validation_step13_verdicts.json") == sha(vpath)]
    assert ok, "REFUSED: no step-13 decision entry hashing the current 2018 verdict file (scripts/evaluate/step10_decide.py)"
    V = json.load(open(vpath)); frozen_a = V["h3_matching"]["chosen_a"]
CELL_RUNS = resolved["cells"]                                       # {cell: {seed: run folder}}
D = Dump(os.path.join(args.eval_dir, f"{args.split}_candidate_scores_PRIVATE.npz"))
enc, pcp, day = D.Z["enc"], D.pcp, D.Z["day"]
cohort_sha = hashlib.sha256(np.asarray(enc, np.int64).tobytes()).hexdigest()
if args.split == "validation": assert cohort_sha == design["cohort_2018"]["enc_sha256"], "2018 cohort differs from the design manifest"
key = {c: {s: "score::" + CELL_RUNS[c][str(s)] for s in S.SEEDS} for c in S.CELLS}
missing = [k for c in key for k in key[c].values() if k not in D.Z.files]; assert not missing, f"runs missing from the dump: {missing}"

from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
mi = load_three_node_inputs(os.path.join(root, design["inputs"]["path"]))
assert sha(os.path.join(root, design["inputs"]["path"])) == design["inputs"]["sha256"], "inputs changed since the design manifest"
H = S.History(mi)
masks = S.slice_masks(mi, H, enc, pcp, day, design["slices"]["thin_network"]["B2"], design["slices"]["rare"]["dx_ids"])
known = mi.enc_insurer[enc] >= 0
codes, uniq = pd.factorize(pcp); B = S.Boot(codes, len(uniq), args.draws)
t0 = time.time(); out = {"split": args.split, "n_referrals": int(D.n_ref), "n_pcps": int(len(uniq)), "cohort_enc_sha256": cohort_sha, "draws": args.draws,
                         "plan_sha256": S.PLAN_SHA256, "design_manifest_sha256": resolved["design_manifest_sha256"],
                         "amendments": [{"id": r["id"], "commit": r["commit"]} for r in AMEND],
                         "resolved_manifest_sha256": sha(os.path.join(args.manifests, "resolved_manifest.json")), "dump_sha256": sha(D.path)}

# ---- 1. native lenses of every cell (before any matching) ---------------------------------------------------------------
NAT = {c: np.stack([S.native(D, key[c][s]) for s in S.SEEDS]) for c in S.CELLS}           # (3 seeds, n_ref, 6)
out["native_lenses"] = {c: {"mean_over_seeds": dict(zip(S.NATIVE_COLS, np.nanmean(NAT[c], axis=1).mean(0).round(6).tolist())),
                            "per_seed": {str(s): dict(zip(S.NATIVE_COLS, np.nanmean(NAT[c][k], axis=0).round(6).tolist())) for k, s in enumerate(S.SEEDS)}}
                        for c in S.CELLS}
print(f"native lenses: {time.time() - t0:.0f}s", flush=True)

# ---- 2. fit components ---------------------------------------------------------------------------------------------------
ALL = np.ones(D.n_ref, bool)
pops = {"all": ALL, **{f"{s}:in": masks[s][0] for s in S.SLICES}, **{f"{s}:rest": masks[s][1] for s in S.SLICES}}


def fit_stats(cell, pop):
    m = pops[pop]; A = np.stack([B.per_pcp(NAT[cell][k][:, 0], m) for k in range(len(S.SEEDS))]); N = B.per_pcp(m.astype(float))
    return B.mean_draws(A, N)


FIT = {}


def fit_value(c, pop):
    if (c, pop) not in FIT: FIT[(c, pop)] = fit_stats(c, pop)
    return FIT[(c, pop)]


def component_fit(comp):
    parts = []
    for c, w in comp["w"].items():
        if comp.get("interaction"):
            a_, b_ = fit_value(c, f"{comp['pop']}:in"), fit_value(c, f"{comp['pop']}:rest"); parts.append([w * (x - y) for x, y in zip(a_, b_)])
        else:
            pop = "all" if comp["pop"] == "all" else f"{comp['pop']}:in"; parts.append([w * x for x in fit_value(c, pop)])
    two, pcp_only, point, per_seed = (sum(p[i] for p in parts) for i in range(4))
    return two, pcp_only, point, per_seed, np.isnan(two)


# ---- 3. H3: matched conditional compatibility ------------------------------------------------------------------------------
GRID = design["matching"]["grid"]; TARGET, TOL = design["matching"]["target_newness"], design["matching"]["tolerance"]
assert GRID == S.MATCH_GRID and TARGET == S.TARGET_NEWNESS and TOL == S.TOLERANCE, "matching settings differ from step13.py"
H3_CELLS = sorted({c for comp in S.UNITS["H3"] + [S.H3_REFERENCE] for c in comp["w"]})
slots = np.minimum(np.diff(D.off), S.K).astype(float); SL = B.per_pcp(slots)
H3 = {}; chosen = {}; attained_point = {}; newness_point = {}; ins_groups = {}; newness_draws = {}
for c in H3_CELLS:
    NEW, NUM, DEN = (np.zeros((len(S.SEEDS), len(GRID), B.n_pcp)) for _ in range(3)); per_ref = []
    for k, s in enumerate(S.SEEDS):
        n_, num_, den_ = S.steered_counts(D, key[c][s], GRID, known); per_ref.append((n_, num_, den_))
        for g in range(len(GRID)):
            NEW[k, g], NUM[k, g], DEN[k, g] = B.per_pcp(n_[:, g]), B.per_pcp(num_[:, g]), B.per_pcp(den_[:, g])
    fz = None if frozen_a is None else np.array([GRID.index(frozen_a[c][str(s)]) for s in S.SEEDS])
    newd = (B.W @ NEW.reshape(-1, B.n_pcp).T).reshape(B.draws, len(S.SEEDS), len(GRID)) / (B.W @ SL)[:, None, None]
    ch, att = S.match(newd, GRID, TARGET, TOL, None if fz is None else np.broadcast_to(fz, (B.draws, len(S.SEEDS))))
    sel = lambda X: np.take_along_axis((B.W @ X.reshape(-1, B.n_pcp).T).reshape(B.draws, len(S.SEEDS), len(GRID)), ch[..., None], axis=-1)[..., 0]
    num_d, den_d = sel(NUM), sel(DEN)
    with np.errstate(invalid="ignore", divide="ignore"): rate = np.where(den_d > 0, num_d / np.where(den_d > 0, den_d, 1), np.nan)
    two = np.take_along_axis(rate, B.S, axis=1).mean(1); att_two = np.take_along_axis(att, B.S, axis=1).all(1)
    got = np.take_along_axis(newd, ch[..., None], axis=-1)[..., 0]                               # achieved newness per draw and seed
    newness_draws[c] = {"per_seed_attainment_share": dict(zip(map(str, S.SEEDS), att.mean(0).round(4).tolist())),
                        "achieved_newness_p5_p50_p95": np.percentile(got, [5, 50, 95]).round(4).tolist(), "tolerance_missed_share": float(1 - att.mean())}
    new0 = NEW.sum(2) / SL.sum()                                                                  # full data: (3, grid)
    ch0, att0 = S.match(new0, GRID, TARGET, TOL, fz)
    num0 = np.take_along_axis(NUM.sum(2), ch0[:, None], 1)[:, 0]; den0 = np.take_along_axis(DEN.sum(2), ch0[:, None], 1)[:, 0]
    rate0 = np.where(den0 > 0, num0 / np.where(den0 > 0, den0, 1), np.nan)
    H3[c] = (two, rate.mean(1), float(np.mean(rate0)), rate0, att_two, att.mean(0))
    chosen[c] = {str(s): GRID[int(ch0[k])] for k, s in enumerate(S.SEEDS)}; attained_point[c] = {str(s): bool(att0[k]) for k, s in enumerate(S.SEEDS)}
    newness_point[c] = {str(s): float(new0[k, ch0[k]]) for k, s in enumerate(S.SEEDS)}
    ins = mi.enc_insurer[enc]; grp = {}
    for v in np.unique(ins[known]):                                   # composition sensitivity (descriptive): by query insurer
        m = ins == v; nn = sum(per_ref[k][0][m, ch0[k]].sum() for k in range(3)); nu = sum(per_ref[k][1][m, ch0[k]].sum() for k in range(3))
        de = sum(per_ref[k][2][m, ch0[k]].sum() for k in range(3)); sl = 3 * slots[m].sum()
        grp[f"insurer_{int(v)}"] = {"referrals": int(m.sum()), "newness": float(nn / sl), "compat_rate": float(nu / de) if de else None}
    ins_groups[c] = {"by_insurer": grp, "numerator_slots": int(num0.sum()), "denominator_slots": int(den0.sum()),
                     "referrals_with_zero_new_picks": {str(s): int((per_ref[k][0][:, ch0[k]] == 0).sum()) for k, s in enumerate(S.SEEDS)}}
    print(f"  H3 {c}: chosen a {chosen[c]}, newness {newness_point[c]} ({time.time() - t0:.0f}s)", flush=True)
out["h3_matching"] = {"grid": GRID, "target_newness": TARGET, "tolerance": TOL, "chosen_a": chosen if frozen_a is None else frozen_a,
                      "frozen_from_2018": frozen_a is not None, "attained_point_estimate": attained_point, "achieved_newness": newness_point,
                      "attainment_share_in_draws": {c: float(H3[c][4].mean()) for c in H3_CELLS}, "newness_in_draws": newness_draws,
                      "ci_note": "H3 intervals use the nearest-grid contrast in draws that miss the tolerance; see newness_in_draws",
                      "composition": ins_groups}


def component_compat(comp):
    cells = list(comp["w"])
    two = sum(w * H3[c][0] for c, w in comp["w"].items()); pcp_only = sum(w * H3[c][1] for c, w in comp["w"].items())
    point = sum(w * H3[c][2] for c, w in comp["w"].items()); per_seed = sum(w * H3[c][3] for c, w in comp["w"].items())
    undefined = np.isnan(two) | (~np.all([H3[c][4] for c in cells], axis=0) if args.split == "validation" else False)
    return two, pcp_only, point, per_seed, undefined


def evaluate_component(comp):
    two, pcp_only, point, per_seed, undefined = (component_compat if comp["outcome"] == "compat" else component_fit)(comp)
    r = {"name": comp["name"], "outcome": comp["outcome"], "point": float(point), "per_seed": dict(zip(map(str, S.SEEDS), np.round(per_seed, 6).tolist())),
         **S.summarise(two, undefined)}
    ok = ~np.isnan(pcp_only); r["pcp_only_ci95"] = [float(np.percentile(pcp_only[ok], 2.5)), float(np.percentile(pcp_only[ok], 97.5))] if ok.any() else None
    r["seed_direction"] = bool(np.all(np.asarray(per_seed) > 0))
    return r


# ---- 4. units, Holm, verdicts ---------------------------------------------------------------------------------------------
units = {}
for u, comps in S.UNITS.items():
    res = [evaluate_component(c) for c in comps]; info = {"components": res}
    if u.startswith("H2_"):
        sl = u[3:]; m = masks[sl][0]; info["slice_n"] = int(m.sum()); info["slice_pcps"] = int(len(np.unique(pcp[m]))); info["rest_n"] = int(masks[sl][1].sum())
        info["testable"] = bool(info["slice_n"] >= S.MIN_SLICE_N and info["slice_pcps"] >= S.MIN_SLICE_PCPS)
    else:
        info["testable"] = True
    info["p_unit"] = float(max(r["p_one_sided"] for r in res)) if info["testable"] else 1.0
    gates = []
    if any(r["undefined_share"] > S.UNDEFINED_GATE for r in res): gates.append("more than 5% of draws undefined for a component")
    if u == "H3" and args.split == "validation":
        cells_ = sorted({c for comp in comps for c in comp["w"]})
        if any(H3[c][5].min() < S.ATTAIN_GATE for c in cells_): gates.append("a model (cell × seed) attains the newness tolerance in < 95% of draws")
        if not all(attained_point[c][str(s)] for c in cells_ for s in S.SEEDS): gates.append("a per-seed 2018 point estimate misses the newness tolerance")
    info["inconclusive_reasons"] = gates; info["seed_direction_all"] = bool(all(r["seed_direction"] for r in res))
    info["practically_equivalent_components"] = [r["name"] for r in res if r["practically_equivalent_tost"]]
    units[u] = info
if args.split == "validation":
    adj = S.holm([units[u]["p_unit"] for u in S.UNITS])
    for u, a_ in zip(S.UNITS, adj):
        x = units[u]; x["p_holm"] = float(a_)
        if not x["testable"]: x["verdict"] = "not testable (slice below the size floor)"
        elif x["inconclusive_reasons"]: x["verdict"] = "inconclusive"
        elif a_ < S.ALPHA and x["seed_direction_all"]: x["verdict"] = "supported"
        else: x["verdict"] = "not supported"
        if u == "H1a" and x["components"][0]["p_one_sided"] >= S.ALPHA: x["note"] = "recovery fraction not established (prerequisite not met)"
else:
    for u, x in units.items():
        was = V["units"][u]["verdict"]; x["verdict_2018"] = was
        pos = all(r["point"] > 0 for r in x["components"])
        if not x["testable"]: x["replication"] = "not replicable (insufficient size)"
        elif was == "supported": x["replication"] = "directional replication" if pos else "not replicated"
        else: x["replication"] = "not applicable (not supported on 2018)"; x["all_points_positive_2019"] = pos
        if u in S.HEADLINE_ONLY_IF_REPLICATED and was == "supported":
            x["headline"] = x["replication"] == "directional replication"
            if not x["headline"]: x["reported_as"] = "supported on 2018, not replicated"
out["units"] = units
out["h3_reference_insmatch"] = evaluate_component(S.H3_REFERENCE)

# ---- 5. descriptives -------------------------------------------------------------------------------------------------------
desc = {"G_beside_H1a_fit": out["native_lenses"]["g"]["mean_over_seeds"]["fit"]}
per_slice = {}
for sl in S.SLICES:
    m = masks[sl][0]
    per_slice[sl] = {c: dict(zip(S.NATIVE_COLS, np.nanmean(NAT[c][:, m], axis=1).mean(0).round(6).tolist())) for c in S.CELLS} if m.any() else None
desc["lenses_by_slice"] = per_slice
m = masks["rare"][0]                                               # A2: the rare slice is descriptive only (no inference)
desc["rare_slice_A2"] = {"n": int(m.sum()), "pcps": int(len(np.unique(pcp[m]))), "rest_n": int(masks["rare"][1].sum()),
                         "graph_minus_comparator_fit": {c: {"slice": float(NAT["graph"][:, m, 0].mean() - NAT[c][:, m, 0].mean()) if m.any() else None,
                                                            "rest": float(NAT["graph"][:, masks["rare"][1], 0].mean() - NAT[c][:, masks["rare"][1], 0].mean())}
                                                        for c in ("ngp", "info", "mlp_info")}}
sens = {}
for nm, folder in resolved.get("sensitivity", {}).items():          # §1 sensitivities (descriptive): native lenses
    k_ = "score::" + folder
    if k_ in D.Z.files: sens[nm] = dict(zip(S.NATIVE_COLS, np.nanmean(S.native(D, k_), axis=0).round(6).tolist()))
desc["sensitivity_runs"] = sens
tgt = mi.enc_target[enc]; lowexp = H.spec_prior(tgt, day) < 10
desc["low_experience_specialist_slice"] = {"n": int(lowexp.sum()), "caveat": "label-defined; MUP volume year caveat; not counted toward any headline",
                                           "fit_in_slice": {c: float(NAT[c][:, lowexp, 0].mean()) for c in S.CELLS} if lowexp.any() else None}
cand_low = H.spec_prior(D.Z["spec"], np.repeat(day, np.diff(D.off))) < 10
avail = float(np.mean([cand_low[a:b].mean() for a, b in D.seg]))
expo = {}
for c in S.CELLS:
    sh = []
    for k, s in enumerate(S.SEEDS):
        base = D.weighted(D.scores(key[c][s])); sh.append(np.mean([cand_low[D.top(base, i, k=S.K)].mean() for i in range(D.n_ref)]))
    expo[c] = float(np.mean(sh))
desc["candidate_side_low_experience_exposure"] = {"top5_share": expo, "eligible_set_availability": avail}
nopat = H.patient_prior(enc) == 0
desc["no_prior_encounter_patients"] = {"n": int(nopat.sum()), "note": "heterogeneity contrast, not a negative control",
                                       "fit": {c: {"group": float(NAT[c][:, nopat, 0].mean()) if nopat.any() else None, "rest": float(NAT[c][:, ~nopat, 0].mean())} for c in S.CELLS}}
out["descriptive"] = desc
out["runtime_s"] = round(time.time() - t0, 1)
json.dump(out, open(out_path, "w"), indent=1, default=float)
print(f"\nwrote {out_path}")
for u, x in units.items():
    tag = x.get("verdict") if args.split == "validation" else f"{x['replication']} (2018: {x['verdict_2018']})"
    print(f"{u:18s} {tag:40s} p_unit {x['p_unit']:.4f}" + (f"  Holm {x['p_holm']:.4f}" if "p_holm" in x else ""))
    for r in x["components"]:
        print(f"    {r['name']:48s} {r['point']:+.4f}  CI95 [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  p {r['p_one_sided']:.4f}  seeds {r['per_seed']}")
