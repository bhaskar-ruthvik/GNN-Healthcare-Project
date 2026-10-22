"""STEP13 phase 1: cell and claim-unit definitions, slices, lenses, the paired two-level bootstrap, H3 matching and the verdict
rules (Implementation/STEP13_GRAPH_VALUE_PLAN.md, FROZEN 2026-10-21, SHA-256 below). The design manifest, the verdict script and
the reviewers read the definitions from this one module.

Outcomes (computed from a candidate-score dump of scripts/evaluate/step10_evaluate.py --dump-scores; candidates.Dump arithmetic):
  fit     native fit at q30: the referred specialist is in the top 5 of 0.7·z(model) + 0.3·z(as-of MIPS) (stable ties)
  compat  H3: among new-to-PCP slots of the top 5 at 0.7·z(model) + 0.3·z(as-of MIPS) + a·[new], the share whose specialist has
          seen the query's insurer before the referral day (slot-weighted; slots with an unknown query insurer excluded), with a
          chosen per model and seed so that newness is closest to TARGET_NEWNESS (re-matched in every draw; frozen for 2019)
Bootstrap (§1): DRAWS draws; per draw the 3 seed IDs are resampled with replacement and applied to every model, and PCPs are
resampled with replacement with the same draw for every model and contrast; every statistic and the matching are recomputed in
the draw. One-sided p = (1 + #draws with contrast ≤ 0 or undefined) / (1 + DRAWS).
"""
from __future__ import annotations

import numpy as np

PLAN = "Implementation/STEP13_GRAPH_VALUE_PLAN.md"
PLAN_SHA256 = "42c11662195dbba811115e704f53c4f15181c343dbbaa2d11c3b64468af36809"
AMENDMENTS = "Implementation/STEP13_AMENDMENTS.md"   # A1 (fine H3 grid), A2 (rare slice descriptive), both 2026-10-21, before any STEP13 run
AMENDMENTS_SHA256 = "00c40d259f9ef54ba8dfce22f402b6223f0caebde91cf7886b9120334f23fffa"   # after A3 (compat criterion), 2026-10-21
SEEDS = (42, 1, 7)                     # paired by seed ID across models (a convention, not shared randomness)
DRAWS = 2000
RNG_PCP, RNG_SEED = 20261021, 20261022
ALPHA, DELTA, K = 0.05, 0.02, 5        # Holm level; TOST margin (2 points); list length
UNDEFINED_GATE, ATTAIN_GATE = 0.05, 0.95
MATCH_GRID = [round(-0.6 + 0.01 * i, 2) for i in range(161)]          # A1: −0.60 … 1.00, step 0.01
TARGET_NEWNESS, TOLERANCE = 0.353, 0.01
MIN_SLICE_N, MIN_SLICE_PCPS = 100, 10

# ---- cells ---------------------------------------------------------------------------------------------------------------
HIST_ABL = ["no_patient_history", "no_predecessor_gap", "no_history_features", "no_patient_specialist_prior", "no_insurer_history"]
CELLS = {
    # existing runs are named by their folder; new runs by run name (resolved to a folder in the resolved manifest)
    "graph": dict(desc="v5 GNN, message passing", kind="gnn",
                  runs={42: "20260929T213930_v5_unmasked_s42", 1: "20260930T094407_v5_unmasked_s1", 7: "s13_graph_s7"},
                  train={7: "--seed 7 --arm model_s13_graph"}),
    "ngp": dict(desc="no message passing; scorer also reads the PCP encoding", kind="gnn",
                runs={s: f"s13_ngp_s{s}" for s in SEEDS}, train={s: f"--seed {s} --no-graph --pcp-in-scorer --arm model_s13_ngp" for s in SEEDS}),
    "g": dict(desc="no message passing, no PCP encoding, explicit history removed (STEP12a gate)", kind="gnn",
              runs={42: "20261013T181555_s12_gate_s42", 1: "20261013T182240_s12_gate_s1", 7: "20261013T182716_s12_gate_s7"}),
    "gnn_nohist": dict(desc="Graph with explicit history removed", kind="gnn",
                       runs={s: f"s13_gnn_nohist_s{s}" for s in SEEDS},
                       train={s: f"--seed {s} " + " ".join(f"--ablation {a}" for a in HIST_ABL) + " --arm model_s13_gnn_nohist" for s in SEEDS}),
    "ngp_nohist": dict(desc="NG+P with explicit history removed", kind="gnn",
                       runs={s: f"s13_ngp_nohist_s{s}" for s in SEEDS},
                       train={s: f"--seed {s} --no-graph --pcp-in-scorer " + " ".join(f"--ablation {a}" for a in HIST_ABL) + " --arm model_s13_ngp_nohist" for s in SEEDS}),
    "table": dict(desc="pure-tabular LightGBM", kind="lgbm",
                  runs={42: "20261007T195708_lgbm_tabular_s42", 1: "20261007T195853_lgbm_tabular_s1", 7: "20261014T174500_s12_lgbm_tabular_lambdarank_L127_M50_R1000_s7"}),
    "flat_rf": dict(desc="receptive-field-matched fixed aggregates, LightGBM", kind="lgbm", feature_set="flat_rf",
                    runs={s: f"s13_lgbm_flat_rf_lambdarank_s{s}" for s in SEEDS}),
    "mlp_rf": dict(desc="MLP on the Flat-RF row", kind="mlp", feature_set="flat_rf", runs=None),
    "info_nohist": dict(desc="practical superset without explicit pair or patient history, LightGBM", kind="lgbm", feature_set="full_nohist",
                        runs={s: f"s13_lgbm_full_nohist_lambdarank_s{s}" for s in SEEDS}),
    "info": dict(desc="practical superset with all GNN inputs + aggregates incl. A3, LightGBM", kind="lgbm", feature_set="full",
                 runs={s: f"s13_lgbm_full_lambdarank_s{s}" for s in SEEDS}),
    "mlp_info": dict(desc="MLP on the Info-table row", kind="mlp", feature_set="full", runs=None),
    "insmatch": dict(desc="Info-table-nohist + the query insurer's share of the specialist's past encounters", kind="lgbm", feature_set="insmatch",
                     runs={s: f"s13_lgbm_insmatch_lambdarank_s{s}" for s in SEEDS}),
}


def lin(**w):
    return {k: float(v) for k, v in w.items()}


# ---- sensitivities (§1; descriptive): Graph s42_dup in place of s42, extra G seeds ---------------------------------------------
SENSITIVITY_RUNS = {"graph_s42_dup": "20260930T140429_v5_unmasked_s42_dup", "g_s2": "20261013T221541_s12_gate_s2",
                    "g_s3": "20261013T223007_s12_gate_s3", "g_s4": "20261013T223531_s12_gate_s4"}

# ---- claim units (§2): each component is a linear combination of cell outcomes, predicted > 0 ------------------------------
SLICES = ("low_history", "thin_network", "rare")         # all three are computed and described
UNIT_SLICES = ("low_history", "thin_network")            # A2: the rare slice is descriptive only (Holm over 5 units)
UNITS = {
    "H1a": [dict(name="prerequisite: Graph − NG+P-nohist", outcome="fit", pop="all", w=lin(graph=1, ngp_nohist=-1)),
            dict(name="GNN-nohist − NG+P-nohist", outcome="fit", pop="all", w=lin(gnn_nohist=1, ngp_nohist=-1)),
            dict(name="joint half-gap", outcome="fit", pop="all", w=lin(gnn_nohist=1, ngp_nohist=-0.5, graph=-0.5))],
    "H1b": [dict(name=f"GNN-nohist − {x}", outcome="fit", pop="all", w={"gnn_nohist": 1.0, x: -1.0}) for x in ("flat_rf", "mlp_rf", "table", "info_nohist")],
    **{f"H2_{s}": [dict(name=f"{kind}: Graph − {c} ({s})", outcome="fit", pop=s, interaction=(kind == "interaction"), w={"graph": 1.0, c: -1.0})
                   for c in ("ngp", "info", "mlp_info") for kind in ("absolute", "interaction")] for s in UNIT_SLICES},
    "H3": [dict(name=f"GNN-nohist − {x} (matched newness)", outcome="compat", pop="all", w={"gnn_nohist": 1.0, x: -1.0})
           for x in ("ngp_nohist", "flat_rf", "mlp_rf", "info_nohist")],
}
H3_REFERENCE = dict(name="GNN-nohist − Ins-match (descriptive)", outcome="compat", pop="all", w={"gnn_nohist": 1.0, "insmatch": -1.0})
HEADLINE_ONLY_IF_REPLICATED = ("H2_low_history", "H2_thin_network")

# ---- slices (§3) ---------------------------------------------------------------------------------------------------------
LOW_HISTORY_MAX = 951                  # PCP prior referrals strictly before the referral day; 2018 lowest-quintile bound (STEP12 groups)
RARE_N = 11                            # the 11 least frequent of the 33 source categories by training-year encounter count


class History:
    """As-of PCP attributes from the encounter arrays (strictly before day d)."""

    BIG = 1 << 20

    def __init__(self, mi):
        self.mi = mi
        self.pk = np.sort(mi.enc_pcp.astype(np.int64) * self.BIG + mi.enc_day.astype(np.int64))
        first = {}
        for p, s, d in zip(mi.enc_pcp, mi.enc_target, mi.enc_day):
            k = (int(p), int(s)); first[k] = min(first.get(k, 1 << 30), int(d))
        by_p = {}
        for (p, _), d in first.items(): by_p.setdefault(p, []).append(d)
        self.pfirst = {p: np.sort(np.asarray(v)) for p, v in by_p.items()}
        self.sk = np.sort(mi.enc_target.astype(np.int64) * self.BIG + mi.enc_day.astype(np.int64))
        self.patk = np.sort(mi.enc_patient.astype(np.int64) * self.BIG + mi.enc_day.astype(np.int64))

    def pcp_prior(self, p, d):
        p = np.asarray(p, np.int64); d = np.asarray(d, np.int64)
        return np.searchsorted(self.pk, p * self.BIG + d) - np.searchsorted(self.pk, p * self.BIG)

    def pcp_net(self, p, d):
        return np.array([int(np.searchsorted(self.pfirst[int(p_)], int(d_))) if int(p_) in self.pfirst else 0 for p_, d_ in zip(p, d)])

    def spec_prior(self, s, d):
        s = np.asarray(s, np.int64); d = np.asarray(d, np.int64)
        return np.searchsorted(self.sk, s * self.BIG + d) - np.searchsorted(self.sk, s * self.BIG)

    def patient_prior(self, enc):
        enc = np.asarray(enc, np.int64); pt = self.mi.enc_patient[enc].astype(np.int64); d = self.mi.enc_day[enc].astype(np.int64)
        return np.searchsorted(self.patk, pt * self.BIG + d) - np.searchsorted(self.patk, pt * self.BIG)


def rare_categories(mi):
    """(rare dx ids, per-id training-year counts): ids 2.. ranked by encounters before the validation year; ties by id."""
    cutoff = int(mi.split_days["validation"][0]); assert cutoff == 17532, "validation year must start 2018-01-01 (day 17532)"
    train = mi.enc_day < cutoff
    cats = np.arange(2, int(mi.enc_dx.max()) + 1); assert len(cats) == 33 == len(mi.dx_vocab.index), "expected the 33 source categories"
    cnt = {int(c): int(((mi.enc_dx[train] == c) & mi.enc_dx_mask[train]).any(axis=1).sum()) for c in cats}
    order = sorted(cnt, key=lambda c: (cnt[c], c))
    return sorted(order[:RARE_N]), cnt


def thin_network_bound(pcp_net_2018):
    """B₂: the 2018 cohort's lower-tertile bound of PCP network size (linear interpolation, as pandas.qcut)."""
    return float(np.quantile(np.asarray(pcp_net_2018, float), 1 / 3))


def slice_masks(mi, H: History, enc, pcp, day, b2: float, rare_ids):
    """{slice: (in-slice mask, rest mask)}; referrals excluded from both are neither."""
    pr = H.pcp_prior(pcp, day); net = H.pcp_net(pcp, day)
    out = {"low_history": (pr <= LOW_HISTORY_MAX, pr > LOW_HISTORY_MAX), "thin_network": (net <= b2, net > b2)}
    rare = np.zeros(len(enc), bool); known = np.zeros(len(enc), bool); rs = set(int(c) for c in rare_ids)
    for i, e in enumerate(enc):
        c = mi.enc_dx[e][mi.enc_dx_mask[e]]
        known[i] = len(c) > 0 and not (c <= 1).any()                          # any unknown code (id 1) → excluded from slice and rest
        rare[i] = known[i] and any(int(x) in rs for x in c)
    out["rare"] = (rare, known & ~rare)
    return out


# ---- per-referral outcomes from a dump ------------------------------------------------------------------------------------
def native(D, key):
    """Per referral: fit, busy1, busy5, new5, gain5 (MIPS gain), dist5 (mean distance of top 5) — no steering."""
    base = D.weighted(D.scores(key)); dist = D.dist; out = np.zeros((D.n_ref, 6))
    for i in range(D.n_ref):
        j = D.top(base, i, k=K)
        out[i] = (float(D.tpos[i] >= 0 and (D.tpos[i] + D.off[i]) in set(j.tolist())), D.busy[j[0]], D.busy[j].mean(), D.new[j].mean(),
                  D.mips[j].mean() - D.refm[i], np.nanmean(dist[j]) if np.isfinite(dist[j]).any() else np.nan)
    return out


NATIVE_COLS = ("fit", "busy1", "busy5", "new5", "gain5", "dist5")


def steered_counts(D, key, grid, known):
    """(new slots, compatible new slots, new slots with a known insurer) per referral × grid value: arrays (n_ref, len(grid)).
    All grid values at once per referral; the same arithmetic and stable ties as Dump.top (base + a·new − 0·busy)."""
    base = D.weighted(D.scores(key)); n = np.zeros((D.n_ref, len(grid))); num = np.zeros_like(n); den = np.zeros_like(n)
    ins = D.Z["ins"] > 0                                                         # read once (npz members decompress on every access)
    A = np.asarray(grid, float)[:, None]
    for i, (a, b) in enumerate(D.seg):
        sc = base[i][None, :] + (A * D.new[a:b][None, :] - 0.0 * D.busy[a:b][None, :])
        top = np.argsort(-sc, axis=1, kind="stable")[:, :K]                       # (grid, K) positions within the referral
        nw = D.new[a:b][top] > 0
        n[i] = nw.sum(1)
        if known[i]: den[i] = nw.sum(1); num[i] = (nw & ins[a:b][top]).sum(1)
    return n, num, den


# ---- bootstrap ---------------------------------------------------------------------------------------------------------------
class Boot:
    """Shared draws: W (draws × PCPs) PCP multiplicities, S (draws × 3) seed-ID indices (into SEEDS)."""

    def __init__(self, pcp_codes, n_pcp, draws=DRAWS):
        idx = np.random.default_rng(RNG_PCP).integers(0, n_pcp, size=(draws, n_pcp))
        self.W = np.stack([np.bincount(r, minlength=n_pcp) for r in idx]).astype(float)
        self.S = np.random.default_rng(RNG_SEED).integers(0, len(SEEDS), size=(draws, len(SEEDS)))
        self.codes, self.n_pcp, self.draws = pcp_codes, n_pcp, draws

    def per_pcp(self, x, mask=None):
        w = np.asarray(x, float) if mask is None else np.where(mask, x, 0.0)
        return np.bincount(self.codes, weights=w, minlength=self.n_pcp)

    def mean_draws(self, A, N):
        """A: (3 seeds, PCPs) sums, N: (PCPs,) counts → (two-level draws, PCP-only draws, point, per-seed points)."""
        num = self.W @ A.T                                                        # (draws, 3)
        den = self.W @ N
        two = np.take_along_axis(num, self.S, axis=1).mean(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (np.where(den > 0, two / np.where(den > 0, den, 1), np.nan), np.where(den > 0, (num.mean(1)) / np.where(den > 0, den, 1), np.nan),
                    A.sum(1).mean() / N.sum() if N.sum() > 0 else np.nan, A.sum(1) / N.sum() if N.sum() > 0 else np.full(len(A), np.nan))


def grid_tie_order(grid):
    """Grid indices in tie-break order: smaller |a|, then smaller a."""
    return np.array(sorted(range(len(grid)), key=lambda g: (abs(grid[g]), grid[g])))


def match(newness, grid, target, tol, frozen=None):
    """newness (..., len(grid)) → (chosen grid index, attained). frozen: fixed grid index (2019)."""
    if frozen is not None:
        ch = np.broadcast_to(np.asarray(frozen), newness.shape[:-1]).copy()
    else:
        order = grid_tie_order(grid); d = np.abs(newness[..., order] - target)
        ch = order[np.argmin(np.where(np.isnan(d), np.inf, d), axis=-1)]
    got = np.take_along_axis(newness, ch[..., None], axis=-1)[..., 0]
    return ch, np.abs(got - target) <= tol + 1e-12


def holm(p):
    """Holm-adjusted p-values (same order as given)."""
    p = np.asarray(p, float); m = len(p); order = np.argsort(p, kind="stable"); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def summarise(two, undefined, alpha=ALPHA, delta=DELTA):
    """Component statistics from two-level draws (two: values or NaN; undefined: draws counted against the claim)."""
    ok = ~np.isnan(two); bad = undefined | ~ok
    p = (1 + int(((two <= 0) & ok & ~undefined).sum() + bad.sum())) / (1 + len(two))
    p_opp = (1 + int(((two >= 0) & ok & ~undefined).sum() + bad.sum())) / (1 + len(two))      # undefined draws count against either claim
    q = lambda lo, hi: [float(np.percentile(two[ok], lo)), float(np.percentile(two[ok], hi))] if ok.any() else [np.nan, np.nan]
    ci95, ci90 = q(2.5, 97.5), q(5, 95)
    return {"p_one_sided": p, "p_opposite_descriptive": p_opp, "ci95": ci95, "ci90": ci90,
            "practically_equivalent_tost": bool(ok.any() and -delta < ci90[0] and ci90[1] < delta),
            "undefined_share": float(bad.mean()), "left_out_of_ci_share": float((~ok).mean())}


def current_code(repo):
    """SHA-256 of every library file (referral_gnn/<file>.py → key referral_gnn/<file>.py) in the repository."""
    import glob
    import hashlib
    import os
    return {f"referral_gnn/{os.path.basename(p)}": hashlib.sha256(open(p, "rb").read()).hexdigest()
            for p in sorted(glob.glob(os.path.join(repo, "referral_gnn", "*.py")))}


def amendment_records(manifests_dir):
    """Post-design amendment records (runs/analysis/step13/amendments/*.json), in the order they were recorded."""
    import glob
    import json
    import os
    return sorted((json.load(open(p)) for p in glob.glob(os.path.join(manifests_dir, "amendments", "*.json"))), key=lambda r: r["seq"])


def expected_code(design, records=()):
    """{repo path: SHA-256} the code must have now: the design manifest's record, updated by each recorded amendment."""
    exp = {f"referral_gnn/{k}": v for k, v in design["code"]["library_files_sha256"].items()}
    exp.update(design["code"]["scripts_sha256"])
    for r in records: exp.update({f: x["after"] for f, x in r["files"].items()})
    return exp


def expected_amendments_sha256(design, records=()):
    return records[-1]["amendments_sha256"] if records else design["amendments_sha256"]


def code_drift(design, repo, records=()):
    """Files whose SHA-256 differs from the design manifest's record (library and every STEP13 script), after the recorded
    post-design amendments; [] when unchanged."""
    import hashlib
    import os
    exp = expected_code(design, records); now = current_code(repo)
    out = [f for f in sorted(set(now) | {k for k in exp if k.startswith("referral_gnn/")}) if now.get(f) != exp.get(f)]
    out += [f for f, v in exp.items() if not f.startswith("referral_gnn/")
            and (not os.path.exists(os.path.join(repo, f)) or hashlib.sha256(open(os.path.join(repo, f), "rb").read()).hexdigest() != v)]
    return out


def tests_record():
    """The machine-readable test definitions frozen in the design manifest (compared again at resolve and verdict time)."""
    import json
    return json.loads(json.dumps({"units": UNITS, "h3_reference": H3_REFERENCE, "draws": DRAWS, "rng": {"pcp": RNG_PCP, "seed": RNG_SEED},
                                  "alpha_holm": ALPHA, "m": len(UNITS), "delta": DELTA, "k": K, "undefined_gate": UNDEFINED_GATE,
                                  "attain_gate": ATTAIN_GATE, "grid": MATCH_GRID, "target_newness": TARGET_NEWNESS, "tolerance": TOLERANCE,
                                  "min_slice": [MIN_SLICE_N, MIN_SLICE_PCPS], "low_history_max": LOW_HISTORY_MAX, "rare_n": RARE_N,
                                  "unit_slices": list(UNIT_SLICES), "headline_only_if_replicated": list(HEADLINE_ONLY_IF_REPLICATED)}))
