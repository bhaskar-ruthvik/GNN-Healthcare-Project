"""Step 10.6 per-input SHAP (STEP10_PROTOCOL A9). Resumable per referral; stoppable between referrals.

Explains f(S) = score(top-1 | S) − score(referred | S) of the base model, for validation first-time referrals of the primary
cohort whose top-1 differs from the referred specialist. 36 input groups; inputs outside S take their "absent" encoding
(specialist inputs on both candidate rows together). Grouped Shapley values by permutation sampling (Castro et al.).

  runs/analysis/step10_6/<run>/per_query_PRIVATE.csv   one row per referral (φ per input for f, top-1 and referred), appended live
  runs/analysis/step10_6/<run>/config.json             sample, groups, permutations; a restart must match it
  --summarize                                          aggregate shares + CIs and charts (from whatever is finished)
Stops before the next referral if runs/STOP_AFTER_CURRENT exists (file removed) or on battery (unless ALLOW_BATTERY=1).
Usage: shap_inputs.py runs/<run> [--n 60] [--perms 12] [--seed 0] | shap_inputs.py runs/<run> --summarize
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd, torch

from referral_gnn.three_node_train import load_three_node_inputs, load_three_node_model  # noqa: E402
from referral_gnn.train import pick_device  # noqa: E402
from referral_gnn.data import SPEC_NUMERIC, PCP_NUMERIC  # noqa: E402
from referral_gnn.step10 import Step10Cohort, load_yardstick  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--n", type=int, default=60); ap.add_argument("--perms", type=int, default=12)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--summarize", action="store_true"); ap.add_argument("--bootstrap", type=int, default=1000)
args = ap.parse_args()
root = str(workspace()); os.chdir(root)
run = os.path.basename(args.run_dir.rstrip("/")); OUT = os.path.join("runs/analysis/step10_6", run); os.makedirs(OUT, exist_ok=True)
PQ, CFG = os.path.join(OUT, "per_query_PRIVATE.csv"), os.path.join(OUT, "config.json")
STOP = "runs/STOP_AFTER_CURRENT"

# ---- groups (A9) -------------------------------------------------------------------------------------------------------
NICE = {"practice_size": "practice size", "years_in_medicare": "years in Medicare", "beneficiary_count": "Medicare patients (count)",
        "service_count": "services billed (count)", "mips_score": "MIPS", "rating": "online rating", "rating_count": "number of ratings",
        "average_panel_age": "patient panel: average age", "average_panel_risk": "patient panel: average risk score",
        "panel_age_lt65": "patient panel: under 65", "panel_age_65_74": "patient panel: 65–74", "panel_age_75_84": "patient panel: 75–84",
        "panel_age_gt84": "patient panel: 85+", "panel_female": "patient panel: female", "panel_male": "patient panel: male"}
GROUPS = [("specialist", f"specialist: {NICE[c]}") for c in SPEC_NUMERIC] + [("specialist", "specialist: specialty"), ("specialist", "specialist: credentials")] + \
         [("pcp", f"PCP: {NICE[c]}") for c in PCP_NUMERIC] + [("pcp", "PCP: credentials")] + \
         [("encounter", g) for g in ("patient: age", "patient: gender", "patient: insurer", "patient: ACO status", "encounter: severity", "patient: race",
                                       "patient: visit history", "encounter: diagnosis categories", "encounter: time and gap")] + \
         [("pair", g) for g in ("pair: PCP–specialist referral history", "pair: patient–specialist prior visits", "pair: PCP–specialist distance",
                                  "pair: patient–specialist distance", "pair: insurer history", "pair: shared hospital affiliation")]
NAMES = [g for _, g in GROUPS]; KIND = dict((g, k) for k, g in GROUPS); G = len(NAMES)

if args.summarize:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    df = pd.read_csv(PQ); n = len(df); cfgd = json.load(open(CFG))
    pats = df.patient.to_numpy(); u, idx = np.unique(pats, return_inverse=True); rng = np.random.default_rng(1)
    B = np.array([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(args.bootstrap)])
    rows = []
    for which in ("diff", "top1", "ref"):
        A = np.abs(df[[f"{which}|{g}" for g in NAMES]].to_numpy()); tot = A.sum(1, keepdims=True); S = A / np.maximum(tot, 1e-12)
        sums = np.stack([np.bincount(idx, weights=S[:, j], minlength=len(u)) for j in range(G)], 1); cnt = np.bincount(idx, minlength=len(u))
        bs = (B @ sums) / (B @ cnt)[:, None]
        for j, g in enumerate(NAMES):
            rows.append({"explained": which, "input": g, "kind": KIND[g], "share_mean_abs": float(S[:, j].mean()), "lo": float(np.percentile(bs[:, j], 2.5)),
                         "hi": float(np.percentile(bs[:, j], 97.5)), "mean_signed": float(df[f"{which}|{g}"].mean())})
    sm = pd.DataFrame(rows); sm.to_csv(os.path.join(OUT, "summary_shares.csv"), index=False)
    json.dump({"n_referrals": n, "additivity_max_abs_error": float(df.additivity_err.max()), **{k: cfgd[k] for k in ("perms", "seed", "n_target", "top1_equals_referred_share")}},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"; COLK = {"specialist": "#2a78d6", "pair": "#eb6834", "encounter": "#1baf7a", "pcp": "#eda100"}
    fam = "Helvetica Neue" if any("Helvetica Neue" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": fam, "font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "figure.facecolor": "white",
                         "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left"})
    d = sm[sm.explained == "diff"].sort_values("share_mean_abs")
    fig, ax = plt.subplots(figsize=(8.5, 0.27 * G + 1.4)); yy = np.arange(G)
    ax.barh(yy, d.share_mean_abs, color=[COLK[k] for k in d.kind], height=0.72, edgecolor="white", linewidth=1.5)
    ax.hlines(yy, d.lo, d.hi, color=INK2, linewidth=1)
    ax.set_yticks(yy); ax.set_yticklabels(d.input, fontsize=8.5); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.set_xlabel("Share of the explanation (mean |SHAP|, 95% patient-cluster CI)")
    ax.set_title(f"What drives the model's top pick over the referred specialist (n = {n})")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c_, label=k) for k, c_ in COLK.items()], loc="lower right", frameon=False)
    fig.text(0.01, -0.01, "Final model (v5, seed 42), base score. Inputs outside a coalition take their 'absent' encoding; correlated inputs (volume, practice size, panel counts) share credit unstably.",
             fontsize=8, color=INK2, va="top")
    fig.savefig(os.path.join(OUT, "shares_diff.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    # distribution of signed φ for the 15 largest inputs
    top = d.sort_values("share_mean_abs", ascending=False).input.head(15).tolist()[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(top) + 1.4)); r2 = np.random.default_rng(0)
    for i, g in enumerate(top):
        v = df[f"diff|{g}"].to_numpy(); ax.scatter(v, i + r2.uniform(-0.25, 0.25, len(v)), s=10, color=COLK[KIND[g]], alpha=0.7, linewidths=0)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.5)
    ax.axvline(0, color=INK2, linewidth=1); ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=8.5); ax.grid(axis="y", visible=False)
    ax.set_xlabel("SHAP value per referral (> 0 pushes the model's top pick above the referred specialist; line = median)")
    ax.set_title("How each input pushes, referral by referral (15 largest)")
    fig.savefig(os.path.join(OUT, "distribution_diff.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    print(sm[sm.explained == "diff"].sort_values("share_mean_abs", ascending=False).round(3).head(20).to_string(index=False)); sys.exit(0)

# ---- setup -----------------------------------------------------------------------------------------------------------------
man = json.load(open(os.path.join(args.run_dir, "run_manifest.json"))); ip = man["inputs_path"]
if not os.path.exists(ip): ip = os.path.join(root, ip[ip.index("prepared_data/"):])
mi = load_three_node_inputs(ip); assert mi.schema_version == "three-node-v5"; device = pick_device("auto")
model = load_three_node_model(args.run_dir, mi, device); model.eval(); bucket = man["train_config"]["graph_bucket"]
y = load_yardstick(mi)
NS = len(SPEC_NUMERIC); V, MISS, AMB = mi.mips_cols; PB = mi.pair_feature_blocks; EB = mi.encounter_blocks

def cutoff_of(q):
    return [c for c, be, qb in mi.buckets("validation", bucket, query_subset=np.array([q]))][0]

def scores_with(bt, q, cands, present):
    """present: bool array over groups. Applies absent encodings, encodes, scores the candidates."""
    x = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bt["x"].items()}
    sn, sc_ = x["specialist_num"], x["specialist_cat"]; ct = torch.as_tensor(cands, device=sn.device)
    for j, c in enumerate(SPEC_NUMERIC):
        if present[j]: continue
        if c == "mips_score":
            sn[ct, V] = torch.as_tensor(mi.mips_imputed_scaled[cands], device=sn.device, dtype=sn.dtype); sn[ct, MISS] = 1.0; sn[ct, AMB] = 0.0; sn[ct, -2:] = 0.0
        else:
            sn[ct, 3 * j] = 0.0; sn[ct, 3 * j + 1] = 1.0; sn[ct, 3 * j + 2] = 0.0
    if not present[NS]: sc_[ct, 0] = 0; sn[ct, 3 * NS] = 0.0           # specialty
    if not present[NS + 1]: sc_[ct, 1] = 0; sn[ct, 3 * NS + 1] = 0.0   # credentials
    o = NS + 2; p = int(mi.enc_pcp[q]); pn = x["pcp_num"]
    for j in range(len(PCP_NUMERIC)):
        if not present[o + j]: pn[p, 3 * j] = 0.0; pn[p, 3 * j + 1] = 1.0; pn[p, 3 * j + 2] = 0.0
    if not present[o + 3]: x["pcp_cat"][p, 0] = 0; pn[p, 3 * len(PCP_NUMERIC)] = 0.0
    o += 4; en, ec = x["encounter_num"], x["encounter_cat"]
    if not present[o]: en[q, 0] = 0.0; en[q, 1] = 1.0; en[q, 2] = 0.0                       # age
    if not present[o + 1]: ec[q, 0] = 0; en[q, 3] = 0.0                                      # gender
    if not present[o + 2]: ec[q, 1] = 0; en[q, 4] = 0.0                                      # patient insurer
    if not present[o + 3]: ec[q, 2] = 0; en[q, 5] = 0.0                                      # ACO status
    if not present[o + 4]: ec[q, 3] = 0; a, b = EB["severity_flags"]; en[q, a:b] = 0.0       # severity
    if not present[o + 5]: ec[q, 4] = 0                                                      # race
    if not present[o + 6]: a, b = EB["patient_history"]; en[q, a:b] = 0.0                    # visit history
    if not present[o + 7]:                                                                   # diagnosis categories
        x["encounter_dx_mask"][q] = False; x["encounter_dx"][q] = 0; a, b = EB["diagnosis_flags"]; en[q, a:b] = 0.0; en[q, a] = 1.0
    if not present[o + 8]: en[q, EB["time"][0]:EB["predecessor_gap"][1]] = 0.0              # time and gap
    o += 9
    pf = mi.pair_features(np.full(len(cands), q), np.asarray(cands))
    cols_of = [PB["referral_history"], PB["patient_specialist"], PB["distance"], PB["patient_distance"], PB["insurer"], PB["affiliation"]]
    for j, cols in enumerate(cols_of):
        if present[o + j]: continue
        pf[:, cols] = 0.0
        for c in cols:
            if c in (1, 3, 5, 8, 10, 12): pf[:, c + 1] = 1.0
    bt2 = dict(bt); bt2["x"] = x
    with torch.no_grad():
        h = model.encode(bt2); c_t = torch.as_tensor(cands, device=device)
        return model.score(h, torch.full_like(c_t, q), c_t, torch.as_tensor(pf, device=device)).float().cpu().numpy()

# ---- sample (fixed; stored in config) ------------------------------------------------------------------------------------
if os.path.exists(CFG):
    cfgd = json.load(open(CFG))
    assert cfgd["groups"] == NAMES and cfgd["perms"] == args.perms and cfgd["seed"] == args.seed and cfgd["n_target"] == args.n, "config differs from the saved run; refusing to resume"
    sample = [tuple(s) for s in cfgd["sample"]]
else:
    R = list(Step10Cohort(mi, "validation", y).referrals()); rng = np.random.default_rng(args.seed); order = rng.permutation(len(R))
    sample, n_eq, n_seen = [], 0, 0
    for i in order:
        r = R[i]; bt = mi.graph_batch(cutoff_of(r.enc), np.array([r.enc]), device=device)
        s = scores_with(bt, r.enc, r.eligible, np.ones(G, bool)); top1 = int(r.eligible[np.argmax(s)]); n_seen += 1
        if top1 == r.target: n_eq += 1; continue
        sample.append((int(r.enc), int(r.patient), top1, int(r.target)))
        if len(sample) == args.n: break
    cfgd = {"run": run, "groups": NAMES, "perms": args.perms, "seed": args.seed, "n_target": args.n, "sample": sample,
            "top1_equals_referred_share": n_eq / n_seen, "screened": n_seen, "protocol": "STEP10_PROTOCOL A9"}
    json.dump(cfgd, open(CFG, "w"), indent=1)
done = set(pd.read_csv(PQ).enc) if os.path.exists(PQ) else set()
print(f"{run}: {len(sample)} referrals in sample, {len(done)} already done; top-1 = referred in {cfgd['top1_equals_referred_share']:.0%} of screened", flush=True)

# ---- main loop (one referral at a time; appended immediately) ------------------------------------------------------------
def on_battery():
    try: return "Battery Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    except Exception: return False
rng_all = np.random.default_rng(args.seed + 1000)
perm_all = {enc: rng_all.permuted(np.tile(np.arange(G), (args.perms, 1)), axis=1) for enc, *_ in sample}   # fixed per referral
for k, (q, pat, top1, ref) in enumerate(sample):
    if q in done: continue
    if os.path.exists(STOP): os.remove(STOP); print("stop requested; exiting before the next referral", flush=True); break
    if os.environ.get("ALLOW_BATTERY") != "1" and on_battery(): print("on battery; exiting before the next referral", flush=True); break
    t0 = time.time(); bt = mi.graph_batch(cutoff_of(q), np.array([q]), device=device); cands = np.array([top1, ref])
    f_all = scores_with(bt, q, cands, np.ones(G, bool)); f_none = scores_with(bt, q, cands, np.zeros(G, bool))
    phi = np.zeros((G, 2))
    for perm in perm_all[q]:
        present = np.zeros(G, bool); prev = f_none
        for gi in perm:
            present[gi] = True; cur = scores_with(bt, q, cands, present); phi[gi] += cur - prev; prev = cur
    phi /= args.perms
    diff = phi[:, 0] - phi[:, 1]; err = abs(diff.sum() - ((f_all[0] - f_all[1]) - (f_none[0] - f_none[1])))
    row = {"enc": q, "patient": pat, "f_all_top1": f_all[0], "f_all_ref": f_all[1], "f_none_top1": f_none[0], "f_none_ref": f_none[1], "additivity_err": err,
           **{f"diff|{g}": diff[j] for j, g in enumerate(NAMES)}, **{f"top1|{g}": phi[j, 0] for j, g in enumerate(NAMES)}, **{f"ref|{g}": phi[j, 1] for j, g in enumerate(NAMES)}}
    pd.DataFrame([row]).to_csv(PQ, mode="a", header=not os.path.exists(PQ), index=False)
    done.add(q); print(f"  referral {len(done)}/{len(sample)} done in {time.time() - t0:.0f}s (additivity error {err:.1e})", flush=True)
print(f"finished {len(done)}/{len(sample)}", flush=True)
