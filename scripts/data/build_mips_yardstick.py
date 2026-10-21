import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import os
"""Step 10.1 — fixed per-specialist MIPS yardstick (STEP10_PROTOCOL §2.3). Reads the raw workbook READ-ONLY; writes a NEW directory.

The prepared bundle nulls every multi-valued MIPS (persistent-ambiguity rule), so the raw values are re-read here:
  columns read: rendering_provider_npi, service_from_date, final MIPS score (nothing else).
Outputs (prepared_data/mips_yardstick_v1/, refuses to overwrite):
  mips_yardstick_PRIVATE.pkl   one row per specialist in the prepared bundle:
      mips_value      mean of distinct positive values (NaN if the specialist only ever has 0 = "no MIPS score")
      mips_min/max    min / max of distinct positive values (bounds, §2.3)
      mips_range      max − min (0 if single)
      n_positive_values, mips_none, mips_multi, mips_partial_zero
      specialty_train label on or before the last training day (else first label seen) — only for the imputation value
      mips_imputed    specialty mean of mips_value over specialists active in training (first seen ≤ last training day);
                      overall training mean if the specialty has none
  mips_values_PRIVATE.pkl      one row per (specialist, distinct value): first and last service day — the source for the v5 as-of inputs
  manifest.json, checks.json   aggregate counts only
Usage: build_mips_yardstick.py [--out prepared_data/mips_yardstick_v1]
"""
import argparse, hashlib, json, os, shutil, sys
from datetime import datetime, timezone
import numpy as np, pandas as pd

root = str(workspace())
BUNDLE = os.path.join(root, "prepared_data", "referral_prepared_20260918T040320_197341Z")
if os.environ.get("REFERRAL_ROOT"):                     # STEP11 snapshots: the workspace's own (single) bundle
    import sys
    from referral_gnn.data import default_bundle_path
    BUNDLE = default_bundle_path(root)
RAW = os.path.join(root, "data", "Base_DataSet _version-09-30-2025_Bhaskar.xlsx")
SHEET = "Final_Base_Data_09_30_2025"
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(root, "prepared_data", "mips_yardstick_v1")); args = ap.parse_args()
if os.path.exists(args.out): sys.exit(f"refusing to overwrite existing {args.out}")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

# ---- raw MIPS (read-only) -------------------------------------------------------------
raw = pd.read_excel(RAW, sheet_name=SHEET, usecols=["rendering_provider_npi", "service_from_date", "final MIPS score"], dtype="string", keep_default_na=False)
n_raw = len(raw)
raw["provider_id"] = raw["rendering_provider_npi"].str.strip()
raw["day"] = pd.to_datetime(raw["service_from_date"], format="ISO8601", errors="coerce").dt.normalize()
raw["m"] = pd.to_numeric(raw["final MIPS score"].replace("", pd.NA), errors="coerce")
bad = raw["m"].isna() | raw["day"].isna() | (raw["m"] < 0) | (raw["m"] > 100)
raw = raw.loc[~bad, ["provider_id", "day", "m"]]

# ---- specialists of the prepared bundle -------------------------------------------------
fs = pd.read_pickle(os.path.join(BUNDLE, "specialist_first_seen_PRIVATE.pkl")); fs.index = fs.index.astype(str)
spec_ids = np.array(sorted(fs.index))
sh = pd.read_pickle(os.path.join(BUNDLE, "specialist_history_PRIVATE.pkl")); sh["provider_id"] = sh["provider_id"].astype(str)
eq = pd.read_pickle(os.path.join(BUNDLE, "evaluation_queries_PRIVATE.pkl"))
train_end = pd.to_datetime(eq.loc[eq["split"].astype(str) == "train", "_service_day"]).max().normalize()   # last training day
raw = raw[raw.provider_id.isin(set(spec_ids))]

# ---- per (specialist, value) and per specialist ---------------------------------------------
vals = raw.groupby(["provider_id", "m"])["day"].agg(first_day="min", last_day="max", n_rows="size").reset_index().rename(columns={"m": "value"})
pos = vals[vals.value > 0].groupby("provider_id")["value"]
y = pd.DataFrame(index=pd.Index(spec_ids, name="provider_id"))
y["n_positive_values"] = pos.nunique().reindex(y.index).fillna(0).astype(int)
y["mips_value"] = pos.mean().reindex(y.index)                  # mean of DISTINCT positive values (vals has one row per value)
y["mips_min"] = pos.min().reindex(y.index); y["mips_max"] = pos.max().reindex(y.index)
y["mips_range"] = (y.mips_max - y.mips_min).fillna(0.0)
has_zero = vals[vals.value == 0].groupby("provider_id").size().reindex(y.index).fillna(0).gt(0)
has_any = vals.groupby("provider_id").size().reindex(y.index).fillna(0).gt(0)
y["mips_none"] = y.n_positive_values.eq(0)
y["mips_multi"] = y.n_positive_values.gt(1)
y["mips_partial_zero"] = has_zero & y.n_positive_values.gt(0)
y["no_raw_rows"] = ~has_any

# ---- imputation value (training information only) ----------------------------------------------
sh["_day"] = pd.to_datetime(sh["_service_day"]).dt.normalize()
lab = sh.dropna(subset=["specialty"]).sort_values("_day")
lab_train = lab[lab._day <= train_end].groupby("provider_id").specialty.last()
lab_first = lab.groupby("provider_id").specialty.first()
y["specialty_train"] = lab_train.reindex(y.index).fillna(lab_first.reindex(y.index)).fillna("missing").astype(str)
active_train = pd.to_datetime(fs.loc[y.index, "first_seen"]).dt.normalize() <= train_end
# STEP11 snapshots (leakage check b, 2026-10-08): the imputation uses only MIPS values first seen on/before the last training
# day (as-of), never the all-time mean. On the real data this is identical to the earlier all-time version (verified for
# the current inputs and S16: 0 difference), but the all-time version would carry post-training values into training inputs.
pos_asof = vals[(vals.value > 0) & (vals.first_day <= train_end)].groupby("provider_id")["value"].mean()
y_asof = pos_asof.reindex(y.index)
tr = y[active_train & y_asof.notna()].assign(mips_value=y_asof[active_train & y_asof.notna()])
spec_mean = tr.groupby("specialty_train").mips_value.mean(); overall = float(tr.mips_value.mean())
y["mips_imputed"] = y.specialty_train.map(spec_mean).fillna(overall)
y["mips_imputed_fallback_overall"] = ~y.specialty_train.isin(spec_mean.index)

# ---- checks (aggregate) -------------------------------------------------------------------
checks = {
    "raw_rows": int(n_raw), "raw_rows_dropped_unparseable_or_out_of_range": int(bad.sum()),
    "specialists": int(len(y)), "specialists_without_raw_rows": int(y.no_raw_rows.sum()),
    "mips_none (only zeros)": int(y.mips_none.sum()), "mips_none_share": round(float(y.mips_none.mean()), 4),
    "mips_multi": int(y.mips_multi.sum()), "mips_partial_zero": int(y.mips_partial_zero.sum()),
    "n_positive_values_distribution": {int(k): int(v) for k, v in y.n_positive_values.value_counts().sort_index().items()},
    "mips_value_quantiles": {str(q): round(float(v), 2) for q, v in y.mips_value.quantile([.05, .25, .5, .75, .95]).items()},
    "multi_range_median_p90": [round(float(y.loc[y.mips_multi, "mips_range"].median()), 1), round(float(y.loc[y.mips_multi, "mips_range"].quantile(.9)), 1)],
    "train_end_day": str(train_end.date()), "imputation_specialties": int(len(spec_mean)), "imputation_overall_mean": round(overall, 2),
    "imputed_via_overall_fallback": int((y.mips_none & y.mips_imputed_fallback_overall).sum()),
}
assert y.mips_value.dropna().between(0, 100).all() and (y.mips_min <= y.mips_value).all() and (y.mips_value <= y.mips_max).all()
assert not y.loc[~y.mips_none, "mips_value"].isna().any() and y.loc[y.mips_none, "mips_value"].isna().all()
assert (y.loc[~y.mips_multi & ~y.mips_none, "mips_range"] == 0).all() and (y.loc[y.mips_multi, "mips_range"] > 0).all()
assert y.mips_imputed.notna().all()

os.makedirs(args.out)
y.reset_index().to_pickle(os.path.join(args.out, "mips_yardstick_PRIVATE.pkl"))
vals.to_pickle(os.path.join(args.out, "mips_values_PRIVATE.pkl"))
shutil.copy(os.path.abspath(__file__), os.path.join(args.out, "build_code_PRIVATE.py"))
json.dump(checks, open(os.path.join(args.out, "checks.json"), "w"), indent=2)
man = {"built_utc": datetime.now(timezone.utc).isoformat(), "protocol": "STEP10_PROTOCOL.md §2.3", "source": os.path.basename(RAW), "source_sheet": SHEET,
       "source_sha256": sha256(RAW), "bundle": os.path.basename(BUNDLE),
       "files": {f: sha256(os.path.join(args.out, f)) for f in ("mips_yardstick_PRIVATE.pkl", "mips_values_PRIVATE.pkl", "checks.json")}}
json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
open(os.path.join(args.out, "COMPLETE.json"), "w").write(json.dumps({"complete": True}))
print(json.dumps(checks, indent=2))
