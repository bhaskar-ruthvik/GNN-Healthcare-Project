"""STEP11 §11.2: provider practice RUCA codes (source MUP columns), one value per provider NPI, for the rural-specialist split.
Reads only the NPI and RUCA columns of the raw workbook; writes prepared_data/provider_ruca_v1/ (private, aggregate per provider).
RUCA 1–3 = metropolitan, 4–10 = non-metropolitan; 0 and 99 = unknown. Each provider has one value (checked 2026-10-07).
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os

import pandas as pd

root = str(workspace()); os.chdir(root)
out = "prepared_data/provider_ruca_v1"
if os.path.exists(out): raise SystemExit(f"{out} exists")
src = "data/Final_Base_DataSet _version-09-30-2025 - Copy-for-Bhaskar.xlsx"
d = pd.read_excel(src, sheet_name="Final_Base_Data_09_30_2025", dtype="string",
                  usecols=["primary_care_provider_npi", "rendering_provider_npi", "Rndrng_Prvdr_RUCA_pcp", "Rndrng_Prvdr_RUCA"])
os.umask(0o077); os.makedirs(out)
summ = {}
for role, npi, col in (("pcp", "primary_care_provider_npi", "Rndrng_Prvdr_RUCA_pcp"), ("specialist", "rendering_provider_npi", "Rndrng_Prvdr_RUCA")):
    t = d[[npi, col]].dropna(); t[col] = pd.to_numeric(t[col], errors="coerce")
    g = t.groupby(npi)[col].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else float("nan")).rename("ruca").reset_index().rename(columns={npi: "npi"})
    g["npi"] = g.npi.str.strip()
    g["group"] = g.ruca.map(lambda v: "unknown" if pd.isna(v) or v <= 0 or v >= 99 else ("metro" if v < 4 else "nonmetro"))
    g.to_pickle(f"{out}/{role}_ruca_PRIVATE.pkl"); summ[role] = g.group.value_counts().to_dict()
json.dump({"source": os.path.basename(src), "columns": ["Rndrng_Prvdr_RUCA_pcp", "Rndrng_Prvdr_RUCA"], "rule": "1-3 metro, 4-10 nonmetro, 0/99 unknown",
           "counts": summ}, open(f"{out}/manifest.json", "w"), indent=1)
print(summ)
