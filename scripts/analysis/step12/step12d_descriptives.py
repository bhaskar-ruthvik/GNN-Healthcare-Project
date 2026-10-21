"""STEP12d descriptives for the deck (aggregates only): learned offset β per table-offset run (torch, checkpoint only) and, in a
separate process, the stacked LightGBM's gain share of the GNN score. → runs/analysis/step12d/heavy_descriptives.json
Usage: python scripts/analysis/step12/step12d_descriptives.py beta   then   python scripts/analysis/step12/step12d_descriptives.py gain"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import glob, json, os, sys
root = str(workspace()); os.chdir(root)
OUT = "runs/analysis/step12d/heavy_descriptives.json"; out = json.load(open(OUT)) if os.path.exists(OUT) else {}
if sys.argv[1] == "beta":
    import torch
    out["offset_beta"] = {}
    for r in sorted(glob.glob("runs/*_s12d_offset_s*")):
        ck = torch.load(os.path.join(r, "best.pt"), map_location="cpu")
        out["offset_beta"][os.path.basename(r)] = {"beta": float(ck["model_state"]["offset_beta"]), "best_epoch": json.load(open(os.path.join(r, "summary.json"))).get("best_epoch")}
else:
    import lightgbm as lgb
    out["stack_gnn_gain_share"] = {}
    for r in sorted(glob.glob("runs/*_s12_lgbm_tabular_lambdarank_stack_L127_M50_R1000_s*")):
        b = lgb.Booster(model_file=os.path.join(r, "model.txt")); g = b.feature_importance("gain"); n = b.feature_name(); tot = g.sum()
        rank = sorted(range(len(g)), key=lambda i: -g[i]).index(n.index("gnn_score")) + 1
        out["stack_gnn_gain_share"][os.path.basename(r)] = {"share": float(g[n.index("gnn_score")] / tot), "rank": rank, "n_features": len(n)}
json.dump(out, open(OUT, "w"), indent=1); print(json.dumps(out, indent=1))
