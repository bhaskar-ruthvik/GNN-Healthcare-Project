"""Build presentation/referral_recommender_this_week.html: template + aggregate data + descriptive charts (base64). Self-contained."""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import base64, json, os
os.chdir(str(workspace())); OUTD = "deliverables/presentation"; os.makedirs(OUTD, exist_ok=True); TPL = code_file("scripts/deliverables/presentation/templates")
t = open(TPL / "template.html").read(); d = open(f"{OUTD}/data.json").read()
imgs = {k: "data:image/png;base64," + base64.b64encode(open(f"runs/analysis/step10_1/{f}", "rb").read()).decode()
        for k, f in [("01", "01_mips_distribution.png"), ("03", "03_referred_vs_available.png"), ("05", "05_load_concentration.png"), ("06", "06_mips_by_load.png"), ("07", "07_distance.png")]}
out = t.replace("/*DATA*/null", d).replace("/*IMAGES*/null", json.dumps(imgs))
open(f"{OUTD}/referral_recommender_this_week.html", "w").write(out); print("wrote", len(out) // 1024, "KB")
