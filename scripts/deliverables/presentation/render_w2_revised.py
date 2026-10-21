"""Build presentation/referral_graph_week2_revised.html: week-2 template + the first deck's base style + aggregate data. Self-contained.
Run from anywhere:  python scripts/deliverables/presentation/render_w2.py ["9 October 2026"]
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import os
import re
import sys

os.chdir(str(workspace())); OUTD = "deliverables/presentation"; os.makedirs(OUTD, exist_ok=True); TPL = code_file("scripts/deliverables/presentation/templates")
base = re.search(r"<style>(.*?)</style>", open(TPL / "template.html").read(), re.S).group(1)       # same look as the first deck
t = open(TPL / "template_w2_revised.html").read(); d = open(f"{OUTD}/data_w2.json").read()
date = sys.argv[1] if len(sys.argv) > 1 else "9 October 2026"
nv = open(f"{OUTD}/netviz_w2.json").read()
out = t.replace("/*BASE_STYLE*/", base).replace("/*DATA*/null", d).replace("/*NETVIZ*/null", nv).replace("/*DATE*/", date)
open(f"{OUTD}/referral_graph_week2_revised.html", "w").write(out); print("wrote referral_graph_week2_revised.html", len(out) // 1024, "KB")
