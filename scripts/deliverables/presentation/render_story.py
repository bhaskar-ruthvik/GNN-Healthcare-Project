"""Build presentation/referral_graph_story.html: story template + the first deck's base style + aggregate data. Self-contained.
Run from anywhere:  python3 scripts/deliverables/presentation/render_story.py ["20 October 2026"] [output.html]
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import os
import re
import sys

os.chdir(str(workspace())); OUTD = "deliverables/presentation"; os.makedirs(OUTD, exist_ok=True); TPL = code_file("scripts/deliverables/presentation/templates")
base = re.search(r"<style>(.*?)</style>", open(TPL / "template.html").read(), re.S).group(1)       # same look as the earlier decks
t = open(TPL / "template_story.html").read(); d = open(f"{OUTD}/data_story.json").read()
date = sys.argv[1] if len(sys.argv) > 1 else "20 October 2026"
out = t.replace("/*BASE_STYLE*/", base).replace("/*DATA*/null", d).replace("/*DATE*/", date)
name = sys.argv[2] if len(sys.argv) > 2 else "referral_graph_story.html"      # later versions go to a new file (never overwrite)
open(f"{OUTD}/{name}", "w").write(out); print("wrote", name, len(out) // 1024, "KB")
