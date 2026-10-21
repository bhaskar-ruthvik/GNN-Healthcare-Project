"""Why does the tabular LightGBM recommend ~10 points more specialists new to the PCP than the graph model, and are they good?
Reads runs/analysis/step11_6_picks/validation_top5_picks_PRIVATE.csv.gz (2018 STEP10 cohort, seed 42 of each model).
Lenses only (MIPS, load, travel, fit) plus case-fit indicators; no MRR. Aggregates only.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os

import numpy as np
import pandas as pd

root = str(workspace()); os.chdir(root)
D = "runs/analysis/step11_6_picks"; p = pd.read_csv(f"{D}/validation_top5_picks_PRIVATE.csv.gz")
p["model"] = np.where(p.run.str.contains("lgbm"), "table", "graph")
p["mips_s"] = p.mips.where(~p.mips_none)
p["better_mips"] = (p.mips > p.ref_mips) & ~p.mips_none
p["good_new"] = p.new_pcp & p.better_mips & ~p.busy
def profile(d):
    return {"slots": len(d), "MIPS (scored)": d.mips_s.mean(), "MIPS above chosen": d.better_mips.mean(), "busiest 10%": d.busy.mean(), "distance mi": d.dist.median(),
            "treated this disease category before": d.cat.mean(), "seen this insurer before": d.ins.mean(), "referrals received before (median)": d.prior_refs.median(),
            "is the chosen specialist": d.is_target.mean()}
out = {}
for var in ("blend0.3", "base"):
    q = p[p.variant == var]; res = {}
    for mname, g in q.groupby("model"):
        res[mname] = {"new share of top-5": g.new_pcp.mean(), "good new (new, higher MIPS than chosen, not busiest)": g.good_new.mean(),
                      "new picks": profile(g[g.new_pcp]), "familiar picks": profile(g[~g.new_pcp])}
    # the extra new picks: table picks new to the PCP that the graph model does not recommend for the same referral,
    # vs the graph model's picks for that referral that the table does not recommend
    key = lambda g: set(zip(g.enc, g.spec))
    gset, tset = key(q[q.model == "graph"]), key(q[q.model == "table"])
    t_only_new = q[(q.model == "table") & q.new_pcp & np.array([(e, s) not in gset for e, s in zip(q.enc, q.spec)])]
    g_only = q[(q.model == "graph") & np.array([(e, s) not in tset for e, s in zip(q.enc, q.spec)])]
    g_only_same_refs = g_only[g_only.enc.isin(t_only_new.enc)]
    res["swap"] = {"table-only new picks": profile(t_only_new), "graph-only picks on the same referrals": profile(g_only_same_refs),
                   "share of graph-only picks that are familiar to the PCP": float((~g_only_same_refs.new_pcp).mean()),
                   "referrals with at least one table-only new pick": int(t_only_new.enc.nunique()), "of referrals": int(q.enc.nunique())}
    out[var] = res
json.dump(out, open(f"{D}/newconn_summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
for var, res in out.items():
    print(f"\n=================== {var}")
    for mname in ("graph", "table"):
        r = res[mname]; print(f"{mname}: new share {r['new share of top-5']:.3f}; good new {r['good new (new, higher MIPS than chosen, not busiest)']:.3f}")
        print(pd.DataFrame({"new picks": r["new picks"], "familiar picks": r["familiar picks"]}).T.to_string(float_format=lambda x: f"{x:.3f}"))
    s = res["swap"]; print(f"\nSWAP ({s['referrals with at least one table-only new pick']:,} of {s['of referrals']:,} referrals); graph-only picks familiar to PCP: {s['share of graph-only picks that are familiar to the PCP']:.3f}")
    print(pd.DataFrame({"table-only new picks": s["table-only new picks"], "graph-only picks (same referrals)": s["graph-only picks on the same referrals"]}).T.to_string(float_format=lambda x: f"{x:.3f}"))
