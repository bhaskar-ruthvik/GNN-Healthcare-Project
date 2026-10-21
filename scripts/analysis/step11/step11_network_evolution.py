"""STEP11 §11.4(a): descriptive evolution of the PCP–specialist referral network, year by year (no model).

Rules (frozen plan): years 2016–2019 (2015 as context); January–October windows for every year, so 2019 (data end
2019-10-29) is comparable; persistence and new-connection shares on a STABLE PANEL of PCPs active in both years.
All referral events are used (the network), with the first-time share reported. Aggregates only.
Output: runs/analysis/step11_4a/{yearly.csv, transitions.csv, summary.json}
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
import os
import sys

import numpy as np
import pandas as pd


from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.data import to_day  # noqa: E402
from referral_gnn.step10 import load_yardstick  # noqa: E402

root = str(workspace()); os.chdir(root)
OUT = "runs/analysis/step11_4a"; os.makedirs(OUT, exist_ok=True)
mi = load_three_node_inputs("prepared_data/three_node_v5/model_inputs_PRIVATE.pkl"); y = load_yardstick(mi)
D = lambda s: int(np.asarray(to_day(np.array([s])))[0])
YEARS = [2015, 2016, 2017, 2018, 2019]
win = {yr: (D(f"{yr}-01-01"), D(f"{yr}-10-30")) for yr in YEARS}          # Jan 1 – Oct 29 inclusive
ev = pd.DataFrame({"day": mi.enc_day, "pcp": mi.enc_pcp, "spec": mi.enc_target, "patient": mi.enc_patient,
                   "sp": mi.enc_target_specialty_asof})
first_time = mi.ps_asof.lookup(mi.enc_patient * mi.n_specialist + mi.enc_target, mi.enc_day) < 0
ev["first_time"] = first_time
ev["year"] = np.nan
for yr, (a, b) in win.items(): ev.loc[(ev.day >= a) & (ev.day < b), "year"] = yr
mips = y.point(np.arange(mi.n_specialist)); none = y.none

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())) if n > 1 and x.sum() > 0 else np.nan

rows = []
for yr in YEARS:
    e = ev[ev.year == yr]
    if not len(e): continue
    edges = e.groupby(["pcp", "spec"]).size()
    net = edges.groupby(level=0).size()
    # concentration within specialty (specialists active this window), median over specialties with >= 10 specialists
    by = e.groupby(["sp", "spec"]).size().reset_index(name="n"); g, top = [], []
    for sp, grp in by.groupby("sp"):
        if len(grp) < 10: continue
        g.append(gini(grp.n)); k = max(1, int(np.ceil(0.1 * len(grp)))); top.append(np.sort(grp.n.to_numpy())[::-1][:k].sum() / grp.n.sum())
    rows.append({"year": yr, "referrals": len(e), "first_time_share": float(e.first_time.mean()), "active_pcps": e.pcp.nunique(),
                 "active_specialists": e.spec.nunique(), "pcp_specialist_edges": len(edges), "median_pcp_network": float(net.median()),
                 "p90_pcp_network": float(net.quantile(0.9)), "median_within_specialty_gini": float(np.median(g)) if g else np.nan,
                 "median_busiest10_share": float(np.median(top)) if top else np.nan, "n_specialties_in_concentration": len(g)})
yearly = pd.DataFrame(rows)

trans = []
for t, t1 in zip(YEARS[:-1], YEARS[1:]):
    a, b = ev[ev.year == t], ev[ev.year == t1]
    if not len(a) or not len(b): continue
    panel = sorted(set(a.pcp) & set(b.pcp))
    keep, new_share, new_ever_share = [], [], []
    hist_before_t1 = ev[ev.day < win[t1][0]]                                  # everything strictly before the t+1 window
    ever = set(zip(hist_before_t1.pcp, hist_before_t1.spec))
    for p in panel:
        sa, sb = set(a.spec[a.pcp == p]), b.spec[b.pcp == p]
        keep.append(len(sa & set(sb)) / len(sa))
        new_share.append(float(np.mean([s not in sa for s in sb])))
        new_ever_share.append(float(np.mean([(p, s) not in ever for s in sb])))
    ea = set(zip(a.pcp, a.spec)); eb = set(zip(b.pcp, b.spec))
    panel_set = set(panel); ea_p = {x for x in ea if x[0] in panel_set}; eb_p = {x for x in eb if x[0] in panel_set}
    retained = ea_p & eb_p; new_edges = eb_p - ea_p
    m_ret = [mips[s] for _, s in retained if not none[s]]; m_new = [mips[s] for _, s in new_edges if not none[s]]
    spec_a, spec_b = set(a.spec), set(b.spec); ever_spec_before = set(hist_before_t1.spec)
    trans.append({"from": t, "to": t1, "stable_panel_pcps": len(panel),
                  "median_pcp_specialists_kept": float(np.median(keep)),
                  "median_share_t1_referrals_to_specialists_not_used_in_t": float(np.median(new_share)),
                  "median_share_t1_referrals_to_specialists_never_used_by_pcp": float(np.median(new_ever_share)),
                  "panel_edge_jaccard": len(ea_p & eb_p) / len(ea_p | eb_p),
                  "specialists_entering": len(spec_b - ever_spec_before), "specialists_not_seen_again_in_t1": len(spec_a - spec_b),
                  "mips_retained_edges_mean": float(np.mean(m_ret)), "mips_new_edges_mean": float(np.mean(m_new)),
                  "n_retained_edges": len(retained), "n_new_edges": len(new_edges)})
transitions = pd.DataFrame(trans)

# community stability on the stable panel: Louvain on the PCP-specialist bipartite graph, ARI of specialists' communities
try:
    import networkx as nx
    from sklearn.metrics import adjusted_rand_score
    ari = []
    for t, t1 in zip(YEARS[:-1], YEARS[1:]):
        a, b = ev[ev.year == t], ev[ev.year == t1]
        if not len(a) or not len(b): continue
        panel = set(a.pcp) & set(b.pcp); labs = []
        for e in (a[a.pcp.isin(panel)], b[b.pcp.isin(panel)]):
            G = nx.Graph()
            for (p, s), w in e.groupby(["pcp", "spec"]).size().items(): G.add_edge(("p", p), ("s", s), weight=w)
            comm = nx.community.louvain_communities(G, weight="weight", seed=20261008)
            labs.append({n[1]: i for i, c in enumerate(comm) for n in c if n[0] == "s"})
        common = sorted(set(labs[0]) & set(labs[1]))
        ari.append({"from": t, "to": t1, "specialists_in_both": len(common),
                    "community_ARI": adjusted_rand_score([labs[0][s] for s in common], [labs[1][s] for s in common])})
    transitions = transitions.merge(pd.DataFrame(ari), on=["from", "to"], how="left")
except Exception as e:  # noqa: BLE001
    print("community stability skipped:", repr(e))

yearly.to_csv(f"{OUT}/yearly.csv", index=False); transitions.to_csv(f"{OUT}/transitions.csv", index=False)
json.dump({"windows": "Jan 1 – Oct 29 each year", "yearly": yearly.to_dict("records"), "transitions": transitions.to_dict("records")},
          open(f"{OUT}/summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(yearly.to_string(index=False, float_format=lambda x: f"{x:.3f}")); print(); print(transitions.T.to_string(float_format=lambda x: f"{x:.3f}"))
