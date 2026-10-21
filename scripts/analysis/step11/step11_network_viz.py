"""Data for the deck's network-over-time slide (presentation/netviz_w2.json). Anonymous: node indices and layout coordinates only.

Windows: January 1 – October 29 of 2016–2019 (as 11.4(a), so 2019 is comparable). Connection = PCP→specialist pair with at least one
referral in the window. Status of a connection in year t: "continued" (present in t−1), "new" (not present in t−1); "dropped" =
present in t−1, absent in t (drawn faintly). For readability the drawing keeps the most-referred specialists and connections with
at least MIN_W referrals in the year; the status of every drawn connection is computed on the FULL connection sets.
Positions: one force-directed layout of the union graph (all four years), fixed across years so changes are visible.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
def _nv_out(): os.makedirs("deliverables/presentation", exist_ok=True); return "deliverables/presentation/netviz_w2.json"
import json
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd


from referral_gnn.three_node_train import load_three_node_inputs  # noqa: E402
from referral_gnn.data import to_day  # noqa: E402
from referral_gnn.step10 import specialty_names  # noqa: E402

root = str(workspace()); os.chdir(root)
N_SPEC, MIN_W = 600, 2
mi = load_three_node_inputs("prepared_data/three_node_v5/model_inputs_PRIVATE.pkl"); names = specialty_names(mi)
D = lambda s: int(np.asarray(to_day(np.array([s])))[0])
YEARS = [2016, 2017, 2018, 2019]
ev = pd.DataFrame({"day": mi.enc_day, "pcp": mi.enc_pcp, "spec": mi.enc_target, "sp": mi.enc_target_specialty_asof})
ev["year"] = np.nan
for y in YEARS + [2015]: ev.loc[(ev.day >= D(f"{y}-01-01")) & (ev.day < D(f"{y}-10-30")), "year"] = y
ev = ev.dropna(subset=["year"]); ev["year"] = ev.year.astype(int)
full = {y: ev[ev.year == y].groupby(["pcp", "spec"]).size() for y in [2015] + YEARS}
# busiest 10% within specialty, per year (by referrals received in the window)
busy = {}
for y in YEARS:
    c = ev[ev.year == y].groupby(["sp", "spec"]).size().reset_index(name="n"); b = set()
    for _, g in c.groupby("sp"):
        k = max(1, int(np.ceil(0.1 * len(g)))); b |= set(g.nlargest(k, "n").spec.tolist())
    busy[y] = b
# drawn nodes: all PCPs active in 2016–2019; the N_SPEC specialists with most referrals over 2016–2019
tot = ev[ev.year.isin(YEARS)].groupby("spec").size().sort_values(ascending=False)
specs = tot.index[:N_SPEC].tolist(); pcps = sorted(ev[ev.year.isin(YEARS)].pcp.unique().tolist())
S, P = set(specs), set(pcps)
# layout on the union graph of drawn connections
G = nx.Graph()
for p in pcps: G.add_node(("p", p))
for s in specs: G.add_node(("s", s))
for y in YEARS:
    for (p, s), w in full[y].items():
        if s in S and p in P and w >= MIN_W: G.add_edge(("p", p), ("s", s), weight=G.get_edge_data(("p", p), ("s", s), {"weight": 0})["weight"] + w)
G.remove_nodes_from([n for n, d in G.degree() if d == 0])                      # nodes with no drawn connection in any year
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()                # main component (small islands distort the scale)
pcps = [p for p in pcps if ("p", p) in G]; specs = [s for s in specs if ("s", s) in G]; S, P = set(specs), set(pcps)
pos = nx.spring_layout(G, k=2.2 / np.sqrt(G.number_of_nodes()), iterations=300, seed=20261009, weight=None)
xy = np.array(list(pos.values())); lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
norm = {n: np.clip((v - lo) / (hi - lo), 0, 1).round(4).tolist() for n, v in pos.items()}
nodes = [{"t": "p", "x": norm[("p", p)][0], "y": norm[("p", p)][1]} for p in pcps] + [{"t": "s", "x": norm[("s", s)][0], "y": norm[("s", s)][1]} for s in specs]
nid = {("p", p): i for i, p in enumerate(pcps)}; nid.update({("s", s): len(pcps) + i for i, s in enumerate(specs)})
years = {}
for y in YEARS:
    prev = set(full[y - 1].index); cur = full[y]
    drawn = [(p, s, int(w)) for (p, s), w in cur.items() if s in S and p in P and w >= MIN_W]
    edges = [[nid[("p", p)], nid[("s", s)], w, 1 if (p, s) in prev else 0] for p, s, w in drawn]     # 1 = continued, 0 = new
    dropped = [[nid[("p", p)], nid[("s", s)]] for (p, s), w in full[y - 1].items() if s in S and p in P and w >= MIN_W and (p, s) not in set(cur.index)]
    recv = ev[ev.year == y].groupby("spec").size(); sent = ev[ev.year == y].groupby("pcp").size()
    size = [int(sent.get(p, 0)) for p in pcps] + [int(recv.get(s, 0)) for s in specs]
    years[str(y)] = {"edges": edges, "dropped": dropped, "size": size, "busy": [i for i, s in enumerate(specs) if s in busy[y]],
                     "all_connections": int(len(cur)), "all_new_share": float(np.mean([k not in prev for k in cur.index])),
                     "drawn_connections": len(edges), "drawn_new_share": float(np.mean([e[3] == 0 for e in edges])) if edges else 0.0}
    print(y, {k: v for k, v in years[str(y)].items() if k in ("all_connections", "all_new_share", "drawn_connections", "drawn_new_share")}, "dropped drawn", len(dropped))
# ---- readable sample: 5 PCPs (seeded draw among PCPs active every year with 60–600 referrals per window); each PCP's top-10
# specialists per year (union across years). Status vs the previous year uses ANY referral by that PCP (full data).
per = ev[ev.year.isin(YEARS)].groupby(["pcp", "year"]).size().unstack(fill_value=0)
mod = per[(per >= 60).all(axis=1) & (per <= 600).all(axis=1)]
pick = sorted(np.random.default_rng(20261009).choice(mod.index.to_numpy(), min(5, len(mod)), replace=False).tolist())
w_all = ev[ev.pcp.isin(pick)].groupby(["pcp", "year", "spec"]).size().reset_index(name="n")
top = w_all[w_all.year.isin(YEARS)].sort_values("n", ascending=False).groupby(["pcp", "year"]).head(10)
sspecs = sorted(top.spec.unique().tolist())
H = nx.Graph()
for p_ in pick: H.add_node(("p", p_))
for _, r in top.iterrows(): H.add_edge(("p", int(r.pcp)), ("s", int(r.spec)))
# deterministic layout: PCPs on a pentagon; each specialist on a ring around its PCP (shared specialists between their PCPs)
cen = {p_: np.array([0.5 + 0.34 * np.cos(2 * np.pi * i / len(pick) - np.pi / 2), 0.5 + 0.34 * np.sin(2 * np.pi * i / len(pick) - np.pi / 2)]) for i, p_ in enumerate(pick)}
owners = top.groupby("spec").pcp.agg(lambda x: sorted(set(x)))
hn = {("p", p_): cen[p_].round(4).tolist() for p_ in pick}
for p_ in pick:
    mine = [s_ for s_ in sspecs if owners[s_] == [p_]]
    out_dir = (cen[p_] - 0.5) / max(np.linalg.norm(cen[p_] - 0.5), 1e-9)
    for j, s_ in enumerate(mine):
        ang = 2 * np.pi * j / max(len(mine), 1); rad = 0.13 + 0.03 * (j % 2)
        hn[("s", s_)] = (cen[p_] + rad * np.array([np.cos(ang), np.sin(ang)]) + 0.02 * out_dir).round(4).tolist()
for s_ in sspecs:
    if len(owners[s_]) > 1:
        m = np.mean([cen[q] for q in owners[s_]], axis=0); hn[("s", s_)] = (0.5 + 0.8 * (m - 0.5)).round(4).tolist()
s_nodes = [{"t": "p", "x": hn[("p", p_)][0], "y": hn[("p", p_)][1]} for p_ in pick] + [{"t": "s", "x": hn[("s", s_)][0], "y": hn[("s", s_)][1]} for s_ in sspecs]
sid = {("p", p_): i for i, p_ in enumerate(pick)}; sid.update({("s", s_): len(pick) + i for i, s_ in enumerate(sspecs)})
s_years = {}
for y in YEARS:
    t = top[top.year == y]; prevany = set(zip(w_all[w_all.year == y - 1].pcp, w_all[w_all.year == y - 1].spec))
    curany = set(zip(w_all[w_all.year == y].pcp, w_all[w_all.year == y].spec)); tprev = top[top.year == y - 1]
    edges = [[sid[("p", int(p_))], sid[("s", int(s_))], int(n), 1 if (p_, s_) in prevany else 0] for p_, s_, n in zip(t.pcp, t.spec, t.n)]
    dropped = [[sid[("p", int(p_))], sid[("s", int(s_))]] for p_, s_ in zip(tprev.pcp, tprev.spec) if (p_, s_) not in curany]
    recv = ev[ev.year == y].groupby("spec").size(); sent = ev[ev.year == y].groupby("pcp").size()
    s_years[str(y)] = {"edges": edges, "dropped": dropped, "size": [int(sent.get(p_, 0)) for p_ in pick] + [int(recv.get(s_, 0)) for s_ in sspecs],
                       "busy": [i for i, s_ in enumerate(sspecs) if s_ in busy[y]], "drawn_connections": len(edges),
                       "drawn_new_share": float(np.mean([e[3] == 0 for e in edges])) if edges else 0.0}
    print("sample", y, len(edges), "edges, new share", round(s_years[str(y)]["drawn_new_share"], 2), "| dropped", len(dropped))
sample = {"n_pcp": len(pick), "n_spec": len(sspecs), "nodes": s_nodes, "years": s_years,
          "note": "5 PCPs (seeded draw among PCPs active every year with 60–600 referrals) and each one's top-10 specialists per year; new = that PCP made no referral to the specialist the previous year; dropped = in last year's top 10 and no referral this year."}
ne = json.load(open("runs/analysis/step11_4a/summary.json"))
out = {"note": f"Top {N_SPEC} specialists by referrals 2016–2019 and every active PCP; connections with ≥ {MIN_W} referrals in the year (Jan–Oct). Status on full connection sets.",
       "n_pcp": len(pcps), "n_spec": len(specs), "nodes": nodes, "years": years, "yearly": ne["yearly"], "transitions": ne["transitions"], "sample": sample}
json.dump(out, open(_nv_out(), "w"), separators=(",", ":"))
print("wrote presentation/netviz_w2.json", os.path.getsize(_nv_out()) // 1024, "KB;", len(pcps), "PCPs,", len(specs), "specialists")
