"""Compose presentation/template_story.html: the week-3 template's style and chart helpers (already checked) + the story-deck slides.
Reused unchanged from template_w3.html: forest(), frontierChart(), renderGroups, renderTopK, renderMech, renderHyb, renderComp,
renderFrontier, renderHeavy, renderBC. New: setting, lenses, imitation, quality weight, graph vs table, 2019 + 10/15 names, summary,
appendix (network turnover, masking); Step 13 graph-value tests, post hoc matched fit, Step 14 three-way blend (23 Oct).
Run from anywhere: python3 scripts/deliverables/presentation/compose_story_template.py"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import os
import re

os.chdir(str(workspace())); OUTD = "deliverables/presentation"; os.makedirs(OUTD, exist_ok=True); TPL = code_file("scripts/deliverables/presentation/templates")
w3 = open(TPL / "template_w3.html").read()
head = w3[: w3.index('<div id="stage">')].replace("<title>Referral Graph — Week 3</title>", "<title>Referral Recommender — Story So Far</title>")
head = head.replace("</style>", """.cards4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.lensrow{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.lens2{background:#f6f6f4;border-radius:16px;padding:12px 16px;border-left:6px solid var(--c)}
.lens2 b{display:block;font-size:16px;margin-bottom:3px;color:#1b1b1f}.lens2 span{font-size:14px;color:#44464f;line-height:1.4}
.appx .tag{background:#e9e8ef!important;color:#44464f!important}
</style>""")
script = w3[w3.index("<script>"):]

def fn(name):
    """Extract a top-level JS function (and a preceding const table if it is the TK/BCM table) from the w3 script."""
    i = script.index(f"function {name}(")
    j = script.index("\nfunction ", i + 10) if "\nfunction " in script[i + 10:] else len(script)
    k = script.find("\n/* ----------", i + 10)
    j = min(x for x in (j, k) if x > 0)
    return script[i:j].rstrip() + "\n"
helpers = script[script.index("const D = /*DATA*/null;"): script.index("/* ---------- slide 2: questions ---------- */")]
helpers = helpers.replace('const state={grpView:"w2",tk:"load_slots",fr:0,hv:0,bc:"busy1"};', 'const state={grpView:"w2",tk:"load_slots",fr:0,hv:0,bc:"busy1",qw:"2019",y19:"5",net:"busy"};')
TK = script[script.index("const TK=["): script.index("function renderTopK(")]
BCM = script[script.index("const BCM=["): script.index("function renderBC(")]
reused = "".join(fn(n) for n in ("renderGroups", "renderMech", "renderHyb", "renderComp", "pareto", "frontierChart", "renderFrontier", "renderHeavy")) + TK + fn("renderTopK") + BCM + fn("renderBC")
reused = reused.replace("steering sets load directly (slides 8–10)", "steering sets load directly (slides 12–14)")
y19_old = fn("renderY19")

SLIDES = r'''
<!-- 1 -->
<section class="slide title">
  <h1 class="big" style="margin-top:110px">Recommending specialists: what a graph model adds, and how to combine it with a table</h1>
  <p class="sub" style="font-size:22px">Referral recommendations judged on quality, load, fit and travel — the story from September to October 2026</p>
  <div class="chips"><span style="background:var(--graph)">Graph</span><span style="background:var(--table)">Table</span><span style="background:var(--blend)">Blend</span></div>
  <p class="sub" style="margin-top:22px">Bhaskar Ruthvik Bikkina · /*DATE*/</p>
  <p class="note">Use ← → or the buttons below. Charts are interactive: hover for details, click the toggles. Every number is read from the result files named at the bottom of each slide; the verification workbook recomputes the headline claims. Code: <a href="https://github.com/bhaskar-ruthvik/GNN-Healthcare-Project" style="color:inherit">github.com/bhaskar-ruthvik/GNN-Healthcare-Project</a>.</p>
</section>

<!-- 2 -->
<section class="slide" data-accent="mix">
  <span class="tag">Setting</span>
  <h1>The question and the data</h1>
  <p class="sub">When a PCP refers a patient, can a recommender propose specialists that are better than the usual choice — higher quality, less congested — while still fitting the case?</p>
  <div class="cards4" id="data-cards"></div>
  <div class="row" style="margin-top:16px">
    <div class="col"><table class="t" id="funnel"></table></div>
    <div class="col" style="flex:0 0 470px;gap:10px" id="fair"></div>
  </div>
  <div class="src">Source: prepared_data/referral_prepared_*/manifest_PRIVATE.json (row counts); runs/analysis/step10_3/{validation,test}_summary.json (funnel)</div>
</section>

<!-- 3 -->
<section class="slide" data-accent="mix">
  <span class="tag">How we judge</span>
  <h1>Four lenses, not one accuracy number</h1>
  <p class="sub">The chosen specialist is a reference, not the right answer. Every model is shown on all of these, on the same referrals.</p>
  <div class="lensrow" id="lenses"></div>
  <div class="panel" style="--tint:#f2f1f6;margin-top:14px"><p class="mini" id="lens-note"></p></div>
  <div class="src">Definitions: STEP10_PROTOCOL.md §2.4; actual-choice references from runs/analysis/step11_1_tabular/lens_compare.json and step12_topk/topk_summary.json</div>
</section>

<!-- 4 -->
<section class="slide" data-accent="load">
  <span class="tag">1 · Imitation</span>
  <h1>A model that imitates PCPs reproduces their habits — and concentrates load</h1>
  <p class="sub">2018 first-time referrals; the graph model as trained (no quality weight), PCP habit, and the PCPs’ actual choices</p>
  <div class="steps" id="imit-steps"></div>
  <div class="panel" style="--tint:var(--load-t);margin-top:14px"><p class="mini" id="imit-read"></p></div>
  <div class="src">Source: runs/analysis/step10_3/validation_arms.csv; runs/analysis/step10_4/reliance/validation_arms.csv (no_pcp_history); runs/analysis/step10_1/summary.json</div>
</section>

<!-- 5 -->
<section class="slide" data-accent="q">
  <span class="tag">2 · Quality</span>
  <h1>A 30% weight on MIPS buys quality at about the PCPs’ own fit</h1>
  <p class="sub">Each point is a ranking: the model’s score mixed with the specialist’s MIPS at the weight shown. Up = higher MIPS than the chosen specialist; right = closer to what PCPs chose.</p>
  <div class="controls" id="qw-year"></div>
  <div class="row"><div class="col"><div class="chart" id="qw-chart"></div><div class="lg" id="qw-legend"></div></div>
    <div class="col" style="flex:0 0 330px;gap:10px" id="qw-side"></div></div>
  <div class="src">Source: runs/analysis/step10_3/{validation,test}_arms.csv (blend λ arms)</div>
</section>

<!-- 6 -->
<section class="slide" data-accent="table">
  <span class="tag">3 · Graph vs table</span>
  <h1>Against a strong table, the graph model fits equally well and concentrates less</h1>
  <p class="sub">Both with the 30% quality weight, same referrals and candidates. Table = LightGBM on claim attributes only (no referral history, no network). Differences with PCP-cluster 95% intervals.</p>
  <div class="row"><div class="col"><table class="t" id="gvt" style="font-size:14px"></table><div class="lg" id="gvt-legend"></div></div>
    <div class="col" style="flex:0 0 330px;gap:10px"><div class="panel" style="--tint:var(--table-t)"><p class="mini" id="gvt-read"></p></div></div></div>
  <div class="src">Source: runs/analysis/step11_1_tabular/lens_compare.json (graph model 2 seeds, table 2 seeds)</div>
</section>

<!-- 7 -->
<section class="slide" data-accent="load">
  <span class="tag">3 · Across groups</span>
  <h1>The lower load holds across patient and doctor groups</h1>
  <p class="sub">Load = share of top picks going to the busiest 10% of specialists; 2018, 30% quality weight</p>
  <div class="controls" id="grp-view"></div>
  <div class="chart" id="grp-chart"></div><div class="lg" id="grp-legend"></div>
  <p class="mini" id="grp-take" style="margin-top:8px"></p>
  <div class="src" id="grp-src"></div>
</section>

<!-- 8 -->
<section class="slide" data-accent="fit">
  <span class="tag">3 · Longer lists</span>
  <h1>With 10–15 names the two converge, except on new connections</h1>
  <p class="sub">2018, 30% quality weight; graph model and table each averaged over 2 seeds; same 5,550 referrals at every list length</p>
  <div class="controls" id="tk-metric"></div>
  <div class="row"><div class="col"><div class="chart" id="tk-chart"></div><div class="lg" id="tk-legend"></div></div>
    <div class="col" style="flex:0 0 360px;gap:10px"><div id="tk-table"></div><div class="panel" style="--tint:var(--fit-t)"><p class="mini" id="tk-read"></p></div></div></div>
  <div class="src">Source: runs/analysis/step12_topk/topk_summary.json (scripts/analysis/step12/step12_topk_summary.py)</div>
</section>

<!-- 9 -->
<section class="slide" data-accent="load">
  <span class="tag">4 · Why</span>
  <h1>Why it concentrates less: not the graph, and not PCP history</h1>
  <p class="sub">Share of top picks going to the busiest 10% of specialists. Steps 2 and 3 compare at equal fit (step 1’s two models already fit about the same), so neither side is just less accurate.</p>
  <div class="steps" id="mech-steps"></div>
  <div class="panel" style="--tint:var(--load-t);margin-top:14px"><p class="mini" id="mech-read"></p></div>
  <div class="src">Source: runs/analysis/step12a/final_summary.json, runs/analysis/step12a_decisions.json (pre-registered tests, scripts/analysis/step12/step12_congestion.py)</div>
</section>

<!-- 10 -->
<section class="slide" data-accent="table">
  <span class="tag">5 · Combining: simple hybrids</span>
  <h1>Simple combinations end up close to one of the two models</h1>
  <p class="sub">2018, 30% quality weight; averaged over 2 seed pairings. Pre-stated test: GNN’s load (top pick and top 5) and quality, plus ≥ 5 points more new connections</p>
  <table class="t" id="hyb-table" style="font-size:14px"></table><div class="lg" id="hyb-legend"></div>
  <div class="src">Source: runs/analysis/step12b/judge_summary.json (scripts/analysis/step12/step12b_judge.py; plan Implementation/STEP12B_HYBRID_PLAN.md)</div>
</section>

<!-- 11 -->
<section class="slide" data-accent="blend">
  <span class="tag">5 · Combining: why it can work</span>
  <h1>The two models disagree in useful ways</h1>
  <p class="sub">2018, 30% quality weight; three seed pairings (ranges show min–max across them)</p>
  <div class="row"><div class="col" style="gap:12px">
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px" id="comp-cards"></div>
      <table class="t" id="comp-table"></table>
      <p class="note" style="margin-top:0">“Picks only one model makes”: specialists in one model’s top 5 but not the other’s, same referrals. MIPS = the specialist’s MIPS score (0–100).</p></div>
    <div class="col" style="flex:0 0 360px;gap:10px"><div class="panel" style="--tint:var(--blend-t)"><p class="mini" id="comp-read"></p></div>
      <div class="panel" style="--tint:#f2f1f6"><p class="mini" id="steer-def"></p></div></div></div>
  <div class="src">Source: runs/analysis/step12c/complementarity.json, runs/analysis/step12c/frontier_points_pair{0,1,2}.csv</div>
</section>

<!-- 12 -->
<section class="slide" data-accent="blend">
  <span class="tag">5 · Combining: steering</span>
  <h1>Steered, the blend beats both models on load and new connections</h1>
  <p class="sub">Every steering setting that keeps fit ≥ 45% (2018); each family’s best trade-offs joined by a line. Down and right is better.</p>
  <div class="controls" id="fr-pair"></div>
  <div class="row"><div class="col"><div class="chart" id="fr-chart"></div><div class="lg" id="fr-legend"></div></div>
    <div class="col" style="flex:0 0 340px;gap:10px"><div class="panel" style="--tint:var(--blend-t)"><p class="mini" id="fr-read"></p></div></div></div>
  <div class="src">Source: runs/analysis/step12c/frontier_points_pair{0,1,2}.csv, runs/analysis/step12c/frontiers.json (scripts/analysis/step12/step12c_frontiers.py)</div>
</section>

<!-- 13 -->
<section class="slide" data-accent="blend">
  <span class="tag">5 · Combining: held-out year</span>
  <h1>It holds on 2019, and with 10 or 15 names</h1>
  <p class="sub" id="y19-sub"></p>
  <div class="controls" id="y19-view"></div>
  <div class="row"><div class="col"><table class="t" id="y19-table"></table><p class="note" style="margin-top:8px" id="y19-note"></p></div>
    <div class="col" style="flex:0 0 340px;gap:10px"><div class="panel" style="--tint:var(--blend-t)"><p class="mini" id="y19-read"></p></div><div class="panel" style="--tint:#fdf3e1"><p class="mini" id="y19-cav"></p></div></div></div>
  <div class="src" id="y19-src"></div>
</section>

<!-- 14 -->
<section class="slide" data-accent="blend">
  <span class="tag">5 · Combining: heavier hybrids</span>
  <h1>Hybrids trained together do not beat the steered blend</h1>
  <p class="sub">Same steering and fit ≥ 45% (2018). Table-offset GNN, dual encoder and stacked LightGBM, 3 seeds each</p>
  <div class="controls" id="hv-pair"></div>
  <div class="row"><div class="col"><div class="chart" id="hv-chart"></div><div class="lg" id="hv-legend"></div></div>
    <div class="col" style="flex:0 0 380px;gap:10px"><div id="hv-table"></div><div class="panel" style="--tint:var(--blend-t)"><p class="mini" id="hv-read"></p></div></div></div>
  <div class="src">Source: runs/analysis/step12d/frontier.json, frontier_points_pair{0,1,2}.csv, heavy_descriptives.json</div>
</section>

<!-- 15 -->
<section class="slide" data-accent="blend">
  <span class="tag">5 · Combining: checks</span>
  <h1>The steered blend lowers load in every group</h1>
  <p class="sub">Steered blend − graph model, by group; 2018, averaged over 3 seed pairings; PCP-cluster 95% intervals</p>
  <div class="controls" id="bc-metric"></div>
  <div class="row"><div class="col"><div class="chart" id="bc-chart"></div><div class="lg" id="bc-legend"></div></div>
    <div class="col" style="flex:0 0 410px;gap:6px"><div class="panel" style="--tint:var(--blend-t);padding:10px 14px"><p class="mini" id="bc-read"></p></div><div class="chart" id="bc-tier" style="flex:0 0 132px"></div><div class="lg" id="bc-tier-legend" style="margin-top:0"></div><div id="bc-exp"></div></div></div>
  <div class="src">Source: runs/analysis/step12c/blend_checks.json (scripts/analysis/step12/step12c_blend_checks.py)</div>
</section>

<!-- 16 -->
<section class="slide" data-accent="graph">
  <span class="tag">6 · What the graph adds: pre-registered tests</span>
  <h1>Learned message passing does not beat the same information as fixed summaries</h1>
  <p class="sub">Each row is one pre-registered comparison (fit, or insurer-compatible new picks at matched newness), averaged over 3 seed pairings, with a two-level bootstrap 95% interval (seeds and PCPs). Right of zero = the graph model is better.</p>
  <div class="controls" id="s13-year"></div>
  <div class="row"><div class="col"><div class="chart" id="s13-chart"></div><div class="lg" id="s13-legend"></div></div>
    <div class="col" style="flex:0 0 400px;gap:10px"><table class="t" id="s13-units" style="font-size:13.5px"></table><div class="panel" style="--tint:var(--graph-t)"><p class="mini" id="s13-read"></p></div></div></div>
  <div class="src" id="s13-src"></div>
</section>

<!-- 17 -->
<section class="slide" data-accent="graph">
  <span class="tag">6 · What the graph adds: information, not propagation</span>
  <h1>Graph summaries cut a table’s load at equal fit, without losing new connections</h1>
  <p class="sub">Top-pick load and newness compared at matched fit (the more accurate model is mixed with random picks down to the other’s fit); 3 seed pairings; PCP-cluster 95% intervals. Post hoc and descriptive: both years had been seen.</p>
  <div class="row"><div class="col"><table class="t" id="ph-table" style="font-size:14px"></table></div>
    <div class="col" style="flex:0 0 380px;gap:10px"><div class="panel" style="--tint:var(--graph-t)"><p class="mini" id="ph-read"></p></div><div class="panel" style="--tint:#fdf3e1"><p class="mini" id="ph-cav"></p></div></div></div>
  <div class="src" id="ph-src"></div>
</section>

<!-- 18 -->
<section class="slide" data-accent="blend">
  <span class="tag">7 · Combining, with graph information</span>
  <h1>Adding the graph-feature table to the steered blend lowers load further</h1>
  <p class="sub">Settings chosen on 2018 by the Step 12c rule (fit ≥ 45% and newness ≥ 35% in all 3 pairings, lowest top-pick load), then checked once on 2019. Top 5; 30% quality weight.</p>
  <div class="row"><div class="col"><table class="t" id="s14-table" style="font-size:14px"></table><p class="note" style="margin-top:8px" id="s14-note"></p></div>
    <div class="col" style="flex:0 0 360px;gap:10px"><div class="panel" style="--tint:var(--blend-t)"><p class="mini" id="s14-read"></p></div><div class="panel" style="--tint:#fdf3e1"><p class="mini" id="s14-cav"></p></div></div></div>
  <div class="src" id="s14-src"></div>
</section>

<!-- 19 -->
<section class="slide" data-accent="mix">
  <span class="tag">Summary</span>
  <h1>What we learned, and what comes next</h1>
  <p class="sub">The best recommender combines a neural ranker, a table and graph information; the open questions decide the next step</p>
  <div class="row"><div class="col" style="flex:0 0 560px;gap:10px" id="take"></div>
    <div class="col" style="gap:0" id="asks"></div></div>
</section>

<!-- 20 -->
<section class="slide appx" data-accent="mix">
  <span class="tag">Appendix A · Network over time</span>
  <h1>The referral network turns over fast; concentration does not</h1>
  <p class="sub">PCPs active in consecutive years; January–October of each year</p>
  <div class="row"><div class="col"><table class="t" id="net-table" style="font-size:14px"></table></div>
    <div class="col" style="flex:0 0 360px"><div class="panel" style="--tint:#f2f1f6"><p class="mini" id="net-read"></p></div></div></div>
  <div class="src">Source: runs/analysis/step11_4a/summary.json (scripts/analysis/step11/step11_network_evolution.py)</div>
</section>

<!-- 21 -->
<section class="slide appx" data-accent="mix">
  <span class="tag">Appendix B · Gender and race</span>
  <h1>Masking gender and race inside the graph model</h1>
  <p class="sub">Masked inside the model, before any embedding is built, in training and scoring. Can gender and race still be read from what the model builds? (AUC 0.5 = chance)</p>
  <div class="row"><div class="col"><table class="t" id="mask-table" style="font-size:15px"></table></div>
    <div class="col" style="flex:0 0 400px"><div class="panel" style="--tint:#f2f1f6"><p class="mini" id="mask-read"></p></div></div></div>
  <div class="src">Source: runs/analysis/step11_3a/probe_summary.json (scripts/analysis/step11/step11_probe.py)</div>
</section>
'''

NEW_JS = r'''
/* ---------- slide 2: setting ---------- */
function renderData(){
  const d=D.data, c=D.cohort;
  html("data-cards",[["Referral events",d.events],["Patients",d.patients],["PCPs",d.pcps],["Specialists",d.specialists]].map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${v.toLocaleString()}</div></div>`).join(""));
  const L={first_time:"First-time referrals (patient never saw that specialist)",unknown_specialty:"− specialty unknown before the referral",non_choice_specialty:"− facility-based (no real choice of doctor)",non_specific_specialty:"− non-specific specialty labels",referred_inactive:"− chosen specialist inactive in the prior year",referred_mips_none:"− chosen specialist has no MIPS score",eligible_lt5:"− fewer than 5 eligible specialists",patient_zip_missing:"− patient ZIP missing",primary:"Referrals evaluated"};
  html("funnel",`<tr><th>Who is evaluated</th><th>2018</th><th>2019</th></tr>`+Object.keys(c["2018"]).map(k=>`<tr${k==="primary"?' class="ref"':""}><td>${L[k]||k}</td><td>${c["2018"][k].toLocaleString()}</td><td>${(c["2019"][k]??0).toLocaleString()}</td></tr>`).join(""));
  html("fair",`<div class="panel" style="--tint:#f2f1f6"><p class="mini"><b>Candidates.</b> For each referral: specialists of the same specialty, active in the last year, within 25 miles of the patient and not already seen by the patient (filled to 5 when fewer qualify). Every model ranks the same candidates.</p>
    <p class="mini"><b>No look-ahead.</b> The model sees only what was known before the referral date: graph edges, provider attributes, pair histories and MIPS as of that day. Trained on referrals up to the end of 2017.</p>
    <p class="mini"><b>Held-out year.</b> 2018 is where every choice is made; 2019 is used once per pre-stated check, after the choice is recorded.</p>
    <p class="mini"><b>Intervals</b> resample whole PCPs (only ${D.data.pcps} PCPs, so they are honest but wide).</p></div>`);
}
/* ---------- slide 3: lenses ---------- */
function renderLenses(){
  const s=D.s11.years, pr=D.topk.practice;
  const L=[["var(--q)","Quality — MIPS gain","Average MIPS points of the recommended specialists minus the MIPS of the specialist actually chosen."],
    ["var(--load)","Load — busiest 10%",`Share of top picks going to the busiest 10% of specialists in that specialty (by past referrals). Actual choices: ${pct(s["2018"].actual_load)} (2018).`],
    ["var(--fit)","Fit — chosen in the list","Share of referrals where the specialist the PCP chose is among the 5 recommended (or 10, 15)."],
    ["var(--travel)","Travel","Distance from the patient to the recommended specialists, in miles."],
    ["#44464f","New to the PCP",`Share of the list the PCP has never referred to before: new connections. Actual choices: ${pct(s["2018"].actual_new)}.`],
    ["var(--blend)","Steering (later slides)","A bonus for specialists new to the PCP and a penalty for the busiest 10%, both known at referral time."]];
  html("lenses",L.map(([c,b,t])=>`<div class="lens2" style="--c:${c}"><b>${b}</b><span>${t}</span></div>`).join(""));
  html("lens-note",`<b>30% quality weight.</b> From slide 5 on, every model ranks by 70% its own score and 30% the specialist’s MIPS (both standardised within the referral’s candidates). This was chosen in advance on 2018 as the setting with the highest MIPS gain at a fit no lower than the PCPs’ own habit. <b>MRR</b> and similar accuracy scores were used only to train and select checkpoints, never to judge a result.`);
}
/* ---------- slide 4: imitation ---------- */
function renderImit(){
  const T=D.s10.tradeoff["2018"], R=D.s10.reliance["2018"], pr=T.practice, ph=T.pcp.base, mb=T.model.base, nh=R["no_pcp_history|base"];
  const bars=rows=>`<div class="bars">${rows.map(([l,v,c])=>`<div><div class="lb">${l}</div><div class="tr"><div class="b" style="width:${(v/0.55*78).toFixed(1)}%;background:${c}"></div><span class="v">${pct(v)}</span></div></div>`).join("")}</div>`;
  const st=[["Load","Top picks on the busiest 10%",bars([["Actual choices",pr.busy1,C.prac],["PCP’s own habit",ph.busy1,"#b9b7b1"],["Graph model, as trained",mb.busy1,C.graph]]),
      `The model sends ${p1(mb.busy1-pr.busy1)} points more top picks to the busiest specialists than PCPs actually did.`],
    ["Connections","Share of the 5 new to the PCP",bars([["Actual choices",pr.new_pcp5,C.prac],["PCP’s own habit",ph.new_pcp5,"#b9b7b1"],["Graph model, as trained",mb.new_pcp5,C.graph]]),
      `It mostly re-ranks specialists the PCP already uses: ${pct(mb.new_pcp5)} new, against ${pct(pr.new_pcp5)} in real choices.`],
    ["Why","Switch off the PCP’s referral history",bars([["Graph model, as trained: load",mb.busy1,C.graph],["Same model, history switched off: load",nh.busy1,"#7fb1ea"],["History switched off: new to PCP",nh.new_pcp5,"#7fb1ea"]]),
      `PCP history drives it: without it, load falls to ${pct(nh.busy1)} and new connections rise to ${pct(nh.new_pcp5)} (fit ${pct(mb.r5)} → ${pct(nh.r5)}).`]];
  html("imit-steps",st.map(([n,q,b,a])=>`<div class="step"><span class="sq">${n}</span><h3>${q}</h3>${b}<p class="ans">${a}</p></div>`).join(""));
  const m=D.s10.practice_mips;
  html("imit-read",`<b>And quality is left on the table.</b> Current referrals are MIPS-neutral: a randomly picked eligible specialist has higher MIPS than the one chosen ${pct(m.random_option_higher)} of the time (chosen ${m.ref_mean.toFixed(1)} vs ${m.elig_mean.toFixed(1)} on average). An imitation model inherits this. So “better than the data” needs a yardstick — the four lenses — and a way to steer.`);
}
/* ---------- slide 5: quality weight ---------- */
function renderQW(){
  toggle("qw-year","qw",[["2018","2018 (choice year)"],["2019","2019 (held out)"]]);
  const T=D.s10.tradeoff[state.qw], V=["base","blend0.1","blend0.2","blend0.3","blend0.5"], lab={"base":"0%","blend0.1":"10%","blend0.2":"20%","blend0.3":"30%","blend0.5":"50%"};
  const {svg,w,h}=frame("qw-chart");const m={l:56,r:20,t:14,b:42};
  const all=[...V.map(v=>T.model[v]),...V.map(v=>T.pcp[v])];const x=scale(0.38,0.52,m.l,w-m.r),y=scale(-2,12,h-m.b,m.t);
  [0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52].forEach(v=>{el("line",{x1:x(v),x2:x(v),y1:m.t,y2:h-m.b,stroke:C.grid},svg);txt(svg,x(v),h-m.b+15,p0(v),{"text-anchor":"middle","font-size":11})});
  ticks(-2,12,7).forEach(v=>{el("line",{x1:m.l,x2:w-m.r,y1:y(v),y2:y(v),stroke:C.grid},svg);txt(svg,m.l-6,y(v)+4,f1(v),{"text-anchor":"end","font-size":11})});
  el("line",{x1:m.l,x2:w-m.r,y1:y(0),y2:y(0),stroke:C.ink2},svg);
  txt(svg,(m.l+w-m.r)/2,h-4,"Fit: chosen specialist in the top 5 →",{"text-anchor":"middle","font-size":11.5,fill:C.ink});
  txt(svg,13,(m.t+h-m.b)/2,"MIPS gain over the chosen specialist (points) →",{transform:`rotate(-90 13 ${(m.t+h-m.b)/2})`,"text-anchor":"middle","font-size":11.5,fill:C.ink});
  const phab=T.pcp.base.r5;el("line",{x1:x(phab),x2:x(phab),y1:m.t,y2:h-m.b,stroke:C.prac,"stroke-dasharray":"5 4"},svg);txt(svg,x(phab)+4,m.t+10,`PCP habit fit ${pct(phab)}`,{"font-size":11});
  [["model",C.graph,"Graph model"],["pcp","#9a988f","PCP habit"]].forEach(([k,c,n])=>{
    el("polyline",{points:V.map(v=>`${x(T[k][v].r5)},${y(T[k][v].gain5)}`).join(" "),fill:"none",stroke:c,"stroke-width":2.5},svg);
    V.forEach(v=>{const p=T[k][v];const big=k==="model"&&v==="blend0.3";const d=el("circle",{cx:x(p.r5),cy:y(p.gain5),r:big?8:5,fill:big?"#fff":c,stroke:c,"stroke-width":big?3:1},svg);
      txt(svg,x(p.r5)+(k==="model"?9:-9),y(p.gain5)+(k==="model"?-6:14),lab[v],{"font-size":11,fill:c,"text-anchor":k==="model"?"start":"end","font-weight":big?700:400});
      d.onmousemove=e=>showTip(e,`<b>${n}, ${lab[v]} MIPS weight</b><br>fit ${pct(p.r5)} · MIPS gain ${f1(p.gain5)}<br>load ${pct(p.busy1)} · new to PCP ${pct(p.new_pcp5)} · median travel ${mi(p.dist5_median)}`);d.onmouseleave=hideTip})});
  legend("qw-legend",[["",C.graph,"graph model at 0–50% MIPS weight"],["","#9a988f","PCP habit at 0–50% MIPS weight"],["","#fff","the 30% setting used from here on",`border:3px solid ${C.graph}`],["ln",C.prac,"fit of the PCPs’ own habit"]]);
  const b=T.model["blend0.3"], R=T.random, M=T.mips;
  html("qw-side",`<div class="card qual"><div class="k">MIPS gain at 30% weight (${state.qw})</div><div class="v">${f1(b.gain5)}</div><div class="d">points over the chosen specialist, interval ${f1(b.gain5_lo)} to ${f1(b.gain5_hi)}</div></div>
    <div class="card fit"><div class="k">Fit at 30% weight</div><div class="v">${pct(b.r5)}</div><div class="d">PCP habit ${pct(T.pcp.base.r5)}; model with no weight ${pct(T.model.base.r5)}</div></div>
    <p class="mini">At equal fit the graph model’s curve sits above the PCP-habit curve: it finds higher-MIPS options among plausible ones. Off the chart: random picks (fit ${pct(R.r5)}, ${f1(R.gain5)}) and “highest MIPS first” (fit ${pct(M.r5)}, ${f1(M.gain5)}).</p>`);
}
/* ---------- slide 6: graph vs table ---------- */
function renderGvT(){
  const Y=D.s11.years, rows=[["Fit","fit",pct,1],["Load (top pick)","load",pct,-1],["New to the PCP","new",pct,1],["MIPS gain","quality",v=>f1(v),1],["Travel, mean (mi)","travel",v=>v.toFixed(1),-1]];
  const df=(k,v)=>k==="quality"?f2(v):k==="travel"?f2(v):pt(v);
  const cls=(c,dir)=>{const lo=c[0],hi=c[1];return (hi<0||lo>0)?((dir<0?hi<0:lo>0)?"good":"bad"):""};
  html("gvt",`<tr><th></th><th colspan="3" style="text-align:center">2018</th><th colspan="3" style="text-align:center">2019 (held out)</th></tr><tr><th>Lens</th><th>graph</th><th>table</th><th>graph − table [95%]</th><th>graph</th><th>table</th><th>graph − table [95%]</th></tr>`+
    rows.map(([l,k,fm,dir])=>`<tr><td>${l}</td>`+["2018","2019"].map(y=>{const r=Y[y][k];return `<td>${fm(r.gnn)}</td><td>${fm(r.table)}</td><td class="${cls(r.ci,dir)}" style="white-space:nowrap">${df(k,r.diff)} [${df(k,r.ci[0])}, ${df(k,r.ci[1])}]</td>`}).join("")+`</tr>`).join(""));
  legend("gvt-legend",[["sq","#e5f6ef","graph model clearly better (interval excludes 0)"],["sq","#fdeee7","table clearly better"],["sq","#fff","no clear difference","border:1px solid #ccc"]]);
  const a=Y["2018"], b=Y["2019"];
  html("gvt-read",`<b>Reading.</b> MIPS gain is level; fit is level on 2018 (the graph model fits ${pt(b.fit.diff)} points better on 2019). The graph model sends fewer top picks to the busiest specialists in both years (${pt(a.load.diff)} and ${pt(b.load.diff)} points; actual choices ${pct(a.actual_load)} / ${pct(b.actual_load)}). The table proposes more specialists new to the PCP (${pt(-a.new.diff)} points) and slightly shorter trips.</p><p class="mini"><b>So each model has its own strength</b>: the graph model spreads load, the table opens new connections. The next slides ask why, and whether the two can be combined.`);
}
/* ---------- slide 13: 2019 and 10/15 names ---------- */
''' + y19_old.replace("function renderY19(){", "function renderY19top5(){") + r'''
function renderY19(){
  toggle("y19-view","y19",[["5","Top 5: 2018 vs 2019"],["k","10 and 15 names"]]);
  if(state.y19==="5"){renderY19top5();html("y19-note","Mean of the three seed pairings; hover a cell for the per-pairing values. Ticks: the pre-stated 2019 criterion for that row held in every pairing. Graph and table here average 3 runs each, so they differ slightly from slides 6–8 (2 runs).");
    html("y19-src","Source: runs/analysis/step12c/point_lenses_validation.json, point_lenses_test2019.json, preregistration_2019.json (recorded before the 2019 pass)");return}
  html("y19-sub",`The same steered blend (50/50, a = ${D.blend.setting.a_new}, b = ${D.blend.setting.b_busy}) with lists of 10 and 15 names; 2018 and 2019 (one pass each, after the setting was recorded)`);
  const E=D.s12e.K, rows=[["Load (top pick)","busy1",pct],["Load (all list slots)","busyK",pct],["New to the PCP","newK",pct],["Fit (chosen in the list)","fit",pct],["MIPS gain","gainK",v=>f1(v)]];
  const cell=(P,arm,k,fm)=>{const v=P.map(p=>p.fixed[arm][k]);return `<td style="white-space:nowrap" title="pairings: ${v.map(fm).join(" / ")}">${fm(mean(v))}</td>`};
  let h_=`<tr><th></th>`+["10","15"].map(K=>["2018","2019"].map(y=>`<th colspan="3" style="text-align:center">${K} names, ${y}</th>`).join("")).join("")+`</tr><tr><th>Lens</th>`+"<th>graph</th><th>table</th><th>blend</th>".repeat(4)+`</tr>`;
  h_+=rows.map(([l,k,fm])=>`<tr><td>${l}</td>`+["10","15"].map(K=>["2018","2019"].map(y=>["GNN","Table","Steered blend"].map(a=>cell(E[K][y],a,k,fm)).join("")).join("")).join("")+`</tr>`).join("");
  const allLow=["10","15"].every(K=>["2018","2019"].every(y=>E[K][y].every(p=>p.ci_load[1]<0)));
  h_+=`<tr><td>Blend load clearly below graph, every pairing</td>`+["10","15"].map(K=>["2018","2019"].map(y=>`<td colspan="3" style="text-align:center">${E[K][y].every(p=>p.ci_load[1]<0)?'<span class="ok">✓</span>':'<span class="nok">✗</span>'}</td>`).join("")).join("")+`</tr>`;
  html("y19-table",h_);
  html("y19-note","Same fixed setting as the top-5 tab, not re-tuned for longer lists. Mean of three seed pairings; hover for each pairing. Stacked LightGBM is not included (its inputs cover top-5 lists only).");
  html("y19-src",`Source: ${D.s12e.src}`);
  const b10=E["10"]["2019"], g10=b10.map(p=>p.fixed.GNN.busy1), s10=b10.map(p=>p.fixed["Steered blend"].busy1);
  html("y19-read",`<b>With longer lists</b> the steered blend keeps its top-pick load at about half the graph model’s (2019, 10 names: ${rng(s10,p0)} vs ${rng(g10,p0)}), puts fewer list slots on the busiest specialists, and matches the table’s share of new connections. ${allLow?"The load reduction is clear in every pairing, both list lengths and both years.":""}`);
  const fd=["10","15"].map(K=>E[K]["2019"].map(p=>p.fixed.GNN.fit-p.fixed["Steered blend"].fit));
  html("y19-cav",`<b>Caveat.</b> As with 5 names, fit is level with the table and a little below the graph model on 2019 (${rng(fd[0],p1)} points with 10 names, ${rng(fd[1],p1)} with 15).`);
}
/* ---------- slide 16: STEP13 pre-registered graph-value tests ---------- */
const mag=v=>(100*Math.abs(v)).toFixed(1);
const S13_HEAD={"H1a":"H1a · recovers history? (fit)","H1b":"H1b · learned vs fixed (fit)","H3":"H3 · insurer-compatible new picks"};
const S13_SHORT={"H1a":["Graph − no-MP (no history)","GNN − no-MP, no history","Joint half-gap"],"H1b":["− Flat-RF (LightGBM)","− MLP on summaries","− pure table","− table + summaries"],"H3":["− no-MP (no history)","− Flat-RF","− MLP on summaries","− table + summaries"]};
function renderS13(){
  if(!state.s13y)state.s13y="2018";toggle("s13-year","s13y",[["2018","2018 (decides)"],["2019","2019 (once)"]]);
  const S=D.s13, key=state.s13y==="2018"?"c2018":"c2019", rows=[];
  ["H1a","H1b","H3"].forEach(u=>S.units[u][key].forEach((c,i)=>rows.push({head:S13_HEAD[u],label:S13_SHORT[u][i],est:c.point,ci:c.ci})));
  forest("s13-chart",rows,{fmt:v=>pt(v),axis:"Graph model − comparator (points; right = graph model better)",labelW:330,colour:r=>r.ci[0]>0?C.graph:r.ci[1]<0?C.table:C.grey});
  html("s13-legend",`<span><i style="background:${C.graph}"></i>interval above 0</span><span><i style="background:${C.table}"></i>interval below 0 (comparator better)</span><span><i style="background:${C.grey}"></i>includes 0</span>`);
  const V={"supported":"✓ supported","not supported":"✗ not supported","inconclusive":"? inconclusive"};
  const NM={"H1a":"Message passing recovers what history columns give","H1b":"Learned propagation beats fixed summaries","H2_low_history":"Graph helps most for low-history PCPs","H2_thin_network":"… and for thin-network PCPs","H3":"More insurer-compatible new picks"};
  html("s13-units",`<tr><th>Claim (2018 decides)</th><th style="white-space:nowrap">Verdict</th><th style="white-space:nowrap">Holm p</th></tr>`+Object.keys(NM).map(u=>`<tr><td>${NM[u]}</td><td style="white-space:nowrap">${V[S.units[u].verdict]||S.units[u].verdict}</td><td>${S.units[u].p_holm.toFixed(2)}</td></tr>`).join(""));
  const h1=S.units.H1a[key], b=S.units.H1b[key], yr=state.s13y;
  html("s13-read",`<b>Reading (${yr}).</b> History columns are worth ${mag(h1[0].point)} points of fit; message passing recovers ${mag(h1[1].point)} of them. Fixed two-hop summaries in LightGBM beat the graph model by ${mag(b[0].point)}; an MLP on them ties it. The gap is tree vs neural, not graph vs no graph. ${yr==="2018"?"2019 shows the same pattern.":"Verdicts are decided on 2018; 2019 shows the same pattern."} Thin-history claims (H2): no edge on 2018, a non-significant hint on 2019.`);
  html("s13-src",`Source: ${S.src}. Holm over 5 claims; a claim needs every component above 0 and every seed agreeing.`);
}
/* ---------- slide 17: post hoc matched fit ---------- */
const PH_ROWS=[["flat_rf","table","Table + graph summaries vs pure table"],["info_nohist","table","Table + summaries + counts vs pure table"],["flat_rf","graph","Table + graph summaries vs full graph model"],["graph","table","Full graph model vs pure table (reference)"],["gnn_nohist","mlp_rf","Learned propagation vs MLP on summaries (no history)"]];
function renderPH(){
  const P=D.s13ph, g=(yr,a,b)=>P[yr].find(r=>r.A===a&&r.B===b), ci=x=>`<span style="color:#7a7a85;font-size:11.5px;white-space:nowrap"> [${pt(x.ci95[0])}, ${pt(x.ci95[1])}]</span>`;
  html("ph-table",`<tr><th>Comparison (A − B)</th><th>Load 2018</th><th>Load 2019</th><th>Newness 2018</th><th>Newness 2019</th></tr>`+PH_ROWS.map(([a,b,l])=>{const r8=g("2018",a,b),r9=g("2019",a,b);
    return `<tr><td>${l}</td><td><b>${pt(r8.matched_top_pick_load_A_minus_B.point)}</b>${ci(r8.matched_top_pick_load_A_minus_B)}</td><td><b>${pt(r9.matched_top_pick_load_A_minus_B.point)}</b>${ci(r9.matched_top_pick_load_A_minus_B)}</td><td>${pt(r8.matched_newness_A_minus_B.point)}</td><td>${pt(r9.matched_newness_A_minus_B.point)}</td></tr>`}).join(""));
  const f=g("2018","flat_rf","table"), f9=g("2019","flat_rf","table"), fg=g("2019","flat_rf","graph");
  html("ph-read",`<b>Reading.</b> Giving the table fixed summaries of each specialist’s and PCP’s past cases cuts its top-pick load by ${mag(f.matched_top_pick_load_A_minus_B.point)} (2018) and ${mag(f9.matched_top_pick_load_A_minus_B.point)} points (2019) at equal fit, with newness unchanged. Against the full graph model it has ${mag(fg.matched_top_pick_load_A_minus_B.point)} points lower load and ${mag(fg.matched_newness_A_minus_B.point)} points more new connections (2019). The network’s information helps; learning it by message passing is not what helps.`);
  html("ph-cav",`<b>Caveat.</b> Chosen after the pre-registered tests, on years already seen: a fair description, not a test. Native fit of the table with summaries is ${mag(f.fit_A-f.fit_B)} (2018) and ${mag(f9.fit_A-f9.fit_B)} points (2019) below the pure table; matching corrects for it.`);
  html("ph-src",`Source: ${P.src} (scripts/analysis/step13/step13_matched_load.py; STEP12a random-mixture matching)`);
}
/* ---------- slide 18: STEP14 three-way blend ---------- */
const S14_NM={"B_GTF":"Graph + table + graph-feature table","B_GT":"Graph + table (Step 12c blend)","B_GF":"Graph + graph-feature table","T":"Table alone","G":"Graph model alone","F":"Graph-feature table alone"};
function renderS14(){
  const S=D.s14, fams=["B_GTF","B_GT","B_GF","T","G","F"];
  html("s14-table",`<tr><th>Recommender (steered)</th><th>a / b (2018)</th><th>Top-pick load</th><th>Newness</th><th>Fit</th><th>MIPS gain</th></tr>`+fams.map(f=>{const c=S.choice[f],y=S.y2019[f];
    if(!c||c.cannot_reach_floors)return `<tr style="color:#8a8984"><td>${S14_NM[f]}</td><td colspan="5">cannot reach fit ≥ 45% and newness ≥ 35% on 2018</td></tr>`;
    const L=y.lenses_mean, b=f===S.proposed?"font-weight:700;background:#e5f6ef":"";
    return `<tr style="${b}"><td>${S14_NM[f]}${f===S.proposed?" ★":""}</td><td>${c.a_new} / ${c.b_busy}</td><td>${pct(L.busy1)}</td><td>${pct(L.new5)}</td><td>${pct(L.fit)}</td><td>${f1(L.gain5)}</td></tr>`}).join(""));
  html("s14-note",`2019, averaged over 3 seed pairings. ★ = chosen on 2018 before 2019 was looked at. a = push toward specialists new to the PCP, b = push away from the busiest 10%.`);
  const d=S.cmp.proposed_minus_B_GT_paired_bootstrap, ci=x=>`[${pt(x.ci95[0])}, ${pt(x.ci95[1])}]`, cim=x=>`[${f2(x.ci95[0])}, ${f2(x.ci95[1])}]`;
  html("s14-read",`<b>Reading.</b> The three-way blend meets the pre-stated test against the Step 12c blend in all three seed pairings: top-pick load ${pt(d.busy1.point)} points ${ci(d.busy1)}, fit ${pt(d.fit.point)} ${ci(d.fit)}, MIPS ${f2(d.gain5.point)} ${cim(d.gain5)}, newness ${pt(d.new5.point)} ${ci(d.new5)}. The graph model alone or the graph-feature table alone cannot reach the floors; combined, they set the best trade-off found.`);
  html("s14-cav",`<b>Caveat.</b> The idea came from Step 13 results that had seen 2019, so 2019 checks the settings and recommender chosen on 2018, not whether to try the graph-feature table. The gain is modest (about 2 points of top-pick load).`);
  html("s14-src",`Source: ${S.src} (scripts/analysis/step14/step14_blend.py)`);
}
/* ---------- slide 19: summary ---------- */
function renderTake(){
  const b19=D.blend.y2019, Y=D.s11.years;
  const tk=[["var(--load)","Imitation reproduces practice.","A recommender trained on past referrals mostly re-ranks the PCP’s usual specialists and puts more top picks on the busiest ones than PCPs do."],
    ["var(--q)","Quality needs a yardstick.",`A 30% MIPS weight gives ${f1(D.s10.tradeoff["2019"].model["blend0.3"].gain5)} MIPS points on 2019 at about the PCPs’ own fit.`],
    ["var(--table)","Graph vs table.",`Similar fit and quality; the graph model has lower load (${pt(Y["2019"].load.diff)} points on 2019), the table more new connections. Not because of message passing or PCP history.`],
    ["var(--graph)","What the graph adds.",`Pre-registered against matched controls: learned message passing does not beat the same neighbourhood information as fixed summaries (both years). The information itself helps: as summaries it cuts a table’s top-pick load by ${mag(D.s13ph["2019"].find(r=>r.A==="flat_rf"&&r.B==="table").matched_top_pick_load_A_minus_B.point)} points at equal fit (2019, post hoc).`],
    ["var(--blend)","Combine and steer.",`The models make different mistakes. Steered toward new, less busy specialists, a blend of graph model, table and graph-feature table reaches 2019 top-pick load ${pct(D.s14.y2019.B_GTF.lenses_mean.busy1)} (Step 12c blend ${pct(D.s14.y2019.B_GT.lenses_mean.busy1)}; graph alone ${rng(b19.map(p=>p.GNN.busy1),p0)}, table ${rng(b19.map(p=>p.Table.busy1),p0)}) at the table’s newness and higher MIPS. Trained-together hybrids do not beat it.`]];
  html("take",tk.map(([c,b,s])=>`<div class="panel" style="--tint:#f6f6f4;border-left:6px solid ${c};padding:12px 18px"><p class="mini"><b>${b}</b> ${s}</p></div>`).join(""));
  const asks=[["var(--blend)","Train for several goals at once","Steering is a weighted sum applied after the model. Multi-objective training (e.g. Tchebycheff or ε-constraint objectives) could build fit, load and newness into the model itself."],
    ["var(--load)","Who sets the steering weights?","How much fit to trade for lower load and new connections is a policy choice."],
    ["var(--graph)","Where could message passing still help?","New PCPs or specialists without history, several networks, larger graphs. 2019 hinted at an edge for thin-history PCPs; it needs a fresh pre-registered test."],
    ["var(--fit)","Open data question","Confirm the year of the specialist volume figures (likely ~2018, overlapping the evaluation year)."]];
  html("asks",asks.map(([c,q,s],i)=>`<div class="q" style="--qc:${c};margin:0 0 9px"><span class="n">${i+1}</span><div><b>${q}${q.endsWith("?")?"":"."}</b> <span style="font-size:15px;color:#44464f">${s}</span></div></div>`).join(""));
}
/* ---------- appendix ---------- */
function renderNet(){
  const N=D.network, Y=N.yearly, T=N.transitions;
  html("net-table",`<tr><th>Year</th><th>referrals</th><th>active PCPs</th><th>active specialists</th><th>busiest 10% receive</th><th>overlap with last year’s connections</th><th>referrals to specialists not used last year</th><th>community agreement (ARI)</th></tr>`+
    Y.map(y=>{const t=T.find(t=>t.to===y.year);return `<tr><td>${y.year}</td><td>${y.referrals.toLocaleString()}</td><td>${y.active_pcps}</td><td>${y.active_specialists.toLocaleString()}</td><td>${pct(y.median_busiest10_share)}</td><td>${t?pct(t.panel_edge_jaccard):"—"}</td><td>${t?pct(t.median_share_t1_referrals_to_specialists_not_used_in_t):"—"}</td><td>${t?t.community_ARI.toFixed(2):"—"}</td></tr>`}).join(""));
  html("net-read",`<b>Reading.</b> Only a minority of PCP–specialist connections recur from one year to the next, and most of a PCP’s referrals go to specialists they did not use the year before; referral communities agree only weakly year to year. Yet the busiest 10% of specialists keep receiving a similar share. The network changes fast; its concentration does not. “Not used last year” can only look back to the 2015 data start, so early years overstate newness; 2015 has few referrals in the data.`);
}
function renderMask(){
  const M=D.masking;
  html("mask-table",`<tr><th>Predicting the patient’s…</th><th>from the model without masking</th><th>from the masked model</th><th>from the masked model’s raw inputs</th></tr>`+
    ["gender","race"].map(k=>`<tr><td>${k}</td><td>${M[k].unmasked.toFixed(2)}</td><td><b>${M[k].masked.toFixed(2)}</b></td><td>${M[k].inputs.toFixed(2)}</td></tr>`).join(""));
  html("mask-read",`<b>Reading.</b> Without masking, gender and race are almost perfectly readable from the model’s embeddings (AUC ${M.gender.unmasked.toFixed(2)} / ${M.race.unmasked.toFixed(2)}). With masking they are barely readable (${M.gender.masked.toFixed(2)} / ${M.race.masked.toFixed(2)}) — close to what the remaining inputs (diagnoses, insurer, location) reveal on their own (${M.gender.inputs.toFixed(2)} / ${M.race.inputs.toFixed(2)}). The graph does not amplify them. Recommendations with and without masking were indistinguishable on every lens (Steps 9–10).`);
}
function renderAll(){const f=[renderData,renderLenses,renderImit,renderQW,renderGvT,renderGroups,renderTopK,renderMech,renderHyb,renderComp,renderFrontier,renderY19,renderHeavy,renderBC,renderS13,renderPH,renderS14,renderTake,renderNet,renderMask];f.forEach(fn=>{try{fn()}catch(e){console.error(fn.name,e)}})}
fit();go((parseInt(location.hash.slice(1))||1)-1);
</script>
</body>
</html>
'''
nav = '''
</div></div>
<div id="tip"></div>
<div class="nav"><button id="prev">‹</button><div class="dots" id="dots"></div><span id="count"></span><button id="next">›</button></div>

<script>
'''
out = head + '<div id="stage"><div class="deck" id="deck">\n' + SLIDES + nav + helpers + reused + NEW_JS
open(TPL / "template_story.html", "w").write(out); print("wrote template_story.html", len(out) // 1024, "KB", out.count("<section"), "slides")
