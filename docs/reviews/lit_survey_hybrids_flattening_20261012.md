# Literature survey: graph flattening, GNN + GBDT hybrids, gating, popularity bias, referral recommenders

2026-10-12. A subagent did the web research. Each citation was checked against arXiv, the publisher or proceedings page, Crossref or PMC, except where noted under "Not verified". Mappings to our setup are the survey's inference. They are not established results.

## 1. Graph flattening (precomputed propagation)

### Methods

| Paper | Link | How it turns the graph into features |
|---|---|---|
| **SGC** (Wu et al., ICML 2019) | https://arxiv.org/abs/1902.07153 | Removes the GCN nonlinearities, leaving a fixed filter S^K·X plus a linear classifier. Each node gets K-hop smoothed features, computed once. |
| **SIGN** (Frasca, Rossi et al., 2020) | https://arxiv.org/abs/2004.11198 | Precomputes [X, AX, A²X, …, PPR·X, triangle-motif·X], one column block per operator and hop. An MLP is trained on top. |
| **NARS** (Yu et al., 2020) | https://arxiv.org/abs/2011.09679 | Heterogeneous version: SIGN-style blocks per sampled relation subset, combined by a learned weighting. |
| **SeHGNN** (Yang et al., AAAI 2023) | https://arxiv.org/abs/2207.02547 | One precomputed mean aggregation per metapath and a single layer, with a small transformer fusing the metapath vectors. Beats HGT-style models and is the closest published "flattened HGT". |
| **GAMLP** (Zhang et al., KDD 2022) | https://arxiv.org/abs/2206.04355 | Multi-hop precomputed features, with node-adaptive attention over hop depth. |
| **node2vec** (Grover & Leskovec, KDD 2016) | https://arxiv.org/abs/1607.00653 | Shallow embeddings from random walks, usable as GBDT columns. |
| **metapath2vec** (Dong et al., KDD 2017) | DOI 10.1145/3097983.3098036 | Same, with walks following a metapath, e.g. PCP–Enc–Spec–Enc–PCP. |
| **BUDDY/ELPH** (Chamberlain et al., ICLR 2023) | https://arxiv.org/abs/2209.15486 | Pairwise subgraph sketches plus SIGN features, scored with a simple model. This is the pairwise counterpart of flattening and the closest fit to our candidate rows. |

### Evidence that fixed aggregation competes with GNNs

- **GraphLand, formerly TabGraphs** (Bazhenov et al., NeurIPS 2025 D&B, https://arxiv.org/abs/2409.14500): GBDTs with graph-based features "can sometimes be very strong baselines".
- **Rubio-Madrigal & Burkholz, "Fixed Aggregation Features Can Rival GNNs"** (ICML 2026, https://arxiv.org/abs/2601.19449): MLPs on training-free aggregations match or beat GNNs on 12 of 14 benchmarks.

### Leakage references

- RelBench (https://arxiv.org/abs/2407.20060): seed-time rule. Only rows timestamped before each example's seed time may enter its features.
- Kaufman et al., "Leakage in data mining", TKDD 2012 (DOI 10.1145/2382577.2382579).

### Our setup

Useful metapaths:

| Metapath | What it gives |
|---|---|
| P→E→S | The PCP's past referral counts and shares to this specialist and to its specialty |
| S–E–P–E–S | Specialist co-referral (item-to-item collaborative filtering) |
| P–E–S–E–P | Similar PCPs (user-to-user collaborative filtering) |
| S←E | The specialist's case-mix profile and its match to the encounter |

Rules:
- Use means with degree normalisation, not sums.
- Compute everything strictly before the weekly cutoff and mask the query edge.
- Fit transductive embeddings per cutoff, or freeze them on the training-period graph.
- Volume features must be point-in-time.
- Report the result as a separate "flattened-graph" model, never as the baseline.

## 2. GNN + GBDT hybrids

- **BGNN** (Ivanov & Prokhorenkova, ICLR 2021, https://arxiv.org/abs/2101.08543)
  - How it works: each epoch adds GBDT trees, and their output feeds the GNN's node features. The next trees fit the GNN's input gradient.
  - Res-GNN variant: GBDT predictions are added to the GNN's features in two stages.
  - Results: BGNN is best with heterogeneous tabular features, GBDT alone with sparse features, and the GNN with homogeneous features.
- **EBBS** (Chen et al., ICLR 2022, https://arxiv.org/abs/2110.13413): boosting with graph smoothing and provable convergence. It is comparable to or better than BGNN and reports that BGNN is unstable.
- **TabGNN** (Guo et al., DLP-KDD 2021 workshop, https://arxiv.org/abs/2108.09127): multiplex-graph embeddings are concatenated with the features for a downstream model.
- **He et al. (Facebook), ADKDD 2014** (DOI 10.1145/2648584.2648589): GBDT leaf indices feed a logistic regression. The hybrid beats either model alone by more than 3%.
- **Correct & Smooth** (Huang et al., ICLR 2021, https://arxiv.org/abs/2010.13993): a base model, then label-propagation post-processing of its residuals and predictions.
- **Stacking GNN embeddings into trees.** Weber et al. 2019 (https://arxiv.org/abs/1908.02591): GCN embeddings added to a random forest gave only a marginal gain (F1 0.788 → 0.796). In RelBench, by contrast, GNNs dominated LightGBM on recommendation tasks.

**Our setup:** BGNN and EBBS are built for node classification and regression, so adapting them to pairwise ranking isn't off the shelf. The patterns that transfer cheaply are:
- stacking, with cross-fitting by time block;
- a GNN fitting the GBDT's residual;
- GBDT leaf indices as features.

## 3. Mixture of experts, gating, late fusion

- **Mowst** (Zeng et al., ICLR 2024, https://arxiv.org/abs/2311.05185): an MLP expert and a GNN expert. A gate built from the MLP's confidence calls the GNN only where the MLP is unsure.
- **Link-MoE** (Ma et al., NeurIPS 2024, https://arxiv.org/abs/2402.08583): a gate routes each node pair to one of several link-prediction experts.
- **ContextGNN** (Yuan et al., ICLR 2025, https://arxiv.org/abs/2411.19513): GNN pair representations score familiar items, two-tower representations score exploratory ones, and the two are fused. This is the closest analogue to our finding (graph = familiar and less congested; table = new connections).
- **FWLS** (Sill et al., 2009, https://arxiv.org/abs/0911.0460): blend weights are linear functions of query meta-features.
- **Background:**
  - Ekstrand & Riedl, RecSys 2012 (DOI 10.1145/2365952.2366002): different algorithms fail on different cases.
  - Burke 2002, hybrid recommender taxonomy (DOI 10.1023/A:1021240730564).
  - Reciprocal Rank Fusion, Cormack et al., SIGIR 2009 (DOI 10.1145/1571941.1572114).
  - Heater, SIGIR 2020 (DOI 10.1145/3397271.3401178).
  - Wide & Deep (https://arxiv.org/abs/1606.07792).

**Our setup:** start with Reciprocal Rank Fusion, then a global linear blend, then an FWLS or small-MLP gate on query features:
- PCP history depth;
- diagnosis group and specialty;
- whether the candidate is new to the PCP;
- the candidate's current load.

A Mowst-style margin gate is a variant. A ContextGNN-style split (GNN for familiar candidates, table for new ones) tests whether the two models complement each other.

## 4. Popularity and degree bias

### Graph convolution can increase concentration

- Zhao et al., SIGIR 2022 (https://arxiv.org/abs/2204.12326): symmetric-normalised aggregation (LightGCN) worsens popularity bias, and more so with depth.
- Chen et al., "How Graph Convolutions Amplify Popularity Bias" (https://arxiv.org/abs/2305.14886).
- Lin et al., WSDM 2025 (https://arxiv.org/abs/2404.12008): spectral view. Popularity sits in the principal component of the score matrix.
- Subramonian et al., ICML 2024 (https://arxiv.org/abs/2309.17417): with symmetric normalisation, GCN link scores grow with the geometric mean of the two endpoints' degrees.
- Aiyappa et al., ICML 2025 (https://arxiv.org/abs/2405.14985): with uniform negative sampling, a degree-only predictor is near-optimal.
- Tang et al., CIKM 2020 (DOI 10.1145/3340531.3411872): degree-related differences in accuracy.

### Graph scoring and training choices can reduce it

- Yin et al., PVLDB 2012 (https://arxiv.org/abs/1205.6700): random walks that surface long-tail items.
- RP3β (Christoffel et al., RecSys 2015, DOI 10.1145/2792838.2800180; Paudel et al., TiiS 2016): divides walk scores by item degree^β.
- Wu et al., sampled-softmax loss (https://arxiv.org/abs/2201.02327): argues sampled softmax mitigates popularity bias.
- Yi et al., RecSys 2019 (DOI 10.1145/3298689.3346996): logQ correction for popularity-skewed negatives.

### Gap

No peer-reviewed paper compares GBDT/LambdaMART with GNN rankers on exposure or concentration.

Candidate explanations for our results (inference, to be tested):
- (a) HGT's attention normalises over neighbours like a mean, so a specialist's representation doesn't grow with its degree.
- (b) Sampled softmax with uniform negatives.
- (c) The table splits directly on specialist-volume attributes.

Our own finding that the no-graph GNN has the same load as the GNN points away from (a) and toward (b) and/or (c).

## 5. Healthcare referral and provider recommendation

- **Fouladvand et al., J. Biomed. Inform. 2023** (DOI 10.1016/j.jbi.2023.104407): a heterogeneous GNN predicts the procedures a specialist orders after a referral, in EHR data. AUC 0.88 vs 0.80 for the best baseline. The target differs from ours.
- **Noshad, Jankovic, Chen, KDD 2020 workshop** (https://arxiv.org/abs/2007.12161): non-graph predecessor.
- **Duarte, Han, Soares 2021** (Naver Labs manuscript; venue not confirmed): referral link prediction with a GNN.
  - Extended version: https://arxiv.org/abs/2312.02387.
  - SHAP ranks degree and centrality as the dominant predictors, which is a popularity signal.
- **Barnett et al., Health Serv Res 2011** (DOI 10.1111/j.1475-6773.2011.01262.x): Medicare shared-patient counts predict self-reported referral ties with AUC 0.73. This validates claims-derived edges.
- **An et al. 2017** (https://arxiv.org/abs/1711.03245): structure of the US referral network.
- **FD-GATDR** (https://arxiv.org/abs/2207.05750): federated heterogeneous graph attention network for doctor recommendation.

**Gap:** no peer-reviewed GNN or GNN + GBDT referral recommender on Medicare claims.

## Ranked shortlist of experiments (judged on the four lenses; the pure-tabular baseline is unchanged)

| # | Experiment | Cost | What it tests |
|---|---|---|---|
| 1 | **Flattened-graph LightGBM** (separate model): point-in-time metapath aggregates at 4, 12 and 52 weeks, co-referral, similar-PCP and case-mix match; degree-normalised, query edge masked | low | How much of the GNN's behaviour fixed aggregation reproduces, and whether mean vs sum normalisation drives load |
| 2 | **Late fusion and gating** of the existing models: Reciprocal Rank Fusion, then a linear blend, then an FWLS or MLP gate, with a margin gate as a variant | low | Whether the two models' strengths complement each other by segment; whether a blend gets the table's new connections without its load. No retraining |
| 3 | **Stack the GNN into the GBDT**: time-cross-fitted GNN pair scores and frozen embeddings as LambdaRank features | low–med | Whether the GBDT can use the GNN's signal |
| 4 | **Residual GNN on a GBDT prior**: LightGBM score as a fixed logit offset, or leaf indices as node features | med | What the graph adds beyond the table |
| 5 | **SeHGNN / SIGN-style precomputed model** with the same scorer and loss as the GNN | med | Whether learned message passing matters compared with fixed propagation |
| 6 | **Popularity-mechanism ablations**: table without volume columns; GNN with sum aggregation; logQ correction in the loss | low | Which mechanism causes the load difference |

Not recommended: BGNN or EBBS joint training. The cost is high, they don't fit pairwise ranking, and EBBS reports BGNN is unstable.

## Not verified

- Venues for LightGCN, Wide & Deep, and the sampled-softmax paper (the TOIS listing).
- The BMC Health Services Research DOI for the extended Duarte paper.
- The NeurIPS 2024 status of the earlier TabGraphs version.
- The FD-GATDR venue.
- Link-MoE's exact set of experts.
- The RecSys Challenge 2022 GNN + LightGBM pipeline, so it is excluded.
