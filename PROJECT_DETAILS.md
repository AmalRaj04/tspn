# TSPN — Implementation & Results Reference

**Purpose of this document**: a complete, standalone description of the TSPN system *as actually built and run*, and every result actually observed. This is NOT a history of what changed from an original plan (see `PROJECT_STATE.md` and `docs/planning/` for that) — everything below describes the current, final state of the implementation and the real numbers it produced. Written so a reader with no other context can understand the system, reproduce the reasoning, and help interpret results or draft paper text.

**Repo**: `github.com/AmalRaj04/tspn`. Code referenced below lives under `src/`; result tables under `results/tables/`; config under `config.py` (repo root).

---

## 0. One-paragraph summary

TSPN (Tariff Shock Propagation Network) is a temporal graph attention network that predicts how a tariff shock on one bilateral trade relationship propagates through the global supply chain to affect producer prices economy-wide, at 3/6/12-month horizons. The graph has 2,408 nodes (43 countries × 56 sectors), built from WIOD input-output tables (2000–2014) extended with UN Comtrade data (2015–2021). Six real tariff events (US Section 232 steel/aluminum, US Section 301 List 1/2 vs. China, EU retaliation, UK Global Tariff post-Brexit) are used with Leave-One-Event-Out cross-validation (6 folds). The model is compared against a static Leontief input-output baseline, a per-node autoregressive baseline, and a graph-free MLP, and against 4 architectural ablations of itself. **Headline finding**: TSPN shows a real, fold-consistent predictive edge over its own ablations at the 3-month horizon specifically, but three independent interpretability probes find no evidence it has learned classical, economically-recognizable input-output structure — a genuine, reportable tension worth building the paper's discussion section around.

---

## 1. The Graph

- **Nodes**: 2,408 = 43 countries × 56 sectors. `node_id = country_idx * 56 + sector_idx`, both indices 0-based into fixed lists in `config.GRAPH["COUNTRY_LIST"]` / `["SECTOR_LIST"]`.
- **Countries (43)**: AUS, AUT, BEL, BGR, BRA, CAN, CHN, CYP, CZE, DEU, DNK, ESP, EST, FIN, FRA, GBR, GRC, HUN, IDN, IND, IRL, ITA, JPN, KOR, LTU, LUX, LVA, MEX, MLT, NLD, NOR, POL, PRT, ROU, RUS, SVK, SVN, SWE, TUR, TWN, USA, HRV, CHE. (No "Rest of World" aggregate node — excluded because no genuine per-node data exists for it.)
- **Sectors (56)**: ISIC/WIOD Rev. 4 classification, e.g. `A01` (crop/animal production), `C24` (basic metals — includes steel), `C29` (motor vehicles), `J58_J60` (publishing/broadcasting), `R_S` (arts/other services), etc. Full list in `config.GRAPH["SECTOR_LIST"]`.
- **Edges**: directed, `src` = supplying (country, sector), `tgt` = buying (country, sector). Built from WIOD's intermediate-use table for 2000–2014, extended for 2015–2021 by rescaling 2014 bilateral flows with UN Comtrade's HS2-level import totals (topology, i.e. which edges exist, is frozen at the 2014 structure; only flow magnitudes update for later years). An edge is kept only if `import_pen_coeff >= 0.001` (flow as a share of the buyer's total input use).
- **Canonical edge set used for all modeling**: exactly the edges present in `edges_2014.parquet` — **157,838 directed edges**. This exact edge index is held fixed across every temporal snapshot and every one of the 6 events (a deliberate design choice so the graph *structure* the model sees never changes, only node/edge *features* do).
- Verified real-world sanity: steel (C24) → automotive (C29) trade flow is ~24x the reverse direction, confirming the src/tgt orientation matches actual supply-chain direction.

## 2. Node Features (9-dimensional, per node per quarter)

| # | Name | Definition |
|---|---|---|
| f0 | `log_gross_output` | log(1 + gross output, USD millions) |
| f1 | `import_penetration` | imports / (gross output + imports − exports); **winsorized** to [0.0, 1.59] at the 1st/99th percentile of 2000–2014 data before normalization |
| f2 | `export_intensity` | exports / gross output; **winsorized** to [0.0, 1.26] |
| f3 | `backward_linkage` | column-sum of the Leontief inverse for that node (total direct+indirect input requirement) |
| f4 | `tariff_exposure` | trade-weighted average applied MFN tariff rate on that node's imports. **0.0 for 2000–2014** (no tariff-schedule data exists that far back); real value for 2015–2021. Left in raw (unnormalized) units deliberately — see §2.1. |
| f5–f8 | `ppi_lag_1..4` | quarter-over-quarter % change in producer price index, lagged 1–4 quarters |

**Normalization**: f0, f1, f2, f3, f5, f6, f7, f8 are z-scored using mean/std computed **only from 2000–2014** (the training-period years, never from the 2015–2021 event-window years, to avoid leaking future-period statistics into normalization). f4 is deliberately **not** z-scored — because it is identically 0.0 for the entire 2000–2014 window, its training-period standard deviation is 0, and z-scoring would divide any real 2015–2021 value by a near-zero denominator, exploding it by orders of magnitude. It is left in its natural, already-bounded raw scale (0.0–~1.0).

### 2.1 Where node feature data comes from by year

- **2000–2014**: real WIOD Socioeconomic Accounts (gross output, value added).
- **2015–2021**: WIOD has no data past 2014, so gross output/value added are proxied by scaling each country's 2014 baseline by its **World Bank nominal GDP growth ratio** (`NY.GDP.MKTP.CD`, no API key needed) between 2014 and the target year. Backward linkage (f3) for these years reuses the frozen 2014 Leontief inverse (structural prior held fixed, matching the frozen edge topology). PPI lags (f5–f8) come from real quarterly PPI data where available.

### 2.2 PPI / label coverage limitation (important, affects everything downstream)

Real PPI data exists for only **29 of 43 countries** (EU members + USA, plus a "WLD" world-commodity benchmark used as a fallback) and **25 of 56 sectors**. 15 countries — including **China, Japan, UK, Canada, South Korea, India, Brazil, Russia, Turkey** — have **zero** PPI coverage. Consequently:
- `has_ppi_lags` is true for only ~12–28% of node-years (rest default to 0.0).
- Every event's label coverage (the fraction of the 2,408 nodes with a usable ground-truth price target) is **27.9%** (672/2,408 nodes) — identical across all 6 events, since they all draw from the same underlying PPI dataset.
- This is a real, accepted data-availability ceiling, not a bug. It bounds what any model — baseline or TSPN — can be evaluated against.

## 3. Edge Features (6-dimensional, per edge per quarter)

| # | Name | Definition |
|---|---|---|
| e0 | `log_trade_flow` | log(1 + bilateral flow, USD) |
| e1 | `import_pen_coeff` | flow / buyer's total input use |
| e2 | `applied_tariff` | the importer's MFN tariff rate on this product (joined on **`src_sector`**, the traded product's own classification — not `tgt_sector`, the buying industry; verified via `build_shock_vectors.py`'s own convention) |
| e3 | `tariff_delta` | **the shock signal**. 0.0 for all quarters except the event quarter itself. Non-zero only on edges directly targeted by a real tariff action. |
| e4 | `product_hhi` | Herfindahl concentration index of the importer's HS2-level import composition for this product. **Approximation, not literally bilateral**: available Comtrade data is HS2-level with no bilateral partner breakdown, so this is the importer's own overall concentration for the relevant HS2 codes, applied uniformly to all suppliers of that product — not a true bilateral HHI. 0.0 for years/sectors with no Comtrade coverage. |
| e5 | `domestic_flag` | 1.0 if src_country == tgt_country else 0.0 |

## 4. The 6 Tariff Events

| Event | Date | Importer(s) | Exporter(s) | Δ tariff |
|---|---|---|---|---|
| `us_232_steel_2018` | 2018-03 | USA | all | +25 pp |
| `us_232_aluminum_2018` | 2018-03 | USA | all | +10 pp |
| `us_301_list1_2018` | 2018-07 | USA | China only | +25 pp |
| `us_301_list2_2018` | 2018-08 | USA | China only | +25 pp |
| `eu_retaliation_2018` | 2018-06 | 27 EU members | USA | varies by product (avg ~25 pp) |
| `uk_global_tariff_2021` | 2021-01 | UK | all | varies, includes both increases and decreases (post-Brexit rate reset vs. EU's Common External Tariff) |

Direct-hit node counts (nodes that are the target of a shocked edge) range from 33 (steel) to 524 (EU retaliation, since it hits 27 importing countries at once).

## 5. Temporal Structure

Each event's model input is **8 quarterly graph snapshots** ending at the event quarter (snapshot index 7 = the event quarter itself; 0 = seven quarters earlier). Quarterly snapshots are built by **linear interpolation** between adjacent annual data points (annual value = that year's Q4; intra-year quarters interpolated). The shock signal (e3) is **zero in snapshots 0–6 and non-zero only in snapshot 7** — verified as a hard invariant on every event, at every stage of the pipeline (this is the mechanism that prevents the model from "seeing the future" during training).

## 6. Labels

For each node and each event, three horizons: `delta_3m`, `delta_6m`, `delta_12m` — the cumulative % change in producer price index from the event quarter to 1/2/4 quarters later, computed by **compounding** the intervening quarterly % changes (not a simple two-point difference). A node's label is only used (`has_label=True`) if all three horizons have real underlying PPI data — partial/some-but-not-all-horizons-available nodes are excluded entirely, not partially imputed.

## 7. PyG Dataset Structure

Each event is one Python object (`TSPNEventGraph`, saved as `data/pyg_datasets/{event}.pt`) containing:
- `temporal_sequence`: list of 8 PyG `Data` objects (x: [2408,9], edge_index: [2,157838] — **identical across all 8 snapshots and all 6 events**, edge_attr: [157838,6])
- `y`: [2408,3] float tensor (3m/6m/12m labels, NaN where unlabeled)
- `label_mask`: [2408] bool
- `direct_hit_mask`: [2408] bool (True only for nodes that are the target of a directly-shocked edge at the event quarter)

---

## 8. Baseline Models (as implemented)

### 8.1 Leontief IO (`src/baselines/leontief_io.py`)
Purely analytic, no learning. For each event: `node_delta_tau[i] = Σ (import_pen_coeff × tariff_delta)` over i's incoming edges at the event quarter (i.e. the direct cost shock a node experiences from its own tariffed imports). Then `predicted_delta_p = L.T @ node_delta_tau × PASS_THROUGH_RATE`, where `L` is the Leontief inverse (2408×2408, computed from the 2014 technical-coefficient matrix). Gives the **same prediction for all 3 horizons** (no temporal component by construction).

`PASS_THROUGH_RATE` was calibrated once, by closed-form OLS, on the **UK Global Tariff 2021 event's 6-month horizon only** (locked procedure) — value: **−5.41858216**. Negative and counterintuitive at first glance; traced to a real data cause, not a bug: the UK event's direct-hit nodes are 100% `GBR` sectors, and `GBR` has zero PPI coverage (see §2.2), so the calibration is effectively fit on indirect/noise correlation among the other labeled UK-event nodes, not a genuine direct pass-through relationship. The UK event itself is **excluded from Leontief's own reported results** (using it to both calibrate and evaluate would be circular) — Leontief's results table has 5 rows, not 6.

### 8.2 Panel VAR / AutoReg (`src/baselines/panel_var.py`)
Per (country, sector) node with sufficient PPI history (≥16 quarters): `statsmodels.tsa.ar_model.AutoReg(ppi_series, lags=4, exog=tariff_series)`. (Note: `statsmodels.tsa.api.VAR` — the originally-intended tool — hard-requires ≥2 endogenous variables and cannot represent a single-variable PPI series; `AutoReg` is the correct substitute for what was actually intended.) Forecasts 4 quarters ahead, compounds into 3m/6m/12m exactly as the true labels are compounded. Nodes without enough history default to a 0.0 prediction (naturally excluded from evaluation anyway since they also lack labels).

### 8.3 MLP no-graph (`src/baselines/mlp_no_graph.py`)
Plain feedforward network, **no graph structure at all** — every node treated independently. Input: 9-dim node features **minus f3** (backward linkage, since that's derived from the full Leontief inverse and would smuggle graph information back in through the back door) **plus** a scalar direct-tariff-exposure feature (the same `node_delta_tau` quantity used in the Leontief baseline) = 9 total input dims. Architecture: `9 → 128 → 64 → 32 → 3`, ReLU activations, same optimizer/training protocol as TSPN itself (Adam, same LR schedule, same LOEO-CV) so the comparison isolates the effect of graph structure specifically, not training procedure differences.

---

## 9. TSPN Architecture (as implemented, `src/models/`)

```
Input: 8 quarterly graph snapshots (x: [2408,9], edge_index: [2,157838], edge_attr: [157838,6] each)

For each of the 8 snapshots independently:
  1. FeatureEmbedding:
       node: Linear(9→128) → BatchNorm1d → ReLU → Dropout(0.1)
       edge: Linear(6→64)  → ReLU → Dropout(0.1)
  2. gat_layer1: TSPNGATLayer (custom, see below) — 128 → 128
  3. gat_layer2: TSPNGATLayer (separate instance, different weights) — 128 → 128
  → append the resulting [2408,128] representation to a sequence list

Stack the 8 resulting [2408,128] representations → GRU (single layer,
unidirectional, hidden_dim=256, node dimension treated as the GRU's batch
dimension) → take only the FINAL hidden state [2408,256] → Dropout(0.2)

Output head: 3 fully INDEPENDENT MLPs (no shared weights), one per horizon:
  Linear(256→128) → ReLU → Dropout(0.2) → Linear(128→64) → ReLU → Linear(64→1)
  → concatenated to [2408, 3]  (columns: 3m, 6m, 12m)
```

### 9.1 TSPNGATLayer — the custom attention mechanism (`tspn_gat_layer.py`)
**Not** standard PyTorch Geometric `GATConv` — that ignores edge features in the attention computation, which would prevent the shock signal (carried in edge features) from ever modulating attention. Per head k (4 heads, 32-dim each):
```
query_i  = W_q_k @ h_i
key_j    = W_k_k @ h_j
edge_k   = W_e_k @ e_ij
score_ij = a_k^T · LeakyReLU([query_i || key_j || edge_k], slope=0.2)
score_ij = score_ij / sqrt(32)          # scaled attention (prevents logit explosion)
alpha_ij = softmax over incoming edges of node i
alpha_ij = Dropout(alpha_ij, p=0.3)     # training only
value_j  = W_v_k @ h_j
m_i      = Σ_j alpha_ij · value_j       # aggregated per head, heads concatenated → 128
h_i_out  = ELU(m_i) + W_res @ h_i_input # residual connection
```
The model stores `gat_layer1.last_alpha` (shape [157838, 4]) after every forward pass — the attention weights actually used, retrievable for interpretability analysis.

### 9.2 Parameter summary
- Node embed dim 128, edge embed dim 64, GAT: 2 layers × 4 heads × 32 dim, GRU hidden 256, output MLPs [256→128→64→1] × 3.
- Full TSPN: **570,307** parameters (verified by direct count). 1-layer ablation: **496,195** (genuinely fewer, since `gat_layer2` doesn't exist for it, not just unused).

---

## 10. Training Methodology

- **Cross-validation**: Leave-One-Event-Out, 6 folds (train on 5 events, validate/test on the 1 held out). Every fold uses a **freshly initialized** model and optimizer.
- **Loss**: `0.50 × MSE_3m + 0.30 × MSE_6m + 0.20 × MSE_12m + 0.01 × mean(|attention_weights|)` (the last term is an L1 sparsity regularizer on attention; automatically skipped — contributes 0 — for the GCN ablation, which has no attention weights).
- **Optimizer**: Adam, lr=1e-3, weight_decay=1e-4. Scheduler: CosineAnnealingWarmRestarts (T_0=50, T_mult=2).
- **Gradient clipping**: warms up from max_norm=5.0 (first 30 epochs) to 1.0 thereafter — prevents the GRU's naturally large early-training gradients from suppressing learning entirely.
- **Early stopping**: patience 20 epochs on validation RMSE at the 6-month horizon, max 200 epochs.
- **Data augmentation** (training folds only, never during validation): random ±1-quarter temporal jitter (50% probability), Gaussian noise on the shock magnitude (σ=0.05), random dropout of low-weight edges (p=0.05 for edges below 0.002 import-penetration), Gaussian label noise (σ=0.01).
- **Hardware**: trained on Google Colab's free T4 GPU. All 6 folds of the full model converged in **~7 minutes wall-clock total**. All genuinely early-stopped between epoch 21–26 (none hit the 200-epoch ceiling).

---

## 11. Ablation Study Design (4 variants, each isolating exactly one change)

| Ablation | What differs from full TSPN |
|---|---|
| `TSPN_GCN` | Both GAT layers replaced with uniform mean aggregation (no learned attention at all — every neighbor weighted equally). Tests whether attention adds value over simple averaging. |
| `TSPN_NoTemporal` | Only ever fed the single event-quarter snapshot (verified: mathematically identical output whether given the full 8-snapshot sequence or a pre-trimmed 1-snapshot sequence, in eval mode). Tests whether temporal/lag history adds value. |
| `TSPN_NoShock` | The shock signal is moved from an edge feature (e3) to a node feature instead — e3 is zeroed everywhere, and a 10th node feature (the same trade-weighted direct-exposure scalar used in the Leontief baseline) is added, correctly timed (0 pre-event, real only at the event quarter). Tests whether bilateral edge-level shock precision matters over a coarser node-level signal. |
| `TSPN_1Layer` | Only 1 GAT layer exists at all (not just unused — genuinely fewer parameters). Tests whether 2-hop (indirect supplier) reasoning adds value over 1-hop (direct suppliers only). |

All 4 trained identically to the full model (same loss, same LOEO-CV, same optimizer/schedule), each fold getting a fresh model instance.

---

## 12. RESULTS

### 12.1 Full comparison table — mean ± std across LOEO folds

| Model | n folds | RMSE_3m | RMSE_6m | RMSE_12m | DirAcc_3m | DirAcc_6m | DirAcc_12m |
|---|---|---|---|---|---|---|---|
| Leontief_IO | 5 | 0.0595±0.0216 | 0.0594±0.0222 | 0.0598±0.0221 | 0.307±0.345 | 0.300±0.175 | 0.284±0.191 |
| Panel_VAR | 6 | 0.0150±0.0062 | 0.0188±0.0123 | 0.0311±0.0375 | 0.506±0.306 | 0.565±0.132 | 0.597±0.146 |
| MLP_no_graph | 6 | 0.0121±0.0067 | 0.0169±0.0112 | 0.0371±0.0360 | 0.618±0.255 | 0.747±0.219 | 0.735±0.143 |
| TSPN_GCN | 6 | 0.0151±0.0052 | 0.0158±0.0086 | 0.0319±0.0361 | 0.459±0.232 | 0.770±0.144 | 0.727±0.194 |
| TSPN_NoTemporal | 6 | 0.0138±0.0048 | 0.0172±0.0096 | 0.0349±0.0349 | 0.709±0.358 | 0.728±0.211 | 0.745±0.221 |
| TSPN_NoShock | 6 | 0.0146±0.0033 | 0.0157±0.0077 | 0.0305±0.0363 | 0.569±0.302 | 0.745±0.185 | 0.743±0.198 |
| TSPN_1Layer | 6 | 0.0150±0.0061 | 0.0147±0.0073 | 0.0313±0.0328 | 0.560±0.299 | 0.752±0.143 | 0.770±0.177 |
| **TSPN (full)** | 6 | **0.0137±0.0049** | 0.0164±0.0090 | 0.0354±0.0335 | **0.719±0.271** | 0.701±0.215 | 0.746±0.222 |

*(RMSE and MAE are on the same scale as the labels — fractional price change, e.g. 0.05 = 5%. R² values were computed but are negative in 82% of all (model, fold, horizon) cells across every model including baselines — median R² is -0.2 to -1.2 depending on horizon, occasionally as low as -107 — because the labeled subsample's variance is small and noisy; RMSE and directional accuracy are the metrics actually worth reasoning about here, not R².)*

**Finding 1 — TSPN vs. naive baseline**: TSPN clearly and decisively beats the static Leontief IO model on every metric, every horizon. Expected and unsurprising, but a necessary sanity check that the whole exercise is measuring something real.

**Finding 2 — TSPN vs. learned baselines (MLP, Panel VAR)**: mixed, not a clean win. TSPN has the best RMSE_6m and best DirAcc_3m/DirAcc_12m among all 8 models, but MLP has the best RMSE_3m, and both MLP and Panel_VAR are competitive-to-better on some other cells. The standard deviations are large relative to the mean differences (e.g. TSPN's RMSE_6m 0.0164±0.0090 vs. MLP's 0.0169±0.0112 overlap heavily) — **not statistically distinguishable at n=6 folds**. This is inherent to the small-N design (only 6 real tariff events exist to hold out), not a flaw in execution.

### 12.2 Ablation study — per-fold win counts (TSPN vs. each ablation, out of 6 folds; higher = TSPN better)

| vs. full TSPN | RMSE_3m | RMSE_6m | RMSE_12m | DirAcc_3m | DirAcc_6m | DirAcc_12m |
|---|---|---|---|---|---|---|
| TSPN_GCN | 4/6 | 2/6 | 3/6 | **6/6** | 1/6 | 4/6 |
| TSPN_NoTemporal | **5/6** | 4/6 | 3/6 | 2/6 | 1/6 | 1/6 |
| TSPN_NoShock | 3/6 | 2/6 | 3/6 | 5/6 | 1/6 | 1/6 |
| TSPN_1Layer | 4/6 | 2/6 | 2/6 | 5/6 | 3/6 | 3/6 |

**Finding 3 — the real, headline architectural result**: TSPN's advantage over its own simplified ablations is concentrated almost entirely at the **3-month horizon**. There it wins RMSE in 3–5 of 6 folds against every ablation, and sweeps the GCN ablation 6/6 on directional accuracy (a clean, defensible "learned attention beats uniform aggregation, at least at short horizon" result). But at 6 and 12 months, that advantage **mostly disappears or reverses**: TSPN beats `TSPN_1Layer` on RMSE_6m in only 2/6 folds and beats `TSPN_GCN` on DirAcc_6m in only 1/6 folds — the "simpler" ablations are frequently *better* at longer horizons.

**Interpretation for the paper**: lead with the 3-month result — it's real and fold-consistent, especially the attention-vs-GCN comparison. Do not claim uniform superiority across all horizons; the data doesn't support that. A defensible explanation: with only 5 training events per fold, the full model's extra capacity (temporal GRU, 2-hop attention, edge-injected shock) may need more data than exists here to pay off on noisier, longer-horizon targets where unmodeled macro factors likely dominate. This is a legitimate finding about small-N regimes and architecture complexity, not a failure.

### 12.3 Interpretability — attention vs. Leontief inverse correlation

For every event, correlation between the model's learned attention weights (averaged across the 4 heads) and the corresponding entry of the classical Leontief inverse matrix, computed over all 157,838 real edges:

| Layer | mean Pearson r | mean Spearman r |
|---|---|---|
| Layer 1 | +0.0502 (std 0.0004) | −0.0687 (std 0.0018) |
| Layer 2 | +0.0520 (std 0.0004) | −0.0644 (std 0.0016) |

Both essentially **negligible** in magnitude (r ≈ 0.05–0.07 on a −1 to 1 scale), and notably the **sign disagrees** between Pearson and Spearman for the theoretically-correct pairing — meaning there isn't even a clean monotonic relationship, just a very weak, inconsistent one. (The p-values are extremely small, e.g. 1e-90, but that's purely an artifact of the enormous sample size — 157,838 edges makes even a trivial correlation "statistically significant"; the effect size itself is what matters here, and it's tiny.) The index-order convention (`L[src,tgt]`, meaning "how much of src's output is embodied in tgt's total output," matched to "how much does tgt's attention weigh src") was independently verified against a known real-world fact (steel→auto flow is 24x the reverse) before trusting this result, and the transposed pairing gives similarly weak correlations, so this isn't an index-order artifact.

**Finding 4**: TSPN's attention does not recover the classical input-output economic structure in any strong or even weakly-consistent sense.

### 12.4 Interpretability — cascade depth (how far does a shock's predicted effect reach?)

For every event, starting from the directly tariff-hit nodes and doing BFS along the supply-chain graph:

| Event | Origin nodes | Hop 1 nodes | Hop 1 attenuation | Hop 2 nodes | Hop 2 attenuation |
|---|---|---|---|---|---|
| us_232_steel_2018 | 33 | 1,483 | 0.98 | 727 | 0.97 |
| us_232_aluminum_2018 | 44 | 1,619 | 0.98 | 580 | 0.97 |
| us_301_list1_2018 | 43 | 1,695 | 0.65 | 505 | 0.55 |
| us_301_list2_2018 | 43 | 1,695 | 0.65 | 505 | 0.55 |
| eu_retaliation_2018 | 524 | 1,513 | 1.44 | 207 | 1.82 |
| uk_global_tariff_2021 | 54 | 1,489 | 0.90 | 698 | 0.94 |

(Attenuation ratio = average |predicted 6m price change| at that hop distance ÷ average at the origin nodes; a ratio near 1.0 means no attenuation, a ratio above 1.0 means the model predicts a *larger* effect further from the shock than at the shock itself.) The graph is extremely densely connected — **nearly all 2,408 nodes are reachable within 2 hops** of any origin set, regardless of how few origin nodes there are. No event shows meaningful attenuation (the significance threshold of 5% of origin magnitude is never crossed within 5 hops for any event).

**Why, traced rather than taken at face value**: the model's predictions have very low variance from node to node (e.g. for `us_232_steel_2018`, 6-month predictions range roughly −0.005 to +0.026 with std 0.0017, and directly-hit nodes are statistically indistinguishable from randomly-sampled non-hit nodes). There was never a strong origin-vs-distance differential to attenuate in the first place — this is a "the model doesn't strongly localize its predictions" finding, not "the shock powerfully floods the entire global economy."

### 12.5 Interpretability — amplifier sectors (attention centrality vs. real trade centrality)

Eigenvector centrality computed on two graphs sharing the same edges: one weighted by learned attention (averaged across all 6 events), one weighted by raw 2014 trade flow.

- **Attention-weighted centrality is nearly uniform across all 2,244 ranked nodes**: ranges only 0.02065–0.02111 (std 0.00002) — from Malta's near-trivial "household employer activities" sector to China's dominant manufacturing sectors, the model's attention treats them almost identically.
- **Trade-weighted centrality correctly shows real economic concentration**: China's manufacturing sectors dominate (e.g. `CHN_F` construction-adjacent sector at 0.49, `CHN_C24` basic metals at 0.38), while small economies' minor sectors sit near zero (e.g. `MLT_T`, `HRV_T` near 1e-12).
- Consequently the "amplification ratio" (attention centrality ÷ trade centrality) is dominated entirely by the denominator — nominally "top amplifiers" are just the smallest-trade-volume nodes (Malta, Croatia, Slovenia's tiniest sectors), and nominal "under-weighted" nodes are just China's biggest sectors. This is a direct restatement of the uniformity finding, not a separate discovery of genuinely important-but-overlooked nodes.
- (Two nodes, `CYP_T` and `DNK_T`, initially showed nonsensical *negative* centrality — traced to both having zero in-edges, which degenerates the eigenvector solver; excluded from the ranking as a known numerical artifact, not a real finding.)

**Finding 5**: consistent with Findings 4 and the cascade-depth result — the model has not learned to replicate the real economy's actual, highly skewed concentration of importance. Its attention is comparatively egalitarian.

---

## 13. Consolidated Narrative for the Paper

1. **The core predictive claim is real but narrow, not sweeping**: TSPN beats a naive analytic baseline decisively, and shows a genuine, fold-consistent edge over both learned baselines and its own architectural ablations — but *only* at the 3-month horizon. At 6 and 12 months, simpler variants of itself are often just as good or better.
2. **The interpretability story is a genuine null/negative result, not a success story** — three independent probes (Leontief correlation, cascade attenuation, eigenvector centrality) all agree the model has not learned classical, sparse, economically-recognizable supply-chain structure. Whatever predictive signal it captures at 3 months appears to be distributed/implicit rather than structured the way classical input-output theory would predict.
3. **These two findings are in tension, and that tension is itself worth discussing**: a model that predicts moderately well without learning recognizable structure raises a real question about *what* it's actually learning — pattern-matching on the input feature distributions rather than genuine causal/structural propagation is a plausible, honest hypothesis to raise in the limitations section.
4. **Data constraints bound everything**: only 6 real tariff events exist to train/validate on (small-N LOEO-CV), and real price labels exist for only 29/43 countries and 25/56 sectors (27.9% node coverage, identical across all events). Both are inherent to what data actually exists in the world, not implementation shortfalls, and should be stated plainly as scope limitations.
5. **Suggested paper structure implication**: lead the results section with the 3-month / attention-vs-GCN finding as the strongest, most defensible claim. Present the 6m/12m ablation results and all three interpretability analyses honestly as a genuine limitation/discussion section, not something to omit or soften. Suggested future-work directions that follow naturally from the findings: attention sparsity regularization (to encourage the model toward more classically-interpretable, concentrated attention), more training events (the field has more real tariff actions since 2021 that could be added), and explicit structural priors that bias the model toward known input-output relationships rather than learning purely from data.

---

## 14. Known, accepted limitations (for the paper's limitations section)

- **Label coverage**: 27.9% of nodes per event have real price-change ground truth; 15 of 43 countries (including China, Japan, UK, Canada, South Korea) have zero PPI coverage.
- **product_hhi (e4)** is an HS2-level, non-bilateral approximation, not the true bilateral HS6 concentration the original design called for (data availability constraint).
- **Small-N cross-validation**: only 6 real tariff events exist; all statistical comparisons between models should be read with wide, overlapping confidence intervals in mind, not as sharp rankings.
- **2015–2021 node features rely on a GDP-scaling proxy** for gross output (WIOD's real data stops at 2014) and a frozen 2014 structural prior for backward linkage — both documented approximations, not measurement error.
- **R² is not a useful metric here** — strongly negative for every model due to the small, noisy labeled subsample; RMSE and directional accuracy are the metrics actually worth reporting/reasoning about.

---

## 15. Where the underlying numbers live (for anyone who wants to re-derive or extend this)

- `results/tables/baselines.csv` — full per-fold baseline results
- `results/tables/all_results.csv` — full per-fold TSPN + ablation results
- `results/tables/interpretability_attention_leontief_correlation.csv`
- `results/tables/cascade_depth.csv`
- `results/tables/amplifier_sectors.csv`
- `config.py` — single source of truth for every hyperparameter/dimension referenced above
- `src/models/`, `src/baselines/`, `src/training/`, `src/analysis/` — full implementation
- `PROJECT_STATE.md` + `docs/planning/` — deviation history and original design intent, if that context is ever needed
