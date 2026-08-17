# TSPN — Locked Implementation Plan
### Phase-by-Phase Technical Specification (Do Not Deviate)

> **Purpose**: Every architectural decision, parameter value, schema definition, formula, and tool choice is finalized here. During implementation you follow this spec exactly. You do not revisit these decisions mid-build. If something needs changing, update this document first, then propagate the change everywhere it appears.

---

## MASTER LOCKED PARAMETERS REFERENCE

### Graph
| Parameter | Locked Value |
|---|---|
| Total nodes | 2,464 (44 countries × 56 sectors) |
| Node ID formula | country_idx × 56 + sector_idx |
| Country indexing | Alphabetical sort of WIOD 44-country list |
| Sector indexing | ISIC code alphabetical sort |
| Edge inclusion threshold | import_pen_coeff >= 0.001 |
| Self-loops | Included |
| Temporal sequence length | 8 quarterly snapshots |
| Quarterly interpolation method | Linear between adjacent annual WIOD tables |
| Expected edge count after filtering | 80,000 – 130,000 per snapshot |

### Node Feature Vector — 9 dimensions, ORDER LOCKED
| Index | Name | Formula |
|---|---|---|
| f[0] | log_gross_output | log(gross_output_usd_millions + 1) |
| f[1] | import_penetration | total_imports / (gross_output + total_imports − total_exports + 1e-9) |
| f[2] | export_intensity | total_exports / (gross_output + 1e-9) |
| f[3] | backward_linkage | column sum of Leontief inverse (I−A)⁻¹ for this node |
| f[4] | tariff_exposure | Σ_j (trade_share_ij × applied_tariff_ij) over all import partners j |
| f[5] | ppi_lag_1 | (PPI[t−1] − PPI[t−2]) / PPI[t−2] |
| f[6] | ppi_lag_2 | (PPI[t−2] − PPI[t−3]) / PPI[t−3] |
| f[7] | ppi_lag_3 | (PPI[t−3] − PPI[t−4]) / PPI[t−4] |
| f[8] | ppi_lag_4 | (PPI[t−4] − PPI[t−5]) / PPI[t−5] |

### Edge Feature Vector — 6 dimensions, ORDER LOCKED
| Index | Name | Formula |
|---|---|---|
| e[0] | log_trade_flow | log(flow_usd_millions + 1) |
| e[1] | import_pen_coeff | flow_ij / total_input_j |
| e[2] | applied_tariff | trade-value-weighted MFN tariff rate for (src_country, tgt_country, sector) |
| e[3] | tariff_delta | new_rate − old_rate (0.0 for non-shocked edges) — THE SHOCK SIGNAL |
| e[4] | product_hhi | Σ_k (trade_share_k)² across HS6 codes in this bilateral-sector pair |
| e[5] | domestic_flag | 1.0 if src_country == tgt_country, else 0.0 |

### Model Architecture — ALL DIMS LOCKED
| Component | Parameter | Value |
|---|---|---|
| Embedding | node_feat_in | 9 |
| Embedding | edge_feat_in | 6 |
| Embedding | node_embed_dim | 128 |
| Embedding | edge_embed_dim | 64 |
| Embedding | node_normalization | BatchNorm1d(128) |
| Embedding | dropout | 0.1 |
| GAT | num_layers | 2 (not 1, not 3) |
| GAT | num_heads | 4 |
| GAT | head_dim | 32 |
| GAT | concat_out_dim | 128 (4 heads × 32) |
| GAT | attention_activation | LeakyReLU(negative_slope=0.2) |
| GAT | aggregation_activation | ELU |
| GAT | attention_dropout | 0.3 |
| GAT | residual_connection | True — Linear(128,128) added to output |
| GRU | input_dim | 128 |
| GRU | hidden_dim | 256 |
| GRU | num_layers | 1 |
| GRU | seq_len | 8 |
| GRU | bidirectional | False |
| GRU | output_dropout | 0.2 |
| MLP | layer_dims per head | [256, 128, 64, 1] |
| MLP | activation | ReLU |
| MLP | dropout | 0.2 after layers 1 and 2 only |
| MLP | num_heads | 3 (independent: 3m, 6m, 12m) |

### Training — ALL VALUES LOCKED
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| LR scheduler | CosineAnnealingWarmRestarts |
| T_0 | 50 |
| T_mult | 2 |
| Gradient clip max_norm | 1.0 |
| Max epochs | 200 |
| Early stopping patience | 20 epochs |
| Early stopping metric | validation RMSE on 6m horizon |
| Loss weight 3m | 0.50 |
| Loss weight 6m | 0.30 |
| Loss weight 12m | 0.20 |
| Loss weight L1_attention | 0.01 |
| Validation protocol | Leave-One-Event-Out CV, 6 folds |
| Weight initialization | Kaiming uniform (PyTorch default for Linear + ReLU) |

### Augmentation — TRAINING ONLY, ALL SPECS LOCKED
| Augmentation | Specification |
|---|---|
| Shock magnitude noise | Gaussian N(0, 0.05²) added to e[3] for non-zero entries only |
| Temporal jitter | 50% probability per epoch: shift sequence by ±1 quarter randomly |
| Edge dropout | Zero edge features (not remove from index) for 5% of edges with import_pen_coeff < 0.002 |
| Label noise | Gaussian N(0, 0.01²) added to all non-null label values |

### Loss Function (exact formula, locked)
```
L = 0.50 × MSE(pred[:,0], label[:,0], mask)
  + 0.30 × MSE(pred[:,1], label[:,1], mask)
  + 0.20 × MSE(pred[:,2], label[:,2], mask)
  + 0.01 × mean(|alpha_weights|)
```

### Evaluation — LOCKED
| Parameter | Value |
|---|---|
| Primary metrics | RMSE, MAE, R², Directional Accuracy |
| All metrics report format | mean ± std across 6 LOEO folds |
| Bootstrap CI | 1000 resamples, 95% CI |
| Cascade significance threshold | 5% of mean direct-hit |Δp| |
| Amplifier centrality algorithm | eigenvector_centrality_numpy (NetworkX) |
| Correlation methods reported | Pearson r AND Spearman ρ, both with p-values |
| Statistical significance level | p < 0.01 |

### Product Stack — LOCKED
| Layer | Choice |
|---|---|
| MVP framework | Streamlit |
| Graph visualization | PyVis |
| Inference export format | ONNX (opset_version=17) |
| Production API | FastAPI + Uvicorn |
| Dev database | SQLite |
| Prod database | Supabase free tier (PostgreSQL) |
| Hosting | Streamlit Community Cloud (free) |
| Experiment tracking | Weights & Biases free tier |
| ONNX target latency | < 3 seconds on CPU |

### Exact Package Versions (Install Exactly These)
```
python          3.10.x
torch           2.1.0
torch_geometric 2.4.0
torch-scatter   2.1.2
torch-sparse    0.6.18
pandas          2.1.0
numpy           1.26.0
scipy           1.11.3
networkx        3.2.1
pyarrow         14.0.0
plotly          5.17.0
streamlit       1.28.0
pyvis           0.3.2
fastapi         0.104.0
uvicorn         0.24.0
onnx            1.15.0
onnxruntime     1.16.3
comtradeapicall 0.2.1
wbdata          0.3.0
openpyxl        3.1.2
wandb           0.16.0
scikit-learn    1.3.2
statsmodels     0.14.0
```

---

## PHASE 0 — Environment and Project Initialization
**Duration**: 2 days
**Output**: Configured repo, config.py, verified installs, folder structure

### Step 0.1 — Create Folder Structure (Exactly This, No Additions)
```
tspn/
├── config.py                     ← SINGLE SOURCE OF TRUTH for all parameters
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   ├── wiod/
│   │   ├── comtrade/
│   │   ├── wits/
│   │   ├── bls_ppi/
│   │   ├── eurostat_ppi/
│   │   ├── commodity_prices/
│   │   ├── tariff_events/
│   │   └── concordance/
│   ├── processed/
│   │   ├── edges/
│   │   ├── node_features/
│   │   ├── tariff_rates/
│   │   ├── shock_vectors/
│   │   └── labels/
│   └── pyg_datasets/
├── src/
│   ├── data/
│   ├── models/
│   ├── baselines/
│   ├── training/
│   └── analysis/
├── notebooks/
├── models/
│   ├── checkpoints/
│   └── onnx/
├── results/
│   ├── tables/
│   └── figures/
└── app/
    ├── components/
    ├── utils/
    └── assets/
```

### Step 0.2 — config.py (Create Before Any Other File)
config.py is the single source of truth. No script may hardcode a value that exists in config.py. The file contains dictionaries or dataclasses for:
- PATHS: all absolute paths for data, models, results
- GRAPH: n_countries, n_sectors, n_nodes, edge_threshold, seq_len, country_list, sector_list, node_id formula
- NODE_FEATURES: dim=9, ordered feature names as list, formulas as comments
- EDGE_FEATURES: dim=6, ordered feature names as list, formulas as comments
- MODEL: every dimension from the locked table above
- TRAINING: every optimizer, scheduler, loss, augmentation value from the locked tables
- EVENTS: list of 6 dicts, each with keys: name, date, hts_file, affected_importers, affected_exporters, description
- EVAL: metrics list, cascade_threshold, significance_level, bootstrap_n
- PRODUCT: framework choices, db paths, onnx paths

### Step 0.3 — Validation Check
Write a single validation script that imports config.py and asserts:
- All paths in PATHS exist or can be created
- All numeric values match the master table in this document
- country_list has exactly 44 entries
- sector_list has exactly 56 entries
- EVENTS list has exactly 6 entries

Run this before proceeding.

### Phase 0 Exit Criteria
- [ ] All packages installed at exact versions
- [ ] Folder structure exists
- [ ] config.py exists, imports without error, passes validation check
- [ ] Git repository initialized with first commit

---

## PHASE 1 — Data Collection
**Duration**: 8 days
**Output**: All raw data files on disk at specified paths

### Step 1.1 — WIOD Download (Day 1)
**Source**: https://www.rug.nl/ggdc/valuechain/wiod/ — no registration required

**Download exactly these files**:
- 17 annual Excel files: WIOT2000_October16_ROW.xlsx through WIOT2016_October16_ROW.xlsx
- 1 socioeconomic accounts file: wiot_sep_16_txt.zip (extract after download)
- Save all to: data/raw/wiod/

**Manual verification after download**: Open WIOT2016_October16_ROW.xlsx. Identify the exact row and column offset where the IO matrix begins. Record these two offsets in config.py as WIOD_MATRIX_ROW_OFFSET and WIOD_MATRIX_COL_OFFSET. The matrix should be approximately 2,464 rows × 2,464 columns (intermediate use only, before final demand columns).

### Step 1.2 — Concordance Files Download (Day 1)
**Source**: https://unstats.un.org/unsd/trade/classifications/correspondence-tables.asp

**Download exactly these three files**:
- HS 2017 → ISIC Rev. 4 correspondence → save as: data/raw/concordance/hs2017_isic4.xlsx
- HS 2012 → ISIC Rev. 4 correspondence → save as: data/raw/concordance/hs2012_isic4.xlsx
- NAICS 2017 → ISIC Rev. 4 correspondence → save as: data/raw/concordance/naics2017_isic4.xlsx

### Step 1.3 — Tariff Event HTS Code Extraction (Days 2–3)
For each event, create one CSV with columns: hts_code (str, 8 digits), product_description (str), delta_tariff_pct (float).

**File 1**: data/raw/tariff_events/us_232_steel_2018.csv
- Source: Federal Register Vol. 83 No. 45 (March 8, 2018) — https://www.federalregister.gov/documents/2018/03/08/2018-04875
- delta_tariff_pct = 25.0 for all rows

**File 2**: data/raw/tariff_events/us_232_aluminum_2018.csv
- Same Federal Register document as above
- delta_tariff_pct = 10.0 for all rows

**File 3**: data/raw/tariff_events/us_301_list1_2018.csv
- Source: Federal Register Vol. 83 No. 119 (June 20, 2018) — https://www.federalregister.gov/documents/2018/06/20/2018-13248
- delta_tariff_pct = 25.0 for all rows
- Approximately 818 HTS-8 codes — copy from Annex table in the document

**File 4**: data/raw/tariff_events/us_301_list2_2018.csv
- Source: Federal Register Vol. 83 No. 155 (August 10, 2018)
- delta_tariff_pct = 25.0 for all rows
- Approximately 284 HTS-8 codes

**File 5**: data/raw/tariff_events/eu_retaliation_2018.csv
- Source: EU Official Journal L 160/1 (June 25, 2018) — https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L:2018:160:TOC
- delta_tariff_pct: use exact rates from the Annex (varies by product — 25% steel, varies for others)
- affected_importer: all EU member states in WIOD country list

**File 6**: data/raw/tariff_events/uk_global_tariff_2021.csv
- Source: https://www.gov.uk/guidance/uk-tariffs-from-1-january-2021 + trade-tariff.service.gov.uk API
- delta_tariff_pct = UK_rate − EU_CET_rate per HTS code (can be negative — reductions are valid)

### Step 1.4 — WITS Tariff Download (Days 3–4)
**Source**: https://wits.worldbank.org/ — free registration required

**Download applied MFN tariff rates for these reporters × years**:
- USA: 2015, 2016, 2017, 2018, 2019, 2020, 2021
- CHN: 2015–2021
- All EU member states in WIOD: 2015–2021 (one download per country)
- CAN: 2017, 2018, 2019
- GBR: 2019, 2020, 2021

**Download type**: Applied Tariff → MFN Applied → all HS6 products → CSV format

**Naming**: data/raw/wits/tariff_{ISO3}_{YEAR}.csv

**Also download**: For USA reporter, partner=CHN specifically: "Effectively Applied" tariff for 2018–2020 (captures Section 301 additional duties on bilateral flows)
- Save as: data/raw/wits/tariff_usa_china_effective_{YEAR}.csv

### Step 1.5 — BLS PPI Download (Day 4)
**Source**: https://www.bls.gov/ppi/data.htm — free API registration at api.bls.gov

**Pull**: All Industry PPI series (prefix PCU) for:
- NAICS 3-digit manufacturing: codes 311–339
- Mining: 211, 212, 213
- Selected services: 481 (air), 483 (water), 484 (truck transport), 4931 (warehousing)
- Time range: January 2014 – December 2023 (monthly)

**Save**: data/raw/bls_ppi/bls_ppi_{NAICS_CODE}.csv — one CSV per series

### Step 1.6 — Eurostat PPI Download (Day 5)
**Source**: https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_inppd_m

**Pull**: All EU member states in WIOD, NACE 2-digit codes B through N, monthly 2014–2023

**Save**: data/raw/eurostat_ppi/eurostat_ppi_all.csv — single file, all countries and sectors

### Step 1.7 — World Bank Commodity Prices (Day 5)
**Source**: https://www.worldbank.org/en/research/commodity-markets → Pink Sheet download

**Download**: CMO-Historical-Data-Monthly.xlsx (no registration)

**Save**: data/raw/commodity_prices/wb_pink_sheet.xlsx

### Step 1.8 — UN Comtrade API Setup (Days 6–8)
**Source**: https://comtradeplus.un.org/ — free registration, 500 calls/hour

**Pull strategy**: HS 2-digit level for all WIOD country pairs, years 2017–2021
- One API call per (reporter, year) combination
- Wait 7.5 seconds between calls (rate limit compliance)
- Parameters: flow=M (imports), classification=HS, commodity=AG2 (all 2-digit aggregates)

**Save naming**: data/raw/comtrade/comtrade_{ISO3}_{YEAR}.parquet

**Also pull** HS6 specifically for tariff-affected codes:
- Reporter=USA, Partner=CHN, years 2018–2020: pull only HS6 codes in Section 301 lists
- Save as: data/raw/comtrade/comtrade_usa_chn_301hs6_{YEAR}.parquet

### Phase 1 Exit Criteria
- [ ] 17 WIOD Excel files in data/raw/wiod/
- [ ] WIOD_MATRIX_ROW_OFFSET and WIOD_MATRIX_COL_OFFSET recorded in config.py
- [ ] 6 tariff event CSVs created with correct delta_tariff_pct values
- [ ] WITS files for USA, CHN, EU countries, CAN, GBR across specified years
- [ ] BLS PPI CSVs for all NAICS codes listed
- [ ] Eurostat PPI downloaded
- [ ] Comtrade Parquet files for all 44 reporters × 5 years (2017–2021)

---

## PHASE 2 — WIOD Processing
**Duration**: 4 days
**Output**: data/processed/edges/edges_{YEAR}.parquet for years 2000–2016, Leontief inverses

### Step 2.1 — WIOD Parser
**File**: src/data/parse_wiod.py
**Run**: 17 times (once per year 2000–2016) via a loop

**Exact parsing sequence (locked)**:
1. Load Excel with openpyxl engine, skiprows = WIOD_MATRIX_ROW_OFFSET from config
2. Extract (country, sector) header pairs from column headers
3. Extract the N×N intermediate use sub-matrix (exclude final demand and value-added rows/columns)
4. Convert to long format: year, src_country, src_sector, tgt_country, tgt_sector, flow_usd
5. Remove rows where flow_usd <= 0
6. Compute tgt_total_input = sum of all inputs into the buyer node (row sum of full buyer input row)
7. Compute import_pen_coeff = flow_usd / tgt_total_input
8. Apply threshold filter: keep only rows where import_pen_coeff >= 0.001
9. Cast dtypes: year→int16, all country/sector columns→category, flow_usd→float32, import_pen_coeff→float32
10. Save as Parquet with snappy compression

**Output schema (locked)**:
```
year             int16
src_country      category
src_sector       category
tgt_country      category
tgt_sector       category
flow_usd         float32   (millions USD)
import_pen_coeff float32   (range: [0.001, 1.0])
```

### Step 2.2 — Leontief Inverse Computation
**File**: src/data/compute_leontief.py

For each year 2000–2016:
1. Load edge table for that year
2. Construct N×N technical coefficients matrix A: A[i,j] = flow_ij / gross_output_j (from socioeconomic accounts)
3. Compute Leontief inverse: L = scipy.linalg.inv(I − A) where I is 2464×2464 identity
4. Compute backward linkages: BL[j] = column sum of L
5. Save L as: data/processed/edges/leontief_{YEAR}.npy — shape (2464, 2464), float32
6. Save BL as: data/processed/edges/backward_linkage_{YEAR}.parquet — columns: country, sector, node_id, backward_linkage

**Note**: Matrix inversion takes 2–5 minutes per year on CPU. Compute once, cache, never recompute unless edge data changes.

### Step 2.3 — Socioeconomic Account Extraction
**File**: src/data/parse_wiod_sea.py

Extract gross_output and value_added per (country, sector, year) from wiot_socioeconomic.xlsx

**Output schema**:
```
year           int16
country        category
sector         category
gross_output   float32   (millions USD)
value_added    float32   (millions USD)
```
Save as: data/processed/edges/socioeconomic_{YEAR}.parquet

### Step 2.4 — Comtrade Graph Extension (2017–2021)
**File**: src/data/extend_with_comtrade.py

**Design decision (locked)**: Use WIOD 2016 sector structure (technical coefficients) as a structural prior. Only update bilateral trade flow magnitudes from Comtrade. Do not attempt to update sector shares. Document this approximation explicitly.

For each year 2017–2021:
1. Load Comtrade HS2-digit aggregates for all reporters
2. Apply HS2→ISIC mapping (approximate: HS2 to ISIC 2-digit sector)
3. For each bilateral (src_country, tgt_country, isic_sector): sum Comtrade flows → flow_usd_comtrade
4. Use WIOD 2016 gross output as denominator for import_pen_coeff (scale by ratio of nominal GDP growth as proxy for gross output growth)
5. Apply same threshold: import_pen_coeff >= 0.001
6. Save with same schema as WIOD edge tables

### Phase 2 Exit Criteria
- [ ] 22 edge Parquet files (2000–2016 WIOD + 2017–2021 Comtrade)
- [ ] 17 Leontief .npy files, all shape (2464, 2464)
- [ ] 17 backward_linkage Parquet files
- [ ] Edge count per year: 80,000–150,000 after threshold
- [ ] No negative import_pen_coeff values
- [ ] All 44 countries present in each year

---

## PHASE 3 — Tariff and Shock Processing
**Duration**: 4 days
**Output**: data/processed/tariff_rates/tariff_rates.parquet, data/processed/shock_vectors/shock_{EVENT}.parquet

### Step 3.1 — HS-ISIC Concordance Builder
**File**: src/data/build_concordance.py

Build Python dict: hs6_to_isic → maps hs6_code (str) → list of (isic_code, weight) tuples

**Weight rule (locked)**:
- HS6 maps to one ISIC: weight = 1.0
- HS6 maps to multiple ISIC sectors: distribute proportionally using WIOD 2016 sector gross output shares — larger sectors receive higher weight
- Weights for each HS6 must sum to 1.0

Also build: naics3_to_isic dict for BLS mapping (mostly 1:1)

**Save**: data/processed/concordance/hs6_isic_weights.json and naics3_isic.json

### Step 3.2 — Sector-Level Tariff Rate Computation
**File**: src/data/compute_tariff_rates.py

**Exact formula per (src_country, tgt_country, isic_sector, year) (locked)**:
```
tariff_rate = Σ_k [ (comtrade_flow_k / total_comtrade_flow_sector) × wits_rate_k ]
```
where k iterates over all HS6 codes mapping to this ISIC sector

**Fallback rules (in priority order)**:
1. If bilateral rate exists in WITS → use it
2. If only MFN rate exists (no bilateral) → use MFN rate
3. If no WITS data for that year → linear interpolation from nearest available years
4. If no WITS data at all for that reporter → set to 0.0 and flag with data_source = 'missing'

**Output schema**:
```
year           int16
src_country    category
tgt_country    category
isic_sector    category
tariff_rate    float32   (decimal: 0.25 = 25%)
data_source    category  (values: wits_bilateral / wits_mfn / interpolated / missing)
```

### Step 3.3 — Shock Vector Builder
**File**: src/data/build_shock_vectors.py

For each of the 6 events defined in config.py:
1. Load event's HTS CSV file
2. For each HTS code → map to ISIC sector(s) using concordance with weights
3. Build affected_pairs: all (src_country, tgt_country, isic_sector) where tgt_country in event.affected_importers AND src_country in event.affected_exporters (or 'all')
4. Compute delta_tariff per affected pair: sum of (concordance_weight × event_delta_tariff_pct / 100) across matching HTS codes
5. All other pairs: delta_tariff = 0.0
6. Add is_direct_hit = True only for pairs with delta_tariff > 0

**Output schema**:
```
event_name     str
src_country    category
tgt_country    category
isic_sector    category
delta_tariff   float32   (0.0 for non-shocked)
is_direct_hit  bool
```
Save as: data/processed/shock_vectors/shock_{event_name}.parquet

### Phase 3 Exit Criteria
- [ ] hs6_isic_weights.json and naics3_isic.json exist and load correctly
- [ ] tariff_rates.parquet covers all WIOD country pairs, all 56 sectors, years 2015–2021
- [ ] Zero null values in tariff_rate column (interpolation fills all gaps)
- [ ] us_232_steel shock: only tgt_country=USA entries are non-zero
- [ ] us_301_list1 shock: only (src=CHN, tgt=USA) entries are non-zero

---

## PHASE 4 — Feature Engineering
**Duration**: 5 days
**Output**: node_features_{YEAR}.parquet, edge_features_{EVENT}_q{0-7}.parquet, labels_{EVENT}.parquet, normalization_stats.json

### Step 4.1 — PPI Unified Table
**File**: src/data/clean_ppi.py

Process BLS, Eurostat, and World Bank commodity data into one unified quarterly table.

**Conversion formula (locked for all three sources)**:
```
ppi_change_t = (ppi_level_t − ppi_level_{t-1}) / ppi_level_{t-1}
Monthly → Quarterly: average of 3 monthly changes within the quarter
```

**Commodity to ISIC mapping (locked, stored in config.py)**:
- Steel HRC → C24
- Aluminum → C24
- Copper → C24
- Iron ore → B (mining)
- Coal → B
- Brent oil → C19
- Wheat, Corn, Soy → A01

**Output schema**:
```
year        int16
quarter     int8    (1–4)
country     category
isic_sector category
ppi_change  float32
source      category  (bls / eurostat / wb_commodity)
```
Save as: data/processed/labels/ppi_quarterly_all.parquet

### Step 4.2 — Node Feature Computation
**File**: src/data/compute_node_features.py

For each (country, sector, year), compute features f[0]–f[8] in exact order from the locked table.

**Missing PPI lag handling (locked)**: For years 2000–2003 where lags reference pre-2000 data (before WIOD coverage), fill with 0.0. Flag these in a separate boolean column: has_ppi_lags.

**Normalization (locked)**:
- Compute mean and std for each of the 9 features using years 2000–2016 ONLY (training period)
- Save stats to: data/processed/node_features/normalization_stats.json
- Apply z-score normalization: f_norm = (f − mean) / std
- normalization_stats.json must be used at inference time — never recompute from test data

**Output schema**:
```
year          int16
country       category
sector        category
node_id       int16   (from config.py mapping: country_idx × 56 + sector_idx)
f0 through f8 float32 (normalized)
has_ppi_lags  bool
```
Save as: data/processed/node_features/node_features_{YEAR}.parquet

### Step 4.3 — Edge Feature Computation
**File**: src/data/compute_edge_features.py

For each event × each of 8 quarterly snapshots:
1. Load edge table for the snapshot year
2. Load tariff_rates for the snapshot year
3. Compute HHI per bilateral-sector from Comtrade HS6 flows: HHI = Σ_k (flow_k / total_flow)²
4. Load shock vector — e[3]=tariff_delta is non-zero only in snapshot 7 (event-time), zero in snapshots 0–6
5. Build 6-dim feature vector in locked order
6. Save as: data/processed/edge_features/edge_features_{event_name}_q{0-7}.parquet

**Schema**:
```
src_id     int16
tgt_id     int16
e0 through e5  float32
```

### Step 4.4 — Label Generation
**File**: src/data/generate_labels.py

For each event at date d_e, and each labeled (country, sector):
```
delta_3m  = (ppi_quarterly[d_e + 1 quarter] − ppi_quarterly[d_e]) / ppi_quarterly[d_e]
delta_6m  = (ppi_quarterly[d_e + 2 quarters] − ppi_quarterly[d_e]) / ppi_quarterly[d_e]
delta_12m = (ppi_quarterly[d_e + 4 quarters] − ppi_quarterly[d_e]) / ppi_quarterly[d_e]
```

**Output schema**:
```
event_name    str
country       category
sector        category
node_id       int16
delta_3m      float32   (null if no PPI)
delta_6m      float32   (null if no PPI)
delta_12m     float32   (null if no PPI)
has_label     bool
label_source  category  (bls / eurostat / wb_commodity / null)
```
Save as: data/processed/labels/labels_{event_name}.parquet

### Phase 4 Exit Criteria
- [ ] Node feature files for all 22 years, each with exactly 2,464 rows
- [ ] normalization_stats.json exists with mean and std for all 9 features
- [ ] Edge feature files for all 6 events × 8 quarters = 48 files
- [ ] has_label=True for at least 60% of nodes in each event
- [ ] No NaN values in node feature tensors
- [ ] delta_6m distribution sanity check: most values in (−0.10, +0.10)

---

## PHASE 5 — PyG Graph Dataset Construction
**Duration**: 3 days
**Output**: data/pyg_datasets/{event_name}.pt for all 6 events

### Step 5.1 — Node and Edge Index Construction
**File**: src/data/build_pyg_dataset.py

**Node index**: Must be consistent across all graphs. Node node_id = country_idx × 56 + sector_idx from config.py. Same node_id = same (country, sector) in every graph and every snapshot.

**Edge index**: (2 × num_edges) LongTensor. edge_index[0] = source node IDs, edge_index[1] = target node IDs. Sort edges by (src_id, tgt_id) before building tensor — this ensures edge feature matrices align correctly across snapshots.

### Step 5.2 — Temporal Sequence Assembly
For each event:
1. Identify 8 quarterly snapshots: event_quarter−7 through event_quarter (inclusive)
2. For each snapshot q in [0..7]: build PyG Data with x=(2464, 9), edge_index=(2, E), edge_attr=(E, 6)
3. Stack as Python list: temporal_sequence = [Data_q0, ..., Data_q7]
4. Build label tensor y: shape (2464, 3), NaN for unlabeled nodes
5. Build label_mask: shape (2464,), bool
6. Build direct_hit_mask: shape (2464,), bool (True for nodes with delta_tariff > 0)

**Package as custom object with attributes**:
```
.temporal_sequence  = list of 8 PyG Data objects
.y                  = FloatTensor (2464, 3)
.label_mask         = BoolTensor (2464,)
.direct_hit_mask    = BoolTensor (2464,)
.event_name         = str
.event_date         = str
```
Save as: data/pyg_datasets/{event_name}.pt using torch.save

### Step 5.3 — Dataset Validation
Write validation function, run before proceeding to Phase 6:
- All 6 .pt files load without error
- Each temporal_sequence has exactly 8 elements
- Each x tensor shape is (2464, 9)
- No NaN in x tensors
- label_mask True fraction >= 0.60 for all events
- e[3] (tariff_delta) is non-zero in snapshot 7, zero in snapshots 0–6 for each event

### Phase 5 Exit Criteria
- [ ] 6 .pt files in data/pyg_datasets/
- [ ] All pass validation function
- [ ] torch.load works on all 6 files without warnings

---

## PHASE 6 — Baseline Models
**Duration**: 5 days
**Output**: results/tables/baselines.csv with all baseline metrics

### Step 6.1 — Leontief IO Baseline
**File**: src/baselines/leontief_io.py

**Calibration (locked procedure)**: Use UK Global Tariff 2021 event as the calibration event for pass_through_rate. Find the scalar that minimizes RMSE_6m on that event. Record this value in config.py as LEONTIEF_PASS_THROUGH_RATE. Then apply that same scalar to all other events — do not re-tune per event.

**Prediction formula (locked)**:
```
node_delta_tau[i] = Σ_j (import_pen_coeff[j→i] × delta_tariff[j→i])  for all incoming edges
predicted_delta_p = L.T @ node_delta_tau × LEONTIEF_PASS_THROUGH_RATE
```
Note: Leontief gives the same vector for all three horizons (no temporal component) — this is expected and correct.

### Step 6.2 — Panel VAR Baseline
**File**: src/baselines/panel_var.py

**Locked settings**: Lag order p=4 (quarters). One VAR per (country, sector) with PPI as endogenous variable and tariff_rate as exogenous. Use statsmodels VAR class with exog parameter.

### Step 6.3 — MLP No-Graph Baseline
**File**: src/baselines/mlp_no_graph.py

**Locked architecture**: Input dim = 9 + 1 = 10 (9 node features + 1 scalar for total direct tariff exposure). Hidden layers: [128, 64, 32]. Output: 3 (one per horizon). Same LOEO-CV, same optimizer, same loss weights as TSPN — controls for training procedure, isolates the effect of graph structure.

### Phase 6 Exit Criteria
- [ ] LEONTIEF_PASS_THROUGH_RATE calibrated and saved to config.py
- [ ] All 3 baselines produce predictions for all 6 events
- [ ] results/tables/baselines.csv populated
- [ ] Leontief directional accuracy in range 0.55–0.70 (sanity check)

---

## PHASE 7 — TSPN Architecture Implementation
**Duration**: 9 days
**Output**: All model files in src/models/, verified end-to-end forward pass

### Step 7.1 — Feature Embedding Module
**File**: src/models/feature_embedding.py
**Class**: FeatureEmbedding

**Node path (locked)**:
```
Linear(9, 128) → BatchNorm1d(128) → ReLU → Dropout(0.1)
```

**Edge path (locked)**:
```
Linear(6, 64) → ReLU → Dropout(0.1)
```

No weight sharing between paths. Kaiming uniform initialization (PyTorch default).

### Step 7.2 — Custom GAT Layer with Edge Features
**File**: src/models/tspn_gat_layer.py
**Class**: TSPNGATLayer (inherits torch_geometric.nn.MessagePassing)

This is NOT standard PyG GATConv. Standard GATConv ignores edge features in attention computation. This custom layer includes edge features.

**Exact attention formula per head k (locked)**:
```
query_i  = W_q_k @ h_i              → dim 32
key_j    = W_k_k @ h_j              → dim 32
edge_kk  = W_e_k @ e_ij             → dim 32  (edge_embed_dim=64 → head_dim=32)
concat   = [query_i || key_j || edge_kk]  → dim 96
score_ij = a_k^T × LeakyReLU(concat, negative_slope=0.2)   → scalar
alpha_ij = softmax over all j in N(i) of score_ij
alpha_ij = Dropout(alpha_ij, p=0.3)  (training only)
```

**Aggregation per head k (locked)**:
```
value_j = W_v_k @ h_j               → dim 32
m_i^k   = Σ_{j∈N(i)} alpha_ij × value_j
```

**Multi-head concat and residual (locked)**:
```
h_i_concat = concat([m_i^1, m_i^2, m_i^3, m_i^4])  → dim 128
h_i_act    = ELU(h_i_concat)
h_i_out    = h_i_act + W_res @ h_i_input              (W_res is Linear(128,128), no bias)
```

**Store attention weights**: `self.last_alpha = alpha_ij_all_edges_all_heads` as class attribute during every forward pass. This is required for interpretability analysis.

### Step 7.3 — GRU Temporal Module
**File**: src/models/tspn_gru.py
**Class**: TSPNTemporalGRU

**Exact specification (locked)**:
- Input: list of 8 node feature matrices, each (2464, 128)
- Stack to tensor: (8, 2464, 128)
- GRU: input_size=128, hidden_size=256, num_layers=1, batch_first=False, bidirectional=False
- Use ONLY the final hidden state (shape 2464, 256) — do NOT use the full output sequence
- Apply Dropout(0.2) to the final hidden state

### Step 7.4 — Multi-Horizon Output Head
**File**: src/models/output_head.py
**Class**: MultiHorizonHead

**Three independent MLPs (locked, no shared weights)**:
```
Each MLP: Linear(256,128) → ReLU → Dropout(0.2) → Linear(128,64) → ReLU → Linear(64,1)
```

Output: stack the three scalar outputs → tensor (2464, 3). Column 0=3m, Column 1=6m, Column 2=12m.

### Step 7.5 — Full TSPN Assembly
**File**: src/models/tspn.py
**Class**: TSPN

**Forward pass sequence (LOCKED — do not reorder)**:
```
1. For each snapshot q in [0..7]:
   a. node_embed_q, edge_embed_q = feature_embedding(x_q, edge_attr_q)
   b. rep1_q = gat_layer1(node_embed_q, edge_index_q, edge_embed_q)
   c. rep2_q = gat_layer2(rep1_q, edge_index_q, edge_embed_q)
   d. append rep2_q to sequence list
2. seq_tensor = stack(sequence list)        → (8, 2464, 128)
3. temporal = gru(seq_tensor)               → (2464, 256)
4. predictions = output_head(temporal)      → (2464, 3)
5. return predictions, gat_layer1.last_alpha
```

**Critical design rules (locked)**:
- gat_layer1 and gat_layer2 are separate instances — they do NOT share weights
- edge_embed from snapshot 7 (event-time) is used for gat_layer1/2 in all snapshots — the same shock signal propagates through all temporal steps in the attention but the node features vary per snapshot
- output_head receives ONLY the final GRU hidden state, not any intermediate states

### Step 7.6 — Forward Pass Sanity Check (Before Any Training)
Run manually on one event:
- Initialize TSPN from config
- Single forward pass → verify output shape (2464, 3)
- Verify no NaN in output
- Verify gat_layer1.last_alpha is populated with correct size
- Run .backward() on a dummy scalar loss — verify all parameters have gradients (no None gradients)

If any check fails, debug before proceeding to Phase 8.

---

## PHASE 8 — Training Infrastructure
**Duration**: 4 days
**Output**: Complete training loop, evaluation functions, W&B logging

### Step 8.1 — Loss Function
**File**: src/training/losses.py

**Exact formula (locked, no deviations)**:
```
def compute_loss(pred, labels, mask, alpha_weights):
    # pred: (2464, 3), labels: (2464, 3), mask: (2464,)
    pred_masked   = pred[mask]        # only labeled nodes
    labels_masked = labels[mask]
    
    mse_3m  = mean((pred_masked[:,0] - labels_masked[:,0])²)
    mse_6m  = mean((pred_masked[:,1] - labels_masked[:,1])²)
    mse_12m = mean((pred_masked[:,2] - labels_masked[:,2])²)
    l1_attn = mean(abs(alpha_weights))
    
    return 0.50*mse_3m + 0.30*mse_6m + 0.20*mse_12m + 0.01*l1_attn
```

### Step 8.2 — Augmentation Functions
**File**: src/training/augmentation.py

Each augmentation is a function taking a PyG Data object and returning an augmented copy. Never modify in-place.

**augment_shock_magnitude(data, sigma=0.05)**:
- Clone data
- For snapshot 7: add N(0, sigma²) noise to edge_attr[:, 3] where edge_attr[:, 3] != 0
- Entries that were zero remain exactly zero

**augment_temporal_jitter(temporal_sequence)**:
- With 50% probability: randomly choose +1 or −1
- If +1: use snapshots [1..7] + repeat snapshot 7 as snapshot 0
- If −1: use snapshot 0 repeated as new snapshot 0 + snapshots [0..6]
- If 0% triggered: return original sequence unchanged

**augment_edge_dropout(data, p=0.05, threshold=0.002)**:
- Clone data
- Build mask: for edges with import_pen_coeff (edge_attr[:,1]) < threshold, zero entire feature vector with probability p
- Multiply edge_attr by mask (zeroing, not removing)

**augment_label_noise(labels, sigma=0.01)**:
- Add N(0, sigma²) to all non-null label values
- Null values remain null

### Step 8.3 — Training Loop
**File**: src/training/train.py

**Exact training loop structure (locked — implement precisely)**:
```
for fold_idx in range(6):
    held_out = events[fold_idx]
    train_events = [e for e in all_events if e.name != held_out.name]
    
    model = TSPN(config.MODEL)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    best_val_rmse = float('inf')
    patience_counter = 0
    
    for epoch in range(200):
        model.train()
        random.shuffle(train_events)
        epoch_loss = 0
        
        for event in train_events:
            seq = augment_temporal_jitter(event.temporal_sequence)
            seq[7] = augment_shock_magnitude(seq[7])
            seq[7] = augment_edge_dropout(seq[7])
            aug_labels = augment_label_noise(event.y)
            
            pred, alpha = model(seq)
            loss = compute_loss(pred, aug_labels, event.label_mask, alpha)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        
        scheduler.step(epoch)
        
        model.eval()
        with torch.no_grad():
            val_pred, val_alpha = model(held_out.temporal_sequence)
            val_rmse_6m = compute_rmse(val_pred[:,1], held_out.y[:,1], held_out.label_mask)
        
        if epoch % 10 == 0:
            wandb.log({'epoch': epoch, 'train_loss': epoch_loss/len(train_events),
                       'val_rmse_6m': val_rmse_6m, 'lr': scheduler.get_last_lr()[0]})
        
        if val_rmse_6m < best_val_rmse:
            best_val_rmse = val_rmse_6m
            patience_counter = 0
            torch.save(model.state_dict(), f'models/checkpoints/tspn_fold{fold_idx}_best.pt')
            np.save(f'results/tables/attention_fold{fold_idx}.npy', val_alpha.cpu().numpy())
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break
    
    # Evaluation on held-out event
    model.load_state_dict(torch.load(f'models/checkpoints/tspn_fold{fold_idx}_best.pt'))
    model.eval()
    with torch.no_grad():
        final_pred, _ = model(held_out.temporal_sequence)
    record_all_metrics(final_pred, held_out, fold_idx)
```

### Step 8.4 — Evaluation Functions
**File**: src/training/evaluate.py

Implement these functions (all compute over labeled nodes only via mask):
```
compute_rmse(pred, labels, mask) → float
compute_mae(pred, labels, mask) → float
compute_r2(pred, labels, mask) → float
compute_directional_accuracy(pred, labels, mask) → float
bootstrap_ci(pred, labels, mask, n=1000, confidence=0.95) → (lower, upper)
record_all_metrics(pred, event_data, fold_idx) → appends to results/tables/all_results.csv
```

### Phase 8 Exit Criteria
- [ ] Training loop runs through all 6 folds without error on dummy data
- [ ] W&B logging shows loss curves
- [ ] Checkpoints saved correctly for each fold
- [ ] attention .npy files saved for each fold

---

## PHASE 9 — Experiments and Training Runs
**Duration**: 7 days
**Output**: results/tables/all_results.csv complete with all models and ablations

### Step 9.1 — Experiment Run Order (Fixed)
Run in this sequence. Record results before starting the next:

1. Leontief IO (CPU, ~10 minutes)
2. Panel VAR (CPU, ~1 hour)
3. MLP no-graph (GPU, ~1 hour all folds)
4. GCN ablation (replace TSPNGATLayer attention with mean aggregation via a config flag)
5. GAT no-temporal ablation (set seq_len=1 in config, use only snapshot 7)
6. GAT no-shock ablation (zero e[3] for all edges, add total_direct_tariff_exposure as f[9])
7. TSPN 1-layer ablation (remove gat_layer2 call from tspn.py forward pass)
8. TSPN full model (the main result)

### Step 9.2 — Compute Management
- Save checkpoint every 10 epochs to Google Drive
- Name checkpoints: {model_name}_fold{n}_epoch{e}_{timestamp}.pt
- If Colab session resets: reload latest checkpoint, reinitialize optimizer (do not try to reload optimizer state), resume training from that epoch
- Split full TSPN training across 3 Colab sessions if needed (2 folds per session)
- Expected time per fold: ~25 minutes on T4 GPU

### Step 9.3 — Results Table
**File**: results/tables/all_results.csv
**Columns**: model_name, fold, val_event, RMSE_3m, RMSE_6m, RMSE_12m, MAE_3m, MAE_6m, MAE_12m, R2_3m, R2_6m, R2_12m, DirAcc_3m, DirAcc_6m, DirAcc_12m

One row per (model, fold). Then a separate summary sheet with mean ± std per model.

---

## PHASE 10 — Interpretability and Analysis
**Duration**: 5 days
**Output**: All paper figures, interpretability metrics, results/tables/interpretability.csv

### Step 10.1 — Attention vs Leontief Correlation
**File**: src/analysis/interpretability.py

For each fold:
1. Load attention_fold{n}.npy — shape (num_edges, num_heads)
2. Average across heads → shape (num_edges,)
3. Load Leontief inverse for the event year: leontief_{YEAR}.npy
4. Map each edge (src_id, tgt_id) to Leontief entry L[src_id, tgt_id]
5. Filter to edges where L[src_id, tgt_id] > 0
6. Compute Pearson r with scipy.stats.pearsonr → (r, p_value)
7. Compute Spearman ρ with scipy.stats.spearmanr → (rho, p_value)
8. Repeat for Layer 1 and Layer 2 alpha separately

**Generate Figure 3**: scatter plot, x=Leontief coefficient, y=attention weight, with regression line and R² annotation. Two panels (Layer 1, Layer 2). Save as results/figures/fig3_attention_leontief.pdf and .png

### Step 10.2 — Cascade Depth Measurement
**File**: src/analysis/cascade_depth.py

For each held-out event:
1. Load model predictions (all nodes)
2. Build NetworkX directed graph from edge_index of snapshot 7
3. Compute BFS distances from all direct-hit nodes
4. For each hop k in [0, 1, 2, 3, 4, 5]: mean(|pred_delta_6m[nodes_at_hop_k]|)
5. Normalize by mean at hop 0
6. Find k* = first k where normalized value < 0.05

**Generate Figure 4**: line plot, x=hop distance, y=normalized avg |Δp_6m|, one line per event. Save as results/figures/fig4_cascade_depth.pdf and .png

### Step 10.3 — Amplifier Sector Analysis
**File**: src/analysis/amplifier_sectors.py

1. Build attention graph: mean alpha_ij weights averaged across all 6 folds
2. Build trade graph: import_pen_coeff from WIOD 2016
3. Eigenvector centrality on both: networkx.eigenvector_centrality_numpy(G, weight='weight')
4. Amplification ratio per node: attn_centrality / (trade_centrality + 1e-9)
5. Rank descending, report top 20 with country, sector, ratio

**Generate Figure 5**: horizontal bar chart of top 15 sectors by amplification ratio. Save as results/figures/fig5_amplifiers.pdf and .png

### Step 10.4 — All Paper Figures Checklist
Generate and save to results/figures/ as both PDF and PNG:
- fig1_pipeline.pdf — data pipeline diagram (created manually in draw.io or matplotlib)
- fig2_graph.pdf — graph structure for one snapshot (NetworkX + matplotlib, USA nodes only subset)
- fig3_attention_leontief.pdf — scatter plot (two panels)
- fig4_cascade_depth.pdf — attenuation curves per event
- fig5_amplifiers.pdf — amplifier sector bar chart
- fig6_model_comparison.pdf — RMSE grouped bar chart (all models × 3 horizons)
- fig7_ablation.pdf — ablation study results (appendix)

---

## PHASE 11 — Research Paper Writing
**Duration**: 25 days
**Writing order is fixed — do not skip ahead**

| Days | Section | Reason for This Order |
|---|---|---|
| 1–3 | §3 Data and Graph Construction | Most factual, anchors your own understanding |
| 4–7 | §4 Model: TSPN | Formalize what you built with notation |
| 8–12 | §5 Experiments | Write around tables and figures already made |
| 13–15 | §6 Interpretability | Economic narrative flows from results |
| 16–18 | §2 Related Work | Easier once you know your full contribution |
| 19–20 | §1 Introduction | Accurate contribution claims only possible now |
| 21–22 | §7 Conclusion + Abstract | Summarize what was actually found |
| 23–25 | Proofread + references | Consistency, notation check, Zotero citations |

**LaTeX setup**: Overleaf free tier. Use the target journal's author template from Day 1 of writing. Do not write in Word and convert — LaTeX from the start.

---

## PHASE 12 — Product MVP
**Duration**: 14 days
**Output**: Live Streamlit app at Streamlit Community Cloud URL

### Step 12.1 — ONNX Export (Day 1)
**File**: src/models/export_onnx.py

Export the TSPN checkpoint with lowest mean RMSE_6m across all 6 folds.

Settings: opset_version=17, dynamic_axes for variable-batch inference.

Verify ONNX inference latency on CPU: target < 3 seconds for one scenario. Profile and optimize if needed.

Save as: models/onnx/tspn_best.onnx

### Step 12.2 — App Build Order (One Feature Per Day, Test Before Next)

**Day 2** — Scenario Builder sidebar:
Three widgets: country multiselect (source), sector select, tariff magnitude slider (0–50%, step 1%). Historical event dropdown that pre-fills widgets. Submit button calls inference wrapper.

**Day 3** — Price Table:
st.dataframe with columns: Country, Sector, Δp(3m)%, Δp(6m)%, Δp(12m)%, Hop Distance, Risk Level. Color-code Risk Level (High/Medium/Low/Negligible). Default sort by |Δp(6m)| descending. Filter widgets for risk level and country.

**Days 4–5** — Graph Visualization (PyVis):
PyVis network, centered on shocked nodes. Node size proportional to |Δp(6m)|, min=5px, max=30px. Node color: red scale for positive Δp, blue for negative, gray for negligible. Edge opacity proportional to import_pen_coeff. Show only nodes with |Δp(6m)| > 0.5%. Hop-distance filter slider.

**Day 6** — Risk Dashboard:
Three Plotly charts: (1) top 10 impacted sectors horizontal bar, (2) world choropleth by national impact score, (3) donut chart manufacturing vs services vs primary.

**Day 7** — Scenario Comparison:
Two-column layout. Left = Scenario A, Right = Scenario B. Below: side-by-side comparison table and aggregate impact scores.

**Day 8** — Historical Event Library:
Dropdown of 6 pre-loaded events. For each: scatter plot of model prediction vs actual BLS PPI change (for US sectors). Model accuracy badge.

**Days 9–10** — Integration Testing:
Test all features end-to-end. Test with edge inputs (0% tariff, 50% tariff). Verify no crashes. Optimize PyVis rendering — cap at 100 nodes on initial render, "show all" button expands.

**Days 11–12** — Polish:
Loading spinners. Error handling for invalid inputs. App title, short methodology description, link to research paper. Add CSV download button to price table.

**Days 13–14** — Deployment:
Push to public GitHub repo. Connect Streamlit Community Cloud at share.streamlit.io. Add requirements.txt. Deploy. Test from separate machine.

---

## PHASE 13 — Ongoing Maintenance (1 hour/day during paper writing)
- Fix app bugs as discovered
- Write README.md with project overview and paper link
- Add "How it works" non-technical explanation page to app
- Once paper is submitted: update app to show paper abstract and link

---

## CRITICAL RULES — NEVER BREAK THESE

1. **config.py is the only source of truth.** No hardcoded values outside it. No exceptions.

2. **Node feature vector order is immutable.** normalization_stats.json is indexed by position. Changing the order corrupts inference silently.

3. **Edge feature vector order is immutable.** The shock signal is ALWAYS e[3]. Any reordering breaks the shock injection mechanism silently.

4. **Never apply augmentation during validation.** Set model.eval() and explicitly disable augmentation. Augmented validation inflates RMSE variance and corrupts early stopping.

5. **Never train on the held-out event.** Verify the LOEO split logic before every training run. A single contaminated fold invalidates all results.

6. **Never recompute Leontief inverses unless edges change.** They are expensive. Cache them. If you change the edge threshold (you should not), regenerate from Phase 2 forward.

7. **Never modify the deployed ONNX model file after deployment.** Version new models: tspn_v2.onnx. Test in staging before switching the live app.

8. **Record the calibrated LEONTIEF_PASS_THROUGH_RATE in config.py immediately after calibration.** If this value is lost, the Leontief baseline numbers cannot be reproduced, breaking the comparison.
