# TSPN — Complete Implementation Guide
### Tariff Shock Propagation Network: From Raw Data to Scalable Product

> **Scope of this document**: Everything you need to go from zero to a working model and a deployable product. No code yet — pure planning, architecture, and strategy. Every data source is free. All compute runs on free-tier cloud. Complexity is a real step up from ecoacoustic work but entirely achievable solo.

---

## SECTION 0 — What You Are Building

You are building two things simultaneously. They share 90% of the same work.

**Track A — Research Paper**: A temporal Graph Neural Network (TSPN) that predicts sector-level price changes following a tariff shock, trained on historical trade war events, with interpretability analysis comparing learned attention weights to the Leontief economic inverse. Target: Journal of International Economics or Nature Computational Science.

**Track B — Product**: A Tariff Risk Intelligence Dashboard where any user can input a tariff scenario (which country, which product, what rate change), see the predicted cascade through the supply chain graph, get sector-level price change forecasts at 3/6/12 months, and download a risk report. Free to host. Useful to trade analysts, procurement teams, policy researchers, and journalists covering trade.

The graph, model, and training pipeline are identical for both tracks. The only difference is Track B adds a frontend, an API layer, and a scenario builder on top of the trained model. Build Track A first. Track B is 3–4 additional weeks after that.

---

## SECTION 1 — Project Architecture (Big Picture)

```
RAW DATA LAYER
├── WIOD (annual IO tables)
├── UN Comtrade (bilateral trade flows)
├── WITS / WTO (tariff schedules)
├── BLS PPI + Eurostat PPI (validation labels)
└── Federal Register / EU Official Journal (tariff event product lists)
         ↓
DATA PROCESSING LAYER
├── WIOD Parser → node and edge tables
├── Tariff Mapper → HS6 → ISIC concordance → sector-level tariff rates
├── Graph Builder → PyG Data objects (one per year, per event)
├── Feature Engineer → 9-dim node vectors, 6-dim edge vectors
└── Label Generator → 3m/6m/12m PPI change per sector per event
         ↓
MODEL LAYER
├── Feature Embedding (linear projection)
├── GAT Layer 1 (1-hop attention, direct supplier effects)
├── GAT Layer 2 (2-hop attention, indirect supplier effects)
├── GRU Temporal Module (8-quarter sequence, lagged dynamics)
└── Multi-horizon MLP Head (predict Δp at 3m, 6m, 12m)
         ↓
EVALUATION LAYER
├── LOEO-CV (Leave-One-Event-Out)
├── Baseline comparison (Leontief, VAR, ablations)
├── Interpretability (attention vs Leontief correlation)
└── Cascade depth + amplifier sector analysis
         ↓
PRODUCT LAYER
├── FastAPI backend (model inference + scenario API)
├── Streamlit frontend (graph viz + dashboard)
├── SQLite / Supabase (scenario history, cached results)
└── Hugging Face Spaces deployment (free hosting)
```

Every layer feeds into the next. You build them in order. You do not touch the product layer until the model layer is solid.

---

## SECTION 2 — Complete Dataset Guide

### 2.1 World Input-Output Database (WIOD)

**What it is**: The backbone of the entire project. Annual inter-sector, inter-country trade flow matrices showing exactly how much each sector in each country buys intermediate goods from every other sector in every other country. This is what builds your graph.

**Coverage**: 43 countries + Rest of World aggregate, 56 ISIC Rev. 4 sectors, years 2000–2016.

**How to get it (completely free)**:
- Go to: `https://www.rug.nl/ggdc/valuechain/wiod/`
- Click "Download" tab
- Under "WIOT Tables": download all 17 Excel files (one per year, 2000–2016) OR the R-data format if you prefer
- Under "Socioeconomic Accounts": download the single Excel file containing gross output, value added, employment per (country, sector, year) — you need this for node features
- No registration required. Direct download.

**File structure**: Each annual Excel file contains one large matrix. Rows = supplying (country, sector) pairs. Columns = purchasing (country, sector) pairs + final demand columns. Cell value = USD millions of intermediate goods purchased. Size per file: approximately 2,500 rows × 2,500 columns.

**Volume**: All 17 files combined = approximately 400MB. Manageable on any machine.

**Sectors list (56 ISIC Rev. 4 sectors)**:
- A01: Crop and animal production
- A02: Forestry and logging
- A03: Fishing and aquaculture
- B: Mining and quarrying
- C10-C12: Food/beverages/tobacco
- C13-C15: Textiles, apparel, leather
- C16: Wood and cork products
- C17: Paper and paper products
- C18: Printing and media
- C19: Coke and refined petroleum
- C20: Chemicals
- C21: Pharmaceuticals
- C22: Rubber and plastics
- C23: Non-metallic mineral products
- C24: Basic metals (this is where steel lives)
- C25: Fabricated metal products
- C26: Computer and electronic products
- C27: Electrical equipment
- C28: Machinery
- C29: Motor vehicles (downstream of steel)
- C30: Other transport equipment
- C31-C32: Furniture and other manufacturing
- C33: Repair and installation
- D35: Electricity, gas, steam
- E36: Water supply
- E37-E39: Waste management
- F: Construction
- G45: Motor vehicle trade
- G46: Wholesale trade
- G47: Retail trade
- H49: Land transport
- H50: Water transport
- H51: Air transport
- H52: Warehousing
- H53: Postal activities
- I: Accommodation and food
- J58-J60: Media and broadcasting
- J61: Telecommunications
- J62-J63: IT and information services
- K64: Financial services
- K65: Insurance
- K66: Financial auxiliaries
- L68: Real estate
- M69-M70: Legal and management consulting
- M71: Architecture and engineering
- M72: Research and development
- M73: Advertising
- M74-M75: Other professional services
- N: Administrative services
- O84: Public administration
- P85: Education
- Q: Health and social work
- R-S: Arts, entertainment, other services
- T: Household activities
- U: International organizations

**Countries list (43 + RoW)**: AUS, AUT, BEL, BGR, BRA, CAN, CHN, CYP, CZE, DEU, DNK, ESP, EST, FIN, FRA, GBR, GRC, HUN, IDN, IND, IRL, ITA, JPN, KOR, LTU, LUX, LVA, MEX, MLT, NLD, NOR, POL, PRT, ROU, RUS, SVK, SVN, SWE, TUR, TWN, USA, RoW.

---

### 2.2 UN Comtrade

**What it is**: Bilateral merchandise trade flows at HS6 commodity level, for all country pairs, annually. You use this to extend the WIOD graph from 2016 to 2021, covering the period when all the major tariff events actually happened.

**How to get it (free)**:
- Register free at: `https://comtradeplus.un.org/`
- After registration: go to "Get Data" → use the API
- Python package: `pip install comtradeapicall`
- Free tier limits: 500 API calls per hour, 250 rows per call (use pagination)
- Alternatively: use the bulk download at `https://comtradeplus.un.org/TradeFlow` — select year, reporter, trade type → downloads as CSV

**What to pull**:
- Reporter: all 43 WIOD countries (make one API call per reporter per year)
- Partner: all countries
- Trade flow: imports (code M)
- Classification: HS (use HS2017 revision)
- Period: 2017, 2018, 2019, 2020, 2021
- Commodity code: TOTAL first to get aggregates, then pull HS2-digit level for each sector mapping

**Volume**: Approximately 5–10 million rows total for all countries × years. Store as Parquet files (much smaller than CSV). Expect ~2–3 GB raw, ~500MB compressed Parquet.

**Smart approach to stay within free limits**: Pull at HS2-digit level (97 codes) rather than HS6. This gives you enough granularity for the ISIC concordance and reduces API calls by ~100x. Only go to HS6 level for the specific tariff-affected product codes where precision matters.

---

### 2.3 WITS / WTO Tariff Data

**What it is**: Applied MFN (Most Favoured Nation) tariff rates at HS6 product level for all WTO members. This gives you the actual tariff rates on every trade flow, which you convert to sector-level weighted average tariffs.

**Two sources (both free)**:

**Source A — World Bank WITS (better for bulk historical data)**:
- URL: `https://wits.worldbank.org/`
- Go to: Data → Trade Outcomes → Tariff → Download bulk data
- TRAINS database: Applied tariff rates, MFN and preferential
- Download: "Tariff Simple Average" by reporter × product × year (CSV format)
- Register free, then bulk download is available
- Coverage: 2000–2021 for most WTO members

**Source B — WTO Tariff Download Facility (better for precise post-2018 data)**:
- URL: `https://tariffdata.wto.org/`
- Download: by member, by HS revision, as CSV
- More current than WITS for the critical 2018–2021 period
- No registration required for basic downloads

**What you actually need**:
- US applied MFN tariffs for all HS6 codes, 2015–2021 (captures pre and post Section 232 + 301)
- China applied MFN tariffs, 2015–2021 (for US-China bilateral)
- EU applied MFN tariffs, 2015–2021 (for retaliatory tariffs)
- Canada applied MFN tariffs, 2017–2019 (for steel/aluminum retaliation)

**Important**: For the US-China Section 301 tariffs, the rates are applied on top of MFN to China-origin goods only. WITS records these as preferential (in reverse — punitive) rates. Pull both MFN and "additional duties" separately.

**Specific tariff event files to download from Federal Register and EU Official Journal**:
- US Section 232 Steel: Federal Register Vol. 83 No. 45 (March 8, 2018) — lists every affected HTS code
- US Section 232 Aluminum: same Federal Register issue
- US Section 301 List 1: Federal Register Vol. 83 No. 119 (June 20, 2018)
- US Section 301 List 2: Federal Register Vol. 83 No. 155 (August 10, 2018)
- US Section 301 List 3: Federal Register Vol. 83 No. 185 (September 21, 2018)
- EU Retaliation: EU Official Journal L 160/1 (June 25, 2018)
- Canada Retaliation: Canada Gazette Part II Vol. 152 No. 14 (July 11, 2018)
- UK Global Tariff: `https://www.gov.uk/guidance/uk-tariffs-from-1-january-2021`

All Federal Register documents are freely accessible at `federalregister.gov`. Each document contains the complete list of affected HTS codes.

---

### 2.4 BLS Producer Price Indices (Validation Labels — US sectors)

**What it is**: Monthly producer price indices for US industries by NAICS code. This is your primary validation label — you measure whether the model correctly predicted how much prices changed after each tariff event.

**How to get it (free)**:
- URL: `https://www.bls.gov/ppi/data.htm`
- Click "Industry Data" → "Industry Multi-Screen Data Search"
- Filter by: Industry NAICS code, "All items"
- Download: all series as CSV from 2000 to 2023
- Alternatively: use BLS Public Data API (`https://api.bls.gov/publicAPI/v2/`) — free registration gives 500 queries/day, 50 series per query
- Key series prefix: `PCU` for Industry PPIs (e.g., PCU331111331111 = iron and steel mills)

**NAICS → ISIC crosswalk**: Download from UN Statistics Division at `https://unstats.un.org/unsd/classifications/Econ/` — look for "ISIC Rev. 4 ↔ NAICS 2017" correspondence table. It's a direct download, no registration.

**What to pull**: All 3-digit NAICS manufacturing sectors and major service sectors. Pull monthly data 2016–2022 to capture pre-shock baseline and full post-shock adjustment period.

---

### 2.5 Eurostat PPI (Validation Labels — EU sectors)

**What it is**: Monthly producer price indices for EU member state industries, by NACE sector code (which aligns closely with ISIC Rev. 4).

**How to get it (free)**:
- URL: `https://ec.europa.eu/eurostat/web/main/data/database`
- Navigate to: Industry, Trade and Services → Short-term business statistics → Industry → Producer prices in industry
- Table code: `sts_inppd_m` (monthly PPI by NACE 2-digit)
- Direct API: `https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_inppd_m`
- Also useful: `prc_ppi_inw` for industry-level PPI weights

**Coverage**: All EU27 countries + Norway + UK, NACE 2-digit sectors, monthly 2000–present.

---

### 2.6 World Bank Commodity Price Data (Validation for Primary Sectors)

**What it is**: Monthly commodity price indices for 70+ primary commodities — steel, aluminum, copper, oil, agricultural goods. Used for validating predictions in sectors where BLS/Eurostat coverage is limited.

**How to get it (free)**:
- URL: `https://www.worldbank.org/en/research/commodity-markets`
- Pink Sheet download: monthly prices as Excel, no registration
- API: World Bank Data API `https://api.worldbank.org/v2/` — completely free, no key

**Key commodity series to use**:
- Steel (HRC): PSTEELHRCE
- Aluminum: PALUM
- Copper: PCOPP
- Iron ore: PIORECR
- Wheat, corn, soybeans (for Section 301 retaliation effects on agriculture)

---

### 2.7 HS6 → ISIC Concordance

**What it is**: The concordance table that maps HS (Harmonized System) commodity codes to ISIC (International Standard Industrial Classification) sectors. This is the bridge between tariff data (which uses HS codes) and your graph (which uses ISIC sectors).

**How to get it (free)**:
- URL: `https://unstats.un.org/unsd/trade/classifications/correspondence-tables.asp`
- Download: "HS 2017 → ISIC Rev. 4" correspondence table (Excel)
- Also useful: "HS 2012 → ISIC Rev. 4" (for older trade data pre-2017 revision)
- File is a simple two-column table: HS6 code → ISIC code(s)

**Critical note**: This is a many-to-many mapping. HS code 7208.10 (flat-rolled iron/steel) maps to ISIC C24 (basic metals). But some HS codes span multiple ISIC sectors. When this happens, split the trade value proportionally using WIOD sector ratios as prior weights.

---

### 2.8 Data You Do NOT Need to Pay For

Everything listed above is free. Here is what you explicitly do not need to buy:

- Bloomberg Terminal: not needed. BLS and Eurostat cover your validation needs.
- Refinitiv / LSEG data: not needed.
- GTAP database (paid CGE model data): you are building an alternative to this.
- S&P Global Trade Intelligence: not needed. Comtrade covers trade flows.
- Panjiva / ImportGenius: not needed. Aggregate sector-level data is enough.

---

## SECTION 3 — Data Processing Pipeline (Step by Step)

### Stage 1: WIOD Raw → Structured Edge Table

**Input**: 17 Excel files (WIOT2000_October16_ROW.xlsx through WIOT2016_October16_ROW.xlsx)

**Process**:
Each Excel file has a specific layout. The first few rows/columns are metadata. The actual IO matrix starts at a fixed offset (approximately row 6, column 5 in the Excel). You parse this into a long-format table with columns: year, supplier_country, supplier_sector, buyer_country, buyer_sector, flow_usd_millions.

You apply a minimum threshold filter: keep only edges where the import penetration coefficient (flow / buyer total input) exceeds 0.001. This eliminates negligible trade relationships and keeps your graph sparse and meaningful. After filtering, expect approximately 80,000–120,000 directed edges per year (down from the theoretical 2,464 × 2,464 = 6 million possible pairs).

**Output**: One Parquet file per year, ~10MB each. Columns: year, src_country, src_sector, tgt_country, tgt_sector, flow_usd_millions, import_pen_coeff.

### Stage 2: Comtrade Raw → Sector-Level Flows (2017–2021)

**Input**: Comtrade CSV/Parquet files by reporter × year

**Process**:
For each bilateral pair (reporter, partner) and each year 2017–2021:
1. Pull all HS2-digit level trade flows
2. Apply the HS → ISIC concordance to map each HS2 subtotal to its ISIC sector
3. Sum trade values within each ISIC sector
4. Compute import penetration coefficients using WIOD 2016 as the denominator baseline (since Comtrade doesn't give gross output — use the last WIOD year as a structural prior and update only the trade flows)

This is an approximation. The assumption is that the sector structure (technical coefficients) changed slowly between 2016 and 2021, but bilateral trade flows shifted due to tariffs. This is a defensible assumption and one you explicitly discuss as a limitation in the paper.

**Output**: One Parquet file per year (2017–2021) with the same schema as Stage 1 output. ~10MB each.

### Stage 3: Tariff Rate Computation

**Input**: WITS bulk download (MFN rates by reporter × HS6 × year), Federal Register event product code lists

**Process**:
For each bilateral pair (exporter, importer) and each ISIC sector and each year:
1. Look up all HS6 codes that map to this ISIC sector (from concordance)
2. For each HS6 code, get the applied tariff rate charged by importer on goods from exporter
3. Compute the weighted average: weight each HS6 rate by the Comtrade trade value for that HS6 in that bilateral pair in that year
4. Result: tariff_rate[exporter][importer][isic_sector][year]

For tariff events specifically, you also compute the tariff delta:
- Pre-event rate: rate in the last quarter before the event date
- Post-event rate: rate effective after the event announcement
- Delta: post - pre (can be 0 for unaffected pairs, positive for tariff increases)

**Output**: tariff_rates.parquet — columns: year, exporter, importer, isic_sector, applied_tariff_rate. And separately, shock_vectors.parquet — one row per (event_name, exporter, importer, isic_sector) with delta_tariff value.

### Stage 4: Node Feature Engineering

**Input**: WIOD socioeconomic accounts (gross output, value added, employment per country-sector-year), PPI data (BLS + Eurostat), Edge table (for computing network statistics)

**Process**:
For each (country, sector, year) combination, compute:

**Feature 1 — Log Gross Output**: Natural log of annual gross output in USD millions. Normalized across all nodes. Captures economic size of the sector.

**Feature 2 — Import Penetration Ratio**: Sum of all imports into this (country, sector) / (gross output + total imports - total exports). Ranges 0 to 1. High values mean this sector is heavily dependent on foreign inputs — highly exposed to supply disruptions.

**Feature 3 — Export Intensity**: Total exports from this sector / gross output. Captures how trade-facing (and therefore tariff-sensitive) this sector is on the output side.

**Feature 4 — Backward Linkage Strength**: Computed from the Leontief inverse matrix (I - A)^{-1} for that year. The backward linkage of sector j = sum of the j-th column of the Leontief inverse. High backward linkage means this sector requires many inputs from throughout the economy. This is your economic baseline signal — and later, comparing it with the model's learned attention is the core interpretability finding.

**Feature 5 — Weighted Tariff Exposure**: For each (country, sector), compute the trade-value-weighted average applied tariff rate across all its import partners and all HS6 codes within the sector. This captures current trade policy exposure even before a shock.

**Features 6–9 — Lagged PPI Changes**: The price change in this sector in each of the 4 preceding annual periods. This gives the model the price dynamics context: is this sector already experiencing inflation, or is it in a deflationary trend? Lagged PPI is one of the strongest features because price changes are autocorrelated.

**Output**: node_features.parquet — columns: year, country, sector, node_id, f1 through f9.

### Stage 5: Edge Feature Engineering

**Input**: Edge table (flows), Tariff rates, Shock vectors, Comtrade HHI computation

**Process**:
For each directed edge (supplier → buyer) in each temporal snapshot, compute:

**Feature 1 — Log Trade Flow**: Natural log of the bilateral flow value in USD millions. Log transform because trade flows are highly skewed.

**Feature 2 — Import Penetration Coefficient**: The edge weight from Stage 1 — what fraction of the buyer's total inputs come from this specific supplier. This is the core economic transmission weight. High coefficient = buyer strongly depends on this supplier = shock at supplier propagates strongly to buyer.

**Feature 3 — Applied Tariff Rate**: The current tariff rate on this bilateral (exporter, importer, sector) pair. Captures the existing trade policy framing between these two nodes.

**Feature 4 — Tariff Delta (SHOCK SIGNAL)**: The change in tariff rate due to an event. Zero for all non-affected edges. Non-zero only for edges directly hit by the tariff event. This is how you inject the shock into the graph. The model learns to propagate this signal through the graph weighted by the attention mechanism.

**Feature 5 — Product Concentration (HHI)**: Herfindahl-Hirschman Index of trade concentration — how concentrated is the trade flow between this pair across HS6 product codes? Low HHI = diversified product mix = more substitutable = shock may attenuate faster. High HHI = concentrated in few products = more vulnerable.

**Feature 6 — Domestic Indicator**: Binary 1/0 for whether supplier and buyer are in the same country. Domestic supply relationships behave differently from international trade flows and should be modeled separately.

**Output**: edge_features.parquet — columns: year, event_name, src_id, tgt_id, f1 through f6.

### Stage 6: Label Generation

**Input**: BLS PPI (monthly), Eurostat PPI (monthly), World Bank commodity prices, tariff event dates

**Process**:
For each tariff event e occurring at date d_e, and for each labeled (country, sector) node:
1. Get PPI value at d_e (event month)
2. Get PPI values at d_e + 3 months, d_e + 6 months, d_e + 12 months
3. Compute percentage change: (PPI[d_e + k] - PPI[d_e]) / PPI[d_e] for k = 3, 6, 12 months

**Coverage gaps**: BLS covers US sectors (NAICS mapped to ISIC). Eurostat covers EU sectors. For non-US/EU countries and for sectors without direct PPI coverage, you have two options:
- Option A: Only compute labels for covered (country, sector) pairs and train on those only (recommended for the research paper — be explicit about this limitation)
- Option B: Use commodity price proxies for manufacturing sectors, and mark them as lower-confidence labels

For the initial build, use Option A. This gives you approximately 60–70% of the 2,464 nodes with reliable labels. That is enough to train and validate the model.

**Output**: labels.parquet — columns: event_name, country, sector, node_id, delta_ppi_3m, delta_ppi_6m, delta_ppi_12m, label_source (BLS/Eurostat/commodity_proxy).

### Stage 7: PyG Graph Dataset Assembly

**Input**: All parquet files from Stages 1–6

**Process**:
For each (event, temporal_snapshot_sequence) pair:
1. Build PyG `Data` object with node feature matrix X (|V| × 9), edge index (2 × |E|), edge feature matrix E (|E| × 6)
2. Build the temporal sequence: 8 quarterly snapshots ending at the event date
3. Attach labels y (|V| × 3) for the three prediction horizons
4. Attach a mask for nodes that have valid labels
5. Save each event as a separate PyG Data file

You will have 6 Data objects (one per tariff event). Each Data object contains the 8-step temporal sequence (list of 8 graphs) plus labels.

**Final dataset size**: Approximately 200–400MB for all 6 events with all feature matrices. Fits comfortably in Google Colab RAM.

---

## SECTION 4 — Graph Construction (Complete Specification)

### 4.1 Node Space

**Definition**: A node is a unique (country, sector) pair.
**Count**: 44 countries × 56 sectors = 2,464 nodes.
**Node IDs**: Integer 0 to 2,463. Mapping: node_id = country_index × 56 + sector_index.
**Representation**: Each node carries a 9-dimensional feature vector (as described in Stage 4 above).

Key nodes to keep track of for analysis:
- (USA, C24) — US basic metals (directly hit by Section 232)
- (CHN, C26) — Chinese computer/electronics (directly hit by Section 301)
- (USA, C29) — US motor vehicles (downstream of steel, indirect hit)
- (DEU, C29) — German motor vehicles (exposed via EU retaliatory context)
- (USA, C20) — US chemicals (exposed through multiple Section 301 lists)

### 4.2 Edge Space

**Definition**: A directed edge from node i to node j means "sector i supplies intermediate inputs to sector j."

**Direction**: i → j means i is the supplier, j is the buyer. A tariff shock at the edge hits the buyer first (their import costs rise), then propagates further downstream as the buyer's output prices rise.

**Edge creation rule**: Include edge (i, j) if the import penetration coefficient import_pen_ij ≥ 0.001 (0.1%). This threshold is a design decision — you ablate it at 0.0005 and 0.002 to test sensitivity.

**Expected edge count after filtering**: 80,000–130,000 directed edges. Average out-degree per node: ~40–50 edges. Graph is sparse but well-connected, as expected for a real trade network.

**Self-loops**: Include domestic within-sector flows (same country, same sector, different production stages). These represent within-sector supply relationships and are present in WIOD.

**Multi-layer view**: The graph can be viewed as two overlapping layers:
- Domestic layer: edges where src_country == tgt_country (within-country supply chains)
- International layer: edges where src_country != tgt_country (cross-border trade flows)

Tariff shocks only affect the international layer edges. This separation is one reason why injecting shocks into edge features (Feature 4) rather than node features is the right design: it precisely targets the affected layer.

### 4.3 Temporal Graph Sequence

**Structure**: 8 quarterly snapshots ending at the event date.

Why 8 quarters: This covers 2 full years of history before the shock. Trade relationships and price dynamics have meaningful annual and semi-annual cycles. The GRU needs enough history to learn these patterns.

**Quarterly interpolation**: WIOD is annual, so you interpolate quarterly snapshots between annual IO tables. Linear interpolation of trade flows between adjacent annual snapshots is sufficient. More sophisticated: use quarterly Comtrade bilateral flows (available from Comtrade) to scale the annual WIOD structure quarterly.

**Snapshot graph for a given quarter q**: Uses the WIOD/Comtrade flows from the surrounding annual period, the WITS tariff rates active in that quarter, and PPI values for node features in that quarter. The result is a consistent graph snapshot with all features populated.

### 4.4 Shock Vector Construction

A shock vector for tariff event e is a sparse mapping: (src_node, tgt_node) → delta_tariff_rate.

Most entries are zero. Non-zero entries correspond to edges where the tariff actually changed.

**Example — US Section 232 Steel (March 8, 2018)**:
- Identify all HTS codes in the steel product list from the Federal Register notice
- Map those HTS codes to ISIC sector C24 (basic metals) using the concordance
- For all (country, USA, C24) edges — i.e., all countries supplying steel to USA — set delta_tariff = +0.25
- Country exemptions: Canada, Mexico, EU countries initially had country-specific treatment (later removed or modified). Reflect this in the shock vector.
- All other edges: delta_tariff = 0.0

The shock vector is then injected into Feature 4 (tariff delta) of the edge feature matrix for the event-time graph snapshot.

**Cascade expectation from this shock**:
- Direct hit: all (*, USA, C24) edges — steel importers in USA
- Hop 1 downstream: (USA, C24) → (USA, C25) fabricated metals, (USA, C29) motor vehicles, (USA, C28) machinery, (USA, F) construction
- Hop 2 downstream: (USA, C29) → (USA, G45) motor vehicle trade, (USA, H49) land transport
- Hop 3+: further downstream, rapidly attenuating

---

## SECTION 5 — Model Architecture (Complete Specification)

### 5.1 Design Principles

**Why GAT over GCN**: Graph Convolutional Networks (GCN) use mean or degree-normalized aggregation — every neighbor is weighted equally. Graph Attention Networks (GAT) learn which neighbors matter more. In supply chain terms, a node has many input suppliers, but only some are critical. The attention weight α_ij should learn to approximate the economic importance of supplier i for buyer j — which is exactly what the Leontief coefficient encodes analytically. Using GAT makes this interpretability comparison possible. Using GCN would collapse this signal.

**Why edge features in the attention computation**: Standard GAT computes attention from node features only: α_ij = f(h_i, h_j). This TSPN includes edge features in the attention: α_ij = f(h_i, h_j, e_ij). Including e_ij means the shock signal (Feature 4: tariff delta) directly modulates the attention weight. When a tariff hits edge (i, j), the attention weight for that edge increases, causing the shock to propagate more strongly along that path. This is the key mechanism that makes the shock injection work.

**Why GRU over Transformer for the temporal module**: Transformers have better long-range attention, but here the sequence length is only 8 and the data volume is small. GRU is lighter, less prone to overfitting on small datasets, and already well-integrated with PyG via the `GRUConv` or custom GRU on top of GAT. Save Transformers for a future version with higher-frequency data.

**Why multi-horizon output (3 separate heads)**: The economic literature on tariff pass-through (Amiti et al. 2019) shows that short-run and long-run price incidence differ — in the short run, importers absorb some of the cost; in the long run, more of it passes to domestic prices. Three separate MLP heads allow the model to learn different temporal response patterns per horizon rather than forcing them to share a single pattern.

### 5.2 Layer-by-Layer Specification

**Layer 0 — Feature Embedding**:
- Node: Linear(9 → 128) + ReLU + Dropout(0.1)
- Edge: Linear(6 → 64) + ReLU + Dropout(0.1)
- Normalization: BatchNorm on node embeddings after projection
- Output dimensions: node_embed ∈ R^(|V| × 128), edge_embed ∈ R^(|E| × 64)

**Layer 1 — GAT Layer 1 (1-hop, direct supplier effects)**:
- Heads: K = 4
- Head dimension: 32 (so concatenated output = 128)
- Attention computation per head k: score_ij = LeakyReLU(a_k^T [W_k · h_i || W_k · h_j || W_e · e_ij])
- Normalization: softmax over j in N(i) for each source node i
- Aggregation: h_i' = concat over k of ELU(Σ_{j∈N(i)} α_ij^k · W_k · h_j)
- Residual connection: h_i' = h_i' + Linear(128, 128)(h_i) [residual from input]
- Output: R^(|V| × 128)
- Dropout on attention weights: 0.3

**Layer 2 — GAT Layer 2 (2-hop, indirect supplier effects)**:
- Same architecture as Layer 1
- Input is h_i' from Layer 1 — now each node encodes information from its 1-hop neighborhood
- After Layer 2, each node encodes information from its 2-hop neighborhood (suppliers-of-suppliers)
- Separate set of weight matrices (not shared with Layer 1)
- Output: R^(|V| × 128)
- Key ablation: run model with 1 layer only and 3 layers only; compare to 2-layer version

**Layer 3 — GRU Temporal Module**:
- Input: sequence [h_i(t-7), h_i(t-6), ..., h_i(t)] from GAT Layer 2, for each node i
- Sequence length: 8 quarterly snapshots
- GRU: input_size=128, hidden_size=256, num_layers=1
- Applied independently per node: the GRU processes the 8-step sequence of node representations
- Implementation: batch all 2,464 nodes and process through GRU together for efficiency
- Output: h_temporal_i ∈ R^256 per node — the final hidden state after processing the full sequence

**Layer 4 — Multi-Horizon MLP Output Heads**:
- Three independent MLPs, one per prediction horizon
- Architecture of each: Linear(256 → 128) + ReLU + Dropout(0.2) + Linear(128 → 64) + ReLU + Linear(64 → 1)
- Applied per node to produce scalar price change prediction
- Output per horizon: Δp̂ ∈ R^(|V|) (one prediction per node)
- Applied only to nodes with valid labels during training (using the label mask)

### 5.3 Loss Function

**Multi-horizon weighted MSE plus attention sparsity regularization**:

The loss is a weighted sum of MSE losses across three horizons, plus an L1 penalty on attention weights to encourage sparse (focused) transmission rather than diffuse spreading.

Horizon weights: 0.50 for 3m, 0.30 for 6m, 0.20 for 12m. The 3m horizon gets more weight because there is less noise in the 3-month PPI measurement (less confounding from other economic factors), giving cleaner training signal.

Attention L1 weight: 0.01. Small enough not to dominate but sufficient to discourage the model from attending to economically irrelevant edges.

Label mask: Only nodes with valid PPI labels contribute to the loss. Unlabeled nodes still receive forward-pass computation (message passing uses them) but their predicted values are excluded from the loss computation.

---

## SECTION 6 — Training Strategy

### 6.1 Leave-One-Event-Out Cross-Validation (LOEO-CV)

This is the correct validation protocol when you have few but large events.

A standard 80/20 train-test split would mean either all training or all testing events are a subset of one shock episode, leading to highly correlated evaluation — a model that overfits to one tariff type (e.g., US-China bilateral) might score well but generalize poorly to a different tariff structure (e.g., UK global tariff restructuring).

LOEO-CV ensures each fold tests on a structurally different event from what was trained on. You run 6 training experiments, each leaving one event as the held-out validation set. Final metric = mean ± standard deviation across 6 folds.

For the research paper, this directly answers: "Does the model generalize across tariff events of different structures, magnitudes, and geographies?"

### 6.2 Training Data Augmentation

With only 6 training events per fold (5 train, 1 test), augmentation is critical. The following augmentations are applied during training only, not during validation:

**Shock magnitude perturbation**: Add Gaussian noise with σ=5 percentage points to each shock magnitude (e.g., a 25% tariff becomes 23%–27% in different augmented copies). Simulates uncertainty in implementation and announcement effects.

**Temporal jitter**: Randomly shift the event date by ±1 quarter. Simulates the gap between tariff announcement and implementation, which varies across events.

**Edge dropout**: Randomly zero out 5% of low-weight edges (import penetration < 0.002) in each training step. Simulates measurement noise in trade flow data and improves robustness.

**Label noise injection**: Add small Gaussian noise (σ=0.01) to PPI labels during training. Standard regularization technique for regression tasks.

### 6.3 Optimizer and Schedule

- Optimizer: Adam with learning rate 1e-3, weight decay 1e-4
- Learning rate schedule: Cosine annealing with warm restarts (T_0=50 epochs, T_mult=2)
- Gradient clipping: max norm = 1.0 (important for GRU stability)
- Early stopping: patience = 20 epochs on validation RMSE
- Training budget: 200 epochs maximum
- Expected wall time on Google Colab T4 GPU: 30–60 minutes per fold

### 6.4 Compute Strategy (Free)

**Google Colab Free Tier**: T4 GPU, 12GB VRAM, 12-hour session limit. Enough for this project.
- Save model checkpoints to Google Drive every 10 epochs to survive session interruptions
- The graph (2,464 nodes, ~100K edges) fits entirely in GPU VRAM
- Full training run (6 folds × 200 epochs) can be done in 3–4 Colab sessions

**Google Colab Pro ($10/month)**: If free tier sessions keep interrupting. Gives you 24-hour sessions and priority GPU access. Optional but helpful.

**Local machine**: If you have any GPU (even a 4GB VRAM card), this runs fine. If CPU-only, expect 3–5x longer training time — still feasible.

---

## SECTION 7 — Baselines and Ablations

### 7.1 External Baselines (What You Compare the Full TSPN Against)

**Baseline 1 — Static Leontief IO Model**:
The standard economics approach. Compute the Leontief inverse matrix L = (I - A)^{-1} from the WIOD technical coefficients matrix A. For a shock vector δτ (sector-level tariff increases), the predicted price changes are: Δp = L^T · δτ · pass_through_rate. The pass_through_rate is the fraction of tariff cost assumed to pass to prices — calibrate this from the pre-2018 historical data. This is your primary economics benchmark. TSPN must beat this on all three horizons to justify the added complexity.

**Baseline 2 — Panel VAR Regression**:
A panel Vector Autoregression treating each (country, sector) as an individual unit with PPI as the time series and tariff rate as an exogenous covariate. Estimated with statsmodels' VAR implementation. This is the standard econometric approach and captures temporal autocorrelation but ignores graph topology.

**Baseline 3 — MLP with No Graph (Node-Features Only)**:
Takes node feature vectors directly and predicts price changes with a 3-layer MLP. No message passing, no graph structure. If TSPN beats this substantially, it proves the graph topology adds predictive value beyond what the node-level features alone contain.

### 7.2 Ablation Variants (What You Test by Removing TSPN Components)

**Ablation 1 — GCN (Replace Attention with Mean Aggregation)**:
Same architecture but replace GAT with GCN (equal-weight mean aggregation over neighbors). Tests whether the learned attention weights add value over treating all suppliers equally.

**Ablation 2 — GAT Without Temporal Module (Single Snapshot)**:
Remove the GRU. Train on single-snapshot graphs at event time only. Tests whether temporal dynamics (lagged price adjustment) add predictive value. Expected: this ablation hurts 12-month predictions most, less effect on 3-month.

**Ablation 3 — GAT Without Shock Injection in Edge Features**:
Instead of injecting δτ into edge feature 4, represent the shock as a global node signal (add δτ contribution to the affected node's features directly). Tests whether edge-level shock injection (bilateral precision) is better than node-level shock representation.

**Ablation 4 — TSPN with 1 GAT Layer Only**:
Remove Layer 2. Tests whether capturing 2-hop neighborhood (indirect supplier effects) adds value over 1-hop (direct suppliers only).

**Ablation 5 — TSPN with 3 GAT Layers**:
Add a third GAT layer (3-hop). Tests whether deeper propagation helps or hurts (possibly due to over-smoothing — a known GNN pathology where deep models blur node representations together).

---

## SECTION 8 — Evaluation Framework

### 8.1 Primary Metrics (Report for Each Baseline, Ablation, and TSPN)

**RMSE** (Root Mean Square Error) by horizon: Main headline metric. Computed per node, averaged across all labeled nodes in the held-out event fold. Report separately for 3m, 6m, 12m.

**MAE** (Mean Absolute Error): More robust to outliers than RMSE. Report alongside RMSE.

**Directional Accuracy**: What fraction of sector predictions got the sign of price change correct (up/down)? This is economically the most interpretable metric — even if the magnitude is off, does the model correctly predict which sectors face price increases vs. decreases?

**R² (Coefficient of Determination)**: Variance in actual price changes explained by the model. Compare to Leontief baseline R² — the improvement in R² is your headline claim.

### 8.2 Interpretability Metrics

**Attention vs Leontief Correlation**:
After training, extract the learned attention weights α_ij for all edges in Layer 1. Compute the Leontief inverse (I - A)^{-1} from the WIOD data. Calculate Pearson and Spearman correlation between α_ij and the corresponding Leontief inverse entry.

If the model has learned the economic structure, this correlation should be:
- Positive and statistically significant (p < 0.01)
- Higher for Layer 1 than Layer 2 (direct effects are better captured)
- Higher for manufacturing sectors than services (where input-output linkages are more physical)

This is the most publishable finding in the paper — the model recovers economic structure without being given it.

**Cascade Depth Measurement**:
For each held-out event, measure the average predicted |Δp| at hop distance 1, 2, 3, 4 from the directly shocked nodes. Plot this attenuation curve. Find the hop k* where average |Δp| drops below 5% of the original shock magnitude. This k* is the empirical cascade depth. Compare it across events and sectors (manufacturing cascades deeper than services).

**Shock Amplifier Sector Identification**:
Compute eigenvector centrality of the learned attention graph (edges weighted by α_ij). Compute eigenvector centrality of the raw trade-flow graph (edges weighted by import_pen_coeff). The ratio = attention_centrality / trade_centrality. Sectors with ratio >> 1 are amplifiers: they transmit more shock than their trade volume implies. Rank and report the top 10 amplifier sectors.

---

## SECTION 9 — Product Features (The Dashboard)

### 9.1 Core User Workflow

A user opens the app and faces a simple 3-step workflow:

**Step 1 — Define a tariff scenario**: Choose the importing country (dropdown of 43 countries), the tariff-affected product sector (dropdown of 56 ISIC sectors), the tariff magnitude (slider: 0% to 50% increase), and the affected exporting country or "all countries." This can represent a historical event (select from library) or a hypothetical future scenario.

**Step 2 — Run the model**: The app runs the trained TSPN model in inference mode with the user's shock vector. This takes 1–3 seconds on CPU (the model is small at inference time). Results are generated for all 2,464 (country, sector) nodes.

**Step 3 — Explore results**: The user sees three interconnected views: the cascade graph visualization, the price prediction table, and the risk score dashboard.

### 9.2 Feature 1 — Tariff Scenario Library

Pre-loaded historical events from the training set, so users can explore what the model has learned:
- US Section 232 Steel & Aluminum (March 2018)
- US-China Trade War Waves 1–4 (July 2018 – May 2019)
- EU Retaliation (June 2018)
- UK Post-Brexit Global Tariff (January 2021)

Each historical event shows: the actual vs. predicted price changes (for sectors with PPI data), with a model accuracy badge. This builds user trust by showing the model was validated.

Also allow users to type in a hypothetical future scenario: "What if the US imposes 25% tariffs on all EU automotive imports?" Users can build custom scenarios in under a minute.

### 9.3 Feature 2 — Cascade Graph Visualization

An interactive network diagram showing the supply chain graph centered on the shock origin. 

Layout: Radial/force-directed graph with the shocked node(s) at the center, rings expanding outward representing hop distance. Edge opacity = trade flow weight. Edge color = red (negative shock propagating) or neutral. Node size = predicted |Δp| magnitude. Node color = magnitude of predicted price change (red = price increase, blue = price decrease).

Filtering controls: Show only nodes with predicted |Δp| > threshold (default 0.5%). Filter by sector type (manufacturing only, services only, all). Filter by country. Zoom and pan.

Interaction: Click any node to see its full prediction breakdown (3m / 6m / 12m), its position in the supply chain, and the top 5 most influential upstream suppliers for this shock (by attention weight on shock edges).

Implementation: PyVis for quick Streamlit integration, or D3.js for the production web app. PyVis is much faster to build and sufficient for the MVP.

### 9.4 Feature 3 — Price Impact Table

A sortable, filterable table showing all sectors with predictions:

Columns: Country | Sector | Direct Exposure | Predicted Δp (3m) | Predicted Δp (6m) | Predicted Δp (12m) | Cascade Hop Distance | Risk Level

Risk level is a computed categorical: High (|Δp| > 2% at 6m), Medium (0.5%–2%), Low (<0.5%), Negligible (< detection threshold).

Sorting: by predicted impact, by risk level, by sector, by country.

Filtering: by sector type, by country, by risk level, by hop distance.

Export: Download filtered table as CSV or PDF.

### 9.5 Feature 4 — Sector Risk Dashboard

High-level summary view for users who want quick insight without graph exploration.

Top 10 most-impacted sectors (bar chart, sorted by predicted 6m Δp).

Cascade depth indicator: how many hops does this shock propagate before dissipating below 1%?

World map: choropleth colored by total national economic exposure (sum of |predicted Δp| × sector output across all sectors in each country).

Sector type breakdown: donut chart showing share of total economic impact in manufacturing vs. services vs. primary.

Amplifier sectors badge: highlight the top 3 amplifier sectors for this specific shock event.

### 9.6 Feature 5 — Scenario Comparison

Side-by-side view of two tariff scenarios.

Use case: "What's worse — a 25% tariff on steel from all countries, or a 10% tariff on all goods from China?" User sets up both scenarios and compares:
- Total global economic impact (aggregate Δp across all sectors)
- Geographic distribution of impact
- Which sectors are more exposed under each scenario
- Cascade depth comparison

This is the feature that makes the tool genuinely useful for policy analysts and procurement strategists.

### 9.7 Feature 6 — My Exposure Calculator

Users enter a simple description of their business exposure: "I am a European auto manufacturer. My supply chain includes steel from Germany, electronics from Japan, and plastics from the US." The app maps this description to (country, sector) pairs and computes a portfolio-weighted tariff risk score.

For the MVP: manual sector selection (dropdown list). For the V2: NLP-powered mapping of free-text supply chain description to ISIC sectors.

Output: Total portfolio risk score (0–100), breakdown by exposure source, recommended risk-reduction actions (diversify supplier country, switch sector sourcing).

### 9.8 Feature 7 — Explanation Modal

For any node in the graph or row in the table, an "Explain this" button opens a modal showing:
- Why did this sector get this risk score?
- Which upstream suppliers are most responsible for the cascaded shock?
- What is the hop path from the shock origin to this sector?
- How does this sector's attention centrality compare to its trade-flow centrality?

This feature is what turns the research model's interpretability findings (attention weights, cascade paths) into a user-facing explanation. It is also a direct demonstration of the paper's interpretability contribution.

---

## SECTION 10 — Complete Technology Stack

### 10.1 Data and ML (All Free)

- **Python 3.10+**: Primary language
- **PyTorch 2.1**: Deep learning framework
- **PyTorch Geometric (PyG) 2.4**: Graph neural network framework — GATConv, GRUConv, Data/Dataset classes
- **Pandas 2.0**: Data manipulation (WIOD parsing, Comtrade processing)
- **NumPy + SciPy**: Numerical computation, Leontief inverse computation
- **NetworkX**: Graph analysis — centrality computation, cascade depth measurement
- **PyArrow**: Parquet file handling (fast storage for large datasets)
- **comtradeapicall**: Python wrapper for UN Comtrade API
- **wbdata / pandas-datareader**: World Bank and BLS data access

### 10.2 Visualization (All Free)

- **PyVis**: Interactive network graph in Python/Streamlit (MVP graph visualization)
- **Plotly**: Interactive charts (bar charts, choropleth maps, scatter plots)
- **D3.js**: Production-quality interactive graph visualization (V2 upgrade from PyVis)
- **Matplotlib + Seaborn**: Research paper figures

### 10.3 Product Backend (Free)

- **FastAPI**: REST API server for model inference
  - Endpoint: POST /predict — takes scenario JSON, returns prediction JSON
  - Endpoint: GET /events — returns historical event library
  - Endpoint: GET /sectors — returns sector and country metadata
- **Uvicorn**: ASGI server for FastAPI
- **ONNX**: Export trained PyTorch model to ONNX format for faster CPU inference (reduces prediction latency from ~3s to ~0.5s)
- **SQLite** (development) / **Supabase PostgreSQL free tier** (production): Stores scenario history, cached predictions, user sessions

### 10.4 Product Frontend (Free)

**MVP (Weeks 1–4 of product phase)**: Streamlit
- Single Python file, no frontend coding required
- Streamlit Cloud free tier hosts it for free
- Sufficient for demo, research presentation, and early user testing
- Deploy via: `streamlit run app.py` locally, then push to Streamlit Cloud

**Production V2 (optional upgrade)**: React + Tailwind CSS
- Cleaner UI, more interactive visualizations
- Hosted on Vercel (free tier) with FastAPI backend on Railway ($5/month)
- Only necessary if you want it to look like a real SaaS product

### 10.5 Compute and Hosting (Free/Nearly Free)

- **Training**: Google Colab free tier (T4 GPU). All training fits in free tier with checkpoint saving to Google Drive.
- **Model serving**: CPU inference on free-tier hosting. ONNX export ensures fast enough latency.
- **Hosting (MVP)**: Streamlit Community Cloud — completely free, unlimited projects, public apps.
- **Hosting (production)**: Hugging Face Spaces (free, supports Streamlit and Gradio, GPU available for paid tier but not needed for inference). Or Render.com free tier.
- **Database**: Supabase free tier (500MB PostgreSQL) — enough for caching scenario results.
- **Version control**: GitHub (free).
- **Experiment tracking**: Weights & Biases free tier (3 projects, unlimited runs) — track training loss curves, hyperparameter experiments.

**Total monthly cost for the full product**: $0 on free tiers. Optional: $10/month Colab Pro for easier training sessions. $5/month Railway for more reliable API hosting.

---

## SECTION 11 — Phase-by-Phase Implementation Roadmap

### Phase 1 — Data Collection and Environment Setup (Weeks 1–2)

**Goal**: All raw data downloaded, environment configured, basic parsing working.

**Environment setup**:
- Create conda environment with Python 3.10
- Install: torch, torch_geometric, pandas, numpy, scipy, networkx, pyarrow, plotly, comtradeapicall, streamlit
- Set up Google Drive folder for data storage and Colab checkpoint saving
- Set up GitHub repository with standard project structure: data/, notebooks/, src/, models/, app/

**WIOD download**:
- Go to rug.nl/ggdc/valuechain/wiod and download all 17 annual Excel files and socioeconomic accounts
- Store in data/raw/wiod/
- Write and test the WIOD parser notebook to verify you can read the matrices correctly
- Check: does your parsed edge table for year 2016 have the expected number of rows (~2,464 nodes)?

**BLS PPI download**:
- Register at bls.gov for API key
- Pull all Industry PPI series at 3-digit NAICS level, 2000–2022
- Store in data/raw/bls_ppi/

**Tariff event product code extraction**:
- Download the 5 Federal Register PDFs for Section 232 and Section 301 Lists 1–3
- Manually extract or copy-paste the HTS code lists (they are structured tables in the Federal Register — typically 2–5 pages of HTS codes per notice)
- Store as CSV files: tariff_events/us_232_steel.csv, tariff_events/us_301_list1.csv, etc.
- Download the UK Global Tariff comparison tool output from gov.uk

**WITS download**:
- Register at wits.worldbank.org
- Download applied MFN tariff data for USA, CHN, EU member states, CAN: years 2015–2021 at HS6 level
- Store in data/raw/wits/

**Milestone check**: You can load any WIOD year file, see the matrix shape (2,464 × 2,464+), and extract a specific bilateral trade flow. You have all Federal Register HTS code lists in CSV format.

---

### Phase 2 — Data Processing Pipeline (Weeks 3–4)

**Goal**: Clean, structured Parquet files for nodes, edges, tariff rates, and labels.

**Build these processing scripts (one script per stage)**:

Script 1 — `parse_wiod.py`: Reads each WIOD Excel → outputs one Parquet file per year with columns (year, src_country, src_sector, tgt_country, tgt_sector, flow_usd, import_pen_coeff). Apply the 0.001 threshold. Log: "Year 2016: N edges after threshold filtering."

Script 2 — `build_hs_isic_concordance.py`: Reads the UN correspondence table → outputs a lookup dictionary: hs6_code → list of isic_codes. Handle many-to-many by storing all ISIC codes with proportional weights.

Script 3 — `compute_tariff_rates.py`: For each WIOD country pair and each ISIC sector and each year: use concordance + WITS rates → compute trade-value-weighted average sector tariff rate. Output: tariff_rates.parquet.

Script 4 — `build_shock_vectors.py`: For each of the 6 tariff events: read the HTS code list → map to ISIC sectors → identify affected bilateral pairs → compute delta_tariff. Output: shock_vectors.parquet.

Script 5 — `compute_node_features.py`: For each (country, sector, year): compute all 9 node features using WIOD socioeconomic accounts, edge data, Leontief inverse, PPI lags. Output: node_features.parquet.

Script 6 — `generate_labels.py`: For each tariff event × each labeled (country, sector): compute 3m/6m/12m PPI change from BLS/Eurostat. Output: labels.parquet.

**Validation checks after Phase 2**:
- Do all labeled nodes have non-null PPI values for all events? (Some will be null — that is expected and handled by the mask)
- Does the total number of non-zero shock vector entries match your expectation based on the Federal Register product lists?
- Is the distribution of import penetration coefficients sensible (most edges < 5%, a few critical edges up to 30%)?

**Milestone check**: You can load the processed Parquet files for any event and inspect the shock vector, node features, and labels. No missing values in edge table. Labels populated for at least 60% of nodes for each event.

---

### Phase 3 — PyG Graph Dataset Construction (Week 5)

**Goal**: Working PyG Data objects for all 6 events, ready to feed into the model.

Build `build_pyg_dataset.py`:
- For each event e:
  - Identify the 8 quarterly snapshots ending at the event date
  - For each snapshot q: build node feature matrix X_q, edge index, edge feature matrix E_q
  - Stack into a temporal sequence: [Data(X_0, E_0), ..., Data(X_7, E_7)]
  - Attach labels y and label mask for the event
  - Save as PyG InMemoryDataset

Build `visualize_graph.py` (for sanity checking):
- Load one event's graph
- Compute basic network statistics: average degree, degree distribution, clustering coefficient
- Draw the graph for a subset (e.g., just USA nodes) to visually verify the structure looks like a supply chain

**Milestone check**: You can iterate through the dataset, access temporal sequences, and verify that the shock vector is non-zero in the correct event graph and zero in non-event graphs.

---

### Phase 4 — Baseline Models (Week 6)

**Goal**: Implement all baselines and record their RMSE/MAE/directional accuracy. These are your performance floor.

Build `baselines/leontief_io.py`: 
- Compute Leontief inverse for each event year
- For each event: multiply shock vector × Leontief inverse × assumed pass-through rate
- Tune pass-through rate using one held-out event (meta-calibration), then apply uniformly
- Record RMSE per horizon per LOEO fold

Build `baselines/panel_var.py`:
- Construct panel: (country, sector) × time series of PPI + tariff rate
- Fit VAR with tariff rate as exogenous variable
- Forecast 3/6/12 months ahead for each event
- Record same metrics

Build `baselines/mlp_no_graph.py`:
- Flatten node features, concatenate shock exposure signal (node-level tariff delta)
- 3-layer MLP, same training protocol as TSPN
- Record same metrics

**Milestone check**: You have a results table with all baselines. Leontief typically achieves around 0.5–1.5% RMSE at 6m (depending on shock magnitude and sector). VAR should be similar. These are your targets to beat with TSPN.

---

### Phase 5 — TSPN Model Implementation (Weeks 7–9)

**Goal**: Working TSPN model with the full architecture, training loop, and LOEO-CV.

Build these model files:

`models/feature_embedding.py`: Linear projections for node and edge features with batch normalization.

`models/tspn_gat_layer.py`: Custom GAT layer that incorporates edge features into the attention computation. The key difference from standard PyG GATConv is that edge features modulate attention weights. Use the `MessagePassing` base class from PyG and implement the attention formula manually.

`models/tspn.py`: Full model class combining embedding → GAT Layer 1 → GAT Layer 2 → GRU → Multi-horizon MLP. Forward pass takes a temporal sequence of graphs.

`training/train.py`: Training loop with LOEO-CV, early stopping, W&B logging, and checkpoint saving to Google Drive.

`training/evaluate.py`: Evaluation functions computing RMSE, MAE, R², directional accuracy with confidence intervals from LOEO folds.

**Debugging order**:
1. Verify forward pass runs without error on one batch (one event, one time step)
2. Verify loss decreases on the first training event (overfitting one event is expected and is a useful sanity check)
3. Verify LOEO-CV loop runs through all 6 folds correctly
4. Verify that predictions are in the right range (price changes should be between roughly -10% and +10% for reasonable shocks)

**Milestone check**: Full TSPN training completes 6 LOEO folds in one Colab session. TSPN RMSE is lower than Leontief and VAR baselines on at least 4 of 6 folds. Training loss curves show normal convergence (decreasing then leveling off, not diverging or stagnating immediately).

---

### Phase 6 — Ablations and Interpretability (Weeks 10–11)

**Goal**: Complete experimental results including all ablations, attention analysis, and economic interpretability findings.

Run all 5 ablations using the same training infrastructure as TSPN. Compile results table.

Build `analysis/interpretability.py`:
- Extract attention weights from trained model for each LOEO fold
- Compute Leontief inverse for the corresponding WIOD year
- Calculate Pearson and Spearman correlations
- Generate scatter plot: attention weight vs Leontief coefficient (this is a key paper figure)

Build `analysis/cascade_depth.py`:
- For each held-out event, compute average predicted |Δp| at each hop distance from shock origin
- Plot attenuation curves per event and averaged across events
- Identify empirical cascade depth k* for each event type

Build `analysis/amplifier_sectors.py`:
- Compute eigenvector centrality on learned attention graph
- Compute eigenvector centrality on raw trade-flow graph
- Compute amplification ratios
- Generate ranked table and sector map

**Milestone check**: You have all numbers for the results section of the paper. The attention vs Leontief correlation is statistically significant (you'd expect r ≈ 0.4–0.7 based on similar work in financial networks). You have an empirical cascade depth finding.

---

### Phase 7 — Research Paper Writing (Weeks 12–16)

**Write sections in this order** (not top to bottom):

Week 12: Section 3 (Data and Graph Construction) — most factual, least ambiguous
Week 12–13: Section 4 (Model) — formalize the architecture with proper notation
Week 13: Section 5 (Experiments) — assemble tables and figures, write around results
Week 14: Section 6 (Interpretability) — economic implications narrative
Week 15: Section 1 (Introduction) and Section 2 (Related Work) — easier once you know your full results
Week 16: Section 7 (Conclusion), abstract, proofreading

**Paper formatting**: LaTeX in Overleaf (free tier is enough for a single paper). Use the target journal's LaTeX template.

---

### Phase 8 — Product MVP (Weeks 17–19, parallel with paper revisions)

**Goal**: Working Streamlit app deployed on Streamlit Community Cloud.

Build `app/app.py`:

Week 17: Static skeleton — layout with sidebar for scenario builder, main area for graph visualization placeholder, table placeholder. Get the UI structure right before adding model calls.

Week 18: Model integration — export trained model to ONNX, wire up the inference call, populate the table with real predictions, build the PyVis graph visualization for a fixed event.

Week 19: Dynamic scenario builder — connect the sidebar controls (country, sector, magnitude sliders) to the shock vector constructor, which feeds into model inference. Add the scenario comparison panel. Add CSV export.

Deploy to Streamlit Community Cloud: push to GitHub, connect Streamlit Cloud to repo, set environment variables. Free, takes 5 minutes.

---

### Phase 9 — Product V2 (Optional, Weeks 20–24)

If you want a more production-grade product after the MVP:
- Migrate backend to FastAPI (better for programmatic API access)
- Migrate graph visualization from PyVis to D3.js (better interactivity)
- Add scenario history and user session persistence via Supabase
- Add the "My Exposure Calculator" feature with NLP sector mapping
- Set up automated data refresh: weekly Comtrade pull, monthly WITS tariff check
- Add email alerts for new tariff announcements matching user-defined exposure profiles

---

## SECTION 12 — Folder Structure

```
tspn/
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/
│   │   ├── wiod/                    # WIOD Excel files (2000–2016)
│   │   ├── comtrade/                # Comtrade Parquet files (2017–2021)
│   │   ├── wits/                    # WITS tariff CSVs
│   │   ├── bls_ppi/                 # BLS PPI CSVs
│   │   ├── eurostat_ppi/            # Eurostat PPI CSVs
│   │   ├── tariff_events/           # Federal Register HTS code CSVs
│   │   └── concordance/             # HS-ISIC correspondence table
│   │
│   ├── processed/
│   │   ├── edges/                   # Parquet edge tables per year
│   │   ├── node_features/           # Parquet node feature tables
│   │   ├── tariff_rates/            # Computed sector tariff rates
│   │   ├── shock_vectors/           # Shock vectors per event
│   │   └── labels/                  # PPI change labels per event
│   │
│   └── pyg_datasets/                # PyG Data objects per event
│
├── src/
│   ├── data/
│   │   ├── parse_wiod.py
│   │   ├── fetch_comtrade.py
│   │   ├── build_concordance.py
│   │   ├── compute_tariff_rates.py
│   │   ├── build_shock_vectors.py
│   │   ├── compute_node_features.py
│   │   ├── generate_labels.py
│   │   └── build_pyg_dataset.py
│   │
│   ├── models/
│   │   ├── feature_embedding.py
│   │   ├── tspn_gat_layer.py        # Custom GAT with edge features
│   │   ├── tspn_gru.py              # Temporal GRU module
│   │   ├── output_head.py           # Multi-horizon MLP head
│   │   └── tspn.py                  # Full model class
│   │
│   ├── baselines/
│   │   ├── leontief_io.py
│   │   ├── panel_var.py
│   │   └── mlp_no_graph.py
│   │
│   ├── training/
│   │   ├── train.py                 # LOEO-CV training loop
│   │   ├── evaluate.py              # Metrics computation
│   │   └── losses.py                # Multi-horizon MSE + attention L1
│   │
│   └── analysis/
│       ├── interpretability.py      # Attention vs Leontief correlation
│       ├── cascade_depth.py         # Cascade attenuation measurement
│       ├── amplifier_sectors.py     # Amplifier sector identification
│       └── paper_figures.py         # Generate all paper figures
│
├── notebooks/
│   ├── 01_wiod_exploration.ipynb    # Sanity-check WIOD data
│   ├── 02_graph_analysis.ipynb      # Network statistics
│   ├── 03_training_runs.ipynb       # Colab training sessions
│   ├── 04_results_analysis.ipynb    # Compile results tables
│   └── 05_paper_figures.ipynb       # Reproduce all figures
│
├── models/
│   ├── checkpoints/                 # Saved model checkpoints per fold
│   └── onnx/                        # ONNX exports for product inference
│
└── app/
    ├── app.py                       # Streamlit main app
    ├── components/
    │   ├── scenario_builder.py      # Sidebar scenario UI
    │   ├── graph_viz.py             # PyVis network component
    │   ├── price_table.py           # Predictions table component
    │   └── risk_dashboard.py        # Risk summary component
    ├── utils/
    │   ├── inference.py             # Model inference wrapper
    │   └── scenario_parser.py       # Scenario JSON → shock vector
    └── assets/
        └── style.css                # Custom CSS
```

---

## SECTION 13 — Complexity Assessment vs Ecoacoustic Work

Your ecoacoustic project involved: audio signal processing, feature extraction (spectral + acoustic indices), ML classification or regression, ecological pattern interpretation. Single modality, relatively homogeneous data, established pipeline tools.

This project involves:

**Data heterogeneity**: Three completely different data types — IO tables, HS-coded trade flows, tariff schedules — requiring custom parsing and integration. Each has a different format, update frequency, and granularity. The HS→ISIC concordance is a non-trivial data engineering problem.

**Graph construction**: Going from raw tabular trade data to a multi-layer directed weighted graph with 2,464 nodes, ~100K edges, and temporal snapshots is more complex than any feature extraction pipeline in ecoacoustics.

**Model architecture**: The TSPN combines three different neural network paradigms (attention networks, graph NNs, recurrent NNs) plus a multi-task output head. More moving parts than a standard classification or regression model.

**Training protocol**: Leave-One-Event-Out is more complex to implement correctly than a standard train-test split. You also need the augmentation strategy to be event-aware.

**Interpretability analysis**: The attention vs Leontief correlation, cascade depth, and amplifier sector analysis are genuinely novel analytical components, not standard evaluation metrics.

**Product component**: A deployable dashboard with graph visualization, real-time inference, and scenario comparison is more complex than serving a standard audio model prediction.

However: the data is all publicly available and free. The graph has only 2,464 nodes — completely manageable. The model is small enough to train on a free Colab GPU in hours. PyTorch Geometric has built-in support for almost everything you need. The product runs on free hosting. This is achievable solo in a 4–5 month timeline.
