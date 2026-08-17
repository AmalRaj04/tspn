# Tariff Shock Propagation via Supply-Chain Graph Networks (TSPN)
## Full Research Brief — Implementation Context Document

---

## 1. Problem Statement

Global tariff policy has become one of the most disruptive forces in international economics. The US-China trade war (2018–2020), US Section 232 steel and aluminum tariffs, post-Brexit UK tariff restructuring, and the sweeping 2025 US universal tariff package have created repeated, large-scale shock events. The central economic problem these events expose is that **tariff shocks don't stay in the sectors they directly hit.**

A 25% tariff on imported steel does not simply raise steel prices. It raises input costs for automotive manufacturers who buy steel, which raises vehicle prices, which feeds into transport costs, which propagates into construction, logistics, and consumer goods — through the supply chain graph. This cascade is entirely mediated by supplier-buyer relationships: who buys from whom, in what quantity, and with what import dependence.

The consequences of failing to model this network effect are serious:

- **Policymakers** underestimate the economy-wide inflationary impact of tariffs because they model only direct (first-order) sector effects, not cascading downstream effects.
- **Firms in downstream sectors** have no early warning system for cost increases — they discover them when contracts come up for renewal and input prices have already risen.
- **Central banks and researchers** misattribute PPI and CPI changes — they observe downstream price effects but cannot trace the network path that generated them.
- **Trade negotiators** cannot quantify the systemic cost of targeting specific sectors, because the cost of a tariff on sector X depends on how central X is in the supply chain graph — not just on X's direct trade volume.

The core research question is: **Can a graph neural network, trained on historical tariff events and supply chain trade flows, learn to predict the multi-horizon, network-mediated price propagation of a new tariff shock?**

---

## 2. What Existing Methods Get Wrong

### 2.1 Computable General Equilibrium (CGE) Models
The dominant tool in trade policy analysis (e.g., GTAP). CGE models impose behavioral assumptions (elasticities, market clearing conditions) that are calibrated, not learned. They are static — they solve for a new equilibrium but don't model the path of adjustment. They cannot learn from historical shock events. They require significant expert parameterization for each new shock scenario. They are also computationally expensive and not easily updated.

### 2.2 Panel Regression / Econometric Models
Standard approach in empirical trade economics (e.g., Amiti, Redding, Weinstein 2019). These regress sector-level price changes on tariff rates, treating tariff as an exogenous covariate. They ignore network topology — every sector is treated as independent. They capture only the average statistical association, not the graph-mediated transmission. They cannot predict future shocks because they are retrospective by design.

### 2.3 Standard Input-Output (Leontief) Models
The Leontief model uses the (I−A)⁻¹ inverse matrix to compute total cost propagation analytically. It is linear, static, and has no learnable components. It treats all transmission coefficients as fixed (the technical coefficients matrix A). It has no temporal component. It cannot model non-linear threshold effects or learn sector-specific shock sensitivity. However, it is the standard economics benchmark and the TSPN attention weights should be compared to it.

### 2.4 What is Missing
No existing method:
1. Treats tariff shock propagation as a **supervised learning problem on a temporal graph**
2. Combines WIOD + UN Comtrade + WITS tariff schedules into a **unified multi-country trade graph dataset**
3. Uses **attention weights** as learned, data-driven transmission coefficients and compares them to the Leontief inverse
4. Predicts **multi-horizon** (3m, 6m, 12m) price changes from a tariff event
5. **Empirically measures cascade depth** — at which graph hop does a shock attenuate below economic significance?
6. Identifies **shock amplifier sectors** — nodes whose attention centrality exceeds their trade-flow centrality, meaning they transmit disproportionately more shock than their trade volume would suggest

---

## 3. Proposed Solution: TSPN Architecture Overview

**Name**: Tariff Shock Propagation Network (TSPN)

**Core idea**: Build a temporal graph where nodes are (country, sector) pairs, directed edges are bilateral trade flows, and tariff events are injected as perturbations to edge features. A Graph Attention Network (GAT) propagates the shock hop-by-hop through the supply chain graph, learning which upstream links matter most. A GRU module captures lagged temporal dynamics (prices adjust slowly, not instantly). A multi-horizon MLP head predicts price changes at t+3, t+6, and t+12 months.

**What makes it novel**:
- Shock injection into edge features on affected bilateral pairs (not global signal)
- Attention weights as learned, data-driven analog of Leontief coefficients
- End-to-end trainable, data-driven — no hand-specified elasticities
- Multi-horizon output captures the pass-through dynamics literature has studied analytically

---

## 4. Dataset Details

### 4.1 World Input-Output Database (WIOD)

**What it provides**: Annual inter-sector, inter-country trade flow matrices. Each year gives a matrix where entry (i, j) = total intermediate goods purchased by sector j in country B from sector i in country A, in USD millions.

**Coverage**: 43 countries + Rest of World, 56 ISIC Rev. 4 sectors, annually from 2000 to 2016.

**Download URL**: https://www.rug.nl/ggdc/valuechain/wiod/

**Files to download**:
- `WIOT[YEAR]_October16_ROW.RData` or the Excel equivalents — one file per year (2000–2016)
- `wiot_sep_16_txt.zip` — supplementary socioeconomic accounts, contains gross output, value added, and employment per (country, sector) per year
- Use the **2016 release** (the November 2016 version is the most widely cited)

**How to parse**: Load the annual IO matrix. Rows = supplying (country, sector). Columns = purchasing (country, sector). The domestic intermediate use block (same country, different sectors) and the international trade block (different countries) are both present. You will use both.

**ISIC sector list**: 56 sectors from Agriculture (A01–A03) through Services (S94–S96). Full list in the WIOD documentation.

**Limitation**: Stops at 2016. You extend to 2021 using UN Comtrade (see 4.2).

---

### 4.2 UN Comtrade

**What it provides**: Bilateral trade flow data at HS6 commodity level for all country pairs.

**Use in this project**: Extend the WIOD trade graph from 2016 to 2021. WIOD gives sector-level flows; Comtrade gives HS6-level flows which you aggregate to ISIC sectors using the concordance (see 4.5). Use WIOD sector structure as a prior and update edge weights with Comtrade annual flows.

**Access**: https://comtradeplus.un.org/ — free registration required. API available.

**Python package**: `comtradeapicall` (pip install). Or use the bulk download facility for full datasets.

**What to pull**:
- Reporter = all countries in WIOD country list
- Partner = all countries
- Trade flow = imports (HS, annual)
- Commodity = AG6 (all HS6 codes)
- Years = 2017, 2018, 2019, 2020, 2021

**Data volume**: Large — filter to the country pairs and sector pairs that appear in WIOD to keep manageable.

---

### 4.3 WITS / WTO Tariff Data

**What it provides**: Applied MFN tariff rates (and preferential rates) at HS6 product level, by reporter country and year.

**Primary source — World Bank WITS**: https://wits.worldbank.org/
- Bulk data download available under "Data" → "Tariff & Trade" → "TRAINS" database
- Pull applied MFN AVE (ad valorem equivalent) rates for all reporters × all HS6 × years 2015–2021

**Secondary source — WTO Tariff Download Facility**: https://tariffdata.wto.org/
- More current than WITS for recent years
- Good for validating post-2018 US, EU, China tariff schedules

**Key event product codes (what to pull specifically)**:

```
US Section 232 Steel (March 2018):
  HTS codes: 7206.10, 7206.90, 7207.11, 7207.12, 7207.19, 7207.20,
             7208.10–7208.90, 7209.15–7209.90, 7210.11–7210.90,
             7211.13–7211.90, 7212.10–7212.60, 7213.10–7213.91,
             7214.10–7214.91, 7215.10–7215.90, 7216.10–7216.99,
             7217.10–7217.90, 7218.10–7218.99, 7219.11–7219.90,
             7220.11–7220.90, 7221.00, 7222.11–7222.40, 7223.00,
             7224.10–7224.90, 7225.11–7225.99, 7226.11–7226.99,
             7227.10–7227.90, 7228.10–7228.80, 7229.20–7229.90
  Tariff increase: +25 percentage points (from near-zero MFN)

US Section 232 Aluminum (March 2018):
  HTS codes: 7601.10, 7601.20, 7604.10–7604.29, 7605.11–7605.29,
             7606.11–7606.92, 7607.11–7607.19, 7608.10–7608.20,
             7609.00, 7610.10–7610.90, 7611.00, 7612.10–7612.90,
             7613.00, 7614.10–7614.90, 7615.11–7615.20, 7616.10–7616.99
  Tariff increase: +10 percentage points

US Section 301 List 1 (July 6, 2018): 818 HTS-8 lines, ~$34B trade from China
  Full list: Federal Register Vol. 83, No. 119 (June 20, 2018)
  Tariff increase: +25 pp on imports from China

US Section 301 List 2 (August 23, 2018): 284 HTS-8 lines, ~$16B trade from China
  Full list: Federal Register Vol. 83, No. 155 (August 10, 2018)
  Tariff increase: +25 pp on imports from China

US Section 301 List 3 (September 24, 2018): ~$200B trade from China
  Tariff: +10 pp initially, raised to +25 pp on May 10, 2019
  Full list: Federal Register Vol. 83, No. 185 (September 21, 2018)

EU Retaliatory Tariffs (June 22, 2018): ~€2.8B of US goods
  Products: steel, agricultural, consumer goods
  Source: EU Official Journal, Implementing Regulation (EU) 2018/886

Canada Retaliatory Tariffs (July 1, 2018): C$16.6B of US goods
  Source: Canada Gazette Part II, Vol. 152, No. 14

UK Global Tariff (January 1, 2021):
  Source: https://www.gov.uk/guidance/uk-tariffs-from-1-january-2021
  Compare to EU CET rates to compute delta_tariff for UK import sectors
```

---

### 4.4 Validation Labels (Price Change Targets)

**BLS Producer Price Indices (US sectors)**:
- URL: https://www.bls.gov/ppi/data.htm → "Industry Data" → download by NAICS code
- Frequency: Monthly (use to compute 3m, 6m, 12m changes post-event)
- Crosswalk: NAICS → ISIC Rev. 4 using UN Statistics Division concordance
- Series to pull: All Industry PPI series for NAICS 2-digit and 3-digit codes
- Format: CSV download, series_id format PCU[NAICS_CODE]

**Eurostat PRODCOM Price Indices (EU sectors)**:
- URL: https://ec.europa.eu/eurostat/web/prodcom/data/database
- Table: `prom_t` (sold production) — use as proxy for producer prices
- Or use Eurostat PPI series: `sts_inppd_m` (monthly PPI by NACE sector)
- NACE Rev. 2 maps cleanly to ISIC Rev. 4 (they share the same classification structure)

**World Bank Commodity Price Data (Pink Sheet)**:
- URL: https://www.worldbank.org/en/research/commodity-markets
- Use for primary commodity sectors: agriculture, metals, energy
- Monthly, covers coal, steel, aluminum, copper, agricultural commodities

**WIOD Socioeconomic Accounts (for 2000–2016 training period)**:
- Available in the `wiot_sep_16_txt.zip` download
- Contains value-added deflators at (country, sector) level
- Use as the price label for the pre-2017 training period

**Label construction**:
```
For each tariff event e occurring at month m:
  label_i_3m  = ppi_i(m+3)  / ppi_i(m)  - 1   # 3-month % change
  label_i_6m  = ppi_i(m+6)  / ppi_i(m)  - 1   # 6-month % change
  label_i_12m = ppi_i(m+12) / ppi_i(m)  - 1   # 12-month % change

Only label sectors where PPI data is available (US, EU sectors primarily).
For other countries, use commodity price indices where applicable.
```

---

### 4.5 HS6 → ISIC Sector Concordance

**Source**: UN Statistics Division
- URL: https://unstats.un.org/unsd/trade/classifications/correspondence-tables.asp
- File: "HS 2017 ↔ ISIC Rev. 4" correspondence table (Excel/CSV)

**Key property**: This is a many-to-many mapping. One HS6 code can map to multiple ISIC sectors, and one ISIC sector maps to many HS6 codes.

**Aggregation approach**: When computing sector-level tariff rates from HS6 tariff rates, weight each HS6 tariff by its share of bilateral trade value within that ISIC sector:

```
tariff_ISIC_ij(t) = Σ_{k ∈ HS6(ISIC)} [trade_value_ijk(t) / total_trade_ISIC_ij(t)] × tariff_k_ij(t)

Where:
  trade_value_ijk = Comtrade bilateral flow for HS6 code k from i to j
  tariff_k_ij = applied MFN tariff rate for HS6 code k charged by j on imports from i
```

---

## 5. Graph Construction Protocol (Step by Step)

### Step 1: Build base graph from WIOD (one graph per year, 2000–2016)

```python
import pandas as pd
import numpy as np

# For each year t:
def build_graph_from_wiod(wiot_matrix, socioeconomic_accounts, year):
    """
    wiot_matrix: DataFrame, shape (N*C, N*C+final_demand)
                 rows = (country, sector) suppliers
                 cols = (country, sector) buyers + final demand
    """
    N_sectors = 56
    N_countries = 44  # 43 + RoW
    N_nodes = N_sectors * N_countries  # ~2,464

    nodes = []
    edges = []

    # Build node list
    for country in countries:
        for sector in sectors:
            gross_output = socioeconomic_accounts.loc[(country, sector, year), 'gross_output']
            total_imports = wiot_matrix.loc[(country, sector)].sum() - domestic_use
            nodes.append({
                'node_id': f"{country}_{sector}",
                'country': country,
                'sector': sector,
                'gross_output': gross_output,
                'total_imports': total_imports,
            })

    # Build edge list (filter by minimum trade share)
    THRESHOLD = 0.001  # 0.1% import penetration
    for supplier in node_list:
        for buyer in node_list:
            flow = wiot_matrix.loc[supplier, buyer]
            buyer_total_input = wiot_matrix.loc[:, buyer].sum()
            if buyer_total_input > 0:
                import_pen = flow / buyer_total_input
                if import_pen > THRESHOLD:
                    edges.append({
                        'source': supplier,
                        'target': buyer,
                        'flow_value': flow,
                        'import_pen_coeff': import_pen,
                        'tariff_rate': get_tariff(supplier, buyer, year),
                        'hhi': compute_hhi(supplier, buyer, year),
                    })
    return nodes, edges
```

### Step 2: Compute node features

```python
def compute_node_features(node, wiod_data, ppi_data, leontief_inverse, year):
    """
    Returns 9-dimensional feature vector per node per year
    """
    cntry, sec = node['country'], node['sector']
    
    gross_output = node['gross_output']
    total_imports = node['total_imports']
    total_exports = wiod_data.get_exports(cntry, sec, year)
    
    import_pen = total_imports / (gross_output + total_imports - total_exports + 1e-9)
    export_intensity = total_exports / (gross_output + 1e-9)
    
    # Backward linkage from Leontief decomposition
    # backward_linkage_i = sum of column i in (I-A)^{-1}
    backward_linkage = leontief_inverse[:, node_index].sum()
    
    # Tariff exposure = weighted avg tariff across all imports
    tariff_exposure = compute_weighted_tariff_exposure(cntry, sec, year)
    
    # Lagged price changes (4 annual lags)
    ppi_lags = [ppi_data.get_change(cntry, sec, year - lag) for lag in range(1, 5)]
    
    return np.array([
        np.log(gross_output + 1),   # log gross output
        import_pen,                  # import penetration
        export_intensity,            # export intensity
        backward_linkage,            # Leontief BL
        tariff_exposure,             # weighted tariff exposure
        ppi_lags[0],                 # PPI change t-1
        ppi_lags[1],                 # PPI change t-2
        ppi_lags[2],                 # PPI change t-3
        ppi_lags[3],                 # PPI change t-4
    ])
    # Dimension: 9
```

### Step 3: Compute edge features and shock vector

```python
def compute_edge_features(edge, tariff_data, shock_event, year):
    """
    Returns 6-dimensional feature vector per edge
    The shock_event = {affected_pairs: {(src, tgt): delta_tariff}}
    """
    src, tgt = edge['source'], edge['target']
    
    flow_value = np.log(edge['flow_value'] + 1)
    import_pen_coeff = edge['import_pen_coeff']
    applied_tariff = tariff_data.get(src, tgt, year)
    
    # SHOCK INJECTION: delta tariff is non-zero only for affected pairs
    tariff_delta = shock_event['affected_pairs'].get((src, tgt), 0.0)
    
    hhi = edge['hhi']  # product concentration index
    is_domestic = 1.0 if src.country == tgt.country else 0.0
    
    return np.array([
        flow_value,          # log trade flow
        import_pen_coeff,    # import penetration coefficient
        applied_tariff,      # current applied tariff rate
        tariff_delta,        # SHOCK: tariff change (0 for non-shocked edges)
        hhi,                 # product concentration
        is_domestic,         # domestic vs international
    ])
    # Dimension: 6
    # Key: tariff_delta is non-zero only on edges directly hit by the tariff event
    # This separates shock origin from steady-state graph structure
```

### Step 4: Build shock events

```python
TARIFF_EVENTS = [
    {
        'name': 'US_232_Steel_Aluminum',
        'date': '2018-03-08',
        'products': ['HTS_7206', 'HTS_7207', ..., 'HTS_7601', 'HTS_7609'],  # full HTS lists
        'delta_tariff': {'steel': 0.25, 'aluminum': 0.10},
        'affected_importers': ['USA'],
        'affected_exporters': 'all',  # all countries exporting these products to USA
    },
    {
        'name': 'US_301_List1',
        'date': '2018-07-06',
        'products': [...],  # 818 HTS lines from Federal Register
        'delta_tariff': 0.25,
        'affected_importers': ['USA'],
        'affected_exporters': ['CHN'],  # China only
    },
    {
        'name': 'US_301_List2',
        'date': '2018-08-23',
        'products': [...],  # 284 HTS lines
        'delta_tariff': 0.25,
        'affected_importers': ['USA'],
        'affected_exporters': ['CHN'],
    },
    {
        'name': 'US_301_List3',
        'date': '2018-09-24',
        'products': [...],  # ~$200B
        'delta_tariff': 0.25,  # Note: started at 0.10, raised to 0.25 May 2019
        'affected_importers': ['USA'],
        'affected_exporters': ['CHN'],
    },
    {
        'name': 'EU_Retaliation',
        'date': '2018-06-22',
        'products': [...],  # EU Official Journal list
        'delta_tariff': 0.25,
        'affected_importers': ['DEU', 'FRA', 'ITA', ...],  # all EU members
        'affected_exporters': ['USA'],
    },
    {
        'name': 'UK_Global_Tariff',
        'date': '2021-01-01',
        'products': 'all',  # full UK schedule vs EU CET
        'delta_tariff': 'computed_from_uk_vs_eu_cet',  # can be positive or negative
        'affected_importers': ['GBR'],
        'affected_exporters': 'all',
    },
]
```

---

## 6. Model Architecture (Full Detail)

### 6.1 High-level structure

```
Input: Graph G = (V, E) with node features X ∈ R^{|V|×9}, edge features E ∈ R^{|edges|×6}
       Shock vector δτ (already injected into E[:, 3] = tariff_delta)
       Temporal sequence: [G_{t-7}, G_{t-6}, ..., G_t] (8 quarterly snapshots)

Stage 1: Feature Embedding
  - Node: Linear(9 → 128) → ReLU → node_embed ∈ R^{|V|×128}
  - Edge: Linear(6 → 64) → ReLU → edge_embed ∈ R^{|edges|×64}

Stage 2: GAT Layer 1 (multi-head attention, K=4 heads)
  - Input: node_embed + edge_embed
  - Attention: α_ij = softmax_j(LeakyReLU(a^T [W·h_i || W·h_j || W_e·e_ij]))
  - Aggregation: h_i' = ELU(Σ_{j∈N(i)} α_ij · W · h_j)
  - Multi-head concat → Linear(4×128 → 128)
  - Output: node_embed_1 ∈ R^{|V|×128}

Stage 3: GAT Layer 2 (same structure, now captures 2-hop neighborhood)
  - Output: node_embed_2 ∈ R^{|V|×128}

Stage 4: GRU Temporal Module
  - Process sequence [node_embed_2(t-7), ..., node_embed_2(t)] per node
  - GRU(input=128, hidden=256)
  - Output: h_temporal ∈ R^{|V|×256}

Stage 5: Multi-horizon MLP Output
  - Δp̂_3m  = MLP_1([256 → 128 → 64 → 1])(h_temporal)
  - Δp̂_6m  = MLP_2([256 → 128 → 64 → 1])(h_temporal)
  - Δp̂_12m = MLP_3([256 → 128 → 64 → 1])(h_temporal)
  - All three MLPs are independent (not shared weights)
```

### 6.2 Attention weight as learned economic coefficient

The attention weight α_ij is computed for each directed edge (supplier i → buyer j). It tells the model: "how much does a shock at node i matter for predicting the price change at node j?"

This is semantically equivalent to what the Leontief inverse coefficient (I−A)⁻¹_{ij} tells you analytically: the total (direct + indirect) input requirement of sector j from sector i, per unit of final demand.

**Key experiment — interpretability**: After training, extract attention weights for Layer 1 (direct effects). Compute Pearson correlation between α_ij and the corresponding Leontief inverse entry (I−A)⁻¹_{ij}. If the model has learned the supply chain structure, this correlation should be significantly positive and higher for Layer 1 than Layer 2.

This is one of the paper's main findings: **the model recovers the economic structure of shock transmission from data, without being told what the Leontief matrix is.**

### 6.3 Shock injection mechanism (important design choice)

The shock signal δτ is injected into edge features (not node features) for a specific reason:

- A tariff is fundamentally a change to a **bilateral trade relationship** — it affects imports along specific (exporter → importer) edges, not all edges from an exporting country.
- By injecting into edge features, the GAT attention mechanism can modulate how much the shock propagates along each edge based on the shock magnitude, the trade flow size, and the learned transmission weight.
- This is unlike a "global shock" where you would simply add a constant to all affected node features — that design loses the edge-specificity of real tariff policy.

### 6.4 Loss function

```python
def tspn_loss(pred_3m, pred_6m, pred_12m, true_3m, true_6m, true_12m, alpha_weights):
    """
    Weighted multi-horizon MSE + attention sparsity regularization
    """
    mse_3m  = F.mse_loss(pred_3m,  true_3m)
    mse_6m  = F.mse_loss(pred_6m,  true_6m)
    mse_12m = F.mse_loss(pred_12m, true_12m)
    
    # Sparsity prior: penalize diffuse attention (prefer focused transmission)
    l1_attn = alpha_weights.abs().mean()
    
    # Weight short-run more heavily (more training signal available)
    total_loss = (0.50 * mse_3m) + (0.30 * mse_6m) + (0.20 * mse_12m) + (0.01 * l1_attn)
    
    return total_loss
```

---

## 7. Training Protocol

### 7.1 Leave-One-Event-Out Cross-Validation (LOEO-CV)

With only 6 tariff events, a standard train/val/test split is inappropriate. Use LOEO-CV:

```
Round 1: Train on [232, 301-L2, 301-L3, EU-ret, UK-GT], Validate on [301-L1]
Round 2: Train on [301-L1, 301-L2, 301-L3, EU-ret, UK-GT], Validate on [232]
Round 3: Train on [232, 301-L1, 301-L3, EU-ret, UK-GT], Validate on [301-L2]
Round 4: Train on [232, 301-L1, 301-L2, EU-ret, UK-GT], Validate on [301-L3]
Round 5: Train on [232, 301-L1, 301-L2, 301-L3, UK-GT], Validate on [EU-ret]
Round 6: Train on [232, 301-L1, 301-L2, 301-L3, EU-ret], Validate on [UK-GT]

Report: mean ± std of RMSE, MAE, directional accuracy across 6 rounds
```

### 7.2 Training hyperparameters

```python
config = {
    'node_embed_dim': 128,
    'edge_embed_dim': 64,
    'gat_heads': 4,
    'gat_layers': 2,          # ablate: 1, 2, 3
    'gat_dropout': 0.3,
    'gru_hidden': 256,
    'gru_seq_len': 8,          # 8 quarterly snapshots = 2 years of history
    'mlp_hidden': [256, 128, 64],
    'lr': 1e-3,
    'lr_scheduler': 'cosine_annealing',
    'weight_decay': 1e-4,
    'epochs': 200,
    'early_stopping_patience': 20,
    'batch_size': 'full_graph',  # one graph per event, no mini-batching
    'loss_weights': [0.5, 0.3, 0.2, 0.01],  # 3m, 6m, 12m, attention L1
}
```

### 7.3 Data augmentation (to mitigate small-N problem)

Since you only have 6 training events, apply these augmentations during training:

- **Shock magnitude perturbation**: Add Gaussian noise σ=0.05 to shock magnitudes (simulates uncertainty in tariff implementation)
- **Temporal jitter**: Randomly shift the event date ±1 quarter (simulates announcement vs. implementation lag uncertainty)
- **Graph edge dropout**: Randomly drop 5% of low-weight edges (simulates measurement noise in trade flows)
- **Country sub-graph sampling**: Randomly mask one non-critical country per training step

---

## 8. Baseline Models

| Baseline | Description | Implementation | Why include |
|---|---|---|---|
| Static Leontief IO | Multiply shock vector by (I−A)⁻¹ matrix | Standard NumPy matrix inversion | Standard econ benchmark; TSPN must beat this on all metrics |
| VAR Panel Regression | Panel VAR on sector price series, tariff rate as exogenous covariate | statsmodels VAR | Standard econometric baseline |
| MLP (no graph) | Node features only, no message passing, no graph | PyTorch MLP | Proves graph structure adds predictive value |
| GCN (no attention) | Same as TSPN but mean aggregation instead of attention | PyG GCNConv | Ablation: proves attention over mean-pooling |
| GAT (no temporal) | GAT layers only, no GRU, single-snapshot | PyG GATConv | Ablation: proves temporal module adds value |
| GAT (no shock injection) | δτ not in edge features; tariff event as global node signal | Modified TSPN | Ablation: proves edge-level shock injection matters |
| TSPN-1hop | TSPN with only 1 GAT layer | Modified TSPN | Ablation: measures value of 2-hop propagation |
| TSPN-3hop | TSPN with 3 GAT layers | Modified TSPN | Ablation: measures whether 3-hop adds over 2-hop |

---

## 9. Evaluation Metrics

### 9.1 Predictive performance (per horizon: 3m, 6m, 12m)

```python
# Primary metrics
RMSE = sqrt(mean((pred - true)^2))   # main headline metric
MAE  = mean(|pred - true|)            # robust to outliers
R²   = 1 - SS_res / SS_tot           # variance explained

# Economic relevance metric
Directional_Accuracy = mean(sign(pred) == sign(true))
# Did we correctly predict whether prices rose or fell? Economically meaningful.

# Sector-level reporting
# Report RMSE broken down by: sector type (manufacturing vs services vs primary)
# and by cascade distance (directly hit sectors, 1-hop, 2-hop)
```

### 9.2 Interpretability metrics

```python
# Correlation of attention weights with Leontief inverse
# Compute after training for each fold

import scipy.stats

def compute_leontief_attention_correlation(alpha_ij_matrix, A_matrix):
    """
    alpha_ij_matrix: learned attention weights, shape (N, N)
    A_matrix: technical coefficients matrix from WIOD, shape (N, N)
    leontief_inverse: (I - A)^{-1}, shape (N, N)
    """
    leontief_inv = np.linalg.inv(np.eye(N) - A_matrix)
    
    # Flatten and filter non-zero trade pairs
    mask = alpha_ij_matrix > 0
    alpha_flat = alpha_ij_matrix[mask]
    leontief_flat = leontief_inv[mask]
    
    pearson_r, p_value = scipy.stats.pearsonr(alpha_flat, leontief_flat)
    spearman_r, _ = scipy.stats.spearmanr(alpha_flat, leontief_flat)
    
    return {'pearson_r': pearson_r, 'spearman_r': spearman_r, 'p_value': p_value}
```

### 9.3 Cascade depth analysis

```python
def measure_cascade_depth(model, shock_event, graph):
    """
    Measure how many hops away from shock origin nodes still show significant price effects.
    'Significant' = predicted |Δp| > 5% of original shock magnitude.
    """
    origin_nodes = get_directly_shocked_nodes(shock_event, graph)
    
    for hop in range(1, 6):
        hop_nodes = get_k_hop_neighbors(origin_nodes, graph, k=hop)
        avg_pred = model.predict(graph, shock_event)[hop_nodes].abs().mean()
        origin_pred = model.predict(graph, shock_event)[origin_nodes].abs().mean()
        
        attenuation_ratio = avg_pred / origin_pred
        print(f"Hop {hop}: attenuation ratio = {attenuation_ratio:.3f}")
        
        if attenuation_ratio < 0.05:
            print(f"Shock effectively dissipates at hop {hop}")
            return hop
```

### 9.4 Shock amplifier sector identification

```python
def identify_amplifier_sectors(model, graph):
    """
    Amplifier sector: node where attention centrality >> trade-flow centrality.
    These sectors transmit disproportionately more shock than their trade volume implies.
    """
    # Attention centrality: eigenvector centrality on attention-weight graph
    attention_graph = build_weighted_graph(model.get_attention_weights())
    attention_centrality = nx.eigenvector_centrality_numpy(attention_graph, weight='weight')
    
    # Trade-flow centrality: eigenvector centrality on raw trade graph
    trade_graph = build_weighted_graph(graph.edge_weights)
    trade_centrality = nx.eigenvector_centrality_numpy(trade_graph, weight='weight')
    
    # Amplification ratio: attention centrality / trade centrality
    for node in nodes:
        ratio = attention_centrality[node] / (trade_centrality[node] + 1e-9)
        node['amplification_ratio'] = ratio
    
    # Report top 10 amplifier sectors
    amplifiers = sorted(nodes, key=lambda x: x['amplification_ratio'], reverse=True)[:10]
    return amplifiers
```

---

## 10. Expected Paper Structure

```
§1 Introduction                    ~1.5 pages
   - Hook: tariff shocks cascade through supply chains
   - Gap: no existing method models this as a GNN problem
   - Contributions: (1) TSPN, (2) dataset+graph construction, (3) attention=Leontief finding, (4) cascade depth measurement

§2 Related Work                    ~1.0 page
   - IO models: Leontief 1951, Acemoglu et al. 2012 "Network Origins of Aggregate Fluctuations"
   - CGE models: Hertel 1997 GTAP framework
   - Tariff pass-through: Amiti, Redding, Weinstein 2019 (AER, on US-China tariffs)
   - GNNs for economic/financial networks
   - Temporal GNNs: T-GCN (Zhao et al. 2019), TGN (Rossi et al. 2020)
   - Gap: no GNN applied to tariff shock propagation

§3 Data and Graph Construction     ~2.0 pages
   - WIOD description and node/edge derivation
   - HS6 → ISIC concordance and tariff aggregation methodology
   - Graph statistics: degree distribution, centrality, sparsity
   - Tariff event descriptions and shock vector construction
   - Validation label construction

§4 Model: TSPN                     ~3.0 pages  [CORE CONTRIBUTION]
   - Formal graph definition and notation
   - Feature embedding layer
   - Multi-head GAT with edge-feature shock injection
   - GRU temporal module
   - Multi-horizon MLP output head
   - Loss function and training details
   - LOEO-CV protocol

§5 Experiments                     ~3.0 pages
   - Main results table (all baselines, all horizons)
   - Ablation study (Table 2: each component removed one at a time)
   - Horizon comparison: which component degrades 12m more than 3m?
   - Sector-type breakdown: manufacturing vs services vs primary
   - Statistical significance testing across LOEO-CV folds

§6 Interpretability and Economic Insights  ~1.5 pages  [CORE CONTRIBUTION]
   - Attention weight vs Leontief correlation (main finding)
   - Cascade depth analysis: effective shock range = N hops
   - Amplifier sector identification: Table of top sectors by amplification ratio
   - Policy implication: systemic cost of targeting amplifier sectors

§7 Conclusion                      ~0.5 page
   - Restate 4 contributions
   - Limitations: WIOD lag, few training events, annual resolution, no firm-level heterogeneity
   - Future work: firm-level graph, commodity price integration, strategic retaliation modeling

Total: ~12.5 pages (standard journal length)
Supplementary: concordance tables, full hyperparameter grid, additional ablations
```

---

## 11. Implementation Stack

```
Language:         Python 3.9+
Graph ML:         PyTorch Geometric (PyG) — GATConv, GRUConv
Deep learning:    PyTorch 2.0+
Data:             Pandas, NumPy, SciPy
Graph analysis:   NetworkX (centrality computation, visualization)
Trade data:       comtradeapicall (Comtrade API), wbdata (World Bank)
IO matrices:      Custom WIOD parser (see parse_wiod.py)
Visualization:    Matplotlib, Seaborn
Compute:          Single GPU sufficient (2,500 nodes, moderate graph density)
Environment:      conda env, requirements.txt to be maintained
```

---

## 12. Implementation Roadmap

### Phase 1 — Data collection and parsing (weeks 1–3)
- [ ] Download WIOD 2016 release (all year files + socioeconomic accounts)
- [ ] Write WIOD parser: extract (country, sector) → (country, sector) flow matrices per year
- [ ] Pull Section 232 and Section 301 product codes from Federal Register
- [ ] Pull WITS tariff rates for 2015–2021 for all country pairs
- [ ] Pull BLS PPI and Eurostat PPI for validation labels
- [ ] Build HS6 → ISIC concordance lookup table

### Phase 2 — Graph construction (weeks 4–5)
- [ ] Build static graphs for 2000–2016 from WIOD
- [ ] Compute all node features (Leontief BL, import pen, PPI lags)
- [ ] Compute all edge features (flow, tariff, HHI)
- [ ] Build shock vectors for all 6 tariff events
- [ ] Extend graph 2017–2021 using Comtrade flows + WITS tariffs
- [ ] Convert to PyG `HeteroData` or `Data` format
- [ ] Verify graph statistics match WIOD documentation

### Phase 3 — Baseline models (week 6)
- [ ] Implement Leontief IO shock model (NumPy matrix inverse)
- [ ] Implement panel VAR regression (statsmodels)
- [ ] Implement MLP-only baseline (no graph)
- [ ] Record baseline RMSE/MAE/direction accuracy as lower-bound targets

### Phase 4 — TSPN implementation (weeks 7–10)
- [ ] Build PyG dataset class with temporal graph sequence
- [ ] Implement feature embedding layers
- [ ] Implement GAT layers with edge feature concatenation (shock injection)
- [ ] Implement GRU temporal module
- [ ] Implement multi-horizon MLP output heads
- [ ] Implement LOEO-CV training loop
- [ ] Log training with W&B or TensorBoard

### Phase 5 — Ablations and interpretability (weeks 11–13)
- [ ] Run all 7 ablation variants
- [ ] Extract attention weights post-training
- [ ] Compute attention vs Leontief correlation per fold
- [ ] Cascade depth measurement per event
- [ ] Amplifier sector identification and ranking
- [ ] Statistical significance (95% CI across LOEO-CV folds)

### Phase 6 — Paper writing (weeks 14–18)
- [ ] §3 Data (write first — most concrete and factual)
- [ ] §4 Model (formalize notation, write pseudocode)
- [ ] §5 Results (tables and figures)
- [ ] §6 Interpretability (economic implications)
- [ ] §1 Introduction and §2 Related Work (write last)
- [ ] §7 Conclusion + limitations + future work
- [ ] Supplementary appendix

---

## 13. Target Journals and Venues

**Primary targets:**
- *Journal of International Economics* — top field journal, methodologically rigorous empirical + modeling papers
- *Review of Economics and Statistics* — strong quantitative methods emphasis
- *Nature Computational Science* — high impact if interpretability finding is strong

**Secondary targets:**
- *Journal of Economic Dynamics and Control* — accepts model-focused economic papers
- *Economic Modelling* — more accessible for novel modeling approaches
- *Economic Letters* — short version for rapid publication

**ML venue (parallel or prior submission):**
- NeurIPS or ICML workshop on ML for Economic Policy
- ICLR workshop on Machine Learning for Financial Markets

---

## 14. Key References to Review

- Acemoglu, D., Carvalho, V.M., Ozdaglar, A., & Tahbaz-Salehi, A. (2012). The network origins of aggregate fluctuations. *Econometrica*, 80(5), 1977–2016.
- Amiti, M., Redding, S.J., & Weinstein, D.E. (2019). The impact of the 2018 tariffs on prices and welfare. *Journal of Economic Perspectives*, 33(4), 187–210.
- Timmer, M.P., et al. (2015). An illustrated user guide to the World Input–Output Database. *Review of International Economics*, 23(3), 575–605.
- Leontief, W. (1951). *The Structure of the American Economy*. Oxford University Press.
- Hertel, T.W. (1997). *Global Trade Analysis: Modeling and Applications*. Cambridge University Press.
- Zhao, L., et al. (2019). T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction. *IEEE Transactions on Intelligent Transportation Systems*, 21(9), 3848–3858.
- Rossi, E., et al. (2020). Temporal Graph Networks for Deep Learning on Dynamic Graphs. *ICML 2020 Workshop on Graph Representation Learning*.
- Veličković, P., et al. (2018). Graph Attention Networks. *ICLR 2018*.
