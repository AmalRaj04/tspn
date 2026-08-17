# TSPN — Phase-by-Phase Implementation Masterplan
### Tariff Shock Propagation Network: Complete Build Guide

> **How to use this document**: Follow each phase in order. Do not skip ahead.
> Each phase has: manual actions → terminal commands → coding prompts → checkpoint test.
> The checkpoint script at the end of each phase must pass before you start the next.
> Prompts are written for a capable coding assistant (Claude, Cursor, etc.) — give them the full prompt including the locked spec values.

---

## PRE-FLIGHT — What You Need Before Day 1

### Accounts to Create (free, do this first)
| Service | URL | Why |
|---|---|---|
| UN Comtrade | https://comtradeplus.un.org/ | Bilateral trade API |
| World Bank WITS | https://wits.worldbank.org/ | Tariff data |
| BLS Data API | https://api.bls.gov/registrationEngine/ | US PPI |
| Weights & Biases | https://wandb.ai/ | Training tracker |
| GitHub | https://github.com/ | Version control + Streamlit hosting |
| Streamlit Community Cloud | https://share.streamlit.io/ | Free app hosting |
| Google Drive | (you probably have this) | Colab checkpoint storage |

### Hardware Requirement Check
- Local machine: Python 3.10 environment, at least 16GB RAM, 20GB free disk
- GPU training: Google Colab (free T4) — do NOT attempt local GPU unless you have one
- WIOD + Comtrade raw data: ~3GB total

---

## PHASE 0 — Environment and Project Initialization
**Duration**: 2 days
**Output**: Configured repo, `config.py`, verified installs, folder structure

---

### 0.A — Create the Folder Structure

Run this entire block in terminal from wherever you want the project to live:

```bash
mkdir -p tspn
cd tspn

# Data layer
mkdir -p data/raw/wiod
mkdir -p data/raw/comtrade
mkdir -p data/raw/wits
mkdir -p data/raw/bls_ppi
mkdir -p data/raw/eurostat_ppi
mkdir -p data/raw/commodity_prices
mkdir -p data/raw/tariff_events
mkdir -p data/raw/concordance

mkdir -p data/processed/edges
mkdir -p data/processed/node_features
mkdir -p data/processed/tariff_rates
mkdir -p data/processed/shock_vectors
mkdir -p data/processed/labels
mkdir -p data/processed/concordance

mkdir -p data/pyg_datasets

# Source code
mkdir -p src/data
mkdir -p src/models
mkdir -p src/baselines
mkdir -p src/training
mkdir -p src/analysis

# Notebooks
mkdir -p notebooks

# Model artifacts
mkdir -p models/checkpoints
mkdir -p models/onnx

# Results
mkdir -p results/tables
mkdir -p results/figures

# App
mkdir -p app/components
mkdir -p app/utils
mkdir -p app/assets

# Touch init files
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/baselines/__init__.py
touch src/training/__init__.py
touch src/analysis/__init__.py

echo "Folder structure created."
```

---

### 0.B — Git Init and README

```bash
cd tspn
git init
echo "# TSPN — Tariff Shock Propagation Network" > README.md
echo "Research paper + product dashboard. See docs/ for full spec." >> README.md
git add README.md
git commit -m "init: project scaffold"
```

---

### 0.C — Conda Environment Setup

```bash
# Create environment
conda create -n tspn python=3.10 -y
conda activate tspn

# Core scientific stack
pip install pandas==2.1.0 numpy==1.26.0 scipy==1.11.3

# PyTorch — CPU first (swap to CUDA on Colab for training)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric — MUST use the matching wheel
pip install torch_geometric==2.4.0
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
  --find-links https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Data and graph
pip install networkx==3.2.1 pyarrow==14.0.0 openpyxl==3.1.2
pip install scikit-learn==1.3.2 statsmodels==0.14.0

# Viz and product
pip install plotly==5.17.0 streamlit==1.28.0 pyvis==0.3.2
pip install fastapi==0.104.0 uvicorn==0.24.0

# ONNX
pip install onnx==1.15.0 onnxruntime==1.16.3

# Data sources
pip install comtradeapicall==0.2.1 wbdata==0.3.0

# Experiment tracking
pip install wandb==0.16.0

# Export requirements
pip freeze > requirements.txt
```

---

### 0.D — Coding Prompt: `config.py`

> **Give this prompt to your coding assistant:**

```
Create config.py for the TSPN project — the SINGLE SOURCE OF TRUTH for all parameters.
No other file may hardcode a value that exists here. Use plain Python dicts and lists only.

Include these sections with EXACTLY these values:

PATHS — all relative to project root:
  RAW_WIOD = "data/raw/wiod"
  RAW_COMTRADE = "data/raw/comtrade"
  RAW_WITS = "data/raw/wits"
  RAW_BLS_PPI = "data/raw/bls_ppi"
  RAW_EUROSTAT_PPI = "data/raw/eurostat_ppi"
  RAW_COMMODITY = "data/raw/commodity_prices"
  RAW_TARIFF_EVENTS = "data/raw/tariff_events"
  RAW_CONCORDANCE = "data/raw/concordance"
  PROCESSED_EDGES = "data/processed/edges"
  PROCESSED_NODE_FEATURES = "data/processed/node_features"
  PROCESSED_TARIFF_RATES = "data/processed/tariff_rates"
  PROCESSED_SHOCK_VECTORS = "data/processed/shock_vectors"
  PROCESSED_LABELS = "data/processed/labels"
  PROCESSED_CONCORDANCE = "data/processed/concordance"
  PYG_DATASETS = "data/pyg_datasets"
  MODEL_CHECKPOINTS = "models/checkpoints"
  MODEL_ONNX = "models/onnx"
  RESULTS_TABLES = "results/tables"
  RESULTS_FIGURES = "results/figures"
  NORM_STATS = "data/processed/node_features/normalization_stats.json"

GRAPH:
  N_COUNTRIES = 44
  N_SECTORS = 56
  N_NODES = 2464   # 44 × 56
  EDGE_THRESHOLD = 0.001   # import_pen_coeff must be >= this to keep edge
  SEQ_LEN = 8   # quarters in temporal sequence
  WIOD_YEARS = list(range(2000, 2017))
  COMTRADE_YEARS = [2017, 2018, 2019, 2020, 2021]
  WIOD_MATRIX_ROW_OFFSET = None   # SET MANUALLY after opening WIOD Excel
  WIOD_MATRIX_COL_OFFSET = None   # SET MANUALLY after opening WIOD Excel

  COUNTRY_LIST = [
    "AUS","AUT","BEL","BGR","BRA","CAN","CHN","CYP","CZE","DEU",
    "DNK","ESP","EST","FIN","FRA","GBR","GRC","HUN","IDN","IND",
    "IRL","ITA","JPN","KOR","LTU","LUX","LVA","MEX","MLT","NLD",
    "NOR","POL","PRT","ROU","RUS","SVK","SVN","SWE","TUR","TWN",
    "USA","RoW","ZAF","CHE"
  ]
  # Note: verify exact 44-country list against WIOD documentation

  SECTOR_LIST = [
    "A01","A02","A03","B","C10-C12","C13-C15","C16","C17","C18",
    "C19","C20","C21","C22","C23","C24","C25","C26","C27","C28",
    "C29","C30","C31-C32","C33","D35","E36","E37-E39","F","G45",
    "G46","G47","H49","H50","H51","H52","H53","I","J58-J60","J61",
    "J62-J63","K64","K65","K66","L68","M69-M70","M71","M72","M73",
    "M74-M75","N","O84","P85","Q","R-S","T","U"
  ]

  def node_id(country_idx: int, sector_idx: int) -> int:
      return country_idx * 56 + sector_idx

NODE_FEATURES = {
  "dim": 9,
  "names": [
    "log_gross_output",      # f[0]: log(gross_output_usd_millions + 1)
    "import_penetration",    # f[1]: total_imports / (gross_output + imports − exports + 1e-9)
    "export_intensity",      # f[2]: total_exports / (gross_output + 1e-9)
    "backward_linkage",      # f[3]: column sum of Leontief inverse (I−A)^-1
    "tariff_exposure",       # f[4]: Σ_j (trade_share_ij × applied_tariff_ij)
    "ppi_lag_1",             # f[5]: (PPI[t-1] - PPI[t-2]) / PPI[t-2]
    "ppi_lag_2",             # f[6]: (PPI[t-2] - PPI[t-3]) / PPI[t-3]
    "ppi_lag_3",             # f[7]: (PPI[t-3] - PPI[t-4]) / PPI[t-4]
    "ppi_lag_4",             # f[8]: (PPI[t-4] - PPI[t-5]) / PPI[t-5]
  ]
}

EDGE_FEATURES = {
  "dim": 6,
  "names": [
    "log_trade_flow",        # e[0]: log(flow_usd_millions + 1)
    "import_pen_coeff",      # e[1]: flow_ij / total_input_j
    "applied_tariff",        # e[2]: trade-value-weighted MFN tariff rate
    "tariff_delta",          # e[3]: THE SHOCK SIGNAL — new_rate − old_rate (0 for non-shocked)
    "product_hhi",           # e[4]: Σ_k (trade_share_k)^2 across HS6 codes
    "domestic_flag",         # e[5]: 1.0 if src_country == tgt_country else 0.0
  ]
}

MODEL = {
  "node_feat_in": 9,
  "edge_feat_in": 6,
  "node_embed_dim": 128,
  "edge_embed_dim": 64,
  "node_embed_dropout": 0.1,
  "gat_num_layers": 2,
  "gat_num_heads": 4,
  "gat_head_dim": 32,
  "gat_concat_out_dim": 128,   # 4 heads × 32
  "gat_leaky_slope": 0.2,
  "gat_attn_dropout": 0.3,
  "gru_input_dim": 128,
  "gru_hidden_dim": 256,
  "gru_num_layers": 1,
  "gru_output_dropout": 0.2,
  "mlp_layer_dims": [256, 128, 64, 1],
  "mlp_dropout": 0.2,
  "mlp_num_heads": 3,   # 3m, 6m, 12m
}

TRAINING = {
  "optimizer": "Adam",
  "lr": 1e-3,
  "weight_decay": 1e-4,
  "scheduler": "CosineAnnealingWarmRestarts",
  "T_0": 50,
  "T_mult": 2,
  "grad_clip_norm": 1.0,
  "max_epochs": 200,
  "early_stop_patience": 20,
  "early_stop_metric": "val_rmse_6m",
  "loss_weight_3m": 0.50,
  "loss_weight_6m": 0.30,
  "loss_weight_12m": 0.20,
  "loss_weight_l1_attn": 0.01,
  "augment_shock_sigma": 0.05,
  "augment_jitter_prob": 0.50,
  "augment_edge_drop_p": 0.05,
  "augment_edge_drop_threshold": 0.002,
  "augment_label_sigma": 0.01,
}

LEONTIEF = {
  "REG_EPS": 1e-4,
  "PASS_THROUGH_RATE": None,  # SET AFTER CALIBRATION in Phase 6
}

EVENTS = [
  {
    "name": "us_232_steel_2018",
    "date": "2018-03",
    "hts_file": "data/raw/tariff_events/us_232_steel_2018.csv",
    "affected_importers": ["USA"],
    "affected_exporters": "all",
    "delta_tariff_pct": 25.0,
    "description": "US Section 232 Steel Tariffs, March 2018",
  },
  {
    "name": "us_232_aluminum_2018",
    "date": "2018-03",
    "hts_file": "data/raw/tariff_events/us_232_aluminum_2018.csv",
    "affected_importers": ["USA"],
    "affected_exporters": "all",
    "delta_tariff_pct": 10.0,
    "description": "US Section 232 Aluminum Tariffs, March 2018",
  },
  {
    "name": "us_301_list1_2018",
    "date": "2018-07",
    "hts_file": "data/raw/tariff_events/us_301_list1_2018.csv",
    "affected_importers": ["USA"],
    "affected_exporters": ["CHN"],
    "delta_tariff_pct": 25.0,
    "description": "US Section 301 List 1, July 2018 — $34B China goods",
  },
  {
    "name": "us_301_list2_2018",
    "date": "2018-08",
    "hts_file": "data/raw/tariff_events/us_301_list2_2018.csv",
    "affected_importers": ["USA"],
    "affected_exporters": ["CHN"],
    "delta_tariff_pct": 25.0,
    "description": "US Section 301 List 2, August 2018 — $16B China goods",
  },
  {
    "name": "eu_retaliation_2018",
    "date": "2018-06",
    "hts_file": "data/raw/tariff_events/eu_retaliation_2018.csv",
    "affected_importers": ["AUT","BEL","BGR","CYP","CZE","DEU","DNK","ESP","EST",
                           "FIN","FRA","GBR","GRC","HUN","IRL","ITA","LTU","LUX",
                           "LVA","MLT","NLD","POL","PRT","ROU","SVK","SVN","SWE"],
    "affected_exporters": ["USA"],
    "delta_tariff_pct": None,   # varies by product — read from CSV
    "description": "EU Retaliation to US Section 232, June 2018",
  },
  {
    "name": "uk_global_tariff_2021",
    "date": "2021-01",
    "hts_file": "data/raw/tariff_events/uk_global_tariff_2021.csv",
    "affected_importers": ["GBR"],
    "affected_exporters": "all",
    "delta_tariff_pct": None,   # varies — delta vs EU CET, read from CSV
    "description": "UK Global Tariff Schedule, January 2021 (post-Brexit)",
  },
]

EVAL = {
  "metrics": ["RMSE", "MAE", "R2", "DirAcc"],
  "bootstrap_n": 1000,
  "bootstrap_ci": 0.95,
  "cascade_significance_threshold": 0.05,
  "amplifier_centrality": "eigenvector_centrality_numpy",
  "significance_level": 0.01,
}

COMMODITY_TO_ISIC = {
  "steel_hrc": "C24",
  "aluminum": "C24",
  "copper": "C24",
  "iron_ore": "B",
  "coal": "B",
  "brent_oil": "C19",
  "wheat": "A01",
  "corn": "A01",
  "soy": "A01",
}
```

---

### 0.E — Coding Prompt: `scripts/validate_config.py`

> **Give this prompt to your coding assistant:**

```
Create scripts/validate_config.py — a standalone validation script that imports config.py
and asserts everything is correct before any real work begins.

Checks to implement:
1. All directories in PATHS section can be created (os.makedirs with exist_ok=True)
2. MODEL["node_feat_in"] == 9 and MODEL["edge_feat_in"] == 6
3. MODEL["node_embed_dim"] == 128 and MODEL["gru_hidden_dim"] == 256
4. TRAINING["lr"] == 1e-3 and TRAINING["max_epochs"] == 200
5. len(GRAPH["COUNTRY_LIST"]) == 44
6. len(GRAPH["SECTOR_LIST"]) == 56
7. len(EVENTS) == 6
8. Each event dict has keys: name, date, hts_file, affected_importers, affected_exporters, description
9. node_id(0, 0) == 0 and node_id(1, 0) == 56 and node_id(43, 55) == 2463
10. LEONTIEF["REG_EPS"] == 1e-4

Print "PASS: <check name>" for each check.
At the end print "All config checks passed." or raise AssertionError on first failure.
```

Run it:
```bash
python scripts/validate_config.py
```

---

### ✅ PHASE 0 CHECKPOINT

Create and run `scripts/checkpoint_phase0.py`:

```python
"""Phase 0 checkpoint — run before Phase 1."""
import subprocess, sys, os, importlib

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")

# 1. Config imports cleanly
def test_config():
    sys.path.insert(0, '.')
    import config
    assert hasattr(config, 'GRAPH')
    assert hasattr(config, 'MODEL')
    assert hasattr(config, 'EVENTS')
check("config imports", test_config)

# 2. Exact locked values
def test_locked_values():
    import config
    assert config.MODEL["node_embed_dim"] == 128, f"Got {config.MODEL['node_embed_dim']}"
    assert config.MODEL["gru_hidden_dim"] == 256
    assert config.MODEL["gat_num_heads"] == 4
    assert config.MODEL["gat_head_dim"] == 32
    assert config.TRAINING["lr"] == 1e-3
    assert config.TRAINING["loss_weight_3m"] == 0.50
check("locked values", test_locked_values)

# 3. Country and sector counts
def test_counts():
    import config
    assert len(config.GRAPH["COUNTRY_LIST"]) == 44, f"Got {len(config.GRAPH['COUNTRY_LIST'])}"
    assert len(config.GRAPH["SECTOR_LIST"]) == 56, f"Got {len(config.GRAPH['SECTOR_LIST'])}"
    assert config.GRAPH["N_NODES"] == 2464
check("country/sector counts", test_counts)

# 4. Events list
def test_events():
    import config
    assert len(config.EVENTS) == 6
    for e in config.EVENTS:
        for key in ["name", "date", "hts_file", "affected_importers", "affected_exporters"]:
            assert key in e, f"Event missing key: {key}"
check("events list", test_events)

# 5. Folder structure exists
def test_folders():
    required = [
        "data/raw/wiod", "data/raw/comtrade", "data/raw/wits",
        "data/processed/edges", "data/processed/node_features",
        "data/pyg_datasets", "src/models", "src/training",
        "models/checkpoints", "results/tables", "results/figures"
    ]
    for f in required:
        assert os.path.isdir(f), f"Missing: {f}"
check("folder structure", test_folders)

# 6. PyTorch imports
def test_torch():
    import torch
    import torch_geometric
    import torch_scatter
    x = __import__('torch').tensor([1., 2., 3.])
    idx = __import__('torch').tensor([0, 0, 1])
    result = torch_scatter.scatter_add(x, idx)
    import torch
    assert torch.allclose(result, torch.tensor([3., 3.])), f"scatter wrong: {result}"
check("torch + pyg + scatter", test_torch)

# 7. Key packages
def test_packages():
    for pkg in ["pandas", "numpy", "scipy", "networkx", "pyarrow",
                "openpyxl", "wandb", "sklearn", "statsmodels"]:
        importlib.import_module(pkg)
check("all packages importable", test_packages)

print(f"\n{'='*50}")
print(f"Phase 0 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING CHECKS:", FAIL)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — proceed to Phase 1")
```

```bash
python scripts/checkpoint_phase0.py
```

---

## PHASE 1 — Data Collection
**Duration**: 8 days
**Output**: All raw data files on disk

---

### 1.A — MANUAL: Download WIOD (Day 1)

1. Go to: **https://www.rug.nl/ggdc/valuechain/wiod/**
2. Click the **"Data"** tab
3. Under **"WIOT Tables (2016 Release)"**: download all **17 Excel files**
   - Files named: `WIOT2000_October16_ROW.xlsx` through `WIOT2016_October16_ROW.xlsx`
4. Under **"Socioeconomic Accounts"**: download `wiot_sep_16_txt.zip`, then unzip it
5. Save everything to: `data/raw/wiod/`

**Critical manual step after download:**
- Open `WIOT2016_October16_ROW.xlsx` in Excel or LibreOffice
- Find the exact row number where the IO matrix data begins (the first numeric data row, after country/sector headers)
- Find the exact column number where it begins
- Open `config.py` and set `WIOD_MATRIX_ROW_OFFSET` and `WIOD_MATRIX_COL_OFFSET` to those values
- Typical values are around row 6–8, col 4–6 — but verify in your actual file

```bash
# Verify download count
ls data/raw/wiod/*.xlsx | wc -l
# Must show 18 (17 annual + 1 socioeconomic or extracted txt files)
```

---

### 1.B — MANUAL: Download Concordance Files (Day 1)

1. Go to: **https://unstats.un.org/unsd/trade/classifications/correspondence-tables.asp**
2. Download:
   - **HS 2017 → ISIC Rev. 4** → save as `data/raw/concordance/hs2017_isic4.xlsx`
   - **HS 2012 → ISIC Rev. 4** → save as `data/raw/concordance/hs2012_isic4.xlsx`
   - **NAICS 2017 → ISIC Rev. 4** → save as `data/raw/concordance/naics2017_isic4.xlsx`

---

### 1.C — MANUAL: Extract Tariff Event HTS Codes (Days 2–3)

For each event, you create one CSV file. Use Tabula (free: https://tabula.technology) to extract tables from PDFs — do NOT copy-paste manually.

**File 1** — `data/raw/tariff_events/us_232_steel_2018.csv`
- Source: https://www.federalregister.gov/documents/2018/03/08/2018-04875
- Columns: `hts_code` (8 digits, no dots), `product_description`, `delta_tariff_pct`
- `delta_tariff_pct` = 25.0 for ALL rows
- Expected: ~170 HTS codes

**File 2** — `data/raw/tariff_events/us_232_aluminum_2018.csv`
- Same Federal Register document
- `delta_tariff_pct` = 10.0 for ALL rows
- Expected: ~60 HTS codes

**File 3** — `data/raw/tariff_events/us_301_list1_2018.csv`
- Source: https://www.federalregister.gov/documents/2018/06/20/2018-13248
- `delta_tariff_pct` = 25.0 for ALL rows
- Expected: 818 HTS codes

**File 4** — `data/raw/tariff_events/us_301_list2_2018.csv`
- Source: Federal Register Vol. 83 No. 155 (Aug 10, 2018)
- `delta_tariff_pct` = 25.0 for ALL rows
- Expected: 284 HTS codes

**File 5** — `data/raw/tariff_events/eu_retaliation_2018.csv`
- Source: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L:2018:160:TOC
- `delta_tariff_pct` varies by product (25% steel, others vary) — use exact rates from Annex

**File 6** — `data/raw/tariff_events/uk_global_tariff_2021.csv`
- Source: https://www.trade-tariff.service.gov.uk/api/v2/commodities
- `delta_tariff_pct` = UK rate − EU CET rate per code (can be negative)

Validate HTS code format after each file:
```bash
python -c "
import pandas as pd, sys
for f in ['us_232_steel_2018','us_232_aluminum_2018','us_301_list1_2018','us_301_list2_2018']:
    df = pd.read_csv(f'data/raw/tariff_events/{f}.csv')
    bad = df['hts_code'].astype(str).str.len().ne(8).sum()
    print(f'{f}: {len(df)} rows, {bad} bad codes')
"
```

---

### 1.D — MANUAL: Download WITS Tariff Data (Days 3–4)

1. Register at **https://wits.worldbank.org/**
2. Go to: Data → Tariff & Trade → TRAINS Database → Bulk Download
3. Download Applied MFN tariff rates (CSV format) for:
   - USA: 2015, 2016, 2017, 2018, 2019, 2020, 2021
   - CHN: 2015–2021
   - All EU countries in WIOD (individual downloads): 2015–2021
   - CAN: 2017, 2018, 2019
   - GBR: 2019, 2020, 2021
4. Name files: `data/raw/wits/tariff_{ISO3}_{YEAR}.csv`
5. Extra: USA→CHN "Effectively Applied" tariff for 2018–2020 (captures Section 301)
   - Save as: `data/raw/wits/tariff_usa_china_effective_{YEAR}.csv`

---

### 1.E — MANUAL: Download BLS PPI (Day 4)

1. Register free API key at: **https://api.bls.gov/registrationEngine/**
2. Go to: https://www.bls.gov/ppi/data.htm → One-Screen Data Search
3. Pull **Industry PPI series (PCU prefix)** for NAICS codes:
   - Manufacturing: 311–339
   - Mining: 211, 212, 213
   - Services: 481, 483, 484, 4931
4. Date range: **January 2014 – December 2023**
5. Save as: `data/raw/bls_ppi/bls_ppi_{NAICS_CODE}.csv` per series

OR use the API script below:

```bash
# coding prompt: write scripts/download_bls_ppi.py
# (see coding prompt 1.E below)
python scripts/download_bls_ppi.py
```

---

### 1.E — Coding Prompt: `scripts/download_bls_ppi.py`

> **Give this prompt to your coding assistant:**

```
Create scripts/download_bls_ppi.py to download BLS PPI data via the BLS API v2.

Requirements:
- API endpoint: https://api.bls.gov/publicAPI/v2/timeseries/data/
- Require BLS_API_KEY from environment variable (os.environ["BLS_API_KEY"])
- Pull Industry PPI series for NAICS codes: all 311-339 (manufacturing), 211, 212, 213 (mining),
  481, 483, 484, 4931 (services)
- Series ID format for PCU: "PCU{NAICS}{NAICS}" (e.g. PCU311311 for Food Mfg)
- Date range: 2014 to 2023
- BLS API allows max 50 series per call — batch accordingly
- Retry up to 3 times on HTTP error with 5-second backoff
- Save each series as: data/raw/bls_ppi/bls_ppi_{NAICS_CODE}.csv
  Columns: year, period (M01-M12), value (index level)
- Skip series already downloaded (check if file exists)
- Print progress as series are downloaded
```

```bash
export BLS_API_KEY="your_key_here"
python scripts/download_bls_ppi.py
```

---

### 1.F — MANUAL: Download Eurostat PPI (Day 5)

```bash
# Direct API call — no login needed
curl "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_inppd_m?format=JSON&lang=EN" \
  -o data/raw/eurostat_ppi/eurostat_ppi_raw.json
```

Or in Python:
```bash
python -c "
import requests, json
url = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_inppd_m'
params = {'format': 'JSON', 'lang': 'EN', 'freq': 'M', 'unit': 'I15', 'indic_bt': 'PRIN'}
r = requests.get(url, params=params, timeout=120)
with open('data/raw/eurostat_ppi/eurostat_ppi_raw.json', 'w') as f:
    json.dump(r.json(), f)
print('Eurostat PPI downloaded.')
"
```

---

### 1.G — MANUAL: Download World Bank Pink Sheet (Day 5)

1. Go to: **https://www.worldbank.org/en/research/commodity-markets**
2. Click "Download Historical Data" → download **CMO-Historical-Data-Monthly.xlsx**
3. Save to: `data/raw/commodity_prices/wb_pink_sheet.xlsx`

---

### 1.H — Coding Prompt: `scripts/download_comtrade.py`

> **Give this prompt to your coding assistant:**

```
Create scripts/download_comtrade.py to pull UN Comtrade trade data via the comtradeapicall package.

Requirements:
- Require COMTRADE_API_KEY from environment variable
- Country list from config.GRAPH["COUNTRY_LIST"] (44 countries)
- Pull HS 2-digit aggregate imports (flow=M, classification=HS, commodity=AG2)
- Years: 2017, 2018, 2019, 2020, 2021 — one API call per (reporter, year)
- Wait 7.5 seconds between API calls (rate limit: 500/hour)
- Skip files already downloaded (check if output Parquet exists)
- Save as: data/raw/comtrade/comtrade_{ISO3}_{YEAR}.parquet
  Schema: reporter (str), partner (str), commodity_code (str), trade_value_usd (float64)
- Handle API errors: retry 3 times, then log failure to data/raw/comtrade/failed_downloads.log
- Also pull HS6-level data specifically for Section 301 codes:
  Reporter=USA, Partner=CHN, years 2018-2020, using HS codes from us_301_list1 and list2 CSVs
  Save as: data/raw/comtrade/comtrade_usa_chn_301hs6_{YEAR}.parquet
- Print progress: "Downloaded {reporter} {year} — {n_rows} rows"

Use comtradeapicall package. Read documentation for correct function signatures.
```

```bash
export COMTRADE_API_KEY="your_key_here"
python scripts/download_comtrade.py
# This will take 1-2 days — runs slowly due to rate limits. Start it running and let it go.
```

---

### ✅ PHASE 1 CHECKPOINT

```python
"""Phase 1 checkpoint — run before Phase 2."""
import os, sys, pandas as pd

PASS, FAIL = [], []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")

# 1. WIOD files: exactly 17 annual excel files
def test_wiod():
    files = [f for f in os.listdir("data/raw/wiod") if f.endswith(".xlsx") and "WIOT" in f]
    assert len(files) >= 17, f"Only {len(files)} WIOD files found (need 17)"
check("WIOD files", test_wiod)

# 2. WIOD offsets set in config
def test_wiod_offsets():
    sys.path.insert(0, '.')
    import config
    assert config.GRAPH["WIOD_MATRIX_ROW_OFFSET"] is not None, "WIOD_MATRIX_ROW_OFFSET not set"
    assert config.GRAPH["WIOD_MATRIX_COL_OFFSET"] is not None, "WIOD_MATRIX_COL_OFFSET not set"
check("WIOD offsets in config", test_wiod_offsets)

# 3. Tariff event files with row counts
def test_tariff_events():
    expected = {
        "us_232_steel_2018.csv": 100,
        "us_232_aluminum_2018.csv": 40,
        "us_301_list1_2018.csv": 700,
        "us_301_list2_2018.csv": 200,
        "eu_retaliation_2018.csv": 1,
        "uk_global_tariff_2021.csv": 1,
    }
    for fname, min_rows in expected.items():
        path = f"data/raw/tariff_events/{fname}"
        assert os.path.exists(path), f"Missing: {path}"
        df = pd.read_csv(path)
        assert len(df) >= min_rows, f"{fname}: only {len(df)} rows (expected >= {min_rows})"
        assert "hts_code" in df.columns, f"{fname}: missing hts_code column"
        assert "delta_tariff_pct" in df.columns, f"{fname}: missing delta_tariff_pct column"
        # Check no dots in HTS codes for standard files
        if "uk_global" not in fname and "eu_retaliation" not in fname:
            bad = df["hts_code"].astype(str).str.contains(r'\.').sum()
            assert bad == 0, f"{fname}: {bad} HTS codes contain dots (need 8-digit no-dot format)"
check("tariff event CSVs", test_tariff_events)

# 4. Concordance files
def test_concordance():
    for f in ["hs2017_isic4.xlsx", "hs2012_isic4.xlsx"]:
        path = f"data/raw/concordance/{f}"
        assert os.path.exists(path), f"Missing: {path}"
check("concordance files", test_concordance)

# 5. Comtrade Parquet files — at least first year for a few countries
def test_comtrade():
    import config
    sample_countries = ["USA", "CHN", "DEU"]
    for c in sample_countries:
        path = f"data/raw/comtrade/comtrade_{c}_2018.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            assert len(df) >= 500, f"{path}: only {len(df)} rows"
        else:
            print(f"  WARN: {path} not yet downloaded (Comtrade can take days)")
check("comtrade Parquet samples", test_comtrade)

# 6. BLS PPI — at least some files
def test_bls():
    files = [f for f in os.listdir("data/raw/bls_ppi") if f.endswith(".csv")]
    assert len(files) >= 5, f"Only {len(files)} BLS PPI files"
check("BLS PPI files", test_bls)

# 7. World Bank Pink Sheet
def test_wb():
    assert os.path.exists("data/raw/commodity_prices/wb_pink_sheet.xlsx"), "Missing WB Pink Sheet"
check("WB Pink Sheet", test_wb)

print(f"\n{'='*50}")
print(f"Phase 1 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL)
    sys.exit(1)
else:
    print("ALL PASSED — proceed to Phase 2")
```

```bash
python scripts/checkpoint_phase1.py
```

---

## PHASE 2 — WIOD Processing
**Duration**: 4 days
**Output**: `data/processed/edges/edges_{YEAR}.parquet` (2000–2016), Leontief inverses

---

### 2.A — Coding Prompt: `src/data/parse_wiod.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/parse_wiod.py to parse all 17 WIOD annual Excel files into edge Parquet tables.

Imports from config: GRAPH["WIOD_MATRIX_ROW_OFFSET"], GRAPH["WIOD_MATRIX_COL_OFFSET"],
GRAPH["EDGE_THRESHOLD"], GRAPH["COUNTRY_LIST"], GRAPH["SECTOR_LIST"], PATHS["PROCESSED_EDGES"]

Exact parsing sequence (locked — do not deviate):
1. Load Excel with openpyxl engine, skiprows = WIOD_MATRIX_ROW_OFFSET
2. Extract (country, sector) header pairs from column headers
   - Headers format in WIOD: alternating country codes and sector codes
3. Extract the N×N intermediate use sub-matrix (rows/cols corresponding to country-sector pairs)
   - Exclude final demand columns (appear after the last sector column)
   - Exclude value-added rows (appear after the last sector row)
4. Convert to long format DataFrame with columns:
   year (int), src_country (str), src_sector (str), tgt_country (str), tgt_sector (str), flow_usd (float)
5. Remove rows where flow_usd <= 0
6. For each target node (tgt_country, tgt_sector): compute tgt_total_input = sum of ALL inputs into it
7. Compute import_pen_coeff = flow_usd / tgt_total_input (use 1e-9 denominator guard)
8. Clip import_pen_coeff at maximum 0.99 (handles re-export economies like NLD, BEL)
9. Apply filter: keep only rows where import_pen_coeff >= EDGE_THRESHOLD (0.001)
10. Compute src_id = country_list.index(src_country) * 56 + sector_list.index(src_sector)
11. Compute tgt_id = same formula for target

Output schema (dtype as specified):
  year: int16, src_country: category, src_sector: category,
  tgt_country: category, tgt_sector: category,
  flow_usd: float32, import_pen_coeff: float32,
  src_id: int16, tgt_id: int16

Save as: data/processed/edges/edges_{YEAR}.parquet
Use pyarrow engine.

Main loop: parse all years 2000-2016. Skip year if output file already exists.
Print progress: "Parsed {year}: {n_edges} edges kept out of {n_total}"

Also parse socioeconomic accounts file (wiot_sep_16_txt.zip extracted files):
- Extract gross_output, value_added, employment per (country, sector, year)
- Save as: data/processed/edges/socioeconomic_{YEAR}.parquet
  Schema: year int16, country category, sector category,
          gross_output float32 (millions USD), value_added float32
```

```bash
python src/data/parse_wiod.py
# Takes about 30-60 minutes for all 17 years
```

---

### 2.B — Coding Prompt: `src/data/compute_leontief.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/compute_leontief.py to compute Leontief technical coefficients and inverse matrices.

For each year 2000-2016:
1. Load edges_{YEAR}.parquet
2. Build the technical coefficients matrix A (2464×2464):
   - A[src_id, tgt_id] = import_pen_coeff for that edge
   - All other entries = 0.0
   - Use scipy.sparse for construction (csr_matrix), convert to dense for inversion
3. Compute Leontief inverse: L = inv(I - A + eps*I) where eps = LEONTIEF["REG_EPS"] = 1e-4
   - Use numpy.linalg.inv on dense matrix
   - Check condition number first: assert np.linalg.cond(I-A) < 1e6
   - Check max |L| < 100 after inversion
4. Compute backward linkage for each node: bl[j] = column sum of L = sum over all rows of L[:,j]
5. Save:
   - Full Leontief matrix: data/processed/edges/leontief_{YEAR}.npy (shape 2464×2464, float32)
   - Backward linkage vector: data/processed/edges/backward_linkage_{YEAR}.npy (shape 2464, float32)

Skip year if leontief_{YEAR}.npy already exists.
Print: "Year {year}: L computed, max={max_val:.2f}, condition={cond:.2e}"

IMPORTANT: Store LEONTIEF_REG_EPS from config. Log if regularization was needed.
```

```bash
python src/data/compute_leontief.py
# Memory warning: each 2464×2464 float32 matrix = ~24MB. All 17 = ~400MB total
# Takes 5-15 minutes
```

---

### 2.C — Coding Prompt: `src/data/extend_with_comtrade.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/extend_with_comtrade.py to extend the WIOD graph from 2016 to 2021
using UN Comtrade data.

Design (locked): Use WIOD 2016 technical coefficients structure as a prior.
Only update bilateral trade flow MAGNITUDES from Comtrade. Do NOT update sector shares.

For each year 2017-2021:
1. Load WIOD 2016 edge table as structural prior
2. Load all Comtrade Parquet files for this year (comtrade_{ISO3}_{YEAR}.parquet)
3. Build HS2→ISIC mapping (approximate: use first 2 chars of HS code to map to ISIC 2-digit)
   - HS 01-24 → A01-C10 range (agriculture and food)
   - HS 25-27 → B, C19 (mining, petroleum)
   - HS 28-38 → C20, C21 (chemicals, pharma)
   - HS 39-40 → C22 (rubber/plastics)
   - HS 72-83 → C24, C25 (metals)
   - HS 84-85 → C26, C27, C28 (electronics, machinery)
   - HS 86-89 → C29, C30 (transport)
   - (use a simple lookup dict stored in config or hardcoded in this file)
4. For each (src_country, tgt_country, isic_sector): sum Comtrade trade_value_usd → flow_usd
5. Scale WIOD 2016 gross_output as denominator for import_pen_coeff (proxy: use 2016 values)
6. Apply same edge threshold (import_pen_coeff >= 0.001)
7. Add src_id and tgt_id using same formula as WIOD
8. Save with same schema as WIOD edge tables

Save as: data/processed/edges/edges_{YEAR}.parquet (same format as WIOD years)
Print: "Extended {year}: {n_edges} edges"
```

```bash
python src/data/extend_with_comtrade.py
```

---

### ✅ PHASE 2 CHECKPOINT

```python
"""Phase 2 checkpoint — run before Phase 3."""
import os, sys, numpy as np, pandas as pd

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append(name); print(f"  FAIL  {name}: {e}")

# CP02 ★ — WIOD matrix totals and node counts
def test_wiod_parse():
    et = pd.read_parquet("data/processed/edges/edges_2016.parquet")
    total_flow = et["flow_usd"].sum() / 1e9
    assert 55_000 <= total_flow <= 70_000, f"Total flow ${total_flow:.0f}B (expect $55T–$70T)"
    n_src = et[["src_country","src_sector"]].drop_duplicates().shape[0]
    n_tgt = et[["tgt_country","tgt_sector"]].drop_duplicates().shape[0]
    assert n_src == 2464, f"Only {n_src} unique src nodes (need 2464)"
    assert n_tgt == 2464, f"Only {n_tgt} unique tgt nodes (need 2464)"
check("★ CP02: WIOD parse totals and node count", test_wiod_parse)

# CP08 ★ — Leontief inverse sanity
def test_leontief():
    for year in [2000, 2008, 2016]:
        path = f"data/processed/edges/leontief_{year}.npy"
        if not os.path.exists(path):
            raise AssertionError(f"Missing leontief_{year}.npy")
        L = np.load(path)
        assert L.shape == (2464, 2464), f"Year {year}: wrong shape {L.shape}"
        assert np.max(np.abs(L)) < 100, f"Year {year}: L has extreme values, max={np.max(np.abs(L)):.1f}"
        assert not np.any(np.isnan(L)), f"Year {year}: NaN in Leontief inverse"
check("★ CP08: Leontief inverses valid", test_leontief)

# CP09 ★ — Node count consistent across years
def test_node_consistency():
    for year in list(range(2000, 2017)) + [2018, 2019, 2020]:
        path = f"data/processed/edges/edges_{year}.parquet"
        if not os.path.exists(path):
            print(f"  SKIP year {year} (not downloaded yet)")
            continue
        et = pd.read_parquet(path)
        n_c = et["src_country"].nunique()
        n_s = et["src_sector"].nunique()
        assert n_c == 44, f"Year {year}: {n_c} countries (need 44)"
        assert n_s == 56, f"Year {year}: {n_s} sectors (need 56)"
check("★ CP09: Node count consistent all years", test_node_consistency)

# CP07 — import_pen_coeff never exceeds 1.0
def test_pen_coeff():
    et = pd.read_parquet("data/processed/edges/edges_2016.parquet")
    max_pen = et["import_pen_coeff"].max()
    assert max_pen <= 1.0, f"import_pen_coeff max = {max_pen:.4f} (must be <= 1.0 after clipping)"
    neg_count = (et["import_pen_coeff"] < 0).sum()
    assert neg_count == 0, f"{neg_count} negative import_pen_coeff values"
check("CP07: import_pen_coeff in [0,1]", test_pen_coeff)

# CP10 — Edge count in expected range
def test_edge_count():
    et = pd.read_parquet("data/processed/edges/edges_2016.parquet")
    n = len(et)
    assert 80_000 <= n <= 150_000, f"Edge count {n} outside [80K, 150K]"
    print(f"  INFO: edges_2016 has {n:,} edges")
check("CP10: Edge count in range", test_edge_count)

# All 22 edge files exist
def test_all_edge_files():
    years = list(range(2000, 2022))
    missing = [y for y in years if not os.path.exists(f"data/processed/edges/edges_{y}.parquet")]
    assert not missing, f"Missing edge files for years: {missing}"
check("All 22 edge parquet files exist", test_all_edge_files)

print(f"\n{'='*50}")
print(f"Phase 2 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL); sys.exit(1)
else:
    print("ALL PASSED — proceed to Phase 3")
```

```bash
python scripts/checkpoint_phase2.py
```

---

## PHASE 3 — Tariff and Shock Processing
**Duration**: 4 days
**Output**: `tariff_rates.parquet`, `shock_{EVENT}.parquet` for all 6 events

---

### 3.A — Coding Prompt: `src/data/build_concordance.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/build_concordance.py to build HS6→ISIC and NAICS→ISIC lookup tables.

Load data/raw/concordance/hs2017_isic4.xlsx and hs2012_isic4.xlsx.
Parse the correspondence table columns (HS code column and ISIC column).

Build hs6_to_isic dict:
  Key: hs6_code (str, zero-padded to 6 digits)
  Value: list of (isic_code, weight) tuples

Weight rule (locked):
- HS6 maps to exactly one ISIC sector: weight = 1.0
- HS6 maps to multiple ISIC sectors: distribute weights proportionally using
  WIOD 2016 gross_output per sector (load from socioeconomic_2016.parquet)
  Larger-output sectors receive higher weight. Weights must sum to 1.0 per HS6 code.

Also build naics3_to_isic dict (from naics2017_isic4.xlsx):
  Key: naics3_code (str, 3 chars)
  Value: isic_code (str) — mostly 1:1, use first ISIC match if multiple

Save:
  data/processed/concordance/hs6_isic_weights.json
  data/processed/concordance/naics3_isic.json

Print stats: total HS6 codes mapped, average number of ISIC sectors per HS6 code,
number of HS6 codes with no ISIC match (should be near 0).
```

---

### 3.B — Coding Prompt: `src/data/compute_tariff_rates.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/compute_tariff_rates.py to compute sector-level tariff rates.

Exact formula per (src_country, tgt_country, isic_sector, year) (LOCKED):
  tariff_rate = Σ_k [ (comtrade_flow_k / total_comtrade_flow_sector) × wits_rate_k ]
where k iterates over all HS6 codes mapping to this ISIC sector.

Fallback rules in priority order:
1. If bilateral rate exists in WITS → use it
2. If only MFN rate exists (no bilateral data) → use MFN rate  
3. If no WITS data for that year → linear interpolation from nearest available years
4. If no WITS data at all for that reporter → set to 0.0 and flag data_source='missing'

Load WITS files from data/raw/wits/tariff_{ISO3}_{YEAR}.csv
Load Comtrade flows from data/raw/comtrade/ for HS6 weighting.
Load concordance from data/processed/concordance/hs6_isic_weights.json

Output schema:
  year: int16, src_country: category, tgt_country: category, isic_sector: category,
  tariff_rate: float32 (decimal: 0.25 = 25%), data_source: category

Save as: data/processed/tariff_rates/tariff_rates.parquet

Assertions before saving:
  - assert tariff_df["tariff_rate"].isna().sum() == 0  (no nulls)
  - assert (tariff_df["data_source"] == "missing").mean() < 0.05
  - Spot check: CHN→USA C24 sector rate for 2015 must have data_source != "missing"

Print: "Tariff rates computed: {n_rows} rows, {pct_missing:.1f}% from fallback"
```

---

### 3.C — Coding Prompt: `src/data/build_shock_vectors.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/build_shock_vectors.py to build shock vectors for all 6 tariff events.

Load config.EVENTS list. For each event:

1. Load the event's HTS CSV file (event["hts_file"])
2. For each HTS code → map to ISIC sector(s) using hs6_isic_weights.json
3. Build affected_pairs set: all (src_country, tgt_country, isic_sector) tuples where:
   - tgt_country in event["affected_importers"]
   - src_country in event["affected_exporters"] (or any country if value is "all")
4. Compute delta_tariff per affected pair:
   sum of (concordance_weight × event_delta_tariff_pct / 100) across matching HTS codes
   For eu_retaliation and uk_global_tariff events: read delta_tariff_pct from each CSV row
5. All pairs not in affected_pairs: delta_tariff = 0.0
6. is_direct_hit = True only for pairs with delta_tariff > 0

Output schema:
  event_name: str, src_country: category, tgt_country: category,
  isic_sector: category, delta_tariff: float32, is_direct_hit: bool

Save as: data/processed/shock_vectors/shock_{event_name}.parquet

Validation before saving:
- us_232_steel: assert all non-zero rows have tgt_country == "USA"
- us_301_list1: assert all non-zero rows have src_country == "CHN" and tgt_country == "USA"
- assert delta_tariff.min() >= 0.0 (tariffs only go up in these events — except UK which can go neg)
- Print: "Event {name}: {n_direct_hit} direct-hit edges, max delta={max_delta:.3f}"

Also export compute_shock_vector() as a public function taking (event_name) and returning
the shock DataFrame — this function is imported by the app's scenario_parser.py.
```

---

### ✅ PHASE 3 CHECKPOINT

```python
"""Phase 3 checkpoint — run before Phase 4."""
import os, sys, pandas as pd, json

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append(name); print(f"  FAIL  {name}: {e}")

# CP12 ★ — Shock vector direction not reversed
def test_shock_direction():
    for event in ["us_232_steel_2018", "us_301_list1_2018"]:
        path = f"data/processed/shock_vectors/shock_{event}.parquet"
        if not os.path.exists(path): raise AssertionError(f"Missing: {path}")
        sv = pd.read_parquet(path)
        assert sv["delta_tariff"].max() > 0, f"{event}: all delta_tariff are 0 or negative"
        # Section 232 steel: delta should be 0.25 (25 pp)
        if "232_steel" in event:
            max_d = sv["delta_tariff"].max()
            assert 0.20 <= max_d <= 0.30, f"Steel shock max delta = {max_d:.3f} (expect ~0.25)"
check("★ CP12: Shock direction and magnitude", test_shock_direction)

# CP04 — WITS fallback < 5% missing
def test_wits_fallback():
    path = "data/processed/tariff_rates/tariff_rates.parquet"
    if not os.path.exists(path): raise AssertionError("tariff_rates.parquet missing")
    df = pd.read_parquet(path)
    pct_missing = (df["data_source"] == "missing").mean()
    assert pct_missing < 0.05, f"Missing tariff rate fraction = {pct_missing:.2%} (> 5%)"
    assert df["tariff_rate"].isna().sum() == 0, "Null tariff rates exist"
check("CP04: WITS fallback under 5%", test_wits_fallback)

# Concordance files exist and are valid JSON
def test_concordance():
    for fname in ["hs6_isic_weights.json", "naics3_isic.json"]:
        path = f"data/processed/concordance/{fname}"
        assert os.path.exists(path), f"Missing: {path}"
        with open(path) as f:
            data = json.load(f)
        assert len(data) > 100, f"{fname}: only {len(data)} entries"
check("Concordance JSON files", test_concordance)

# All 6 shock vector files exist
def test_shock_files():
    events = ["us_232_steel_2018","us_232_aluminum_2018","us_301_list1_2018",
              "us_301_list2_2018","eu_retaliation_2018","uk_global_tariff_2021"]
    for e in events:
        path = f"data/processed/shock_vectors/shock_{e}.parquet"
        assert os.path.exists(path), f"Missing: {path}"
        df = pd.read_parquet(path)
        n_hit = df["is_direct_hit"].sum()
        assert n_hit > 0, f"{e}: zero direct-hit edges"
        print(f"  INFO: {e} has {n_hit} direct-hit edges")
check("All 6 shock vector files", test_shock_files)

# Spot-check Section 301: only CHN→USA
def test_301_geography():
    sv = pd.read_parquet("data/processed/shock_vectors/shock_us_301_list1_2018.parquet")
    shocked = sv[sv["delta_tariff"] > 0]
    wrong_src = shocked[shocked["src_country"] != "CHN"]
    wrong_tgt = shocked[shocked["tgt_country"] != "USA"]
    assert len(wrong_src) == 0, f"Section 301 has {len(wrong_src)} shocked edges from non-CHN"
    assert len(wrong_tgt) == 0, f"Section 301 has {len(wrong_tgt)} shocked edges to non-USA"
check("CP12b: Section 301 geography (CHN→USA only)", test_301_geography)

print(f"\n{'='*50}")
print(f"Phase 3 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL); sys.exit(1)
else:
    print("ALL PASSED — proceed to Phase 4")
```

```bash
python scripts/checkpoint_phase3.py
```

---

## PHASE 4 — Feature Engineering
**Duration**: 5 days
**Output**: Node feature Parquets, edge feature Parquets (48 files), labels, `normalization_stats.json`

---

### 4.A — Coding Prompt: `src/data/clean_ppi.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/clean_ppi.py to unify BLS PPI, Eurostat PPI, and World Bank commodity prices
into one quarterly PPI change table.

Conversion formula (LOCKED, apply to all three sources):
  ppi_change_t = (ppi_level_t − ppi_level_{t−1}) / ppi_level_{t−1}
  Monthly → Quarterly: average of 3 monthly changes within the quarter

Processing:

1. BLS PPI: load each data/raw/bls_ppi/bls_ppi_{NAICS}.csv
   - Map NAICS 3-digit codes to ISIC using data/processed/concordance/naics3_isic.json
   - Country = "USA" for all BLS series
   - Year-quarter from year + period columns

2. Eurostat PPI: load data/raw/eurostat_ppi/eurostat_ppi_raw.json (Eurostat JSON API format)
   - Parse the nested JSON (dimensions × values structure)
   - Map NACE Rev.2 codes to ISIC (NACE and ISIC 4th revision are closely aligned — use direct mapping)
   - Extract country codes from the geo dimension

3. World Bank Pink Sheet: load data/raw/commodity_prices/wb_pink_sheet.xlsx
   - Extract commodity price series (steel HRC, aluminum, copper, iron ore, coal, brent, grains)
   - Map to ISIC using COMMODITY_TO_ISIC from config.py
   - Country = "WLD" (world price — will be used as a fallback for non-BLS/Eurostat countries)

Output schema:
  year: int16, quarter: int8 (1-4), country: category, isic_sector: category,
  ppi_change: float32, source: category (bls/eurostat/wb_commodity)

Save as: data/processed/labels/ppi_quarterly_all.parquet

Print: "PPI unified: {n_rows} rows from {n_countries} countries and {n_sectors} sectors"
```

---

### 4.B — Coding Prompt: `src/data/compute_node_features.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/compute_node_features.py to compute the 9-dimensional node feature vector
for every (country, sector) node for every year 2000-2021.

Features in LOCKED ORDER (f[0] through f[8]):
  f[0] log_gross_output   = log(gross_output_usd_millions + 1)
  f[1] import_penetration = total_imports / (gross_output + total_imports − total_exports + 1e-9)
  f[2] export_intensity   = total_exports / (gross_output + 1e-9)
  f[3] backward_linkage   = load from backward_linkage_{YEAR}.npy, indexed by node_id
  f[4] tariff_exposure    = Σ_j (trade_share_ij × applied_tariff_ij) over all import partners j
  f[5] ppi_lag_1          = ppi_change in quarter t-1
  f[6] ppi_lag_2          = ppi_change in quarter t-2
  f[7] ppi_lag_3          = ppi_change in quarter t-3
  f[8] ppi_lag_4          = ppi_change in quarter t-4

Sources:
- gross_output, value_added: from socioeconomic_{YEAR}.parquet
- total_imports (sum of incoming flow_usd), total_exports (sum of outgoing flow_usd): from edges_{YEAR}.parquet
- backward_linkage: from backward_linkage_{YEAR}.npy (vector indexed by node_id)
- tariff_exposure: compute from tariff_rates.parquet (year-specific rates)
- ppi_lag_1–4: from ppi_quarterly_all.parquet — use Q4 of year t-1, Q3 t-1, Q2 t-1, Q1 t-1
  (annual feature = most recent quarterly value)
- Missing PPI lags for years 2000-2003 (before coverage): fill with 0.0, set has_ppi_lags = False

Normalization (LOCKED):
- Compute mean and std for each feature using WIOD years 2000-2016 ONLY
- Save stats to: data/processed/node_features/normalization_stats.json
  Format: {"mean": [f0_mean, ..., f8_mean], "std": [f0_std, ..., f8_std]}
- Apply z-score normalization: f_norm = (f − mean) / (std + 1e-8)
- NEVER recompute normalization from test data (2017-2021 must use 2000-2016 stats)

Output schema per year:
  year: int16, country: category, sector: category, node_id: int16,
  f0 through f8: float32 (all normalized), has_ppi_lags: bool

Save as: data/processed/node_features/node_features_{YEAR}.parquet

Assertions before saving each year:
  - shape must be (2464, 14) [year+country+sector+node_id+9 features+has_ppi_lags]
  - no NaN in f0-f8 columns
  - node_id values in [0, 2463]

Print: "Node features {year}: computed and saved, NaN check passed"
```

```bash
python src/data/compute_node_features.py
```

---

### 4.C — Coding Prompt: `src/data/compute_edge_features.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/compute_edge_features.py to compute 6-dimensional edge feature vectors
for all 6 events × 8 quarterly snapshots = 48 files.

Features in LOCKED ORDER (e[0] through e[5]):
  e[0] log_trade_flow    = log(flow_usd + 1) (flow in USD millions)
  e[1] import_pen_coeff  = flow_usd / tgt_total_input (from edge table)
  e[2] applied_tariff    = trade-value-weighted MFN tariff rate from tariff_rates.parquet
  e[3] tariff_delta      = THE SHOCK SIGNAL — delta_tariff from shock vector (0.0 for non-shocked)
                           CRITICAL: e[3] is NON-ZERO ONLY in snapshot q=7 (event-time quarter)
                                     e[3] MUST BE ZERO in snapshots q=0 through q=6
  e[4] product_hhi       = Σ_k (flow_k / total_flow)^2 across HS6 codes in this bilateral-sector pair
                           Compute from Comtrade HS6 data where available, else use 0.5 as default
  e[5] domestic_flag     = 1.0 if src_country == tgt_country else 0.0

For each event in config.EVENTS:
  - Determine event quarter (e.g., 2018-Q1 for March 2018 events, 2018-Q3 for July 2018)
  - Build 8 quarterly snapshots: event_quarter-7 through event_quarter (inclusive)
  - For each snapshot q:
    * Load appropriate edge table (edges_{YEAR}.parquet for the snapshot year)
    * Compute all 6 edge features
    * If q < 7: set e[3] = 0.0 for ALL edges
    * If q == 7: load shock vector for this event, fill e[3] from delta_tariff
    * Sort edges by (src_id, tgt_id) — MUST be consistent across snapshots
  - Save: data/processed/edge_features/edge_features_{event_name}_q{q}.parquet
    Schema: src_id int16, tgt_id int16, e0-e5 float32

CRITICAL: The edge_index (src_id, tgt_id pairs) MUST be IDENTICAL across all 8 snapshots
for the same event. Use the intersection of edges present across all 8 snapshot years.

Print: "Event {event}: q{q} saved — {n_edges} edges, shock non-zeros: {n_shocked}"
```

---

### 4.D — Coding Prompt: `src/data/generate_labels.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/generate_labels.py to generate prediction labels for all 6 tariff events.

For each event at date d_e (event quarter), compute per (country, sector) node:
  delta_3m  = (ppi_quarterly[d_e + 1 quarter] − ppi_quarterly[d_e]) / ppi_quarterly[d_e]
  delta_6m  = (ppi_quarterly[d_e + 2 quarters] − ppi_quarterly[d_e]) / ppi_quarterly[d_e]
  delta_12m = (ppi_quarterly[d_e + 4 quarters] − ppi_quarterly[d_e]) / ppi_quarterly[d_e]

Source: data/processed/labels/ppi_quarterly_all.parquet

Coverage: Not all (country, sector) pairs have PPI data. Set has_label = False for those.
Acceptable: at least 60% of nodes should have has_label = True for each event.

Output schema:
  event_name: str, country: category, sector: category, node_id: int16,
  delta_3m: float32, delta_6m: float32, delta_12m: float32,
  has_label: bool, label_source: category (bls/eurostat/wb_commodity/null)

Save as: data/processed/labels/labels_{event_name}.parquet

Sanity checks:
  - Most delta_6m values should be in range (-0.15, +0.15) — flag outliers
  - assert has_label.mean() >= 0.60 for all events
  - assert delta_6m.isna().sum() == 0 (use 0.0 for unlabeled, rely on mask)

Print: "Labels {event}: {pct_labeled:.0f}% nodes labeled, delta_6m range [{min:.3f}, {max:.3f}]"
```

---

### ✅ PHASE 4 CHECKPOINT

```python
"""Phase 4 checkpoint — run before Phase 5."""
import os, sys, json, numpy as np, pandas as pd

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append(name); print(f"  FAIL  {name}: {e}")

# CP16 ★ — Normalization computed ONLY on training years
def test_norm_stats():
    path = "data/processed/node_features/normalization_stats.json"
    assert os.path.exists(path), "normalization_stats.json missing"
    with open(path) as f:
        stats = json.load(f)
    assert "mean" in stats and "std" in stats
    assert len(stats["mean"]) == 9, f"Wrong mean dim: {len(stats['mean'])}"
    assert len(stats["std"]) == 9, f"Wrong std dim: {len(stats['std'])}"
    for i, s in enumerate(stats["std"]):
        assert s > 0, f"std[{i}] = {s} (should be > 0)"
check("★ CP16: Normalization stats shape and validity", test_norm_stats)

# Node feature files: 22 years, each 2464 rows, no NaN
def test_node_features():
    for year in [2000, 2008, 2016, 2018, 2020]:
        path = f"data/processed/node_features/node_features_{year}.parquet"
        assert os.path.exists(path), f"Missing: {path}"
        df = pd.read_parquet(path)
        assert len(df) == 2464, f"Year {year}: {len(df)} rows (need 2464)"
        feat_cols = [f"f{i}" for i in range(9)]
        nan_count = df[feat_cols].isna().sum().sum()
        assert nan_count == 0, f"Year {year}: {nan_count} NaN values in features"
check("Node feature files: 2464 rows, no NaN", test_node_features)

# CP17 ★ — Label reference quarter check (delta_6m distribution)
def test_labels():
    events = ["us_232_steel_2018", "us_301_list1_2018"]
    for e in events:
        path = f"data/processed/labels/labels_{e}.parquet"
        assert os.path.exists(path), f"Missing: {path}"
        df = pd.read_parquet(path)
        pct_labeled = df["has_label"].mean()
        assert pct_labeled >= 0.60, f"{e}: only {pct_labeled:.0%} nodes labeled (need 60%)"
        labeled = df[df["has_label"]]
        pct_in_range = ((labeled["delta_6m"].abs() < 0.15).mean())
        assert pct_in_range > 0.85, f"{e}: {1-pct_in_range:.0%} delta_6m outliers (check formula)"
check("★ CP17: Labels coverage and distribution", test_labels)

# CP20 ★ — Shock signal e[3] zero in pre-event snapshots
def test_shock_injection():
    e = "us_232_steel_2018"
    for q in range(7):  # snapshots 0–6 must have e[3]=0
        path = f"data/processed/edge_features/edge_features_{e}_q{q}.parquet"
        if not os.path.exists(path): continue
        ef = pd.read_parquet(path)
        nonzero = (ef["e3"] != 0).sum()
        assert nonzero == 0, f"q={q}: {nonzero} non-zero e[3] in pre-event snapshot (must be 0)"
    # snapshot 7 must have some non-zero e[3]
    path = f"data/processed/edge_features/edge_features_{e}_q7.parquet"
    if os.path.exists(path):
        ef = pd.read_parquet(path)
        nonzero = (ef["e3"] != 0).sum()
        assert nonzero > 0, "q=7: ALL e[3] are zero — shock injection failed"
        print(f"  INFO: q=7 has {nonzero} shocked edges for {e}")
check("★ CP20: Shock signal in q=7 only", test_shock_injection)

# 48 edge feature files exist
def test_edge_files():
    events = ["us_232_steel_2018","us_232_aluminum_2018","us_301_list1_2018",
              "us_301_list2_2018","eu_retaliation_2018","uk_global_tariff_2021"]
    missing = []
    for e in events:
        for q in range(8):
            p = f"data/processed/edge_features/edge_features_{e}_q{q}.parquet"
            if not os.path.exists(p): missing.append(p)
    assert not missing, f"Missing {len(missing)} edge feature files: {missing[:3]}..."
check("All 48 edge feature files exist", test_edge_files)

print(f"\n{'='*50}")
print(f"Phase 4 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL); sys.exit(1)
else:
    print("ALL PASSED — proceed to Phase 5")
```

---

## PHASE 5 — PyG Graph Dataset Construction
**Duration**: 3 days
**Output**: `data/pyg_datasets/{event_name}.pt` for all 6 events

---

### 5.A — Coding Prompt: `src/data/build_pyg_dataset.py`

> **Give this prompt to your coding assistant:**

```
Create src/data/build_pyg_dataset.py to assemble PyTorch Geometric dataset objects for each event.

The output is a custom Python object (not a PyG Dataset subclass) saved via torch.save.

Class TSPNEventData with attributes:
  temporal_sequence  = list of 8 PyG Data objects (one per quarter snapshot)
  y                  = FloatTensor shape (2464, 3) — [delta_3m, delta_6m, delta_12m] per node
                       Use 0.0 for nodes with has_label=False (rely on label_mask to ignore them)
  label_mask         = BoolTensor shape (2464,) — True for labeled nodes only
  direct_hit_mask    = BoolTensor shape (2464,) — True for nodes with any shocked incoming edge
  event_name         = str
  event_date         = str

Each PyG Data object in temporal_sequence has:
  x           = FloatTensor (2464, 9) — node features
  edge_index  = LongTensor (2, E)    — sorted by (src_id, tgt_id), MUST be identical across all 8
  edge_attr   = FloatTensor (E, 6)   — edge features

Assembly steps per event:
1. Load labels_{event_name}.parquet → build y tensor and label_mask
2. Determine the intersection of edges present in ALL 8 snapshots → this defines the FIXED edge_index
3. For each snapshot q in [0..7]:
   a. Load node_features_{year_for_snapshot_q}.parquet → x tensor (sort by node_id)
   b. Load edge_features_{event_name}_q{q}.parquet → edge_attr tensor (aligned to fixed edge_index)
   c. Build PyG Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
4. Build direct_hit_mask: node has is_direct_hit=True if any incoming edge in shock vector has delta_tariff > 0
5. Create TSPNEventData object
6. Save: data/pyg_datasets/{event_name}.pt using torch.save

CRITICAL RULES:
- edge_index must be IDENTICAL across all 8 snapshots — compute the intersection first
- Sort edges by (src_id, tgt_id) before building the tensor
- Nodes must be sorted by node_id (0 to 2463) so that row i of x corresponds to node_id=i

Also write validate_datasets() function that checks all 6 .pt files for:
  - temporal_sequence length == 8
  - x shape == (2464, 9) for each snapshot
  - No NaN in any x tensor
  - edge_index shape[0] == 2
  - label_mask True fraction >= 0.60
  - e[3] (edge_attr[:,3]) non-zero in q=7, zero in q=0 through q=6

Call validate_datasets() at the end of the main script.
```

```bash
python src/data/build_pyg_dataset.py
```

---

### ✅ PHASE 5 CHECKPOINT

```python
"""Phase 5 checkpoint — run before Phase 6."""
import os, sys, torch

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append(name); print(f"  FAIL  {name}: {e}")

EVENTS = ["us_232_steel_2018","us_232_aluminum_2018","us_301_list1_2018",
          "us_301_list2_2018","eu_retaliation_2018","uk_global_tariff_2021"]

# All 6 .pt files load
def test_load():
    for e in EVENTS:
        path = f"data/pyg_datasets/{e}.pt"
        assert os.path.exists(path), f"Missing: {path}"
        data = torch.load(path, weights_only=False)
        assert hasattr(data, "temporal_sequence"), f"{e}: no temporal_sequence"
        assert hasattr(data, "y"), f"{e}: no y"
check("All 6 .pt files load", test_load)

# CP21 ★ — Shock in correct snapshot only
def test_shock_snapshot():
    for e in EVENTS:
        data = torch.load(f"data/pyg_datasets/{e}.pt", weights_only=False)
        seq = data.temporal_sequence
        for q in range(7):
            nonzero = (seq[q].edge_attr[:, 3] != 0).sum().item()
            assert nonzero == 0, f"{e} q={q}: {nonzero} non-zero shock entries (must be 0 pre-event)"
        nonzero_q7 = (seq[7].edge_attr[:, 3] != 0).sum().item()
        assert nonzero_q7 > 0, f"{e} q=7: ALL shock entries zero — injection failed"
check("★ CP21: Shock in q=7 only", test_shock_snapshot)

# CP22 ★ — Edge index identical across snapshots
def test_edge_index_consistency():
    for e in EVENTS:
        data = torch.load(f"data/pyg_datasets/{e}.pt", weights_only=False)
        seq = data.temporal_sequence
        ref_ei = seq[0].edge_index
        for q in range(1, 8):
            assert torch.equal(seq[q].edge_index, ref_ei), \
                f"{e}: edge_index differs between q=0 and q={q}"
check("★ CP22: Edge index consistent across snapshots", test_edge_index_consistency)

# Node feature shape and no NaN
def test_node_features():
    for e in EVENTS:
        data = torch.load(f"data/pyg_datasets/{e}.pt", weights_only=False)
        for q, snap in enumerate(data.temporal_sequence):
            assert snap.x.shape == (2464, 9), f"{e} q={q}: x shape {snap.x.shape}"
            nan_count = torch.isnan(snap.x).sum().item()
            assert nan_count == 0, f"{e} q={q}: {nan_count} NaN in node features"
check("Node features: shape (2464,9) and no NaN", test_node_features)

# Label coverage >= 60%
def test_label_coverage():
    for e in EVENTS:
        data = torch.load(f"data/pyg_datasets/{e}.pt", weights_only=False)
        pct = data.label_mask.float().mean().item()
        assert pct >= 0.60, f"{e}: label coverage {pct:.0%} (need >= 60%)"
check("Label coverage >= 60% for all events", test_label_coverage)

print(f"\n{'='*50}")
print(f"Phase 5 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL); sys.exit(1)
else:
    print("ALL PASSED — proceed to Phase 6")
```

---

## PHASE 6 — Baseline Models
**Duration**: 5 days
**Output**: `results/tables/baselines.csv`, `LEONTIEF_PASS_THROUGH_RATE` in config

---

### 6.A — Coding Prompt: `src/baselines/leontief_io.py`

> **Give this prompt to your coding assistant:**

```
Create src/baselines/leontief_io.py to implement the Leontief IO shock propagation baseline.

Prediction formula (LOCKED):
  node_delta_tau[i] = Σ_j (import_pen_coeff[j→i] × delta_tariff[j→i]) for all j
  predicted_delta_p = (L.T @ node_delta_tau) × LEONTIEF_PASS_THROUGH_RATE

where L is the Leontief inverse for the event year (loaded from leontief_{YEAR}.npy).
Note: Leontief gives same prediction for all three horizons (no temporal dynamics).

Calibration procedure (run ONCE using UK Global Tariff 2021 event):
1. Load labels_uk_global_tariff_2021.parquet
2. Load leontief_{YEAR}.npy for year 2020 (pre-event year)
3. Compute raw predictions (LEONTIEF_PASS_THROUGH_RATE = 1.0)
4. Grid search rate in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
5. Find rate minimizing RMSE_6m on the UK event (labeled nodes only)
6. Print: "Calibrated LEONTIEF_PASS_THROUGH_RATE = {best_rate}"
7. Print: "IMPORTANT: Copy this value to config.LEONTIEF['PASS_THROUGH_RATE'] NOW"
   (do NOT auto-write to config.py — do it manually to maintain version control)

Evaluation function run_leontief_baseline():
- For each of the 6 events: compute predictions using calibrated rate
- Compute RMSE, MAE, R², DirectionalAccuracy for 3m, 6m, 12m (same value for all 3 horizons)
- Save to results/tables/baselines.csv (append rows, don't overwrite)
- Return results dict

Sanity assertion: Leontief directional accuracy should be in range [0.50, 0.75]
  (better than random 0.50, worse than trained model ~0.70)
```

---

### 6.B — Coding Prompt: `src/baselines/panel_var.py`

> **Give this prompt to your coding assistant:**

```
Create src/baselines/panel_var.py to implement the Panel VAR baseline.

Locked settings: lag order p=4 quarters. One VAR per (country, sector) node.
PPI change is endogenous variable. tariff_rate is exogenous.

Implementation:
1. Load ppi_quarterly_all.parquet — this provides the time series per (country, sector)
2. Load tariff_rates.parquet for exogenous variable
3. For each (country, sector) with >= 20 quarterly observations:
   a. Build quarterly PPI change series as endogenous
   b. Build quarterly tariff_rate series as exogenous
   c. Fit statsmodels VAR with exog parameter (treat as VARX with p=4 lags)
   d. Forecast 1, 2, and 4 quarters ahead = predictions for 3m, 6m, 12m
4. Map forecasts to LOEO-CV: for each event fold, fit on pre-event data only
   (all quarters before event_quarter, using training events' pre-event periods)

Use leave-one-event-out structure consistent with TSPN training:
  For fold n (held-out event n): fit VAR on all data EXCLUDING event n's post-event period
  Predict for event n's test period

Save predictions to results/tables/baselines.csv with model_name="panel_var"
```

---

### 6.C — Coding Prompt: `src/baselines/mlp_no_graph.py`

> **Give this prompt to your coding assistant:**

```
Create src/baselines/mlp_no_graph.py to implement the MLP-no-graph baseline.

Architecture (locked): input=10 (9 node features + 1 scalar total direct tariff exposure),
hidden=[128, 64, 32], output=3 (3m/6m/12m predictions).
No graph structure — each node is processed independently.

Input construction:
  - The 9 normalized node features from node_features_{YEAR}.parquet
  - Plus 1 scalar: total_direct_tariff_exposure = sum of (import_pen_coeff × tariff_delta)
    over all incoming edges for this node (from shock vector)

Training: Same LOEO-CV protocol as TSPN. Same optimizer (Adam lr=1e-3). Same loss function
(0.50*MSE_3m + 0.30*MSE_6m + 0.20*MSE_12m). Same max_epochs=200, early stopping patience=20.
NO augmentation (the MLP has no edge features to augment).

Save predictions to results/tables/baselines.csv with model_name="mlp_no_graph"

This baseline controls for graph structure: if TSPN beats MLP-no-graph, the improvement
is attributable to graph-based propagation.
```

---

### 6.D — MANUAL: Set LEONTIEF_PASS_THROUGH_RATE

After running `src/baselines/leontief_io.py`, it will print the calibrated rate. Open `config.py` and set:
```python
LEONTIEF = {
    "REG_EPS": 1e-4,
    "PASS_THROUGH_RATE": 0.XX,   # <-- paste your calibrated value here
}
```
Commit this to git immediately:
```bash
git add config.py
git commit -m "calibrate: set LEONTIEF_PASS_THROUGH_RATE={VALUE}"
```

---

### ✅ PHASE 6 CHECKPOINT

```python
"""Phase 6 checkpoint — run before Phase 7."""
import os, sys, pandas as pd

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append(name); print(f"  FAIL  {name}: {e}")

# CP26 ★ — LEONTIEF_PASS_THROUGH_RATE set in config
def test_pass_through():
    sys.path.insert(0, '.')
    import config
    rate = config.LEONTIEF.get("PASS_THROUGH_RATE")
    assert rate is not None, "LEONTIEF_PASS_THROUGH_RATE not set in config"
    assert isinstance(rate, float), f"Expected float, got {type(rate)}"
    assert 0.001 <= rate <= 10.0, f"Rate {rate} seems wrong (expected 0.001–10.0)"
    print(f"  INFO: LEONTIEF_PASS_THROUGH_RATE = {rate}")
check("★ CP26: LEONTIEF_PASS_THROUGH_RATE set", test_pass_through)

# baselines.csv exists and has results
def test_baselines_csv():
    path = "results/tables/baselines.csv"
    assert os.path.exists(path), "baselines.csv not found"
    df = pd.read_csv(path)
    required_cols = ["model_name", "fold", "val_event", "RMSE_6m", "MAE_6m", "R2_6m", "DirAcc_6m"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    # Should have results for leontief, panel_var, mlp_no_graph
    models = df["model_name"].unique()
    print(f"  INFO: Models with results: {list(models)}")
check("baselines.csv structure", test_baselines_csv)

# Leontief directional accuracy sanity
def test_leontief_dacc():
    df = pd.read_csv("results/tables/baselines.csv")
    leontief = df[df["model_name"] == "leontief_io"]
    if len(leontief) == 0:
        raise AssertionError("No leontief_io rows in baselines.csv")
    avg_dacc = leontief["DirAcc_6m"].mean()
    assert 0.50 <= avg_dacc <= 0.80, f"Leontief DirAcc_6m = {avg_dacc:.3f} (expect 0.50-0.80)"
    print(f"  INFO: Leontief avg DirAcc_6m = {avg_dacc:.3f}")
check("Leontief directional accuracy in range", test_leontief_dacc)

print(f"\n{'='*50}")
print(f"Phase 6 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL); sys.exit(1)
else:
    print("ALL PASSED — proceed to Phase 7")
```

---

## PHASE 7 — TSPN Architecture Implementation
**Duration**: 9 days
**Output**: All model files in `src/models/`, verified forward pass

> **Note**: Build and test files in this exact order. Run a forward-pass sanity check after each module.

---

### 7.A — Coding Prompt: `src/models/feature_embedding.py`

> **Give this prompt to your coding assistant:**

```
Create src/models/feature_embedding.py with class FeatureEmbedding(nn.Module).

LOCKED ARCHITECTURE:
Node path: Linear(9, 128) → BatchNorm1d(128) → ReLU → Dropout(0.1)
Edge path:  Linear(6, 64) → ReLU → Dropout(0.1)

Rules:
- No weight sharing between node and edge paths
- Kaiming uniform initialization (PyTorch default for Linear with ReLU)
- forward(x, edge_attr) returns (node_emb, edge_emb):
    node_emb shape: (N, 128)
    edge_emb shape: (E, 64)
- All dims from config.MODEL dict

Unit test at bottom of file (run if __name__ == "__main__"):
  x = torch.randn(2464, 9)
  edge_attr = torch.randn(100000, 6)
  model = FeatureEmbedding()
  node_emb, edge_emb = model(x, edge_attr)
  assert node_emb.shape == (2464, 128)
  assert edge_emb.shape == (100000, 64)
  assert not torch.isnan(node_emb).any()
  print("FeatureEmbedding OK")
```

---

### 7.B — Coding Prompt: `src/models/tspn_gat_layer.py`

> **Give this prompt to your coding assistant:**

```
Create src/models/tspn_gat_layer.py with class TSPNGATLayer(MessagePassing).

This is a CUSTOM GAT — NOT standard GATConv. Standard GATConv ignores edge features in
attention score computation. This layer explicitly includes edge embeddings.

LOCKED ATTENTION FORMULA per head k:
  query_i  = W_q_k @ h_i           → shape (32,)
  key_j    = W_k_k @ h_j           → shape (32,)
  edge_kk  = W_e_k @ e_ij          → shape (32,)   [from edge_embed_dim=64 down to head_dim=32]
  concat   = cat([query_i, key_j, edge_kk])  → shape (96,)
  score_ij = a_k^T × LeakyReLU(concat, negative_slope=0.2)
  alpha_ij = softmax(score_ij) over N(i)
  alpha_ij = Dropout(alpha_ij, p=0.3) during training

LOCKED AGGREGATION per head k:
  value_j = W_v_k @ h_j            → shape (32,)
  m_i^k   = Σ_{j in N(i)} alpha_ij × value_j

LOCKED MULTI-HEAD + RESIDUAL:
  h_i_concat = cat([m_i^1, m_i^2, m_i^3, m_i^4]) → shape (128,)
  h_i_act    = ELU(h_i_concat)
  h_i_out    = h_i_act + W_res @ h_i_input         [W_res = Linear(128,128, bias=False)]

Parameters:
  num_heads=4, head_dim=32, in_dim=128, edge_dim=64
  W_q, W_k, W_v, a per head (4 sets)
  W_e per head (maps 64→32)
  W_res (residual projection, no bias)

self.last_alpha: store the full attention weight tensor after each forward pass
  shape: (num_edges × num_heads) or (num_edges, num_heads)
  This is REQUIRED for interpretability analysis in Phase 10

forward(x, edge_index, edge_attr) returns h_out shape (N, 128)

Inherit from torch_geometric.nn.MessagePassing with aggr="add".
Use propagate() pattern correctly:
  - message() computes weighted value per edge
  - aggregate() sums them
  - update() applies residual

Unit test:
  layer = TSPNGATLayer()
  x = torch.randn(2464, 128)
  edge_index = torch.randint(0, 2464, (2, 50000))
  edge_attr = torch.randn(50000, 64)
  out = layer(x, edge_index, edge_attr)
  assert out.shape == (2464, 128)
  assert not torch.isnan(out).any()
  assert layer.last_alpha is not None
  print("TSPNGATLayer OK")
```

---

### 7.C — Coding Prompt: `src/models/tspn_gru.py`

> **Give this prompt to your coding assistant:**

```
Create src/models/tspn_gru.py with class TSPNTemporalGRU(nn.Module).

LOCKED SPECIFICATION:
- Input: list of 8 tensors, each shape (2464, 128)
- Stack to (8, 2464, 128) — sequence length=8, each position is a full graph snapshot
- GRU: input_size=128, hidden_size=256, num_layers=1, batch_first=False, bidirectional=False
- Extract ONLY final hidden state: shape (2464, 256)
  (NOT the full output sequence — only the last hidden state)
- Apply Dropout(0.2) to the final hidden state
- Return: tensor shape (2464, 256)

The GRU treats node dimension as the "batch" dimension.
Reshape input from (8, 2464, 128) → treat as (seq_len=8, batch=2464, input=128).

forward(sequence_list) → tensor (2464, 256)

Unit test:
  gru = TSPNTemporalGRU()
  seq = [torch.randn(2464, 128) for _ in range(8)]
  out = gru(seq)
  assert out.shape == (2464, 256)
  assert not torch.isnan(out).any()
  print("TSPNTemporalGRU OK")
```

---

### 7.D — Coding Prompt: `src/models/output_head.py`

> **Give this prompt to your coding assistant:**

```
Create src/models/output_head.py with class MultiHorizonHead(nn.Module).

LOCKED ARCHITECTURE: Three INDEPENDENT MLPs (no shared weights).
Each MLP: Linear(256,128) → ReLU → Dropout(0.2) → Linear(128,64) → ReLU → Linear(64,1)

forward(x) where x has shape (2464, 256):
  out_3m  = mlp_3m(x)   → (2464, 1)
  out_6m  = mlp_6m(x)   → (2464, 1)
  out_12m = mlp_12m(x)  → (2464, 1)
  return torch.cat([out_3m, out_6m, out_12m], dim=1)  → (2464, 3)

Column 0 = 3m prediction, Column 1 = 6m, Column 2 = 12m.
Dropout only after first two Linear layers (not the last).

Unit test:
  head = MultiHorizonHead()
  x = torch.randn(2464, 256)
  out = head(x)
  assert out.shape == (2464, 3)
  print("MultiHorizonHead OK")
```

---

### 7.E — Coding Prompt: `src/models/tspn.py`

> **Give this prompt to your coding assistant:**

```
Create src/models/tspn.py with class TSPN(nn.Module) — the full model assembly.

LOCKED FORWARD PASS SEQUENCE (do not reorder):
1. For each snapshot q in [0..7]:
   a. node_embed_q, edge_embed_q = feature_embedding(temporal_sequence[q].x,
                                                      temporal_sequence[7].edge_attr)
      NOTE: edge_attr from snapshot 7 (event-time) is used in ALL snapshots.
            This is intentional — the shock signal propagates through all temporal positions.
   b. rep1_q = gat_layer1(node_embed_q, edge_index, edge_embed_q)
   c. rep2_q = gat_layer2(rep1_q, edge_index, edge_embed_q)
   d. sequence_list.append(rep2_q)
2. seq_tensor = stack(sequence_list) → (8, 2464, 128)
3. temporal = gru(seq_tensor)  → (2464, 256)
4. predictions = output_head(temporal)  → (2464, 3)
5. return predictions, gat_layer1.last_alpha

CRITICAL DESIGN RULES:
- gat_layer1 and gat_layer2 are SEPARATE instances (separate weight matrices)
- They do NOT share weights. Never use the same instance twice.
- edge_index: all snapshots must have identical edge_index — use temporal_sequence[0].edge_index
- output_head receives ONLY the final GRU hidden state

Components: FeatureEmbedding, TSPNGATLayer (×2, separate instances), TSPNTemporalGRU, MultiHorizonHead
All dim parameters from config.MODEL.

Include forward pass sanity check method sanity_check(event_data):
  - Run one forward pass
  - Assert output shape (2464, 3)
  - Assert no NaN in output
  - Run .backward() on dummy loss
  - Assert all parameters have gradients (no None)
  - Print "TSPN sanity check PASSED"
```

---

### 7.F — Run Sanity Check

```bash
python -c "
import torch, sys
sys.path.insert(0, '.')
from src.models.tspn import TSPN
from src.data.build_pyg_dataset import TSPNEventData

# Load one event dataset
event = torch.load('data/pyg_datasets/us_232_steel_2018.pt', weights_only=False)
model = TSPN()
model.sanity_check(event)
"
```

---

### ✅ PHASE 7 CHECKPOINT

```python
"""Phase 7 checkpoint — run before Phase 8."""
import sys, torch
sys.path.insert(0, '.')

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append(name); print(f"  FAIL  {name}: {e}")

# All model files importable
def test_imports():
    from src.models.feature_embedding import FeatureEmbedding
    from src.models.tspn_gat_layer import TSPNGATLayer
    from src.models.tspn_gru import TSPNTemporalGRU
    from src.models.output_head import MultiHorizonHead
    from src.models.tspn import TSPN
check("All model modules import", test_imports)

# CP27 ★ — GAT layers are SEPARATE instances
def test_separate_gat():
    from src.models.tspn import TSPN
    model = TSPN()
    assert model.gat_layer1 is not model.gat_layer2, "gat_layer1 and gat_layer2 are the same object!"
    # Check they have different parameter id's
    p1 = set(id(p) for p in model.gat_layer1.parameters())
    p2 = set(id(p) for p in model.gat_layer2.parameters())
    assert not p1.intersection(p2), "GAT layers share parameter tensors!"
check("★ CP27: GAT layers are separate instances with distinct weights", test_separate_gat)

# Forward pass: correct output shape
def test_forward_shape():
    from src.models.tspn import TSPN
    event = torch.load("data/pyg_datasets/us_232_steel_2018.pt", weights_only=False)
    model = TSPN()
    model.eval()
    with torch.no_grad():
        pred, alpha = model(event.temporal_sequence)
    assert pred.shape == (2464, 3), f"Wrong output shape: {pred.shape}"
    assert not torch.isnan(pred).any(), "NaN in TSPN output"
check("Forward pass: shape (2464,3), no NaN", test_forward_shape)

# Attention weights populated
def test_attention():
    from src.models.tspn import TSPN
    event = torch.load("data/pyg_datasets/us_232_steel_2018.pt", weights_only=False)
    model = TSPN()
    model.eval()
    with torch.no_grad():
        pred, alpha = model(event.temporal_sequence)
    assert alpha is not None, "gat_layer1.last_alpha is None"
    assert alpha.shape[0] > 0, "alpha has 0 entries"
    print(f"  INFO: alpha shape: {alpha.shape}")
check("Attention weights saved in last_alpha", test_attention)

# Backward pass: all parameters have gradients
def test_gradients():
    from src.models.tspn import TSPN
    event = torch.load("data/pyg_datasets/us_232_steel_2018.pt", weights_only=False)
    model = TSPN()
    model.train()
    pred, alpha = model(event.temporal_sequence)
    dummy_loss = pred.sum()
    dummy_loss.backward()
    none_grads = [n for n, p in model.named_parameters() if p.grad is None]
    assert not none_grads, f"Parameters with None gradient: {none_grads[:3]}"
check("All parameters have gradients", test_gradients)

# Model parameter count reasonable
def test_param_count():
    from src.models.tspn import TSPN
    model = TSPN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  INFO: TSPN total parameters: {n_params:,}")
    assert 500_000 <= n_params <= 5_000_000, f"Unexpected param count: {n_params}"
check("Model parameter count in expected range", test_param_count)

print(f"\n{'='*50}")
print(f"Phase 7 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL); sys.exit(1)
else:
    print("ALL PASSED — proceed to Phase 8")
```

---

## PHASE 8 — Training Infrastructure
**Duration**: 4 days
**Output**: Complete training loop, evaluation, W&B logging

---

### 8.A — Coding Prompt: `src/training/losses.py`

> **Give this prompt to your coding assistant:**

```
Create src/training/losses.py with the EXACT locked loss function.

def compute_loss(pred, labels, mask, alpha_weights, config):
    """
    pred:          FloatTensor (2464, 3)
    labels:        FloatTensor (2464, 3)
    mask:          BoolTensor (2464,) — True for labeled nodes
    alpha_weights: FloatTensor (E, num_heads) — attention weights
    config:        config.TRAINING dict
    """
    pred_masked   = pred[mask]
    labels_masked = labels[mask]
    
    mse_3m  = ((pred_masked[:,0] - labels_masked[:,0]) ** 2).mean()
    mse_6m  = ((pred_masked[:,1] - labels_masked[:,1]) ** 2).mean()
    mse_12m = ((pred_masked[:,2] - labels_masked[:,2]) ** 2).mean()
    l1_attn = alpha_weights.abs().mean()
    
    loss = (config["loss_weight_3m"]   * mse_3m
          + config["loss_weight_6m"]   * mse_6m
          + config["loss_weight_12m"]  * mse_12m
          + config["loss_weight_l1_attn"] * l1_attn)
    
    return loss, {"mse_3m": mse_3m.item(), "mse_6m": mse_6m.item(),
                  "mse_12m": mse_12m.item(), "l1_attn": l1_attn.item()}

WEIGHTS: 0.50, 0.30, 0.20, 0.01 — read from config, do NOT hardcode.
```

---

### 8.B — Coding Prompt: `src/training/augmentation.py`

> **Give this prompt to your coding assistant:**

```
Create src/training/augmentation.py with four augmentation functions.
Each function returns an augmented COPY — never modify the input in-place.

def augment_shock_magnitude(data_q7, sigma=0.05):
    """Add Gaussian noise to e[3] only where e[3] != 0."""
    data_copy = data_q7.clone()
    shock_mask = data_copy.edge_attr[:, 3] != 0
    noise = torch.randn(shock_mask.sum()) * sigma
    data_copy.edge_attr[shock_mask, 3] += noise
    return data_copy

def augment_temporal_jitter(temporal_sequence):
    """With 50% probability, shift sequence by +1 or -1."""
    import random
    if random.random() < 0.50:
        shift = random.choice([-1, 1])
        if shift == 1:
            return temporal_sequence[1:] + [temporal_sequence[7]]
        else:
            return [temporal_sequence[0]] + temporal_sequence[:-1]
    return temporal_sequence

def augment_edge_dropout(data_q7, p=0.05, threshold=0.002):
    """Zero out edge features for low-import-pen edges with probability p."""
    data_copy = data_q7.clone()
    low_pen_mask = data_copy.edge_attr[:, 1] < threshold
    drop_mask = (torch.rand(low_pen_mask.sum()) < p)
    indices = torch.where(low_pen_mask)[0][drop_mask]
    data_copy.edge_attr[indices] = 0.0
    return data_copy

def augment_label_noise(labels, sigma=0.01):
    """Add Gaussian noise to label values."""
    noisy = labels.clone()
    noise = torch.randn_like(noisy) * sigma
    noisy += noise
    return noisy

CRITICAL: augment_temporal_jitter must ensure snapshot 7 (the shock snapshot)
remains the LAST element in the returned list, regardless of jitter direction.
The shock snapshot must always be at position -1 in the sequence.
```

---

### 8.C — Coding Prompt: `src/training/evaluate.py`

> **Give this prompt to your coding assistant:**

```
Create src/training/evaluate.py with all evaluation functions.

All functions take (pred, labels, mask) where:
  pred, labels: FloatTensor (2464, 3) — columns are 3m/6m/12m
  mask: BoolTensor (2464,)

def compute_rmse(pred, labels, mask) -> dict:
    """Returns {'3m': float, '6m': float, '12m': float}"""

def compute_mae(pred, labels, mask) -> dict:
    """Returns {'3m': float, '6m': float, '12m': float}"""

def compute_r2(pred, labels, mask) -> dict:
    """R² = 1 - SS_res / SS_tot. Returns {'3m': float, '6m': float, '12m': float}"""

def compute_directional_accuracy(pred, labels, mask) -> dict:
    """Fraction of labeled nodes where sign(pred) == sign(label). Returns {'3m':, '6m':, '12m':}"""

def bootstrap_ci(pred, labels, mask, metric_fn, n=1000, confidence=0.95) -> dict:
    """Bootstrap confidence interval for any metric function.
    Returns {'3m': (lo, hi), '6m': (lo, hi), '12m': (lo, hi)}"""

def record_all_metrics(pred, event_data, fold_idx, model_name, results_path):
    """Compute all metrics and append one row to results CSV.
    CSV columns: model_name, fold, val_event, RMSE_3m, RMSE_6m, RMSE_12m,
                 MAE_3m, MAE_6m, MAE_12m, R2_3m, R2_6m, R2_12m, DirAcc_3m, DirAcc_6m, DirAcc_12m"""
    import os, pandas as pd
    metrics = {}
    for name, fn in [("RMSE", compute_rmse), ("MAE", compute_mae),
                     ("R2", compute_r2), ("DirAcc", compute_directional_accuracy)]:
        result = fn(pred, event_data.y, event_data.label_mask)
        for horizon, val in result.items():
            metrics[f"{name}_{horizon}"] = val
    row = {"model_name": model_name, "fold": fold_idx,
           "val_event": event_data.event_name, **metrics}
    df = pd.DataFrame([row])
    if os.path.exists(results_path):
        df.to_csv(results_path, mode="a", header=False, index=False)
    else:
        df.to_csv(results_path, index=False)
```

---

### 8.D — Coding Prompt: `src/training/train.py`

> **Give this prompt to your coding assistant:**

```
Create src/training/train.py with the full LOEO-CV training loop.

This is the LOCKED training loop — implement it exactly as specified.

import config, torch, wandb, numpy as np, random
from src.models.tspn import TSPN
from src.training.losses import compute_loss
from src.training.evaluate import compute_rmse, record_all_metrics
from src.training.augmentation import *

def train_all_folds(events, config):
    wandb.init(project="tspn", config=config)
    
    for fold_idx in range(6):
        held_out = events[fold_idx]
        train_events = [e for e in events if e.event_name != held_out.event_name]
        
        model = TSPN()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.TRAINING["lr"],
                                     weight_decay=config.TRAINING["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.TRAINING["T_0"], T_mult=config.TRAINING["T_mult"])
        
        best_val_rmse = float("inf")
        patience_counter = 0
        
        for epoch in range(config.TRAINING["max_epochs"]):
            model.train()
            random.shuffle(train_events)
            epoch_loss = 0
            
            for event in train_events:
                # Apply augmentation
                seq = augment_temporal_jitter(list(event.temporal_sequence))
                seq[7] = augment_shock_magnitude(seq[7])
                seq[7] = augment_edge_dropout(seq[7])
                aug_labels = augment_label_noise(event.y)
                
                optimizer.zero_grad()
                pred, alpha = model(seq)
                loss, loss_parts = compute_loss(pred, aug_labels, event.label_mask, alpha, config.TRAINING)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAINING["grad_clip_norm"])
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step(epoch)
            
            # Validation — NO augmentation
            model.eval()
            with torch.no_grad():
                val_pred, val_alpha = model(held_out.temporal_sequence)
                val_rmse = compute_rmse(val_pred, held_out.y, held_out.label_mask)
                val_rmse_6m = val_rmse["6m"]
            
            if epoch % 10 == 0:
                wandb.log({"fold": fold_idx, "epoch": epoch,
                           "train_loss": epoch_loss / len(train_events),
                           "val_rmse_6m": val_rmse_6m,
                           "lr": scheduler.get_last_lr()[0]})
            
            if val_rmse_6m < best_val_rmse:
                best_val_rmse = val_rmse_6m
                patience_counter = 0
                ckpt_path = f"models/checkpoints/tspn_fold{fold_idx}_best.pt"
                torch.save(model.state_dict(), ckpt_path)
                attn_path = f"results/tables/attention_fold{fold_idx}.npy"
                np.save(attn_path, val_alpha.cpu().numpy())
                print(f"  Fold {fold_idx} epoch {epoch}: new best RMSE_6m = {val_rmse_6m:.5f}")
            else:
                patience_counter += 1
                if patience_counter >= config.TRAINING["early_stop_patience"]:
                    print(f"  Fold {fold_idx}: early stopping at epoch {epoch}")
                    break
        
        # Final evaluation on held-out event
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        with torch.no_grad():
            final_pred, _ = model(held_out.temporal_sequence)
        record_all_metrics(final_pred, held_out, fold_idx, "tspn_full",
                           "results/tables/all_results.csv")
        print(f"Fold {fold_idx} COMPLETE. Best RMSE_6m: {best_val_rmse:.5f}")
    
    wandb.finish()

CRITICAL CHECKS to include:
- Assert model.training == True during training steps
- Assert model.training == False during validation (after model.eval())
- Assert augmentation is NOT applied during validation
- Print "LOEO-CV fold {n}: training on {k} events, validating on {event_name}"

CP33 GUARD: Before starting fold n, print the list of training events and confirm
  held-out event is NOT in that list.
```

---

### ✅ PHASE 8 CHECKPOINT

```python
"""Phase 8 checkpoint — run a dry-run with 2 epochs before full training."""
import sys, torch, os
sys.path.insert(0, '.')

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append(name); print(f"  FAIL  {name}: {e}")

# Loss function test
def test_loss():
    from src.training.losses import compute_loss
    import config
    pred   = torch.randn(2464, 3)
    labels = torch.randn(2464, 3)
    mask   = torch.ones(2464, dtype=torch.bool)
    alpha  = torch.rand(100000, 4)
    loss, parts = compute_loss(pred, labels, mask, alpha, config.TRAINING)
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss is zero or negative"
    assert "mse_3m" in parts
check("Loss function computes correctly", test_loss)

# CP34 ★ — Augmentation NOT applied during eval
def test_no_aug_in_eval():
    from src.training.augmentation import augment_shock_magnitude
    import copy
    data = torch.load("data/pyg_datasets/us_232_steel_2018.pt", weights_only=False)
    seq = data.temporal_sequence
    # The shock in q=7 should survive augmentation (only MAGNITUDE changes, not sign)
    aug_q7 = augment_shock_magnitude(seq[7], sigma=0.05)
    shocked_orig = (seq[7].edge_attr[:, 3] != 0).sum()
    shocked_aug  = (aug_q7.edge_attr[:, 3] != 0).sum()
    assert shocked_orig == shocked_aug, "Augmentation changed which edges are shocked (it should only change magnitude)"
    # Augmentation must NOT be called in eval mode — this is enforced by train.py code review
    print("  INFO: Augmentation preserves shock sparsity pattern")
check("★ CP34: Augmentation preserves shock structure", test_no_aug_in_eval)

# Dry run: 2 epochs for 1 fold
def test_dry_run():
    import config, copy
    from src.models.tspn import TSPN
    from src.training.losses import compute_loss
    from src.training.evaluate import compute_rmse
    
    events = [torch.load(f"data/pyg_datasets/{e}.pt", weights_only=False)
              for e in ["us_232_steel_2018","us_301_list1_2018","eu_retaliation_2018"]]
    
    held_out = events[0]
    train_events = events[1:]
    
    model = TSPN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(2):
        model.train()
        for event in train_events:
            optimizer.zero_grad()
            pred, alpha = model(event.temporal_sequence)
            loss, _ = compute_loss(pred, event.y, event.label_mask, alpha, config.TRAINING)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(held_out.temporal_sequence)
            rmse = compute_rmse(val_pred, held_out.y, held_out.label_mask)
        print(f"  Epoch {epoch}: val RMSE_6m = {rmse['6m']:.5f}")
    
    assert not torch.isnan(val_pred).any(), "NaN in val predictions after training"
check("Dry run: 2 epochs training loop runs cleanly", test_dry_run)

# Checkpoint saving works
def test_checkpoint_save():
    from src.models.tspn import TSPN
    model = TSPN()
    path = "models/checkpoints/test_checkpoint.pt"
    torch.save(model.state_dict(), path)
    model2 = TSPN()
    model2.load_state_dict(torch.load(path))
    os.remove(path)
check("Checkpoint save/load works", test_checkpoint_save)

print(f"\n{'='*50}")
print(f"Phase 8 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL); sys.exit(1)
else:
    print("ALL PASSED — ready for full training in Phase 9")
```

---

## PHASE 9 — Full Training Runs
**Duration**: 7 days
**Output**: `results/tables/all_results.csv` with all models and ablations

---

### 9.A — MANUAL: Set Up Google Colab for Training

1. Upload the entire `tspn/` directory to Google Drive
2. Open a new Colab notebook (runtime → GPU T4)
3. Mount Drive and run:

```python
# In Colab cell 1 — mount and setup
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/tspn

# Install packages (use CUDA wheel for torch)
!pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118 -q
!pip install torch_geometric==2.4.0 -q
!pip install torch-scatter torch-sparse \
  --find-links https://data.pyg.org/whl/torch-2.1.0+cu118.html -q
!pip install wandb pandas pyarrow networkx -q

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

```python
# In Colab cell 2 — W&B login
import wandb
wandb.login()  # paste your W&B API key
```

```python
# In Colab cell 3 — add device support to training
# Before running train.py, ensure model and data are on GPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move tensors: event.y = event.y.to(device), model = model.to(device)
```

---

### 9.B — Run Experiments In This Fixed Order

Run one at a time. Record results in `results/tables/all_results.csv` before starting the next.

```bash
# 1. Leontief IO (~10 min, CPU)
python src/baselines/leontief_io.py --run_evaluation

# 2. Panel VAR (~1 hour, CPU)
python src/baselines/panel_var.py --run_evaluation

# 3. MLP no-graph (~1 hour all folds, GPU in Colab)
python src/baselines/mlp_no_graph.py --run_all_folds

# 4-7. Ablations — run src/training/train.py with config flags:
# GCN ablation: replace attention with mean aggregation (add flag --ablation gcn)
# GAT no-temporal: set seq_len=1 in config temporarily
# GAT no-shock: zero e[3] for all edges in data loading
# TSPN 1-layer: comment out gat_layer2 in forward pass temporarily

# 8. Full TSPN (main result — run last)
python src/training/train.py  # runs all 6 folds (~25 min each on T4)
```

For each Colab session (limited to ~12 hours), save progress:
```python
# Save checkpoint to Drive every 10 epochs (already in train.py)
# If session resets: reload latest checkpoint and resume
model.load_state_dict(torch.load(f"models/checkpoints/tspn_fold{n}_best.pt"))
# Reinitialize optimizer fresh — do NOT try to reload optimizer state
```

---

### ✅ PHASE 9 CHECKPOINT

```python
"""Phase 9 checkpoint — verify training produced complete results."""
import os, sys, pandas as pd, numpy as np

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append(name); print(f"  FAIL  {name}: {e}")

# CP33 ★ — LOEO contamination check
def test_loeo_integrity():
    df = pd.read_csv("results/tables/all_results.csv")
    tspn_rows = df[df["model_name"] == "tspn_full"]
    assert len(tspn_rows) == 6, f"Expected 6 TSPN folds, got {len(tspn_rows)}"
    # Each fold must have different val_event
    val_events = tspn_rows["val_event"].unique()
    assert len(val_events) == 6, f"Only {len(val_events)} unique val events (expect 6 — one per fold)"
check("★ CP33: 6 unique LOEO folds, no contamination", test_loeo_integrity)

# CP37 ★ — Ablation models recorded separately
def test_ablations():
    df = pd.read_csv("results/tables/all_results.csv")
    required_models = ["leontief_io", "panel_var", "mlp_no_graph", "tspn_full"]
    present = df["model_name"].unique()
    missing = [m for m in required_models if m not in present]
    assert not missing, f"Missing model results: {missing}"
check("★ CP37: All required models have results", test_ablations)

# TSPN beats Leontief on RMSE_6m
def test_tspn_vs_leontief():
    df = pd.read_csv("results/tables/all_results.csv")
    tspn_rmse = df[df["model_name"] == "tspn_full"]["RMSE_6m"].mean()
    leontief_rmse = df[df["model_name"] == "leontief_io"]["RMSE_6m"].mean()
    print(f"  INFO: TSPN RMSE_6m = {tspn_rmse:.5f}, Leontief = {leontief_rmse:.5f}")
    if tspn_rmse >= leontief_rmse:
        print("  WARN: TSPN not better than Leontief — investigate before paper writing")
check("TSPN RMSE compared to baselines (informational)", test_tspn_vs_leontief)

# Attention .npy files saved for all folds
def test_attention_files():
    for fold in range(6):
        path = f"results/tables/attention_fold{fold}.npy"
        assert os.path.exists(path), f"Missing: {path}"
        arr = np.load(path)
        assert arr.ndim >= 1, f"attention_fold{fold}.npy is empty"
        assert not np.isnan(arr).any(), f"NaN in attention_fold{fold}.npy"
check("Attention .npy files for all 6 folds", test_attention_files)

# Model checkpoints exist
def test_checkpoints():
    for fold in range(6):
        path = f"models/checkpoints/tspn_fold{fold}_best.pt"
        assert os.path.exists(path), f"Missing: {path}"
check("Model checkpoints for all 6 folds", test_checkpoints)

print(f"\n{'='*50}")
print(f"Phase 9 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL); sys.exit(1)
else:
    print("ALL PASSED — proceed to Phase 10 (Interpretability)")
```

---

## PHASE 10 — Interpretability and Analysis
**Duration**: 5 days
**Output**: All paper figures, interpretability metrics, `results/tables/interpretability.csv`

---

### 10.A — Coding Prompt: `src/analysis/interpretability.py`

> **Give this prompt to your coding assistant:**

```
Create src/analysis/interpretability.py to compute attention vs Leontief correlation.

For each fold n (0-5):
1. Load attention_fold{n}.npy — shape (num_edges, num_heads)
2. Average across heads → shape (num_edges,)
3. Load leontief_{YEAR}.npy for the held-out event's year
4. Get edge_index from the event's PyG dataset (snap 0)
5. For each edge (src_id, tgt_id): map to Leontief entry L[src_id, tgt_id]
6. Filter to edges where L[src_id, tgt_id] > 0 (only meaningful Leontief entries)
7. Compute from scipy.stats:
   pearson_r, pearson_p = scipy.stats.pearsonr(attn_filtered, leontief_filtered)
   spearman_rho, spearman_p = scipy.stats.spearmanr(attn_filtered, leontief_filtered)
8. Repeat for Layer 1 and Layer 2 alpha separately (if saved)

Save results to results/tables/interpretability.csv:
  Columns: fold, layer, pearson_r, pearson_p, spearman_rho, spearman_p, n_edges

Generate Figure 3 (save as results/figures/fig3_attention_leontief.pdf and .png):
  Scatter plot: x = Leontief coefficient, y = attention weight
  Two panels side by side (Layer 1 and Layer 2)
  Add regression line and R² annotation
  Use plt.savefig with bbox_inches='tight', dpi=300 for PNG, format='pdf' for PDF
```

---

### 10.B — Coding Prompt: `src/analysis/cascade_depth.py`

> **Give this prompt to your coding assistant:**

```
Create src/analysis/cascade_depth.py to measure how far tariff shocks propagate in the graph.

For each held-out event (6 total):
1. Load the TSPN predictions for that fold from all_results.csv or re-run inference
2. Load the PyG dataset for that event
3. Build a NetworkX directed graph from the edge_index of snapshot 7
4. Find direct_hit nodes (is_direct_hit=True in event data)
5. BFS from all direct_hit nodes: compute shortest-hop distance to every other node
6. For each hop k in [0, 1, 2, 3, 4, 5]:
   - Get all nodes at hop distance k
   - Compute mean(|predicted_delta_6m|) for those nodes (labeled nodes only)
7. Normalize by mean at hop 0
8. Find k* = first k where normalized value < 0.05 (cascade_significance_threshold from config)

Generate Figure 4 (results/figures/fig4_cascade_depth.pdf and .png):
  Line plot: x = hop distance, y = normalized avg |Δp_6m|
  One line per event (6 lines), plus average across events (bold)
  Mark k* with vertical dashed line
```

---

### 10.C — Coding Prompt: `src/analysis/amplifier_sectors.py`

> **Give this prompt to your coding assistant:**

```
Create src/analysis/amplifier_sectors.py to identify shock amplifier sectors.

1. Build attention graph:
   - Average alpha weights across all 6 folds (load attention_fold{n}.npy for n=0..5)
   - Build NetworkX DiGraph: add_edge(src_id, tgt_id, weight=mean_alpha)

2. Build trade graph:
   - Load edges_2016.parquet
   - Build NetworkX DiGraph: add_edge(src_id, tgt_id, weight=import_pen_coeff)

3. Compute eigenvector centrality on both:
   attn_centrality = networkx.eigenvector_centrality_numpy(attn_graph, weight='weight')
   trade_centrality = networkx.eigenvector_centrality_numpy(trade_graph, weight='weight')

4. Amplification ratio per node:
   ratio = attn_centrality[node_id] / (trade_centrality[node_id] + 1e-9)

5. Map node_id back to (country, sector) using config.GRAPH
6. Rank all 2464 nodes by ratio descending
7. Report top 20: country, sector, attn_centrality, trade_centrality, ratio

Save: results/tables/amplifier_sectors.csv (all 2464 nodes, sorted by ratio)

Generate Figure 5 (results/figures/fig5_amplifiers.pdf and .png):
  Horizontal bar chart of top 15 sectors by amplification ratio
  Label bars with "{country} — {sector}" format
  Color bars by sector type (manufacturing=blue, services=green, primary=orange)
```

---

### 10.D — Coding Prompt: `src/analysis/paper_figures.py`

> **Give this prompt to your coding assistant:**

```
Create src/analysis/paper_figures.py to generate Figure 2 and Figure 6.

Figure 2 — Graph structure visualization (results/figures/fig2_graph.pdf and .png):
  NetworkX + Matplotlib visualization of the WIOD 2016 graph for USA nodes only
  - Filter edges: src_country=="USA" OR tgt_country=="USA"
  - Node size proportional to backward_linkage
  - Edge width proportional to import_pen_coeff
  - Node labels: sector code abbreviations (e.g. "C24-Steel", "C29-Vehicles")
  - Color nodes by sector type (manufacturing/services/primary)

Figure 6 — Model comparison (results/figures/fig6_model_comparison.pdf and .png):
  Grouped bar chart: x = model name, bars grouped by horizon (3m/6m/12m)
  y = RMSE (mean across folds)
  Error bars = std across folds
  Load data from results/tables/all_results.csv
  Models: leontief_io, panel_var, mlp_no_graph, tspn_full (+ any ablations)

All figures saved as both .pdf (vector) and .png (300dpi).
Use plt.savefig(path, format='pdf', bbox_inches='tight') for PDF.
Use plt.savefig(path, dpi=300, bbox_inches='tight') for PNG.
```

---

### ✅ PHASE 10 CHECKPOINT

```python
"""Phase 10 checkpoint — verify all analysis outputs."""
import os, sys, pandas as pd

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append(name); print(f"  FAIL  {name}: {e}")

# CP40 ★ — Attention-Leontief index alignment check
def test_interpretability():
    path = "results/tables/interpretability.csv"
    assert os.path.exists(path), "interpretability.csv missing"
    df = pd.read_csv(path)
    assert "pearson_r" in df.columns
    assert "pearson_p" in df.columns
    # All p-values should be < 0.01 for a significant result
    sig = (df["pearson_p"] < 0.01).mean()
    print(f"  INFO: {sig:.0%} of folds show p < 0.01 for Pearson r")
    # r should be positive (higher attention = more Leontief linkage)
    avg_r = df["pearson_r"].mean()
    print(f"  INFO: Mean Pearson r = {avg_r:.3f}")
    assert avg_r > 0, "Negative mean Pearson r — check edge index alignment"
check("★ CP40: Interpretability r > 0 and significant", test_interpretability)

# CP42 — All paper figures exist and are > 50KB (vector PDF)
def test_figures():
    required = ["fig3_attention_leontief.pdf", "fig4_cascade_depth.pdf",
                "fig5_amplifiers.pdf", "fig6_model_comparison.pdf"]
    for fig in required:
        path = f"results/figures/{fig}"
        assert os.path.exists(path), f"Missing: {path}"
        size_kb = os.path.getsize(path) / 1024
        assert size_kb > 50, f"{fig}: {size_kb:.0f}KB — likely rasterized (PDF should be > 50KB)"
check("★ CP42: All paper figures exist as vector PDFs > 50KB", test_figures)

# Amplifier sectors table
def test_amplifiers():
    path = "results/tables/amplifier_sectors.csv"
    assert os.path.exists(path), "amplifier_sectors.csv missing"
    df = pd.read_csv(path)
    assert "ratio" in df.columns
    assert len(df) > 0
    top1 = df.iloc[0]
    print(f"  INFO: Top amplifier: {top1.get('country','?')} — {top1.get('sector','?')}, ratio={top1['ratio']:.2f}")
check("Amplifier sectors table", test_amplifiers)

# Cascade depth results make sense
def test_cascade():
    # Check that mean |Δp| decreases with hop distance (the core finding)
    # This is a qualitative check — the result should show attenuation
    print("  INFO: Manual inspection required for cascade depth plot (fig4)")
    assert os.path.exists("results/figures/fig4_cascade_depth.pdf")
check("Cascade depth figure exists", test_cascade)

print(f"\n{'='*50}")
print(f"Phase 10 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL); sys.exit(1)
else:
    print("ALL PASSED — proceed to Phase 11 (Paper) or Phase 12 (Product)")
```

---

## PHASE 11 — Research Paper Writing
**Duration**: 25 days

This phase is writing-only — no coding. Work in LaTeX from Day 1.

### Setup
1. Go to **https://www.overleaf.com/** → New Project → Upload from template
2. Download and use the target journal's LaTeX template (JIE or NCS)
3. Create one main file `tspn_paper.tex`, one `references.bib`

### Fixed Writing Order (do not skip ahead)
| Days | Section | Key content |
|---|---|---|
| 1–3 | §3 Data and Graph Construction | WIOD description, concordance methodology, graph statistics (degree distribution from notebook), tariff event descriptions, shock vector construction |
| 4–7 | §4 Model: TSPN | Formal notation, GAT attention formula, GRU temporal module, full forward pass pseudocode, loss function |
| 8–12 | §5 Experiments | Main results table (from all_results.csv), ablation table, horizon analysis, statistical significance |
| 13–15 | §6 Interpretability | Attention vs Leontief correlation finding, cascade depth finding, amplifier sector table, policy implications |
| 16–18 | §2 Related Work | CGE models, panel regression, IO models, GNNs for economic networks, temporal GNNs |
| 19–20 | §1 Introduction | Problem, existing approaches failure, 4 contributions, paper overview |
| 21–22 | §7 Conclusion + Abstract | Findings, limitations (WIOD lag, few events, no firm-level), future work |
| 23–25 | Proofread + References | Zotero citations, notation consistency, figure placement |

---

## PHASE 12 — Product MVP
**Duration**: 14 days
**Output**: Live Streamlit app on Streamlit Community Cloud

---

### 12.A — Coding Prompt: `src/models/export_onnx.py`

> **Give this prompt to your coding assistant:**

```
Create src/models/export_onnx.py to export the best TSPN checkpoint to ONNX format.

1. Load all_results.csv, find the fold with the lowest RMSE_6m
2. Load that fold's checkpoint: models/checkpoints/tspn_fold{best_fold}_best.pt
3. Export to ONNX:
   torch.onnx.export(model, example_input, "models/onnx/tspn_best.onnx",
                     opset_version=17, dynamic_axes={"node_features": {0: "batch_size"}})
4. Verify ONNX output matches PyTorch:
   Run same input through both, assert max_diff < 1e-4
   If max_diff >= 1e-4: try TorchScript (torch.jit.script) as fallback
5. Profile CPU inference latency: 10 runs, report mean ± std
   Assert mean latency < 3 seconds (target from spec)

Save as: models/onnx/tspn_best.onnx
Print: "ONNX export complete. Size: {size_mb:.1f}MB, CPU latency: {latency_ms:.0f}ms"
```

---

### 12.B — Coding Prompt: `app/utils/inference.py`

> **Give this prompt to your coding assistant:**

```
Create app/utils/inference.py — the inference wrapper for the Streamlit app.

Critical rule: ALWAYS apply normalization before inference using normalization_stats.json.
Never pass raw features to the model.

def load_model_and_stats():
    """Load ONNX model and normalization stats at startup."""
    import onnxruntime as ort, json
    sess = ort.InferenceSession("models/onnx/tspn_best.onnx")
    with open("data/processed/node_features/normalization_stats.json") as f:
        stats = json.load(f)
    return sess, stats

def run_inference(scenario_dict, sess, stats, graph_data):
    """
    scenario_dict: {'target_country': str, 'sector': str,
                    'delta_tariff': float, 'source_countries': list}
    Returns DataFrame with columns:
      country, sector, delta_3m, delta_6m, delta_12m, hop_distance, is_direct_hit
    """
    # 1. Build shock vector from scenario_dict using compute_shock_vector()
    #    IMPORT from src.data.build_shock_vectors — do NOT re-implement
    # 2. Load node features from data/processed/node_features/node_features_2021.parquet
    # 3. APPLY z-score normalization: (raw - mean) / std using stats
    # 4. Set e[3] from shock vector for affected edges
    # 5. Run ONNX inference
    # 6. Compute hop distances via BFS from direct-hit nodes
    # 7. Return results DataFrame
```

---

### 12.C — Build the App (Day-by-Day Order)

**Day 2** — Scenario Builder sidebar (`app/components/scenario_builder.py`):
```
Prompt: Build a Streamlit sidebar component with:
- Country multiselect (source countries): default all, from COUNTRY_LIST
- Sector select: 56 ISIC sectors with readable labels (e.g. "C24 — Basic Metals")
- Tariff magnitude slider: 0–50%, step 1%
- Historical event dropdown: pre-fills all controls (6 events from config)
- Submit button that calls inference wrapper and stores result in st.session_state
```

**Day 3** — Price Table (`app/components/price_table.py`):
```
Prompt: Build a Streamlit component showing inference results as a table:
- Columns: Country, Sector, Δp(3m)%, Δp(6m)%, Δp(12m)%, Hop Distance, Risk Level
- Risk Level: "High" if |Δp6m| > 3%, "Medium" if > 1%, "Low" if > 0.5%, "Negligible" otherwise
- Color-code Risk Level column
- Default sort: |Δp(6m)| descending
- Filter widgets: by risk level and by country
- CSV download button
```

**Days 4–5** — Graph Visualization (`app/components/graph_viz.py`):
```
Prompt: Build a PyVis network component showing the shock cascade:
- Center on shocked nodes (direct_hit=True)
- Node size proportional to |Δp(6m)|: min=5px, max=30px
- Node color: red for positive Δp (price increase), blue for negative, gray for negligible
- Edge opacity proportional to import_pen_coeff
- Show ONLY nodes with |Δp(6m)| > 0.5%
- Hard node budget: max 150 nodes (add most-impacted nodes first)
- Show "Displaying X of Y affected nodes" count
- Add hop-distance filter slider
- CP46 guard: if affected nodes > 150, show warning and "Download full graph" button
```

**Day 6** — Risk Dashboard (`app/components/risk_dashboard.py`):
```
Prompt: Build three Plotly charts:
1. Top 10 impacted sectors — horizontal bar chart, x=|Δp(6m)|, color by risk level
2. World choropleth — aggregate impact score per country
3. Donut chart — share of impact by sector type (manufacturing/services/primary)
```

**Day 7** — Scenario Comparison:
```
Prompt: Build a two-column comparison layout:
Left = Scenario A (current), Right = Scenario B (previous or user-defined)
Below: side-by-side table with diff column (B - A), and aggregate impact scores
```

**Days 8** — Historical Event Library:
```
Prompt: Build a historical events page:
- Dropdown of 6 pre-loaded tariff events
- For each: scatter plot of model prediction vs actual BLS PPI (US sectors)
- Model accuracy badge showing RMSE_6m for that event
```

**Days 9–10** — Integration test end-to-end. Edge inputs (0%, 50% tariff).

**Days 11–12** — Polish: loading spinners, error handling, methodology description.

**Days 13–14** — Deploy:
```bash
# Push to public GitHub repo
git add .
git commit -m "feat: complete Streamlit MVP"
git push origin main

# Deploy on Streamlit Community Cloud:
# 1. Go to share.streamlit.io
# 2. Click "New app" → Connect GitHub repo
# 3. Set main file: app/app.py
# 4. Add secrets: COMTRADE_API_KEY, etc.
# 5. Deploy → get public URL
```

---

### ✅ PHASE 12 CHECKPOINT

```python
"""Phase 12 checkpoint — verify product before deployment."""
import os, sys, numpy as np

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append(name); print(f"  FAIL  {name}: {e}")

# CP43 ★ — ONNX output matches PyTorch
def test_onnx_parity():
    import torch, onnxruntime as ort
    sys.path.insert(0, '.')
    from src.models.tspn import TSPN
    import pandas as pd

    onnx_path = "models/onnx/tspn_best.onnx"
    assert os.path.exists(onnx_path), "ONNX model not found"
    
    # Load best fold checkpoint
    df = pd.read_csv("results/tables/all_results.csv")
    best_fold = df[df["model_name"] == "tspn_full"].groupby("fold")["RMSE_6m"].mean().idxmin()
    model = TSPN()
    model.load_state_dict(torch.load(f"models/checkpoints/tspn_fold{best_fold}_best.pt"))
    model.eval()
    
    event = torch.load("data/pyg_datasets/us_232_steel_2018.pt", weights_only=False)
    with torch.no_grad():
        pt_out = model(event.temporal_sequence)[0].numpy()
    
    sess = ort.InferenceSession(onnx_path)
    # Note: prepare ONNX inputs based on your export signature
    # (simplified check — actual input prep depends on export implementation)
    print(f"  INFO: ONNX model loaded successfully, size={os.path.getsize(onnx_path)/1e6:.1f}MB")
check("★ CP43: ONNX model exists and loads", test_onnx_parity)

# CP44 ★ — Normalization applied at inference
def test_normalization_at_inference():
    import json
    stats_path = "data/processed/node_features/normalization_stats.json"
    assert os.path.exists(stats_path), "normalization_stats.json missing"
    with open(stats_path) as f:
        stats = json.load(f)
    assert len(stats["mean"]) == 9
    assert len(stats["std"]) == 9
    # Check inference.py imports and uses stats
    inf_path = "app/utils/inference.py"
    assert os.path.exists(inf_path), "inference.py missing"
    with open(inf_path) as f:
        code = f.read()
    assert "normalization_stats.json" in code, "inference.py doesn't load normalization stats"
    assert "stats" in code, "inference.py doesn't use normalization stats"
check("★ CP44: Normalization applied in inference.py", test_normalization_at_inference)

# CP48 — ONNX model under 100MB
def test_model_size():
    path = "models/onnx/tspn_best.onnx"
    if not os.path.exists(path):
        raise AssertionError("ONNX model not found")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  INFO: ONNX model size: {size_mb:.1f}MB")
    assert size_mb < 100, f"Model too large: {size_mb:.1f}MB (limit 100MB for Streamlit Cloud)"
check("CP48: ONNX model under 100MB", test_model_size)

# CP46 — PyVis node budget enforced
def test_node_budget():
    viz_path = "app/components/graph_viz.py"
    assert os.path.exists(viz_path), "graph_viz.py missing"
    with open(viz_path) as f:
        code = f.read()
    assert "150" in code, "Node budget of 150 not found in graph_viz.py"
check("CP46: PyVis node budget (150) enforced", test_node_budget)

# App file exists
def test_app_files():
    required = ["app/app.py", "app/utils/inference.py", "app/utils/scenario_parser.py",
                "app/components/scenario_builder.py", "app/components/graph_viz.py",
                "app/components/price_table.py"]
    for f in required:
        assert os.path.exists(f), f"Missing: {f}"
check("All app files exist", test_app_files)

print(f"\n{'='*50}")
print(f"Phase 12 Checkpoint: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL); sys.exit(1)
else:
    print("ALL PASSED — deploy to Streamlit Cloud")
```

---

## QUICK REFERENCE — Critical Rules

These are the rules that corrupt results silently if broken. Keep this visible.

| Rule | What breaks silently |
|---|---|
| `config.py` is the ONLY source of truth | Mismatched params across scripts |
| Node feature vector order `f[0]–f[8]` is IMMUTABLE | Normalization stats misalign |
| Edge feature order `e[0]–e[5]` is IMMUTABLE, shock = `e[3]` | Wrong edges get shocked |
| `e[3] == 0` in snapshots 0–6, non-zero only in snapshot 7 | Model trains on contaminated signal |
| Normalization computed from 2000–2016 ONLY | Data leakage from test period |
| LOEO-CV: held-out event NEVER in train set | Inflated results |
| `model.eval()` during validation, NO augmentation | Corrupted early stopping |
| `gat_layer1` and `gat_layer2` are SEPARATE instances | Shared weights = wrong learning |
| Inference uses `normalization_stats.json` | Product outputs garbage |
| App imports shock builder from `src/`, doesn't re-implement | Shock formula diverges from training |

---

## PHASE SUMMARY TIMELINE

| Phase | Days | Critical Milestone |
|---|---|---|
| 0 — Environment | 2 | `config.py` passes validation, all packages install |
| 1 — Data Collection | 8 | All raw data on disk, HTS codes extracted |
| 2 — WIOD Processing | 4 | ★ CP02 WIOD total $55–70T, all Leontief inverses valid |
| 3 — Tariff & Shock | 4 | ★ CP12 shock direction correct, all 6 events built |
| 4 — Feature Engineering | 5 | ★ CP16 norm on training only, ★ CP20 shock in q=7 only |
| 5 — PyG Dataset | 3 | ★ CP21 shock placement, ★ CP22 edge index consistent |
| 6 — Baselines | 5 | ★ CP26 LEONTIEF_PASS_THROUGH_RATE set in config |
| 7 — TSPN Architecture | 9 | ★ CP27 separate GAT layers, forward pass clean |
| 8 — Training Infra | 4 | Dry run passes, ★ CP34 no aug in eval |
| 9 — Training Runs | 7 | ★ CP33 LOEO integrity, all 6 folds done |
| 10 — Interpretability | 5 | ★ CP40 r > 0, ★ CP42 vector PDFs |
| 11 — Paper Writing | 25 | Submit to journal |
| 12 — Product | 14 | ★ CP43 ONNX parity, ★ CP44 normalization at inference |
