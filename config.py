"""TSPN Configuration File.

This is the SINGLE SOURCE OF TRUTH for all parameters.
No other file may hardcode a value that exists here.
Only plain Python dicts and lists are used.
"""

# PATHS — all relative to project root
PATHS = {
    "RAW_WIOD": "data/raw/wiod",
    "RAW_COMTRADE": "data/raw/comtrade",
    "RAW_WITS": "data/raw/wits",
    "RAW_BLS_PPI": "data/raw/bls_ppi",
    "RAW_EUROSTAT_PPI": "data/raw/eurostat_ppi",
    "RAW_COMMODITY": "data/raw/commodity_prices",
    "RAW_TARIFF_EVENTS": "data/raw/tariff_events",
    "RAW_CONCORDANCE": "data/raw/concordance",
    "PROCESSED_EDGES": "data/processed/edges",
    "PROCESSED_NODE_FEATURES": "data/processed/node_features",
    "PROCESSED_TARIFF_RATES": "data/processed/tariff_rates",
    "PROCESSED_SHOCK_VECTORS": "data/processed/shock_vectors",
    "PROCESSED_LABELS": "data/processed/labels",
    "PROCESSED_CONCORDANCE": "data/processed/concordance",
    "PYG_DATASETS": "data/pyg_datasets",
    "MODEL_CHECKPOINTS": "models/checkpoints",
    "MODEL_ONNX": "models/onnx",
    "RESULTS_TABLES": "results/tables",
    "RESULTS_FIGURES": "results/figures",
    "NORM_STATS": "data/processed/node_features/normalization_stats.json",
}

# GRAPH parameters
GRAPH = {
    "N_COUNTRIES": 44,
    "N_SECTORS": 56,
    "N_NODES": 2464,   # 44 × 56
    "EDGE_THRESHOLD": 0.001,   # import_pen_coeff must be >= this to keep edge
    "SEQ_LEN": 8,   # quarters in temporal sequence
    "WIOD_YEARS": list(range(2000, 2015)),
    "COMTRADE_YEARS": [2017, 2018, 2019, 2020, 2021],
    "WIOD_MATRIX_ROW_OFFSET": 6,   # SET MANUALLY after opening WIOD Excel
    "WIOD_MATRIX_COL_OFFSET": 4,   # SET MANUALLY after opening WIOD Excel
    "COUNTRY_LIST": [
        "AUS", "AUT", "BEL", "BGR", "BRA", "CAN", "CHN", "CYP", "CZE", "DEU",
        "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HUN", "IDN", "IND",
        "IRL", "ITA", "JPN", "KOR", "LTU", "LUX", "LVA", "MEX", "MLT", "NLD",
        "NOR", "POL", "PRT", "ROU", "RUS", "SVK", "SVN", "SWE", "TUR", "TWN",
        "USA", "ROW", "HRV", "CHE"
    ],
    "SECTOR_LIST": [
        "A01", "A02", "A03", "B", "C10_C12", "C13_C15", "C16", "C17", "C18",
        "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28",
        "C29", "C30", "C31_C32", "C33", "D35", "E36", "E37_E39", "F", "G45",
        "G46", "G47", "H49", "H50", "H51", "H52", "H53", "I", "J58", "J59_J60", "J61",
        "J62_J63", "K64", "K65", "K66", "L68", "M69_M70", "M71", "M72", "M73",
        "M74_M75", "N", "O84", "P85", "Q", "R_S", "T", "U"
    ]
}


def node_id(country_idx: int, sector_idx: int) -> int:
    return country_idx * 56 + sector_idx


# NODE_FEATURES parameters
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

# EDGE_FEATURES parameters
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

# MODEL parameters
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

# TRAINING parameters
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

# LEONTIEF parameters
LEONTIEF = {
    "REG_EPS": 1e-4,
    "PASS_THROUGH_RATE": None,  # SET AFTER CALIBRATION in Phase 6
}

# EVENTS parameters
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
        "affected_importers": [
            "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST",
            "FIN", "FRA", "GBR", "GRC", "HUN", "IRL", "ITA", "LTU", "LUX",
            "LVA", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE"
        ],
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

# EVAL parameters
EVAL = {
    "metrics": ["RMSE", "MAE", "R2", "DirAcc"],
    "bootstrap_n": 1000,
    "bootstrap_ci": 0.95,
    "cascade_significance_threshold": 0.05,
    "amplifier_centrality": "eigenvector_centrality_numpy",
    "significance_level": 0.01,
}

# COMMODITY_TO_ISIC mapping
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
