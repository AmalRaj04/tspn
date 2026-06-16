# scripts/audit_tariff_feature.py

import pandas as pd
from pathlib import Path

NODE_DIR = Path("data/processed/node_features")

for fp in sorted(NODE_DIR.glob("node_features_*.parquet")):

    year = int(fp.stem.split("_")[-1])

    df = pd.read_parquet(fp)

    n_unique = df["f4"].nunique()

    print(
        year,
        "unique_f4_values =",
        n_unique
    )