# scripts/audit_ppi_lag_coverage.py

import pandas as pd
from pathlib import Path

NODE_DIR = Path("data/processed/node_features")

print("=" * 80)
print("PPI LAG COVERAGE AUDIT")
print("=" * 80)

for fp in sorted(NODE_DIR.glob("node_features_*.parquet")):

    year = int(fp.stem.split("_")[-1])

    df = pd.read_parquet(fp)

    total = len(df)

    has_ppi = int(df["has_ppi_lags"].sum())
    no_ppi = total - has_ppi

    pct = 100 * has_ppi / total

    print(
        f"{year}: "
        f"has_ppi={has_ppi:4d} "
        f"missing={no_ppi:4d} "
        f"coverage={pct:6.2f}%"
    )

print("\nDONE")