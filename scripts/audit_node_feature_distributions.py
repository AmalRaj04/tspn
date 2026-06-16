# scripts/audit_node_feature_distributions.py

import pandas as pd
from pathlib import Path

NODE_DIR = Path("data/processed/node_features")

FEATURES = [f"f{i}" for i in range(9)]

print("=" * 100)
print("NODE FEATURE DISTRIBUTION AUDIT")
print("=" * 100)

for fp in sorted(NODE_DIR.glob("node_features_*.parquet")):

    year = int(fp.stem.split("_")[-1])

    df = pd.read_parquet(fp)

    print("\n" + "=" * 80)
    print(f"YEAR {year}")
    print("=" * 80)

    for col in FEATURES:

        s = df[col]

        print(
            f"{col:>3} | "
            f"mean={s.mean():>8.4f} "
            f"std={s.std():>8.4f} "
            f"min={s.min():>8.4f} "
            f"max={s.max():>8.4f}"
        )

print("\nDONE")