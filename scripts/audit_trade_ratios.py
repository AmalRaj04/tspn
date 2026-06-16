# scripts/audit_trade_ratios.py

import pandas as pd

df = pd.read_parquet(
    "data/processed/node_features/node_features_2014.parquet"
)

for col in ["f1", "f2"]:

    print("\n", col)

    print(df[col].describe())

    print("\nTOP 20")

    print(
        df.sort_values(col, ascending=False)[
            ["country", "sector", col]
        ].head(20)
    )