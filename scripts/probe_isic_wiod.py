# scripts/probe_isic_wiod.py

import pandas as pd

sea = pd.read_parquet(
    "data/processed/edges/socioeconomic_2014.parquet"
)

print("Unique WIOD sectors:")
print(sorted(sea["sector"].unique()))

print("\nCounts:")
print(sea["sector"].value_counts().head(20))

isic = pd.read_csv(

    "data/raw/concordance/isic4-cpc21.txt"

)

print(

    sorted(

        isic["ISIC4code"]

        .astype(str)

        .str.zfill(4)

        .unique()

    )[:50]

)

print(

    sorted(

        isic["ISIC4code"]

        .astype(str)

        .str.zfill(4)

        .unique()

    )[-50:]

)