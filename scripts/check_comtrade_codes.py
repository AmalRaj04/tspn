import pandas as pd

df = pd.read_parquet(
    "data/raw/comtrade/comtrade_AUS_2019.parquet"
)

print("=" * 80)
print("COMMODITY CODE LENGTHS")
print("=" * 80)

lengths = (
    df["commodity_code"]
    .astype(str)
    .str.strip()
    .str.len()
)

print(lengths.value_counts().sort_index())

print("\n" + "=" * 80)
print("SAMPLE CODES")
print("=" * 80)

print(
    sorted(
        df["commodity_code"]
        .astype(str)
        .unique()
    )[:50]
)

print("\nUnique codes:", df["commodity_code"].nunique())