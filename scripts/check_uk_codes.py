import pandas as pd

df = pd.read_csv(
    "data/raw/tariff_events/uk-tariff-2021-01-01--v4.0.1527--measures-on-declarable-commodities.csv",
    usecols=["commodity__code"]
)

codes = (
    df["commodity__code"]
    .astype(str)
    .str.replace(".0", "", regex=False)
)

print("Min length:", codes.str.len().min())
print("Max length:", codes.str.len().max())

print("\nExamples:")
print(codes.head(20).tolist())