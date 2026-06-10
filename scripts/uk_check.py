import pandas as pd

df = pd.read_csv(
    "data/raw/tariff_events/uk-tariff-2021-01-01--v4.0.1527--measures-on-declarable-commodities.csv"
)

ukgt = df[
    df["measure__type__description"] == "Third country duty"
]

print("Rows:", len(ukgt))

print("\nSample duty expressions:")
print(
    ukgt["measure__duty_expression"]
    .dropna()
    .value_counts()
    .head(30)
)