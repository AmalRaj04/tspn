import pandas as pd

df = pd.read_csv(
    "data/raw/tariff_events/uk-tariff-2021-01-01--v4.0.1527--measures-on-declarable-commodities.csv",
    low_memory=False
)

uk = df[
    df["measure__type__description"] == "Third country duty"
]

pct = uk[
    uk["measure__duty_expression"]
    .astype(str)
    .str.match(r"^\d+(\.\d+)?%$")
]

print("Third country duty rows:", len(uk))
print("Pure percentage rows:", len(pct))

print("\nSample:")
print(
    pct[
        ["commodity__code", "measure__duty_expression"]
    ].head(20)
)