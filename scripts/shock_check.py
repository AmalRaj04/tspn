import pandas as pd

df = pd.read_parquet(
    "data/processed/tariff_rates/sector_tariffs.parquet"
)

print(df.groupby("country")["tariff_rate"].mean().sort_values())
print(df.groupby("country")["tariff_rate"].mean().sort_values(ascending=False))