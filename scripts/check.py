import pandas as pd

df = pd.read_parquet(
    "data/processed/shock_vectors/shock_eu_retaliation_2018.parquet"
)

print(
    df.groupby("sector")["delta_tariff"]
      .first()
      .sort_values(ascending=False)
      .head(20)
)

csv = pd.read_csv(

    "data/raw/tariff_events/eu_retaliation_2018.csv"

)

print(csv["delta_tariff_pct"].describe())

print(csv["delta_tariff_pct"].value_counts().head(20))