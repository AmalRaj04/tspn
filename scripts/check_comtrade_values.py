import pandas as pd

df = pd.read_parquet(
    "data/raw/comtrade/comtrade_AUS_2019.parquet"
)

print("=" * 80)
print("TRADE VALUE SUMMARY")
print("=" * 80)

print(df["trade_value_usd"].describe())

print("\n" + "=" * 80)
print("ZERO VALUES")
print("=" * 80)

print(
    (df["trade_value_usd"] == 0).sum()
)

print("\n" + "=" * 80)
print("TOP 20 FLOWS")
print("=" * 80)

print(
    df.sort_values(
        "trade_value_usd",
        ascending=False
    )
    .head(20)
)

print("\n" + "=" * 80)
print("TOTAL TRADE")
print("=" * 80)

print(df["trade_value_usd"].sum())