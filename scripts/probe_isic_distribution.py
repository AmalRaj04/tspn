import pandas as pd

df = pd.read_csv(
    "data/raw/concordance/isic4-cpc21.txt"
)

df["ISIC4code"] = (
    df["ISIC4code"]
    .astype(str)
    .str.zfill(4)
)

print("Unique ISIC:")
print(df["ISIC4code"].nunique())

print("\nFirst 100:")
print(sorted(df["ISIC4code"].unique())[:100])

print("\nLast 100:")
print(sorted(df["ISIC4code"].unique())[-100:])