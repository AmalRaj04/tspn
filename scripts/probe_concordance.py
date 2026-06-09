# src/data/probe_concordance.py

import pandas as pd

print("\n" + "=" * 80)
print("HS2017 -> CPC21")
print("=" * 80)

hs = pd.read_csv(
    "data/raw/concordance/CPC21-HS2017.csv"
)

print("Shape:", hs.shape)
print("Columns:")
print(list(hs.columns))
print("\nHead:")
print(hs.head())

print("\nUnique HS codes:", hs.iloc[:, 0].nunique())
print("Unique CPC codes:", hs.iloc[:, 2].nunique())


print("\n" + "=" * 80)
print("ISIC4 -> CPC21")
print("=" * 80)

isic = pd.read_csv(
    "data/raw/concordance/isic4-cpc21.txt"
)

print("Shape:", isic.shape)
print("Columns:")
print(list(isic.columns))
print("\nHead:")
print(isic.head())

print("\nUnique ISIC codes:", isic["ISIC4code"].nunique())
print("Unique CPC codes:", isic["CPC21code"].nunique())


print("\n" + "=" * 80)
print("NAICS -> ISIC4")
print("=" * 80)

naics_raw = pd.read_excel(
    "data/raw/concordance/2017_NAICS_to_ISIC_4.xlsx",
    header=None
)

print("Raw shape:", naics_raw.shape)

for i in range(15):
    print(f"\nROW {i}")
    print(list(naics_raw.iloc[i].values))


print("\n" + "=" * 80)
print("SOCIOECONOMIC 2014")
print("=" * 80)

sea = pd.read_parquet(
    "data/processed/edges/socioeconomic_2014.parquet"
)

print("Columns:")
print(list(sea.columns))

print("\nUnique sectors:")
print(sorted(sea["sector"].unique()))

print("\nRows:", len(sea))