
import pandas as pd
from config import GRAPH

df = pd.read_parquet(
    "data/processed/edges/socioeconomic_2014.parquet"
)

actual = set(df["country"].astype(str).unique())
expected = set(GRAPH["COUNTRY_LIST"])

print("EXPECTED:", len(expected))
print("ACTUAL:", len(actual))

print("\nMISSING:")
print(sorted(expected - actual))

print("\nEXTRA:")
print(sorted(actual - expected))