# scripts/check_comtrade_extension.py

import pandas as pd

a = pd.read_parquet(
    "data/processed/edges/edges_2015.parquet"
)

b = pd.read_parquet(
    "data/processed/edges/edges_2021.parquet"
)

same = (a["flow_usd"] == b["flow_usd"]).all()

print("ALL FLOWS IDENTICAL:", same)

if not same:
    diff = (a["flow_usd"] != b["flow_usd"]).sum()
    print("DIFFERENT ROWS:", diff)