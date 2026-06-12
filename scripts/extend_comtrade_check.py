import pandas as pd

a = pd.read_parquet(
    "data/processed/edges/edges_2017.parquet"
)

b = pd.read_parquet(
    "data/processed/edges/edges_2021.parquet"
)

print(
    (a["import_pen_coeff"] != b["import_pen_coeff"]).mean()
)