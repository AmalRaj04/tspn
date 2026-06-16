import pandas as pd

df = pd.read_parquet(
    "data/processed/edges/socioeconomic_2000.parquet"
)

print("ROWS:", len(df))

for col in df.columns:
    print(col)

if "node_id" in df.columns:
    print("\nNODE ID STATS")
    print("MIN:", df["node_id"].min())
    print("MAX:", df["node_id"].max())
    print("UNIQUE:", df["node_id"].nunique())

    print("\nHEAD")
    print(
        df[
            ["node_id", "country", "sector"]
        ].head(20)
    )
else:
    print("\nNO node_id COLUMN FOUND")