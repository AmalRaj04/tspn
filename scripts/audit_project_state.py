from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(".")

DIRECTORIES = [
    "data/processed/edges",
    "data/processed/tariff_rates",
    "data/processed/shock_vectors",
    "data/processed/concordance",
    "data/processed/socioeconomic",
    "data/processed/backward_linkage",
    "data/processed/leontief",
    "data/processed/node_features",
    "data/processed/labels",
]

print("=" * 80)
print("TSPN PROJECT AUDIT")
print("=" * 80)

for d in DIRECTORIES:

    p = ROOT / d

    print("\n")
    print("=" * 80)
    print(f"DIRECTORY: {d}")
    print("=" * 80)

    if not p.exists():
        print("MISSING DIRECTORY")
        continue

    files = sorted(p.glob("*"))

    print(f"FILE COUNT: {len(files)}")

    for f in files:

        print("\n--------------------------------------------------")
        print(f"FILE: {f.name}")
        print(f"SIZE_MB: {round(f.stat().st_size / 1024 / 1024, 2)}")

        suffix = f.suffix.lower()

        try:

            if suffix == ".parquet":

                df = pd.read_parquet(f)

                print(f"ROWS: {len(df)}")
                print(f"COLS: {len(df.columns)}")

                print("DTYPES:")
                print(df.dtypes)

                print("COLUMNS:")
                print(list(df.columns))

                print("HEAD:")
                print(df.head(3))

                if "year" in df.columns:
                    print(
                        "YEAR RANGE:",
                        df["year"].min(),
                        df["year"].max()
                    )

                for col in [
                    "country",
                    "src_country",
                    "tgt_country",
                    "sector"
                ]:
                    if col in df.columns:
                        print(
                            f"{col}_UNIQUE:",
                            df[col].nunique()
                        )

            elif suffix == ".npy":

                arr = np.load(f, allow_pickle=True)

                print("SHAPE:", arr.shape)
                print("DTYPE:", arr.dtype)

                if arr.size > 0:
                    print("MIN:", np.min(arr))
                    print("MAX:", np.max(arr))

            elif suffix == ".json":

                with open(f) as fp:
                    obj = json.load(fp)

                print("TYPE:", type(obj).__name__)

                if isinstance(obj, dict):
                    print("TOP LEVEL KEYS:", len(obj))

                    keys = list(obj.keys())[:5]

                    print("SAMPLE KEYS:", keys)

                    sample_key = keys[0] if keys else None

                    if sample_key:
                        print(
                            "SAMPLE VALUE:",
                            str(obj[sample_key])[:500]
                        )

        except Exception as e:

            print("ERROR:", e)

print("\n")
print("=" * 80)
print("SPECIAL PHASE 4 CHECKS")
print("=" * 80)

special_files = [
    "data/processed/tariff_rates/sector_tariffs.parquet",
    "data/processed/shock_vectors/shock_us_232_steel_2018.parquet",
    "data/processed/edges/edges_2014.parquet",
    "data/processed/edges/edges_2021.parquet",
]

for sf in special_files:

    p = Path(sf)

    print("\n", sf)

    if p.exists():
        print("FOUND")
    else:
        print("MISSING")

print("\nAUDIT COMPLETE")