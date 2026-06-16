from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "EDGES": ROOT / "data/processed/edges/edges_2000.parquet",
    "SOCIOECONOMIC": ROOT / "data/processed/edges/socioeconomic_2000.parquet",
    "TARIFF": ROOT / "data/processed/tariff_rates/sector_tariffs.parquet",
    "BACKWARD_LINKAGE": ROOT / "data/processed/leontief/backward_linkage_2000.npy",
}


def audit_parquet(path, name):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    if not path.exists():
        print("NOT FOUND:", path)
        return

    df = pd.read_parquet(path)

    print("PATH:", path)
    print("ROWS:", len(df))
    print("COLS:", len(df.columns))

    print("\nCOLUMN NAMES:")
    for c in df.columns:
        print(" ", c)

    print("\nDTYPES:")
    print(df.dtypes)

    print("\nHEAD:")
    print(df.head())

    print("\nNULL COUNTS:")
    print(df.isnull().sum())

    print("\nMEMORY MB:")
    print(round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))

    print("\nUNIQUE COUNTS:")
    for c in df.columns[:20]:
        try:
            print(f"{c}: {df[c].nunique()}")
        except Exception:
            pass


def audit_npy(path):
    print("\n" + "=" * 100)
    print("BACKWARD LINKAGE")
    print("=" * 100)

    if not path.exists():
        print("NOT FOUND:", path)
        return

    arr = np.load(path)

    print("PATH:", path)
    print("SHAPE:", arr.shape)
    print("DTYPE:", arr.dtype)

    print("\nFIRST 20 VALUES:")
    print(arr[:20])

    print("\nMIN:", np.nanmin(arr))
    print("MAX:", np.nanmax(arr))
    print("MEAN:", np.nanmean(arr))
    print("STD:", np.nanstd(arr))

    print("\nNAN COUNT:")
    print(np.isnan(arr).sum())

    print("\nINF COUNT:")
    print(np.isinf(arr).sum())


if __name__ == "__main__":

    audit_parquet(FILES["EDGES"], "EDGES_2000")

    audit_parquet(
        FILES["SOCIOECONOMIC"],
        "SOCIOECONOMIC_2000"
    )

    audit_parquet(
        FILES["TARIFF"],
        "TARIFF_RATES"
    )

    audit_npy(
        FILES["BACKWARD_LINKAGE"]
    )

    print("\nAUDIT COMPLETE")