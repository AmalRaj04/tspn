# scripts/audit_ppi_inputs.py

from pathlib import Path
import pandas as pd

DIRS = [
    "data/raw/bls_ppi",
    "data/raw/eurostat_ppi",
]

for d in DIRS:
    p = Path(d)

    print("\n" + "=" * 80)
    print(d)
    print("=" * 80)

    if not p.exists():
        print("MISSING")
        continue

    files = sorted(p.glob("*"))
    print("FILE COUNT:", len(files))

    for f in files:
        print("\nFILE:", f.name)
        print("SIZE_MB:", round(f.stat().st_size / 1024 / 1024, 2))

        try:

            if f.suffix.lower() == ".csv":
                df = pd.read_csv(f, nrows=5)

                print("COLS:", list(df.columns))
                print("HEAD:")
                print(df.head())

            elif f.suffix.lower() in [".xlsx", ".xls"]:
                xl = pd.ExcelFile(f)

                print("SHEETS:")
                print(xl.sheet_names)

            elif f.suffix.lower() == ".json":
                print("JSON FILE")

        except Exception as e:
            print("ERROR:", e)