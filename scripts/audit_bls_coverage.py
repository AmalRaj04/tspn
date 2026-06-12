import os
import pandas as pd

path = "data/raw/bls_ppi"

for f in sorted(os.listdir(path)):
    if not f.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(path, f))

    sector = f.replace("bls_ppi_", "").replace(".csv", "")

    print(
        sector,
        len(df),
        "OK" if len(df) > 0 else "MISSING"
    )