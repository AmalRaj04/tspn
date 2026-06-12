import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PINK_SHEET = ROOT / "data/raw/commodity_prices/wb_pink_sheet.xlsx"
NAICS_ISIC = ROOT / "data/processed/concordance/naics3_isic.json"

print("=" * 100)
print("WORLD BANK PINK SHEET AUDIT")
print("=" * 100)

if not PINK_SHEET.exists():
    print(f"NOT FOUND: {PINK_SHEET}")
else:
    print(f"FILE: {PINK_SHEET}")
    print(f"SIZE_MB: {round(PINK_SHEET.stat().st_size / 1024 / 1024, 2)}")

    xls = pd.ExcelFile(PINK_SHEET)

    print("\nSHEETS:")
    for sheet in xls.sheet_names:
        print(f"  - {sheet}")

    print("\n" + "=" * 80)

    for sheet in xls.sheet_names:
        print(f"\nSHEET: {sheet}")
        print("-" * 80)

        try:
            df = pd.read_excel(PINK_SHEET, sheet_name=sheet)

            print(f"ROWS: {len(df)}")
            print(f"COLS: {len(df.columns)}")

            print("\nCOLUMN NAMES:")
            for col in df.columns:
                print(f"  {col}")

            print("\nHEAD:")
            print(df.head(5))

            print("\nNON-NULL COUNTS:")
            print(df.count())

        except Exception as e:
            print(f"ERROR READING SHEET: {e}")

print("\n\n")
print("=" * 100)
print("NAICS → ISIC CONCORDANCE AUDIT")
print("=" * 100)

if not NAICS_ISIC.exists():
    print(f"NOT FOUND: {NAICS_ISIC}")
else:
    print(f"FILE: {NAICS_ISIC}")
    print(f"SIZE_KB: {round(NAICS_ISIC.stat().st_size / 1024, 2)}")

    with open(NAICS_ISIC, "r") as f:
        mapping = json.load(f)

    print(f"\nTYPE: {type(mapping)}")
    print(f"TOTAL MAPPINGS: {len(mapping)}")

    print("\nFIRST 50 MAPPINGS:")
    print("-" * 80)

    for i, (k, v) in enumerate(mapping.items()):
        print(f"{k} -> {v}")

        if i >= 49:
            break

    print("\nUNIQUE ISIC VALUES:")
    unique_isic = sorted(set(str(v) for v in mapping.values()))

    print(f"COUNT: {len(unique_isic)}")

    for x in unique_isic:
        print(x)

print("\n\nAUDIT COMPLETE")