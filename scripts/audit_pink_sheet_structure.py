import pandas as pd

FILE = "data/raw/commodity_prices/wb_pink_sheet.xlsx"

df = pd.read_excel(
    FILE,
    sheet_name="Monthly Prices",
    header=None
)

print("=" * 120)
print("SHAPE")
print("=" * 120)
print(df.shape)

print("\n")
print("=" * 120)
print("FIRST 25 ROWS")
print("=" * 120)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

print(df.head(25))

print("\n")
print("=" * 120)
print("ROWS 0-15")
print("=" * 120)

for i in range(15):
    print(f"\nROW {i}")
    print(df.iloc[i].tolist())