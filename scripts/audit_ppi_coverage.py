import pandas as pd

PPI_PATH = "data/processed/labels/ppi_quarterly_all.parquet"

df = pd.read_parquet(PPI_PATH)

print("\n==============================")
print("BASIC SUMMARY")
print("==============================")
print("Rows:", len(df))
print("Countries:", df["country"].nunique())
print("Sectors:", df["isic_sector"].nunique())
print("Years:", sorted(df["year"].unique())[:5], "...", sorted(df["year"].unique())[-5:])

print("\n==============================")
print("SOURCE COVERAGE")
print("==============================")
print(df["source"].value_counts())

print("\n==============================")
print("COUNTRY COVERAGE")
print("==============================")
country_cov = (
    df.groupby("country")
      .size()
      .sort_values(ascending=False)
)

print(country_cov)

print("\n==============================")
print("SECTOR COVERAGE")
print("==============================")
sector_cov = (
    df.groupby("isic_sector")
      .size()
      .sort_values(ascending=False)
)

print(sector_cov)

print("\n==============================")
print("NODE COVERAGE")
print("==============================")

covered_nodes = (
    df[["country", "isic_sector"]]
      .drop_duplicates()
)

n_covered = len(covered_nodes)

TOTAL_NODES = 44 * 56

print(f"Covered nodes : {n_covered}")
print(f"Total nodes   : {TOTAL_NODES}")
print(f"Coverage      : {100*n_covered/TOTAL_NODES:.2f}%")

print("\n==============================")
print("MISSING COUNTRIES")
print("==============================")

all_countries = {
    "AUS","AUT","BEL","BGR","BRA","CAN","CHE","CHN","CYP","CZE",
    "DEU","DNK","ESP","EST","FIN","FRA","GBR","GRC","HRV","HUN",
    "IDN","IND","IRL","ITA","JPN","KOR","LTU","LUX","LVA","MEX",
    "MLT","NLD","NOR","POL","PRT","ROU","ROW","RUS","SVK","SVN",
    "SWE","TUR","TWN","USA"
}

present = set(df["country"].unique())

missing = sorted(all_countries - present)

print(missing)

print("\n==============================")
print("MISSING SECTORS")
print("==============================")

all_sectors = {
    "A01","A02","A03","B","C10_C12","C13_C15","C16","C17","C18",
    "C19","C20","C21","C22","C23","C24","C25","C26","C27","C28",
    "C29","C30","C31_C32","C33","D35","E36","E37_E39","F",
    "G45","G46","G47","H49","H50","H51","H52","H53",
    "I","J58","J59_J60","J61","J62_J63","K64","K65","K66",
    "L68","M69_M70","M71","M72","M73","M74_M75",
    "N","O84","P85","Q","R_S","T"
}

present = set(df["isic_sector"].unique())

missing = sorted(all_sectors - present)

print(missing)