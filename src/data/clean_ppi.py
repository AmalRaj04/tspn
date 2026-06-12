"""Unify BLS PPI, Eurostat PPI, and World Bank commodity prices into a single quarterly dataset.

The script processes monthly price level data, calculates monthly percentage changes,
aggregates them to calendar quarters (by taking the mean of monthly changes),
maps sectors to WIOD-56 classification, and saves the clean dataset to Parquet.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Project root & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

# ---------------------------------------------------------------------------
# ISO-2 to ISO-3 Country Mapping
# ---------------------------------------------------------------------------
ISO2_TO_ISO3 = {
    "AT": "AUT",
    "BE": "BEL",
    "BG": "BGR",
    "CY": "CYP",
    "CZ": "CZE",
    "DE": "DEU",
    "DK": "DNK",
    "EE": "EST",
    "EL": "GRC",
    "ES": "ESP",
    "FI": "FIN",
    "FR": "FRA",
    "HR": "HRV",
    "HU": "HUN",
    "IE": "IRL",
    "IT": "ITA",
    "LT": "LTU",
    "LU": "LUX",
    "LV": "LVA",
    "MT": "MLT",
    "NL": "NLD",
    "PL": "POL",
    "PT": "PRT",
    "RO": "ROU",
    "SE": "SWE",
    "SI": "SVN",
    "SK": "SVK",
    "UK": "GBR",
    "GB": "GBR"
}

# Eurostat NACE -> WIOD-56 sector mapping
MANUFACTURING_SECTORS = [
    "C10_C12", "C13_C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22",
    "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31_C32", "C33"
]

NACE_TO_WIOD = {
    "B": ["B"],
    "C": MANUFACTURING_SECTORS,
    "MANUFACTURING": MANUFACTURING_SECTORS,
    "D": ["D35"],
    "D35": ["D35"]
}

# World Bank Excel column name to config commodity key
EXCEL_COMMODITY_MAP = {
    "Aluminum": "aluminum",
    "Copper": "copper",
    "Iron ore, cfr spot": "iron_ore",
    "Coal, Australian": "coal",
    "Crude oil, Brent": "brent_oil",
    "Maize": "corn",
    "Wheat, US SRW": "wheat",
    "Wheat, US HRW": "wheat",
    "Soybeans": "soy"
}


def process_series_levels_to_quarterly_changes(df: pd.DataFrame, val_col: str = "value") -> pd.DataFrame:
    """Computes monthly percentage changes and aggregates to quarterly changes.

    Ensures percentage changes are only calculated between consecutive calendar months.
    """
    if df.empty:
        return pd.DataFrame(columns=["year", "quarter", "ppi_change"])

    # Clean and parse columns
    df = df.dropna(subset=["year", "month", val_col]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=[val_col])

    if df.empty:
        return pd.DataFrame(columns=["year", "quarter", "ppi_change"])

    # Create PeriodIndex for chronological sorting & gap detection
    df["period"] = pd.PeriodIndex(year=df["year"], month=df["month"], freq="M")
    df = df.sort_values("period").drop_duplicates(subset=["period"])
    df = df.set_index("period")

    # Reindex to a complete monthly grid to make gaps explicit as NaNs
    all_periods = pd.period_range(start=df.index.min(), end=df.index.max(), freq="M")
    df = df.reindex(all_periods)

    # Compute percentage changes: (level_t - level_{t-1}) / level_{t-1}
    df["prev_val"] = df[val_col].shift(1)
    df["monthly_change"] = (df[val_col] - df["prev_val"]) / df["prev_val"]

    # Reconstruct year and month from index
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)

    # Drop rows where monthly_change is missing (which includes the shifted first row)
    df = df.dropna(subset=["monthly_change"])
    if df.empty:
        return pd.DataFrame(columns=["year", "quarter", "ppi_change"])

    # Group by year and quarter and take the mean of monthly changes
    agg = df.groupby(["year", "quarter"])["monthly_change"].mean().reset_index()
    agg.rename(columns={"monthly_change": "ppi_change"}, inplace=True)
    return agg


# ---------------------------------------------------------------------------
# 1. BLS PPI Loader
# ---------------------------------------------------------------------------
def load_bls_ppi(raw_dir: str, concordance_path: str) -> tuple[pd.DataFrame, float]:
    """Loads and cleans BLS PPI series.

    Returns the quarterly aggregated DataFrame and the percentage of missing mappings.
    """
    if not os.path.exists(concordance_path):
        print(f"Warning: concordance file not found at {concordance_path}")
        concordance = {}
    else:
        with open(concordance_path, "r") as f:
            concordance = json.load(f)

    records = []
    pattern = os.path.join(raw_dir, "bls_ppi_*.csv")
    csv_files = glob.glob(pattern)

    for fpath in csv_files:
        filename = os.path.basename(fpath)
        m = re.search(r"bls_ppi_(\d+)\.csv", filename)
        if not m:
            continue
        naics_code = m.group(1)
        naics3 = naics_code[:3]

        if os.path.getsize(fpath) < 20:
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

        if df.empty or "period" not in df.columns or "value" not in df.columns:
            continue

        # Extract month from period
        df = df[df["period"].str.startswith("M", na=False)].copy()
        df["month"] = df["period"].str[1:].astype(int)
        df = df[(df["month"] >= 1) & (df["month"] <= 12)]

        if df.empty:
            continue

        q_df = process_series_levels_to_quarterly_changes(df, val_col="value")
        if q_df.empty:
            continue

        # Map to WIOD-56 sector
        sector = concordance.get(naics3)

        q_df["country"] = "USA"
        q_df["isic_sector"] = sector
        q_df["source"] = "bls"

        records.append(q_df)

    if not records:
        return pd.DataFrame(columns=["year", "quarter", "country", "isic_sector", "ppi_change", "source"]), 0.0

    final_df = pd.concat(records, ignore_index=True)
    n_missing = final_df["isic_sector"].isna().sum()
    pct_missing = (n_missing / len(final_df)) * 100 if len(final_df) > 0 else 0.0

    # Filter out missing sector mappings for downstream
    clean_df = final_df.dropna(subset=["isic_sector"]).copy()
    return clean_df, pct_missing


# ---------------------------------------------------------------------------
# 2. Eurostat PPI Loader
# ---------------------------------------------------------------------------
def parse_eurostat_date(date_str: str) -> tuple[int | None, int | None]:
    if not isinstance(date_str, str):
        return None, None
    if len(date_str) >= 7 and date_str[4] == '-':
        try:
            return int(date_str[:4]), int(date_str[5:7])
        except ValueError:
            pass
    if len(date_str) >= 7 and 'M' in date_str:
        parts = date_str.split('M')
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None


def parse_eurostat_file(fpath: str) -> list[dict]:
    try:
        with open(fpath, "r") as f:
            data = json.load(f)
    except Exception:
        return []

    if not isinstance(data, dict) or "error" in data:
        return []

    datasets = []
    if all(k in data for k in ["id", "size", "dimension"]):
        datasets.append(data)
    elif "dataset" in data:
        datasets.append(data["dataset"])
    else:
        for v in data.values():
            if isinstance(v, dict) and all(dk in v for dk in ["id", "size", "dimension"]):
                datasets.append(v)

    records = []
    filename = os.path.basename(fpath)
    fn_geo = None
    fn_nace = None
    # Filename fallback: e.g. AT_B.json -> geo="AT", nace="B"
    m = re.search(r"^([A-Za-z]{2,3})_([A-Za-z0-9_]+)\.json$", filename)
    if m:
        fn_geo = m.group(1).upper()
        fn_nace = m.group(2).upper()
        if fn_nace == "MANUFACTURING":
            fn_nace = "C"

    for ds in datasets:
        dims = ds.get("id", [])
        sizes = ds.get("size", [])
        dimension = ds.get("dimension", {})

        cat_lists = {}
        for dim_name in dims:
            dim = dimension.get(dim_name, {})
            cat = dim.get("category", {})
            idx_info = cat.get("index")
            if isinstance(idx_info, list):
                cat_lists[dim_name] = idx_info
            elif isinstance(idx_info, dict):
                cat_lists[dim_name] = sorted(idx_info.keys(), key=lambda k: idx_info[k])
            else:
                label_info = cat.get("label", {})
                cat_lists[dim_name] = list(label_info.keys())

        # Pad category list if smaller than size
        for i, dim_name in enumerate(dims):
            lst = cat_lists.get(dim_name, [])
            if len(lst) < sizes[i]:
                cat_lists[dim_name] = lst + [None] * (sizes[i] - len(lst))

        def get_coords(flat_idx, sizes):
            coords = []
            curr = flat_idx
            for s in reversed(sizes):
                coords.append(curr % s)
                curr = curr // s
            return list(reversed(coords))

        value_obj = ds.get("value", {})
        if isinstance(value_obj, dict):
            iterator = value_obj.items()
        elif isinstance(value_obj, list):
            iterator = enumerate(value_obj)
        else:
            continue

        geo_key = None
        nace_key = None
        time_key = None

        for key in dims:
            kl = key.lower()
            if "geo" in kl or kl == "country":
                geo_key = key
            elif "nace" in kl or "sector" in kl or kl == "activity":
                nace_key = key
            elif "time" in kl or "date" in kl or kl == "period":
                time_key = key

        if not geo_key or not nace_key or not time_key:
            for d_name in dims:
                label = dimension.get(d_name, {}).get("label", "").lower()
                if "geopolitical" in label or "country" in label:
                    geo_key = d_name
                elif "nace" in label or "sector" in label or "activity" in label:
                    nace_key = d_name
                elif "time" in label or "period" in label:
                    time_key = d_name

        for flat_idx_str, val in iterator:
            if val is None or val == "NaN" or val == ":":
                continue
            try:
                flat_idx = int(flat_idx_str)
                coords = get_coords(flat_idx, sizes)
                if any(c >= sizes[i] for i, c in enumerate(coords)):
                    continue

                geo_val = cat_lists[geo_key][coords[dims.index(geo_key)]] if geo_key else None
                nace_val = cat_lists[nace_key][coords[dims.index(nace_key)]] if nace_key else None
                time_val = cat_lists[time_key][coords[dims.index(time_key)]] if time_key else None

                if not geo_val:
                    geo_val = fn_geo
                if not nace_val:
                    nace_val = fn_nace

                if geo_val and nace_val and time_val:
                    records.append({
                        "geo": geo_val,
                        "nace": nace_val,
                        "time": time_val,
                        "value": float(val)
                    })
            except Exception:
                continue

    return records


def map_country(geo: str) -> str | None:
    geo_upper = geo.upper().strip()
    if len(geo_upper) == 3:
        return geo_upper
    return ISO2_TO_ISO3.get(geo_upper, None)


def map_nace_sector(nace: str) -> list[str]:
    nace_upper = nace.upper().strip()
    return NACE_TO_WIOD.get(nace_upper, [])


def load_eurostat_ppi(raw_dir: str) -> tuple[pd.DataFrame, float]:
    """Loads and parses Eurostat JSON-stat files.

    Returns the quarterly aggregated DataFrame and the percentage of missing mappings.
    """
    all_json_files = glob.glob(os.path.join(raw_dir, "*.json"))
    records = []

    for fpath in all_json_files:
        filename = os.path.basename(fpath)
        if filename == "eurostat_ppi_raw.json":
            continue

        file_records = parse_eurostat_file(fpath)
        if not file_records:
            continue

        df = pd.DataFrame(file_records)
        for (geo, nace), group_df in df.groupby(["geo", "nace"]):
            iso3_country = map_country(geo)
            if not iso3_country:
                continue

            parsed_dates = [parse_eurostat_date(d) for d in group_df["time"]]
            group_df["year"] = [d[0] for d in parsed_dates]
            group_df["month"] = [d[1] for d in parsed_dates]

            q_df = process_series_levels_to_quarterly_changes(group_df, val_col="value")
            if q_df.empty:
                continue

            wiod_sectors = map_nace_sector(nace)
            if not wiod_sectors:
                row_df = q_df.copy()
                row_df["country"] = iso3_country
                row_df["isic_sector"] = None
                row_df["source"] = "eurostat"
                records.append(row_df)
            else:
                for sector in wiod_sectors:
                    row_df = q_df.copy()
                    row_df["country"] = iso3_country
                    row_df["isic_sector"] = sector
                    row_df["source"] = "eurostat"
                    records.append(row_df)

    if not records:
        return pd.DataFrame(columns=["year", "quarter", "country", "isic_sector", "ppi_change", "source"]), 0.0

    final_df = pd.concat(records, ignore_index=True)
    n_missing = final_df["isic_sector"].isna().sum()
    pct_missing = (n_missing / len(final_df)) * 100 if len(final_df) > 0 else 0.0

    clean_df = final_df.dropna(subset=["isic_sector"]).copy()
    return clean_df, pct_missing


# ---------------------------------------------------------------------------
# 3. World Bank Pink Sheet Loader
# ---------------------------------------------------------------------------
def parse_wb_date(date_str: str) -> tuple[int | None, int | None]:
    if not isinstance(date_str, str):
        return None, None
    if len(date_str) >= 7 and "M" in date_str:
        parts = date_str.split("M")
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None


def load_wb_commodity_prices(fpath: str) -> tuple[pd.DataFrame, float]:
    """Loads and cleans World Bank Pink Sheet commodity prices.

    Returns the quarterly aggregated DataFrame and the percentage of missing mappings.
    """
    try:
        df = pd.read_excel(fpath, sheet_name="Monthly Prices", skiprows=4)
    except Exception as e:
        print(f"Error reading World Bank Pink Sheet: {e}")
        return pd.DataFrame(columns=["year", "quarter", "country", "isic_sector", "ppi_change", "source"]), 0.0

    if df.empty:
        return pd.DataFrame(columns=["year", "quarter", "country", "isic_sector", "ppi_change", "source"]), 0.0

    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df = df.iloc[1:].copy()

    parsed_dates = [parse_wb_date(d) for d in df["date"]]
    df["year"] = [d[0] for d in parsed_dates]
    df["month"] = [d[1] for d in parsed_dates]
    df = df.dropna(subset=["year", "month"])

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    records = []

    for col_name, config_key in EXCEL_COMMODITY_MAP.items():
        if col_name not in df.columns:
            print(f"World Bank: Column '{col_name}' not found in sheet — skipping.")
            continue

        series_df = df[["year", "month", col_name]].copy()
        series_df[col_name] = pd.to_numeric(series_df[col_name], errors="coerce")
        series_df = series_df.dropna(subset=[col_name])

        if series_df.empty:
            continue

        q_df = process_series_levels_to_quarterly_changes(series_df, val_col=col_name)
        if q_df.empty:
            continue

        sector = config.COMMODITY_TO_ISIC.get(config_key)

        q_df["country"] = "WLD"
        q_df["isic_sector"] = sector
        q_df["source"] = "wb_commodity"

        records.append(q_df)

    if not records:
        return pd.DataFrame(columns=["year", "quarter", "country", "isic_sector", "ppi_change", "source"]), 0.0

    final_df = pd.concat(records, ignore_index=True)
    n_missing = final_df["isic_sector"].isna().sum()
    pct_missing = (n_missing / len(final_df)) * 100 if len(final_df) > 0 else 0.0

    clean_df = final_df.dropna(subset=["isic_sector"]).copy()
    return clean_df, pct_missing


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    bls_raw_dir = os.path.join(PROJECT_ROOT, config.PATHS["RAW_BLS_PPI"])
    concordance_path = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_CONCORDANCE"], "naics3_to_wiod56.json")
    eurostat_raw_dir = os.path.join(PROJECT_ROOT, config.PATHS["RAW_EUROSTAT_PPI"])
    wb_file_path = os.path.join(PROJECT_ROOT, config.PATHS["RAW_COMMODITY"], "wb_pink_sheet.xlsx")

    print("=== Processing BLS PPI ===")
    bls_df, bls_missing = load_bls_ppi(bls_raw_dir, concordance_path)

    print("=== Processing Eurostat PPI ===")
    eurostat_df, eurostat_missing = load_eurostat_ppi(eurostat_raw_dir)

    print("=== Processing World Bank Commodity Prices ===")
    wb_df, wb_missing = load_wb_commodity_prices(wb_file_path)

    # Combine all dataframes
    all_dfs = []
    if not bls_df.empty:
        all_dfs.append(bls_df)
    if not eurostat_df.empty:
        all_dfs.append(eurostat_df)
    if not wb_df.empty:
        all_dfs.append(wb_df)

    if not all_dfs:
        print("Error: No data successfully processed from any source.")
        sys.exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates and aggregate duplicates by taking the mean.
    # Grouping by keys guarantees a single unique change value per (year, quarter, country, isic_sector)
    combined_df = combined_df.groupby(["year", "quarter", "country", "isic_sector", "source"], observed=True).mean().reset_index()

    # Enforce correct output schemas and column types
    combined_df["year"] = combined_df["year"].astype("int16")
    combined_df["quarter"] = combined_df["quarter"].astype("int8")
    combined_df["country"] = combined_df["country"].astype("category")
    combined_df["isic_sector"] = combined_df["isic_sector"].astype("category")
    combined_df["ppi_change"] = combined_df["ppi_change"].astype("float32")
    combined_df["source"] = combined_df["source"].astype("category")

    # Drop any leftover duplicates and NaNs
    combined_df = combined_df.drop_duplicates(subset=["year", "quarter", "country", "isic_sector"])
    combined_df = combined_df.dropna(subset=["year", "quarter", "country", "isic_sector", "ppi_change"])

    # Sort final output
    combined_df = combined_df.sort_values(["year", "quarter", "country", "isic_sector"]).reset_index(drop=True)

    # Save output to parquet
    out_dir = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_LABELS"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ppi_quarterly_all.parquet")
    combined_df.to_parquet(out_path, index=False)

    # Counts
    n_rows = len(combined_df)
    n_countries = combined_df["country"].nunique()
    n_sectors = combined_df["isic_sector"].nunique()

    bls_rows = len(combined_df[combined_df["source"] == "bls"])
    eurostat_rows = len(combined_df[combined_df["source"] == "eurostat"])
    wb_rows = len(combined_df[combined_df["source"] == "wb_commodity"])

    # Output Validation Report
    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    print(f"PPI unified: {n_rows} rows from {n_countries} countries and {n_sectors} sectors")
    print(f"BLS rows: {bls_rows} | Eurostat rows: {eurostat_rows} | World Bank rows: {wb_rows}")
    print(f"Missing sector mappings percentage:")
    print(f"  BLS: {bls_missing:.2f}%")
    print(f"  Eurostat: {eurostat_missing:.2f}%")
    print(f"  World Bank: {wb_missing:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
