"""parse_wiod.py — WIOD → Parquet edge tables.

Parses all annual WIOD Excel files (2000–2014 by default) into edge-level
Parquet tables stored at PATHS["PROCESSED_EDGES"]/edges_{YEAR}.parquet.

Also parses the WIOD Socioeconomic Accounts file (wiot_sep_16_txt.zip
extracted text files, or the Excel SEA file) into per-year Parquet tables
stored at PATHS["PROCESSED_EDGES"]/socioeconomic_{YEAR}.parquet.

Usage
-----
    # Full run (skips years already parsed):
    python src/data/parse_wiod.py

    # Test mode (single file only — set TEST_MODE = True below):
    python src/data/parse_wiod.py

Configuration
-------------
All constants imported from config.py (project root). No hardcoded values
other than the test-mode toggle and the test file placeholder.
"""

import os
import sys

import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Test-mode toggle
# ---------------------------------------------------------------------------
TEST_MODE = False                       # ← set True to run one year only
TEST_FILE = "WIOT2014_Nov16_ROW.xlsb"       # ← placeholder: change as needed

# ---------------------------------------------------------------------------
# Project root & config import
# ---------------------------------------------------------------------------
_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config                                                    # noqa: E402

ROW_OFFSET     = config.GRAPH["WIOD_MATRIX_ROW_OFFSET"]        # 6 (0-indexed)
COL_OFFSET     = config.GRAPH["WIOD_MATRIX_COL_OFFSET"]        # 4
EDGE_THRESHOLD = config.GRAPH["EDGE_THRESHOLD"]                 # 0.001
COUNTRY_LIST   = config.GRAPH["COUNTRY_LIST"]                   # 44 entries
SECTOR_LIST    = config.GRAPH["SECTOR_LIST"]                    # 56 entries
N_SECTORS      = config.GRAPH["N_SECTORS"]                      # 56
WIOD_YEARS     = config.GRAPH["WIOD_YEARS"]                     # 2000–2014
RAW_WIOD_DIR   = os.path.join(PROJECT_ROOT, config.PATHS["RAW_WIOD"])
OUT_DIR        = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_EDGES"])

os.makedirs(OUT_DIR, exist_ok=True)

N_IO = len(COUNTRY_LIST) * len(SECTOR_LIST)   # max 44 × 56 = 2464 nodes

# Country & sector → index lookup (used for src_id / tgt_id)
_COUNTRY_IDX = {c: i for i, c in enumerate(COUNTRY_LIST)}
_SECTOR_IDX  = {s: i for i, s in enumerate(SECTOR_LIST)}


# ---------------------------------------------------------------------------
# Helper: compute node id
# ---------------------------------------------------------------------------
def normalize_sector(sector: str) -> str:
    if sector is None:
        return ""
    return str(sector).strip().replace("-", "_")

def _node_id(country: str, sector: str) -> int:
    return _COUNTRY_IDX[country] * N_SECTORS + _SECTOR_IDX[sector]


# ---------------------------------------------------------------------------
# Step 1-11: parse a single WIOD annual XLSB file → edge DataFrame
# ---------------------------------------------------------------------------
def parse_single_year(xlsb_path: str, year: int) -> pd.DataFrame:
    """Parse one WIOD annual file.  Returns a clean edge DataFrame."""

    # ------------------------------------------------------------------
    # Step 1. Load Excel with pyxlsb engine, skip header rows
    #         skiprows = ROW_OFFSET rows so that row 0 of the returned
    #         DataFrame is the *country* header row.
    #         We then use COL_OFFSET to find where numeric columns start.
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_excel(
            xlsb_path,
            engine="pyxlsb",
            sheet_name=0,   # first sheet
            header=None,    # read everything as raw — NO skiprows;
            # ROW_OFFSET is the first *data* row; header rows are above it
        )

    # ------------------------------------------------------------------
    # Step 2. Extract (country, sector) header pairs from columns.
    #
    # WIOD layout (0-indexed absolute rows, confirmed by structure probe):
    #   row 2  → sector codes  ("A01", "A01", … repeated per country)
    #   row 4  → country codes ("AUS", "AUS", … repeated per sector)
    #   row ROW_OFFSET (6) → first data row
    #
    # We scan rows 0..ROW_OFFSET to find which row has the most sector
    # matches and which has the most country matches.  This is robust to
    # minor layout differences across WIOD vintage files.
    # ------------------------------------------------------------------
    countries_set = set(COUNTRY_LIST)
    sectors_set   = set(SECTOR_LIST)
  

    def _best_row(df_full, up_to_row, lookup_set):
        """Return index of the row (among rows 0..up_to_row-1) with most matches."""
        best_row, best_count = 0, 0
        for r in range(min(up_to_row, len(df_full))):
            count = sum(1 for v in df_full.iloc[r] if normalize_sector(v) in lookup_set)
            if count > best_count:
                best_count, best_row = count, r
        return best_row

    # Search for headers only in the pre-data rows
    sec_row_local = _best_row(raw, ROW_OFFSET, sectors_set)
    cou_row_local = _best_row(raw, ROW_OFFSET, countries_set)

    sector_hdr  = raw.iloc[sec_row_local]
    country_hdr = raw.iloc[cou_row_local]

    # Identify which column positions carry valid (country, sector) pairs
    cs_pairs = []   # list of (col_pos, country, sector)
    for col_pos in range(COL_OFFSET, len(sector_hdr)):
        sec = normalize_sector(sector_hdr.iloc[col_pos])
        cou = str(country_hdr.iloc[col_pos])
        if sec in sectors_set and cou in countries_set:
            cs_pairs.append((col_pos, cou, sec))

    if not cs_pairs:
        raise ValueError(f"[{year}] No valid (country, sector) column headers found.")

    # Map from column position → (country, sector)
    col_to_cs = {col: (c, s) for col, c, s in cs_pairs}
    valid_cols = [col for col, _, _ in cs_pairs]

    # ------------------------------------------------------------------
    # Step 3. Extract the N×N intermediate use sub-matrix.
    #
    # Data rows: rows in `raw` whose *row-label* (col 1 or 2) carries a
    # valid (country, sector) pair — i.e. we exclude value-added rows.
    # Data cols: only the valid_cols identified above (exclude final demand).
    # ------------------------------------------------------------------
    # First data row is always ROW_OFFSET (known from config / structure probe)
    first_data_raw_row = ROW_OFFSET

    # Build index of row → (country, sector) for data rows
    row_cs_map = {}   # raw_row_idx → (country, sector)
    for r_idx in range(first_data_raw_row, len(raw)):
        row = raw.iloc[r_idx]
        # Try several candidate label columns (0..COL_OFFSET-1)
        row_sec, row_cou = None, None
        for c in range(COL_OFFSET):
            val = normalize_sector(row.iloc[c])
            if val in sectors_set:
                row_sec = val
            if val in countries_set:
                row_cou = val
        if row_sec is not None and row_cou is not None:
            row_cs_map[r_idx] = (row_cou, row_sec)

    if not row_cs_map:
        raise ValueError(f"[{year}] No valid (country, sector) row labels found.")

    data_row_indices = list(row_cs_map.keys())


    # Extract numeric sub-matrix: data_rows × valid_cols
    sub = raw.iloc[data_row_indices][valid_cols].copy()
    sub = sub.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # ------------------------------------------------------------------
    # Step 4. Convert to long format.
    # ------------------------------------------------------------------
    records = []
    tgt_countries = [col_to_cs[c][0] for c in valid_cols]
    tgt_sectors   = [col_to_cs[c][1] for c in valid_cols]
    tgt_cs        = list(zip(tgt_countries, tgt_sectors))

    sub_values = sub.values   # numpy array (n_src_rows × n_tgt_cols)

    src_list = [row_cs_map[r] for r in data_row_indices]   # (country, sector)

    # Pre-compute tgt_total_input for Step 6 (sum over ALL data rows per tgt col)
    col_sums = sub_values.sum(axis=0)   # shape: (n_tgt_cols,)

    for src_i, (src_c, src_s) in enumerate(src_list):
        row_vals = sub_values[src_i]
        for tgt_j, (tgt_c, tgt_s) in enumerate(tgt_cs):
            flow = float(row_vals[tgt_j])
            # Step 5: remove non-positive flows early to save memory
            if flow <= 0.0:
                continue
            records.append((year, src_c, src_s, tgt_c, tgt_s, flow, float(col_sums[tgt_j])))

    if not records:
        raise ValueError(f"[{year}] No positive-flow edges found.")

    df = pd.DataFrame(
        records,
        columns=["year", "src_country", "src_sector",
                 "tgt_country", "tgt_sector", "flow_usd", "tgt_total_input"],
    )

    # ------------------------------------------------------------------
    # Step 7. Compute import_pen_coeff = flow_usd / tgt_total_input
    # ------------------------------------------------------------------
    df["import_pen_coeff"] = df["flow_usd"] / (df["tgt_total_input"] + 1e-9)

    # ------------------------------------------------------------------
    # Step 8. Clip at 0.99
    # ------------------------------------------------------------------
    df["import_pen_coeff"] = df["import_pen_coeff"].clip(upper=0.99)

    # ------------------------------------------------------------------
    # Save raw edges before filtering
    # ------------------------------------------------------------------
    df_raw = df.copy()
    if "tgt_total_input" in df_raw.columns:
        df_raw.drop(columns=["tgt_total_input"], inplace=True)
    df_raw["src_id"] = df_raw["src_country"].map(_COUNTRY_IDX) * N_SECTORS + df_raw["src_sector"].map(_SECTOR_IDX)
    df_raw["tgt_id"] = df_raw["tgt_country"].map(_COUNTRY_IDX) * N_SECTORS + df_raw["tgt_sector"].map(_SECTOR_IDX)
    df_raw["year"]            = df_raw["year"].astype("int16")
    df_raw["src_country"]     = df_raw["src_country"].astype("category")
    df_raw["src_sector"]      = df_raw["src_sector"].astype("category")
    df_raw["tgt_country"]     = df_raw["tgt_country"].astype("category")
    df_raw["tgt_sector"]      = df_raw["tgt_sector"].astype("category")
    df_raw["flow_usd"]        = df_raw["flow_usd"].astype("float32")
    df_raw["import_pen_coeff"]= df_raw["import_pen_coeff"].astype("float32")
    df_raw["src_id"]          = df_raw["src_id"].astype("int16")
    df_raw["tgt_id"]          = df_raw["tgt_id"].astype("int16")
    raw_out_path = os.path.join(OUT_DIR, f"edges_raw_{year}.parquet")
    df_raw.to_parquet(raw_out_path, engine="pyarrow", index=False)
    n_raw_edges = len(df_raw)
    print(f"Parsed {year}: {n_raw_edges} raw edges kept.")
    # ------------------------------------------------------------------
    # Step 9. Apply edge threshold filter
    # ------------------------------------------------------------------
    n_total = len(df)
    df = df[df["import_pen_coeff"] >= EDGE_THRESHOLD].copy()
    n_edges = len(df)

    # Drop tgt_total_input helper column (not in output schema)
    df.drop(columns=["tgt_total_input"], inplace=True)

    # ------------------------------------------------------------------
    # Steps 10-11. Compute src_id and tgt_id
    # ------------------------------------------------------------------
    df["src_id"] = df.apply(
        lambda r: _node_id(r["src_country"], r["src_sector"]), axis=1
    )
    df["tgt_id"] = df.apply(
        lambda r: _node_id(r["tgt_country"], r["tgt_sector"]), axis=1
    )

    # ------------------------------------------------------------------
    # Cast to output schema dtypes
    # ------------------------------------------------------------------
    df["year"]            = df["year"].astype("int16")
    df["src_country"]     = df["src_country"].astype("category")
    df["src_sector"]      = df["src_sector"].astype("category")
    df["tgt_country"]     = df["tgt_country"].astype("category")
    df["tgt_sector"]      = df["tgt_sector"].astype("category")
    df["flow_usd"]        = df["flow_usd"].astype("float32")
    df["import_pen_coeff"]= df["import_pen_coeff"].astype("float32")
    df["src_id"]          = df["src_id"].astype("int16")
    df["tgt_id"]          = df["tgt_id"].astype("int16")

    print(f"Parsed {year}: {n_edges} edges kept out of {n_total}")
    return df


# ---------------------------------------------------------------------------
# Socioeconomic accounts parser
# ---------------------------------------------------------------------------
# DATA sheet layout (Socio_Economic_Accounts.xlsx):
#   col 0  → country   (ISO-3 code, e.g. "AUS")
#   col 1  → variable  ("GO", "VA", "II", "EMP", "EMPE", "CAP", …)
#   col 2  → description (sector long name — ignored)
#   col 3  → code       (ISIC sector code, e.g. "A01")
#   col 4+ → year columns (2000.0, 2001.0, …, 2014.0) as float headers
#
# We filter variable == "GO" for gross_output and variable == "VA" for
# value_added, then melt year columns into long format.
# ---------------------------------------------------------------------------
_SEA_FILENAME = "Socio_Economic_Accounts.xlsx"
_SEA_SHEET    = "DATA"
_SEA_VAR_GO   = "GO"     # gross output variable code
_SEA_VAR_VA   = "VA"     # value added variable code


def parse_socioeconomic(raw_wiod_dir: str, out_dir: str) -> None:
    """Parse Socio_Economic_Accounts.xlsx → per-year Parquet files.

    Output per year: PATHS["PROCESSED_EDGES"]/socioeconomic_{YEAR}.parquet
    Schema: year int16, country category, sector category,
            gross_output float32 (millions USD), value_added float32
    """
    sea_path = os.path.join(raw_wiod_dir, _SEA_FILENAME)
    if not os.path.exists(sea_path):
        print(f"\n[SEA] File not found: {sea_path} — skipping.")
        return

    print(f"\n[SEA] Reading {sea_path} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_excel(
            sea_path,
            sheet_name=_SEA_SHEET,
            header=0,          # row 0 is the header
            engine="openpyxl",
        )

    # Normalise column names
    raw.columns = [str(c).strip() for c in raw.columns]

    # The first four columns carry fixed labels; the rest are year columns
    # whose header values come out as "2000.0", "2001.0" etc. from openpyxl.
    # Rename them to canonical names and convert year headers to int strings.
    fixed_cols = list(raw.columns[:4])   # ["country", "variable", "description", "code"]
    year_cols_raw = list(raw.columns[4:])  # e.g. ["2000.0", "2001.0", …]

    # Build a mapping from raw year-col name → int year (keep only WIOD years)
    year_col_map = {}   # raw_col_name → int year
    for col in year_cols_raw:
        try:
            yr = int(float(col))
            if yr in WIOD_YEARS:
                year_col_map[col] = yr
        except (ValueError, TypeError):
            continue

    if not year_col_map:
        print("[SEA] No matching year columns found — skipping.")
        return

    # Standardise the key columns (use positional names for safety)
    raw = raw.rename(columns={
        fixed_cols[0]: "country",
        fixed_cols[1]: "variable",
        fixed_cols[2]: "description",
        fixed_cols[3]: "sector",
    })

    # Drop rows with missing country/sector
    raw.dropna(subset=["country", "sector", "variable"], inplace=True)
    raw["country"]  = raw["country"].astype(str).str.strip()
    raw["sector"]   = raw["sector"].astype(str).str.strip()
    raw["variable"] = raw["variable"].astype(str).str.strip()

    # Select only GO and VA rows
    go_df = raw[raw["variable"] == _SEA_VAR_GO][["country", "sector"] + list(year_col_map.keys())].copy()
    va_df = raw[raw["variable"] == _SEA_VAR_VA][["country", "sector"] + list(year_col_map.keys())].copy()

    # Melt to long format
    def _melt(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        melted = df.melt(
            id_vars=["country", "sector"],
            value_vars=list(year_col_map.keys()),
            var_name="_yr_raw",
            value_name=value_name,
        )
        melted["year"] = melted["_yr_raw"].map(year_col_map)
        melted.drop(columns=["_yr_raw"], inplace=True)
        melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce")
        return melted[["country", "sector", "year", value_name]]

    go_long = _melt(go_df, "gross_output")
    va_long = _melt(va_df, "value_added")

    # Merge on country × sector × year
    sea = go_long.merge(va_long, on=["country", "sector", "year"], how="left")

    # Save one Parquet file per year
    for yr in sorted(year_col_map.values()):
        out_path = os.path.join(out_dir, f"socioeconomic_{yr}.parquet")
        if os.path.exists(out_path):
            print(f"[SEA] {yr}: already exists, skipping.")
            continue

        grp = sea[sea["year"] == yr].copy()
        grp["year"]         = grp["year"].astype("int16")
        grp["country"]      = grp["country"].astype("category")
        grp["sector"]       = grp["sector"].astype("category")
        grp["gross_output"] = grp["gross_output"].astype("float32")
        grp["value_added"]  = grp["value_added"].astype("float32")
        grp.to_parquet(out_path, engine="pyarrow", index=False)
        print(f"[SEA] Saved socioeconomic_{yr}.parquet  ({len(grp)} rows)")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    years_to_run = WIOD_YEARS if not TEST_MODE else [int(TEST_FILE[4:8])]

    for year in years_to_run:
        out_path = os.path.join(OUT_DIR, f"edges_{year}.parquet")
        raw_out_path = os.path.join(OUT_DIR, f"edges_raw_{year}.parquet")
        if os.path.exists(out_path) and os.path.exists(raw_out_path):
            print(f"Skipping {year}: {out_path} and {raw_out_path} already exist.")
            continue

        # WIOD file naming convention: WIOT{YEAR}_Nov16_ROW.xlsb
        fname = TEST_FILE if TEST_MODE else f"WIOT{year}_Nov16_ROW.xlsb"
        fpath = os.path.join(RAW_WIOD_DIR, fname)

        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found — skipping {year}.")
            continue

        try:
            df = parse_single_year(fpath, year)
            df.to_parquet(out_path, engine="pyarrow", index=False)
        except Exception as e:
            print(f"Error parsing {year}: {e}")

    # Parse socioeconomic accounts
    parse_socioeconomic(RAW_WIOD_DIR, OUT_DIR)


if __name__ == "__main__":
    main()
