import os
import sys
from pyxlsb import open_workbook

# Add project root to sys.path to import config.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config


def probe_wiod_structure():
    file_path = os.path.join(project_root, config.PATHS["RAW_WIOD"], "WIOT2014_Nov16_ROW.xlsb")
    if not os.path.exists(file_path):
        print(f"Error: WIOD file not found at {file_path}")
        sys.exit(1)

    countries_set = set(config.GRAPH["COUNTRY_LIST"])
    sectors_set = set(config.GRAPH["SECTOR_LIST"])

    sector_row_idx = None
    country_row_idx = None
    first_data_row_idx = None

    with open_workbook(file_path) as wb:
        # Get sheet "2014" or fall back to the first sheet
        sheet_name = "2014" if "2014" in wb.sheets else wb.sheets[0]
        with wb.get_sheet(sheet_name) as sheet:
            # Read first 300 rows
            rows = []
            for idx, r in enumerate(sheet.rows()):
                rows.append([cell.v for cell in r])
                if idx >= 299:
                    break

    # Find sector header row and country header row based on matching codes
    best_sector_row = -1
    max_sector_matches = 0
    best_country_row = -1
    max_country_matches = 0

    for r_idx, row in enumerate(rows):
        sector_matches = sum(1 for val in row if str(val) in sectors_set)
        country_matches = sum(1 for val in row if str(val) in countries_set)

        if sector_matches > max_sector_matches:
            max_sector_matches = sector_matches
            best_sector_row = r_idx

        if country_matches > max_country_matches:
            max_country_matches = country_matches
            best_country_row = r_idx

    sector_row_idx = best_sector_row
    country_row_idx = best_country_row

    # Determine where the data columns start (col_start)
    # It starts at the first column where both sector and country are valid config values
    col_start = None
    if sector_row_idx is not None and country_row_idx is not None:
        sector_row = rows[sector_row_idx]
        country_row = rows[country_row_idx]
        for c_idx in range(min(len(sector_row), len(country_row))):
            if str(sector_row[c_idx]) in sectors_set and str(country_row[c_idx]) in countries_set:
                col_start = c_idx
                break

    # Find the first numeric data row after the headers at the data column
    if col_start is not None:
        header_max_row = max(sector_row_idx, country_row_idx)
        for r_idx in range(header_max_row + 1, len(rows)):
            val = rows[r_idx][col_start]
            if isinstance(val, (int, float)):
                first_data_row_idx = r_idx
                break

    # Gather the unique sectors and countries from the headers starting from col_start
    detected_sectors = []
    detected_countries = []

    if col_start is not None:
        # Scan columns starting from col_start until we hit empty values
        sector_row = rows[sector_row_idx]
        country_row = rows[country_row_idx]
        c = col_start
        while c < len(sector_row) and c < len(country_row):
            sec_val = sector_row[c]
            cou_val = country_row[c]
            if sec_val is None or cou_val is None:
                break
            if str(sec_val) in sectors_set:
                if sec_val not in detected_sectors:
                    detected_sectors.append(sec_val)
            if str(cou_val) in countries_set:
                if cou_val not in detected_countries:
                    detected_countries.append(cou_val)
            c += 1

        num_cols = c - col_start
    else:
        num_cols = 0

    # Calculate estimated matrix dimensions
    # Number of rows = (number of countries * number of sectors)
    # Number of columns = (number of countries * number of sectors)
    # Plus whatever final demand / other columns exist, but here we can estimate based on detected sectors/countries
    num_countries = len(detected_countries)
    num_sectors = len(detected_sectors)
    est_rows = num_countries * num_sectors
    est_cols = num_countries * num_sectors

    # Print results in required format
    print(f"Sector header row: {sector_row_idx}")
    print(f"Country header row: {country_row_idx}")
    print(f"First data row: {first_data_row_idx}")
    print(f"Detected sectors: {detected_sectors}")
    print(f"Detected countries: {detected_countries}")
    print(f"Estimated matrix dimensions: {est_rows} x {est_cols}")


if __name__ == "__main__":
    probe_wiod_structure()
