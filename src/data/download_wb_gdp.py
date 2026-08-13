"""download_wb_gdp.py — Fetch World Bank nominal GDP (current USD) per country.

Used to proxy WIOD gross_output / value_added growth for 2015-2021, since
WIOD's Socioeconomic Accounts stop at 2014 (see PROJECT_STATE.md §2 decision
5). Indicator NY.GDP.MKTP.CD requires no API key and no registration.

Output: data/raw/wb_gdp/gdp_current_usd.parquet
    Schema: country (str, ISO3), year (int), gdp_usd (float)

Usage:
    python src/data/download_wb_gdp.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

import pandas as pd

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

COUNTRY_LIST = config.GRAPH["COUNTRY_LIST"]
OUT_DIR = os.path.join(PROJECT_ROOT, config.PATHS["RAW_WB_GDP"])
OUT_PATH = os.path.join(OUT_DIR, "gdp_current_usd.parquet")

INDICATOR = "NY.GDP.MKTP.CD"
DATE_RANGE = "2000:2021"
BASE_URL = "https://api.worldbank.org/v2/country/{iso3}/indicator/{ind}?format=json&date={date}&per_page=100"


def fetch_country(iso3: str) -> list[dict]:
    url = BASE_URL.format(iso3=iso3, ind=INDICATOR, date=DATE_RANGE)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                payload = json.loads(r.read())
            break
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  {iso3}: attempt {attempt+1} failed ({e}), retrying...")
            time.sleep(2)
    else:
        print(f"  {iso3}: FAILED after 3 attempts")
        return []

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        print(f"  {iso3}: no data returned by WB API")
        return []

    rows = []
    for rec in payload[1]:
        val = rec.get("value")
        yr = rec.get("date")
        if val is None or yr is None:
            continue
        rows.append({"country": iso3, "year": int(yr), "gdp_usd": float(val)})
    return rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(OUT_PATH):
        print(f"Already exists: {OUT_PATH} — delete it to refetch.")
        return

    all_rows: list[dict] = []
    for country in COUNTRY_LIST:
        print(f"Fetching GDP for {country}...")
        rows = fetch_country(country)
        all_rows.extend(rows)
        print(f"  {country}: {len(rows)} year-observations")

    df = pd.DataFrame(all_rows)
    df["country"] = df["country"].astype("category")
    df["year"] = df["year"].astype("int16")
    df["gdp_usd"] = df["gdp_usd"].astype("float64")
    df.to_parquet(OUT_PATH, engine="pyarrow", index=False)

    n_countries = df["country"].nunique()
    print(f"\nSaved {len(df)} rows for {n_countries}/{len(COUNTRY_LIST)} countries -> {OUT_PATH}")
    missing = set(COUNTRY_LIST) - set(df["country"].astype(str).unique())
    if missing:
        print(f"WARNING: no GDP data for: {sorted(missing)} — will fall back to flat "
              f"(unscaled) gross_output for these countries in 2015-2021.")


if __name__ == "__main__":
    main()
