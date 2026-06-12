"""Download BLS Industry PPI series via the BLS Public Data API v2.

Pulls monthly Producer Price Index data for selected NAICS industry codes
(2014–2023) and saves one CSV per series.

Usage
-----
    export BLS_API_KEY=<your-key>
    python scripts/download_bls_ppi.py
"""

from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()

import os

api_key = os.getenv("BLS_API_KEY")  
import sys
import time
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Project root & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, config.PATHS["RAW_BLS_PPI"])

START_YEAR = "2014"
END_YEAR = "2023"
BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5.0

# NAICS industry codes: manufacturing 311–339, mining 211–213, selected services.
MANUFACTURING_NAICS = [str(code) for code in range(311, 340)]
MINING_NAICS = ["211", "212", "213"]
SERVICES_NAICS = ["481", "483", "484", "4931"]
NAICS_CODES = MANUFACTURING_NAICS + MINING_NAICS + SERVICES_NAICS

MONTHLY_PERIODS = {f"M{month:02d}" for month in range(1, 13)}


def _series_id(naics_code: str) -> str:
    """
    Build BLS aggregate 3-digit NAICS PPI series ID.

    Examples:
        311 -> PCU311---311---
        327 -> PCU327---327---
        331 -> PCU331---331---
    """
    return f"PCU{naics_code}---{naics_code}---"


def _output_path(naics_code: str) -> str:
    return os.path.join(OUTPUT_DIR, f"bls_ppi_{naics_code}.csv")


def _fetch_batch(
    session: requests.Session,
    api_key: str,
    series_ids: list[str],
) -> dict[str, Any]:
    """POST a batch of up to 50 series IDs to the BLS API v2."""
    payload = {
        "seriesid": series_ids,
        "startyear": START_YEAR,
        "endyear": END_YEAR,
        "registrationkey": api_key,
    }
    response = session.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def _fetch_batch_with_retries(
    session: requests.Session,
    api_key: str,
    series_ids: list[str],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            body = _fetch_batch(session, api_key, series_ids)
            status = body.get("status", "")
            if status != "REQUEST_SUCCEEDED":
                messages = body.get("message", [])
                raise RuntimeError(
                    f"BLS API returned status {status!r}: {messages}"
                )
            return body
        except (requests.HTTPError, requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                print(
                    f"  HTTP/API error (attempt {attempt}/{MAX_RETRIES}): {exc} "
                    f"— retrying in {RETRY_BACKOFF_SEC:.0f}s"
                )
                time.sleep(RETRY_BACKOFF_SEC)

    raise RuntimeError(
        f"BLS API batch failed after {MAX_RETRIES} retries: {last_error}"
    ) from last_error


def _series_id_to_naics(series_id: str) -> str | None:
    """
    PCU311---311--- -> 311
    """
    if not series_id.startswith("PCU"):
        return None

    body = series_id[3:]

    if "---" not in body:
        return None

    return body[:3]


def _parse_series_data(series: dict[str, Any]) -> pd.DataFrame:
    """Convert one BLS series payload to a monthly DataFrame."""
    rows: list[dict[str, Any]] = []
    for point in series.get("data", []):
        period = point.get("period", "")
        if period not in MONTHLY_PERIODS:
            continue
        value_raw = point.get("value", "")
        if value_raw in ("", "-", None):
            continue
        rows.append(
            {
                "year": int(point["year"]),
                "period": period,
                "value": float(value_raw),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["year", "period", "value"])

    df = pd.DataFrame(rows)
    return df.sort_values(["year", "period"], kind="mergesort").reset_index(drop=True)


def _save_series(naics_code: str, df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(_output_path(naics_code), index=False)


def download_bls_ppi() -> None:
    api_key = os.environ["BLS_API_KEY"]

    pending: list[str] = []
    for naics_code in NAICS_CODES:
        out_path = _output_path(naics_code)
        if os.path.exists(out_path):
            print(f"Skipping {naics_code} — file already exists")
            continue
        pending.append(naics_code)

    if not pending:
        print("All series already downloaded.")
        return

    print(
        f"Downloading {len(pending)} BLS PPI series "
        f"({START_YEAR}–{END_YEAR}) in batches of {BATCH_SIZE}"
    )

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch_naics = pending[batch_start : batch_start + BATCH_SIZE]
        batch_ids = [_series_id(code) for code in batch_naics]
        id_to_naics = dict(zip(batch_ids, batch_naics))

        print(
            f"Fetching batch {batch_start // BATCH_SIZE + 1} "
            f"({len(batch_ids)} series)..."
        )
        body = _fetch_batch_with_retries(session, api_key, batch_ids)

        results = body.get("Results", {})
        series_list = results.get("series", []) if isinstance(results, dict) else []

        returned_ids = set()
        for series in series_list:
            series_id = series.get("seriesID", "")
            naics_code = id_to_naics.get(series_id) or _series_id_to_naics(series_id)
            if naics_code is None:
                print(f"  Warning: could not map series ID {series_id!r}")
                continue

            returned_ids.add(series_id)
            df = _parse_series_data(series)
            _save_series(naics_code, df)
            print(f"Downloaded {naics_code} — {len(df)} rows")

        missing = set(batch_ids) - returned_ids
        for series_id in sorted(missing):
            naics_code = id_to_naics[series_id]
            _save_series(naics_code, pd.DataFrame(columns=["year", "period", "value"]))
            print(f"Downloaded {naics_code} — 0 rows (no data returned)")


def main() -> None:
    try:
        os.environ["BLS_API_KEY"]
    except KeyError:
        raise SystemExit(
            "BLS_API_KEY environment variable is required. "
            "Export your BLS registration key before running."
        ) from None

    download_bls_ppi()


if __name__ == "__main__":
    main()
