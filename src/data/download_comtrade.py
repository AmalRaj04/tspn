"""Download UN Comtrade trade data via comtradeapicall.

Pulls HS 2-digit aggregate imports for all countries in config.GRAPH["COUNTRY_LIST"],
plus HS6-level USA→CHN imports for Section 301 tariff lists.

Usage
-----
    export COMTRADE_API_KEY=<your-key>
    python scripts/download_comtrade.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

try:
    import comtradeapicall
except ModuleNotFoundError:
    raise SystemExit(
        "comtradeapicall is not installed for this Python interpreter.\n"
        "Activate the tspn conda env and use 'python' (not 'python3'):\n"
        "  conda activate tspn\n"
        "  pip install comtradeapicall==1.3.1\n"
        "  python scripts/download_comtrade.py"
    ) from None

import pandas as pd
 
# ---------------------------------------------------------------------------
# Project root & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

COUNTRY_LIST = config.GRAPH["COUNTRY_LIST"]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, config.PATHS["RAW_COMTRADE"])
FAILED_LOG = os.path.join(OUTPUT_DIR, "failed_downloads.log")

YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
SECTION_301_YEARS = [2018, 2019, 2020]
RATE_LIMIT_SEC = 7.5
MAX_RETRIES = 3
HS6_BATCH_SIZE = 100

HS_CODE_COLUMNS = (
    "hts_code",
    "hts",
    "HTS",
    "hs6",
    "hs_code",
    "commodity_code",
    "cmdCode",
    "cmd_code",
)


def _require_api_key() -> str:
    api_key = os.environ.get("COMTRADE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "COMTRADE_API_KEY environment variable is required. "
            "Export your UN Comtrade subscription key before running."
        )
    return api_key


def _iso3_to_comtrade_code(iso3: str) -> str | None:
    code = comtradeapicall.convertCountryIso3ToCode(iso3)
    return code if code else None


def _normalize_hs6(code: object) -> str | None:
    raw = str(code).strip().replace(".", "").replace(" ", "")
    if not raw or not raw.isdigit():
        return None
    return raw.zfill(6)[:6]


def _load_section_301_hs6_codes() -> list[str]:
    paths: list[str] = []
    for event in config.EVENTS:
        if event["name"] in ("us_301_list1_2018", "us_301_list2_2018"):
            paths.append(os.path.join(PROJECT_ROOT, event["hts_file"]))

    if not paths:
        raise SystemExit("No Section 301 HTS file paths found in config.EVENTS.")

    codes: set[str] = set()
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(f"Section 301 HTS file not found: {path}")
        df = pd.read_csv(path, dtype=str)
        col = next((c for c in HS_CODE_COLUMNS if c in df.columns), None)
        if col is None:
            col = df.columns[0]
        for value in df[col].dropna():
            hs6 = _normalize_hs6(value)
            if hs6 is not None:
                codes.add(hs6)

    if not codes:
        raise SystemExit("No HS6 codes found in Section 301 list CSV files.")

    return sorted(codes)


def _normalize_trade_df(raw: pd.DataFrame | None) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=["reporter", "partner", "commodity_code", "trade_value_usd"]
        )

    reporter_col = "reporterISO" if "reporterISO" in raw.columns else "reporterCode"
    partner_col = "partnerISO" if "partnerISO" in raw.columns else "partnerCode"

    out = pd.DataFrame(
        {
            "reporter": raw[reporter_col].astype(str),
            "partner": raw[partner_col].astype(str),
            "commodity_code": raw["cmdCode"].astype(str),
            "trade_value_usd": pd.to_numeric(raw["primaryValue"], errors="coerce"),
        }
    )
    return out.dropna(subset=["trade_value_usd"]).reset_index(drop=True)


def _log_failure(label: str, detail: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(FAILED_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{timestamp} FAILED {label}: {detail}\n")


def _fetch_with_retries(
    api_key: str,
    *,
    period: str,
    reporter_code: str,
    cmd_code: str,
    partner_code: str | None,
    label: str,
) -> pd.DataFrame | None:
    last_error = "unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = comtradeapicall.getFinalData(
                subscription_key=api_key,
                typeCode="C",
                freqCode="A",
                clCode="HS",
                period=period,
                reporterCode=reporter_code,
                cmdCode=cmd_code,
                flowCode="M",
                partnerCode=partner_code,
                partner2Code=None,
                customsCode=None,
                motCode=None,
                maxRecords=250000,
                format_output="JSON",
                aggregateBy=None,
                breakdownMode="classic",
                countOnly=None,
                includeDesc=False,
            )
            if raw is None:
                last_error = "API returned no data (check printed API error message)"
            elif isinstance(raw, pd.DataFrame):
                return raw
            else:
                last_error = f"Unexpected response type: {type(raw).__name__}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        if attempt < MAX_RETRIES:
            time.sleep(RATE_LIMIT_SEC)

    _log_failure(label, f"{last_error} after {MAX_RETRIES} retries")
    return None


def _save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def download_hs2_imports(api_key: str) -> None:
    for reporter in COUNTRY_LIST:
        reporter_code = _iso3_to_comtrade_code(reporter)
        if not reporter_code:
            _log_failure(
                f"{reporter}",
                "Could not map ISO3 reporter code via convertCountryIso3ToCode",
            )
            continue

        for year in YEARS:
            out_path = os.path.join(
                OUTPUT_DIR, f"comtrade_{reporter}_{year}.parquet"
            )
            label = f"{reporter} {year}"

            if os.path.exists(out_path):
                print(f"Skipping {label} — file already exists")
                continue

            raw = _fetch_with_retries(
                api_key,
                period=str(year),
                reporter_code=reporter_code,
                cmd_code="AG2",
                partner_code=None,
                label=label,
            )
            time.sleep(RATE_LIMIT_SEC)

            if raw is None:
                continue

            df = _normalize_trade_df(raw)
            _save_parquet(df, out_path)
            print(f"Downloaded {reporter} {year} — {len(df)} rows")


def download_section_301_hs6(api_key: str, hs6_codes: list[str]) -> None:
    reporter_code = _iso3_to_comtrade_code("USA")
    partner_code = _iso3_to_comtrade_code("CHN")
    if not reporter_code or not partner_code:
        raise SystemExit("Could not map USA or CHN to Comtrade country codes.")

    for year in SECTION_301_YEARS:
        out_path = os.path.join(
            OUTPUT_DIR, f"comtrade_usa_chn_301hs6_{year}.parquet"
        )
        label = f"usa_chn_301hs6 {year}"

        if os.path.exists(out_path):
            print(f"Skipping {label} — file already exists")
            continue

        frames: list[pd.DataFrame] = []
        failed = False

        for start in range(0, len(hs6_codes), HS6_BATCH_SIZE):
            batch = hs6_codes[start : start + HS6_BATCH_SIZE]
            batch_label = f"{label} batch {start // HS6_BATCH_SIZE + 1}"
            raw = _fetch_with_retries(
                api_key,
                period=str(year),
                reporter_code=reporter_code,
                cmd_code=",".join(batch),
                partner_code=partner_code,
                label=batch_label,
            )
            time.sleep(RATE_LIMIT_SEC)

            if raw is None:
                failed = True
                break
            frames.append(_normalize_trade_df(raw))

        if failed:
            continue

        df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else _normalize_trade_df(None)
        )
        _save_parquet(df, out_path)
        print(f"Downloaded usa_chn_301hs6 {year} — {len(df)} rows")


def main() -> None:
    api_key = _require_api_key()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Downloading HS2 aggregate imports for {len(COUNTRY_LIST)} countries, "
          f"years {YEARS[0]}–{YEARS[-1]}")
    download_hs2_imports(api_key)

    hs6_codes = _load_section_301_hs6_codes()
    print(
        f"Downloading USA→CHN HS6 imports for {len(hs6_codes)} Section 301 codes, "
        f"years {SECTION_301_YEARS[0]}–{SECTION_301_YEARS[-1]}"
    )
    download_section_301_hs6(api_key, hs6_codes)


if __name__ == "__main__":
    main()
