"""Compute baseline sector-level MFN tariff rates.

Pipeline
--------
For every (country, year) with a WITS tariff file:
    1. Load hs6 → mfn_rate (percentage → decimal).
    2. Map each HS6 code to WIOD-56 sectors using concordance weights.
    3. Compute weighted-average tariff per (country, year, sector).

For country-years without data:
    - Interpolate linearly between nearest available observed years.
    - Fill remaining gaps with 0.0 and label "missing".

Output
------
data/processed/tariff_rates/sector_tariffs.parquet
    year        : int16
    country     : category
    sector      : category
    tariff_rate : float32       (decimal; 5 % → 0.05)
    data_source : category      ("observed" | "interpolated" | "missing")
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: locate project root regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants from config
# ---------------------------------------------------------------------------
COUNTRY_LIST: list[str] = config.GRAPH["COUNTRY_LIST"]   # 44 countries
SECTOR_LIST: list[str] = config.GRAPH["SECTOR_LIST"]     # 56 WIOD sectors
COMTRADE_YEARS: list[int] = config.GRAPH["COMTRADE_YEARS"]  # [2017..2021]

RAW_WITS = ROOT / config.PATHS["RAW_WITS"]
PROC_TARIFF = ROOT / config.PATHS["PROCESSED_TARIFF_RATES"]
PROC_CONC = ROOT / config.PATHS["PROCESSED_CONCORDANCE"]

PROC_TARIFF.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_concordance() -> dict[str, dict[str, float]]:
    """Load hs6_to_wiod56_weights.json → {hs6: {sector: weight}}."""
    path = PROC_CONC / "hs6_to_wiod56_weights.json"
    with open(path) as fh:
        return json.load(fh)


def _parse_wits_files() -> dict[str, dict[int, Path]]:
    """Scan RAW_WITS for tariff_{ISO3}_{YEAR}.csv files.

    Returns:
        {country_iso3: {year: Path}}
    """
    available: dict[str, dict[int, Path]] = {}
    for fp in sorted(RAW_WITS.glob("tariff_*_*.csv")):
        stem = fp.stem          # e.g. "tariff_USA_2018"
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        _, iso3, year_str = parts
        try:
            year = int(year_str)
        except ValueError:
            continue
        available.setdefault(iso3, {})[year] = fp
    return available


def _load_wits_schedule(path: Path) -> pd.Series:
    """Read one WITS CSV; return Series {hs6_str: mfn_rate_decimal}.

    - hs6 codes are stored as plain strings (may need zero-padding).
    - mfn_rate is in percentage form in the file; we convert to decimal.
    """
    df = pd.read_csv(path, dtype={"hs6": str, "mfn_rate": float})
    df["hs6"] = df["hs6"].str.strip().str.zfill(6)
    df = df.dropna(subset=["mfn_rate"])
    df["mfn_rate"] = df["mfn_rate"] / 100.0          # % → decimal
    df = df.drop_duplicates(subset="hs6", keep="first")
    return df.set_index("hs6")["mfn_rate"]


def _compute_sector_tariffs(
    schedule: pd.Series,
    concordance: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute weighted-average tariff per WIOD sector from one HS6 schedule.

    For each sector s:
        tariff_rate[s] = Σ(weight_hs6_s × rate_hs6) / Σ(weight_hs6_s)

    Args:
        schedule:    {hs6: mfn_rate_decimal}
        concordance: {hs6: {sector: weight}}

    Returns:
        {sector: tariff_rate_decimal}
    """
    # Accumulators
    weighted_sum: dict[str, float] = {s: 0.0 for s in SECTOR_LIST}
    weight_total: dict[str, float] = {s: 0.0 for s in SECTOR_LIST}

    for hs6, rate in schedule.items():
        mapping = concordance.get(hs6)
        if mapping is None:
            continue
        for sector, w in mapping.items():
            if sector not in weighted_sum:
                continue                   # guard against unknown sectors
            weighted_sum[sector] += w * rate
            weight_total[sector] += w

    result: dict[str, float] = {}
    for sector in SECTOR_LIST:
        wt = weight_total[sector]
        result[sector] = weighted_sum[sector] / wt if wt > 0.0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Computing sector-level MFN tariff rates …\n")

    concordance = _load_concordance()
    print(f"Concordance loaded: {len(concordance):,} HS6 codes.")

    wits_index = _parse_wits_files()
    countries_with_data = sorted(wits_index.keys())
    print(f"WITS files found for {len(countries_with_data)} countries: {countries_with_data}\n")

    # ------------------------------------------------------------------
    # Phase 1: compute observed tariffs for every available (country, year)
    # ------------------------------------------------------------------
    # observed[country][year][sector] = tariff_rate
    observed: dict[str, dict[int, dict[str, float]]] = {}

    for country, year_files in sorted(wits_index.items()):
        observed[country] = {}
        for year, fp in sorted(year_files.items()):
            schedule = _load_wits_schedule(fp)
            sector_rates = _compute_sector_tariffs(schedule, concordance)
            observed[country][year] = sector_rates
            print(f"  {country} {year}: {len(schedule):,} HS6 codes processed.")

    # ------------------------------------------------------------------
    # Phase 2: build full (country, year, sector) table for COMTRADE_YEARS
    # ------------------------------------------------------------------
    all_years: list[int] = COMTRADE_YEARS
    records: list[dict] = []

    n_observed = 0
    n_interpolated = 0
    n_missing = 0

    for country in COUNTRY_LIST:
        country_obs = observed.get(country, {})          # {} if no data at all
        observed_years = sorted(country_obs.keys())

        for year in all_years:
            if year in country_obs:
                # ── Case 1: Directly observed ──────────────────────────────
                sector_rates = country_obs[year]
                source = "observed"
                for sector in SECTOR_LIST:
                    records.append({
                        "year": year,
                        "country": country,
                        "sector": sector,
                        "tariff_rate": float(sector_rates.get(sector, 0.0)),
                        "data_source": source,
                    })
                n_observed += 1

            elif observed_years:
                # ── Case 2: Interpolate between nearest years ──────────────
                lo_years = [y for y in observed_years if y < year]
                hi_years = [y for y in observed_years if y > year]

                if lo_years and hi_years:
                    y_lo = max(lo_years)
                    y_hi = min(hi_years)
                    alpha = (year - y_lo) / (y_hi - y_lo)    # in (0, 1)
                    rates_lo = country_obs[y_lo]
                    rates_hi = country_obs[y_hi]
                    for sector in SECTOR_LIST:
                        r_lo = rates_lo.get(sector, 0.0)
                        r_hi = rates_hi.get(sector, 0.0)
                        interp = r_lo + alpha * (r_hi - r_lo)
                        records.append({
                            "year": year,
                            "country": country,
                            "sector": sector,
                            "tariff_rate": float(interp),
                            "data_source": "interpolated",
                        })

                elif hi_years:
                    # Only higher years available → use nearest higher year
                    y_near = min(hi_years)
                    for sector in SECTOR_LIST:
                        records.append({
                            "year": year,
                            "country": country,
                            "sector": sector,
                            "tariff_rate": float(country_obs[y_near].get(sector, 0.0)),
                            "data_source": "interpolated",
                        })

                else:
                    # Only lower years available → use nearest lower year
                    y_near = max(lo_years)
                    for sector in SECTOR_LIST:
                        records.append({
                            "year": year,
                            "country": country,
                            "sector": sector,
                            "tariff_rate": float(country_obs[y_near].get(sector, 0.0)),
                            "data_source": "interpolated",
                        })

                n_interpolated += 1

            else:
                # ── Case 3: No data at all → zero-fill ────────────────────
                for sector in SECTOR_LIST:
                    records.append({
                        "year": year,
                        "country": country,
                        "sector": sector,
                        "tariff_rate": 0.0,
                        "data_source": "missing",
                    })
                n_missing += 1

    # ------------------------------------------------------------------
    # Phase 3: Build DataFrame and cast types
    # ------------------------------------------------------------------
    print(f"\nBuilding DataFrame from {len(records):,} records …")
    df = pd.DataFrame.from_records(records)

    df["year"] = df["year"].astype("int16")
    df["country"] = df["country"].astype("category")
    df["sector"] = df["sector"].astype("category")
    df["tariff_rate"] = df["tariff_rate"].astype("float32")
    df["data_source"] = df["data_source"].astype("category")

    # ------------------------------------------------------------------
    # Phase 4: Validation
    # ------------------------------------------------------------------
    print("\nRunning validation checks …")

    nan_count = df["tariff_rate"].isna().sum()
    assert nan_count == 0, f"Found {nan_count} NaN tariff_rate values!"
    print("  ✅ No NaN tariff_rate values.")

    bad_sectors = ~df["sector"].isin(SECTOR_LIST)
    assert not bad_sectors.any(), (
        f"Unknown sectors: {df.loc[bad_sectors, 'sector'].unique().tolist()}"
    )
    print("  ✅ All sectors are valid WIOD-56 members.")

    neg_rates = df["tariff_rate"] < 0
    assert not neg_rates.any(), (
        f"Found {neg_rates.sum()} negative tariff_rate values!"
    )
    print("  ✅ All tariff_rate values are >= 0.")

    # ------------------------------------------------------------------
    # Phase 5: Summary statistics
    # ------------------------------------------------------------------
    total_country_years = len(COUNTRY_LIST) * len(all_years)
    pct_interpolated = 100 * n_interpolated / total_country_years
    pct_missing = 100 * n_missing / total_country_years

    print("\n" + "=" * 60)
    print("TARIFF RATES SUMMARY")
    print("=" * 60)
    print(f"  Countries processed    : {len(COUNTRY_LIST)}")
    print(f"  Years processed        : {all_years}")
    print(f"  Rows produced          : {len(df):,}")
    print(f"  Country-years observed : {n_observed}  ({100*n_observed/total_country_years:.1f}%)")
    print(f"  Country-years interp.  : {n_interpolated}  ({pct_interpolated:.1f}%)")
    print(f"  Country-years missing  : {n_missing}  ({pct_missing:.1f}%)")

    # Top 10 highest sector tariffs
    top10 = (
        df[df["data_source"] == "observed"]
        .nlargest(10, "tariff_rate")[["country", "year", "sector", "tariff_rate"]]
        .reset_index(drop=True)
    )
    top10["tariff_rate_pct"] = (top10["tariff_rate"] * 100).round(2)
    print("\n  Top 10 highest observed sector tariffs:")
    print(top10[["country", "year", "sector", "tariff_rate_pct"]].to_string(index=False))
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 6: Save
    # ------------------------------------------------------------------
    out_path = PROC_TARIFF / "sector_tariffs.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path.relative_to(ROOT)}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
