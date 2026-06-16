"""Compute the normalized node feature matrix for every available WIOD year.

Features (in locked order):
    f[0]  log_gross_output
    f[1]  import_penetration
    f[2]  export_intensity
    f[3]  backward_linkage
    f[4]  tariff_exposure
    f[5]  ppi_lag_1  (Q4 of year-1)
    f[6]  ppi_lag_2  (Q3 of year-1)
    f[7]  ppi_lag_3  (Q2 of year-1)
    f[8]  ppi_lag_4  (Q1 of year-1)

Node universe per year: socioeconomic_YEAR.parquet (2408 nodes, 43 countries × 56 sectors).
ROW rows are never fabricated.

node_id is computed via config.node_id(country_idx, sector_idx) and used
to index backward_linkage_YEAR.npy (length 2464 = 44 × 56).
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Index look-ups built once from config (authoritative order)
# ---------------------------------------------------------------------------
COUNTRY_IDX: dict[str, int] = {c: i for i, c in enumerate(config.GRAPH["COUNTRY_LIST"])}
SECTOR_IDX: dict[str, int] = {s: i for i, s in enumerate(config.GRAPH["SECTOR_LIST"])}


def normalize_sector(raw: str) -> str:
    """Normalize sector codes that use hyphens to the underscore form used by config.

    Examples
    --------
    'C10-C12' -> 'C10_C12'
    'E37-E39' -> 'E37_E39'
    'C13-C15' -> 'C13_C15'
    All other strings are returned unchanged.
    """
    return str(raw).replace("-", "_")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _sea_path(year: int) -> str:
    return os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_EDGES"], f"socioeconomic_{year}.parquet")


def _edges_path(year: int) -> str:
    return os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_EDGES"], f"edges_{year}.parquet")


def _bl_path(year: int) -> str:
    return os.path.join(PROJECT_ROOT, "data", "processed", "leontief", f"backward_linkage_{year}.npy")


TARIFF_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_TARIFF_RATES"], "sector_tariffs.parquet")
PPI_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_LABELS"], "ppi_quarterly_all.parquet")
NF_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_NODE_FEATURES"])
NORM_STATS_PATH = os.path.join(PROJECT_ROOT, config.PATHS["NORM_STATS"])


# ---------------------------------------------------------------------------
# Year discovery
# ---------------------------------------------------------------------------
def discover_years() -> list[int]:
    """Return available years by scanning socioeconomic_*.parquet files."""
    pattern = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_EDGES"], "socioeconomic_*.parquet")
    files = glob.glob(pattern)
    years = []
    for f in files:
        m = re.search(r"socioeconomic_(\d{4})\.parquet", os.path.basename(f))
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


# ---------------------------------------------------------------------------
# Tariff lookup: build a dict {(country, normalized_sector) -> tariff_rate}
# Since tariff data covers 2017-2021 and WIOD years are 2000-2014 there is
# no overlap; all tariff_exposure values will be 0.0. The lookup is built
# defensively so it works if future data is added.
# ---------------------------------------------------------------------------
def build_tariff_lookup() -> dict[tuple[str, str], float]:
    """Load sector_tariffs.parquet and return a dict keyed by (country, sector)."""
    if not os.path.exists(TARIFF_PATH):
        print(f"  [tariff] File not found: {TARIFF_PATH} — tariff_exposure will be 0.0")
        return {}
    tar = pd.read_parquet(TARIFF_PATH)
    tar["sector"] = tar["sector"].astype(str).apply(normalize_sector)
    # Average over years/data_sources if duplicates exist for same (country, sector)
    tar_agg = (
        tar.groupby(["country", "sector"], observed=True)["tariff_rate"]
        .mean()
        .reset_index()
    )
    lookup: dict[tuple[str, str], float] = {
        (str(row.country), str(row.sector)): float(row.tariff_rate)
        for row in tar_agg.itertuples(index=False)
    }
    return lookup


# ---------------------------------------------------------------------------
# PPI lookup: build {(country, sector, year, quarter) -> ppi_change}
# and a WLD fallback {(sector, year, quarter) -> ppi_change}
# ---------------------------------------------------------------------------
def build_ppi_lookup() -> tuple[
    dict[tuple[str, str, int, int], float],
    dict[tuple[str, int, int], float],
]:
    """Return (country_ppi_lookup, wld_ppi_lookup) dicts."""
    if not os.path.exists(PPI_PATH):
        print(f"  [ppi] File not found: {PPI_PATH} — PPI lags will be 0.0")
        return {}, {}

    ppi = pd.read_parquet(PPI_PATH)
    ppi["isic_sector"] = ppi["isic_sector"].astype(str).apply(normalize_sector)
    ppi["country"] = ppi["country"].astype(str)
    ppi["year"] = ppi["year"].astype(int)
    ppi["quarter"] = ppi["quarter"].astype(int)

    country_lookup: dict[tuple[str, str, int, int], float] = {}
    wld_lookup: dict[tuple[str, int, int], float] = {}

    for row in ppi.itertuples(index=False):
        key_c = (row.country, row.isic_sector, int(row.year), int(row.quarter))
        val = float(row.ppi_change)
        country_lookup[key_c] = val
        if row.country == "WLD":
            key_w = (row.isic_sector, int(row.year), int(row.quarter))
            wld_lookup[key_w] = val

    return country_lookup, wld_lookup


def get_ppi_lag(
    country: str,
    sector: str,
    lag_year: int,
    lag_quarter: int,
    country_ppi: dict,
    wld_ppi: dict,
) -> tuple[float, bool]:
    """Return (ppi_change, found) for a given lag point.

    Priority:
      1. Exact country + sector
      2. WLD + sector
      3. 0.0 (not found)
    """
    key_c = (country, sector, lag_year, lag_quarter)
    if key_c in country_ppi:
        return country_ppi[key_c], True
    key_w = (sector, lag_year, lag_quarter)
    if key_w in wld_ppi:
        return wld_ppi[key_w], True
    return 0.0, False


# ---------------------------------------------------------------------------
# Per-year feature computation
# ---------------------------------------------------------------------------
def compute_features_for_year(
    year: int,
    tariff_lookup: dict,
    country_ppi: dict,
    wld_ppi: dict,
) -> pd.DataFrame:
    """Build the raw (un-normalized) feature DataFrame for one year."""

    # -----------------------------------------------------------------------
    # 1. Load socioeconomic (authoritative node list)
    # -----------------------------------------------------------------------
    sea = pd.read_parquet(_sea_path(year))
    sea["sector_norm"] = sea["sector"].astype(str).apply(normalize_sector)
    sea["country"] = sea["country"].astype(str)
    sea["gross_output"] = sea["gross_output"].astype(float).clip(lower=0.0)
    sea["value_added"] = sea["value_added"].astype(float)

    # -----------------------------------------------------------------------
    # 2. Assign node_id using config indices
    # -----------------------------------------------------------------------
    def _node_id(country: str, sector_norm: str) -> int | None:
        ci = COUNTRY_IDX.get(country)
        si = SECTOR_IDX.get(sector_norm)
        if ci is None or si is None:
            return None
        return config.node_id(ci, si)

    sea["node_id"] = [
        _node_id(row.country, row.sector_norm)
        for row in sea.itertuples(index=False)
    ]

    missing_ids = sea["node_id"].isna().sum()
    if missing_ids > 0:
        print(f"  [WARN] {missing_ids} rows could not be mapped to a node_id and will be dropped.")
        sea = sea.dropna(subset=["node_id"])
    sea["node_id"] = sea["node_id"].astype(int)

    # -----------------------------------------------------------------------
    # 3. Trade aggregations: exports and imports from edges
    # -----------------------------------------------------------------------
    edges = pd.read_parquet(_edges_path(year))
    edges["flow_usd"] = edges["flow_usd"].astype(float).clip(lower=0.0)
    edges["src_country"] = edges["src_country"].astype(str)
    edges["src_sector"] = edges["src_sector"].astype(str).apply(normalize_sector)
    edges["tgt_country"] = edges["tgt_country"].astype(str)
    edges["tgt_sector"] = edges["tgt_sector"].astype(str).apply(normalize_sector)

    exports = (
        edges.groupby(["src_country", "src_sector"], observed=True)["flow_usd"]
        .sum()
        .rename("total_exports")
        .reset_index()
        .rename(columns={"src_country": "country", "src_sector": "sector_norm"})
    )

    imports = (
        edges.groupby(["tgt_country", "tgt_sector"], observed=True)["flow_usd"]
        .sum()
        .rename("total_imports")
        .reset_index()
        .rename(columns={"tgt_country": "country", "tgt_sector": "sector_norm"})
    )

    sea = sea.merge(exports, on=["country", "sector_norm"], how="left")
    sea = sea.merge(imports, on=["country", "sector_norm"], how="left")
    sea["total_exports"] = sea["total_exports"].fillna(0.0).clip(lower=0.0)
    sea["total_imports"] = sea["total_imports"].fillna(0.0).clip(lower=0.0)

    # -----------------------------------------------------------------------
    # 4. Backward linkage (from pre-computed .npy, indexed by node_id)
    # -----------------------------------------------------------------------
    bl_arr = np.load(_bl_path(year)).astype(np.float32)  # shape (2464,)
    sea["backward_linkage"] = [float(bl_arr[nid]) for nid in sea["node_id"]]

    # -----------------------------------------------------------------------
    # 5. Build features f0-f4
    # -----------------------------------------------------------------------
    go = sea["gross_output"].values.astype(np.float64)
    te = sea["total_exports"].values.astype(np.float64)
    ti = sea["total_imports"].values.astype(np.float64)

    f0 = np.log1p(go)                                           # log_gross_output
    f1 = ti / (go + ti - te + 1e-9)                            # import_penetration
    f2 = te / (go + 1e-9)                                      # export_intensity
    f3 = sea["backward_linkage"].values.astype(np.float64)     # backward_linkage

    # f4: tariff_exposure (country-sector, year-agnostic since no WIOD-year tariff data)
    f4 = np.array([
        tariff_lookup.get((row.country, row.sector_norm), 0.0)
        for row in sea.itertuples(index=False)
    ], dtype=np.float64)

    # -----------------------------------------------------------------------
    # 6. PPI lags: Q4, Q3, Q2, Q1 of (year - 1)
    # -----------------------------------------------------------------------
    lag_year = year - 1
    lag_quarters = [4, 3, 2, 1]   # f[5]=Q4(y-1), f[6]=Q3(y-1), f[7]=Q2(y-1), f[8]=Q1(y-1)

    ppi_lags = [[], [], [], []]
    ppi_found_flags = []

    for row in sea.itertuples(index=False):
        row_found = False
        lags_for_row = []
        for lq in lag_quarters:
            val, found = get_ppi_lag(
                row.country, row.sector_norm, lag_year, lq, country_ppi, wld_ppi
            )
            lags_for_row.append(val)
            if found:
                row_found = True
        ppi_found_flags.append(row_found)
        for i, v in enumerate(lags_for_row):
            ppi_lags[i].append(v)

    f5 = np.array(ppi_lags[0], dtype=np.float64)
    f6 = np.array(ppi_lags[1], dtype=np.float64)
    f7 = np.array(ppi_lags[2], dtype=np.float64)
    f8 = np.array(ppi_lags[3], dtype=np.float64)
    has_ppi_lags = np.array(ppi_found_flags, dtype=bool)

    # -----------------------------------------------------------------------
    # 7. Assemble output DataFrame
    # -----------------------------------------------------------------------
    out = pd.DataFrame({
        "year":        pd.array([year] * len(sea), dtype="int16"),
        "country":     pd.Categorical(sea["country"].values),
        "sector":      pd.Categorical(sea["sector"].astype(str).values),
        "node_id":     pd.array(sea["node_id"].values, dtype="int16"),
        "f0":          f0.astype("float32"),
        "f1":          f1.astype("float32"),
        "f2":          f2.astype("float32"),
        "f3":          f3.astype("float32"),
        "f4":          f4.astype("float32"),
        "f5":          f5.astype("float32"),
        "f6":          f6.astype("float32"),
        "f7":          f7.astype("float32"),
        "f8":          f8.astype("float32"),
        "has_ppi_lags": has_ppi_lags,
    })

    return out


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
FEATURE_COLS = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]


def compute_normalization_stats(all_frames: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std across all training years."""
    combined = pd.concat(all_frames, ignore_index=True)
    means = combined[FEATURE_COLS].mean().values.astype(np.float64)
    stds = combined[FEATURE_COLS].std(ddof=0).values.astype(np.float64)
    return means, stds


def apply_normalization(
    df: pd.DataFrame, means: np.ndarray, stds: np.ndarray
) -> pd.DataFrame:
    """Apply z-score normalization in-place: (x - mean) / (std + 1e-8)."""
    df = df.copy()
    for i, col in enumerate(FEATURE_COLS):
        df[col] = ((df[col].values.astype(np.float64) - means[i]) / (stds[i] + 1e-8)).astype("float32")
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame, year: int, expected_rows: int) -> None:
    """Run assertions and print the validation line."""
    assert len(df) == expected_rows, (
        f"Year {year}: expected {expected_rows} rows, got {len(df)}"
    )

    nan_cols = [c for c in FEATURE_COLS if df[c].isna().any()]
    assert not nan_cols, f"Year {year}: NaN found in {nan_cols}"

    assert df["node_id"].between(0, 2463).all(), f"Year {year}: node_id out of range"

    dup = df.duplicated(subset=["country", "sector"]).sum()
    assert dup == 0, f"Year {year}: {dup} duplicate (country, sector) pairs"

    ppi_coverage = float(df["has_ppi_lags"].mean()) * 100

    print(
        f"  Node features {year}: rows={len(df)}  nan_check=passed  "
        f"ppi_coverage={ppi_coverage:.1f}%"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(NF_DIR, exist_ok=True)

    years = discover_years()
    if not years:
        print("No socioeconomic parquet files found. Exiting.")
        sys.exit(1)

    print(f"Discovered {len(years)} WIOD years: {years[0]}–{years[-1]}")

    # -- Load shared data once -----------------------------------------------
    print("Loading tariff data...")
    tariff_lookup = build_tariff_lookup()

    print("Loading PPI data...")
    country_ppi, wld_ppi = build_ppi_lookup()

    # -- Compute raw features for all years ----------------------------------
    raw_frames: dict[int, pd.DataFrame] = {}
    for year in years:
        print(f"Computing raw features for {year}...")
        df = compute_features_for_year(year, tariff_lookup, country_ppi, wld_ppi)
        raw_frames[year] = df

    # -- Normalization stats computed on ALL available years (= training years) --
    print("Computing normalization statistics...")
    means, stds = compute_normalization_stats(list(raw_frames.values()))

    # Save normalization stats
    norm_stats = {
        "mean": means.tolist(),
        "std":  stds.tolist(),
    }
    os.makedirs(os.path.dirname(NORM_STATS_PATH), exist_ok=True)
    with open(NORM_STATS_PATH, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization stats → {NORM_STATS_PATH}")

    # -- Apply normalization and save ----------------------------------------
    expected_rows = None
    for year in years:
        raw_df = raw_frames[year]
        if expected_rows is None:
            expected_rows = len(raw_df)

        norm_df = apply_normalization(raw_df, means, stds)
        validate(norm_df, year, expected_rows)

        out_path = os.path.join(NF_DIR, f"node_features_{year}.parquet")
        norm_df.to_parquet(out_path, index=False)

    print(f"\nDone. {len(years)} node feature files saved to {NF_DIR}/")


if __name__ == "__main__":
    main()
