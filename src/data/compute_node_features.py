"""Compute the normalized node feature matrix for every year 2000-2021.

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

Node universe per year: 2,408 nodes (43 countries x 56 sectors). ROW is
never fabricated (see PROJECT_STATE.md §2 decision 1).

Years 2000-2014 (WIOD): socioeconomic_{YEAR}.parquet is the authoritative
node list and gross_output/value_added source.

Years 2015-2021 (Comtrade-extension period): WIOD's Socioeconomic Accounts
do not exist past 2014, so gross_output/value_added are proxied by scaling
the 2014 baseline with each country's World Bank nominal-GDP growth ratio
(see PROJECT_STATE.md §2 decision 5, src/data/download_wb_gdp.py). This is
a documented approximation, in the same spirit as extend_with_comtrade.py's
frozen-topology design.

f4 (tariff_exposure) is genuinely time-varying: 0.0 for years with no WITS
coverage (2000-2014, flagged has_tariff_data=False), and the real
trade-value-weighted rate from sector_tariffs.parquet for 2015-2021 where
it exists (PROJECT_STATE.md §2 decision 4).

Backward linkage (f3) for 2015-2021 reuses backward_linkage_2014.npy, since
the structural prior (technical coefficients) is frozen at 2014 for the
Comtrade-extension years — consistent with extend_with_comtrade.py.

Normalization statistics (mean/std) are computed strictly from WIOD years
(2000-2014, the training period) to avoid leaking 2015-2021 event-window
statistics into training-time normalization (CP16).
"""

from __future__ import annotations

import json
import os
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
N_NODES = config.GRAPH["N_NODES"]

WIOD_YEARS: list[int] = sorted(config.GRAPH["WIOD_YEARS"])          # 2000-2014
COMTRADE_YEARS: list[int] = sorted(config.GRAPH["COMTRADE_YEARS"])  # 2015-2021
ALL_YEARS: list[int] = sorted(set(WIOD_YEARS) | set(COMTRADE_YEARS))
LAST_WIOD_YEAR = max(WIOD_YEARS)


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
    """Backward linkage file for a year. 2015-2021 reuse the 2014 structural prior."""
    bl_year = year if year <= LAST_WIOD_YEAR else LAST_WIOD_YEAR
    return os.path.join(PROJECT_ROOT, "data", "processed", "leontief", f"backward_linkage_{bl_year}.npy")


TARIFF_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_TARIFF_RATES"], "sector_tariffs.parquet")
PPI_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_LABELS"], "ppi_quarterly_all.parquet")
GDP_PATH = os.path.join(PROJECT_ROOT, config.PATHS["RAW_WB_GDP"], "gdp_current_usd.parquet")
NF_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_NODE_FEATURES"])
NORM_STATS_PATH = os.path.join(PROJECT_ROOT, config.PATHS["NORM_STATS"])


# ---------------------------------------------------------------------------
# GDP scaling (for 2015-2021 socioeconomic proxy)
# ---------------------------------------------------------------------------
def build_gdp_ratio_lookup() -> dict[tuple[str, int], float]:
    """Return {(country, year): gdp[year] / gdp[LAST_WIOD_YEAR]}.

    Missing (country, year) pairs default to ratio 1.0 (flat 2014 value held
    constant) — this only affects countries WB has no GDP series for (e.g. TWN).
    """
    if not os.path.exists(GDP_PATH):
        print(f"  [gdp] File not found: {GDP_PATH} — 2015-2021 gross_output will be "
              f"held flat at {LAST_WIOD_YEAR} values. Run src/data/download_wb_gdp.py first.")
        return {}

    gdp = pd.read_parquet(GDP_PATH)
    gdp["country"] = gdp["country"].astype(str)
    gdp["year"] = gdp["year"].astype(int)

    base = gdp[gdp["year"] == LAST_WIOD_YEAR].set_index("country")["gdp_usd"].to_dict()

    ratios: dict[tuple[str, int], float] = {}
    for row in gdp.itertuples(index=False):
        base_val = base.get(row.country)
        if base_val is None or base_val <= 0:
            continue
        ratios[(row.country, int(row.year))] = float(row.gdp_usd) / float(base_val)
    return ratios


def load_or_synthesize_sea(year: int, gdp_ratios: dict[tuple[str, int], float]) -> pd.DataFrame:
    """Return the (country, sector, gross_output, value_added) node table for a year.

    year <= LAST_WIOD_YEAR: load socioeconomic_{year}.parquet directly.
    year >  LAST_WIOD_YEAR: synthesize from socioeconomic_{LAST_WIOD_YEAR}.parquet,
        scaling gross_output/value_added by the country's GDP growth ratio.
    """
    if year <= LAST_WIOD_YEAR:
        sea = pd.read_parquet(_sea_path(year))
        sea["is_gdp_scaled"] = False
        return sea

    base = pd.read_parquet(_sea_path(LAST_WIOD_YEAR)).copy()
    base["country"] = base["country"].astype(str)
    ratio = base["country"].map(lambda c: gdp_ratios.get((c, year), 1.0))
    base["gross_output"] = (base["gross_output"].astype(float) * ratio).astype("float32")
    base["value_added"] = (base["value_added"].astype(float) * ratio).astype("float32")
    base["year"] = np.int16(year)
    base["is_gdp_scaled"] = True
    return base


# ---------------------------------------------------------------------------
# Tariff lookup: {(country, sector, year) -> tariff_rate}
# sector_tariffs.parquet only covers 2015-2021 (WITS coverage window), so
# any (country, sector, year) not found returns has_tariff_data=False and
# f4=0.0 — this correctly makes f4=0.0 for all of 2000-2014.
# ---------------------------------------------------------------------------
def build_tariff_lookup() -> dict[tuple[str, str, int], float]:
    if not os.path.exists(TARIFF_PATH):
        print(f"  [tariff] File not found: {TARIFF_PATH} — tariff_exposure will be 0.0 for all years")
        return {}
    tar = pd.read_parquet(TARIFF_PATH)
    tar["sector"] = tar["sector"].astype(str).apply(normalize_sector)
    tar["country"] = tar["country"].astype(str)
    tar["year"] = tar["year"].astype(int)
    lookup: dict[tuple[str, str, int], float] = {
        (str(row.country), str(row.sector), int(row.year)): float(row.tariff_rate)
        for row in tar.itertuples(index=False)
    }
    return lookup


def get_tariff_exposure(
    country: str, sector: str, year: int, tariff_lookup: dict
) -> tuple[float, bool]:
    key = (country, sector, year)
    if key in tariff_lookup:
        return tariff_lookup[key], True
    return 0.0, False


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
    gdp_ratios: dict,
) -> pd.DataFrame:
    """Build the raw (un-normalized) feature DataFrame for one year."""

    # -----------------------------------------------------------------------
    # 1. Load (or synthesize) socioeconomic node list
    # -----------------------------------------------------------------------
    sea = load_or_synthesize_sea(year, gdp_ratios)
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
    bl_arr = np.load(_bl_path(year)).astype(np.float32)  # shape (N_NODES,)
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

    # f4: tariff_exposure — 0.0 pre-2015 (no WITS coverage), real value 2015-2021
    f4_vals = []
    has_tariff_flags = []
    for row in sea.itertuples(index=False):
        val, found = get_tariff_exposure(row.country, row.sector_norm, year, tariff_lookup)
        f4_vals.append(val)
        has_tariff_flags.append(found)
    f4 = np.array(f4_vals, dtype=np.float64)
    has_tariff_data = np.array(has_tariff_flags, dtype=bool)

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
        "year":            pd.array([year] * len(sea), dtype="int16"),
        "country":         pd.Categorical(sea["country"].values),
        "sector":          pd.Categorical(sea["sector_norm"].values),   # underscore form (fix #13)
        "node_id":         pd.array(sea["node_id"].values, dtype="int16"),
        "f0":              f0.astype("float32"),
        "f1":              f1.astype("float32"),
        "f2":              f2.astype("float32"),
        "f3":              f3.astype("float32"),
        "f4":              f4.astype("float32"),
        "f5":              f5.astype("float32"),
        "f6":              f6.astype("float32"),
        "f7":              f7.astype("float32"),
        "f8":              f8.astype("float32"),
        "has_ppi_lags":    has_ppi_lags,
        "has_tariff_data": has_tariff_data,
        "is_gdp_scaled":   bool(sea["is_gdp_scaled"].iloc[0]) if len(sea) else False,
    })

    return out


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
FEATURE_COLS = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]

# f4 (tariff_exposure) is EXCLUDED from z-score normalization. By design
# (PROJECT_STATE.md §2 decision 4) f4 is identically 0.0 for every WIOD
# training year (2000-2014) — no WITS coverage exists before 2015 — so its
# training-period mean/std are degenerate (0.0, 0.0). Z-scoring against a
# ~0 std would divide any real 2015-2021 tariff-exposure value by ~1e-8 and
# explode it by ~8 orders of magnitude. f4 is already a naturally bounded
# decimal rate (0.0-~1.0), so it is left in raw units; mean=0.0/std=1.0 are
# still recorded in normalization_stats.json as an explicit identity marker
# for downstream consumers, not because it was actually fitted.
ZSCORE_COLS = ["f0", "f1", "f2", "f3", "f5", "f6", "f7", "f8"]

# f1 (import_penetration) and f2 (export_intensity) both divide by
# (gross_output + ...), which is locked in config.py and cannot be
# redesigned here. For sectors with near-zero gross_output — common in
# small economies/niche sectors, and made worse for 2015-2021 by the
# Comtrade-extension's per-target scale factor (see PROJECT_STATE.md §1.2
# finding #17) — the raw ratio can explode to |value| > 100 even in the
# 2000-2014 training data itself (verified: p0=-174, p100=+23 for f1;
# p0=-0.32, p100=+111 for f2, in raw units). Left unclipped, z-scoring
# would produce extreme feature values that risk NaN loss during GAT/GRU
# training (CP35). Winsorize at [p1, p99] of the TRAINING period only
# (same CP16-consistent principle as normalization), applied uniformly to
# all years — the same pattern the project already uses for
# import_pen_coeff (CP07, clipped at 0.99) and backward linkage (CP18).
WINSORIZE_COLS = ["f1", "f2"]
WINSORIZE_LOWER_PCT = 1.0
WINSORIZE_UPPER_PCT = 99.0


def compute_winsorize_bounds(
    raw_frames: dict[int, pd.DataFrame],
) -> dict[str, tuple[float, float]]:
    """Compute [p1, p99] clip bounds per WINSORIZE_COLS, from WIOD_YEARS only."""
    train_frames = [raw_frames[y] for y in WIOD_YEARS if y in raw_frames]
    assert train_frames, "No WIOD-year frames available to compute winsorize bounds"
    combined = pd.concat(train_frames, ignore_index=True)
    bounds: dict[str, tuple[float, float]] = {}
    for col in WINSORIZE_COLS:
        vals = combined[col].values.astype(np.float64)
        lo = float(np.percentile(vals, WINSORIZE_LOWER_PCT))
        hi = float(np.percentile(vals, WINSORIZE_UPPER_PCT))
        bounds[col] = (lo, hi)
    return bounds


def apply_winsorize(
    df: pd.DataFrame, bounds: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        df[col] = df[col].clip(lower=lo, upper=hi).astype("float32")
    return df


def compute_normalization_stats(
    raw_frames: dict[int, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std strictly from WIOD_YEARS (2000-2014).

    CP16: normalization must never be computed from years that overlap the
    Comtrade-extension / event window (2015-2021), or test-period statistics
    leak into training-time normalization. f4 is excluded (see ZSCORE_COLS).
    """
    train_frames = [raw_frames[y] for y in WIOD_YEARS if y in raw_frames]
    assert train_frames, "No WIOD-year frames available to compute normalization stats"
    combined = pd.concat(train_frames, ignore_index=True)

    means = np.zeros(len(FEATURE_COLS), dtype=np.float64)
    stds = np.ones(len(FEATURE_COLS), dtype=np.float64)
    for i, col in enumerate(FEATURE_COLS):
        if col in ZSCORE_COLS:
            means[i] = combined[col].mean()
            stds[i] = combined[col].std(ddof=0)
    return means, stds


def apply_normalization(
    df: pd.DataFrame, means: np.ndarray, stds: np.ndarray
) -> pd.DataFrame:
    """Apply z-score normalization: (x - mean) / (std + 1e-8) for ZSCORE_COLS,
    identity (raw passthrough) for f4."""
    df = df.copy()
    for i, col in enumerate(FEATURE_COLS):
        if col in ZSCORE_COLS:
            df[col] = ((df[col].values.astype(np.float64) - means[i]) / (stds[i] + 1e-8)).astype("float32")
        # f4: left as-is (raw value already assembled in compute_features_for_year)
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

    assert df["node_id"].between(0, N_NODES - 1).all(), f"Year {year}: node_id out of range"

    dup = df.duplicated(subset=["country", "sector"]).sum()
    assert dup == 0, f"Year {year}: {dup} duplicate (country, sector) pairs"

    bad_sectors = set(df["sector"].astype(str).unique()) - set(config.GRAPH["SECTOR_LIST"])
    assert not bad_sectors, f"Year {year}: sector codes not in config.SECTOR_LIST: {bad_sectors}"

    ppi_coverage = float(df["has_ppi_lags"].mean()) * 100
    tariff_coverage = float(df["has_tariff_data"].mean()) * 100

    print(
        f"  Node features {year}: rows={len(df)}  nan_check=passed  "
        f"ppi_coverage={ppi_coverage:.1f}%  tariff_coverage={tariff_coverage:.1f}%"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(NF_DIR, exist_ok=True)

    years = ALL_YEARS
    print(f"Computing node features for {len(years)} years: {years[0]}-{years[-1]}")
    print(f"  WIOD years (real socioeconomic data): {WIOD_YEARS[0]}-{WIOD_YEARS[-1]}")
    print(f"  Comtrade-extension years (GDP-scaled proxy): {COMTRADE_YEARS[0]}-{COMTRADE_YEARS[-1]}")

    # -- Load shared data once -----------------------------------------------
    print("Loading tariff data...")
    tariff_lookup = build_tariff_lookup()

    print("Loading PPI data...")
    country_ppi, wld_ppi = build_ppi_lookup()

    print("Loading GDP ratios...")
    gdp_ratios = build_gdp_ratio_lookup()

    # -- Compute raw features for all years ----------------------------------
    raw_frames: dict[int, pd.DataFrame] = {}
    for year in years:
        print(f"Computing raw features for {year}...")
        df = compute_features_for_year(year, tariff_lookup, country_ppi, wld_ppi, gdp_ratios)
        raw_frames[year] = df

    # -- Winsorize f1/f2 at training-period [p1,p99] before normalizing ------
    print("Computing winsorize bounds for f1/f2 (2000-2014 only)...")
    winsorize_bounds = compute_winsorize_bounds(raw_frames)
    for col, (lo, hi) in winsorize_bounds.items():
        print(f"  {col}: clip to [{lo:.4f}, {hi:.4f}]")
    for year in years:
        raw_frames[year] = apply_winsorize(raw_frames[year], winsorize_bounds)

    # -- Normalization stats computed on WIOD_YEARS ONLY (training period) --
    print("Computing normalization statistics (2000-2014 only, CP16)...")
    means, stds = compute_normalization_stats(raw_frames)

    norm_stats = {
        "mean": means.tolist(),
        "std": stds.tolist(),
        "computed_from_years": WIOD_YEARS,
        "feature_names": FEATURE_COLS,
        "zscore_features": ZSCORE_COLS,
        "identity_features": [c for c in FEATURE_COLS if c not in ZSCORE_COLS],
        "winsorize_bounds": {c: list(b) for c, b in winsorize_bounds.items()},
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
