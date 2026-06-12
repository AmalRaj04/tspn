"""Build tariff shock vectors for all events defined in config.EVENTS.

Pipeline
--------
For each event:
  1. Load the event's HTS CSV file.
  2. Auto-detect the HTS code column.
  3. Truncate HTS codes to 6-digit HS6.
  4. Look up WIOD-56 sector weights from concordance.
  5. Distribute the delta_tariff across sectors proportionally.
  6. Expand over all affected (src, tgt) country pairs.
  7. Save as shock_{event_name}.parquet.

Public API
----------
    compute_shock_vector(event_name: str) -> pd.DataFrame
    build_all_shock_vectors() -> None

These are importable by scenario_parser.py, feature_builder.py, and the
inference pipeline.

Output schema (per file)
------------------------
    event_name   : str        (scalar constant per file)
    src_country  : category   (exporting country)
    tgt_country  : category   (importing country)
    sector       : category   (WIOD-56 code)
    delta_tariff : float32    (decimal; 0.25 = 25 pp shock)
    is_direct_hit: bool       (delta_tariff != 0)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COUNTRY_LIST: list[str] = config.GRAPH["COUNTRY_LIST"]
SECTOR_LIST: list[str] = config.GRAPH["SECTOR_LIST"]

RAW_EVENTS = ROOT / "data/raw/tariff_events"
PROC_SHOCK = ROOT / config.PATHS["PROCESSED_SHOCK_VECTORS"]
PROC_CONC = ROOT / config.PATHS["PROCESSED_CONCORDANCE"]

PROC_SHOCK.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Concordance loader (cached at module level)
# ---------------------------------------------------------------------------
_CONCORDANCE: dict[str, dict[str, float]] | None = None


def _load_concordance() -> dict[str, dict[str, float]]:
    """Return {hs6: {wiod_sector: weight}}, loaded once."""
    global _CONCORDANCE
    if _CONCORDANCE is None:
        path = PROC_CONC / "hs6_to_wiod56_weights.json"
        with open(path) as fh:
            _CONCORDANCE = json.load(fh)
    return _CONCORDANCE


# ---------------------------------------------------------------------------
# HTS column detection
# ---------------------------------------------------------------------------
_HTS_COLUMN_CANDIDATES = ["hts_code", "product_code", "commodity__code"]


def _detect_hts_column(df: pd.DataFrame) -> str:
    """Return the name of the HTS code column, raising if not found."""
    for candidate in _HTS_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Cannot detect HTS column. Found columns: {df.columns.tolist()}. "
        f"Expected one of: {_HTS_COLUMN_CANDIDATES}"
    )


# ---------------------------------------------------------------------------
# Sector shock computation
# ---------------------------------------------------------------------------

def _compute_event_sector_shocks(
    hts_df: pd.DataFrame,
    hts_col: str,
    concordance: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute the weighted-average delta_tariff per WIOD sector for one event.

    For every HTS row that maps to the concordance:
        hs6   = first 6 digits of hts_code
        delta = delta_tariff_pct / 100

    For each (sector, weight) in concordance[hs6]:
        weighted_sum[sector]   += weight × delta
        weight_total[sector]   += weight

    Final sector shock = weighted_sum[sector] / weight_total[sector]

    This produces the weight-normalised mean delta-tariff per sector rather
    than a running total that grows with the number of HTS lines.

    Returns {wiod_sector: weighted_average_delta_tariff}.
    """
    weighted_sum: dict[str, float] = defaultdict(float)
    weight_total: dict[str, float] = defaultdict(float)

    sector_set = set(SECTOR_LIST)

    for _, row in hts_df.iterrows():
        raw_code = str(row[hts_col]).strip().zfill(8)
        hs6 = raw_code[:6]

        mapping = concordance.get(hs6)
        if mapping is None:
            continue

        delta_pct = float(row["delta_tariff_pct"])
        delta = delta_pct / 100.0

        for sector, weight in mapping.items():
            if sector in sector_set:
                weighted_sum[sector] += weight * delta
                weight_total[sector] += weight

    # Normalise
    result: dict[str, float] = {}
    for sector in weighted_sum:
        wt = weight_total[sector]
        result[sector] = weighted_sum[sector] / wt if wt > 0.0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Country-pair expansion
# ---------------------------------------------------------------------------

def _resolve_countries(spec: list[str] | str) -> list[str]:
    """Expand "all" to the full COUNTRY_LIST; otherwise return as-is."""
    if spec == "all":
        return COUNTRY_LIST
    return list(spec)


def _expand_to_pairs(
    event_name: str,
    sector_shocks: dict[str, float],
    importers: list[str],
    exporters: list[str],
) -> pd.DataFrame:
    """Build the rows table for all (src, tgt, sector) triples.

    Only generates rows for sectors that appear in sector_shocks (i.e. have
    at least one mapped HTS code).  Missing sectors are treated as zero shock
    by downstream consumers.
    """
    records: list[dict[str, Any]] = []

    for tgt in importers:          # tgt = importing country
        for src in exporters:      # src = exporting country
            for sector, delta in sector_shocks.items():
                records.append(
                    {
                        "event_name": event_name,
                        "src_country": src,
                        "tgt_country": tgt,
                        "sector": sector,
                        "delta_tariff": float(delta),
                    }
                )

    if not records:
        # Return an empty but correctly-typed DataFrame
        return pd.DataFrame(
            columns=["event_name", "src_country", "tgt_country",
                     "sector", "delta_tariff", "is_direct_hit"]
        )

    df = pd.DataFrame.from_records(records)
    df["is_direct_hit"] = df["delta_tariff"] != 0.0

    # Cast types
    df["src_country"] = df["src_country"].astype("category")
    df["tgt_country"] = df["tgt_country"].astype("category")
    df["sector"] = df["sector"].astype("category")
    df["delta_tariff"] = df["delta_tariff"].astype("float32")
    df["is_direct_hit"] = df["is_direct_hit"].astype(bool)

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(event: dict, df: pd.DataFrame) -> None:
    """Run per-event validation assertions."""
    name = event["name"]

    # Sector membership
    bad = ~df["sector"].isin(SECTOR_LIST)
    assert not bad.any(), (
        f"[{name}] Unknown WIOD sectors: {df.loc[bad,'sector'].unique().tolist()}"
    )

    # Event-specific importer/exporter checks
    if name == "us_232_steel_2018":
        assert (df["tgt_country"] == "USA").all(), \
            f"[{name}] Expected all tgt_country == USA"

    elif name == "us_232_aluminum_2018":
        assert (df["tgt_country"] == "USA").all(), \
            f"[{name}] Expected all tgt_country == USA"

    elif name == "us_301_list1_2018":
        assert (df["src_country"] == "CHN").all(), \
            f"[{name}] Expected all src_country == CHN"
        assert (df["tgt_country"] == "USA").all(), \
            f"[{name}] Expected all tgt_country == USA"

    elif name == "us_301_list2_2018":
        assert (df["src_country"] == "CHN").all(), \
            f"[{name}] Expected all src_country == CHN"
        assert (df["tgt_country"] == "USA").all(), \
            f"[{name}] Expected all tgt_country == USA"

    # Delta tariff sign check (allow negatives only for UK event)
    if name != "uk_global_tariff_2021":
        assert df["delta_tariff"].min() >= 0, (
            f"[{name}] Unexpected negative delta_tariff "
            f"(min={df['delta_tariff'].min():.4f})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_shock_vector(event_name: str) -> pd.DataFrame:
    """Compute and return the shock DataFrame for a single named event.

    Args:
        event_name: Must match one of the ``name`` fields in config.EVENTS.

    Returns:
        DataFrame with columns:
            event_name, src_country, tgt_country, sector,
            delta_tariff, is_direct_hit
    """
    event = next(
        (e for e in config.EVENTS if e["name"] == event_name), None
    )
    if event is None:
        raise ValueError(
            f"Unknown event '{event_name}'. "
            f"Available: {[e['name'] for e in config.EVENTS]}"
        )
    return _process_event(event, verbose=False)


def build_all_shock_vectors() -> None:
    """Build and save shock vectors for every event in config.EVENTS."""
    print("Building shock vectors for all tariff events …\n")
    concordance = _load_concordance()
    print(f"Concordance loaded: {len(concordance):,} HS6 codes.\n")

    for event in config.EVENTS:
        _process_event(event, verbose=True, concordance=concordance)

    print("\nDone. Shock vectors saved to:", PROC_SHOCK.relative_to(ROOT))


# ---------------------------------------------------------------------------
# Internal processing (shared by public API)
# ---------------------------------------------------------------------------

def _process_event(
    event: dict,
    *,
    verbose: bool = True,
    concordance: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Core processing logic for one event.

    Args:
        event:       Entry from config.EVENTS.
        verbose:     Whether to print progress/reporting lines.
        concordance: Pre-loaded concordance dict (loaded if None).

    Returns:
        The processed shock DataFrame.
    """
    if concordance is None:
        concordance = _load_concordance()

    name = event["name"]
    hts_file = ROOT / event["hts_file"]

    # ------------------------------------------------------------------
    # Load event CSV
    # ------------------------------------------------------------------
    hts_df = pd.read_csv(hts_file, dtype=str)

    # If the event has a fixed delta_tariff_pct (not per-row), inject it
    fixed_delta = event.get("delta_tariff_pct")
    if fixed_delta is not None and "delta_tariff_pct" not in hts_df.columns:
        hts_df["delta_tariff_pct"] = str(fixed_delta)

    if "delta_tariff_pct" not in hts_df.columns:
        raise ValueError(
            f"[{name}] No delta_tariff_pct column found and none set in "
            f"config.EVENTS. Columns: {hts_df.columns.tolist()}"
        )

    hts_col = _detect_hts_column(hts_df)
    hts_df["delta_tariff_pct"] = hts_df["delta_tariff_pct"].astype(float)

    # ------------------------------------------------------------------
    # Compute per-sector shocks
    # ------------------------------------------------------------------
    sector_shocks = _compute_event_sector_shocks(hts_df, hts_col, concordance)

    # ------------------------------------------------------------------
    # Resolve country pairs
    # ------------------------------------------------------------------
    importers = _resolve_countries(event["affected_importers"])
    exporters = _resolve_countries(event["affected_exporters"])

    # ------------------------------------------------------------------
    # Expand to (src, tgt, sector) rows
    # ------------------------------------------------------------------
    df = _expand_to_pairs(name, sector_shocks, importers, exporters)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    _validate(event, df)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = PROC_SHOCK / f"shock_{name}.parquet"
    df.to_parquet(out_path, index=False)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    if verbose:
        n_sectors = df["sector"].nunique()
        n_rows = len(df)
        max_d = float(df["delta_tariff"].max()) if n_rows else 0.0
        min_d = float(df["delta_tariff"].min()) if n_rows else 0.0
        print(
            f"Event  {name}\n"
            f"  affected sectors : {n_sectors}\n"
            f"  affected pairs   : {n_rows:,}\n"
            f"  max delta tariff : {max_d:.4f}\n"
            f"  min delta tariff : {min_d:.4f}\n"
            f"  saved → {out_path.relative_to(ROOT)}\n"
        )

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_all_shock_vectors()
