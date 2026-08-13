"""generate_labels.py — Build event-relative 3m/6m/12m PPI-change labels.

For each of the 6 tariff events, and each (country, sector) node, computes
the cumulative PPI change from the event quarter to the event quarter + k
quarters (k=1 for 3m, k=2 for 6m, k=4 for 12m), per Locked Plan §4.4 (CP17:
labels must be aligned to the event's actual quarter, not an adjacent one).

ppi_quarterly_all.parquet stores quarter-over-quarter PERCENT CHANGE
(ppi_change), not price levels. A k-quarter-ahead cumulative change is
reconstructed by compounding the intervening quarterly changes:

    delta_k = prod(1 + ppi_change[event_quarter + i] for i in 1..k) - 1

Country coverage in ppi_quarterly_all.parquet is limited (29/44 countries;
see PROJECT_STATE.md §1.2 finding #15 and §2 decision 3, accepted as a
scoped limitation). Each quarter step first tries (country, sector); if
missing, falls back to (WLD, sector) — the same two-tier priority already
used for the PPI-lag node features in compute_node_features.py. has_label
is True only if ALL quarters needed for the 12m horizon were found (CP24:
never mark has_label=True with a partially-null horizon) — this
automatically covers 3m and 6m too, since they are partial products of the
same quarterly series.

Output: data/processed/labels/labels_{event_name}.parquet
    Schema: event_name, country, sector, node_id,
            delta_3m, delta_6m, delta_12m (float32, NaN if not has_label),
            has_label (bool), label_source (category: bls/eurostat/wb_commodity/null)

Usage:
    python src/data/generate_labels.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if _here not in sys.path:
    sys.path.insert(0, _here)

import config  # noqa: E402
from quarterly_interpolation import parse_event_date, step_back_quarters  # noqa: E402

PPI_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_LABELS"], "ppi_quarterly_all.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_LABELS"])

COUNTRY_LIST = config.GRAPH["COUNTRY_LIST"]
SECTOR_LIST = config.GRAPH["SECTOR_LIST"]
COUNTRY_IDX = {c: i for i, c in enumerate(COUNTRY_LIST)}
SECTOR_IDX = {s: i for i, s in enumerate(SECTOR_LIST)}

HORIZONS = {"delta_3m": 1, "delta_6m": 2, "delta_12m": 4}   # k = quarters ahead


def normalize_sector(raw: str) -> str:
    return str(raw).replace("-", "_")


def step_forward_quarters(year: int, quarter: int, n: int) -> tuple[int, int]:
    return step_back_quarters(year, quarter, -n)


def build_ppi_change_lookup() -> tuple[dict, dict]:
    """Return (country_lookup, wld_lookup):
    country_lookup[(country, sector, year, quarter)] -> (ppi_change, source)
    wld_lookup[(sector, year, quarter)] -> (ppi_change, source)
    """
    if not os.path.exists(PPI_PATH):
        raise FileNotFoundError(f"{PPI_PATH} not found — run clean_ppi.py first")

    ppi = pd.read_parquet(PPI_PATH)
    ppi["isic_sector"] = ppi["isic_sector"].astype(str).apply(normalize_sector)
    ppi["country"] = ppi["country"].astype(str)
    ppi["source"] = ppi["source"].astype(str)
    ppi["year"] = ppi["year"].astype(int)
    ppi["quarter"] = ppi["quarter"].astype(int)

    country_lookup: dict[tuple[str, str, int, int], tuple[float, str]] = {}
    wld_lookup: dict[tuple[str, int, int], tuple[float, str]] = {}

    for row in ppi.itertuples(index=False):
        val = (float(row.ppi_change), row.source)
        country_lookup[(row.country, row.isic_sector, int(row.year), int(row.quarter))] = val
        if row.country == "WLD":
            wld_lookup[(row.isic_sector, int(row.year), int(row.quarter))] = val

    return country_lookup, wld_lookup


def get_quarter_change(
    country: str, sector: str, year: int, quarter: int, country_lookup: dict, wld_lookup: dict
) -> tuple[float, str] | None:
    key_c = (country, sector, year, quarter)
    if key_c in country_lookup:
        return country_lookup[key_c]
    key_w = (sector, year, quarter)
    if key_w in wld_lookup:
        return wld_lookup[key_w]
    return None


def compute_node_labels(
    country: str, sector: str, event_year: int, event_quarter: int,
    country_lookup: dict, wld_lookup: dict,
) -> dict:
    """Compute delta_3m/6m/12m for one (country, sector) via compounding."""
    max_k = max(HORIZONS.values())
    cumulative = 1.0
    per_quarter_source: list[str] = []
    quarter_deltas: dict[int, float] = {}   # k -> cumulative delta

    complete = True
    for i in range(1, max_k + 1):
        y, q = step_forward_quarters(event_year, event_quarter, i)
        found = get_quarter_change(country, sector, y, q, country_lookup, wld_lookup)
        if found is None:
            complete = False
            break
        change, source = found
        cumulative *= (1.0 + change)
        per_quarter_source.append(source)
        if i in HORIZONS.values():
            quarter_deltas[i] = cumulative - 1.0

    if not complete:
        return {
            "delta_3m": np.nan, "delta_6m": np.nan, "delta_12m": np.nan,
            "has_label": False, "label_source": None,
        }

    return {
        "delta_3m": quarter_deltas[HORIZONS["delta_3m"]],
        "delta_6m": quarter_deltas[HORIZONS["delta_6m"]],
        "delta_12m": quarter_deltas[HORIZONS["delta_12m"]],
        "has_label": True,
        "label_source": per_quarter_source[0],   # source of the first (3m) quarter
    }


def build_event_labels(event: dict, country_lookup: dict, wld_lookup: dict) -> pd.DataFrame:
    event_year, event_quarter = parse_event_date(event["date"])

    rows = []
    for country in COUNTRY_LIST:
        ci = COUNTRY_IDX[country]
        for sector in SECTOR_LIST:
            si = SECTOR_IDX[sector]
            node_id = config.node_id(ci, si)
            labels = compute_node_labels(
                country, sector, event_year, event_quarter, country_lookup, wld_lookup
            )
            rows.append({
                "event_name": event["name"],
                "country": country,
                "sector": sector,
                "node_id": node_id,
                **labels,
            })

    df = pd.DataFrame(rows)
    df["event_name"] = df["event_name"].astype("category")
    df["country"] = df["country"].astype("category")
    df["sector"] = df["sector"].astype("category")
    df["node_id"] = df["node_id"].astype("int16")
    df["delta_3m"] = df["delta_3m"].astype("float32")
    df["delta_6m"] = df["delta_6m"].astype("float32")
    df["delta_12m"] = df["delta_12m"].astype("float32")
    df["has_label"] = df["has_label"].astype(bool)
    df["label_source"] = df["label_source"].astype("category")
    return df


def validate(df: pd.DataFrame, event_name: str) -> float:
    labeled = df[df["has_label"]]
    assert not labeled[["delta_3m", "delta_6m", "delta_12m"]].isna().any().any(), (
        f"{event_name}: NaN found among rows marked has_label=True (CP24 violation)"
    )
    unlabeled = df[~df["has_label"]]
    assert unlabeled[["delta_3m", "delta_6m", "delta_12m"]].isna().all().all(), (
        f"{event_name}: has_label=False rows should have all-NaN deltas"
    )
    coverage = float(df["has_label"].mean())
    return coverage


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading PPI quarterly change lookup...")
    country_lookup, wld_lookup = build_ppi_change_lookup()

    coverage_report = []
    for event in config.EVENTS:
        print(f"\nGenerating labels for {event['name']} (date {event['date']})...")
        df = build_event_labels(event, country_lookup, wld_lookup)
        coverage = validate(df, event["name"])
        coverage_report.append((event["name"], coverage, int(df["has_label"].sum()), len(df)))

        out_path = os.path.join(OUT_DIR, f"labels_{event['name']}.parquet")
        df.to_parquet(out_path, index=False)
        flag = "" if coverage >= 0.60 else "  <-- BELOW 60% TARGET (CP19, accepted per PROJECT_STATE.md decision 3)"
        print(f"  coverage: {coverage:.1%} ({int(df['has_label'].sum())}/{len(df)} nodes){flag}")
        print(f"  Saved -> {out_path}")

    print("\n" + "=" * 60)
    print("LABEL COVERAGE SUMMARY (per event)")
    print("=" * 60)
    for name, cov, n_labeled, n_total in coverage_report:
        flag = "OK" if cov >= 0.60 else "BELOW 60%"
        print(f"  {name:28s} {cov:6.1%}  ({n_labeled}/{n_total})  [{flag}]")


if __name__ == "__main__":
    main()
