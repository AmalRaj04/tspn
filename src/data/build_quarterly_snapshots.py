"""build_quarterly_snapshots.py — Build 8 quarterly node-feature snapshots per event.

For each of the 6 tariff events, identifies the 8 quarters ending at the
event's quarter (Locked Plan §5.2: event_quarter-7 .. event_quarter
inclusive) and linearly interpolates node_features_{YEAR}.parquet between
adjacent years to produce quarterly node feature snapshots (see
quarterly_interpolation.py for the interpolation convention).

Output: data/processed/node_features_quarterly/{event_name}.parquet
    One file per event, all 8 snapshots stacked.
    Schema: event_name, snapshot_idx (0-7), year, quarter, node_id,
            country, sector, f0..f8, has_ppi_lags, has_tariff_data

Usage:
    python src/data/build_quarterly_snapshots.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if _here not in sys.path:
    sys.path.insert(0, _here)

import config  # noqa: E402
from quarterly_interpolation import (  # noqa: E402
    parse_event_date,
    quarter_sequence,
    interpolate_frame,
)

NF_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_NODE_FEATURES"])
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "node_features_quarterly")
FEATURE_COLS = [f"f{i}" for i in range(9)]
N_NODES = config.GRAPH["N_NODES"]

_year_cache: dict[int, pd.DataFrame] = {}


def load_node_features(year: int) -> pd.DataFrame:
    if year not in _year_cache:
        path = os.path.join(NF_DIR, f"node_features_{year}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"node_features_{year}.parquet not found — quarter sequence "
                f"needs years outside the available range. Path: {path}"
            )
        _year_cache[year] = pd.read_parquet(path)
    return _year_cache[year]


def build_event_snapshots(event: dict) -> pd.DataFrame:
    event_year, event_quarter = parse_event_date(event["date"])
    quarters = quarter_sequence(event_year, event_quarter)

    frames = []
    for idx, (year, quarter) in enumerate(quarters):
        df_curr = load_node_features(year)
        if quarter == 4:
            df_q = df_curr[["node_id", "country", "sector"] + FEATURE_COLS].copy()
            flags = df_curr[["node_id", "has_ppi_lags", "has_tariff_data"]]
        else:
            df_prev = load_node_features(year - 1)
            interp = interpolate_frame(
                df_prev, df_curr, key_cols=["node_id"], value_cols=FEATURE_COLS, quarter=quarter
            )
            meta = df_curr[["node_id", "country", "sector"]]
            df_q = interp.merge(meta, on="node_id", how="left")
            df_q = df_q[["node_id", "country", "sector"] + FEATURE_COLS]
            flags = df_curr[["node_id", "has_ppi_lags", "has_tariff_data"]]

        df_q = df_q.merge(flags, on="node_id", how="left")
        df_q["event_name"] = event["name"]
        df_q["snapshot_idx"] = idx
        df_q["year"] = year
        df_q["quarter"] = quarter
        frames.append(df_q)

    out = pd.concat(frames, ignore_index=True)
    cols = ["event_name", "snapshot_idx", "year", "quarter", "node_id", "country", "sector"] + \
        FEATURE_COLS + ["has_ppi_lags", "has_tariff_data"]
    out = out[cols]
    out["node_id"] = out["node_id"].astype("int16")
    out["snapshot_idx"] = out["snapshot_idx"].astype("int8")
    out["year"] = out["year"].astype("int16")
    out["quarter"] = out["quarter"].astype("int8")
    for c in FEATURE_COLS:
        out[c] = out[c].astype("float32")
    return out


def validate(df: pd.DataFrame, event_name: str) -> None:
    assert df["snapshot_idx"].nunique() == 8, f"{event_name}: expected 8 snapshots"
    for idx in range(8):
        n = len(df[df["snapshot_idx"] == idx])
        assert n == N_NODES, f"{event_name} snapshot {idx}: {n} rows (expected {N_NODES})"
    nan_cols = [c for c in FEATURE_COLS if df[c].isna().any()]
    assert not nan_cols, f"{event_name}: NaN in {nan_cols}"
    print(f"  {event_name}: 8 snapshots x {N_NODES} nodes, "
          f"years {df['year'].min()}-{df['year'].max()}, validation passed")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for event in config.EVENTS:
        print(f"Building quarterly node-feature snapshots for {event['name']} "
              f"(event date {event['date']})...")
        df = build_event_snapshots(event)
        validate(df, event["name"])
        out_path = os.path.join(OUT_DIR, f"{event['name']}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"  Saved -> {out_path}")

    print(f"\nDone. {len(config.EVENTS)} event quarterly-snapshot files saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
