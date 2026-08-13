"""compute_edge_features.py — Build the 6-dim edge feature matrix per event x 8 quarters.

Features (locked order, config.EDGE_FEATURES):
    e[0] log_trade_flow    = log1p(quarterly-interpolated flow_usd)
    e[1] import_pen_coeff  = quarterly-interpolated import_pen_coeff
    e[2] applied_tariff    = importer's (tgt_country) MFN rate on the traded
                             product's sector (= src_sector, the classification
                             of what's being imported — see note below), from
                             sector_tariffs.parquet. 0.0 for years <2015 (no
                             WITS coverage).
    e[3] tariff_delta      = THE SHOCK SIGNAL. 0.0 for snapshot_idx 0-6.
                             At snapshot_idx 7 (event quarter) only: joined
                             from shock_{event}.parquet on
                             (src_country, tgt_country, sector=src_sector).
    e[4] product_hhi       = Herfindahl index of the importer's (tgt_country)
                             HS2-level import composition within the HS2
                             codes mapped to this edge's src_sector (see
                             scoped-limitation note below). 0.0 for years
                             without Comtrade data (<2015).
    e[5] domestic_flag     = 1.0 if src_country == tgt_country else 0.0

Sector-matching note (important, see PROJECT_STATE.md): a tariff is levied
by the importer on goods classified under the traded PRODUCT's sector. In a
WIOD edge (src_country, src_sector) -> (tgt_country, tgt_sector), src_sector
is that product classification (the exporting industry), not tgt_sector
(the purchasing industry) — confirmed by reading build_shock_vectors.py,
whose "sector" column is the tariffed product's WIOD sector. e2 and e3 are
therefore joined on (tgt_country, src_sector), not (tgt_country, tgt_sector).

Scoped limitation — product_hhi (e4): the true spec ("HHI across HS6 codes
in this bilateral-sector pair") is not achievable with the data actually on
disk. data/raw/comtrade/comtrade_{ISO3}_{YEAR}.parquet is HS2-level only (97
codes, verified) and has no bilateral partner breakdown (partner column is
always None — extend_with_comtrade.py's own docstring confirms Comtrade was
pulled without partner disaggregation). e4 is therefore computed as the
IMPORTER's own HS2-level import concentration within the HS2 codes mapped to
the edge's sector (via extend_with_comtrade.HS2_SECTOR_MAP), applied
uniformly to every source country trading into that (tgt_country, sector)
node — not truly bilateral, and only present for goods-producing sectors
that have an HS2 mapping at all (most services sectors get e4=0.0).

Output: data/processed/edge_features/edge_features_{event_name}_q{0-7}.parquet
    Schema: src_id, tgt_id, src_country, src_sector, tgt_country, tgt_sector,
            e0, e1, e2, e3, e4, e5

Usage:
    python src/data/compute_edge_features.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if _here not in sys.path:
    sys.path.insert(0, _here)

import config  # noqa: E402
from quarterly_interpolation import parse_event_date, quarter_sequence, interpolate_frame  # noqa: E402
from extend_with_comtrade import HS2_SECTOR_MAP  # noqa: E402

EDGES_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_EDGES"])
TARIFF_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_TARIFF_RATES"], "sector_tariffs.parquet")
SHOCK_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_SHOCK_VECTORS"])
RAW_COMTRADE = os.path.join(PROJECT_ROOT, config.PATHS["RAW_COMTRADE"])
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "edge_features")

EDGE_KEY = ["src_id", "tgt_id", "src_country", "src_sector", "tgt_country", "tgt_sector"]

# Reverse of HS2_SECTOR_MAP: sector -> list of hs2 codes that (partially) map to it
SECTOR_TO_HS2: dict[str, list[int]] = defaultdict(list)
for _hs2, _mapping in HS2_SECTOR_MAP.items():
    for _sector, _share in _mapping:
        SECTOR_TO_HS2[_sector].append(_hs2)

_edges_cache: dict[int, pd.DataFrame] = {}
_comtrade_cache: dict[tuple[str, int], dict[int, float]] = {}


def load_edges(year: int) -> pd.DataFrame:
    if year not in _edges_cache:
        path = os.path.join(EDGES_DIR, f"edges_{year}.parquet")
        df = pd.read_parquet(path)
        for c in ["src_country", "src_sector", "tgt_country", "tgt_sector"]:
            df[c] = df[c].astype(str)
        _edges_cache[year] = df
    return _edges_cache[year]


def load_tariff_lookup() -> dict[tuple[str, str, int], float]:
    if not os.path.exists(TARIFF_PATH):
        print(f"  [tariff] {TARIFF_PATH} not found — e2 will be 0.0 for all years")
        return {}
    tar = pd.read_parquet(TARIFF_PATH)
    tar["country"] = tar["country"].astype(str)
    tar["sector"] = tar["sector"].astype(str)
    tar["year"] = tar["year"].astype(int)
    return {
        (r.country, r.sector, int(r.year)): float(r.tariff_rate)
        for r in tar.itertuples(index=False)
    }


def load_shock_lookup(event_name: str) -> dict[tuple[str, str, str], float]:
    path = os.path.join(SHOCK_DIR, f"shock_{event_name}.parquet")
    if not os.path.exists(path):
        print(f"  [shock] {path} not found — e3 will be 0.0")
        return {}
    sv = pd.read_parquet(path)
    sv["src_country"] = sv["src_country"].astype(str)
    sv["tgt_country"] = sv["tgt_country"].astype(str)
    sv["sector"] = sv["sector"].astype(str)
    return {
        (r.src_country, r.tgt_country, r.sector): float(r.delta_tariff)
        for r in sv.itertuples(index=False)
        if r.delta_tariff != 0.0
    }


def comtrade_hs2_totals(tgt_country: str, year: int) -> dict[int, float]:
    """Return {hs2: total_trade_value_usd} for one (country, year), cached."""
    key = (tgt_country, year)
    if key in _comtrade_cache:
        return _comtrade_cache[key]
    path = os.path.join(RAW_COMTRADE, f"comtrade_{tgt_country}_{year}.parquet")
    if not os.path.exists(path):
        _comtrade_cache[key] = {}
        return {}
    df = pd.read_parquet(path)
    df = df.dropna(subset=["trade_value_usd"])
    df = df[df["trade_value_usd"] > 0]
    totals: dict[int, float] = defaultdict(float)
    for row in df.itertuples(index=False):
        try:
            hs2 = int(str(row.commodity_code).strip().lstrip("0") or "0")
        except (ValueError, TypeError):
            continue
        totals[hs2] += float(row.trade_value_usd)
    _comtrade_cache[key] = dict(totals)
    return _comtrade_cache[key]


def hhi_for_sector(tgt_country: str, sector: str, year: int) -> float:
    hs2_codes = SECTOR_TO_HS2.get(sector)
    if not hs2_codes:
        return 0.0
    totals = comtrade_hs2_totals(tgt_country, year)
    if not totals:
        return 0.0
    vals = [totals.get(h, 0.0) for h in hs2_codes]
    total = sum(vals)
    if total <= 0:
        return 0.0
    return float(sum((v / total) ** 2 for v in vals))


def build_edge_flow_features(year: int, quarter: int) -> pd.DataFrame:
    """Interpolated e0 (log flow) and e1 (import_pen_coeff) for one quarter."""
    df_curr = load_edges(year)
    if quarter == 4:
        base = df_curr[EDGE_KEY + ["flow_usd", "import_pen_coeff"]].copy()
    else:
        df_prev = load_edges(year - 1)
        base = interpolate_frame(
            df_prev, df_curr, key_cols=EDGE_KEY,
            value_cols=["flow_usd", "import_pen_coeff"], quarter=quarter,
        )
    base["e0"] = np.log1p(base["flow_usd"].clip(lower=0.0))
    base["e1"] = base["import_pen_coeff"].clip(lower=0.0, upper=1.0)
    return base.drop(columns=["flow_usd", "import_pen_coeff"])


def build_quarter_edge_features(
    event: dict,
    year: int,
    quarter: int,
    snapshot_idx: int,
    tariff_lookup: dict,
    shock_lookup: dict,
) -> pd.DataFrame:
    df = build_edge_flow_features(year, quarter)

    # e2: importer's applied tariff on the traded product's sector (= src_sector)
    has_tariff_year = year >= 2015
    if has_tariff_year:
        df["e2"] = [
            tariff_lookup.get((tgt, src_sec, year), 0.0)
            for tgt, src_sec in zip(df["tgt_country"], df["src_sector"])
        ]
    else:
        df["e2"] = 0.0

    # e3: shock signal — ONLY injected at the event-quarter snapshot (idx 7), CP21
    if snapshot_idx == 7 and shock_lookup:
        df["e3"] = [
            shock_lookup.get((src, tgt, src_sec), 0.0)
            for src, tgt, src_sec in zip(df["src_country"], df["tgt_country"], df["src_sector"])
        ]
    else:
        df["e3"] = 0.0

    # e4: importer's HS2-level import concentration for this edge's sector
    if has_tariff_year:  # Comtrade files exist for the same 2015+ years
        hhi_cache: dict[tuple[str, str], float] = {}
        e4_vals = []
        for tgt, src_sec in zip(df["tgt_country"], df["src_sector"]):
            k = (tgt, src_sec)
            if k not in hhi_cache:
                hhi_cache[k] = hhi_for_sector(tgt, src_sec, year)
            e4_vals.append(hhi_cache[k])
        df["e4"] = e4_vals
    else:
        df["e4"] = 0.0

    # e5: domestic flag
    df["e5"] = (df["src_country"] == df["tgt_country"]).astype("float32")

    out = df[EDGE_KEY + ["e0", "e1", "e2", "e3", "e4", "e5"]].copy()
    for c in ["e0", "e1", "e2", "e3", "e4", "e5"]:
        out[c] = out[c].astype("float32")
    out["src_id"] = out["src_id"].astype("int16")
    out["tgt_id"] = out["tgt_id"].astype("int16")
    return out


def validate(df: pd.DataFrame, event_name: str, snapshot_idx: int) -> None:
    assert not df[["e0", "e1", "e2", "e3", "e4", "e5"]].isna().any().any(), (
        f"{event_name} q{snapshot_idx}: NaN in edge features"
    )
    if snapshot_idx < 7:
        assert (df["e3"] == 0.0).all(), (
            f"{event_name} q{snapshot_idx}: shock leaked into pre-event snapshot (CP21)"
        )
    max_id = config.GRAPH["N_NODES"] - 1
    assert df["src_id"].between(0, max_id).all() and df["tgt_id"].between(0, max_id).all(), (
        f"{event_name} q{snapshot_idx}: node id out of range"
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading tariff lookup...")
    tariff_lookup = load_tariff_lookup()

    for event in config.EVENTS:
        event_name = event["name"]
        print(f"\nBuilding edge features for {event_name} (date {event['date']})...")
        shock_lookup = load_shock_lookup(event_name)
        n_direct_shocks = sum(1 for v in shock_lookup.values() if v != 0.0)
        print(f"  {n_direct_shocks} nonzero shock entries loaded")

        event_year, event_quarter = parse_event_date(event["date"])
        quarters = quarter_sequence(event_year, event_quarter)

        for idx, (year, quarter) in enumerate(quarters):
            df = build_quarter_edge_features(
                event, year, quarter, idx, tariff_lookup, shock_lookup
            )
            validate(df, event_name, idx)
            n_shocked = int((df["e3"] != 0.0).sum())
            out_path = os.path.join(OUT_DIR, f"edge_features_{event_name}_q{idx}.parquet")
            df.to_parquet(out_path, index=False)
            print(f"  q{idx} ({year}Q{quarter}): {len(df):,} edges, "
                  f"{n_shocked} shocked edges -> {os.path.basename(out_path)}")

    print(f"\nDone. Edge features saved to {OUT_DIR}/ (6 events x 8 quarters = 48 files)")


if __name__ == "__main__":
    main()
