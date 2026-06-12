"""Extend the WIOD production network from 2014 to 2017-2021 using UN Comtrade.

Design
------
WIOD edges only run to 2014.  Comtrade files give total-import-by-HS2 for each
reporter country (no bilateral partner breakdown — the download used
partner_code=None / "all").  We therefore cannot reconstruct bilateral flows
from Comtrade alone.

Strategy: WIOD-2014 structural prior + Comtrade scaling
  1. Use edges_2014.parquet as the fixed structural template (topology, src_ids,
     tgt_ids, domestic flows unchanged).
  2. Build a Comtrade import-share vector per (tgt_country, tgt_sector, year)
     from the HS2-aggregate Comtrade files.
  3. Scale each bilateral WIOD-2014 flow by:
         scale = comtrade_total_import(tgt, sector, year)
                 / wiod_total_import(tgt, sector, 2014)
     preserving WIOD bilateral structure while refreshing import magnitudes.
  4. Domestic (src==tgt) flows are left at WIOD-2014 values (no Comtrade equiv).
  5. Recompute import_pen_coeff using scaled flows and WIOD-2014 denominator.
  6. Apply the edge threshold from config.GRAPH["EDGE_THRESHOLD"].

HS2 → WIOD sector mapping is used to allocate HS2-level Comtrade imports to
WIOD sectors.  Trade value is split equally across all sectors in the mapping.

Output schema (identical to existing WIOD edge tables)
------------------------------------------------------
    year              int16
    src_country       category
    src_sector        category
    tgt_country       category
    tgt_sector        category
    flow_usd          float32
    import_pen_coeff  float32
    src_id            int16
    tgt_id            int16

Note: the spec requested columns src_country/tgt_country/sector/src_id/tgt_id
but the actual WIOD schema — which ALL downstream scripts depend on — uses
src_sector/tgt_sector.  We match the real schema exactly.

Output files
------------
    data/processed/edges/edges_{year}.parquet   for year in 2017..2021
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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
COUNTRY_LIST: list[str] = config.GRAPH["COUNTRY_LIST"]   # 44 countries
SECTOR_LIST: list[str] = config.GRAPH["SECTOR_LIST"]     # 56 WIOD sectors
EDGE_THRESHOLD: float = config.GRAPH["EDGE_THRESHOLD"]   # 0.001
TARGET_YEARS: list[int] = config.GRAPH["COMTRADE_YEARS"] # [2017..2021]

COUNTRY_IDX: dict[str, int] = {c: i for i, c in enumerate(COUNTRY_LIST)}
SECTOR_IDX: dict[str, int] = {s: i for i, s in enumerate(SECTOR_LIST)}

RAW_COMTRADE = ROOT / config.PATHS["RAW_COMTRADE"]
PROC_EDGES = ROOT / config.PATHS["PROCESSED_EDGES"]

# ---------------------------------------------------------------------------
# HS2 → WIOD sector mapping
# ---------------------------------------------------------------------------
# Each HS2 chapter range maps to one or more WIOD-56 sectors.
# Trade value is split equally among mapped sectors.
# The mapping covers only goods-producing sectors (as in Comtrade HS data).

_HS2_TO_SECTORS_RAW: list[tuple[range, list[str]]] = [
    (range(1,  25),  ["A01", "A03", "C10_C12"]),
    (range(25, 28),  ["B", "C19"]),
    (range(28, 39),  ["C20", "C21"]),
    (range(39, 41),  ["C22"]),
    (range(41, 72),  ["C13_C15", "C16", "C17"]),
    (range(72, 84),  ["C24", "C25"]),
    (range(84, 86),  ["C26", "C27", "C28"]),
    (range(86, 90),  ["C29", "C30"]),
]

# Build hs2 (int) → [(sector, share)] for fast look-up
HS2_SECTOR_MAP: dict[int, list[tuple[str, float]]] = {}
for rng, sectors in _HS2_TO_SECTORS_RAW:
    share = 1.0 / len(sectors)
    for hs2 in rng:
        HS2_SECTOR_MAP[hs2] = [(s, share) for s in sectors]

# ---------------------------------------------------------------------------
# Node-ID helpers  (LOCKED: same as rest of project)
# ---------------------------------------------------------------------------

def node_id(country: str, sector: str) -> int:
    return COUNTRY_IDX[country] * 56 + SECTOR_IDX[sector]


# ---------------------------------------------------------------------------
# Step 1: Load WIOD 2014 structural prior
# ---------------------------------------------------------------------------

def load_wiod_prior() -> pd.DataFrame:
    """Load edges_2014 and return as the structural template."""
    path = PROC_EDGES / "edges_2014.parquet"
    df = pd.read_parquet(path)
    # Ensure string types for merge operations
    df["src_country"] = df["src_country"].astype(str)
    df["tgt_country"] = df["tgt_country"].astype(str)
    df["src_sector"]  = df["src_sector"].astype(str)
    df["tgt_sector"]  = df["tgt_sector"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Step 2: Comtrade total-import aggregation per (tgt_country, sector)
# ---------------------------------------------------------------------------

def _load_comtrade_year(year: int) -> dict[str, dict[str, float]]:
    """Return {tgt_country: {sector: total_import_usd}} for one year.

    Reads all comtrade_{ISO3}_{year}.parquet files and distributes HS2-level
    trade values across sectors using HS2_SECTOR_MAP.
    """
    result: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    country_set = set(COUNTRY_LIST)
    n_files = 0

    for country in COUNTRY_LIST:
        fp = RAW_COMTRADE / f"comtrade_{country}_{year}.parquet"
        if not fp.exists():
            continue

        df = pd.read_parquet(fp)
        if df.empty:
            continue

        df = df.dropna(subset=["trade_value_usd"])
        df = df[df["trade_value_usd"] > 0]
        if df.empty:
            continue

        for _, row in df.iterrows():
            try:
                hs2 = int(str(row["commodity_code"]).strip().lstrip("0") or "0")
            except (ValueError, TypeError):
                continue

            mapping = HS2_SECTOR_MAP.get(hs2)
            if mapping is None:
                continue

            val = float(row["trade_value_usd"])
            for sector, share in mapping:
                result[country][sector] += val * share

        n_files += 1

    return dict(result), n_files  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Step 3: WIOD-2014 total bilateral imports per (tgt_country, tgt_sector)
# ---------------------------------------------------------------------------

def compute_wiod_import_totals(wiod: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Sum bilateral (src!=tgt) flow_usd per (tgt_country, tgt_sector)."""
    bilateral = wiod[wiod["src_country"] != wiod["tgt_country"]]
    totals: dict[tuple[str, str], float] = (
        bilateral.groupby(["tgt_country", "tgt_sector"])["flow_usd"]
        .sum()
        .to_dict()
    )
    return totals


# ---------------------------------------------------------------------------
# Step 4: Scale flows and build year edge table
# ---------------------------------------------------------------------------

def build_year_edges(
    year: int,
    wiod: pd.DataFrame,
    wiod_import_totals: dict[tuple[str, str], float],
    comtrade_totals: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Build the scaled edge table for one year.

    For bilateral edges:
        scale = comtrade_total(tgt, sector, year) / wiod_total(tgt, sector, 2014)
        new_flow = wiod_flow * scale

    Domestic edges keep WIOD-2014 values unchanged.
    """
    rows = wiod.copy()

    # Determine which rows are bilateral
    is_bilateral = rows["src_country"] != rows["tgt_country"]

    # Compute scale factors per (tgt_country, tgt_sector)
    def scale_for(tgt_country: str, tgt_sector: str) -> float:
        ct_val = comtrade_totals.get(tgt_country, {}).get(tgt_sector, None)
        if ct_val is None or ct_val == 0.0:
            return 1.0   # no Comtrade data → keep WIOD-2014 flow unchanged
        wiod_val = wiod_import_totals.get((tgt_country, tgt_sector), 0.0)
        if wiod_val == 0.0:
            return 1.0
        return float(ct_val) / float(wiod_val)

    # Vectorised scale computation for bilateral rows
    bilateral_idx = rows.index[is_bilateral]
    if len(bilateral_idx) > 0:
        bil = rows.loc[bilateral_idx, ["tgt_country", "tgt_sector", "flow_usd"]].copy()
        bil["_scale"] = [
            scale_for(r["tgt_country"], r["tgt_sector"])
            for _, r in bil.iterrows()
        ]
        rows.loc[bilateral_idx, "flow_usd"] = (
            bil["flow_usd"].values * bil["_scale"].values
        ).astype("float32")

    # ── Recompute import_pen_coeff ──────────────────────────────────────────
    # Denominator: total incoming flow per (tgt_country, tgt_sector) in 2014.
    # We add all flows (incl. domestic) as the denominator proxy, consistent
    # with WIOD methodology.
    denom = (
        rows.groupby(["tgt_country", "tgt_sector"])["flow_usd"]
        .transform("sum")
    ).values + 1e-9

    rows["import_pen_coeff"] = (rows["flow_usd"].values / denom).astype("float32")

    # ── Filter by edge threshold ────────────────────────────────────────────
    rows = rows[rows["import_pen_coeff"] >= EDGE_THRESHOLD].copy()

    # ── Update year field ───────────────────────────────────────────────────
    rows["year"] = np.int16(year)

    # ── Recompute node IDs (locked formula) ────────────────────────────────
    rows["src_id"] = [
        node_id(c, s)
        for c, s in zip(rows["src_country"], rows["src_sector"])
    ]
    rows["tgt_id"] = [
        node_id(c, s)
        for c, s in zip(rows["tgt_country"], rows["tgt_sector"])
    ]
    rows["src_id"] = rows["src_id"].astype("int16")
    rows["tgt_id"] = rows["tgt_id"].astype("int16")

    # ── Cast dtypes ─────────────────────────────────────────────────────────
    rows["src_country"] = rows["src_country"].astype("category")
    rows["tgt_country"] = rows["tgt_country"].astype("category")
    rows["src_sector"]  = rows["src_sector"].astype("category")
    rows["tgt_sector"]  = rows["tgt_sector"].astype("category")
    rows["flow_usd"]    = rows["flow_usd"].astype("float32")
    rows["import_pen_coeff"] = rows["import_pen_coeff"].astype("float32")

    # ── Sort ────────────────────────────────────────────────────────────────
    rows = rows.sort_values(["src_id", "tgt_id"]).reset_index(drop=True)

    # ── Ensure correct column order ─────────────────────────────────────────
    cols = [
        "year", "src_country", "src_sector",
        "tgt_country", "tgt_sector",
        "flow_usd", "import_pen_coeff",
        "src_id", "tgt_id",
    ]
    return rows[cols]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, year: int) -> None:
    """Run all required assertions on the year edge table."""
    assert len(df) > 10_000, (
        f"[{year}] Too few edges: {len(df)}"
    )
    assert float(df["flow_usd"].sum()) > 0, (
        f"[{year}] Total flow_usd is zero"
    )
    assert df["src_country"].nunique() >= 40, (
        f"[{year}] Only {df['src_country'].nunique()} src_countries"
    )
    assert df["tgt_country"].nunique() >= 40, (
        f"[{year}] Only {df['tgt_country'].nunique()} tgt_countries"
    )
    assert float(df["import_pen_coeff"].min()) >= 0, (
        f"[{year}] Negative import_pen_coeff"
    )
    assert df["src_id"].between(0, 2463).all(), (
        f"[{year}] src_id out of [0, 2463]"
    )
    assert df["tgt_id"].between(0, 2463).all(), (
        f"[{year}] tgt_id out of [0, 2463]"
    )
    assert not df.isna().any().any(), (
        f"[{year}] NaN values detected"
    )
    all_sectors = set(df["src_sector"].unique()) | set(df["tgt_sector"].unique())
    assert all_sectors.issubset(set(SECTOR_LIST)), (
        f"[{year}] Unknown sectors: {all_sectors - set(SECTOR_LIST)}"
    )
    print(f"  ✅ All 9 validation checks passed for {year}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Extending WIOD to 2017-2021 using Comtrade …\n")

    # Load structural prior once
    print("Loading WIOD 2014 structural prior …")
    wiod = load_wiod_prior()
    print(f"  edges_2014: {len(wiod):,} edges, {wiod['src_country'].nunique()} countries\n")

    wiod_import_totals = compute_wiod_import_totals(wiod)

    total_edges = 0
    total_files = 0
    year_results: list[dict] = []

    for year in TARGET_YEARS:
        print(f"Processing year {year} …")

        comtrade_totals, n_files = _load_comtrade_year(year)
        total_files += n_files
        print(f"  Loaded {n_files} Comtrade files.")

        df = build_year_edges(year, wiod, wiod_import_totals, comtrade_totals)

        # Validate
        validate(df, year)

        # Reporting
        n_edges = len(df)
        total_edges += n_edges
        n_src_ctry = df["src_country"].nunique()
        n_tgt_ctry = df["tgt_country"].nunique()
        all_sectors = sorted(
            set(df["src_sector"].unique()) | set(df["tgt_sector"].unique())
        )
        total_flow = float(df["flow_usd"].sum())
        mean_ipc = float(df["import_pen_coeff"].mean())

        print(f"  Extended {year}: {n_edges:,} edges")
        print(f"    src countries : {n_src_ctry}")
        print(f"    tgt countries : {n_tgt_ctry}")
        print(f"    sectors       : {len(all_sectors)}")
        print(f"    total flow_usd: {total_flow:,.0f} M USD")
        print(f"    mean imp_pen  : {mean_ipc:.4f}")

        # Save
        out_path = PROC_EDGES / f"edges_{year}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved → {out_path.relative_to(ROOT)}\n")

        year_results.append({"year": year, "n_edges": n_edges})

    # Final summary
    print("=" * 50)
    print("COMTRADE EXTENSION SUMMARY")
    print("=" * 50)
    print(f"Years processed : {TARGET_YEARS[0]}-{TARGET_YEARS[-1]}")
    print(f"Files loaded    : {total_files}")
    print(f"Total edges     : {total_edges:,}")
    print(f"Node universe   : {config.GRAPH['N_NODES']}")
    print("Validation      : PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
