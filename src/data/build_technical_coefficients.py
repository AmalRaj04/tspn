"""build_technical_coefficients.py — Construct WIOD technical coefficient matrix A.

For each year 2000-2014, constructs the WIOD technical coefficient matrix A (shape 2464x2464)
and saves it as a SciPy sparse CSR matrix in:
data/processed/technical_coefficients/A_{YEAR}.npz
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz

# ---------------------------------------------------------------------------
# Project root & config import
# ---------------------------------------------------------------------------
_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

# Paths and Constants from config
RAW_EDGES_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_EDGES"])
OUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/technical_coefficients")
WIOD_YEARS = config.GRAPH.get("WIOD_YEARS", list(range(2000, 2015)))
N_NODES = config.GRAPH.get("N_NODES", 2464)
COUNTRY_LIST = config.GRAPH["COUNTRY_LIST"]
SECTOR_LIST = config.GRAPH["SECTOR_LIST"]

# Map countries and sectors to indices
_COUNTRY_IDX = {c: i for i, c in enumerate(COUNTRY_LIST)}
_SECTOR_IDX = {s: i for i, s in enumerate(SECTOR_LIST)}

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)


def build_coefficients_for_year(year: int) -> None:
    # Paths
    edges_path = os.path.join(RAW_EDGES_DIR, f"edges_raw_{year}.parquet")
    sea_path = os.path.join(RAW_EDGES_DIR, f"socioeconomic_{year}.parquet")
    out_path = os.path.join(OUT_DIR, f"A_{year}.npz")

    # Skip year if output already exists
    if os.path.exists(out_path):
        print(f"Skipping {year}: {out_path} already exists.")
        return

    # Check input files exist
    if not os.path.exists(edges_path):
        print(f"Skipping {year}: {edges_path} not found.")
        return
    if not os.path.exists(sea_path):
        print(f"Skipping {year}: {sea_path} not found.")
        return

    # 1. Load edges_raw_{YEAR}.parquet
    df_edges = pd.read_parquet(edges_path, columns=["src_id", "tgt_id", "flow_usd"])

    # 2. Load socioeconomic_{YEAR}.parquet
    df_sea = pd.read_parquet(sea_path, columns=["country", "sector", "gross_output"])

    # Convert country and sector to string to match indexing
    df_sea["country"] = df_sea["country"].astype(str)
    df_sea["sector"] = df_sea["sector"].astype(str)

    # 3. Reconstruct node_id using config node mapping
    df_sea["country_idx"] = df_sea["country"].map(_COUNTRY_IDX)
    df_sea["sector_idx"] = df_sea["sector"].map(_SECTOR_IDX)
    
    # Drop rows that don't match any index in configuration lists
    df_sea = df_sea.dropna(subset=["country_idx", "sector_idx"])
    
    df_sea["node_id"] = df_sea.apply(
        lambda r: config.node_id(int(r["country_idx"]), int(r["sector_idx"])), axis=1
    )

    # 4. Create gross output vector x of length 2464
    x = np.zeros(N_NODES, dtype=np.float64)
    for nid, go in zip(df_sea["node_id"].astype(int), df_sea["gross_output"].astype(float)):
        if 0 <= nid < N_NODES:
            x[nid] = go

    # 5. Build sparse technical coefficient matrix:
    # A[src_id, tgt_id] = flow_usd / gross_output[tgt_id]
    # If gross_output[tgt_id] <= 0: skip edge.
    src_ids = df_edges["src_id"].values
    tgt_ids = df_edges["tgt_id"].values
    flows = df_edges["flow_usd"].values

    # Look up target gross outputs
    tgt_gos = x[tgt_ids]
    valid_mask = tgt_gos > 0.0

    src_filtered = src_ids[valid_mask]
    tgt_filtered = tgt_ids[valid_mask]
    flows_filtered = flows[valid_mask]
    tgt_gos_filtered = tgt_gos[valid_mask]

    coefficients = flows_filtered / tgt_gos_filtered

    # 6. Construct A as scipy.sparse csr_matrix.
    # Construct as COO first, then convert to CSR (sums duplicate entries automatically)
    A_coo = coo_matrix((coefficients, (src_filtered, tgt_filtered)), shape=(N_NODES, N_NODES))
    A = A_coo.tocsr()

    # 7. Save using scipy.sparse.save_npz()
    save_npz(out_path, A)

    # 8. Validation
    nnz = A.nnz
    density = nnz / (N_NODES * N_NODES)
    max_coeff = A.data.max() if nnz > 0 else 0.0
    min_pos_coeff = A.data[A.data > 0].min() if len(A.data[A.data > 0]) > 0 else 0.0

    print(f"Year: {year}")
    print(f"  Nonzero entries: {nnz}")
    print(f"  Density: {density:.6f}")
    print(f"  Max coefficient: {max_coeff:.6f}")
    print(f"  Min positive coefficient: {min_pos_coeff:.6f}")

    # Assertions
    assert A.shape == (N_NODES, N_NODES), f"Shape is {A.shape}, expected ({N_NODES}, {N_NODES})"
    assert not np.isnan(A.data).any(), f"[{year}] NaN detected in coefficients"
    assert not np.isinf(A.data).any(), f"[{year}] Inf detected in coefficients"
    assert max_coeff < 10.0, f"[{year}] Max coefficient {max_coeff:.4f} >= 10"


def main() -> None:
    # Filter years to 2000-2014 only
    years = [y for y in WIOD_YEARS if 2000 <= y <= 2014]
    for year in years:
        try:
            build_coefficients_for_year(year)
        except Exception as e:
            print(f"Error building coefficients for {year}: {e}")
            raise e


if __name__ == "__main__":
    main()
