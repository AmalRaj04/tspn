"""compute_leontief.py — Compute Leontief Inverse & Backward Linkages.

For each year in WIOD_YEARS (2000-2014), loads technical coefficient matrix A,
computes the Leontief inverse L = (I - A + REG_EPS * I)^-1, and saves both L
and column sums (backward linkages) to:
data/processed/leontief/leontief_{YEAR}.npy
data/processed/leontief/backward_linkage_{YEAR}.npy
"""

import os
import sys
import gc
import numpy as np
from scipy.sparse import load_npz

# ---------------------------------------------------------------------------
# Test-mode toggle
# ---------------------------------------------------------------------------
TEST_MODE = False                          # ← set True to run one year only
TEST_FILE = "A_2014.npz"                   # ← placeholder: change as needed

# ---------------------------------------------------------------------------
# Project root & config import
# ---------------------------------------------------------------------------
_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

# Constants from config
WIOD_YEARS = config.GRAPH.get("WIOD_YEARS", list(range(2000, 2015)))
N_NODES = config.GRAPH.get("N_NODES", 2464)
REG_EPS = config.LEONTIEF["REG_EPS"]

INPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/technical_coefficients")
OUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/leontief")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)


def compute_leontief_for_year(year: int) -> None:
    # Output paths
    leontief_path = os.path.join(OUT_DIR, f"leontief_{year}.npy")
    bl_path = os.path.join(OUT_DIR, f"backward_linkage_{year}.npy")

    # 9. Skip year if output already exists
    if os.path.exists(leontief_path) and os.path.exists(bl_path):
        print(f"Skipping {year}: {leontief_path} and {bl_path} already exist.")
        return

    # 1. Load data/processed/technical_coefficients/A_{YEAR}.npz
    coeff_path = os.path.join(INPUT_DIR, f"A_{year}.npz")
    if not os.path.exists(coeff_path):
        print(f"Skipping {year}: {coeff_path} not found.")
        return

    A_sparse = load_npz(coeff_path)

    # 2. Convert sparse matrix A to dense float64
    A = A_sparse.toarray().astype(np.float64)

    # 3. Build M = I - A + REG_EPS * I
    I = np.eye(N_NODES, dtype=np.float64)
    M = (1.0 + REG_EPS) * I - A

    # 4. Compute condition number
    cond = np.linalg.cond(M)
    if cond > 1e6:
        print(f"Warning (Year {year}): condition number is large ({cond:.2e}) but continuing.")

    # 5. Compute L = np.linalg.inv(M)
    L = np.linalg.inv(M)

    # 6. Validation
    # Assertions
    assert L.shape == (N_NODES, N_NODES), f"Shape is {L.shape}, expected ({N_NODES}, {N_NODES})"
    assert not np.isnan(L).any(), f"[{year}] NaN detected in Leontief inverse"
    assert not np.isinf(L).any(), f"[{year}] Inf detected in Leontief inverse"

    max_val = np.max(np.abs(L))
    mean_val = np.mean(np.abs(L))

    # Print validation information
    print(f"Validation for {year}:")
    print(f"  max(abs(L)): {max_val:.6f}")
    print(f"  mean(abs(L)): {mean_val:.6f}")
    print(f"  condition number: {cond:.2e}")

    if max_val > 100:
        print(f"Warning (Year {year}): max(abs(L)) is large ({max_val:.2f}).")

    # 7. Compute backward linkage: bl = L.sum(axis=0)
    bl = L.sum(axis=0)

    # 8. Save L and bl as float32
    np.save(leontief_path, L.astype(np.float32))
    np.save(bl_path, bl.astype(np.float32))

    # Output required print statement:
    print(f"Year {year}: L computed, max={max_val:.2f}, condition={cond:.2e}")

    # 10. Free memory after each year
    del A_sparse
    del A
    del M
    del L
    gc.collect()


def main() -> None:
    # Filter years to 2000-2014 only
    wiod_years = [y for y in WIOD_YEARS if 2000 <= y <= 2014]

    if TEST_MODE:
        try:
            # Extract year from A_YYYY.npz
            test_year = int(TEST_FILE.split("_")[1].split(".")[0])
            years_to_run = [test_year]
        except Exception:
            years_to_run = [2014]
    else:
        years_to_run = wiod_years

    for year in years_to_run:
        try:
            compute_leontief_for_year(year)
        except Exception as e:
            print(f"Error computing Leontief for {year}: {e}")
            raise e


if __name__ == "__main__":
    main()
