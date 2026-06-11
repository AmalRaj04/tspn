"""Build weighted HS6 → WIOD56 concordance.

Pipeline:
    HS 2017  →  CPC 2.1  →  ISIC Rev.4  →  WIOD-56 sector

Inputs
------
data/raw/concordance/CPC21-HS2017.csv
data/raw/concordance/isic4-cpc21.txt
data/raw/concordance/2017_NAICS_to_ISIC_4.xlsx
data/processed/edges/socioeconomic_2014.parquet

Outputs  (data/processed/concordance/)
-------
hs6_to_cpc21.json
cpc21_to_isic4.json
isic4_to_wiod56.json
naics3_to_wiod56.json
hs6_to_wiod56_weights.json
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: locate project root regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402  (project-level config.py)

# ---------------------------------------------------------------------------
# Paths (from config)
# ---------------------------------------------------------------------------
RAW_CONC = ROOT / config.PATHS["RAW_CONCORDANCE"]
PROC_EDGES = ROOT / config.PATHS["PROCESSED_EDGES"]
PROC_CONC = ROOT / config.PATHS["PROCESSED_CONCORDANCE"]
PROC_CONC.mkdir(parents=True, exist_ok=True)

SECTOR_LIST: list[str] = config.GRAPH["SECTOR_LIST"]  # 56 canonical WIOD codes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_sector(raw: str) -> str:
    """Convert parquet sector names to canonical config.SECTOR_LIST format.

    Some WIOD sector codes are stored with hyphens in the parquet
    (e.g. 'C10-C12', 'E37-E39') but config.py uses underscores
    ('C10_C12', 'E37_E39').  Normalise to underscore form so look-ups
    against SECTOR_LIST always succeed.
    """
    return raw.replace("-", "_")


# ---------------------------------------------------------------------------
# STEP 1 — Load HS 2017 → CPC 2.1
# ---------------------------------------------------------------------------

def build_hs6_to_cpc21() -> dict[str, list[str]]:
    """Read CPC21-HS2017.csv and return {hs6: [cpc, ...]}.

    HS codes are normalised to 6-character zero-padded strings (decimal
    points removed).  CPC codes are stored as plain strings.
    """
    csv_path = RAW_CONC / "CPC21-HS2017.csv"
    df = pd.read_csv(csv_path, dtype=str)

    # Identify the two relevant columns
    hs_col = "HS 2017"
    cpc_col = "CPC Ver. 2.1"

    mapping: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        hs_raw = str(row[hs_col]).strip()
        cpc_raw = str(row[cpc_col]).strip()

        if not hs_raw or not cpc_raw or hs_raw == "nan" or cpc_raw == "nan":
            continue

        # Normalise HS: strip decimal point, zero-pad to 6 digits
        hs6 = hs_raw.replace(".", "").zfill(6)

        if cpc_raw not in mapping[hs6]:
            mapping[hs6].append(cpc_raw)

    return dict(mapping)


# ---------------------------------------------------------------------------
# STEP 2 — Load CPC 2.1 → ISIC Rev.4
# ---------------------------------------------------------------------------

def build_cpc21_to_isic4() -> dict[str, list[str]]:
    """Read isic4-cpc21.txt and return {cpc: [isic4, ...]}.

    ISIC codes are stored as 4-character zero-padded strings.
    """
    txt_path = RAW_CONC / "isic4-cpc21.txt"

    # The file has a CSV-like structure with quoted fields
    df = pd.read_csv(txt_path, dtype=str)

    # Strip surrounding whitespace / quotes from column names
    df.columns = [c.strip().strip('"') for c in df.columns]

    isic_col = "ISIC4code"
    cpc_col = "CPC21code"

    mapping: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        cpc_raw = str(row[cpc_col]).strip().strip('"')
        isic_raw = str(row[isic_col]).strip().strip('"')

        if not cpc_raw or not isic_raw or cpc_raw == "nan" or isic_raw == "nan":
            continue

        # Zero-pad ISIC to 4 digits
        isic4 = isic_raw.zfill(4)

        if isic4 not in mapping[cpc_raw]:
            mapping[cpc_raw].append(isic4)

    return dict(mapping)


# ---------------------------------------------------------------------------
# STEP 3 — Build ISIC Rev.4 → WIOD-56
# ---------------------------------------------------------------------------

def _isic_prefix(isic4: str) -> int:
    """Return the integer ISIC division (first two digits)."""
    return int(isic4[:2])


def build_isic4_to_wiod56() -> dict[str, str]:
    """Deterministically map every ISIC-4 code to a WIOD-56 sector.

    Mapping rules are derived from ISIC Rev.4 divisions as specified in
    the project requirements.  Returns {isic4: wiod_sector}.
    """
    # -----------------------------------------------------------------------
    # Division-level rules (first two digits of ISIC code)
    # Single-division → single sector
    # -----------------------------------------------------------------------
    DIVISION_MAP: dict[int, str] = {
        # Agriculture, forestry and fishing
        1: "A01",   # 01xx
        2: "A02",   # 02xx
        3: "A03",   # 03xx
        # Mining and quarrying
        5: "B", 6: "B", 7: "B", 8: "B", 9: "B",  # 05xx–09xx
        # Manufacturing — food/bev/tobacco
        16: "C16",
        17: "C17",
        18: "C18",
        19: "C19",
        20: "C20",
        21: "C21",
        22: "C22",
        23: "C23",
        24: "C24",
        25: "C25",
        26: "C26",
        27: "C27",
        28: "C28",
        29: "C29",
        30: "C30",
        33: "C33",
        # Electricity, gas, steam
        35: "D35",
        # Water supply; sewerage
        36: "E36",
        # Construction
        41: "F", 42: "F", 43: "F",
        # Wholesale / retail trade
        45: "G45",
        46: "G46",
        47: "G47",
        # Transport & storage
        49: "H49",
        50: "H50",
        51: "H51",
        52: "H52",
        53: "H53",
        # Accommodation / food services
        55: "I", 56: "I",
        # Publishing / IT / telecom
        58: "J58",
        61: "J61",
        # Finance and insurance
        64: "K64",
        65: "K65",
        66: "K66",
        # Real estate
        68: "L68",
        # Professional services
        71: "M71",
        72: "M72",
        73: "M73",
        # Public admin & defence
        84: "O84",
        # Education
        85: "P85",
        # Activities of households
        97: "T",
        # Extraterritorial organisations
        99: "U",
    }

    # Multi-division groups (ranges → single sector)
    RANGE_MAP: list[tuple[int, int, str]] = [
        (10, 12, "C10_C12"),
        (13, 15, "C13_C15"),
        (31, 32, "C31_C32"),
        (37, 39, "E37_E39"),
        (59, 60, "J59_J60"),
        (62, 63, "J62_J63"),
        (69, 70, "M69_M70"),
        (74, 75, "M74_M75"),
        (77, 82, "N"),
        (86, 88, "Q"),
        (90, 96, "R_S"),
    ]

    # Build a full look-up table for every possible two-digit ISIC division
    FULL_MAP: dict[int, str] = {}
    for lo, hi, sector in RANGE_MAP:
        for d in range(lo, hi + 1):
            FULL_MAP[d] = sector
    FULL_MAP.update(DIVISION_MAP)

    # Now iterate over all possible ISIC4 codes that actually appear in the
    # CPC→ISIC mapping and assign WIOD sectors.
    # We build the mapping lazily from codes that will be needed downstream,
    # but for the saved JSON we also include all codes we encounter in the
    # CPC→ISIC file to maximise coverage.
    mapping: dict[str, str] = {}

    # Enumerate every 4-digit code from 0000–9999 and map those we know
    for div in FULL_MAP:
        sector = FULL_MAP[div]
        for unit in range(100):   # 00–99 third/fourth digit combinations
            isic4 = f"{div:02d}{unit:02d}"
            mapping[isic4] = sector

    return mapping


# ---------------------------------------------------------------------------
# STEP 4 — Load WIOD gross output and aggregate
# ---------------------------------------------------------------------------

def build_sector_output() -> dict[str, float]:
    """Sum gross_output across all countries by WIOD sector.

    Returns {wiod_sector: total_output}.

    Parquet sector names may use hyphens ('C10-C12'); normalise to
    the underscore form used in config.SECTOR_LIST.
    """
    parquet_path = PROC_EDGES / "socioeconomic_2014.parquet"
    df = pd.read_parquet(parquet_path, columns=["sector", "gross_output"])

    df["sector"] = df["sector"].astype(str).str.replace("-", "_", regex=False)
    agg = df.groupby("sector")["gross_output"].sum()

    return {s: float(v) for s, v in agg.items()}


# ---------------------------------------------------------------------------
# STEP 5 — Build HS6 → WIOD-56 weighted mapping
# ---------------------------------------------------------------------------

def build_hs6_to_wiod56_weights(
    hs6_to_cpc21: dict[str, list[str]],
    cpc21_to_isic4: dict[str, list[str]],
    isic4_to_wiod56: dict[str, str],
    sector_output: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute output-weighted HS6 → WIOD-56 probability vectors.

    For each HS6 code:
        HS6 → {CPC} → {ISIC4} → {WIOD sector}

    Weight of each reachable WIOD sector =
        gross_output(sector) / Σ gross_output(reachable sectors)

    When only one sector is reachable the weight is set to 1.0.
    """
    result: dict[str, dict[str, float]] = {}

    for hs6, cpc_list in hs6_to_cpc21.items():
        # Collect all reachable WIOD sectors
        sector_outputs: dict[str, float] = {}

        for cpc in cpc_list:
            isic_list = cpc21_to_isic4.get(cpc, [])
            for isic4 in isic_list:
                wiod = isic4_to_wiod56.get(isic4)
                if wiod is None:
                    continue
                if wiod not in sector_outputs:
                    # Use sector output as the weight base; fall back to 1.0
                    sector_outputs[wiod] = sector_output.get(wiod, 1.0)

        if not sector_outputs:
            # Unmappable HS6 — skip (no entry produced)
            continue

        total = sum(sector_outputs.values())
        if total == 0.0:
            # Guard against all-zero outputs (shouldn't happen, but be safe)
            n = len(sector_outputs)
            weights = {s: 1.0 / n for s in sector_outputs}
        elif len(sector_outputs) == 1:
            (sector,) = sector_outputs
            weights = {sector: 1.0}
        else:
            weights = {s: v / total for s, v in sector_outputs.items()}

        result[hs6] = weights

    return result


# ---------------------------------------------------------------------------
# STEP 6 — NAICS 3-digit → WIOD-56 mapping
# ---------------------------------------------------------------------------

def build_naics3_to_wiod56(
    isic4_to_wiod56: dict[str, str],
) -> dict[str, str]:
    """Map NAICS 3-digit codes to WIOD-56 sectors via ISIC 4.

    Strategy: for each NAICS 6-digit code take the first valid ISIC mapping,
    derive the WIOD sector, then aggregate to the 3-digit NAICS prefix using
    first-valid-wins.
    """
    xlsx_path = RAW_CONC / "2017_NAICS_to_ISIC_4.xlsx"
    df = pd.read_excel(xlsx_path, dtype=str)

    # The NAICS column has embedded newlines in its header; identify it by
    # checking which column actually contains 6-digit numeric codes.
    # The ISIC column is labelled "ISIC 4.0".
    col_naics: str | None = None
    for col in df.columns:
        sample = df[col].dropna().head(20)
        numeric_6 = sample.apply(lambda x: str(x).strip().isdigit() and len(str(x).strip()) >= 6)
        if numeric_6.any():
            col_naics = col
            break
    if col_naics is None:
        # Fallback: take the column whose name contains "NAICS" and has
        # no "TITLE" / "Part of" in it, using the one with newlines first.
        candidates = [c for c in df.columns if "NAICS" in c and "TITLE" not in c and "Part" not in c]
        col_naics = candidates[0] if candidates else df.columns[1]

    col_isic = next((c for c in df.columns if "ISIC 4" in c and "Revision" not in c), None)
    if col_isic is None:
        col_isic = next(c for c in df.columns if "ISIC" in c and "Revision" not in c and "Title" not in c)

    mapping: dict[str, str] = {}   # naics3 → wiod sector

    for _, row in df.iterrows():
        naics_raw = str(row[col_naics]).strip()
        isic_raw = str(row[col_isic]).strip()

        # Skip header-like rows, non-numeric NAICS, or missing values
        if naics_raw in ("nan", "", "0") or isic_raw in ("nan", ""):
            continue
        if not naics_raw.isdigit():
            continue

        naics6 = naics_raw.zfill(6)
        naics3 = naics6[:3]

        # Already have a mapping for this 3-digit code — skip (first-valid-wins)
        if naics3 in mapping:
            continue

        # Zero-pad ISIC to 4 digits
        try:
            isic4 = str(int(float(isic_raw))).zfill(4)
        except (ValueError, OverflowError):
            continue

        wiod = isic4_to_wiod56.get(isic4)
        if wiod is None:
            continue

        mapping[naics3] = wiod

    return mapping


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    hs6_to_cpc21: dict[str, list[str]],
    cpc21_to_isic4: dict[str, list[str]],
    isic4_to_wiod56: dict[str, str],
    hs6_to_wiod56_weights: dict[str, dict[str, float]],
) -> None:
    """Run validation checks and print a summary report."""

    print("\n" + "=" * 60)
    print("CONCORDANCE VALIDATION REPORT")
    print("=" * 60)

    # 1. Total HS6 codes mapped
    n_hs6 = len(hs6_to_wiod56_weights)
    print(f"[1] Total HS6 codes mapped           : {n_hs6:,}")

    # 2. Total CPC codes mapped
    n_cpc = len(cpc21_to_isic4)
    print(f"[2] Total CPC codes with ISIC mapping : {n_cpc:,}")

    # 3. Total ISIC codes mapped (distinct codes appearing as keys in isic4_to_wiod56
    #    that are actually reachable via HS6→CPC→ISIC)
    all_isic_reachable: set[str] = set()
    for cpc_list in hs6_to_cpc21.values():
        for cpc in cpc_list:
            all_isic_reachable.update(cpc21_to_isic4.get(cpc, []))
    mapped_isic = {i for i in all_isic_reachable if i in isic4_to_wiod56}
    print(f"[3] Total reachable ISIC codes mapped : {len(mapped_isic):,}")

    # 4. Total distinct WIOD sectors reached
    wiod_sectors_reached: set[str] = set()
    for weights in hs6_to_wiod56_weights.values():
        wiod_sectors_reached.update(weights.keys())
    print(f"[4] Total WIOD-56 sectors reached     : {len(wiod_sectors_reached):,}")

    # 5. Average WIOD sectors per HS6
    avg_sectors = (
        sum(len(w) for w in hs6_to_wiod56_weights.values()) / n_hs6
        if n_hs6 > 0
        else 0.0
    )
    print(f"[5] Average WIOD sectors per HS6     : {avg_sectors:.3f}")

    # 6. Verify weight vectors sum to 1.0 ± 1e-6
    bad_hs6 = [
        hs6
        for hs6, w in hs6_to_wiod56_weights.items()
        if abs(sum(w.values()) - 1.0) > 1e-6
    ]
    if bad_hs6:
        print(f"[6] ❌ Weight-sum check FAILED for {len(bad_hs6)} HS6 codes!")
        for hs6 in bad_hs6[:5]:
            print(f"       {hs6}: sum = {sum(hs6_to_wiod56_weights[hs6].values()):.8f}")
    else:
        print(f"[6] ✅ All {n_hs6:,} HS6 weight vectors sum to 1.0 ± 1e-6")

    # 7. Verify every output WIOD sector is in config.SECTOR_LIST
    unknown_sectors = wiod_sectors_reached - set(SECTOR_LIST)
    if unknown_sectors:
        print(f"[7] ❌ Unknown WIOD sectors found: {sorted(unknown_sectors)}")
    else:
        print(f"[7] ✅ All WIOD sectors are valid members of config.SECTOR_LIST")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building HS6 → WIOD-56 concordance …\n")

    # ------------------------------------------------------------------
    # Step 1 — HS6 → CPC 2.1
    # ------------------------------------------------------------------
    print("STEP 1  Reading HS 2017 → CPC 2.1 …")
    hs6_to_cpc21 = build_hs6_to_cpc21()
    print(f"        {len(hs6_to_cpc21):,} HS6 codes loaded.")

    out_path = PROC_CONC / "hs6_to_cpc21.json"
    with open(out_path, "w") as fh:
        json.dump(hs6_to_cpc21, fh, indent=2)
    print(f"        Saved → {out_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Step 2 — CPC 2.1 → ISIC Rev.4
    # ------------------------------------------------------------------
    print("\nSTEP 2  Reading CPC 2.1 → ISIC Rev.4 …")
    cpc21_to_isic4 = build_cpc21_to_isic4()
    print(f"        {len(cpc21_to_isic4):,} CPC codes loaded.")

    out_path = PROC_CONC / "cpc21_to_isic4.json"
    with open(out_path, "w") as fh:
        json.dump(cpc21_to_isic4, fh, indent=2)
    print(f"        Saved → {out_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Step 3 — ISIC Rev.4 → WIOD-56
    # ------------------------------------------------------------------
    print("\nSTEP 3  Building deterministic ISIC Rev.4 → WIOD-56 map …")
    isic4_to_wiod56 = build_isic4_to_wiod56()
    print(f"        {len(isic4_to_wiod56):,} ISIC-4 codes mapped.")

    out_path = PROC_CONC / "isic4_to_wiod56.json"
    with open(out_path, "w") as fh:
        json.dump(isic4_to_wiod56, fh, indent=2)
    print(f"        Saved → {out_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Step 4 — WIOD gross output aggregation
    # ------------------------------------------------------------------
    print("\nSTEP 4  Aggregating WIOD gross output by sector …")
    sector_output = build_sector_output()
    print(f"        {len(sector_output):,} sectors with output data.")
    total_go = sum(sector_output.values())
    print(f"        Total global gross output : {total_go:,.1f} (million USD)")

    # ------------------------------------------------------------------
    # Step 5 — HS6 → WIOD-56 weighted mapping
    # ------------------------------------------------------------------
    print("\nSTEP 5  Computing HS6 → WIOD-56 weighted mapping …")
    hs6_to_wiod56_weights = build_hs6_to_wiod56_weights(
        hs6_to_cpc21, cpc21_to_isic4, isic4_to_wiod56, sector_output
    )
    print(f"        {len(hs6_to_wiod56_weights):,} HS6 codes successfully mapped.")

    out_path = PROC_CONC / "hs6_to_wiod56_weights.json"
    with open(out_path, "w") as fh:
        json.dump(hs6_to_wiod56_weights, fh, indent=2)
    print(f"        Saved → {out_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Step 6 — NAICS 3-digit → WIOD-56
    # ------------------------------------------------------------------
    print("\nSTEP 6  Building NAICS 3-digit → WIOD-56 map …")
    naics3_to_wiod56 = build_naics3_to_wiod56(isic4_to_wiod56)
    print(f"        {len(naics3_to_wiod56):,} NAICS-3 codes mapped.")

    out_path = PROC_CONC / "naics3_to_wiod56.json"
    with open(out_path, "w") as fh:
        json.dump(naics3_to_wiod56, fh, indent=2)
    print(f"        Saved → {out_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    validate(hs6_to_cpc21, cpc21_to_isic4, isic4_to_wiod56, hs6_to_wiod56_weights)

    print("Done.  All concordance files written to:")
    print(f"  {PROC_CONC.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
