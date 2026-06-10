"""Compute UK Global Tariff (2021) deltas vs EU CET from official sources.

Compares UK third-country ad valorem duties (Jan 2021) against EU conventional
rates of duty from Commission Implementing Regulation (EU) 2020/1577.

Usage
-----
    python src/data/extract_uk_global_tariff_2021.py

Output
------
    data/raw/tariff_events/uk_global_tariff_2021.csv
"""

from __future__ import annotations

import os
import re
import sys

import pandas as pd
import pdfplumber

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

TARIFF_DIR = os.path.join(PROJECT_ROOT, config.PATHS["RAW_TARIFF_EVENTS"])

UK_MEASURES_CSV = "uk-tariff-2021-01-01--v4.0.1527--measures-on-declarable-commodities.csv"
EU_CET_PDF = "CELEX_32020R1577_EN_TXT.pdf"
OUTPUT_CSV = os.path.join(TARIFF_DIR, "uk_global_tariff_2021.csv")

PURE_PCT_RE = re.compile(r"^\d+(\.\d+)?%$")
CN_LINE_RE = re.compile(r"^(\d{4}\s+\d{2}\s+\d{2})\b")
DOT_LEADER_RE = re.compile(r"\.{2,}\s*(.+)$")

# Explicit percentage at the start of the duty field.
EXPLICIT_PCT_RE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*%")
# Compound ad valorem + specific duty, e.g. "12,8 % + 176,8 €/100 kg" or "10,2 + 93,1 €/".
COMPOUND_AD_VALOREM_RE = re.compile(r"^\(?(\d+(?:[.,]\d+)?)\s*(?:%)?\s*\+")
# Pure ad valorem in the (%) column without a literal percent sign, e.g. "11,5 p/st".
PURE_AD_VALOREM_RE = re.compile(
    r"^(\d+(?:[.,]\d+)?)\s+(?:p/st|p/|kg\b|l\b|m2|m3|—|-\s|$)"
)


def _source_path(filename: str) -> str:
    path = os.path.join(TARIFF_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required source file not found: {path}")
    return path


def normalize_commodity_code(raw_code: object) -> str:
    """Normalize UK commodity code to 10 digits (preserving leading zeros)."""
    code = str(raw_code).strip()
    if code.endswith(".0"):
        code = code[:-2]
    return code.zfill(10)


def commodity_to_cn8(raw_code: object) -> str:
    """Derive 8-digit CN code from a UK commodity code."""
    return normalize_commodity_code(raw_code)[:8]


def parse_pct_duty(expression: str) -> float:
    """Convert a pure UK duty string like '6%' or '12.8%' to float."""
    return float(expression.rstrip("%"))


def extract_uk_rates(measures_path: str) -> pd.DataFrame:
    """STEP 1: UK third-country pure ad valorem duties by CN8."""
    df = pd.read_csv(measures_path, low_memory=False)

    required = {
        "commodity__code",
        "measure__type__description",
        "measure__duty_expression",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"UK measures CSV missing columns: {sorted(missing)}")

    uk = df[df["measure__type__description"] == "Third country duty"].copy()
    uk["duty_expression"] = uk["measure__duty_expression"].astype(str)
    uk = uk[uk["duty_expression"].str.match(PURE_PCT_RE, na=False)].copy()

    uk["cn8"] = uk["commodity__code"].map(commodity_to_cn8)
    uk["uk_rate_pct"] = uk["duty_expression"].map(parse_pct_duty)

    rates = (
        uk.groupby("cn8", as_index=False)["uk_rate_pct"]
        .max()
        .sort_values("cn8", kind="mergesort")
        .reset_index(drop=True)
    )
    return rates


def _parse_eu_duty_component(duty_field: str) -> float | None:
    """Extract the ad valorem percentage from an EU conventional-rate field.

    The PDF column is headed "Conventional rate of duty (%)". Values may appear as:
      - "12,8 %" or "6 %"          (explicit percentage)
      - "12,8 % + 176,8 €/100 kg"  (compound — first component is ad valorem %)
      - "10,2 + 93,1 €/100 kg"     (compound without literal % on first term)
      - "11,5 p/st"                (pure ad valorem in the % column)
      - "Free"                     (zero ad valorem)

    Specific-only duties (e.g. "26,2 €/100 kg") return None and are skipped.
    """
    duty = duty_field.strip()
    if not duty:
        return None

    if re.match(r"^Free\b", duty, re.IGNORECASE):
        return 0.0

    match = EXPLICIT_PCT_RE.match(duty)
    if match:
        return float(match.group(1).replace(",", "."))

    match = COMPOUND_AD_VALOREM_RE.match(duty)
    if match:
        return float(match.group(1).replace(",", "."))

    # Pure ad valorem without €/ specific component.
    if "€" not in duty:
        match = PURE_AD_VALOREM_RE.match(duty)
        if match:
            return float(match.group(1).replace(",", "."))

    return None


def extract_eu_rates(pdf_path: str) -> pd.DataFrame:
    """STEP 2: EU CET conventional ad valorem rates by CN8 from the 2021 CN PDF."""
    rows: list[tuple[str, float]] = []
    seen: set[str] = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                cn_match = CN_LINE_RE.match(line)
                if not cn_match:
                    continue

                duty_match = DOT_LEADER_RE.search(line)
                if not duty_match:
                    continue

                rate = _parse_eu_duty_component(duty_match.group(1))
                if rate is None:
                    continue

                cn8 = cn_match.group(1).replace(" ", "")
                if cn8 in seen:
                    continue

                seen.add(cn8)
                rows.append((cn8, rate))

    if not rows:
        raise RuntimeError("No EU CET rates extracted from PDF.")

    return pd.DataFrame(rows, columns=["cn8", "eu_rate_pct"])


def compute_delta(uk_rates: pd.DataFrame, eu_rates: pd.DataFrame) -> pd.DataFrame:
    """STEP 3: Inner join and compute UK − EU tariff delta."""
    merged = uk_rates.merge(eu_rates, on="cn8", how="inner")
    merged["delta_tariff_pct"] = merged["uk_rate_pct"] - merged["eu_rate_pct"]
    return merged


def build_output(merged: pd.DataFrame) -> pd.DataFrame:
    """STEP 4: Format final output CSV."""
    out = merged.rename(columns={"cn8": "hts_code"})[["hts_code", "delta_tariff_pct"]]
    out["delta_tariff_pct"] = out["delta_tariff_pct"].round(4)
    return out.sort_values("hts_code", kind="mergesort").reset_index(drop=True)


def validate_output(df: pd.DataFrame) -> list[str]:
    """Return product codes that are not exactly 8 digits."""
    return [
        str(code)
        for code in df["hts_code"]
        if not re.fullmatch(r"\d{8}", str(code))
    ]


def print_validation_report(
    uk_rates: pd.DataFrame,
    eu_rates: pd.DataFrame,
    merged: pd.DataFrame,
    output: pd.DataFrame,
    bad_codes: list[str],
) -> None:
    uk_cn8 = set(uk_rates["cn8"])
    eu_cn8 = set(eu_rates["cn8"])
    matched_cn8 = set(merged["cn8"])

    print(f"UK rows: {len(uk_rates)}")
    print(f"EU rows: {len(eu_rates)}")
    print(f"Matched rows: {len(merged)}")
    print(f"Unmatched UK rows: {len(uk_cn8 - eu_cn8)}")
    print(f"Unmatched EU rows: {len(eu_cn8 - uk_cn8)}")
    print()
    print(f"Min delta: {output['delta_tariff_pct'].min():.4f}")
    print(f"Max delta: {output['delta_tariff_pct'].max():.4f}")
    print(f"Mean delta: {output['delta_tariff_pct'].mean():.4f}")
    print()
    print(f"Bad codes count: {len(bad_codes)}")
    if bad_codes:
        print(f"Bad codes: {bad_codes[:20]}")


def main() -> None:
    os.makedirs(TARIFF_DIR, exist_ok=True)

    uk_rates = extract_uk_rates(_source_path(UK_MEASURES_CSV))
    eu_rates = extract_eu_rates(_source_path(EU_CET_PDF))

    merged = compute_delta(uk_rates, eu_rates)
    output = build_output(merged)

    bad_codes = validate_output(output)
    if bad_codes:
        raise RuntimeError(
            f"Validation failed: {len(bad_codes)} invalid hts_code values."
        )

    output.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print()
    print_validation_report(uk_rates, eu_rates, merged, output, bad_codes)


if __name__ == "__main__":
    main()
