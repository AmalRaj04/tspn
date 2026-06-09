"""Extract EU Section 232 retaliation tariffs from Regulation (EU) 2018/886.

Parses CN 2018 codes and additional duty rates from Annex I and Annex II of
Commission Implementing Regulation (EU) 2018/886 (CELEX 32018R0886).

Usage
-----
    python src/data/extract_eu_retaliation_2018.py

Output
------
    data/raw/tariff_events/eu_retaliation_2018.csv
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

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
SOURCE_PDF = "CELEX_32018R0886_EN_TXT.pdf"
OUTPUT_CSV = os.path.join(TARIFF_DIR, "eu_retaliation_2018.csv")

ANNEX_I_EFFECTIVE = "2018-06-22"
ANNEX_II_EFFECTIVE = "2021-06-01"

# Annex table rows: "0710 40 00 25 %" (CN 2018 code + additional duty).
CN_DUTY_LINE_RE = re.compile(
    r"^(\d{4}\s+\d{2}\s+\d{2})\s+(\d+(?:\.\d+)?)\s*%\s*$"
)

ANNEX_I_HEADER_RE = re.compile(r"^ANNEX I\s*$", re.MULTILINE)
ANNEX_II_HEADER_RE = re.compile(r"^ANNEX II\s*$", re.MULTILINE)

# Footnote at the end of each annex table — stop parsing before this marker.
FOOTNOTE_START_RE = re.compile(
    r"^\(1\)\s+The nomenclature codes are taken from the Combined Nomenclature",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass(frozen=True)
class TariffRow:
    product_code: str
    delta_tariff_pct: float
    annex: str
    effective_date: str


def _pdf_path() -> str:
    path = os.path.join(TARIFF_DIR, SOURCE_PDF)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required source PDF not found: {path}")
    return path


def normalize_cn_code(spaced_code: str) -> str:
    """Convert '0710 40 00' → '07104000'."""
    return spaced_code.replace(" ", "")


def _effective_date_for_annex(annex: str) -> str:
    if annex == "I":
        return ANNEX_I_EFFECTIVE
    if annex == "II":
        return ANNEX_II_EFFECTIVE
    raise ValueError(f"Unknown annex: {annex!r}")


def _trim_before_footnote(text: str) -> str:
    """Exclude annex footnote prose that follows the tariff table."""
    match = FOOTNOTE_START_RE.search(text)
    if match:
        return text[: match.start()]
    return text


def extract_rows_from_annex_text(text: str, annex: str) -> list[TariffRow]:
    """Extract CN code / duty pairs from annex table text."""
    effective_date = _effective_date_for_annex(annex)
    table_text = _trim_before_footnote(text)
    rows: list[TariffRow] = []

    for line in table_text.splitlines():
        line = line.strip()
        if not line or line.startswith("CN 2018"):
            continue

        match = CN_DUTY_LINE_RE.match(line)
        if not match:
            continue

        product_code = normalize_cn_code(match.group(1))
        duty = float(match.group(2))
        rows.append(
            TariffRow(
                product_code=product_code,
                delta_tariff_pct=duty,
                annex=annex,
                effective_date=effective_date,
            )
        )

    return rows


def extract_from_pdf(pdf_path: str) -> list[TariffRow]:
    """Walk the PDF, switching annex context at ANNEX I / ANNEX II headers."""
    all_rows: list[TariffRow] = []
    current_annex: str | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""

            if ANNEX_I_HEADER_RE.search(text):
                current_annex = "I"
            elif ANNEX_II_HEADER_RE.search(text):
                current_annex = "II"

            if current_annex is None:
                continue

            all_rows.extend(extract_rows_from_annex_text(text, current_annex))

    if not all_rows:
        raise RuntimeError("No tariff rows extracted from PDF annexes.")

    return all_rows


def rows_to_dataframe(rows: list[TariffRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_code": row.product_code,
                "delta_tariff_pct": row.delta_tariff_pct,
                "annex": row.annex,
                "effective_date": row.effective_date,
            }
            for row in rows
        ],
        columns=["product_code", "delta_tariff_pct", "annex", "effective_date"],
    )


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (same code, duty, annex, and effective date)."""
    return df.drop_duplicates(
        subset=["product_code", "delta_tariff_pct", "annex", "effective_date"],
        keep="first",
    ).reset_index(drop=True)


def validate_dataframe(df: pd.DataFrame) -> tuple[list[str], int]:
    """Return invalid product codes and count of rows with missing duty."""
    invalid_codes = [
        str(code)
        for code in df["product_code"]
        if not re.fullmatch(r"\d{8}", str(code))
    ]
    missing_duty = int(df["delta_tariff_pct"].isna().sum())
    return invalid_codes, missing_duty


def print_validation_report(
    raw_df: pd.DataFrame,
    final_df: pd.DataFrame,
    invalid_codes: list[str],
    missing_duty: int,
) -> None:
    exact_dupes = len(raw_df) - len(
        raw_df.drop_duplicates(
            subset=["product_code", "delta_tariff_pct", "annex", "effective_date"]
        )
    )
    code_counts = final_df["product_code"].value_counts()
    codes_in_multiple_annexes = code_counts[code_counts > 1]

    print(f"Rows extracted: {len(raw_df)}")
    print(f"Rows after deduplication: {len(final_df)}")
    print(f"Unique codes: {final_df['product_code'].nunique()}")
    print(f"Exact duplicate rows removed: {exact_dupes}")
    print(
        "Codes appearing in both annexes: "
        f"{len(codes_in_multiple_annexes)}"
    )
    if len(codes_in_multiple_annexes) > 0:
        print(f"  Examples: {codes_in_multiple_annexes.head(10).index.tolist()}")
    print(f"Missing duty values: {missing_duty}")
    print(
        f"Invalid codes (length != 8): "
        f"{invalid_codes if invalid_codes else '[]'}"
    )

    annex_counts = final_df.groupby("annex").size()
    print(f"Annex I rows: {annex_counts.get('I', 0)}")
    print(f"Annex II rows: {annex_counts.get('II', 0)}")


def main() -> None:
    os.makedirs(TARIFF_DIR, exist_ok=True)

    pdf_path = _pdf_path()
    rows = extract_from_pdf(pdf_path)
    raw_df = rows_to_dataframe(rows)
    final_df = deduplicate_rows(raw_df)

    final_df = final_df.sort_values(
        by=["annex", "product_code"], kind="mergesort"
    ).reset_index(drop=True)

    invalid_codes, missing_duty = validate_dataframe(final_df)
    if invalid_codes:
        raise RuntimeError(
            f"Validation failed: {len(invalid_codes)} invalid product codes."
        )
    if missing_duty:
        raise RuntimeError(f"Validation failed: {missing_duty} missing duties.")

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print()
    print_validation_report(raw_df, final_df, invalid_codes, missing_duty)


if __name__ == "__main__":
    main()
