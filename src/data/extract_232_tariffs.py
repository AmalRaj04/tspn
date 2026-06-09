"""Extract US Section 232 (2018) steel and aluminum HTS codes from official sources.

Derives tariff scope from Federal Register proclamations (9704 aluminum, 9705 steel)
and expands to 8-digit HTS lines from HTSUS 2018 Revision 1.2 chapter PDFs.

Usage
-----
    python src/data/extract_232_tariffs.py

Outputs
-------
    data/raw/tariff_events/us_232_steel_2018.csv
    data/raw/tariff_events/us_232_aluminum_2018.csv
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

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

# Source documents (local filenames in data/raw/tariff_events/)
STEEL_PROCLAMATION = "2018-05478.pdf"       # Proclamation 9705 — steel
ALUMINUM_PROCLAMATION = "2018-05477.pdf"    # Proclamation 9704 — aluminum
CHAPTER_72 = "Chapter 72_2018HTSARevision1_2.pdf"
CHAPTER_73 = "Chapter 73_2018HTSARevision1_2.pdf"
CHAPTER_76 = "Chapter 76_2018HTSARevision1_2.pdf"

STEEL_OUTPUT = os.path.join(TARIFF_DIR, "us_232_steel_2018.csv")
ALUMINUM_OUTPUT = os.path.join(TARIFF_DIR, "us_232_aluminum_2018.csv")

STEEL_TARIFF_PCT = 25.0
ALUMINUM_TARIFF_PCT = 10.0

EXPECTED_STEEL = 170
EXPECTED_ALUMINUM = 60

# Annex note 16(b)(ii): these 7216 subheadings are excluded from steel coverage.
STEEL_EXCLUDED_8 = frozenset({"72166100", "72166900", "72169100"})

# Regex: tariff line subheading (XXXX.XX.XX) with optional statistical suffix.
HTS_SUBHEADING_RE = re.compile(
    r"\b(\d{4}\.\d{2}\.\d{2})\b(?:\s+(\d{2})\b)?"
)

# Duty / unit markers that terminate article descriptions in HTS tables.
DESC_TERMINATOR_RE = re.compile(
    r"\.{2,}|(?:\bkg\b|\bNo\.|\bFree\b|\b\d+(?:\.\d+)?%)"
)

SKIP_LINE_RE = re.compile(
    r"^(Harmonized Tariff|Annotated for|Heading/|Subheading|fix Quantity|General|Special|XV|"
    r"\d{2}-\d+|Notes|SECTION )",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumericRange:
    """Inclusive numeric range on a dotted HTS prefix (e.g. 7206.10 → 720610)."""

    start: int
    end: int

    def contains_prefix(self, prefix6: int) -> bool:
        return self.start <= prefix6 <= self.end


@dataclass
class SteelScope:
    ranges_6: list[NumericRange] = field(default_factory=list)
    excluded_8: frozenset[str] = STEEL_EXCLUDED_8


@dataclass
class AluminumScope:
    headings_4: frozenset[str] = frozenset(
        {"7601", "7604", "7605", "7606", "7607", "7608", "7609"}
    )
    # 7616.99.51.60 and 7616.99.51.70 → 8-digit 76169951 (stat suffix stripped).
    specific_8: frozenset[str] = frozenset({"76169951"})


def _pdf_path(filename: str) -> str:
    path = os.path.join(TARIFF_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required source file not found: {path}")
    return path


def _dotted_to_int6(code: str) -> int:
    """Convert dotted HTS fragment to 6-digit integer (e.g. '7206.10' → 720610)."""
    parts = code.strip().split(".")
    if len(parts) == 2:
        return int(parts[0]) * 100 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 10_000 + int(parts[1]) * 100 + int(parts[2])
    raise ValueError(f"Cannot convert dotted code to 6-digit integer: {code!r}")


def _hts_to_8(hts_dotted: str) -> str:
    """Convert XXXX.XX.XX to 8-digit HTS (statistical suffix dropped)."""
    return hts_dotted.replace(".", "")


def _extract_description(line: str, match_end: int) -> str:
    """Pull article description text following an HTS code on a tariff table line."""
    rest = line[match_end:].strip()
    rest = DESC_TERMINATOR_RE.split(rest, maxsplit=1)[0]
    rest = re.sub(r"\s+", " ", rest).strip(" .,;")
    return rest


def _read_pdf_text(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def parse_steel_scope_from_proclamation(text: str) -> SteelScope:
    """Parse steel article 6-digit ranges from Proclamation 9705 clause (1)."""
    match = re.search(
        r"6-digit level as:\s*(.+?),\s*including any subsequent revisions",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError("Could not locate steel scope definition in proclamation text.")

    scope_text = re.sub(r"\s+", " ", match.group(1))
    ranges: list[NumericRange] = []

    # Split on commas and the word "and" before the final range segment.
    segments = re.split(r",\s*|\s+and\s+", scope_text)
    for segment in segments:
        segment = segment.strip().rstrip(".")
        segment = re.sub(r"^and\s+", "", segment, flags=re.IGNORECASE)
        if not segment:
            continue

        through = re.match(
            r"(\d{4}\.\d{2}(?:\.\d{2})?)\s+through\s+(\d{4}\.\d{2}(?:\.\d{2})?)",
            segment,
            flags=re.IGNORECASE,
        )
        if through:
            start = _dotted_to_int6(through.group(1))
            end = _dotted_to_int6(through.group(2))
            ranges.append(NumericRange(start, end))
            continue

        singleton = re.match(r"(\d{4}\.\d{2}(?:\.\d{2})?)", segment)
        if singleton:
            value = _dotted_to_int6(singleton.group(1))
            ranges.append(NumericRange(value, value))
            continue

        raise ValueError(f"Unrecognized steel scope segment: {segment!r}")

    return SteelScope(ranges_6=ranges)


def parse_aluminum_scope_from_proclamation(text: str) -> AluminumScope:
    """Validate aluminum scope headings from Proclamation 9704 clause (1)."""
    normalized = re.sub(r"\s+", " ", text)
    expected_patterns = [
        r"unwrought aluminum \(HTS\s*7601\)",
        r"aluminum bars, rods, and profiles \(HTS\s*7604\)",
        r"aluminum wire \(HTS\s*7605\)",
        r"\(HTS\s*7606 and 7607\)",
        r"\(HTS\s*7608 and 7609\)",
        r"7616\.99\.51\.60",
        r"7616\.99\.51\.70",
    ]
    missing = [pat for pat in expected_patterns if not re.search(pat, normalized, re.I)]
    if missing:
        raise ValueError(
            "Proclamation text missing expected aluminum scope phrases: "
            + ", ".join(missing)
        )
    return AluminumScope()


def extract_hts_lines_from_chapter(pdf_path: str) -> dict[str, str]:
    """Extract 8-digit HTS codes and descriptions from an HTSUS chapter PDF."""
    entries: dict[str, str] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line or SKIP_LINE_RE.match(line):
                    continue

                for match in HTS_SUBHEADING_RE.finditer(line):
                    hts8 = _hts_to_8(match.group(1))
                    if len(hts8) != 8 or not hts8.isdigit():
                        continue

                    desc = _extract_description(line, match.end())
                    if hts8 not in entries or (desc and not entries[hts8]):
                        entries[hts8] = desc

    return entries


def _prefix6(hts8: str) -> int:
    return int(hts8[:6])


def in_steel_scope(hts8: str, scope: SteelScope) -> bool:
    if hts8 in scope.excluded_8:
        return False
    prefix = _prefix6(hts8)
    return any(r.contains_prefix(prefix) for r in scope.ranges_6)


def in_aluminum_scope(hts8: str, scope: AluminumScope) -> bool:
    if hts8[:4] in scope.headings_4:
        return True
    return hts8 in scope.specific_8


def filter_steel_codes(
    chapter_entries: dict[str, str], scope: SteelScope
) -> dict[str, str]:
    return {
        code: desc
        for code, desc in chapter_entries.items()
        if in_steel_scope(code, scope)
    }


def filter_aluminum_codes(
    chapter_entries: dict[str, str], scope: AluminumScope
) -> dict[str, str]:
    return {
        code: desc
        for code, desc in chapter_entries.items()
        if in_aluminum_scope(code, scope)
    }


def _entries_to_dataframe(
    entries: dict[str, str], delta_tariff_pct: float
) -> pd.DataFrame:
    rows = [
        {
            "hts_code": code,
            "product_description": entries[code],
            "delta_tariff_pct": delta_tariff_pct,
        }
        for code in sorted(entries)
    ]
    return pd.DataFrame(rows, columns=["hts_code", "product_description", "delta_tariff_pct"])


def _validate_codes(df: pd.DataFrame, label: str) -> list[str]:
    bad = [
        str(code)
        for code in df["hts_code"]
        if not re.fullmatch(r"\d{8}", str(code))
    ]
    return bad


def _print_validation(
    steel_df: pd.DataFrame,
    aluminum_df: pd.DataFrame,
    steel_scope: SteelScope,
) -> None:
    bad_steel = _validate_codes(steel_df, "steel")
    bad_aluminum = _validate_codes(aluminum_df, "aluminum")

    print(f"Steel rows: {len(steel_df)}")
    print(f"Aluminum rows: {len(aluminum_df)}")
    print(f"Bad steel codes: {bad_steel if bad_steel else '[]'}")
    print(f"Bad aluminum codes: {bad_aluminum if bad_aluminum else '[]'}")

    steel_n = len(steel_df)
    aluminum_n = len(aluminum_df)

    if abs(steel_n - EXPECTED_STEEL) > EXPECTED_STEEL * 0.3:
        print()
        print("DIAGNOSTIC — steel count differs from expected ~170:")
        print(
            f"  Extracted {steel_n} unique 8-digit lines from Chapters 72–73 "
            f"that fall within the Proclamation 9705 6-digit ranges."
        )
        print(
            "  The proclamation defines scope at the 6-digit level; expanding to "
            "all 8-digit statistical reporting lines in HTSUS 2018 Rev 1.2 yields "
            "more rows than condensed Commerce/CBP reference lists (~170)."
        )
        print(f"  Proclamation ranges parsed: {steel_scope.ranges_6}")
        print(f"  Excluded 8-digit codes (Annex note 16): {sorted(STEEL_EXCLUDED_8)}")
        print(f"  First 20 steel codes: {steel_df['hts_code'].head(20).tolist()}")

    if abs(aluminum_n - EXPECTED_ALUMINUM) > EXPECTED_ALUMINUM * 0.3:
        print()
        print("DIAGNOSTIC — aluminum count differs from expected ~60:")
        print(
            f"  Extracted {aluminum_n} unique 8-digit lines from Chapter 76 for "
            f"headings {sorted(AluminumScope().headings_4)} plus 76169951 "
            f"(7616.99.51.60 / .70 castings & forgings)."
        )
        print(
            "  HTSUS 2018 Rev 1.2 Chapter 76 contains a finite set of 8-digit "
            "reporting lines under those headings; statistical suffix variants "
            "collapse to the same 8-digit code per project rules."
        )
        print(
            "  Other 7616 subheadings (nails, cloth, luggage frames, etc.) are "
            "outside Proclamation 9704 scope."
        )
        print(f"  First 20 aluminum codes: {aluminum_df['hts_code'].head(20).tolist()}")


def main() -> None:
    os.makedirs(TARIFF_DIR, exist_ok=True)

    # 1. Read proclamations — legal scope is source of truth.
    steel_proclamation_text = _read_pdf_text(_pdf_path(STEEL_PROCLAMATION))
    aluminum_proclamation_text = _read_pdf_text(_pdf_path(ALUMINUM_PROCLAMATION))

    steel_scope = parse_steel_scope_from_proclamation(steel_proclamation_text)
    aluminum_scope = parse_aluminum_scope_from_proclamation(aluminum_proclamation_text)

    # 2. Extract detailed tariff lines from HTS chapter PDFs.
    ch72 = extract_hts_lines_from_chapter(_pdf_path(CHAPTER_72))
    ch73 = extract_hts_lines_from_chapter(_pdf_path(CHAPTER_73))
    ch76 = extract_hts_lines_from_chapter(_pdf_path(CHAPTER_76))

    steel_chapters = {**ch72, **ch73}

    # 3. Expand proclamation scope to individual 8-digit HTS codes.
    steel_entries = filter_steel_codes(steel_chapters, steel_scope)
    aluminum_entries = filter_aluminum_codes(ch76, aluminum_scope)

    if not steel_entries:
        raise RuntimeError("No steel HTS codes matched proclamation scope.")
    if not aluminum_entries:
        raise RuntimeError("No aluminum HTS codes matched proclamation scope.")

    # 4. Write output CSVs.
    steel_df = _entries_to_dataframe(steel_entries, STEEL_TARIFF_PCT)
    aluminum_df = _entries_to_dataframe(aluminum_entries, ALUMINUM_TARIFF_PCT)

    steel_df.to_csv(STEEL_OUTPUT, index=False)
    aluminum_df.to_csv(ALUMINUM_OUTPUT, index=False)

    print(f"Saved {STEEL_OUTPUT}")
    print(f"Saved {ALUMINUM_OUTPUT}")
    print()
    _print_validation(steel_df, aluminum_df, steel_scope)


if __name__ == "__main__":
    main()
