"""quarterly_interpolation.py — Shared annual -> quarterly interpolation utilities.

WIOD/Comtrade data (edges, node features, tariff rates) is annual, but the
locked spec requires SEQ_LEN=8 quarterly graph snapshots per event (Locked
Plan §4.3/§5.2). No official quarterly WIOD/Comtrade data exists, so
intra-year quarters are constructed by linear interpolation between adjacent
annual snapshots (per the original guide's documented design), used by both
build_quarterly_snapshots.py (node features) and compute_edge_features.py
(edge features).

Convention (documented, since the original guide did not pin one down):
each annual value is treated as observed AT Q4 of that year. A quarter q in
{1,2,3,4} of year Y is interpolated between annual(Y-1) [=Q4 of Y-1] and
annual(Y) [=Q4 of Y] at fraction q/4:

    value(Y, q) = annual(Y-1) * (1 - q/4) + annual(Y) * (q/4)

so value(Y, 4) == annual(Y) exactly (no interpolation needed).
"""

from __future__ import annotations

import pandas as pd


def month_to_quarter(month: int) -> int:
    return (month - 1) // 3 + 1


def parse_event_date(date_str: str) -> tuple[int, int]:
    """'2018-03' -> (2018, 1) i.e. (year, quarter)."""
    year_str, month_str = date_str.split("-")
    return int(year_str), month_to_quarter(int(month_str))


def step_back_quarters(year: int, quarter: int, n: int) -> tuple[int, int]:
    """Return the (year, quarter) that is n quarters before (year, quarter)."""
    total_q = year * 4 + (quarter - 1) - n
    return total_q // 4, total_q % 4 + 1


def quarter_sequence(event_year: int, event_quarter: int) -> list[tuple[int, int]]:
    """8 (year, quarter) pairs: index 0 = event_quarter-7 ... index 7 = event_quarter."""
    return [step_back_quarters(event_year, event_quarter, 7 - i) for i in range(8)]


def interp_fraction(quarter: int) -> float:
    return quarter / 4.0


def interpolate_frame(
    df_prev: pd.DataFrame,
    df_curr: pd.DataFrame,
    key_cols: list[str],
    value_cols: list[str],
    quarter: int,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Outer-merge df_prev/df_curr on key_cols, linearly interpolate value_cols.

    Rows present in only one frame are treated as fill_value in the other
    (e.g. an edge that clears the threshold in one year but not the other).
    quarter == 4 returns df_curr's values directly (exact annual observation,
    df_prev unused).
    """
    frac = interp_fraction(quarter)
    if frac >= 1.0 - 1e-9:
        return df_curr[key_cols + value_cols].reset_index(drop=True)

    merged = df_prev[key_cols + value_cols].merge(
        df_curr[key_cols + value_cols], on=key_cols, how="outer", suffixes=("_prev", "_curr")
    )
    for c in value_cols:
        prev_col = merged[f"{c}_prev"].fillna(fill_value)
        curr_col = merged[f"{c}_curr"].fillna(fill_value)
        merged[c] = prev_col * (1.0 - frac) + curr_col * frac
        merged.drop(columns=[f"{c}_prev", f"{c}_curr"], inplace=True)
    return merged[key_cols + value_cols].reset_index(drop=True)
