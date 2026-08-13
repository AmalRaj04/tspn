"""panel_var.py — Panel autoregressive baseline: PPI endogenous, tariff exogenous.

Implements Locked Plan §6.2 ("Lag order p=4 (quarters). One VAR per
(country, sector) with PPI as endogenous variable and tariff_rate as
exogenous.") with one necessary tool substitution, documented here and in
PROJECT_STATE.md: statsmodels.tsa.api.VAR requires >=2 endogenous columns
("Only gave one variable to VAR" -- verified) and cannot represent a
genuinely single-variable (PPI-only) endogenous series as the spec
describes. statsmodels.tsa.ar_model.AutoReg is the correct tool for an
AR(4) model with a single endogenous series and an exogenous regressor,
which is what "PPI as endogenous, tariff_rate as exogenous" actually
specifies; it is used here instead, with the same lag order (4).

Per node (country, sector):
  - Endogenous series: ppi_change history (from ppi_quarterly_all.parquet),
    all quarters strictly before the event quarter.
  - Exogenous series: quarterly tariff level, 0.0 pre-2015 (no WITS
    coverage) and interpolated from sector_tariffs.parquet 2015-2021 (same
    interpolation convention as quarterly_interpolation.py); held flat at
    the 2021 value for quarters beyond 2021 (last known schedule persists).
  - A node is only attempted if it has >= MIN_OBS quarters of PPI history
    before the event (AR(4) + 1 exog regressor needs a reasonable degrees-
    of-freedom margin); nodes without a real PPI series (the same ~72% PPI-
    coverage gap already documented in PROJECT_STATE.md decision 3) are
    left at a 0.0 prediction and are naturally excluded from evaluation
    anyway since evaluation only uses label_mask.
  - Forecast 4 quarters ahead with the fitted model + exog_oos = the
    node's own tariff series for those quarters, then compound the
    quarterly forecasts into cumulative delta_3m/6m/12m exactly as
    generate_labels.py compounds the true labels, for an apples-to-apples
    comparison.

Usage:
    python src/baselines/panel_var.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.ar_model import AutoReg

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
SRC_DATA = os.path.join(PROJECT_ROOT, "src", "data")
for p in (PROJECT_ROOT, SRC_DATA, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
from tspn_event_graph import TSPNEventGraph  # noqa: E402,F401
from metrics import evaluate_prediction, record_all_metrics  # noqa: E402
from quarterly_interpolation import parse_event_date, step_back_quarters, interp_fraction  # noqa: E402
from leontief_io import load_event_graphs  # noqa: E402

N_NODES = config.GRAPH["N_NODES"]
COUNTRY_LIST = config.GRAPH["COUNTRY_LIST"]
SECTOR_LIST = config.GRAPH["SECTOR_LIST"]
COUNTRY_IDX = {c: i for i, c in enumerate(COUNTRY_LIST)}
SECTOR_IDX = {s: i for i, s in enumerate(SECTOR_LIST)}

PPI_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_LABELS"], "ppi_quarterly_all.parquet")
TARIFF_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_TARIFF_RATES"], "sector_tariffs.parquet")

MODEL_NAME = "Panel_VAR"
LAG_ORDER = 4
MIN_OBS = 16
LAST_TARIFF_YEAR = 2021
HORIZONS = {"delta_3m": 1, "delta_6m": 2, "delta_12m": 4}


def normalize_sector(raw: str) -> str:
    return str(raw).replace("-", "_")


def step_forward(year: int, quarter: int, n: int) -> tuple[int, int]:
    return step_back_quarters(year, quarter, -n)


# ---------------------------------------------------------------------------
# Build per-(country, sector) PPI history (quarter -> ppi_change)
# ---------------------------------------------------------------------------
def load_ppi_series() -> dict[tuple[str, str], dict[tuple[int, int], float]]:
    ppi = pd.read_parquet(PPI_PATH)
    ppi["isic_sector"] = ppi["isic_sector"].astype(str).apply(normalize_sector)
    ppi["country"] = ppi["country"].astype(str)
    ppi = ppi[ppi["country"] != "WLD"]   # WLD is a labeling fallback, not a real graph node
    ppi["year"] = ppi["year"].astype(int)
    ppi["quarter"] = ppi["quarter"].astype(int)

    series: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
    for row in ppi.itertuples(index=False):
        key = (row.country, row.isic_sector)
        series.setdefault(key, {})[(int(row.year), int(row.quarter))] = float(row.ppi_change)
    return series


# ---------------------------------------------------------------------------
# Build per-(country, sector) quarterly tariff exogenous series
# ---------------------------------------------------------------------------
def load_tariff_annual() -> dict[tuple[str, str], dict[int, float]]:
    tar = pd.read_parquet(TARIFF_PATH)
    tar["country"] = tar["country"].astype(str)
    tar["sector"] = tar["sector"].astype(str)
    tar["year"] = tar["year"].astype(int)
    out: dict[tuple[str, str], dict[int, float]] = {}
    for row in tar.itertuples(index=False):
        out.setdefault((row.country, row.sector), {})[int(row.year)] = float(row.tariff_rate)
    return out


def tariff_at_quarter(annual: dict[int, float], year: int, quarter: int) -> float:
    """0.0 pre-2015; interpolated 2015-2021; flat-held at 2021 value beyond."""
    if year > LAST_TARIFF_YEAR:
        year, quarter = LAST_TARIFF_YEAR, 4
    prev_val = annual.get(year - 1, 0.0 if year - 1 < 2015 else annual.get(year, 0.0))
    curr_val = annual.get(year, 0.0)
    if year < 2015:
        return 0.0
    frac = interp_fraction(quarter)
    if frac >= 1.0 - 1e-9:
        return curr_val
    return prev_val * (1.0 - frac) + curr_val * frac


# ---------------------------------------------------------------------------
# Fit + forecast for one (country, sector) at one event
# ---------------------------------------------------------------------------
def fit_and_forecast(
    ppi_hist: dict[tuple[int, int], float],
    tariff_annual: dict[int, float],
    event_year: int,
    event_quarter: int,
) -> dict | None:
    # History: all quarters strictly before the event quarter, chronological
    history_keys = sorted(k for k in ppi_hist if k < (event_year, event_quarter))
    if len(history_keys) < MIN_OBS:
        return None

    ppi_series = pd.Series([ppi_hist[k] for k in history_keys])
    exog_series = pd.Series([tariff_at_quarter(tariff_annual, y, q) for y, q in history_keys])

    max_k = max(HORIZONS.values())
    future_keys = [step_forward(event_year, event_quarter, i) for i in range(1, max_k + 1)]
    exog_future = pd.Series([tariff_at_quarter(tariff_annual, y, q) for y, q in future_keys])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = AutoReg(ppi_series, lags=LAG_ORDER, exog=exog_series, old_names=False)
            res = model.fit()
            n = len(ppi_series)
            fc = res.predict(start=n, end=n + max_k - 1, exog_oos=exog_future)
        except Exception:
            return None

    fc_vals = fc.values
    if np.isnan(fc_vals).any() or np.isinf(fc_vals).any():
        return None

    cumulative = 1.0
    deltas = {}
    for i in range(1, max_k + 1):
        cumulative *= (1.0 + fc_vals[i - 1])
        if i in HORIZONS.values():
            deltas[i] = cumulative - 1.0
    return {
        "delta_3m": deltas[HORIZONS["delta_3m"]],
        "delta_6m": deltas[HORIZONS["delta_6m"]],
        "delta_12m": deltas[HORIZONS["delta_12m"]],
    }


def predict_event(
    event: dict,
    ppi_series_by_node: dict,
    tariff_annual_by_node: dict,
) -> tuple[torch.Tensor, int, int]:
    event_year, event_quarter = parse_event_date(event["date"])
    pred = np.zeros((N_NODES, 3), dtype=np.float32)
    n_fit, n_skip = 0, 0

    for (country, sector), ppi_hist in ppi_series_by_node.items():
        ci = COUNTRY_IDX.get(country)
        si = SECTOR_IDX.get(sector)
        if ci is None or si is None:
            continue
        node_id = config.node_id(ci, si)
        tariff_annual = tariff_annual_by_node.get((country, sector), {})

        result = fit_and_forecast(ppi_hist, tariff_annual, event_year, event_quarter)
        if result is None:
            n_skip += 1
            continue
        n_fit += 1
        pred[node_id, 0] = result["delta_3m"]
        pred[node_id, 1] = result["delta_6m"]
        pred[node_id, 2] = result["delta_12m"]

    return torch.tensor(pred, dtype=torch.float32), n_fit, n_skip


def main() -> None:
    print("Loading PPI history and tariff series...")
    ppi_series_by_node = load_ppi_series()
    tariff_annual_by_node = load_tariff_annual()
    print(f"  {len(ppi_series_by_node)} (country, sector) pairs with PPI history")

    print("Loading PyG event graphs (for y / label_mask only)...")
    graphs = load_event_graphs()

    event_names = [e["name"] for e in config.EVENTS]
    for fold_idx, event in enumerate(config.EVENTS):
        name = event["name"]
        print(f"\nFold {fold_idx}: {name} (date {event['date']})")
        pred, n_fit, n_skip = predict_event(event, ppi_series_by_node, tariff_annual_by_node)
        print(f"  AR(4)+exog fit for {n_fit} nodes, skipped {n_skip} "
              f"(insufficient history or fit failure)")

        g = graphs[name]
        m = evaluate_prediction(pred, g.y, g.label_mask)
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")
        record_all_metrics(MODEL_NAME, fold_idx, name, m)

    print(f"\nDone. {MODEL_NAME} baseline recorded to results/tables/baselines.csv (6 folds).")


if __name__ == "__main__":
    main()
