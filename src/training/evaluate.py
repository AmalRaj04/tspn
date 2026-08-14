"""evaluate.py — TSPN evaluation functions.

Locked Plan §8.4. Function signatures match the training loop pseudocode's
actual call sites exactly, e.g.:
    val_rmse_6m = compute_rmse(val_pred[:, 1], held_out.y[:, 1], held_out.label_mask)
i.e. pred/labels are full-length (N_NODES,) 1-D tensors for a single
horizon (already column-sliced by the caller), and mask is the full-length
(N_NODES,) boolean label_mask -- masking happens inside these functions,
not before calling them.

Reuses the core metric math from src/baselines/metrics.py (already used by
every Phase 6 baseline) rather than duplicating RMSE/MAE/R2/DirAcc a second
time, so baseline and TSPN numbers are computed identically and are
directly comparable.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
SRC_BASELINES = os.path.join(PROJECT_ROOT, "src", "baselines")
for p in (PROJECT_ROOT, SRC_BASELINES):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
from metrics import (  # noqa: E402
    _to_numpy,
    compute_rmse as _rmse,
    compute_mae as _mae,
    compute_r2 as _r2,
    compute_directional_accuracy as _diracc,
    evaluate_prediction,
    record_all_metrics as _record_all_metrics,
)

RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "tables", "all_results.csv")


def _masked_arrays(pred, labels, mask) -> tuple[np.ndarray, np.ndarray]:
    p = _to_numpy(pred)
    l = _to_numpy(labels)
    m = _to_numpy(mask).astype(bool)
    return p[m], l[m]


def compute_rmse(pred, labels, mask) -> float:
    p, l = _masked_arrays(pred, labels, mask)
    return _rmse(p, l)


def compute_mae(pred, labels, mask) -> float:
    p, l = _masked_arrays(pred, labels, mask)
    return _mae(p, l)


def compute_r2(pred, labels, mask) -> float:
    p, l = _masked_arrays(pred, labels, mask)
    return _r2(p, l)


def compute_directional_accuracy(pred, labels, mask) -> float:
    p, l = _masked_arrays(pred, labels, mask)
    return _diracc(p, l)


def bootstrap_ci(
    pred, labels, mask, n: int | None = None, confidence: float | None = None, metric_fn=None, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap CI for a metric (default RMSE) over the masked (labeled) nodes.

    n / confidence default to config.EVAL["bootstrap_n"] / ["bootstrap_ci"].
    """
    n = config.EVAL["bootstrap_n"] if n is None else n
    confidence = config.EVAL["bootstrap_ci"] if confidence is None else confidence
    metric_fn = _rmse if metric_fn is None else metric_fn

    p, l = _masked_arrays(pred, labels, mask)
    n_samples = len(p)
    if n_samples == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        stats[i] = metric_fn(p[idx], l[idx])

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(stats, 100 * alpha))
    upper = float(np.percentile(stats, 100 * (1.0 - alpha)))
    return lower, upper


def record_all_metrics(pred, event_data, fold_idx: int, model_name: str = "TSPN") -> dict:
    """pred: (N_NODES, 3). event_data: TSPNEventGraph (uses .y, .label_mask,
    .event_name). Appends one row to results/tables/all_results.csv
    (dedup-safe on (model_name, fold), CP39 -- reuses metrics.record_all_metrics)."""
    m = evaluate_prediction(pred, event_data.y, event_data.label_mask)
    _record_all_metrics(model_name, fold_idx, event_data.event_name, m, results_path=RESULTS_PATH)
    return m
