"""metrics.py — Shared evaluation metrics and results-table writer for baselines.

Implements the locked evaluation spec (Locked Plan §8.4 / EVAL config):
RMSE, MAE, R2, Directional Accuracy, all computed over labeled nodes only
(via label_mask). record_all_metrics() appends one row per (model, fold) to
results/tables/baselines.csv, matching the schema in Locked Plan §9.3, and
deduplicates on (model_name, fold) to avoid the "duplicate rows from a
restarted run" failure mode (CP39).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "tables", "baselines.csv")

HORIZON_NAMES = ["3m", "6m", "12m"]


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def compute_r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_directional_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.sign(pred) == np.sign(true)))


def compute_all_metrics_for_horizon(pred: np.ndarray, true: np.ndarray) -> dict:
    return {
        "RMSE": compute_rmse(pred, true),
        "MAE": compute_mae(pred, true),
        "R2": compute_r2(pred, true),
        "DirAcc": compute_directional_accuracy(pred, true),
    }


def evaluate_prediction(pred, y, label_mask) -> dict:
    """pred, y: (N_NODES, 3) arrays/tensors. label_mask: (N_NODES,) bool.

    Returns a flat dict with RMSE_3m, RMSE_6m, ..., DirAcc_12m keys, computed
    only over rows where label_mask is True.
    """
    pred_np = _to_numpy(pred)
    y_np = _to_numpy(y)
    mask_np = _to_numpy(label_mask).astype(bool)

    pred_masked = pred_np[mask_np]
    y_masked = y_np[mask_np]

    out: dict = {}
    for h_idx, h_name in enumerate(HORIZON_NAMES):
        m = compute_all_metrics_for_horizon(pred_masked[:, h_idx], y_masked[:, h_idx])
        for metric_name, val in m.items():
            out[f"{metric_name}_{h_name}"] = val
    return out


def record_all_metrics(
    model_name: str, fold_idx: int, val_event: str, metrics: dict, results_path: str = RESULTS_PATH
) -> None:
    """Append one row to results/tables/baselines.csv, deduplicating on
    (model_name, fold) — CP39."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    row = {"model_name": model_name, "fold": fold_idx, "val_event": val_event, **metrics}
    new_row_df = pd.DataFrame([row])

    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
        results = results[~((results["model_name"] == model_name) & (results["fold"] == fold_idx))]
        results = pd.concat([results, new_row_df], ignore_index=True)
    else:
        results = new_row_df

    results.to_csv(results_path, index=False)


def summarize(results_path: str = RESULTS_PATH) -> pd.DataFrame:
    """Mean +/- std per model across folds, for all metric columns."""
    results = pd.read_csv(results_path)
    metric_cols = [c for c in results.columns if c not in ("model_name", "fold", "val_event")]
    summary = results.groupby("model_name")[metric_cols].agg(["mean", "std"])
    return summary
