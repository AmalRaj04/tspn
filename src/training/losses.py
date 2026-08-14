"""losses.py — TSPN training loss.

Locked Plan §8.1 (exact formula):
    L = 0.50 * MSE(pred[:,0], label[:,0], mask)
      + 0.30 * MSE(pred[:,1], label[:,1], mask)
      + 0.20 * MSE(pred[:,2], label[:,2], mask)
      + 0.01 * mean(|alpha_weights|)

Weights are pulled from config.TRAINING by default so config.py stays the
single source of truth (no hardcoded 0.50/0.30/0.20/0.01 duplicated here).
"""

from __future__ import annotations

import os
import sys

import torch

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402


def compute_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    alpha_weights: torch.Tensor,
    training_config: dict | None = None,
) -> torch.Tensor:
    """pred, labels: (N_NODES, 3). mask: (N_NODES,) bool. alpha_weights: (E, H)."""
    cfg = training_config or config.TRAINING

    pred_masked = pred[mask]
    labels_masked = labels[mask]

    mse_3m = torch.mean((pred_masked[:, 0] - labels_masked[:, 0]) ** 2)
    mse_6m = torch.mean((pred_masked[:, 1] - labels_masked[:, 1]) ** 2)
    mse_12m = torch.mean((pred_masked[:, 2] - labels_masked[:, 2]) ** 2)
    l1_attn = alpha_weights.abs().mean()

    return (
        cfg["loss_weight_3m"] * mse_3m
        + cfg["loss_weight_6m"] * mse_6m
        + cfg["loss_weight_12m"] * mse_12m
        + cfg["loss_weight_l1_attn"] * l1_attn
    )
