"""mlp_no_graph.py — MLP baseline with no graph structure (node features only).

Implements Locked Plan §6.3, corrected per Risk Checkpoints CP31: f3
(backward_linkage) is EXCLUDED from the input because it is derived from
the full Leontief inverse and therefore encodes complete graph structure --
including it would give this "no-graph" baseline indirect access to the
graph, making the graph-vs-no-graph comparison unfair.

Input (9-dim, not the Locked Plan's literal "9+1=10" -- CP31's f3 removal
supersedes that count): f0, f1, f2, f4, f5, f6, f7, f8 (8 node features,
event-quarter snapshot) + 1 scalar for direct tariff exposure on incoming
edges (trade-value-weighted: sum of import_pen_coeff * tariff_delta over
incoming edges at the event quarter -- the same "node_delta_tau" quantity
computed for the Leontief baseline, reused here via leontief_io.py).

Architecture (locked): Linear(9->128) -> ReLU -> Linear(128->64) -> ReLU ->
Linear(64->32) -> ReLU -> Linear(32->3).

Training protocol (locked, "same as TSPN" -- controls for training
procedure so the comparison isolates the effect of graph structure): Adam
lr=1e-3, weight_decay=1e-4, CosineAnnealingWarmRestarts(T_0=50, T_mult=2),
grad_clip_norm=1.0, max_epochs=200, early stopping patience=20 on
val_rmse_6m, loss weights 0.50/0.30/0.20 for 3m/6m/12m (the L1-attention
term in config.TRAINING is TSPN-specific and does not apply here).
Leave-One-Event-Out CV: 6 folds, train on the pooled labeled nodes from 5
events, validate/report on the 6th.

Usage:
    python src/baselines/mlp_no_graph.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
SRC_DATA = os.path.join(PROJECT_ROOT, "src", "data")
for p in (PROJECT_ROOT, SRC_DATA, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
from tspn_event_graph import TSPNEventGraph  # noqa: E402,F401
from metrics import evaluate_prediction, record_all_metrics  # noqa: E402
from leontief_io import load_event_graphs, compute_node_delta_tau  # noqa: E402

N_NODES = config.GRAPH["N_NODES"]
FEATURE_IDX = [0, 1, 2, 4, 5, 6, 7, 8]   # f3 excluded (CP31)
INPUT_DIM = len(FEATURE_IDX) + 1          # + tariff-exposure scalar
HIDDEN_DIMS = [128, 64, 32]
MODEL_NAME = "MLP_no_graph"

T = config.TRAINING
SEED = 0


class MLPNoGraph(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [INPUT_DIM] + HIDDEN_DIMS
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_event_features(event_name: str, graphs: dict) -> torch.Tensor:
    """(N_NODES, INPUT_DIM) feature matrix at the event-quarter (q7) snapshot."""
    g = graphs[event_name]
    x_full = g.temporal_sequence[7].x.numpy()          # (N_NODES, 9)
    x_sel = x_full[:, FEATURE_IDX]                       # (N_NODES, 8)
    tau = compute_node_delta_tau(g.temporal_sequence[7])  # (N_NODES,)
    x = np.concatenate([x_sel, tau[:, None]], axis=1)     # (N_NODES, 9)
    return torch.tensor(x, dtype=torch.float32)


def weighted_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse_3m = torch.mean((pred[:, 0] - y[:, 0]) ** 2)
    mse_6m = torch.mean((pred[:, 1] - y[:, 1]) ** 2)
    mse_12m = torch.mean((pred[:, 2] - y[:, 2]) ** 2)
    return T["loss_weight_3m"] * mse_3m + T["loss_weight_6m"] * mse_6m + T["loss_weight_12m"] * mse_12m


def rmse_6m(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred[:, 1] - y[:, 1]) ** 2)).item())


def run_fold(fold_idx: int, held_out: dict, train_events: list[str], features: dict, graphs: dict) -> None:
    held_out_name = held_out["name"]
    print(f"\nFold {fold_idx}: held out = {held_out_name}, train on {train_events}")

    torch.manual_seed(SEED)
    model = MLPNoGraph()
    optimizer = torch.optim.Adam(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T["T_0"], T_mult=T["T_mult"])

    # Pool labeled training examples across the 5 train events
    train_X, train_y = [], []
    for name in train_events:
        g = graphs[name]
        mask = g.label_mask
        train_X.append(features[name][mask])
        train_y.append(g.y[mask])
    train_X = torch.cat(train_X, dim=0)
    train_y = torch.cat(train_y, dim=0)

    val_g = graphs[held_out_name]
    val_X = features[held_out_name][val_g.label_mask]
    val_y = val_g.y[val_g.label_mask]

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(T["max_epochs"]):
        model.train()
        optimizer.zero_grad()
        pred = model(train_X)
        loss = weighted_loss(pred, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=T["grad_clip_norm"])
        optimizer.step()
        scheduler.step(epoch)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_rmse = rmse_6m(val_pred, val_y)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= T["early_stop_patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        full_pred = model(features[held_out_name])   # (N_NODES, 3), all nodes

    m = evaluate_prediction(full_pred, val_g.y, val_g.label_mask)
    print(f"  best epoch val_rmse_6m={best_val_rmse:.4f} (stopped at epoch {epoch})")
    for k, v in m.items():
        print(f"    {k}: {v:.4f}")
    record_all_metrics(MODEL_NAME, fold_idx, held_out_name, m)


def main() -> None:
    print("Loading PyG event graphs...")
    graphs = load_event_graphs()

    print("Building per-event (9-dim) feature matrices (f3 excluded, CP31)...")
    features = {name: build_event_features(name, graphs) for name in graphs}

    event_names = [e["name"] for e in config.EVENTS]
    for fold_idx, held_out in enumerate(config.EVENTS):
        train_events = [n for n in event_names if n != held_out["name"]]
        assert len(train_events) == 5, f"fold {fold_idx}: expected 5 train events, got {len(train_events)}"
        assert held_out["name"] not in train_events, f"CONTAMINATION: fold {fold_idx}"
        run_fold(fold_idx, held_out, train_events, features, graphs)

    print(f"\nDone. {MODEL_NAME} baseline recorded to results/tables/baselines.csv (6 folds).")


if __name__ == "__main__":
    main()
