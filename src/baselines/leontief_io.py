"""leontief_io.py — Static Leontief input-output baseline.

Implements Locked Plan §6.1:

    node_delta_tau[i] = sum_{j: (j,i) is an incoming edge} import_pen_coeff[j->i] * delta_tariff[j->i]
    predicted_delta_p  = L.T @ node_delta_tau * PASS_THROUGH_RATE

Leontief gives ONE vector, applied identically to all three horizons (3m,
6m, 12m) -- no temporal component. This is expected and correct per spec.

L is data/processed/leontief/leontief_2014.npy: the project's single frozen
structural prior (2408x2408, ROW-free), used consistently for every event
regardless of its actual date -- consistent with the design already used
throughout Phase 2/4/5 (WIOD-2014 topology frozen; only magnitudes update).

Calibration (locked procedure, CP26): PASS_THROUGH_RATE is calibrated ONLY
on the UK Global Tariff 2021 event, by minimizing RMSE on the 6m horizon.
Since the relationship is linear in the scalar rate, the minimizer has a
closed form (ordinary least squares for one coefficient):

    rate = sum(pred_unit_i * y_6m_i) / sum(pred_unit_i^2)   over labeled i

That same rate is then applied uniformly to all 6 events -- never re-tuned
per event. The UK event's own prediction is still computed (useful for
later interpretability comparisons in Phase 10) but is EXCLUDED from
results/tables/baselines.csv, since reporting a model's accuracy on its own
calibration event is circular evaluation (CP26). This script only prints
the calibrated rate -- config.py's LEONTIEF["PASS_THROUGH_RATE"] must be
updated by hand immediately after running this (Locked Plan critical rule
#8), which is done as a deliberate, reviewed edit, not an unattended
self-modification of the single-source-of-truth config file.

Usage:
    python src/baselines/leontief_io.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
SRC_DATA = os.path.join(PROJECT_ROOT, "src", "data")
for p in (PROJECT_ROOT, SRC_DATA, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
from tspn_event_graph import TSPNEventGraph  # noqa: E402,F401  (needed for unpickling)
from metrics import evaluate_prediction, record_all_metrics  # noqa: E402

N_NODES = config.GRAPH["N_NODES"]
PYG_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PYG_DATASETS"])
LEONTIEF_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "leontief", "leontief_2014.npy")

CALIBRATION_EVENT = "uk_global_tariff_2021"
MODEL_NAME = "Leontief_IO"


def load_leontief() -> np.ndarray:
    L = np.load(LEONTIEF_PATH).astype(np.float64)
    assert L.shape == (N_NODES, N_NODES), f"leontief_2014.npy shape {L.shape}, expected ({N_NODES},{N_NODES})"
    return L


def load_event_graphs() -> dict[str, TSPNEventGraph]:
    graphs = {}
    for event in config.EVENTS:
        path = os.path.join(PYG_DIR, f"{event['name']}.pt")
        graphs[event["name"]] = torch.load(path, weights_only=False)
    return graphs


def compute_node_delta_tau(snapshot_q7) -> np.ndarray:
    """node_delta_tau[i] = sum over incoming edges (j->i) of e1 * e3."""
    edge_index = snapshot_q7.edge_index.numpy()   # (2, E): [0]=src, [1]=tgt
    edge_attr = snapshot_q7.edge_attr.numpy()      # (E, 6)
    contrib = edge_attr[:, 1] * edge_attr[:, 3]    # import_pen_coeff * tariff_delta
    tgt = edge_index[1]
    tau = np.zeros(N_NODES, dtype=np.float64)
    np.add.at(tau, tgt, contrib)
    return tau


def predict_unit(L: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return L.T @ tau


def calibrate_rate(pred_unit: np.ndarray, y_6m: np.ndarray, label_mask: np.ndarray) -> float:
    pu = pred_unit[label_mask]
    yt = y_6m[label_mask]
    denom = float(np.sum(pu * pu))
    if denom == 0.0:
        return 0.0
    return float(np.sum(pu * yt) / denom)


def main() -> None:
    print("Loading Leontief inverse (leontief_2014.npy, frozen structural prior)...")
    L = load_leontief()

    print("Loading PyG event graphs...")
    graphs = load_event_graphs()

    print("\nComputing unit shock-propagation vector (pre-calibration) for every event...")
    unit_preds: dict[str, np.ndarray] = {}
    for name, g in graphs.items():
        tau = compute_node_delta_tau(g.temporal_sequence[7])
        unit_preds[name] = predict_unit(L, tau)
        n_nonzero_tau = int((tau != 0.0).sum())
        print(f"  {name}: {n_nonzero_tau} nodes with nonzero incoming shock exposure")

    print(f"\nCalibrating PASS_THROUGH_RATE on {CALIBRATION_EVENT} (6m horizon, closed-form OLS)...")
    uk_graph = graphs[CALIBRATION_EVENT]
    y_uk = uk_graph.y.numpy()
    mask_uk = uk_graph.label_mask.numpy()
    rate = calibrate_rate(unit_preds[CALIBRATION_EVENT], y_uk[:, 1], mask_uk)
    print(f"  Calibrated LEONTIEF_PASS_THROUGH_RATE = {rate:.8f}")
    print(f"  ACTION REQUIRED: set config.LEONTIEF['PASS_THROUGH_RATE'] = {rate:.8f}")

    print(f"\nEvaluating (rate applied uniformly to all 6 events, same vector for all 3 horizons)...")
    event_names = [e["name"] for e in config.EVENTS]
    diracc_values = []
    for event in config.EVENTS:
        name = event["name"]
        g = graphs[name]
        pred_vec = unit_preds[name] * rate
        pred_3h = np.stack([pred_vec, pred_vec, pred_vec], axis=1)   # (N_NODES, 3)

        m = evaluate_prediction(pred_3h, g.y.numpy(), g.label_mask.numpy())
        print(f"\n  {name}:")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

        if name == CALIBRATION_EVENT:
            print(f"    (EXCLUDED from results/tables/baselines.csv -- used for calibration, CP26)")
            continue

        fold_idx = event_names.index(name)
        record_all_metrics(MODEL_NAME, fold_idx, name, m)
        diracc_values.append(m["DirAcc_6m"])

    print(f"\n{'='*60}")
    mean_diracc = float(np.mean(diracc_values))
    print(f"Mean DirAcc_6m across the 5 non-calibration folds: {mean_diracc:.4f}")
    if not (0.55 <= mean_diracc <= 0.70):
        print(f"  NOTE: outside the original sanity-check range [0.55, 0.70] -- "
              f"informational only, not a failure (see PROJECT_STATE.md).")
    print("Done. Leontief_IO baseline recorded to results/tables/baselines.csv "
          "(5 folds, UK excluded per CP26).")


if __name__ == "__main__":
    main()
