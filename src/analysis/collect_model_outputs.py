"""collect_model_outputs.py — Recover per-event predictions + both GAT
layers' attention weights from the already-trained full-TSPN checkpoints.

TSPN.forward() only ever RETURNS gat_layer1.last_alpha (Locked Plan §7.5
step 5: "return predictions, gat_layer1.last_alpha") -- layer 2's attention
was never persisted by src/training/train.py, even though it's sitting
right there on the model object after any forward call. Rather than treat
"we only have layer 1" as a permanent limitation, this script reloads each
fold's best checkpoint (already trained, no retraining needed) and runs
ONE clean eval-mode forward pass on its held-out event to recover BOTH
gat_layer1.last_alpha and gat_layer2.last_alpha, plus the raw prediction
vector (also never saved -- only aggregate metrics were).

Each fold's model only ever saw its OWN held-out event at inference time
here too (same LOEO discipline as training) -- fold i's checkpoint predicts
on EVENT_NAMES[i], never any other event, so these outputs are genuinely
held-out, not just architecturally convenient.

Output: results/analysis/{event_name}_outputs.npz
    predictions   (N_NODES, 3) float32
    alpha_layer1  (E, num_heads) float32
    alpha_layer2  (E, num_heads) float32
    edge_src      (E,) int64  -- edge_index[0], same canonical template for every event
    edge_tgt      (E,) int64  -- edge_index[1]

Usage:
    python src/analysis/collect_model_outputs.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
SRC_DATA = os.path.join(PROJECT_ROOT, "src", "data")
SRC_MODELS = os.path.join(PROJECT_ROOT, "src", "models")
for p in (PROJECT_ROOT, SRC_DATA, SRC_MODELS):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
from tspn_event_graph import TSPNEventGraph  # noqa: E402,F401
from tspn import TSPN  # noqa: E402

EVENT_NAMES = [e["name"] for e in config.EVENTS]
PYG_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PYG_DATASETS"])
CKPT_DIR = os.path.join(PROJECT_ROOT, config.PATHS["MODEL_CHECKPOINTS"])
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")

DEVICE = torch.device("cpu")   # analysis is a handful of forward passes -- CPU is fine


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    for fold_idx, event_name in enumerate(EVENT_NAMES):
        ckpt_path = os.path.join(CKPT_DIR, f"tspn_fold{fold_idx}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {event_name}: no checkpoint at {ckpt_path}")
            continue

        graph: TSPNEventGraph = torch.load(
            os.path.join(PYG_DIR, f"{event_name}.pt"), weights_only=False
        )
        graph.temporal_sequence = [d.to(DEVICE) for d in graph.temporal_sequence]

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = TSPN(config.MODEL).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        with torch.no_grad():
            predictions, _ = model(graph.temporal_sequence)

        alpha_layer1 = model.gat_layer1.last_alpha
        alpha_layer2 = model.gat_layer2.last_alpha
        edge_index = graph.temporal_sequence[0].edge_index   # identical across all 8 snapshots (CP22)

        assert alpha_layer1 is not None and alpha_layer2 is not None
        assert alpha_layer1.shape == alpha_layer2.shape
        assert edge_index.shape[1] == alpha_layer1.shape[0]

        out_path = os.path.join(OUT_DIR, f"{event_name}_outputs.npz")
        np.savez(
            out_path,
            predictions=predictions.detach().cpu().numpy().astype(np.float32),
            alpha_layer1=alpha_layer1.detach().cpu().numpy().astype(np.float32),
            alpha_layer2=alpha_layer2.detach().cpu().numpy().astype(np.float32),
            edge_src=edge_index[0].detach().cpu().numpy().astype(np.int64),
            edge_tgt=edge_index[1].detach().cpu().numpy().astype(np.int64),
        )
        print(f"  {event_name} (fold {fold_idx}, checkpoint epoch {ckpt['epoch']}, "
              f"val_rmse_6m={ckpt['val_rmse_6m']:.4f}) -> {out_path}")

    print(f"\nDone. Outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
