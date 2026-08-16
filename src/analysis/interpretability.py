"""interpretability.py — Attention weights vs. Leontief inverse correlation.

Research Brief §9.2 / Locked Plan Phase 10. Main interpretability finding
the paper is built around: does the model's learned attention recover the
economic structure of shock transmission (the Leontief inverse) without
being told what it is?

CP40 (index-order risk, verified here by construction, not assumed): our
technical coefficient matrix is A[src,tgt] = flow_usd / gross_output[tgt]
(src/data/build_technical_coefficients.py) -- the standard economics
convention, input required from src per unit of tgt's own output. L =
(I-A)^-1 inherits that orientation: L[src,tgt] is the total (direct +
indirect) output of src embodied in tgt's total output, i.e. how
economically important src is as an input to tgt. For a GAT edge
(src,tgt), attention alpha measures how much tgt's representation update
weighs src. The non-transposed, correct pairing is therefore
alpha <-> L[src, tgt] -- both describe "how important is src to tgt" from
tgt's point of view. This is the SAME direction already used and verified
in the Leontief baseline (src/baselines/leontief_io.py's
predicted_delta_p = L.T @ node_delta_tau propagates a cost shock from
upstream sector i to downstream dependent k via L[i,k]).

Since alpha is only ever defined on the 157,838 real edges (not a dense
2408x2408 matrix with structural zeros for non-edges), there is no
separate "mask nonzero pairs" step -- L[edge_src, edge_tgt] and alpha are
already aligned 1:1 over exactly the edges that exist.

Reads results/analysis/{event}_outputs.npz (src/analysis/
collect_model_outputs.py) and data/processed/leontief/leontief_2014.npy.

Usage:
    python src/analysis/interpretability.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

EVENT_NAMES = [e["name"] for e in config.EVENTS]
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")
LEONTIEF_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "leontief", "leontief_2014.npy")
RESULTS_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")


def compute_correlation(alpha_per_edge: np.ndarray, edge_src: np.ndarray, edge_tgt: np.ndarray, L: np.ndarray) -> dict:
    leontief_vals = L[edge_src, edge_tgt]   # vectorized gather, L[src, tgt] -- not transposed (CP40)
    pearson_r, pearson_p = stats.pearsonr(alpha_per_edge, leontief_vals)
    spearman_r, spearman_p = stats.spearmanr(alpha_per_edge, leontief_vals)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n_edges": int(len(alpha_per_edge)),
    }


def main() -> None:
    print("Loading Leontief inverse (leontief_2014.npy, frozen structural prior)...")
    L = np.load(LEONTIEF_PATH).astype(np.float64)

    rows = []
    for event_name in EVENT_NAMES:
        path = os.path.join(ANALYSIS_DIR, f"{event_name}_outputs.npz")
        if not os.path.exists(path):
            print(f"  SKIP {event_name}: no outputs file (run collect_model_outputs.py first)")
            continue

        data = np.load(path)
        edge_src, edge_tgt = data["edge_src"], data["edge_tgt"]

        for layer_name in ["alpha_layer1", "alpha_layer2"]:
            alpha = data[layer_name].mean(axis=1).astype(np.float64)   # average across heads
            result = compute_correlation(alpha, edge_src, edge_tgt, L)
            rows.append({"event": event_name, "layer": layer_name.replace("alpha_", ""), **result})
            print(f"  {event_name:26s} {layer_name:12s} pearson_r={result['pearson_r']:+.4f} "
                  f"(p={result['pearson_p']:.2e})  spearman_r={result['spearman_r']:+.4f} "
                  f"(p={result['spearman_p']:.2e})")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    out_path = os.path.join(RESULTS_TABLES, "interpretability_attention_leontief_correlation.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("Summary (mean across events, per layer):")
    for layer in ["layer1", "layer2"]:
        sub = df[df["layer"] == layer]
        print(f"  {layer}: mean pearson_r={sub['pearson_r'].mean():+.4f} "
              f"(std={sub['pearson_r'].std():.4f}), "
              f"mean spearman_r={sub['spearman_r'].mean():+.4f} (std={sub['spearman_r'].std():.4f})")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
