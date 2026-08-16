"""cascade_depth.py — How many hops does a tariff shock's price effect reach?

Research Brief §9.3 / Locked Plan Phase 10. Measures the k-hop attenuation
of the model's predicted price change, starting from the directly-shocked
nodes (direct_hit_mask: targets of an edge with nonzero e[3] at the event
quarter -- i.e. nodes whose own import costs rose).

Propagation direction: a directly-hit node's costs rising makes IT a more
expensive SUPPLIER to whoever buys from it next. So hop-1 neighbors are
nodes reached by following the graph's edges FORWARD from an origin node
(origin as src, neighbor as tgt) -- standard multi-source BFS along the
edge_src -> edge_tgt direction, not the reverse.

"Significant" (Brief's definition): predicted |delta p| at a hop's nodes
exceeds 5% of the origin nodes' own average |delta p| (config.EVAL
["cascade_significance_threshold"]). The horizon used is 6m (the
early-stopping metric throughout this project, and the middle of the
three prediction horizons).

Reads results/analysis/{event}_outputs.npz (predictions + the fixed edge
structure) and each event's PyG dataset (for direct_hit_mask).

Usage:
    python src/analysis/cascade_depth.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
SRC_DATA = os.path.join(PROJECT_ROOT, "src", "data")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DATA not in sys.path:
    sys.path.insert(0, SRC_DATA)

import config  # noqa: E402
from tspn_event_graph import TSPNEventGraph  # noqa: E402,F401

EVENT_NAMES = [e["name"] for e in config.EVENTS]
N_NODES = config.GRAPH["N_NODES"]
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")
PYG_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PYG_DATASETS"])
RESULTS_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")

SIGNIFICANCE_THRESHOLD = config.EVAL["cascade_significance_threshold"]
MAX_HOP = 5
HORIZON_IDX = 1   # 6m


def build_forward_adjacency(edge_src: np.ndarray, edge_tgt: np.ndarray) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    for s, t in zip(edge_src, edge_tgt):
        adj[int(s)].append(int(t))
    return adj


def bfs_hop_distances(origin_nodes: set[int], adj: dict[int, list[int]], max_hop: int) -> dict[int, list[int]]:
    """Multi-source BFS from origin_nodes. Returns {hop: [node_ids at exactly that hop]}."""
    visited = {n: 0 for n in origin_nodes}
    queue = deque((n, 0) for n in origin_nodes)
    hop_groups: dict[int, list[int]] = defaultdict(list)

    while queue:
        node, dist = queue.popleft()
        if dist >= max_hop:
            continue
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                hop_groups[dist + 1].append(neighbor)
                queue.append((neighbor, dist + 1))

    return hop_groups


def measure_cascade_depth(event_name: str) -> dict:
    outputs = np.load(os.path.join(ANALYSIS_DIR, f"{event_name}_outputs.npz"))
    predictions = outputs["predictions"]   # (N_NODES, 3)
    edge_src, edge_tgt = outputs["edge_src"], outputs["edge_tgt"]

    graph: TSPNEventGraph = torch.load(os.path.join(PYG_DIR, f"{event_name}.pt"), weights_only=False)
    origin_nodes = set(np.where(graph.direct_hit_mask.numpy())[0].tolist())

    if not origin_nodes:
        return {"event": event_name, "n_origin_nodes": 0, "effective_cascade_hop": None}

    pred_abs = np.abs(predictions[:, HORIZON_IDX])
    origin_pred = pred_abs[list(origin_nodes)].mean()

    adj = build_forward_adjacency(edge_src, edge_tgt)
    hop_groups = bfs_hop_distances(origin_nodes, adj, MAX_HOP)

    row = {"event": event_name, "n_origin_nodes": len(origin_nodes), "origin_pred_6m": float(origin_pred)}
    effective_hop = None
    for hop in range(1, MAX_HOP + 1):
        nodes_at_hop = hop_groups.get(hop, [])
        if not nodes_at_hop:
            row[f"hop{hop}_n_nodes"] = 0
            row[f"hop{hop}_attenuation_ratio"] = None
            continue
        hop_pred = pred_abs[nodes_at_hop].mean()
        ratio = float(hop_pred / origin_pred) if origin_pred > 0 else None
        row[f"hop{hop}_n_nodes"] = len(nodes_at_hop)
        row[f"hop{hop}_attenuation_ratio"] = ratio
        if effective_hop is None and ratio is not None and ratio < SIGNIFICANCE_THRESHOLD:
            effective_hop = hop

    row["effective_cascade_hop"] = effective_hop
    return row


def main() -> None:
    rows = []
    for event_name in EVENT_NAMES:
        path = os.path.join(ANALYSIS_DIR, f"{event_name}_outputs.npz")
        if not os.path.exists(path):
            print(f"  SKIP {event_name}: no outputs file (run collect_model_outputs.py first)")
            continue
        row = measure_cascade_depth(event_name)
        rows.append(row)

        print(f"\n{event_name}: {row['n_origin_nodes']} origin nodes, "
              f"origin |pred_6m| = {row.get('origin_pred_6m', float('nan')):.5f}")
        for hop in range(1, MAX_HOP + 1):
            n = row.get(f"hop{hop}_n_nodes", 0)
            ratio = row.get(f"hop{hop}_attenuation_ratio")
            ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
            print(f"  hop {hop}: {n} nodes, attenuation ratio = {ratio_str}")
        if row["effective_cascade_hop"] is not None:
            print(f"  -> shock effectively dissipates at hop {row['effective_cascade_hop']} "
                  f"(ratio < {SIGNIFICANCE_THRESHOLD})")
        else:
            print(f"  -> shock does not dissipate below {SIGNIFICANCE_THRESHOLD} within {MAX_HOP} hops")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    out_path = os.path.join(RESULTS_TABLES, "cascade_depth.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
