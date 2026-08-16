"""amplifier_sectors.py — Nodes whose learned-attention centrality exceeds
their raw trade-flow centrality.

Research Brief §9.4 / Locked Plan Phase 10. An "amplifier" node is one the
model's attention treats as more central to shock transmission than its
raw trade volume alone would suggest -- i.e. the model has learned that
this node punches above its trade-volume weight in propagating price
effects, which is exactly the kind of structural insight a static
Leontief/trade-volume view would miss.

Uses eigenvector centrality (config.EVAL["amplifier_centrality"]) on two
weighted directed graphs built from the SAME fixed edge structure (CP22):
  - attention graph: weight = attention alpha (layer 1, averaged across
    heads and across all 6 folds' events for robustness -- see main()).
  - trade graph: weight = raw flow_usd from edges_2014.parquet (the
    project's frozen structural prior, same file the canonical edge
    template itself comes from).

amplification_ratio = attention_centrality / (trade_centrality + eps)

Usage:
    python src/analysis/amplifier_sectors.py
"""

from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

EVENT_NAMES = [e["name"] for e in config.EVENTS]
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")
EDGES_2014_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PROCESSED_EDGES"], "edges_2014.parquet")
RESULTS_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")

COUNTRY_LIST = config.GRAPH["COUNTRY_LIST"]
SECTOR_LIST = config.GRAPH["SECTOR_LIST"]
N_SECTORS = config.GRAPH["N_SECTORS"]


def node_label(node_id: int) -> str:
    country_idx, sector_idx = divmod(node_id, N_SECTORS)
    return f"{COUNTRY_LIST[country_idx]}_{SECTOR_LIST[sector_idx]}"


def average_attention_across_events() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average layer-1 attention (already head-averaged) across all 6 events'
    outputs -- same canonical edge structure (CP22), so a simple elementwise
    mean is well-defined."""
    alphas = []
    edge_src = edge_tgt = None
    for event_name in EVENT_NAMES:
        path = os.path.join(ANALYSIS_DIR, f"{event_name}_outputs.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        alphas.append(data["alpha_layer1"].mean(axis=1))
        if edge_src is None:
            edge_src, edge_tgt = data["edge_src"], data["edge_tgt"]
    assert alphas, "no per-event outputs found -- run collect_model_outputs.py first"
    return edge_src, edge_tgt, np.mean(alphas, axis=0)


def build_weighted_digraph(edge_src: np.ndarray, edge_tgt: np.ndarray, weight: np.ndarray) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_weighted_edges_from(zip(edge_src.tolist(), edge_tgt.tolist(), weight.tolist()))
    return g


def main() -> None:
    print("Averaging layer-1 attention across all 6 events...")
    edge_src, edge_tgt, mean_alpha = average_attention_across_events()

    print("Loading edges_2014.parquet for the raw trade-weighted graph...")
    edges = pd.read_parquet(EDGES_2014_PATH, columns=["src_id", "tgt_id", "flow_usd"])
    flow_lookup = {(int(r.src_id), int(r.tgt_id)): float(r.flow_usd) for r in edges.itertuples(index=False)}
    trade_weight = np.array([flow_lookup.get((int(s), int(t)), 0.0) for s, t in zip(edge_src, edge_tgt)])

    print(f"Building attention-weighted and trade-weighted graphs "
          f"({len(edge_src):,} edges each)...")
    attention_graph = build_weighted_digraph(edge_src, edge_tgt, mean_alpha)
    trade_graph = build_weighted_digraph(edge_src, edge_tgt, trade_weight)

    print("Computing eigenvector centrality (attention graph)...")
    attention_centrality = nx.eigenvector_centrality_numpy(attention_graph, weight="weight")
    print("Computing eigenvector centrality (trade graph)...")
    trade_centrality = nx.eigenvector_centrality_numpy(trade_graph, weight="weight")

    # Nodes with zero in-degree receive no centrality flow at all and make
    # numpy's eigenvector solver produce degenerate values (including sign-
    # flipped negative "centrality", found empirically -- e.g. CYP_T/DNK_T,
    # both true zero-in-edge nodes, not an amplification finding). Filtered
    # out rather than silently left in the ranked table.
    in_degree = attention_graph.in_degree()
    zero_indegree_nodes = {n for n, d in in_degree if d == 0}
    if zero_indegree_nodes:
        print(f"  Excluding {len(zero_indegree_nodes)} zero-in-degree nodes from the "
              f"ranking (degenerate eigenvector centrality, not real amplification): "
              f"{sorted(node_label(n) for n in zero_indegree_nodes)}")

    nodes = sorted((set(attention_centrality) | set(trade_centrality)) - zero_indegree_nodes)
    rows = []
    for node in nodes:
        ac = attention_centrality.get(node, 0.0)
        tc = trade_centrality.get(node, 0.0)
        ratio = ac / (tc + 1e-9)
        rows.append({
            "node_id": node,
            "label": node_label(node),
            "attention_centrality": ac,
            "trade_centrality": tc,
            "amplification_ratio": ratio,
        })

    df = pd.DataFrame(rows).sort_values("amplification_ratio", ascending=False)
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    out_path = os.path.join(RESULTS_TABLES, "amplifier_sectors.csv")
    df.to_csv(out_path, index=False)

    print(f"\nTop 10 amplifier nodes (attention centrality >> trade centrality):")
    print(df.head(10)[["label", "attention_centrality", "trade_centrality", "amplification_ratio"]]
          .to_string(index=False))

    print(f"\nBottom 10 (trade centrality >> attention centrality -- \"under-weighted\" by the model):")
    print(df.tail(10)[["label", "attention_centrality", "trade_centrality", "amplification_ratio"]]
          .to_string(index=False))

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
