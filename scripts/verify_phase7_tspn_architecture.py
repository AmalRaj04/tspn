"""verify_phase7_tspn_architecture.py — Hard-gate verification of Phase 7.

Covers Locked Plan §7.6 (forward-pass sanity check) and the architecture-
specific Risk Checkpoints: CP27 (GAT layers must not share weights), CP28
(dropout disabled in eval mode -> deterministic), CP29 (no None gradients),
CP30 (attention does not collapse to a single neighbor).

Usage:
    python scripts/verify_phase7_tspn_architecture.py
"""

import math
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DATA = os.path.join(PROJECT_ROOT, "src", "data")
SRC_MODELS = os.path.join(PROJECT_ROOT, "src", "models")
for p in (PROJECT_ROOT, SRC_DATA, SRC_MODELS):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
from tspn_event_graph import TSPNEventGraph  # noqa: E402,F401
from tspn import TSPN  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")


N_NODES = config.GRAPH["N_NODES"]
EVENT_PATH = os.path.join(PROJECT_ROOT, config.PATHS["PYG_DATASETS"], "us_232_steel_2018.pt")
_graph = torch.load(EVENT_PATH, weights_only=False)


def test_forward_shape_and_nan():
    torch.manual_seed(0)
    model = TSPN()
    model.eval()
    with torch.no_grad():
        pred, alpha = model(_graph.temporal_sequence)
    assert pred.shape == (N_NODES, 3), f"pred shape {pred.shape}"
    assert not torch.isnan(pred).any(), "NaN in predictions"
    assert alpha is not None and alpha.shape[1] == config.MODEL["gat_num_heads"], \
        f"last_alpha shape {alpha.shape if alpha is not None else None}"
check("§7.6: forward pass runs, output (N_NODES,3), no NaN, last_alpha populated", test_forward_shape_and_nan)


def test_cp27_layers_not_aliased():
    model = TSPN()
    assert model.gat_layer1 is not model.gat_layer2, "gat_layer1 and gat_layer2 are the same object"
    assert id(model.gat_layer1) != id(model.gat_layer2)

    before_1 = model.gat_layer1.W_q.weight.data.clone()
    before_2 = model.gat_layer2.W_q.weight.data.clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    pred, alpha = model(_graph.temporal_sequence)
    loss = pred[_graph.label_mask].sum() + 0.01 * alpha.abs().mean()
    loss.backward()
    optimizer.step()

    after_1 = model.gat_layer1.W_q.weight.data
    after_2 = model.gat_layer2.W_q.weight.data
    diverged = not torch.equal(after_1 - before_1, after_2 - before_2)
    assert diverged, "CP27 VIOLATION: gat_layer1 and gat_layer2 updated identically -- they share weights"
check("CP27: gat_layer1/gat_layer2 are separate instances that diverge after a training step", test_cp27_layers_not_aliased)


def test_cp28_eval_deterministic():
    torch.manual_seed(0)
    model = TSPN()
    model.eval()
    with torch.no_grad():
        out1, _ = model(_graph.temporal_sequence)
        out2, _ = model(_graph.temporal_sequence)
    max_diff = (out1 - out2).abs().max().item()
    assert max_diff < 1e-7, f"model is stochastic in eval mode: max diff = {max_diff}"
check("CP28: identical output across repeated eval-mode forward passes (dropout disabled)", test_cp28_eval_deterministic)


def test_cp29_no_none_gradients():
    torch.manual_seed(0)
    model = TSPN()
    model.train()
    pred, alpha = model(_graph.temporal_sequence)
    loss = pred[_graph.label_mask].sum() + 0.01 * alpha.abs().mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no grad for: {missing}"
check("CP29: every parameter receives a gradient after backward()", test_cp29_no_none_gradients)


def test_cp30_attention_scaling_applied():
    """Confirm the 1/sqrt(head_dim) scaling is actually present in the score
    computation (not just that attention happens not to have collapsed on
    this one random init, which wouldn't prove the scaling exists)."""
    import inspect
    from tspn_gat_layer import TSPNGATLayer
    src = inspect.getsource(TSPNGATLayer.message)
    assert "self.scale" in src and "/ self.scale" in src, "no explicit scaling by sqrt(head_dim) found in message()"
    layer = TSPNGATLayer(head_dim=32)
    assert abs(layer.scale - math.sqrt(32)) < 1e-9, f"scale={layer.scale}, expected sqrt(32)"
check("CP30: attention scores scaled by 1/sqrt(head_dim) before softmax", test_cp30_attention_scaling_applied)


def test_attention_not_collapsed():
    """At random init (no training yet), attention should not already be
    degenerate (all weight on one neighbor) -- a coarse sanity check."""
    torch.manual_seed(0)
    model = TSPN()
    model.eval()
    with torch.no_grad():
        _, alpha = model(_graph.temporal_sequence)
    alpha_mean = alpha.mean(dim=1)   # (E,), averaged across heads
    tgt = _graph.temporal_sequence[0].edge_index[1]
    # crude per-target max share check on a sample of targets
    import torch as T
    unique_tgts = tgt.unique()[:200]
    max_shares = []
    for t in unique_tgts:
        mask = tgt == t
        vals = alpha_mean[mask]
        if len(vals) > 1:
            max_shares.append((vals.max() / vals.sum()).item())
    mean_max_share = sum(max_shares) / len(max_shares)
    print(f"    mean max-neighbor attention share (sampled 200 targets): {mean_max_share:.3f}")
    assert mean_max_share < 0.95, f"attention appears collapsed at init: mean max share={mean_max_share:.3f}"
check("attention not degenerate at random init (coarse pre-training sanity check)", test_attention_not_collapsed)


print(f"\n{'='*60}")
print(f"Phase 7 verification: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILING:", FAIL)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Phase 7 architecture complete, ready for Phase 8")
