"""augmentation.py — Training-only data augmentation.

Locked Plan §8.2. Every function clones its input rather than mutating in
place (CP23 discipline: never mutate a Data object that might be aliased
elsewhere -- see temporal-jitter note below for why this matters here
specifically, beyond the general principle). NEVER call any of these
during validation/evaluation (CP34) -- only inside the training loop's
per-event augmentation step.

Spec-ambiguity note (documented per PROJECT_STATE.md's established pattern
of resolving contradictions in the original Locked Plan rather than
guessing silently): augment_temporal_jitter's "+1: use snapshots [1..7] +
repeat snapshot 7 as snapshot 0" / "-1: use snapshot 0 repeated as new
snapshot 0 + snapshots [0..6]" is inconsistently worded across the two
branches. The two candidate literal readings for +1 disagreed on whether
the repeated element lands at position 0 or position 7; only the
position-7 reading is consistent with (a) the -1 branch's parallel
structure (repeated element at the boundary being extrapolated INTO -- the
start when shifting backward, the end when shifting forward) and (b) the
training loop immediately doing `seq[7] = augment_shock_magnitude(seq[7])`
right after jitter, which only makes sense if position 7 still holds the
real event-quarter snapshot after jitter. Implemented as:
    +1 (shift forward): [s1, s2, s3, s4, s5, s6, s7, s7]  (drop s0, pad end)
    -1 (shift backward): [s0, s0, s1, s2, s3, s4, s5, s6]  (drop s7, pad start)
Each element of the output is an independent deepcopy -- s7 (or s0) is
duplicated at two positions, so if it were the SAME object reference at
both positions, augmenting one (e.g. augment_shock_magnitude on the seq[7]
slot) would silently also mutate the other slot's data. This is exactly
the aliasing failure mode CP23 describes, just reachable through jitter's
duplication rather than list multiplication.
"""

from __future__ import annotations

import copy
import os
import random
import sys

import torch
from torch_geometric.data import Data

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402

T = config.TRAINING


def augment_shock_magnitude(data: Data, sigma: float | None = None) -> Data:
    """Add N(0, sigma^2) noise to edge_attr[:,3] where nonzero. Zero entries stay zero."""
    sigma = T["augment_shock_sigma"] if sigma is None else sigma
    out = copy.deepcopy(data)
    e3 = out.edge_attr[:, 3]
    nonzero_mask = e3 != 0.0
    noise = torch.randn_like(e3) * sigma
    e3_new = e3.clone()
    e3_new[nonzero_mask] = e3[nonzero_mask] + noise[nonzero_mask]
    out.edge_attr = out.edge_attr.clone()
    out.edge_attr[:, 3] = e3_new
    return out


def augment_temporal_jitter(temporal_sequence: list[Data], jitter_prob: float | None = None) -> list[Data]:
    """50% chance of a +-1 quarter shift (see module docstring for the resolved
    interpretation); otherwise returns an unchanged (deep-copied) sequence."""
    jitter_prob = T["augment_jitter_prob"] if jitter_prob is None else jitter_prob
    assert len(temporal_sequence) == 8, f"expected 8 snapshots, got {len(temporal_sequence)}"

    if random.random() >= jitter_prob:
        return [copy.deepcopy(d) for d in temporal_sequence]

    direction = random.choice([1, -1])
    if direction == 1:
        # drop s0, keep s1..s7, pad the end with a duplicate of s7
        new_seq = [temporal_sequence[i] for i in range(1, 8)] + [temporal_sequence[7]]
    else:
        # drop s7, keep s0..s6, pad the start with a duplicate of s0
        new_seq = [temporal_sequence[0]] + [temporal_sequence[i] for i in range(0, 7)]

    return [copy.deepcopy(d) for d in new_seq]


def augment_edge_dropout(data: Data, p: float | None = None, threshold: float | None = None) -> Data:
    """Zero out (not remove) the entire feature vector of low-weight edges
    (import_pen_coeff < threshold) with probability p each."""
    p = T["augment_edge_drop_p"] if p is None else p
    threshold = T["augment_edge_drop_threshold"] if threshold is None else threshold

    out = copy.deepcopy(data)
    import_pen = out.edge_attr[:, 1]
    low_weight = import_pen < threshold
    drop_roll = torch.rand(out.edge_attr.size(0)) < p
    drop_mask = low_weight & drop_roll

    keep = (~drop_mask).float().unsqueeze(-1)   # (E, 1)
    out.edge_attr = out.edge_attr.clone() * keep
    return out


def augment_label_noise(labels: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Add N(0, sigma^2) to non-NaN label values only; NaN entries stay NaN."""
    sigma = T["augment_label_sigma"] if sigma is None else sigma
    not_nan = ~torch.isnan(labels)
    noise = torch.randn_like(labels) * sigma
    out = labels.clone()
    out[not_nan] = labels[not_nan] + noise[not_nan]
    return out
