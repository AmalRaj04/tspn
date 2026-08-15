"""train_ablations.py — Ablation training: GCN, no-temporal, no-shock, 1-layer.

Locked Plan §9.1 run order items 4, 5, 6, 7. Reuses losses.py,
augmentation.py, evaluate.py exactly as train.py's main TSPN training does
-- same loss, same augmentation functions, same metrics, same
results/tables/all_results.csv (distinguished by model_name). CP33/34/35/36
are handled identically to train.py (same validate(), same checkpoint
metadata, same contamination guard).

CP37 (Critical): each (ablation, fold) pair gets a FRESH model instance and
optimizer -- model_ctor() is called fresh inside the fold loop, never
reused or copied across folds or ablations.

Each ablation's INPUT DATA is precomputed once per ablation (not per fold,
not per epoch): the no-shock ablation's x/edge_attr transform
(ablation_data.shock_as_node_feature) is deterministic, so transforming
once and reusing across all 6 folds x N epochs is correct and avoids
redundant work. The standard augmentation pipeline (temporal jitter, shock-
magnitude noise, edge dropout, label noise) then runs on top of that
per-ablation base sequence every epoch, identically to train.py. Note
augment_shock_magnitude only perturbs edges where e[3] != 0 -- since the
no-shock ablation's e[3] is always 0 after the transform, that specific
augmentation naturally becomes a no-op for it with no special-casing
needed.

Usage:
    python src/training/train_ablations.py                          # all 4 ablations, all 6 folds
    python src/training/train_ablations.py --ablations TSPN_GCN      # just one
    python src/training/train_ablations.py --max_epochs 3            # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

_here = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
SRC_DATA = os.path.join(PROJECT_ROOT, "src", "data")
SRC_MODELS = os.path.join(PROJECT_ROOT, "src", "models")
for p in (PROJECT_ROOT, SRC_DATA, SRC_MODELS, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
from tspn_event_graph import TSPNEventGraph  # noqa: E402,F401
from tspn import TSPN  # noqa: E402
from tspn_ablations import TSPNGCNAblation, TSPNNoTemporalAblation, TSPN1LayerAblation  # noqa: E402
from ablation_data import shock_as_node_feature  # noqa: E402
from losses import compute_loss  # noqa: E402
from augmentation import (  # noqa: E402
    augment_temporal_jitter,
    augment_shock_magnitude,
    augment_edge_dropout,
    augment_label_noise,
)
import evaluate  # noqa: E402

T = config.TRAINING
EVENT_NAMES = [e["name"] for e in config.EVENTS]
PYG_DIR = os.path.join(PROJECT_ROOT, config.PATHS["PYG_DATASETS"])
CKPT_DIR = os.path.join(PROJECT_ROOT, config.PATHS["MODEL_CHECKPOINTS"])
RESULTS_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Locked Plan §9.1 items 4-7. Each entry: (model constructor, data transform
# or None). model_ctor takes no arguments -- config.MODEL is closed over --
# so calling it always returns a brand-new, randomly-initialized instance (CP37).
ABLATIONS: dict[str, dict] = {
    "TSPN_GCN": {
        "model_ctor": lambda: TSPNGCNAblation(config.MODEL),
        "data_transform": None,
    },
    "TSPN_NoTemporal": {
        "model_ctor": lambda: TSPNNoTemporalAblation(config.MODEL),
        "data_transform": None,
    },
    "TSPN_NoShock": {
        "model_ctor": lambda: TSPN({**config.MODEL, "node_feat_in": 10}),
        "data_transform": shock_as_node_feature,
    },
    "TSPN_1Layer": {
        "model_ctor": lambda: TSPN1LayerAblation(config.MODEL),
        "data_transform": None,
    },
}


def load_all_graphs() -> dict[str, TSPNEventGraph]:
    graphs = {}
    for name in EVENT_NAMES:
        path = os.path.join(PYG_DIR, f"{name}.pt")
        g: TSPNEventGraph = torch.load(path, weights_only=False)
        g.temporal_sequence = [d.to(DEVICE) for d in g.temporal_sequence]
        g.y = g.y.to(DEVICE)
        g.label_mask = g.label_mask.to(DEVICE)
        g.direct_hit_mask = g.direct_hit_mask.to(DEVICE)
        graphs[name] = g
    return graphs


class RunLogger:
    """Identical design to train.py's RunLogger (rewrites the CSV every
    epoch, resume-safe) -- duplicated rather than imported to keep this
    file runnable standalone without coupling to train.py's module state."""

    def __init__(self, use_wandb: bool, project: str, run_name: str, csv_path: str):
        self.csv_path = csv_path
        self.rows: list[dict] = []
        self._existing_rows = pd.read_csv(csv_path).to_dict("records") if os.path.exists(csv_path) else []
        self.wandb = None
        if use_wandb:
            try:
                import wandb as _wandb
                _wandb.init(project=project, name=run_name, reinit=True, config=T)
                self.wandb = _wandb
            except Exception as e:
                print(f"  [RunLogger] wandb unavailable ({type(e).__name__}: {e}); logging to local CSV only.")

    def log(self, row: dict) -> None:
        self.rows.append(row)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        combined = pd.DataFrame(self._existing_rows + self.rows)
        combined = combined.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")
        combined.to_csv(self.csv_path, index=False)
        if self.wandb is not None:
            try:
                self.wandb.log(row)
            except Exception:
                pass

    def finish(self) -> None:
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass


def validate(model, event_data: TSPNEventGraph) -> tuple[float, torch.Tensor, torch.Tensor | None]:
    """CP34: model.eval() is the literal first line. No augmentation, raw data."""
    model.eval()
    with torch.no_grad():
        val_pred, val_alpha = model(event_data.temporal_sequence)
        val_rmse_6m = evaluate.compute_rmse(val_pred[:, 1], event_data.y[:, 1], event_data.label_mask)
    return val_rmse_6m, val_pred, val_alpha


def train_one_ablation_fold(
    ablation_name: str,
    fold_idx: int,
    graphs: dict[str, TSPNEventGraph],
    base_sequences: dict[str, list],
    max_epochs: int | None = None,
    use_wandb: bool = False,
    seed: int = 0,
    resume: bool = True,
) -> dict:
    model_ctor = ABLATIONS[ablation_name]["model_ctor"]

    held_out_name = EVENT_NAMES[fold_idx]
    train_names = [n for n in EVENT_NAMES if n != held_out_name]

    # CP33: hard-fail contamination check, every fold, no exceptions.
    if len(train_names) != 5:
        raise RuntimeError(f"Fold {fold_idx}: expected 5 train events, got {len(train_names)}")
    if held_out_name in train_names:
        raise RuntimeError(f"CONTAMINATION: {held_out_name} present in its own fold's training set")

    max_epochs = T["max_epochs"] if max_epochs is None else max_epochs

    torch.manual_seed(seed)
    random.seed(seed)

    model = model_ctor().to(DEVICE)   # CP37: fresh instance every fold
    optimizer = torch.optim.Adam(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T["T_0"], T_mult=T["T_mult"])

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{ablation_name}_fold{fold_idx}_best.pt")
    resume_state_path = os.path.join(CKPT_DIR, f"{ablation_name}_fold{fold_idx}_resume_state.json")

    best_val_rmse = float("inf")
    patience_counter = 0
    start_epoch = 0

    if resume and os.path.exists(resume_state_path):
        with open(resume_state_path) as f:
            state = json.load(f)
        already_early_stopped = state.get("early_stopped", False)
        budget_exhausted = (state["epoch"] + 1) >= max_epochs
        if already_early_stopped or budget_exhausted:
            reason = "early-stopped" if already_early_stopped else f"reached max_epochs={max_epochs}"
            print(f"  {ablation_name} fold {fold_idx}: resume state says already done ({reason} at "
                  f"epoch {state['epoch']}, val_rmse_6m={state['best_val_rmse']:.4f}) -- evaluating only.")
            start_epoch = max_epochs
        elif os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            best_val_rmse = state["best_val_rmse"]
            patience_counter = state["patience_counter"]
            start_epoch = state["epoch"] + 1
            print(f"  {ablation_name} fold {fold_idx}: RESUMING from epoch {start_epoch}")

    train_log_path = os.path.join(RESULTS_TABLES, f"train_log_{ablation_name}_fold{fold_idx}.csv")
    logger = RunLogger(use_wandb, project="tspn-ablations",
                        run_name=f"{ablation_name}_fold{fold_idx}_{held_out_name}", csv_path=train_log_path)

    held_out_data = TSPNEventGraph(
        temporal_sequence=base_sequences[held_out_name],
        y=graphs[held_out_name].y,
        label_mask=graphs[held_out_name].label_mask,
        direct_hit_mask=graphs[held_out_name].direct_hit_mask,
        event_name=held_out_name,
        event_date=graphs[held_out_name].event_date,
    )

    for epoch in range(start_epoch, max_epochs):
        model.train()
        random.shuffle(train_names)
        epoch_loss = 0.0

        for name in train_names:
            event = graphs[name]
            seq = augment_temporal_jitter(base_sequences[name])
            seq[7] = augment_shock_magnitude(seq[7])
            seq[7] = augment_edge_dropout(seq[7])
            aug_labels = augment_label_noise(event.y)

            pred, alpha = model(seq)
            if alpha is None:   # GCN ablation has no attention to regularize
                loss = compute_loss(pred, aug_labels, event.label_mask, torch.zeros(1, device=DEVICE))
            else:
                loss = compute_loss(pred, aug_labels, event.label_mask, alpha)

            if torch.isnan(loss):
                raise RuntimeError(
                    f"CP35: NaN loss at {ablation_name} fold {fold_idx} epoch {epoch} event {name}"
                )

            optimizer.zero_grad()
            loss.backward()
            clip_norm = T["grad_clip_warmup_norm"] if epoch < T["grad_clip_warmup_epochs"] else T["grad_clip_norm"]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step(epoch)

        val_rmse_6m, _, val_alpha = validate(model, held_out_data)

        stopping = False
        if val_rmse_6m < best_val_rmse:
            best_val_rmse = val_rmse_6m
            patience_counter = 0
            torch.save({"epoch": epoch, "val_rmse_6m": val_rmse_6m, "model_state": model.state_dict()}, ckpt_path)
            if val_alpha is not None:
                np.save(os.path.join(RESULTS_TABLES, f"attention_{ablation_name}_fold{fold_idx}.npy"),
                        val_alpha.detach().cpu().numpy())
        else:
            patience_counter += 1
            if patience_counter >= T["early_stop_patience"]:
                stopping = True

        logger.log({
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_names),
            "val_rmse_6m": val_rmse_6m,
            "lr": scheduler.get_last_lr()[0],
            "grad_clip_norm": clip_norm,
        })

        with open(resume_state_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "best_val_rmse": best_val_rmse,
                "patience_counter": patience_counter,
                "early_stopped": stopping,
            }, f)

        if stopping:
            break

    logger.finish()

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    with open(resume_state_path) as f:
        total_epochs_run = json.load(f)["epoch"] + 1
    print(f"  {ablation_name} fold {fold_idx}: best checkpoint epoch {ckpt['epoch']}, "
          f"val_rmse_6m={ckpt['val_rmse_6m']:.4f} ({total_epochs_run} epochs total)")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        final_pred, _ = model(held_out_data.temporal_sequence)

    metrics = evaluate.record_all_metrics(final_pred, held_out_data, fold_idx, model_name=ablation_name)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="*", default=None, choices=list(ABLATIONS.keys()),
                         help="subset of ablations to run (default: all 4)")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print("Loading PyG event graphs...")
    graphs = load_all_graphs()

    ablation_names = args.ablations if args.ablations else list(ABLATIONS.keys())
    folds = args.folds if args.folds is not None else list(range(6))

    for ablation_name in ablation_names:
        transform = ABLATIONS[ablation_name]["data_transform"]
        base_sequences = {
            name: (transform(g.temporal_sequence) if transform else g.temporal_sequence)
            for name, g in graphs.items()
        }

        print(f"\n{'#'*60}\n# Ablation: {ablation_name}\n{'#'*60}")
        for fold_idx in folds:
            held_out_name = EVENT_NAMES[fold_idx]
            print(f"\nFold {fold_idx}: held out = {held_out_name}")
            metrics = train_one_ablation_fold(
                ablation_name, fold_idx, graphs, base_sequences,
                max_epochs=args.max_epochs, use_wandb=args.use_wandb,
                seed=args.seed, resume=not args.no_resume,
            )
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")

    print(f"\nDone. {len(ablation_names)} ablation(s) x {len(folds)} fold(s) recorded to results/tables/all_results.csv.")


if __name__ == "__main__":
    main()
