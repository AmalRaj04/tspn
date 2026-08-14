"""train.py — TSPN training: 6-fold Leave-One-Event-Out cross-validation.

Locked Plan §8.3 loop structure, implemented with the risk-doc fixes
called out inline:

  CP33 (Critical): hard RuntimeError (not a warning/log) if the held-out
        event is ever found in its own fold's training set.
  CP34 (Critical): validate() has model.eval() as its literal first line
        and is called on the RAW event.temporal_sequence -- augmentation
        functions are never called anywhere in the validation path.
  CP35 (High): grad clipping happens BEFORE optimizer.step() (loop order
        below); NaN loss raises immediately rather than corrupting a
        checkpoint.
  CP36 (High): checkpoints save {epoch, val_rmse_6m, model_state}, not a
        bare state_dict, and the metadata is printed when reloaded for
        final evaluation.
  CP38 (Medium): grad-clip warmup (config.TRAINING["grad_clip_warmup_*"]).
  CP39 (Medium): result-row dedup is handled by evaluate.record_all_metrics
        (reuses src/baselines/metrics.record_all_metrics).

W&B logging is attempted opportunistically: wandb is broken in this
environment (`ModuleNotFoundError: No module named 'pkg_resources'` on
import, not just unauthenticated -- verified directly), so training must
not hard-depend on it. Every fold's per-epoch metrics are always written
to results/tables/train_log_fold{N}.csv (loss curves are inspectable
locally regardless of wandb availability); wandb.log is also attempted if
`use_wandb=True` and import/init succeed, silently skipped otherwise.

Usage:
    python src/training/train.py                       # full 200-epoch x 6-fold run
    python src/training/train.py --max_epochs 3         # smoke test
    python src/training/train.py --folds 0 2            # only specific folds
"""

from __future__ import annotations

import argparse
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


class RunLogger:
    """Always logs to a local list (-> CSV). Opportunistically also logs to
    wandb if requested and available; never lets wandb break training."""

    def __init__(self, use_wandb: bool, project: str, run_name: str):
        self.rows: list[dict] = []
        self.wandb = None
        if use_wandb:
            try:
                import wandb as _wandb
                _wandb.init(project=project, name=run_name, reinit=True, config=T)
                self.wandb = _wandb
            except Exception as e:
                print(f"  [RunLogger] wandb unavailable ({type(e).__name__}: {e}); "
                      f"logging to local CSV only.")

    def log(self, row: dict) -> None:
        self.rows.append(row)
        if self.wandb is not None:
            try:
                self.wandb.log(row)
            except Exception:
                pass

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(self.rows).to_csv(path, index=False)

    def finish(self) -> None:
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass


def load_all_graphs() -> dict[str, TSPNEventGraph]:
    graphs = {}
    for name in EVENT_NAMES:
        path = os.path.join(PYG_DIR, f"{name}.pt")
        graphs[name] = torch.load(path, weights_only=False)
    return graphs


def validate(model: TSPN, event_data: TSPNEventGraph) -> tuple[float, torch.Tensor, torch.Tensor]:
    """CP34: model.eval() is the literal first line. No augmentation, raw data."""
    model.eval()
    with torch.no_grad():
        val_pred, val_alpha = model(event_data.temporal_sequence)
        val_rmse_6m = evaluate.compute_rmse(val_pred[:, 1], event_data.y[:, 1], event_data.label_mask)
    return val_rmse_6m, val_pred, val_alpha


def train_one_fold(
    fold_idx: int,
    graphs: dict[str, TSPNEventGraph],
    max_epochs: int | None = None,
    use_wandb: bool = False,
    seed: int = 0,
) -> dict:
    held_out_name = EVENT_NAMES[fold_idx]
    train_names = [n for n in EVENT_NAMES if n != held_out_name]

    # CP33: hard-fail contamination check, every fold, no exceptions.
    if len(train_names) != 5:
        raise RuntimeError(f"Fold {fold_idx}: expected 5 train events, got {len(train_names)}")
    if held_out_name in train_names:
        raise RuntimeError(f"CONTAMINATION: {held_out_name} present in its own fold's training set")

    max_epochs = T["max_epochs"] if max_epochs is None else max_epochs
    held_out = graphs[held_out_name]

    torch.manual_seed(seed)
    random.seed(seed)

    model = TSPN(config.MODEL)
    optimizer = torch.optim.Adam(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T["T_0"], T_mult=T["T_mult"])

    logger = RunLogger(use_wandb, project="tspn", run_name=f"fold{fold_idx}_{held_out_name}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"tspn_fold{fold_idx}_best.pt")

    best_val_rmse = float("inf")
    patience_counter = 0
    last_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        random.shuffle(train_names)
        epoch_loss = 0.0

        for name in train_names:
            event = graphs[name]
            seq = augment_temporal_jitter(event.temporal_sequence)
            seq[7] = augment_shock_magnitude(seq[7])
            seq[7] = augment_edge_dropout(seq[7])
            aug_labels = augment_label_noise(event.y)

            pred, alpha = model(seq)
            loss = compute_loss(pred, aug_labels, event.label_mask, alpha)

            if torch.isnan(loss):
                raise RuntimeError(
                    f"CP35: NaN loss at fold {fold_idx} epoch {epoch} event {name} -- "
                    f"pred max={pred.abs().max().item():.4f}, "
                    f"label max={event.y[event.label_mask].abs().max().item():.4f}"
                )

            optimizer.zero_grad()
            loss.backward()
            clip_norm = T["grad_clip_warmup_norm"] if epoch < T["grad_clip_warmup_epochs"] else T["grad_clip_norm"]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step(epoch)

        val_rmse_6m, _, val_alpha = validate(model, held_out)
        last_epoch = epoch

        logger.log({
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_names),
            "val_rmse_6m": val_rmse_6m,
            "lr": scheduler.get_last_lr()[0],
            "grad_clip_norm": clip_norm,
        })

        if val_rmse_6m < best_val_rmse:
            best_val_rmse = val_rmse_6m
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "val_rmse_6m": val_rmse_6m, "model_state": model.state_dict()},
                ckpt_path,
            )
            np.save(
                os.path.join(RESULTS_TABLES, f"attention_fold{fold_idx}.npy"),
                val_alpha.detach().cpu().numpy(),
            )
        else:
            patience_counter += 1
            if patience_counter >= T["early_stop_patience"]:
                break

    logger.save_csv(os.path.join(RESULTS_TABLES, f"train_log_fold{fold_idx}.csv"))
    logger.finish()

    # CP36: reload the BEST checkpoint (not the last-epoch model still in memory),
    # print its metadata to confirm what's actually being evaluated.
    ckpt = torch.load(ckpt_path, weights_only=False)
    print(f"  Fold {fold_idx}: loading best checkpoint -- "
          f"epoch {ckpt['epoch']}, val_rmse_6m={ckpt['val_rmse_6m']:.4f} "
          f"(trained {last_epoch + 1} epochs total)")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        final_pred, _ = model(held_out.temporal_sequence)

    metrics = evaluate.record_all_metrics(final_pred, held_out, fold_idx, model_name="TSPN")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=None, help="override config.TRAINING['max_epochs']")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="specific fold indices to run (default: all 6)")
    parser.add_argument("--use_wandb", action="store_true", help="attempt wandb logging (best-effort)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading PyG event graphs...")
    graphs = load_all_graphs()

    folds = args.folds if args.folds is not None else list(range(6))
    for fold_idx in folds:
        held_out_name = EVENT_NAMES[fold_idx]
        print(f"\n{'='*60}\nFold {fold_idx}: held out = {held_out_name}\n{'='*60}")
        metrics = train_one_fold(
            fold_idx, graphs, max_epochs=args.max_epochs, use_wandb=args.use_wandb, seed=args.seed
        )
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    print(f"\nDone. TSPN training recorded to results/tables/all_results.csv "
          f"for {len(folds)} fold(s).")


if __name__ == "__main__":
    main()
