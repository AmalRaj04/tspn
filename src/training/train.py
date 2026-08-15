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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_makedirs(path: str) -> None:
    """os.makedirs(path, exist_ok=True) that also tolerates FileExistsError --
    exist_ok=True alone isn't sufficient when path is a symlink into Colab's
    Drive FUSE mount (hit for real mid-run, see RunLogger.log())."""
    try:
        os.makedirs(path, exist_ok=True)
    except FileExistsError:
        pass
RESULTS_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")


class RunLogger:
    """Logs to a CSV that is rewritten (small file, cheap) after EVERY epoch,
    not just once at the end of a fold -- a fold killed mid-training (as
    actually happened once already, see PROJECT_STATE.md finding #33) must
    not lose its loss-curve history for the epochs it did complete, the same
    resume-safety principle already applied to the resume-state JSON and
    checkpoint files. Opportunistically also logs to wandb if requested and
    available; never lets wandb break training."""

    def __init__(self, use_wandb: bool, project: str, run_name: str, csv_path: str):
        self.csv_path = csv_path
        self.rows: list[dict] = []
        if os.path.exists(csv_path):
            self._existing_rows = pd.read_csv(csv_path).to_dict("records")
        else:
            self._existing_rows = []
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
        _safe_makedirs(os.path.dirname(self.csv_path))
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


def load_all_graphs() -> dict[str, TSPNEventGraph]:
    """Loads each event's PyG dataset (saved as plain CPU tensors during
    Phase 5) and moves everything onto DEVICE once, up front -- so every
    downstream forward/backward pass runs on GPU when one is available
    (e.g. Colab's T4) without needing per-call .to(DEVICE) calls scattered
    through the training loop."""
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
    resume: bool = True,
) -> dict:
    """Compute-management note (Locked Plan §9.2): a full 200-epoch x 6-fold run
    on this CPU-only setup is a multi-day operation, so resuming after an
    interruption matters in practice, not just in theory. Resume state is a
    small JSON sidecar saved after every epoch (epoch, best_val_rmse,
    patience_counter, done) -- separate from the (larger) best-model
    checkpoint, which only saves on improvement. On resume: reload the best
    model's WEIGHTS from the checkpoint, but always build a FRESH optimizer
    and scheduler (per §9.2's own explicit guidance -- "do not try to reload
    optimizer state"), and continue the epoch loop and early-stopping counters
    from the sidecar. If a fold's sidecar says done=True, it's skipped
    entirely and only re-evaluated -- safe to re-invoke this script freely."""
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

    model = TSPN(config.MODEL).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T["T_0"], T_mult=T["T_mult"])

    _safe_makedirs(CKPT_DIR)
    _safe_makedirs(RESULTS_TABLES)
    ckpt_path = os.path.join(CKPT_DIR, f"tspn_fold{fold_idx}_best.pt")
    resume_state_path = os.path.join(CKPT_DIR, f"tspn_fold{fold_idx}_resume_state.json")

    best_val_rmse = float("inf")
    patience_counter = 0
    start_epoch = 0

    if resume and os.path.exists(resume_state_path):
        with open(resume_state_path) as f:
            state = json.load(f)
        # "Done" is evaluated fresh against the CURRENT call's max_epochs, not
        # a boolean frozen at save time -- early stopping means genuinely done
        # regardless of max_epochs, but merely having reached a PRIOR run's
        # (possibly smaller) max_epochs must not block a later run that asks
        # for more epochs. (Caught by testing: an earlier version stored a
        # single frozen `done` flag and incorrectly refused to resume past a
        # prior smoke test's smaller epoch cap.)
        already_early_stopped = state.get("early_stopped", False)
        budget_exhausted = (state["epoch"] + 1) >= max_epochs
        if already_early_stopped or budget_exhausted:
            reason = "early-stopped" if already_early_stopped else f"reached max_epochs={max_epochs}"
            print(f"  Fold {fold_idx}: resume state says already done ({reason} at "
                  f"epoch {state['epoch']}, val_rmse_6m={state['best_val_rmse']:.4f}) -- "
                  f"skipping training, evaluating existing checkpoint only.")
            start_epoch = max_epochs   # skip the training loop entirely
        elif os.path.exists(ckpt_path):
            # map_location=DEVICE: a checkpoint saved on one device (e.g. CPU
            # locally) must load cleanly on another (e.g. Colab's GPU) --
            # portable checkpoints are the whole point of moving environments.
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            best_val_rmse = state["best_val_rmse"]
            patience_counter = state["patience_counter"]
            start_epoch = state["epoch"] + 1
            print(f"  Fold {fold_idx}: RESUMING from epoch {start_epoch} "
                  f"(best_val_rmse={best_val_rmse:.4f}, patience={patience_counter}/{T['early_stop_patience']}, "
                  f"fresh optimizer/scheduler per §9.2)")

    train_log_path = os.path.join(RESULTS_TABLES, f"train_log_fold{fold_idx}.csv")
    logger = RunLogger(use_wandb, project="tspn", run_name=f"fold{fold_idx}_{held_out_name}", csv_path=train_log_path)

    for epoch in range(start_epoch, max_epochs):
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

        logger.log({
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_names),
            "val_rmse_6m": val_rmse_6m,
            "lr": scheduler.get_last_lr()[0],
            "grad_clip_norm": clip_norm,
        })

        stopping = False
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
                stopping = True

        # Resume-state sidecar, updated every epoch (§9.2) -- cheap relative to
        # an epoch's training cost, and is what makes a mid-run interruption
        # recoverable without losing the epoch/patience bookkeeping.
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

    # CP36: reload the BEST checkpoint (not the last-epoch model still in memory),
    # print its metadata to confirm what's actually being evaluated.
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    with open(resume_state_path) as f:
        total_epochs_run = json.load(f)["epoch"] + 1
    print(f"  Fold {fold_idx}: loading best checkpoint -- "
          f"epoch {ckpt['epoch']}, val_rmse_6m={ckpt['val_rmse_6m']:.4f} "
          f"(trained {total_epochs_run} epochs total, across resumes if any)")
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
    parser.add_argument("--no_resume", action="store_true",
                         help="ignore any existing resume-state/checkpoint and retrain each fold from scratch")
    args = parser.parse_args()

    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print("Loading PyG event graphs...")
    graphs = load_all_graphs()

    folds = args.folds if args.folds is not None else list(range(6))
    for fold_idx in folds:
        held_out_name = EVENT_NAMES[fold_idx]
        print(f"\n{'='*60}\nFold {fold_idx}: held out = {held_out_name}\n{'='*60}")
        metrics = train_one_fold(
            fold_idx, graphs, max_epochs=args.max_epochs, use_wandb=args.use_wandb,
            seed=args.seed, resume=not args.no_resume,
        )
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    print(f"\nDone. TSPN training recorded to results/tables/all_results.csv "
          f"for {len(folds)} fold(s).")


if __name__ == "__main__":
    main()
