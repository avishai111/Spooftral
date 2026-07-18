"""
utils/training_helpers_audio_adapter_head.py
=============================================
Training and checkpoint helpers for the Audio-Adapter + MLP Classification
Head paradigm (AudioAdapterHeadClassifier).

This module mirrors the CAG training_evaluate_classification.py interface
but drives the classification-head forward pass instead of scoring label
log-probabilities.

Key functions
-------------
save_epoch_checkpoint()
    Atomic two-pass checkpoint strategy:
      Pass 1 (before eval): save weights immediately after the train epoch so
        a crash during evaluation does not lose the epoch's gradient update.
      Pass 2 (after eval): overwrite checkpoint-last.pt with updated eval-epoch
        trackers; epoch-N file is NOT rewritten (weights are unchanged).

train_one_epoch()
    Single epoch loop: forward pass → CE loss → backward → gradient clip →
    optimizer step → scheduler step.  Handles gradient accumulation and
    NaN-loss detection.

evaluate_epoch()
    Inference-only loop over a DataLoader, collecting logits and labels for
    EER / accuracy / F1 computation via helper_function.compute_eer.

log_dev_metrics() / log_eval_metrics()
    Append per-epoch EER and classification metrics to the run's log files
    and W&B, then save best-checkpoint artefacts when EER improves.
"""
import glob
import logging
import os
from typing import Any, Iterable, Optional, Tuple

from utils.helper_function import log_metrics


def save_epoch_checkpoint(
    *,
    epoch: int,
    global_step: int,
    best_dev_eer: float,
    best_eval_eer: float,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    config_dict: dict,
    output_dir: str,
    save_best_only: bool,
    logger: logging.Logger,
    save_checkpoint,
    last_eval_epoch: Optional[int] = None,
    last_test_eval_epoch: Optional[int] = None,
    save_epoch_file: bool = True,
    cached_state_dict: Optional[dict] = None,
) -> None:
    """
    Save the rolling checkpoint for the completed epoch.

    Called twice per epoch:
      1. BEFORE evaluations (save_epoch_file=True):
         Persists model weights immediately after training ends.  A crash during
         the subsequent evaluation will not lose the epoch's trained weights —
         the process resumes from this checkpoint, detects the missing eval via
         last_eval_epoch / last_test_eval_epoch, and re-runs it automatically.

      2. AFTER evaluations (save_epoch_file=False):
         Overwrites checkpoint-last.pt with updated last_eval_epoch /
         last_test_eval_epoch so the next resume knows which evals completed.
         checkpoint-epochN.pt is NOT re-written (the weights are identical).

    cached_state_dict
    -----------------
    Pass the result of model.state_dict() computed once before Pass 1.
    Both Pass 1 and Pass 2 (and log_dev/eval_metrics) reuse the same dict so
    model.state_dict() is called exactly once per epoch instead of 3-4 times.
    If None, the function calls model.state_dict() itself (fallback for callers
    that don't cache, e.g. the initial_eval_epochs block).

    Checkpoint epoch field
    ----------------------
    We store epoch+1 (the next epoch to resume from).  load_checkpoint_if_available
    treats a checkpoint without epoch_completed as "legacy: epoch field is already
    the next epoch to run", so resuming from this file starts at the correct epoch.

    Eval tracker fields
    -------------------
    last_eval_epoch / last_test_eval_epoch store epoch+1 after each split's eval
    completes (matching the convention in training_evaluate_classification.py).
    On resume, load_checkpoint_if_available reads these back; the resume block in
    main() compares them to start_epoch to detect which evals were missed.
    """
    payload = {
        "epoch": epoch + 1,  # next epoch to run on resume (legacy-compatible)
        "global_step": int(global_step),
        "best_dev_eer": float(best_dev_eer),
        "best_eval_eer": float(best_eval_eer),
        "model_state_dict": cached_state_dict if cached_state_dict is not None else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "configs": config_dict,
        # Eval trackers: None means "not evaluated yet" (or legacy checkpoint).
        # Set to epoch+1 after each split's evaluation completes.
        "last_eval_epoch": last_eval_epoch,
        "last_test_eval_epoch": last_test_eval_epoch,
    }

    # checkpoint-last.pt always reflects the most recent state and is what a
    # resume run looks for first.
    last_path = os.path.join(output_dir, "checkpoint-last.pt")
    save_checkpoint(last_path, payload)

    if save_epoch_file:
        # Epoch marker file lets shell scripts quickly read the last completed
        # epoch without deserialising the full .pt file.
        try:
            with open(os.path.join(output_dir, "checkpoint-last.epoch"), "w") as f:
                f.write(f"{epoch + 1}\n")
        except Exception as exc:
            logger.warning(f"Failed to write checkpoint-last epoch marker: {exc}")

        # save_best_only trims disk usage by skipping the per-epoch snapshot, keeping
        # only the rolling checkpoint-last.pt.
        if not save_best_only:
            epoch_path = os.path.join(output_dir, f"checkpoint-epoch{epoch + 1}.pt")
            save_checkpoint(epoch_path, payload)
            logger.info(f"Saved rolling checkpoints: {last_path}, {epoch_path}")
        else:
            logger.info(f"Saved rolling checkpoint: {last_path}")
    else:
        logger.info(f"Updated checkpoint-last.pt with eval trackers: {last_path}")


# Coerces a config-provided list of epoch indices (possibly strings/floats/junk)
# into a sorted list of unique non-negative ints.
def normalize_initial_epochs(val: Optional[Iterable[Any]]) -> list[int]:
    epochs = []
    for item in val or []:
        try:
            epoch = int(item)
        except (TypeError, ValueError):
            continue
        if epoch >= 0:
            epochs.append(epoch)
    return sorted(set(epochs))


# Returns the most recently modified file matching pattern in output_dir, or
# None if no files match.
def latest_metrics_file(output_dir: str, pattern: str) -> Optional[str]:
    files = glob.glob(os.path.join(output_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def last_logged_epoch(log_path: Optional[str]) -> Optional[int]:
    """
    Parse the last epoch number logged in a metrics file.
    Returns integer epoch_index (0-based) or None if not found.
    """
    if not log_path or not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except Exception:
        return None
    for line in reversed(lines):
        line = line.strip()
        # Log headers are written as "Epoch {epoch+1}:" (1-indexed for readability);
        # subtract 1 to recover the 0-indexed epoch.
        if line.lower().startswith("epoch"):
            try:
                parts = line.split()
                num_str = parts[1].rstrip(":")
                return int(num_str) - 1
            except Exception:
                continue
    return None


def log_dev_metrics(
    metrics: Optional[dict],
    *,
    epoch: Optional[int],
    label: Optional[str],
    metrics_log_file: str,
    logger: logging.Logger,
    wandb_run: Any,
    global_step: int,
    best_dev_eer: float,
    best_dev_path: Optional[str],
    output_dir: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any = None,
    config_dict: dict,
    start_epoch: int,
    save_checkpoint,
    cached_state_dict: Optional[dict] = None,
) -> Tuple[float, Optional[str]]:
    if not metrics:
        return best_dev_eer, best_dev_path
    log_metrics(metrics_log_file, epoch=epoch, metrics=metrics)
    prefix = f"{label} " if label else ""
    if epoch is not None:
        logger.info(f"{prefix}Dev metrics (epoch {epoch + 1}): {metrics}")
    else:
        logger.info(f"{prefix}Dev metrics: {metrics}")
    if wandb_run:
        wandb_run.log({f"dev/{k}": v for k, v in metrics.items()}, step=global_step)
    if metrics.get("eer", float("inf")) < best_dev_eer:
        best_dev_eer = metrics["eer"]
        payload = {
            "epoch": epoch + 1 if epoch is not None else start_epoch + 1,
            "global_step": int(global_step),
            "best_dev_eer": float(best_dev_eer),
            "model_state_dict": cached_state_dict if cached_state_dict is not None else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "configs": config_dict,
        }
        best_dev_path = os.path.join(output_dir, "best_adapter_head_dev.pt")
        save_checkpoint(best_dev_path, payload)
        logger.info(f"Saved new best DEV checkpoint to {best_dev_path}")
    return best_dev_eer, best_dev_path


def log_eval_metrics(
    metrics: Optional[dict],
    *,
    epoch: Optional[int],
    label: Optional[str],
    eval_log_file: str,
    logger: logging.Logger,
    wandb_run: Any,
    global_step: int,
    best_eval_eer: float,
    best_eval_path: Optional[str],
    output_dir: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any = None,
    config_dict: dict,
    start_epoch: int,
    save_checkpoint,
    cached_state_dict: Optional[dict] = None,
) -> Tuple[float, Optional[str]]:
    if not metrics:
        return best_eval_eer, best_eval_path
    log_metrics(eval_log_file, epoch=epoch, metrics=metrics)
    prefix = f"{label} " if label else ""
    if epoch is not None:
        logger.info(f"{prefix}Eval metrics (epoch {epoch + 1}): {metrics}")
    else:
        logger.info(f"{prefix}Eval metrics: {metrics}")
    if wandb_run:
        wandb_run.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)
    if metrics.get("eer", float("inf")) < best_eval_eer:
        best_eval_eer = metrics["eer"]
        payload = {
            "epoch": epoch + 1 if epoch is not None else start_epoch + 1,
            "global_step": int(global_step),
            "best_eval_eer": float(best_eval_eer),
            "model_state_dict": cached_state_dict if cached_state_dict is not None else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "configs": config_dict,
        }
        best_eval_path = os.path.join(output_dir, "best_adapter_head_eval.pt")
        save_checkpoint(best_eval_path, payload)
        logger.info(f"Saved new best EVAL checkpoint to {best_eval_path}")
    return best_eval_eer, best_eval_path
