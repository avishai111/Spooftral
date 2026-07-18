"""
checkpoint_utils.py
===================
Checkpoint auto-discovery, loading, and resume utilities for Voxtral training.

Two public functions are exposed:
  split_metrics_example         : Strip inline example data from a metrics dict.
  resume_checkpoint_evaluation  : Re-run evaluation immediately after resume.
  load_checkpoint_if_available  : Auto-discover and fully restore training state.

Module-level constants
----------------------
_EPOCH_PAT : compiled regex matching "checkpoint-epochN.pt" file names.
_STEP_PAT  : compiled regex matching "checkpoint-stepN.pt" file names.
Both are compiled once at import time to avoid repeated compilation overhead
inside the checkpoint-discovery loop.
"""

import os
import re
import random
from typing import Optional
import numpy as np
import torch

# Pre-compiled regexes for checkpoint filename parsing.
# Compiled at module level so load_checkpoint_if_available can call .search()
# without recompiling on every invocation.
_EPOCH_PAT = re.compile(r"checkpoint-epoch(\d+)\.pt$")
_STEP_PAT  = re.compile(r"checkpoint-step(\d+)\.pt$")

from utils.helper_function import log_eval_stats_table


# ---------------------------------------------------------------------------
# split_metrics_example
# ---------------------------------------------------------------------------

def _to_byte_tensor(s):
    if isinstance(s, torch.Tensor):
        return s.cpu().byte()
    return torch.ByteTensor(bytearray(s))


def split_metrics_example(metrics_dict):
    """
    Separate metrics payload from inline example data.

    evaluate_epoch_classification_as_generation can optionally embed a decoded
    example prediction inside the returned metrics dict.  Most logging helpers
    expect a scalar-valued metrics dict; this function extracts the example
    entry so callers can log them separately.

    Parameters
    ----------
    metrics_dict : dict or None
        The dict returned by an evaluate_epoch function.  May contain an
        "example" or "example_prediction" key.

    Local variables
    ---------------
    example      : dict or None — the extracted example entry.
    metrics_clean: dict — metrics without the example keys.

    Returns
    -------
    metrics_clean : dict
        A copy of metrics_dict with "example" and "example_prediction" removed.
        Returns an empty dict if metrics_dict is None.
    example       : dict or None
        The example dict that was embedded in metrics_dict, or None if absent.
    """
    if metrics_dict is None:
        return {}, None
    # Check both key names for backwards compatibility.
    example = metrics_dict.get("example", None)
    if example is None:
        example = metrics_dict.get("example_prediction", None)
    # Build a clean copy without the example keys.
    metrics_clean = {
        k: v
        for k, v in metrics_dict.items()
        if k not in ("example", "example_prediction")
    }
    return metrics_clean, example


# ---------------------------------------------------------------------------
# resume_checkpoint_evaluation
# ---------------------------------------------------------------------------

def resume_checkpoint_evaluation(
    model, evaluate_epoch, processor, DEVICE,
    start_epoch, optimizer_step, run_by_epochs,
    dev_loader, train_eval_loader, eval_loader, asvspoof2021_eval_loader,
    eval_use_amp, eval_amp_dtype, debug_log_path, eval_debug_log_path,
    enable_debug_logging, log_eval_examples,
    resume_dev_eval_needed, resume_train_eval_needed, resume_test_eval_needed, resume_asvspoof2021_eval_needed,
    log_metrics, metrics_log_file, train_eval_log_file, eval_log_file, asvspoof2021_eval_log_file,
    log_wandb, print_asvspoof2021_info, configs, log_eval_stats_enabled,
    last_eval_epoch, last_eval_step, last_train_eval_epoch, last_train_eval_step,
    last_test_eval_epoch, last_test_eval_step, last_asvspoof2021_eval_epoch, last_asvspoof2021_eval_step,
    generation_cfg=None,
):
    """
    Run evaluation immediately after resuming from a checkpoint.

    When training resumes from a saved checkpoint the model may have already
    been evaluated at that exact epoch/step in a prior run.  Calling this
    function re-runs those evaluations so the W&B dashboard and log files have
    a complete record before training continues.  Each eval split is guarded by
    its own resume_*_needed flag so only the splits that were actually needed
    are evaluated.

    Parameters
    ----------
    model                     : nn.Module — the loaded model wrapper.
    evaluate_epoch            : callable — the evaluation function (CAG scorer).
    processor                 : HF Processor — tokenizer/feature extractor.
    DEVICE                    : torch.device — target compute device.
    start_epoch               : int — epoch number of the loaded checkpoint.
    optimizer_step            : int — global step of the loaded checkpoint.
    run_by_epochs             : bool — True if training is epoch-based,
                                False if step-based.
    dev_loader                : DataLoader — development set loader.
    train_eval_loader         : DataLoader or None — training-set eval loader.
    eval_loader               : DataLoader or None — test set loader.
    asvspoof2021_eval_loader  : DataLoader or None — ASVspoof2021 loader.
    eval_use_amp              : bool — enable AMP during evaluation.
    eval_amp_dtype            : str — "fp16" or "bf16".
    debug_log_path            : str or None — path for dev-set debug logs.
    eval_debug_log_path       : str or None — path for test-set debug logs.
    enable_debug_logging      : bool — enable verbose debug prints.
    log_eval_examples         : bool — whether to log decoded example predictions.
    resume_dev_eval_needed    : bool — True if dev evaluation should run.
    resume_train_eval_needed  : bool — True if train-set evaluation should run.
    resume_test_eval_needed   : bool — True if test-set evaluation should run.
    resume_asvspoof2021_eval_needed : bool — True if ASVspoof2021 eval should run.
    log_metrics               : callable or None — function to write metrics to disk.
    metrics_log_file          : str — path to the dev metrics log file.
    train_eval_log_file       : str or None — path to the train-eval log file.
    eval_log_file             : str or None — path to the test-set log file.
    asvspoof2021_eval_log_file: str or None — path to the ASVspoof2021 log file.
    log_wandb                 : callable — function to log a payload dict to W&B.
    print_asvspoof2021_info   : callable — prints context info for ASVspoof2021 eval.
    configs                   : config object — training config with feature flags.
    log_eval_stats_enabled    : bool — whether to collect per-attack/codec stats.
    last_eval_epoch           : int — last dev-eval epoch before this resume.
    last_eval_step            : int — last dev-eval step before this resume.
    last_train_eval_epoch     : int — last train-eval epoch before this resume.
    last_train_eval_step      : int — last train-eval step before this resume.
    last_test_eval_epoch      : int — last test-eval epoch before this resume.
    last_test_eval_step       : int — last test-eval step before this resume.
    last_asvspoof2021_eval_epoch : int — last ASVspoof2021 eval epoch.
    last_asvspoof2021_eval_step  : int — last ASVspoof2021 eval step.
    generation_cfg            : dict or None — CAG scoring config forwarded to
                                evaluate_epoch.

    Local variables
    ---------------
    resume_eval_step      : int or None — step for tqdm label (None in epoch mode).
    resume_desc           : str — human-readable "epoch N" or "step N" string.
    resume_metrics        : dict — raw metrics from dev evaluation.
    resume_example        : dict or None — decoded dev example.
    resume_error          : Exception or None — dev eval exception (if any).
    resume_metrics_clean  : dict — dev metrics without example entries.
    resume_dev_log_payload: dict — dev metrics prefixed with "val/".
    train_eval_metrics    : dict — raw metrics from train-set evaluation.
    train_eval_example    : dict or None — decoded train example.
    train_eval_metrics_clean : dict — train metrics without example entries.
    resume_train_log_payload : dict — train metrics prefixed with "train_eval/".
    eval_metrics          : dict — raw metrics from test-set evaluation.
    eval_example          : dict or None — decoded test example.
    eval_stats            : dict or None — per-attack/codec eval stats.
    eval_metrics_clean    : dict — test metrics without example entries.
    resume_test_log_payload : dict — test metrics prefixed with "eval/".
    asv2021_metrics       : dict — raw ASVspoof2021 metrics.
    asv2021_example       : dict or None — decoded ASVspoof2021 example.
    asv2021_metrics_clean : dict — ASVspoof2021 metrics without example entries.
    resume_asv2021_log_payload : dict — ASVspoof2021 metrics prefixed with
                                 "asvspoof2021/".

    Returns
    -------
    tuple of 8 ints
        Updated last-eval tracking counters in the order:
        (last_eval_epoch, last_eval_step,
         last_train_eval_epoch, last_train_eval_step,
         last_test_eval_epoch, last_test_eval_step,
         last_asvspoof2021_eval_epoch, last_asvspoof2021_eval_step).
        These are passed back to the training loop to prevent duplicate
        evaluations at the resumed step.
    """
    model.eval()
    # Choose between step-based and epoch-based identifiers for log messages.
    resume_eval_step = optimizer_step if not run_by_epochs else None
    resume_desc = f"epoch {start_epoch}" if run_by_epochs else f"step {resume_eval_step}"
    print(f"[Resume] Evaluating loaded checkpoint for {resume_desc} before continuing training...")

    # --- Dev set evaluation ---
    if resume_dev_eval_needed:
        resume_metrics = None
        resume_example = None
        resume_error = None
        try:
            resume_metrics = evaluate_epoch(
                model, dev_loader, processor, DEVICE,
                use_amp=eval_use_amp, amp_dtype=eval_amp_dtype,
                debug_log_path=debug_log_path, eval_step=resume_eval_step,
                enable_debug_logging=enable_debug_logging,
                return_example=log_eval_examples,
                generation_cfg=generation_cfg,
            )
            resume_metrics_clean, resume_example = split_metrics_example(resume_metrics)
            resume_dev_log_payload = {f"val/{k}": v for k, v in resume_metrics_clean.items()}
            print(f"[Resume Eval] Dev  {resume_desc}: {resume_dev_log_payload}")

            if log_metrics is not None:
                if run_by_epochs:
                    log_metrics(log_file=metrics_log_file, epoch=start_epoch - 1, metrics=resume_metrics_clean, examples=resume_example)
                else:
                    log_metrics(log_file=metrics_log_file, step=optimizer_step, metrics=resume_metrics_clean, examples=resume_example)
            log_wandb(resume_dev_log_payload, step=optimizer_step)
            if resume_example:
                print(f"[Resume Eval] Dev example: {resume_example}")
            # Advance the last-eval tracker so the main loop skips this step.
            if run_by_epochs:
                last_eval_epoch = max(last_eval_epoch or 0, start_epoch)
            else:
                last_eval_step = max(last_eval_step or 0, optimizer_step)
        except Exception as exc:
            resume_error = exc
            print(f"[Resume Eval] Dev {resume_desc} failed: {exc}")
            log_wandb({"val/error": str(exc)}, step=optimizer_step)

    # --- Train-set evaluation ---
    if (
        resume_train_eval_needed
        and configs is not None
        and getattr(configs, "eval_on_training_set_during_training", False)
        and train_eval_loader is not None
    ):
        train_eval_metrics = None
        train_eval_example = None
        try:
            train_eval_metrics = evaluate_epoch(
                model, train_eval_loader, processor, DEVICE,
                use_amp=eval_use_amp, amp_dtype=eval_amp_dtype,
                debug_log_path=debug_log_path, eval_step=resume_eval_step,
                enable_debug_logging=enable_debug_logging,
                return_example=log_eval_examples,
                generation_cfg=generation_cfg,
            )
            train_eval_metrics_clean, train_eval_example = split_metrics_example(train_eval_metrics)
            resume_train_log_payload = {f"train_eval/{k}": v for k, v in train_eval_metrics_clean.items()}
            print(f"[Resume Eval] Train {resume_desc}: {resume_train_log_payload}")
            if log_metrics is not None and train_eval_log_file is not None:
                if run_by_epochs:
                    log_metrics(log_file=train_eval_log_file, epoch=start_epoch - 1, metrics=train_eval_metrics_clean, examples=train_eval_example)
                else:
                    log_metrics(log_file=train_eval_log_file, step=optimizer_step, metrics=train_eval_metrics_clean, examples=train_eval_example)
            log_wandb(resume_train_log_payload, step=optimizer_step)
            if train_eval_example:
                print(f"[Resume Eval] Train example: {train_eval_example}")
            if run_by_epochs:
                last_train_eval_epoch = max(last_train_eval_epoch or 0, start_epoch)
            else:
                last_train_eval_step = max(last_train_eval_step or 0, optimizer_step)
        except Exception as exc:
            print(f"[Resume Eval] Train {resume_desc} failed: {exc}")
            log_wandb({"train_eval/error": str(exc)}, step=optimizer_step)

    # --- Test (eval) set evaluation ---
    if (
        resume_test_eval_needed
        and configs is not None
        and getattr(configs, "eval_on_test_during_training", False)
        and eval_loader is not None
    ):
        eval_metrics = None
        eval_example = None
        eval_stats = None
        try:
            eval_metrics = evaluate_epoch(
                model, eval_loader, processor, DEVICE,
                use_amp=eval_use_amp, amp_dtype=eval_amp_dtype,
                debug_log_path=eval_debug_log_path, eval_step=resume_eval_step,
                enable_debug_logging=enable_debug_logging,
                return_example=log_eval_examples,
                return_eval_stats=log_eval_stats_enabled,
                generation_cfg=generation_cfg,
            )
            # Extract per-attack/codec stats before cleaning the metrics dict.
            if log_eval_stats_enabled and isinstance(eval_metrics, dict):
                eval_stats = eval_metrics.pop("eval_stats", None)
            eval_metrics_clean, eval_example = split_metrics_example(eval_metrics)
            resume_test_log_payload = {f"eval/{k}": v for k, v in eval_metrics_clean.items()}
            print(f"[Resume Eval] Eval {resume_desc}: {resume_test_log_payload}")
            if log_metrics is not None:
                if run_by_epochs:
                    log_metrics(log_file=eval_log_file, epoch=start_epoch - 1, metrics=eval_metrics_clean, examples=eval_example)
                else:
                    log_metrics(log_file=eval_log_file, step=optimizer_step, metrics=eval_metrics_clean, examples=eval_example)
            log_wandb(resume_test_log_payload, step=optimizer_step)
            # Log the per-attack/codec table to the eval log file if available.
            if eval_stats and eval_log_file is not None:
                log_eval_stats_table(
                    log_file=eval_log_file,
                    header=f"Resume {resume_desc} (eval set)",
                    threshold=eval_stats.get("threshold_at_eer"),
                    attack_stats=eval_stats.get("attack"),
                    codec_stats=eval_stats.get("codec"),
                )
            if eval_example:
                print(f"[Resume Eval] Eval example: {eval_example}")
            if run_by_epochs:
                last_test_eval_epoch = max(last_test_eval_epoch or 0, start_epoch)
            else:
                last_test_eval_step = max(last_test_eval_step or 0, optimizer_step)
        except Exception as exc:
            print(f"[Resume Eval] Eval {resume_desc} failed: {exc}")
            log_wandb({"eval/error": str(exc)}, step=optimizer_step)

    # --- ASVspoof2021 evaluation ---
    if (
        resume_asvspoof2021_eval_needed
        and configs is not None
        and getattr(configs, "eval_on_asvspoof2021_during_training", False)
        and asvspoof2021_eval_loader is not None
    ):
        print_asvspoof2021_info(f"resume {resume_desc}")
        asv2021_metrics = None
        asv2021_example = None
        try:
            asv2021_metrics = evaluate_epoch(
                model, asvspoof2021_eval_loader, processor, DEVICE,
                use_amp=eval_use_amp, amp_dtype=eval_amp_dtype,
                debug_log_path=eval_debug_log_path, eval_step=resume_eval_step,
                enable_debug_logging=enable_debug_logging,
                return_example=log_eval_examples,
                generation_cfg=generation_cfg,
            )
            asv2021_metrics_clean, asv2021_example = split_metrics_example(asv2021_metrics)
            resume_asv2021_log_payload = {f"asvspoof2021/{k}": v for k, v in asv2021_metrics_clean.items()}
            print(f"[Resume Eval] ASVspoof2021 {resume_desc}: {resume_asv2021_log_payload}")
            if log_metrics is not None and asvspoof2021_eval_log_file is not None:
                if run_by_epochs:
                    log_metrics(log_file=asvspoof2021_eval_log_file, epoch=start_epoch - 1, metrics=asv2021_metrics_clean, examples=asv2021_example)
                else:
                    log_metrics(log_file=asvspoof2021_eval_log_file, step=optimizer_step, metrics=asv2021_metrics_clean, examples=asv2021_example)
            log_wandb(resume_asv2021_log_payload, step=optimizer_step)
            if asv2021_example:
                print(f"[Resume Eval] ASVspoof2021 example: {asv2021_example}")
            if run_by_epochs:
                last_asvspoof2021_eval_epoch = max(last_asvspoof2021_eval_epoch or 0, start_epoch)
            else:
                last_asvspoof2021_eval_step = max(last_asvspoof2021_eval_step or 0, optimizer_step)
        except Exception as exc:
            print(f"[Resume Eval] ASVspoof2021 {resume_desc} failed: {exc}")
            log_wandb({"asvspoof2021/error": str(exc)}, step=optimizer_step)

    # Return the updated last-eval trackers so the main loop can skip duplicate evals.
    return (
        last_eval_epoch, last_eval_step,
        last_train_eval_epoch, last_train_eval_step,
        last_test_eval_epoch, last_test_eval_step,
        last_asvspoof2021_eval_epoch, last_asvspoof2021_eval_step,
    )


# ---------------------------------------------------------------------------
# load_checkpoint_if_available
# ---------------------------------------------------------------------------

def load_checkpoint_if_available(
    model,
    optimizer,
    scheduler,
    scaler,
    configs,
    wandb_enabled,
    wandb_run,
    resume_ckpt_path,
    output_dir,
    device,
    steps_per_epoch,
    start_epoch=0,
    start_step=0,
    best_eer=100.0,
    best_dev_eer=100.0,
    best_eval_eer=100.0,
    last_eval_epoch=None,
    last_eval_step=None,
    last_train_eval_epoch=None,
    last_train_eval_step=None,
    last_test_eval_epoch=None,
    last_test_eval_step=None,
    last_asvspoof2021_eval_epoch=None,
    last_asvspoof2021_eval_step=None,
):
    """
    Auto-discover and load a checkpoint, restoring the full training state.

    Discovery priority:
      1. resume_ckpt_path (explicit path to a .pt/.bin file or directory).
      2. output_dir scan for "checkpoint-last.pt" or the highest-numbered
         checkpoint-epochN.pt / checkpoint-stepN.pt file.
    If no checkpoint is found, the function returns immediately with all
    default values so the caller can start training from scratch.

    Restore order:
      1. Model weights (strict=True; falls back through multiple state-dict keys).
      2. PEFT adapters (loaded from an "adapters/" sub-directory if present).
      3. Optimizer state (raises RuntimeError on mismatch).
      4. Scheduler state (raises RuntimeError on mismatch).
      5. GradScaler state (soft warning on failure).
      6. Resume metadata (epoch, step, best EER, last-eval counters).
      7. W&B run reconnect (when wandb_enabled and wandb_run is None).

    Parameters
    ----------
    model                     : nn.Module — the model to restore weights into.
    optimizer                 : Optimizer or None — restored when present in ckpt.
    scheduler                 : LRScheduler or None — restored when present in ckpt.
    scaler                    : GradScaler or None — restored when present in ckpt.
    configs                   : config object — training config; used for wandb init
                                and resume epoch fallback.
    wandb_enabled             : bool — whether W&B logging is active.
    wandb_run                 : wandb.Run or None — existing W&B run handle.
    resume_ckpt_path          : str or None — explicit checkpoint path or directory.
                                Pass None to trigger auto-discovery from output_dir.
    output_dir                : str — directory where checkpoints are saved.
    device                    : torch.device — map_location for torch.load.
    steps_per_epoch           : int — used to derive start_step from start_epoch
                                when no "global_step" key is present in the ckpt.
    start_epoch               : int — default if ckpt has no "epoch" key.
    start_step                : int — default if ckpt has no "global_step" key.
    best_eer                  : float — running best EER (overwritten by ckpt).
    best_dev_eer              : float or None — best dev EER (overwritten by ckpt).
    best_eval_eer             : float or None — best test EER (overwritten by ckpt).
    last_eval_epoch           : int or None — last dev-eval epoch.
    last_eval_step            : int or None — last dev-eval step.
    last_train_eval_epoch     : int or None — last train-eval epoch.
    last_train_eval_step      : int or None — last train-eval step.
    last_test_eval_epoch      : int or None — last test-eval epoch.
    last_test_eval_step       : int or None — last test-eval step.
    last_asvspoof2021_eval_epoch : int or None — last ASVspoof2021 eval epoch.
    last_asvspoof2021_eval_step  : int or None — last ASVspoof2021 eval step.

    Local variables
    ---------------
    ckpt_path      : str or None — resolved path to the checkpoint file to load.
    candidate_dir  : str or None — output_dir if it exists as a directory.
    cand           : list of str — candidate checkpoint file names found in the dir.
    parsed         : list of (kind, number, name) tuples for numbered checkpoints.
    latest_name    : str — file name of the highest-numbered checkpoint found.
    m_epoch        : re.Match or None — regex match for epoch-numbered files.
    m_step         : re.Match or None — regex match for step-numbered files.
    ckpt           : dict or raw state_dict — loaded checkpoint object.
    wandb_id       : str or None — W&B run ID stored in the checkpoint.
    state_options  : list of (key, state_dict) — candidate state-dict entries.
    key            : str — name of the state-dict key tried.
    state          : dict — model state_dict candidate.
    last_error     : Exception or None — last model-load exception.
    model_to_check : nn.Module — model after stripping DDP wrapper.
    adapters_dir   : str — path to PEFT adapters sub-directory.

    Returns
    -------
    dict with keys:
        "ckpt_path"                  : str or None — resolved checkpoint path.
        "wandb_run"                  : wandb.Run or None — possibly reconnected.
        "start_epoch"                : int — epoch to resume from.
        "start_step"                 : int — global step to resume from.
        "best_eer"                   : float — best EER seen so far.
        "best_dev_eer"               : float — best dev EER seen so far.
        "best_eval_eer"              : float — best eval EER seen so far.
        "last_eval_epoch"            : int or None.
        "last_eval_step"             : int or None.
        "last_train_eval_epoch"      : int or None.
        "last_train_eval_step"       : int or None.
        "last_test_eval_epoch"       : int or None.
        "last_test_eval_step"        : int or None.
        "last_asvspoof2021_eval_epoch": int or None.
        "last_asvspoof2021_eval_step" : int or None.
    """
    # --- Step 1: Resolve the checkpoint path ---
    ckpt_path = resume_ckpt_path
    if ckpt_path is None:
        # Auto-discover from output_dir.
        candidate_dir = output_dir if os.path.isdir(output_dir) else None
        if candidate_dir:
            cand = [
                f for f in os.listdir(candidate_dir)
                if f.startswith("checkpoint") and f.endswith((".bin", ".pt", ".ckpt", ".pth"))
            ]
            if cand:
                if "checkpoint-last.pt" in cand:
                    ckpt_path = os.path.join(candidate_dir, "checkpoint-last.pt")
                    print(f"[Auto Resume] Found checkpoint-last.pt, will load: {ckpt_path}")
                else:
                    # Defer directory resolution to the next block.
                    ckpt_path = candidate_dir
                    print(f"[Auto Resume] Found existing checkpoint directory, will try to load latest from: {ckpt_path}")

    # No checkpoint found anywhere — return defaults so training starts fresh.
    if ckpt_path is None:
        return {
            "ckpt_path": None,
            "wandb_run": wandb_run,
            "start_epoch": start_epoch,
            "start_step": start_step,
            "best_eer": best_eer,
            "best_dev_eer": best_dev_eer,
            "best_eval_eer": best_eval_eer,
            "last_eval_epoch": last_eval_epoch,
            "last_eval_step": last_eval_step,
            "last_train_eval_epoch": last_train_eval_epoch,
            "last_train_eval_step": last_train_eval_step,
            "last_test_eval_epoch": last_test_eval_epoch,
            "last_test_eval_step": last_test_eval_step,
            "last_asvspoof2021_eval_epoch": last_asvspoof2021_eval_epoch,
            "last_asvspoof2021_eval_step": last_asvspoof2021_eval_step,
        }

    # --- Step 2: If ckpt_path is a directory, find the best file inside it ---
    if os.path.isdir(ckpt_path):
        cand = [f for f in os.listdir(ckpt_path) if f.endswith((".bin", ".pt", ".ckpt", ".pth"))]
        if not cand:
            raise FileNotFoundError(f"No checkpoint file in directory: {ckpt_path}")

        # Sort all candidates by mtime so the most recently written file wins.
        # "checkpoint-last.pt" is only preferred if it also has the highest mtime.
        # Guard against TOCTOU: a file may disappear between listdir and stat.
        all_mtimes = {}
        for _name in cand:
            try:
                all_mtimes[_name] = os.path.getmtime(os.path.join(ckpt_path, _name))
            except OSError:
                continue  # file disappeared between listdir and stat
        cand = list(all_mtimes.keys())
        if "checkpoint-last.pt" in cand:
            last_mtime = all_mtimes["checkpoint-last.pt"]
            max_mtime_among_all = max(all_mtimes.values())
            if last_mtime >= max_mtime_among_all:
                # checkpoint-last.pt is the newest file (a tie is resolved in its favor) — use it.
                ckpt_path = os.path.join(ckpt_path, "checkpoint-last.pt")
                print(f"[Auto Resume] Loading checkpoint-last.pt (newest by mtime): {ckpt_path}")
            else:
                # A numbered checkpoint is newer; fall through to mtime sort below.
                numbered = [n for n in cand if n != "checkpoint-last.pt"]
                numbered.sort(key=lambda n: all_mtimes[n])
                latest_name = numbered[-1]
                ckpt_path = os.path.join(ckpt_path, latest_name)
                print(f"[Auto Resume] Loading latest checkpoint file (by mtime, newer than checkpoint-last.pt): {ckpt_path}")
        else:
            # No checkpoint-last.pt — sort all parseable epoch/step files by mtime.
            parsed = []
            for name in cand:
                m_epoch = _EPOCH_PAT.search(name)
                m_step = _STEP_PAT.search(name)
                if m_epoch:
                    parsed.append(("epoch", int(m_epoch.group(1)), name))
                elif m_step:
                    parsed.append(("step", int(m_step.group(1)), name))

            if parsed:
                # Sort by mtime (not by epoch/step number) to pick the most recently written.
                parsed.sort(key=lambda x: all_mtimes[x[2]])
                latest_name = parsed[-1][2]
            else:
                # No parseable numbers — fall back to mtime sort over all candidates.
                cand.sort(key=lambda n: all_mtimes[n])
                latest_name = cand[-1]

            ckpt_path = os.path.join(ckpt_path, latest_name)
            print(f"[Auto Resume] Loading latest checkpoint file (by mtime): {ckpt_path}")

    # --- Step 3: Load the checkpoint from disk ---
    print(f"Loading checkpoint from {ckpt_path}")
    # Try the safe weights_only=True path first; fall back for checkpoints that
    # contain non-tensor Python objects (e.g. config dicts pickled before the
    # weights_only flag was introduced).
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        print("[Warning] weights_only=True failed (checkpoint contains non-tensor objects); "
              "falling back to weights_only=False.")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # --- Step 4: Reconnect W&B run if needed ---
    if wandb_enabled and wandb_run is None:
        # Local import avoids importing wandb at module level.
        from utils.config import init_wandb
        wandb_id = ckpt.get("wandb_id")
        if wandb_id is None:
            print("[Resume] W&B: checkpoint has no wandb_id (original run had W&B disabled or predates id saving). Starting a new W&B run.")
        setattr(configs, "wandb_id", wandb_id)
        wandb_run = init_wandb(
            configs,
            project=configs.wandb.project,
            run_name=configs.wandb.name,
            resume=(wandb_id is not None),
            wandb_cfg=configs.wandb,
        )

    # --- Step 5: Load model weights ---
    # Partial-state checkpoints (save_only_adapters / save_only_trainable) store a
    # subset of weights under "adapter_state_dict" or "trainable_state_dict" instead
    # of the full "wrapper_state_dict".  For these we merge the saved subset into the
    # current (HF-initialised) model state and call load_state_dict once with the
    # merged full dict, preserving strict=True.
    _loaded_via_partial_state = False
    _partial_key = None
    if isinstance(ckpt, dict):
        if "adapter_state_dict" in ckpt and "wrapper_state_dict" not in ckpt:
            _partial_key = "adapter_state_dict"
        elif "trainable_state_dict" in ckpt and "wrapper_state_dict" not in ckpt:
            _partial_key = "trainable_state_dict"

    if _partial_key is not None:
        saved_state   = ckpt[_partial_key]
        current_state = model.state_dict()

        # Only check for missing expected keys: unexpected keys are caught by strict=True below.
        if _partial_key == "adapter_state_dict":
            expected_keys = {k for k in current_state if "lora_" in k or "dora_" in k}
        else:  # trainable_state_dict
            expected_keys = {n for n, p in model.named_parameters() if p.requires_grad}
        missing = expected_keys - set(saved_state.keys())
        if missing:
            raise RuntimeError(
                f"[Checkpoint] {_partial_key} is missing {len(missing)} expected key(s). "
                f"First few: {sorted(missing)[:5]}"
            )
        # Merge saved subset into the full current state, then load with strict=True.
        # strict=True catches any unexpected keys left in saved_state.
        model.load_state_dict({**current_state, **saved_state}, strict=True)
        print(f"[Checkpoint] Loaded {_partial_key} (strict=True, {len(saved_state)} keys merged).")
        _loaded_via_partial_state = True
    else:
        # Try multiple state-dict key names for backwards compatibility with older
        # checkpoint formats that used different key conventions.
        state_options = []
        if isinstance(ckpt, dict):
            for key in ("wrapper_state_dict", "model_state_dict", "state_dict"):
                if key in ckpt:
                    state_options.append((key, ckpt[key]))
        if not state_options:
            # The checkpoint itself is a bare state dict (no wrapper dict).
            state_options.append(("raw", ckpt))

        # Also try stripping/adding 'model.' prefix to handle format mismatches:
        # - Strip: wrapper-saved checkpoint (keys prefixed 'model.') → bare PEFT model
        # - Add:   bare PEFT checkpoint → wrapper (keys need 'model.' prefix)
        derived_options = []
        for key, state in state_options:
            if any(k.startswith("model.") for k in state):
                stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
                if stripped:
                    derived_options.append((key + "_strip_model_prefix", stripped))
            else:
                prefixed = {"model." + k: v for k, v in state.items()}
                derived_options.append((key + "_add_model_prefix", prefixed))
        all_state_options = state_options + derived_options

        last_error: Optional[Exception] = None
        for key, state in all_state_options:
            try:
                model.load_state_dict(state, strict=True)
                print(f"Loaded {key} into model (strict=True, all keys matched).")
                last_error = None
                break
            except Exception as e:
                last_error = e
                continue  # try next key variant

        if last_error is not None:
            raise RuntimeError(
                f"Failed to load checkpoint weights (strict=True): {last_error}. "
                "The checkpoint architecture does not match the model."
            ) from last_error

    # --- Step 6: Optionally load PEFT adapters ---
    # Skipped when a partial-state checkpoint was already loaded (adapter weights
    # were restored from the .pt file directly; loading from the adapter directory
    # would override them with potentially stale weights).
    # Use output_dir/adapters/ rather than a path relative to the checkpoint
    # file itself, so the adapter directory is always found regardless of which
    # numbered checkpoint file was selected.
    model_to_check = model.module if hasattr(model, "module") else model
    if not _loaded_via_partial_state and hasattr(model_to_check, "peft_config"):
        adapters_dir = os.path.join(output_dir, "adapters")
        if os.path.isdir(adapters_dir):
            try:
                model_to_check.load_adapter(adapters_dir)
                print(f"[Checkpoint] Loaded PEFT adapters from {adapters_dir}")
            except Exception as exc:
                print(f"[Checkpoint] WARNING: Failed to load PEFT adapters from {adapters_dir}: {exc}")
        else:
            print(f"[Checkpoint] No adapter directory found at {adapters_dir}; skipping adapter load.")

    # --- Step 7: Restore optimizer / scheduler / scaler states ---
    if isinstance(ckpt, dict):
        if "optimizer" in ckpt and optimizer is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("Loaded optimizer state.")
            except Exception as e:
                # A mismatch here typically means the architecture or param groups
                # changed.  Silently continuing would produce wrong LRs and no
                # momentum history — better to fail loudly.
                raise RuntimeError(
                    f"[Checkpoint] Optimizer state load failed: {e}. "
                    "Pass optimizer=None to skip loading, or delete the checkpoint if the architecture changed."
                ) from e
            else:
                # Move optimizer state tensors to the current device to avoid
                # device mismatches when resuming on a different GPU configuration.
                _current_device = next(model.parameters()).device
                for _state in optimizer.state.values():
                    for _k, _v in _state.items():
                        if isinstance(_v, torch.Tensor):
                            _state[_k] = _v.to(_current_device)
        elif optimizer is not None:
            print("Warning: optimizer state not found in checkpoint; starting optimizer from scratch.")

        if "scheduler" in ckpt and scheduler is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
                print("Loaded scheduler state.")
            except Exception as e:
                raise RuntimeError(
                    f"[Checkpoint] Scheduler state load failed: {e}. "
                    "Pass scheduler=None to skip loading, or delete the checkpoint if the architecture changed."
                ) from e
        elif scheduler is not None:
            print("Warning: scheduler state not found in checkpoint; starting scheduler from scratch.")

        # GradScaler state is non-critical — a mismatch (e.g. fp32 → bf16 switch)
        # only affects the loss-scale warm-up, not correctness.
        if "scaler" in ckpt and scaler is not None and ckpt.get("scaler") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
                print("Loaded scaler state.")
            except Exception as e:
                print("Warning: failed to load scaler state:", e)
        elif scaler is not None:
            print("Warning: scaler state not found in checkpoint; starting scaler from scratch.")

        # --- Step 8: Restore last-eval tracking counters ---
        # These are used by the training loop to avoid evaluating at the same
        # step/epoch twice after resume.
        last_eval_epoch = ckpt.get("last_eval_epoch", last_eval_epoch)
        last_eval_step = ckpt.get("last_eval_step", last_eval_step)
        last_train_eval_epoch = ckpt.get("last_train_eval_epoch", last_train_eval_epoch)
        last_train_eval_step = ckpt.get("last_train_eval_step", last_train_eval_step)
        last_test_eval_epoch = ckpt.get("last_test_eval_epoch", last_test_eval_epoch)
        last_test_eval_step = ckpt.get("last_test_eval_step", last_test_eval_step)
        last_asvspoof2021_eval_epoch = ckpt.get("last_asvspoof2021_eval_epoch", last_asvspoof2021_eval_epoch)
        last_asvspoof2021_eval_step = ckpt.get("last_asvspoof2021_eval_step", last_asvspoof2021_eval_step)

        # --- Step 9: Restore epoch / step counters ---
        if "epoch" in ckpt:
            # Invariant: ckpt["epoch"] always stores the raw loop variable (0-indexed).
            #   epoch_completed=True  → epoch finished; resume from epoch+1.
            #   epoch_completed=False → epoch was interrupted; replay it from its start.
            # Legacy checkpoints (no epoch_completed field) stored epoch+1 rather than the
            # raw loop variable, so treat the stored value as the next epoch directly.
            if "epoch_completed" not in ckpt:
                start_epoch = int(ckpt["epoch"])  # legacy: stored epoch+1, already the next epoch
            elif ckpt["epoch_completed"]:
                start_epoch = int(ckpt["epoch"]) + 1  # resume from the epoch after the completed one
            else:
                start_epoch = int(ckpt["epoch"])  # interrupted mid-epoch; replay it from its start
            print(f"Resuming from epoch stored in ckpt: start_epoch={start_epoch} (epoch_completed={ckpt.get('epoch_completed', 'legacy')})")
        else:
            # Very old checkpoint pre-dating the epoch-key convention; start from 0.
            start_epoch = 0

        if "global_step" in ckpt:
            start_step = int(ckpt["global_step"])
        else:
            # Derive from epoch when no explicit step is stored.
            start_step = start_epoch * steps_per_epoch

        # --- Step 10: Restore best-EER tracking ---
        # Always restore best_dev_eer and best_eval_eer directly from the
        # checkpoint, defaulting to 100.0 (worst-case) when the key is absent.
        # Do NOT fall back to best_eer, which may already be an improved value
        # that would skew model-saving thresholds incorrectly.
        best_eer = float(ckpt.get("best_eer", best_eer))
        # Prefer "best_dev_eer" (canonical key); fall back to "best_eer" for checkpoints
        # written before the key was introduced, then to 100.0 as worst-case default.
        _bde = ckpt.get("best_dev_eer")
        if _bde is None:
            _bde = ckpt.get("best_eer", 100.0)
        best_dev_eer = float(_bde if _bde is not None else 100.0)
        _bee = ckpt.get("best_eval_eer")
        best_eval_eer = float(_bee if _bee is not None else 100.0)
        print(f"[Resume] Will continue training from epoch {start_epoch}")
        print(f"[Resume] Starting from global step {start_step}")
        print(f"[Resume] Best EER so far: {best_eer:.4f}")

        # --- Step 11: Restore RNG states (C3) ---
        rng_states = ckpt.get("rng_states")
        if rng_states is not None:
            torch.set_rng_state(_to_byte_tensor(rng_states["torch"]))
            _saved_cuda = rng_states.get("cuda")
            if _saved_cuda is not None and torch.cuda.is_available():
                _n_now = torch.cuda.device_count()
                if isinstance(_saved_cuda, list):
                    _saved_cuda = [_to_byte_tensor(s) for s in _saved_cuda]
                    if len(_saved_cuda) == _n_now:
                        torch.cuda.set_rng_state_all(_saved_cuda)
                    elif _saved_cuda:
                        print(f"[Resume WARNING] cuda RNG device count mismatch (saved={len(_saved_cuda)}, now={_n_now}); restoring device 0 only.")
                        torch.cuda.set_rng_state(_saved_cuda[0], device=0)
            np.random.set_state(rng_states["numpy"])
            random.setstate(rng_states["python"])
            print("[Resume] RNG states restored from checkpoint.")
        else:
            print("[Resume] No RNG states found in checkpoint; data order after resume may differ.")

        # --- Step 12: Warn if critical config keys changed since last checkpoint ---
        saved_cfg = ckpt.get("configs") or ckpt.get("config_dict")
        if saved_cfg and configs is not None:
            critical_keys = [
                "lr", "batch_size", "BATCH_SIZE", "lora_rank", "LORA_RANK",
                "lora_alpha", "model_name",
                "use_bf16", "use_amp", "augmentation_only_spoof",
            ]
            for key in critical_keys:
                saved_val = saved_cfg.get(key)
                current_val = getattr(configs, key, None)
                if saved_val is not None and current_val is not None and saved_val != current_val:
                    print(f"[Resume WARNING] Config key '{key}' changed: saved={saved_val!r} vs current={current_val!r}")

            # --- Step 12b: Validate generation_classification nested config ---
            # label_texts and normalize_logprob_by_tokens live under the
            # generation_classification sub-dict in both the YAML and saved configs;
            # the flat-key loop above never reaches them.
            # saved_cfg is always a plain dict (JSON-serialised).  If the key is
            # absent or not a dict (legacy checkpoint), skip quietly.
            _saved_gen_cfg = saved_cfg.get("generation_classification")
            if not isinstance(_saved_gen_cfg, dict):
                _saved_gen_cfg = None

            # configs.generation_classification is a plain dict when loaded directly
            # from YAML, but may be a SimpleNamespace/object in some code paths;
            # normalise to dict so .get() calls below work uniformly.
            _cur_gen_raw = getattr(configs, "generation_classification", None)
            if isinstance(_cur_gen_raw, dict):
                _cur_gen_cfg = _cur_gen_raw
            elif _cur_gen_raw is not None and hasattr(_cur_gen_raw, "__dict__"):
                _cur_gen_cfg = vars(_cur_gen_raw)
            else:
                _cur_gen_cfg = None

            # Only run the nested checks when both sides have the sub-dict; if either
            # is missing (non-CAG run, legacy checkpoint) the comparisons are undefined.
            if _saved_gen_cfg is not None and _cur_gen_cfg is not None:
                # normalize_logprob_by_tokens — warn on change (may silently destroy EER)
                _saved_norm = _saved_gen_cfg.get("normalize_logprob_by_tokens")
                _cur_norm = _cur_gen_cfg.get("normalize_logprob_by_tokens")
                if _saved_norm is not None and _cur_norm is not None and _saved_norm != _cur_norm:
                    print(
                        f"[Resume WARNING] generation_classification.normalize_logprob_by_tokens "
                        f"changed: saved={_saved_norm!r} vs current={_cur_norm!r}"
                    )

                # label_texts — raise (not warn) when label order is inverted, because
                # an inverted order silently flips the entire EER direction.
                _saved_lt_raw = _saved_gen_cfg.get("label_texts") or {}
                _cur_lt_raw = _cur_gen_cfg.get("label_texts") or {}
                # Guard: if either side has no label_texts the comparison is meaningless.
                if _saved_lt_raw and _cur_lt_raw:
                    try:
                        # int(k) normalises JSON string keys ("0","1") → int keys.
                        _saved_lt = {int(k): v for k, v in _saved_lt_raw.items()}
                        _cur_lt = {int(k): v for k, v in _cur_lt_raw.items()}
                    except (TypeError, ValueError, AttributeError):
                        # Malformed label_texts (not a dict, non-integer keys, etc.) —
                        # skip the check rather than crashing on an optional guard.
                        _saved_lt, _cur_lt = {}, {}
                    # Exact inversion: saved[0]==current[1] AND saved[1]==current[0].
                    # Both dicts must contain keys 0 and 1 before the cross-comparison.
                    if (
                        {0, 1} <= set(_saved_lt) and {0, 1} <= set(_cur_lt)
                        and _saved_lt.get(0) == _cur_lt.get(1)
                        and _saved_lt.get(1) == _cur_lt.get(0)
                    ):
                        raise ValueError(
                            f"[Resume ERROR] generation_classification.label_texts order is "
                            f"INVERTED relative to the checkpoint: checkpoint trained with "
                            f"{{0: {_saved_lt[0]!r}, 1: {_saved_lt[1]!r}}} but current config "
                            f"has {{0: {_cur_lt[0]!r}, 1: {_cur_lt[1]!r}}}. Resuming with "
                            f"swapped labels silently inverts EER direction across the entire run. "
                            f"Fix: restore label_texts to the checkpoint's order or start fresh."
                        )
                    elif _saved_lt != _cur_lt:
                        print(
                            f"[Resume WARNING] generation_classification.label_texts changed: "
                            f"saved={_saved_lt!r} vs current={_cur_lt!r}"
                        )

    # Return a flat dict so callers can unpack exactly the fields they need.
    return {
        "ckpt_path": ckpt_path,
        "wandb_run": wandb_run,
        "start_epoch": start_epoch,
        "start_step": start_step,
        "best_eer": best_eer,
        "best_dev_eer": best_dev_eer,
        "best_eval_eer": best_eval_eer,
        "last_eval_epoch": last_eval_epoch,
        "last_eval_step": last_eval_step,
        "last_train_eval_epoch": last_train_eval_epoch,
        "last_train_eval_step": last_train_eval_step,
        "last_test_eval_epoch": last_test_eval_epoch,
        "last_test_eval_step": last_test_eval_step,
        "last_asvspoof2021_eval_epoch": last_asvspoof2021_eval_epoch,
        "last_asvspoof2021_eval_step": last_asvspoof2021_eval_step,
    }
