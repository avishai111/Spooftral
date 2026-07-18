"""
utils/training_helpers.py
==========================
Shared training-loop utility functions used by both the CAG and
classification-head training loops.

Functions
---------
next_available(path)
    Returns a non-conflicting file path by appending _1, _2, … when the
    target already exists.

safe_int(val)
    Safely converts a value to int, returning None on failure.

dprint(debug_log_path, enable_debug_logging, *args)
    Debug print that optionally mirrors output to a log file.

eprint(debug_log_path, enable_debug_logging, *args)
    Error-style print with optional file logging.

save_checkpoint(...)
    Atomic checkpoint save (write to tmp then os.replace) with fsync for
    power-loss safety.

print_asvspoof2021_eval_info(...)
    Prints a formatted summary of ASVspoof2021 eval parameters to stdout.
"""
import os

import torch


def next_available(path):
    """
    Return a non-conflicting file path by appending _N if needed.
    - If path does not exist, returns it unchanged; otherwise increments suffix until free.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        alt = f"{base}_{i}{ext}"
        if not os.path.exists(alt):
            return alt
        i += 1


def safe_int(val):
    """
    Safely convert value to int or return None on failure.
    - Catches TypeError/ValueError to avoid raising in loose parsing paths.
    """
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def dprint(debug_log_path, enable_debug_logging, *args, **kwargs):
    """
    Debug print that can mirror to file.
    - No-op when enable_debug_logging is False.
    - Writes to debug_log_path if provided; otherwise prints to stdout.
    """
    if not enable_debug_logging:
        return
    if debug_log_path is None:
        print(*args, **kwargs)
        return
    with open(debug_log_path, "a", encoding="utf-8") as f:
        print(*args, file=f, **{k: v for k, v in kwargs.items() if k != "file"})


def eprint(debug_log_path, enable_debug_logging, *args):
    """
    Error-style print with optional file logging.
    - No-op when enable_debug_logging is False.
    - Prints to stdout and appends to debug_log_path when provided.
    """
    if not enable_debug_logging:
        return
    msg = " ".join(str(a) for a in args)
    print(msg)
    if debug_log_path:
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def print_asvspoof2021_eval_info(configs, asvspoof2021_eval_loader, stage):
    """
    Print ASVspoof2021 evaluation configuration summary.
    - Extracts audio dir, protocol, and batch size from configs/loader.
    - No-op when configs is None.
    """
    if configs is None:
        return
    audio_dir = getattr(configs, "asvspoof2021_eval_audio_dir", None)
    protocol = getattr(configs, "asvspoof2021_eval_protocol", None)
    batch_size = None
    if asvspoof2021_eval_loader is not None:
        batch_size = getattr(asvspoof2021_eval_loader, "batch_size", None)
    if batch_size is None:
        batch_size = getattr(configs, "eval_batch_size", None)
    print(
        f"[ASVspoof2021] {stage} audio_dir={audio_dir} eval_protocol={protocol} batch_size={batch_size}",
        flush=True,
    )


def save_checkpoint(checkpoint_path, payload):
    """
    Persist checkpoint payload to disk atomically.
    - Writes to a .tmp file first; fsyncs the tmp file for power-loss durability;
      then renames to the final path (POSIX atomic); then fsyncs the parent dir
      so the rename itself is durable.
    - Cleans up the .tmp file on any exception to avoid leaving partial files.
    - Prevents corrupt/zero-byte checkpoints when killed mid-save.
    """
    tmp_path = checkpoint_path + ".tmp"
    parent_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    try:
        torch.save(payload, tmp_path)
        # Flush the tmp file's data to disk before rename so a crash immediately
        # after cannot leave a zero-byte or partial file at the final path.
        with open(tmp_path, "rb") as _f:
            os.fsync(_f.fileno())
        # Atomically rename tmp → final path.
        os.replace(tmp_path, checkpoint_path)
        # Sync the parent directory so the rename itself is durable.
        dir_fd = os.open(parent_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise
