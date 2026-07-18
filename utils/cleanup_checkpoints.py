#!/usr/bin/env python3
"""
Checkpoint Cleanup Utility

Keeps only the best N checkpoints to save disk space.

Usage:
  python cleanup_checkpoints.py --output_dir /path/to/output --keep_best 5
"""
import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable, List, Set


def _sorted_checkpoints(output_dir: Path, pattern: str) -> List[Path]:
    """Return checkpoint paths sorted by modification time (newest first)."""
    checkpoints = list(output_dir.glob(pattern))
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints


def _fmt_size_mb(paths: Iterable[Path]) -> float:
    """Return total size of all paths in MB."""
    return sum(p.stat().st_size for p in paths) / (1024 * 1024)


def _find_protected_checkpoints(output_path: Path) -> Set[Path]:
    """
    Return the set of checkpoint paths that must never be deleted:
    - checkpoint-last.pt (the current training state)
    - The metric-best checkpoint referenced by best_dev_metrics.json / best_eval_metrics.json
    - All .pt files inside best_dev/ and best_eval/ subdirectories
    """
    protected: Set[Path] = set()

    # Always protect the running-last checkpoint
    last = output_path / "checkpoint-last.pt"
    if last.exists():
        protected.add(last)

    # Protect metric-best files discovered via their metadata JSON
    for metrics_file in ("best_metrics.json", "best_dev_metrics.json", "best_eval_metrics.json"):
        mf = output_path / metrics_file
        if not mf.exists():
            continue
        try:
            with open(mf, encoding="utf-8") as f:
                meta = json.load(f)
            # Different training scripts have historically written the checkpoint reference
            # under different key names; check each in turn.
            ckpt_name = meta.get("checkpoint_name") or meta.get("ckpt_path") or meta.get("path")
            if ckpt_name:
                candidate = Path(ckpt_name)
                if not candidate.is_absolute():
                    candidate = output_path / candidate
                if candidate.exists():
                    protected.add(candidate)
        except Exception as _e:
            warnings.warn(
                f"[Cleanup] Failed to parse {mf}: {_e}. "
                "The best checkpoint referenced by this file may not be protected. "
                "Check for a corrupted or partial JSON write."
            )

    # Also protect every .pt file under best_dev/ and best_eval/ subdirs
    # Best-checkpoint snapshots live in dedicated subdirectories and are always kept
    # regardless of age.
    for subdir in ("best_dev", "best_eval"):
        for p in (output_path / subdir).glob("*.pt"):
            protected.add(p)

    return protected


def cleanup_old_checkpoints(output_dir: str, keep_best_n: int = 5, pattern: str = "*.pt") -> None:
    """
    Delete old checkpoints, keeping only the keep_best_n most recent deletable ones.

    Protected (never deleted): checkpoint-last.pt, metric-best checkpoints referenced
    by best_dev_metrics.json / best_eval_metrics.json, and all .pt files in best_dev/
    and best_eval/ subdirectories. Only unprotected files are candidates for deletion.
    Raises FileNotFoundError if output_dir does not exist.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    protected = _find_protected_checkpoints(output_path)
    if protected:
        print(f"[Cleanup] Protected checkpoints (will never delete):")
        for p in sorted(protected):
            print(f"  - {p.name}")

    all_checkpoints = _sorted_checkpoints(output_path, pattern)
    # Only consider files that are not protected
    deletable = [c for c in all_checkpoints if c not in protected]

    if len(deletable) <= keep_best_n:
        print(f"Found {len(all_checkpoints)} checkpoints ({len(protected)} protected), "
              f"keeping all deletable (limit={keep_best_n})")
        return

    # deletable is sorted newest-first, so the first keep_best_n entries are the most
    # recent ones being retained.
    to_keep = deletable[:keep_best_n]
    to_remove = deletable[keep_best_n:]

    print(f"Keeping {len(to_keep)} most recent deletable checkpoints:")
    for ckpt in to_keep:
        print(f"  - {ckpt.name}")

    print(f"\nRemoving {len(to_remove)} old checkpoints:")
    total_freed = _fmt_size_mb(to_remove)
    for ckpt in to_remove:
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
        # missing_ok guards against the file already having been removed since the listing was taken.
        ckpt.unlink(missing_ok=True)

    print(f"\nFreed {total_freed:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup old checkpoints")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory with checkpoints")
    parser.add_argument("--keep_best", type=int, default=5, help="Number of checkpoints to keep")
    parser.add_argument("--pattern", type=str, default="*.pt", help="File pattern to match")

    args = parser.parse_args()

    cleanup_old_checkpoints(
        args.output_dir,
        keep_best_n=args.keep_best,
        pattern=args.pattern,
    )
