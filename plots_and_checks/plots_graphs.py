#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matches: "Epoch 1 Step 50 Loss 0.7111"
TRAIN_LOSS_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+Step\s+(\d+)\s+Loss\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)", re.IGNORECASE)
EER_PATTERN = re.compile(
    r"eer:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", re.IGNORECASE
)


def find_log_files(log_dir: Path, stem: str) -> List[Path]:
    """Return sorted log files like stem.txt, stem_1.txt, stem_2.txt, ..."""
    candidates = list(log_dir.glob(f"{stem}.txt")) + list(log_dir.glob(f"{stem}_*.txt"))
    # Deduplicate while preserving order.
    seen = set()
    files: List[Path] = []
    for path in sorted(candidates):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            files.append(path)
    return files


def parse_training_logs(
    log_dir: Path,
    stem_suffix: str = "",
) -> Tuple[List[int], List[float], List[int], List[float]]:
    """Return step-level loss and epoch-averaged loss."""
    step_records: List[Tuple[int, float]] = []
    epoch_buckets: defaultdict[int, List[float]] = defaultdict(list)
    files = find_log_files(log_dir, f"train_log{stem_suffix}")
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                match = TRAIN_LOSS_PATTERN.search(line)
                if not match:
                    continue
                epoch_str, step_str, loss_str = match.groups()
                epoch = int(epoch_str)
                step = int(step_str)
                loss = float(loss_str)
                step_records.append((step, loss))
                epoch_buckets[epoch].append(loss)
    step_records.sort(key=lambda item: item[0])  # Stable sort keeps original order on ties.
    steps = [step for step, _ in step_records]
    losses = [loss for _, loss in step_records]
    epoch_losses = [(epoch, sum(vals) / len(vals)) for epoch, vals in epoch_buckets.items()]
    epoch_losses.sort(key=lambda item: item[0])
    epoch_numbers = [epoch for epoch, _ in epoch_losses]
    epoch_means = [loss for _, loss in epoch_losses]
    return steps, losses, epoch_numbers, epoch_means


def parse_metrics_logs(log_dir: Path, split: str, stem_suffix: str = "") -> Tuple[List[int], List[float]]:
    """Extract (epoch, eer) pairs for the requested split (dev or eval)."""
    files = find_log_files(log_dir, f"metrics_log_{split}{stem_suffix}")
    records: List[Tuple[int, float]] = []
    for path in files:
        current_epoch = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                epoch_match = EPOCH_PATTERN.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                eer_match = EER_PATTERN.search(line)
                if eer_match and current_epoch is not None:
                    records.append((current_epoch, float(eer_match.group(1))))
    records.sort(key=lambda item: item[0])
    epochs = [epoch for epoch, _ in records]
    eers = [eer for _, eer in records]
    return epochs, eers


def add_series_plot(
    ax,
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    missing_msg: str,
) -> None:
    if x and y:
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=3)
    else:
        ax.text(0.5, 0.5, missing_msg, ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)


def build_figure(
    steps: Sequence[int],
    losses: Sequence[float],
    epoch_numbers: Sequence[int],
    epoch_means: Sequence[float],
    dev_epochs: Sequence[int],
    dev_eers: Sequence[float],
    eval_epochs: Sequence[int],
    eval_eers: Sequence[float],
):
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)
    add_series_plot(
        axes[0],
        steps,
        losses,
        title="Training loss vs step",
        xlabel="Step",
        ylabel="Loss",
        missing_msg="No training loss data found",
    )
    add_series_plot(
        axes[1],
        epoch_numbers,
        epoch_means,
        title="Training loss vs epoch",
        xlabel="Epoch",
        ylabel="Loss",
        missing_msg="No training loss data found",
    )
    add_series_plot(
        axes[2],
        dev_epochs,
        dev_eers,
        title="EER (dev) vs epoch",
        xlabel="Epoch",
        ylabel="EER",
        missing_msg="No dev metrics found",
    )
    add_series_plot(
        axes[3],
        eval_epochs,
        eval_eers,
        title="EER (eval) vs epoch",
        xlabel="Epoch",
        ylabel="EER",
        missing_msg="No eval metrics found",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot training loss and EER metrics from log files."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="/path/to/voxtral_clean/voxtral_clean-generation-asvspoof2021-aug-bfloat16-2-constant-classification_as_generation-with-aug2/",
        help="Directory containing train_log*.txt and metrics_log_* files (default: %(default)s).",
    )
    parser.add_argument(
        "--stem",
        default="",
        help=(
            "Optional suffix appended to log file stems. "
            "For example, '--stem _1' will look for train_log_1*.txt and metrics_log_{dev,eval}_1*.txt."
        ),
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.exists() or not log_dir.is_dir():
        raise SystemExit(f"Log directory does not exist or is not a directory: {log_dir}")

    steps, losses, epoch_numbers, epoch_means = parse_training_logs(
        log_dir, stem_suffix=args.stem
    )
    dev_epochs, dev_eers = parse_metrics_logs(log_dir, "dev", stem_suffix=args.stem)
    eval_epochs, eval_eers = parse_metrics_logs(log_dir, "eval", stem_suffix=args.stem)

    fig = build_figure(
        steps, losses, epoch_numbers, epoch_means, dev_epochs, dev_eers, eval_epochs, eval_eers
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = log_dir / f"training_plots_{timestamp}.png"
    fig.savefig(output_path)
    print(f"Saved plots to {output_path}")


if __name__ == "__main__":
    main()
