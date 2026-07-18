#!/usr/bin/env python3
"""
Visualize audio-adapter vs. LLM-decoder embeddings with PaCMAP.

Reproduces the paper's Figure 2: PaCMAP is fit on the ASVspoof5 training-set
embeddings only, then applied (out-of-sample transform, using the same fitted
basis) to the dev and eval splits, for both the audio-adapter ("audio_projector"
stage) and LLM-decoder ("llama" stage) representations. Points are colored by
bonafide/spoof.

Inputs are the NPZ bundles produced by aggregate_embeddings_to_npz.py, one per
(stage, split) pair.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed when running on a cluster
import matplotlib.pyplot as plt

try:
    import pacmap
except ImportError as exc:
    raise ImportError(
        "The 'pacmap' package is required for this script (pip install pacmap>=0.7)."
    ) from exc


# ------------ PARAMETERS (EDIT HERE) ------------
AUDIO_ADAPTER_NPZ = {
    "train": "/path/to/ASVspoof5_embeds/audio_projector/total_train_embeddings.npz",
    "dev": "/path/to/ASVspoof5_embeds/audio_projector/total_dev_embeddings.npz",
    "eval": "/path/to/ASVspoof5_embeds/audio_projector/total_eval_embeddings.npz",
}
LLM_NPZ = {
    "train": "/path/to/ASVspoof5_embeds/llama/total_train_embeddings.npz",
    "dev": "/path/to/ASVspoof5_embeds/llama/total_dev_embeddings.npz",
    "eval": "/path/to/ASVspoof5_embeds/llama/total_eval_embeddings.npz",
}
OUTPUT_FILE = "pacmap_asvspoof5_projections.png"
RANDOM_STATE = 42
N_NEIGHBORS = 10
MAX_POINTS_PER_SPLIT = None  # set an int to subsample very large splits
# -----------------------------------------------

SPLIT_TITLES = {"train": "Training set", "dev": "Dev. set", "eval": "Eval. set"}


def _load_split(npz_path):
    """Load embeddings (N, D) and lower-cased label strings from an NPZ bundle."""
    with np.load(npz_path, allow_pickle=True) as d:
        embeddings = d["embeddings"].astype(np.float32)
        labels = np.array([str(l).lower() for l in d["labels"]])
    return embeddings, labels


def _subsample(embeddings, labels, max_points, rng):
    """Randomly subsample a split to at most max_points rows."""
    if max_points is None or len(labels) <= max_points:
        return embeddings, labels
    idx = rng.choice(len(labels), size=max_points, replace=False)
    return embeddings[idx], labels[idx]


def _fit_transform_splits(npz_paths, rng):
    """Fit PaCMAP on the train split, transform dev/eval onto the same basis."""
    train_emb, train_labels = _load_split(npz_paths["train"])
    train_emb, train_labels = _subsample(train_emb, train_labels, MAX_POINTS_PER_SPLIT, rng)

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE)
    train_2d = reducer.fit_transform(train_emb, init="pca")

    projections = {"train": (train_2d, train_labels)}
    for split in ("dev", "eval"):
        emb, labels = _load_split(npz_paths[split])
        emb, labels = _subsample(emb, labels, MAX_POINTS_PER_SPLIT, rng)
        emb_2d = reducer.transform(emb, basis=train_emb)
        projections[split] = (emb_2d, labels)
    return projections


def _plot_panel(ax, points_2d, labels, title):
    """Scatter one PaCMAP panel, spoof points first so bonafide draws on top."""
    bonafide_mask = np.isin(labels, ["bonafide", "genuine"])
    spoof_mask = labels == "spoof"
    ax.scatter(points_2d[spoof_mask, 0], points_2d[spoof_mask, 1], s=3, alpha=0.5, c="tab:red", label="spoof")
    ax.scatter(points_2d[bonafide_mask, 0], points_2d[bonafide_mask, 1], s=3, alpha=0.5, c="tab:blue", label="bonafide")
    ax.set_title(title)
    ax.set_xlabel("PACMAP-1")
    ax.set_ylabel("PACMAP-2")


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    print("Fitting PaCMAP on audio-adapter embeddings (train) and projecting dev/eval...")
    audio_adapter_proj = _fit_transform_splits(AUDIO_ADAPTER_NPZ, rng)

    print("Fitting PaCMAP on LLM-decoder embeddings (train) and projecting dev/eval...")
    llm_proj = _fit_transform_splits(LLM_NPZ, rng)

    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    for row, split in enumerate(("train", "dev", "eval")):
        points_2d, labels = audio_adapter_proj[split]
        _plot_panel(axes[row, 0], points_2d, labels, f"Audio Adapter — {SPLIT_TITLES[split]}")
        points_2d, labels = llm_proj[split]
        _plot_panel(axes[row, 1], points_2d, labels, f"LLM — {SPLIT_TITLES[split]}")

    handles, plot_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, plot_labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_FILE, dpi=200)
    print(f"Saved PaCMAP projection figure to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
