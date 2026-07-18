#!/usr/bin/env python3
"""
Compare embedding files and compute distance metrics between them.

NPZ_PATH_1 and NPZ_PATH_2 can be:
  - Single paths (str): Compare one pair
  - Lists of paths: Compare NPZ_PATH_1[i] vs NPZ_PATH_2[i] for each i

Metrics computed:
  - Average Cosine Similarity (row-wise)
  - Average L2 (Euclidean) Distance
  - Standard Deviation of Differences
  - CKA (Centered Kernel Alignment)
  - Pairwise Distance Correlation
  - Min/Max Cosine and L2
"""

import os
import sys

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


# ------------ PARAMETERS (EDIT HERE) ------------
# Can be a single path (str) or a list of paths
# If lists, compares NPZ_PATH_1[i] vs NPZ_PATH_2[i] for each i
NPZ_PATH_1 = "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/total_eval_embeddings.npz"
NPZ_PATH_2 = "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama2/total_eval_embeddings.npz"

# Example with lists:
# NPZ_PATH_1 = [
#     "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/total_train_embeddings.npz",
#     "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/total_dev_embeddings.npz",
#     "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/total_eval_embeddings.npz",
# ]
# NPZ_PATH_2 = [
#     "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama2/total_train_embeddings.npz",
#      "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama2/total_dev_embeddings.npz",
#     "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama2/total_eval_embeddings.npz",
# ]

# Output file path (leave empty to skip saving)
OUTPUT_FILE = "/gpfs0/bgu-benshimo/users/wavishay/voxtral/calc_embeddings_projector_codes/comparison_results.txt"

# Use float16 to reduce memory usage (True = ~50% less RAM)
USE_FLOAT16 = True
# -----------------------------------------------


_output_file = None


def log(msg: str = ""):
    """Print to console and optionally write to file."""
    print(msg)
    if _output_file:
        _output_file.write(msg + "\n")
        _output_file.flush()


def load_embeddings(npz_path: str) -> np.ndarray:
    """Load embeddings from an NPZ file."""
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        if "embeddings" not in data.files:
            raise ValueError(f"Missing 'embeddings' key in: {npz_path}")
        embeddings = data["embeddings"]
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape: {embeddings.shape}")

    if USE_FLOAT16:
        embeddings = embeddings.astype(np.float16)

    return embeddings


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear CKA (Centered Kernel Alignment).

    CKA measures similarity between representation spaces.
    Values range from 0 (completely different) to 1 (identical structure).
    """
    # Center the matrices
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Compute Gram matrices
    XTX = X @ X.T
    YTY = Y @ Y.T

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    hsic_xy = np.sum(XTX * YTY)
    hsic_xx = np.sum(XTX * XTX)
    hsic_yy = np.sum(YTY * YTY)

    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def compute_pairwise_distance_correlation(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute Spearman correlation between pairwise distances.

    Measures if relative distances are preserved between embeddings.
    High correlation = similar neighborhood structure.
    """
    # Compute pairwise L2 distances using optimized C implementation
    dist1 = pdist(emb1, metric='euclidean')
    dist2 = pdist(emb2, metric='euclidean')

    corr, _ = spearmanr(dist1, dist2)
    return corr


def measure_closeness(emb1: np.ndarray, emb2: np.ndarray) -> dict:
    """
    Compute distance metrics between two embedding arrays.

    Args:
        emb1: First embedding array (N x D)
        emb2: Second embedding array (N x D)

    Returns:
        Dictionary with distance metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if USE_FLOAT16 else torch.float32
    t1 = torch.tensor(emb1, dtype=dtype).to(device)
    t2 = torch.tensor(emb2, dtype=dtype).to(device)

    # 1. Cosine Similarity (row-wise)
    cos = torch.nn.functional.cosine_similarity(t1, t2, dim=1)
    avg_cos = cos.mean().item()
    min_cos = cos.min().item()
    max_cos = cos.max().item()

    # 2. L2 (Euclidean) Distance
    l2_dist = torch.norm(t1 - t2, p=2, dim=1)
    avg_l2 = l2_dist.mean().item()
    min_l2 = l2_dist.min().item()
    max_l2 = l2_dist.max().item()

    # 3. Standard Deviation of Differences
    diff_std = (t1 - t2).std().item()

    # 4. CKA - Centered Kernel Alignment
    cka = compute_cka(emb1, emb2)

    # 5. Pairwise Distance Correlation
    pairwise_corr = compute_pairwise_distance_correlation(emb1, emb2)

    return {
        "Avg Cosine": avg_cos,
        "Min Cosine": min_cos,
        "Max Cosine": max_cos,
        "Avg L2": avg_l2,
        "Min L2": min_l2,
        "Max L2": max_l2,
        "Diff Std": diff_std,
        "CKA": cka,
        "Pairwise Dist Corr": pairwise_corr,
    }


def compare_pair(npz_path_1: str, npz_path_2: str, pair_idx: int = None):
    """Compare two embedding files and log results."""
    if pair_idx is not None:
        log(f"\n{'#' * 60}")
        log(f"# Comparison {pair_idx + 1}")
        log(f"{'#' * 60}")

    log(f"\nLoading embeddings from: {npz_path_1}")
    emb1 = load_embeddings(npz_path_1)
    log(f"  Shape: {emb1.shape}, dtype: {emb1.dtype}")

    log(f"Loading embeddings from: {npz_path_2}")
    emb2 = load_embeddings(npz_path_2)
    log(f"  Shape: {emb2.shape}, dtype: {emb2.dtype}")

    # Validate shapes match
    if emb1.shape[1] != emb2.shape[1]:
        raise ValueError(
            f"Embedding dimensions do not match: {emb1.shape[1]} vs {emb2.shape[1]}"
        )

    if emb1.shape[0] != emb2.shape[0]:
        log(f"\nWARNING: Row counts differ ({emb1.shape[0]} vs {emb2.shape[0]})")
        min_rows = min(emb1.shape[0], emb2.shape[0])
        log(f"Comparing first {min_rows} rows only.\n")
        emb1 = emb1[:min_rows]
        emb2 = emb2[:min_rows]

    log("\nComputing distance metrics...")
    metrics = measure_closeness(emb1, emb2)

    log("\n" + "=" * 60)
    log("Results:")
    log("=" * 60)

    log("\n[Cosine Similarity] (1 = identical, 0 = orthogonal)")
    log(f"  Average: {metrics['Avg Cosine']:.6f}")
    log(f"  Min:     {metrics['Min Cosine']:.6f}")
    log(f"  Max:     {metrics['Max Cosine']:.6f}")

    log("\n[L2 Distance] (0 = identical)")
    log(f"  Average: {metrics['Avg L2']:.6f}")
    log(f"  Min:     {metrics['Min L2']:.6f}")
    log(f"  Max:     {metrics['Max L2']:.6f}")

    log("\n[Additional Metrics]")
    log(f"  Diff Std Dev:           {metrics['Diff Std']:.6f}")
    log(f"  CKA:                    {metrics['CKA']:.6f}  (1 = same structure)")
    log(f"  Pairwise Dist Corr:     {metrics['Pairwise Dist Corr']:.6f}  (1 = same neighborhoods)")

    log("=" * 60)
    return metrics


def main():
    global _output_file

    # Convert to lists if single paths
    paths_1 = NPZ_PATH_1 if isinstance(NPZ_PATH_1, list) else [NPZ_PATH_1]
    paths_2 = NPZ_PATH_2 if isinstance(NPZ_PATH_2, list) else [NPZ_PATH_2]

    # Validate
    if not paths_1 or not paths_1[0]:
        raise ValueError("NPZ_PATH_1 is empty.")
    if not paths_2 or not paths_2[0]:
        raise ValueError("NPZ_PATH_2 is empty.")
    if len(paths_1) != len(paths_2):
        raise ValueError(
            f"NPZ_PATH_1 and NPZ_PATH_2 must have the same length. "
            f"Got {len(paths_1)} vs {len(paths_2)}."
        )

    # Open output file if specified
    if OUTPUT_FILE:
        _output_file = open(OUTPUT_FILE, "w", encoding="utf-8")
        log(f"Saving results to: {OUTPUT_FILE}")

    try:
        n_comparisons = len(paths_1)
        log(f"Will perform {n_comparisons} comparison(s)")

        all_metrics = []
        for i in range(n_comparisons):
            metrics = compare_pair(paths_1[i], paths_2[i], pair_idx=i if n_comparisons > 1 else None)
            all_metrics.append(metrics)

        # Print summary if multiple comparisons
        if n_comparisons > 1:
            log("\n" + "#" * 60)
            log("# SUMMARY")
            log("#" * 60)
            log(f"\n{'Pair':<6} {'Avg Cosine':<12} {'Avg L2':<12} {'CKA':<12} {'Pairwise Corr':<14}")
            log("-" * 56)
            for i, m in enumerate(all_metrics):
                log(f"{i+1:<6} {m['Avg Cosine']:<12.6f} {m['Avg L2']:<12.6f} {m['CKA']:<12.6f} {m['Pairwise Dist Corr']:<14.6f}")
    finally:
        if _output_file:
            _output_file.close()
            _output_file = None


if __name__ == "__main__":
    main()
