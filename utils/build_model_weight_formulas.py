"""
Class weighting formula for imbalanced spoof/bonafide datasets.
"""
import math
import torch


def compute_weights_sqrt_inverse(dataset, device):
    """
    Compute class weights using square-root inverse frequency, a softer alternative to
    plain inverse frequency (total / n_i) that prevents extreme weights on highly imbalanced sets.

    Formula:
        weight_i = sqrt(total / n_i)
        then normalized so weights sum to 2.0 (one per class on average).
        clamped at 10.0 to prevent training instability.

    Why sqrt instead of plain inverse:
        Plain inverse (total/n_i) amplifies the minority class very aggressively.
        Taking the square root compresses the ratio — e.g. a 10:1 imbalance gives
        weight ratio sqrt(10) ≈ 3.2x instead of 10x — which is more stable for training.

    When to use:
        Moderate imbalance (roughly 2:1 to 10:1). For severe imbalance (>10:1) consider
        oversampling instead.

    Returns:
        Tensor of shape [2] with [bonafide_weight, spoof_weight] on the given device.
        Falls back to [1.0, 1.0] if either class is empty.
    """
    n_bona = len(dataset.bonafide_indices)
    n_spoof = len(dataset.spoof_indices)
    total = n_bona + n_spoof

    print(f"[Weights] Bonafide: {n_bona}, Spoof: {n_spoof}, Total: {total}")

    if n_bona == 0 or n_spoof == 0:
        print("[WARN] Zero class count; using equal weights.")
        return torch.tensor([1.0, 1.0], dtype=torch.float, device=device)

    weight_bona = math.sqrt(total / n_bona)
    weight_spoof = math.sqrt(total / n_spoof)

    weights = torch.tensor([weight_bona, weight_spoof], dtype=torch.float, device=device)
    weights = weights / weights.sum() * 2.0
    weights = weights.clamp(max=10.0)

    print(f"[Weights] sqrt_inverse: bonafide={weight_bona:.3f}, spoof={weight_spoof:.3f} → final={weights.tolist()}")
    return weights
