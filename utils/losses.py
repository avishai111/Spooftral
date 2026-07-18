"""
losses.py
=========
Core loss functions used throughout the Voxtral training pipeline.

Two public functions are exposed:
  - fixed_cross_entropy  : a cross-entropy wrapper that supports optional manual
                           normalisation by a user-supplied sample count.
  - for_causal_lm_loss   : causal (auto-regressive) LM cross-entropy that handles
                           label shifting, fp32 upcasting, and VRAM-safe deletion.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# fixed_cross_entropy
# ---------------------------------------------------------------------------

def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """
    Cross-entropy loss with optional manual normalisation by batch item count.

    Mirrors the HuggingFace ForCausalLMLoss convention: when *num_items_in_batch*
    is provided and *reduction* is not "none", the raw sum is divided by
    *num_items_in_batch* instead of by the number of valid (non-ignore) tokens.
    This is useful for variable-length batches where a simple mean would give
    different gradients depending on how many padding tokens were appended.

    Parameters
    ----------
    source : torch.Tensor, shape (N, C)
        Raw logit scores for each of the N positions across C classes.
    target : torch.Tensor, shape (N,)
        Integer class indices.  Positions equal to *ignore_index* contribute
        zero loss and are excluded from normalisation.
    num_items_in_batch : torch.Tensor | None
        If provided (and reduction != "none"), the summed loss is divided by
        this value instead of by N.  Matches HF's packed-sequence contract.
    ignore_index : int
        Token id that marks positions to skip (default -100, the PyTorch
        convention for masked cross-entropy).
    reduction : str
        One of "mean" | "sum" | "none".
        - "mean"  → average over valid tokens (or divide by num_items_in_batch).
        - "sum"   → raw sum over valid tokens.
        - "none"  → per-token losses returned as a tensor of the same shape as target.
    **kwargs
        Forwarded verbatim to torch.nn.functional.cross_entropy.

    Returns
    -------
    torch.Tensor
        Scalar loss (reduction == "mean" or "sum") or per-token tensor
        (reduction == "none").
    """
    # Reject unsupported reduction strings early so the error message is clear.
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Unsupported reduction: {reduction}")

    # When the caller wants a manually-normalised mean, compute the sum first
    # and then divide by num_items_in_batch below.  This avoids an extra pass.
    ce_reduction = "sum" if (num_items_in_batch is not None and reduction != "none") else reduction

    # Compute the raw cross-entropy.  ignore_index causes those positions to
    # contribute 0 to the sum and to be excluded from the "mean" denominator.
    loss = F.cross_entropy(source, target, ignore_index=ignore_index, reduction=ce_reduction, **kwargs)

    # "none" path: return per-token tensor without any further aggregation.
    if reduction == "none":
        return loss

    # Manual normalisation: caller supplied the effective batch size so we
    # divide the sum by that instead of by the token count.
    if num_items_in_batch is not None and ce_reduction == "sum":
        # Move the normalisation scalar to the same device as the loss to avoid
        # device-mismatch errors when training on GPU.  Clamp to ≥1 to guard
        # against divide-by-zero (empty batch after filtering) which would
        # produce NaN/Inf and silently corrupt accumulated gradients.
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device).clamp_min(1)
        else:
            num_items_in_batch = max(1, num_items_in_batch)
        loss = loss / num_items_in_batch

    return loss


# ---------------------------------------------------------------------------
# for_causal_lm_loss
# ---------------------------------------------------------------------------

def for_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int | None = None,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    reduction: str = "mean",
    _skip_upcast: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Causal LM cross-entropy loss with label shifting and optional normalisation.

    In an autoregressive model logit[t] predicts token[t+1], so labels must be
    shifted left by one position before the cross-entropy is computed.  This
    function handles that shift, flattens the (B, L, V) logit tensor to
    (B*L, V), and delegates to fixed_cross_entropy.

    VRAM contract
    -------------
    Preferred calling pattern (from score_label_logprob):
      1. Caller upcasts logits to fp32 and immediately deletes the original.
      2. Caller passes _skip_upcast=True so this function never creates a second
         fp32 copy.
    This guarantees only ONE copy of the logit tensor is in memory at any time.

    When _skip_upcast=False (default / backward-compatible), this function
    upcasts internally, which transiently holds both the original low-precision
    tensor and the fp32 copy simultaneously.  This is acceptable only when the
    caller has already released its reference before calling.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, L, V)
        Model output logits for every sequence position.  B = batch size,
        L = sequence length, V = vocabulary size.
    labels : torch.Tensor, shape (B, L)
        Integer token ids.  Positions with value *ignore_index* (-100) are
        excluded from the loss.
    vocab_size : int | None
        Vocabulary dimension.  Inferred from logits.size(-1) when None.
    num_items_in_batch : torch.Tensor | None
        Optional manual normalisation divisor (see fixed_cross_entropy).
    ignore_index : int
        Token id to exclude from the loss (default -100).
    shift_labels : torch.Tensor | None
        Pre-shifted label tensor of shape (B, L) — same shape as labels — where
        position t holds the target for logit t (i.e. the original labels shifted
        left by one, with the last position filled with ignore_index).  When
        provided the internal shift step is skipped entirely.  Useful if the
        caller has already performed the shift.  NOTE: passing (B, L-1) will
        cause a shape mismatch in cross_entropy since logits_flat has B*L rows.
    reduction : str
        "mean" | "sum" | "none" — controls aggregation (see fixed_cross_entropy).
    _skip_upcast : bool
        Set True when the caller has already upcasted logits to fp32.  Prevents
        a redundant copy inside this function.
    **kwargs
        Forwarded to fixed_cross_entropy / F.cross_entropy.

    Returns
    -------
    torch.Tensor
        Scalar loss (reduction in {"mean","sum"}) or per-token tensor of
        shape (B, L) when reduction == "none".  The last column is always
        zero because shift_labels[..., -1] is padded with ignore_index.
    """
    # Infer vocab size from the logit tensor when not provided explicitly.
    if vocab_size is None:
        vocab_size = logits.size(-1)

    # Shape guard: catch dimension mismatches before they produce cryptic
    # errors deep inside the flat-view cross-entropy call.
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"for_causal_lm_loss expects logits (B, L, V) and labels (B, L); "
            f"got logits {tuple(logits.shape)} and labels {tuple(labels.shape)}."
        )
    if logits.size(0) != labels.size(0) or logits.size(1) != labels.size(1):
        raise ValueError(
            f"Batch/sequence dimension mismatch: "
            f"logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}."
        )

    # Upcast to fp32 for numerical stability.  When _skip_upcast=True the
    # caller guarantees the tensor is already fp32, so no copy is needed.
    if not _skip_upcast and logits.dtype in (torch.float16, torch.bfloat16):
        # Rebinds the local name; the caller must have released its reference
        # before calling so only this fp32 copy exists in VRAM.
        logits = logits.float()

    # Label shifting: each logit at position t predicts the token at t+1.
    # Pad with ignore_index so the last position has no target, then slice
    # from index 1 onward to align token t+1 with logit t.
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)   # (B, L+1)
        shift_labels = labels[..., 1:].contiguous()           # (B, L)

    # Flatten (B, L, V) → (B*L, V) for cross-entropy.
    logits_flat = logits.view(-1, vocab_size)

    # Free the fp32 copy immediately after the flat view is created so that
    # the flat view is the only live reference to this data.
    del logits

    # Move shifted labels to the same device as logits (handles CPU inference).
    shift_labels_flat = shift_labels.view(-1).to(logits_flat.device)

    # Compute the actual cross-entropy loss.
    loss = fixed_cross_entropy(
        logits_flat,
        shift_labels_flat,
        num_items_in_batch=num_items_in_batch,
        ignore_index=ignore_index,
        reduction=reduction,
        **kwargs,
    )

    # Reshape per-token losses back to (B, L) so callers can apply masking.
    if reduction == "none":
        return loss.view_as(shift_labels)
    return loss
