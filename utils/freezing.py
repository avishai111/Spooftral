"""
freezing.py
===========
Model freeze-state inspection utilities for Voxtral.

Public API
----------
  verify_model_frozen  : Print a requires_grad summary (read-only).
"""


# ---------------------------------------------------------------------------
# verify_model_frozen
# ---------------------------------------------------------------------------

def verify_model_frozen(model) -> None:
    """
    Print a requires_grad summary without modifying the model.

    Intended as a read-only diagnostic tool, typically called before/after
    inference to confirm the model's parameters are in the expected frozen
    state. This function NEVER calls requires_grad_() — it only inspects the
    current state.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch module.  Typically model_wrapper or model_wrapper.model.

    Local variables
    ---------------
    trainable : list of str — names of parameters with requires_grad=True.
    total     : int — total number of scalar parameters across all tensors.

    Returns
    -------
    None
        Prints a human-readable summary to stdout.
    """
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        print(f"[VERIFY] {len(trainable)} parameter(s) have requires_grad=True:")
        # Print up to the first 10 names to keep the output manageable.
        for n in trainable[:10]:
            print(f"  {n}")
        if len(trainable) > 10:
            print(f"  ... and {len(trainable) - 10} more")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"[VERIFY] All {total:,} parameters are frozen. Inference-only mode confirmed.")
