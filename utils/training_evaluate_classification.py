"""
training_evaluate_classification.py
=====================================
Main training loop and supporting helpers for Classification-as-Generation (CAG)
audio deepfake detection with Voxtral.

Entry point
-----------
  train_classification_model : Full training loop (epoch-based or step-based).

Supporting helpers
------------------
  get_lm_loss_from_outputs   : Extract per-sample LM NLL from a model output.
  create_wandb_logger        : Build a W&B logging closure with step offset.

Architecture overview
---------------------
For each batch:
  1. Audio encoder runs ONCE, producing shared embeddings for both label candidates.
  2. The two candidate sequences ([prompt|bonafide] and [prompt|spoof]) are combined
     into a single 2B-row batch.
  3. A single LLM forward pass computes both log-probabilities simultaneously.
  4. Softmax CE over [logP_label0, logP_label1] vs. the true label forms the primary
     loss, with optional margin and teacher-forcing auxiliary losses.

Training is controlled exclusively via YAML config (generation_cls_cfg).
"""

import json
import math
import os
import shutil
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils.helper_function import maybe_log_eval_stats
from utils.checkpoint_utils import resume_checkpoint_evaluation, split_metrics_example
from utils.generation_text_bonafide_spoof_classification import (
    build_cag_candidate,
    build_cag_candidates_batched,  
    make_lm_labels,
    score_label_logprob,
    pad_right,  
)


from utils.training_helpers import (
    dprint as _dprint,
    next_available,
    print_asvspoof2021_eval_info,
    save_checkpoint,
    safe_int,
)
from utils.config_helper_functions import resolve_label_texts

# ======== Eval =========
from utils.eval_classification import (
    evaluate_epoch_classification_as_generation,
)
try:
    # Prefer the resolved DEBUG flag from utils.config when available
    from utils import config as _global_config
except Exception:  
    _global_config = None


# ======== Helper Functions =========

def get_lm_loss_from_outputs(outs):
    """
    Extract per-sample LM loss from a model output object.

    Checks for the preferred 'lm_loss_per_sample' key first (attached by
    score_label_logprob when return_outputs=True), then falls back to 'lm_loss'
    and 'loss' for compatibility with older output formats.

    Note: the training loop uses score_label_logprob(..., return_lm_loss=True)
    which returns the loss tensor directly — the full output object is never
    kept alive.  This helper exists for external callers and evaluation utilities
    that still consume the full output dict/object.

    Parameters
    ----------
    outs : dict, object, or None
        Model output.  Accepts both dict-like and attribute-based containers.

    Local variables
    ---------------
    cand : torch.Tensor or None — the candidate loss tensor found in outs.

    Returns
    -------
    torch.Tensor or None
        Per-sample LM loss tensor, or None if not present.
    """
    if outs is None:
        return None
    if isinstance(outs, dict):
        cand = outs.get("lm_loss_per_sample")
        if cand is None:
            cand = outs.get("lm_loss")
        return cand
    cand = getattr(outs, "lm_loss_per_sample", None)
    if cand is None:
        cand = getattr(outs, "lm_loss", None)
    return cand


def create_wandb_logger(wandb_run, wandb_step_offset=0):
    """
    Create a W&B logging closure with an optional step offset.

    When a training run is resumed, the W&B run may have already logged up to
    a higher step than the current optimizer_step.  Logging with raw
    optimizer_step values would create a gap or overwrite existing points.
    wandb_step_offset shifts all logged steps forward to maintain monotonicity.

    Parameters
    ----------
    wandb_run          : wandb.Run or None
        Active W&B run handle.  When None, the closure is a no-op.
    wandb_step_offset  : int
        Added to every logged step to keep the run's step sequence monotonic
        after a resume.  Typically computed from wandb_run.summary["_step"].

    Local variables (inside closure)
    ---------------------------------
    payload   : dict — the metrics dict to log.
    step      : int or None — optimizer step at log time.
    log_step  : int or None — step + wandb_step_offset.

    Returns
    -------
    Callable[[dict, Optional[int]], None]
        log_wandb(payload, step=None) — logs payload to W&B if wandb_run is set.
    """
    def log_wandb(payload, step=None):
        if wandb_run is None:
            return
        log_step = step + wandb_step_offset if step is not None else None
        wandb_run.log(payload, step=log_step)
    return log_wandb



# ======== Train =========
def train_classification_model(
    model,
    train_loader,
    dev_loader,
    eval_loader,
    processor,
    optimizer,
    scheduler,
    DEVICE,
    NUM_EPOCHS,
    OUTPUT_DIR,
    train_eval_loader=None,
    LOG_EVERY=100,
    EVAL_EVERY=1,
    SAVE_EVERY = 1,
    use_fp16=False,
    use_bf16=False,
    use_amp = False,
    scaler=None,
    max_grad_norm=None,
    accum=1,
    result_file=None,
    metrics_log_file=None,
    eval_log_file=None,
    train_eval_log_file=None,
    evaluate_epoch=evaluate_epoch_classification_as_generation,
    log_metrics=None,
    configs=None,
    config_dict=None,
    log_eval_examples=True,
    start_epoch=0, 
    best_eer=100.0,  
    best_eval_eer=100.0,  
    wandb_run=None,
    run_by_epochs=True,
    max_train_steps=None,
    eval_every_steps=None,
    save_every_steps=None,
    start_step=0,
    enable_debug_logging=True,
    save_lr_and_grad_norm_debug=False,
    resume_eval_before_train=False,
    last_eval_epoch=None,
    last_eval_step=None,
    last_test_eval_epoch=None,
    last_test_eval_step=None,
    last_train_eval_epoch=None,
    last_train_eval_step=None,
    resume_dev_eval_needed=False,
    resume_test_eval_needed=False,
    resume_train_eval_needed=False,
    resume_asvspoof2021_eval_needed=False,
    asvspoof2021_eval_loader=None,
    asvspoof2021_eval_log_file=None,
    last_asvspoof2021_eval_epoch=None,
    last_asvspoof2021_eval_step=None,
    generation_cls_cfg=None,
    use_weighted_loss=False,
    class_weights=None,
    initial_eval_steps=None,
    initial_eval_steps_eval=None,
):
    """
    Main training loop for Classification-as-Generation (CAG) models.

    Architecture overview
    ---------------------
    For each batch the loop executes one Whisper encoder pass (shared across both
    label candidates) and ONE LLM forward pass over a combined 2B-sample batch
    [cand_label0 | cand_label1].  This halves LLM kernel launches compared to
    running candidates separately, and lets gradient checkpointing amortise
    activation recomputation across both.

    Tokenization (build_cag_candidate, pad_sequence) runs BEFORE the AMP context
    because it is CPU-only and does not benefit from mixed-precision autocast.

    The AMP context (torch.autocast) is created once before the epoch loop and
    reused across all batch steps — not recreated per batch.

    Loss components (all read from YAML / generation_cls_cfg)
    ---------------------------------------------------------
    1. CAG cross-entropy: softmax over [logP(label_0), logP(label_1)] vs. true label.
       Always operates on length-normalised log-probs (normalize_logprob_by_tokens)
       to prevent bias toward shorter label strings.

    2. Margin loss (optional, use_margin_loss: True):
       Hinge loss — relu(margin - (logP_correct - logP_wrong)).
       Uses the same length-normalised logprobs as CE so the margin threshold is
       consistent regardless of label token count.
       Per-class margins (margin_bonafide, margin_spoof) are assigned by
       matching label_texts_map values against "bonafide"/"spoof" substrings,
       NOT by hardcoded index. This preserves correctness when the YAML order
       is swapped or custom label names are used.

    3. Teacher-forcing loss (optional, teacher_forcing_loss: True):
       Average NLL of generating the true label token-by-token.
       Only the ground-truth label's NLL is included — the wrong candidate's
       loss is never added. Extracted as a scalar via return_lm_loss=True,
       so the full logit tensor is never kept alive during this computation.

    Resume behaviour
    ----------------
    * best_eval_eer is restored from the checkpoint (passed in as an argument
      by the main script) so spurious "best eval" checkpoint saves do not occur
      after a resume. The previous bug where this defaulted to 100.0 is fixed.

    VRAM strategy
    -------------
    * score_label_logprob deletes its logits reference before for_causal_lm_loss
      upcasts to fp32, preventing a 3× spike on the largest tensor per step.
    * Individual per-candidate tensors (cand_b, attn_b, mask_b, …) are deleted
      immediately after being folded into the combined 2B batch.
    * The teacher-forcing NLL is a scalar per sample — the full output object is
      never stored.

    Debug mode
    ----------
    When debug_mode=True the loop scores only the true label (not both candidates)
    to cut memory. The resulting loss is an unconditional LM NLL, NOT the same
    quantity as production CE. It is logged under "debug_lm_nll" and printed via
    dprint() to make the incomparability explicit.

    Args
    ----
        generation_cls_cfg : dict from YAML generation_classification section.
        use_weighted_loss  : apply per-class weights to the CE loss.
        class_weights      : tensor of shape (num_classes,); auto-computed or from YAML.
        best_eval_eer      : initial best eval EER; must be restored from checkpoint
                             by the caller (main script) before passing here.
    """
    # ===== SETUP OUTPUT DIRECTORY AND LOG FILES =====
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create default log file paths if not provided
    if result_file is None:
        result_file = os.path.join(OUTPUT_DIR, "train_log.txt")
    if metrics_log_file is None:
        metrics_log_file = os.path.join(OUTPUT_DIR, "dev_metrics.log")
    if eval_log_file is None:
        eval_log_file = os.path.join(OUTPUT_DIR, "test_metrics.log")
    if train_eval_log_file is None and train_eval_loader is not None:
        train_eval_log_file = os.path.join(OUTPUT_DIR, "metrics_train_log.txt")
    if asvspoof2021_eval_log_file is None and asvspoof2021_eval_loader is not None:
        asvspoof2021_eval_log_file = os.path.join(OUTPUT_DIR, "asvspoof2021_metrics.log")


    # ===== CREATE LOG FILES WITH AUTO-INCREMENTING NAMES =====
    # On fresh runs, next_available increments the filename to avoid overwriting.
    # On resume (start_epoch > 0 or start_step > 0), keep the existing filenames so
    # training history stays in one file instead of being fragmented across log_1, log_2, etc.
    _is_resume = start_epoch > 0 or start_step > 0
    if not _is_resume:
        result_file = next_available(result_file)
        metrics_log_file = next_available(metrics_log_file)
        eval_log_file = next_available(eval_log_file)
        if train_eval_log_file is not None and train_eval_loader is not None:
            train_eval_log_file = next_available(train_eval_log_file)
        if asvspoof2021_eval_log_file is not None:
            asvspoof2021_eval_log_file = next_available(asvspoof2021_eval_log_file)

    debug_log_path = os.path.join(OUTPUT_DIR, "debug.log")   
    eval_debug_log_path =  os.path.join(OUTPUT_DIR, "eval_debug.log") 
    lr_grad_log_path = None
    if save_lr_and_grad_norm_debug:
        lr_grad_log_path = os.path.join(OUTPUT_DIR, "lr_grad_norm_debug.txt")
        with open(lr_grad_log_path, "w", encoding="utf-8") as f:
            f.write("step\tlr\tgrad_norm\n")

    # ===== SETUP DEBUG LOGGING =====
    # Attach debug log paths to model if it supports them
    target = model.module if hasattr(model, "module") else model
    if enable_debug_logging and hasattr(target, "debug_log_path"):
        target.debug_log_path = debug_log_path
        

    # Bind debug log path and flag; message args are passed at call time
    dprint = partial(_dprint, debug_log_path, enable_debug_logging)


    # ===== CLASSIFICATION-AS-GENERATION (CAG) CONFIGURATION =====
    # Parse CAG-specific parameters: label texts, temperature, normalization, etc.
    gen_cfg = generation_cls_cfg or {}
    label_texts_map = resolve_label_texts(gen_cfg.get("label_texts", {}), context_label="Training")
    
    max_answer_tokens = gen_cfg.get("max_answer_tokens", None)
    cag_loss_weight = float(gen_cfg.get("loss_weight", 1.0))
    temperature = float(gen_cfg.get("temperature", 1.0) or 1.0)
    normalize_logprob_by_tokens = bool(gen_cfg.get("normalize_logprob_by_tokens", True))
    use_teacher_forcing_loss = bool(gen_cfg.get("teacher_forcing_loss", False))
    teacher_forcing_lambda = gen_cfg.get("teacher_forcing_loss_lambda", gen_cfg.get("teacher_forcing_lambda", 0.0))
    try:
        teacher_forcing_lambda = float(teacher_forcing_lambda)
    except Exception:
        teacher_forcing_lambda = 0.0
    if teacher_forcing_lambda < 0.0:
        raise ValueError(
            f"teacher_forcing_lambda={teacher_forcing_lambda} is negative. "
            "Set a value in [0.0, 1.0] in the YAML."
        )

    if use_teacher_forcing_loss and teacher_forcing_lambda > 0.0:
        import math as _math
        import warnings as _warnings
        _ce_max = _math.log(max(2, len(label_texts_map)))  # e.g. ln(2) ≈ 0.693
        _estimated_tf_nll = 3.0  # typical initial LM NLL for a LoRA fine-tune
        if teacher_forcing_lambda * _estimated_tf_nll > 2.0 * _ce_max:
            _warnings.warn(
                f"teacher_forcing_lambda={teacher_forcing_lambda:.3f}: the "
                f"teacher-forcing loss (~{_estimated_tf_nll:.1f} nats × lambda "
                f"≈ {teacher_forcing_lambda * _estimated_tf_nll:.2f} nats) will "
                f"likely dominate the CAG cross-entropy loss (max {_ce_max:.3f} "
                f"nats for {len(label_texts_map)} classes). "
                f"Consider lambda < {_ce_max / _estimated_tf_nll:.3f} for balanced "
                f"training, or confirm the dominance is intentional.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Margin-based loss (optional) — enforces a minimum score gap between classes.
    # Margins are read from YAML so the threshold is fully configurable without code changes.
    use_margin_loss = bool(getattr(configs, "use_margin_loss", False)) if configs is not None else False
    margin_bonafide_raw = getattr(configs, "margin_bonafide", None) if configs is not None else None
    margin_spoof_raw = getattr(configs, "margin_spoof", None) if configs is not None else None
    try:
        margin_bonafide = float(margin_bonafide_raw) if margin_bonafide_raw is not None else 1.0
    except Exception:
        margin_bonafide = 1.0
    try:
        margin_spoof = float(margin_spoof_raw) if margin_spoof_raw is not None else 1.0
    except Exception:
        margin_spoof = 1.0
    lambda_margin = float(getattr(configs, "lambda_margin", 0.3)) if configs is not None else 0.3
    use_weighted_aux_losses = bool(getattr(configs, "use_weighted_aux_losses", False)) if configs is not None else False

    _SPOOF_SYNONYMS = ("spoof", "fake", "synthetic", "generated", "artificial")
    _BONA_SYNONYMS = ("bonafide", "real", "genuine", "authentic", "natural")
    _spoof_key = next(
        (k for k, v in label_texts_map.items() if any(w in str(v).lower() for w in _SPOOF_SYNONYMS)), 1
    )
    _bonafide_key = next(
        (k for k, v in label_texts_map.items() if any(w in str(v).lower() for w in _BONA_SYNONYMS)), 0
    )
    print(
        f"[Margin Loss] Semantic mapping — bonafide key: {_bonafide_key} "
        f"('{label_texts_map.get(_bonafide_key)}'), "
        f"spoof key: {_spoof_key} ('{label_texts_map.get(_spoof_key)}')"
    )



    print_asv2021_info = partial(print_asvspoof2021_eval_info, configs, asvspoof2021_eval_loader)
    debug_source = None
    if _global_config is not None:
        debug_source = getattr(_global_config, "DEBUG", None)
    if debug_source is None:
        debug_source = getattr(configs, "DEBUG", None)
    if debug_source is None:
        debug_source = getattr(configs, "debug", False)
    debug_mode = bool(debug_source)
    save_best_only = False
    if configs is not None:
        model_cfg = getattr(configs, "model", None)
        if isinstance(model_cfg, dict):
            save_best_only = bool(model_cfg.get("save_best_only", False))
        else:
            save_best_only = bool(getattr(model_cfg, "save_best_only", False))
        save_best_only = save_best_only or bool(getattr(configs, "save_best_only", False))
    save_only_trainable = False
    save_only_adapters = False
    if configs is not None:
        model_cfg = getattr(configs, "model", None)
        if isinstance(model_cfg, dict):
            save_only_trainable = bool(model_cfg.get("save_only_trainable", False))
            save_only_adapters  = bool(model_cfg.get("save_only_adapters", False))
        else:
            save_only_trainable = bool(getattr(model_cfg, "save_only_trainable", False))
            save_only_adapters  = bool(getattr(model_cfg, "save_only_adapters", False))
        save_only_trainable = save_only_trainable or bool(getattr(configs, "save_only_trainable", False))
        save_only_adapters  = save_only_adapters  or bool(getattr(configs, "save_only_adapters", False))
    if save_only_trainable and save_only_adapters:
        raise ValueError("save_only_trainable and save_only_adapters cannot both be True.")
    if save_only_adapters:
        _m_init = model.module if hasattr(model, "module") else model
        _non_adapter_trainable = [
            n for n, p in _m_init.named_parameters()
            if p.requires_grad and "lora_" not in n and "dora_" not in n
        ]
        if _non_adapter_trainable:
            raise ValueError(
                f"save_only_adapters=True but {len(_non_adapter_trainable)} trainable "
                f"non-adapter parameter(s) exist (e.g. {_non_adapter_trainable[:3]}). "
                "Use save_only_trainable=True instead, or freeze these parameters."
            )
    if debug_mode:
        # shorten answer span and disable teacher forcing to save memory
        if max_answer_tokens is None:
            max_answer_tokens = 4
        else:
            max_answer_tokens = min(max_answer_tokens, 4)
        print("[DEBUG] teacher_forcing_loss disabled in debug mode.")
        use_teacher_forcing_loss = False

    last_eval_epoch = -1 if last_eval_epoch is None else int(last_eval_epoch)
    last_eval_step = -1 if last_eval_step is None else int(last_eval_step)
    last_train_eval_epoch = -1 if last_train_eval_epoch is None else int(last_train_eval_epoch)
    last_train_eval_step = -1 if last_train_eval_step is None else int(last_train_eval_step)
    last_test_eval_epoch = -1 if last_test_eval_epoch is None else int(last_test_eval_epoch)
    last_test_eval_step = -1 if last_test_eval_step is None else int(last_test_eval_step)
    last_asvspoof2021_eval_epoch = -1 if last_asvspoof2021_eval_epoch is None else int(last_asvspoof2021_eval_epoch)
    last_asvspoof2021_eval_step = -1 if last_asvspoof2021_eval_step is None else int(last_asvspoof2021_eval_step)

    # ===== INITIALIZE MODEL AND DEBUG PRINT FUNCTION =====
    model = model.to(DEVICE)
    model.train()


    # ===== MIXED PRECISION TRAINING SETUP =====
    if use_amp and use_bf16:
        raise ValueError("use_amp (fp16) and use_bf16 are mutually exclusive — set only one in the config.")
    # Initialize gradient scaler for FP16/BF16 training
    if scaler is None:
        scaler = torch.amp.GradScaler(enabled=(use_amp or use_fp16))

    eval_use_amp = bool(use_amp or use_fp16 or use_bf16)
    eval_amp_dtype = "bf16" if use_bf16 else "fp16"

    # ===== COMPUTE TRAINING STEPS AND RESUME OFFSET =====
    # Calculate total steps per epoch and handle resume logic
    steps_per_epoch = math.ceil(len(train_loader) / max(1, accum))
    # When resuming in step-based mode, use start_step directly if provided
    # Only derive from start_epoch if we're in epoch-based mode and start_step is 0
    if start_step and start_step > 0:
        optimizer_step = int(start_step)
    elif start_epoch > 0:
        optimizer_step = start_epoch * steps_per_epoch
    else:
        optimizer_step = 0

    # Keep W&B logging steps monotonic when resuming runs that already logged further
    wandb_step_offset = 0
    if wandb_run is not None:
        candidates = [
            safe_int(getattr(wandb_run, "step", None)),
            safe_int(getattr(wandb_run, "_step", None)),
        ]
        try:
            candidates.append(safe_int(wandb_run.summary.get("_step")))
            candidates.append(safe_int(wandb_run.summary.get("global_step")))
            candidates.append(safe_int(wandb_run.summary.get("train/optimizer_step")))
        except Exception:
            pass
        candidates = [c for c in candidates if c is not None]
        if candidates:
            wandb_resume_step = max(candidates)
            if wandb_resume_step > optimizer_step:
                wandb_step_offset = wandb_resume_step - optimizer_step
                print(f"[W&B] Resuming run from step {wandb_resume_step}; shifting log steps by +{wandb_step_offset} to keep monotonic.")

    # ===== CREATE WANDB LOGGER =====
    # Create W&B logging function with resume offset
    log_wandb = create_wandb_logger(wandb_run, wandb_step_offset)

    log_eval_stats_enabled = bool(
        (configs is not None and getattr(configs, "log_eval_stats", False))
        or (_global_config is not None and getattr(_global_config, "LOG_EVAL_STATS", False))
    )

    _consecutive_eval_failures = 0  # tracks back-to-back non-OOM eval failures

    def _safe_eval_split(split_name: str, fn, *args, **kwargs):
        """
        Run one evaluation split without letting a failure stop training.

        Wraps the evaluate_epoch call in a try/except so that a crash on one
        split (e.g. OOM on the large test set) does not abort the other splits
        or the training loop itself.

        OOM is always re-raised immediately (the model is in an unknown state).
        Other exceptions are swallowed up to 2 consecutive failures; on the
        3rd consecutive failure training is aborted to prevent silent loss of
        all checkpoints (e.g. a systematic ValueError in the eval pipeline
        that would never resolve on its own).

        Parameters
        ----------
        split_name : str — human-readable label used in the warning message.
        fn         : callable — the evaluate_epoch function to call.
        *args, **kwargs : forwarded to fn.

        Returns
        -------
        (metrics, None)   on success — metrics dict from fn.
        (None, exception) on failure — exception for the caller to inspect.
        """
        nonlocal _consecutive_eval_failures
        try:
            result = fn(*args, **kwargs), None
            _consecutive_eval_failures = 0  # reset counter on any success
            return result
        except torch.cuda.OutOfMemoryError as exc:  # OOM during eval is a hard error
            # Re-raise OOM rather than swallowing — a silent OOM leaves the model in
            # an unknown state and continuing would produce meaningless checkpoints.
            print(f"[FATAL] {split_name} evaluation OOM: {exc}", flush=True)
            raise
        except Exception as exc:  # pragma: no cover - defensive path
            import traceback
            tb_str = traceback.format_exc()
            _consecutive_eval_failures += 1
            print(
                f"[Warn] {split_name} evaluation failed "
                f"({_consecutive_eval_failures} consecutive): {exc}\n{tb_str}",
                flush=True,
            )
            if _consecutive_eval_failures >= 3:
                raise RuntimeError(
                    f"Evaluation failed {_consecutive_eval_failures} consecutive times "
                    f"(last split: {split_name}). Aborting training to prevent silent "
                    f"loss of all checkpoints. Last error: {exc}"
                ) from exc
            return None, exc

    # ===== OPTIONAL INITIAL EVALUATION STEPS =====
    def _normalize_initial_steps(val):
        try:
            return sorted({int(s) for s in (val or []) if s is not None and int(s) >= 0})
        except Exception:
            return []

    initial_eval_steps = _normalize_initial_steps(initial_eval_steps)
    initial_eval_steps_set = set(initial_eval_steps)
    initial_eval_done = set()

    initial_eval_steps_eval = _normalize_initial_steps(initial_eval_steps_eval)
    initial_eval_steps_eval_set = set(initial_eval_steps_eval)
    initial_eval_eval_done = set()

    def _run_initial_dev_eval(step_val: int):
        nonlocal best_eer, last_eval_step
        if evaluate_epoch is None or dev_loader is None:
            return
        print(f"[Initial Eval] step {step_val} on dev", flush=True)
        metrics, dev_err = _safe_eval_split(
            "Dev initial",
            evaluate_epoch,
            model,
            dev_loader,
            processor,
            DEVICE,
            use_amp=eval_use_amp,
            amp_dtype=eval_amp_dtype,
            debug_log_path=debug_log_path,
            enable_debug_logging=enable_debug_logging,
            generation_cfg=generation_cls_cfg,
            return_example=log_eval_examples,
            eval_step=step_val,
        )
        if dev_err or metrics is None:
            return
        metrics_clean, example_entry = split_metrics_example(metrics)
        dev_eval_log_payload = {f"val/{k}": v for k, v in metrics_clean.items()}
        print(f"[Initial Eval] step {step_val}: {dev_eval_log_payload}", flush=True)
        if example_entry:
            print(f"[Eval example] {example_entry}", flush=True)
        if log_metrics is not None:
            log_metrics(log_file=metrics_log_file, step=step_val, metrics=metrics_clean, examples=example_entry)
        log_wandb(dev_eval_log_payload, step=step_val)
        last_eval_step = max(last_eval_step, step_val)
        eer = metrics_clean.get("eer")
        if eer is not None and math.isfinite(eer) and eer < best_eer:
            best_eer = eer
            if wandb_run:
                wandb_run.summary["best_eer"] = best_eer

    def _run_initial_eval_eval(step_val: int):
        nonlocal best_eval_eer, last_test_eval_step
        if evaluate_epoch is None or eval_loader is None:
            return
        print(f"[Initial Eval] step {step_val} on eval", flush=True)
        metrics, eval_err = _safe_eval_split(
            "Eval initial",
            evaluate_epoch,
            model,
            eval_loader,
            processor,
            DEVICE,
            use_amp=eval_use_amp,
            amp_dtype=eval_amp_dtype,
            debug_log_path=debug_log_path,
            enable_debug_logging=enable_debug_logging,
            generation_cfg=generation_cls_cfg,
            return_example=log_eval_examples,
            eval_step=step_val,
            return_eval_stats=log_eval_stats_enabled,
        )
        if eval_err or metrics is None:
            return
        eval_stats = None
        if log_eval_stats_enabled and isinstance(metrics, dict):
            eval_stats = metrics.pop("eval_stats", None)
        metrics_clean, example_entry = split_metrics_example(metrics)
        eval_log_payload = {f"eval/{k}": v for k, v in metrics_clean.items()}
        print(f"[Initial Eval] step {step_val}: {eval_log_payload}", flush=True)
        if example_entry:
            print(f"[Eval example - test] {example_entry}", flush=True)
        if log_metrics is not None:
            log_metrics(log_file=eval_log_file, step=step_val, metrics=metrics_clean, examples=example_entry)
        log_wandb(eval_log_payload, step=step_val)
        last_test_eval_step = max(last_test_eval_step, step_val)
        eval_eer = metrics_clean.get("eer")
        if eval_eer is not None and math.isfinite(eval_eer) and eval_eer < best_eval_eer:
            best_eval_eer = eval_eer
            if wandb_run:
                wandb_run.summary["best_eval_eer"] = best_eval_eer
        if eval_stats:
            maybe_log_eval_stats(
                enabled=log_eval_stats_enabled,
                log_file=eval_log_file,
                header=f"Initial step {step_val} (eval set)",
                eval_stats=eval_stats,
            )

    # ===== RESUME EVALUATION LOGIC =====

    stop_training = False
    # debug_mode was resolved once above from _global_config / configs; do NOT
    # re-derive it here — a second getattr(configs, "debug", False) would silently
    # overwrite the value when configs.debug is absent (defaults to False) even if
    # the global DEBUG flag was True.  

    # When resuming from a checkpoint, optionally evaluate before continuing
    resuming_from_checkpoint = bool(start_epoch) or bool(start_step)
    if resuming_from_checkpoint and resume_eval_before_train and evaluate_epoch is not None:
        try:
            (last_eval_epoch, last_eval_step, last_train_eval_epoch, last_train_eval_step,
             last_test_eval_epoch, last_test_eval_step, last_asvspoof2021_eval_epoch, last_asvspoof2021_eval_step) = resume_checkpoint_evaluation(
                model=model,
                evaluate_epoch=evaluate_epoch,
                processor=processor,
                DEVICE=DEVICE,
                start_epoch=start_epoch,
                optimizer_step=optimizer_step,
                run_by_epochs=run_by_epochs,
                dev_loader=dev_loader,
                train_eval_loader=train_eval_loader,
                eval_loader=eval_loader,
                asvspoof2021_eval_loader=asvspoof2021_eval_loader,
                eval_use_amp=eval_use_amp,
                eval_amp_dtype=eval_amp_dtype,
                debug_log_path=debug_log_path,
                eval_debug_log_path=eval_debug_log_path,
                enable_debug_logging=enable_debug_logging,
                log_eval_examples=log_eval_examples,
                resume_dev_eval_needed=resume_dev_eval_needed,
                resume_train_eval_needed=resume_train_eval_needed,
                resume_test_eval_needed=resume_test_eval_needed,
                resume_asvspoof2021_eval_needed=resume_asvspoof2021_eval_needed,
                log_metrics=log_metrics,
                metrics_log_file=metrics_log_file,
                train_eval_log_file=train_eval_log_file,
                eval_log_file=eval_log_file,
                asvspoof2021_eval_log_file=asvspoof2021_eval_log_file,
                log_wandb=log_wandb,
                print_asvspoof2021_info=print_asv2021_info,
                configs=configs,
                log_eval_stats_enabled=log_eval_stats_enabled,
                last_eval_epoch=last_eval_epoch,
                last_eval_step=last_eval_step,
                last_train_eval_epoch=last_train_eval_epoch,
                last_train_eval_step=last_train_eval_step,
                last_test_eval_epoch=last_test_eval_epoch,
                last_test_eval_step=last_test_eval_step,
                last_asvspoof2021_eval_epoch=last_asvspoof2021_eval_epoch,
                last_asvspoof2021_eval_step=last_asvspoof2021_eval_step,
                generation_cfg=generation_cls_cfg,
            )
            model.train()
        except Exception as e:
            print(f"[Resume Eval] Failed: {e}")
            model.train()
    elif resuming_from_checkpoint:
        resume_desc = f"epoch {start_epoch}" if run_by_epochs else f"step {optimizer_step}"
        print(f"[Resume] Continuing training from {resume_desc} (initial evaluation skipped).")

    if not run_by_epochs and max_train_steps is None:
        raise ValueError("max_train_steps must be provided when training by steps.")

    if not run_by_epochs and max_train_steps is not None and optimizer_step >= max_train_steps:
        print(f"[Train] Already reached max_train_steps ({optimizer_step} >= {max_train_steps}). Exiting.")
        return best_eer

    # Run baseline evaluations before any training updates if requested
    if 0 in initial_eval_steps and (not resuming_from_checkpoint or optimizer_step == 0):
        _run_initial_dev_eval(0)
        initial_eval_done.add(0)
    if 0 in initial_eval_steps_eval and (not resuming_from_checkpoint or optimizer_step == 0):
        _run_initial_eval_eval(0)
        initial_eval_eval_done.add(0)

    # ===== CHECKPOINT HELPER =====
    def _build_checkpoint(epoch_num, step_num, extra=None, epoch_completed=False):
        """
        Build the full checkpoint dict for saving.

        Reads best_eer and all last_*_epoch / last_*_step counters from the
        enclosing scope's closure so the saved values always reflect the state
        at the moment of saving, not at function definition time.

        Parameters
        ----------
        epoch_num       : int — epoch number to embed in the checkpoint.
                          Always the raw 0-indexed loop variable at save time, for both
                          end-of-epoch and mid-epoch saves.  The completed epoch in 1-based
                          counting is epoch_num+1; resume logic uses epoch_completed to decide
                          whether to continue from epoch_num+1 or replay epoch_num.
        step_num        : int — global optimizer step to embed.
        extra           : dict or None — additional key-value pairs merged into the
                          checkpoint dict.  Used for keys like "best_dev_eer" that are
                          not always present in the base dict.
        epoch_completed : bool — True when saving at the end of a completed epoch,
                          False when saving mid-epoch (step-based periodic save).
                          Stored in the checkpoint so the resume logic can distinguish
                          "resume from next epoch" vs "replay current epoch from start".

        Local variables
        ---------------
        m         : nn.Module — model with any DDP wrapper stripped.
        ckpt      : dict — the assembled checkpoint dict.

        Returns
        -------
        dict
            Complete checkpoint dict suitable for torch.save.
        """
        import random
        import numpy as np
        m = model.module if hasattr(model, "module") else model
        # Determine which model weights to include in the checkpoint.
        # state_dict() is called once and reused across all branches.
        _full_state = m.state_dict()
        if save_only_adapters:
            _model_state_entry = {"adapter_state_dict": {
                k: v for k, v in _full_state.items() if "lora_" in k or "dora_" in k
            }}
        elif save_only_trainable:
            _trainable_names = {n for n, p in m.named_parameters() if p.requires_grad}
            _model_state_entry = {"trainable_state_dict": {
                k: v for k, v in _full_state.items() if k in _trainable_names
            }}
        else:
            _model_state_entry = {"wrapper_state_dict": _full_state}  # full wrapper state
        ckpt = {
            "epoch": epoch_num,
            "epoch_completed": bool(epoch_completed),
            "global_step": int(step_num),
            "best_eer": float(best_eer),
            "best_dev_eer": float(best_eer),
            "best_eval_eer": float(best_eval_eer),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if (use_amp or use_fp16) and scaler is not None else None,
            **_model_state_entry,
            "configs": config_dict,
            "wandb_id": getattr(configs, "wandb_id", None),
            "last_eval_epoch": last_eval_epoch,
            "last_eval_step": last_eval_step,
            "last_train_eval_epoch": last_train_eval_epoch,
            "last_train_eval_step": last_train_eval_step,
            "last_test_eval_epoch": last_test_eval_epoch,
            "last_test_eval_step": last_test_eval_step,
            "last_asvspoof2021_eval_epoch": last_asvspoof2021_eval_epoch,
            "last_asvspoof2021_eval_step": last_asvspoof2021_eval_step,
            "rng_states": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        if extra:
            ckpt.update(extra)
        return ckpt

    def _save_best_checkpoint(
        model_save_dir: str,
        adapters_dir: str,
        ckpt_path: str,
        metrics_json_path: str,
        eer: float,
        metrics: dict,
        label: str,
        epoch_num: int,
        step_num: int,
        extra_ckpt: dict = None,
    ) -> None:
        """
        Save a best-model checkpoint with model weights, adapters, and processor.

        Canonical implementation shared across all four "best checkpoint" save sites
        (step/epoch × dev/eval).  Previously duplicated verbatim at each site; now
        changes only need to happen here.

        Saves in this order:
          1. PEFT adapters (if model uses PEFT) or full model via save_pretrained.
          2. Processor (tokenizer + feature extractor) via save_pretrained.
          3. Full training state checkpoint (.pt) via save_checkpoint.
          4. Metrics JSON for quick inspection without loading the checkpoint.

        Parameters
        ----------
        model_save_dir    : str — directory for model files (created if absent).
        adapters_dir      : str — directory for PEFT adapter files.
        ckpt_path         : str — path for the .pt state checkpoint.
        metrics_json_path : str — path for the metrics JSON file.
        eer               : float — EER value for the log message.
        metrics           : dict — metrics dict written to JSON.
        label             : str — human-readable label (e.g. "best dev (step)").
        epoch_num         : int — epoch number passed to _build_checkpoint.
        step_num          : int — global step passed to _build_checkpoint.
        extra_ckpt        : dict or None — extra keys merged into the checkpoint
                            (e.g. {"best_dev_eer": float}).

        Local variables
        ---------------
        m           : nn.Module — DDP-unwrapped model.
        peft_target : nn.Module or None — the PEFT-wrapped sub-module, if any.
        base        : nn.Module — base model for non-PEFT save_pretrained.
        ckpt        : dict — checkpoint dict from _build_checkpoint.

        Returns
        -------
        None
        """
        m = model.module if hasattr(model, "module") else model
        peft_target = (
            m if hasattr(m, "peft_config")
            else m.model if hasattr(getattr(m, "model", None), "peft_config")
            else None
        )
        os.makedirs(model_save_dir, exist_ok=True)

        # 1. Save the training-state checkpoint FIRST via save_checkpoint (tmp+rename).
        ckpt = _build_checkpoint(epoch_num, step_num, extra=extra_ckpt, epoch_completed=True)
        save_checkpoint(ckpt_path, ckpt)

        # 2. Save PEFT adapters atomically (write to .tmp, then rename).
        if peft_target is not None:
            tmp_adapters = adapters_dir + ".tmp"
            if os.path.exists(tmp_adapters):
                shutil.rmtree(tmp_adapters)
            peft_target.save_pretrained(tmp_adapters)
            if os.path.exists(adapters_dir):
                shutil.rmtree(adapters_dir)
            os.rename(tmp_adapters, adapters_dir)
        else:
            os.makedirs(model_save_dir, exist_ok=True)
            base = getattr(m, "model", m)
            if hasattr(base, "save_pretrained"):
                base.save_pretrained(model_save_dir)
            elif hasattr(m, "save_pretrained"):
                m.save_pretrained(model_save_dir)

        # 3. Save processor.
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(model_save_dir)

        # 4. Save metrics JSON atomically (include checkpoint_name for cleanup protection).
        metrics_with_name = dict(metrics)
        metrics_with_name["checkpoint_name"] = os.path.basename(ckpt_path)
        tmp_json = metrics_json_path + ".tmp"
        with open(tmp_json, "w") as _f:
            json.dump(metrics_with_name, _f, indent=2)
        os.replace(tmp_json, metrics_json_path)

        print(f"Saved {label} checkpoint (EER: {eer:.2f}%) → {model_save_dir}")

    # _pad_right replaced by imported pad_right from generation module.

    # ===== PRECOMPUTE LABEL TOKENS ONCE =====
    # label_texts_map and max_answer_tokens are constant for the entire run.
    # Tokenising the same strings inside the per-sample loop caused ~400K
    # redundant tokenizer calls per training run (bs×2 per batch).
    # Fail loudly if the YAML did not define both label entries — hardcoded
    # fallbacks ("bonafide"/"spoof") would silently diverge from the config.
    _label0_text = label_texts_map.get(0) or label_texts_map.get(int(0))
    _label1_text = label_texts_map.get(1) or label_texts_map.get(int(1))
    if not _label0_text or not _label1_text:
        raise ValueError(
            f"generation_classification.label_texts must define entries for keys 0 and 1. "
            f"Got: {dict(label_texts_map)}"
        )
    _tok = processor.tokenizer
    _label0_ids = _tok(_label0_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    _label1_ids = _tok(_label1_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if max_answer_tokens is not None and max_answer_tokens > 0:
        _label0_ids = _label0_ids[:max_answer_tokens]
        _label1_ids = _label1_ids[:max_answer_tokens]

    _label0_ids = _label0_ids.to(DEVICE)
    _label1_ids = _label1_ids.to(DEVICE)

    # Invariant #8: without per-token normalization, the shorter label wins by raw
    # log-prob sum unconditionally, silently biasing every score and destroying EER.
    if not normalize_logprob_by_tokens and _label0_ids.shape[0] != _label1_ids.shape[0]:
        raise ValueError(
            f"normalize_logprob_by_tokens=False with labels of different token lengths is "
            f"mathematically unsound: '{_label0_text}' tokenizes to {_label0_ids.shape[0]} "
            f"token(s) but '{_label1_text}' tokenizes to {_label1_ids.shape[0]} token(s). "
            f"The shorter label accumulates a higher total log-prob on every sample regardless "
            f"of content, silently biasing EER. "
            f"Fix: set normalize_logprob_by_tokens: true in your YAML (CLAUDE.md invariant #8)."
        )

    # Validate margin loss + token normalization compatibility.
    # NOTE: the general check above already raises for the length-mismatch case, so
    # the inner raise here can only be reached when normalize_logprob_by_tokens=False
    # AND both labels happen to tokenize to the same length — in which case the inner
    # condition is False and nothing is raised.  Kept for its specific error message.
    if use_margin_loss and not normalize_logprob_by_tokens:
        if _label0_ids.shape[0] != _label1_ids.shape[0]:
            raise ValueError(
                f"use_margin_loss=True with normalize_logprob_by_tokens=False is "
                f"mathematically unsound: '{_label0_text}' tokenizes to "
                f"{_label0_ids.shape[0]} token(s) but '{_label1_text}' tokenizes "
                f"to {_label1_ids.shape[0]} token(s). The margin threshold is "
                f"compared against total (not per-token) log-probabilities, causing "
                f"systematic length bias toward the shorter label. "
                f"Fix: set normalize_logprob_by_tokens: true in your YAML, or make "
                f"both labels tokenize to the same length."
            )

    # ===== MAIN TRAINING LOOP =====
    # Outer loop: epochs, Inner loop: batches
    if run_by_epochs:
        max_epochs = NUM_EPOCHS
    else:
        remaining_steps = max(0, max_train_steps - optimizer_step)
        max_epochs = start_epoch + max(1, math.ceil(remaining_steps / max(1, steps_per_epoch)))

    # Build the AMP autocast context once before the training loop.
    _amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if use_amp or use_fp16:
        _amp_ctx = torch.autocast(device_type=_amp_device_type, dtype=torch.float16)
    elif use_bf16:
        _amp_ctx = torch.autocast(device_type=_amp_device_type, dtype=torch.bfloat16)
    else:
        _amp_ctx = nullcontext()

    # Move class_weights to device once before the training loop.
    if use_weighted_loss and class_weights is not None:
        class_weights = class_weights.to(DEVICE)

    for epoch in range(start_epoch, max_epochs):
        # ===== EPOCH INITIALIZATION =====
        # Reset metrics and prepare for new epoch
        if enable_debug_logging and torch.cuda.is_available():
            torch.cuda.empty_cache()             # clear unused blocks
            torch.cuda.reset_peak_memory_stats() # reset counters

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_cag_lm_loss = 0.0
        running_tf_loss = 0.0
        running_margin_loss = 0.0
        step_in_epoch = 0
        valid_steps = 0 
        

        epoch_desc = f"Epoch {epoch+1}/{max_epochs}" if run_by_epochs else f"Epoch {epoch+1} (step-mode)"
        loop = tqdm(train_loader, desc=epoch_desc, leave=True)
        for step, batch in enumerate(loop):
            # Determine if this is the last DataLoader step before checking for empty batches,
            # so accumulated gradients are always flushed even if the final batch is empty.
            _is_last_loader_step = (step + 1 == len(train_loader))

            batch = {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

            # Validate batch is non-empty to prevent crashes on malformed data.
            if "input_ids" not in batch or batch["input_ids"].size(0) == 0:
                dprint(f"[WARNING] Skipping empty batch at step {step}")
                if _is_last_loader_step and step_in_epoch > 0 and (step_in_epoch % accum) != 0:
                    # Last step is empty but gradients were accumulated — flush them now
                    # with partial-window rescaling so they don't bleed into the next epoch.
                    _flush_use_scaler = (use_amp or use_fp16) and scaler is not None and scaler.is_enabled()
                    if _flush_use_scaler:
                        scaler.unscale_(optimizer)
                    if max_grad_norm is not None and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if _flush_use_scaler:
                        _flush_prev_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        _flush_opt_ran = scaler.get_scale() >= _flush_prev_scale
                    else:
                        optimizer.step()
                        _flush_opt_ran = True
                    optimizer.zero_grad(set_to_none=True)
                    if _flush_opt_ran:
                        if scheduler is not None:
                            scheduler.step()
                        optimizer_step += 1
                    dprint(f"[WARNING] Flushed accumulated gradients on empty last batch (step_in_epoch={step_in_epoch})")
                continue
            valid_steps += 1  # count only batches that reach the loss computation

            # ===== GRAD-ACCUM WINDOW SIZE =====
            # Each micro-batch loss must be divided by the number of micro-batches
            # contributing to its optimizer step: `accum` for full windows, fewer for
            # the final (partial) window of the epoch. Dividing everything by `accum`
            # and rescaling only the last micro-batch afterwards (previous approach)
            # weighted the partial window's members inconsistently — the last
            # micro-batch dominated the final optimizer step of every epoch.
            _pos_in_window = step_in_epoch % accum               # 0-based position in window
            _remaining_loader_steps = len(train_loader) - step   # including current step
            if _pos_in_window + _remaining_loader_steps <= accum:
                # Final window: only the remaining loader steps contribute. If a
                # remaining batch is skipped as empty this slightly overestimates the
                # divisor (conservative; never crashes).
                _window_size = max(1, _pos_in_window + _remaining_loader_steps)
            else:
                _window_size = accum

            # ===== CPU PREP: build candidate sequences (tokenization, no AMP benefit) =====
            # build_cag_candidate and pad_sequence are CPU-only operations.
            # Running them outside the AMP autocast context avoids unnecessary
            # AMP bookkeeping overhead and makes the GPU section of the step cleaner.
            device = batch["input_ids"].device
            bs = batch["input_ids"].size(0)
            label0_token_ids = _label0_ids
            label1_token_ids = _label1_ids
            pad_id = processor.tokenizer.pad_token_id

            tf_loss = None
            tf_loss_value = 0.0
            margin_loss = None
            margin_loss_value = 0.0

            if debug_mode:
                # Lighter path in debug: score only the true label to cut memory
                cand_t_list, attn_t_list, mask_t_list = [], [], []
                for i in range(bs):
                    base_ids = batch["input_ids"][i]
                    base_mask = batch["attention_mask"][i]
                    lbl_idx = int(batch["classification_labels"][i].item())
                    lbl_precomp = label0_token_ids if lbl_idx == 0 else label1_token_ids
                    cand_t_i, attn_t_i, mask_t_i = build_cag_candidate(
                        processor, "", label_texts_map[lbl_idx],
                        base_ids, base_mask,
                        max_answer_tokens=max_answer_tokens,
                        precomputed_label_tokens=lbl_precomp,
                    )
                    cand_t_list.append(cand_t_i)
                    attn_t_list.append(attn_t_i)
                    mask_t_list.append(mask_t_i)
                cand_t = pad_sequence(cand_t_list, batch_first=True, padding_value=pad_id)
                attn_t = pad_sequence(attn_t_list, batch_first=True, padding_value=0)
                mask_t = pad_sequence(mask_t_list, batch_first=True, padding_value=False)
                pos_t = torch.arange(cand_t.size(1), device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
                labels_t = make_lm_labels(cand_t, mask_t)
            else:
                # Build CAG candidate sequences for the full batch in one vectorized call.
                cand_b, attn_b, mask_b = build_cag_candidates_batched(
                    batch["input_ids"], batch["attention_mask"], label0_token_ids, pad_id
                )
                labels_b = make_lm_labels(cand_b, mask_b)

                cand_s, attn_s, mask_s = build_cag_candidates_batched(
                    batch["input_ids"], batch["attention_mask"], label1_token_ids, pad_id
                )
                labels_s = make_lm_labels(cand_s, mask_s)

                # Pad both to L_max and concatenate into a single 2B-sample batch so
                # the LLM runs once — halves kernel launches and lets gradient
                # checkpointing amortise recomputation across both candidates.
                L_b, L_s = cand_b.size(1), cand_s.size(1)
                L_max = max(L_b, L_s)

                # Delete individual tensors immediately after they are folded into
                # the combined batch; avoids holding ~3× peak memory at this point.
                cand_all = torch.cat([pad_right(cand_b, L_max, pad_id), pad_right(cand_s, L_max, pad_id)], dim=0)
                del cand_b, cand_s
                attn_all = torch.cat([pad_right(attn_b, L_max, 0), pad_right(attn_s, L_max, 0)], dim=0)
                del attn_b, attn_s
                mask_all = torch.cat([pad_right(mask_b.long(), L_max, 0).bool(),
                                      pad_right(mask_s.long(), L_max, 0).bool()], dim=0)
                del mask_b, mask_s
                labels_all = torch.cat([pad_right(labels_b, L_max, -100), pad_right(labels_s, L_max, -100)], dim=0)
                del labels_b, labels_s
                pos_all = torch.arange(L_max, device=device, dtype=torch.long).unsqueeze(0).expand(2 * bs, -1)

            with _amp_ctx:
                # ===== GPU: audio encoding (shared across both label candidates) =====
                _model_unwrapped = model.module if hasattr(model, "module") else model
                audio_embeds_cache = _model_unwrapped.compute_audio_embeds(
                    input_features=batch.get("input_features"),
                    input_ids=batch["input_ids"],
                    raw_audio_input=batch.get("raw_audio_input"),
                    raw_audio_lengths=batch.get("raw_audio_lengths"),
                )

                if debug_mode:
                    fwd_kwargs_dbg = dict(precomputed_audio_embeds=audio_embeds_cache)
                    logp_t = score_label_logprob(
                        model, cand_t, attn_t, mask_t,
                        labels=labels_t,
                        temperature=temperature,
                        normalize_by_length=normalize_logprob_by_tokens,
                        position_ids=pos_t,
                        **fwd_kwargs_dbg,
                    )
                    del audio_embeds_cache, fwd_kwargs_dbg
                    # Debug path: unconditional NLL of the true label — logged separately
                    # as "debug_lm_nll" since it is not comparable to the production CE loss.
                    _lm_nll = -logp_t.mean()
                    loss_cls_value = float(_lm_nll.detach().item())
                    dprint(f"[DEBUG] lm_nll (not comparable to prod CE): {loss_cls_value:.4f}")
                    total_loss = _lm_nll * cag_loss_weight
                    total_loss = total_loss / _window_size
                    total_loss_value = float(total_loss.detach().item() * _window_size)
                else:
                    # Double audio embeds along batch dim to match the 2B combined input.
                    fwd_kwargs = {}
                    if audio_embeds_cache is not None:
                        fwd_kwargs["precomputed_audio_embeds"] = torch.cat(
                            [audio_embeds_cache, audio_embeds_cache], dim=0
                        )
                    del audio_embeds_cache

                    # Single LLM forward pass for both label candidates.
                    # return_lm_loss=True returns only the per-sample scalar NLL (shape B),
                    # never the full logits tensor, so no large output object is kept alive.
                    _ret_all = score_label_logprob(
                        model, cand_all, attn_all, mask_all,
                        labels=labels_all,
                        temperature=temperature,
                        normalize_by_length=normalize_logprob_by_tokens,
                        position_ids=pos_all,
                        return_lm_loss=use_teacher_forcing_loss,
                        **fwd_kwargs,
                    )
                    del fwd_kwargs

                    if use_teacher_forcing_loss:
                        logp_all, lm_loss_all = _ret_all
                        # Split by label index (0/1) rather than semantics — label order is YAML-configurable.
                        lm_loss_0 = lm_loss_all[:bs]
                        lm_loss_1 = lm_loss_all[bs:]
                        del lm_loss_all
                    else:
                        logp_all = _ret_all
                        lm_loss_0 = lm_loss_1 = None
                    # Drop the combined-batch references now that we have the splits.
                    del _ret_all

                    logp_b = logp_all[:bs]
                    logp_s = logp_all[bs:]
                    del logp_all

                    # Normalization is handled inside score_label_logprob; no post-hoc division here.
                    logpt = torch.stack([logp_b, logp_s], dim=1)

                    # Apply class weights if configured (already moved to DEVICE before the loop).
                    weight = class_weights if (use_weighted_loss and class_weights is not None) else None
                    # Renamed `cls_labels` (not `labels`) to avoid shadowing the LM label
                    # tensors (labels_all, labels_b, labels_s) in the same scope.
                    cls_labels = batch["classification_labels"].long()

                    # ===== LOSS COMPUTATION =====
                    loss_cls = F.cross_entropy(logpt, cls_labels, weight=weight)

                    # Per-sample weights for auxiliary losses (margin, teacher forcing)
                    sample_weights = None
                    if use_weighted_aux_losses and use_weighted_loss and weight is not None:
                        sample_weights = weight[cls_labels] / weight.mean()

                    ce_loss = loss_cls * cag_loss_weight
                    total_loss = ce_loss

                    if use_margin_loss:
                        # Margin loss uses `1 - cls_labels` for the wrong-class index — only valid for binary (N=2).
                        N_labels = logpt.size(1)
                        if N_labels != 2:
                            raise NotImplementedError(
                                f"use_margin_loss with `1 - cls_labels` wrong-class indexing "
                                f"only supports binary classification (N=2). Got N={N_labels}. "
                                f"Extend to multi-class before enabling margin loss."
                            )
                        # Use YAML-derived semantic keys so margin loss is correct regardless of label order in the YAML.
                        #
                        # diff = score(true label) - score(wrong label) for every sample.
                        # For a correct prediction, diff > 0.  We penalise when diff < margin.
                        logp_correct = logpt[torch.arange(bs, device=logpt.device), cls_labels]
                        # 1 - cls_labels flips 0<->1 to index the other (wrong) class column.
                        logp_wrong   = logpt[torch.arange(bs, device=logpt.device), 1 - cls_labels]
                        diff = logp_correct - logp_wrong

                        # Assign the per-sample margin based on the true semantic class.
                        margin_bonafide_tensor = logpt.new_tensor(margin_bonafide)
                        margin_spoof_tensor    = logpt.new_tensor(margin_spoof)
                        is_spoof = cls_labels == _spoof_key
                        margin_tensor = torch.where(is_spoof, margin_spoof_tensor, margin_bonafide_tensor)

                        margin_loss_per_sample = F.relu(margin_tensor - diff)
                        if sample_weights is not None:
                            margin_loss_per_sample = margin_loss_per_sample * sample_weights
                        margin_loss = margin_loss_per_sample.mean()
                        total_loss = total_loss + logpt.new_tensor(lambda_margin) * margin_loss
                        margin_loss_value = float(margin_loss.detach().item())

                    if use_teacher_forcing_loss:
                        if lm_loss_0 is not None and lm_loss_1 is not None:
                            loss_0_per = lm_loss_0.to(loss_cls.device, loss_cls.dtype)
                            loss_1_per = lm_loss_1.to(loss_cls.device, loss_cls.dtype)
                            if loss_0_per.dim() == 0:
                                loss_0_per = loss_0_per.expand(bs)
                            if loss_1_per.dim() == 0:
                                loss_1_per = loss_1_per.expand(bs)
                            # Select teacher-forcing NLL for the true label using its index.
                            tf_loss_per_sample = torch.where(cls_labels == 0, loss_0_per, loss_1_per)
                            if sample_weights is not None:
                                tf_loss_per_sample = tf_loss_per_sample * sample_weights
                            tf_loss = tf_loss_per_sample.mean()
                            total_loss = total_loss + loss_cls.new_tensor(teacher_forcing_lambda) * tf_loss
                            tf_loss_value = float(tf_loss.detach().item())

                    total_loss = total_loss / _window_size
                    loss_cls_value = float(loss_cls.detach().item())
                    if tf_loss is None:
                        tf_loss_value = 0.0
                    total_loss_value = float(total_loss.detach().item() * _window_size)




            # ===== BACKWARD PASS =====
            # total_loss is already divided by the true window size (_window_size),
            # so no partial-window rescaling is needed here.

            # NaN guard — skip backward on non-finite loss to protect model parameters.
            if not torch.isfinite(total_loss):
                import warnings
                warnings.warn(
                    f"Non-finite loss detected ({total_loss.item()}); "
                    "skipping backward and clearing grads."
                )
                optimizer.zero_grad(set_to_none=True)
                if (use_amp or use_fp16) and scaler is not None and scaler.is_enabled():
                    scaler.update()
                continue

            # Use gradient scaling for mixed precision training
            use_scaler = (use_amp or use_fp16) and scaler.is_enabled()
            if use_scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            step_in_epoch += 1

            # ===== GRADIENT ACCUMULATION BOUNDARY =====
            # Only update weights every 'accum' steps or at end of epoch.
            # use _is_last_loader_step (hoisted before empty-batch skip) instead of
            # raw (step + 1 == len(train_loader)) so the boundary is correct even when
            # earlier steps were skipped (step counter diverges from step_in_epoch).
            boundary = ((step_in_epoch % accum) == 0) or _is_last_loader_step
            if boundary:
                # Unscale gradients before clipping
                if use_scaler:
                    scaler.unscale_(optimizer)

                # Gradient norm (global) for clipping/logging
                grad_norm_value = None
                should_clip = max_grad_norm is not None and max_grad_norm > 0
                should_log_grad_norm = save_lr_and_grad_norm_debug and lr_grad_log_path is not None
                if should_clip or should_log_grad_norm:
                    clip_val = max_grad_norm if should_clip else float("inf")
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    grad_norm_value = float(grad_norm_tensor.detach().item())

                # Optimizer step with gradient scaling
                if use_scaler:
                    _prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    _opt_ran = scaler.get_scale() >= _prev_scale
                else:
                    optimizer.step()
                    _opt_ran = True

                # Reset gradients and update learning rate (only when optimizer actually ran)
                optimizer.zero_grad(set_to_none=True)
                if _opt_ran:
                    if scheduler is not None:
                        scheduler.step()
                    optimizer_step += 1
                
                # Optional ad-hoc initial eval steps (e.g., step 0/handful of steps)
                if optimizer_step in initial_eval_steps_set and optimizer_step not in initial_eval_done:
                    _run_initial_dev_eval(optimizer_step)
                    initial_eval_done.add(optimizer_step)
                if optimizer_step in initial_eval_steps_eval_set and optimizer_step not in initial_eval_eval_done:
                    _run_initial_eval_eval(optimizer_step)
                    initial_eval_eval_done.add(optimizer_step)
                
                # Optional: Log learning rate and gradient norm.
                # Log max LR across all param groups so no group is silently ignored.
                if save_lr_and_grad_norm_debug and lr_grad_log_path is not None:
                    _all_lrs = [g.get("lr", float("nan")) for g in optimizer.param_groups]
                    current_lr_debug = max(_all_lrs) if _all_lrs else float("nan")
                    with open(lr_grad_log_path, "a", encoding="utf-8") as f:
                        grad_norm_to_log = grad_norm_value if grad_norm_value is not None else float("nan")
                        f.write(f"{optimizer_step}\t{current_lr_debug}\t{grad_norm_to_log}\n")

                # ===== STEP-BASED PERIODIC CHECKPOINT SAVE =====
                # Step-based periodic save
                if (not run_by_epochs) and save_every_steps and save_every_steps > 0 and (optimizer_step % save_every_steps == 0):
                    # Mid-epoch saves store the raw loop variable with epoch_completed=False
                    # so the resume logic knows this epoch is not done and replays it.
                    ckpt = _build_checkpoint(epoch, optimizer_step, epoch_completed=False)
                    if save_best_only:
                        checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint-last.pt")
                        save_checkpoint(checkpoint_path, ckpt)
                        print(f"[Periodic Save] Saved rolling checkpoint to {checkpoint_path} at step {optimizer_step}")
                    else:
                        checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint-step{optimizer_step}.pt")
                        save_checkpoint(checkpoint_path, ckpt)
                        print(f"[Periodic Save] Saved checkpoint to {OUTPUT_DIR} at step {optimizer_step}")

                # Step-based evaluation
                if (not run_by_epochs) and eval_every_steps and eval_every_steps > 0 and evaluate_epoch is not None and (optimizer_step % eval_every_steps == 0):
                    # Print start marker so you see that evaluation actually began (and flush immediately)
                    print(f"[Eval start] step {optimizer_step} on dev", flush=True)
                    metrics, dev_err = _safe_eval_split(
                        f"Dev step {optimizer_step}",
                        evaluate_epoch,
                        model, dev_loader, processor, DEVICE,
                        use_amp=eval_use_amp, amp_dtype=eval_amp_dtype,
                        debug_log_path=debug_log_path, eval_step=optimizer_step,
                        enable_debug_logging=enable_debug_logging,
                        generation_cfg=generation_cls_cfg,
                        return_example=log_eval_examples,
                    )
                    # Restore training mode after eval — evaluate_epoch leaves the model in eval().
                    model.train()
                    if not dev_err and metrics is not None:
                        metrics_clean, example_entry = split_metrics_example(metrics)
                        dev_eval_log_payload = {f"val/{k}": v for k, v in metrics_clean.items()}
                        print(f"Eval step {optimizer_step}: {dev_eval_log_payload}", flush=True)
                        if example_entry:
                            print(f"[Eval example] {example_entry}", flush=True)

                        if log_metrics is not None:
                            log_metrics(log_file=metrics_log_file, step=optimizer_step, metrics=metrics_clean, examples=example_entry)
                        
                        log_wandb(dev_eval_log_payload, step=optimizer_step)
                        last_eval_step = max(last_eval_step, optimizer_step)
                        
                        
                        eer = metrics_clean.get("eer")

                        if eer is not None and math.isfinite(eer) and eer < best_eer:
                            best_eer = eer
                            if wandb_run:
                                wandb_run.summary["best_eer"] = best_eer
                            _save_best_checkpoint(
                                model_save_dir=OUTPUT_DIR,
                                adapters_dir=os.path.join(OUTPUT_DIR, "adapters"),
                                ckpt_path=os.path.join(OUTPUT_DIR, f"checkpoint-step{optimizer_step}.pt"),
                                metrics_json_path=os.path.join(OUTPUT_DIR, "best_metrics.json"),
                                eer=eer, metrics=metrics_clean, label="best dev (step)",
                                epoch_num=epoch, step_num=optimizer_step,
                            )

                    # ===== OPTIONAL: TRAINING SET EVALUATION (STEP-BASED) =====
                    # Evaluate on training subset if configured
                    if configs is not None and getattr(configs, "eval_on_training_set_during_training", False) and train_eval_loader is not None:
                        train_eval_metrics, train_eval_err = _safe_eval_split(
                            f"Train eval step {optimizer_step}",
                            evaluate_epoch,
                            model,
                            train_eval_loader,
                            processor,
                            DEVICE,
                            use_amp=eval_use_amp,
                            amp_dtype=eval_amp_dtype,
                            debug_log_path=eval_debug_log_path,
                            enable_debug_logging=enable_debug_logging,
                            generation_cfg=generation_cls_cfg,

                            return_example=log_eval_examples,
                        )
                        # Restore training mode after train-set eval.
                        model.train()
                        if not train_eval_err and train_eval_metrics is not None:
                            train_eval_metrics_clean, train_eval_example = split_metrics_example(train_eval_metrics)
                            train_eval_log_payload = {f"train_eval/{k}": v for k, v in train_eval_metrics_clean.items()}
                            print(f"[Eval log] Step {optimizer_step}: {train_eval_log_payload}")
                            if train_eval_example:
                                print(f"[Eval example - train] {train_eval_example}")
                            if log_metrics is not None and train_eval_log_file is not None:
                                log_metrics(log_file=train_eval_log_file, step=optimizer_step, metrics=train_eval_metrics_clean, examples=train_eval_example)
                            log_wandb(train_eval_log_payload, step=optimizer_step)
                            last_train_eval_step = max(last_train_eval_step, optimizer_step)

                    # ===== OPTIONAL: TEST SET EVALUATION (STEP-BASED) =====
                    # Evaluate on test set if configured
                    if configs is not None and getattr(configs, "eval_on_test_during_training", False) and eval_loader is not None:
                        eval_metrics, eval_err = _safe_eval_split(
                            f"Eval step {optimizer_step}",
                            evaluate_epoch,
                            model,
                            eval_loader,
                            processor,
                            DEVICE,
                            use_amp=eval_use_amp,
                            amp_dtype=eval_amp_dtype,
                            debug_log_path=eval_debug_log_path,
                            enable_debug_logging=enable_debug_logging,
                            generation_cfg=generation_cls_cfg,

                            return_example=log_eval_examples,
                            return_eval_stats=log_eval_stats_enabled,
                        )
                        # restore training mode after test-set eval.
                        model.train()
                        if not eval_err and eval_metrics is not None:
                            eval_stats = None
                            if log_eval_stats_enabled and isinstance(eval_metrics, dict):
                                eval_stats = eval_metrics.pop("eval_stats", None)
                            eval_metrics_clean, eval_example = split_metrics_example(eval_metrics)
                            test_eval_log_payload = {f"eval/{k}": v for k, v in eval_metrics_clean.items()}
                            print(f"[Eval log] Step {optimizer_step}: {test_eval_log_payload}")
                            if eval_example:
                                print(f"[Eval example - test] {eval_example}")
                            if log_metrics is not None:
                                log_metrics(log_file=eval_log_file, step=optimizer_step, metrics=eval_metrics_clean, examples=eval_example)
                            log_wandb(test_eval_log_payload, step=optimizer_step)
                            last_test_eval_step = max(last_test_eval_step, optimizer_step)
                            # Save best-eval checkpoint in step-based mode (mirrors epoch-based logic).
                            _step_eval_eer = eval_metrics_clean.get("eer")
                            if _step_eval_eer is not None and math.isfinite(_step_eval_eer) and _step_eval_eer < best_eval_eer:
                                best_eval_eer = _step_eval_eer
                                if wandb_run:
                                    wandb_run.summary["best_eval_eer"] = best_eval_eer
                                _best_eval_dir = os.path.join(OUTPUT_DIR, "best_eval")
                                _save_best_checkpoint(
                                    model_save_dir=_best_eval_dir,
                                    adapters_dir=os.path.join(OUTPUT_DIR, "adapters_best_eval"),
                                    ckpt_path=os.path.join(_best_eval_dir, "checkpoint_best_eval.pt"),
                                    metrics_json_path=os.path.join(OUTPUT_DIR, "best_eval_metrics.json"),
                                    eer=_step_eval_eer, metrics=eval_metrics_clean, label="best eval (step)",
                                    epoch_num=epoch, step_num=optimizer_step,
                                )
                            if eval_stats:
                                maybe_log_eval_stats(
                                    enabled=log_eval_stats_enabled,
                                    log_file=eval_log_file,
                                    header=f"Step {optimizer_step} (eval set)",
                                    eval_stats=eval_stats,
                                )

                    # ===== OPTIONAL: ASVSPOOF2021 EVALUATION (STEP-BASED) =====
                    # Evaluate on ASVspoof2021 dataset if configured
                    if configs is not None and getattr(configs, "eval_on_asvspoof2021_during_training", False) and asvspoof2021_eval_loader is not None:
                        print_asv2021_info(f"step {optimizer_step}")
                        asv2021_metrics, asv2021_err = _safe_eval_split(
                            f"ASVspoof2021 step {optimizer_step}",
                            evaluate_epoch,
                            model,
                            asvspoof2021_eval_loader,
                            processor,
                            DEVICE,
                            use_amp=eval_use_amp,
                            amp_dtype=eval_amp_dtype,
                            debug_log_path=eval_debug_log_path,
                            enable_debug_logging=enable_debug_logging,
                            generation_cfg=generation_cls_cfg,

                            return_example=log_eval_examples,
                        )
                        # restore training mode after ASVspoof2021 eval.
                        model.train()
                        if not asv2021_err and asv2021_metrics is not None:
                            asv2021_metrics_clean, asv2021_example = split_metrics_example(asv2021_metrics)
                            asv2021_log_payload = {f"asvspoof2021/{k}": v for k, v in asv2021_metrics_clean.items()}
                            print(f"[Eval log] Step {optimizer_step}: {asv2021_log_payload}")
                            if asv2021_example:
                                print(f"[Eval example - asvspoof2021] {asv2021_example}")
                            if log_metrics is not None and asvspoof2021_eval_log_file is not None:
                                log_metrics(log_file=asvspoof2021_eval_log_file, step=optimizer_step, metrics=asv2021_metrics_clean, examples=asv2021_example)
                            log_wandb(asv2021_log_payload, step=optimizer_step)
                            last_asvspoof2021_eval_step = max(last_asvspoof2021_eval_step, optimizer_step)

                if (not run_by_epochs) and max_train_steps and optimizer_step >= max_train_steps:
                    stop_training = True
                    break
                

            # ===== TRAINING METRICS LOGGING =====
            # Update running losses and compute averages
            running_loss += total_loss_value
            running_cag_lm_loss += loss_cls_value
            if use_teacher_forcing_loss:
                running_tf_loss += tf_loss_value
            if use_margin_loss:
                running_margin_loss += margin_loss_value
            # Average over valid_steps only — excludes skipped empty batches.
            _denom = max(1, valid_steps)
            avg_loss        = running_loss        / _denom
            avg_cls_loss    = running_cag_lm_loss / _denom
            avg_tf_loss     = (running_tf_loss     / _denom) if use_teacher_forcing_loss else 0.0
            avg_margin_loss = (running_margin_loss / _denom) if use_margin_loss          else 0.0
            if use_teacher_forcing_loss and use_margin_loss:
                loop.set_postfix(loss=f"{avg_loss:.4f}", cag_lm=f"{avg_cls_loss:.4f}", margin=f"{avg_margin_loss:.4f}", tf=f"{avg_tf_loss:.4f}")
            elif use_teacher_forcing_loss:
                loop.set_postfix(loss=f"{avg_loss:.4f}", cag_lm=f"{avg_cls_loss:.4f}", tf=f"{avg_tf_loss:.4f}")
            elif use_margin_loss:
                loop.set_postfix(loss=f"{avg_loss:.4f}", cag_lm=f"{avg_cls_loss:.4f}", margin=f"{avg_margin_loss:.4f}")
            else:
                loop.set_postfix(loss=f"{avg_loss:.4f}", cag_lm=f"{avg_cls_loss:.4f}")

            micro_step = epoch * len(train_loader) + step
            # Collect LR from every param group so no group is silently ignored in W&B.
            if scheduler is not None:
                _all_lrs = scheduler.get_last_lr()
            else:
                _all_lrs = [g.get("lr", float("nan")) for g in optimizer.param_groups]
            current_lr = max(_all_lrs) if _all_lrs else float("nan")
            logged_step = optimizer_step + wandb_step_offset
            wandb_payload = {
                "train/loss": total_loss_value,
                "train/avg_loss": avg_loss,
                "train/lr": current_lr,
                "epoch": epoch + 1,
                "step": logged_step,
                "train/optimizer_step": optimizer_step,
                "train/micro_step": micro_step,
            }
            # Log per-group LRs when multiple param groups are active.
            if len(_all_lrs) > 1:
                for _gi, _glr in enumerate(_all_lrs):
                    wandb_payload[f"train/lr_group{_gi}"] = _glr
            wandb_payload["train/cag_lm_loss"] = loss_cls_value
            wandb_payload["train/cag_avg_lm_loss"] = avg_cls_loss
            if use_margin_loss:
                wandb_payload["train/margin_loss"] = margin_loss_value
                wandb_payload["train/margin_avg_loss"] = avg_margin_loss
            if use_teacher_forcing_loss:
                wandb_payload["train/teacher_forcing_lm_loss"] = tf_loss_value
                wandb_payload["train/teacher_forcing_avg_loss"] = avg_tf_loss
            log_wandb(wandb_payload, step=optimizer_step)


            if (step + 1) % LOG_EVERY == 0:
                with open(result_file, "a") as f:
                    margin_txt = f" L_margin {avg_margin_loss:.4f}" if use_margin_loss else ""
                    tf_txt = f" L_tf {avg_tf_loss:.4f}" if use_teacher_forcing_loss else ""
                    if run_by_epochs:
                        f.write(
                            f"Epoch {epoch+1} Step {step+1} Loss {avg_loss:.4f} "
                            f"L_cag {avg_cls_loss:.4f}{margin_txt}{tf_txt}\n"
                        )
                    else:
                        f.write(
                            f"Step {optimizer_step} (epoch {epoch+1} batch {step+1}) Loss {avg_loss:.4f} "
                            f"L_cag {avg_cls_loss:.4f}{margin_txt}{tf_txt}\n"
                        )
                    
            if debug_mode:
                break

        if stop_training:
            break

        if debug_mode and evaluate_epoch is not None:
            metrics = evaluate_epoch(
                model,
                dev_loader,
                processor,
                DEVICE,
                use_amp=eval_use_amp,
                amp_dtype=eval_amp_dtype,
                debug_log_path=debug_log_path,
                enable_debug_logging=enable_debug_logging,
                generation_cfg=generation_cls_cfg,


                return_example=log_eval_examples,
            )
            print(f"[DEBUG] Eval after single training step: {metrics}")
            break

        # ===== EPOCH-BASED PERIODIC CHECKPOINT SAVE =====
        # Save checkpoint at regular epoch intervals.
        # Track the path so we can re-save AFTER the eval blocks below, persisting
        # the freshly-updated last_*_eval_epoch counters into the same file (otherwise
        # resume would re-evaluate this epoch because the saved counters are stale).
        _periodic_save_path = None
        if run_by_epochs and SAVE_EVERY is not None and SAVE_EVERY > 0 and ((epoch + 1) % SAVE_EVERY == 0):
            ckpt = _build_checkpoint(epoch, optimizer_step, epoch_completed=True)
            if save_best_only:
                _periodic_save_path = os.path.join(OUTPUT_DIR, "checkpoint-last.pt")
                save_checkpoint(_periodic_save_path, ckpt)
                print(f"[Periodic Save] Saved rolling checkpoint to {_periodic_save_path} at epoch {epoch+1}")
            else:
                _periodic_save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch+1}.pt")
                save_checkpoint(_periodic_save_path, ckpt)
                print(f"[Periodic Save] Saved checkpoint to {OUTPUT_DIR} at epoch {epoch+1}")

        # ===== DEVELOPMENT SET EVALUATION =====
        # Evaluate on dev set at regular intervals
        if run_by_epochs and (epoch + 1) % EVAL_EVERY == 0 and evaluate_epoch is not None:
            metrics, dev_err = _safe_eval_split(
                f"Dev epoch {epoch+1}",
                evaluate_epoch,
                model, dev_loader, processor, DEVICE,
                use_amp=eval_use_amp, amp_dtype=eval_amp_dtype,
                debug_log_path=debug_log_path, enable_debug_logging=enable_debug_logging,
                generation_cfg=generation_cls_cfg,
                return_example=log_eval_examples,
            )
            # restore training mode after epoch-level dev eval.
            model.train()
            if not dev_err and metrics is not None:
                metrics_clean, example_entry = split_metrics_example(metrics)
                dev_eval_log_payload = {f"val/{k}": v for k, v in metrics_clean.items()}
                print(f"Eval epoch {epoch+1}: {dev_eval_log_payload}")
                if example_entry:
                    print(f"[Eval example] {example_entry}")

                if log_metrics is not None:
                    log_metrics(log_file=metrics_log_file, epoch=epoch, metrics=metrics_clean, examples=example_entry)
                
                log_wandb(dev_eval_log_payload, step=optimizer_step)
                last_eval_epoch = max(last_eval_epoch, epoch + 1)

                # ===== SAVE BEST DEV MODEL CHECKPOINT =====
                # Save checkpoint when dev EER improves
                eer = metrics_clean.get("eer")

                if eer is not None and math.isfinite(eer) and eer < best_eer:
                    best_eer = eer
                    if wandb_run:
                        wandb_run.summary["best_eer"] = best_eer
                        wandb_run.summary["best_dev_eer"] = best_eer
                    _best_dev_dir = os.path.join(OUTPUT_DIR, "best_dev")
                    _save_best_checkpoint(
                        model_save_dir=_best_dev_dir,
                        adapters_dir=os.path.join(OUTPUT_DIR, "adapters_best_dev"),
                        ckpt_path=os.path.join(_best_dev_dir, "checkpoint_best_dev.pt"),
                        metrics_json_path=os.path.join(OUTPUT_DIR, "best_dev_metrics.json"),
                        eer=eer, metrics=metrics_clean, label="best dev (epoch)",
                        epoch_num=epoch, step_num=optimizer_step,
                        extra_ckpt={"best_dev_eer": float(best_eer)},
                    )

            # ===== OPTIONAL: TRAINING SET EVALUATION =====
            # Evaluate on training subset if configured
            if configs is not None and getattr(configs, "eval_on_training_set_during_training", False) and train_eval_loader is not None:
                train_eval_metrics, train_eval_err = _safe_eval_split(
                    f"Train eval epoch {epoch+1}",
                    evaluate_epoch,
                    model,
                    train_eval_loader,
                    processor,
                    DEVICE,
                    use_amp=eval_use_amp,
                    amp_dtype=eval_amp_dtype,
                    debug_log_path=eval_debug_log_path,
                    enable_debug_logging=enable_debug_logging,
                    generation_cfg=generation_cls_cfg,

                    return_example=log_eval_examples,
                )
                # restore training mode after train-eval.
                model.train()
                if not train_eval_err and train_eval_metrics is not None:
                    train_eval_metrics_clean, train_eval_example = split_metrics_example(train_eval_metrics)
                    train_eval_log_payload = {f"train_eval/{k}": v for k, v in train_eval_metrics_clean.items()}
                    print(f"[Eval log] Epoch {epoch+1}: {train_eval_log_payload}")
                    if train_eval_example:
                        print(f"[Eval example - train] {train_eval_example}")
                    if log_metrics is not None and train_eval_log_file is not None:
                        log_metrics(log_file=train_eval_log_file, epoch=epoch, metrics=train_eval_metrics_clean, examples=train_eval_example)
                    log_wandb(train_eval_log_payload, step=optimizer_step)
                    last_train_eval_epoch = max(last_train_eval_epoch, epoch + 1)

            # ===== OPTIONAL: TEST SET EVALUATION =====
            # Evaluate on test set if configured
            if evaluate_epoch is not None and eval_loader is not None and configs is not None and getattr(configs, "eval_on_test_during_training", False):
                eval_metrics, eval_err = _safe_eval_split(
                    f"Eval epoch {epoch+1}",
                    evaluate_epoch,
                    model, eval_loader, processor, DEVICE,
                    use_amp=eval_use_amp, amp_dtype=eval_amp_dtype,
                    debug_log_path=eval_debug_log_path, enable_debug_logging=enable_debug_logging,
                    generation_cfg=generation_cls_cfg,return_example=log_eval_examples,
                    return_eval_stats=log_eval_stats_enabled
                )
                #restore training mode after epoch-level test eval.
                model.train()
                if not eval_err and eval_metrics is not None:
                    eval_stats = None
                    if log_eval_stats_enabled and isinstance(eval_metrics, dict):
                        eval_stats = eval_metrics.pop("eval_stats", None)
                    eval_metrics_clean, eval_example = split_metrics_example(eval_metrics)
                    test_eval_log_payload = {f"eval/{k}": v for k, v in eval_metrics_clean.items()}
                    print(f"Epoch {epoch+1}: {test_eval_log_payload}")
                    if eval_example:
                        print(f"[Eval example] {eval_example}")
                    if log_metrics is not None:
                        log_metrics(log_file=eval_log_file, epoch=epoch, metrics=eval_metrics_clean, examples=eval_example)
                    log_wandb(test_eval_log_payload, step=optimizer_step)
                    last_test_eval_epoch = max(last_test_eval_epoch, epoch + 1)
                    if eval_stats:
                        maybe_log_eval_stats(
                            enabled=log_eval_stats_enabled,
                            log_file=eval_log_file,
                            header=f"Epoch {epoch+1} (eval set)",
                            eval_stats=eval_stats,
                        )
                    
                    # Save separate checkpoint for best EVAL EER.
                    eval_eer = eval_metrics_clean.get("eer")
                    if eval_eer is not None and math.isfinite(eval_eer) and eval_eer < best_eval_eer:
                        best_eval_eer = eval_eer
                        if wandb_run:
                            wandb_run.summary["best_eval_eer"] = best_eval_eer
                        _best_eval_dir = os.path.join(OUTPUT_DIR, "best_eval")
                        _save_best_checkpoint(
                            model_save_dir=_best_eval_dir,
                            adapters_dir=os.path.join(OUTPUT_DIR, "adapters_best_eval"),
                            ckpt_path=os.path.join(_best_eval_dir, "checkpoint_best_eval.pt"),
                            metrics_json_path=os.path.join(OUTPUT_DIR, "best_eval_metrics.json"),
                            eer=eval_eer, metrics=eval_metrics_clean, label="best eval (epoch)",
                            epoch_num=epoch, step_num=optimizer_step,
                        )


            if configs is not None and getattr(configs, "eval_on_asvspoof2021_during_training", False) and asvspoof2021_eval_loader is not None:
                print_asv2021_info(f"epoch {epoch+1}")
                asv2021_metrics, asv2021_err = _safe_eval_split(
                    f"ASVspoof2021 epoch {epoch+1}",
                    evaluate_epoch,
                    model,
                    asvspoof2021_eval_loader,
                    processor,
                    DEVICE,
                    use_amp=eval_use_amp,
                    amp_dtype=eval_amp_dtype,
                    debug_log_path=eval_debug_log_path,
                    enable_debug_logging=enable_debug_logging,
                    generation_cfg=generation_cls_cfg,
                    return_example=log_eval_examples,
                )
                # restore training mode after epoch-level ASVspoof2021 eval.
                model.train()
                if not asv2021_err and asv2021_metrics is not None:
                    asv2021_metrics_clean, asv2021_example = split_metrics_example(asv2021_metrics)
                    asv2021_log_payload = {f"asvspoof2021/{k}": v for k, v in asv2021_metrics_clean.items()}
                    print(f"[Eval log] Epoch {epoch+1}: {asv2021_log_payload}")
                    if asv2021_example:
                        print(f"[Eval example - asvspoof2021] {asv2021_example}")
                    if log_metrics is not None and asvspoof2021_eval_log_file is not None:
                        log_metrics(log_file=asvspoof2021_eval_log_file, epoch=epoch, metrics=asv2021_metrics_clean, examples=asv2021_example)
                    log_wandb(asv2021_log_payload, step=optimizer_step)
                    last_asvspoof2021_eval_epoch = max(last_asvspoof2021_eval_epoch, epoch + 1)

        # Re-save the periodic checkpoint AFTER eval blocks so the fresh
        # last_*_eval_epoch counters are persisted. Without this, resume would
        # see stale counters and re-evaluate this epoch on the same model state.
        if _periodic_save_path is not None:
            ckpt = _build_checkpoint(epoch, optimizer_step, epoch_completed=True)
            save_checkpoint(_periodic_save_path, ckpt)
            print(f"[Periodic Re-save] Updated {_periodic_save_path} with fresh eval counters")

        # Rolling checkpoint: save only when the periodic SAVE_EVERY block did not
        # already write this epoch, to avoid writing the same file twice per epoch.
        _periodic_already_ran = (
            run_by_epochs
            and SAVE_EVERY is not None
            and SAVE_EVERY > 0
            and (epoch + 1) % SAVE_EVERY == 0
        )
        if save_best_only and not _periodic_already_ran:
            ckpt = _build_checkpoint(epoch, optimizer_step, epoch_completed=True)
            checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint-last.pt")
            save_checkpoint(checkpoint_path, ckpt)
            print(f"[Rolling Save] Saved checkpoint-last to {checkpoint_path} at epoch {epoch+1}")

    return best_eer
