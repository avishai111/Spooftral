import logging
import math
import os
from contextlib import nullcontext

# Resolve config path before importing utils.config
_script_dir = os.path.dirname(__file__)
_config_name = "hyperparameters_5_classification_head.yaml"
_config_in_configs = os.path.join(_script_dir, "configs", _config_name)
_config_legacy = os.path.join(_script_dir, _config_name)
_default_cfg_path = _config_in_configs if os.path.exists(_config_in_configs) else _config_legacy
os.environ["VOXTRAL_CONFIG"] = os.environ.get("VOXTRAL_CONFIG", _default_cfg_path)

import numpy as np
import torch
import torch.nn.functional as F
import torch.serialization
from tqdm import tqdm
import bitsandbytes as bnb
import torch.nn as nn

from utils.helper_function import (
    compute_eer,
    log_metrics,
    add_tekken_once_filter,
    Config as HFConfig,
)
from utils.checkpoint_utils import load_checkpoint_if_available, resume_checkpoint_evaluation
from utils.build_model_for_classification_head import (
    build_classification_head_model,
    compute_class_weights_from_dataset,
)
from utils.dataset import ASVspoofIndexDataset, collate_fn, make_dataloader
from utils.config import *
from utils.config_helper_functions import save_all_configs, print_config_summary, log_run_settings
from utils.training_helpers import save_checkpoint, next_available
from utils.training_helpers_audio_adapter_head import (
    normalize_initial_epochs,
    latest_metrics_file,
    last_logged_epoch,
    log_dev_metrics,
    log_eval_metrics,
    save_epoch_checkpoint,
)

# Registers the custom Config class as safe to unpickle so checkpoints containing
# it can be loaded under torch.load's weights_only safety check.
torch.serialization.add_safe_globals([HFConfig])
add_tekken_once_filter()


# Enable TF32 where possible
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def _maybe_autocast(device: str, use_amp: bool, use_fp16: bool, use_bf16: bool):
    if device.startswith("cuda"):
        if use_fp16 or use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _move_batch_for_adapter_head(batch, device: str, use_raw_audio: bool):
    """
    Move only required tensors to GPU for adapter-head training.
    - Internal Whisper (encoder_type=None): move input_features + classification_labels
    - External encoder (wavlm/wav2vec2/hubert): move raw_audio_input + classification_labels
    """
    # NOTE: We intentionally do NOT move input_ids/labels/attention_mask here because
    # the adapter-head path does not use the LLM. Keeping them on CPU saves VRAM.
    keys_to_device = {"classification_labels"}
    if use_raw_audio:
        keys_to_device.add("raw_audio_input")
    else:
        keys_to_device.add("input_features")

    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and k in keys_to_device:
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def _quantize_language_model_4bit(lm: nn.Module, compute_dtype: torch.dtype = torch.float16) -> nn.Module:
    """
    Quantize Linear layers inside the language model to 4-bit, leaving other modules untouched.
    Returns a new module tree with Linear replaced by bnb Linear4bit.
    """
    def _to_linear4bit(linear: nn.Linear) -> nn.Module:
        # Try modern API first
        if hasattr(bnb.nn, "Linear4bit") and hasattr(bnb.nn.Linear4bit, "from_float"):
            return bnb.nn.Linear4bit.from_float(linear, compute_dtype=compute_dtype)
        # Fallback: instantiate and copy weights/bias manually
        lin4 = bnb.nn.Linear4bit(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=compute_dtype,
        )
        lin4.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and hasattr(lin4, "bias") and lin4.bias is not None:
            lin4.bias.data.copy_(linear.bias.data)
        return lin4

    def _convert(module: nn.Module) -> nn.Module:
        for name, child in list(module.named_children()):
            module.add_module(name, _convert(child))
        if isinstance(module, nn.Linear):
            return _to_linear4bit(module)
        return module

    quantized = _convert(lm)
    for p in quantized.parameters():
        p.requires_grad = False
    return quantized


def _prepare_dataset(dataset_type: str, protocol: str, audio_dir: str, is_train: bool, fixed_prompt_text=None):
    """
    Build one ASVspoof dataset split with the current global config.
    This is a thin wrapper so we don't duplicate constructor args.
    """
    return ASVspoofIndexDataset(
        AUGMENTATION_PROB=AUGMENTATION_PROB,
        path_to_protocol=protocol,
        data_dir=audio_dir,
        dataset_type=dataset_type,
        processor=processor,
        prompts=PROMPTS,
        is_train=is_train,
        augmentation_on=AUGMENTATION_ON,
        augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
        reasoning=False,
        SPECIAL_TOKEN_TASK=SPECIAL_TOKEN_TASK,
        dtype=dtype,
        CONFIG_AUGMENTATION=CONFIG_AUGMENTATION,
        encoder_type=ENCODER_TYPE if ENCODER_TYPE not in ("none", None) else None,
        pad_features_mode=PAD_FEATURES_MODE,
        pad_waveform_mode=PAD_WAVEFORM_MODE,
        constant_prompts_flag=CONSTANT_PROMPTS_FLAG,
        three_random_prompts=THREE_RANDOM_PROMPTS_FLAG,
        random_prompt_variants=THREE_RANDOM_PROMPT_VARIANTS,
        fixed_prompt_text=fixed_prompt_text,
        N_audio_tokens=N_AUDIO_TOKENS,
        constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
        label_texts=LABEL_TEXTS,
    )


def _forward_and_loss_adapter_head(
    model,
    batch,
    criterion,
    use_raw_audio: bool,
    device: str,
    use_amp: bool,
    use_fp16: bool,
    use_bf16: bool,
):
    """
    Forward + loss for adapter-head training/eval under the same autocast context.
    This keeps numerical behavior consistent with the original implementation.
    """
    with _maybe_autocast(device, use_amp, use_fp16, use_bf16):
        if use_raw_audio:
            outputs = model(
                raw_audio_input=batch.get("raw_audio_input"),
                raw_audio_lengths=batch.get("raw_audio_length"),
            )
        else:
            outputs = model(
                input_features=batch.get("input_features"),
            )
        logits = outputs["logits"]
        loss = criterion(logits, batch["classification_labels"])
    return outputs, logits, loss


def _evaluate_adapter_head(model, loader, criterion, device, use_raw_audio: bool, use_amp=False, use_fp16=False, use_bf16=False):
    """
    Run evaluation loop for adapter-head classification.
    Returns basic metrics computed from logits (loss, EER, accuracy, mean spoof prob).
    """
    if loader is None:
        return None
    model.eval()
    losses, probs, labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = _move_batch_for_adapter_head(batch, device, use_raw_audio)
            outputs, logits, loss = _forward_and_loss_adapter_head(
                model, batch, criterion, use_raw_audio, device, use_amp, use_fp16, use_bf16
            )
            losses.append(loss.detach().item())
            # Class index 1 = spoof, index 0 = bonafide, matching the label convention used throughout.
            p_spoof = torch.softmax(logits, dim=-1)[:, 1]
            probs.extend(p_spoof.detach().cpu().tolist())
            labels.extend(batch["classification_labels"].detach().cpu().tolist())
            # Proactively release large tensors between iterations (helps reduce peak VRAM in long evals).
            del outputs, logits, loss

    # EER is reported as a percentage; an empty or NaN result falls back to the
    # worst-case 100.0 so downstream best-checkpoint comparisons still behave sensibly.
    _raw_eer = compute_eer(labels, probs) if labels else float("nan")
    eer = float(_raw_eer) * 100.0 if (labels and not math.isnan(float(_raw_eer))) else 100.0
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = float(np.mean(np.equal(preds, labels))) if labels else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_prob = float(np.mean(probs)) if probs else 0.0
    return {
        "loss": avg_loss,
        "eer": float(eer),
        "accuracy": acc,
        "avg_spoof_prob": avg_prob,
    }


def _resolve_adapter_head_cfg():
    """
    Normalize the audio_adapter_head config section into a plain dict.
    Also enforces the enable flag check.
    """
    adapter_head_cfg = getattr(configs, "audio_adapter_head", {}) or {}
    if not isinstance(adapter_head_cfg, dict) and hasattr(adapter_head_cfg, "__dict__"):
        adapter_head_cfg = dict(adapter_head_cfg.__dict__)
    if not adapter_head_cfg.get("enable", False):
        print("audio_adapter_head.enable must be True in the YAML to run this experiment.")
    return adapter_head_cfg


def _setup_logging(output_dir: str):
    """
    Create log files and a module logger for this run.
    Returns (logger, result_file, metrics_log_file, eval_log_file).
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "adapter_head_training.log")),
        ],
    )
    logger = logging.getLogger(__name__)
    result_file = next_available(os.path.join(output_dir, "train_log.txt"))
    metrics_log_file = next_available(os.path.join(output_dir, "metrics_log_dev.txt"))
    eval_log_file = next_available(os.path.join(output_dir, "metrics_log_eval.txt"))
    return logger, result_file, metrics_log_file, eval_log_file


def _setup_precision(precision: str):
    """
    Configure mixed-precision flags and the GradScaler based on PRECISION string.
    Returns (use_fp16, use_bf16, use_amp, scaler).
    """
    use_fp16 = precision == "float16"
    use_bf16 = precision == "bfloat16"
    use_amp = precision == "AMP"
    scaler = torch.amp.GradScaler(enabled=(use_amp or use_fp16))
    return use_fp16, use_bf16, use_amp, scaler


def _build_adapter_head_model(adapter_head_cfg, drop_language_model: bool, build_device: str, pretrained_ckpt, logger):
    """
    Build the adapter-head model with the requested config.
    Handles LLM removal before moving the model to GPU.
    """
    model = build_classification_head_model(
        base_model_name=VOXTRAL_MODEL,
        device=build_device,
        dtype=dtype,
        finetune_mode=finetune_mode,  # Passed to AudioAdapterHeadClassifier for audio encoder DoRA
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=TARGET_MODULES,
        use_4bit=USE_4BIT,
        precision=PRECISION,
        load_model_path=pretrained_ckpt,
        load_pretrained_audio_stack=bool(adapter_head_cfg.get("load_pretrained_audio_stack", True)),
        language_model_only_lora=LANGUAGE_MODEL_ONLY_LORA,
        encoder_type=ENCODER_TYPE,
        audio_encoder_tuning=adapter_head_cfg.get("audio_encoder_tuning", audio_encoder_mode),
        k_downsample=K_DOWNSAMPLE,
        num_classes=NUM_CLASSES,
        pooling_strategy=adapter_head_cfg.get("pooling_strategy", "mean"),
        head_hidden_dim=adapter_head_cfg.get("head_hidden_dim", 256),
        head_dropout=adapter_head_cfg.get("head_dropout", 0.1),
        train_audio_adapter=adapter_head_cfg.get("train_audio_adapter", True),
        drop_language_model=drop_language_model,
    )

    # Ensure LM is removed before any GPU move when requested
    if drop_language_model and hasattr(model, "_remove_language_model_from_device"):
        model._remove_language_model_from_device()

    # Debug: verify LM is gone before we move to GPU (should be False when drop_language_model=True)
    lm_exists_pre = hasattr(model, "model") and hasattr(model.model, "language_model") and model.model.language_model is not None
    print(f"[DEBUG] language_model exists (post-build, pre-move): {lm_exists_pre}")

    # Debug: compare param counts before/after .to(DEVICE) to detect unexpected changes
    total_params_before_move = sum(p.numel() for p in model.parameters())
    print(f"[DEBUG] Total params before .to(DEVICE): {total_params_before_move:,}")

    # Move to GPU after LLM was removed (if applicable)
    if drop_language_model and build_device == "cpu":
        model = model.to(DEVICE)
        torch.cuda.empty_cache()
        logger.info(f"Moved model to {DEVICE} after dropping language model.")
        total_params_after_move = sum(p.numel() for p in model.parameters())
        print(f"[DEBUG] Total params after .to(DEVICE): {total_params_after_move:,}")
        if total_params_after_move != total_params_before_move:
            print(
                f"[DEBUG][WARN] Param count changed after .to(DEVICE): "
                f"{total_params_before_move:,} -> {total_params_after_move:,}"
            )

    return model


def _debug_model_summary(model):
    """
    Print a structured debug summary of the model:
    - LLM presence and device
    - audio_tower stats
    - GPU memory usage
    - total/trainable parameter counts
    """
    print("\n" + "=" * 80)
    print("DEBUG: Model structure after building")
    print("=" * 80)

    # Check if language_model exists
    lm_exists = hasattr(model.model, "language_model") and model.model.language_model is not None
    print(f"language_model exists: {lm_exists}")
    if lm_exists:
        lm = model.model.language_model
        lm_params = sum(p.numel() for p in lm.parameters())
        print(f"  WARNING: language_model still has {lm_params:,} parameters!")
        print(f"  language_model device: {next(lm.parameters()).device}")

    # Check for any parameters that still include "language_model" in their name
    # This catches cases where references linger even if language_model attr is removed.
    lm_named_params = [(n, p) for n, p in model.named_parameters() if "language_model" in n]
    if lm_named_params:
        devices = {str(p.device) for _, p in lm_named_params}
        print(f"  WARNING: Found {len(lm_named_params)} params with 'language_model' in name on devices: {devices}")
        for name, p in lm_named_params[:10]:
            print(f"    {name} -> {p.device}")
    else:
        print("  No parameters with 'language_model' in name.")

    # Check audio_tower
    if hasattr(model.model, "audio_tower") and model.model.audio_tower is not None:
        at = model.model.audio_tower
        at_params = sum(p.numel() for p in at.parameters())
        at_trainable = sum(p.numel() for p in at.parameters() if p.requires_grad)
        print(f"audio_tower exists: True")
        print(f"  Total params: {at_params:,}")
        print(f"  Trainable params: {at_trainable:,}")
        print(f"  audio_tower is PeftModel: {type(at).__name__}")

    # GPU memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print("\nGPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")

    # Total model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 80 + "\n")


def _maybe_adjust_language_model(model, adapter_head_cfg, drop_language_model: bool, logger):
    """
    Optional: quantize or offload the language model (if it is still present).
    Returns flags for logging purposes.
    """
    quantize_lm_cfg = bool(adapter_head_cfg.get("quantize_language_model_4bit", False))
    offload_lm_cfg = bool(adapter_head_cfg.get("offload_language_model", False))
    quantize_lm_4bit = quantize_lm_cfg and not drop_language_model
    offload_lm = offload_lm_cfg and not drop_language_model

    if drop_language_model:
        logger.info("Language model removed for adapter-head training; skipping LM quantization/offload.")
        return quantize_lm_4bit, offload_lm

    if quantize_lm_4bit:
        lm_ref = getattr(model, "language_model", None)
        lm_parent = model
        if lm_ref is None and hasattr(model, "model") and hasattr(model.model, "language_model"):
            lm_ref = model.model.language_model
            lm_parent = model.model
        if lm_ref is None:
            raise ValueError("quantize_language_model_4bit was requested but model has no language_model.")
        lm_parent.language_model = _quantize_language_model_4bit(lm_ref, compute_dtype=torch.float16)
        offload_lm = True
    if offload_lm:
        lm_ref = getattr(model, "language_model", None)
        if lm_ref is None and hasattr(model, "model") and hasattr(model.model, "language_model"):
            lm_ref = model.model.language_model
        if lm_ref is not None:
            lm_ref.to("cpu")
            torch.cuda.empty_cache()
    return quantize_lm_4bit, offload_lm


def _build_datasets():
    """
    Create train/dev/eval datasets for the current DATASET config.
    Keeps logic for dataset-type normalization in one place.
    """
    # ASVspoof2021 has no official training partition, so training always draws from
    # the 2019 protocol/audio even when evaluating on 2021.
    train_dataset_type = "asvspoof2019" if configs.DATASET == "asvspoof2021" else configs.DATASET
    dev_dataset_type = train_dataset_type
    train_dataset = _prepare_dataset(train_dataset_type, TRAIN_PROTOCOL, TRAIN_AUDIO_DIR, is_train=True)
    dev_dataset = _prepare_dataset(dev_dataset_type, DEV_PROTOCOL, DEV_AUDIO_DIR, is_train=False, fixed_prompt_text=EVAL_PROMPT_TEXT)
    eval_dataset = _prepare_dataset(configs.DATASET, EVAL_PROTOCOL, EVAL_AUDIO_DIR, is_train=False, fixed_prompt_text=EVAL_PROMPT_TEXT)
    return train_dataset, dev_dataset, eval_dataset


def _build_dataloaders(train_dataset, dev_dataset, eval_dataset):
    """
    Construct DataLoaders with consistent settings and computed eval batch size.
    Returns (train_loader, dev_loader, test_loader, eval_batch_size_cfg).
    """
    train_loader = make_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        ensure_bonafide_in_batch=ENSURE_BONAFIDE_IN_BATCH,
        balance_bonafide_spoof=BALANCE_BONAFIDE_SPOOF,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    # Default eval batch size to double the training batch size: evaluation runs
    # without gradients and needs less memory per sample.
    eval_batch_size_cfg = getattr(configs, "eval_batch_size", None)
    if eval_batch_size_cfg is None or eval_batch_size_cfg <= 0:
        eval_batch_size_cfg = BATCH_SIZE * 2
    eval_batch_size_cfg = max(1, int(eval_batch_size_cfg))

    dev_loader = make_dataloader(
        dev_dataset,
        batch_size=eval_batch_size_cfg,
        shuffle=False,
        ensure_bonafide_in_batch=False,
        balance_bonafide_spoof=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    test_loader = make_dataloader(
        eval_dataset,
        batch_size=eval_batch_size_cfg,
        shuffle=False,
        ensure_bonafide_in_batch=False,
        balance_bonafide_spoof=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    return train_loader, dev_loader, test_loader, eval_batch_size_cfg


def _compute_class_weights(train_dataset, adapter_head_cfg, logger):
    """
    Compute or load class weights for the classification head loss.
    Returns (class_weights, head_use_weighted).
    """
    head_use_weighted = bool(adapter_head_cfg.get("use_weighted_loss", False))
    class_weights_cfg = getattr(configs, "class_weights", None)
    class_weights = None
    if head_use_weighted:
        if not isinstance(class_weights_cfg, dict) and hasattr(class_weights_cfg, "__dict__"):
            class_weights_cfg = class_weights_cfg.__dict__
        if isinstance(class_weights_cfg, dict):
            bona = class_weights_cfg.get("bonafide", None)
            spoof = class_weights_cfg.get("spoof", None)
            if bona is not None and spoof is not None:
                try:
                    class_weights = torch.tensor([float(bona), float(spoof)], device=DEVICE, dtype=torch.float)
                    logger.info(f"Using class weights from YAML: bonafide={bona}, spoof={spoof}")
                except Exception as exc:
                    logger.warning(f"Invalid class weights in YAML ({exc}); falling back to auto-computed.")
        if class_weights is None:
            class_weights = compute_class_weights_from_dataset(train_dataset, DEVICE)
            logger.info(f"Auto-computed class weights: {class_weights.tolist()}")
    return class_weights, head_use_weighted


def _compute_training_schedule(train_loader):
    """
    Compute steps_per_epoch, total_train_steps, and warmup_steps based on config.
    Handles both epoch-based and step-based training.
    """
    steps_per_epoch = math.ceil(len(train_loader) / max(1, gradient_accumulation_steps))
    if RUNNING_BY_EPOCHS:
        total_train_steps = steps_per_epoch * NUM_EPOCHS
    else:
        if MAX_TRAIN_STEPS is None:
            raise ValueError("max_train_steps must be set when train_by == 'steps'")
        total_train_steps = MAX_TRAIN_STEPS
    warmup_steps = int(total_train_steps * WARMUP_RATIO)
    return steps_per_epoch, total_train_steps, warmup_steps


def _setup_optimizer_scheduler(model, total_train_steps: int, warmup_steps: int):
    """
    Build optimizer and scheduler using config defaults.
    """
    optimizer = get_optimizer(
        None,
        lr=configs.lr,
        weight_decay=configs.weight_decay,
        params=[p for p in model.parameters() if p.requires_grad],
    )
    scheduler = get_scheduler(optimizer=optimizer, num_training_steps=total_train_steps, warmup_steps=warmup_steps)
    return optimizer, scheduler


def _log_trainable_params(model, logger):
    """
    Log the names of all trainable parameters for debugging.
    """
    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  {name}")


def main():
    """
    Main training entrypoint for adapter-head classification.
    The flow is:
      1) load config + logging + precision
      2) build model (optionally drop LLM)
      3) build datasets/loaders
      4) setup optimizer/scheduler
      5) train with periodic eval + checkpoints
    """
    adapter_head_cfg = _resolve_adapter_head_cfg()
    save_best_only = bool(getattr(getattr(configs, "model", None), "save_best_only", False))
    drop_language_model = bool(adapter_head_cfg.get("drop_language_model", False))

    # ---- Logging + precision ----
    logger, result_file, metrics_log_file, eval_log_file = _setup_logging(OUTPUT_DIR)
    use_fp16, use_bf16, use_amp, scaler = _setup_precision(PRECISION)

    # ---- Model build ----
    pretrained_ckpt = adapter_head_cfg.get("pretrained_checkpoint", None)
    
    # When dropping LLM, load on CPU first to avoid OOM, then move to GPU after LLM is removed
    build_device = "cpu" if drop_language_model else DEVICE
    
    model = _build_adapter_head_model(
        adapter_head_cfg=adapter_head_cfg,
        drop_language_model=drop_language_model,
        build_device=build_device,
        pretrained_ckpt=pretrained_ckpt,
        logger=logger,
    )
    
    # ============================================================================
    # DEBUG: Print memory usage and model structure to diagnose OOM issues
    # ============================================================================
    _debug_model_summary(model)

    print(model)

    # ---- Optional LM adjustments (only if LM is present) ----
    quantize_lm_4bit, offload_lm = _maybe_adjust_language_model(
        model=model,
        adapter_head_cfg=adapter_head_cfg,
        drop_language_model=drop_language_model,
        logger=logger,
    )

    # ---- Trainable params log ----
    _log_trainable_params(model, logger)

    # Cap this process at 99% of GPU memory, leaving a small reserve to reduce the
    # chance of a fatal out-of-memory allocation.
    torch.cuda.set_per_process_memory_fraction(0.99, device=0)
    print(model)

    # ---- Data ----
    train_dataset, dev_dataset, eval_dataset = _build_datasets()
    class_weights, head_use_weighted = _compute_class_weights(train_dataset, adapter_head_cfg, logger)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_loader, dev_loader, test_loader, eval_batch_size_cfg = _build_dataloaders(
        train_dataset, dev_dataset, eval_dataset
    )
    steps_per_epoch, total_train_steps, warmup_steps = _compute_training_schedule(train_loader)
    optimizer, scheduler = _setup_optimizer_scheduler(model, total_train_steps, warmup_steps)

    # Save configs before training
    save_all_configs(
        OUTPUT_DIR,
        configs,
        module_path="utils.config",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=criterion,
    )

    # W&B setup (defer init until after resume)
    wandb_run = None
    wandb_cfg = getattr(configs, "wandb", {}) or {}
    wandb_enabled = wandb_cfg.get("enable") if isinstance(wandb_cfg, dict) else getattr(wandb_cfg, "enable", True)
    wandb_enabled = True if wandb_enabled is None else bool(wandb_enabled)

    logger.info("Configuration summary (audio_adapter_head):")
    loss_log_params = {
        "use_weighted_loss": head_use_weighted,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "pooling_strategy": adapter_head_cfg.get("pooling_strategy", "mean"),
        "audio_encoder_tuning": adapter_head_cfg.get("audio_encoder_tuning", audio_encoder_mode),
        "quantize_language_model_4bit": bool(quantize_lm_4bit),
        "offload_language_model": bool(offload_lm),
        "drop_language_model": drop_language_model,
        "save_best_only": save_best_only,
    }
    print_config_summary(configs, "adapter_head", loss_log_params)
    log_run_settings("adapter_head", loss_log_params)

    # Build config_dict for checkpoint saving
    config_dict = {k: v for k, v in vars(configs).items() if not k.startswith("_")}

    start_epoch = 0
    start_step = 0
    global_step = 0
    best_dev_eer = float("inf")
    best_eval_eer = float("inf")
    best_dev_path = None
    best_eval_path = None
    last_final_path = None
    best_checkpoint_path = None

    resume_state = load_checkpoint_if_available(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        configs=configs,
        wandb_enabled=wandb_enabled,
        wandb_run=wandb_run,
        resume_ckpt_path=None,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        steps_per_epoch=steps_per_epoch,
        start_epoch=start_epoch,
        start_step=start_step,
        best_eer=float("inf"),
        best_dev_eer=best_dev_eer,
        best_eval_eer=best_eval_eer,
        last_eval_epoch=None,
        last_eval_step=None,
        last_train_eval_epoch=None,
        last_train_eval_step=None,
        last_test_eval_epoch=None,
        last_test_eval_step=None,
        last_asvspoof2021_eval_epoch=None,
        last_asvspoof2021_eval_step=None,
    )

    resume_ckpt_path = resume_state["ckpt_path"]
    wandb_run = resume_state["wandb_run"]
    start_epoch = resume_state["start_epoch"]
    start_step = resume_state["start_step"]
    global_step = int(start_step)
    best_dev_eer = resume_state.get("best_dev_eer", best_dev_eer)
    best_eval_eer = resume_state.get("best_eval_eer", best_eval_eer)
    last_eval_epoch = resume_state["last_eval_epoch"]
    last_eval_step = resume_state["last_eval_step"]
    last_test_eval_epoch = resume_state["last_test_eval_epoch"]
    last_test_eval_step = resume_state["last_test_eval_step"]

    if resume_ckpt_path:
        logger.info(
            f"Resumed training from {resume_ckpt_path} "
            f"(next_epoch {start_epoch + 1}, global_step {global_step}, "
            f"best_dev_eer {best_dev_eer:.4f}, best_eval_eer {best_eval_eer:.4f})"
        )
    else:
        logger.info("Starting fresh training run (no checkpoint to resume).")

    if wandb_run is None and wandb_enabled:
        wandb_run = init_wandb(
            configs,
            project=configs.wandb.project,
            run_name=configs.wandb.name,
            resume=bool(getattr(configs, "resume", False)),
            wandb_cfg=configs.wandb,
        )

    if os.path.exists(os.path.join(OUTPUT_DIR, "best_adapter_head_dev.pt")):
        best_dev_path = os.path.join(OUTPUT_DIR, "best_adapter_head_dev.pt")
    if os.path.exists(os.path.join(OUTPUT_DIR, "best_adapter_head_eval.pt")):
        best_eval_path = os.path.join(OUTPUT_DIR, "best_adapter_head_eval.pt")

    # Decide which input path the adapter-head should use:
    # - Internal Whisper: use mel features only
    # - External encoders: use raw waveform only
    use_raw_audio = ENCODER_TYPE not in (None, "none")

    # Margin loss config — mirrors the CAG implementation in training_evaluate_classification.py.
    # diff = logit_correct - logit_wrong  (equivalent to log_softmax diff since log-Z cancels)
    # loss = relu(margin_per_class - diff).mean()
    use_margin_loss = bool(getattr(configs, "use_margin_loss", False))
    try:
        margin_bonafide = float(getattr(configs, "margin_bonafide", 1.0) or 1.0)
    except Exception:
        margin_bonafide = 1.0
    try:
        margin_spoof = float(getattr(configs, "margin_spoof", 1.0) or 1.0)
    except Exception:
        margin_spoof = 1.0
    try:
        lambda_margin = float(getattr(configs, "lambda_margin", 0.3) or 0.3)
    except Exception:
        lambda_margin = 0.3

    # Derive semantic keys from label_texts (same synonym logic as CAG script).
    _SPOOF_SYNONYMS = ("spoof", "fake", "synthetic", "generated", "artificial")
    _BONA_SYNONYMS  = ("bonafide", "real", "genuine", "authentic", "natural")
    # Fall back to the standard ASVspoof convention (0=bonafide, 1=spoof) if no label
    # text matches a known synonym.
    _spoof_key = next(
        (k for k, v in LABEL_TEXTS.items() if any(w in str(v).lower() for w in _SPOOF_SYNONYMS)), 1
    )
    _bonafide_key = next(
        (k for k, v in LABEL_TEXTS.items() if any(w in str(v).lower() for w in _BONA_SYNONYMS)), 0
    )
    if use_margin_loss:
        logger.info(
            f"Margin loss enabled — bonafide key: {_bonafide_key}, spoof key: {_spoof_key}, "
            f"margin_bonafide: {margin_bonafide}, margin_spoof: {margin_spoof}, lambda: {lambda_margin}"
        )

    def _evaluate_epoch_adapter(
        model,
        loader,
        processor,
        DEVICE,
        use_amp,
        amp_dtype,
        debug_log_path,
        eval_step,
        enable_debug_logging,
        return_example=False,
        generation_cfg=None,
        return_eval_stats=False,
    ):
        del processor, amp_dtype, debug_log_path, eval_step, enable_debug_logging, return_example, generation_cfg, return_eval_stats
        return _evaluate_adapter_head(model, loader, criterion, DEVICE, use_raw_audio, use_amp, use_fp16, use_bf16)

    # Auto-detect evaluations that were missed due to a crash between training end
    # and eval completion.  We check checkpoint tracker fields first (set to epoch+1
    # after each split finishes); fall back to log-file parsing for legacy checkpoints
    # that pre-date the tracker fields.
    #
    # Condition: eval was *expected* at the last completed epoch AND the tracker
    # shows it never finished (last_*_epoch < start_epoch).
    if resume_ckpt_path and start_epoch > 0:
        # Mirror the exact should_eval logic from the training loop so we only
        # check for a missed eval when one was actually scheduled at this epoch.
        if RUNNING_BY_EPOCHS:
            if bool(EVAL_EVERY) and EVAL_EVERY > 0:
                # Eval runs whenever (epoch+1) % EVAL_EVERY == 0.
                # start_epoch = epoch+1, so this is equivalent.
                _eval_expected = start_epoch % EVAL_EVERY == 0
            else:
                # No EVAL_EVERY set: only the last epoch triggers eval.
                _eval_expected = start_epoch == NUM_EPOCHS
        else:
            # Step-based training: only the last epoch triggers eval.
            _eval_expected = start_epoch == NUM_EPOCHS
        if _eval_expected:
            if last_eval_epoch is not None:
                # Checkpoint tracker available: compare directly.
                resume_dev_eval_needed = last_eval_epoch < start_epoch
            else:
                # Legacy checkpoint: fall back to log-file parsing.
                last_dev_epoch = last_logged_epoch(latest_metrics_file(OUTPUT_DIR, "metrics_log_dev*.txt"))
                resume_dev_eval_needed = not (last_dev_epoch is not None and last_dev_epoch >= start_epoch - 1)

            if last_test_eval_epoch is not None:
                resume_test_eval_needed = last_test_eval_epoch < start_epoch
            else:
                last_eval_epoch_log = last_logged_epoch(latest_metrics_file(OUTPUT_DIR, "metrics_log_eval*.txt"))
                resume_test_eval_needed = not (last_eval_epoch_log is not None and last_eval_epoch_log >= start_epoch - 1)
        else:
            resume_dev_eval_needed = False
            resume_test_eval_needed = False
    else:
        resume_dev_eval_needed = False
        resume_test_eval_needed = False

    if resume_dev_eval_needed or resume_test_eval_needed:
        logger.info(
            f"Detected missed evaluations at epoch {start_epoch}: "
            f"dev={'needed' if resume_dev_eval_needed else 'ok'}, "
            f"test={'needed' if resume_test_eval_needed else 'ok'}"
        )

        resume_metrics = {}

        def _log_metrics_capture(log_file, epoch=None, step=None, metrics=None, examples=None):
            log_metrics(log_file=log_file, epoch=epoch, step=step, metrics=metrics, examples=examples)
            if log_file == metrics_log_file:
                resume_metrics["dev"] = metrics or {}
            elif log_file == eval_log_file:
                resume_metrics["eval"] = metrics or {}

        def _log_wandb(payload, step=None):
            if wandb_run is not None:
                wandb_run.log(payload, step=step)

        eval_use_amp = bool(use_amp or use_fp16 or use_bf16)
        eval_amp_dtype = "bf16" if use_bf16 else "fp16"

        (
            last_eval_epoch,
            last_eval_step,
            _last_train_eval_epoch,
            _last_train_eval_step,
            last_test_eval_epoch,
            last_test_eval_step,
            _last_asvspoof2021_eval_epoch,
            _last_asvspoof2021_eval_step,
        ) = resume_checkpoint_evaluation(
            model=model,
            evaluate_epoch=_evaluate_epoch_adapter,
            processor=None,
            DEVICE=DEVICE,
            start_epoch=start_epoch,
            optimizer_step=global_step,
            run_by_epochs=RUNNING_BY_EPOCHS,
            dev_loader=dev_loader,
            train_eval_loader=None,
            eval_loader=test_loader,
            asvspoof2021_eval_loader=None,
            eval_use_amp=eval_use_amp,
            eval_amp_dtype=eval_amp_dtype,
            debug_log_path=None,
            eval_debug_log_path=None,
            enable_debug_logging=False,
            log_eval_examples=False,
            resume_dev_eval_needed=resume_dev_eval_needed,
            resume_train_eval_needed=False,
            resume_test_eval_needed=resume_test_eval_needed,
            resume_asvspoof2021_eval_needed=False,
            log_metrics=_log_metrics_capture,
            metrics_log_file=metrics_log_file,
            train_eval_log_file=None,
            eval_log_file=eval_log_file,
            asvspoof2021_eval_log_file=None,
            log_wandb=_log_wandb,
            print_asvspoof2021_info=lambda *_args, **_kwargs: None,
            configs=configs,
            log_eval_stats_enabled=False,
            last_eval_epoch=last_eval_epoch,
            last_eval_step=last_eval_step,
            last_train_eval_epoch=-1,
            last_train_eval_step=-1,
            last_test_eval_epoch=last_test_eval_epoch,
            last_test_eval_step=last_test_eval_step,
            last_asvspoof2021_eval_epoch=-1,
            last_asvspoof2021_eval_step=-1,
            generation_cfg=None,
        )

        # Update best checkpoints if the resumed evals produced better EER.
        dev_eer = resume_metrics.get("dev", {}).get("eer")
        eval_eer = resume_metrics.get("eval", {}).get("eer")
        if dev_eer is not None and dev_eer < best_dev_eer:
            best_dev_eer = float(dev_eer)
            best_dev_path = os.path.join(OUTPUT_DIR, "best_adapter_head_dev.pt")
            save_checkpoint(best_dev_path, {
                "epoch": start_epoch, "global_step": int(global_step),
                "best_dev_eer": float(best_dev_eer), "best_eval_eer": float(best_eval_eer),
                "model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "configs": config_dict,
            })
            logger.info(f"[Resume eval] Improved dev EER {dev_eer:.4f}; saved {best_dev_path}")
        if eval_eer is not None and eval_eer < best_eval_eer:
            best_eval_eer = float(eval_eer)
            best_eval_path = os.path.join(OUTPUT_DIR, "best_adapter_head_eval.pt")
            save_checkpoint(best_eval_path, {
                "epoch": start_epoch, "global_step": int(global_step),
                "best_dev_eer": float(best_dev_eer), "best_eval_eer": float(best_eval_eer),
                "model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "configs": config_dict,
            })
            logger.info(f"[Resume eval] Improved eval EER {eval_eer:.4f}; saved {best_eval_path}")

        # Persist the updated eval trackers so the next resume won't re-run these evals.
        save_epoch_checkpoint(
            epoch=start_epoch - 1,
            global_step=global_step,
            best_dev_eer=best_dev_eer,
            best_eval_eer=best_eval_eer,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config_dict=config_dict,
            output_dir=OUTPUT_DIR,
            save_best_only=save_best_only,
            logger=logger,
            save_checkpoint=save_checkpoint,
            last_eval_epoch=last_eval_epoch,
            last_test_eval_epoch=last_test_eval_epoch,
            save_epoch_file=False,  # weights unchanged; only trackers need updating
        )

    # initial_eval_epochs lets the YAML request extra dev/eval passes (e.g. epoch 0)
    # outside the regular EVAL_EVERY schedule, useful for a pre-training baseline.
    initial_eval_epochs = normalize_initial_epochs(getattr(configs, "initial_eval_epochs", None))
    initial_eval_epochs_eval = normalize_initial_epochs(getattr(configs, "initial_eval_epochs_eval", None))
    initial_eval_epochs_set = set(initial_eval_epochs)
    initial_eval_epochs_eval_set = set(initial_eval_epochs_eval)
    initial_eval_epochs_done = set()
    initial_eval_epochs_eval_done = set()

    for epoch in range(start_epoch, NUM_EPOCHS):
        if epoch in initial_eval_epochs_set and epoch not in initial_eval_epochs_done:
            dev_metrics = _evaluate_adapter_head(model, dev_loader, criterion, DEVICE, use_raw_audio, use_amp, use_fp16, use_bf16)
            best_dev_eer, best_dev_path = log_dev_metrics(
                dev_metrics,
                epoch=epoch,
                label="Initial",
                metrics_log_file=metrics_log_file,
                logger=logger,
                wandb_run=wandb_run,
                global_step=global_step,
                best_dev_eer=best_dev_eer,
                best_dev_path=best_dev_path,
                output_dir=OUTPUT_DIR,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config_dict=config_dict,
                start_epoch=start_epoch,
                save_checkpoint=save_checkpoint,
            )
            initial_eval_epochs_done.add(epoch)
        if epoch in initial_eval_epochs_eval_set and epoch not in initial_eval_epochs_eval_done:
            eval_metrics = _evaluate_adapter_head(model, test_loader, criterion, DEVICE, use_raw_audio, use_amp, use_fp16, use_bf16)
            best_eval_eer, best_eval_path = log_eval_metrics(
                eval_metrics,
                epoch=epoch,
                label="Initial",
                eval_log_file=eval_log_file,
                logger=logger,
                wandb_run=wandb_run,
                global_step=global_step,
                best_eval_eer=best_eval_eer,
                best_eval_path=best_eval_path,
                output_dir=OUTPUT_DIR,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config_dict=config_dict,
                start_epoch=start_epoch,
                save_checkpoint=save_checkpoint,
            )
            initial_eval_epochs_eval_done.add(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_margin_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=True)
        for step, batch in enumerate(loop):
            batch = _move_batch_for_adapter_head(batch, DEVICE, use_raw_audio)
            outputs, logits, loss = _forward_and_loss_adapter_head(
                model, batch, criterion, use_raw_audio, DEVICE, use_amp, use_fp16, use_bf16
            )

            margin_loss_value = 0.0
            if use_margin_loss:
                cls_labels = batch["classification_labels"]
                bs = logits.size(0)
                arange = torch.arange(bs, device=logits.device)
                logp_correct = logits[arange, cls_labels]
                logp_wrong   = logits[arange, 1 - cls_labels]
                diff = logp_correct - logp_wrong
                is_spoof = cls_labels == _spoof_key
                margin_tensor = torch.where(
                    is_spoof,
                    logits.new_tensor(margin_spoof),
                    logits.new_tensor(margin_bonafide),
                )
                margin_loss = F.relu(margin_tensor - diff).mean()
                loss = loss + logits.new_tensor(lambda_margin) * margin_loss
                margin_loss_value = float(margin_loss.detach().item())

            # Scale the loss down so summing gradients over gradient_accumulation_steps
            # batches is equivalent to averaging over one larger effective batch.
            loss = loss / gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_value = loss.detach().item()
            running_loss += loss_value
            running_margin_loss += margin_loss_value
            current_loss = loss_value * gradient_accumulation_steps
            avg_loss_so_far = running_loss * gradient_accumulation_steps / (step + 1)
            avg_margin_so_far = running_margin_loss / (step + 1)
            if use_margin_loss:
                loop.set_postfix(loss=f"{current_loss:.4f}", avg=f"{avg_loss_so_far:.4f}", margin=f"{avg_margin_so_far:.4f}")
            else:
                loop.set_postfix(loss=f"{current_loss:.4f}", avg=f"{avg_loss_so_far:.4f}")
            # Release large tensors once they're no longer needed in this step.
            del outputs, logits, loss

            if LOG_EVERY and (step + 1) % LOG_EVERY == 0:
                with open(result_file, "a") as f:
                    margin_txt = f" L_margin {avg_margin_so_far:.4f}" if use_margin_loss else ""
                    f.write(
                        f"Epoch {epoch + 1} Step {step + 1} "
                        f"Loss {avg_loss_so_far:.4f} (current {current_loss:.4f}){margin_txt}\n"
                    )
            # Only step the optimizer/scheduler on accumulation boundaries (every
            # gradient_accumulation_steps batches, or the final batch of the epoch).
            boundary = ((step + 1) % gradient_accumulation_steps == 0) or ((step + 1) == len(train_loader))
            if boundary:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm if max_grad_norm is not None else float("inf"))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if not RUNNING_BY_EPOCHS and global_step >= total_train_steps:
                    break

        # Multiply back by gradient_accumulation_steps to undo the earlier division and
        # report the loss on its original (unscaled) scale.
        avg_train_loss = running_loss * gradient_accumulation_steps / max(1, len(train_loader))
        avg_train_margin_loss = running_margin_loss / max(1, len(train_loader))
        epoch_metrics = {"train_loss": avg_train_loss}
        if use_margin_loss:
            epoch_metrics["train_margin_loss"] = avg_train_margin_loss
        log_metrics(result_file, epoch=epoch, metrics=epoch_metrics)
        if wandb_run:
            wandb_payload = {"train/loss": avg_train_loss, "epoch": epoch}
            if use_margin_loss:
                wandb_payload["train/margin_loss"] = avg_train_margin_loss
            wandb_run.log(wandb_payload)

        # Evaluate every EVAL_EVERY epochs when set, otherwise only after the final epoch.
        should_eval = False
        if RUNNING_BY_EPOCHS:
            if EVAL_EVERY and EVAL_EVERY > 0:
                should_eval = ((epoch + 1) % EVAL_EVERY == 0)
            elif epoch == NUM_EPOCHS - 1:
                should_eval = True
        elif epoch == NUM_EPOCHS - 1:
            should_eval = True

        # Compute state_dict once; reuse across Pass 1, log_dev/eval_metrics, and
        # Pass 2 to avoid 3-4 expensive serializations (~3-8s each) per eval epoch.
        cached_state_dict = model.state_dict()

        # Pass 1: save weights before evaluations.  If a crash happens mid-eval,
        # the model weights for this epoch are preserved and the resume block above
        # will detect the missing eval via last_eval_epoch / last_test_eval_epoch.
        save_epoch_checkpoint(
            epoch=epoch,
            global_step=global_step,
            best_dev_eer=best_dev_eer,
            best_eval_eer=best_eval_eer,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config_dict=config_dict,
            output_dir=OUTPUT_DIR,
            save_best_only=save_best_only,
            logger=logger,
            save_checkpoint=save_checkpoint,
            last_eval_epoch=last_eval_epoch,
            last_test_eval_epoch=last_test_eval_epoch,
            save_epoch_file=True,
            cached_state_dict=cached_state_dict,
        )

        if should_eval:
            dev_metrics = _evaluate_adapter_head(model, dev_loader, criterion, DEVICE, use_raw_audio, use_amp, use_fp16, use_bf16)
            best_dev_eer, best_dev_path = log_dev_metrics(
                dev_metrics,
                epoch=epoch,
                label=None,
                metrics_log_file=metrics_log_file,
                logger=logger,
                wandb_run=wandb_run,
                global_step=global_step,
                best_dev_eer=best_dev_eer,
                best_dev_path=best_dev_path,
                output_dir=OUTPUT_DIR,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config_dict=config_dict,
                start_epoch=start_epoch,
                save_checkpoint=save_checkpoint,
                cached_state_dict=cached_state_dict,
            )
            # Mark dev eval complete so a crash before test eval is detectable on resume.
            last_eval_epoch = epoch + 1

            test_metrics = _evaluate_adapter_head(model, test_loader, criterion, DEVICE, use_raw_audio, use_amp, use_fp16, use_bf16)
            best_eval_eer, best_eval_path = log_eval_metrics(
                test_metrics,
                epoch=epoch,
                label=None,
                eval_log_file=eval_log_file,
                logger=logger,
                wandb_run=wandb_run,
                global_step=global_step,
                best_eval_eer=best_eval_eer,
                best_eval_path=best_eval_path,
                output_dir=OUTPUT_DIR,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config_dict=config_dict,
                start_epoch=start_epoch,
                save_checkpoint=save_checkpoint,
                cached_state_dict=cached_state_dict,
            )
            last_test_eval_epoch = epoch + 1

            # Pass 2: overwrite checkpoint-last.pt with updated eval trackers.
            # checkpoint-epochN.pt is skipped (save_epoch_file=False) — weights
            # are identical to pass 1 so there is nothing to re-write.
            save_epoch_checkpoint(
                epoch=epoch,
                global_step=global_step,
                best_dev_eer=best_dev_eer,
                best_eval_eer=best_eval_eer,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config_dict=config_dict,
                output_dir=OUTPUT_DIR,
                save_best_only=save_best_only,
                logger=logger,
                save_checkpoint=save_checkpoint,
                last_eval_epoch=last_eval_epoch,
                last_test_eval_epoch=last_test_eval_epoch,
                save_epoch_file=False,
                cached_state_dict=cached_state_dict,
            )

        del cached_state_dict

        if not RUNNING_BY_EPOCHS and global_step >= total_train_steps:
            break

    # Final save
    if not save_best_only:
        final_payload = {
            "epoch": NUM_EPOCHS,
            "global_step": int(global_step),
            "best_dev_eer": float(best_dev_eer),
            "best_eval_eer": float(best_eval_eer),
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "configs": config_dict,
            "last_eval_epoch": last_eval_epoch,
            "last_test_eval_epoch": last_test_eval_epoch,
        }
        last_final_path = os.path.join(OUTPUT_DIR, "adapter_head_final.pt")
        save_checkpoint(last_final_path, final_payload)
    else:
        logger.info(
            f"save_best_only=True; final checkpoint skipped. "
            f"Best DEV: {best_dev_path}, Best EVAL: {best_eval_path}"
        )

    logger.info(
        f"Checkpoint summary — best_dev: {best_dev_path}, best_eval: {best_eval_path}, final: {last_final_path}"
    )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
