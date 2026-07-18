#!/usr/bin/env python3
"""
Unified training script for Voxtral audio spoofing detection.

Usage:
    python scripts/train.py --config configs/hyperparameters_2021_LA_2.yaml
    python scripts/train.py -c configs/hyperparameters_05_no_margin.yaml

This is an alternative to launching the per-experiment
more_training_voxtral_scripts/training_voxtral_generation_tokens*.py scripts
directly: a single entry point that accepts the config file as an argument.
"""
import argparse
import json
import logging
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Voxtral model for audio spoofing detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., configs/hyperparameters_05_no_margin.yaml)",
    )
    return parser.parse_args()

args = parse_args()

# Resolve config path (support both relative and absolute paths)
config_path = Path(args.config)
if not config_path.is_absolute():
    # Try relative to current working directory first
    if not config_path.exists():
        # Try relative to project root (parent of scripts/)
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / args.config

if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {args.config}")

# Set the config path environment variable BEFORE any other imports
os.environ["VOXTRAL_CONFIG"] = str(config_path.resolve())
print(f"Using config: {config_path.resolve()}")

# Now import everything else (they depend on VOXTRAL_CONFIG). This has to happen
# at module scope rather than inside main(): `from utils.config import *` is a
# SyntaxError inside a function body, and utils.config reads VOXTRAL_CONFIG at
# import time, so the env var above must already be set before this runs.
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from torch.utils.data import Subset
print(transformers.__version__)

from utils.helper_function import (
    get_resume_checkpoint_info,
    log_metrics,
    add_tekken_once_filter,
    Config as HFConfig,
)
add_tekken_once_filter()

from utils.build_model import (
    build_voxtral_model,
    VoxtralStageWrapper_classification,
    compute_class_weights_from_dataset
)
from utils.checkpoint_utils import load_checkpoint_if_available
from utils.training_evaluate_classification import (
    train_classification_model,
    evaluate_epoch_classification_as_generation,
)
from utils.dataset import ASVspoofIndexDataset, collate_fn, make_dataloader
from utils.config import *
from utils.config_helper_functions import save_all_configs, log_run_settings, print_config_summary
import torch.serialization
torch.serialization.add_safe_globals([HFConfig])
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")

N_audio_tokens = N_AUDIO_TOKENS


def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    if torch.cuda.is_available():
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ========== LOGGING ==========
    log_dir = OUTPUT_DIR if 'OUTPUT_DIR' in dir() else '.'
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
        ]
    )
    logger = logging.getLogger(__name__)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    result_file = os.path.join(OUTPUT_DIR, "train_log.txt")
    metrics_log_file = os.path.join(OUTPUT_DIR, "metrics_log_dev.txt")
    eval_log_file = os.path.join(OUTPUT_DIR, "metrics_log_eval.txt")
    train_eval_log_file = os.path.join(OUTPUT_DIR, "metrics_train_log.txt")
    asvspoof2021_eval_log_file = os.path.join(OUTPUT_DIR, "metrics_log_asvspoof2021.txt")

    # ========== Resume if possible ==========
    cur_ckpts = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if os.path.isfile(os.path.join(OUTPUT_DIR, f)) and f.endswith((".bin", ".pt", ".ckpt", ".pth"))
    ]
    resume_epoch = None
    resume_ckpt_path = None
    if len(cur_ckpts) > 0:
        resume_epoch, resume_ckpt_path = get_resume_checkpoint_info(cur_ckpts, configs)
        print(f"Resuming from epoch or step: {resume_epoch}, loading model from {resume_ckpt_path}")

    gen_cls_cfg = getattr(configs, "generation_classification", {}) or {}

    # ========== Build model ==========
    model = build_voxtral_model(
        finetune_mode=finetune_mode,
        base_model_name=VOXTRAL_MODEL,
        dtype=dtype,
        device=DEVICE,
        load_pretrained_base=True,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=TARGET_MODULES,
        use_4bit=USE_4BIT,
        precision=PRECISION,
        resume=resume_epoch,
        load_model_path=resume_ckpt_path,
        language_model_only_lora=LANGUAGE_MODEL_ONLY_LORA,
    )

    encoder_type = ENCODER_TYPE if ENCODER_TYPE not in ("none", "None") else None

    model = VoxtralStageWrapper_classification(
        model,
        encoder_type=encoder_type,
        trainable_audio_encoder=TRAINABLE_AUDIO_ENCODER,
        k_downsample=K_DOWNSAMPLE,
        num_classes=NUM_CLASSES,
        finetune_mode=finetune_mode,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=TARGET_MODULES,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.model.config.use_cache = False

    try:
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.model.gradient_checkpointing_enable()

    # ========== LOAD DATA ==========
    train_dataset_type = "asvspoof2019" if configs.DATASET == "asvspoof2021" else configs.DATASET
    dev_dataset_type = train_dataset_type

    train_dataset = ASVspoofIndexDataset(
        AUGMENTATION_PROB = AUGMENTATION_PROB,
        path_to_protocol=TRAIN_PROTOCOL,
        data_dir=TRAIN_AUDIO_DIR,
        dataset_type=train_dataset_type,
        processor=processor,
        prompts=PROMPTS,
        is_train=True,
        augmentation_on=AUGMENTATION_ON,
        augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
        reasoning=False,
        SPECIAL_TOKEN_TASK=SPECIAL_TOKEN_TASK,
        dtype=dtype,
        CONFIG_AUGMENTATION=CONFIG_AUGMENTATION,
        encoder_type=encoder_type,
        pad_features_mode=PAD_FEATURES_MODE,
        pad_waveform_mode=PAD_WAVEFORM_MODE,
        constant_prompts_flag=CONSTANT_PROMPTS_FLAG,
        three_random_prompts=THREE_RANDOM_PROMPTS_FLAG,
        random_prompt_variants=THREE_RANDOM_PROMPT_VARIANTS,
        N_audio_tokens=N_audio_tokens,
        constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
        label_texts=LABEL_TEXTS,
    )
    if DEBUG:
        print(train_dataset[0])

    dev_dataset = ASVspoofIndexDataset(
        AUGMENTATION_PROB = AUGMENTATION_PROB,
        path_to_protocol=DEV_PROTOCOL,
        data_dir=DEV_AUDIO_DIR,
        dataset_type=dev_dataset_type,
        processor=processor,
        prompts=PROMPTS,
        is_train=False,
        augmentation_on=False,
        augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
        reasoning=False,
        SPECIAL_TOKEN_TASK=SPECIAL_TOKEN_TASK,
        dtype=dtype,
        CONFIG_AUGMENTATION=CONFIG_AUGMENTATION,
        encoder_type=encoder_type,
        pad_features_mode=PAD_FEATURES_MODE,
        pad_waveform_mode=PAD_WAVEFORM_MODE,
        constant_prompts_flag=CONSTANT_PROMPTS_FLAG,
        three_random_prompts=THREE_RANDOM_PROMPTS_FLAG,
        random_prompt_variants=THREE_RANDOM_PROMPT_VARIANTS,
        fixed_prompt_text=EVAL_PROMPT_TEXT,
        N_audio_tokens=N_audio_tokens,
        constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
        label_texts=LABEL_TEXTS,
    )

    eval_dataset = ASVspoofIndexDataset(
        AUGMENTATION_PROB = AUGMENTATION_PROB,
        path_to_protocol=EVAL_PROTOCOL,
        data_dir=EVAL_AUDIO_DIR,
        dataset_type=configs.DATASET,
        processor=processor,
        prompts=PROMPTS,
        is_train=False,
        augmentation_on=False,
        augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
        reasoning=False,
        SPECIAL_TOKEN_TASK=SPECIAL_TOKEN_TASK,
        dtype=dtype,
        CONFIG_AUGMENTATION=CONFIG_AUGMENTATION,
        encoder_type=encoder_type,
        pad_features_mode=PAD_FEATURES_MODE,
        pad_waveform_mode=PAD_WAVEFORM_MODE,
        constant_prompts_flag=CONSTANT_PROMPTS_FLAG,
        three_random_prompts=THREE_RANDOM_PROMPTS_FLAG,
        random_prompt_variants=THREE_RANDOM_PROMPT_VARIANTS,
        fixed_prompt_text=EVAL_PROMPT_TEXT,
        N_audio_tokens=N_audio_tokens,
        constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
        label_texts=LABEL_TEXTS,
    )

    train_eval_dataset = None
    small_train_eval_dataset = None
    if getattr(configs, "eval_on_training_set_during_training", False):
        train_eval_dataset = ASVspoofIndexDataset(
            AUGMENTATION_PROB = AUGMENTATION_PROB,
            path_to_protocol=TRAIN_PROTOCOL,
            data_dir=TRAIN_AUDIO_DIR,
            dataset_type=train_dataset_type,
            processor=processor,
            prompts=PROMPTS,
            is_train=False,
            augmentation_on=False,
            augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
            reasoning=False,
            SPECIAL_TOKEN_TASK=SPECIAL_TOKEN_TASK,
            dtype=dtype,
            CONFIG_AUGMENTATION=CONFIG_AUGMENTATION,
            encoder_type=encoder_type,
            pad_features_mode=PAD_FEATURES_MODE,
            pad_waveform_mode=PAD_WAVEFORM_MODE,
            constant_prompts_flag=CONSTANT_PROMPTS_FLAG,
            three_random_prompts=THREE_RANDOM_PROMPTS_FLAG,
            random_prompt_variants=THREE_RANDOM_PROMPT_VARIANTS,
            fixed_prompt_text=EVAL_PROMPT_TEXT,
            N_audio_tokens=N_audio_tokens,
            constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
            label_texts=LABEL_TEXTS,
        )

    asvspoof2021_eval_dataset = None
    small_asvspoof2021_dataset = None
    if getattr(configs, "eval_on_asvspoof2021_during_training", False):
        if ASVSPOOF2021_EVAL_PROTOCOL is None or ASVSPOOF2021_EVAL_AUDIO_DIR is None:
            raise ValueError("ASVSPOOF2021 evaluation requested but asvspoof2021_eval_protocol or asvspoof2021_eval_audio_dir is missing in the config.")
        asvspoof2021_eval_dataset = ASVspoofIndexDataset(
            AUGMENTATION_PROB = AUGMENTATION_PROB,
            path_to_protocol=ASVSPOOF2021_EVAL_PROTOCOL,
            data_dir=ASVSPOOF2021_EVAL_AUDIO_DIR,
            dataset_type="asvspoof2021",
            processor=processor,
            prompts=PROMPTS,
            is_train=False,
            augmentation_on=False,
            augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
            reasoning=False,
            SPECIAL_TOKEN_TASK=SPECIAL_TOKEN_TASK,
            dtype=dtype,
            CONFIG_AUGMENTATION=CONFIG_AUGMENTATION,
            encoder_type=encoder_type,
            pad_features_mode=PAD_FEATURES_MODE,
            pad_waveform_mode=PAD_WAVEFORM_MODE,
            constant_prompts_flag=CONSTANT_PROMPTS_FLAG,
            three_random_prompts=THREE_RANDOM_PROMPTS_FLAG,
            random_prompt_variants=THREE_RANDOM_PROMPT_VARIANTS,
            fixed_prompt_text=EVAL_PROMPT_TEXT,
            N_audio_tokens=N_audio_tokens,
            constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
            label_texts=LABEL_TEXTS,
        )

    # ========== DEBUGGING ==========
    def create_batch(samples, processor, device, bn_safe=False):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        pad_id = processor.tokenizer.pad_token_id

        batch = {}
        B = len(samples)

        if "input_features" in samples[0]:
            input_features = torch.stack([s["input_features"] for s in samples])
            batch["input_features"] = input_features.to(device)

        if "input_ids" in samples[0]:
            input_ids = [s["input_ids"] for s in samples]
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
            batch["input_ids"] = input_ids_padded.to(device)

        if "attention_mask" in samples[0]:
            attention_mask = [s["attention_mask"] for s in samples]
            attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
            batch["attention_mask"] = attention_mask_padded.to(device)

        if "labels" in samples[0]:
            labels_list = [s["labels"] for s in samples]
            labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=pad_id)
            labels_padded = labels_padded.to(device)
            labels_padded[labels_padded == pad_id] = -100
            batch["labels"] = labels_padded

        if "raw_audio_input" in samples[0]:
            raw_audio_inputs = [s["raw_audio_input"] for s in samples]
            raw_audio_inputs = pad_sequence(raw_audio_inputs, batch_first=True, padding_value=0.0)
            batch["raw_audio_input"] = raw_audio_inputs.to(device)

        if "classification_labels" in samples[0]:
            cls_labels = torch.tensor([s["classification_labels"] for s in samples], dtype=torch.long)
            batch["classification_labels"] = cls_labels.to(device)

        if "input_ids" in batch:
            Bpad, L = batch["input_ids"].size()
            position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(Bpad, -1)
            batch["position_ids"] = position_ids

        def maybe_list(key):
            if key in samples[0]:
                batch[key] = [s[key] for s in samples]

        for k in ["file_id", "file_ids", "speaker_id", "speaker_ids",
                  "gender", "genders", "attack_type", "attack_types", "codec", "codecs"]:
            maybe_list(k)

        if bn_safe:
            tensor_keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor)]
            list_keys   = [k for k, v in batch.items() if isinstance(v, list)]
            if len(samples) == 1:
                for k in tensor_keys:
                    x = batch[k]
                    rep = [2] + [1] * (x.dim() - 1)
                    batch[k] = x.repeat(*rep)
                for k in list_keys:
                    lst = batch[k]
                    batch[k] = lst + lst

        return batch

    if DEBUG:
        plot_debug = False
        sample1 = train_dataset[0]
        sample2 = train_dataset[0]
        if plot_debug:
            input_features = sample1["input_features"]
            input_features_np = input_features.squeeze().cpu().numpy()
            img = plt.imshow(input_features_np,origin="lower",aspect="auto",interpolation="nearest",cmap="magma")
            plt.colorbar(img, fraction=0.046, pad=0.04)
            plt.ylabel("Frequency bins")
            plt.xlabel("Frames")
            plt.title("Spectrogram debug")
            plt.tight_layout()
            plt.savefig("debug_sample0_spectrogram.png", dpi=300)
            plt.close()
        batch = create_batch([sample1,sample2], processor=processor, device=DEVICE, bn_safe=False)
        decoded = processor.tokenizer.batch_decode(batch["input_ids"].cpu(), skip_special_tokens=False)
        for i, txt in enumerate(decoded):
            print(f"[DEBUG] Decoded input_ids batch[{i}]: {txt}")

        try:
            outputs = model(
                input_ids=batch.get("input_ids"),
                input_features=batch.get("input_features"),
                attention_mask=batch.get("attention_mask"),
                position_ids=batch.get("position_ids"),
                raw_audio_input=batch.get("raw_audio_input"),
            )
            print("Forward pass successful!")
        except Exception as e:
            print(f"Forward CAG debug failed: {e}")

    if DEBUG:
        EVAL_BATCH_SIZE = 2
        small_dev_dataset = Subset(dev_dataset, list(range(EVAL_BATCH_SIZE)))
        small_eval_dataset = Subset(dev_dataset, list(range(EVAL_BATCH_SIZE)))
        if train_eval_dataset is not None:
            subset_len = min(EVAL_BATCH_SIZE, len(train_eval_dataset))
            small_train_eval_dataset = Subset(train_eval_dataset, list(range(subset_len)))
        if asvspoof2021_eval_dataset is not None:
            subset_len = min(EVAL_BATCH_SIZE, len(asvspoof2021_eval_dataset))
            small_asvspoof2021_dataset = Subset(asvspoof2021_eval_dataset, list(range(subset_len)))
        print("DEBUG MODE: Using smaller batch size and fewer epochs for quick testing.")

    # =========== Dataloaders =============
    if not DEBUG:
        eval_batch_size_cfg = getattr(configs, "eval_batch_size", None)
        if eval_batch_size_cfg is None or eval_batch_size_cfg <= 0:
            eval_batch_size_cfg = BATCH_SIZE * 2
        EVAL_BATCH_SIZE = max(1, int(eval_batch_size_cfg))
    print(f"Eval batch size set to {EVAL_BATCH_SIZE} (train batch size {BATCH_SIZE})")

    train_loader = make_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        ensure_bonafide_in_batch=ENSURE_BONAFIDE_IN_BATCH,
        balance_bonafide_spoof=BALANCE_BONAFIDE_SPOOF,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    dev_loader = make_dataloader(
        small_dev_dataset if DEBUG else dev_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        ensure_bonafide_in_batch=False,
        balance_bonafide_spoof=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    eval_loader = make_dataloader(
        small_eval_dataset if DEBUG else eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        ensure_bonafide_in_batch=False,
        balance_bonafide_spoof=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    train_eval_loader = None
    if train_eval_dataset is not None:
        dataset_for_loader = small_train_eval_dataset if DEBUG and small_train_eval_dataset is not None else train_eval_dataset
        train_eval_loader = make_dataloader(
            dataset_for_loader,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            ensure_bonafide_in_batch=False,
            balance_bonafide_spoof=False,
            drop_last=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=collate_fn,
        )

    asvspoof2021_eval_loader = None
    if asvspoof2021_eval_dataset is not None:
        dataset_for_loader = small_asvspoof2021_dataset if DEBUG and small_asvspoof2021_dataset is not None else asvspoof2021_eval_dataset
        asvspoof2021_eval_loader = make_dataloader(
            dataset_for_loader,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            ensure_bonafide_in_batch=False,
            balance_bonafide_spoof=False,
            drop_last=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=collate_fn,
        )

    if train_eval_loader is None:
        train_eval_log_file = None

    # ========== Optimizer ===========
    optimizer = get_optimizer(
        model,
        lr=configs.lr,
        weight_decay=configs.weight_decay
    )

    # ========== Scheduler ==========
    grad_accum_steps = gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
    print("Steps per epoch:", len(train_loader))
    print("Effective steps per epoch with grad-accum:", len(train_loader) // grad_accum_steps)
    if RUNNING_BY_EPOCHS:
        total_train_steps = steps_per_epoch * NUM_EPOCHS
    else:
        if MAX_TRAIN_STEPS is None:
            raise ValueError("When train_by == 'steps', you must set max_train_steps in the YAML.")
        total_train_steps = MAX_TRAIN_STEPS
    warmup_steps = int(total_train_steps * WARMUP_RATIO)
    scheduler = get_scheduler(
        optimizer=optimizer,
        num_training_steps=total_train_steps,
        warmup_steps=warmup_steps,
    )

    GENERATION_CLASSIFICATION_CFG = getattr(configs, "generation_classification", {}) or {}
    use_cag_mode = GENERATION_CLASSIFICATION_CFG.get("enable", True)

    loss_fn = None
    class_weights = None
    cag_use_weighted = bool(GENERATION_CLASSIFICATION_CFG.get("use_weighted_loss", False))
    USE_WEIGHTED_LOSS = cag_use_weighted
    class_weights_cfg = getattr(configs, "class_weights", None)
    class_weights_source = "none"
    if cag_use_weighted:
        if isinstance(class_weights_cfg, dict):
            bona_w = class_weights_cfg.get("bonafide")
            spoof_w = class_weights_cfg.get("spoof")
            if bona_w is not None and spoof_w is not None:
                try:
                    bona_w = float(bona_w)
                    spoof_w = float(spoof_w)
                    class_weights = torch.tensor([bona_w, spoof_w], device=DEVICE, dtype=torch.float)
                    class_weights_source = "yaml"
                    print(f"[CAG] Using user-provided class weights: bonafide={bona_w}, spoof={spoof_w}")
                except Exception as exc:
                    print(f"[CAG] Invalid class_weights override ({exc}); falling back to auto-computed weights.")
        if class_weights is None:
            class_weights = compute_class_weights_from_dataset(train_dataset, DEVICE)
            class_weights_source = "auto"
            print(f"[CAG] Class weights enabled (auto-computed):", class_weights)

    def _flatten_cfg(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {k: _flatten_cfg(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_flatten_cfg(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return {k: _flatten_cfg(v) for k, v in obj.__dict__.items()}
        return str(obj)

    _loss_type = "lm_cag" if use_cag_mode else "cross_entropy"
    _loss_params = {
        "loss_weight": GENERATION_CLASSIFICATION_CFG.get("loss_weight", 1.0),
        "use_weighted_loss": USE_WEIGHTED_LOSS,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "class_weights_source": class_weights_source,
    } if use_cag_mode else {}
    print_config_summary(configs, _loss_type, _loss_params)
    print("[Config Dump] Effective configuration before training:")
    try:
        print(json.dumps(_flatten_cfg(configs), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"[Config Dump] Failed to serialize configs: {exc}")

    use_fp16 = (PRECISION == "float16")
    use_bf16 = (PRECISION == "bfloat16")
    use_amp = (PRECISION == "AMP")

    scaler = torch.amp.GradScaler(enabled=(use_amp or use_fp16))

    loss_log_type = _loss_type
    loss_log_params = _loss_params
    log_run_settings(loss_log_type, loss_log_params)

    training_function = train_classification_model
    evaluate_epoch = evaluate_epoch_classification_as_generation

    save_all_configs(
        OUTPUT_DIR,
        configs,
        module_path="utils.config",
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    wandb_run = None
    wandb_cfg = getattr(configs, "wandb", {}) or {}
    wandb_enabled = wandb_cfg.get("enable") if isinstance(wandb_cfg, dict) else getattr(wandb_cfg, "enable", True)
    wandb_enabled = True if wandb_enabled is None else bool(wandb_enabled)

    # ======= Resume training =======
    start_epoch = 0
    start_step = 0
    best_eer = 100.0
    resume_eval_before_train = False
    last_eval_epoch = None
    last_eval_step = None
    resume_dev_eval_needed = False
    resume_test_eval_needed = False
    resume_train_eval_needed = False
    resume_asvspoof2021_eval_needed = False
    last_test_eval_epoch = None
    last_test_eval_step = None
    last_train_eval_epoch = None
    last_train_eval_step = None
    last_asvspoof2021_eval_epoch = None
    last_asvspoof2021_eval_step = None

    resume_state = load_checkpoint_if_available(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        configs=configs,
        wandb_enabled=wandb_enabled,
        wandb_run=wandb_run,
        resume_ckpt_path=resume_ckpt_path,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        steps_per_epoch=steps_per_epoch,
        start_epoch=start_epoch,
        start_step=start_step,
        best_eer=best_eer,
        last_eval_epoch=last_eval_epoch,
        last_eval_step=last_eval_step,
        last_train_eval_epoch=last_train_eval_epoch,
        last_train_eval_step=last_train_eval_step,
        last_test_eval_epoch=last_test_eval_epoch,
        last_test_eval_step=last_test_eval_step,
        last_asvspoof2021_eval_epoch=last_asvspoof2021_eval_epoch,
        last_asvspoof2021_eval_step=last_asvspoof2021_eval_step,
    )

    ckpt_path = resume_state["ckpt_path"]
    wandb_run = resume_state["wandb_run"]
    start_epoch = resume_state["start_epoch"]
    start_step = resume_state["start_step"]
    best_eer = resume_state["best_eer"]
    last_eval_epoch = resume_state["last_eval_epoch"]
    last_eval_step = resume_state["last_eval_step"]
    last_train_eval_epoch = resume_state["last_train_eval_epoch"]
    last_train_eval_step = resume_state["last_train_eval_step"]
    last_test_eval_epoch = resume_state["last_test_eval_epoch"]
    last_test_eval_step = resume_state["last_test_eval_step"]
    last_asvspoof2021_eval_epoch = resume_state["last_asvspoof2021_eval_epoch"]
    last_asvspoof2021_eval_step = resume_state["last_asvspoof2021_eval_step"]

    def _to_int_or_none(val):
        try:
            return int(val)
        except Exception:
            return None

    last_eval_epoch = _to_int_or_none(last_eval_epoch)
    last_eval_step = _to_int_or_none(last_eval_step)
    last_train_eval_epoch = _to_int_or_none(last_train_eval_epoch)
    last_train_eval_step = _to_int_or_none(last_train_eval_step)
    last_test_eval_epoch = _to_int_or_none(last_test_eval_epoch)
    last_test_eval_step = _to_int_or_none(last_test_eval_step)
    last_asvspoof2021_eval_epoch = _to_int_or_none(last_asvspoof2021_eval_epoch)
    last_asvspoof2021_eval_step = _to_int_or_none(last_asvspoof2021_eval_step)

    if ckpt_path is not None and not configs.only_eval:
        if RUNNING_BY_EPOCHS:
            if EVAL_EVERY and EVAL_EVERY > 0 and start_epoch > 0 and (start_epoch % EVAL_EVERY == 0):
                resume_dev_eval_needed = (last_eval_epoch is None or last_eval_epoch < start_epoch)
                if getattr(configs, "eval_on_training_set_during_training", False) and train_eval_loader is not None:
                    resume_train_eval_needed = (last_train_eval_epoch is None or last_train_eval_epoch < start_epoch)
                if getattr(configs, "eval_on_test_during_training", False):
                    resume_test_eval_needed = (last_test_eval_epoch is None or last_test_eval_epoch < start_epoch)
                if getattr(configs, "eval_on_asvspoof2021_during_training", False) and asvspoof2021_eval_loader is not None:
                    resume_asvspoof2021_eval_needed = (last_asvspoof2021_eval_epoch is None or last_asvspoof2021_eval_epoch < start_epoch)
                resume_eval_before_train = (
                    resume_dev_eval_needed
                    or resume_train_eval_needed
                    or resume_test_eval_needed
                    or resume_asvspoof2021_eval_needed
                )
        else:
            if EVAL_EVERY_STEPS and EVAL_EVERY_STEPS > 0 and start_step > 0 and (start_step % EVAL_EVERY_STEPS == 0):
                resume_dev_eval_needed = (last_eval_step is None or last_eval_step < start_step)
                if getattr(configs, "eval_on_training_set_during_training", False) and train_eval_loader is not None:
                    resume_train_eval_needed = (last_train_eval_step is None or last_train_eval_step < start_step)
                if getattr(configs, "eval_on_test_during_training", False):
                    resume_test_eval_needed = (last_test_eval_step is None or last_test_eval_step < start_step)
                if getattr(configs, "eval_on_asvspoof2021_during_training", False) and asvspoof2021_eval_loader is not None:
                    resume_asvspoof2021_eval_needed = (last_asvspoof2021_eval_step is None or last_asvspoof2021_eval_step < start_step)
                resume_eval_before_train = (
                    resume_dev_eval_needed
                    or resume_train_eval_needed
                    or resume_test_eval_needed
                    or resume_asvspoof2021_eval_needed
                )
        if resume_eval_before_train:
            pending_desc = f"epoch {start_epoch}" if RUNNING_BY_EPOCHS else f"step {start_step}"
            print(f"[Resume] Evaluation pending for {pending_desc}; will run before continuing training.")

    model = model.to(DEVICE)

    if wandb_run is None and wandb_enabled:
        wandb_run = init_wandb(
            configs,
            project=configs.wandb.project,
            run_name=configs.wandb.name,
            resume=bool(getattr(configs, "resume", False)),
            wandb_cfg=configs.wandb,
        )

    watch_log = wandb_cfg.get("watch_log", "none") if isinstance(wandb_cfg, dict) else getattr(wandb_cfg, "watch_log", "none")
    watch_log_freq = wandb_cfg.get("watch_log_freq") if isinstance(wandb_cfg, dict) else getattr(wandb_cfg, "watch_log_freq", None)
    log_freq = watch_log_freq if watch_log_freq is not None else getattr(configs, "log_freq", LOG_EVERY)
    if wandb_run and watch_log not in (None, "", "none") and log_freq and log_freq > 0:
        wandb_run.watch(model, log=watch_log, log_freq=log_freq)

    initial_eval_steps = getattr(configs, "initial_eval_steps", None)
    initial_eval_steps_eval = getattr(configs, "initial_eval_steps_eval", None)

    base_train_kwargs = dict(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        eval_loader=eval_loader,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        DEVICE=DEVICE,
        NUM_EPOCHS=NUM_EPOCHS,
        OUTPUT_DIR=OUTPUT_DIR,
        LOG_EVERY=LOG_EVERY,
        EVAL_EVERY=EVAL_EVERY,
        SAVE_EVERY=SAVE_EVERY,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        use_amp=use_amp,
        scaler=scaler,
        max_grad_norm=max_grad_norm,
        accum=gradient_accumulation_steps,
        result_file=result_file,
        metrics_log_file=metrics_log_file,
        eval_log_file=eval_log_file,
        evaluate_epoch=evaluate_epoch,
        log_metrics=log_metrics,
        configs=configs,
        config_dict=config_dict,
        start_epoch=start_epoch,
        best_eer=best_eer,
        wandb_run=wandb_run,
        enable_debug_logging=getattr(configs, "debug_logging", False),
        generation_cls_cfg=GENERATION_CLASSIFICATION_CFG,
    )

    if training_function is train_classification_model:
        base_train_kwargs.update(
            run_by_epochs=RUNNING_BY_EPOCHS,
            max_train_steps=MAX_TRAIN_STEPS,
            eval_every_steps=EVAL_EVERY_STEPS,
            save_every_steps=SAVE_EVERY_STEPS,
            start_step=start_step,
            log_eval_examples=getattr(configs, "log_eval_examples", True),
            save_lr_and_grad_norm_debug=getattr(configs, "save_lr_and_grad_norm_debug", False),
            resume_eval_before_train=resume_eval_before_train,
            last_eval_epoch=last_eval_epoch,
            last_eval_step=last_eval_step,
            train_eval_loader=train_eval_loader,
            train_eval_log_file=train_eval_log_file,
            last_train_eval_epoch=last_train_eval_epoch,
            last_train_eval_step=last_train_eval_step,
            resume_dev_eval_needed=resume_dev_eval_needed,
            resume_test_eval_needed=resume_test_eval_needed,
            resume_train_eval_needed=resume_train_eval_needed,
            last_test_eval_epoch=last_test_eval_epoch,
            last_test_eval_step=last_test_eval_step,
            asvspoof2021_eval_loader=asvspoof2021_eval_loader,
            asvspoof2021_eval_log_file=asvspoof2021_eval_log_file,
            last_asvspoof2021_eval_epoch=last_asvspoof2021_eval_epoch,
            last_asvspoof2021_eval_step=last_asvspoof2021_eval_step,
            resume_asvspoof2021_eval_needed=resume_asvspoof2021_eval_needed,
            use_weighted_loss=USE_WEIGHTED_LOSS,
            class_weights=class_weights,
            initial_eval_steps=initial_eval_steps,
            initial_eval_steps_eval=initial_eval_steps_eval,
        )

    print("Trainable layers:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    print("Trainable layers finish")

    training_function(**base_train_kwargs)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
