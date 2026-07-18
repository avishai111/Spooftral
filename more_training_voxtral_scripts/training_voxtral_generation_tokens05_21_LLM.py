"""
training_voxtral_generation_reasoning_tokens05_21_LLM.py
=========================================================
CAG training script for ASVspoof5 with margin variant 21 and LLM-only adapter tuning.

Same as training_voxtral_generation_reasoning_tokens05_21.py but configured via
hyperparameters_05_dif_margins21_LLM.yaml, which applies PEFT adapters exclusively
to the language model (language_model_only_lora=True) and keeps the audio encoder frozen.
"""
import json
import logging
import os

# Help avoid CUDA fragmentation on small GPUs (must be set before CUDA allocs)
#os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Resolve config path from env, falling back to configs/ directory (new structure)
_script_dir = os.path.dirname(__file__)
_config_name = "hyperparameters_05_dif_margins21_LLM.yaml"
_config_in_configs = os.path.join(_script_dir, "configs", _config_name)
_config_legacy = os.path.join(_script_dir, _config_name)
_default_cfg_path = _config_in_configs if os.path.exists(_config_in_configs) else _config_legacy
os.environ["VOXTRAL_CONFIG"] = os.environ.get("VOXTRAL_CONFIG", _default_cfg_path)

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from torch.utils.data import Subset
# Core HF building blocks used throughout training and evaluation
# Surface key library version for debugging reproducibility
print(transformers.__version__)
from utils.helper_function import (
    get_resume_checkpoint_info,
    log_metrics,
    bump_last_eval_epoch_from_log,
    get_best_eer_from_log,
    add_tekken_once_filter,
    Config as HFConfig,
)
# Only keep the first Tekken tokenizer log per process to avoid spam
add_tekken_once_filter()
from utils.build_model import (
    build_voxtral_model,
    VoxtralStageWrapper_classification,
    compute_class_weights_from_dataset,
    load_checkpoint_into_model,
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
# Registers the project's Config class as an allowed type for PyTorch's safe
# (weights_only) checkpoint loading, since saved checkpoints embed a Config object.
torch.serialization.add_safe_globals([HFConfig])
import warnings
# Limits the routine torchcodec informational warning to a single print instead
warnings.filterwarnings(
    "once",
    message=".*torchcodec.*"
)

N_audio_tokens = N_AUDIO_TOKENS

print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
# Enable TensorFloat-32 matmul/cuDNN kernels for faster float32 compute on
# supported GPUs, at a small cost to numerical precision.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ========== Reproducibility (H4) ==========
_SEED = int(getattr(configs, "seed", 42))
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)
print(f"[Reproducibility] Global seed set to {_SEED}")

# ========== LOGGING: send logs to console + file ==========
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


# ========== Create dir ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Worker Init Function ==========
def worker_init_fn(worker_id):
    """
    README: Seed NumPy/random for each DataLoader worker.
    - Derives worker_seed from torch.initial_seed and seeds numpy/random for reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# Each evaluation split (dev, held-out eval, train-set-as-eval, ASVspoof2021) gets
# its own metrics log file, which also lets resume logic track progress per split.
result_file = os.path.join(OUTPUT_DIR, "train_log.txt")
metrics_log_file = os.path.join(OUTPUT_DIR, "metrics_log_dev.txt")
eval_log_file = os.path.join(OUTPUT_DIR, "metrics_log_eval.txt")  # new file just for eval-only logs
train_eval_log_file = os.path.join(OUTPUT_DIR, "metrics_train_log.txt")
asvspoof2021_eval_log_file = os.path.join(OUTPUT_DIR, "metrics_log_asvspoof2021.txt")


# ========== Resume if possible ==========
# Check for existing checkpoints and auto-select latest for resume
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

# Build base Voxtral with LoRA/DoRA/QLoRA and wrap for classification-as-generation
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
    language_model_only_lora=LANGUAGE_MODEL_ONLY_LORA,
)

# Normalize encoder selection from config (treat "none"/None as no external encoder)
encoder_type = ENCODER_TYPE if ENCODER_TYPE not in ("none", "None") else None


# Attach classification wrapper and optional external encoder projection
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
    projector_peft=PROJECTOR_PEFT,
)
if resume_ckpt_path:
    load_checkpoint_into_model(model, resume_ckpt_path, device=DEVICE)
    print(f"Resumed from checkpoint: {resume_ckpt_path}")

    
# ==== Define special tokens if not already defined ====
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
# KV-caching only helps autoregressive generation, not the forward/backward passes
# used during training, so it is turned off here.
model.model.config.use_cache = False

if GRADIENT_CHECKPOINTING:
    try:
        # Gradient checkpointing trades extra recomputation in the backward pass for lower
        # activation memory; the fallback below supports older Transformers versions.
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.model.gradient_checkpointing_enable()
print(f"[Config] gradient_checkpointing: {GRADIENT_CHECKPOINTING}")

# # ========== LOAD DATA ==========

# Route train/dev through 2019 protocols when running the 2021 eval setup.
train_dataset_type = "asvspoof2019" if configs.DATASET == "asvspoof2021" else configs.DATASET
dev_dataset_type = train_dataset_type

try:
    dataset_aug_prob = float(AUGMENTATION_PROB)
except (TypeError, ValueError) as exc:
    raise ValueError(
        f"AUGMENTATION_PROB must be numeric in [0, 1], got: {AUGMENTATION_PROB!r}"
    ) from exc
if not (0.0 <= dataset_aug_prob <= 1.0):
    logger.warning(
        "AUGMENTATION_PROB=%s is outside [0, 1]. Clamping for dataset sampling.",
        dataset_aug_prob,
    )
    dataset_aug_prob = max(0.0, min(1.0, dataset_aug_prob))

train_dataset = ASVspoofIndexDataset(
    path_to_protocol=TRAIN_PROTOCOL,
    data_dir=TRAIN_AUDIO_DIR,
    dataset_type=train_dataset_type,
    processor=processor,
    prompts=PROMPTS,
    AUGMENTATION_PROB=dataset_aug_prob,
    is_train=True,
    augmentation_on=AUGMENTATION_ON,
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
    augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
    constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
    label_texts=LABEL_TEXTS,
)
if DEBUG:
    print(train_dataset[0])
# Dev follows the same dataset as train (2019 when 2021 eval is active)
dev_dataset = ASVspoofIndexDataset(
    path_to_protocol=DEV_PROTOCOL,
    data_dir=DEV_AUDIO_DIR,
    dataset_type=dev_dataset_type,
    processor=processor,
    prompts=PROMPTS,
    AUGMENTATION_PROB=dataset_aug_prob,
    is_train=False,
    augmentation_on=False,
    reasoning=False,
    SPECIAL_TOKEN_TASK=SPECIAL_TOKEN_TASK,
    dtype=dtype,
    CONFIG_AUGMENTATION=CONFIG_AUGMENTATION,
    encoder_type=encoder_type,
    pad_features_mode=PAD_FEATURES_MODE,
    pad_waveform_mode=PAD_WAVEFORM_MODE,
    constant_prompts_flag=True,
    three_random_prompts=False,
    random_prompt_variants=THREE_RANDOM_PROMPT_VARIANTS,
    N_audio_tokens=N_audio_tokens,
    augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
    constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
    label_texts=LABEL_TEXTS,
)
# Eval remains on the configured dataset (2021 when requested)
eval_dataset = ASVspoofIndexDataset(
    path_to_protocol=EVAL_PROTOCOL,
    data_dir=EVAL_AUDIO_DIR,
    dataset_type=configs.DATASET,
    processor=processor,
    prompts=PROMPTS,
    AUGMENTATION_PROB=dataset_aug_prob,
    is_train=False,
    augmentation_on=False,
    reasoning=False,
    SPECIAL_TOKEN_TASK=SPECIAL_TOKEN_TASK,
    dtype=dtype,
    CONFIG_AUGMENTATION=CONFIG_AUGMENTATION,
    encoder_type=encoder_type,
    pad_features_mode=PAD_FEATURES_MODE,
    pad_waveform_mode=PAD_WAVEFORM_MODE,
    constant_prompts_flag=True,
    three_random_prompts=False,
    random_prompt_variants=THREE_RANDOM_PROMPT_VARIANTS,
    N_audio_tokens=N_audio_tokens,
    augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
    constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
    label_texts=LABEL_TEXTS,
)

train_eval_dataset = None
small_train_eval_dataset = None
if getattr(configs, "eval_on_training_set_during_training", False):
    train_eval_dataset = ASVspoofIndexDataset(
        path_to_protocol=TRAIN_PROTOCOL,
        data_dir=TRAIN_AUDIO_DIR,
        dataset_type=train_dataset_type,
        processor=processor,
        prompts=PROMPTS,
        AUGMENTATION_PROB=dataset_aug_prob,
        is_train=False,
        augmentation_on=False,
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
        augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
        constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
        label_texts=LABEL_TEXTS,
    )

asvspoof2021_eval_dataset = None
small_asvspoof2021_dataset = None
if getattr(configs, "eval_on_asvspoof2021_during_training", False):
    if ASVSPOOF2021_EVAL_PROTOCOL is None or ASVSPOOF2021_EVAL_AUDIO_DIR is None:
        raise ValueError("ASVSPOOF2021 evaluation requested but asvspoof2021_eval_protocol or asvspoof2021_eval_audio_dir is missing in the config.")
    asvspoof2021_eval_dataset = ASVspoofIndexDataset(
        path_to_protocol=ASVSPOOF2021_EVAL_PROTOCOL,
        data_dir=ASVSPOOF2021_EVAL_AUDIO_DIR,
        dataset_type="asvspoof2021",
        processor=processor,
        prompts=PROMPTS,
        AUGMENTATION_PROB=dataset_aug_prob,
        is_train=False,
        augmentation_on=False,
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
        augmentation_only_spoof=AUGMENTATION_ONLY_SPOOF,
        constant_prompt_override=CONSTANT_PROMPT_OVERRIDE,
        label_texts=LABEL_TEXTS,
    )


# ========== DEBUGGING: single forward pass ==========
    # Create batch by stacking tensors

def create_batch(samples, processor, device, bn_safe=False):
    """
    Build a padded batch dict from dataset samples.

    Pads variable-length fields (input_ids, attention_mask, labels,
    raw_audio_input) to the longest sample in the batch. Fixed-shape tensors
    (input_features) are stacked directly. Tensor data is moved to device;
    metadata lists (file_id, attack_type, …) stay on CPU as Python lists.

    Labels padding note: labels are padded with -100 (the CE ignore index)
    rather than pad_token_id. When pad_token_id == eos_token_id — which is
    the case when pad_token is set to eos_token — padding with pad_token_id
    and then masking with `labels[labels == pad_id] = -100` would corrupt
    every real EOS token inside the sequence. Direct -100 padding avoids
    this collision entirely.

    Args:
        samples  : list of dataset sample dicts.
        processor: HF processor whose tokenizer provides pad_token_id.
        device   : torch.device to move tensors to.
        bn_safe  : if True and batch_size == 1, duplicate the sample along
                   dim=0 so BatchNorm layers do not crash on a size-1 batch.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence
    pad_id = processor.tokenizer.pad_token_id

    batch = {}
    B = len(samples)

    # ---- tensors that are already same shape → stack
    if "input_features" in samples[0]:
        input_features = torch.stack([s["input_features"] for s in samples])  # (B, ...)
        batch["input_features"] = input_features.to(device)

    # ---- variable-length (pad to max)
    if "input_ids" in samples[0]:
        input_ids = [s["input_ids"] for s in samples]  # list of (Li,)
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        batch["input_ids"] = input_ids_padded.to(device)

    if "attention_mask" in samples[0]:
        attention_mask = [s["attention_mask"] for s in samples]  # list of (Li,)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        batch["attention_mask"] = attention_mask_padded.to(device)

    if "labels" in samples[0]:
        labels_list = [s["labels"] for s in samples]  # list of (Li,)
        # [FIX] Pad directly with -100 (the CE ignore index) instead of pad_id.
        # When pad_token_id == eos_token_id (set above via pad_token = eos_token),
        # the previous pattern `pad_sequence(..., padding_value=pad_id)` followed by
        # `labels[labels == pad_id] = -100` corrupted every real EOS token inside the
        # sequence. Padding with -100 from the start avoids any such collision.
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        batch["labels"] = labels_padded.to(device)

    if "raw_audio_input" in samples[0]:
        raw_audio_inputs = [s["raw_audio_input"] for s in samples]  # list of (Ti,)
        raw_audio_inputs = pad_sequence(raw_audio_inputs, batch_first=True, padding_value=0.0)
        batch["raw_audio_input"] = raw_audio_inputs.to(device)

    # ---- classification labels (scalars)
    if "classification_labels" in samples[0]:
        cls_labels = torch.tensor([s["classification_labels"] for s in samples], dtype=torch.long)
        batch["classification_labels"] = cls_labels.to(device)

    # ---- position ids (match input_ids length if present)
    if "input_ids" in batch:
        Bpad, L = batch["input_ids"].size()
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(Bpad, -1)
        batch["position_ids"] = position_ids

    # ---- metadata kept as Python lists (not moved to device)
    def maybe_list(key):
        if key in samples[0]:
            batch[key] = [s[key] for s in samples]

    for k in ["file_id", "file_ids", "speaker_id", "speaker_ids",
              "gender", "genders", "attack_type", "attack_types", "codec", "codecs"]:
        maybe_list(k)

    # ---- (optional) BN-safe duplication when batch size == 1
    if bn_safe:
        # detect tensor fields and duplicate along dim=0
        tensor_keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor)]
        list_keys   = [k for k, v in batch.items() if isinstance(v, list)]
        if len(samples) == 1:
            for k in tensor_keys:
                x = batch[k]
                # repeat along batch dim
                rep = [2] + [1] * (x.dim() - 1)
                batch[k] = x.repeat(*rep)
            for k in list_keys:
                lst = batch[k]
                batch[k] = lst + lst  # duplicate metadata

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
    # Decode text side to verify prompt/input tokenization (keep special tokens to inspect audio placeholders)
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

# use smaller eval batch by default to reduce peak memory; clamp so it never exceeds the train batch size
if not DEBUG:
    # Default eval batch size to double the train batch size, since evaluation has no
    # optimizer/gradient state and so uses less memory per sample.
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
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
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
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn,
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
print("Steps per epoch:", len(train_loader)) #-> (2580+22800)/32 -> 793 STEPS ~ 800STEPS
print("Effective steps per epoch with grad-accum:", len(train_loader) // grad_accum_steps)
if RUNNING_BY_EPOCHS:
    total_train_steps = steps_per_epoch * NUM_EPOCHS
else:
    if MAX_TRAIN_STEPS is None:
        raise ValueError("When train_by == 'steps', you must set max_train_steps in the YAML.")
    total_train_steps = MAX_TRAIN_STEPS
# Linear LR warmup runs for the configured fraction of total training steps before
# the scheduler moves into its decay phase.
warmup_steps = int(total_train_steps * WARMUP_RATIO)
scheduler = get_scheduler(
    optimizer=optimizer,
    num_training_steps=total_train_steps,
    warmup_steps=warmup_steps,
)


# Determine if we're using CAG mode (only supported configuration)
GENERATION_CLASSIFICATION_CFG = getattr(configs, "generation_classification", {}) or {}
use_cag_mode = GENERATION_CLASSIFICATION_CFG.get("enable", True)


# CAG uses standard CrossEntropy by default.
loss_fn = None
class_weights = None
class_weights_source = "none"
# Compute weights for CAG mode if enabled
cag_use_weighted =  bool(GENERATION_CLASSIFICATION_CFG.get("use_weighted_loss", False))
USE_WEIGHTED_LOSS = cag_use_weighted
class_weights_cfg = getattr(configs, "class_weights", None)
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
    """
    README: Flatten config objects/containers for readable dumps.
    - Recursively handles dicts, lists/tuples, objects with __dict__, and primitives.
    - Falls back to str(obj) for unrecognized types.
    """
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
except Exception as exc:  # pragma: no cover
    print(f"[Config Dump] Failed to serialize configs: {exc}")


# ---------- AMP / grad accumulation ----------

use_fp16 = (PRECISION == "float16")
use_bf16 = (PRECISION == "bfloat16")
use_amp = (PRECISION  == "AMP")

# Loss scaling via GradScaler is only needed to avoid gradient underflow in
# float16/AMP; bfloat16's wider dynamic range doesn't need it.
scaler = torch.amp.GradScaler(enabled=(use_amp or use_fp16))

loss_log_type = _loss_type
loss_log_params = _loss_params
log_run_settings(loss_log_type, loss_log_params)

# ======== Train ==========
training_function = train_classification_model
evaluate_epoch = evaluate_epoch_classification_as_generation

# ========== Save The Configs ==========

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


# ======= Resume training: load full checkpoint (model + opt + sched + scaler) =======
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
best_eval_eer = float(resume_state.get("best_eval_eer", 100.0))
last_eval_epoch = resume_state["last_eval_epoch"]
last_eval_step = resume_state["last_eval_step"]
last_train_eval_epoch = resume_state["last_train_eval_epoch"]
last_train_eval_step = resume_state["last_train_eval_step"]
last_test_eval_epoch = resume_state["last_test_eval_epoch"]
last_test_eval_step = resume_state["last_test_eval_step"]
last_asvspoof2021_eval_epoch = resume_state["last_asvspoof2021_eval_epoch"]
last_asvspoof2021_eval_step = resume_state["last_asvspoof2021_eval_step"]

# Determine if an evaluation should run before resuming training
def _to_int_or_none(val):
    """
    README: Best-effort conversion to int, returning None on failure.
    - Helps normalize resume metadata that may be strings or missing.
    """
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

last_eval_epoch = bump_last_eval_epoch_from_log(last_eval_epoch, metrics_log_file, "dev")
last_train_eval_epoch = bump_last_eval_epoch_from_log(last_train_eval_epoch, train_eval_log_file, "train")
last_test_eval_epoch = bump_last_eval_epoch_from_log(last_test_eval_epoch, eval_log_file, "test")
last_asvspoof2021_eval_epoch = bump_last_eval_epoch_from_log(last_asvspoof2021_eval_epoch, asvspoof2021_eval_log_file, "asvspoof2021")

# Correct best_eer / best_eval_eer using the actual minimum from the log files.
# Guards against a stale checkpoint where best_eer was not updated before a
# crash/resume (e.g. saved mid-epoch before eval, or an earlier run overwrote
# the checkpoint with a worse best_eer).
_log_best_dev_eer = get_best_eer_from_log(metrics_log_file)
if _log_best_dev_eer is not None and _log_best_dev_eer < best_eer:
    print(f"[Resume] Correcting best_eer from dev log: {best_eer:.4f} → {_log_best_dev_eer:.4f}")
    best_eer = _log_best_dev_eer

_log_best_eval_eer = get_best_eer_from_log(eval_log_file)
if _log_best_eval_eer is not None and _log_best_eval_eer < best_eval_eer:
    print(f"[Resume] Correcting best_eval_eer from eval log: {best_eval_eer:.4f} → {_log_best_eval_eer:.4f}")
    best_eval_eer = _log_best_eval_eer

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


# ensure model is on device (in case build moved it earlier)
model = model.to(DEVICE)

if wandb_run is None and wandb_enabled:
    wandb_run = init_wandb(
        configs,
        project=configs.wandb.project,
        run_name=configs.wandb.name,
        resume=bool(getattr(configs, "resume", False)),
        wandb_cfg=configs.wandb,
    )

# W&B watch disabled by default to avoid extra overhead; set watch_log in configs.wandb to re-enable
watch_log = wandb_cfg.get("watch_log", "none") if isinstance(wandb_cfg, dict) else getattr(wandb_cfg, "watch_log", "none")
watch_log_freq = wandb_cfg.get("watch_log_freq") if isinstance(wandb_cfg, dict) else getattr(wandb_cfg, "watch_log_freq", None)
log_freq = watch_log_freq if watch_log_freq is not None else getattr(configs, "log_freq", LOG_EVERY)
if wandb_run and watch_log not in (None, "", "none") and log_freq and log_freq > 0:
    wandb_run.watch(model, log=watch_log, log_freq=log_freq)


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
    best_eval_eer=best_eval_eer,  # [FIX] Pass restored best_eval_eer so train fn does not reset it to 100.0
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
        initial_eval_steps=getattr(configs, "initial_eval_steps", None),
        initial_eval_steps_eval=getattr(configs, "initial_eval_steps_eval", None),
    )

print("Trainable layers:\n")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

print("Trainable layers finish")

# Hands off to the CAG training loop in utils.training_evaluate_classification,
# which runs training end-to-end: periodic evaluation, checkpointing, logging.
training_function(**base_train_kwargs)


if wandb_run:
    wandb_run.finish()

    
# ========== OPTIONAL: EER PLOTTING ==========
# If your eval still uses EER or ROC metrics, use your helper functions on model generations.
# plot_single_eer_file("train_metrics.txt", "Train", "train_eer.png")
# plot_single_eer_file("dev_metrics.txt", "Dev", "dev_eer.png")
# plot_single_eer_file("eval_metrics.txt", "Eval", "eval_eer.png")
