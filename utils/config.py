"""
utils/config.py
===============
Module-level configuration loader for all Voxtral training runs.

Responsibilities
----------------
- Reads the YAML pointed to by VOXTRAL_CONFIG env var and exposes every
  hyperparameter as a module-level constant (star-imported by training scripts).
- Validates critical invariants at import time (label ordering, projector_peft
  compatibility, language_model_only_lora consistency) so errors surface before
  any GPU/W&B initialisation.
- Builds the WaveformAugmentationConfig (CONFIG_AUGMENTATION) and tokeniser/
  processor objects shared across dataset workers.
- Provides get_optimizer() and get_scheduler() factory helpers.

Import contract
---------------
  Training scripts MUST set os.environ["VOXTRAL_CONFIG"] before importing this
  module.  Importing without it raises ValueError immediately.
"""
import dataclasses
import json
import math
import os
import random
import uuid
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import torch
import transformers
import wandb
import yaml
from typing import Any
from dotenv import load_dotenv
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoProcessor
try:
    from config_helper_functions import save_all_configs, extract_label_texts_from_gen_cfg
    from augmentation import augment_waveform, WaveformAugmentationConfig, augment_spectrogram
    from helper_function import (
        compute_eer, extract_label_from_output,Config,
        compute_prob_for_sequence, append_metrics_to_file, pad_mel_spec, 
        plot_single_eer_file, get_resume_checkpoint_info,load_protocol_ASVspoof5,autocast_ctx ,log_metrics, load_text_prompts

    )
    
    from build_model import build_voxtral_model

    
except ImportError:
    from utils.config_helper_functions import save_all_configs, extract_label_texts_from_gen_cfg
    from utils.augmentation import augment_waveform, WaveformAugmentationConfig, augment_spectrogram
    from utils.helper_function import (
        compute_eer, extract_label_from_output,Config,
        compute_prob_for_sequence, append_metrics_to_file, pad_mel_spec, 
        plot_single_eer_file, get_resume_checkpoint_info,load_protocol_ASVspoof5,autocast_ctx ,log_metrics, load_text_prompts

    )
    
    from utils.build_model import build_voxtral_model

    
from huggingface_hub import login

# Load optional .env (honors VOXTRAL_DOTENV or falls back to project root)
DOTENV_PATH = Path(os.environ.get("VOXTRAL_DOTENV", Path(__file__).resolve().parent.parent / ".env"))
if DOTENV_PATH.exists():
    # override=False: an already-exported shell/CI environment variable wins over .env.
    load_dotenv(DOTENV_PATH, override=False)

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    _hf_offline = os.environ.get("HF_HUB_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE", "0")).strip().lower() in ("1", "true")
    if _hf_offline:
        print("HF_HUB_OFFLINE is set; skipping HuggingFace login (model already cached).")
    else:
        login(hf_token)
else:
    print("Warning: HF_TOKEN not set; skipping HuggingFace login.")

WANDB_API_KEY = os.getenv("WANDB_API_KEY")

def _to_plain_dict(cfg):
    """
    Convert a config-like object into a plain Python dict.
    - Supports dataclasses, OmegaConf containers, Config objects, and plain dicts/objects.
    - Recurses through __dict__ when available; falls back to returning cfg unchanged.
    """
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return {k: _to_plain_dict(v) for k, v in cfg.__dict__.items()}
    return cfg  # fallback


def print_default(name: str, value: Any) -> None:
    """Print a message when a default value is used."""
    print(f"[DEFAULT] {name} not in YAML -> using default {value}")

def init_wandb(cfg, project, run_name=None, resume=False, wandb_cfg=None):
    """
    Initialize Weights & Biases with stable wandb_id for resume.
    - Serializes cfg to dict (via _to_plain_dict) and ensures wandb_id is stored on cfg/checkpoints.
    - Honors resume flag/policy and optional wandb_cfg (entity/dir/mode/resume).
    - Returns the initialized wandb run handle.
    """
    cfg_dict = _to_plain_dict(cfg)
    # reuse wandb_id if present and resume requested; otherwise create new
    wandb_id = getattr(cfg, "wandb_id", None) or cfg_dict.get("wandb_id")
    if resume and wandb_id is None:
        wandb_id = str(uuid.uuid4())
    if not resume:
        wandb_id = str(uuid.uuid4())
    # store back on cfg so it’s saved in checkpoints
    setattr(cfg, "wandb_id", wandb_id)
    cfg_dict["wandb_id"] = wandb_id

    entity = getattr(wandb_cfg, "entity", None) if wandb_cfg is not None else None
    mode = getattr(wandb_cfg, "mode", None) if wandb_cfg is not None else None
    dir_path = getattr(wandb_cfg, "dir", None) if wandb_cfg is not None else None
    resume_policy = getattr(wandb_cfg, "resume", None) if wandb_cfg is not None else None
    if resume and resume_policy is None:
        resume_policy = "allow"

    run = wandb.init(
        project=project,
        name=run_name,
        config=cfg_dict,
        id=wandb_id,
        resume=resume_policy,
        entity=entity,
        mode=mode,
        dir=dir_path,
    )
    return run


# ========== loading yaml config ==========
# Config path must be set by the training script via VOXTRAL_CONFIG environment variable
config_path = os.environ.get("VOXTRAL_CONFIG")
if config_path is None:
    raise ValueError(
        "VOXTRAL_CONFIG environment variable must be set before importing utils.config. "
        "Set it in your training script before any imports."
    )
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)

# YAML represents Python None as the string "None" when quoting is inconsistent; normalize to actual None.
for k, v in config_dict.items():
    if isinstance(v, str) and v.strip() == "None":
        config_dict[k] = None

# Backfill flat ASVSpoof paths from nested dual-encoder config if needed.
asvspoof_cfg = config_dict.get("asvspoof_dataset")
if isinstance(asvspoof_cfg, dict):
    dataset_name = config_dict.get("DATASET") or asvspoof_cfg.get("version")
    if dataset_name:
        config_dict.setdefault("DATASET", dataset_name)
        dataset_paths = asvspoof_cfg.get(dataset_name, {}) or {}
        for key in (
            "train_protocol",
            "dev_protocol",
            "eval_protocol",
            "train_audio_dir",
            "dev_audio_dir",
            "eval_audio_dir",
        ):
            if key in dataset_paths:
                config_dict.setdefault(key, dataset_paths.get(key))
        if dataset_name == "asvspoof2021":
            config_dict.setdefault("asvspoof2021_eval_protocol", dataset_paths.get("eval_protocol"))
            config_dict.setdefault("asvspoof2021_eval_audio_dir", dataset_paths.get("eval_audio_dir"))

configs = Config(config_dict)

# Force supported configuration: classification with CAG
DATASET = configs.DATASET #"asvspoof5" # or "asvspoof5" or "custom"
# ========== SEEDING ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = configs.debug
print(f"Debug mode: {DEBUG}")
print(f"Running on device: {DEVICE}")
USE_4BIT = configs.use_4bit

OPTIMIZER_NAME = configs.optimizer
BETAS = tuple(configs.betas)   # make sure it's a tuple
EPS = float(configs.eps)
MOMENTUM = float(configs.momentum)

TRAIN_BY = str(getattr(configs, "train_by", "epochs")).lower()
if getattr(configs, "train_by", None) is None:
    print_default("train_by", "epochs")
RUNNING_BY_EPOCHS = TRAIN_BY != "steps"
NUM_WORKERS = configs.NUM_WORKERS if not DEBUG else 1
BATCH_SIZE = configs.BATCH_SIZE if not DEBUG else 1
NUM_EPOCHS = configs.NUM_EPOCHS if not DEBUG else 1
MAX_TRAIN_STEPS = getattr(configs, "max_train_steps", None)
if MAX_TRAIN_STEPS is not None:
    try:
        MAX_TRAIN_STEPS = int(MAX_TRAIN_STEPS)
    except Exception:
        raise ValueError(f"Invalid max_train_steps value: {MAX_TRAIN_STEPS}")

# ========== CONFIG ==========
PRECISION = configs.precision
NAME_PRECISION = "4bit" if USE_4BIT else PRECISION
NUM_OF_TOKENS = configs.NUM_OF_TOKENS
N_AUDIO_TOKENS = getattr(configs, "N_audio_tokens", 375)
if getattr(configs, "N_audio_tokens", None) is None:
    print_default("N_audio_tokens", 375)
K_DOWNSAMPLE = getattr(configs, "k_downsample", 4)
if getattr(configs, "k_downsample", None) is None:
    print_default("k_downsample", 4)
# Resolve audio encoder mode placeholder if present (needed for classification-head configs)
audio_encoder_mode = getattr(configs, "audio_encoder_mode", None)
if audio_encoder_mode is None:
    chm = getattr(configs, "classification_head_mode", None)
    if isinstance(chm, dict):
        audio_encoder_mode = chm.get("audio_encoder_mode")
    elif chm is not None and hasattr(chm, "audio_encoder_mode"):
        audio_encoder_mode = getattr(chm, "audio_encoder_mode", None)
if audio_encoder_mode is None:
    audio_encoder_mode = "unknown"
setattr(configs, "audio_encoder_mode", audio_encoder_mode)

_format_vars = {
    "NUM_OF_TOKENS": NUM_OF_TOKENS,
    "DATASET": DATASET,
    "precision": NAME_PRECISION,
    "audio_encoder_mode": audio_encoder_mode,
}
configs.project = configs.project.format(**_format_vars)
configs.name = configs.name.format(**_format_vars)

VOXTRAL_MODEL = configs.model_id
OUTPUT_DIR = os.path.join(configs.save_path, configs.name)


finetune_mode = configs.finetune_mode
TARGET_MODULES = configs.target_modules
# PEFT (LoRA/DoRA) hyperparameters: fall back to sensible defaults when the YAML omits them.
LORA_RANK = int(getattr(configs, "lora_rank", 16) or 16)
if getattr(configs, "lora_rank", None) is None:
    print_default("lora_rank", 16)
LORA_ALPHA = int(getattr(configs, "lora_alpha", 32) or 32)
if getattr(configs, "lora_alpha", None) is None:
    print_default("lora_alpha", 32)
LORA_DROPOUT = float(getattr(configs, "lora_dropout", 0.05) or 0.05)
if getattr(configs, "lora_dropout", None) is None:
    print_default("lora_dropout", 0.05)

TRAIN_PROTOCOL = configs.train_protocol
DEV_PROTOCOL   = configs.dev_protocol
EVAL_PROTOCOL  = configs.eval_protocol

TRAIN_AUDIO_DIR = configs.train_audio_dir
DEV_AUDIO_DIR   = configs.dev_audio_dir
EVAL_AUDIO_DIR  = configs.eval_audio_dir
ASVSPOOF2021_EVAL_PROTOCOL = getattr(configs, "asvspoof2021_eval_protocol", None)
ASVSPOOF2021_EVAL_AUDIO_DIR = getattr(configs, "asvspoof2021_eval_audio_dir", None)
EVAL_ON_TRAINING_SET_DURING_TRAINING = bool(getattr(configs, "eval_on_training_set_during_training", False))
if getattr(configs, "eval_on_training_set_during_training", None) is None:
    print_default("eval_on_training_set_during_training", False)
EVAL_ON_ASVSPOOF2021_DURING_TRAINING = bool(getattr(configs, "eval_on_asvspoof2021_during_training", False))
if getattr(configs, "eval_on_asvspoof2021_during_training", None) is None:
    print_default("eval_on_asvspoof2021_during_training", False)

ENSURE_BONAFIDE_IN_BATCH = bool(getattr(configs, "ensure_bonafide_in_batch", False))
if getattr(configs, "ensure_bonafide_in_batch", None) is None:
    print_default("ensure_bonafide_in_batch", False)
BALANCE_BONAFIDE_SPOOF = bool(getattr(configs, "balance_bonafide_spoof", False))
if getattr(configs, "balance_bonafide_spoof", None) is None:
    print_default("balance_bonafide_spoof", False)

PAD_WAVEFORM_MODE = configs.PAD_WAVEFORM_MODE
PAD_FEATURES_MODE = configs.PAD_FEATURES_MODE

CONSTANT_PROMPTS_FLAG = configs.constant_prompts_flag
CONSTANT_PROMPT_OVERRIDE = getattr(configs, "constant_prompt_override", None) or None

LABEL_TEXTS = extract_label_texts_from_gen_cfg(getattr(configs, "generation_classification", {}) or {})
# Validate label_texts is fully populated before any model/W&B/DataLoader setup runs.
# Only enforce when generation_classification.enable is explicitly True; configs that set
# enable: false (e.g. wavlm adapter) legitimately omit label_texts.
_gen_cls_block = getattr(configs, "generation_classification", None)
if _gen_cls_block is not None:
    _gen_cls_enabled = (
        _gen_cls_block.get("enable", False)
        if isinstance(_gen_cls_block, dict)
        else bool(getattr(_gen_cls_block, "enable", False))
    )
    if _gen_cls_enabled and (not LABEL_TEXTS or 0 not in LABEL_TEXTS or 1 not in LABEL_TEXTS):
        raise ValueError(
            "generation_classification.label_texts must define both key 0 (bonafide) "
            "and key 1 (spoof) in the YAML. Got: " + repr(LABEL_TEXTS)
        )
# Assert at config load time that label_texts[1] is the spoof class.
# EER computation assumes pos_label=1 is spoof; warn loudly if that assumption may be wrong.
_SPOOF_WORDS = {"spoof", "spoofed", "fake", "synthetic", "generated"}
_label1 = str(LABEL_TEXTS.get(1, "")).lower() if isinstance(LABEL_TEXTS, dict) else ""
if LABEL_TEXTS and not any(w in _label1 for w in _SPOOF_WORDS):
    raise ValueError(
        f"label_texts[1]={LABEL_TEXTS.get(1)!r} does not appear to be the spoof class. "
        "EER computation assumes pos_label=1 is spoof (bonafide=0, spoof=1). "
        "Fix the label_texts YAML key — swapping the order silently inverts EER direction."
    )
THREE_RANDOM_PROMPTS_FLAG = bool(
    getattr(configs, "3_random_prompts", False) or getattr(configs, "three_random_prompts", False)
)
TRAINABLE_AUDIO_ENCODER = configs.trainable_audio_encoder
_audio_mode = str(TRAINABLE_AUDIO_ENCODER).strip().lower() if isinstance(TRAINABLE_AUDIO_ENCODER, str) else ("trainable" if TRAINABLE_AUDIO_ENCODER else "frozen")
# LANGUAGE_MODEL_ONLY_LORA controls whether LoRA/DoRA adapters are applied to the LM only or also to
# the audio encoder. It is derived exclusively from trainable_audio_encoder:
#   "dora"         → inject DoRA into Whisper as well as the LM  (language_model_only_lora=False)
#   anything else  → inject LoRA/DoRA into the LM only           (language_model_only_lora=True)
# Note: language_model_only_lora=True does NOT imply the audio encoder is frozen.
# Whether Whisper trains end-to-end is still controlled by trainable_audio_encoder (True/False/"trainable").
# NOTE: the YAML field `language_model_only_lora` is validated against the derived value
# and raises ValueError on mismatch. Omit it from the YAML entirely to avoid conflicts.
LANGUAGE_MODEL_ONLY_LORA = (_audio_mode != "dora")
_yaml_lm_only_lora = getattr(configs, "language_model_only_lora", None)
if _yaml_lm_only_lora is not None and bool(_yaml_lm_only_lora) != LANGUAGE_MODEL_ONLY_LORA:
    raise ValueError(
        f"YAML language_model_only_lora={_yaml_lm_only_lora} conflicts with the value derived from "
        f"trainable_audio_encoder={TRAINABLE_AUDIO_ENCODER!r} (derived: {LANGUAGE_MODEL_ONLY_LORA}). "
        "Remove language_model_only_lora from the YAML — it is always derived from trainable_audio_encoder."
    )
# PROJECTOR_PEFT — whether to wrap multi_modal_projector in LoRA/DoRA adapters.
#   False (default): projector trained end-to-end (legacy, bit-for-bit backward-compat).
#   True           : projector base weights frozen; only LoRA/DoRA adapters on
#                    linear_1 / linear_2 are trainable. Requires finetune_mode in
#                    ('lora','dora') — VoxtralStageWrapper_classification raises
#                    ValueError otherwise. Adapter r/alpha/dropout are shared with
#                    the LM PEFT (lora_rank / lora_alpha / lora_dropout).
# Read from YAML key `projector_peft`; falls back to False so older YAMLs that
# predate this flag continue to work unchanged.
PROJECTOR_PEFT = bool(getattr(configs, "projector_peft", False))
# Config-level guard: catch projector_peft + wrong finetune_mode before model/GPU/W&B init.
# (The model wrapper also checks this, but failing earlier saves startup overhead.)
if PROJECTOR_PEFT and finetune_mode not in ("lora", "dora"):
    raise ValueError(
        f"projector_peft=True requires finetune_mode in ('lora', 'dora'), "
        f"got finetune_mode={finetune_mode!r}. Fix the YAML."
    )
FREEZE_ALL = getattr(configs, "freeze_all", False)
LOG_EVAL_STATS = bool(getattr(configs, "log_eval_stats", False))
ENCODER_TYPE = getattr(configs, "encoder_type", None)
if isinstance(ENCODER_TYPE, str) and ENCODER_TYPE.lower() != "none":
    ENCODER_TYPE = ENCODER_TYPE.lower()

# ========== Model configs ==========
LOG_EVERY = configs.model['log_every']
EVAL_EVERY = configs.model['eval_every']
SAVE_EVERY = configs.model['save_every']
EVAL_EVERY_STEPS = configs.model.get('eval_every_steps')
SAVE_EVERY_STEPS = configs.model.get('save_every_steps')
if EVAL_EVERY_STEPS is not None:
    EVAL_EVERY_STEPS = int(EVAL_EVERY_STEPS)
if SAVE_EVERY_STEPS is not None:
    SAVE_EVERY_STEPS = int(SAVE_EVERY_STEPS)



NUM_CLASSES = configs.model['num_classes']


# ========== scheduler configs =========
LR_SCHEDULER_TYPE = configs.lr_scheduler_type
WARMUP_RATIO = configs.scheduler['warmup_ratio']
COSINE_CYCLES = configs.scheduler['cosine_cycles']
POLY_POWER = configs.scheduler['poly_power']
EXP_GAMMA = configs.scheduler['exp_gamma']
STEP_SIZE = configs.scheduler['step_size']
STEP_GAMMA = configs.scheduler['step_gamma']



gradient_accumulation_steps = configs.gradient_accumulation_steps if hasattr(configs, "gradient_accumulation_steps") else 1
max_grad_norm =  configs.max_grad_norm if hasattr(configs, "max_grad_norm") else 1
GRADIENT_CHECKPOINTING = getattr(configs, "gradient_checkpointing", True)


# ========== AUGMENTATION ==========
AUGMENTATION_ON = configs.augmentation_on
AUGMENTATION_PROB = configs.augmentation_prob
AUGMENTATION_ONLY_SPOOF = bool(getattr(configs, "augmentation_only_spoof", False))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch_dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "AMP": torch.float32,
}
dtype = torch_dtype_map[PRECISION]

# ========== PROMPTS ==========
# Resolve prompts path from YAML/env and fail fast if missing to avoid silent training with no prompts.
_prompts_path = os.environ.get("VOXTRAL_PROMPTS_PATH", getattr(configs, "prompts_path", None))
if _prompts_path is None:
    # Default to prompts file colocated with utils/ (utils/prompts_1000.txt)
    _prompts_path = str(Path(__file__).resolve().parent / "prompts_1000.txt")
if not os.path.exists(_prompts_path):
    raise FileNotFoundError(f"Prompts file not found at '{_prompts_path}'. Set 'prompts_path' in YAML or VOXTRAL_PROMPTS_PATH.")
PROMPTS = load_text_prompts(file_path=_prompts_path)
if not PROMPTS:
    raise ValueError(f"Prompts list is empty from '{_prompts_path}'.")
THREE_RANDOM_PROMPT_VARIANTS = []
if THREE_RANDOM_PROMPTS_FLAG:
    raw_variants = (
        getattr(configs, "three_random_prompt_variants", None)
        or getattr(configs, "random_prompt_variants", None)
        or getattr(configs, "three_random_prompts_variants", None)
    )
    if isinstance(raw_variants, str):
        raw_variants = [raw_variants]
    if not raw_variants:
        raw_variants = PROMPTS[:3]
    THREE_RANDOM_PROMPT_VARIANTS = list(raw_variants)
EVAL_PROMPT_TEXT = getattr(configs, "eval_prompt_text", None)
if isinstance(EVAL_PROMPT_TEXT, str):
    EVAL_PROMPT_TEXT = EVAL_PROMPT_TEXT.strip()
if not EVAL_PROMPT_TEXT:
    EVAL_PROMPT_TEXT = None

# ========== PROCESSOR & MODEL ==========
processor = AutoProcessor.from_pretrained(VOXTRAL_MODEL)

# Right padding required: candidate sequences are built as [prompt tokens | label tokens],
# so label tokens must stay at the end of each row for the label-position mask to line up.
processor.tokenizer.padding_side = "right"

# Some Voxtral processor configs ship with normalization disabled; force it on so mel
# features match what the audio tower was pretrained on.
processor.feature_extractor.do_normalize = True
print(f"[Config] processor.feature_extractor.do_normalize set to {processor.feature_extractor.do_normalize}")

vocab = processor.tokenizer.get_vocab()

# ========== Tokens ==========
SPECIAL_TOKEN_TASK = "<SPECIAL_20>" # Task token

print(f"check special tokens!")

wandb_cfg = configs.wandb if hasattr(configs, "wandb") else None
if isinstance(wandb_cfg, dict):
    wandb_cfg = Config(wandb_cfg)
if wandb_cfg is not None:
    # allow formatting placeholders in wandb name/project using same tokens
    if getattr(wandb_cfg, "name", None):
        wandb_cfg.name = wandb_cfg.name.format(NUM_OF_TOKENS=NUM_OF_TOKENS, DATASET=DATASET, precision=NAME_PRECISION, audio_encoder_mode=audio_encoder_mode)
    if getattr(wandb_cfg, "project", None):
        wandb_cfg.project = wandb_cfg.project.format(NUM_OF_TOKENS=NUM_OF_TOKENS, DATASET=DATASET, precision=NAME_PRECISION, audio_encoder_mode=audio_encoder_mode)
    # keep synced on configs for downstream access
    configs.wandb = wandb_cfg


# ========== AUGMENTATION CONFIG ==========
# CONFIG_AUGMENTATION is always built unconditionally; the dataset ignores it
# when augmentation_on=False.

def _cfg_get(name, default):
    val = getattr(configs, name, None)
    if val is None:
        print_default(name, default)
        return default
    return val

AUG_ADD_NOISE = bool(_cfg_get("AUG_ADD_NOISE", True))
AUG_NOISE_STD = float(_cfg_get("AUG_NOISE_STD", 0.2))

# gain perturbation
GAIN_PERTURB = bool(_cfg_get("GAIN_PERTURB", True))
GAIN_MIN_DB = float(_cfg_get("GAIN_MIN_DB", -5.0))
GAIN_MAX_DB = float(_cfg_get("GAIN_MAX_DB", 5.0))

# time shift perturbation
TIMESHIFT_PERTURB = bool(_cfg_get("TIMESHIFT_PERTURB", True))
TIMESHIFT_MAX_FRAC = float(_cfg_get("TIMESHIFT_MAX_FRAC", 0.25))
TIMESHIFT_WRAP = bool(_cfg_get("TIMESHIFT_WRAP", True))

# speed perturbation
AUG_SPEED_PERTURB = bool(_cfg_get("AUG_SPEED_PERTURB", True))
AUG_SPEED_MIN = float(_cfg_get("AUG_SPEED_MIN", 0.90))
AUG_SPEED_MAX = float(_cfg_get("AUG_SPEED_MAX", 1.10))

# time masking
AUG_TIME_MASK = bool(_cfg_get("AUG_TIME_MASK", False))            # usually for spectrograms, skipped for raw 1-D input
AUG_TIME_MASK_PARAM = int(_cfg_get("AUG_TIME_MASK_PARAM", 86))

# frequency masking
AUG_FREQ_MASK = bool(_cfg_get("AUG_FREQ_MASK", False))            # usually for spectrograms, skipped for raw 1-D input
AUG_FREQ_MASK_PARAM = int(_cfg_get("AUG_FREQ_MASK_PARAM", 48))

# codec perturbation
CODEC_PERTURB = bool(_cfg_get("CODEC_PERTURB", True))
_codec_choices = _cfg_get(
    "CODEC_CHOICES",
    ("mp3", "aac", "m4a", "opus", "flac", "alac", "amr", "speex", "g726", "g711a", "g711u", "g722", "wav"),
)
if isinstance(_codec_choices, str):
    CODEC_CHOICES = (_codec_choices,)
else:
    CODEC_CHOICES = tuple(_codec_choices)
BITRATE_KBPS_MIN = int(_cfg_get("BITRATE_KBPS_MIN", 4))
BITRATE_KBPS_MAX = int(_cfg_get("BITRATE_KBPS_MAX", 86))
KEEP_ORIGINAL_SR_AFTER_CODEC = bool(_cfg_get("KEEP_ORIGINAL_SR_AFTER_CODEC", True))
PREFER_FFMPEG = bool(_cfg_get("PREFER_FFMPEG", True))

# waveform dropout
WAVEFORM_DROPOUT = bool(_cfg_get("WAVEFORM_DROPOUT", True))
WAVEFORM_DROPOUT_MAX_FRAC = float(_cfg_get("WAVEFORM_DROPOUT_MAX_FRAC", 0.05))
WAVEFORM_DROPOUT_NUM_SPANS = int(_cfg_get("WAVEFORM_DROPOUT_NUM_SPANS", 2))

# bandpass
BANDPASS_PERTURB = bool(_cfg_get("BANDPASS_PERTURB", True))
BANDPASS_MIN_LOW_HZ = float(_cfg_get("BANDPASS_MIN_LOW_HZ", 200.0))
BANDPASS_MAX_HIGH_HZ = float(_cfg_get("BANDPASS_MAX_HIGH_HZ", 6000.0))
BANDPASS_MIN_WIDTH_HZ = float(_cfg_get("BANDPASS_MIN_WIDTH_HZ", 1000.0))

# clipping
CLIP_PERTURB = bool(_cfg_get("CLIP_PERTURB", True))
CLIP_MIN = float(_cfg_get("CLIP_MIN", 0.5))
CLIP_MAX = float(_cfg_get("CLIP_MAX", 0.98))

# RawBoost augmentation (convolutive + impulsive + SSI noise)
RAWBOOST_ENABLE = bool(_cfg_get("RAWBOOST_ENABLE", _cfg_get("RAWBOOST_AUGMENTATION", False)))
RAWBOOST_ALGO = int(_cfg_get("RAWBOOST_ALGO", 4))
RAWBOOST_N_F = int(_cfg_get("RAWBOOST_N_F", 5))
RAWBOOST_N_BANDS = int(_cfg_get("RAWBOOST_N_BANDS", 5))
RAWBOOST_MIN_F = float(_cfg_get("RAWBOOST_MIN_F", 20.0))
RAWBOOST_MAX_F = float(_cfg_get("RAWBOOST_MAX_F", 8000.0))
RAWBOOST_MIN_BW = float(_cfg_get("RAWBOOST_MIN_BW", 100.0))
RAWBOOST_MAX_BW = float(_cfg_get("RAWBOOST_MAX_BW", 1000.0))
RAWBOOST_MIN_COEFF = int(_cfg_get("RAWBOOST_MIN_COEFF", 2))
RAWBOOST_MAX_COEFF = int(_cfg_get("RAWBOOST_MAX_COEFF", 32))
RAWBOOST_MIN_G = float(_cfg_get("RAWBOOST_MIN_G", 0.0))
RAWBOOST_MAX_G = float(_cfg_get("RAWBOOST_MAX_G", 15.0))
RAWBOOST_MIN_BIAS = float(_cfg_get("RAWBOOST_MIN_BIAS", 5.0))
RAWBOOST_MAX_BIAS = float(_cfg_get("RAWBOOST_MAX_BIAS", 20.0))
RAWBOOST_P = float(_cfg_get("RAWBOOST_P", 0.6))
RAWBOOST_G_SD = float(_cfg_get("RAWBOOST_G_SD", 2.0))
RAWBOOST_SNR_MIN = float(_cfg_get("RAWBOOST_SNR_MIN", 10.0))
RAWBOOST_SNR_MAX = float(_cfg_get("RAWBOOST_SNR_MAX", 40.0))

# RawBoost/codec selection probabilities (new names with backward compat)
def _prob_get(new_name: str, old_name: str, default: float) -> float:
    if hasattr(configs, new_name):
        return float(_cfg_get(new_name, default))
    if hasattr(configs, old_name):
        return float(_cfg_get(old_name, default))
    return float(default)

probs_defined = any(
    hasattr(configs, name)
    for name in ("prob_rawboost_no_codec", "prob_codec_no_rawboost", "prob_both", "prob_rawboost_only", "prob_codec_only")
)
if probs_defined:
    PROB_RAWBOOST_NO_CODEC = _prob_get("prob_rawboost_no_codec", "prob_rawboost_only", 0.0)
    PROB_CODEC_NO_RAWBOOST = _prob_get("prob_codec_no_rawboost", "prob_codec_only", 0.0)
    PROB_BOTH = _prob_get("prob_both", "prob_both", 0.0)
else:
    if RAWBOOST_ENABLE and CODEC_PERTURB:
        PROB_RAWBOOST_NO_CODEC = 0.0
        PROB_CODEC_NO_RAWBOOST = 1.0
        PROB_BOTH = 0.0
    elif RAWBOOST_ENABLE:
        PROB_RAWBOOST_NO_CODEC = 1.0
        PROB_CODEC_NO_RAWBOOST = 0.0
        PROB_BOTH = 0.0
    elif CODEC_PERTURB:
        PROB_RAWBOOST_NO_CODEC = 0.0
        PROB_CODEC_NO_RAWBOOST = 1.0
        PROB_BOTH = 0.0
    else:
        PROB_RAWBOOST_NO_CODEC = 0.0
        PROB_CODEC_NO_RAWBOOST = 0.0
        PROB_BOTH = 0.0

# selective augmentation mode
SELECTIVE_AUGMENTATION = bool(_cfg_get("SELECTIVE_AUGMENTATION", False))
MAX_AUGMENTATIONS_PER_SAMPLE = int(_cfg_get("MAX_AUGMENTATIONS_PER_SAMPLE", 2))
MAX_AUGMENTATIONS_PER_SPECTROGRAM = int(_cfg_get("MAX_AUGMENTATIONS_PER_SPECTROGRAM", 2))

CONFIG_AUGMENTATION = WaveformAugmentationConfig(
    add_noise=AUG_ADD_NOISE,
    noise_std=AUG_NOISE_STD,

    gain_perturb=GAIN_PERTURB,
    gain_min_db=GAIN_MIN_DB,
    gain_max_db=GAIN_MAX_DB,

    timeshift_perturb=TIMESHIFT_PERTURB,
    timeshift_max_frac=TIMESHIFT_MAX_FRAC,
    timeshift_wrap=TIMESHIFT_WRAP,

    speed_perturb=AUG_SPEED_PERTURB,
    speed_min=AUG_SPEED_MIN,
    speed_max=AUG_SPEED_MAX,

    time_mask=AUG_TIME_MASK,
    time_mask_param=AUG_TIME_MASK_PARAM,

    freq_mask=AUG_FREQ_MASK,
    freq_mask_param=AUG_FREQ_MASK_PARAM,

    codec_perturb=CODEC_PERTURB,
    codec_choices=CODEC_CHOICES,
    bitrate_kbps_min=BITRATE_KBPS_MIN,
    bitrate_kbps_max=BITRATE_KBPS_MAX,
    keep_original_sr_after_codec=KEEP_ORIGINAL_SR_AFTER_CODEC,
    prefer_ffmpeg=PREFER_FFMPEG,

    waveform_dropout=WAVEFORM_DROPOUT,
    waveform_dropout_max_frac=WAVEFORM_DROPOUT_MAX_FRAC,
    waveform_dropout_num_spans=WAVEFORM_DROPOUT_NUM_SPANS,
    bandpass_perturb=BANDPASS_PERTURB,
    bandpass_min_low_hz=BANDPASS_MIN_LOW_HZ,
    bandpass_max_high_hz=BANDPASS_MAX_HIGH_HZ,
    bandpass_min_width_hz=BANDPASS_MIN_WIDTH_HZ,

    clip_perturb=CLIP_PERTURB,
    clip_min=CLIP_MIN,
    clip_max=CLIP_MAX,

    rawboost_enable=RAWBOOST_ENABLE,
    rawboost_algorithm=RAWBOOST_ALGO,
    rawboost_n_f=RAWBOOST_N_F,
    rawboost_n_bands=RAWBOOST_N_BANDS,
    rawboost_min_f=RAWBOOST_MIN_F,
    rawboost_max_f=RAWBOOST_MAX_F,
    rawboost_min_bw=RAWBOOST_MIN_BW,
    rawboost_max_bw=RAWBOOST_MAX_BW,
    rawboost_min_coeff=RAWBOOST_MIN_COEFF,
    rawboost_max_coeff=RAWBOOST_MAX_COEFF,
    rawboost_min_g=RAWBOOST_MIN_G,
    rawboost_max_g=RAWBOOST_MAX_G,
    rawboost_min_bias=RAWBOOST_MIN_BIAS,
    rawboost_max_bias=RAWBOOST_MAX_BIAS,
    rawboost_p=RAWBOOST_P,
    rawboost_g_sd=RAWBOOST_G_SD,
    rawboost_snr_min=RAWBOOST_SNR_MIN,
    rawboost_snr_max=RAWBOOST_SNR_MAX,
    prob_rawboost_no_codec=PROB_RAWBOOST_NO_CODEC,
    prob_codec_no_rawboost=PROB_CODEC_NO_RAWBOOST,
    prob_both=PROB_BOTH,

    selective_augmentation=SELECTIVE_AUGMENTATION,
    max_augmentations_per_sample=MAX_AUGMENTATIONS_PER_SAMPLE,
    max_augmentations_per_spectrogram=MAX_AUGMENTATIONS_PER_SPECTROGRAM,
)

def get_optimizer(model=None, lr: float = 0.0, weight_decay: float = 0.0, params=None):
    """
    Build optimizer using global config defaults.
    - Selects optimizer by OPTIMIZER_NAME (auto -> AdamW or AdamW8bit when USE_4BIT).
    - Accepts model or explicit params/param groups; requires one or the other.
    - Applies global BETAS/EPS/MOMENTUM settings and weight decay as provided.
    """
    name = OPTIMIZER_NAME.lower()
    if params is None:
        if model is None:
            raise ValueError("get_optimizer requires either model or params.")
        params = model.parameters()

    # "auto" picks AdamW (8-bit if USE_4BIT and bnb available)
    if name == "auto":
        if USE_4BIT:
            name = "adamw8bit"
        else:
            name = "adamw"

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=BETAS,
            eps=EPS,
            weight_decay=weight_decay,
        )
    elif name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=BETAS,
            eps=EPS,
            weight_decay=weight_decay,
        )
    elif name == "adamw8bit":
        return bnb.optim.AdamW8bit(
            params,
            lr=lr,
            betas=BETAS,
            eps=EPS,
            weight_decay=weight_decay,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=MOMENTUM,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")



def get_scheduler(optimizer, num_training_steps: int, warmup_steps: int = None):
    """
    Build learning-rate scheduler with optional warmup.
    - Uses warmup_steps when provided, otherwise WARMUP_RATIO * num_training_steps.
    - Returns LambdaLR applying linear/cosine/polynomial/exp/step decay based on LR_SCHEDULER_TYPE.
    """
    if warmup_steps is None:
        num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
    else:
        num_warmup_steps = int(warmup_steps)

    def lr_lambda(current_step: int):

        # ---------------------------
        #   Warmup
        # ---------------------------
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)

        # after warmup
        step = current_step - num_warmup_steps
        total = max(1, num_training_steps - num_warmup_steps)
        progress = step / total
        progress = min(max(progress, 0.0), 1.0)

        # ---------------------------
        #   Schedulers
        # ---------------------------

        if LR_SCHEDULER_TYPE == "linear":
            return 1.0 - progress

        elif LR_SCHEDULER_TYPE == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        elif LR_SCHEDULER_TYPE == "cosine_with_restarts":
            if "COSINE_CYCLES" not in globals():
                print("[WARNING] cosine_cycles not in config; defaulting to 1")
            cycles = COSINE_CYCLES if "COSINE_CYCLES" in globals() else 1
            return 0.5 * (1.0 + math.cos(math.pi * ((progress * cycles) % 1.0)))

        elif LR_SCHEDULER_TYPE == "polynomial":
            if "POLY_POWER" not in globals():
                print("[WARNING] poly_power not in config; defaulting to 2.0")
            power = POLY_POWER if "POLY_POWER" in globals() else 2.0
            return (1.0 - progress) ** power

        elif LR_SCHEDULER_TYPE == "exponential":
            if "EXP_GAMMA" not in globals():
                print("[WARNING] exp_gamma not in config; defaulting to 0.95")
            gamma = EXP_GAMMA if "EXP_GAMMA" in globals() else 0.95
            # decay smoothly from 1.0 to ~gamma across training (avoid gamma**step collapse)
            return gamma ** progress

        elif LR_SCHEDULER_TYPE == "step":
            # decay every STEP_SIZE steps
            if "STEP_SIZE" not in globals():
                print("[WARNING] step_size not in config; defaulting to 1000")
            if "STEP_GAMMA" not in globals():
                print("[WARNING] step_gamma not in config; defaulting to 0.1")
            step_size = STEP_SIZE if "STEP_SIZE" in globals() else 1000
            gamma = STEP_GAMMA if "STEP_GAMMA" in globals() else 0.1
            return gamma ** (step // step_size)

        elif LR_SCHEDULER_TYPE == "constant":
            return 1.0

        elif LR_SCHEDULER_TYPE == "constant_with_warmup":
            return 1.0

        elif LR_SCHEDULER_TYPE == "none":
            return 1.0

        else:
            raise ValueError(f"Unknown LR_SCHEDULER_TYPE: {LR_SCHEDULER_TYPE}")

    return LambdaLR(optimizer, lr_lambda)
