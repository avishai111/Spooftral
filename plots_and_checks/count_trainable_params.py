#!/usr/bin/env python3
"""
count_trainable_params.py
=========================
Loads two model configurations and prints a detailed breakdown of trainable
parameters per component and per adapter type (DoRA A/B, DoRA magnitude, base weights).

Usage
-----
  python count_trainable_params.py

Edit CAG_CONFIG and HEAD_CONFIG below to point at any two YAML files before running.
The full report is also written to param_count_report.txt in the project root.

Notes
-----
- utils.config is NOT imported; configs are parsed directly from YAML.
  This avoids the VOXTRAL_CONFIG env-var requirement.
- No GPU required — runs entirely on CPU.
- For the head model, load_pretrained_audio_stack=False is used to trigger
  the zero-LM meta-device fast path (avoids allocating 6 GB for the LLM).
- For the CAG model, load_pretrained_base=False builds the architecture with
  random weights (identical param count to pretrained; no network/disk I/O).
"""
import os
import sys

# Ensure project root is importable as a package root
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ===========================================================================
# EDIT THESE PATHS before running
# ===========================================================================

# CAG model (VoxtralStageWrapper_classification)
CAG_CONFIG = "configs/hyperparameters_05_dif_margins21.yaml"

# Classification-head model (AudioAdapterHeadClassifier)
HEAD_CONFIG = "configs/hyperparameters_5_classification_head.yaml"

# Output report filename (saved in the project root)
REPORT_FILE = "param_count_report.txt"

# ===========================================================================

import yaml
import torch
from collections import defaultdict, OrderedDict

# ---------------------------------------------------------------------------
# Component patterns (ordered; first match wins per parameter name)
# ---------------------------------------------------------------------------
#   Each entry: (display_label, list_of_substrings_that_identify_this_component)
_COMPONENTS = [
    ("Audio Encoder  (audio_tower)",         ["audio_tower"]),
    ("Audio Projector (multi_modal_projector)", ["multi_modal_projector"]),
    ("LLM  (language_model)",                 ["language_model"]),
    ("Classification Head",                   ["classification_head"]),
    ("Ext. Encoder Projection",               ["audio_encoder_projection"]),
    ("Ext. Audio Encoder",                    ["audio_encoder"]),
    ("Other",                                 []),          # catch-all
]


def _subtype(param_name: str) -> str:
    """Classify a parameter by its adapter role."""
    if "lora_magnitude_vector" in param_name:
        return "DoRA magnitude vector"
    if "lora_A" in param_name or "lora_B" in param_name:
        return "LoRA/DoRA adapter (A/B)"
    return "base weights"


def _component(param_name: str) -> str:
    """Return the display label for the first matching component."""
    for label, patterns in _COMPONENTS:
        if not patterns:                        # catch-all "Other"
            return label
        if any(p in param_name for p in patterns):
            return label
    return "Other"                              # should never reach here


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_params(model: torch.nn.Module):
    """
    Walk model.named_parameters() and bucket each param by (component, subtype).

    Returns
    -------
    stats : OrderedDict  {component_label: {subtype_label: {"trainable": int, "total": int}}}
    total_trainable : int
    total_all : int
    """
    # Pre-populate in component order so the output table has a stable order
    stats = OrderedDict(
        (label, defaultdict(lambda: {"trainable": 0, "total": 0}))
        for label, _ in _COMPONENTS
    )

    for name, param in model.named_parameters():
        n = param.numel()
        comp  = _component(name)
        stype = _subtype(name)
        stats[comp][stype]["total"] += n
        if param.requires_grad:
            stats[comp][stype]["trainable"] += n

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_all       = sum(p.numel() for p in model.parameters())
    return stats, total_trainable, total_all


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_W_COMP  = 42    # component column width
_W_TRAIN = 16    # trainable column width
_W_TOTAL = 16    # total column width
_W_PCT   =  9    # percentage column width
_LINE    = _W_COMP + _W_TRAIN + _W_TOTAL + _W_PCT + 3  # +3 for spaces


def _fmt(n: int) -> str:
    return f"{n:>{_W_TRAIN},}"


def _fmt_total(n: int) -> str:
    return f"{n:>{_W_TOTAL},}"


def _pct(num: int, den: int) -> str:
    if den == 0:
        return f"{'N/A':>{_W_PCT}}"
    return f"{100*num/den:>{_W_PCT-1}.2f}%"


def _pct_total(num: int, den: int) -> str:
    if den == 0:
        return f"{'N/A':>{_W_PCT}}"
    return f"{100*num/den:>{_W_PCT-1}.4f}%"


def format_report(
    model_name: str,
    config_path: str,
    stats: OrderedDict,
    total_trainable: int,
    total_all: int,
) -> str:
    sep   = "=" * _LINE
    dash  = "-" * _LINE
    hdr   = (f"{'Component':<{_W_COMP}}"
             f"{'Trainable':>{_W_TRAIN}}"
             f"{'Total':>{_W_TOTAL}}"
             f"{'  %Train':>{_W_PCT}}")

    lines = [
        sep,
        f"  MODEL : {model_name}",
        f"  CONFIG: {os.path.relpath(config_path, _PROJECT_ROOT)}",
        sep,
        hdr,
        dash,
    ]

    for comp_label, stype_dict in stats.items():
        comp_train = sum(v["trainable"] for v in stype_dict.values())
        comp_total = sum(v["total"]     for v in stype_dict.values())
        if comp_total == 0:
            continue

        # Component row (no numbers, just label to group sub-rows)
        lines.append(
            f"{'  ' + comp_label:<{_W_COMP}}"
            f"{_fmt(comp_train)}"
            f"{_fmt_total(comp_total)}"
            f"{_pct(comp_train, comp_total)}"
        )

        # Sub-type rows (sorted for stability: adapters first, then base)
        order = ["LoRA/DoRA adapter (A/B)", "DoRA magnitude vector", "base weights"]
        present = sorted(stype_dict.keys(), key=lambda k: order.index(k) if k in order else 99)
        for stype in present:
            counts = stype_dict[stype]
            if counts["total"] == 0:
                continue
            frozen_tag = "  [frozen]" if counts["trainable"] == 0 and counts["total"] > 0 else ""
            lines.append(
                f"{'    └─ ' + stype:<{_W_COMP}}"
                f"{_fmt(counts['trainable'])}"
                f"{_fmt_total(counts['total'])}"
                f"{_pct(counts['trainable'], counts['total'])}"
                f"{frozen_tag}"
            )

    lines.append(dash)
    lines.append(
        f"{'  TOTAL TRAINABLE':<{_W_COMP}}"
        f"{_fmt(total_trainable)}"
        f"{_fmt_total(total_all)}"
        f"{_pct_total(total_trainable, total_all)}"
    )
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # YAML represents None as the string "None" when quotes are inconsistent
    for k, v in list(cfg.items()):
        if isinstance(v, str) and v.strip() == "None":
            cfg[k] = None
    return cfg


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16":     torch.bfloat16,
    "float16":  torch.float16,
    "fp16":     torch.float16,
    "float32":  torch.float32,
    "fp32":     torch.float32,
    "amp":      torch.float32,
}


# ---------------------------------------------------------------------------
# Model builders (no utils.config dependency)
# ---------------------------------------------------------------------------

def build_cag_model(cfg_path: str, device: str = "cpu"):
    """Build VoxtralStageWrapper_classification from a CAG YAML config."""
    from utils.build_model import build_voxtral_model, VoxtralStageWrapper_classification

    cfg = _load_yaml(cfg_path)

    finetune_mode           = cfg.get("finetune_mode", "lora")
    base_model_name         = cfg.get("model_id", "mistralai/Voxtral-Mini-3B-2507")
    precision               = str(cfg.get("precision", "bfloat16")).lower()
    dtype                   = _DTYPE_MAP.get(precision, torch.bfloat16)
    use_4bit                = bool(cfg.get("use_4bit", False))
    lora_rank               = int(cfg.get("lora_rank", 16))
    lora_alpha              = int(cfg.get("lora_alpha", 32))
    lora_dropout            = float(cfg.get("lora_dropout", 0.05))
    target_modules          = cfg.get("target_modules",
                                      ["q_proj", "k_proj", "v_proj", "o_proj"])
    trainable_audio_encoder = cfg.get("trainable_audio_encoder", False)
    encoder_type            = cfg.get("encoder_type", None)
    k_downsample            = int(cfg.get("k_downsample", 4))
    projector_peft          = bool(cfg.get("projector_peft", False))
    model_sec               = cfg.get("model", {}) or {}
    num_classes             = int(model_sec.get("num_classes", 2)
                                  if isinstance(model_sec, dict) else 2)

    # Derive language_model_only_lora (mirrors utils/config.py logic)
    _am = (str(trainable_audio_encoder).strip().lower()
           if isinstance(trainable_audio_encoder, str)
           else ("trainable" if trainable_audio_encoder else "frozen"))
    language_model_only_lora = (_am != "dora")

    print(f"\n[CAG] finetune_mode={finetune_mode!r}  "
          f"trainable_audio_encoder={trainable_audio_encoder!r}  "
          f"language_model_only_lora={language_model_only_lora}  "
          f"projector_peft={projector_peft}")

    base_model = build_voxtral_model(
        finetune_mode=finetune_mode,
        base_model_name=base_model_name,
        dtype=dtype,
        device=device,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=target_modules,
        use_4bit=use_4bit,
        precision=precision,
        load_pretrained_base=False,   # architecture only; weights irrelevant for counting
        language_model_only_lora=language_model_only_lora,
    )

    wrapper = VoxtralStageWrapper_classification(
        base_model,
        encoder_type=encoder_type,
        trainable_audio_encoder=trainable_audio_encoder,
        k_downsample=k_downsample,
        num_classes=num_classes,
        finetune_mode=finetune_mode,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=target_modules,
        projector_peft=projector_peft,
    )
    return wrapper


def build_head_model(cfg_path: str, device: str = "cpu"):
    """Build AudioAdapterHeadClassifier from a classification-head YAML config."""
    from utils.build_model_for_classification_head import build_classification_head_model

    cfg = _load_yaml(cfg_path)

    finetune_mode   = cfg.get("finetune_mode", "lora")
    base_model_name = cfg.get("model_id", "mistralai/Voxtral-Mini-3B-2507")
    precision       = str(cfg.get("precision", "bfloat16")).lower()
    dtype           = _DTYPE_MAP.get(precision, torch.bfloat16)
    use_4bit        = bool(cfg.get("use_4bit", False))
    lora_rank       = int(cfg.get("lora_rank", 16))
    lora_alpha      = int(cfg.get("lora_alpha", 32))
    lora_dropout    = float(cfg.get("lora_dropout", 0.05))
    target_modules  = cfg.get("target_modules",
                               ["q_proj", "k_proj", "v_proj", "o_proj"])
    encoder_type    = cfg.get("encoder_type", None)
    k_downsample    = int(cfg.get("k_downsample", 4))

    head_sec = cfg.get("audio_adapter_head", {}) or {}
    audio_encoder_tuning      = head_sec.get("audio_encoder_tuning", "frozen")
    pooling_strategy          = head_sec.get("pooling_strategy", "mean")
    head_hidden_dim           = int(head_sec.get("head_hidden_dim", 256))
    head_dropout              = float(head_sec.get("head_dropout", 0.1))
    train_audio_adapter       = bool(head_sec.get("train_audio_adapter", True))
    drop_language_model       = bool(head_sec.get("drop_language_model", False))

    model_sec   = cfg.get("model", {}) or {}
    num_classes = int(model_sec.get("num_classes", 2)
                      if isinstance(model_sec, dict) else 2)

    print(f"\n[Head] finetune_mode={finetune_mode!r}  "
          f"audio_encoder_tuning={audio_encoder_tuning!r}  "
          f"drop_language_model={drop_language_model}  "
          f"train_audio_adapter={train_audio_adapter}  "
          f"head_hidden_dim={head_hidden_dim}")

    model = build_classification_head_model(
        base_model_name=base_model_name,
        device=device,
        dtype=dtype,
        finetune_mode=finetune_mode,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=target_modules,
        use_4bit=use_4bit,
        precision=precision,
        # load_pretrained_audio_stack=False: when drop_language_model=True this triggers
        # the meta-device fast path — LLM is never allocated, audio stack is
        # materialised with random weights on the target device. Param count is identical.
        load_pretrained_audio_stack=False,
        language_model_only_lora=True,   # head paradigm: LM adapters irrelevant (LM frozen/dropped)
        encoder_type=encoder_type,
        audio_encoder_tuning=audio_encoder_tuning,
        k_downsample=k_downsample,
        num_classes=num_classes,
        pooling_strategy=pooling_strategy,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        train_audio_adapter=train_audio_adapter,
        drop_language_model=drop_language_model,
    )
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _step(msg: str) -> None:
    """Print a clearly visible progress step."""
    print(f"\n>>> {msg}", flush=True)


def main():
    # Resolve relative paths from the project root
    cag_cfg  = CAG_CONFIG  if os.path.isabs(CAG_CONFIG)  else os.path.join(_PROJECT_ROOT, CAG_CONFIG)
    head_cfg = HEAD_CONFIG if os.path.isabs(HEAD_CONFIG) else os.path.join(_PROJECT_ROOT, HEAD_CONFIG)

    device = "cpu"   # CPU is sufficient for parameter counting; avoids GPU allocation

    _step(f"Configs\n    CAG  : {os.path.relpath(cag_cfg,  _PROJECT_ROOT)}"
          f"\n    Head : {os.path.relpath(head_cfg, _PROJECT_ROOT)}")
    _step(f"Device : {device}  (no GPU needed)")

    reports = []

    # ── spootral ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[1/6] Importing model builders...")
    print("=" * 60, flush=True)

    _step("[2/6] Building spootral base model  (VoxtralForConditionalGeneration + DoRA)...")
    cag_model = build_cag_model(cag_cfg, device=device)

    _step("[3/6] Counting spootral parameters...")
    cag_stats, cag_train, cag_total = count_params(cag_model)
    print(f"      trainable: {cag_train:,}  /  total: {cag_total:,}  "
          f"({100*cag_train/cag_total:.4f}%)", flush=True)

    cag_report = format_report(
        "spootral  —  CAG / DoRA  (VoxtralStageWrapper_classification)",
        cag_cfg,
        cag_stats,
        cag_train,
        cag_total,
    )
    reports.append(cag_report)
    del cag_model   # free RAM before building the next model
    _step("spootral done — RAM freed.")

    # ── pootral-enc ───────────────────────────────────────────────────────────
    _step("[4/6] Building pootral-enc base model  (meta-device fast path, LLM never allocated)...")
    head_model = build_head_model(head_cfg, device=device)

    _step("[5/6] Counting pootral-enc parameters...")
    head_stats, head_train, head_total = count_params(head_model)
    print(f"      trainable: {head_train:,}  /  total: {head_total:,}  "
          f"({100*head_train/head_total:.4f}%)", flush=True)

    head_report = format_report(
        "pootral-enc  —  AudioAdapterHead / DoRA  (AudioAdapterHeadClassifier)",
        head_cfg,
        head_stats,
        head_train,
        head_total,
    )
    reports.append(head_report)
    del head_model

    # ── Print & save ──────────────────────────────────────────────────────────
    _step("[6/6] Writing report...")
    full_report = "\n\n".join(reports)
    print("\n\n")
    print(full_report)

    out_path = REPORT_FILE if os.path.isabs(REPORT_FILE) else os.path.join(_PROJECT_ROOT, REPORT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_report + "\n")
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
