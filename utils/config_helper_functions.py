"""
utils/config_helper_functions.py
=================================
Config persistence, display, and label-text resolution helpers.

Key functions
-------------
save_all_configs()
    Serialises all uppercase module-level constants from utils.config plus the
    configs object attributes into all_configs.json in output_dir.  Appends an
    auto-incrementing suffix when the file already exists to avoid overwrites.

log_run_settings() / print_config_summary()
    Pretty-print the effective hyperparameters to stdout at run start.

extract_label_texts_from_gen_cfg()
    Extracts {0: str, 1: str} label mapping from the generation_classification
    YAML block; returns an empty dict when the block is absent so the caller can
    apply its own default.

resolve_label_texts()
    Validates and normalises a label_texts dict to {0: str, 1: str}, emitting
    warnings when expected keys are absent and using "bonafide"/"spoof" defaults.
"""
import os
import json
import importlib

def save_all_configs(
    output_dir: str,
    configs,
    module_path="utils.config",
    model=None,
    loss_fn=None,
    optimizer=None,
    scheduler=None
):
    """
    Persist merged config variables to JSON for reproducibility.
    - Collects uppercase module vars (excluding PROMPTS) plus attributes from configs object.
    - Adds class names for model/loss/optimizer/scheduler when provided.
    - Writes to all_configs.json (auto-incrementing copy if file exists) under output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = importlib.import_module(module_path)

    # Convention: only uppercase names in utils.config are treated as constants worth
    # recording; PROMPTS is excluded since it's a large text template, not a hyperparameter.
    cfg_vars = {k: v for k, v in vars(cfg).items() if k.isupper() and k != "PROMPTS"}
    config_obj_vars = {k: v for k, v in vars(configs).items() if not k.startswith("__")}
    all_vars = {**cfg_vars, **config_obj_vars}

    if model is not None:
        all_vars["MODEL_CLASS"] = model.__class__.__name__
        # Prefer a model-provided parameter summary (e.g. from a LoRA/adapter wrapper) when
        # available; otherwise compute trainable vs. total parameter counts directly.
        if hasattr(model, "_trainable_params_summary"):
            all_vars.update(model._trainable_params_summary)
        else:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            all_vars["TRAINABLE_PARAMS"] = trainable
            all_vars["TOTAL_PARAMS"] = total
            all_vars["TRAINABLE_PARAMS_PCT"] = round(100 * trainable / total, 4) if total > 0 else 0
    if loss_fn is not None:
        all_vars["LOSS_FN"] = loss_fn.__class__.__name__
    if optimizer is not None:
        all_vars["OPTIMIZER"] = optimizer.__class__.__name__
    if scheduler is not None:
        all_vars["SCHEDULER"] = scheduler.__class__.__name__

    # Not every config value is JSON-serializable (e.g. torch dtypes, class instances);
    # stringify anything that isn't so the dump always succeeds.
    def safe(val):
        try:
            json.dumps(val)
            return val
        except Exception:
            return str(val)

    all_vars = {k: safe(v) for k, v in all_vars.items()}

    config_path = os.path.join(output_dir, "all_configs.json")

    # if the file already exists, create an additional numbered copy
    if os.path.exists(config_path):
        i = 1
        while True:
            alt_path = os.path.join(output_dir, f"all_configs_{i}.json")
            if not os.path.exists(alt_path):
                config_path = alt_path
                break
            i += 1

    with open(config_path, "w") as f:
        json.dump(all_vars, f, indent=2)
    print(f"Saved {len(all_vars)} configs to {config_path}")


def log_run_settings(loss_type, loss_params):
    """
    Print a lightweight summary of training loss settings.
    - Logs loss type and params; exceptions are suppressed to avoid run failures.
    """
    try:
        print(
            "[Run Settings]",
            f"loss_type={loss_type}, loss_params={loss_params}",
        )
    except Exception:
        pass




def print_config_summary(configs, loss_type, loss_params):
    """
    Summarize key training config at startup.
    - Extracts precision, num_classes, loss details, training regime, and max steps.
    - Prints a single summary dict; exceptions suppressed for robustness.
    """
    try:
        summary = {
            "precision": getattr(configs, "precision", None),
            "num_classes": getattr(configs, "NUM_CLASSES", None) or getattr(configs, "num_classes", None),
            "loss_type": loss_type,
            "loss_params": loss_params,
            "train_by": str(getattr(configs, "train_by", "epochs")).lower(),
            "max_train_steps": getattr(configs, "max_train_steps", None),
        }
        print(f"[Config Summary] {summary}")
    except Exception:
        pass


def resolve_label_texts(raw_dict: dict, context_label: str = "") -> dict:
    """Normalise a label_texts dict and fill in bonafide/spoof defaults with visible warnings."""
    lt = {int(k): v for k, v in (raw_dict or {}).items()}
    prefix = f"{context_label} " if context_label else ""
    if 0 not in lt:
        print(f"[LABEL_TEXTS FALLBACK] {prefix}label_texts[0] not set -> using default 'bonafide'")
        lt[0] = "bonafide"
    if 1 not in lt:
        print(f"[LABEL_TEXTS FALLBACK] {prefix}label_texts[1] not set -> using default 'spoof'")
        lt[1] = "spoof"
    print(f"[LABEL_TEXTS] {prefix}labels: 0='{lt[0]}', 1='{lt[1]}'")
    return lt


def extract_label_texts_from_gen_cfg(gen_cfg) -> dict:
    """Extract label_texts from a generation_classification config (dict or object), without defaults."""
    if isinstance(gen_cfg, dict):
        raw = gen_cfg.get("label_texts", {}) or {}
    else:
        raw = getattr(gen_cfg, "label_texts", {}) or {}
    return {int(k): v for k, v in raw.items()} if raw else {}
