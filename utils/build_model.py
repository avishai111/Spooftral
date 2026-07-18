"""
utils/build_model.py
====================
Model construction for the Classification-as-Generation (CAG) paradigm.

Key exports
-----------
build_voxtral_model()
    Loads VoxtralForConditionalGeneration from HuggingFace, applies the chosen
    PEFT strategy (LoRA / DoRA / AdaLoRA / QLoRA / frozen / full), and
    optionally loads a weights-only checkpoint.  All loads use strict=True.

VoxtralStageWrapper_classification
    nn.Module wrapper that combines the base Voxtral LLM with an audio encoder
    (built-in Whisper tower or an external wav2vec2/HuBERT/WavLM).  Used
    exclusively for CAG: no classification head is attached; labels are scored
    by comparing log P("bonafide") vs log P("spoof").
    Supports projector_peft for PEFT-wrapping the audio→LM bridge module.

VoxtralModalProjector_Encoder_Audio
    Replacement projector used when k_downsample != 4.  Projects from
    k_downsample*1280 dimensions to the LM hidden size (3072 for Mistral-3B)
    via Linear → GELU → Linear.

load_checkpoint_into_model()
    Loads a training or weights-only checkpoint into an existing model,
    auto-adapting key prefixes (strip/add 'model.') to handle wrapper-vs-bare
    key namespace mismatches.

_adapt_state_keys()
    Heuristic prefix stripping/adding that maximises state-dict key overlap
    before strict=True load.

External encoder note
---------------------
wav2vec2 / hubert / wavlm paths use 'out_proj' (not 'o_proj') as the
attention output module.  _set_audio_trainable() in the wrapper already
handles this — do not unify the names.
"""
import os
import re
import torch
import torch.nn as nn
from transformers import (
    VoxtralForConditionalGeneration,
    AutoConfig,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
try:
    from peft import AdaLoraConfig
except Exception:
    AdaLoraConfig = None
from transformers.activations import GELUActivation
from utils.build_model_weight_formulas import compute_weights_sqrt_inverse
# Local cache dir for the downloaded base model weights; override via VOXTRAL_CACHE_DIR.
CACHE_DIR = os.environ.get("VOXTRAL_CACHE_DIR", "/path/to/voxtral/base_model/")


def _adapt_state_keys(state: dict, model) -> dict:
    """Align saved checkpoint keys to the current model's state_dict using coverage scoring."""
    model_keys = set(model.state_dict().keys())
    if not state:
        return state

    def _overlap(d):
        return sum(1 for k in d if k in model_keys)

    best_state = state
    best_overlap = _overlap(state)

    # Already matching well enough
    if best_overlap / max(len(state), 1) > 0.5:
        return state

    # Try stripping known prefixes
    for prefix in ("model.", "base_model.model.", "module.", "wrapper."):
        candidate = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}
        n = _overlap(candidate)
        if n > best_overlap:
            best_overlap = n
            best_state = candidate
            print(f"[Checkpoint] Stripped prefix '{prefix}' from saved keys (coverage {n}/{len(state)}).")

    # Try adding known prefixes
    for prefix in ("model.", "base_model.model."):
        candidate = {prefix + k: v for k, v in state.items()}
        n = _overlap(candidate)
        if n > best_overlap:
            best_overlap = n
            best_state = candidate
            print(f"[Checkpoint] Added prefix '{prefix}' to saved keys (coverage {n}/{len(state)}).")

    if best_overlap == 0:
        print("[Checkpoint] WARNING: Could not auto-align state_dict keys. Loading as-is.")

    return best_state


def _checkpoint_sort_key(filename: str) -> int:
    """
    Return a numeric sort key from a checkpoint filename.

    Extracts the step or epoch number so that checkpoint-step1000.pt sorts
    after checkpoint-step200.pt (numeric order), not before it (alphabetic order).
    Falls back to 0 for unrecognised names so those sort to the front.
    """
    m = re.search(r"step(\d+)", filename)
    if m:
        return int(m.group(1))
    m = re.search(r"epoch(\d+)", filename)
    if m:
        return int(m.group(1))
    return 0


class VoxtralModalProjector_Encoder_Audio(nn.Module):
    def __init__(self, input_dim, out_dim=3072):
        """Two-layer projector: input_dim → out_dim via Linear → GELU → Linear (no bias)."""
        super().__init__()
        self.input_dim = input_dim
        self.linear_1 = nn.Linear(self.input_dim, out_dim, bias=False)
        self.act = GELUActivation()  # Using GELU activation
        self.linear_2 = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def compute_class_weights_from_dataset(dataset, device):
    """Compatibility shim — delegates to compute_weights_sqrt_inverse."""
    return compute_weights_sqrt_inverse(dataset, device)


def build_voxtral_model(
    finetune_mode, base_model_name, dtype, device,
    lora_rank=16, lora_alpha=32, lora_dropout=0.05,
    lora_target_modules=None,
    use_4bit=False,
    precision="float16",
    resume=None,
    load_model_path=None,
    load_pretrained_base=True,
    language_model_only_lora=True,
):
    """
    Load and configure a Voxtral model with adapters, quantization, and checkpoint handling.

    finetune_mode controls which adapter strategy is applied to the language model:
      - "lora"    : LoRA adapters on lora_target_modules (default: q/k/v/o_proj).
      - "dora"    : DoRA adapters (magnitude + direction, requires recent PEFT).
      - "adalora" : AdaLoRA — adaptive rank allocation across layers.
      - "frozen"  : no adapters; all weights frozen (eval or classification-head mode).
      - any other : full fine-tuning of the entire model (high VRAM cost).

    language_model_only_lora=True (default):
        Adapters are injected only into the LM. The audio tower and projector remain
        as plain nn.Module parameters and are trained directly (no PEFT wrapping).
        Set to False to also wrap the audio encoder in PEFT adapters.

    Quantization (use_4bit=True):
        Loads in 4-bit NF4 with double quantization (QLoRA setup).
        prepare_model_for_kbit_training is called automatically to enable
        gradient checkpointing through quantized layers.

    precision: controls the compute dtype — "bfloat16", "float16", "float32", or "amp".

    Checkpoint loading:
        - resume: path to a training checkpoint dict (full optimizer/scheduler state).
        - load_model_path: path to a weights-only checkpoint (model state dict only).
        - All loads use strict=True — mismatched keys raise immediately.

    Returns the configured model (PeftModel if LoRA/DoRA/AdaLoRA, else plain nn.Module).
    """
    cache_dir = CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Normalize precision to a torch dtype so we actually honor the precision flag when loading
    precision_key = str(precision).lower() if precision is not None else None
    precision_to_dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "amp": torch.float32,  # autocast runs from float32 params
    }
    dtype = precision_to_dtype.get(precision_key, dtype)
    if dtype is None:
        raise ValueError("Please provide a torch dtype or a valid precision (float16, float32, bfloat16, amp/fp16/fp32/bf16).")

    mode = str(finetune_mode).strip().lower().replace("-", "_")
    if mode in ("adaptive_ranking", "adaptive_rank", "adaptive_lora", "adalora"):
        mode = "adalora"
    
    # Always use CAUSAL_LM task type for classification-as-generation
    task_type = TaskType.CAUSAL_LM
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    if mode in ("lora", "dora"):
        lora_config_kwargs = {
            "r": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": lora_target_modules,
            "bias": "none",
            "task_type": task_type,
        }
        if mode == "dora":
            lora_config_kwargs["use_dora"] = True
        try:
            lora_config = LoraConfig(**lora_config_kwargs)
        except TypeError as e:
            if mode == "dora":
                raise ValueError("finetune_mode='dora' requires a PEFT version with DoRA support (use_dora).") from e
            raise
        adapter_name = "DoRA" if mode == "dora" else "LoRA"
        if mode == "lora":
            print(f"[Task_Type] = {task_type}")
        if load_pretrained_base:
            if use_4bit:
                base_model = VoxtralForConditionalGeneration.from_pretrained(
                    base_model_name,
                    cache_dir=cache_dir,
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                try:
                    from peft import prepare_model_for_kbit_training
                    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
                    print("[build_model] Applied prepare_model_for_kbit_training for QLoRA/4-bit.")
                except ImportError:
                    print("[build_model] WARNING: peft.prepare_model_for_kbit_training not available; skipping. QLoRA + gradient checkpointing may produce no gradients.")
            else:
                base_model = VoxtralForConditionalGeneration.from_pretrained(
                    base_model_name,
                    cache_dir=cache_dir,
                    dtype=dtype,
                    device_map=device,
                    low_cpu_mem_usage=True,
                )
                print(f"[build_model] Model loaded on {device}.", flush=True)
        else:
            if use_4bit:
                raise ValueError("Cannot initialize from config with use_4bit=True. Set load_pretrained_base=True for 4-bit.")
            cfg = AutoConfig.from_pretrained(base_model_name, cache_dir=cache_dir)
            base_model = VoxtralForConditionalGeneration._from_config(cfg).to(device).to(dtype)

        # Apply adapter configuration based on language_model_only_lora flag
        if language_model_only_lora:
            # Apply adapters ONLY to the language model, not audio components
            print(f"[{adapter_name}] Applying {adapter_name} adapters ONLY to language_model (audio encoder trainability set by trainable_audio_encoder)")
            base_model.language_model = get_peft_model(base_model.language_model, lora_config)
            base_model.language_model.enable_input_require_grads()
        else:
            # Apply adapters to the entire model (old behavior)
            print(f"[{adapter_name}] Applying {adapter_name} adapters to ENTIRE model (including audio components)")
            base_model = get_peft_model(base_model, lora_config)
            base_model.enable_input_require_grads()
        
    elif mode == "adalora":
        if AdaLoraConfig is None:
            raise ImportError("finetune_mode='adalora' requires peft with AdaLoRA support installed.")
        adalora_config = AdaLoraConfig(
            init_r=lora_rank,
            target_r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            task_type=task_type,
        )
        if load_pretrained_base:
            if use_4bit:
                base_model = VoxtralForConditionalGeneration.from_pretrained(
                    base_model_name,
                    cache_dir=cache_dir,
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                try:
                    from peft import prepare_model_for_kbit_training
                    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
                    print("[build_model] Applied prepare_model_for_kbit_training for QLoRA/4-bit.")
                except ImportError:
                    print("[build_model] WARNING: peft.prepare_model_for_kbit_training not available; skipping. QLoRA + gradient checkpointing may produce no gradients.")
            else:
                base_model = VoxtralForConditionalGeneration.from_pretrained(
                    base_model_name,
                    cache_dir=cache_dir,
                    dtype=dtype,
                    device_map=device,
                    low_cpu_mem_usage=True,
                )
                print(f"[build_model] Model loaded on {device}.", flush=True)
        else:
            if use_4bit:
                raise ValueError("Cannot initialize from config with use_4bit=True. Set load_pretrained_base=True for 4-bit.")
            cfg = AutoConfig.from_pretrained(base_model_name, cache_dir=cache_dir)
            base_model = VoxtralForConditionalGeneration._from_config(cfg).to(device).to(dtype)

        if language_model_only_lora:
            print(f"[AdaLoRA] Applying adaptive rank adapters ONLY to language_model (audio components will be fully trainable)")
            base_model.language_model = get_peft_model(base_model.language_model, adalora_config)
            base_model.language_model.enable_input_require_grads()
        else:
            print(f"[AdaLoRA] Applying adaptive rank adapters to ENTIRE model (including audio components)")
            base_model = get_peft_model(base_model, adalora_config)
            base_model.enable_input_require_grads()

    elif mode == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type=task_type,
        )
        if not load_pretrained_base:
            raise ValueError("QLoRA requires loading a quantized pretrained base. Set load_pretrained_base=True.")
        base_model = VoxtralForConditionalGeneration.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            device_map="auto",
        )
        try:
            from peft import prepare_model_for_kbit_training
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
            print("[build_model] Applied prepare_model_for_kbit_training for QLoRA/4-bit.")
        except ImportError:
            print("[build_model] WARNING: peft.prepare_model_for_kbit_training not available; skipping. QLoRA + gradient checkpointing may produce no gradients.")

        if language_model_only_lora:
            print(f"[QLoRA] Applying QLoRA adapters ONLY to language_model")
            base_model.language_model = get_peft_model(base_model.language_model, lora_config)
            base_model.language_model.enable_input_require_grads()
        else:
            print(f"[QLoRA] Applying QLoRA adapters to ENTIRE model (including audio components)")
            base_model = get_peft_model(base_model, lora_config)
            base_model.enable_input_require_grads()
        
    else:
        if load_pretrained_base:
            if use_4bit:
                base_model = VoxtralForConditionalGeneration.from_pretrained(
                    base_model_name,
                    cache_dir=cache_dir,
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                try:
                    from peft import prepare_model_for_kbit_training
                    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
                    print("[build_model] Applied prepare_model_for_kbit_training for QLoRA/4-bit.")
                except ImportError:
                    print("[build_model] WARNING: peft.prepare_model_for_kbit_training not available; skipping. QLoRA + gradient checkpointing may produce no gradients.")
            else:
                base_model = VoxtralForConditionalGeneration.from_pretrained(
                    base_model_name,
                    cache_dir=cache_dir,
                    dtype=dtype,
                    device_map=device,
                    low_cpu_mem_usage=True,
                )
                print(f"[build_model] Model loaded on {device}.", flush=True)
        else:
            if use_4bit:
                raise ValueError("Cannot initialize from config with use_4bit=True. Set load_pretrained_base=True for 4-bit.")
            cfg = AutoConfig.from_pretrained(base_model_name, cache_dir=cache_dir)
            base_model = VoxtralForConditionalGeneration._from_config(cfg).to(device).to(dtype)

        if mode in ["frozen", "classifier"]:
            for p in base_model.parameters():
                p.requires_grad = False

    if load_model_path:
        ckpt_path = load_model_path
        if os.path.isdir(ckpt_path):
            cand = [f for f in os.listdir(ckpt_path) if f.endswith((".bin", ".pt", ".ckpt", ".pth"))]
            cand.sort(key=_checkpoint_sort_key)
            if not cand:
                raise FileNotFoundError(f"No checkpoint file in directory: {ckpt_path}")
            ckpt_path = os.path.join(ckpt_path, cand[-1])

        print(f"Loading model weights from: {ckpt_path}")
        #Try weights_only first for security, fall back if needed
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        state_options = []
        if isinstance(ckpt, dict):
            for key in ("wrapper_state_dict", "model_state_dict", "state_dict"):
                if key in ckpt:
                    state_options.append(ckpt[key])
        if not state_options:
            state_options.append(ckpt)

        # Also try stripping 'model.' prefix: checkpoints saved from VoxtralStageWrapper_classification
        # store the inner model as self.model, so all keys are prefixed with 'model.'.
        # Strip that prefix to match the bare PEFT/base model's key namespace.
        prefixed = []
        for state in state_options:
            if any(k.startswith("model.") for k in state):
                stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
                if stripped:
                    prefixed.append(stripped)
        state_options = state_options + prefixed

        last_error = None
        for state in state_options:
            try:
                state = _adapt_state_keys(state, base_model)
                base_model.load_state_dict(state, strict=True)
                print(f"Loaded model weights from: {ckpt_path}")
                last_error = None
                break
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            raise RuntimeError(
                f"Failed to load checkpoint weights from {ckpt_path}: {last_error}. "
                "The checkpoint architecture does not match the model. "
                "Check that PEFT/LoRA wrapping is identical between save and load."
            ) from last_error

        if resume is not None:
            print(f"Resume requested from epoch {resume}")

    return base_model


def load_checkpoint_into_model(model: nn.Module, ckpt_path: str, device: str = "cpu") -> None:
    """
    Load weights into an existing model from a checkpoint path.

    Accepts a directory (picks the latest matching file) or an explicit file path.
    Tries wrapper_state_dict / model_state_dict / state_dict keys before the raw dict.
    Auto-adapts key prefixes via _adapt_state_keys (strip/add 'model.' etc.).
    Always strict=True — raises RuntimeError if no format produces a clean load.
    """
    if os.path.isdir(ckpt_path):
        cand = [f for f in os.listdir(ckpt_path) if f.endswith((".bin", ".pt", ".ckpt", ".pth"))]
        cand.sort(key=_checkpoint_sort_key)
        if not cand:
            raise FileNotFoundError(f"No checkpoint file in directory: {ckpt_path}")
        ckpt_path = os.path.join(ckpt_path, cand[-1])

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_options = []
    if isinstance(ckpt, dict):
        for key in ("wrapper_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt:
                state_options.append((key, ckpt[key]))
    if not state_options:
        state_options.append(("raw", ckpt))

    derived = []
    for key, state in state_options:
        # Format 2: strip 'model.' prefix — wrapper checkpoint → bare PEFT model
        if any(k.startswith("model.") for k in state):
            stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
            if stripped:
                derived.append((key + "_strip_model_prefix", stripped,
                                 "Format 2 (strip 'model.' prefix): wrapper-saved checkpoint → bare PEFT model"))
        # Format 3: add 'model.' prefix — bare PEFT checkpoint → wrapper
        if not any(k.startswith("model.") for k in state):
            prefixed = {"model." + k: v for k, v in state.items()}
            derived.append((key + "_add_model_prefix", prefixed,
                             "Format 3 (add 'model.' prefix): bare-PEFT checkpoint → wrapper"))

    state_options_full = [
        (key, state, "Format 1 (direct, strict=True): checkpoint keys match model as-is")
        for key, state in state_options
    ] + derived

    last_error = None
    for key, state, fmt_desc in state_options_full:
        try:
            state = _adapt_state_keys(state, model)
            model.load_state_dict(state, strict=True)
            print(f"[Checkpoint] Loaded '{key}' from: {ckpt_path}")
            print(f"[Checkpoint] {fmt_desc}")
            return
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(
        f"Failed to load checkpoint from {ckpt_path}: {last_error}. "
        "The checkpoint architecture does not match the model. "
        "Check that PEFT/LoRA wrapping is identical between save and load."
    ) from last_error




# ============================================================================
# CAG MODEL WRAPPER
# Combines the Voxtral LLM with an audio encoder for Classification-as-Generation.
# The LM scores log-probs of label strings ("bonafide"/"spoof").
# ============================================================================

class VoxtralStageWrapper_classification(nn.Module):
    def __init__(
        self,
        base_model,
        encoder_type=None,
        trainable_audio_encoder=False,
        k_downsample=4,
        num_classes=2,
        finetune_mode="lora",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
        projector_peft=False,
    ):
        """
        Wrapper for Classification-as-Generation (CAG) mode — no classification head.

        The model scores log-probabilities of label strings ("bonafide", "spoof") directly
        from the LM and picks the highest as the prediction. The LM is assumed to already
        have LoRA/DoRA injected by build_voxtral_model before this wrapper is created.

        encoder_type selects the audio encoder:
          - None / "none"  : uses Voxtral's built-in Whisper-based audio tower (default).
          - "wav2vec2"     : facebook/wav2vec2-xls-r-300m
          - "hubert"       : facebook/hubert-large-ls960-ft
          - "wavlm"        : microsoft/wavlm-large
          External encoders are followed by a learned VoxtralModalProjector_Encoder_Audio
          that projects their hidden size into the LM hidden size (3072 for Mistral-3B).

        trainable_audio_encoder controls the audio encoder's gradient flow:
          True / "trainable" — full fine-tuning of all encoder weights.
          False / "frozen"   — encoder frozen; LoRA/DoRA adapters injected instead
                               (finetune_mode + lora_rank/alpha/dropout apply here).
          "dora"             — base weights frozen, only DoRA adapters train.

        k_downsample: temporal downsampling factor applied before the projector.
          Default is 4 (matches Voxtral's built-in projector stride). Changing this
          replaces model.multi_modal_projector with a new VoxtralModalProjector_Encoder_Audio
          sized to k_downsample * 1280 → LM hidden size.

        projector_peft (default False): freezes the base weights of multi_modal_projector
          and wraps it in LoRA/DoRA adapters (target_modules=['linear_1','linear_2']).
          This is the parameter-efficient alternative to training the projector end-to-end.
          Motivation: the projector is the bottleneck through which all audio information
          flows into the LM, so unconstrained updates here can shift the LM's input
          distribution in ways the LM-side LoRA cannot compensate for. PEFT-wrapping it
          preserves the original linear mapping while still allowing task adaptation.

          Requires finetune_mode in ('lora', 'dora') — raises ValueError otherwise.
          Adapter hyperparameters (rank, alpha, dropout) are shared with the LM PEFT.
          Default False = legacy end-to-end training of the projector (bit-for-bit
          backward-compatible; old checkpoints continue to load under strict=True).
          Applies uniformly to both Voxtral's default projector (k_downsample=4) and the
          custom VoxtralModalProjector_Encoder_Audio (k_downsample!=4); both expose
          'linear_1' and 'linear_2' Linear submodules.
        """
        super().__init__()
        self.model = base_model
        self.model_dtype = next(base_model.parameters()).dtype
        self.trainable_audio_encoder = trainable_audio_encoder
        self.finetune_mode = finetune_mode
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules if lora_target_modules is not None else ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.projector_peft = projector_peft

        self.device = self.model.device
        # `.model` here resolves through PeftModel's attribute delegation to the wrapped
        # base_model when the LM is LoRA/DoRA-adapted, and directly to the LM otherwise.
        classification_hidden_size = self.model.language_model.model.config.hidden_size # 3072
        self.is_peft_lm = hasattr(self.model.language_model, "peft_config") or "peft" in self.model.language_model.__class__.__name__.lower()
        self.k_downsample = k_downsample  # downsample audio frames before projection 
        self.num_classes = num_classes

        if self.k_downsample != 4:
            self.model.multi_modal_projector = VoxtralModalProjector_Encoder_Audio(input_dim = self.k_downsample*1280, out_dim = classification_hidden_size).to(self.model_dtype).to(self.model.device)

        # ── Optional PEFT wrap of the audio → LM bridge (multi_modal_projector) ───────────
        # When projector_peft=True, freeze the projector's base Linear weights and inject
        # LoRA/DoRA adapters into its 'linear_1' and 'linear_2' submodules. This is a
        # catastrophic-forgetting mitigation: the projector is the single bottleneck
        # through which all audio information enters the LM, so unconstrained updates
        # here can shift the LM input distribution in ways the LM-side LoRA cannot
        # compensate for. Restricting the projector to low-rank deltas preserves its
        # original linear mapping while still allowing task adaptation.
        #
        # Target-module names ('linear_1', 'linear_2') are identical in:
        #   - Voxtral's built-in VoxtralMultiModalProjector (k_downsample==4)
        #   - the custom VoxtralModalProjector_Encoder_Audio (k_downsample!=4 — wrapped above)
        # so a single LoraConfig covers both projector implementations.
        #
        # IMPORTANT — task_type is intentionally omitted from LoraConfig. PEFT's
        # TaskType.FEATURE_EXTRACTION routes forward() through a `input_ids=` kwarg via
        # PeftModelForFeatureExtraction, but the projector's forward signature is
        # `forward(self, x)` (single positional tensor). Passing task_type breaks the
        # forward call with `TypeError: forward() got an unexpected keyword argument
        # 'input_ids'`. Plain `get_peft_model(module, LoraConfig(...))` returns a
        # PeftModel whose forward delegates positionally — which is what we need.
        #
        # Hyperparameter sharing: r / alpha / dropout match the LM PEFT (no separate
        # YAML keys) — keeps the configuration surface minimal.
        #
        # finetune_mode guard: this block requires 'lora' or 'dora'. Other modes
        # ('frozen', 'adalora', full FT) raise ValueError immediately so the user fails
        # fast rather than silently leaving the projector with all-frozen weights and
        # no adapters.
        if self.projector_peft:
            proj_mode = str(self.finetune_mode).strip().lower().replace("-", "_")
            if proj_mode not in ("lora", "dora"):
                raise ValueError(
                    f"projector_peft=True requires finetune_mode in ('lora', 'dora'); got '{self.finetune_mode}'."
                )
            projector_lora_kwargs = {
                "r": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": ["linear_1", "linear_2"],
                "bias": "none",
            }
            if proj_mode == "dora":
                projector_lora_kwargs["use_dora"] = True
            try:
                projector_lora_config = LoraConfig(**projector_lora_kwargs)
                self.model.multi_modal_projector = get_peft_model(
                    self.model.multi_modal_projector, projector_lora_config
                )
                proj_adapter_name = "DoRA" if proj_mode == "dora" else "LoRA"
                print(f"[Projector {proj_adapter_name}] Applied {proj_adapter_name} to multi_modal_projector")
                print(f"[Projector {proj_adapter_name}] Target modules: ['linear_1', 'linear_2']  r={self.lora_rank}, alpha={self.lora_alpha}")
            except TypeError as e:
                if proj_mode == "dora":
                    raise ValueError("projector_peft=True with finetune_mode='dora' requires a PEFT version with DoRA support.") from e
                raise

        # Normalize external encoder selection
        if isinstance(encoder_type, str):
            encoder_type = encoder_type.lower()
            if encoder_type == "none":
                encoder_type = None
        self.encoder_type = encoder_type
        self.encoder_hidden_size = None

        if encoder_type == "wav2vec2":
            from transformers import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m", cache_dir=CACHE_DIR)
        elif encoder_type == "hubert":
            from transformers import HubertModel
            self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir=CACHE_DIR)
        elif encoder_type == "wavlm":
            from transformers import WavLMModel
            self.audio_encoder = WavLMModel.from_pretrained("microsoft/wavlm-large", cache_dir=CACHE_DIR)
        elif encoder_type is None:
            self.audio_encoder = None
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Store encoder_hidden_size BEFORE applying PEFT (PEFT wrapping can change config access)
        if self.audio_encoder is not None:
            self.encoder_hidden_size = self.audio_encoder.config.hidden_size

        # Apply LoRA/DoRA to audio encoder when trainable_audio_encoder=False (same logic as LM)
        # trainable_audio_encoder=True means full fine-tuning, False means use LoRA/DoRA
        if self.audio_encoder is not None and not self.trainable_audio_encoder:
            # Audio encoders (WavLM, Wav2Vec2, HuBERT) use "out_proj" not "o_proj" like LLMs
            # Map o_proj -> out_proj for audio encoder compatibility
            audio_target_modules = []
            for mod in self.lora_target_modules:
                if mod == "o_proj":
                    audio_target_modules.append("out_proj")
                else:
                    audio_target_modules.append(mod)

            mode = str(self.finetune_mode).strip().lower().replace("-", "_")
            lora_config_kwargs = {
                "r": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": audio_target_modules,
                "bias": "none",
                "task_type": TaskType.FEATURE_EXTRACTION,
            }
            if mode == "dora":
                lora_config_kwargs["use_dora"] = True

            try:
                audio_lora_config = LoraConfig(**lora_config_kwargs)
                self.audio_encoder = get_peft_model(self.audio_encoder, audio_lora_config)
                adapter_name = "DoRA" if mode == "dora" else "LoRA"
                print(f"[Audio Encoder {adapter_name}] Applied {adapter_name} to {encoder_type} encoder")
                print(f"[Audio Encoder {adapter_name}] Target modules: {audio_target_modules}")
            except TypeError as e:
                if mode == "dora":
                    raise ValueError("finetune_mode='dora' for audio encoder requires a PEFT version with DoRA support.") from e
                raise
        

        
        if self.encoder_type is not None:
            # encoder_hidden_size already set before PEFT wrapping
            self.audio_encoder = self.audio_encoder.to(self.model_dtype).to(self.model.device)
            # Use a projection sized for the external encoder (hidden_size -> LM hidden)
            self.audio_encoder_projection = VoxtralModalProjector_Encoder_Audio(
                input_dim=self.encoder_hidden_size,
                out_dim=classification_hidden_size,
            ).to(self.model_dtype).to(self.model.device)
            self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(375)


        self._set_audio_trainable()

        # freeze_llm must be set as an instance attribute externally (e.g. model.freeze_llm = True)
        # before calling this method; defaults to False (no additional freeze).
        if getattr(self, "freeze_llm", False):
            for p in self.model.language_model.model.parameters():
                p.requires_grad = False

        lm_t, lm_n = self._count_params(self.model.language_model)
        at_t, at_n = (self._count_params(self.model.audio_tower) if hasattr(self.model, "audio_tower") else (0, 0))
        proj_t, proj_n = (self._count_params(self.model.multi_modal_projector) if hasattr(self.model, "multi_modal_projector") else (0, 0))
        total_t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_n = sum(p.numel() for p in self.parameters())

        print(f"[Trainable Params]")
        print(f"  LM:                 {lm_t:>12,} / {lm_n:>14,}  ({100*lm_t/lm_n:.2f}%)")
        print(f"  Whisper:            {at_t:>12,} / {at_n:>14,}  ({100*at_t/at_n:.2f}%)" if at_n else "  Whisper:            N/A")
        print(f"  Projector:          {proj_t:>12,} / {proj_n:>14,}  ({100*proj_t/proj_n:.2f}%)" if proj_n else "  Projector:          N/A")
        print(f"  TOTAL:              {total_t:>12,} / {total_n:>14,}  ({100*total_t/total_n:.4f}%)")

        self._trainable_params_summary = {
            "trainable_lm": lm_t, "total_lm": lm_n,
            "trainable_whisper": at_t, "total_whisper": at_n,
            "trainable_projector": proj_t, "total_projector": proj_n,
            "trainable_total": total_t, "total_params": total_n,
            "trainable_pct": round(100 * total_t / total_n, 4) if total_n else 0,
        }


    def _set_audio_trainable(self) -> None:
        """
        Apply the trainable_audio_encoder flag to the relevant sub-modules.

        Modes (built-in Whisper audio_tower, encoder_type=None)
        --------------------------------------------------------
        "trainable" / True  : full fine-tuning of all audio_tower parameters.
        "frozen"    / False : audio_tower fully frozen (no adapters).
        "dora"              : base weights frozen; only DoRA/LoRA adapter params trainable.

        Modes (external encoder, encoder_type in {wav2vec2, hubert, wavlm})
        ---------------------------------------------------------------------
        True   : full fine-tuning of all external audio_encoder parameters.
        False  : PEFT adapters were injected during __init__; requires_grad is
                 managed by PEFT (base weights frozen, adapter params trainable).
                 This method does NOT override requires_grad in this case.

        Multimodal projector
        --------------------
        Its requires_grad pattern depends on self.projector_peft:
          projector_peft=False (legacy): every projector parameter is set trainable —
            full end-to-end training of the audio→LM bridge.
          projector_peft=True:  only adapter parameters (names containing 'lora_' or
            'dora_') are trainable; the underlying linear_1/linear_2 base weights stay
            frozen at their pretrained values. This is the parameter-efficient option
            for the projector and the only one consistent with the get_peft_model wrap
            performed earlier in __init__.

        The audio_encoder_projection (used only when an external encoder is attached)
        is unaffected by projector_peft and remains trainable.
        """
        _v = self.trainable_audio_encoder
        audio_mode = str(_v).strip().lower() if isinstance(_v, str) else ("trainable" if _v else "frozen")
        if self.encoder_type is None:
            if hasattr(self.model, "audio_tower"):
                if audio_mode == "dora":
                    for name, p in self.model.audio_tower.named_parameters():
                        p.requires_grad = "lora_" in name or "dora_" in name
                    status = "DoRA adapters only (base weights frozen)"
                elif audio_mode == "trainable":
                    for p in self.model.audio_tower.parameters():
                        p.requires_grad = True
                    status = "TRAINABLE (full fine-tuning)"
                else:  # "frozen"
                    for p in self.model.audio_tower.parameters():
                        p.requires_grad = False
                    status = "FROZEN"
                print(f"[Audio Encoder] Voxtral audio_tower (Whisper): {status}")
            if hasattr(self.model, "multi_modal_projector"):
                if self.projector_peft:
                    for name, p in self.model.multi_modal_projector.named_parameters():
                        p.requires_grad = ("lora_" in name) or ("dora_" in name)
                else:
                    for p in self.model.multi_modal_projector.parameters():
                        p.requires_grad = True
        else:
            if self.audio_encoder is not None:
                # trainable_audio_encoder=True: full fine-tuning.
                # trainable_audio_encoder=False: PEFT manages requires_grad; don't override.
                if self.trainable_audio_encoder:
                    for p in self.audio_encoder.parameters():
                        p.requires_grad = True
            if hasattr(self, "audio_encoder_projection"):
                for p in self.audio_encoder_projection.parameters():
                    p.requires_grad = True

    @staticmethod
    def _count_params(module: nn.Module) -> tuple:
        """Return (trainable_params, total_params) for *module*."""
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        return trainable, total


    def compute_audio_embeds(
        self,
        input_features=None,
        input_ids=None,
        raw_audio_input=None,
        raw_audio_lengths=None,
    ):
        """
        Pre-compute audio embeddings for caching between multiple forward passes.

        Call once per batch before scoring bonafide/spoof candidates to avoid
        running the Whisper encoder (or external encoder) twice on identical audio.

        Returns audio_embeds tensor [B, T_audio, H], or None if inputs are absent.
        """
        audio_embeds = None

        if input_features is not None and input_ids is not None and self.encoder_type is None:
            _v = self.trainable_audio_encoder
            _audio_mode = str(_v).strip().lower() if isinstance(_v, str) else ("trainable" if _v else "frozen")
            if _audio_mode in ("trainable", "dora"):
                audio_outputs = self.model.audio_tower(input_features)
            else:
                with torch.no_grad():
                    audio_outputs = self.model.audio_tower(input_features)
            audio_hidden_states = audio_outputs.last_hidden_state          # [B, T, D]
            B, T, D = audio_hidden_states.shape
            Tt = (T // self.k_downsample) * self.k_downsample
            x = audio_hidden_states[:, :Tt, :].contiguous().view(
                B, Tt // self.k_downsample, self.k_downsample * D
            )
            x = x.view(-1, self.k_downsample * D)
            audio_embeds = self.model.multi_modal_projector(x)             # [B*(T/k), H]
            audio_embeds = audio_embeds.view(B, Tt // self.k_downsample, -1)  # [B, T/k, H]

        elif (
            input_features is not None
            and input_ids is not None
            and self.encoder_type is not None
            and raw_audio_input is not None
        ):
            raw_audio_input = raw_audio_input.to(self.model.device).to(self.model_dtype)
            is_peft_audio_encoder = hasattr(self.audio_encoder, "peft_config")
            _v2 = self.trainable_audio_encoder
            _ext_mode = str(_v2).strip().lower() if isinstance(_v2, str) else ("trainable" if _v2 else "frozen")
            enable_grad = (_ext_mode in ("trainable", "dora")) or is_peft_audio_encoder
            batch_size = len(raw_audio_lengths)
            raw_audio_lengths_t = torch.as_tensor(raw_audio_lengths, device=raw_audio_input.device)
            mask = (
                torch.arange(raw_audio_input.size(1), device=raw_audio_input.device)[None, :]
                < raw_audio_lengths_t[:, None]
            )
            if enable_grad:
                audio_hidden_states = self.audio_encoder(raw_audio_input, attention_mask=mask).last_hidden_state
            else:
                with torch.no_grad():
                    audio_hidden_states = self.audio_encoder(raw_audio_input, attention_mask=mask).last_hidden_state
            audio_hidden_states = audio_hidden_states.transpose(1, 2)                # [B, H_enc, T]
            audio_hidden_states = self.adaptive_avg_pool(audio_hidden_states)        # [B, H_enc, 375]
            audio_hidden_states = (
                audio_hidden_states.transpose(1, 2).contiguous().view(-1, self.encoder_hidden_size)
            )
            audio_embeds = self.audio_encoder_projection(audio_hidden_states)        # [B*375, H]
            audio_embeds = audio_embeds.view(batch_size, -1, audio_embeds.size(-1))

        if audio_embeds is None and input_features is not None and input_ids is not None:
            if self.encoder_type is not None and raw_audio_input is None:
                raise ValueError(
                    "compute_audio_embeds: encoder_type is set but raw_audio_input is None. "
                    "Provide raw_audio_input when using an external encoder."
                )

        return audio_embeds

    def forward(
        self,
        input_ids=None,
        raw_audio_lengths=None,
        raw_audio_input=None,
        input_features=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        precomputed_audio_embeds=None,
        **kwargs,
    ):
        """
        Classification-as-generation forward pass.
        - If precomputed_audio_embeds is provided (pre-cached from compute_audio_embeds()),
          audio encoding is skipped entirely — avoids running Whisper twice per batch.
        - Otherwise falls back to computing audio embeddings inline.
        - Inserts audio embeddings at audio token positions, then runs the language model.
        - Returns dict with logits, loss (if present), and raw LM output for downstream scoring.
        """
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Use cached audio embeds when available; compute otherwise.
        if precomputed_audio_embeds is not None:
            audio_embeds = precomputed_audio_embeds
        else:
            audio_embeds = self.compute_audio_embeds(
                input_features=input_features,
                input_ids=input_ids,
                raw_audio_input=raw_audio_input,
                raw_audio_lengths=raw_audio_lengths,
            )

        if audio_embeds is None:
            raise ValueError("audio_embeds was not computed for CAG forward; check inputs.")

        audio_embeds = audio_embeds.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        audio_token_mask = input_ids == self.model.config.audio_token_id

        if not audio_token_mask.any():
            raise ValueError(
                "No audio token positions found in input_ids. "
                "Audio embeddings cannot be injected into the sequence. "
                "Check that the tokenizer properly inserts audio placeholder tokens."
            )


        tokens_per_sample = audio_token_mask.sum(dim=1)          # (B,)
        embeds_per_sample = torch.full(
            (input_ids.size(0),), audio_embeds.size(1),
            dtype=tokens_per_sample.dtype, device=tokens_per_sample.device,
        )
        mismatch = (tokens_per_sample != embeds_per_sample).nonzero(as_tuple=False)
        if mismatch.numel() > 0:
            b = int(mismatch[0].item())
            raise ValueError(
                f"Sample {b}: audio token positions ({int(tokens_per_sample[b].item())}) "
                f"!= audio embed frames ({int(embeds_per_sample[b].item())}). "
                f"Check processor config and k_downsample={self.k_downsample}."
            )

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[audio_token_mask] = audio_embeds.reshape(-1, audio_embeds.size(-1))

        if position_ids is None:
            seq_len = inputs_embeds.size(1)
            position_ids = (
                torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(inputs_embeds.size(0), -1)
            )

        # Default to no KV cache: the CAG path never consumes past_key_values, and
        # use_cache=None would fall back to the LM's own text config (usually True) —
        # allocating a full DynamicCache per eval forward that is immediately discarded.
        # (Setting model.config.use_cache on the top-level VoxtralConfig does NOT
        # propagate here because language_model is called directly.)
        if use_cache is None:
            use_cache = False

        lm_out = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
            labels=labels,
            **kwargs,
        )

        if isinstance(lm_out, dict):
            lm_logits = lm_out.get("logits", None)
            if lm_logits is None:
                lm_logits = lm_out.get("lm_logits", None)
            lm_loss = lm_out.get("loss", None)
            if lm_loss is None:
                lm_loss = lm_out.get("lm_loss", None)
        else:
            lm_logits = getattr(lm_out, "logits", None)
            lm_loss = getattr(lm_out, "loss", None)

        return {
            "lm_logits": lm_logits,
            "lm_loss": lm_loss,
            "raw_output": lm_out,
        }
