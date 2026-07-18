"""
utils/build_model_for_classification_head.py
=============================================
Model construction for the Audio-Adapter + MLP Classification Head paradigm.

This module is the counterpart to build_model.py for the alternative training
paradigm where a lightweight MLP head replaces the LM's generation scoring.
The language model is always frozen and can be dropped from GPU memory entirely
(drop_language_model=True) to save VRAM.

Key exports
-----------
AudioAdapterHeadClassifier
    nn.Module: audio encoder → multi_modal_projector → pool → MLP head → logits.
    Supports four audio encoder tuning modes: frozen / finetune / lora / dora.
    Supports external encoders (wavlm, wav2vec2, hubert) or the built-in Whisper tower.

build_classification_head_model()
    Factory that calls build_voxtral_model(), then wraps the result in
    AudioAdapterHeadClassifier with the desired tuning and head configuration.

Paradigm invariants
-------------------
- The LM (language_model) is frozen and not part of the forward pass.
- Classification labels are integer class indices (bonafide=0, spoof=1).
- Checkpoints saved here are NOT loadable into VoxtralStageWrapper_classification
  (different state_dict key namespaces) — strict=True enforces this.
- drop_language_model=True deletes the LM after wrapping; any code that later
  accesses model.language_model must guard with hasattr().
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers.activations import GELUActivation

from utils.build_model import (
    build_voxtral_model,
    compute_class_weights_from_dataset,
    VoxtralModalProjector_Encoder_Audio,
    CACHE_DIR,
)

__all__ = [
    "build_classification_head_model",
    "compute_class_weights_from_dataset",
    "AudioAdapterHeadClassifier",
]


class AudioAdapterHeadClassifier(nn.Module):
    """
    Wraps a Voxtral base model with a lightweight MLP classification head for spoof detection,
    bypassing the language model entirely.

    Architecture:
      audio encoder → multi_modal_projector (adapter) → pool → MLP head → num_classes logits

    Language model: always frozen and optionally offloaded to CPU (drop_language_model=True)
    to reclaim GPU memory when the LM is not needed.

    Audio encoder tuning modes (audio_encoder_tuning):
      - "frozen"   : all encoder weights frozen; only the adapter + head are trained.
      - "finetune" : full encoder fine-tuning (high VRAM cost).
      - "lora"     : LoRA adapters injected into the encoder; base weights frozen.
      - "dora"     : DoRA adapters (magnitude + direction decomposition); base weights frozen.

    Encoder type (encoder_type):
      - None / "none" : uses the Voxtral internal Whisper-based audio tower.
      - "wavlm" / "wav2vec2" / "hubert" : loads an external HuggingFace encoder instead.

    Pooling strategy (pooling_strategy): how adapter time steps are collapsed before the head.
      - "mean" : average across time (default).
      - "max"  : max across time.
      - "last" : take the last time step.

    Head: two-layer MLP with GELU activation, dropout, and a final linear → num_classes.
    """

    def __init__(
        self,
        base_model,
        encoder_type=None,
        audio_encoder_tuning="frozen",
        k_downsample=4,
        num_classes=2,
        pooling_strategy="mean",
        head_hidden_dim=256,
        head_dropout=0.1,
        train_audio_adapter=True,
        finetune_mode="lora",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
        drop_language_model=False,
    ):
        super().__init__()
        self.model = base_model
        # Dtype/device are read from the model's parameters rather than a .dtype/.device
        # attribute, since not all HF model wrappers expose those directly.
        self.model_dtype = next(base_model.parameters()).dtype
        self.device = getattr(self.model, "device", next(base_model.parameters()).device)
        self.k_downsample = k_downsample
        self.num_classes = num_classes
        self.pooling_strategy = str(pooling_strategy or "mean").lower()
        self.audio_encoder_tuning = str(audio_encoder_tuning or "frozen").lower()
        self.train_audio_adapter = bool(train_audio_adapter)
        self.trainable_audio_encoder = self.audio_encoder_tuning in ("finetune", "lora", "dora")
        self.drop_language_model = bool(drop_language_model)

        self.finetune_mode = finetune_mode
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules if lora_target_modules is not None else ["q_proj", "k_proj", "v_proj", "o_proj"]

        # Normalizes the "none" string (as it might come from a YAML config) to Python None,
        # meaning "use the internal Whisper tower".
        self.encoder_type = None
        if isinstance(encoder_type, str) and encoder_type.lower() != "none":
            self.encoder_type = encoder_type.lower()

        classification_hidden_size = self._resolve_hidden_size()
        if self.drop_language_model:
            self._remove_language_model_from_device()
        self.audio_encoder = None
        self.encoder_hidden_size = None
        self.adaptive_avg_pool = None

        # ============================================================================
        # STEP 1: Setup external audio encoder (if using wavlm/wav2vec2/hubert)
        # ============================================================================
        self._setup_external_encoder(classification_hidden_size)

        # ============================================================================
        # STEP 2: Apply LoRA/DoRA to internal Whisper (if encoder_type is None)
        # ============================================================================
        self._apply_lora_to_internal_whisper()

        # Simple MLP classification head (optional hidden layer)
        self.classification_head = self._build_classification_head(
            classification_hidden_size,
            head_hidden_dim,
            head_dropout,
            num_classes,
        )

        # Freeze language model; this path only optimizes audio stack + head
        if hasattr(self.model, "language_model") and self.model.language_model is not None:
            for p in self.model.language_model.parameters():
                p.requires_grad = False

        # Respect adapter training flag
        if hasattr(self.model, "multi_modal_projector"):
            for p in self.model.multi_modal_projector.parameters():
                p.requires_grad = self.train_audio_adapter

        # Configure audio encoder training mode
        self._configure_audio_trainability()

    def _resolve_hidden_size(self):
        """
        Resolve the LM hidden size needed to size the MLP classification head's input dimension.

        The MLP head connects to the audio adapter output, which is projected to the LM's
        hidden size (hidden_size in the text config). We read this from the config object
        rather than from live parameters so the LM can be offloaded or deleted before this
        call returns.

        Tries config locations in priority order:
          1. lm.model.config.hidden_size  (nested HF model)
          2. lm.config.hidden_size        (top-level LM config)
          3. model.config.text_config.hidden_size  (multimodal wrapper)
          4. model.config.hidden_size     (fallback)

        Raises ValueError if none of the above yield a value.
        """
        lm = getattr(self.model, "language_model", None)
        cfg_candidates = []
        if lm is not None:
            if hasattr(lm, "model") and hasattr(lm.model, "config"):
                cfg_candidates.append(lm.model.config)
            if hasattr(lm, "config"):
                cfg_candidates.append(lm.config)
        model_cfg = getattr(self.model, "config", None)
        if model_cfg is not None:
            cfg_candidates.append(getattr(model_cfg, "text_config", None))
            cfg_candidates.append(model_cfg)

        for cfg in cfg_candidates:
            if cfg is None:
                continue
            hidden_size = getattr(cfg, "hidden_size", None)
            if hidden_size is not None:
                return hidden_size

        raise ValueError("Unable to infer language model hidden size for classification head.")

    def _remove_language_model_from_device(self):
        """
        Offload the language model to CPU and delete its reference to free GPU memory.

        Used when drop_language_model=True — in adapter-head mode the LM is never called
        during forward, so keeping it on GPU wastes VRAM that could go to larger batches.

        Removes up to four references that HuggingFace multimodal wrappers may hold:
          - model.language_model         (top-level)
          - model.model.language_model   (nested base_model wrapper)
          - model.lm_head                (tied output projection)
          - model.model.lm_head          (nested variant)

        Each removal first moves to CPU (so PyTorch can release CUDA memory), then tries
        delattr; if that fails due to the module being a registered attribute, sets it to
        None instead. Calls torch.cuda.empty_cache() if anything was removed.
        """
        def _offload_and_delete(parent: nn.Module, attr_name: str) -> bool:
            mod = getattr(parent, attr_name, None)
            if mod is None:
                return False
            try:
                mod.to("cpu")
            except Exception:
                pass
            try:
                delattr(parent, attr_name)
                return True
            except Exception:
                try:
                    setattr(parent, attr_name, None)
                    return True
                except Exception:
                    return False

        removed = False
        # Remove top-level LM references (common in HF wrappers)
        removed |= _offload_and_delete(self.model, "language_model")
        # Remove nested LM references (e.g., base_model.model.language_model)
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Module):
            removed |= _offload_and_delete(self.model.model, "language_model")
        # Drop LM heads if present to avoid dangling params
        removed |= _offload_and_delete(self.model, "lm_head")
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Module):
            removed |= _offload_and_delete(self.model.model, "lm_head")

        if removed:
            torch.cuda.empty_cache()

    # ================================================================================
    # NEW MODULAR METHODS FOR AUDIO ENCODER SETUP AND LORA/DORA APPLICATION
    # ================================================================================
    
    def _build_lora_config(self, target_modules: list) -> LoraConfig:
        """
        Build a LoRA/DoRA configuration for audio encoders.
        
        Args:
            target_modules: List of module names to apply adapters to (e.g., ["q_proj", "k_proj", "v_proj", "out_proj"])
        
        Returns:
            LoraConfig configured for feature extraction with optional DoRA support based on self.finetune_mode.
        
        Note:
            - If finetune_mode is "dora", DoRA (Weight-Decomposed Low-Rank Adaptation) is enabled.
            - DoRA typically provides better performance than LoRA by decomposing weights into magnitude and direction.
        """
        mode = str(self.finetune_mode).strip().lower().replace("-", "_")
        
        lora_config_kwargs = {
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": target_modules,
            "bias": "none",
            "task_type": TaskType.FEATURE_EXTRACTION,
        }
        
        # Enable DoRA if finetune_mode is "dora"
        if mode == "dora":
            lora_config_kwargs["use_dora"] = True
        
        return LoraConfig(**lora_config_kwargs)
    
    def _map_target_modules_for_audio(self) -> list:
        """
        Map LoRA target module names to audio encoder naming conventions.
        
        Whisper and other audio encoders use "out_proj" instead of "o_proj" for output projection.
        This method translates the standard LLM module names to audio encoder equivalents.
        
        Returns:
            List of module names adapted for audio encoders.
        """
        audio_target_modules = []
        for mod in self.lora_target_modules:
            # Whisper/Audio encoders use "out_proj" instead of "o_proj"
            if mod == "o_proj":
                audio_target_modules.append("out_proj")
            else:
                audio_target_modules.append(mod)
        return audio_target_modules
    
    def _setup_external_encoder(self, classification_hidden_size: int):
        """
        Load and configure an external audio encoder (WavLM, Wav2Vec2, or HuBERT).
        
        This method is called when encoder_type is set to one of the supported external encoders.
        It handles:
            1. Loading the pretrained encoder model
            2. Applying LoRA/DoRA adapters if audio_encoder_tuning == "lora"
            3. Setting up the projection layer to match classification head dimensions
        
        Args:
            classification_hidden_size: Target dimension for the audio encoder projection output.
        
        Note:
            External encoders are useful when you want to use a different audio representation
            than the internal Whisper encoder (audio_tower). This is common for tasks where
            self-supervised speech representations (WavLM, Wav2Vec2) perform better.
        """
        # ---- Load external encoder based on encoder_type ----
        if self.encoder_type == "wav2vec2":
            from transformers import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-xls-r-300m", cache_dir=CACHE_DIR
            )
        elif self.encoder_type == "hubert":
            from transformers import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(
                "facebook/hubert-large-ls960-ft", cache_dir=CACHE_DIR
            )
        elif self.encoder_type == "wavlm":
            from transformers import WavLMModel
            self.audio_encoder = WavLMModel.from_pretrained(
                "microsoft/wavlm-large", cache_dir=CACHE_DIR
            )
        
        # ---- Apply LoRA/DoRA if configured ----
        if self.audio_encoder is not None:
            self.encoder_hidden_size = self.audio_encoder.config.hidden_size
            
            if self.audio_encoder_tuning in ("lora", "dora"):
                audio_target_modules = self._map_target_modules_for_audio()
                lora_config = self._build_lora_config(audio_target_modules)
                
                adapter_name = "DoRA" if "dora" in str(self.finetune_mode).lower() else "LoRA"
                print(f"[External Encoder {adapter_name}] Applying {adapter_name} to {self.encoder_type}")
                print(f"[External Encoder {adapter_name}] Target modules: {audio_target_modules}")
                
                self.audio_encoder = get_peft_model(self.audio_encoder, lora_config)
                try:
                    self.audio_encoder.print_trainable_parameters()
                except Exception:
                    pass
            
            # Move to correct device and dtype
            self.audio_encoder = self.audio_encoder.to(self.model_dtype).to(self.device)
            
            # Projection layer: external encoder dim -> classification hidden dim
            self.audio_encoder_projection = VoxtralModalProjector_Encoder_Audio(
                input_dim=self.encoder_hidden_size,
                out_dim=classification_hidden_size,
            ).to(self.model_dtype).to(self.device)
            
            # Adaptive pooling to match expected sequence length
            # 375 is the number of audio tokens Voxtral produces per 30s chunk (Whisper's 1500
            # encoder frames downsampled by k_downsample=4); keeps external-encoder output aligned.
            self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(375)
    
    def _apply_lora_to_internal_whisper(self):
        """
        Apply LoRA/DoRA adapters to the internal Whisper audio_tower.
        
        This method is called when:
            - encoder_type is None (using internal Whisper from the Voxtral model)
            - audio_encoder_tuning is "lora" or "dora"

        This method is a no-op when:
            - encoder_type is set (external encoder handles its own LoRA in _setup_external_encoder)
            - audio_encoder_tuning is "frozen" (nothing to inject)
            - audio_encoder_tuning is "finetune" (full fine-tuning, no adapters needed)
        
        Key Design Decision:
            When drop_language_model=True, the base model's LLM is NOT wrapped in PEFT adapters
            (finetune_mode="frozen" is used for build_voxtral_model). This saves ~3B parameters
            from being wrapped in PeftModel, which would consume massive GPU memory.
            
            Instead, DoRA/LoRA is applied ONLY to the audio_tower here, which is what we actually
            want to train. This is the key optimization that reduces memory from >80GB to ~10-15GB.
        
        Whisper Module Naming:
            Whisper uses "out_proj" instead of "o_proj" for the output projection in attention layers.
            This is handled by _map_target_modules_for_audio().
        
        Raises:
            ValueError: If audio_tower doesn't exist on the model when this method is called.
        """
        # Only apply if using internal Whisper (encoder_type is None) and LoRA/DoRA mode is enabled
        if self.encoder_type is not None or self.audio_encoder_tuning not in ("lora", "dora"):
            return
        
        if not hasattr(self.model, "audio_tower") or self.model.audio_tower is None:
            raise ValueError(
                "audio_encoder_tuning='lora' with encoder_type=None requires model.audio_tower to exist. "
                "Make sure the base model has an audio_tower component."
            )
        
        # Get the audio tower (Whisper encoder)
        audio_tower = self.model.audio_tower
        
        # Map module names for Whisper architecture
        whisper_target_modules = self._map_target_modules_for_audio()
        
        # Build and apply LoRA/DoRA config
        lora_config = self._build_lora_config(whisper_target_modules)
        
        adapter_name = "DoRA" if "dora" in str(self.finetune_mode).lower() else "LoRA"
        print(f"[Internal Whisper {adapter_name}] Applied {adapter_name} to audio_tower")
        print(f"[Internal Whisper {adapter_name}] Target modules: {whisper_target_modules}")
        print(f"[Internal Whisper {adapter_name}] LoRA rank: {self.lora_rank}, alpha: {self.lora_alpha}")
        
        # Wrap audio_tower in PEFT model
        self.model.audio_tower = get_peft_model(audio_tower, lora_config)
        
        # Enable gradient checkpointing to save memory (PEFT wrappers may hide the base encoder)
        candidates = []
        candidates.append(self.model.audio_tower)
        if hasattr(self.model.audio_tower, "get_base_model"):
            try:
                candidates.append(self.model.audio_tower.get_base_model())
            except Exception:
                pass
        if hasattr(self.model.audio_tower, "base_model"):
            candidates.append(getattr(self.model.audio_tower, "base_model"))
        if hasattr(self.model.audio_tower, "model"):
            candidates.append(getattr(self.model.audio_tower, "model"))

        enabled = False
        for cand in candidates:
            if cand is None:
                continue
            if hasattr(cand, "gradient_checkpointing_enable"):
                try:
                    cand.gradient_checkpointing_enable()
                    enabled = True
                except Exception:
                    pass
            if hasattr(cand, "config") and hasattr(cand.config, "use_cache"):
                try:
                    cand.config.use_cache = False
                except Exception:
                    pass
            if hasattr(cand, "enable_input_require_grads"):
                try:
                    cand.enable_input_require_grads()
                except Exception:
                    pass

        if enabled:
            print(f"[Internal Whisper {adapter_name}] Gradient checkpointing enabled (base encoder)")
        else:
            print(f"[Internal Whisper {adapter_name}] Could not enable gradient checkpointing on audio_tower")
        
        try:
            self.model.audio_tower.print_trainable_parameters()
        except Exception:
            pass

    # ================================================================================
    # END OF NEW MODULAR METHODS
    # ================================================================================

    @staticmethod
    def _build_classification_head(hidden_size: int, head_hidden_dim: int, head_dropout: float, num_classes: int) -> nn.Module:
        """
        Build a lightweight MLP head.
        - If head_hidden_dim <= 0: single linear layer to num_classes.
        - Else: Linear -> GELU -> Dropout -> Linear.
        """
        head_layers = []
        if head_hidden_dim is not None and head_hidden_dim > 0:
            head_layers.append(nn.Linear(hidden_size, head_hidden_dim))
            head_layers.append(GELUActivation())
            head_layers.append(nn.Dropout(head_dropout))
            head_layers.append(nn.Linear(head_hidden_dim, num_classes))
        else:
            head_layers.append(nn.Linear(hidden_size, num_classes))
        return nn.Sequential(*head_layers)

    def _iter_audio_params(self):
        """Yield (name, param) for whichever audio encoder is active."""
        if self.encoder_type is None and hasattr(self.model, "audio_tower"):
            yield from self.model.audio_tower.named_parameters()
        elif self.audio_encoder is not None:
            yield from self.audio_encoder.named_parameters()
        else:
            return

    def _configure_audio_trainability(self):
        """
        Set requires_grad according to audio_encoder_tuning.
        - finetune: all audio params trainable
        - frozen: all audio params frozen
        - lora: only LoRA params trainable
        """
        mode = self.audio_encoder_tuning
        if mode == "finetune":
            for _, p in self._iter_audio_params():
                p.requires_grad = True
        elif mode == "frozen":
            for _, p in self._iter_audio_params():
                p.requires_grad = False
        # PEFT names its injected weight matrices with "lora_A"/"lora_B" in the parameter path
        # (DoRA's magnitude params follow the same convention), so this substring check works for both.
        elif mode in ("lora", "dora"):
            params = list(self._iter_audio_params())
            found_lora = False
            for name, p in params:
                if "lora" in name.lower():
                    p.requires_grad = True
                    found_lora = True
                else:
                    p.requires_grad = False
            if not found_lora:
                raise ValueError(f"audio_encoder_tuning='{mode}' was requested, but no LoRA parameters were found on the audio encoder.")

    def _compute_internal_audio_embeds(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings from the internal Whisper audio_tower.
        Uses the model's multi_modal_projector after downsampling.
        """
        if input_features is None:
            raise ValueError("input_features required for internal audio tower.")
        grad_enabled = self.training and self.audio_encoder_tuning in ("finetune", "lora", "dora")
        with torch.set_grad_enabled(grad_enabled):
            # When audio_tower is wrapped in PeftModel, must use keyword argument
            # Groups every k_downsample consecutive Whisper frames into one vector (matching
            # Voxtral's audio-token merge ratio) before the projector; Tt drops any trailing partial group.
            audio_outputs = self.model.audio_tower(input_features=input_features)
            audio_hidden_states = audio_outputs.last_hidden_state
            B, T, D = audio_hidden_states.shape
            Tt = (T // self.k_downsample) * self.k_downsample
            x = audio_hidden_states[:, :Tt, :].contiguous().view(B, Tt // self.k_downsample, self.k_downsample * D)
            x = x.view(-1, self.k_downsample * D)
            audio_embeds = self.model.multi_modal_projector(x)
            audio_embeds = audio_embeds.view(B, Tt // self.k_downsample, -1)
            return audio_embeds

    def _compute_external_audio_embeds(self, raw_audio_input: torch.Tensor, raw_audio_lengths) -> torch.Tensor:
        """
        Compute embeddings from an external encoder (wavlm/wav2vec2/hubert).
        Uses adaptive pooling + projection to match classification hidden size.
        """
        if raw_audio_input is None or raw_audio_lengths is None:
            raise ValueError("raw_audio_input and raw_audio_lengths required for external encoder path.")
        raw_audio_input = raw_audio_input.to(self.device).to(self.model_dtype)
        batch_size = len(raw_audio_lengths)
        raw_audio_lengths_t = torch.as_tensor(raw_audio_lengths, device=raw_audio_input.device)
        mask = (torch.arange(raw_audio_input.size(1), device=raw_audio_input.device)[None, :] < raw_audio_lengths_t[:, None])
        grad_enabled = self.training and self.audio_encoder_tuning in ("finetune", "lora", "dora")
        with torch.set_grad_enabled(grad_enabled):
            audio_hidden_states = self.audio_encoder(raw_audio_input, attention_mask=mask).last_hidden_state
        # Transpose to (B, H, T) so AdaptiveAvgPool1d can pool over the time axis to a fixed
        # 375 steps, then transpose back and flatten for the projector.
        audio_hidden_states = audio_hidden_states.transpose(1, 2)
        audio_hidden_states = self.adaptive_avg_pool(audio_hidden_states)
        audio_hidden_states = audio_hidden_states.transpose(1, 2).contiguous().view(-1, self.encoder_hidden_size)
        audio_embeds = self.audio_encoder_projection(audio_hidden_states)
        audio_embeds = audio_embeds.view(batch_size, -1, audio_embeds.size(-1))
        return audio_embeds

    def _compute_audio_embeds(self, input_features=None, raw_audio_input=None, raw_audio_lengths=None):
        """
        Dispatch to internal or external encoder based on encoder_type.
        """
        if self.encoder_type is None:
            return self._compute_internal_audio_embeds(input_features)
        if self.audio_encoder is not None:
            return self._compute_external_audio_embeds(raw_audio_input, raw_audio_lengths)
        raise ValueError("No audio encoder available to compute embeddings.")

    def _pool_audio(self, audio_embeds):
        """
        Pool time dimension of audio embeddings to a single vector.
        """
        if self.pooling_strategy == "max":
            return torch.max(audio_embeds, dim=1).values
        if self.pooling_strategy == "last":
            return audio_embeds[:, -1, :]
        # default mean pooling
        return torch.mean(audio_embeds, dim=1)

    def forward(
        self,
        input_features=None,
        raw_audio_input=None,
        raw_audio_lengths=None,
        labels=None,
        return_audio_embeds: bool = False,
        return_pooled: bool = False,
        **kwargs,
    ):
        """
        Adapter-head forward pass:
        - compute audio embeddings
        - pool to a vector
        - classify with MLP head
        """
        audio_embeds = self._compute_audio_embeds(
            input_features=input_features,
            raw_audio_input=raw_audio_input,
            raw_audio_lengths=raw_audio_lengths,
        )
        pooled = self._pool_audio(audio_embeds)
        logits = self.classification_head(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        # By default we return only logits/loss to avoid keeping large tensors alive.
        # Enable return_audio_embeds/return_pooled explicitly for debugging or analysis.
        out = {"logits": logits, "loss": loss}
        if return_audio_embeds:
            out["audio_embeds"] = audio_embeds
        if return_pooled:
            out["pooled"] = pooled
        return out
            
         
    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        raw_audio_input=None,
        raw_audio_lengths=None,  
        input_features=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        **kwargs
    ):
        """
        Generation path for classification-as-generation.
        - Prepares audio embeddings and scatters them into audio token placeholders.
        - Calls language model (PEFT-aware) to generate logits/loss/outputs under inference mode.
        - Returns dict mirroring forward: logits, loss (if present), and raw output.
        """
        if self.drop_language_model or not hasattr(self.model, "language_model") or self.model.language_model is None:
            raise RuntimeError("Language model is disabled for this adapter-head wrapper; generation is unavailable.")
        audio_embeds = None

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # get_audio_features() is the base HF model's own method, which already runs the audio
        # tower plus multi_modal_projector internally (unlike the manual steps used elsewhere).
        if input_features is not None and input_ids is not None and self.encoder_type is None:
            with torch.enable_grad() if self.trainable_audio_encoder else torch.no_grad():
                audio_embeds = self.model.get_audio_features(input_features)
            
        if input_features is not None and input_ids is not None and self.encoder_type is not None and raw_audio_input is not None:
            raw_audio_input = raw_audio_input.to(self.model.device).to(self.model_dtype)
            # Enable gradients if: full fine-tuning OR using LoRA/DoRA (PEFT model needs gradients)
            is_peft_audio_encoder = hasattr(self.audio_encoder, "peft_config")
            enable_grad = self.trainable_audio_encoder or is_peft_audio_encoder
            with torch.enable_grad() if enable_grad else torch.no_grad():
                batch_size = len(raw_audio_lengths)
                raw_audio_lengths_t = torch.as_tensor(raw_audio_lengths, device=raw_audio_input.device)
                mask = (torch.arange(raw_audio_input.size(1), device=raw_audio_input.device)[None, :] < raw_audio_lengths_t[:, None])
                audio_hidden_states = self.audio_encoder(raw_audio_input,attention_mask=mask).last_hidden_state         # [B, time_steps, encoder_hidden_size]
                audio_hidden_states = audio_hidden_states.transpose(1, 2)                                               # [B, encoder_hidden_size, time_steps]
                audio_hidden_states = self.adaptive_avg_pool(audio_hidden_states)                                       # [B, encoder_hidden_size, 375]
                audio_hidden_states = (audio_hidden_states.transpose(1, 2).contiguous().view(-1, self.encoder_hidden_size))
                audio_embeds = self.audio_encoder_projection(audio_hidden_states)                                       # [B*375, 3072]
                audio_embeds = audio_embeds.view(batch_size, -1, audio_embeds.size(-1))

        if audio_embeds is None:
            raise ValueError("audio_embeds was not computed for generation; check inputs.")

        audio_embeds = audio_embeds.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        audio_token_mask = input_ids == self.model.config.audio_token_id

        # Validate that audio tokens exist in the input
        if not audio_token_mask.any():
            import warnings
            warnings.warn(
                "No audio token positions found in input_ids during generate(). "
                "Audio embeddings will not be injected into the sequence.",
                RuntimeWarning
            )

        # Clone inputs_embeds to avoid modifying the original tensor (defensive programming)
        inputs_embeds = inputs_embeds.clone()

        # External-encoder embeddings can have a different token count per sample, so each is
        # sliced to its own count in a loop; Whisper's count is uniform, so masked_scatter suffices.
        if self.encoder_type is not None: #wav2vec, hubert, etc.
            B, T, D = inputs_embeds.shape
            for b in range(B):
                n = audio_token_mask[b].sum()
                inputs_embeds[b, audio_token_mask[b]] = audio_embeds[b, :n]

        else: #whisper model
            # replace text-audio token placeholders with audio embeddings
            audio_token_mask_scatter = (input_ids == self.model.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask_scatter.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        if hasattr(self.model, "language_model"):
            lm_ref = self.model.language_model
            is_peft_lm = hasattr(lm_ref, "peft_config") or "peft" in lm_ref.__class__.__name__.lower()
        else:
            lm_ref = None
            is_peft_lm = False

        if is_peft_lm:
            lm_out = self.model.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        else:
            lm_out = self.model.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        
        if isinstance(lm_out, dict):
            # Checks both the standard HF output keys and Voxtral-wrapper-specific alternates so
            # this works across different LM output classes.
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


# Treats "none"/"null" strings (as they might appear in a YAML/JSON config) the same
# as Python None, letting config files disable the external-encoder path.
def _normalize_encoder_type(encoder_type: Optional[str]) -> Optional[str]:
    if encoder_type is None:
        return None
    if isinstance(encoder_type, str):
        enc = encoder_type.lower()
        if enc in ("none", "null"):
            return None
        return encoder_type
    return None


def build_classification_head_model(
    *,
    # Base model + PEFT options
    base_model_name: str,
    device: str,
    dtype: torch.dtype,
    finetune_mode: str = "lora",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[Sequence[str]] = None,
    use_4bit: bool = False,
    precision: str = "float16",
    load_model_path: Optional[str] = None,
    load_pretrained_audio_stack: bool = True,
    language_model_only_lora: bool = True,
    # Adapter-head specifics
    encoder_type: Optional[str] = None,
    audio_encoder_tuning: str = "frozen",
    k_downsample: int = 4,
    num_classes: int = 2,
    pooling_strategy: str = "mean",
    head_hidden_dim: int = 256,
    head_dropout: float = 0.1,
    train_audio_adapter: bool = True,
    drop_language_model: bool = False,
) -> AudioAdapterHeadClassifier:
    """
    Build a Voxtral base model and wrap it with the adapter-head classifier.
    
    This is the main entry point for creating a classification model that uses
    only the audio encoder (Whisper or external) without the full LLM.
    
    Memory Optimization Strategy (when drop_language_model=True):
    ================================================================
    Problem:
        The Voxtral model contains a ~3B parameter LLM that is not needed for 
        audio classification. If we apply LoRA/DoRA to the LLM before dropping it,
        we waste ~80GB of GPU memory on parameters we'll delete anyway.
    
    Solution:
        1. Load the base model with finetune_mode="frozen" (no PEFT on LLM)
        2. Load on CPU so the LLM never touches GPU memory
        3. Remove the LLM reference (in AudioAdapterHeadClassifier.__init__)
        4. Apply DoRA/LoRA ONLY to the audio encoder
        5. Move the remaining model to GPU
    
    This reduces memory usage from 80GB to ~10-15GB.
    
    Args:
        base_model_name: HuggingFace model name (e.g., "mistralai/Voxtral-Mini-3B-2507")
        device: Target device for the final model (e.g., "cuda:0")
        dtype: Model precision (e.g., torch.bfloat16)
        finetune_mode: "lora", "dora", or "frozen" - determines adapter type for audio encoder
        lora_rank: LoRA/DoRA rank (higher = more parameters, more capacity)
        lora_alpha: LoRA/DoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: Which modules to apply adapters to (e.g., ["q_proj", "k_proj", "v_proj", "o_proj"])
        use_4bit: Whether to use 4-bit quantization for base model
        precision: "float16" or "bfloat16"
        load_model_path: Optional path to a saved checkpoint
        load_pretrained_audio_stack: Whether to load pretrained weights
        language_model_only_lora: If True, apply PEFT only to LLM (ignored when drop_language_model=True)
        encoder_type: None (internal Whisper), "wavlm", "wav2vec2", or "hubert"
        audio_encoder_tuning: "frozen", "finetune", or "lora" (for audio encoder training mode)
        k_downsample: Downsampling factor for audio features
        num_classes: Number of output classes for classification
        pooling_strategy: "mean", "max", or "last" for temporal pooling
        head_hidden_dim: Hidden dimension of the MLP classification head
        head_dropout: Dropout probability in the classification head
        train_audio_adapter: Whether to train the multi_modal_projector
        drop_language_model: Remove LLM to save VRAM (CRITICAL for memory optimization)
    
    Returns:
        AudioAdapterHeadClassifier model ready for training
    
    Example Configuration (YAML):
        finetune_mode: 'dora'
        audio_adapter_head:
            drop_language_model: True
            audio_encoder_tuning: 'lora'  # This will apply DoRA to Whisper
    """
    # ==========================================================================
    # STEP 1: Determine loading strategy based on drop_language_model
    # ==========================================================================

    if drop_language_model and not load_pretrained_audio_stack and not use_4bit:
        # -----------------------------------------------------------------------
        # FAST PATH: meta-device — the LLM is never allocated (0 CPU/GPU RAM).
        #
        # When all weights come from a checkpoint (load_pretrained_audio_stack=False)
        # and the LLM will be dropped anyway, allocating 6 GB of CPU RAM for Mistral
        # just to immediately delete it is pure waste.  Using torch.device("meta")
        # creates a zero-cost skeleton; we delete the LLM references from that
        # skeleton (still cost-free), then to_empty(device) materialises only the
        # ~1.5 GB audio stack directly on the target device.
        # -----------------------------------------------------------------------
        from transformers import VoxtralForConditionalGeneration as _VFC, AutoConfig as _AC
        print(f"[Memory Optimization] drop_language_model=True + load_pretrained_audio_stack=False:")
        print(f"  - Meta-device fast path: LLM never allocated (saves ~6 GB CPU RAM)")
        print(f"  - Audio stack (~1.5 GB) materialised directly on {device}")

        # Resolve local snapshot path so AutoConfig never pings the HF Hub.
        # Passing a repo ID triggers a remote check even with HF_HUB_OFFLINE=1
        # (same issue as MistralCommonTokenizer — see _parse_inference_config).
        from pathlib import Path as _Path
        _cfg_load_path = base_model_name
        _ref_file = _Path(CACHE_DIR) / ("models--" + base_model_name.replace("/", "--")) / "refs" / "main"
        if _ref_file.is_file():
            _commit = _ref_file.read_text().strip()
            _snapshot = _Path(CACHE_DIR) / ("models--" + base_model_name.replace("/", "--")) / "snapshots" / _commit
            if _snapshot.is_dir():
                _cfg_load_path = str(_snapshot)
        _cfg = _AC.from_pretrained(_cfg_load_path, cache_dir=CACHE_DIR)
        with torch.device("meta"):
            _meta_base = _VFC(_cfg)

        # Delete LLM and lm_head from meta — cost-free, no actual tensors exist yet.
        # Mirror the same four attribute paths that _remove_language_model_from_device uses.
        def _meta_drop(parent, attr):
            if parent is not None and hasattr(parent, attr):
                try:
                    delattr(parent, attr)
                except Exception:
                    try:
                        setattr(parent, attr, None)
                    except Exception:
                        pass

        for _attr in ("language_model", "lm_head"):
            _meta_drop(_meta_base, _attr)
            _nested = getattr(_meta_base, "model", None)
            if isinstance(_nested, nn.Module):
                _meta_drop(_nested, _attr)

        # Materialise only the remaining audio modules (audio_tower + multi_modal_projector)
        # on the target device, then cast to the requested dtype.
        base_model = _meta_base.to_empty(device=device).to(dtype)
        print(f"[Memory Optimization] Audio stack on {device}.", flush=True)

    else:
        # -----------------------------------------------------------------------
        # ORIGINAL PATH
        # -----------------------------------------------------------------------
        # When we plan to drop the LLM, load on CPU first so it never touches GPU.
        base_load_device = "cpu" if drop_language_model else device

        # When dropping LLM, use 'frozen' mode so the 3B LLM is NOT wrapped in PEFT.
        base_finetune_mode = "frozen" if drop_language_model else finetune_mode

        if drop_language_model:
            print(f"[Memory Optimization] drop_language_model=True:")
            print(f"  - Loading base model on: {base_load_device}")
            print(f"  - Base model finetune_mode: '{base_finetune_mode}' (LLM not wrapped in PEFT)")
            print(f"  - Audio encoder finetune_mode: '{finetune_mode}' (DoRA/LoRA on audio encoder)")

        # ==========================================================================
        # STEP 2: Load the base Voxtral model
        # ==========================================================================
        base_model = build_voxtral_model(
            finetune_mode=base_finetune_mode,  # "frozen" when dropping LLM
            base_model_name=base_model_name,
            dtype=dtype,
            device=base_load_device,  # CPU when dropping LLM
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            use_4bit=use_4bit,
            precision=precision,
            load_model_path=load_model_path,
            load_pretrained_base=load_pretrained_audio_stack,
            language_model_only_lora=language_model_only_lora,
        )

    # ==========================================================================
    # STEP 3: Normalize encoder type and create classifier wrapper
    # ==========================================================================
    encoder_type = _normalize_encoder_type(encoder_type)

    # AudioAdapterHeadClassifier receives the ORIGINAL finetune_mode for audio encoder DoRA
    model = AudioAdapterHeadClassifier(
        base_model,
        encoder_type=encoder_type,
        audio_encoder_tuning=audio_encoder_tuning,
        k_downsample=k_downsample,
        num_classes=num_classes,
        pooling_strategy=pooling_strategy,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        train_audio_adapter=train_audio_adapter,
        finetune_mode=finetune_mode,  # ORIGINAL mode for audio encoder DoRA
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        drop_language_model=drop_language_model,
    )
    
    # ==========================================================================
    # STEP 4: Move to target device
    # ==========================================================================
    return model.to(device)
