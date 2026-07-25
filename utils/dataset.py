"""
utils/dataset.py
================
Dataset, batch-sampler, and DataLoader factory for Voxtral ASVspoof training.

Key classes
-----------
ASVspoofIndexDataset
    Supports ASVspoof5, ASVspoof2019 LA, and ASVspoof2021 LA/DF protocols.
    Builds complete input_ids sequences (BOS/INST/audio-tokens/prompt/ENDINST)
    and runs Whisper feature extraction, optional augmentation, and padding.
    Label convention: bonafide=0, spoof=1 (pos_label=1 for EER computation).

BalancedBatchSampler
    Yields batches with equal bonafide/spoof halves; cycles shorter class.
    Uses torch.randperm(generator=...) for reproducibility.

EnsureAtLeastOneBonafideBatchSampler
    Guarantees at least one bonafide sample per batch by replacing one item
    when a randomly-drawn batch is all-spoof.

make_dataloader()
    Factory that wires a dataset to the appropriate sampler and DataLoader.

Padding modes
-------------
pad_waveform_mode  — none | tile_repeat | tile_repeat_aug | tile_wav | tile_wav_aug
pad_features_mode  — zero | repeat_last | repeat_last_aug | tile | tile_aug
"""
from __future__ import annotations

import itertools
import math
import os
import random
import warnings
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoProcessor

# Prefer the ffmpeg backend when installed, since codec augmentation and general
# audio loading downstream rely on ffmpeg's broader codec support.
print(f"Available torchaudio backends: {torchaudio.list_audio_backends()}")
if "ffmpeg" in torchaudio.list_audio_backends():
    print("Setting torchaudio backend to 'ffmpeg'")
    torchaudio.set_audio_backend("ffmpeg")

print(transformers.__version__)

try:
    from config import *
    from augmentation import augment_waveform, augment_spectrogram
    from config_helper_functions import resolve_label_texts

except ImportError:
    from utils.augmentation import augment_waveform, augment_spectrogram

    try:
        from utils.config import *
    except ValueError:
        pass  # VOXTRAL_CONFIG not set; inference callers pass all params explicitly
    from utils.augmentation import augment_waveform  # type: ignore
    from utils.config_helper_functions import resolve_label_texts



# Value used to pad mel spectrograms when masking; ~-1.0 matches the typical silence floor in Whisper-style normalized log-mel.
#SILENCE_PAD_VALUE = -1.0
# NOTE: These module-level constants must match the processor's feature_extractor.
# ASVspoofIndexDataset overrides them with self._sample_rate / self._n_samples derived
# from the processor at construction time.
SAMPLE_RATE = 16_000
N_SAMPLES = SAMPLE_RATE * 30  # 30s window like Whisper
_AUDIO_DECODE_WARNED = set()


def _validate_waveform_normalization_mode(mode: str) -> str:
    mode = str(mode or "none").lower().strip()
    if mode not in {"none", "minmax"}:
        raise ValueError(f"Unsupported waveform_normalization mode {mode!r}; supported: none, minmax")
    return mode


def _apply_waveform_normalization(waveform: torch.Tensor, mode: str) -> torch.Tensor:
    mode = _validate_waveform_normalization_mode(mode)
    if mode == "none":
        return waveform
    # "minmax" mode rescales the waveform into [0, 1] using its own min/max;
    # clamp_min below avoids a divide-by-zero for constant/silent signals.
    w_min = waveform.min()
    w_max = waveform.max()
    denom = (w_max - w_min).clamp_min(1e-8)
    return (waveform - w_min) / denom


def _load_audio_with_fallback(path: str):
    """
    Load audio with torchcodec fallback handling.
    - Attempts torchaudio.load_with_torchcodec, then torchaudio.load on failure.
    - Warns once per path when falling back; raises RuntimeError if all decoders fail.
    """
    last_err = None
    for name, loader in (
        ("torchcodec", lambda p: torchaudio.load_with_torchcodec(p)),
        ("torchaudio.load", lambda p: torchaudio.load(p)),
    ):
        try:
            return loader(path)
        except Exception as exc:
            last_err = exc
            if path not in _AUDIO_DECODE_WARNED:
                warnings.warn(f"{name} failed for {path}: {exc}. Trying fallback decoder.", RuntimeWarning)
                _AUDIO_DECODE_WARNED.add(path)
    raise RuntimeError(f"Failed to decode audio at {path}. Last error: {last_err}") from last_err

def pad_or_trim_waveform(waveform: torch.Tensor, target_len: int = N_SAMPLES) -> torch.Tensor:
    """
    Zero-pad or trim waveform to fixed length.
    - Ensures channel dimension; truncates or pads with zeros to target_len.
    - Returns tensor shaped [C, target_len].
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    C, N = waveform.shape
    if N > target_len:
        waveform = waveform[:, :target_len]
    elif N < target_len:
        pad = waveform.new_zeros((C, target_len - N))
        waveform = torch.cat([waveform, pad], dim=1)
    return waveform


def tile_waveform_to_length(waveform: torch.Tensor, target_len: int = N_SAMPLES) -> torch.Tensor:
    """
    Tile waveform until reaching target_len.
    - Repeats audio content, trimming any excess; preserves channel dimension.
    - Returns tensor shaped [C, target_len].
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    C, N = waveform.shape
    if N >= target_len:
        return waveform[:, :target_len]
    reps = math.ceil(target_len / N)
    return waveform.repeat(1, reps)[:, :target_len]


def _estimate_valid_feat_len(valid_samples: int, hop: int, t_cap: int) -> int:
    """
    Convert valid sample count to feature frames with cap.
    - Computes ceil(valid_samples / hop) and clamps to t_cap.
    """
    return min(t_cap, int(math.ceil(valid_samples / float(hop))))

def _repeat_pad_features(input_features: torch.Tensor,
                         T_valid: int,
                         mode: str = "repeat_last",
                         pad_aug: bool = False,
                         aug_fn=None,
                         aug_prob: float = 0) -> torch.Tensor:
    """
    Pad spectrogram-like features beyond valid length with optional augmentation.
    - Supports repeat_last/tile modes; `_aug` suffix or pad_aug applies aug_fn on padded spans.
    - aug_fn operates on [F, T_chunk]; aug_prob controls probability per chunk.
    - Returns input unchanged when T_valid is out of range or no padding needed.
    """
    F, T_feat = input_features.shape
    if T_valid >= T_feat or T_valid <= 0:
        return input_features

    valid = input_features[:, :T_valid].contiguous()
    rem = T_feat - T_valid
    if rem <= 0:
        return input_features

    use_aug = pad_aug or mode.endswith("_aug")
    base_mode = mode.replace("_aug", "")

    def maybe_aug(x: torch.Tensor) -> torch.Tensor:
        if use_aug and aug_fn is not None and (aug_prob >= 1.0 or random.random() < aug_prob):
            return aug_fn(x)
        return x

    if base_mode == "repeat_last":
        pad = valid[:, -1:].expand(F, rem)
        pad = maybe_aug(pad)
        return torch.cat([valid, pad], dim=1)

    if base_mode == "tile":
        n_full, n_tail = divmod(rem, T_valid)
        parts = []
        for _ in range(n_full):
            chunk = maybe_aug(valid)
            parts.append(chunk)
        if n_tail:
            chunk = maybe_aug(valid[:, :n_tail])
            parts.append(chunk)
        pad = torch.cat(parts, dim=1) if parts else valid[:, :0]
        return torch.cat([valid, pad], dim=1)

    return input_features

def _tile_waveform_with_aug(
    waveform: torch.Tensor,       # [C, N] or [N]
    samples_needed: int,
    sr: int = 16000,
    do_aug: bool = False,
    aug_prob: float = 1.0,
    aug_cfg=None,
    aug_fn=augment_waveform,
    waveform_augments_enabled: bool = True,
) -> torch.Tensor:
    """
    Tile waveform with optional augmentation until reaching samples_needed.
    - Repeats base chunk, optionally applying aug_fn with aug_prob on each appended piece.
    - Ensures output shape [C, samples_needed], trimming excess at the end.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, N]
    C, N = waveform.shape
    if N == 0:
        raise ValueError("Waveform is empty; cannot tile to a target length.")
    if N >= samples_needed:
        return waveform[:, :samples_needed]

    pieces = [waveform]
    total = N
    base = waveform

    # Tile the base chunk repeatedly, each time applying augmentation with probability aug_prob
    while total < samples_needed:
        chunk = base
        if do_aug and aug_fn is not None and (aug_prob >= 1.0 or random.random() < aug_prob):
            # Important: work on a copy to avoid corrupting the base chunk
            chunk = aug_fn(
                waveform=base.clone(),
                sr=sr,
                config=aug_cfg,
                waveform_augments_enabled=waveform_augments_enabled,
            )
            # Preserve [C, T] dimensions
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)
        pieces.append(chunk)
        total += chunk.shape[-1]

    out = torch.cat(pieces, dim=-1)
    return out[:, :samples_needed]



# ========== BatchSamplers ==========

# Batches start out plainly shuffled, then get patched after the fact: any batch with
# no bonafide sample has one spoof slot swapped for a bonafide one.
class EnsureAtLeastOneBonafideBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset, batch_size: int, drop_last: bool = False, generator=None):
        if len(dataset.bonafide_indices) == 0:
            raise ValueError("No bonafide samples in dataset; cannot ensure presence per batch.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        n = len(self.dataset)
        all_idx = list(range(n))
        if self.generator is not None:
            perm = torch.randperm(len(all_idx), generator=self.generator).tolist()
        else:
            perm = torch.randperm(len(all_idx)).tolist()
        all_idx = [all_idx[i] for i in perm]
        # Separately shuffled pool of bonafide indices to draw replacements from when a batch
        # turns out all-spoof; bon_ptr cycles through it round-robin across batches.
        bon_set = set(self.dataset.bonafide_indices)
        bon_pool = self.dataset.bonafide_indices.copy()
        if self.generator is not None:
            perm = torch.randperm(len(bon_pool), generator=self.generator).tolist()
        else:
            perm = torch.randperm(len(bon_pool)).tolist()
        bon_pool = [bon_pool[i] for i in perm]
        bon_ptr = 0

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch = all_idx[start:end]
            has_bon = any(i in bon_set for i in batch)
            if not has_bon:
                b = bon_pool[bon_ptr]
                bon_ptr = (bon_ptr + 1) % len(bon_pool)
                if len(batch) == 0:
                    batch = [b]
                else:
                    # Replace one item to guarantee a bonafide sample in the batch
                    if b not in batch:
                        if self.generator is not None:
                            replace_idx = int(torch.randint(len(batch), (1,), generator=self.generator).item())
                        else:
                            replace_idx = int(torch.randint(len(batch), (1,)).item())
                        batch[replace_idx] = b
            yield batch

    def __len__(self):
        n = len(self.dataset)
        return (n // self.batch_size) if self.drop_last else math.ceil(n / self.batch_size)


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset, batch_size: int, drop_last: bool = False, generator=None):
        if batch_size < 2:
            raise ValueError("BalancedBatchSampler requires batch_size >= 2.")
        if len(dataset.bonafide_indices) == 0 or len(dataset.spoof_indices) == 0:
            raise ValueError("Both classes must exist to build balanced batches.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.extra = batch_size - 2 * self.half  # 0 if even batch size, 1 if odd
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        bon = self.dataset.bonafide_indices.copy()
        spf = self.dataset.spoof_indices.copy()
        if self.generator is not None:
            perm = torch.randperm(len(bon), generator=self.generator).tolist()
        else:
            perm = torch.randperm(len(bon)).tolist()
        bon = [bon[i] for i in perm]
        if self.generator is not None:
            perm = torch.randperm(len(spf), generator=self.generator).tolist()
        else:
            perm = torch.randperm(len(spf)).tolist()
        spf = [spf[i] for i in perm]
        # Cyclic iterators let the smaller class wrap around and repeat across the epoch so
        # every batch still gets its half; num_batches below covers the larger class fully.
        cyc_bon = itertools.cycle(bon)
        cyc_spf = itertools.cycle(spf)

        if self.drop_last:
            num_batches = min(len(bon) // self.half, len(spf) // self.half)
        else:
            num_batches = max(math.ceil(len(bon) / self.half), math.ceil(len(spf) / self.half))

        for _ in range(num_batches):
            batch = [next(cyc_bon) for _ in range(self.half)] + [next(cyc_spf) for _ in range(self.half)]
            if self.extra == 1:
                if len(bon) >= len(spf):
                    batch.append(next(cyc_bon))
                else:
                    batch.append(next(cyc_spf))
            perm = torch.randperm(len(batch), generator=self.generator).tolist()
            batch = [batch[i] for i in perm]
            yield batch

    def __len__(self):
        bon = len(self.dataset.bonafide_indices)
        spf = len(self.dataset.spoof_indices)
        if self.drop_last:
            return min(bon // self.half, spf // self.half)
        return max(math.ceil(bon / self.half), math.ceil(spf / self.half))


# ========== DATASET ==========
class ASVspoofIndexDataset(Dataset):
    """
        Initializes the ASVspoofIndexDataset instance.

        This constructor sets up the dataset configuration, including file paths, 
        label mappings (bonafide=0, spoof=1), and data augmentation strategies. 
        It also retrieves special token IDs (e.g., [INST], [AUDIO]) from the 
        processor's tokenizer to prepare inputs for multi-modal language models.

        Args:
            path_to_protocol (str): Path to the protocol file containing file lists and labels.
            data_dir (str): Directory containing the raw audio files.
            dataset_type (str): The type of dataset (e.g., 'train', 'dev', 'eval').
            processor (object): The processor/tokenizer object used for text and audio encoding.
            prompts (list/dict): A collection of text prompts to be used with the audio.
            is_train (bool, optional): Whether this is a training dataset. Defaults to True.
            augmentation_on (bool, optional): Whether to apply audio augmentations. Defaults to True.
            reasoning (bool, optional): Whether to include reasoning steps in the output. Defaults to False.
             SPECIAL_TOKEN_TASK (str, optional): A special token used to denote the task type.
            dtype (type, optional): Data type for the tensors.
            CONFIG_AUGMENTATION (dict, optional): Configuration parameters for augmentation.
            AUGMENTATION_PROB (float, optional): Probability of applying augmentation. Defaults to 0.9.
            encoder_type (str, optional): Type of audio encoder. If not None, waveform is kept.
            
            pad_features_mode (str, optional): Strategy for padding features. 
            'zero' pads with silence; 
            (pad_waveform_mode='none' with pad_features_mode='zero' mirrors Whisper’s default: 
            trim/pad waveform to 30s and silence-pad the mel/features (no tiling or aug).)
            'repeat_last' repeats the final valid frame; 
            'repeat_last_aug' does the same but can augment the repeated tail; 
            'tile' tiles valid frames to fill; 
            'tile_aug' tiles with augmentation on the tiled chunks.
            
            pad_waveform_mode (str, optional): Strategy for padding waveforms. 
            'none' = standard trim/zero-pad; 
            'tile_wav' tiles waveform only when features are short; 
            'tile_wav_aug' tiles with augmentation on each tile; 
            'tile_repeat' always repeats waveform to N_SAMPLES without per-repeat aug; 
            'tile_repeat_aug' repeats to N_SAMPLES with augmentation applied per repeat.
    """
    def __init__(self, path_to_protocol,data_dir, dataset_type, processor, prompts, AUGMENTATION_PROB,
                 is_train=True, augmentation_on=True, reasoning=False,
                 SPECIAL_TOKEN_TASK = "<SPECIAL_20>", dtype=None, 
                 CONFIG_AUGMENTATION = None, 
                 encoder_type = None, pad_features_mode = 'zero', pad_waveform_mode = 'none',
                 constant_prompts_flag = False, three_random_prompts: bool = False,
                 random_prompt_variants: Optional[list[str]] = None,
                 fixed_prompt_text: Optional[str] = None,
                 constant_prompt_override: Optional[str] = None,
                 label_texts: Optional[dict] = None,
                 N_audio_tokens = 375,
                 augmentation_only_spoof: bool = False,
                 wavlm_waveform_normalization: str = "none",
                 whisper_waveform_normalization: str = "none"):
        self.path_to_protocol = path_to_protocol
        self.data_dir = data_dir
        self.processor = processor
        # Ensure tokenizer has a pad token (Mistral/Voxtral ships with pad_token_id=None)
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            import warnings
            warnings.warn("pad_token_id was None; set to eos_token_id. Labels are padded with -100 so EOS-as-pad is safe for the label stream, but verify input_ids masking.")
        # Right-padding is required for causal LM training (attention mask convention)
        if self.processor.tokenizer.padding_side != "right":
            import warnings
            warnings.warn(f"padding_side is {self.processor.tokenizer.padding_side!r}; expected 'right' for training. Generation eval may need 'left'.")
        if prompts is None or len(prompts) == 0:
            raise ValueError("Prompts list is empty; set a valid prompts_path in the YAML or VOXTRAL_PROMPTS_PATH.")
        self.prompts = prompts
        self.is_train = is_train
        self.augmentation_on = augmentation_on
        self.reasoning = reasoning
        self.use_three_random_prompts = bool(three_random_prompts)
        self.mapping = {
            "bonafide": 0,
            "spoof": 1,
        }
        # Map numeric class index -> output text (e.g. {0: "real", 1: "fake"})
        _lt = resolve_label_texts(label_texts or {}, context_label="Dataset")
        # normalized protocol label -> output text used for tokenization
        self._label_output_map = {norm: _lt[idx] for norm, idx in self.mapping.items()}

        self.SPECIAL_TOKEN_TASK = SPECIAL_TOKEN_TASK 
        self.dtype = dtype
        self.CONFIG_AUGMENTATION = CONFIG_AUGMENTATION
        self.AUGMENTATION_PROB = AUGMENTATION_PROB
        self.augmentation_only_spoof = bool(augmentation_only_spoof)
        # Normalize encoder choice; treat "none"/None as built-in Voxtral tower
        if isinstance(encoder_type, str):
            encoder_type = encoder_type.lower()
            if encoder_type == "none":
                encoder_type = None
        self.encoder_type = encoder_type
        self.keep_waveform = self.encoder_type is not None
        self.dataset_type = dataset_type
        self.PAD_FEATURES_MODE = pad_features_mode  # 'zero', "repeat_last", "repeat_last_aug", "tile", "tile_aug"
        self.PAD_WAVEFORM_MODE = pad_waveform_mode
        self.N_audio_tokens = N_audio_tokens
        # Two independent normalization policies: one for the waveform fed to an optional
        # secondary encoder (e.g. WavLM), another for the waveform used for Whisper mel features.
        self.wavlm_waveform_normalization = _validate_waveform_normalization_mode(wavlm_waveform_normalization)
        self.whisper_waveform_normalization = _validate_waveform_normalization_mode(whisper_waveform_normalization)
        
        self._tile_waveform_aug_per_repeat = (pad_waveform_mode == "tile_repeat_aug")
        self._hop = getattr(getattr(self.processor, "feature_extractor", object()), "hop_length", 160)
        # Derive sampling rate from the processor's feature extractor when available
        _fe = getattr(self.processor, "feature_extractor", None)
        _sr = getattr(_fe, "sampling_rate", None)
        if _sr is not None and _sr != SAMPLE_RATE:
            import warnings
            warnings.warn(f"Processor sampling_rate={_sr} differs from module-level SAMPLE_RATE={SAMPLE_RATE}. Using processor value.")
        self._sample_rate = _sr if _sr is not None else SAMPLE_RATE
        # chunk_length is the max audio duration (s) the feature extractor expects (30s for
        # Whisper); times sample rate gives the fixed sample count every waveform is padded/trimmed to.
        _chunk_len = getattr(_fe, "chunk_length", None)
        self._n_samples = int(self._sample_rate * _chunk_len) if _chunk_len is not None else (self._sample_rate * 30)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
        # lightweight per-worker cache to avoid re-running feature_extractor on the same file
        self._feat_cache = {}
        self._feat_cache_size = 128  # LRU size

        self.vocab = self.processor.tokenizer.get_vocab()
        self._bos_id       =  self.vocab.get("<s>", getattr(self.processor.tokenizer, "bos_token_id", None)) # Begin-of-sequence token. Starts a new sample for the language model.
        self._inst_id      =  self.vocab["[INST]"] # Wrapper that marks a single user instruction/turn. Everything between them is the user’s message for the decoder to condition on. Many Mistral-style chat templates require this exact framing.
        self._endinst_id   =  self.vocab["[/INST]"]
        self._begin_aud_id =  self.vocab.get("[BEGIN_AUDIO]")     # “the next part is audio.”
        self._aud_frame_id =  self.vocab["[AUDIO]"]
        self._T_rapped_audio_tokens = N_audio_tokens # number of audio tokens to use
        # static tensors reused per sample
        self._bos_inst_tensor = torch.tensor([self._bos_id , self._inst_id], dtype=torch.long)
        self._begin_aud_tensor = torch.tensor([self._begin_aud_id], dtype=torch.long) if self._begin_aud_id is not None else None
        # Fixed-length run of [AUDIO] placeholder tokens; the audio tower later fills these
        # positions with encoded audio embeddings instead of text embeddings.
        self._audio_block = torch.full((self._T_rapped_audio_tokens,), self._aud_frame_id, dtype=torch.long)
        self._labels = []
        self._bonafide_indices = []
        self._spoof_indices = []
        # caches to avoid repeated tokenization work
        self._prompt_id_cache = []
        self._three_prompt_id_cache = []
        self._label_token_cache = {}
        self._steps_token_cache = {}

        self._space_token = self.processor.tokenizer(" ", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.constant_prompts_flag = constant_prompts_flag
        self.constant_prompts_ids = []
        self.prompt_variants = list(random_prompt_variants) if random_prompt_variants else []
        self.fixed_prompt_text = fixed_prompt_text.strip() if isinstance(fixed_prompt_text, str) else None
        self.fixed_prompt_ids = None
        self.constant_prompt_override = constant_prompt_override.strip() if isinstance(constant_prompt_override, str) else None



        if self.path_to_protocol is not None:
            # All protocol files (ASVspoof5, ASVspoof2019/2021, In_The_Wild) are
            # whitespace-separated despite the .tsv extension on ASVspoof5's files
            with open(self.path_to_protocol, "r") as f:
                audio_info = [info.strip().split() for info in f.readlines()]
            self.audio_info = audio_info
        else:
            # Gracefully handle missing protocol by using an empty dataset
            self.audio_info = []
        
        # Pre-scans every protocol line once at construction time so the bonafide/spoof index
        # lists are ready for BalancedBatchSampler and EnsureAtLeastOneBonafideBatchSampler.
        for i in range(len(self.audio_info)):
            y = self._label_from_line(i)
            self._labels.append(y)
            if y == 0:
                self._bonafide_indices.append(i)
            else:
                self._spoof_indices.append(i)

        # pre-tokenize output label strings (uses label_texts mapping if provided)
        for norm_lbl, output_text in self._label_output_map.items():
            self._label_token_cache[norm_lbl] = self.processor.tokenizer(
                output_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]

        # pre-tokenize prompts once (classification-only)
        for p in self.prompts:
            prompt_text = f"{self.SPECIAL_TOKEN_TASK} {p} "
            ids = self.processor.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            self._prompt_id_cache.append(ids)
        
        if self.use_three_random_prompts:
            variants = self.prompt_variants if self.prompt_variants else self.prompts[:3]
            if not variants:
                raise ValueError("three_random_prompts is enabled but no prompt variants were provided.")
            for p in variants:
                prompt_text = f"{self.SPECIAL_TOKEN_TASK} {p} "
                ids = self.processor.tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                self._three_prompt_id_cache.append(ids)

        if self.fixed_prompt_text:
            prompt_text = f"{self.SPECIAL_TOKEN_TASK} {self.fixed_prompt_text} "
            self.fixed_prompt_ids = self.processor.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
        
        if self.constant_prompts_flag:
            _const_text = self.constant_prompt_override if self.constant_prompt_override else "Is this speech spoof or bonafide?"
            prompt_text = f"{self.SPECIAL_TOKEN_TASK} {_const_text} "
            self.constant_prompts_ids = self.processor.tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
       


    
    def _normalize_label_text(self, label: str) -> str:
        """Normalize protocol label variants to canonical bonafide/spoof strings."""
        lbl = label.lower()
        if lbl in ("bonafide", "bona-fide", "genuine"):
            return "bonafide"
        if lbl in ("spoof", "spoofed"):
            return "spoof"
        raise ValueError(f"Unsupported label text: {label}")

    def _label_from_line(self, idx: int) -> int:
        line = self.audio_info[idx]
        if self.dataset_type == "asvspoof5":
            if len(line) <= 8:
                raise IndexError(f"Malformed ASVspoof5 protocol line at index {idx}: expected at least 9 fields, got {len(line)}")
            lbl = line[8]
        elif self.dataset_type == "asvspoof2019" or (self.dataset_type == "asvspoof2021" and "2019" in self.path_to_protocol):
            lbl = line[-1]
        elif self.dataset_type == "asvspoof2021":
            if len(line) <= 5:
                raise IndexError(f"Malformed ASVspoof2021 protocol line at index {idx}: expected at least 6 fields, got {len(line)}")
            lbl = line[5]
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        normalized_label = self._normalize_label_text(lbl)
        return self.mapping[normalized_label]


    def _get_cached_feats(self, key):
        if key not in self._feat_cache:
            return None
        feats, t_valid = self._feat_cache[key]
        # move to end for LRU
        self._feat_cache.pop(key, None)
        self._feat_cache[key] = (feats, t_valid)
        return feats.clone(), t_valid

    def _put_cached_feats(self, key, feats, t_valid):
        # store on CPU to keep cache light
        self._feat_cache[key] = (feats.cpu(), t_valid)
        if len(self._feat_cache) > self._feat_cache_size:
            # drop oldest
            self._feat_cache.pop(next(iter(self._feat_cache)))


    def __len__(self):
        return len(self.audio_info)
    
    @property
    def bonafide_indices(self):
        return self._bonafide_indices

    @property
    def spoof_indices(self):
        return self._spoof_indices

    # ASVspoof5 protocol columns: speaker_id, file_id, gender, ..., codec (6),
    # attack_type (7), label (8).
    def _parse_line_asvspoof5(self, idx):
        audio_info_line = self.audio_info[idx]
        speaker_id = audio_info_line[0]
        file_id = audio_info_line[1]
        gender = audio_info_line[2]
        codec = audio_info_line[6]
        attack_type = audio_info_line[7]
        label = audio_info_line[8]  # spoof or bonafide
        label_text = self._normalize_label_text(label)
        steps = None
        audio_path = os.path.join(self.data_dir, file_id + ".flac")
        return audio_path, steps, label_text, file_id, speaker_id, gender, attack_type, codec

    # ASVspoof2019 protocol columns: speaker_id, file_id, ..., attack_type (second-to-last),
    # label (last).
    def _parse_line_asvspoof2019(self, idx):
        audio_info_line = self.audio_info[idx]
        speaker_id = audio_info_line[0]
        file_id = audio_info_line[1]
        attack_type = audio_info_line[-2]
        label = audio_info_line[-1]
        label_text = self._normalize_label_text(label)
        steps = None
        audio_path = os.path.join(self.data_dir, file_id + ".flac")

        return audio_path, steps, label_text, file_id, speaker_id, None, attack_type, None


    # ASVspoof2021 protocol columns: speaker_id, file_id, codec (2), ..., attack_type (4),
    # label (5).
    def _parse_line_asvspoof2021(self, idx):
        audio_info_line = self.audio_info[idx]
        speaker_id = audio_info_line[0]
        file_id = audio_info_line[1]
        codec = audio_info_line[2]
        attack_type = audio_info_line[4]
        label = audio_info_line[5]  # spoof or bonafide
        label_text = self._normalize_label_text(label)
        steps = None

        audio_path = os.path.join(self.data_dir, file_id + ".flac")
        return audio_path, steps, label_text, file_id, speaker_id, None, attack_type, codec
    
    def _parse_line_In_The_Wild(self, idx):
        audio_info_line = self.audio_info[idx]
        file_id = audio_info_line[0]
        speaker_id = audio_info_line[1]
        label = audio_info_line[2]
        label_text = self._normalize_label_text(label)
        steps = audio_info_line[3] if self.reasoning and len(audio_info_line) > 3 else None
        audio_path = os.path.join(self.data_dir, file_id + ".flac")

        return audio_path, steps, label_text, file_id, speaker_id, None, None, None

    def _parse_entry(self, idx):
        """Dispatch to the correct line parser based on dataset_type."""
        if self.dataset_type == "asvspoof5":
            return self._parse_line_asvspoof5(idx)
        if self.dataset_type == "asvspoof2019" or (self.dataset_type == "asvspoof2021" and "2019" in self.path_to_protocol):
            return self._parse_line_asvspoof2019(idx)
        if self.dataset_type == "asvspoof2021":
            return self._parse_line_asvspoof2021(idx)
        if self.dataset_type == "In_The_Wild":
            return self._parse_line_In_The_Wild(idx)
        raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _load_waveform(self, path):
        """Load audio, downmix to mono, and resample to the model sample rate."""
        waveform, sr = _load_audio_with_fallback(path)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self._sample_rate:
            resampler = self._resamplers.get(sr)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self._sample_rate)
                self._resamplers[sr] = resampler
            waveform = resampler(waveform)
            sr = self._sample_rate
        return waveform, sr

    def _maybe_normalize_waveform(self, waveform):
        """Apply the explicit WavLM waveform-normalization policy."""
        return _apply_waveform_normalization(waveform, self.wavlm_waveform_normalization)

    def _should_apply_augmentation(self) -> bool:
        """Decide once per sample whether to run any augmentation."""
        return (
            self.is_train
            and self.augmentation_on
            and self.CONFIG_AUGMENTATION is not None
            and random.random() < self.AUGMENTATION_PROB
        )

    def _maybe_augment_waveform(self, waveform, apply_aug: bool, disable_rawboost_codec: bool = False, waveform_augments_enabled: bool = True):
        """Optionally apply waveform-level augmentation during training."""
        if (
            not apply_aug
            or not self.augmentation_on
            or self.CONFIG_AUGMENTATION is None
            or not self.is_train
        ):
            return waveform
        return augment_waveform(
            waveform=waveform,
            sr=self._sample_rate,
            config=self.CONFIG_AUGMENTATION,
            force_disable_rawboost=disable_rawboost_codec,
            force_disable_codec=disable_rawboost_codec,
            waveform_augments_enabled=waveform_augments_enabled,
        )

    def _prepare_waveform(self, waveform, sr, pad_mode, apply_aug: bool, waveform_augments_enabled: bool = True):
        """Pad/trim or tile the waveform according to pad mode; track effective length."""
        orig_len = waveform.shape[-1]
        waveform_t = waveform
        sample_len = min(orig_len, self._n_samples)

        if pad_mode in ("none", "tile_repeat", "tile_repeat_aug"):
            if pad_mode in ("tile_repeat", "tile_repeat_aug"):
                do_aug = (
                    self._tile_waveform_aug_per_repeat
                    and self.augmentation_on
                    and self.CONFIG_AUGMENTATION is not None
                    and self.is_train
                    and apply_aug
                    and waveform_augments_enabled
                )
                if do_aug:
                    waveform_t = _tile_waveform_with_aug(
                        waveform=waveform_t,
                        samples_needed=self._n_samples,
                        sr=sr,
                        do_aug=True,
                        aug_prob=1.0,
                        aug_cfg=self.CONFIG_AUGMENTATION,
                        aug_fn=augment_waveform,
                        waveform_augments_enabled=waveform_augments_enabled,
                    )
                else:
                    waveform_t = tile_waveform_to_length(waveform_t, target_len=self._n_samples)
                sample_len = min(waveform_t.shape[-1], self._n_samples)
            else:
                waveform_t = pad_or_trim_waveform(waveform_t, target_len=self._n_samples)
                sample_len = min(orig_len, self._n_samples)

        return waveform_t, sample_len

    def _prepare_waveform_with_aug_strategy(self, waveform, sr, pad_mode, apply_aug: bool, waveform_augments_enabled: bool = True):
        """
        Keep the base waveform intact for tiling modes (augmentation happens per tile),
        and only apply augmentation upfront when we use a single copy.
        """
        tiling_modes = ("tile_repeat", "tile_repeat_aug", "tile_wav", "tile_wav_aug")
        if pad_mode in tiling_modes:
            return self._prepare_waveform(waveform, sr, pad_mode, apply_aug, waveform_augments_enabled)

        force_disable_rawboost_codec = (not self.is_train) or (not waveform_augments_enabled)
        waveform = self._maybe_augment_waveform(
            waveform,
            apply_aug=apply_aug,
            disable_rawboost_codec=force_disable_rawboost_codec,
            waveform_augments_enabled=waveform_augments_enabled,
        )
        return self._prepare_waveform(waveform, sr, pad_mode, apply_aug, waveform_augments_enabled)

    def _feature_cache_key(self, pad_mode, file_id):
        """Return cache key for features; disable cache when random repeat aug is active."""
        if self.is_train and self.augmentation_on:
            return None  # keep augmentation stochastic; never cache train features
        if pad_mode not in ("none", "tile_repeat", "tile_repeat_aug"):
            return None
        use_random_repeat_aug = (
            pad_mode in ("tile_repeat", "tile_repeat_aug")
            and self._tile_waveform_aug_per_repeat
            and self.augmentation_on
            and self.CONFIG_AUGMENTATION is not None
            and self.is_train
        )
        return None if use_random_repeat_aug else (file_id, self.whisper_waveform_normalization)

    def _extract_features(self, waveform_np, sr, sample_len):
        """Run feature extractor and compute number of valid frames."""
        feats = self.processor.feature_extractor(waveform_np, sampling_rate=sr, return_tensors="pt")
        input_features = feats["input_features"][0]
        T_valid = _estimate_valid_feat_len(sample_len, hop=self._hop, t_cap=input_features.shape[1])
        return input_features, T_valid

    def _get_features(self, waveform_np, sr, sample_len, cache_key):
        """Fetch features from cache when available, otherwise compute and store."""
        cached = self._get_cached_feats(cache_key) if cache_key is not None else None
        if cached is not None:
            return cached

        input_features, T_valid = self._extract_features(waveform_np, sr, sample_len)
        if cache_key is not None:
            self._put_cached_feats(cache_key, input_features, T_valid)
        return input_features, T_valid

    def _maybe_tile_waveform_features(self, waveform_t, waveform_np, sr, input_features, sample_len, T_valid, apply_aug: bool, waveform_augments_enabled: bool = True):
        """Tile waveform when feature length is short and tile mode is requested."""
        if getattr(self, "PAD_WAVEFORM_MODE", "none") not in ("tile_wav", "tile_wav_aug"):
            return waveform_t, waveform_np, input_features, sample_len, T_valid

        T_feat = input_features.shape[1]
        if T_valid >= T_feat:
            return waveform_t, waveform_np, input_features, sample_len, T_valid

        do_aug = (
            self.PAD_WAVEFORM_MODE == "tile_wav_aug"
            and self.is_train
            and self.augmentation_on
            and self.CONFIG_AUGMENTATION is not None
            and apply_aug
            and waveform_augments_enabled
        )
        waveform_t = _tile_waveform_with_aug(
            waveform=waveform_t,
            samples_needed=T_feat * self._hop,
            sr=sr,
            do_aug=do_aug,
            aug_prob=1.0 if do_aug else 0.0,
            aug_cfg=self.CONFIG_AUGMENTATION,
            aug_fn=augment_waveform,
            waveform_augments_enabled=waveform_augments_enabled,
        )
        sample_len = min(waveform_t.shape[-1], self._n_samples)
        waveform_np = waveform_t.squeeze().numpy()
        input_features, T_valid = self._extract_features(waveform_np, sr, sample_len)
        return waveform_t, waveform_np, input_features, sample_len, T_valid

    def _select_prompt_ids(self, idx):
        """Pick prompt ids (random for train) or use constant prompt when flagged."""
        if not self.is_train and self.fixed_prompt_ids is not None:
            return self.fixed_prompt_ids
        if self.use_three_random_prompts:
            cache = self._three_prompt_id_cache if self._three_prompt_id_cache else self._prompt_id_cache
            if self.is_train:
                return random.choice(cache)
            prompt_idx = idx % len(cache)
            return cache[prompt_idx]
        if self.constant_prompts_flag:
            return self.constant_prompts_ids
        # Default: select from cached prompts
        prompt_idx = random.randrange(len(self._prompt_id_cache)) if self.is_train else (idx % len(self._prompt_id_cache))
        return self._prompt_id_cache[prompt_idx]

    def _build_input_ids(self, prompt_ids):
        """Assemble BOS/INST, audio tokens, prompt, and ENDINST into one ids tensor."""
        pieces = [self._bos_inst_tensor]
        if self._begin_aud_tensor is not None:
            pieces.append(self._begin_aud_tensor)
        pieces.append(self._audio_block)
        pieces.append(prompt_ids)
        pieces.append(torch.tensor([self._endinst_id], dtype=torch.long))
        return torch.cat(pieces, dim=0)

    def _maybe_augment_features(self, input_features, T_valid, apply_aug: bool, allow_feature_augments: bool = True):
        """Optionally apply spec augmentation; respect valid region when no waveform tiling."""
        if (
            not apply_aug
            or not allow_feature_augments
            or not (self.augmentation_on and self.CONFIG_AUGMENTATION is not None and self.is_train)
        ):
            return input_features
        if not (self.CONFIG_AUGMENTATION.freq_mask or self.CONFIG_AUGMENTATION.time_mask):
            return input_features

        if getattr(self, "PAD_WAVEFORM_MODE", "none") == "none":
            aug_valid = input_features[:, :T_valid].contiguous()
            aug_valid = augment_spectrogram(
                aug_valid,
                config=self.CONFIG_AUGMENTATION,
                mask_fill=input_features.min(),
            )
            return torch.cat([aug_valid, input_features[:, T_valid:]], dim=1)

        return augment_spectrogram(
            input_features,
            config=self.CONFIG_AUGMENTATION,
            mask_fill=input_features.min(),
        )

    def _maybe_repeat_pad_features(self, input_features, T_valid, apply_aug: bool, allow_feature_augments: bool = True):
        """Pad mel tail via repeat/tile strategies instead of zeros when requested."""
        if getattr(self, "PAD_FEATURES_MODE", "zero") not in ("repeat_last", "repeat_last_aug", "tile", "tile_aug"):
            return input_features
        return _repeat_pad_features(
            input_features=input_features,
            T_valid=T_valid,
            mode=self.PAD_FEATURES_MODE,
            aug_fn=(lambda x: augment_spectrogram(x, config=self.CONFIG_AUGMENTATION, mask_fill=input_features.min())) if self.CONFIG_AUGMENTATION is not None else None,
            aug_prob=float(getattr(self, "AUGMENTATION_PROB", 1.0)) if (self.is_train and self.augmentation_on and apply_aug and allow_feature_augments) else 0.0,
        )

    def _build_labels(self, idx, steps, label):
        """Concatenate optional reasoning steps, a space, and the label tokens."""
        tokens_list = []
        if steps is not None:
            if idx not in self._steps_token_cache:
                self._steps_token_cache[idx] = self.processor.tokenizer(
                    steps, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
            tokens_list.append(self._steps_token_cache[idx])
            tokens_list.append(self._space_token)

        normalized_label = self._normalize_label_text(label)
        tokens_list.append(self._label_token_cache[normalized_label])
        return torch.cat(tokens_list, dim=0)

  
    def __getitem__(self, idx):
        path, steps, label, file_id, speaker_id, gender, attack_type, codec = self._parse_entry(idx)
        normalized_label = self._normalize_label_text(label)
        classification_labels = self.mapping[normalized_label]
        # When augmentation_only_spoof is enabled, bonafide (label 0) samples skip
        # augmentation entirely so the model always sees genuine audio unaltered.
        apply_aug = self._should_apply_augmentation()
        waveform_augments_enabled = not (self.augmentation_only_spoof and classification_labels == 0)
        # Mirror waveform logic: spectrogram augments also spoof-only when flag is set
        feature_augments_enabled = not (self.augmentation_only_spoof and classification_labels == 0)
        waveform, sr = self._load_waveform(path)
        pad_mode = getattr(self, "PAD_WAVEFORM_MODE", "none")
        waveform_t, sample_len = self._prepare_waveform_with_aug_strategy(
            waveform,
            sr,
            pad_mode,
            apply_aug,
            waveform_augments_enabled=waveform_augments_enabled,
        )

        prompt_ids = self._select_prompt_ids(idx)
        feature_waveform_t = _apply_waveform_normalization(
            waveform_t,
            self.whisper_waveform_normalization,
        )
        waveform_np = feature_waveform_t.squeeze().numpy()

        cache_key = self._feature_cache_key(pad_mode, file_id)
        input_features, T_valid = self._get_features(waveform_np, sr, sample_len, cache_key)
        feature_waveform_t, waveform_np, input_features, sample_len, T_valid = self._maybe_tile_waveform_features(
            feature_waveform_t, waveform_np, sr, input_features, sample_len, T_valid, apply_aug, waveform_augments_enabled=waveform_augments_enabled
        )

        input_ids = self._build_input_ids(prompt_ids)
        input_features = self._maybe_augment_features(input_features, T_valid, apply_aug, allow_feature_augments=feature_augments_enabled)
        input_features = self._maybe_repeat_pad_features(input_features, T_valid, apply_aug, allow_feature_augments=feature_augments_enabled)

        concat_steps_labels_tokens = self._build_labels(idx, steps, normalized_label)
        attention_mask = torch.ones_like(input_ids)
        # Raw audio for WavLM/wav2vec: use actual length only (no 30s zero-padding).
        raw_source_waveform_t = feature_waveform_t if pad_mode in ("tile_wav", "tile_wav_aug") else waveform_t
        raw_waveform_t = _apply_waveform_normalization(raw_source_waveform_t, self.wavlm_waveform_normalization)
        raw_audio_input = raw_waveform_t.squeeze()[:sample_len].float() if self.keep_waveform else torch.tensor([0.0])

        return {
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "input_features": input_features,
            "raw_audio_input": raw_audio_input,
            "raw_audio_lengths": sample_len,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": concat_steps_labels_tokens,
            "classification_labels": classification_labels,
            "file_id": file_id,
            "speaker_id": speaker_id,
            "gender": gender,
            "attack_type": attack_type,
            "codec": codec
        }
        

# ========== COLLATE FUNCTION ==========
def collate_fn(batch):
    """
    Collate batch items into padded tensors for training.
    - Pads input_ids/attention_mask to max length with pad_token_id; labels padded with -100.
    - Pads input_features to max T across the batch; pads raw_audio_input to max length.
    - Expects pad_token_id in each item; raises if missing.
    """
    ref_keys = set(batch[0].keys())
    for i, item in enumerate(batch[1:], 1):
        if set(item.keys()) != ref_keys:
            import warnings
            warnings.warn(f"collate_fn: sample {i} has different keys than sample 0. Missing: {ref_keys - set(item.keys())}. Extra: {set(item.keys()) - ref_keys}.")
    pad_token_id = batch[0].get("pad_token_id")
    if pad_token_id is None:
        raise ValueError("pad_token_id is missing from batch items; ensure dataset provides it.")
    # Verify all features have the same shape before stacking; if not, pad to max T
    feat_shapes = [item["input_features"].shape for item in batch]
    if len(set(feat_shapes)) > 1:
        # Variable time dimension — pad to the longest
        max_T = max(s[-1] for s in feat_shapes)
        padded = []
        for item in batch:
            f = item["input_features"]
            if f.shape[-1] < max_T:
                f = torch.nn.functional.pad(f, (0, max_T - f.shape[-1]))
            padded.append(f)
        input_features = torch.stack(padded)
    else:
        input_features = torch.stack([item["input_features"] for item in batch])
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    classification_labels = torch.tensor([item["classification_labels"] for item in batch], dtype=torch.long)
    raw_audio_inputs = [item["raw_audio_input"] for item in batch]
    
    file_ids = [item["file_id"] for item in batch]
    speaker_ids = [item["speaker_id"] for item in batch]
    genders = [item["gender"] for item in batch]
    attack_types = [item["attack_type"] for item in batch]
    codecs = [item["codec"] for item in batch]
    lengths  = [item["raw_audio_lengths"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    raw_audio_inputs = pad_sequence(raw_audio_inputs, batch_first=True, padding_value=0.0)
    
    return {
        "input_features": input_features,
        "raw_audio_input": raw_audio_inputs,
        "raw_audio_lengths": lengths,
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "classification_labels": classification_labels,
        "file_ids": file_ids,
        "speaker_ids": speaker_ids,
        "genders": genders,
        "attack_types": attack_types,
        "codecs": codecs
    }



def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    ensure_bonafide_in_batch: bool = False,
    balance_bonafide_spoof: bool = False,
    drop_last: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    collate_fn: Optional[Callable] = collate_fn,   # default: the module-level collate_fn
    worker_init_fn: Optional[Callable] = None,  # for per-worker seeding/reproducibility
    generator: Optional[torch.Generator] = None,  
):
    """
    Build a DataLoader with optional balanced or bonafide-guaranteed samplers.
    - balance_bonafide_spoof: uses BalancedBatchSampler (equal spoof/bonafide halves).
    - ensure_bonafide_in_batch: uses EnsureAtLeastOneBonafideBatchSampler.
    - Disables persistent_workers and prefetch_factor automatically when num_workers=0.
    - Passes through custom collate_fn and worker_init_fn for reproducibility.
    """
    if balance_bonafide_spoof and ensure_bonafide_in_batch:
        raise ValueError(
            "balance_bonafide_spoof and ensure_bonafide_in_batch "
            "cannot both be True – choose only one of them."
        )

    # persistent_workers requires num_workers > 0; PyTorch raises if both are set
    if num_workers == 0:
        persistent_workers = False
        # prefetch_factor is only valid when num_workers > 0
        prefetch_factor = None

    if balance_bonafide_spoof:
        sampler = BalancedBatchSampler(dataset, batch_size=batch_size, drop_last=drop_last, generator=generator)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
        )

    if ensure_bonafide_in_batch:
        sampler = EnsureAtLeastOneBonafideBatchSampler(dataset, batch_size=batch_size, drop_last=drop_last, generator=generator)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,  # Pass through for reproducibility
        )

    # Regular DataLoader without special samplers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,  
        generator=generator,  
    )


if __name__ == "__main__":
    # ========== loading yaml config ==========
    SPECIAL_TOKEN_TASK = "<SPECIAL_20>" # Task token
    torch.manual_seed(configs.seed)
    random.seed(configs.seed)
    np.random.seed(configs.seed)

    # ========== SEEDING ==========
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEBUG = True
    USE_4BIT = True 
    RUNNING_BY_EPOCHS = True
    NUM_WORKERS = 16 if not DEBUG else 1
    BATCH_SIZE = 4 if not DEBUG else 2
    NUM_EPOCHS = 10 if not DEBUG else 1

    # ========== CONFIG ==========
    NUM_OF_TOKENS = configs.NUM_OF_TOKENS
    configs.project = configs.project.format(NUM_OF_TOKENS=NUM_OF_TOKENS)
    configs.name = configs.name.format(NUM_OF_TOKENS=NUM_OF_TOKENS)

    VOXTRAL_MODEL = configs.model_id
    OUTPUT_DIR = os.path.join(configs.save_path, configs.name)
    PRECISION = configs.precision

    finetune_mode = configs.finetune_mode
    TARGET_MODULES = configs.target_modules

    TRAIN_PROTOCOL = configs.train_protocol
    DEV_PROTOCOL   = configs.dev_protocol
    EVAL_PROTOCOL  = configs.eval_protocol

    TRAIN_AUDIO_DIR = configs.train_audio_dir
    DEV_AUDIO_DIR   = configs.dev_audio_dir
    EVAL_AUDIO_DIR  = configs.eval_audio_dir
 
    reasoning = False
    AUGMENTATION_ON = True
    # Use PROMPTS already loaded via configs
    processor = AutoProcessor.from_pretrained(VOXTRAL_MODEL)
    processor.tokenizer.padding_side = "right"
    processor.feature_extractor.do_normalize = True

    vocab = processor.tokenizer.get_vocab()


    train_dataset = ASVspoofIndexDataset(path_to_protocol = TRAIN_PROTOCOL, data_dir = TRAIN_AUDIO_DIR, dataset_type = configs.DATASET, processor = processor, prompts = PROMPTS, is_train=True, 
                                        augmentation_on=AUGMENTATION_ON, reasoning = reasoning, 
                                        SPECIAL_TOKEN_TASK = SPECIAL_TOKEN_TASK, dtype = dtype, CONFIG_AUGMENTATION=CONFIG_AUGMENTATION, 
                                        pad_features_mode = 'zero', # YAML: standard zero-padding
                                        pad_waveform_mode = 'none', # YAML: standard
                                        constant_prompts_flag = True)
    
    # pad_features_mode (str, optional): Strategy for padding features. 
    # 'zero' pads with silence; 
    # (pad_waveform_mode='none' with pad_features_mode='zero' mirrors Whisper’s default: 
    # trim/pad waveform to 30s and silence-pad the mel/features (no tiling or aug).)
    # 'repeat_last' repeats the final valid frame; 
    # 'repeat_last_aug' does the same but can augment the repeated tail; 
    #  'tile' tiles valid frames to fill; 
    #   'tile_aug' tiles with augmentation on the tiled chunks.
            
    #  pad_waveform_mode (str, optional): Strategy for padding waveforms. 
    #  'none' = standard trim/zero-pad; 
    #  'tile_wav' tiles waveform only when features are short; 
    #  'tile_wav_aug' tiles with augmentation on each tile; 
    #  'tile_repeat' always repeats waveform to N_SAMPLES without per-repeat aug; 
    #  'tile_repeat_aug' repeats to N_SAMPLES with augmentation applied per repeat.


    # pad_features_mode=’zero’: fill the end of mel/features with silence (zeros). With pad_waveform_mode=’none’ this matches default Whisper behaviour — trim/pad waveform to 30s then silent-fill the mel.

    # repeat_last: repeat the last valid frame until reaching the full feature length.

    # repeat_last_aug: like repeat_last, but each repeated chunk may undergo spectral augmentation with the configured probability.

    # tile: tile the valid feature region (consecutive chunks) until the full length is filled.

    # tile_aug: like tile, but each tiled chunk may undergo augmentation.

    # pad_waveform_mode=’none’: standard trim/pad of the waveform to N_SAMPLES (zero-padding).

    # tile_wav: if features are still short after extraction, tile the waveform to fill and re-extract features (no augmentation per tile).

    # tile_wav_aug: like tile_wav, but each waveform tile may undergo augmentation.

    # tile_repeat: always tile the waveform to N_SAMPLES (even if already close), no augmentation per tile.

    # tile_repeat_aug: always tile to N_SAMPLES, each tile may undergo augmentation.
        

    # train_dataset = ASVspoofIndexDataset(path_to_protocol = TRAIN_PROTOCOL, data_dir = TRAIN_AUDIO_DIR, dataset_type = configs.DATASET, processor = processor, prompts = PROMPTS, is_train=True, augmentation_on=AUGMENTATION_ON, reasoning = reasoning, 
    #                                SPECIAL_TOKEN_TASK = SPECIAL_TOKEN_TASK, dtype = dtype, CONFIG_AUGMENTATION=CONFIG_AUGMENTATION, pad_features_mode = 'zero', pad_waveform_mode = "tile_wav_aug" )
    

   # train_loader = make_dataloader(train_dataset, batch_size=2, ensure_bonafide_in_batch=False, balance_bonafide_spoof=True ,drop_last=False,num_workers = 0)


    indx = 10
    # Extract one sample
   # sample = next(iter(train_loader))

   
    # Extract one sample
    sample = train_dataset[indx]

    # Example: if your dataset returns {"audio": tensor, "text": str, ...}
    input_features = sample["input_features"] if isinstance(sample, dict) else sample[0]

    # Convert to numpy
    if isinstance(input_features, torch.Tensor):
        input_features_np = input_features.squeeze().cpu().numpy()
    else:
        input_features_np = input_features
    
    labels = sample["classification_labels"]  # shape: [batch_size]

    # num_bonafide = (labels == 0).sum().item()
    # num_spoof = (labels == 1).sum().item()

    # print("bonafide (0):", num_bonafide)
    # print("spoof (1):", num_spoof)


    img = plt.imshow(
        input_features_np,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma"
    )
    plt.colorbar(img, fraction=0.046, pad=0.04)
    plt.ylabel("Frequency bins")
    plt.xlabel("Frames")
    plt.title(f"Spectrogram of train_dataset[{indx}] (features)")
    # Save as image
    plt.tight_layout()
    plt.savefig(f"train_dataset_{indx}_spectrogram.png", dpi=300)
    plt.close()
    print(f"Saved spectrogram to train_dataset_{indx}_spectrogram.png")


    # dev_dataset = ASVspoofIndexDataset(path_to_protocol = DEV_PROTOCOL, data_dir = DEV_AUDIO_DIR, dataset_type = configs.DATASET, processor = processor, prompts = PROMPTS, is_train=False, augmentation_on=False , 
    #                              reasoning = reasoning, SPECIAL_TOKEN_TASK = SPECIAL_TOKEN_TASK, dtype = dtype, CONFIG_AUGMENTATION=CONFIG_AUGMENTATION)

    # eval_dataset = ASVspoofIndexDataset(path_to_protocol = EVAL_PROTOCOL, data_dir = EVAL_AUDIO_DIR, dataset_type = configs.DATASET, processor = processor, prompts = PROMPTS, is_train=False, augmentation_on=False , 
    #                              reasoning = reasoning, SPECIAL_TOKEN_TASK = SPECIAL_TOKEN_TASK, dtype = dtype, CONFIG_AUGMENTATION=CONFIG_AUGMENTATION)
    # eval_dataset[11]
