"""
Augmentation helpers for Voxtral audio spoofing detection.

Available augmentations:
- Additive Gaussian noise
- Speed perturbation (resampling to simulate faster/slower speech)
- Time masking (mask random spans of time, works on spectrograms)
- Frequency masking (mask random frequency bins, works on spectrograms)
- Codec re-encoding (see utils/codec.py for codec-specific logic)
- RawBoost augmentation (optinal)
- Gain/volume perturbation
- Time shift perturbation
- Bandpass filtering
- Waveform dropout
- Clipping/saturation

"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
import sys
import torch
import torchaudio
from typing import Optional, Sequence, Union, Tuple
import numpy as np
from types import SimpleNamespace

# Import codec utilities from codec.py to avoid duplication
from .codec import (
    # Constants
    CODEC_BUNDLES,
    TELEPHONY_CODECS as TELEPHONY,
    TORCHAUDIO_SAFE_CODECS,
    CODEC_BITRATE_CAPS,
    MONO_ONLY_CODECS,
    # FFmpeg detection
    have_ffmpeg as _have_ffmpeg,
    get_ffmpeg_supported_encoders as _ffmpeg_supported_encoder_names,
    # Codec mapping functions
    get_ffmpeg_encoder_name as _ffmpeg_encoder_name_for,
    get_codec_extension as _codec_ext,
    get_target_sample_rate as _target_sr_for,
    filter_supported_codecs as _filter_supported_codecs,
    # Re-encoding functions
    reencode_audio,
    reencode_with_torchaudio as _reencode_with_torchaudio_impl,
)

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio\\.io\\._streaming_media_decoder\\.StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)


_HAS_TORCHCODEC_LOAD = hasattr(torchaudio, "load_with_torchcodec")


def _to_cpu_f32(x: torch.Tensor) -> torch.Tensor:
    """
    Move tensor to CPU float32 for codec IO.
    - Detaches to avoid autograd ties, casts to float32, and moves to CPU.
    - Keeps shape; caller is responsible for restoring dtype/device if needed.
    """
    return x.detach().to(dtype=torch.float32, device="cpu")


def _ta_load(path: str) -> tuple[torch.Tensor, int]:
    """
    Load audio using torchcodec when available.
    - Tries torchaudio.load_with_torchcodec first, falling back to torchaudio.load on error or absence.
    - Returns waveform tensor and sample rate; caller handles device/dtype adjustments.
    """
    if _HAS_TORCHCODEC_LOAD:
        try:
            return torchaudio.load_with_torchcodec(path)
        except Exception:
            pass
    return torchaudio.load(path)


# -------------------------- AUGMENTATION CORE -------------------------- #

_PROB_SUM_FLOAT_TOLERANCE = 1e-6  # guard against floating-point rounding when checking prob_sum <= 1.0

@dataclass
class WaveformAugmentationConfig:
    """Holds all options for waveform augmentation."""
    # Additive noise
    add_noise: bool = False
    noise_std: float = 0.0

    # Gain perturbation
    gain_perturb: bool = False
    gain_min_db: float = -6.0
    gain_max_db: float = 6.0

    # Time shift
    timeshift_perturb: bool = False
    timeshift_max_frac: float = 0.2  # fraction of total length to shift
    timeshift_wrap: bool = True      # True = circular roll, False = zero-pad

    # Speed perturbation
    speed_perturb: bool = False
    speed_min: float = 0.9
    speed_max: float = 1.1
    
    # Time masking
    time_mask: bool = False
    time_mask_param: int = 80
    
    # Freq masking
    freq_mask: bool = False
    freq_mask_param: int = 8
    
    # Codec augmentation (re-encoding with lossy codec)
    codec_perturb: bool = False
    # g711a/g711u/g722/g726 are telephony/VoIP codecs and amr/speex are low-bitrate voice
    # codecs, included so augmentation can simulate phone-call channel conditions too.
    codec_choices: Sequence[str] = ("mp3" , "aac", "m4a", "opus", "flac", "alac", "amr", "speex" , "g726", "g711a", "g711u", "g722", "wav")
    bitrate_kbps_min: int = 8
    bitrate_kbps_max: int = 64
    # Resamples back to the input rate after codecs that force their own sample rate
    # (e.g. telephony codecs at 8 kHz); prefer_ffmpeg favors the ffmpeg encoder path
    # over torchaudio's built-in codecs when both are available.
    keep_original_sr_after_codec: bool = True
    prefer_ffmpeg: bool = True


    # Band-limited filtering on the signal (telephony / different bandwidth)
    bandpass_perturb: bool = False
    bandpass_min_low_hz: float = 200.0
    bandpass_max_high_hz: float = 6000.0
    bandpass_min_width_hz: float = 1000.0

    # Time-domain "holes" in the waveform (similar to time masking but on the raw signal)
    waveform_dropout: bool = False
    waveform_dropout_max_frac: float = 0.1   # up to 10% of signal can be zeroed per span
    waveform_dropout_num_spans: int = 1      # number of dropout spans per sample

    # Clipping / saturation
    clip_perturb: bool = False
    clip_min: float = 0.5    # minimum clip level
    clip_max: float = 0.98   # maximum clip level

# These fields configure RawBoost's three noise stages: linear/non-linear convolutive
# distortion, stationary/non-stationary band noise, and additive impulsive noise.
    # RawBoost augmentation (from utils/RawBoost.py)
    rawboost_enable: bool = False
    rawboost_algorithm: int = 4  # default: run all three stages in series
    rawboost_n_f: int = 5
    rawboost_n_bands: int = 5
    rawboost_min_f: float = 20.0
    rawboost_max_f: float = 8000.0
    rawboost_min_bw: float = 100.0
    rawboost_max_bw: float = 1000.0
    rawboost_min_coeff: int = 2
    rawboost_max_coeff: int = 32
    rawboost_min_g: float = 0.0
    rawboost_max_g: float = 15.0
    rawboost_min_bias: float = 5.0
    rawboost_max_bias: float = 20.0
    rawboost_p: float = 0.6
    rawboost_g_sd: float = 2.0
    rawboost_snr_min: float = 10.0
    rawboost_snr_max: float = 40.0

# These three probabilities partition each sample into rawboost-only, codec-only, or
# both applied; remaining probability mass is the chance neither is applied.
    # Probabilities for RawBoost/codec selection inside the augmentation branch
    prob_rawboost_no_codec: float = 0.0
    prob_codec_no_rawboost: float = 0.0
    prob_both: float = 0.0

    # Selective augmentation mode (NEW)
    selective_augmentation: bool = False  # If True, randomly select subset of augmentations instead of stacking all
    max_augmentations_per_sample: int = 2  # Maximum number of augmentations to apply per sample (if selective_augmentation=True)
    max_augmentations_per_spectrogram: int = 2  # Maximum number of spectrogram augmentations to apply per sample (if selective_augmentation=True)
    # Note: codec is always separate - when selected, only codec applies (no stacking with others)


_DEFAULT_CONFIG = WaveformAugmentationConfig()

# Constants imported from codec.py:
# - CODEC_BUNDLES (voip, consumer, lossless, all)
# - TELEPHONY (aliased from TELEPHONY_CODECS)
# - TORCHAUDIO_SAFE_CODECS
# - CODEC_BITRATE_CAPS
# - MONO_ONLY_CODECS


def set_default_augmentation_config(config: WaveformAugmentationConfig) -> None:
    """
    Replace the module-level default augmentation config.
    - Validates the input type before updating the shared default.
    - Affects subsequent calls that omit an explicit config.
    """
    if not isinstance(config, WaveformAugmentationConfig):
        raise TypeError("config must be a WaveformAugmentationConfig")
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = config


def get_default_augmentation_config() -> WaveformAugmentationConfig:
    """
    Return the current default augmentation config.
    - Provides a safe accessor without copying; caller should clone if mutating.
    """
    return _DEFAULT_CONFIG


def _ensure_2d(waveform: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """
    Guarantee waveform is shaped (channels, time).
    - Adds a channel dimension for 1-D input; validates 2-D input; raises otherwise.
    - Returns the adjusted tensor and a flag indicating whether a channel was added.
    """
    if waveform.dim() == 1:
        return waveform.unsqueeze(0), True
    if waveform.dim() == 2:
        return waveform, False
    raise ValueError("waveform must be 1-D or 2-D (channels, time)")


def _cast_like(x: torch.Tensor, ref_dtype: torch.dtype) -> torch.Tensor:
    """
    Cast a tensor to the reference dtype when possible.
    - Returns the original tensor if already matching; otherwise attempts a safe cast.
    - Leaves tensor unchanged on type errors to avoid breaking callers.
    """
    if x.dtype == ref_dtype:
        return x
    try:
        return x.to(ref_dtype)
    except (TypeError, RuntimeError):
        return x


def _rand_uniform(low: float, high: float) -> float:
    """
    Draw a uniform random float using torch RNG.
    - Honors torch.manual_seed for reproducibility.
    - Returns Python float sampled from [low, high).
    """
    return float(torch.empty((), device="cpu").uniform_(low, high).item())


def _rand_int_inclusive(low: int, high: int) -> int:
    """
    Draw an inclusive random integer using torch RNG.
    - Uses torch.randint so global torch seeds govern reproducibility.
    - Inclusive bounds: returns any value in [low, high].
    """
    return int(torch.randint(low, high + 1, (1,), device="cpu").item())


def _rand_choice(seq):
    """
    Torch-backed choice from a non-empty sequence.
    - Samples an index uniformly with torch.randint for seed control.
    - Raises ValueError for empty sequences to avoid silent failures.
    """
    if not seq:
        raise ValueError("Cannot choose from an empty sequence.")
    idx = _rand_int_inclusive(0, len(seq) - 1)
    return seq[idx]


def _apply_gain(x: torch.Tensor, db: float) -> torch.Tensor:
    """
    Apply scalar gain in decibels.
    - Converts dB to linear scale and multiplies the waveform.
    - Keeps dtype/device; caller handles clipping if desired.
    """
    return x * (10.0 ** (db / 20.0))

def _gain_augment(x: torch.Tensor, cfg: WaveformAugmentationConfig) -> torch.Tensor:
    """
    Randomly adjust volume based on config bounds.
    - No-op when gain_perturb is disabled.
    - Draws gain uniformly between gain_min_db and gain_max_db and applies per-channel.
    """
    if not cfg.gain_perturb:
        return x
    gain_db = _rand_uniform(cfg.gain_min_db, cfg.gain_max_db)
    x2d, squeezed = _ensure_2d(x)
    y = _apply_gain(x2d, gain_db)
    return y.squeeze(0) if squeezed else y

def _timeshift_augment(x: torch.Tensor, cfg: WaveformAugmentationConfig) -> torch.Tensor:
    """
    Randomly shift waveform left/right without changing length.
    - Controlled by cfg.timeshift_perturb with magnitude capped by timeshift_max_frac of total samples.
    - In wrap mode it circularly rolls the signal; otherwise it zero-pads the exposed side.
    - Returns the input unchanged when disabled, the shift is zero, or the computed max shift is non-positive.
    """
    if not cfg.timeshift_perturb or cfg.timeshift_max_frac <= 0:
        return x
    x2d, squeezed = _ensure_2d(x)
    C, T = x2d.shape
    max_shift = int(T * cfg.timeshift_max_frac)
    if max_shift > T:
        max_shift = T
    if max_shift <= 0:
        return x
    shift = _rand_int_inclusive(-max_shift, max_shift)
    if shift == 0:
        return x
    if cfg.timeshift_wrap:
        y = torch.roll(x2d, shifts=shift, dims=1)
    else:
        if shift > 0:
            pad = torch.zeros(C, shift, dtype=x.dtype, device=x.device)
            y = torch.cat([pad, x2d[:, :-shift]], dim=1)
        else:
            shift = -shift
            pad = torch.zeros(C, shift, dtype=x.dtype, device=x.device)
            y = torch.cat([x2d[:, shift:], pad], dim=1)
    return y.squeeze(0) if squeezed else y


def _pick_codec_and_bitrate(
    cfg: WaveformAugmentationConfig,
    choices_override: Optional[Sequence[str]] = None,
) -> tuple[str, int]:
    """
    Select a codec and bitrate for augmentation.
    - Uses choices_override when provided, otherwise cfg.codec_choices (string coerced to tuple).
    - Filters and caches supported codecs based on ffmpeg availability; falls back to safe defaults.
    - Samples bitrate uniformly within [bitrate_kbps_min, bitrate_kbps_max].
    """
    if choices_override is None:
        if isinstance(cfg.codec_choices, str):
            cfg.codec_choices = [cfg.codec_choices]
        choices = tuple(cfg.codec_choices)
    else:
        choices = tuple(choices_override)
    if not choices:
        raise ValueError("codec_choices must be non-empty when codec_perturb=True")
    have_ffmpeg = _have_ffmpeg()
    # One-time filtering and cache on cfg; reset if list changed or ffmpeg availability changed
    if (
        getattr(cfg, "_filtered_codec_choices_src", None) != choices
        or getattr(cfg, "_filtered_codec_choices_ffmpeg", None) != have_ffmpeg
        or not hasattr(cfg, "_filtered_codec_choices")
    ):
        cfg._filtered_codec_choices_src = choices
        cfg._filtered_codec_choices_ffmpeg = have_ffmpeg
        cfg._filtered_codec_choices = _filter_supported_codecs(choices)
        if not cfg._filtered_codec_choices:
            # If everything failed, fall back to safe minimum
            cfg._filtered_codec_choices = tuple(c for c in choices if c in {"mp3", "aac", "opus", "ogg", "flac", "wav", "m4a", "wma"})
            print(f"Warning: No supported codecs found from choices {choices}. Falling back to {cfg._filtered_codec_choices}.", file=sys.stderr)
        if not cfg._filtered_codec_choices:
            # Last-resort: keep training running even if user passed only unsupported codecs.
            cfg._filtered_codec_choices = ("wav",)
        if not have_ffmpeg:
            torchaudio_safe = tuple(c for c in cfg._filtered_codec_choices if c in {"wav", "flac"})
            if torchaudio_safe:
                if set(torchaudio_safe) != set(cfg._filtered_codec_choices):
                    print(f"Warning: FFmpeg not available; restricting codec choices to {torchaudio_safe}.", file=sys.stderr)
                cfg._filtered_codec_choices = torchaudio_safe
            if not cfg._filtered_codec_choices:
                cfg._filtered_codec_choices = ("wav",)
    
    codec = _rand_choice(list(cfg._filtered_codec_choices))
    min_kbps = max(1, int(cfg.bitrate_kbps_min))
    max_kbps = max(1, int(cfg.bitrate_kbps_max))
    if min_kbps > max_kbps:
        print(f"Warning: bitrate_kbps_min ({min_kbps}) > bitrate_kbps_max ({max_kbps}); swapping.", file=sys.stderr)
        min_kbps, max_kbps = max_kbps, min_kbps
    cap = CODEC_BITRATE_CAPS.get(codec)
    if cap is not None and max_kbps > cap:
        max_kbps = cap
        if min_kbps > max_kbps:
            min_kbps = max_kbps
    # Uniform random bitrate between min and max (inclusive)
    kbps = _rand_int_inclusive(min_kbps, max_kbps)
    return codec, kbps



# Note: The following functions are now imported from codec.py:
# - _have_ffmpeg (from have_ffmpeg)
# - _ffmpeg_supported_encoder_names (from get_ffmpeg_supported_encoders)
# - _ffmpeg_encoder_name_for (from get_ffmpeg_encoder_name)
# - _filter_supported_codecs (from filter_supported_codecs)
# - _codec_ext (from get_codec_extension)
# - _target_sr_for (from get_target_sample_rate)

def _reencode_with_ffmpeg(x: torch.Tensor, sr: int, codec: str, bitrate_kbps: int) -> tuple[torch.Tensor, int]:
    """
    Re-encode a waveform via ffmpeg for codec augmentation.
    Delegates to utils/codec.py which uses pipe-based encoding (no disk I/O) when possible,
    falling back to file-based encoding only for codecs that require container formats.
    """
    # Use the imported reencode_audio from codec.py
    return reencode_audio(x, sr, codec, bitrate_kbps, prefer_pipe=True)


def _reencode_with_torchaudio(x: torch.Tensor, sr: int, codec: str) -> tuple[torch.Tensor, int]:
    """
    Fallback codec re-encode using torchaudio save/load (no ffmpeg).
    Delegates to utils/codec.py for implementation.
    """
    return _reencode_with_torchaudio_impl(x, sr, codec)
        
        
def _codec_augment(x: torch.Tensor, sr: int, cfg: WaveformAugmentationConfig) -> tuple[torch.Tensor, int, str, int]:
    """
    Apply codec perturbation with ffmpeg when preferred/available.
    - Chooses codec and bitrate via _pick_codec_and_bitrate, optionally restricting to torchaudio-safe codecs.
    - Handles mono-only codecs, resamples as needed, and optionally restores the caller's original sample rate.
    - Returns augmented waveform, output sample rate, codec name, and bitrate; falls back gracefully on failure.
    """
    x2d, squeezed = _ensure_2d(x)
    orig_sr = sr
    have_ffmpeg = _have_ffmpeg()
    use_ffmpeg = cfg.prefer_ffmpeg and have_ffmpeg
    raw_choices = cfg.codec_choices
    if isinstance(raw_choices, str):
        raw_choices = (raw_choices,)
    torchaudio_choices = tuple(c for c in raw_choices if c in TORCHAUDIO_SAFE_CODECS)
    choices_override: Optional[Sequence[str]] = None
    codec = "none"
    kbps = 0

    if not use_ffmpeg:
        if torchaudio_choices:
            choices_override = torchaudio_choices
        elif have_ffmpeg:
            use_ffmpeg = True
            print("Warning: No torchaudio-safe codecs available; falling back to ffmpeg for codec augmentation.", file=sys.stderr)
        else:
            raise RuntimeError(
                "Codec augmentation requested but no torchaudio-safe codecs available and ffmpeg not present. "
                "Install ffmpeg or use torchaudio-safe codecs (wav, flac)."
            )
    try:
        codec, kbps = _pick_codec_and_bitrate(cfg, choices_override=choices_override)

        if use_ffmpeg:
            y, new_sr = _reencode_with_ffmpeg(x2d, sr, codec, kbps)
        else:
            print("Warning: Using torchaudio for codec augmentation instead of ffmpeg.", file=sys.stderr)
            y, new_sr = _reencode_with_torchaudio(x2d, sr, codec)

        # Some codecs force their own sample rate (e.g. narrowband telephony at 8 kHz);
        # resample back to the caller's original rate so downstream tensor shapes stay consistent.
        if cfg.keep_original_sr_after_codec and new_sr != orig_sr:
            y = torchaudio.functional.resample(y, orig_freq=new_sr, new_freq=orig_sr)
            new_sr = orig_sr
        return (y.squeeze(0) if squeezed else y), int(new_sr), codec, kbps
    except Exception as exc:
        raise RuntimeError(
            f"Codec augmentation failed (codec={codec}, kbps={kbps}): {exc}. "
            "Check ffmpeg installation or codec availability."
        ) from exc


def _build_rawboost_args(cfg: WaveformAugmentationConfig, sr: int) -> SimpleNamespace:
    """Create a SimpleNamespace with RawBoost parameters clamped to sane ranges."""
    # 0.49 * sr keeps the highest filter frequency safely below the Nyquist limit (sr / 2).
    max_f = min(float(getattr(cfg, "rawboost_max_f", 8000.0)), 0.49 * sr)
    min_f = max(1.0, min(float(getattr(cfg, "rawboost_min_f", 20.0)), max_f - 1.0))
    min_bw = max(1.0, float(getattr(cfg, "rawboost_min_bw", 100.0)))
    max_bw = max(min_bw, float(getattr(cfg, "rawboost_max_bw", 1000.0)))
    snr_min = float(getattr(cfg, "rawboost_snr_min", 10.0))
    snr_max = max(snr_min, float(getattr(cfg, "rawboost_snr_max", 40.0)))

    args = SimpleNamespace(
        N_f=int(getattr(cfg, "rawboost_n_f", 5)),
        nBands=int(getattr(cfg, "rawboost_n_bands", 5)),
        minF=min_f,
        maxF=max_f,
        minBW=min_bw,
        maxBW=max_bw,
        minCoeff=int(getattr(cfg, "rawboost_min_coeff", 2)),
        maxCoeff=int(getattr(cfg, "rawboost_max_coeff", 32)),
        minG=float(getattr(cfg, "rawboost_min_g", 0.0)),
        maxG=float(getattr(cfg, "rawboost_max_g", 15.0)),
        minBiasLinNonLin=float(getattr(cfg, "rawboost_min_bias", 5.0)),
        maxBiasLinNonLin=float(getattr(cfg, "rawboost_max_bias", 20.0)),
        P=float(getattr(cfg, "rawboost_p", 0.6)),
        g_sd=float(getattr(cfg, "rawboost_g_sd", 2.0)),
        SNRmin=snr_min,
        SNRmax=snr_max,
    )
    return args


def _rawboost_augment(x: torch.Tensor, sr: int, cfg: WaveformAugmentationConfig) -> torch.Tensor:
    """
    Apply RawBoost augmentation (convolutive/impulsive/SSI noise) from utils/RawBoost.py.
    - Controlled via cfg.rawboost_enable/rawboost_algorithm; safe no-op when disabled or on import failure.
    - Operates per-channel on CPU/NumPy and returns tensor on original device/dtype.
    """
    if not getattr(cfg, "rawboost_enable", False):
        return x

    algo = int(getattr(cfg, "rawboost_algorithm", 0) or 0)
    if algo <= 0 or sr <= 0:
        return x

    try:
        from utils.data_utils_rawboost import process_Rawboost_feature  # type: ignore
    except Exception:
        try:
            from data_utils_rawboost import process_Rawboost_feature  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"Warning: RawBoost augmentation unavailable ({exc}). Skipping RawBoost.", file=sys.stderr)
            return x

    args = _build_rawboost_args(cfg, sr)
    x2d, squeezed = _ensure_2d(x)

    def _apply_channel(chan: torch.Tensor) -> torch.Tensor:
        feat_np = chan.detach().cpu().numpy()
        boosted = process_Rawboost_feature(feat_np, sr, args, algo)
        return torch.as_tensor(boosted, dtype=chan.dtype, device=chan.device)

    try:
        boosted = torch.stack([_apply_channel(c) for c in x2d], dim=0)
    except Exception as exc:
        print(f"Warning: RawBoost augmentation failed (algo={algo}): {exc}. Skipping RawBoost.", file=sys.stderr)
        return x

    return boosted.squeeze(0) if squeezed else boosted


def augment_waveform(
    waveform: torch.Tensor,
    sr: int,
    config: Optional[WaveformAugmentationConfig] = None,
    return_sr: bool = False,
    force_disable_rawboost: bool = False,
    force_disable_codec: bool = False,
    waveform_augments_enabled: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    """
    Apply waveform-level augmentations without mutating the input tensor.
    - Validates tensor input/sample rate, upcasts non-floating types to float32, and preserves original dtype on return.
    - Supports RawBoost (convolutive/impulsive/SSI noise), noise, gain, time shift, dropout, bandpass, clipping, speed, and codec perturbations.
    - RawBoost/codec application is chosen via prob_rawboost_no_codec/prob_codec_no_rawboost/prob_both when augmentation is active; RawBoost always precedes codec when both are selected.
    - Selective mode picks a random subset (codec remains separate) instead of stacking everything.
    - Returns augmented waveform, optionally alongside effective sample rate when return_sr=True.
    """
    # Input validation
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"waveform must be a torch.Tensor, got {type(waveform)}")
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}")

    cfg = config or _DEFAULT_CONFIG
    # Augmentation math (noise, filtering, resampling) runs in float32 for numerical
    # stability; the original dtype (fp16/bf16 or integer) is restored on the way out.
    input_dtype = waveform.dtype
    input_device = waveform.device
    input_is_floating = waveform.is_floating_point()
    needs_upcast = (not input_is_floating) or input_dtype in (torch.float16, torch.bfloat16)
    if needs_upcast:
        waveform = waveform.to(torch.float32)
    effective_sr = int(sr)
    out = waveform.clone()
    x2d, added_channel = _ensure_2d(out)

    # Decide RawBoost/codec application based on probabilities and enable flags
    allow_rawboost = (
        getattr(cfg, "rawboost_enable", False)
        and int(getattr(cfg, "rawboost_algorithm", 0)) > 0
        and not force_disable_rawboost
    )
    allow_codec = bool(cfg.codec_perturb and cfg.codec_choices) and not force_disable_codec

    def _prob_from_cfg(new_name: str, old_name: str, default: float) -> float:
        val = getattr(cfg, new_name, None)
        if val is None:
            val = getattr(cfg, old_name, None)
        if val is None:
            val = default
        return max(0.0, float(val or 0.0))

    prob_rawboost_no_codec = _prob_from_cfg("prob_rawboost_no_codec", "prob_rawboost_only", 0.0)
    prob_codec_no_rawboost = _prob_from_cfg("prob_codec_no_rawboost", "prob_codec_only", 0.0)
    prob_both = _prob_from_cfg("prob_both", "prob_both", 0.0)

    if not allow_rawboost:
        prob_rawboost_no_codec = 0.0
        prob_both = 0.0
    if not allow_codec:
        prob_codec_no_rawboost = 0.0
        prob_both = 0.0

    prob_sum = prob_rawboost_no_codec + prob_codec_no_rawboost + prob_both
    if prob_sum > 1.0 + _PROB_SUM_FLOAT_TOLERANCE:
        # Normalize if user-supplied probabilities sum to >1.0
        prob_rawboost_no_codec /= prob_sum
        prob_codec_no_rawboost /= prob_sum
        prob_both /= prob_sum
        prob_sum = prob_rawboost_no_codec + prob_codec_no_rawboost + prob_both
    # Draws one random number and compares it against cumulative probability thresholds
    # to pick exactly one outcome: rawboost-only, codec-only, both, or neither.
    prob_none = max(0.0, 1.0 - prob_sum)
    r = torch.rand(1, device="cpu").item()
    apply_rawboost = False
    apply_codec = False
    if r < prob_rawboost_no_codec:
        apply_rawboost = allow_rawboost
    elif r < prob_rawboost_no_codec + prob_codec_no_rawboost:
        apply_codec = allow_codec
    elif r < prob_rawboost_no_codec + prob_codec_no_rawboost + prob_both:
        apply_rawboost = allow_rawboost
        apply_codec = allow_codec
    else:
        # prob_none branch; neither RawBoost nor codec
        _ = prob_none

    # SELECTIVE AUGMENTATION MODE
    if cfg.selective_augmentation:
        # Apply RawBoost upfront when selected to guarantee ordering before codec
        remaining_slots = cfg.max_augmentations_per_sample
        if apply_rawboost:
            x2d = _rawboost_augment(x2d, effective_sr, cfg)
            remaining_slots = max(0, remaining_slots - 1)

        # Build list of available augmentations (excluding RawBoost/codec which are handled separately)
        available_augs: list[str] = []
        if waveform_augments_enabled:
            if cfg.add_noise and cfg.noise_std > 0:
                available_augs.append('noise')
            if cfg.gain_perturb:
                available_augs.append('gain')
            if cfg.timeshift_perturb:
                available_augs.append('timeshift')
            if cfg.waveform_dropout:
                available_augs.append('waveform_dropout')
            if cfg.bandpass_perturb:
                available_augs.append('bandpass')
            if cfg.clip_perturb:
                available_augs.append('clip')
            if cfg.speed_perturb:
                available_augs.append('speed')
        
        # Randomly decides how many (1..max_select) and which available augmentations to
        # apply this sample, instead of always stacking every enabled one.
        max_select = min(remaining_slots, len(available_augs))
        num_to_select = _rand_int_inclusive(1, max_select) if max_select > 0 else 0
        if num_to_select > 0:
            indices = torch.randperm(len(available_augs))[:num_to_select].tolist()
            selected_augs = [available_augs[i] for i in indices]
        else:
            selected_augs = []
        
        # Apply selected augmentations
        if 'noise' in selected_augs:
            x2d = x2d + torch.randn_like(x2d) * float(cfg.noise_std)
        if 'gain' in selected_augs:
            x2d = _gain_augment(x2d, cfg)
        if 'timeshift' in selected_augs:
            x2d = _timeshift_augment(x2d, cfg)
        if 'waveform_dropout' in selected_augs:
            x2d = _waveform_dropout_augment(x2d, cfg)
        if 'bandpass' in selected_augs:
            x2d = _bandpass_augment(x2d, sr, cfg)
        if 'clip' in selected_augs:
            x2d = _clipping_augment(x2d, cfg)
        if 'speed' in selected_augs:
            if cfg.speed_min <= 0 or cfg.speed_max <= 0:
                raise ValueError("speed_min/max must be positive")
            if cfg.speed_min > cfg.speed_max:
                raise ValueError("speed_min cannot be greater than speed_max")
            speed = _rand_uniform(float(cfg.speed_min), float(cfg.speed_max))
            new_sr = max(1, int(round(effective_sr / speed)))
            x2d = torchaudio.functional.resample(x2d, orig_freq=effective_sr, new_freq=new_sr)
            effective_sr = new_sr
        
        # Codec augmentation (applied after other waveform transforms)
        if apply_codec:
            x2d, codec_sr, codec, kbps = _codec_augment(x2d, effective_sr, cfg)
            # _codec_augment already handles resampling when keep_original_sr_after_codec=True
            # So we just need to update effective_sr to match what was returned
            effective_sr = codec_sr

    else:
        # ORIGINAL MODE: Stack all enabled augmentations (optionally skip non-rawboost/codec)
        if apply_rawboost:
            x2d = _rawboost_augment(x2d, effective_sr, cfg)

        if waveform_augments_enabled:
            # 1) Additive noise
            if cfg.add_noise and cfg.noise_std > 0:
                x2d = x2d + torch.randn_like(x2d) * float(cfg.noise_std)
            
            # 2) Gain perturbation
            if cfg.gain_perturb:
                x2d = _gain_augment(x2d, cfg)
            
            # 3) Time shift perturbation
            if cfg.timeshift_perturb:
                x2d = _timeshift_augment(x2d, cfg)

            # 3.5) Waveform dropout (signal-level time masking)
            if cfg.waveform_dropout:
                x2d = _waveform_dropout_augment(x2d, cfg)

            # 3.6) Band-limited filtering (telephony / different bandwidth)
            if cfg.bandpass_perturb:
                x2d = _bandpass_augment(x2d, sr, cfg)

            # 3.7) Clipping / saturation
            if cfg.clip_perturb:
                x2d = _clipping_augment(x2d, cfg)
                

            # 4) Random speed perturbation
            if cfg.speed_perturb:
                if cfg.speed_min <= 0 or cfg.speed_max <= 0:
                    raise ValueError("speed_min/max must be positive")
                if cfg.speed_min > cfg.speed_max:
                    raise ValueError("speed_min cannot be greater than speed_max")

                speed = _rand_uniform(float(cfg.speed_min), float(cfg.speed_max))
                new_sr = max(1, int(round(effective_sr / speed)))

                x2d = torchaudio.functional.resample(x2d, orig_freq=effective_sr, new_freq=new_sr)
                effective_sr = new_sr
        
        # 5) Codec perturbation
        if apply_codec:
            x2d, codec_sr, codec, kbps = _codec_augment(x2d, effective_sr, cfg)
            #print(f"Applied codec augmentation: codec={codec}, bitrate={kbps} kbps", file=sys.stderr)
            # _codec_augment already handles resampling when keep_original_sr_after_codec=True
            # So we just need to update effective_sr to match what was returned
            effective_sr = codec_sr

    result = x2d.squeeze(0) if added_channel else x2d
    result = result.to(input_device)
    if input_is_floating:
        result = _cast_like(result, input_dtype)
    if return_sr:
        return result, int(effective_sr)
    return result


def augment_spectrogram(
    spec: torch.Tensor,
    config: Optional[WaveformAugmentationConfig] = None,
    mask_fill: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply SpecAugment-style time/freq masking to spectrogram-like tensors.
    - Requires spec.dim() >= 2; otherwise returns input unchanged.
    - mask_fill or config.mask_fill controls the value used for masked spans (defaults to tensor min).
    - Selective mode randomly chooses a subset of time/freq masks up to max_augmentations_per_spectrogram.
    """
    if not isinstance(spec, torch.Tensor):
        raise TypeError("spec must be a torch.Tensor")

    cfg = config or _DEFAULT_CONFIG
    if spec.dim() < 2:
        return spec

    out = spec
    cloned = False

    # value to write into masked spans
    # Falls back from an explicit mask_fill argument, to a mask_fill attribute on the
    # config, to the spectrogram's own minimum value (approximating silence in log-mel features).
    pad_val: Optional[float] = mask_fill
    if pad_val is None and config is not None and hasattr(config, "mask_fill"):
        pad_val = getattr(config, "mask_fill")

    def _resolve_pad_val() -> float:
        nonlocal pad_val
        if pad_val is None:
            pad_val = float(spec.min().item())
        return pad_val

    # SELECTIVE AUGMENTATION MODE
    if cfg.selective_augmentation:
        # Build list of available spectrogram augmentations
        available_augs = []
        if cfg.time_mask and cfg.time_mask_param > 0:
            available_augs.append('time_mask')
        if cfg.freq_mask and cfg.freq_mask_param > 0:
            available_augs.append('freq_mask')
        
        # Select subset of augmentations to apply
        if available_augs:
            max_select = min(cfg.max_augmentations_per_spectrogram, len(available_augs))
            num_to_select = _rand_int_inclusive(1, max_select) if max_select > 0 else 0
            
            if num_to_select > 0:
                indices = torch.randperm(len(available_augs))[:num_to_select].tolist()
                selected_augs = [available_augs[i] for i in indices]
            else:
                selected_augs = []
        else:
            selected_augs = []
        
        # Apply only selected augmentations
        if 'time_mask' in selected_augs:
            T = out.shape[-1]
            max_w = min(int(cfg.time_mask_param), T)
            if max_w > 0 and T > 1:
                w = int(torch.randint(1, max_w + 1, (1,), device=out.device).item())
                start = int(torch.randint(0, max(1, T - w + 1), (1,), device=out.device).item())
                if not cloned:
                    out = out.clone()
                    cloned = True
                out[..., start:start + w] = _resolve_pad_val()
        
        if 'freq_mask' in selected_augs:
            F = out.shape[-2]
            max_f = min(int(cfg.freq_mask_param), F)
            if max_f > 0 and F > 1:
                f = int(torch.randint(1, max_f + 1, (1,), device=out.device).item())
                start_f = int(torch.randint(0, max(1, F - f + 1), (1,), device=out.device).item())
                if not cloned:
                    out = out.clone()
                    cloned = True
                out[..., start_f:start_f + f, :] = _resolve_pad_val()
    
    else:
        # ORIGINAL MODE: Stack all enabled augmentations
        # 3) Time masking (spectrogram-like tensors only)
        if cfg.time_mask and cfg.time_mask_param > 0:
            T = out.shape[-1]
            max_w = min(int(cfg.time_mask_param), T)
            if max_w > 0 and T > 1:
                w = int(torch.randint(1, max_w + 1, (1,), device=out.device).item())
                start = int(torch.randint(0, max(1, T - w + 1), (1,), device=out.device).item())
                if not cloned:
                    out = out.clone()
                    cloned = True
                out[..., start:start + w] = _resolve_pad_val()

        # 4) Frequency masking (spectrogram-like tensors only)
        if cfg.freq_mask and cfg.freq_mask_param > 0:
            F = out.shape[-2]
            max_f = min(int(cfg.freq_mask_param), F)
            if max_f > 0 and F > 1:
                f = int(torch.randint(1, max_f + 1, (1,), device=out.device).item())
                start_f = int(torch.randint(0, max(1, F - f + 1), (1,), device=out.device).item())
                if not cloned:
                    out = out.clone()
                    cloned = True
                out[..., start_f:start_f + f, :] = _resolve_pad_val()

    return out


def _waveform_dropout_augment(
    x: torch.Tensor,
    cfg: WaveformAugmentationConfig
) -> torch.Tensor:
    """
    Zero-out random waveform spans (signal-level time masking).
    - Enabled via waveform_dropout; span length capped by waveform_dropout_max_frac of signal length.
    - Repeats for waveform_dropout_num_spans times; returns input unchanged when disabled or spans invalid.
    """
    if not cfg.waveform_dropout or cfg.waveform_dropout_max_frac <= 0.0:
        return x

    x2d, squeezed = _ensure_2d(x)  # (C, T)
    C, T = x2d.shape
    max_span = int(T * cfg.waveform_dropout_max_frac)
    if max_span <= 0 or cfg.waveform_dropout_num_spans <= 0:
        return x

    for _ in range(int(cfg.waveform_dropout_num_spans)):
        span = _rand_int_inclusive(1, max_span)
        start = _rand_int_inclusive(0, max(0, T - span))
        x2d[:, start:start + span] = 0.0

    return x2d.squeeze(0) if squeezed else x2d


def _bandpass_augment(
    x: torch.Tensor,
    sr: int,
    cfg: WaveformAugmentationConfig
) -> torch.Tensor:
    """
    Apply random band-limited filtering to simulate channel effects.
    - Randomly chooses bandpass, lowpass, or highpass using cfg bounds; skips when disabled or bounds invalid.
    - Uses torchaudio biquad filters and preserves original dimensionality (squeezes back if needed).
    """
    if not cfg.bandpass_perturb:
        return x

    x2d, squeezed = _ensure_2d(x)  # (C, T)
    if sr <= 0:
        return x

    y = x2d
    r = _rand_uniform(0.0, 1.0)

    # Telephony / narrow channel: bandpass ~300–3400 Hz
    if r < 0.33:
        low = max(20.0, cfg.bandpass_min_low_hz)
        high = min(0.45 * sr, cfg.bandpass_max_high_hz)
        if high <= low + cfg.bandpass_min_width_hz:
            return x
        low_cut = _rand_uniform(low, high - cfg.bandpass_min_width_hz)
        high_cut = _rand_uniform(low_cut + cfg.bandpass_min_width_hz, high)

        y = torchaudio.functional.highpass_biquad(y, sr, cutoff_freq=low_cut)
        y = torchaudio.functional.lowpass_biquad(y, sr, cutoff_freq=high_cut)

    # Lowpass only
    elif r < 0.66:
        upper = min(0.45 * sr, cfg.bandpass_max_high_hz)
        lower = max(20.0, cfg.bandpass_min_low_hz)
        if upper <= lower:
            return x
        cutoff = _rand_uniform(lower, upper)
        y = torchaudio.functional.lowpass_biquad(y, sr, cutoff_freq=cutoff)

    # Highpass only
    else:
        upper = min(0.45 * sr, cfg.bandpass_max_high_hz)
        lower = max(50.0, cfg.bandpass_min_low_hz)
        if upper <= lower:
            return x
        cutoff = _rand_uniform(lower, upper)
        y = torchaudio.functional.highpass_biquad(y, sr, cutoff_freq=cutoff)

    return y.squeeze(0) if squeezed else y


def _clipping_augment(
    x: torch.Tensor,
    cfg: WaveformAugmentationConfig
) -> torch.Tensor:
    """
    Apply random clipping / saturation.
    - Enabled via clip_perturb; picks clamp level between clip_min and clip_max, clamping symmetrically.
    - Normalizes by the clamp level to keep amplitudes roughly within [-1, 1].
    """
    if not cfg.clip_perturb:
        return x

    level_min = max(0.1, float(cfg.clip_min))
    level_max = min(1.0, float(cfg.clip_max))
    if level_max <= level_min:
        return x

    level = _rand_uniform(level_min, level_max)

    x2d, squeezed = _ensure_2d(x)
    y = torch.clamp(x2d, -level, level)

    # Normalize back so range is approximately [-1, 1]
    y = y / level

    return y.squeeze(0) if squeezed else y


__all__ = [
    "WaveformAugmentationConfig",
    "augment_waveform",
    "augment_spectrogram",
    "get_default_augmentation_config",
    "set_default_augmentation_config",
]


# ------------------------------ TEST BLOCK ----------------------------- #

# Manual smoke-test entry point: loads one example file and applies the configured
# augmentations to inspect their effect; not used during actual training runs.
if __name__ == "__main__":
    INPUT_PATH = pathlib.Path("/path/to/projects/ASVspoof5/flac_E_prog/E_0000000191.flac")          # point to your input file
    OUTPUT_PATH = None #pathlib.Path("example.aug.wav")     # set to None to skip saving
    TARGET_SAMPLE_RATE = 16000                         # e.g., 16000 or None to keep original
    SEED = 1234                                       # set to None for non-deterministic
    
    # noise
    AUG_ADD_NOISE = True
    AUG_NOISE_STD = 0.02  # YAML: reduced from 0.05
    
    # gain perturbation
    GAIN_PERTURB = True
    GAIN_MIN_DB = -5.0  # YAML value
    GAIN_MAX_DB = 5.0   # YAML value
    
    # time shift perturbation
    TIMESHIFT_PERTURB = True
    TIMESHIFT_MAX_FRAC = 0.1  # YAML: reduced from 0.25
    TIMESHIFT_WRAP = True


    # speed perturbation
    AUG_SPEED_PERTURB = True
    AUG_SPEED_MIN = 0.97  # YAML: reduced from 0.95
    AUG_SPEED_MAX = 1.03  # YAML: reduced from 1.05

    # time masking
    AUG_TIME_MASK = True
    AUG_TIME_MASK_PARAM = 48  # YAML: user's choice

    # frequency masking
    AUG_FREQ_MASK = True
    AUG_FREQ_MASK_PARAM = 8  # YAML: conservative value for anti-spoofing

    # codec perturbation
    CODEC_PERTURB = True
    CODEC_CHOICES = ("mp3", "aac", "opus", "flac", "wav", "g722", "amr")  # YAML: balanced mix, removed ultra-aggressive codecs
    BITRATE_KBPS_MIN = 64   # YAML: increased from 32
    BITRATE_KBPS_MAX = 128  # YAML: increased from 96
    KEEP_ORIGINAL_SR_AFTER_CODEC = True
    PREFER_FFMPEG = True


     # waveform dropout
    WAVEFORM_DROPOUT = True
    WAVEFORM_DROPOUT_MAX_FRAC = 0.02  # YAML: reduced from 0.03
    WAVEFORM_DROPOUT_NUM_SPANS = 1    # YAML value

    # bandpass
    BANDPASS_PERTURB = True
    BANDPASS_MIN_LOW_HZ = 200.0       # YAML values
    BANDPASS_MAX_HIGH_HZ = 6000.0
    BANDPASS_MIN_WIDTH_HZ = 1000.0

    # clipping
    CLIP_PERTURB = False  # YAML: disabled
    CLIP_MIN = 0.5
    CLIP_MAX = 0.98
    
    # selective augmentation mode (NEW)
    SELECTIVE_AUGMENTATION = True  # YAML: enabled to prevent stacking
    MAX_AUGMENTATIONS_PER_SAMPLE = 1  # YAML: one augmentation at a time


    CFG = WaveformAugmentationConfig(
        add_noise=AUG_ADD_NOISE,
        noise_std=AUG_NOISE_STD,

        gain_perturb=GAIN_PERTURB,
        gain_min_db = GAIN_MIN_DB,
        gain_max_db = GAIN_MAX_DB,

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
        
        selective_augmentation=SELECTIVE_AUGMENTATION,
        max_augmentations_per_sample=MAX_AUGMENTATIONS_PER_SAMPLE,
        
    )

    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    waveform, sr = _ta_load(INPUT_PATH)

    # Optional resample before augmentation
    target_sr = TARGET_SAMPLE_RATE or sr
    if target_sr != sr:
        x, squeezed = _ensure_2d(waveform)
        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=int(target_sr))
        waveform = x.squeeze(0) if squeezed else x
        sr = int(target_sr)

    iters = 10
    for i in range(iters):
        print(f"\n--- Augmentation iteration {i+1}/{iters} ---")
        augmented = augment_waveform(waveform, sr, config=CFG)
        print(f"Augmented waveform {i+1}: {augmented}")

    # Save or just print stats
    if OUTPUT_PATH is not None:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        to_save, _ = _ensure_2d(augmented)
        torchaudio.save(OUTPUT_PATH, to_save, sample_rate=sr)
        print(f"Saved: {OUTPUT_PATH}")

    print("Aug stats")
    print(f"shape: {tuple(augmented.shape)}")
    print(f"dtype: {augmented.dtype}")
    print(f"mean: {float(augmented.mean().item()):.6f}")
    print(f"std:  {float(augmented.std().item()):.6f}")
    print(f"sample_rate: {sr}")
