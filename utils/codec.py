"""
Codec Augmentation Utilities for Audio Processing
==================================================

This module provides codec re-encoding functionality for audio augmentation
in speech spoofing detection training. Re-encoding audio through lossy codecs
simulates real-world transmission artifacts and improves model robustness.

Architecture Overview
---------------------
The module implements two encoding strategies:

1. **PIPE MODE** (Primary, Fast):
   - Streams audio directly through ffmpeg via stdin/stdout pipes
   - Zero disk I/O - everything stays in memory
   - Best performance for training loops
   
2. **FILE MODE** (Fallback, Reliable):
   - Uses temporary files on disk
   - Required for codecs needing container formats (AAC, ALAC, WMA)
   - Slower but more compatible

Supported Codec Bundles
-----------------------
- **VOIP**: g711a, g711u, g722, gsm, amr, speex, opus
- **Consumer**: mp3, aac, m4a, ogg, wma, ac3, flac, alac
- **Lossless**: wav, flac, alac, wavpack

Usage Example
-------------
    >>> import torch
    >>> from utils.codec import reencode_audio
    >>> 
    >>> # Load or create audio waveform
    >>> waveform = torch.randn(1, 16000)  # 1 second mono audio at 16kHz
    >>> 
    >>> # Re-encode through MP3 at 64 kbps
    >>> augmented, new_sr = reencode_audio(waveform, 16000, 'mp3', 64)
    >>> print(f"Output: {augmented.shape} at {new_sr}Hz")

Dependencies
------------
- FFmpeg (required for most codecs)
- PyTorch
- torchaudio
- numpy

"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np
import torch
import torchaudio


# =============================================================================
# CONSTANTS - Codec Classification and Configuration
# =============================================================================

CODEC_BUNDLES = {
    # VOIP/Telephony codecs - typically lower sample rates (8-16 kHz)
    "voip": (
        "g711a",    # G.711 A-law: 8 kHz, 64 kbps PCM (European standard)
        "g711u",    # G.711 μ-law: 8 kHz, 64 kbps PCM (North American standard)
        "g722",     # G.722: 16 kHz wideband, 64 kbps
        "gsm",      # GSM Full Rate: 8 kHz, 13 kbps (mobile telephony)
        "amr",      # AMR-NB: 8 kHz, 4.75-12.2 kbps (3G mobile)
        "speex",    # Speex: 8/16 kHz variable (open-source VoIP)
        "opus",     # Opus: up to 48 kHz, 6-510 kbps (modern standard)
    ),
    
    # Consumer audio codecs - higher quality, various sample rates
    "consumer": (
        "mp3",      # MPEG Layer 3: ubiquitous lossy format
        "aac",      # Advanced Audio Coding: better than MP3 at same bitrate
        "m4a",      # M4A container with AAC audio
        "ogg",      # Ogg Vorbis: open-source lossy format
        "wma",      # Windows Media Audio: Microsoft's format
        "ac3",      # Dolby Digital: 5.1 surround sound codec
        "flac",     # Free Lossless Audio Codec: no quality loss
        "alac",     # Apple Lossless: Apple's lossless format
    ),
    
    # Lossless codecs - for reference/archival
    "lossless": (
        "wav",      # Waveform Audio: uncompressed PCM
        "flac",     # Free Lossless Audio Codec
        "alac",     # Apple Lossless Audio Codec
        "wavpack",  # WavPack: hybrid lossless/lossy
    ),
}

# Combined bundle with all supported codecs (de-duplicated)
# dict.fromkeys(...) is used purely as an order-preserving way to de-duplicate codecs
# that appear in more than one bundle (e.g. flac/alac are in both consumer and lossless).
CODEC_BUNDLES["all"] = tuple(dict.fromkeys(
    CODEC_BUNDLES["voip"]
    + CODEC_BUNDLES["consumer"]
    + CODEC_BUNDLES["lossless"]
    + ("amrwb", "g726")  # Additional telephony codecs
))

# Telephony codecs that require 8 kHz mono audio
TELEPHONY_CODECS = {"g711a", "g711u", "g726", "amr"}

# Codecs safely handled by torchaudio alone (no ffmpeg needed)
TORCHAUDIO_SAFE_CODECS = {"wav", "flac"}

# Maximum bitrate caps for specific codecs (kbps)
CODEC_BITRATE_CAPS = {
    "opus": 256,  # Opus caps at 256 kbps for stereo
}

# Codecs that only support mono audio
MONO_ONLY_CODECS = {"amr", "amrwb", "g726", "g711a", "g711u", "g722", "gsm"}

# Codecs that work well with pipe-based encoding (no container format needed)
PIPE_FRIENDLY_CODECS = {"mp3", "flac", "wav", "g711a", "g711u", "opus", "ogg"}

# Codecs that REQUIRE file-based encoding (need proper container format)
# These codecs fail with pipe mode due to container/muxer requirements
FILE_ONLY_CODECS = {"m4a", "aac", "alac", "wma", "ac3", "amr", "amrwb"}


# =============================================================================
# FFMPEG DETECTION - Check for FFmpeg Availability
# =============================================================================

@lru_cache(maxsize=1)
def have_ffmpeg() -> bool:
    """
    Check if ffmpeg is available on the system PATH.
    
    This function is cached - it only runs the check once per process.
    
    Returns:
        bool: True if ffmpeg is installed and executable, False otherwise.
        
    Example:
        >>> if have_ffmpeg():
        ...     print("FFmpeg ready for codec augmentation")
        ... else:
        ...     print("FFmpeg not found - install it or use wav/flac only")
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except FileNotFoundError:
        return False  # ffmpeg not in PATH
    except subprocess.CalledProcessError:
        return False  # ffmpeg exists but failed
    except Exception:
        return False  # Any other error


@lru_cache(maxsize=1)
def get_ffmpeg_supported_encoders() -> set[str]:
    """
    Get the set of audio encoder names supported by the local FFmpeg installation.
    
    This function parses the output of `ffmpeg -encoders` to determine which
    codecs are actually available. Some FFmpeg builds may lack certain encoders
    (e.g., libmp3lame, libopencore_amrnb) depending on compile-time options.
    
    Returns:
        set[str]: Set of encoder names (e.g., {'libmp3lame', 'aac', 'pcm_s16le'}).
                  Returns empty set if FFmpeg is unavailable or parsing fails.
                  
    Note:
        Result is cached - safe to call frequently without performance penalty.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, 
            text=True, 
            check=True
        )
        output = result.stdout
    except Exception:
        return set()

    encoders = set()
    for line in output.splitlines():
        # Format: " A..... pcm_s16le            PCM signed 16-bit little-endian"
        # First char: A=Audio, V=Video, S=Subtitle
        # Next 5 chars: capabilities flags
        # Then encoder name, then description
        match = re.match(r"\s*[A\.][V\.A-Z\.]{5}\s+(\S+)", line)
        if match:
            encoders.add(match.group(1))
    
    return encoders


# =============================================================================
# CODEC MAPPINGS - Translate Codec Names to FFmpeg Parameters
# =============================================================================

def get_ffmpeg_encoder_name(codec: str) -> str:
    """
    Map a logical codec label to its FFmpeg encoder argument.
    
    FFmpeg uses specific library names for encoding that differ from the
    codec's common name. This function translates user-friendly codec names
    to the correct -acodec parameter value.
    
    Args:
        codec: User-friendly codec name (e.g., 'mp3', 'opus', 'aac')
        
    Returns:
        str: FFmpeg encoder name (e.g., 'libmp3lame', 'libopus', 'aac')
        
    Example:
        >>> get_ffmpeg_encoder_name('mp3')
        'libmp3lame'
        >>> get_ffmpeg_encoder_name('g711a')
        'pcm_alaw'
    """
    mapping = {
        # Consumer codecs
        "mp3":     "libmp3lame",   # LAME MP3 encoder (most common)
        "aac":     "aac",          # FFmpeg's native AAC encoder
        "m4a":     "aac",          # M4A is just AAC in MP4 container
        "opus":    "libopus",      # Opus encoder library
        "ogg":     "libvorbis",    # Vorbis encoder for OGG container
        "wma":     "wmav2",        # Windows Media Audio v2
        "ac3":     "ac3",          # Dolby Digital encoder
        "flac":    "flac",         # FLAC encoder (lossless)
        "alac":    "alac",         # Apple Lossless encoder
        "wavpack": "wavpack",      # WavPack encoder
        "wav":     "pcm_s16le",    # Raw PCM (16-bit little-endian)
        
        # Telephony codecs
        "g711a":   "pcm_alaw",     # G.711 A-law (European telephony)
        "g711u":   "pcm_mulaw",    # G.711 μ-law (North American)
        "g722":    "g722",         # G.722 wideband
        "gsm":     "libgsm",       # GSM Full Rate
        "g726":    "adpcm_g726",   # G.726 ADPCM
        "amr":     "libopencore_amrnb",  # AMR Narrowband
        "amrwb":   "libvo_amrwbenc",     # AMR Wideband
        "speex":   "libspeex",     # Speex codec
    }
    return mapping.get(codec, codec)


def get_codec_extension(codec: str) -> str:
    """
    Map codec name to the preferred file extension for that format.
    
    This is used when creating temporary files during file-based encoding.
    The extension must match what FFmpeg expects for the container format.
    
    Args:
        codec: Codec name (e.g., 'mp3', 'aac', 'opus')
        
    Returns:
        str: File extension including the dot (e.g., '.mp3', '.m4a', '.opus')
        
    Example:
        >>> get_codec_extension('aac')
        '.m4a'
        >>> get_codec_extension('opus')
        '.opus'
    """
    mapping = {
        "mp3":     ".mp3",
        "opus":    ".opus",
        "ogg":     ".ogg",     # Vorbis in OGG container
        "aac":     ".m4a",     # AAC typically uses M4A container
        "m4a":     ".m4a",
        "wma":     ".wma",
        "flac":    ".flac",
        "alac":    ".m4a",     # ALAC uses M4A/MP4 container
        "ac3":     ".ac3",
        "amr":     ".amr",     # AMR-NB native container
        "amrwb":   ".awb",     # AMR-WB uses .awb extension
        "speex":   ".ogg",     # Speex typically in OGG container
        "g726":    ".g726",
        "g711a":   ".wav",     # A-law PCM in WAV container
        "g711u":   ".wav",     # μ-law PCM in WAV container
        "g722":    ".g722",
        "gsm":     ".gsm",
        "wavpack": ".wv",
        "wav":     ".wav",
    }
    return mapping.get(codec, f".{codec}")


def get_codec_format(codec: str) -> str:
    """
    Get FFmpeg format/muxer name for pipe-based streaming.
    
    When using pipes (stdin/stdout), FFmpeg needs explicit format specification
    since it can't auto-detect from file extension. This returns the -f argument.
    
    Args:
        codec: Codec name
        
    Returns:
        str: FFmpeg format name for pipe mode
        
    Note:
        Not all codecs support pipe mode - some require proper container files.
    """
    mapping = {
        "mp3":    "mp3",     # MP3 frames can be streamed
        "flac":   "flac",    # FLAC supports streaming
        "wav":    "wav",     # WAV with header
        "opus":   "opus",    # Opus in OGG container
        "ogg":    "ogg",     # OGG container format
        "g711a":  "alaw",    # Raw A-law samples
        "g711u":  "mulaw",   # Raw μ-law samples
        "gsm":    "gsm",     # GSM format
        "g722":   "g722",    # G.722 format
    }
    return mapping.get(codec, codec)


def get_target_sample_rate(codec: str) -> int:
    """
    Determine the target sample rate required by a codec.
    
    Different codecs have specific sample rate requirements:
    - Telephony codecs: 8 kHz (narrowband) or 16 kHz (wideband)
    - Consumer codecs: Usually flexible, we default to 16 kHz
    
    Args:
        codec: Codec name
        
    Returns:
        int: Target sample rate in Hz (8000, 16000, etc.)
        
    Example:
        >>> get_target_sample_rate('amr')  # Narrowband
        8000
        >>> get_target_sample_rate('g722')  # Wideband
        16000
    """
    # 8 kHz narrowband codecs
    if codec in {"g711a", "g711u", "gsm", "g726", "amr"}:
        return 8000
    
    # 16 kHz wideband codecs
    if codec in {"g722", "amrwb", "speex"}:
        return 16000
    
    # Default to 16 kHz for other codecs (good balance of quality/size)
    return 16000


def filter_supported_codecs(codecs: Sequence[str]) -> Tuple[str, ...]:
    """
    Filter a list of codecs to only those supported by the local FFmpeg build.
    
    Some FFmpeg installations lack certain encoder libraries (e.g., libmp3lame,
    libopencore_amrnb). This function checks against the actual available 
    encoders and removes unsupported codecs.
    
    Args:
        codecs: Sequence of codec names to filter
        
    Returns:
        Tuple[str, ...]: Subset of input codecs that are actually usable
        
    Example:
        >>> filter_supported_codecs(['mp3', 'amr', 'opus', 'unknown_codec'])
        ('mp3', 'opus')  # if AMR encoder not installed
    """
    supported_encoders = get_ffmpeg_supported_encoders()
    
    # If we can't determine supported encoders, return all (optimistic)
    if not supported_encoders:
        return tuple(codecs)

    kept = []
    for codec in codecs:
        encoder = get_ffmpeg_encoder_name(codec)
        # Check if encoder is available (special case for g726)
        # Some FFmpeg builds list the G.726 encoder under the plain name "g726" rather than
        # "adpcm_g726", so both spellings are checked.
        if encoder in supported_encoders or (codec == "g726" and "g726" in supported_encoders):
            kept.append(codec)
    
    return tuple(kept)


# =============================================================================
# AUDIO I/O HELPERS - Tensor/Bytes Conversion Utilities
# =============================================================================

def _to_cpu_f32(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to CPU float32 for codec I/O operations.
    
    FFmpeg and torchaudio expect audio on CPU in float32 format.
    This helper ensures consistent tensor format.
    
    Args:
        x: Input tensor (any device, any dtype)
        
    Returns:
        torch.Tensor: CPU float32 tensor (detached from computation graph)
    """
    return x.detach().to(dtype=torch.float32, device="cpu")


def _tensor_to_pcm_bytes(waveform: torch.Tensor) -> bytes:
    """
    Convert a PyTorch audio tensor to raw PCM s16le bytes for FFmpeg stdin.
    
    FFmpeg expects interleaved samples when reading raw PCM:
    For stereo: [L0, R0, L1, R1, L2, R2, ...]
    
    Args:
        waveform: Audio tensor of shape (channels, samples) or (samples,)
                  Values should be in range [-1.0, 1.0]
                  
    Returns:
        bytes: Raw PCM data as signed 16-bit little-endian samples
        
    Note:
        Values are clipped to [-1.0, 1.0] before conversion to prevent overflow.
    """
    audio_np = waveform.cpu().numpy()
    
    # Ensure 2D shape
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(1, -1)
    
    # Clamp to valid range and convert to int16
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    # Interleave channels: (C, T) -> (T, C) -> bytes
    # FFmpeg expects samples in LRLRLR order for stereo
    return audio_int16.T.tobytes()


def _pcm_bytes_to_tensor(data: bytes, channels: int) -> torch.Tensor:
    """
    Convert raw PCM s16le bytes back to a PyTorch audio tensor.
    
    This is the inverse of _tensor_to_pcm_bytes(). Parses interleaved
    samples from FFmpeg stdout back into tensor format.
    
    Args:
        data: Raw PCM bytes (signed 16-bit little-endian, interleaved)
        channels: Number of audio channels (1=mono, 2=stereo)
        
    Returns:
        torch.Tensor: Audio tensor of shape (channels, samples)
                      Values normalized to range [-1.0, 1.0]
                      
    Note:
        Handles truncated data gracefully by trimming to complete samples.
    """
    audio_int16 = np.frombuffer(data, dtype=np.int16)
    
    # Calculate complete samples and trim any partial data
    n_samples = len(audio_int16) // channels
    audio_int16 = audio_int16[:n_samples * channels]
    
    # De-interleave: reshape to (T, C) then transpose to (C, T)
    audio_int16 = audio_int16.reshape(n_samples, channels).T
    
    # Convert to float32 normalized to [-1.0, 1.0]
    audio_f32 = audio_int16.astype(np.float32) / 32767.0
    
    return torch.from_numpy(audio_f32)


# =============================================================================
# PIPE-BASED RE-ENCODING - Fast In-Memory Processing
# =============================================================================

def reencode_with_pipe(
    waveform: torch.Tensor,
    sample_rate: int,
    codec: str,
    bitrate_kbps: int = 64,
) -> Tuple[torch.Tensor, int]:
    """
    Re-encode audio via FFmpeg using stdin/stdout pipes - NO DISK I/O.
    
    This is the fastest re-encoding method as it keeps all audio data in memory.
    Uses a two-pass approach:
    1. ENCODE: Raw PCM → Compressed format (via ffmpeg encode)
    2. DECODE: Compressed → Raw PCM (via ffmpeg decode)
    
    The audio quality degradation from the encode-decode round-trip simulates
    real-world transmission through the codec, which is useful for training
    robust spoofing detection models.
    
    Args:
        waveform: Input audio tensor of shape (channels, samples) or (samples,)
                  Should be float32 with values in [-1.0, 1.0]
        sample_rate: Input sample rate in Hz (e.g., 16000)
        codec: Target codec name (e.g., 'mp3', 'opus', 'flac')
        bitrate_kbps: Target bitrate in kbps (default: 64)
                      Ignored for lossless codecs like flac
        
    Returns:
        Tuple[torch.Tensor, int]: (re-encoded waveform, output sample rate)
        
    Raises:
        RuntimeError: If encoding or decoding fails, or produces empty output
        
    Example:
        >>> audio = torch.randn(1, 16000)  # 1s mono at 16kHz
        >>> augmented, sr = reencode_with_pipe(audio, 16000, 'mp3', 64)
        >>> print(f"MP3 augmentation: {sr}Hz, shape {augmented.shape}")
        
    Note:
        Not all codecs support pipe mode. Use reencode_audio() for automatic
        fallback to file-based encoding when needed.
    """
    original_device = waveform.device
    
    # Normalize input: ensure (C, T) shape, float32, CPU
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (T,) -> (1, T)
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)
    waveform = waveform.cpu()
    
    channels = waveform.shape[0]
    original_channels = channels
    
    # Downmix to mono for codecs that require it
    if channels > 1 and codec in MONO_ONLY_CODECS:
        waveform = waveform.mean(dim=0, keepdim=True)
        channels = 1
    
    # Determine target sample rate for this codec
    target_sr = get_target_sample_rate(codec)
    if codec in TELEPHONY_CODECS or codec == "gsm":
        target_sr = 8000
    else:
        target_sr = max(target_sr, 16000)
    
    codec_lib = get_ffmpeg_encoder_name(codec)
    codec_fmt = get_codec_format(codec)
    
    # Convert tensor to raw PCM bytes for stdin
    pcm_input = _tensor_to_pcm_bytes(waveform)
    
    # =========================================================================
    # STEP 1: ENCODE - stdin (raw PCM) → stdout (compressed audio)
    # =========================================================================
    cmd_encode = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        # Input specification: raw PCM from stdin
        "-f", "s16le",              # Format: signed 16-bit little-endian
        "-ar", str(sample_rate),    # Input sample rate
        "-ac", str(channels),       # Input channels
        "-i", "pipe:0",             # Read from stdin
        # Output settings
        "-ar", str(target_sr),      # Resample to target rate
        "-ac", str(channels),       # Keep channel count
    ]
    
    # Add codec-specific encoding parameters
    if codec == "mp3":
        cmd_encode += ["-acodec", codec_lib, "-b:a", f"{bitrate_kbps}k"]
    elif codec == "opus":
        # Opus works well at lower bitrates for speech
        opus_bitrate = min(bitrate_kbps, 64)
        cmd_encode += ["-acodec", codec_lib, "-b:a", f"{opus_bitrate}k", "-application", "voip"]
    elif codec == "flac":
        # Lossless - compression level affects speed/size, not quality
        cmd_encode += ["-acodec", codec_lib, "-compression_level", "5"]
    elif codec in {"g711a", "g711u"}:
        # PCM codecs don't use bitrate
        cmd_encode += ["-acodec", codec_lib]
    elif codec == "ogg":
        # Vorbis quality scale (0-10, higher = better)
        cmd_encode += ["-acodec", codec_lib, "-q:a", "3"]
    else:
        # Default: use specified bitrate
        cmd_encode += ["-acodec", codec_lib, "-b:a", f"{bitrate_kbps}k"]
    
    # Output format for pipe
    cmd_encode += ["-f", codec_fmt, "pipe:1"]
    
    # Run encoder
    try:
        encode_result = subprocess.run(
            cmd_encode,
            input=pcm_input,
            capture_output=True,
            timeout=30,  # 30 second timeout prevents hangs
        )
        if encode_result.returncode != 0:
            error_msg = encode_result.stderr.decode('utf-8', errors='replace')
            raise RuntimeError(f"Encode failed for {codec}: {error_msg}")
        encoded_audio = encode_result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Encode timeout for codec {codec} (>30s)")
    
    if len(encoded_audio) == 0:
        raise RuntimeError(f"Encode produced empty output for codec {codec}")
    
    # =========================================================================
    # STEP 2: DECODE - stdin (compressed) → stdout (raw PCM)
    # =========================================================================
    cmd_decode = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        # Input: compressed audio from stdin
        "-f", codec_fmt,
        "-i", "pipe:0",
        # Output: raw PCM to stdout
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-ac", str(channels),
        "pipe:1",
    ]
    
    # Run decoder
    try:
        decode_result = subprocess.run(
            cmd_decode,
            input=encoded_audio,
            capture_output=True,
            timeout=30,
        )
        if decode_result.returncode != 0:
            error_msg = decode_result.stderr.decode('utf-8', errors='replace')
            raise RuntimeError(f"Decode failed for {codec}: {error_msg}")
        pcm_output = decode_result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Decode timeout for codec {codec} (>30s)")
    
    if len(pcm_output) == 0:
        raise RuntimeError(f"Decode produced empty output for codec {codec}")
    
    # Convert PCM bytes back to tensor
    output = _pcm_bytes_to_tensor(pcm_output, channels)
    
    # Restore original channel count if we downmixed
    if original_channels > 1 and codec in MONO_ONLY_CODECS:
        output = output.repeat(original_channels, 1)
    
    return output.to(original_device), target_sr


# =============================================================================
# FILE-BASED RE-ENCODING - Reliable Fallback for Complex Codecs
# =============================================================================

# Check for newer torchaudio codec APIs
# Feature-detection shim for newer torchaudio releases that expose a TorchCodec-backed
# save/load API alongside (or instead of) the classic torchaudio.save/load.
_HAS_TORCHCODEC_SAVE = hasattr(torchaudio, "save_with_torchcodec")
_HAS_TORCHCODEC_LOAD = hasattr(torchaudio, "load_with_torchcodec")


def save_wav(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """
    Save waveform to a WAV file using the best available torchaudio method.
    
    Prefers save_with_torchcodec when available (newer torchaudio versions),
    falls back to standard save() otherwise.
    
    Args:
        path: Output file path
        waveform: Audio tensor (channels, samples)
        sample_rate: Sample rate in Hz
    """
    if _HAS_TORCHCODEC_SAVE:
        torchaudio.save_with_torchcodec(path, _to_cpu_f32(waveform), sample_rate)
    else:
        torchaudio.save(path, _to_cpu_f32(waveform), sample_rate)


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio from file using the best available torchaudio method.
    
    Prefers load_with_torchcodec when available, falls back to standard load().
    
    Args:
        path: Input audio file path
        
    Returns:
        Tuple[torch.Tensor, int]: (waveform tensor, sample rate)
    """
    if _HAS_TORCHCODEC_LOAD:
        try:
            return torchaudio.load_with_torchcodec(path)
        except Exception:
            pass  # Fall through to standard load
    return torchaudio.load(path)


def reencode_with_files(
    waveform: torch.Tensor,
    sample_rate: int,
    codec: str,
    bitrate_kbps: int
) -> Tuple[torch.Tensor, int]:
    """
    Re-encode audio via FFmpeg using temporary files on disk.
    
    This is the FALLBACK method for codecs that require proper container formats
    (e.g., AAC/M4A, ALAC, WMA, AMR). These codecs need file headers/atoms that
    can't be streamed through pipes.
    
    Process:
    1. Save input as WAV file
    2. Encode WAV → target codec file
    3. Decode target codec → WAV file
    4. Load and return the decoded audio
    5. Clean up all temporary files
    
    Args:
        waveform: Input audio tensor (channels, samples) or (samples,)
        sample_rate: Input sample rate in Hz
        codec: Target codec name
        bitrate_kbps: Target bitrate in kbps

    Returns:
        Tuple[torch.Tensor, int]: (re-encoded waveform, output sample rate)
        
    Note:
        Slower than pipe mode due to disk I/O. Use reencode_audio() for
        automatic selection of the best method.
    """
    original_device = waveform.device
    
    # Normalize input
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)
    waveform = waveform.to("cpu")

    original_channels = waveform.shape[0]
    
    # Downmix for mono-only codecs
    if original_channels > 1 and codec in MONO_ONLY_CODECS:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_channels = waveform.shape[0]

    # Create temporary files (all will be cleaned up in finally block)
    # The three temp files map to the three stages of the round trip: the raw input WAV,
    # the codec-encoded file, and the WAV decoded back from it.
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    enc_tmp = tempfile.NamedTemporaryFile(suffix=get_codec_extension(codec), delete=False)
    dec_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()
    enc_tmp.close()
    dec_tmp.close()

    try:
        # Step 1: Save input as WAV
        save_wav(wav_tmp.name, waveform, sample_rate)

        codec_lib = get_ffmpeg_encoder_name(codec)

        # Handle special bitrate requirements for speech codecs
        if codec == "amr":
            # AMR-NB only supports specific bitrates
            # These are the fixed sets of standardized 3GPP AMR-NB/AMR-WB coding-mode bitrates;
            amr_kbps_allowed = (4.75, 5.15, 5.9, 6.7, 7.4, 7.95, 10.2, 12.2)
            bitrate_kbps = float(min(amr_kbps_allowed, key=lambda v: abs(v - bitrate_kbps)))
        elif codec == "amrwb":
            # AMR-WB only supports specific bitrates
            amrwb_kbps_allowed = (6.60, 8.85, 12.65, 14.25, 15.85, 18.25, 19.85, 23.05, 23.85)
            bitrate_kbps = float(min(amrwb_kbps_allowed, key=lambda v: abs(v - bitrate_kbps)))

        # Determine target sample rate
        target_sr = get_target_sample_rate(codec)
        if codec in TELEPHONY_CODECS or codec == "gsm":
            target_sr = 8000
        else:
            target_sr = max(target_sr, 16000)

        # Vorbis requires minimum sample rate
        if codec == "ogg" and target_sr < 22050:
            target_sr = 24000

        # Build encoding command
        cmd_enc = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", wav_tmp.name,
            "-ar", str(target_sr),
            "-ac", str(target_channels),
        ]

        # Codec-specific bitrate handling
        g726_kb = None
        if codec not in {"g711a", "g711u", "wav", "flac", "gsm", "wma", "g722"}:
            if codec == "g726":
                # G.726 only supports specific bitrates
                allowed = [16, 24, 32, 40]
                g726_kb = min(allowed, key=lambda v: abs(v - bitrate_kbps))
            elif codec == "ogg":
                # Vorbis uses quality scale instead of bitrate
                # Approximate linear mapping from a 56 kbps floor to Vorbis's 0-10 quality scale,
                # stepping one quality point per ~16 kbps.
                kbps_eff = max(56, int(bitrate_kbps))
                q = max(0, min(5, round((kbps_eff - 56) / 16) + 2))
                cmd_enc += ["-q:a", str(int(q))]
            else:
                cmd_enc += ["-b:a", f"{bitrate_kbps}k"]

        # Add codec-specific options
        if codec == "opus":
            opus_bitrate = 16  # Good for speech
            cmd_enc += ["-acodec", codec_lib, "-b:a", f"{opus_bitrate}k", "-vbr", "on", "-application", "voip"]
        elif codec in {"g711a", "g711u"}:
            cmd_enc += ["-acodec", codec_lib, "-f", "wav"]
            # Update extension for PCM codecs
            os.remove(enc_tmp.name)
            enc_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            enc_tmp.close()
        elif codec == "g726":
            kb = g726_kb if g726_kb is not None else 32
            cmd_enc += ["-acodec", "adpcm_g726", "-f", "wav", "-b:a", f"{kb}k"]
            os.remove(enc_tmp.name)
            enc_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            enc_tmp.close()
        elif codec == "gsm":
            cmd_enc += ["-acodec", codec_lib, "-f", "gsm"]
        # Enforces a higher minimum bitrate for wideband WMA (32 kHz+) than narrowband,
        # since WMA quality degrades sharply below these thresholds.
        elif codec == "wma":
            floor = 64 if target_sr >= 32000 else 24
            kb = max(int(bitrate_kbps), floor)
            cmd_enc += ["-acodec", codec_lib, "-b:a", f"{kb}k", "-f", "asf"]
        elif codec == "flac":
            cmd_enc += ["-acodec", codec_lib, "-compression_level", "5"]
        elif codec == "amr":
            cmd_enc += ["-acodec", codec_lib, "-strict", "experimental"]
        elif codec == "ogg":
            cmd_enc += ["-acodec", codec_lib]
        elif codec == "g722":
            cmd_enc += ["-acodec", codec_lib]
        else:
            cmd_enc += ["-acodec", codec_lib]

        cmd_enc += [enc_tmp.name]

        # Step 2: Encode
        subprocess.run(cmd_enc, check=True, capture_output=True, text=True)

        # Step 3: Decode back to WAV
        cmd_dec = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", enc_tmp.name,
            "-f", "wav",
            "-acodec", "pcm_s16le",
            "-ar", str(target_sr),
            "-ac", str(target_channels),
            dec_tmp.name,
        ]
        subprocess.run(cmd_dec, check=True, capture_output=True, text=True)

        # Step 4: Load decoded audio
        y, new_sr = load_audio(dec_tmp.name)

        # Restore original channel count
        if original_channels > 1 and codec in MONO_ONLY_CODECS and y.shape[0] == 1:
            y = y.repeat(original_channels, 1)

        # Ensure correct sample rate
        # Some codecs decode back at a container-imposed rate that can differ from the
        # requested target_sr; resample to guarantee the returned rate always matches it.
        if new_sr != target_sr:
            y = torchaudio.functional.resample(y, orig_freq=new_sr, new_freq=target_sr)
            new_sr = target_sr

        return y.to(original_device), int(new_sr)

    finally:
        # Step 5: Clean up ALL temporary files
        for path in (wav_tmp.name, enc_tmp.name, dec_tmp.name):
            try:
                os.remove(path)
            except OSError:
                pass  # File may not exist or already deleted


# =============================================================================
# SMART RE-ENCODING - Automatic Method Selection
# =============================================================================

def reencode_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    codec: str,
    bitrate_kbps: int = 64,
    prefer_pipe: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Smart re-encoding function that automatically chooses the best method.
    
    This is the RECOMMENDED entry point for codec augmentation. It:
    1. Checks if FFmpeg is available
    2. Tries fast pipe mode first (if supported by codec)
    3. Falls back to file mode for problematic codecs
    4. Falls back to torchaudio for basic codecs if FFmpeg unavailable
    
    Args:
        waveform: Input audio tensor (channels, samples) or (samples,)
                  Should be float32 with values in [-1.0, 1.0]
        sample_rate: Input sample rate in Hz
        codec: Target codec name (e.g., 'mp3', 'opus', 'aac', 'flac')
        bitrate_kbps: Target bitrate in kbps (default: 64)
        prefer_pipe: If True, try pipe mode first (default: True)
                     Set False to always use file mode
        
    Returns:
        Tuple[torch.Tensor, int]: (re-encoded waveform, output sample rate)
        
    Raises:
        RuntimeError: If FFmpeg unavailable and codec not supported by torchaudio
        
    Example:
        >>> # Basic usage
        >>> augmented, sr = reencode_audio(audio, 16000, 'mp3', 64)
        
        >>> # Force file mode (more compatible but slower)
        >>> augmented, sr = reencode_audio(audio, 16000, 'aac', 96, prefer_pipe=False)
        
        >>> # Works even without FFmpeg for basic codecs
        >>> augmented, sr = reencode_audio(audio, 16000, 'flac')
    """
    # Check FFmpeg availability
    # wav/flac work without FFmpeg since torchaudio's own backends read/write them
    # natively; every other codec needs FFmpeg for encode/decode.
    if not have_ffmpeg():
        if codec in TORCHAUDIO_SAFE_CODECS:
            return reencode_with_torchaudio(waveform, sample_rate, codec)
        raise RuntimeError(
            f"FFmpeg not available and codec '{codec}' not supported by torchaudio. "
            f"Install FFmpeg or use one of: {TORCHAUDIO_SAFE_CODECS}"
        )
    
    # Try pipe mode first for supported codecs (faster, no disk I/O)
    if prefer_pipe and codec not in FILE_ONLY_CODECS:
        try:
            return reencode_with_pipe(waveform, sample_rate, codec, bitrate_kbps)
        except Exception:
            # Pipe mode failed - fall through to file mode
            pass
    
    # Use file-based encoding (always works, but slower)
    return reencode_with_files(waveform, sample_rate, codec, bitrate_kbps)


def reencode_with_torchaudio(
    waveform: torch.Tensor,
    sample_rate: int,
    codec: str
) -> Tuple[torch.Tensor, int]:
    """
    Fallback re-encoding using only torchaudio (no FFmpeg required).
    
    This provides basic codec support for environments without FFmpeg.
    Only supports codecs that torchaudio can natively handle: wav, flac.
    
    Args:
        waveform: Input audio tensor
        sample_rate: Input sample rate
        codec: Codec name (should be in TORCHAUDIO_SAFE_CODECS)
        
    Returns:
        Tuple[torch.Tensor, int]: (re-encoded waveform, output sample rate)
        
    Note:
        This still uses a temporary file, but doesn't require FFmpeg.
    """
    original_device = waveform.device

    # Normalize input
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    original_channels = waveform.shape[0]
    if original_channels > 1 and codec in MONO_ONLY_CODECS:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Determine target sample rate
    target_sr = get_target_sample_rate(codec)
    if codec in TELEPHONY_CODECS or codec == "gsm":
        target_sr = 8000
    else:
        target_sr = max(target_sr, 16000)

    # Resample if needed
    if target_sr != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=target_sr
        )

    # Save and reload through torchaudio
    tmp = tempfile.NamedTemporaryFile(suffix=get_codec_extension(codec), delete=False)
    tmp.close()
    
    try:
        torchaudio.save(tmp.name, _to_cpu_f32(waveform), target_sr, format=codec)
        y, new_sr = load_audio(tmp.name)
        
        # Restore channels if needed
        if original_channels > 1 and codec in MONO_ONLY_CODECS and y.shape[0] == 1:
            y = y.repeat(original_channels, 1)
            
        # Ensure correct sample rate
        if new_sr != target_sr:
            y = torchaudio.functional.resample(y, orig_freq=new_sr, new_freq=target_sr)
            new_sr = target_sr
            
        return y.to(original_device), int(new_sr)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


# =============================================================================
# LEGACY ALIASES - Backwards Compatibility
# =============================================================================

# Keep old function name for backwards compatibility with existing code
reencode_with_ffmpeg = reencode_with_files


# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    # ===== Constants =====
    "CODEC_BUNDLES",          # Dict of codec groups: voip, consumer, lossless, all
    "TELEPHONY_CODECS",       # Set of 8kHz telephony codecs
    "TORCHAUDIO_SAFE_CODECS", # Set of codecs usable without FFmpeg
    "CODEC_BITRATE_CAPS",     # Dict of per-codec max bitrates
    "MONO_ONLY_CODECS",       # Set of mono-only codecs
    "PIPE_FRIENDLY_CODECS",   # Set of codecs that work with pipe mode
    "FILE_ONLY_CODECS",       # Set of codecs requiring file mode
    
    # ===== Detection Functions =====
    "have_ffmpeg",            # Check if FFmpeg is available
    "get_ffmpeg_supported_encoders",  # Get set of available encoders
    
    # ===== Codec Mapping Functions =====
    "get_ffmpeg_encoder_name",  # Codec name -> FFmpeg encoder flag
    "get_codec_extension",      # Codec name -> file extension
    "get_codec_format",         # Codec name -> FFmpeg format name
    "get_target_sample_rate",   # Codec name -> required sample rate
    "filter_supported_codecs",  # Filter codecs by FFmpeg capabilities
    
    # ===== I/O Functions =====
    "save_wav",               # Save waveform to WAV file
    "load_audio",             # Load audio from file
    
    # ===== Re-encoding Functions (NEW API) =====
    "reencode_audio",         # RECOMMENDED: Smart auto-select method
    "reencode_with_pipe",     # Fast pipe mode (no disk I/O)
    "reencode_with_files",    # Reliable file mode (disk I/O)
    "reencode_with_torchaudio",  # Pure torchaudio fallback
    
    # ===== Legacy =====
    "reencode_with_ffmpeg",   # Alias for reencode_with_files (backwards compat)
]
