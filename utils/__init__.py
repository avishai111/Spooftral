"""
Voxtral utility modules for audio spoofing detection.

This package contains:
- augmentation: Audio augmentation (noise, speed, codec, RawBoost, etc.)
- codec: Codec re-encoding utilities (FFmpeg integration)
- config: Configuration loading and management
- dataset: ASVspoof dataset loaders and data utilities
- build_model: Model building and wrapper classes
- helper_function: Utility functions for training/evaluation
- training_evaluate_classification: Training and evaluation loops
- checkpoint_utils: Checkpoint save/load utilities
- losses: Loss functions
- freezing: Model parameter freezing utilities

Note: Modules are imported explicitly where needed rather than here,
to avoid circular import issues with config.py which requires
VOXTRAL_CONFIG to be set first.
"""

# Convenience re-exports for common helpers
from .helper_function import get_audio_files, original_voxtral_inference  # noqa: F401
from .codec import load_audio  # noqa: F401
