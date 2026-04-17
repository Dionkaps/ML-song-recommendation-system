"""
Pretrained audio embedding extraction package.

Provides MusicNN, MERT, and EnCodecMAE extractors plus an orchestrator
that produces per-model and fused CSV embedding files compatible with
the existing clustering pipeline.
"""

from .base import (
    METADATA_COLUMNS,
    SUPPORTED_AUDIO_EXTENSIONS,
    BaseExtractor,
    default_audio_dir,
    default_output_dir,
    extract_ids_from_filename,
    l2_normalize,
    resolve_device,
)
from .encodecmae_extractor import EnCodecMAEExtractor
from .mert_extractor import MERTExtractor
from .musicnn_extractor import MusicNNExtractor
from .orchestrator import (
    AVAILABLE_MODELS,
    MODEL_DIMS,
    PretrainedEmbeddingExtractor,
)

__all__ = [
    "AVAILABLE_MODELS",
    "BaseExtractor",
    "EnCodecMAEExtractor",
    "MERTExtractor",
    "METADATA_COLUMNS",
    "MODEL_DIMS",
    "MusicNNExtractor",
    "PretrainedEmbeddingExtractor",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "default_audio_dir",
    "default_output_dir",
    "extract_ids_from_filename",
    "l2_normalize",
    "resolve_device",
]
