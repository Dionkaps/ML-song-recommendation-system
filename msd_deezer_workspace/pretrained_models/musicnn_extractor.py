"""
MusicNN penultimate-layer embedding extractor (200-dim).

Reference:
  Pons & Serra (2019) "musicnn: Pre-trained CNNs for music audio tagging"
  https://github.com/jordipons/musicnn
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .base import BaseExtractor, SuppressStderr


# musicnn relies on the legacy Keras 2 API (tf.keras.layers.batch_normalization,
# etc.) that was removed in Keras 3 (shipped with TF 2.16+). Setting this env
# var *before* TensorFlow is imported routes `tf.keras` through the `tf-keras`
# shim package, restoring Keras 2 semantics for the musicnn call.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


logger = logging.getLogger(__name__)


class MusicNNExtractor(BaseExtractor):
    """MusicNN penultimate-layer embedding extractor.

    Uses the `MSD_musicnn` model variant trained on Million Song Dataset
    auto-tagging (50 tags). The penultimate dense layer captures a compact
    representation encoding timbral and temporal patterns learned from
    ~200 K labeled songs.

    The extractor processes audio in 3-second patches (with default overlap)
    and returns the mean-pooled penultimate activations across all patches.

    Notes:
      - musicnn uses TensorFlow and handles its own audio loading/resampling.
      - Runs comfortably on CPU; GPU gives marginal speedup for this size.
    """

    name = "musicnn"
    sample_rate = 16000
    embedding_dim = 200

    MODEL_VARIANT = "MSD_musicnn"
    INPUT_LENGTH_SEC = 3

    def __init__(self) -> None:
        try:
            from musicnn.extractor import extractor  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "musicnn package not found. Install with:\n"
                "  pip install musicnn\n"
                "(Also requires a compatible TensorFlow 2.x install.)"
            ) from exc

        self._extractor_fn = extractor
        logger.info("MusicNN extractor initialized (200-dim, CPU)")

    def extract(self, audio_path: str) -> np.ndarray:
        with SuppressStderr():
            _taggram, _tags, features = self._extractor_fn(
                file_name=audio_path,
                model=self.MODEL_VARIANT,
                input_length=self.INPUT_LENGTH_SEC,
                input_overlap=None,   # default overlap
                extract_features=True,
            )

        # features['penultimate'] shape: (num_patches, 200)
        penultimate = np.asarray(features["penultimate"], dtype=np.float32)
        if penultimate.ndim == 1:
            return penultimate
        return penultimate.mean(axis=0).astype(np.float32)
