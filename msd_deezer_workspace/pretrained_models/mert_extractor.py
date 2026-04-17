"""
MERT-v1-95M embedding extractor (768-dim).

Reference:
  Li et al. (2023) "MERT: Acoustic Music Understanding Model with
  Large-Scale Self-supervised Training" (ICLR 2024)
  https://arxiv.org/abs/2306.00107
  https://huggingface.co/m-a-p/MERT-v1-95M
"""

from __future__ import annotations

import logging

import librosa
import numpy as np

from .base import BaseExtractor, resolve_device


logger = logging.getLogger(__name__)


class MERTExtractor(BaseExtractor):
    """MERT-v1-95M embedding extractor.

    Loads the 95M-parameter variant: 12 Transformer encoder layers, hidden
    size 768. The embedding is the mean across all 13 hidden states
    (1 input-embedding layer + 12 encoder layers), then mean-pooled over
    time frames. This captures both low-level acoustic information (early
    layers) and high-level musical semantics (later layers).

    Chosen over the 330M variant for efficiency -- the 330M model shows
    inverse-scaling on several downstream tasks per the MERT paper, and
    consumes ~4x the memory for marginal gains.

    Runs on CUDA, MPS, or CPU. CPU is slow (~20-30 s/song) but functional
    for small-scale testing on laptops.
    """

    name = "mert"
    sample_rate = 24000
    embedding_dim = 768

    HF_MODEL_NAME = "m-a-p/MERT-v1-95M"

    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_device(device)

        try:
            import torch  # type: ignore
            from transformers import AutoModel, Wav2Vec2FeatureExtractor  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "torch + transformers required for MERT. Install with:\n"
                "  pip install torch transformers"
            ) from exc

        self._torch = torch

        logger.info(
            "Loading MERT-v1-95M on %s (first run downloads ~380 MB)...",
            self.device,
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.HF_MODEL_NAME, trust_remote_code=True,
        )
        self.model = (
            AutoModel.from_pretrained(self.HF_MODEL_NAME, trust_remote_code=True)
            .to(self.device)
            .eval()
        )
        logger.info("MERT extractor initialized (768-dim, %s)", self.device)

    def extract(self, audio_path: str) -> np.ndarray:
        y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        inputs = self.processor(
            y, sampling_rate=self.sample_rate, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # hidden_states: tuple of 13 tensors, each (1, time_steps, 768)
            all_hidden = self._torch.stack(outputs.hidden_states)  # (13, 1, T, 768)
            # Mean across layers (dim 0) and time (dim 2)
            embedding = all_hidden.mean(dim=(0, 2)).squeeze(0)     # (768,)

        return embedding.detach().cpu().numpy().astype(np.float32)
