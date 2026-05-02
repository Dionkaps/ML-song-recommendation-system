"""
MERT-v1-330M embedding extractor (1024-dim).

Reference:
  Li et al. (2023) "MERT: Acoustic Music Understanding Model with
  Large-Scale Self-supervised Training" (ICLR 2024)
  https://arxiv.org/abs/2306.00107
  https://huggingface.co/m-a-p/MERT-v1-330M
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

import librosa
import numpy as np

from .base import BaseExtractor, resolve_device


logger = logging.getLogger(__name__)


class MERTExtractor(BaseExtractor):
    """MERT-v1-330M embedding extractor.

    Loads the 330M-parameter variant: 24 Transformer encoder layers, hidden
    size 1024. The embedding is the mean across all 25 hidden states
    (1 input-embedding layer + 24 encoder layers), then mean-pooled over
    time frames. This captures both low-level acoustic information (early
    layers) and high-level musical semantics (later layers).

    Runs on CUDA, MPS, or CPU. CPU is slow (~60-90 s/song) but functional
    for small-scale testing on laptops.

    GPU saturation features (when device == "cuda"):
      * TF32 matmul + cuDNN autotune for fixed-shape conv kernels.
      * `torch.inference_mode()` + bfloat16 autocast for the forward pass
        (bf16 falls back to fp16 on Ampere/older where bf16 isn't native).
      * Native batched inference via `extract_batch_from_arrays`, used by
        the orchestrator to push N songs through one forward pass.
    """

    name = "mert"
    sample_rate = 24000
    embedding_dim = 1024

    HF_MODEL_NAME = "m-a-p/MERT-v1-330M"

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
        self._is_cuda = self.device.startswith("cuda")
        self._autocast_dtype = self._pick_autocast_dtype()

        if self._is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        logger.info(
            "Loading MERT-v1-330M on %s (first run downloads ~1.3 GB)...",
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
        logger.info(
            "MERT extractor initialized (1024-dim, %s, autocast=%s)",
            self.device,
            getattr(self._autocast_dtype, "__repr__", lambda: "off")(),
        )

    def _pick_autocast_dtype(self):
        """Pick the best autocast dtype for the active device.

        bfloat16 is preferred on A100/H100/Ada/MPS-bf16 (no scaler needed,
        wider dynamic range). Falls back to float16 elsewhere on CUDA.
        Returns None for CPU (autocast on CPU yields little for transformers
        and adds overhead).
        """
        torch = self._torch
        if not self._is_cuda:
            return None
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    def _autocast_ctx(self):
        if self._autocast_dtype is None or not self._is_cuda:
            return nullcontext()
        return self._torch.autocast(device_type="cuda", dtype=self._autocast_dtype)

    # ── Per-file path (kept for single-song callers) ───────────────────

    def extract(self, audio_path: str) -> np.ndarray:
        y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return self.extract_from_array(y)

    def extract_from_array(self, y: np.ndarray) -> np.ndarray:
        """Same as `extract` but skips the librosa.load step.

        Used by the orchestrator's async prefetch pipeline so the GPU isn't
        idled by disk I/O between songs.
        """
        return self.extract_batch_from_arrays([y])[0]

    # ── Batched path (used by the orchestrator for GPU saturation) ─────

    def extract_batch_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Run a single forward pass over a batch of N waveforms.

        Returns an array of shape (N, 1024) dtype float32. All inputs in a
        batch should be the same length (the preprocessing pipeline crops
        every song to 29 s @ 24 kHz, so this holds in practice). When
        lengths differ, the HF processor pads to the longest sample and
        the attention_mask is honored when computing the time-axis mean.
        """
        if not arrays:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        torch = self._torch

        # The HF Wav2Vec2 feature extractor accepts a list of 1-D arrays
        # and returns padded tensors plus an attention_mask.
        inputs = self.processor(
            list(arrays),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        device_inputs = {
            k: v.to(self.device, non_blocking=self._is_cuda)
            for k, v in inputs.items()
        }
        attention_mask = device_inputs.get("attention_mask")

        with torch.inference_mode(), self._autocast_ctx():
            outputs = self.model(**device_inputs, output_hidden_states=True)
            # hidden_states: tuple of 25 tensors, each (B, T, 1024)
            stacked = torch.stack(outputs.hidden_states, dim=0)  # (L, B, T, 1024)

            if attention_mask is not None and stacked.shape[2] > 1:
                # Reduce the audio attention mask to the model's token
                # resolution by simple striding -- MERT downsamples by a
                # factor of (input_samples / num_tokens), and the ratio is
                # roughly constant across the batch.
                token_count = stacked.shape[2]
                mask = self._downsample_mask(attention_mask, token_count)
                # mask: (B, T) -> (1, B, T, 1)
                mask = mask.to(stacked.dtype).unsqueeze(0).unsqueeze(-1)
                weighted = stacked * mask
                token_sum = weighted.sum(dim=2)            # (L, B, 1024)
                denom = mask.sum(dim=2).clamp_min(1e-6)    # (1, B, 1)
                time_mean = token_sum / denom              # (L, B, 1024)
            else:
                time_mean = stacked.mean(dim=2)            # (L, B, 1024)

            embedding = time_mean.mean(dim=0)              # (B, 1024)
            # Cast + move to CPU inside inference_mode so we never touch
            # an inference tensor from the outer scope. numpy() then yields
            # a plain ndarray that the rest of the pipeline can handle.
            host = embedding.to(dtype=torch.float32, device="cpu").numpy()

        return host.astype(np.float32, copy=False)

    def _downsample_mask(self, attention_mask, target_len: int):
        """Stride-sample a (B, S_audio) mask down to (B, target_len)."""
        torch = self._torch
        if attention_mask.shape[1] == target_len:
            return attention_mask
        # Linearly map each output index to a source index.
        idx = torch.linspace(
            0, attention_mask.shape[1] - 1, steps=target_len,
            device=attention_mask.device,
        ).long()
        return attention_mask.index_select(dim=1, index=idx)
