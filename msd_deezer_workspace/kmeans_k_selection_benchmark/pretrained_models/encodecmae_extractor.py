"""
EnCodecMAE-large-st embedding extractor (1024-dim).

Reference:
  Pepino, Riera & Ferrer (2023) "EnCodecMAE: Leveraging neural codecs for
  universal audio representation learning" (Interspeech 2025)
  https://arxiv.org/abs/2309.07391
  https://github.com/habla-liaa/encodecmae
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

import librosa
import numpy as np

from .base import BaseExtractor, resolve_device


logger = logging.getLogger(__name__)


class EnCodecMAEExtractor(BaseExtractor):
    """EnCodecMAE-large-st embedding extractor.

    Loads the `large-st` variant: 20 Transformer encoder layers with
    hidden size 1024, pre-trained to reconstruct EnCodec RVQ discrete
    codes from masked mel-spectrogram patches, with an additional
    self-training stage (hence the "-st" suffix).

    Audio is resampled to 24 kHz and passed through the encoder; the
    output is mean-pooled over the time axis.

    Note: The `encodecmae` package's public API has varied between
    versions. We try several common call patterns (HEAR
    `get_scene_embeddings`, `extract_features_from_array`, etc.). If
    extraction fails on your installed version, adjust `extract()`.

    GPU saturation features (when device == "cuda"):
      * TF32 matmul + cuDNN autotune for the encoder.
      * `torch.inference_mode()` + bfloat16 autocast across the whole
        batch loop (the encodecmae library's API is per-array, so we
        amortize autocast overhead by entering it once per batch).
      * `extract_batch_from_arrays` is the orchestrator's entry point;
        it loops over the batch under one autocast context.
    """

    name = "encodecmae"
    sample_rate = 24000
    embedding_dim = 1024

    # Mel-spectrogram 256-bin input, EnCodec-large encoder, with
    # self-training stage. See `encodecmae.hub.get_available_models()` for
    # all variants hosted at huggingface.co/lpepino/encodecmae-v2.
    MODEL_VARIANT = "mel256-ec-large_st"

    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_device(device)

        try:
            import torch  # type: ignore
            from encodecmae import load_model  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "encodecmae package not found. Install from source:\n"
                "  git clone https://github.com/habla-liaa/encodecmae.git\n"
                "  cd encodecmae && pip install -e .\n"
                "Also requires: pip install torch"
            ) from exc

        self._torch = torch
        self._is_cuda = self.device.startswith("cuda")
        self._autocast_dtype = self._pick_autocast_dtype()

        if self._is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        logger.info(
            "Loading EnCodecMAE %s on %s (first run downloads weights)...",
            self.MODEL_VARIANT, self.device,
        )
        self.model = load_model(self.MODEL_VARIANT, device=self.device)
        # Best-effort eval mode: not all encodecmae wrappers expose .eval().
        if hasattr(self.model, "eval"):
            try:
                self.model.eval()
            except Exception:
                pass
        logger.info(
            "EnCodecMAE extractor initialized (1024-dim, %s, autocast=%s)",
            self.device,
            getattr(self._autocast_dtype, "__repr__", lambda: "off")(),
        )

    def _pick_autocast_dtype(self):
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

    # ── Per-file path ──────────────────────────────────────────────────

    def extract(self, audio_path: str) -> np.ndarray:
        y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return self.extract_from_array(y, audio_path=audio_path)

    def extract_from_array(
        self, y: np.ndarray, audio_path: str | None = None,
    ) -> np.ndarray:
        """Same as `extract` but skips the librosa.load step.

        Used by the orchestrator's async prefetch pipeline so the GPU isn't
        idled by disk I/O between songs. `audio_path` is optional and only
        used if the installed encodecmae API version falls back to the
        file-path call path.
        """
        # no_grad (not inference_mode) because the third-party encodecmae
        # library is opaque and may internally toggle autograd in hooks;
        # inference_mode would refuse the re-enable. The bf16 autocast is
        # where the real speedup comes from anyway.
        with self._torch.no_grad(), self._autocast_ctx():
            features = self._call_model(y, audio_path or "")
        return self._features_to_embedding(features)

    # ── Batched path (used by the orchestrator) ────────────────────────

    def extract_batch_from_arrays(
        self,
        arrays: list[np.ndarray],
        audio_paths: list[str] | None = None,
    ) -> np.ndarray:
        """Run the encoder over a batch of N waveforms.

        encodecmae's high-level `extract_features_from_array` API is
        per-array, so we loop within a single inference_mode + autocast
        context. That alone gives a meaningful speedup on the GPU because
        the autocast cost is amortized and bfloat16 matmul kicks in for
        the encoder's transformer layers.

        Returns an array of shape (N, 1024) dtype float32.
        """
        if not arrays:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if audio_paths is None:
            audio_paths = [""] * len(arrays)

        embeddings: list[np.ndarray] = []
        # no_grad (not inference_mode) because the third-party encodecmae
        # library is opaque and may internally toggle autograd in hooks;
        # inference_mode would refuse the re-enable. The bf16 autocast is
        # where the real speedup comes from anyway.
        with self._torch.no_grad(), self._autocast_ctx():
            for y, path in zip(arrays, audio_paths):
                features = self._call_model(y, path)
                embeddings.append(self._features_to_embedding(features))
        return np.stack(embeddings).astype(np.float32)

    # ── Internal helpers ───────────────────────────────────────────────

    def _features_to_embedding(self, features) -> np.ndarray:
        if isinstance(features, self._torch.Tensor):
            # bf16/fp16 autocast outputs cast back to float32 for the
            # numpy round-trip + downstream l2_normalize.
            features = features.float().detach().cpu().numpy()

        features = np.asarray(features, dtype=np.float32)

        # Collapse batch then mean-pool over time if needed
        while features.ndim > 2:
            features = features.squeeze(0)
        if features.ndim == 2:
            embedding = features.mean(axis=0)
        else:
            embedding = features

        return embedding.astype(np.float32)

    def _call_model(self, y_array: np.ndarray, audio_path: str):
        """Try known API patterns on the EncodecMAE model object."""
        model = self.model

        # Canonical habla-liaa API (current): takes a 1-D numpy array at fs
        if hasattr(model, "extract_features_from_array"):
            return model.extract_features_from_array(y_array)
        # Alternative: load directly from file path
        if hasattr(model, "extract_features_from_file"):
            return model.extract_features_from_file(audio_path)
        # HEAR benchmark wrappers expose this method
        if hasattr(model, "get_scene_embeddings"):
            waveform = self._torch.from_numpy(y_array).unsqueeze(0).float().to(self.device)
            return model.get_scene_embeddings(waveform)
        # Older / alternative API names
        if hasattr(model, "extract_features"):
            return model.extract_features(y_array)
        if hasattr(model, "get_features"):
            return model.get_features(y_array)
        # Last resort: model is directly callable
        waveform = self._torch.from_numpy(y_array).unsqueeze(0).float().to(self.device)
        return model(waveform)
