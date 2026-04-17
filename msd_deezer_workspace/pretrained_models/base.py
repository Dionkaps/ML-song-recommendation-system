"""
Shared utilities and the BaseExtractor ABC for pretrained embedding extractors.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


# ── Workspace-wide constants ──────────────────────────────────────────────

# Must match clustering/shared.py METADATA_COLUMNS for seamless clustering integration
METADATA_COLUMNS = [
    "file",
    "audio_path",
    "raw_feature_path",
    "msd_track_id",
    "deezer_track_id",
    "sample_rate",
    "duration_sec",
    "frames",
]

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


# ── Device resolution ─────────────────────────────────────────────────────

def resolve_device(requested: str) -> str:
    """Resolve a device spec to a concrete PyTorch device string.

    Accepts "auto" (default choice: cuda > mps > cpu), "cuda", "cpu", "mps",
    or an explicit device like "cuda:0". If torch is unavailable we fall
    back to "cpu" so MusicNN-only runs don't require a torch install.
    """
    if requested != "auto":
        return requested

    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Filename ID extraction ────────────────────────────────────────────────

def extract_ids_from_filename(filename: str) -> dict[str, str]:
    """Parse MSD track ID and Deezer track ID from a filename."""
    msd_match = re.search(r"\[(TR[A-Z0-9]+)\]", filename)
    deezer_match = re.search(r"\[deezer-(\d+)\]", filename, flags=re.IGNORECASE)
    return {
        "msd_track_id": msd_match.group(1) if msd_match else "",
        "deezer_track_id": deezer_match.group(1) if deezer_match else "",
    }


# ── Numerical helpers ─────────────────────────────────────────────────────

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector to unit norm; passes through on zero norm."""
    norm = float(np.linalg.norm(v))
    if norm < 1e-10:
        return v
    return (v / norm).astype(v.dtype, copy=False)


# ── stderr suppression (for TF C-level warnings) ──────────────────────────

class SuppressStderr:
    """Context manager that silences stdout/stderr from native backends.

    Used to keep the progress bar clean while musicnn/TensorFlow emits
    deprecation warnings or mpg123 emits decoder noise.
    """

    def __init__(self) -> None:
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self) -> "SuppressStderr":
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
        return self

    def __exit__(self, *_: object) -> None:
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


# ── BaseExtractor ABC ─────────────────────────────────────────────────────

class BaseExtractor(ABC):
    """Common interface for all pretrained embedding extractors.

    Subclasses must set the three class attributes (`name`, `sample_rate`,
    `embedding_dim`) and implement `extract(audio_path) -> (embedding_dim,)`.
    """

    name: str = ""
    sample_rate: int = 0
    embedding_dim: int = 0

    @abstractmethod
    def extract(self, audio_path: str) -> np.ndarray:
        """Extract a fixed-length embedding from an audio file.

        Returns:
            np.ndarray of shape (embedding_dim,), dtype float32.
        """
        ...

    def describe(self) -> str:
        return f"{self.name} ({self.embedding_dim}-dim @ {self.sample_rate} Hz)"


# ── Runtime paths ─────────────────────────────────────────────────────────

def workspace_root() -> Path:
    """Return the msd_deezer_workspace/ directory (parent of this package)."""
    return Path(__file__).resolve().parents[1]


def default_audio_dir() -> Path:
    return workspace_root() / "audio"


def default_output_dir() -> Path:
    return workspace_root() / "pretrained_embeddings"
