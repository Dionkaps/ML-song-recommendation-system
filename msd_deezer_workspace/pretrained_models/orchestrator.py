"""
PretrainedEmbeddingExtractor -- orchestrates extraction across multiple
pretrained audio models.

Processes audio files sequentially (GPU models share VRAM, can't be
parallelized via ProcessPoolExecutor like the handcrafted pipeline)
and saves per-song NPZ files incrementally for crash-safe resume.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from tqdm import tqdm

from .base import (
    SUPPORTED_AUDIO_EXTENSIONS,
    BaseExtractor,
    default_output_dir,
)
from .csv_writer import (
    DURATION_KEY,
    SAMPLE_RATE_KEY,
    generate_all_csvs,
)
from .encodecmae_extractor import EnCodecMAEExtractor
from .mert_extractor import MERTExtractor
from .musicnn_extractor import MusicNNExtractor


logger = logging.getLogger(__name__)


# ── Model registry ────────────────────────────────────────────────────────

AVAILABLE_MODELS: dict[str, type[BaseExtractor]] = {
    "musicnn": MusicNNExtractor,
    "mert": MERTExtractor,
    "encodecmae": EnCodecMAEExtractor,
}

MODEL_DIMS: dict[str, int] = {
    name: cls.embedding_dim for name, cls in AVAILABLE_MODELS.items()
}


# ── Orchestrator ──────────────────────────────────────────────────────────

class PretrainedEmbeddingExtractor:
    """Extract pretrained embeddings for all songs in a directory.

    Parameters
    ----------
    output_dir : path-like
        Where NPZ files and CSV summaries are written.
    models : list[str] | None
        Subset of AVAILABLE_MODELS keys to use. None = all three.
    device : str
        "auto" (default), "cuda", "cpu", "mps", or explicit "cuda:N".
        MusicNN ignores this (always CPU/TF).
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        models: list[str] | None = None,
        device: str = "auto",
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else default_output_dir()
        self.raw_output_dir = self.output_dir / "raw"
        self.device_request = device

        if models is None:
            models = list(AVAILABLE_MODELS.keys())

        self.extractors: dict[str, BaseExtractor] = {}
        for model_name in models:
            if model_name not in AVAILABLE_MODELS:
                logger.warning("Unknown model '%s' -- skipping", model_name)
                continue
            try:
                cls = AVAILABLE_MODELS[model_name]
                if model_name == "musicnn":
                    self.extractors[model_name] = cls()
                else:
                    self.extractors[model_name] = cls(device=device)
            except Exception as exc:
                logger.warning(
                    "Could not load %s (%s) -- skipping this model",
                    model_name, exc,
                )

        if not self.extractors:
            raise RuntimeError(
                "No models could be loaded. Check that at least one model's "
                "dependencies are installed (see requirements-pretrained.txt)."
            )

        self.active_model_names = list(self.extractors.keys())
        logger.info("Active models: %s", ", ".join(self.active_model_names))

    # ── Resume helpers ────────────────────────────────────────────────

    def _existing_models_in_npz(self, audio_stem: str) -> dict[str, np.ndarray]:
        """Load already-extracted embeddings from an existing NPZ, if any."""
        npz_path = self.raw_output_dir / f"{audio_stem}.npz"
        if not npz_path.exists():
            return {}
        try:
            with np.load(npz_path) as data:
                # Only take keys matching our active model names (skip metadata keys)
                return {
                    key: np.asarray(data[key]).copy()
                    for key in data.files
                    if key in AVAILABLE_MODELS
                }
        except Exception as exc:
            logger.warning("Could not load existing NPZ %s: %s", npz_path.name, exc)
            return {}

    # ── Single file processing ────────────────────────────────────────

    def process_file(
        self,
        file_path: str,
        skip_existing: bool = True,
    ) -> dict[str, Any]:
        """Extract embeddings from a single audio file using all active models.

        Returns a dict with keys: file, status, embeddings, errors,
        duration_sec, sample_rate.
        """
        path = Path(file_path)
        result: dict[str, Any] = {
            "file": path.name,
            "status": "success",
            "embeddings": {},
            "errors": {},
            "duration_sec": 0.0,
            "sample_rate": 22050,
        }

        # Record duration once from the audio file itself
        try:
            result["duration_sec"] = round(float(librosa.get_duration(path=str(path))), 6)
        except Exception:
            try:
                # Fallback for older librosa versions that use `filename=`
                result["duration_sec"] = round(float(librosa.get_duration(filename=str(path))), 6)
            except Exception:
                result["duration_sec"] = 0.0

        # Resume: reuse any embeddings already in the NPZ
        preloaded: dict[str, np.ndarray] = {}
        if skip_existing:
            preloaded = self._existing_models_in_npz(path.stem)

        for name, extractor in self.extractors.items():
            if name in preloaded:
                result["embeddings"][name] = preloaded[name]
                continue
            try:
                embedding = extractor.extract(str(path))
                expected_dim = extractor.embedding_dim
                if embedding.shape[0] != expected_dim:
                    logger.warning(
                        "%s produced %d-dim embedding (expected %d) for %s",
                        name, embedding.shape[0], expected_dim, path.name,
                    )
                result["embeddings"][name] = embedding.astype(np.float32)
            except Exception as exc:
                result["errors"][name] = str(exc)
                logger.error("Error extracting %s from %s: %s", name, path.name, exc)

        if not result["embeddings"]:
            result["status"] = "error"
            return result

        # Persist NPZ with all embeddings plus cached metadata
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = self.raw_output_dir / f"{path.stem}.npz"
        npz_data = {name: emb for name, emb in result["embeddings"].items()}
        npz_data[DURATION_KEY] = np.float32(result["duration_sec"])
        npz_data[SAMPLE_RATE_KEY] = np.int32(result["sample_rate"])
        np.savez_compressed(npz_path, **npz_data)

        return result

    # ── Directory processing ──────────────────────────────────────────

    def process_directory(
        self,
        directory: str | Path,
        skip_existing: bool = True,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Extract embeddings for every audio file in a directory.

        Parameters
        ----------
        directory : path-like
            Directory containing preprocessed audio.
        skip_existing : bool
            If True, reuses embeddings already present in the NPZ for
            each song (crash-safe resume).
        limit : int | None
            If set, processes only the first N files (useful for testing).
        """
        input_dir = Path(directory)
        if not input_dir.exists():
            logger.error("Directory not found: %s", directory)
            return {"error": "Directory not found"}

        files: list[Path] = []
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            files.extend(input_dir.glob(f"*{ext}"))
        files = sorted(files)

        if limit is not None and limit > 0:
            files = files[: int(limit)]

        if not files:
            logger.warning("No audio files found in %s", input_dir)
            return {"total": 0}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)

        stats: dict[str, Any] = {
            "total": len(files),
            "processed": 0,
            "errors": 0,
            "per_model_success": {name: 0 for name in self.active_model_names},
            "per_model_errors": {name: 0 for name in self.active_model_names},
            "device_request": self.device_request,
        }

        print(f"\n{'=' * 60}")
        print("PRETRAINED EMBEDDING EXTRACTION")
        print(f"{'=' * 60}")
        print(f"Files to process: {len(files)}")
        print(f"Active models:    {', '.join(self.active_model_names)}")
        print(f"Device request:   {self.device_request}")
        print(f"Output directory: {self.output_dir}")
        print(f"Resume mode:      {'ON' if skip_existing else 'OFF'}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        for file_path in tqdm(files, desc="Extracting embeddings"):
            result = self.process_file(str(file_path), skip_existing=skip_existing)

            if result["status"] == "error":
                stats["errors"] += 1
            else:
                stats["processed"] += 1
                for name in self.active_model_names:
                    if name in result["embeddings"]:
                        stats["per_model_success"][name] += 1
                    if name in result["errors"]:
                        stats["per_model_errors"][name] += 1

        elapsed = time.time() - start_time

        print("\nGenerating CSV summaries...")
        csv_paths = generate_all_csvs(
            self.raw_output_dir, self.output_dir,
            input_dir, self.active_model_names,
        )

        stats["elapsed_sec"] = round(elapsed, 1)
        stats["csv_files"] = csv_paths

        summary_path = self.output_dir / "extraction_summary.json"
        summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        print(f"\n{'=' * 60}")
        print("EXTRACTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total files:  {stats['total']}")
        print(f"Processed:    {stats['processed']}")
        print(f"Errors:       {stats['errors']}")
        print(f"Elapsed:      {elapsed:.1f}s ({elapsed / max(len(files), 1):.2f}s/file)")
        print()
        for name in self.active_model_names:
            ok = stats["per_model_success"][name]
            err = stats["per_model_errors"][name]
            print(f"  {name:12s}  OK: {ok}  Errors: {err}")
        print()
        for label, path in csv_paths.items():
            print(f"  {label}: {path}")
        print(f"{'=' * 60}\n")

        return stats

    # ── CSV-only mode ─────────────────────────────────────────────────

    def generate_csvs_only(self, audio_dir: str | Path) -> dict[str, str]:
        """Regenerate CSVs from existing NPZ files without re-extracting."""
        audio_dir = Path(audio_dir)
        print(f"\nRegenerating CSVs from {self.raw_output_dir}...")
        csv_paths = generate_all_csvs(
            self.raw_output_dir, self.output_dir,
            audio_dir, self.active_model_names,
        )
        for label, path in csv_paths.items():
            print(f"  {label}: {path}")
        print()
        return csv_paths
