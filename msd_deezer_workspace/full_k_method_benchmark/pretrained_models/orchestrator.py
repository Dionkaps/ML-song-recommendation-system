"""
PretrainedEmbeddingExtractor -- orchestrates extraction across multiple
pretrained audio models.

Runs GPU-friendly extractors (MERT, EnCodecMAE) in batches so a single
dedicated GPU stays saturated. CPU-only extractors (MusicNN) still run
one song at a time -- batching them via TF doesn't help the GPU pipeline.

Per-song NPZ files are written incrementally so a crash mid-batch only
loses that batch's work, not the entire run.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from tqdm import tqdm

from safety import assert_inside_workspace

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

    # ── Dynamic batch-size sizer tunables ──────────────────────────────
    # Probe free VRAM every N successful batches. Too small = overhead;
    # too large = slow to react when a peer GPU process exits.
    VRAM_PROBE_EVERY_BATCHES = 8
    # Headroom multiplier: grow only when free VRAM >= factor * recent_peak.
    # 1.5 leaves enough slack for allocator fragmentation + the next batch.
    VRAM_HEADROOM_FACTOR = 1.5
    # On first probe (no peak recorded yet), grow if free VRAM >= this
    # fraction of total VRAM.
    VRAM_BOOTSTRAP_FRACTION = 0.30

    def __init__(
        self,
        output_dir: str | Path | None = None,
        models: list[str] | None = None,
        device: str = "auto",
    ) -> None:
        self.output_dir = assert_inside_workspace(
            output_dir if output_dir else default_output_dir(),
            "pretrained_output_dir",
        )
        self.raw_output_dir = self.output_dir / "raw"
        self.device_request = device

        # Per-model state for the dynamic batch sizer. `_effective_batch`
        # tracks the current in-use batch size (may grow on free VRAM,
        # shrink on OOM). `_recent_peak` is the high-water mark of VRAM
        # usage at the current batch size. `_grow_counter` drives the
        # "every N batches" probe cadence.
        self._effective_batch: dict[str, int] = {}
        self._recent_peak: dict[str, int] = {}
        self._grow_counter: dict[str, int] = {}

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

    def _file_duration_sec(self, path: Path) -> float:
        try:
            return round(float(librosa.get_duration(path=str(path))), 6)
        except Exception:
            try:
                return round(float(librosa.get_duration(filename=str(path))), 6)
            except Exception:
                return 0.0

    # ── Single file processing (kept for external callers) ─────────────

    def process_file(
        self,
        file_path: str,
        skip_existing: bool = True,
        preloaded_arrays: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Extract embeddings from a single audio file using all active models.

        Kept for external callers / tests. The directory loop now goes
        through `_process_batch` for GPU saturation, but single-file
        callers still get a working entry point here.
        """
        path = Path(file_path)
        result: dict[str, Any] = {
            "file": path.name,
            "status": "success",
            "embeddings": {},
            "errors": {},
            "duration_sec": self._file_duration_sec(path),
            # Pretrained copy produced by DualAudioPreprocessor is 24 kHz.
            # This is the sample rate of the file on disk, not the model's
            # internal rate (each extractor resamples to its own native rate
            # if it differs -- MusiCNN to 16 kHz, MERT/EnCodecMAE stay at 24).
            "sample_rate": 24000,
        }

        preloaded = self._existing_models_in_npz(path.stem) if skip_existing else {}

        for name, extractor in self.extractors.items():
            if name in preloaded:
                result["embeddings"][name] = preloaded[name]
                continue
            try:
                if (
                    preloaded_arrays is not None
                    and name in preloaded_arrays
                    and hasattr(extractor, "extract_from_array")
                ):
                    embedding = extractor.extract_from_array(preloaded_arrays[name])
                else:
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

        self._persist_npz(path.stem, result["embeddings"], result["duration_sec"], result["sample_rate"])
        return result

    def _persist_npz(
        self,
        audio_stem: str,
        embeddings: dict[str, np.ndarray],
        duration_sec: float,
        sample_rate: int,
    ) -> None:
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = self.raw_output_dir / f"{audio_stem}.npz"
        npz_data: dict[str, np.ndarray] = {
            name: emb.astype(np.float32) for name, emb in embeddings.items()
        }
        npz_data[DURATION_KEY] = np.float32(duration_sec)
        npz_data[SAMPLE_RATE_KEY] = np.int32(sample_rate)
        np.savez_compressed(npz_path, **npz_data)

    # ── Audio prefetching ─────────────────────────────────────────────

    def _gpu_prefetch_targets(self) -> list[str]:
        """Return the names of active extractors that support extract_from_array.

        MusicNN has its own TF runtime that reads files internally, so it is
        excluded. When no GPU extractor supports the array-based path, the
        prefetch pipeline short-circuits and behavior is identical to the
        pre-prefetch code path.
        """
        return [
            name for name, ex in self.extractors.items()
            if hasattr(ex, "extract_from_array")
        ]

    def _prefetch_arrays(
        self, file_path: Path, target_names: list[str],
    ) -> dict[str, np.ndarray]:
        """Load the waveform for each target extractor at its native sample rate.

        Returns a dict keyed by extractor name. Entries for which loading
        fails are simply omitted -- the batch path will mark those
        (file, model) pairs as errors.
        """
        arrays: dict[str, np.ndarray] = {}
        for name in target_names:
            extractor = self.extractors[name]
            try:
                y, _ = librosa.load(
                    str(file_path), sr=extractor.sample_rate, mono=True,
                )
                arrays[name] = y
            except Exception as exc:
                logger.warning(
                    "prefetch %s for %s failed: %s", name, file_path.name, exc,
                )
        return arrays

    # ── Batched GPU dispatch ──────────────────────────────────────────

    def _maybe_grow_batch(self, name: str, cap: int) -> None:
        """Probe free VRAM and possibly double `self._effective_batch[name]`.

        Called at the start of each `_run_model_batch` invocation. Growth
        fires only once per `VRAM_PROBE_EVERY_BATCHES` successful calls,
        to keep the overhead negligible. The decision uses `_recent_peak`
        (the high-water mark of VRAM at the current batch) with a
        `VRAM_HEADROOM_FACTOR` safety margin, or a bootstrap fraction
        before the first peak is recorded.

        This is the mechanism that makes the co-resident GPU scenario
        elastic: when a sibling worker finishes, VRAM frees up, and the
        next probe detects it and scales the remaining worker's batch up.
        """
        current = self._effective_batch.get(name, 1)
        if current >= cap:
            return

        count = self._grow_counter.get(name, 0) + 1
        if count < self.VRAM_PROBE_EVERY_BATCHES:
            self._grow_counter[name] = count
            return
        self._grow_counter[name] = 0

        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                return
            free, total = torch.cuda.mem_get_info()
        except Exception:
            return

        peak = self._recent_peak.get(name, 0)
        if peak > 0:
            needed = int(peak * self.VRAM_HEADROOM_FACTOR)
        else:
            needed = int(total * self.VRAM_BOOTSTRAP_FRACTION)

        if free >= needed:
            new = min(cap, current * 2)
            if new > current:
                logger.info(
                    "%s: VRAM free=%.1f/%.1fGB (peak~%.1fGB), "
                    "growing batch %d -> %d",
                    name, free / 1e9, total / 1e9, peak / 1e9, current, new,
                )
                self._effective_batch[name] = new
                # Reset the peak so the next successful batch records
                # usage at the NEW size, not the old one.
                self._recent_peak[name] = 0

    def _snapshot_peak(self, name: str) -> None:
        """Record the CUDA peak-allocated-memory figure for the current model.

        Called after each successful batch dispatch. Resetting + reading
        on every call gives us the peak that the last batch alone caused
        (instead of the cumulative peak across all batches), which is what
        the sizer needs to decide whether a 2x batch would fit.
        """
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                return
            peak = int(torch.cuda.max_memory_allocated())
            if peak > self._recent_peak.get(name, 0):
                self._recent_peak[name] = peak
            # Reset the stat window so the next batch's peak stands alone.
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _run_model_batch(
        self,
        name: str,
        arrays: list[np.ndarray],
        base_batch_size: int,
        max_batch_size: int,
    ) -> list[np.ndarray | None]:
        """Run `name` over `arrays` in dynamically-sized mini-batches.

        Starts from `self._effective_batch[name]` (seeded from
        `base_batch_size` on first call) and can grow up to
        `max_batch_size` when free VRAM allows, or halve down to 1 on
        CUDA OOM. On a persistent non-OOM failure at batch=1, marks the
        chunk as failed and advances. Returns a list of embeddings (one
        per input array) with None for entries that errored out.
        """
        extractor = self.extractors[name]
        results: list[np.ndarray | None] = [None] * len(arrays)
        if not arrays:
            return results

        # Seed the per-model batch state on first call.
        if name not in self._effective_batch:
            self._effective_batch[name] = max(1, int(base_batch_size))

        # Attempt to grow before dispatching this batch (cheap if we're
        # not at the probe boundary yet).
        self._maybe_grow_batch(name, max(1, int(max_batch_size)))

        # Choose the call path based on what the extractor exposes.
        has_batch_api = hasattr(extractor, "extract_batch_from_arrays")
        has_array_api = hasattr(extractor, "extract_from_array")

        i = 0
        current_bs = self._effective_batch[name]
        while i < len(arrays):
            chunk_end = min(i + current_bs, len(arrays))
            chunk = arrays[i:chunk_end]
            try:
                if has_batch_api:
                    out = extractor.extract_batch_from_arrays(chunk)
                    for k, emb in enumerate(out):
                        results[i + k] = np.asarray(emb, dtype=np.float32)
                elif has_array_api:
                    for k, y in enumerate(chunk):
                        results[i + k] = np.asarray(
                            extractor.extract_from_array(y), dtype=np.float32,
                        )
                else:
                    # No array-based entry point at all (shouldn't happen
                    # for the registered GPU extractors). Caller should
                    # have routed these through process_file() instead.
                    raise RuntimeError(
                        f"{name} has no array-based extractor; cannot batch."
                    )
                # Successful dispatch: refresh VRAM peak for the sizer.
                self._snapshot_peak(name)
                i = chunk_end
            except Exception as exc:  # noqa: BLE001 - we want to see CUDA OOM too
                if self._is_oom(exc) and current_bs > 1:
                    new_bs = max(1, current_bs // 2)
                    logger.warning(
                        "%s OOM at batch=%d -> halving to %d and retrying",
                        name, current_bs, new_bs,
                    )
                    self._cuda_empty_cache()
                    current_bs = new_bs
                    # Persist the shrink so future calls start from here
                    # instead of re-attempting the oversized batch.
                    self._effective_batch[name] = current_bs
                    # Invalidate the recorded peak; it was taken at the
                    # too-large batch that we just retreated from.
                    self._recent_peak[name] = 0
                    continue
                # Either non-OOM or already at batch=1: mark every song in
                # the chunk as failed and move on.
                logger.error(
                    "%s batch [%d:%d] failed: %s", name, i, chunk_end, exc,
                )
                for k in range(len(chunk)):
                    results[i + k] = None
                i = chunk_end

        return results

    def _is_oom(self, exc: BaseException) -> bool:
        """Detect CUDA OOM without importing torch at module level."""
        if exc.__class__.__name__ == "OutOfMemoryError":
            return True
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda oom" in msg

    def _cuda_empty_cache(self) -> None:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _process_batch(
        self,
        batch_files: list[Path],
        batch_arrays: list[dict[str, np.ndarray] | None],
        skip_existing: bool,
        batch_size: int,
        max_batch_size: int,
        stats: dict[str, Any],
    ) -> None:
        """Drive one batch of B files through every active model.

        For each model, gathers the songs in the batch that still need it
        (skip_existing + missing-prefetch handling), invokes the model in
        a single forward pass, then stitches results back into per-song
        NPZ files.
        """
        if not batch_files:
            return

        per_song_existing: list[dict[str, np.ndarray]] = [
            self._existing_models_in_npz(p.stem) if skip_existing else {}
            for p in batch_files
        ]
        per_song_embeddings: list[dict[str, np.ndarray]] = [
            dict(existing) for existing in per_song_existing
        ]
        per_song_errors: list[dict[str, str]] = [{} for _ in batch_files]
        per_song_duration: list[float] = [
            self._file_duration_sec(p) for p in batch_files
        ]

        for name, extractor in self.extractors.items():
            # Decide which song indices in the batch still need this model.
            needs: list[int] = []
            inputs: list[np.ndarray] = []
            for idx, path in enumerate(batch_files):
                if name in per_song_embeddings[idx]:
                    continue  # already extracted (from prior NPZ)

                # GPU extractors prefer the prefetched waveform.
                preloaded = batch_arrays[idx] or {}
                if name in preloaded and hasattr(extractor, "extract_from_array"):
                    needs.append(idx)
                    inputs.append(preloaded[name])
                    continue

                # Either prefetch failed for this (song, model) or the
                # extractor is path-based (MusicNN). Fall back per-song.
                try:
                    emb = extractor.extract(str(path))
                    expected_dim = extractor.embedding_dim
                    if emb.shape[0] != expected_dim:
                        logger.warning(
                            "%s produced %d-dim embedding (expected %d) for %s",
                            name, emb.shape[0], expected_dim, path.name,
                        )
                    per_song_embeddings[idx][name] = emb.astype(np.float32)
                except Exception as exc:
                    per_song_errors[idx][name] = str(exc)
                    logger.error(
                        "Error extracting %s from %s: %s", name, path.name, exc,
                    )

            if not needs:
                continue

            # Run the batched forward pass. _run_model_batch handles OOM
            # shrinkage and free-VRAM growth via self._effective_batch.
            batch_out = self._run_model_batch(
                name, inputs, batch_size, max_batch_size,
            )
            for slot, idx in enumerate(needs):
                emb = batch_out[slot]
                if emb is None:
                    per_song_errors[idx][name] = "batch extraction failed"
                    continue
                expected_dim = extractor.embedding_dim
                if emb.shape[0] != expected_dim:
                    logger.warning(
                        "%s produced %d-dim embedding (expected %d) for %s",
                        name, emb.shape[0], expected_dim, batch_files[idx].name,
                    )
                per_song_embeddings[idx][name] = emb.astype(np.float32)

        # Persist per-song NPZs and update stats.
        for idx, path in enumerate(batch_files):
            embs = per_song_embeddings[idx]
            errs = per_song_errors[idx]
            if not embs:
                stats["errors"] += 1
                continue
            self._persist_npz(
                path.stem, embs, per_song_duration[idx], 24000,
            )
            stats["processed"] += 1
            for name in self.active_model_names:
                if name in embs:
                    stats["per_model_success"][name] += 1
                if name in errs:
                    stats["per_model_errors"][name] += 1

    # ── Directory processing ──────────────────────────────────────────

    def process_directory(
        self,
        directory: str | Path,
        skip_existing: bool = True,
        limit: int | None = None,
        shard_index: int = 0,
        num_shards: int = 1,
        generate_csvs: bool = True,
        prefetch: int = 16,
        batch_size: int = 16,
        max_batch_size: int | None = None,
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
            Applied before sharding, so --limit N --num-shards K distributes
            N files across K shards.
        shard_index : int
            Zero-based index of this shard when running in sharded mode.
            Each shard picks every Nth file from the sorted list.
        num_shards : int
            Total number of parallel shards (1 = no sharding, default).
            Use to run multiple independent workers against the same output
            directory -- e.g., several CPU-bound MusicNN workers.
        generate_csvs : bool
            If True, regenerate CSVs from all NPZs after extraction. Must be
            False when `num_shards > 1` because concurrent shards would race
            each other on the CSV files. The launcher/merge script is
            responsible for producing CSVs once every shard has finished.
        prefetch : int
            Number of songs to pre-decode in the background for GPU extractors
            (MERT, EnCodecMAE). Keeps the GPU fed while the next song's audio
            is being read from disk. 0 disables prefetch. Default: 16.
        batch_size : int
            Starting mini-batch size for GPU-friendly extractors. Songs
            are accumulated into batches of this size and pushed through
            each model in a single forward pass. The effective batch
            grows up to `max_batch_size` when free VRAM allows (that is
            what picks up the slack when a sibling GPU worker finishes)
            and halves on CUDA OOM. Default: 16.
        max_batch_size : int | None
            Ceiling for the dynamic batch sizer. Defaults to
            `4 * batch_size` (e.g. starting batch 16 -> cap 64). Set
            equal to `batch_size` to disable growth and keep the batch
            fixed.
        """
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if not 0 <= shard_index < num_shards:
            raise ValueError(
                f"shard_index must be in [0, {num_shards}), got {shard_index}"
            )
        if num_shards > 1 and generate_csvs:
            logger.warning(
                "generate_csvs=True with num_shards=%d would race concurrent "
                "shards on CSV writes; forcing generate_csvs=False.",
                num_shards,
            )
            generate_csvs = False

        input_dir = Path(directory)
        if not input_dir.exists():
            logger.error("Directory not found: %s", directory)
            return {"error": "Directory not found"}

        files: list[Path] = []
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            files.extend(input_dir.glob(f"*{ext}"))
        files = sorted(files)
        total_before_shard = len(files)

        if limit is not None and limit > 0:
            files = files[: int(limit)]

        if num_shards > 1:
            files = [f for i, f in enumerate(files) if i % num_shards == shard_index]

        if not files:
            if num_shards > 1:
                logger.warning(
                    "Shard %d/%d has no files to process (source had %d).",
                    shard_index, num_shards, total_before_shard,
                )
            else:
                logger.warning("No audio files found in %s", input_dir)
            return {"total": 0}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = max(1, int(batch_size))
        if max_batch_size is None:
            max_batch_size = batch_size * 4
        max_batch_size = max(batch_size, int(max_batch_size))

        stats: dict[str, Any] = {
            "total": len(files),
            "processed": 0,
            "errors": 0,
            "per_model_success": {name: 0 for name in self.active_model_names},
            "per_model_errors": {name: 0 for name in self.active_model_names},
            "device_request": self.device_request,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "batch_size": batch_size,
            "max_batch_size": max_batch_size,
        }

        print(f"\n{'=' * 60}")
        print("PRETRAINED EMBEDDING EXTRACTION")
        print(f"{'=' * 60}")
        if num_shards > 1:
            print(f"Shard:            {shard_index}/{num_shards} "
                  f"({len(files)} of {total_before_shard} files)")
        print(f"Files to process: {len(files)}")
        print(f"Active models:    {', '.join(self.active_model_names)}")
        print(f"Device request:   {self.device_request}")
        print(f"Batch size:       start={batch_size}, cap={max_batch_size} (dynamic)")
        print(f"Output directory: {self.output_dir}")
        print(f"Resume mode:      {'ON' if skip_existing else 'OFF'}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        gpu_targets = self._gpu_prefetch_targets()
        # Prefetch the next P*B songs so a full batch is always queued.
        # When user disables prefetch entirely (prefetch=0), we still load
        # waveforms synchronously inside the batch path below.
        use_prefetch = prefetch > 0 and len(gpu_targets) > 0
        prefetch_window = max(prefetch, batch_size) if use_prefetch else 0

        executor: ThreadPoolExecutor | None = None
        pending: dict[int, Future[dict[str, np.ndarray]]] = {}

        if use_prefetch:
            executor = ThreadPoolExecutor(
                max_workers=prefetch, thread_name_prefix="audio-prefetch",
            )
            # Seed the queue with up to `prefetch_window` lookahead.
            for i in range(min(prefetch_window, len(files))):
                pending[i] = executor.submit(
                    self._prefetch_arrays, files[i], gpu_targets,
                )
            print(
                f"Audio prefetch:   {prefetch} thread(s), "
                f"window={prefetch_window} feeding {', '.join(gpu_targets)}\n"
            )

        # Synchronous fallback for songs whose prefetch failed or wasn't
        # used: load on demand inside the batch.
        def _ensure_arrays(
            idx: int, file_path: Path,
        ) -> dict[str, np.ndarray] | None:
            if use_prefetch and idx in pending:
                try:
                    return pending.pop(idx).result()
                except Exception as exc:
                    logger.warning(
                        "prefetch failed for %s: %s -- falling back to sync load",
                        file_path.name, exc,
                    )
            if not gpu_targets:
                return None
            return self._prefetch_arrays(file_path, gpu_targets)

        try:
            with tqdm(total=len(files), desc="Extracting embeddings") as pbar:
                for batch_start in range(0, len(files), batch_size):
                    batch_end = min(batch_start + batch_size, len(files))
                    batch_files = files[batch_start:batch_end]

                    batch_arrays: list[dict[str, np.ndarray] | None] = []
                    for offset, file_path in enumerate(batch_files):
                        idx = batch_start + offset
                        arrays = _ensure_arrays(idx, file_path)
                        batch_arrays.append(arrays)
                        # Keep the prefetch window saturated.
                        if use_prefetch and executor is not None:
                            next_i = idx + prefetch_window
                            if next_i < len(files) and next_i not in pending:
                                pending[next_i] = executor.submit(
                                    self._prefetch_arrays,
                                    files[next_i], gpu_targets,
                                )

                    self._process_batch(
                        batch_files, batch_arrays,
                        skip_existing=skip_existing,
                        batch_size=batch_size,
                        max_batch_size=max_batch_size,
                        stats=stats,
                    )
                    pbar.update(len(batch_files))
        finally:
            if executor is not None:
                # cancel_futures avoids blocking on the remaining prefetch
                # tasks when the caller Ctrl-C's mid-loop.
                executor.shutdown(wait=False, cancel_futures=True)

        elapsed = time.time() - start_time

        csv_paths: dict[str, str] = {}
        if generate_csvs:
            print("\nGenerating CSV summaries...")
            csv_paths = generate_all_csvs(
                self.raw_output_dir, self.output_dir,
                input_dir, self.active_model_names,
            )
        else:
            print(
                "\nSkipping CSV generation (sharded run). Run the merge "
                "script after all shards finish to produce the CSVs."
            )

        stats["elapsed_sec"] = round(elapsed, 1)
        stats["csv_files"] = csv_paths

        if num_shards > 1:
            summary_name = (
                f"extraction_summary_shard{shard_index:02d}of{num_shards:02d}.json"
            )
        else:
            summary_name = "extraction_summary.json"
        summary_path = self.output_dir / summary_name
        summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        print(f"\n{'=' * 60}")
        print("EXTRACTION SUMMARY")
        print(f"{'=' * 60}")
        if num_shards > 1:
            print(f"Shard:        {shard_index}/{num_shards}")
        print(f"Total files:  {stats['total']}")
        print(f"Processed:    {stats['processed']}")
        print(f"Errors:       {stats['errors']}")
        print(f"Batch size:   start={stats['batch_size']}, "
              f"cap={stats['max_batch_size']}, "
              f"final={self._effective_batch}")
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
