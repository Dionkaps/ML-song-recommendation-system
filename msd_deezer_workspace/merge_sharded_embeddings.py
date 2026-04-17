"""
Merge per-model pretrained-embedding output directories into a unified one.

When `extract_pretrained_embeddings.py` is run in parallel across multiple
output directories (one per model, plus optional CPU-shard workers for
MusicNN), each directory holds `raw/<song>.npz` files containing only its
model's embedding array (plus `__duration_sec__` / `__sample_rate__` meta
keys).

This script walks every source directory, merges the per-model NPZs for
each song stem into a single unified NPZ under `--output-dir/raw/`, and
then calls `generate_all_csvs` to emit the per-model CSVs plus the fused
1992-dim `feature_vectors.csv` that is drop-in compatible with the
clustering pipeline.

The merge is idempotent: re-running over an already-merged output directory
simply rewrites the unified NPZs and refreshes the CSVs.

Usage:
    python merge_sharded_embeddings.py \
        --sources pretrained_embeddings_musicnn \
                  pretrained_embeddings_mert \
                  pretrained_embeddings_encodecmae \
        --output-dir pretrained_embeddings \
        --audio-dir audio
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from pretrained_models import (
    AVAILABLE_MODELS,
    default_audio_dir,
    default_output_dir,
)
from pretrained_models.csv_writer import (
    DURATION_KEY,
    SAMPLE_RATE_KEY,
    generate_all_csvs,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_META_KEYS = {DURATION_KEY, SAMPLE_RATE_KEY}


def _collect_stems(source_raws: list[Path]) -> list[str]:
    """Return the sorted union of song stems present in any source raw/ dir."""
    stems: set[str] = set()
    for raw in source_raws:
        stems.update(p.stem for p in raw.glob("*.npz"))
    return sorted(stems)


def _merge_stem(
    stem: str, source_raws: list[Path], target_raw: Path,
) -> tuple[bool, set[str]]:
    """Merge every source NPZ for `stem` into one unified NPZ.

    Returns (wrote_file, models_present). `wrote_file` is False if no source
    contained the stem (should not happen when stems come from `_collect_stems`).
    """
    merged: dict[str, np.ndarray] = {}
    duration: float | None = None
    sample_rate: int | None = None
    found = False
    models_present: set[str] = set()

    for raw in source_raws:
        src = raw / f"{stem}.npz"
        if not src.exists():
            continue
        found = True
        try:
            with np.load(src) as data:
                for key in data.files:
                    if key == DURATION_KEY:
                        if duration is None:
                            duration = float(data[key])
                    elif key == SAMPLE_RATE_KEY:
                        if sample_rate is None:
                            sample_rate = int(data[key])
                    else:
                        # Embedding array, keyed by model name
                        merged[key] = np.asarray(data[key]).copy()
                        models_present.add(key)
        except Exception as exc:
            logger.warning("Could not read %s: %s", src, exc)

    if not found:
        return False, models_present

    target_raw.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = dict(merged)
    out[DURATION_KEY] = np.float32(duration if duration is not None else 0.0)
    out[SAMPLE_RATE_KEY] = np.int32(sample_rate if sample_rate is not None else 22050)
    np.savez_compressed(target_raw / f"{stem}.npz", **out)
    return True, models_present


def _canonical_model_order(models_present: set[str]) -> list[str]:
    """Return models_present ordered by AVAILABLE_MODELS (canonical), then extras."""
    canonical = [m for m in AVAILABLE_MODELS.keys() if m in models_present]
    extras = sorted(m for m in models_present if m not in AVAILABLE_MODELS)
    return canonical + extras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-model sharded NPZ outputs into a unified embedding directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help=(
            "One or more per-model output directories. Each must contain a "
            "raw/ subfolder full of <song>.npz files written by "
            "extract_pretrained_embeddings.py."
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(default_output_dir()),
        help=(
            "Unified output directory. Merged NPZs go to <output-dir>/raw/ "
            "and CSVs go to <output-dir>/. Default: pretrained_embeddings/"
        ),
    )
    parser.add_argument(
        "--audio-dir", default=str(default_audio_dir()),
        help=(
            "Directory containing the original audio files (used to fill "
            "the audio_path column of the emitted CSVs). Default: audio/"
        ),
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help=(
            "Skip the NPZ merge step and go straight to CSV regeneration. "
            "Use only when <output-dir>/raw/ already contains unified NPZs "
            "from a previous merge run."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_dirs = [Path(s).resolve() for s in args.sources]
    source_raws: list[Path] = []
    for src in source_dirs:
        raw = src / "raw"
        if not raw.is_dir():
            logger.error("Source directory has no raw/ subfolder: %s", src)
            return 2
        source_raws.append(raw)

    output_dir = Path(args.output_dir).resolve()
    target_raw = output_dir / "raw"
    audio_dir = Path(args.audio_dir).resolve()

    all_models_present: set[str] = set()

    if args.skip_merge:
        if not target_raw.is_dir():
            logger.error(
                "--skip-merge requested but %s does not exist.", target_raw,
            )
            return 3
        # Scan existing merged NPZs for model keys so CSV gen knows what to emit.
        for npz_path in target_raw.glob("*.npz"):
            try:
                with np.load(npz_path) as data:
                    for key in data.files:
                        if key not in _META_KEYS:
                            all_models_present.add(key)
            except Exception as exc:
                logger.warning("Could not inspect %s: %s", npz_path, exc)
    else:
        stems = _collect_stems(source_raws)
        if not stems:
            logger.error(
                "No NPZ files found across %d source raw/ directories.",
                len(source_raws),
            )
            return 3

        logger.info(
            "Merging %d song(s) from %d source(s) into %s",
            len(stems), len(source_raws), target_raw,
        )

        written = 0
        for stem in stems:
            ok, models = _merge_stem(stem, source_raws, target_raw)
            if ok:
                written += 1
                all_models_present.update(models)

        logger.info(
            "Merged %d NPZ(s). Models found across merged set: %s",
            written, sorted(all_models_present) or "(none)",
        )

    if not all_models_present:
        logger.error("No model embeddings were found -- cannot generate CSVs.")
        return 4

    active_models = _canonical_model_order(all_models_present)
    logger.info("Generating CSVs for models (order): %s", active_models)

    if not audio_dir.is_dir():
        logger.warning(
            "Audio directory %s does not exist; the audio_path column in "
            "the CSVs will be left blank.", audio_dir,
        )

    paths = generate_all_csvs(target_raw, output_dir, audio_dir, active_models)
    if not paths:
        logger.error("CSV generation produced no files.")
        return 5

    print()
    print("Output:")
    for label, path in paths.items():
        print(f"  {label:10s}  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
