"""
CLI entry point for pretrained embedding extraction.

Extracts song-level embeddings from three complementary pretrained models
and writes clustering-ready CSVs:

  1. MusicNN    (Pons & Serra, 2019)   --  200-dim, supervised CNN, CPU-friendly
  2. MERT-v1-95M (Li et al., 2023)     --  768-dim, self-supervised Transformer
  3. EnCodecMAE-large-st (Pepino, 2023) -- 1024-dim, self-supervised MAE

Outputs (under --output-dir, default: pretrained_embeddings/):
  raw/<song>.npz           per-song NPZ with one array per model
  musicnn_vectors.csv      MusicNN embeddings   (metadata + 200 cols)
  mert_vectors.csv         MERT embeddings      (metadata + 768 cols)
  encodecmae_vectors.csv   EnCodecMAE embeddings (metadata + 1024 cols)
  feature_vectors.csv      Fused embeddings     (metadata + 1992 cols;
                           L2-normalized per-model before concatenation)

The fused CSV is directly compatible with the existing clustering pipeline:

  python run_kmeans_clustering.py --features-path pretrained_embeddings/

Examples:

  # Full extraction on the DGX server (auto-detects CUDA)
  python extract_pretrained_embeddings.py

  # Local laptop test: 5 files, MusicNN only, CPU (no GPU needed)
  python extract_pretrained_embeddings.py --models musicnn --limit 5 --device cpu

  # Verify all models load without extracting anything (useful before long runs)
  python extract_pretrained_embeddings.py --check-models

  # Rebuild CSVs after changing fusion parameters (no re-extraction)
  python extract_pretrained_embeddings.py --csv-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

from pretrained_models import (
    AVAILABLE_MODELS,
    PretrainedEmbeddingExtractor,
    default_audio_dir,
    default_output_dir,
    resolve_device,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
logging.getLogger("audioread").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.ERROR)


def _parse_models(csv: str) -> list[str]:
    names = [m.strip().lower() for m in csv.split(",") if m.strip()]
    unknown = [n for n in names if n not in AVAILABLE_MODELS]
    if unknown:
        raise SystemExit(
            f"Unknown model(s): {unknown}. "
            f"Choose from: {list(AVAILABLE_MODELS.keys())}"
        )
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pretrained audio embeddings (MusicNN + MERT + EnCodecMAE) "
            "for clustering-ready MSD/Deezer previews."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", default=str(default_audio_dir()),
        help="Directory containing preprocessed audio files (default: audio/)",
    )
    parser.add_argument(
        "--output-dir", default=str(default_output_dir()),
        help="Directory for embedding outputs (default: pretrained_embeddings/)",
    )
    parser.add_argument(
        "--models", default=",".join(AVAILABLE_MODELS.keys()),
        help=(
            "Comma-separated list of models to use "
            f"(choices: {','.join(AVAILABLE_MODELS.keys())}; "
            "default: all three)"
        ),
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu", "mps"],
        help=(
            "Device for GPU-based models (MERT, EnCodecMAE). "
            "'auto' picks cuda > mps > cpu. MusicNN always uses its own "
            "TensorFlow runtime (typically CPU). Default: auto"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help=(
            "Process only the first N audio files (useful for testing). "
            "Default: process all files."
        ),
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-extract every file even if embeddings already exist in NPZ.",
    )
    parser.add_argument(
        "--csv-only", action="store_true",
        help="Only regenerate CSVs from existing NPZ files (skip extraction).",
    )
    parser.add_argument(
        "--check-models", action="store_true",
        help=(
            "Attempt to load each requested model and report status, then "
            "exit without processing any files. Use this before a long run "
            "to verify your environment is set up correctly."
        ),
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help=(
            "Zero-based index of this shard when running in sharded mode. "
            "Used with --num-shards. Default: 0."
        ),
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help=(
            "Total number of parallel shards. Each worker picks every Nth "
            "file from the sorted list. Useful for running multiple CPU "
            "MusicNN workers in parallel against the same output directory. "
            "When >1, CSV generation is skipped in-process -- run "
            "merge_sharded_embeddings.py after all shards finish. Default: 1."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def run_check_models(model_names: list[str], device: str) -> int:
    """Load each requested model and report pass/fail. Returns exit code."""
    resolved = resolve_device(device)
    print(f"\n{'=' * 60}")
    print("MODEL ENVIRONMENT CHECK")
    print(f"{'=' * 60}")
    print(f"Requested device: {device}  (resolved: {resolved})")
    print(f"Models to check:  {', '.join(model_names)}")
    print(f"{'=' * 60}\n")

    failures = 0
    for name in model_names:
        print(f"[{name}] loading...", flush=True)
        try:
            cls = AVAILABLE_MODELS[name]
            if name == "musicnn":
                inst = cls()
            else:
                inst = cls(device=device)
            print(f"  OK -- {inst.describe()}\n")
            del inst
        except Exception as exc:
            failures += 1
            print(f"  FAIL -- {type(exc).__name__}: {exc}\n")

    if failures:
        print(f"{failures}/{len(model_names)} model(s) failed to load.")
        print("Install missing dependencies from requirements-pretrained.txt")
        return 1

    print("All models loaded successfully. You are ready to run extraction.")
    return 0


def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.num_shards < 1:
        logger.error("--num-shards must be >= 1, got %d", args.num_shards)
        return 2
    if not 0 <= args.shard_index < args.num_shards:
        logger.error(
            "--shard-index must be in [0, %d), got %d",
            args.num_shards, args.shard_index,
        )
        return 2
    if args.num_shards > 1 and args.csv_only:
        logger.error("--csv-only cannot be combined with --num-shards > 1")
        return 2

    model_names = _parse_models(args.models)

    # Fast path: just verify models can load and exit.
    if args.check_models:
        return run_check_models(model_names, args.device)

    input_dir = Path(args.input_dir)
    if not args.csv_only and not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 2

    try:
        extractor = PretrainedEmbeddingExtractor(
            output_dir=args.output_dir,
            models=model_names,
            device=args.device,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 3

    if args.csv_only:
        extractor.generate_csvs_only(input_dir)
        return 0

    extractor.process_directory(
        input_dir,
        skip_existing=not args.no_resume,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        generate_csvs=(args.num_shards == 1),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
