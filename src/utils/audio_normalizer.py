"""
Legacy compatibility wrapper for audio preprocessing.

This module used to implement a duration-only MP3 normalizer. The supported
workspace path is now `scripts/run_audio_preprocessing.py` backed by
`src.audio_preprocessing.AudioPreprocessor`.

The helper below delegates to the supported pipeline so older imports do not
silently use stale logic.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv
from src.audio_preprocessing.processor import AudioPreprocessor


def normalize_audio_files(
    audio_dir: str,
    verbose: bool = True,
    max_workers: Optional[int] = None,
    target_duration: float = fv.baseline_target_duration_seconds,
    target_lufs: float = fv.baseline_target_lufs,
    max_peak_db: float = fv.baseline_max_true_peak_dbtp,
) -> Dict:
    """
    Run the supported preprocessing pipeline on a directory of audio files.

    Args:
        audio_dir: Directory containing local audio files.
        verbose: Whether to print a compatibility notice.
        max_workers: Optional worker count for parallel preprocessing.
        target_duration: Target duration in seconds.
        target_lufs: Target integrated loudness in LUFS.
        max_peak_db: Sample-peak ceiling in dBFS.

    Returns:
        The stats dictionary returned by `AudioPreprocessor.process_directory`.
    """
    if verbose:
        print(
            "src/utils/audio_normalizer.py is a legacy compatibility wrapper. "
            "Prefer scripts/run_audio_preprocessing.py for the supported entrypoint."
        )

    processor = AudioPreprocessor(
        target_duration=target_duration,
        target_lufs=target_lufs,
        max_true_peak=max_peak_db,
        sample_rate=fv.baseline_sample_rate,
    )
    return processor.process_directory(audio_dir, max_workers=max_workers)


def main() -> Dict:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper around the supported audio preprocessing pipeline."
        )
    )
    parser.add_argument("--audio-dir", default=str(PROJECT_ROOT / "audio_files"))
    parser.add_argument(
        "--target-duration",
        type=float,
        default=fv.baseline_target_duration_seconds,
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=fv.baseline_target_lufs,
    )
    parser.add_argument(
        "--max-peak-db",
        "--max-true-peak",
        dest="max_peak_db",
        type=float,
        default=fv.baseline_max_true_peak_dbtp,
        help="Sample-peak safety ceiling in dBFS.",
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    stats = normalize_audio_files(
        audio_dir=args.audio_dir,
        max_workers=args.workers,
        target_duration=args.target_duration,
        target_lufs=args.target_lufs,
        max_peak_db=args.max_peak_db,
    )
    print(stats)
    return stats


if __name__ == "__main__":
    main()
