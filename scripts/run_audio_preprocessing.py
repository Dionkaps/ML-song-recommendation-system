import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv
from src.audio_preprocessing.processor import AudioPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Run the supported audio preprocessing pipeline on the local audio library."
    )
    parser.add_argument("--audio-dir", default="audio_files")
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
        help=(
            "Sample-peak safety ceiling in dBFS. "
            "The legacy --max-true-peak alias is kept for compatibility."
        ),
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    processor = AudioPreprocessor(
        target_duration=args.target_duration,
        target_lufs=args.target_lufs,
        max_true_peak=args.max_peak_db,
    )
    stats = processor.process_directory(
        args.audio_dir,
        max_workers=args.workers,
        return_details=args.details,
    )

    print("\nPreprocessing stage finished.")
    print(stats)


if __name__ == "__main__":
    main()
