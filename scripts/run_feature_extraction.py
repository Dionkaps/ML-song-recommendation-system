import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extract_features import run_feature_extraction


def main():
    parser = argparse.ArgumentParser(
        description="Run handcrafted feature extraction on a local audio directory."
    )
    parser.add_argument("--audio-dir", default="audio_files")
    parser.add_argument("--results-dir", default="output/features")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--executor",
        choices=["auto", "thread", "process"],
        default="auto",
        help="Executor backend to use for parallel feature extraction.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute features even when a complete bundle already exists.",
    )
    args = parser.parse_args()

    run_feature_extraction(
        audio_dir=args.audio_dir,
        results_dir=args.results_dir,
        workers=args.workers,
        executor_type=args.executor,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
