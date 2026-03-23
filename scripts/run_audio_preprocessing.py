import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv
from src.audio_preprocessing.processor import AudioPreprocessor


def _resolve_output_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    return value


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
    parser.add_argument(
        "--details-output",
        default="",
        help=(
            "Optional UTF-8 JSON path for the full preprocessing stats payload, "
            "including per-file details when --details is enabled."
        ),
    )
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
    summary = stats
    if isinstance(stats, dict) and "details" in stats:
        summary = {key: value for key, value in stats.items() if key != "details"}

    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=True))

    if args.details:
        detail_rows = 0
        if isinstance(stats, dict):
            detail_rows = len(stats.get("details") or [])
        print(f"Detailed preprocessing records collected: {detail_rows}")

    if args.details_output:
        output_path = _resolve_output_path(args.details_output)
        output_path.write_text(
            json.dumps(_json_ready(stats), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Detailed preprocessing JSON written to: {output_path}")
    elif args.details:
        print(
            "Tip: pass --details-output <path> to persist the full UTF-8 "
            "per-file preprocessing payload."
        )


if __name__ == "__main__":
    main()
