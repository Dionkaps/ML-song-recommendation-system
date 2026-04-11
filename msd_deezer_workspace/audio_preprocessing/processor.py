"""
Audio Preprocessor - Main orchestrator for audio preprocessing pipeline.

Pipeline Steps:
1. Load audio file (resampled to target sample rate, converted to mono)
2. Duration validation and cropping (default: 29 seconds)
3. Loudness normalization (ITU-R BS.1770, default: -14 LUFS)
4. Sample-peak safety cap (default: -1.0 dBFS)
5. Save processed audio
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict

import librosa
import soundfile as sf
from tqdm import tqdm

try:
    from .duration_handler import DurationHandler
    from .loudness_normalizer import LoudnessNormalizer
except ImportError:  # pragma: no cover
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR.parent) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR.parent))
    from audio_preprocessing.duration_handler import DurationHandler
    from audio_preprocessing.loudness_normalizer import LoudnessNormalizer


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_DIR = WORKSPACE_DIR / "audio"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
logging.getLogger("audioread").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.ERROR)


class _SuppressStderr:
    """Context manager to suppress stderr (catches C-level warnings from mpg123)."""

    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class AudioPreprocessor:
    SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")

    def __init__(
        self,
        target_duration: float = 29.0,
        target_lufs: float = -14.0,
        max_true_peak: float = -1.0,
        sample_rate: int = 22050,
    ):
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.sample_rate = sample_rate
        self.duration_handler = DurationHandler(target_duration=target_duration)
        self.loudness_normalizer = LoudnessNormalizer(
            target_lufs=target_lufs,
            max_true_peak=max_true_peak,
        )

    def process_file(self, file_path: str) -> Dict:
        path = Path(file_path)
        result = {
            "file": path.name,
            "status": "success",
            "actions": [],
            "original_duration": 0.0,
            "final_duration": 0.0,
            "original_lufs": None,
            "final_lufs": None,
            "original_peak_db": None,
            "final_peak_db": None,
            "gain_applied_db": 0.0,
            "error": None,
        }

        try:
            with _SuppressStderr():
                y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            result["original_duration"] = round(duration, 3)

            y, action = self.duration_handler.process(y, sr)
            if action == "too_short":
                os.remove(path)
                result["status"] = "removed"
                result["actions"].append("removed_too_short")
                return result

            if action == "cropped":
                result["actions"].append("cropped")

            result["final_duration"] = round(librosa.get_duration(y=y, sr=sr), 3)

            y, norm_stats = self.loudness_normalizer.process(y, sr)
            result["original_lufs"] = (
                round(norm_stats["original_lufs"], 3) if norm_stats["original_lufs"] is not None else None
            )
            result["final_lufs"] = (
                round(norm_stats["final_lufs"], 3) if norm_stats["final_lufs"] is not None else None
            )
            result["original_peak_db"] = (
                round(norm_stats.get("original_peak_db", 0), 3)
                if norm_stats.get("original_peak_db") is not None
                else None
            )
            result["final_peak_db"] = (
                round(norm_stats.get("final_peak_db", 0), 3)
                if norm_stats.get("final_peak_db") is not None
                else None
            )
            result["gain_applied_db"] = round(norm_stats.get("gain_applied_db", 0), 3)
            result["actions"].extend(norm_stats["actions"])

            if path.suffix.lower() == ".wav":
                output_path = path
            else:
                output_path = path.with_suffix(".wav")

            sf.write(str(output_path), y, sr, subtype="PCM_16")

            if output_path != path and path.exists():
                path.unlink()
                result["actions"].append("converted_to_wav")
                result["file"] = output_path.name
                result["output_file"] = output_path.name

        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            logger.error("Error processing %s: %s", path.name, exc)

        return result

    def process_directory(self, directory: str, max_workers: int | None = None, return_details: bool = False) -> Dict:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error("Directory not found: %s", directory)
            return {"error": "Directory not found"}

        files = []
        for extension in self.SUPPORTED_AUDIO_EXTENSIONS:
            files.extend(dir_path.glob(f"*{extension}"))
        files = sorted(files)

        if not files:
            logger.warning("No audio files found in %s", directory)
            return {"total": 0}

        if max_workers is None:
            default_cap = int(os.environ.get("MAX_WORKERS", 16))
            max_workers = min(cpu_count(), default_cap)

        stats = {
            "total": len(files),
            "processed": 0,
            "removed": 0,
            "errors": 0,
            "cropped": 0,
            "normalized": 0,
            "peak_limited": 0,
            "removed_files": [],
            "details": [] if return_details else None,
        }

        print(f"\n{'=' * 60}")
        print("AUDIO PREPROCESSING")
        print(f"{'=' * 60}")
        print(f"Files to process: {len(files)}")
        print(f"Workers: {max_workers}")
        print(f"Target Duration: {self.duration_handler.target_duration}s")
        print(f"Target Loudness: {self.loudness_normalizer.target_lufs} LUFS")
        print(
            "Peak Ceiling: "
            f"{self.loudness_normalizer.max_true_peak} dBFS sample peak "
            "(legacy config name: max_true_peak)"
        )
        print(f"{'=' * 60}\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_file, str(file_path)): file_path for file_path in files}

            for future in tqdm(as_completed(future_to_file), total=len(files), desc="Preprocessing"):
                result = future.result()

                if return_details:
                    stats["details"].append(result)

                if result["status"] == "success":
                    stats["processed"] += 1
                    if "cropped" in result["actions"]:
                        stats["cropped"] += 1
                    if "normalized" in result["actions"]:
                        stats["normalized"] += 1
                    if "peak_limited" in result["actions"]:
                        stats["peak_limited"] += 1
                elif result["status"] == "removed":
                    stats["removed"] += 1
                    stats["removed_files"].append(result["file"])
                elif result["status"] == "error":
                    stats["errors"] += 1

        print(f"\n{'=' * 60}")
        print("PREPROCESSING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total files:      {stats['total']}")
        print(f"Processed:        {stats['processed']}")
        print(f"  - Cropped:      {stats['cropped']}")
        print(f"  - Normalized:   {stats['normalized']}")
        print(f"  - Peak Limited: {stats['peak_limited']}")
        print(f"Removed (short):  {stats['removed']}")
        print(f"Errors:           {stats['errors']}")
        print(f"{'=' * 60}\n")

        return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio preprocessing: crop, loudness-normalize, and apply a sample-peak safety cap."
    )
    parser.add_argument("--dir", default=str(DEFAULT_AUDIO_DIR), help="Directory containing audio files")
    parser.add_argument("--lufs", type=float, default=-14.0, help="Target LUFS")
    parser.add_argument("--duration", type=float, default=29.0, help="Target duration in seconds")
    parser.add_argument("--peak", type=float, default=-1.0, help="Sample-peak ceiling in dBFS")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--workers", type=int, help="Parallel worker count")
    parser.add_argument("--details", action="store_true", help="Return per-file details")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AudioPreprocessor(
        target_duration=args.duration,
        target_lufs=args.lufs,
        max_true_peak=args.peak,
        sample_rate=args.sample_rate,
    )
    processor.process_directory(args.dir, max_workers=args.workers, return_details=args.details)


if __name__ == "__main__":
    main()
