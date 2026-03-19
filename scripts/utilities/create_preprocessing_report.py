"""
Create a before/after CSV report for the supported audio preprocessing pipeline.

The report uses integrated loudness plus sample-peak measurements. It does not
claim oversampled true-peak analysis.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable

import librosa
import numpy as np
import pyloudnorm as pyln
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv
from src.audio_preprocessing import AudioPreprocessor


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


def iter_audio_files(audio_dir: Path) -> Iterable[Path]:
    files = []
    for extension in SUPPORTED_AUDIO_EXTENSIONS:
        files.extend(audio_dir.glob(f"*{extension}"))
    return sorted(files)


def measure_audio_stats(file_path: Path, sample_rate: int = fv.baseline_sample_rate) -> Dict:
    """Measure audio characteristics without modifying the file."""
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        duration = len(audio) / sr
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(audio)
        peak_linear = np.abs(audio).max() if len(audio) else 0.0
        sample_peak_db = 20 * np.log10(peak_linear) if peak_linear > 0 else -np.inf
        return {
            "filename": file_path.name,
            "basename": file_path.stem,
            "duration": round(duration, 3),
            "lufs": round(lufs, 2) if np.isfinite(lufs) else None,
            "sample_peak_db": round(sample_peak_db, 2) if np.isfinite(sample_peak_db) else None,
            "success": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "filename": file_path.name,
            "basename": file_path.stem,
            "duration": None,
            "lufs": None,
            "sample_peak_db": None,
            "success": False,
            "error": str(exc),
        }


def create_preprocessing_report(audio_dir: Path, output_csv: Path) -> None:
    print("=" * 80)
    print("AUDIO PREPROCESSING REPORT GENERATOR")
    print("=" * 80)
    print(f"\nAudio directory: {audio_dir}")
    print(f"Output CSV: {output_csv}")
    print()

    audio_files = list(iter_audio_files(audio_dir))
    if not audio_files:
        print("No audio files found.")
        return

    print(f"Found {len(audio_files)} audio files")
    print()

    print("Step 1/3: Measuring original audio characteristics...")
    before_stats = {}
    for file_path in tqdm(audio_files, desc="Analyzing before"):
        before_stats[file_path.stem] = measure_audio_stats(file_path)
    print(f"Measured {len(before_stats)} files")
    print()

    print("Step 2/3: Running audio preprocessing...")
    print(
        "Target: "
        f"{fv.baseline_target_duration_seconds}s, "
        f"{fv.baseline_target_lufs} LUFS, "
        f"{fv.baseline_max_true_peak_dbtp} dBFS sample-peak ceiling"
    )
    print()

    preprocessor = AudioPreprocessor(
        target_duration=fv.baseline_target_duration_seconds,
        target_lufs=fv.baseline_target_lufs,
        max_true_peak=fv.baseline_max_true_peak_dbtp,
        sample_rate=fv.baseline_sample_rate,
    )
    processing_results = preprocessor.process_directory(str(audio_dir), return_details=True)
    detail_rows = {
        Path(detail.get("file", "")).stem: detail
        for detail in processing_results.get("details") or []
        if detail.get("file")
    }
    print()
    print(f"Processed: {processing_results.get('processed', 0)} files")
    print(f"Errors: {processing_results.get('errors', 0)} files")
    print(f"Removed (too short): {processing_results.get('removed', 0)} files")
    print()

    print("Step 3/3: Measuring processed audio characteristics...")
    after_stats = {}
    for file_path in tqdm(iter_audio_files(audio_dir), desc="Analyzing after"):
        after_stats[file_path.stem] = measure_audio_stats(file_path)
    print(f"Measured {len(after_stats)} files")
    print()

    print("Creating CSV report...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "basename",
            "before_filename",
            "after_filename",
            "status",
            "before_duration_s",
            "after_duration_s",
            "duration_change_s",
            "before_lufs",
            "after_lufs",
            "lufs_change",
            "before_sample_peak_db",
            "after_sample_peak_db",
            "peak_change_db",
            "gain_applied_db",
            "was_cropped",
            "was_removed",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        all_basenames = sorted(set(before_stats) | set(after_stats))
        removed_files = {
            Path(filename).stem
            for filename in processing_results.get("removed_files", [])
        }

        for basename in all_basenames:
            before = before_stats.get(basename, {})
            after = after_stats.get(basename, {})
            detail = detail_rows.get(basename, {})

            if basename in removed_files:
                status = "REMOVED"
            elif before.get("success") and after.get("success"):
                status = "SUCCESS"
            elif not before.get("success"):
                status = "ERROR_BEFORE"
            else:
                status = "ERROR_AFTER"

            before_duration = before.get("duration")
            after_duration = after.get("duration")
            before_lufs = before.get("lufs")
            after_lufs = after.get("lufs")
            before_peak = before.get("sample_peak_db")
            after_peak = after.get("sample_peak_db")

            duration_change = (
                round(after_duration - before_duration, 3)
                if before_duration is not None and after_duration is not None
                else None
            )
            lufs_change = (
                round(after_lufs - before_lufs, 2)
                if before_lufs is not None and after_lufs is not None
                else None
            )
            peak_change = (
                round(after_peak - before_peak, 2)
                if before_peak is not None and after_peak is not None
                else None
            )

            writer.writerow(
                {
                    "basename": basename,
                    "before_filename": before.get("filename"),
                    "after_filename": after.get("filename"),
                    "status": status,
                    "before_duration_s": before_duration,
                    "after_duration_s": after_duration,
                    "duration_change_s": duration_change,
                    "before_lufs": before_lufs,
                    "after_lufs": after_lufs,
                    "lufs_change": lufs_change,
                    "before_sample_peak_db": before_peak,
                    "after_sample_peak_db": after_peak,
                    "peak_change_db": peak_change,
                    "gain_applied_db": detail.get("gain_applied_db"),
                    "was_cropped": "cropped" in (detail.get("actions") or []),
                    "was_removed": basename in removed_files,
                    "error": before.get("error") or after.get("error") or detail.get("error"),
                }
            )

    print(f"CSV report saved to: {output_csv}")
    print()

    successful = [
        basename
        for basename in sorted(set(before_stats) & set(after_stats))
        if before_stats[basename].get("success") and after_stats[basename].get("success")
    ]
    if successful:
        lufs_after = [
            after_stats[basename]["lufs"]
            for basename in successful
            if after_stats[basename].get("lufs") is not None
        ]
        durations_after = [
            after_stats[basename]["duration"]
            for basename in successful
            if after_stats[basename].get("duration") is not None
        ]
        peaks_after = [
            after_stats[basename]["sample_peak_db"]
            for basename in successful
            if after_stats[basename].get("sample_peak_db") is not None
        ]

        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Successful files: {len(successful)}")
        if durations_after:
            print(
                "Duration after: "
                f"{np.mean(durations_after):.2f}s +/- {np.std(durations_after):.2f}s"
            )
        if lufs_after:
            at_target = sum(abs(value - fv.baseline_target_lufs) <= 0.5 for value in lufs_after)
            print(
                "Loudness after: "
                f"{np.mean(lufs_after):.2f} +/- {np.std(lufs_after):.2f} LUFS"
            )
            print(f"Within +/-0.5 LUFS of target: {at_target}/{len(lufs_after)}")
        if peaks_after:
            print(
                "Sample peak after: "
                f"{np.mean(peaks_after):.2f} dBFS mean, {np.max(peaks_after):.2f} dBFS max"
            )
    print("=" * 80)
    print("Report generation complete.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preprocessing and generate a before/after CSV report."
    )
    parser.add_argument("--audio-dir", default="audio_files")
    parser.add_argument(
        "--output-csv",
        default=str(Path("output") / "preprocessing_report.csv"),
    )
    args = parser.parse_args()

    create_preprocessing_report(Path(args.audio_dir), Path(args.output_csv))


if __name__ == "__main__":
    main()
