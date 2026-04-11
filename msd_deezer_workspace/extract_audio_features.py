"""
Audio feature extraction for clustering-ready MSD/Deezer previews.

Outputs:
1. One compressed `.npz` file per song with raw feature arrays
2. One `feature_vectors.csv` file with fixed-length summary statistics
   suitable for downstream clustering experiments
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np
from tqdm import tqdm


WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO_DIR = WORKSPACE_DIR / "audio"
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "features"
DEFAULT_RAW_DIRNAME = "raw"
DEFAULT_SUMMARY_FILENAME = "feature_vectors.csv"
DEFAULT_EXTRACTION_SUMMARY = "extraction_summary.json"
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
logging.getLogger("audioread").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.ERROR)


class _SuppressStderr:
    """Context manager to suppress stderr from lower-level audio loaders."""

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


class AudioFeatureExtractor:
    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 20,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if n_mfcc <= 0:
            raise ValueError("n_mfcc must be positive")

        self.output_dir = Path(output_dir)
        self.raw_output_dir = self.output_dir / DEFAULT_RAW_DIRNAME
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def summary_fieldnames(self) -> list[str]:
        fields = [
            "file",
            "audio_path",
            "raw_feature_path",
            "msd_track_id",
            "deezer_track_id",
            "sample_rate",
            "duration_sec",
            "frames",
        ]

        for prefix in ("mfcc", "delta_mfcc", "delta2_mfcc"):
            for index in range(1, self.n_mfcc + 1):
                fields.append(f"{prefix}_{index:02d}_mean")
                fields.append(f"{prefix}_{index:02d}_std")

        for index in range(1, 13):
            fields.append(f"chroma_{index:02d}_mean")
            fields.append(f"chroma_{index:02d}_std")

        for prefix in (
            "spectral_centroid",
            "spectral_rolloff",
            "spectral_flux",
            "spectral_flatness",
            "spectral_bandwidth",
            "zero_crossing_rate",
            "beat_strength",
        ):
            fields.append(f"{prefix}_mean")
            fields.append(f"{prefix}_std")

        # Tzanetakis rhythm features (track-level scalars)
        fields.append("tempo_bpm")
        fields.append("tempogram_peak1_bpm")
        fields.append("tempogram_peak1_amp")
        fields.append("tempogram_peak2_bpm")
        fields.append("tempogram_peak2_amp")
        fields.append("tempogram_peak_ratio")
        fields.append("tempogram_sum")

        # Tzanetakis low-energy rate
        fields.append("low_energy_rate")

        return fields

    def _extract_ids_from_filename(self, filename: str) -> dict[str, str]:
        msd_match = re.search(r"\[(TR[A-Z0-9]+)\]", filename)
        deezer_match = re.search(r"\[deezer-(\d+)\]", filename, flags=re.IGNORECASE)
        return {
            "msd_track_id": msd_match.group(1) if msd_match else "",
            "deezer_track_id": deezer_match.group(1) if deezer_match else "",
        }

    def _compute_feature_payload(self, y: np.ndarray, sr: int) -> dict[str, np.ndarray | int | float]:
        spectrum = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        safe_norm = np.linalg.norm(spectrum, axis=0, keepdims=True)
        normalized_spectrum = spectrum / np.maximum(safe_norm, 1e-10)
        spectrum_diff = np.diff(normalized_spectrum, axis=1, prepend=normalized_spectrum[:, :1])
        spectral_flux = np.sqrt(np.sum(np.square(spectrum_diff), axis=0, dtype=np.float64)).astype(np.float32)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        ).astype(np.float32)
        delta_mfcc = librosa.feature.delta(mfcc, order=1).astype(np.float32)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2).astype(np.float32)
        spectral_centroid = librosa.feature.spectral_centroid(
            S=spectrum,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )[0].astype(np.float32)
        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=spectrum,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )[0].astype(np.float32)
        spectral_flatness = librosa.feature.spectral_flatness(S=spectrum)[0].astype(np.float32)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )[0].astype(np.float32)
        chroma = librosa.feature.chroma_stft(
            S=spectrum,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        ).astype(np.float32)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            S=spectrum,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )[0].astype(np.float32)
        beat_strength = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
        ).astype(np.float32)

        # --- Tzanetakis rhythm features ---
        tempo_arr = librosa.feature.tempo(
            onset_envelope=beat_strength,
            sr=sr,
            hop_length=self.hop_length,
        )
        tempo_bpm = float(tempo_arr[0]) if np.ndim(tempo_arr) >= 1 else float(tempo_arr)

        # Tempogram-based beat histogram (autocorrelation method)
        tempogram = librosa.feature.tempogram(
            onset_envelope=beat_strength,
            sr=sr,
            hop_length=self.hop_length,
        )
        # Global beat histogram: average over time axis
        beat_histogram = tempogram.mean(axis=1).astype(np.float64)
        # BPM axis for the tempogram bins
        tempo_frequencies = librosa.tempo_frequencies(
            tempogram.shape[0],
            sr=sr,
            hop_length=self.hop_length,
        )

        # Restrict to musically plausible BPM range [30, 300] (Tzanetakis convention)
        bpm_min, bpm_max = 30.0, 300.0
        plausible_mask = (tempo_frequencies >= bpm_min) & (tempo_frequencies <= bpm_max)
        bh = beat_histogram.copy()
        bh[~plausible_mask] = 0.0
        # Also zero out bin 0 (DC / inf BPM)
        bh[0] = 0.0

        peak1_idx = int(np.argmax(bh))
        peak1_amp = float(bh[peak1_idx])
        peak1_bpm = float(tempo_frequencies[peak1_idx])

        bh_copy = bh.copy()
        # Suppress a neighbourhood around peak1 to find a distinct second peak
        suppress_start = max(0, peak1_idx - 5)
        suppress_end = min(len(bh_copy), peak1_idx + 6)
        bh_copy[suppress_start:suppress_end] = 0.0
        peak2_idx = int(np.argmax(bh_copy))
        peak2_amp = float(bh_copy[peak2_idx])
        peak2_bpm = float(tempo_frequencies[peak2_idx])

        peak_ratio = float(peak2_amp / peak1_amp) if peak1_amp > 1e-10 else 0.0
        # Sum only within plausible range for consistent energy measure
        tempogram_sum = float(beat_histogram[plausible_mask].sum())

        # --- Tzanetakis low-energy rate ---
        rms = librosa.feature.rms(
            y=y,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )[0].astype(np.float32)
        low_energy_rate = float(np.mean(rms < rms.mean()))

        return {
            "sample_rate": int(sr),
            "duration_sec": float(librosa.get_duration(y=y, sr=sr)),
            "mfcc": mfcc,
            "delta_mfcc": delta_mfcc,
            "delta2_mfcc": delta2_mfcc,
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff": spectral_rolloff,
            "spectral_flux": spectral_flux,
            "spectral_flatness": spectral_flatness,
            "spectral_bandwidth": spectral_bandwidth,
            "zero_crossing_rate": zero_crossing_rate,
            "chroma": chroma,
            "beat_strength": beat_strength,
            "rms": rms,
            "tempo_bpm": tempo_bpm,
            "tempogram": tempogram.astype(np.float32),
            "tempogram_peak1_bpm": peak1_bpm,
            "tempogram_peak1_amp": peak1_amp,
            "tempogram_peak2_bpm": peak2_bpm,
            "tempogram_peak2_amp": peak2_amp,
            "tempogram_peak_ratio": peak_ratio,
            "tempogram_sum": tempogram_sum,
            "low_energy_rate": low_energy_rate,
        }

    def _build_summary_row(
        self,
        file_path: Path,
        raw_feature_path: Path,
        payload: dict[str, np.ndarray | int | float],
    ) -> dict[str, Any]:
        ids = self._extract_ids_from_filename(file_path.name)
        row: dict[str, Any] = {
            "file": file_path.name,
            "audio_path": str(file_path.resolve()),
            "raw_feature_path": str(raw_feature_path.resolve()),
            "msd_track_id": ids["msd_track_id"],
            "deezer_track_id": ids["deezer_track_id"],
            "sample_rate": int(payload["sample_rate"]),
            "duration_sec": round(float(payload["duration_sec"]), 6),
            "frames": int(np.asarray(payload["mfcc"]).shape[1]),
        }

        for prefix in ("mfcc", "delta_mfcc", "delta2_mfcc"):
            matrix = np.asarray(payload[prefix], dtype=np.float32)
            means = matrix.mean(axis=1)
            stds = matrix.std(axis=1)
            for index, (mean_value, std_value) in enumerate(zip(means, stds), start=1):
                row[f"{prefix}_{index:02d}_mean"] = round(float(mean_value), 8)
                row[f"{prefix}_{index:02d}_std"] = round(float(std_value), 8)

        chroma = np.asarray(payload["chroma"], dtype=np.float32)
        chroma_means = chroma.mean(axis=1)
        chroma_stds = chroma.std(axis=1)
        for index, (mean_value, std_value) in enumerate(zip(chroma_means, chroma_stds), start=1):
            row[f"chroma_{index:02d}_mean"] = round(float(mean_value), 8)
            row[f"chroma_{index:02d}_std"] = round(float(std_value), 8)

        for prefix in (
            "spectral_centroid",
            "spectral_rolloff",
            "spectral_flux",
            "spectral_flatness",
            "spectral_bandwidth",
            "zero_crossing_rate",
            "beat_strength",
        ):
            values = np.asarray(payload[prefix], dtype=np.float32)
            row[f"{prefix}_mean"] = round(float(values.mean()), 8)
            row[f"{prefix}_std"] = round(float(values.std()), 8)

        # Tzanetakis rhythm scalars
        row["tempo_bpm"] = round(float(payload["tempo_bpm"]), 4)
        row["tempogram_peak1_bpm"] = round(float(payload["tempogram_peak1_bpm"]), 4)
        row["tempogram_peak1_amp"] = round(float(payload["tempogram_peak1_amp"]), 8)
        row["tempogram_peak2_bpm"] = round(float(payload["tempogram_peak2_bpm"]), 4)
        row["tempogram_peak2_amp"] = round(float(payload["tempogram_peak2_amp"]), 8)
        row["tempogram_peak_ratio"] = round(float(payload["tempogram_peak_ratio"]), 8)
        row["tempogram_sum"] = round(float(payload["tempogram_sum"]), 8)

        # Tzanetakis low-energy rate
        row["low_energy_rate"] = round(float(payload["low_energy_rate"]), 8)

        return row

    def process_file(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        result: Dict[str, Any] = {
            "file": path.name,
            "status": "success",
            "error": None,
            "summary_row": None,
        }

        try:
            self.raw_output_dir.mkdir(parents=True, exist_ok=True)
            with _SuppressStderr():
                y, sr = librosa.load(path, sr=self.sample_rate, mono=True)

            payload = self._compute_feature_payload(y, sr)
            raw_feature_path = self.raw_output_dir / f"{path.stem}.npz"
            np.savez_compressed(
                raw_feature_path,
                sample_rate=np.int32(payload["sample_rate"]),
                duration_sec=np.float32(payload["duration_sec"]),
                mfcc=np.asarray(payload["mfcc"], dtype=np.float32),
                delta_mfcc=np.asarray(payload["delta_mfcc"], dtype=np.float32),
                delta2_mfcc=np.asarray(payload["delta2_mfcc"], dtype=np.float32),
                spectral_centroid=np.asarray(payload["spectral_centroid"], dtype=np.float32),
                spectral_rolloff=np.asarray(payload["spectral_rolloff"], dtype=np.float32),
                spectral_flux=np.asarray(payload["spectral_flux"], dtype=np.float32),
                spectral_flatness=np.asarray(payload["spectral_flatness"], dtype=np.float32),
                spectral_bandwidth=np.asarray(payload["spectral_bandwidth"], dtype=np.float32),
                zero_crossing_rate=np.asarray(payload["zero_crossing_rate"], dtype=np.float32),
                chroma=np.asarray(payload["chroma"], dtype=np.float32),
                beat_strength=np.asarray(payload["beat_strength"], dtype=np.float32),
                rms=np.asarray(payload["rms"], dtype=np.float32),
                tempogram=np.asarray(payload["tempogram"], dtype=np.float32),
                tempo_bpm=np.float32(payload["tempo_bpm"]),
                tempogram_peak1_bpm=np.float32(payload["tempogram_peak1_bpm"]),
                tempogram_peak1_amp=np.float32(payload["tempogram_peak1_amp"]),
                tempogram_peak2_bpm=np.float32(payload["tempogram_peak2_bpm"]),
                tempogram_peak2_amp=np.float32(payload["tempogram_peak2_amp"]),
                tempogram_peak_ratio=np.float32(payload["tempogram_peak_ratio"]),
                tempogram_sum=np.float32(payload["tempogram_sum"]),
                low_energy_rate=np.float32(payload["low_energy_rate"]),
            )
            result["summary_row"] = self._build_summary_row(path, raw_feature_path, payload)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            logger.error("Error extracting features from %s: %s", path.name, exc)

        return result

    def process_directory(self, directory: str | Path, max_workers: int | None = None, return_details: bool = False) -> Dict[str, Any]:
        input_dir = Path(directory)
        if not input_dir.exists():
            logger.error("Directory not found: %s", directory)
            return {"error": "Directory not found"}

        files = []
        for extension in SUPPORTED_AUDIO_EXTENSIONS:
            files.extend(input_dir.glob(f"*{extension}"))
        files = sorted(files)

        if not files:
            logger.warning("No audio files found in %s", input_dir)
            return {"total": 0}

        if max_workers is None:
            max_workers = min(cpu_count(), 8)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)

        stats: Dict[str, Any] = {
            "total": len(files),
            "processed": 0,
            "errors": 0,
            "details": [] if return_details else None,
        }
        summary_rows: list[dict[str, Any]] = []

        print(f"\n{'=' * 60}")
        print("AUDIO FEATURE EXTRACTION")
        print(f"{'=' * 60}")
        print(f"Files to process: {len(files)}")
        print(f"Workers: {max_workers}")
        print(f"Output directory: {self.output_dir}")
        print(f"Sample rate: {self.sample_rate}")
        print(f"MFCC count: {self.n_mfcc}")
        print(f"{'=' * 60}\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_file, str(file_path)): file_path for file_path in files}

            for future in tqdm(as_completed(future_to_file), total=len(files), desc="Extracting features"):
                result = future.result()
                if return_details:
                    stats["details"].append(result)

                if result["status"] == "success":
                    stats["processed"] += 1
                    if result["summary_row"] is not None:
                        summary_rows.append(result["summary_row"])
                else:
                    stats["errors"] += 1

        summary_rows.sort(key=lambda row: row["file"])
        summary_csv_path = self.output_dir / DEFAULT_SUMMARY_FILENAME
        with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.summary_fieldnames())
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        summary_payload = {
            "total": stats["total"],
            "processed": stats["processed"],
            "errors": stats["errors"],
            "output_dir": str(self.output_dir.resolve()),
            "raw_dir": str(self.raw_output_dir.resolve()),
            "summary_csv": str(summary_csv_path.resolve()),
        }
        (self.output_dir / DEFAULT_EXTRACTION_SUMMARY).write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )

        print(f"\n{'=' * 60}")
        print("FEATURE EXTRACTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total files:   {stats['total']}")
        print(f"Processed:     {stats['processed']}")
        print(f"Errors:        {stats['errors']}")
        print(f"Feature CSV:   {summary_csv_path}")
        print(f"Raw features:  {self.raw_output_dir}")
        print(f"{'=' * 60}\n")

        return summary_payload | {"details": stats["details"] if return_details else None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clustering-ready audio features and raw feature arrays from the final audio directory."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_AUDIO_DIR), help="Directory containing final audio files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where extracted features will be stored")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate for feature extraction")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length")
    parser.add_argument("--n-mfcc", type=int, default=20, help="Number of MFCC coefficients")
    parser.add_argument("--workers", type=int, help="Parallel worker count")
    parser.add_argument("--details", action="store_true", help="Include per-file results in the returned stats")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractor = AudioFeatureExtractor(
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mfcc=args.n_mfcc,
    )
    extractor.process_directory(args.input_dir, max_workers=args.workers, return_details=args.details)


if __name__ == "__main__":
    main()
