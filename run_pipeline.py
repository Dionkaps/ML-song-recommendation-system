import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from config import feature_vars as fv


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


def run(cmd: List[str]) -> None:
    """Execute a command and stop the pipeline if it fails."""
    print("\n$", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def iter_audio_files(audio_dir: Path) -> Iterable[Path]:
    for extension in SUPPORTED_AUDIO_EXTENSIONS:
        yield from audio_dir.glob(f"*{extension}")


def count_audio_backed_rows(csv_path: Path) -> int:
    """Count rows that currently claim a local audio file."""
    if not csv_path.exists():
        return 0

    count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            has_audio_raw = str(row.get("has_audio", "")).strip()
            has_audio = has_audio_raw.lower()
            filename = str(row.get("filename", "")).strip()

            if has_audio_raw:
                if has_audio in {"true", "1", "yes"}:
                    count += 1
            elif filename:
                count += 1

    return count


def resolve_metadata_csv(project_root: Path) -> Path | None:
    candidates = [
        project_root / "data" / "songs.csv",
        project_root / "data" / "songs_data_with_genre.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_download_step(project_root: Path, py: str) -> None:
    print("\n" + "=" * 60)
    print("DOWNLOAD STEP: Songs with Genre Information")
    print("=" * 60)

    million_song_csv = project_root / "data" / "millionsong_dataset.csv"
    tar_path = project_root / "data" / "millionsongsubset.tar.gz"
    if not million_song_csv.exists():
        if tar_path.exists():
            print(f"Found Million Song Dataset archive: {tar_path}")
            print("Extracting dataset and generating CSV...")
            run([py, "src/data_collection/extract_millionsong_dataset.py"])
        else:
            print("ERROR: Million Song Dataset CSV not found.")
            print(f"Expected: {million_song_csv}")
            print("OR")
            print(f"Archive: {tar_path}")
            print("Please ensure the dataset exists before downloading.")
            sys.exit(1)

    print(f"Found Million Song Dataset: {million_song_csv}")
    print("\nDownloading songs from Deezer...")
    print("  - Only songs with genres will be downloaded")
    print("  - Failed downloads will be cleaned up automatically")
    print("  - Results saved to unified metadata in data/songs.csv")
    print("  - Audio files saved to audio_files/")

    run([py, "src/data_collection/deezer-song.py"])

    print("\n" + "=" * 60)
    print("Validating download results...")
    print("=" * 60)

    audio_dir = project_root / "audio_files"
    active_csv = resolve_metadata_csv(project_root)
    audio_files = sorted(iter_audio_files(audio_dir)) if audio_dir.exists() else []
    audio_count = len(audio_files)

    if active_csv is None:
        print("WARNING: No songs CSV was found after download.")
    else:
        csv_count = count_audio_backed_rows(active_csv)
        print(f"Audio-backed metadata rows: {csv_count} in {active_csv.name}")

        if audio_dir.exists():
            print(f"Detected audio files: {audio_count}")
            if csv_count > 0 and audio_count != csv_count:
                diff = abs(audio_count - csv_count)
                print(f"\nWARNING: Metadata/audio count mismatch detected ({diff} files).")
                print("Running orphaned-audio cleanup against unified metadata...")
                run([py, "scripts/utilities/cleanup_orphaned_files.py", "--auto-confirm"])
            else:
                print("Metadata and on-disk audio counts are aligned.")
        else:
            print("WARNING: audio_files/ directory not found.")

    print("=" * 60)
    print("Download step complete!")
    print("=" * 60 + "\n")


def run_preprocessing_step(py: str) -> None:
    print("\n" + "=" * 60)
    print("AUDIO PREPROCESSING")
    print("=" * 60)
    print("Using the supported preprocessing runner and baseline settings:")
    print(f"  - duration: {fv.baseline_target_duration_seconds}s")
    print(f"  - loudness target: {fv.baseline_target_lufs} LUFS")
    print(
        "  - peak ceiling: "
        f"{fv.baseline_max_true_peak_dbtp} dBFS sample peak "
        "(legacy config name keeps max_true_peak)"
    )
    print(f"  - sample rate: {fv.baseline_sample_rate} Hz")
    print(f"  - mono: {fv.baseline_force_mono}")
    print(f"  - output subtype: {fv.baseline_output_subtype}")

    run(
        [
            py,
            "scripts/run_audio_preprocessing.py",
            "--audio-dir",
            "audio_files",
            "--target-duration",
            str(fv.baseline_target_duration_seconds),
            "--target-lufs",
            str(fv.baseline_target_lufs),
            "--max-peak-db",
            str(fv.baseline_max_true_peak_dbtp),
        ]
    )

    print("=" * 60)
    print("Audio preprocessing complete!")
    print("=" * 60 + "\n")


def run_extract_step(py: str) -> None:
    print("\nExtracting handcrafted audio features...")
    print("Active baseline note: clustering uses the spectral_plus_beat subset.")
    run([py, "src/features/extract_features.py"])
    print("\nFor deep embedding extraction (EnCodecMAE, MERT, MusiCNN), run:")
    print("  python run_extraction.py")


def run_plot_step(py: str) -> None:
    print("\nGenerating feature plots from extracted arrays...")
    run(
        [
            py,
            "scripts/visualization/ploting.py",
            "--features_dir",
            "output/features",
            "--plots_dir",
            "output/plots",
        ]
    )


def run_cluster_step(py: str, clustering_method: str) -> None:
    if clustering_method == "hdbscan":
        print("\nUsing HDBSCAN clustering method")
        run([py, "src/clustering/hdbscan.py"])
    elif clustering_method == "gmm":
        print("\nUsing GMM clustering method")
        run([py, "src/clustering/gmm.py"])
    elif clustering_method == "vade":
        print("\nUsing VaDE clustering method")
        run([py, "src/clustering/vade.py"])
    else:
        print("\nUsing K-Means clustering method")
        run([py, "src/clustering/kmeans.py"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the supported audio analysis pipeline end-to-end."
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=["download", "preprocess", "extract", "process", "plot", "cluster"],
        default=[],
        help="Pipeline steps to skip.",
    )
    parser.add_argument(
        "--clustering-method",
        choices=["kmeans", "hdbscan", "gmm", "vade"],
        default=fv.default_clustering_method,
        help="Clustering method to use for the final step.",
    )
    args = parser.parse_args()

    if "process" in args.skip:
        print(
            "Note: the historical standalone 'process' stage no longer exists in the "
            "supported baseline. The skip flag is kept only for compatibility."
        )

    py = sys.executable
    project_root = Path(__file__).resolve().parent

    if "download" not in args.skip:
        run_download_step(project_root, py)

    if "preprocess" not in args.skip:
        run_preprocessing_step(py)

    if "extract" not in args.skip:
        run_extract_step(py)

    if "plot" not in args.skip:
        run_plot_step(py)

    if "cluster" not in args.skip:
        run_cluster_step(py, args.clustering_method)


if __name__ == "__main__":
    main()
