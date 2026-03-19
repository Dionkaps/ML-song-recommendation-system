"""
Cleanup orphaned local audio files that do not map to unified metadata.

The current workspace keeps processed audio primarily as WAV files, but this
script also understands legacy MP3/FLAC/M4A files. Matching is done by audio
basename so a processed `.wav` file still matches metadata rows that may have
originated from older preview downloads.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

AUDIO_FOLDER = Path("audio_files")
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def resolve_csv_file() -> Path | None:
    candidates = [
        Path("data") / "songs.csv",
        Path("data") / "songs_data_with_genre.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def csv_row_has_audio(row: Dict[str, str]) -> bool:
    has_audio_raw = str(row.get("has_audio", "")).strip()
    has_audio = has_audio_raw.lower()
    if has_audio_raw:
        return has_audio in {"true", "1", "yes"}

    filename = str(row.get("filename", "")).strip()
    return bool(filename)


def row_to_audio_basename(row: Dict[str, str]) -> str | None:
    explicit_basename = str(row.get("audio_basename", "")).strip()
    if explicit_basename:
        return explicit_basename

    filename = str(row.get("filename", "")).strip()
    if filename:
        return Path(filename).stem

    return None


def load_csv_audio_basenames(csv_file: Path) -> Set[str]:
    basenames: Set[str] = set()
    with csv_file.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not csv_row_has_audio(row):
                continue
            basename = row_to_audio_basename(row)
            if basename:
                basenames.add(basename)
    return basenames


def iter_audio_files(audio_folder: Path) -> Iterable[Path]:
    if not audio_folder.exists():
        return []
    return sorted(
        file_path
        for file_path in audio_folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def collect_audio_file_index(audio_folder: Path) -> Dict[str, List[Path]]:
    indexed: Dict[str, List[Path]] = {}
    for path in iter_audio_files(audio_folder):
        indexed.setdefault(path.stem, []).append(path)
    return indexed


def summarize_extensions(paths: Iterable[Path]) -> str:
    counts = Counter(path.suffix.lower() for path in paths)
    if not counts:
        return "none"
    return ", ".join(f"{ext}:{count}" for ext, count in sorted(counts.items()))


def cleanup_orphaned_files(auto_confirm: bool = False, dry_run: bool = False) -> Dict[str, int]:
    csv_file = resolve_csv_file()
    if csv_file is None:
        print("Error: no metadata CSV was found in data/.")
        return {"deleted": 0, "orphaned": 0, "catalog_audio_rows": 0, "audio_files": 0}

    if not AUDIO_FOLDER.exists():
        print(f"Error: {AUDIO_FOLDER}/ folder not found.")
        return {"deleted": 0, "orphaned": 0, "catalog_audio_rows": 0, "audio_files": 0}

    catalog_basenames = load_csv_audio_basenames(csv_file)
    indexed_audio = collect_audio_file_index(AUDIO_FOLDER)
    all_audio_files = [path for paths in indexed_audio.values() for path in paths]
    orphaned_basenames = sorted(set(indexed_audio) - catalog_basenames)
    orphaned_files = [path for basename in orphaned_basenames for path in indexed_audio[basename]]

    print(f"Using metadata CSV: {csv_file}")
    print(f"Audio-backed metadata rows: {len(catalog_basenames)}")
    print(
        f"Detected local audio files: {len(all_audio_files)} "
        f"({summarize_extensions(all_audio_files)})"
    )

    if not orphaned_files:
        print("\nNo orphaned audio files found. Local audio matches metadata basenames.")
        return {
            "deleted": 0,
            "orphaned": 0,
            "catalog_audio_rows": len(catalog_basenames),
            "audio_files": len(all_audio_files),
        }

    print(f"\nFound {len(orphaned_files)} orphaned audio files across {len(orphaned_basenames)} basenames:")
    print("=" * 60)
    for path in orphaned_files[:10]:
        print(f"  - {path.name}")
    if len(orphaned_files) > 10:
        print(f"  ... and {len(orphaned_files) - 10} more")
    print("=" * 60)

    if dry_run:
        print("\nDry run only. No files were deleted.")
        return {
            "deleted": 0,
            "orphaned": len(orphaned_files),
            "catalog_audio_rows": len(catalog_basenames),
            "audio_files": len(all_audio_files),
        }

    response = "yes" if auto_confirm else input(
        f"\nDelete all {len(orphaned_files)} orphaned audio files? (yes/no): "
    ).strip().lower()

    if response != "yes":
        print("\nCleanup cancelled. No files were deleted.")
        return {
            "deleted": 0,
            "orphaned": len(orphaned_files),
            "catalog_audio_rows": len(catalog_basenames),
            "audio_files": len(all_audio_files),
        }

    deleted = 0
    failed = 0
    for path in orphaned_files:
        try:
            path.unlink()
            deleted += 1
        except Exception as exc:
            print(f"Failed to delete {path.name}: {exc}")
            failed += 1

    remaining_audio_files = list(iter_audio_files(AUDIO_FOLDER))
    print("\nCleanup complete.")
    print(f"  - Deleted: {deleted}")
    if failed:
        print(f"  - Failed: {failed}")
    print(
        f"  - Remaining audio files: {len(remaining_audio_files)} "
        f"({summarize_extensions(remaining_audio_files)})"
    )
    print(f"  - Audio-backed metadata rows: {len(catalog_basenames)}")

    return {
        "deleted": deleted,
        "failed": failed,
        "orphaned": len(orphaned_files),
        "catalog_audio_rows": len(catalog_basenames),
        "audio_files": len(remaining_audio_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove local audio files that are not represented by unified metadata."
    )
    parser.add_argument("--auto-confirm", "-y", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cleanup_orphaned_files(auto_confirm=args.auto_confirm, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
