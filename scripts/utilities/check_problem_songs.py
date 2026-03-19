#!/usr/bin/env python
"""Check whether a shortlist of songs still has metadata, local audio, and features."""

import csv
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")

problem_songs = [
    "-M- - Je suis une cigarette",
    "2 Chainz - Good Drank",
    "21 Savage - Bank Account",
    "21 Savage - No Heart",
    "21 Savage - out for the night (part 2)",
    "21 Savage - Savage Mode",
    "2Pac - Dear Mama",
    "2Pac - All Eyez On Me",
]


def resolve_csv_path() -> Path:
    unified = Path("data") / "songs.csv"
    if unified.exists():
        return unified
    return Path("data") / "songs_data_with_genre.csv"


def find_audio_file(song_basename: str) -> Path | None:
    for extension in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = Path("audio_files") / f"{song_basename}{extension}"
        if candidate.exists():
            return candidate
    return None


print("=" * 60)
print("Checking Problem Songs")
print("=" * 60)

csv_path = resolve_csv_path()
with csv_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

print(f"\nUsing CSV: {csv_path}")
print(f"Total rows in CSV: {len(rows)}")

print("\n" + "=" * 60)
print("Results:")
print("=" * 60)

for song in problem_songs:
    print(f"\nSong: {song}")

    matches = []
    for row in rows:
        filename = str(row.get("filename", "")).strip()
        audio_basename = str(row.get("audio_basename", "")).strip()
        if song == audio_basename or song == Path(filename).stem:
            matches.append(row)

    if matches:
        first_match = matches[0]
        print(f"  [YES] Found in CSV: {first_match.get('filename', 'N/A')}")
        print(f"        Genre: {str(first_match.get('genre', 'N/A'))[:60]}")
        print(f"        has_audio: {first_match.get('has_audio', 'N/A')}")
    else:
        print("  [NO]  NOT in CSV")

    audio_path = find_audio_file(song)
    if audio_path is not None:
        print(f"  [YES] Audio file EXISTS: {audio_path.name}")
    else:
        print("  [NO]  Audio file MISSING")

    feature_path = Path("output") / "features" / f"{song}_mfcc.npy"
    if feature_path.exists():
        print(f"  [YES] Feature file EXISTS: {feature_path.name}")
    else:
        print("  [NO]  Feature file MISSING")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("\nThese warnings usually occur when:")
print("  - Feature files exist but")
print("  - Audio files are missing, or")
print("  - Songs are not represented in the active metadata CSV")
print("\nIf the audio library has drifted, run:")
print("  python scripts/utilities/cleanup_orphaned_features.py --confirm")
print("  python scripts/utilities/cleanup_orphaned_files.py --dry-run")
print("=" * 60)
