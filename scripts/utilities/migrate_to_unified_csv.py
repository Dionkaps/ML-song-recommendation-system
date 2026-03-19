"""Rebuild the canonical unified songs.csv from local metadata sources."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.song_metadata import build_unified_songs_dataframe


def migrate_to_unified_csv(
    project_root: str = None,
    data_dir: str = "data",
    audio_dir: str = "audio_files",
    results_dir: str = "output/features",
) -> Path:
    """Build and persist the unified songs.csv plus a summary JSON."""

    resolved_root = Path(project_root) if project_root else PROJECT_ROOT
    unified, summary = build_unified_songs_dataframe(
        project_root=str(resolved_root),
        data_dir=data_dir,
        audio_dir=audio_dir,
        results_dir=results_dir,
    )

    data_root = resolved_root / data_dir
    data_root.mkdir(parents=True, exist_ok=True)

    output_csv = data_root / "songs.csv"
    output_summary = data_root / "songs_schema_summary.json"

    unified.to_csv(output_csv, index=False)
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Unified metadata rebuilt successfully.")
    print(f"Rows written: {len(unified)}")
    print(f"Audio-backed rows: {int(unified['has_audio'].sum())}")
    print(f"Rows with MSD track id: {int(unified['msd_track_id'].astype(str).str.strip().ne('').sum())}")
    print(
        "Audio-backed rows with numeric MSD features: "
        f"{summary['audio_rows_with_numeric_msd_features']}"
    )
    print(f"Unified CSV: {output_csv}")
    print(f"Schema summary: {output_summary}")
    return output_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild data/songs.csv from the local legacy metadata sources."
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--audio-dir", default="audio_files")
    parser.add_argument("--results-dir", default="output/features")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    migrate_to_unified_csv(
        project_root=args.project_root,
        data_dir=args.data_dir,
        audio_dir=args.audio_dir,
        results_dir=args.results_dir,
    )
