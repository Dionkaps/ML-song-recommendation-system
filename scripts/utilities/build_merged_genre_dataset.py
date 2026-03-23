from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.genre_taxonomy import (  # noqa: E402
    EXCLUSION_REASON_COLUMN,
    INCLUDE_IN_MRS_COLUMN,
    PRIMARY_TAGS_COLUMN,
    SECONDARY_TAGS_COLUMN,
    build_assigned_songs_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a taxonomy-aware songs CSV using the merged-genre mapping and "
            "mark non-genre-only rows as excluded from the MRS dataset."
        )
    )
    parser.add_argument("--source-songs-csv", default="data/songs.csv")
    parser.add_argument(
        "--mapping-csv",
        default="data/acoustically_coherent_merged_genres_corrected.csv",
    )
    parser.add_argument("--output-csv", default="data/songs_with_merged_genres.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_assigned_songs_frame(
        source_songs_csv=args.source_songs_csv,
        mapping_csv_path=args.mapping_csv,
    )
    output_path = Path(args.output_csv)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    summary = {
        "output_csv": str(output_path),
        "rows": int(len(frame)),
        "rows_with_audio": int(frame["has_audio"].astype(bool).sum()),
        "rows_included_in_mrs": int(frame[INCLUDE_IN_MRS_COLUMN].astype(bool).sum()),
        "audio_rows_included_in_mrs": int(
            (frame["has_audio"].astype(bool) & frame[INCLUDE_IN_MRS_COLUMN].astype(bool)).sum()
        ),
        "excluded_rows": int((~frame[INCLUDE_IN_MRS_COLUMN].astype(bool)).sum()),
        "excluded_non_genre_only_rows": int(
            frame[EXCLUSION_REASON_COLUMN].fillna("").eq("non_genre_only").sum()
        ),
        "unique_primary_genres": int(
            frame["primary_genre"].fillna("").replace("", pd.NA).dropna().nunique()
        ),
        "unique_primary_tag_sets": int(
            frame[PRIMARY_TAGS_COLUMN].fillna("").replace("", pd.NA).dropna().nunique()
        ),
        "unique_secondary_tags": int(
            len(
                {
                    tag
                    for text in frame[SECONDARY_TAGS_COLUMN].fillna("").astype(str)
                    for tag in [token.strip() for token in text.split(",") if token.strip()]
                }
            )
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
