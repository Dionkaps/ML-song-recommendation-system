from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.utils.song_metadata import load_unified_songs


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SOURCE_SONGS_CSV = PROJECT_ROOT / "data" / "songs.csv"
DEFAULT_MERGED_GENRE_MAP_CSV = (
    PROJECT_ROOT / "data" / "acoustically_coherent_merged_genres_corrected.csv"
)
DEFAULT_ASSIGNED_SONGS_CSV = PROJECT_ROOT / "data" / "songs_with_merged_genres.csv"

NON_GENRE_PREFIX = "non_genre_"
PRIMARY_TAGS_COLUMN = "mapped_primary_genres"
SECONDARY_TAGS_COLUMN = "mapped_secondary_tags"
ALL_TAGS_COLUMN = "mapped_all_tags"
INCLUDE_IN_MRS_COLUMN = "include_in_mrs"
EXCLUSION_REASON_COLUMN = "mrs_exclusion_reason"
UNMAPPED_SOURCE_TAGS_COLUMN = "unmapped_source_genres"


def split_tag_list(value: Any, delimiter: str = ",") -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(delimiter) if token.strip()]


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        ordered.append(token)
        seen.add(token)
    return ordered


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def is_non_genre_tag(tag: Any) -> bool:
    return str(tag or "").strip().startswith(NON_GENRE_PREFIX)


def load_merged_genre_lookup(
    mapping_csv_path: Optional[str] = None,
) -> Dict[str, str]:
    path = Path(mapping_csv_path) if mapping_csv_path else DEFAULT_MERGED_GENRE_MAP_CSV
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Merged genre mapping CSV not found: {path}")

    lookup: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            merged_genre = str(row.get("merged_genre", "") or "").strip()
            source_tags = split_tag_list(row.get("consists_of", ""), delimiter="|")
            for source_tag in source_tags:
                existing = lookup.get(source_tag)
                if existing is not None and existing != merged_genre:
                    raise ValueError(
                        "Conflicting merged-genre mapping for "
                        f"'{source_tag}': '{existing}' vs '{merged_genre}'"
                    )
                lookup[source_tag] = merged_genre
    return lookup


def assign_merged_genres(
    raw_genre_text: Any,
    lookup: Dict[str, str],
) -> Dict[str, Any]:
    source_tags = split_tag_list(raw_genre_text, delimiter=",")
    mapped_tags = [lookup.get(tag, tag) for tag in source_tags]
    mapped_tags = dedupe_keep_order(mapped_tags)
    primary_tags = [tag for tag in mapped_tags if not is_non_genre_tag(tag)]
    secondary_tags = [tag for tag in mapped_tags if is_non_genre_tag(tag)]
    unmapped_source_tags = [tag for tag in source_tags if tag not in lookup]
    all_tags = dedupe_keep_order([*primary_tags, *secondary_tags])
    return {
        "source_tags": source_tags,
        "primary_tags": primary_tags,
        "secondary_tags": secondary_tags,
        "all_tags": all_tags,
        "primary_genre": primary_tags[0] if primary_tags else "",
        "include_in_mrs": bool(primary_tags),
        "exclusion_reason": "" if primary_tags else "non_genre_only",
        "unmapped_source_tags": unmapped_source_tags,
    }


def build_assigned_songs_frame(
    source_songs_csv: Optional[str] = None,
    mapping_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    source_path = Path(source_songs_csv) if source_songs_csv else DEFAULT_SOURCE_SONGS_CSV
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"Source songs CSV not found: {source_path}")

    lookup = load_merged_genre_lookup(mapping_csv_path)
    frame = load_unified_songs(str(source_path)).copy()

    original_genres = frame["genre"].astype(str).tolist()
    original_primary_genres = frame["primary_genre"].astype(str).tolist()
    original_genre_counts = frame["genre_count"].astype(int).tolist()
    assignments = [assign_merged_genres(value, lookup) for value in original_genres]

    frame["original_genre"] = original_genres
    frame["original_primary_genre"] = original_primary_genres
    frame["original_genre_count"] = original_genre_counts
    frame["genre"] = [", ".join(row["all_tags"]) for row in assignments]
    frame["primary_genre"] = [row["primary_genre"] for row in assignments]
    frame["genre_count"] = [len(row["all_tags"]) for row in assignments]
    frame[PRIMARY_TAGS_COLUMN] = [", ".join(row["primary_tags"]) for row in assignments]
    frame[SECONDARY_TAGS_COLUMN] = [
        ", ".join(row["secondary_tags"]) for row in assignments
    ]
    frame[ALL_TAGS_COLUMN] = [", ".join(row["all_tags"]) for row in assignments]
    frame["mapped_primary_genre_count"] = [
        len(row["primary_tags"]) for row in assignments
    ]
    frame["mapped_secondary_tag_count"] = [
        len(row["secondary_tags"]) for row in assignments
    ]
    frame[INCLUDE_IN_MRS_COLUMN] = [row["include_in_mrs"] for row in assignments]
    frame[EXCLUSION_REASON_COLUMN] = [row["exclusion_reason"] for row in assignments]
    frame[UNMAPPED_SOURCE_TAGS_COLUMN] = [
        ", ".join(row["unmapped_source_tags"]) for row in assignments
    ]
    frame["merged_genre_map_file"] = (
        Path(mapping_csv_path).name if mapping_csv_path else DEFAULT_MERGED_GENRE_MAP_CSV.name
    )
    return frame


def write_assigned_songs_csv(
    output_csv_path: Optional[str] = None,
    source_songs_csv: Optional[str] = None,
    mapping_csv_path: Optional[str] = None,
) -> Path:
    output_path = Path(output_csv_path) if output_csv_path else DEFAULT_ASSIGNED_SONGS_CSV
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = build_assigned_songs_frame(
        source_songs_csv=source_songs_csv,
        mapping_csv_path=mapping_csv_path,
    )
    frame.to_csv(output_path, index=False)
    return output_path


def _is_stale(path: Path, dependencies: Iterable[Path]) -> bool:
    if not path.exists():
        return True
    target_mtime = path.stat().st_mtime
    for dependency in dependencies:
        if dependency.exists() and dependency.stat().st_mtime > target_mtime:
            return True
    return False


def resolve_pipeline_songs_csv(
    requested_csv_path: Optional[str] = None,
    mapping_csv_path: Optional[str] = None,
) -> Path:
    if requested_csv_path:
        requested_path = Path(requested_csv_path)
        if not requested_path.is_absolute():
            requested_path = PROJECT_ROOT / requested_path
    else:
        requested_path = DEFAULT_SOURCE_SONGS_CSV

    source_path = DEFAULT_SOURCE_SONGS_CSV
    mapping_path = (
        Path(mapping_csv_path)
        if mapping_csv_path
        else DEFAULT_MERGED_GENRE_MAP_CSV
    )
    if not mapping_path.is_absolute():
        mapping_path = PROJECT_ROOT / mapping_path

    if requested_path == source_path:
        assigned_path = DEFAULT_ASSIGNED_SONGS_CSV
        if _is_stale(assigned_path, dependencies=[source_path, mapping_path]):
            write_assigned_songs_csv(
                output_csv_path=str(assigned_path),
                source_songs_csv=str(source_path),
                mapping_csv_path=str(mapping_path),
            )
        return assigned_path

    if requested_path == DEFAULT_ASSIGNED_SONGS_CSV:
        if _is_stale(requested_path, dependencies=[source_path, mapping_path]):
            write_assigned_songs_csv(
                output_csv_path=str(requested_path),
                source_songs_csv=str(source_path),
                mapping_csv_path=str(mapping_path),
            )
        return requested_path

    return requested_path


def load_taxonomy_songs(
    csv_path: Optional[str] = None,
    audio_only: bool = False,
    eligible_only: bool = False,
) -> pd.DataFrame:
    resolved_path = resolve_pipeline_songs_csv(csv_path)
    frame = load_unified_songs(str(resolved_path))

    if eligible_only:
        if INCLUDE_IN_MRS_COLUMN in frame.columns:
            include_mask = frame[INCLUDE_IN_MRS_COLUMN].apply(coerce_bool)
        else:
            include_mask = frame["primary_genre"].astype(str).str.strip().ne("")
        frame = frame[include_mask].copy()

    if audio_only and "has_audio" in frame.columns:
        frame = frame[frame["has_audio"].apply(coerce_bool)].copy()

    frame = frame.sort_values(
        by=["has_audio", "audio_basename", "filename"],
        ascending=[False, True, True],
        kind="stable",
    )
    return frame
