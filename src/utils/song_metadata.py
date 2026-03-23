import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


UNIFIED_SONG_COLUMNS: Tuple[str, ...] = (
    "msd_track_id",
    "msd_artist",
    "msd_title",
    "deezer_artist",
    "deezer_title",
    "filename",
    "audio_basename",
    "audio_extension",
    "has_audio",
    "genre",
    "primary_genre",
    "genre_count",
    "key",
    "mode",
    "loudness",
    "tempo",
    "key_confidence",
    "mode_confidence",
    "metadata_origin",
    "genre_source",
    "audio_match_source",
)

_STRING_COLUMNS: Tuple[str, ...] = (
    "msd_track_id",
    "msd_artist",
    "msd_title",
    "deezer_artist",
    "deezer_title",
    "filename",
    "audio_basename",
    "audio_extension",
    "genre",
    "primary_genre",
    "metadata_origin",
    "genre_source",
    "audio_match_source",
)

_NUMERIC_COLUMNS: Tuple[str, ...] = (
    "genre_count",
    "key",
    "mode",
    "loudness",
    "tempo",
    "key_confidence",
    "mode_confidence",
)


def normalize_metadata_text(text: Any) -> str:
    """Normalize free text for conservative metadata matching."""

    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().strip()
    value = value.replace("&", " and ")
    value = value.replace("`", "'").replace("’", "'")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\[[^\]]*\]", " ", value)
    value = re.sub(r"[^a-z0-9']+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def split_artist_title(text: Any) -> Tuple[str, str]:
    """Split a display string like 'Artist - Title' into its components."""

    value = str(text or "").strip()
    if " - " in value:
        artist, title = value.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", value


def first_nonempty(*values: Any) -> str:
    """Return the first non-empty string-like value."""

    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _genre_tokens(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _nonempty_string_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def ensure_unified_songs_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a songs dataframe to the canonical unified schema."""

    df = frame.copy()
    had_primary_genre_column = "primary_genre" in df.columns
    had_genre_count_column = "genre_count" in df.columns

    rename_map: Dict[str, str] = {}
    if "track_id" in df.columns and "msd_track_id" not in df.columns:
        rename_map["track_id"] = "msd_track_id"
    if "artist" in df.columns and "msd_artist" not in df.columns and "deezer_artist" not in df.columns:
        rename_map["artist"] = "deezer_artist" if "filename" in df.columns else "msd_artist"
    if "title" in df.columns and "msd_title" not in df.columns and "deezer_title" not in df.columns:
        rename_map["title"] = "deezer_title" if "filename" in df.columns else "msd_title"
    if rename_map:
        df = df.rename(columns=rename_map)

    for column in UNIFIED_SONG_COLUMNS:
        if column not in df.columns:
            if column == "has_audio":
                df[column] = False
            elif column == "genre_count":
                df[column] = 0
            else:
                df[column] = ""

    for column in _STRING_COLUMNS:
        df[column] = df[column].where(df[column].notna(), "").astype(str).str.strip()

    for column in _NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    filename_values = df["filename"].where(df["filename"].notna(), "").astype(str).str.strip()
    has_filename = filename_values.ne("")
    df.loc[has_filename, "audio_basename"] = filename_values.loc[has_filename].apply(
        lambda value: Path(value).stem
    )
    df.loc[has_filename, "audio_extension"] = filename_values.loc[has_filename].apply(
        lambda value: Path(value).suffix.lower()
    )

    df["has_audio"] = df["has_audio"].apply(_coerce_bool)

    if had_primary_genre_column:
        df["primary_genre"] = (
            df["primary_genre"]
            .where(df["primary_genre"].notna(), "")
            .astype(str)
            .str.strip()
        )
    else:
        df["primary_genre"] = df["genre"].apply(
            lambda value: _genre_tokens(value)[0] if _genre_tokens(value) else ""
        )

    derived_genre_counts = df["genre"].apply(lambda value: len(_genre_tokens(value)))
    if had_genre_count_column:
        parsed_counts = pd.to_numeric(df["genre_count"], errors="coerce")
        df["genre_count"] = (
            parsed_counts.where(parsed_counts.notna(), derived_genre_counts)
            .fillna(0)
            .astype(int)
        )
    else:
        df["genre_count"] = derived_genre_counts.astype(int)

    ordered = list(UNIFIED_SONG_COLUMNS)
    extras = [column for column in df.columns if column not in ordered]
    return df[ordered + extras].copy()


def load_unified_songs(csv_path: str) -> pd.DataFrame:
    """Load and normalize a unified songs CSV."""

    return ensure_unified_songs_schema(pd.read_csv(csv_path))


def load_legacy_msd_catalog(csv_path: str) -> pd.DataFrame:
    """Load the Million Song catalog CSV in either legacy or current schema."""

    df = pd.read_csv(csv_path)
    columns = [str(column).strip() for column in df.columns]
    df.columns = columns

    if {"track_id", "title", "artist", "genre"}.issubset(df.columns):
        normalized = df[["track_id", "artist", "title", "genre"]].rename(
            columns={
                "track_id": "msd_track_id",
                "artist": "msd_artist",
                "title": "msd_title",
            }
        )
    elif {"title", "artist", "genre"}.issubset(df.columns):
        normalized = df[["artist", "title", "genre"]].rename(
            columns={
                "artist": "msd_artist",
                "title": "msd_title",
            }
        )
        normalized["msd_track_id"] = ""
    else:
        raise ValueError(
            f"Unsupported millionsong dataset schema in {csv_path}: {columns}"
        )

    normalized["source_row_number"] = np.arange(len(normalized), dtype=np.int32)
    normalized["msd_track_id"] = (
        normalized["msd_track_id"]
        .where(normalized["msd_track_id"].notna(), "")
        .astype(str)
        .str.strip()
    )
    normalized["msd_artist"] = (
        normalized["msd_artist"]
        .where(normalized["msd_artist"].notna(), "")
        .astype(str)
        .str.strip()
    )
    normalized["msd_title"] = (
        normalized["msd_title"]
        .where(normalized["msd_title"].notna(), "")
        .astype(str)
        .str.strip()
    )
    normalized["genre"] = (
        normalized["genre"].where(normalized["genre"].notna(), "").astype(str).str.strip()
    )
    return normalized


def load_msd_feature_rows(csv_path: str) -> pd.DataFrame:
    """Load extracted MSD numeric metadata in canonical column names."""

    df = pd.read_csv(csv_path)
    if "track_id" not in df.columns:
        raise ValueError(f"Expected 'track_id' column in {csv_path}")

    normalized = df.rename(columns={"track_id": "msd_track_id"})
    columns = [
        "msd_track_id",
        "key",
        "mode",
        "loudness",
        "tempo",
        "key_confidence",
        "mode_confidence",
    ]
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = np.nan
    return normalized[columns].copy()


def load_legacy_downloaded_songs(csv_path: str) -> pd.DataFrame:
    """Load the legacy downloaded-song CSV in normalized column names."""

    df = pd.read_csv(csv_path)
    expected = {"title", "artist", "filename", "genre"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Unsupported downloaded-song schema in {csv_path}: {df.columns.tolist()}"
        )

    normalized = df.rename(
        columns={
            "title": "deezer_title",
            "artist": "deezer_artist",
        }
    )[["deezer_artist", "deezer_title", "filename", "genre"]].copy()
    for column in normalized.columns:
        normalized[column] = (
            normalized[column].where(normalized[column].notna(), "").astype(str).str.strip()
        )
    normalized["audio_basename"] = normalized["filename"].apply(lambda value: Path(value).stem)
    return normalized


def load_checkpoint_download_rows(json_path: str) -> pd.DataFrame:
    """Load downloaded-song metadata from the richer checkpoint JSON when available."""

    path = Path(json_path)
    columns = [
        "deezer_artist",
        "deezer_title",
        "filename",
        "genre",
        "audio_basename",
        "msd_track_id",
        "msd_artist",
        "msd_title",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, str]] = []
    for item in payload.get("downloaded_songs", []):
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename", "") or "").strip()
        audio_basename = str(item.get("audio_basename", "") or "").strip()
        if not audio_basename and filename:
            audio_basename = Path(filename).stem
        rows.append(
            {
                "deezer_artist": str(item.get("deezer_artist", item.get("artist", "")) or "").strip(),
                "deezer_title": str(item.get("deezer_title", item.get("title", "")) or "").strip(),
                "filename": filename,
                "genre": str(item.get("genre", "") or "").strip(),
                "audio_basename": audio_basename,
                "msd_track_id": str(item.get("msd_track_id", "") or "").strip(),
                "msd_artist": str(item.get("msd_artist", "") or "").strip(),
                "msd_title": str(item.get("msd_title", "") or "").strip(),
            }
        )

    checkpoint_rows = pd.DataFrame(rows, columns=columns)
    if checkpoint_rows.empty:
        return checkpoint_rows

    checkpoint_rows = checkpoint_rows[
        checkpoint_rows["audio_basename"].astype(str).str.strip().ne("")
    ].copy()
    checkpoint_rows["_has_track_id"] = checkpoint_rows["msd_track_id"].astype(str).str.strip().ne("")
    checkpoint_rows["_has_genre"] = checkpoint_rows["genre"].astype(str).str.strip().ne("")
    checkpoint_rows = checkpoint_rows.sort_values(
        by=["_has_track_id", "_has_genre", "audio_basename", "filename"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    checkpoint_rows = checkpoint_rows.drop_duplicates(subset=["audio_basename"], keep="first")
    checkpoint_rows = checkpoint_rows.drop(columns=["_has_track_id", "_has_genre"])
    return checkpoint_rows.reset_index(drop=True)


def load_legacy_match_rows(csv_path: str) -> pd.DataFrame:
    """Load the legacy audio-to-MSD match table."""

    df = pd.read_csv(csv_path)
    columns = [
        "filename",
        "msd_track_id",
        "match_type",
        "match_score",
        "msd_artist",
        "msd_title",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    normalized = df[columns].copy()
    for column in ("filename", "msd_track_id", "match_type", "msd_artist", "msd_title"):
        normalized[column] = (
            normalized[column].where(normalized[column].notna(), "").astype(str).str.strip()
        )
    normalized["audio_basename"] = normalized["filename"].apply(lambda value: Path(value).stem)
    normalized["match_score"] = pd.to_numeric(normalized["match_score"], errors="coerce")
    return normalized


def load_audio_mapping_rows(csv_path: str) -> pd.DataFrame:
    """Load the legacy filename to MSD track-id mapping."""

    df = pd.read_csv(csv_path)
    if not {"filename", "track_id"}.issubset(df.columns):
        raise ValueError(
            f"Unsupported audio mapping schema in {csv_path}: {df.columns.tolist()}"
        )
    normalized = df.rename(columns={"track_id": "msd_track_id"})[
        ["filename", "msd_track_id"]
    ].copy()
    normalized["filename"] = (
        normalized["filename"].where(normalized["filename"].notna(), "").astype(str).str.strip()
    )
    normalized["msd_track_id"] = (
        normalized["msd_track_id"]
        .where(normalized["msd_track_id"].notna(), "")
        .astype(str)
        .str.strip()
    )
    normalized["audio_basename"] = normalized["filename"].apply(lambda value: Path(value).stem)
    return normalized


def load_audio_library_rows(audio_dir: str) -> pd.DataFrame:
    """Enumerate the current workspace audio library."""

    audio_root = Path(audio_dir)
    rows: List[Dict[str, str]] = []
    if not audio_root.exists():
        return pd.DataFrame(columns=["filename", "audio_basename", "audio_extension"])

    for path in sorted(audio_root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a"}:
            continue
        rows.append(
            {
                "filename": path.name,
                "audio_basename": path.stem,
                "audio_extension": path.suffix.lower(),
            }
        )
    return pd.DataFrame(rows)


def load_cached_genre_map(path: str) -> Dict[str, str]:
    """Load the cached audio-basename to primary-genre map when available."""

    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    payload = np.load(cache_path, allow_pickle=True).item()
    return {
        str(key): str(value)
        for key, value in payload.items()
        if str(key).strip()
    }


def _dedupe_msd_catalog(msd_catalog: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Collapse the MSD catalog to one row per non-empty track id."""

    original_rows = len(msd_catalog)
    exact_dedup = msd_catalog.drop_duplicates(
        subset=["msd_track_id", "msd_artist", "msd_title", "genre"]
    ).copy()
    exact_duplicates_removed = original_rows - len(exact_dedup)

    nonempty = exact_dedup[exact_dedup["msd_track_id"].ne("")].copy()
    missing_track_id_rows = exact_dedup[exact_dedup["msd_track_id"].eq("")].copy()

    conflicting_track_ids = 0
    if not nonempty.empty:
        for _, group in nonempty.groupby("msd_track_id"):
            unique_rows = group[["msd_artist", "msd_title", "genre"]].drop_duplicates()
            if len(unique_rows) > 1:
                conflicting_track_ids += 1

    nonempty["_has_genre"] = nonempty["genre"].ne("")
    nonempty["_title_len"] = nonempty["msd_title"].str.len()
    nonempty["_artist_len"] = nonempty["msd_artist"].str.len()
    nonempty = nonempty.sort_values(
        by=["_has_genre", "_title_len", "_artist_len", "source_row_number"],
        ascending=[False, False, False, True],
    )
    deduped_nonempty = nonempty.drop_duplicates(subset=["msd_track_id"], keep="first")
    track_id_duplicates_removed = len(nonempty) - len(deduped_nonempty)

    deduped = pd.concat(
        [
            deduped_nonempty.drop(columns=["_has_genre", "_title_len", "_artist_len"]),
            missing_track_id_rows,
        ],
        ignore_index=True,
    )

    summary = {
        "msd_catalog_rows_original": int(original_rows),
        "msd_catalog_rows_after_exact_dedup": int(len(exact_dedup)),
        "msd_catalog_exact_duplicates_removed": int(exact_duplicates_removed),
        "msd_catalog_track_id_duplicates_removed": int(track_id_duplicates_removed),
        "msd_catalog_conflicting_track_id_groups": int(conflicting_track_ids),
        "msd_catalog_rows_missing_track_id": int(len(missing_track_id_rows)),
        "msd_catalog_rows_final": int(len(deduped)),
    }
    return deduped, summary


def _build_unique_name_lookup(
    frame: pd.DataFrame,
    artist_column: str,
    title_column: str,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    indices: Dict[str, int] = {}

    for idx, row in frame.iterrows():
        key = normalize_metadata_text(
            f"{row.get(artist_column, '')} - {row.get(title_column, '')}"
        )
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        if key not in indices:
            indices[key] = int(idx)

    return {key: indices[key] for key, count in counts.items() if count == 1}


def _apply_audio_assignment(
    frame: pd.DataFrame,
    row_idx: int,
    audio_filename: str,
    audio_basename: str,
    audio_extension: str,
    deezer_artist: str,
    deezer_title: str,
    genre: str,
    metadata_origin: str,
    genre_source: str,
    audio_match_source: str,
) -> None:
    if deezer_artist and not str(frame.at[row_idx, "deezer_artist"]).strip():
        frame.at[row_idx, "deezer_artist"] = deezer_artist
    if deezer_title and not str(frame.at[row_idx, "deezer_title"]).strip():
        frame.at[row_idx, "deezer_title"] = deezer_title
    if genre and not str(frame.at[row_idx, "genre"]).strip():
        frame.at[row_idx, "genre"] = genre
        if genre_source:
            frame.at[row_idx, "genre_source"] = genre_source

    frame.at[row_idx, "filename"] = audio_filename
    frame.at[row_idx, "audio_basename"] = audio_basename
    frame.at[row_idx, "audio_extension"] = audio_extension
    frame.at[row_idx, "has_audio"] = True
    frame.at[row_idx, "metadata_origin"] = metadata_origin
    frame.at[row_idx, "audio_match_source"] = audio_match_source


def build_unified_songs_dataframe(
    project_root: str,
    data_dir: str = "data",
    audio_dir: str = "audio_files",
    results_dir: str = "output/features",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Rebuild the canonical unified songs dataframe from local source files."""

    project_root_path = Path(project_root)
    data_root = project_root_path / data_dir
    backup_root = data_root / "backup_old_csvs"
    audio_root = project_root_path / audio_dir
    results_root = project_root_path / results_dir

    msd_catalog = load_legacy_msd_catalog(str(data_root / "millionsong_dataset.csv"))
    msd_catalog, summary = _dedupe_msd_catalog(msd_catalog)

    features_path = (
        data_root / "msd_features.csv"
        if (data_root / "msd_features.csv").exists()
        else backup_root / "msd_features.csv"
    )
    if features_path.exists():
        msd_features = load_msd_feature_rows(str(features_path)).drop_duplicates(
            subset=["msd_track_id"],
            keep="first",
        )
    else:
        msd_features = pd.DataFrame(
            columns=[
                "msd_track_id",
                "key",
                "mode",
                "loudness",
                "tempo",
                "key_confidence",
                "mode_confidence",
            ]
        )

    unified = msd_catalog.merge(msd_features, on="msd_track_id", how="left")
    unified["deezer_artist"] = ""
    unified["deezer_title"] = ""
    unified["filename"] = ""
    unified["audio_basename"] = ""
    unified["audio_extension"] = ""
    unified["has_audio"] = False
    unified["metadata_origin"] = "msd_catalog"
    unified["genre_source"] = np.where(unified["genre"].ne(""), "msd_catalog", "")
    unified["audio_match_source"] = ""

    for column in ("key", "mode", "loudness", "tempo", "key_confidence", "mode_confidence"):
        if column not in unified.columns:
            unified[column] = np.nan

    downloaded_path = (
        data_root / "songs_data_with_genre.csv"
        if (data_root / "songs_data_with_genre.csv").exists()
        else backup_root / "songs_data_with_genre.csv"
    )
    checkpoint_path = project_root_path / "download_checkpoint_with_genre.json"
    matches_path = (
        data_root / "msd_matches.csv"
        if (data_root / "msd_matches.csv").exists()
        else backup_root / "msd_matches.csv"
    )
    mapping_path = (
        data_root / "audio_msd_mapping.csv"
        if (data_root / "audio_msd_mapping.csv").exists()
        else backup_root / "audio_msd_mapping.csv"
    )

    downloaded = (
        load_legacy_downloaded_songs(str(downloaded_path))
        if downloaded_path.exists()
        else pd.DataFrame(
            columns=["deezer_artist", "deezer_title", "filename", "genre", "audio_basename"]
        )
    )
    checkpoint_downloaded = load_checkpoint_download_rows(str(checkpoint_path))
    matches = (
        load_legacy_match_rows(str(matches_path))
        if matches_path.exists()
        else pd.DataFrame(
            columns=["filename", "msd_track_id", "match_type", "match_score", "msd_artist", "msd_title", "audio_basename"]
        )
    )
    mapping = (
        load_audio_mapping_rows(str(mapping_path))
        if mapping_path.exists()
        else pd.DataFrame(columns=["filename", "msd_track_id", "audio_basename"])
    )

    if not downloaded.empty:
        downloaded = downloaded.merge(
            matches[["filename", "msd_track_id", "match_type", "match_score", "msd_artist", "msd_title"]],
            on="filename",
            how="left",
        )
        downloaded = downloaded.merge(
            mapping.rename(columns={"msd_track_id": "mapped_track_id"})[
                ["filename", "mapped_track_id"]
            ],
            on="filename",
            how="left",
        )
        downloaded["resolved_track_id"] = downloaded["msd_track_id"].where(
            downloaded["msd_track_id"].notna()
            & downloaded["msd_track_id"].astype(str).str.strip().ne(""),
            downloaded["mapped_track_id"],
        )
    else:
        downloaded["resolved_track_id"] = []

    if not checkpoint_downloaded.empty:
        checkpoint_enrichment = checkpoint_downloaded.rename(
            columns={
                "deezer_artist": "checkpoint_deezer_artist",
                "deezer_title": "checkpoint_deezer_title",
                "filename": "checkpoint_filename",
                "genre": "checkpoint_genre",
                "msd_track_id": "checkpoint_track_id",
                "msd_artist": "checkpoint_msd_artist",
                "msd_title": "checkpoint_msd_title",
            }
        )
        if downloaded.empty:
            downloaded = checkpoint_enrichment.rename(
                columns={
                    "checkpoint_deezer_artist": "deezer_artist",
                    "checkpoint_deezer_title": "deezer_title",
                    "checkpoint_filename": "filename",
                    "checkpoint_genre": "genre",
                    "checkpoint_track_id": "resolved_track_id",
                    "checkpoint_msd_artist": "msd_artist",
                    "checkpoint_msd_title": "msd_title",
                }
            ).copy()
            downloaded["msd_track_id"] = downloaded["resolved_track_id"]
            downloaded["match_type"] = ""
            downloaded["match_score"] = np.nan
            downloaded["mapped_track_id"] = ""
        else:
            downloaded = downloaded.merge(
                checkpoint_enrichment,
                on="audio_basename",
                how="left",
            )

            for column, checkpoint_column in (
                ("deezer_artist", "checkpoint_deezer_artist"),
                ("deezer_title", "checkpoint_deezer_title"),
                ("filename", "checkpoint_filename"),
                ("genre", "checkpoint_genre"),
                ("msd_artist", "checkpoint_msd_artist"),
                ("msd_title", "checkpoint_msd_title"),
            ):
                downloaded[column] = downloaded[column].where(
                    _nonempty_string_mask(downloaded[column]),
                    downloaded[checkpoint_column],
                )

            downloaded["resolved_track_id"] = downloaded["resolved_track_id"].where(
                _nonempty_string_mask(downloaded["resolved_track_id"]),
                downloaded["checkpoint_track_id"],
            )
            downloaded["msd_track_id"] = downloaded["msd_track_id"].where(
                _nonempty_string_mask(downloaded["msd_track_id"]),
                downloaded["checkpoint_track_id"],
            )

            for column in (
                "checkpoint_deezer_artist",
                "checkpoint_deezer_title",
                "checkpoint_filename",
                "checkpoint_genre",
                "checkpoint_track_id",
                "checkpoint_msd_artist",
                "checkpoint_msd_title",
            ):
                if column in downloaded.columns:
                    downloaded = downloaded.drop(columns=[column])

            missing_checkpoint_rows = checkpoint_downloaded[
                ~checkpoint_downloaded["audio_basename"].isin(downloaded["audio_basename"])
            ].copy()
            if not missing_checkpoint_rows.empty:
                missing_checkpoint_rows["msd_track_id"] = missing_checkpoint_rows["msd_track_id"]
                missing_checkpoint_rows["match_type"] = ""
                missing_checkpoint_rows["match_score"] = np.nan
                missing_checkpoint_rows["mapped_track_id"] = ""
                missing_checkpoint_rows["resolved_track_id"] = missing_checkpoint_rows["msd_track_id"]
                downloaded = pd.concat([downloaded, missing_checkpoint_rows], ignore_index=True, sort=False)

    downloaded_fallback_by_stem: Dict[str, Dict[str, str]] = {}
    if not downloaded.empty:
        downloaded_fallback = downloaded.copy()
        downloaded_fallback["_has_genre"] = (
            downloaded_fallback["genre"].astype(str).str.strip().ne("")
        )
        downloaded_fallback = downloaded_fallback.sort_values(
            by=["_has_genre"],
            ascending=[False],
            kind="stable",
        )
        for _, row in downloaded_fallback.iterrows():
            stem = str(row.get("audio_basename", "") or "").strip()
            if not stem or stem in downloaded_fallback_by_stem:
                continue
            downloaded_fallback_by_stem[stem] = {
                "deezer_artist": str(row.get("deezer_artist", "") or "").strip(),
                "deezer_title": str(row.get("deezer_title", "") or "").strip(),
                "genre": str(row.get("genre", "") or "").strip(),
            }

    audio_rows = load_audio_library_rows(str(audio_root))
    audio_by_stem = {
        row["audio_basename"]: {
            "filename": row["filename"],
            "audio_extension": row["audio_extension"],
        }
        for _, row in audio_rows.iterrows()
    }
    cached_genre_map = load_cached_genre_map(str(results_root / "genre_map.npy"))

    track_id_lookup = {
        str(track_id): int(idx)
        for idx, track_id in unified["msd_track_id"].items()
        if str(track_id).strip()
    }
    unique_msd_name_lookup = _build_unique_name_lookup(
        unified,
        artist_column="msd_artist",
        title_column="msd_title",
    )
    mapping_by_stem = {
        str(row["audio_basename"]): str(row["msd_track_id"])
        for _, row in mapping.iterrows()
        if str(row["audio_basename"]).strip() and str(row["msd_track_id"]).strip()
    }

    assigned_audio_stems = set()
    assignment_counts = {
        "legacy_download": 0,
        "legacy_audio_mapping": 0,
        "normalized_audio_to_msd": 0,
        "audio_only_cache": 0,
    }

    if not downloaded.empty:
        for _, row in downloaded.iterrows():
            resolved_track_id = str(row.get("resolved_track_id", "") or "").strip()
            stem = str(row.get("audio_basename", "") or "").strip()
            current_audio = audio_by_stem.get(stem)

            row_idx = track_id_lookup.get(resolved_track_id)
            if row_idx is None:
                match_key = normalize_metadata_text(
                    f"{row.get('deezer_artist', '')} - {row.get('deezer_title', '')}"
                )
                row_idx = unique_msd_name_lookup.get(match_key)

            if row_idx is None:
                continue

            if current_audio is not None and (
                not bool(unified.at[row_idx, "has_audio"])
                or str(unified.at[row_idx, "audio_basename"]).strip() == stem
            ):
                _apply_audio_assignment(
                    frame=unified,
                    row_idx=row_idx,
                    audio_filename=current_audio["filename"],
                    audio_basename=stem,
                    audio_extension=current_audio["audio_extension"],
                    deezer_artist=str(row.get("deezer_artist", "") or ""),
                    deezer_title=str(row.get("deezer_title", "") or ""),
                    genre=str(row.get("genre", "") or ""),
                    metadata_origin="msd_catalog+legacy_download",
                    genre_source="legacy_download" if str(row.get("genre", "") or "").strip() else "",
                    audio_match_source="legacy_download",
                )
                assigned_audio_stems.add(stem)
                assignment_counts["legacy_download"] += 1
            else:
                if str(row.get("deezer_artist", "") or "").strip():
                    unified.at[row_idx, "deezer_artist"] = str(row.get("deezer_artist", "") or "")
                if str(row.get("deezer_title", "") or "").strip():
                    unified.at[row_idx, "deezer_title"] = str(row.get("deezer_title", "") or "")
                if (
                    not str(unified.at[row_idx, "genre"]).strip()
                    and str(row.get("genre", "") or "").strip()
                ):
                    unified.at[row_idx, "genre"] = str(row.get("genre", "") or "")
                    unified.at[row_idx, "genre_source"] = "legacy_download"
                if not bool(unified.at[row_idx, "has_audio"]) and stem:
                    unified.at[row_idx, "filename"] = str(row.get("filename", "") or "")
                    unified.at[row_idx, "audio_basename"] = stem
                    unified.at[row_idx, "audio_extension"] = Path(
                        str(row.get("filename", "") or "")
                    ).suffix.lower()
                    unified.at[row_idx, "metadata_origin"] = "msd_catalog+legacy_download"

    extra_rows: List[Dict[str, Any]] = []
    for audio_basename, audio_info in audio_by_stem.items():
        if audio_basename in assigned_audio_stems:
            continue

        artist_name, title_name = split_artist_title(audio_basename)
        downloaded_fallback = downloaded_fallback_by_stem.get(audio_basename, {})
        cached_genre = cached_genre_map.get(audio_basename, "")
        fallback_genre = first_nonempty(downloaded_fallback.get("genre"), cached_genre)
        fallback_artist = first_nonempty(
            downloaded_fallback.get("deezer_artist"),
            artist_name,
        )
        fallback_title = first_nonempty(
            downloaded_fallback.get("deezer_title"),
            title_name,
        )
        fallback_genre_source = (
            "legacy_download"
            if str(downloaded_fallback.get("genre", "")).strip()
            else ("genre_cache" if cached_genre else "")
        )
        fallback_audio_match_source = (
            "legacy_download_fallback"
            if downloaded_fallback
            else "genre_cache_only"
        )
        track_id = mapping_by_stem.get(audio_basename, "")
        row_idx = track_id_lookup.get(track_id) if track_id else None
        if row_idx is not None and (
            not bool(unified.at[row_idx, "has_audio"])
            or str(unified.at[row_idx, "audio_basename"]).strip() == audio_basename
        ):
            _apply_audio_assignment(
                frame=unified,
                row_idx=row_idx,
                audio_filename=audio_info["filename"],
                audio_basename=audio_basename,
                audio_extension=audio_info["audio_extension"],
                deezer_artist=artist_name,
                deezer_title=title_name,
                genre=fallback_genre,
                metadata_origin="msd_catalog+audio_mapping",
                genre_source=fallback_genre_source,
                audio_match_source="legacy_audio_mapping",
            )
            assigned_audio_stems.add(audio_basename)
            assignment_counts["legacy_audio_mapping"] += 1
            continue

        match_key = normalize_metadata_text(audio_basename)
        row_idx = unique_msd_name_lookup.get(match_key)
        if row_idx is not None and (
            not bool(unified.at[row_idx, "has_audio"])
            or str(unified.at[row_idx, "audio_basename"]).strip() == audio_basename
        ):
            _apply_audio_assignment(
                frame=unified,
                row_idx=row_idx,
                audio_filename=audio_info["filename"],
                audio_basename=audio_basename,
                audio_extension=audio_info["audio_extension"],
                deezer_artist=artist_name,
                deezer_title=title_name,
                genre=fallback_genre,
                metadata_origin="msd_catalog+normalized_audio",
                genre_source=fallback_genre_source,
                audio_match_source="normalized_audio_to_msd",
            )
            assigned_audio_stems.add(audio_basename)
            assignment_counts["normalized_audio_to_msd"] += 1
            continue

        extra_rows.append(
            {
                "msd_track_id": "",
                "msd_artist": "",
                "msd_title": "",
                "deezer_artist": fallback_artist,
                "deezer_title": fallback_title,
                "filename": audio_info["filename"],
                "audio_basename": audio_basename,
                "audio_extension": audio_info["audio_extension"],
                "has_audio": True,
                "genre": fallback_genre,
                "key": np.nan,
                "mode": np.nan,
                "loudness": np.nan,
                "tempo": np.nan,
                "key_confidence": np.nan,
                "mode_confidence": np.nan,
                "metadata_origin": "audio_only_cache",
                "genre_source": fallback_genre_source,
                "audio_match_source": fallback_audio_match_source,
            }
        )
        assigned_audio_stems.add(audio_basename)
        assignment_counts["audio_only_cache"] += 1

    if extra_rows:
        unified = pd.concat([unified, pd.DataFrame(extra_rows)], ignore_index=True)

    unified = ensure_unified_songs_schema(unified)
    unified = unified.sort_values(
        by=["has_audio", "audio_basename", "msd_track_id", "deezer_artist", "deezer_title"],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    audio_rows_with_track_id = int(
        (
            unified["has_audio"]
            & unified["msd_track_id"].astype(str).str.strip().ne("")
        ).sum()
    )
    audio_rows_with_numeric_msd = int(
        (
            unified["has_audio"]
            & unified["key"].notna()
            & unified["mode"].notna()
            & unified["loudness"].notna()
            & unified["tempo"].notna()
        ).sum()
    )

    summary.update(
        {
            "msd_feature_rows": int(len(msd_features)),
            "legacy_download_rows": int(len(downloaded)),
            "legacy_match_rows": int(len(matches)),
            "legacy_audio_mapping_rows": int(len(mapping)),
            "current_audio_rows": int(len(audio_rows)),
            "current_audio_assigned_via_legacy_download": int(assignment_counts["legacy_download"]),
            "current_audio_assigned_via_legacy_audio_mapping": int(assignment_counts["legacy_audio_mapping"]),
            "current_audio_assigned_via_normalized_msd_match": int(assignment_counts["normalized_audio_to_msd"]),
            "current_audio_audio_only_rows": int(assignment_counts["audio_only_cache"]),
            "unified_rows_final": int(len(unified)),
            "unified_rows_with_audio": int(unified["has_audio"].sum()),
            "unified_rows_with_msd_track_id": int(
                unified["msd_track_id"].astype(str).str.strip().ne("").sum()
            ),
            "audio_rows_with_msd_track_id": int(audio_rows_with_track_id),
            "audio_rows_with_numeric_msd_features": int(audio_rows_with_numeric_msd),
            "audio_rows_without_msd_track_id": int(
                unified["has_audio"].sum() - audio_rows_with_track_id
            ),
            "audio_rows_without_numeric_msd_features": int(
                unified["has_audio"].sum() - audio_rows_with_numeric_msd
            ),
        }
    )
    return unified, summary


def build_audio_metadata_frame(
    base_names: Sequence[str],
    songs_csv_path: str = "data/songs.csv",
) -> pd.DataFrame:
    """Return metadata rows aligned to a list of audio basenames."""

    lookup: Dict[str, Dict[str, Any]] = {}
    csv_path = Path(songs_csv_path)

    if csv_path.exists():
        unified = load_unified_songs(str(csv_path))
        preferred = unified.sort_values(
            by=["has_audio", "msd_track_id", "filename"],
            ascending=[False, True, True],
            kind="stable",
        )
        for _, row in preferred.iterrows():
            key_candidates = [
                str(row.get("audio_basename", "") or "").strip(),
                Path(str(row.get("filename", "") or "")).stem,
            ]
            artist_name = first_nonempty(row.get("deezer_artist"), row.get("msd_artist"))
            title_name = first_nonempty(row.get("deezer_title"), row.get("msd_title"))
            if not artist_name or not title_name:
                fallback_artist, fallback_title = split_artist_title(
                    first_nonempty(
                        row.get("audio_basename"),
                        Path(str(row.get("filename", "") or "")).stem,
                    )
                )
                artist_name = first_nonempty(artist_name, fallback_artist)
                title_name = first_nonempty(title_name, fallback_title)

            payload = {
                "Artist": artist_name,
                "Title": title_name,
                "Filename": first_nonempty(row.get("filename")),
                "AudioBasename": first_nonempty(row.get("audio_basename")),
                "MSDTrackID": first_nonempty(row.get("msd_track_id")),
                "GenreList": first_nonempty(row.get("genre")),
                "PrimaryGenre": first_nonempty(row.get("primary_genre"), "unknown"),
                "PrimaryGenres": first_nonempty(
                    row.get("mapped_primary_genres"),
                    row.get("genre"),
                ),
                "SecondaryTags": first_nonempty(row.get("mapped_secondary_tags")),
                "AllGenreTags": first_nonempty(
                    row.get("mapped_all_tags"),
                    row.get("genre"),
                ),
                "OriginalGenreList": first_nonempty(
                    row.get("original_genre"),
                    row.get("genre"),
                ),
                "OriginalPrimaryGenre": first_nonempty(
                    row.get("original_primary_genre"),
                    row.get("primary_genre"),
                    "unknown",
                ),
                "IncludeInMRS": _coerce_bool(row.get("include_in_mrs", True)),
                "HasAudio": _coerce_bool(row.get("has_audio")),
            }
            for key in key_candidates:
                if key and key not in lookup:
                    lookup[key] = payload

    rows: List[Dict[str, Any]] = []
    for base_name in base_names:
        metadata = lookup.get(str(base_name))
        if metadata is None:
            artist_name, title_name = split_artist_title(base_name)
            metadata = {
                "Artist": artist_name,
                "Title": title_name,
                "Filename": "",
                "AudioBasename": str(base_name),
                "MSDTrackID": "",
                "GenreList": "",
                "PrimaryGenre": "unknown",
                "PrimaryGenres": "",
                "SecondaryTags": "",
                "AllGenreTags": "",
                "OriginalGenreList": "",
                "OriginalPrimaryGenre": "unknown",
                "IncludeInMRS": False,
                "HasAudio": False,
            }
        rows.append(metadata)
    return pd.DataFrame(rows)
