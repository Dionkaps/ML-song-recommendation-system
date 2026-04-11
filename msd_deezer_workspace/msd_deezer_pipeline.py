from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUBSET_DIR = SCRIPT_DIR.parent / "millionsongsubset" / "MillionSongSubset"
DEFAULT_CSV_PATH = SCRIPT_DIR / "data" / "msd_deezer_matches.csv"
DEFAULT_AUDIO_DIR = SCRIPT_DIR / "audio"
DEFAULT_CACHE_PATH = SCRIPT_DIR / "cache" / "deezer_search_cache.json"

DEEZE_SEARCH_URL = "https://api.deezer.com/search/track"
DEEZE_TRACK_URL = "https://api.deezer.com/track/{}"
SEARCH_TIMEOUT_SEC = 15
PREVIEW_TIMEOUT_SEC = 20
SEARCH_RESULT_LIMIT = 8
MAX_AUDIO_FILENAME_LENGTH = 180
DOWNLOAD_RETRY_ATTEMPTS = 4
DOWNLOAD_RETRY_BASE_DELAY_SEC = 2.0
DOWNLOAD_RETRY_MAX_DELAY_SEC = 20.0
SESSION_RETRY_BASE_DELAY_SEC = 5.0
SESSION_RETRY_MAX_DELAY_SEC = 60.0
CSV_WRITE_RETRY_ATTEMPTS = 12
CSV_WRITE_RETRY_BASE_DELAY_SEC = 0.25
CSV_WRITE_RETRY_MAX_DELAY_SEC = 3.0

CSV_FIELDNAMES = [
    "msd_row_index",
    "msd_file_path",
    "msd_track_id",
    "msd_song_id",
    "msd_title",
    "msd_artist_name",
    "msd_release",
    "msd_year",
    "msd_genre",
    "msd_artist_id",
    "msd_artist_mbid",
    "msd_artist_7digitalid",
    "msd_artist_playmeid",
    "msd_track_7digitalid",
    "msd_release_7digitalid",
    "msd_artist_location",
    "msd_artist_latitude",
    "msd_artist_longitude",
    "msd_artist_familiarity",
    "msd_artist_hotttnesss",
    "msd_song_hotttnesss",
    "msd_duration",
    "msd_tempo",
    "msd_loudness",
    "msd_key",
    "msd_key_confidence",
    "msd_mode",
    "msd_mode_confidence",
    "msd_time_signature",
    "msd_time_signature_confidence",
    "msd_danceability",
    "msd_energy",
    "msd_end_of_fade_in",
    "msd_start_of_fade_out",
    "msd_analysis_sample_rate",
    "msd_audio_md5",
    "msd_artist_terms",
    "msd_artist_mbtags",
    "msd_artist_mbtags_with_count",
    "deezer_track_id",
    "deezer_title",
    "deezer_artist",
    "deezer_album",
    "deezer_link",
    "deezer_isrc",
    "deezer_duration",
    "deezer_rank",
    "deezer_explicit_lyrics",
    "deezer_match_query",
    "deezer_match_score",
    "deezer_title_overlap",
    "deezer_artist_overlap",
    "deezer_duration_delta",
    "deezer_match_status",
    "deezer_download_status",
    "deezer_audio_filename",
    "deezer_audio_path",
    "deezer_error",
    "deezer_last_processed_utc",
]

DEEZE_MATCH_VALUE_FIELDS = [
    "deezer_track_id",
    "deezer_title",
    "deezer_artist",
    "deezer_album",
    "deezer_link",
    "deezer_isrc",
    "deezer_duration",
    "deezer_rank",
    "deezer_explicit_lyrics",
    "deezer_match_query",
    "deezer_match_score",
    "deezer_title_overlap",
    "deezer_artist_overlap",
    "deezer_duration_delta",
    "deezer_audio_filename",
    "deezer_audio_path",
]


def make_blank_row() -> dict[str, str]:
    return {field: "" for field in CSV_FIELDNAMES}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_optional_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", "").strip()
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    return text


def normalize_scalar(value: Any) -> str:
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray, str)):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            value = value.decode(errors="replace")

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}".rstrip("0").rstrip(".")

    if value is None:
        return ""

    return clean_optional_text(value)


def join_normalized(values: Any) -> str:
    items = []
    for value in values:
        text = normalize_scalar(value)
        if text:
            items.append(text)
    return "|".join(items)


def join_tag_counts(tags: Any, counts: Any) -> str:
    pairs = []
    for tag, count in zip(tags, counts):
        tag_text = normalize_scalar(tag)
        count_text = normalize_scalar(count) or "0"
        if tag_text:
            pairs.append(f"{tag_text}:{count_text}")
    return "|".join(pairs)


def progress(iterable: Any, total: int | None = None, desc: str = "") -> Any:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="song")
    return iterable


def atomic_write_rows(rows: list[dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = csv_path.with_name(f"{csv_path.name}.{time.time_ns()}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            normalized_row = make_blank_row()
            normalized_row.update({key: clean_optional_text(row.get(key, "")) for key in CSV_FIELDNAMES})
            writer.writerow(normalized_row)
        handle.flush()
        os.fsync(handle.fileno())

    last_error: Exception | None = None
    for attempt in range(1, CSV_WRITE_RETRY_ATTEMPTS + 1):
        try:
            temp_path.replace(csv_path)
            return
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            last_error = exc
            if getattr(exc, "winerror", None) != 5:
                break

        if attempt < CSV_WRITE_RETRY_ATTEMPTS:
            delay = min(
                CSV_WRITE_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)),
                CSV_WRITE_RETRY_MAX_DELAY_SEC,
            )
            time.sleep(delay)

    recovery_path = csv_path.with_name(f"{csv_path.stem}.pending{csv_path.suffix}")
    try:
        temp_path.replace(recovery_path)
    except OSError:
        pass

    details = (
        f"Could not replace {csv_path} after {CSV_WRITE_RETRY_ATTEMPTS} attempts. "
        "Another program or monitoring process may still have the file open."
    )
    if recovery_path.exists():
        details += f" Latest snapshot preserved at {recovery_path}."
    raise PermissionError(details) from last_error


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = make_blank_row()
            for field in CSV_FIELDNAMES:
                row[field] = clean_optional_text(raw_row.get(field, ""))
            rows.append(row)
    return rows


def load_existing_row_map(csv_path: Path) -> dict[str, dict[str, str]]:
    existing_rows = {}
    for row in load_csv_rows(csv_path):
        track_id = clean_optional_text(row.get("msd_track_id", ""))
        if track_id:
            existing_rows[track_id] = row
    return existing_rows


def load_search_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_search_cache(cache_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(cache_path)


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", filename).strip().rstrip(".")


def truncate_filename_component(text: str, max_length: int) -> str:
    text = sanitize_filename(text)
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[: max_length - 3].rstrip(" ._") + "..."


def normalize_search_text(text: Any) -> str:
    text = str(text or "").lower()
    text = text.replace("&", " and ")
    text = text.replace("_", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def clean_title_for_search(title: Any) -> str:
    title = str(title or "").replace("_", " ")
    title = re.sub(
        r"\(([^)]*(version|mix|remaster|live|edit|mono|stereo|album|explicit|clean|instrumental|demo)[^)]*)\)",
        " ",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"\[[^\]]*\]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip(" -")


def clean_artist_for_search(artist: Any) -> str:
    artist = str(artist or "").replace("_", " ")
    artist = re.sub(r"\b(feat(?:uring)?|ft\.?|with)\b.*$", " ", artist, flags=re.IGNORECASE)
    artist = re.sub(r"\([^)]*\)", " ", artist)
    artist = re.sub(r"\s+", " ", artist)
    return artist.strip(" -")


def token_overlap_ratio(expected: Any, actual: Any) -> float:
    expected_tokens = set(normalize_search_text(expected).split())
    actual_tokens = set(normalize_search_text(actual).split())
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & actual_tokens) / len(expected_tokens)


def normalize_float(text: Any) -> float | None:
    try:
        value = float(str(text).strip())
    except (AttributeError, TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def score_candidate(song_name: str, artist_name: str, expected_duration: float | None, candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_title = candidate.get("title", "")
    candidate_artist = candidate.get("artist", {}).get("name", "")

    title_overlap = token_overlap_ratio(clean_title_for_search(song_name), candidate_title)
    artist_overlap = token_overlap_ratio(clean_artist_for_search(artist_name), candidate_artist)

    expected_title_norm = normalize_search_text(clean_title_for_search(song_name))
    candidate_title_norm = normalize_search_text(candidate_title)
    expected_artist_norm = normalize_search_text(clean_artist_for_search(artist_name))
    candidate_artist_norm = normalize_search_text(candidate_artist)

    title_exactish = (
        expected_title_norm
        and candidate_title_norm
        and (
            expected_title_norm == candidate_title_norm
            or expected_title_norm in candidate_title_norm
            or candidate_title_norm in expected_title_norm
        )
    )
    artist_exactish = (
        expected_artist_norm
        and candidate_artist_norm
        and (
            expected_artist_norm == candidate_artist_norm
            or expected_artist_norm in candidate_artist_norm
            or candidate_artist_norm in expected_artist_norm
        )
    )

    duration_delta = None
    candidate_duration = normalize_float(candidate.get("duration"))
    if expected_duration is not None and candidate_duration is not None:
        duration_delta = abs(expected_duration - candidate_duration)

    score = 0.0
    score += 4.0 if title_exactish else 0.0
    score += 3.0 if artist_exactish else 0.0
    score += title_overlap * 3.0
    score += artist_overlap * 2.5
    score += 1.0 if candidate.get("preview") else -5.0

    if duration_delta is not None:
        if duration_delta <= 5:
            score += 2.0
        elif duration_delta <= 10:
            score += 1.0
        elif duration_delta >= 30:
            score -= 4.0
        elif duration_delta >= 20:
            score -= 2.0

    return {
        "score": score,
        "title_overlap": title_overlap,
        "artist_overlap": artist_overlap,
        "title_exactish": title_exactish,
        "artist_exactish": artist_exactish,
        "duration_delta": duration_delta,
    }


def is_confident_candidate(metrics: dict[str, Any], artist_name: str = "") -> bool:
    if not metrics:
        return False
    duration_delta = metrics.get("duration_delta")
    if duration_delta is not None and duration_delta > 30:
        return False
    if metrics["title_exactish"] and (metrics["artist_exactish"] or metrics["artist_overlap"] >= 0.5):
        return duration_delta is None or duration_delta <= 12
    if metrics["title_overlap"] >= 0.75 and (not artist_name or metrics["artist_overlap"] >= 0.6):
        return duration_delta is None or duration_delta <= 12
    return False


def build_search_queries(song_name: str, artist_name: str) -> list[str]:
    raw_title = str(song_name or "").strip()
    raw_artist = str(artist_name or "").strip()
    clean_title = clean_title_for_search(raw_title)
    clean_artist = clean_artist_for_search(raw_artist)

    queries = []
    for query in [
        f'track:"{raw_title}" artist:"{raw_artist}"' if raw_title and raw_artist else "",
        f'track:"{clean_title}" artist:"{clean_artist}"' if clean_title and clean_artist else "",
        f"{clean_title} {clean_artist}".strip(),
        f"{raw_title} {clean_artist}".strip(),
        f"{clean_title} {raw_artist}".strip(),
        clean_title,
    ]:
        if query and query not in queries:
            queries.append(query)
    return queries


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "msd-deezer-pipeline/1.0"})
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = SEARCH_TIMEOUT_SEC,
    max_retries: int = 3,
) -> tuple[Any | None, str | None]:
    backoff = 1.0
    last_error = None

    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt + 1 < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            return None, last_error

        if response.status_code == 200:
            try:
                return response.json(), None
            except ValueError as exc:
                return None, f"Invalid JSON response: {exc}"

        last_error = f"HTTP {response.status_code}"
        if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < max_retries:
            time.sleep(backoff)
            backoff *= 2
            continue

        snippet = response.text[:200].replace("\n", " ").strip()
        return None, f"{last_error}: {snippet}"

    return None, last_error or "Request failed"


def fetch_track_by_id(session: requests.Session, track_id: Any) -> dict[str, Any] | None:
    track_id = clean_optional_text(track_id)
    if not track_id:
        return None
    payload, error = get_json(session, DEEZE_TRACK_URL.format(track_id), timeout=SEARCH_TIMEOUT_SEC)
    if error or not isinstance(payload, dict) or payload.get("error"):
        return None
    return payload if payload.get("preview") else None


def run_search_query(session: requests.Session, query: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not query:
        return [], None
    payload, error = get_json(
        session,
        DEEZE_SEARCH_URL,
        params={"q": query, "limit": SEARCH_RESULT_LIMIT},
        timeout=SEARCH_TIMEOUT_SEC,
    )
    if error:
        return None, error
    if not isinstance(payload, dict):
        return [], "Unexpected Deezer payload"
    return payload.get("data", []) or [], None


def search_song(
    session: requests.Session,
    song_name: str,
    artist_name: str,
    expected_duration: float | None,
    search_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    cache_key = f"{song_name}|{artist_name}"
    cached = search_cache.get(cache_key)
    if isinstance(cached, dict):
        refreshed = fetch_track_by_id(session, cached.get("track_id"))
        if refreshed:
            metrics = score_candidate(song_name, artist_name, expected_duration, refreshed)
            if is_confident_candidate(metrics, artist_name):
                return {"song": refreshed, "metrics": metrics, "query": "__cache__"}, None
        search_cache.pop(cache_key, None)

    had_successful_query = False
    query_errors: list[str] = []
    candidates: dict[str, dict[str, Any]] = {}

    for query in build_search_queries(song_name, artist_name):
        results, error = run_search_query(session, query)
        if error:
            query_errors.append(f"{query}: {error}")
            continue

        had_successful_query = True
        for candidate in results or []:
            track_id = clean_optional_text(candidate.get("id"))
            if not track_id:
                continue
            metrics = score_candidate(song_name, artist_name, expected_duration, candidate)
            existing = candidates.get(track_id)
            if existing is None or metrics["score"] > existing["metrics"]["score"]:
                candidates[track_id] = {"song": candidate, "metrics": metrics, "query": query}

    if not candidates:
        if not had_successful_query and query_errors:
            return None, "; ".join(query_errors)
        return None, None

    ranked = sorted(candidates.values(), key=lambda item: item["metrics"]["score"], reverse=True)
    best = ranked[0]
    if not is_confident_candidate(best["metrics"], artist_name):
        return None, None

    search_cache[cache_key] = {"track_id": best["song"].get("id")}
    return best, None


def download_file(session: requests.Session, url: str, destination: Path) -> tuple[bool, str | None]:
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=PREVIEW_TIMEOUT_SEC) as response:
            if response.status_code != 200:
                return False, f"Preview HTTP {response.status_code}"

            destination.parent.mkdir(parents=True, exist_ok=True)
            bytes_written = 0
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_written += len(chunk)

        if bytes_written <= 0:
            return False, "Preview download was empty"

        temp_path.replace(destination)
        return True, None
    except requests.RequestException as exc:
        return False, str(exc)
    except OSError as exc:
        return False, f"Filesystem error: {exc}"
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def ensure_preview_download(
    session: requests.Session,
    matched_song: dict[str, Any],
    output_path: Path,
) -> tuple[bool, str | None, dict[str, Any]]:
    if output_path.exists() and output_path.stat().st_size > 0:
        return True, None, matched_song

    current_song = matched_song
    last_error = "Could not obtain a valid Deezer preview URL"

    for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
        preview_url = clean_optional_text(current_song.get("preview"))
        if preview_url:
            ok, error = download_file(session, preview_url, output_path)
            if ok:
                return True, None, current_song
            if error:
                last_error = error
        else:
            last_error = "Missing Deezer preview URL"

        refreshed = fetch_track_by_id(session, current_song.get("id"))
        if refreshed:
            current_song = refreshed

            refreshed_preview = clean_optional_text(current_song.get("preview"))
            if refreshed_preview:
                ok, error = download_file(session, refreshed_preview, output_path)
                if ok:
                    return True, None, current_song
                if error:
                    last_error = error
            else:
                last_error = "Missing Deezer preview URL after refresh"

        if attempt < DOWNLOAD_RETRY_ATTEMPTS:
            delay = min(
                DOWNLOAD_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)),
                DOWNLOAD_RETRY_MAX_DELAY_SEC,
            )
            time.sleep(delay)

    return False, f"Download failed after {DOWNLOAD_RETRY_ATTEMPTS} attempts: {last_error}", current_song


def build_audio_filename(msd_track_id: str, deezer_song: dict[str, Any]) -> str:
    title = clean_optional_text(deezer_song.get("title")) or "Unknown Title"
    artist = clean_optional_text(deezer_song.get("artist", {}).get("name")) or "Unknown Artist"
    deezer_track_id = clean_optional_text(deezer_song.get("id"))

    suffix_parts = []
    if msd_track_id:
        suffix_parts.append(msd_track_id)
    if deezer_track_id:
        suffix_parts.append(f"deezer-{deezer_track_id}")

    suffix = f" [{' ] ['.join(suffix_parts)}]" if suffix_parts else ""
    suffix = suffix.replace("[ ", "[").replace(" ]", "]")
    extension = ".mp3"
    max_base_length = max(24, MAX_AUDIO_FILENAME_LENGTH - len(suffix) - len(extension))
    base = truncate_filename_component(f"{artist} - {title}", max_base_length)

    if suffix_parts:
        return f"{base}{suffix}{extension}"
    return f"{base}{extension}"


def clear_deezer_match_fields(row: dict[str, str]) -> None:
    for field in DEEZE_MATCH_VALUE_FIELDS:
        row[field] = ""


def should_skip_match(row: dict[str, str], audio_dir: Path, retry_no_match: bool, redownload_existing: bool) -> bool:
    download_status = clean_optional_text(row.get("deezer_download_status", ""))
    match_status = clean_optional_text(row.get("deezer_match_status", ""))
    audio_filename = clean_optional_text(row.get("deezer_audio_filename", ""))

    if download_status == "downloaded" and audio_filename and not redownload_existing:
        audio_path = audio_dir / audio_filename
        if audio_path.exists() and audio_path.stat().st_size > 0:
            return True

    if match_status == "no_match" and not retry_no_match:
        return True

    return False


def populate_match_fields(
    row: dict[str, str],
    matched_song: dict[str, Any],
    metrics: dict[str, Any],
    query: str,
    audio_path: Path | None,
    csv_base_dir: Path,
) -> None:
    row["deezer_track_id"] = normalize_scalar(matched_song.get("id"))
    row["deezer_title"] = clean_optional_text(matched_song.get("title"))
    row["deezer_artist"] = clean_optional_text(matched_song.get("artist", {}).get("name"))
    row["deezer_album"] = clean_optional_text(matched_song.get("album", {}).get("title"))
    row["deezer_link"] = clean_optional_text(matched_song.get("link"))
    row["deezer_isrc"] = clean_optional_text(matched_song.get("isrc"))
    row["deezer_duration"] = normalize_scalar(matched_song.get("duration"))
    row["deezer_rank"] = normalize_scalar(matched_song.get("rank"))
    row["deezer_explicit_lyrics"] = normalize_scalar(matched_song.get("explicit_lyrics"))
    row["deezer_match_query"] = query
    row["deezer_match_score"] = normalize_scalar(metrics.get("score"))
    row["deezer_title_overlap"] = normalize_scalar(metrics.get("title_overlap"))
    row["deezer_artist_overlap"] = normalize_scalar(metrics.get("artist_overlap"))
    row["deezer_duration_delta"] = normalize_scalar(metrics.get("duration_delta"))
    if audio_path is not None:
        row["deezer_audio_filename"] = audio_path.name
        try:
            row["deezer_audio_path"] = audio_path.relative_to(csv_base_dir).as_posix()
        except ValueError:
            row["deezer_audio_path"] = str(audio_path.resolve())


def process_match_row(
    row: dict[str, str],
    session: requests.Session,
    search_cache: dict[str, dict[str, Any]],
    audio_dir: Path,
    csv_path: Path,
) -> str:
    song_name = clean_optional_text(row.get("msd_title", ""))
    artist_name = clean_optional_text(row.get("msd_artist_name", ""))
    expected_duration = normalize_float(row.get("msd_duration", ""))

    row["deezer_last_processed_utc"] = utc_now_iso()
    row["deezer_error"] = ""

    match, error = search_song(session, song_name, artist_name, expected_duration, search_cache)
    if error:
        clear_deezer_match_fields(row)
        row["deezer_match_status"] = "search_error"
        row["deezer_download_status"] = ""
        row["deezer_error"] = error
        return "search_error"

    if not match:
        clear_deezer_match_fields(row)
        row["deezer_match_status"] = "no_match"
        row["deezer_download_status"] = ""
        row["deezer_error"] = ""
        return "no_match"

    matched_song = match["song"]
    metrics = match["metrics"]
    query = clean_optional_text(match["query"])
    output_filename = build_audio_filename(clean_optional_text(row.get("msd_track_id", "")), matched_song)
    output_path = audio_dir / output_filename

    ok, download_error, freshest_song = ensure_preview_download(session, matched_song, output_path)
    audio_path_for_csv = output_path if ok or output_path.exists() else None
    populate_match_fields(
        row,
        freshest_song,
        metrics,
        query,
        audio_path_for_csv,
        csv_path.parent,
    )
    row["deezer_match_status"] = "matched"
    if ok:
        row["deezer_download_status"] = "downloaded"
        row["deezer_error"] = ""
        return "downloaded"

    row["deezer_download_status"] = "download_failed"
    row["deezer_error"] = clean_optional_text(download_error)
    return "download_failed"


def extract_song_row(file_path: Path, subset_dir: Path, existing_row: dict[str, str] | None, row_index: int) -> dict[str, str]:
    row = make_blank_row()
    row["msd_row_index"] = str(row_index)
    row["msd_file_path"] = file_path.relative_to(subset_dir).as_posix()

    with h5py.File(file_path, "r") as h5_file:
        metadata_song = h5_file["metadata"]["songs"][0]
        analysis_song = h5_file["analysis"]["songs"][0]
        musicbrainz_song = h5_file["musicbrainz"]["songs"][0]

        artist_terms = h5_file["metadata"]["artist_terms"][:]
        artist_mbtags = h5_file["musicbrainz"]["artist_mbtags"][:]
        artist_mbtags_count = h5_file["musicbrainz"]["artist_mbtags_count"][:]

        row.update(
            {
                "msd_track_id": normalize_scalar(analysis_song["track_id"]) or file_path.stem,
                "msd_song_id": normalize_scalar(metadata_song["song_id"]),
                "msd_title": normalize_scalar(metadata_song["title"]),
                "msd_artist_name": normalize_scalar(metadata_song["artist_name"]),
                "msd_release": normalize_scalar(metadata_song["release"]),
                "msd_year": normalize_scalar(musicbrainz_song["year"]),
                "msd_genre": normalize_scalar(metadata_song["genre"]),
                "msd_artist_id": normalize_scalar(metadata_song["artist_id"]),
                "msd_artist_mbid": normalize_scalar(metadata_song["artist_mbid"]),
                "msd_artist_7digitalid": normalize_scalar(metadata_song["artist_7digitalid"]),
                "msd_artist_playmeid": normalize_scalar(metadata_song["artist_playmeid"]),
                "msd_track_7digitalid": normalize_scalar(metadata_song["track_7digitalid"]),
                "msd_release_7digitalid": normalize_scalar(metadata_song["release_7digitalid"]),
                "msd_artist_location": normalize_scalar(metadata_song["artist_location"]),
                "msd_artist_latitude": normalize_scalar(metadata_song["artist_latitude"]),
                "msd_artist_longitude": normalize_scalar(metadata_song["artist_longitude"]),
                "msd_artist_familiarity": normalize_scalar(metadata_song["artist_familiarity"]),
                "msd_artist_hotttnesss": normalize_scalar(metadata_song["artist_hotttnesss"]),
                "msd_song_hotttnesss": normalize_scalar(metadata_song["song_hotttnesss"]),
                "msd_duration": normalize_scalar(analysis_song["duration"]),
                "msd_tempo": normalize_scalar(analysis_song["tempo"]),
                "msd_loudness": normalize_scalar(analysis_song["loudness"]),
                "msd_key": normalize_scalar(analysis_song["key"]),
                "msd_key_confidence": normalize_scalar(analysis_song["key_confidence"]),
                "msd_mode": normalize_scalar(analysis_song["mode"]),
                "msd_mode_confidence": normalize_scalar(analysis_song["mode_confidence"]),
                "msd_time_signature": normalize_scalar(analysis_song["time_signature"]),
                "msd_time_signature_confidence": normalize_scalar(analysis_song["time_signature_confidence"]),
                "msd_danceability": normalize_scalar(analysis_song["danceability"]),
                "msd_energy": normalize_scalar(analysis_song["energy"]),
                "msd_end_of_fade_in": normalize_scalar(analysis_song["end_of_fade_in"]),
                "msd_start_of_fade_out": normalize_scalar(analysis_song["start_of_fade_out"]),
                "msd_analysis_sample_rate": normalize_scalar(analysis_song["analysis_sample_rate"]),
                "msd_audio_md5": normalize_scalar(analysis_song["audio_md5"]),
                "msd_artist_terms": join_normalized(artist_terms),
                "msd_artist_mbtags": join_normalized(artist_mbtags),
                "msd_artist_mbtags_with_count": join_tag_counts(artist_mbtags, artist_mbtags_count),
            }
        )

    if existing_row:
        for field in CSV_FIELDNAMES:
            if field.startswith("deezer_"):
                row[field] = clean_optional_text(existing_row.get(field, ""))

    return row


def extract_metadata_csv(subset_dir: Path, csv_path: Path, preserve_existing: bool) -> tuple[list[dict[str, str]], list[str]]:
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset directory does not exist: {subset_dir}")

    h5_files = sorted(subset_dir.rglob("*.h5"))
    existing_rows = load_existing_row_map(csv_path) if preserve_existing else {}
    extracted_rows: list[dict[str, str]] = []
    errors: list[str] = []

    iterator = progress(enumerate(h5_files), total=len(h5_files), desc="Extracting MSD metadata")
    for row_index, file_path in iterator:
        try:
            existing_row = existing_rows.get(file_path.stem)
            row = extract_song_row(file_path, subset_dir, existing_row, row_index)
            existing_row = existing_rows.get(row["msd_track_id"])
            if existing_row:
                for field in CSV_FIELDNAMES:
                    if field.startswith("deezer_"):
                        row[field] = clean_optional_text(existing_row.get(field, ""))
            extracted_rows.append(row)
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")

    atomic_write_rows(extracted_rows, csv_path)
    return extracted_rows, errors


def match_and_download(
    rows: list[dict[str, str]],
    csv_path: Path,
    audio_dir: Path,
    cache_path: Path,
    start_index: int,
    limit: int | None,
    save_every: int,
    request_delay: float,
    retry_no_match: bool,
    redownload_existing: bool,
) -> dict[str, int]:
    if not rows:
        return {
            "attempted": 0,
            "downloaded": 0,
            "no_match": 0,
            "download_failed": 0,
            "search_errors": 0,
            "skipped_downloaded": 0,
            "skipped_no_match": 0,
        }

    session = build_session()
    search_cache = load_search_cache(cache_path)
    stats = {
        "attempted": 0,
        "downloaded": 0,
        "no_match": 0,
        "download_failed": 0,
        "search_errors": 0,
        "skipped_downloaded": 0,
        "skipped_no_match": 0,
    }
    final_outcomes: dict[int, str] = {}
    processed_since_save = 0
    audio_dir.mkdir(parents=True, exist_ok=True)
    selected_indices: list[int] = []

    for row_index in range(start_index, len(rows)):
        row = rows[row_index]
        if should_skip_match(row, audio_dir, retry_no_match=retry_no_match, redownload_existing=redownload_existing):
            if clean_optional_text(row.get("deezer_download_status")) == "downloaded":
                stats["skipped_downloaded"] += 1
            elif clean_optional_text(row.get("deezer_match_status")) == "no_match":
                stats["skipped_no_match"] += 1
            continue

        selected_indices.append(row_index)
        if limit is not None and len(selected_indices) >= limit:
            break

    if not selected_indices:
        return stats

    try:
        attempt = 1
        pending_indices = list(selected_indices)

        while pending_indices:
            if attempt > 1:
                print(f"\nRetry attempt {attempt - 1}: retrying {len(pending_indices)} failed song(s).")

            completed_in_attempt = 0
            failed_indices_this_attempt: list[int] = []
            iterator = progress(
                pending_indices,
                total=len(pending_indices),
                desc=f"Matching on Deezer (attempt {attempt})",
            )

            for row_index in iterator:
                row = rows[row_index]
                outcome = process_match_row(
                    row=row,
                    session=session,
                    search_cache=search_cache,
                    audio_dir=audio_dir,
                    csv_path=csv_path,
                )
                final_outcomes[row_index] = outcome

                if outcome == "downloaded":
                    completed_in_attempt += 1
                else:
                    failed_indices_this_attempt.append(row_index)

                processed_since_save += 1
                if processed_since_save >= max(1, save_every):
                    atomic_write_rows(rows, csv_path)
                    save_search_cache(cache_path, search_cache)
                    processed_since_save = 0

                if request_delay > 0:
                    time.sleep(request_delay)

            print(
                f"Attempt {attempt} complete: {completed_in_attempt} success, "
                f"{len(failed_indices_this_attempt)} failed"
            )

            if not failed_indices_this_attempt:
                print("All songs in this session completed successfully.")
                break

            if completed_in_attempt == 0:
                print("No successful downloads in this attempt - stopping retries.")
                break

            attempt += 1
            pending_indices = failed_indices_this_attempt
            backoff_time = min(
                SESSION_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 2)),
                SESSION_RETRY_MAX_DELAY_SEC,
            )
            print(f"Waiting {backoff_time:.0f} seconds before retry...\n")
            time.sleep(backoff_time)
    finally:
        atomic_write_rows(rows, csv_path)
        save_search_cache(cache_path, search_cache)

    stats["attempted"] = len(selected_indices)
    stats["downloaded"] = sum(1 for outcome in final_outcomes.values() if outcome == "downloaded")
    stats["no_match"] = sum(1 for outcome in final_outcomes.values() if outcome == "no_match")
    stats["download_failed"] = sum(1 for outcome in final_outcomes.values() if outcome == "download_failed")
    stats["search_errors"] = sum(1 for outcome in final_outcomes.values() if outcome == "search_error")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Million Song Subset metadata and enrich it with Deezer matches and preview downloads."
    )
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR, help="Path to the MillionSongSubset folder.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH, help="Path to the output CSV.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR, help="Folder for downloaded preview MP3 files.")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH, help="Path to the Deezer search cache JSON.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip the HDF5-to-CSV extraction step.")
    parser.add_argument("--extract-only", action="store_true", help="Only build the metadata CSV and stop before Deezer matching.")
    parser.add_argument(
        "--force-rebuild-csv",
        action="store_true",
        help="Rebuild the CSV from scratch without preserving existing Deezer columns.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="CSV row index to start Deezer matching from.")
    parser.add_argument("--limit", type=int, help="Maximum number of non-skipped songs to attempt during the Deezer stage.")
    parser.add_argument("--save-every", type=int, default=25, help="Persist the CSV and cache after this many attempts.")
    parser.add_argument("--request-delay", type=float, default=0.35, help="Delay between Deezer attempts in seconds.")
    parser.add_argument(
        "--retry-no-match",
        action="store_true",
        help="Retry rows that were already marked as no_match in a previous run.",
    )
    parser.add_argument(
        "--redownload-existing",
        action="store_true",
        help="Redownload previews even if the MP3 file already exists locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subset_dir = args.subset_dir.resolve()
    csv_path = args.csv_path.resolve()
    audio_dir = args.audio_dir.resolve()
    cache_path = args.cache_path.resolve()

    rows: list[dict[str, str]]
    extraction_errors: list[str] = []

    if args.skip_extract:
        rows = load_csv_rows(csv_path)
        if not rows:
            raise FileNotFoundError(f"CSV not found or empty: {csv_path}")
    else:
        rows, extraction_errors = extract_metadata_csv(
            subset_dir=subset_dir,
            csv_path=csv_path,
            preserve_existing=not args.force_rebuild_csv,
        )
        print(f"Metadata CSV written to: {csv_path}")
        print(f"Extracted rows: {len(rows)}")
        if extraction_errors:
            print(f"Extraction errors: {len(extraction_errors)}")
            for message in extraction_errors[:10]:
                print(f"  - {message}")

    if args.extract_only:
        return 0

    stats = match_and_download(
        rows=rows,
        csv_path=csv_path,
        audio_dir=audio_dir,
        cache_path=cache_path,
        start_index=max(0, args.start_index),
        limit=args.limit,
        save_every=max(1, args.save_every),
        request_delay=max(0.0, args.request_delay),
        retry_no_match=args.retry_no_match,
        redownload_existing=args.redownload_existing,
    )

    print(f"Updated CSV: {csv_path}")
    print(f"Audio folder: {audio_dir}")
    print("Deezer stage summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
