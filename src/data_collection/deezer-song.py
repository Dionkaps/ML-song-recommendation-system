import requests
import argparse
import os
import sys
import re
import time
import multiprocessing
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from src.utils.song_metadata import ensure_unified_songs_schema

DEEZE_AUDIO_FOLDER = "audio_files"
UNIFIED_CSV = os.path.join("data", "songs.csv")  # Unified CSV with all song data
CHECKPOINT_FILE = "download_checkpoint_with_genre.json"
INPUT_CSV = os.path.join("data", "millionsong_dataset.csv")
SEARCH_CACHE_FILE = "deezer_search_cache.json"  # Cache successful searches

# Global search cache
_search_cache = {}
_download_registry_by_track_id = {}
_download_registry_by_filename = {}

DEEZE_SEARCH_URL = "https://api.deezer.com/search/track"
DEEZE_TRACK_URL = "https://api.deezer.com/track/{}"
SEARCH_TIMEOUT_SEC = 10
PREVIEW_TIMEOUT_SEC = 15
SEARCH_RESULT_LIMIT = 8


def clean_optional_text(value):
    """Return a stripped string while treating null-ish values as empty."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    return text


def normalize_input_row(row):
    """Normalize CSV row keys so BOM/quoted headers do not break track-id loading."""
    if not isinstance(row, dict):
        return row

    normalized = {}
    key_aliases = {
        "track_id": "track_id",
        "title": "title",
        "artist": "artist",
        "genre": "genre",
    }
    for raw_key, value in row.items():
        key_text = str(raw_key or "").replace("\ufeff", "").strip()
        key_text = key_text.strip('"').strip("'")
        canonical = key_aliases.get(key_text.lower(), key_text)
        if canonical not in normalized:
            normalized[canonical] = clean_optional_text(value)
    return normalized


def load_search_cache():
    """Load cached search results from previous runs"""
    global _search_cache
    if os.path.exists(SEARCH_CACHE_FILE):
        try:
            with open(SEARCH_CACHE_FILE, 'r', encoding='utf-8') as f:
                payload = json.load(f)
                _search_cache = payload if isinstance(payload, dict) else {}
        except Exception:
            _search_cache = {}
    return _search_cache


def save_search_cache():
    """Save search cache to file for reuse in future runs"""
    try:
        with open(SEARCH_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_search_cache, f, indent=2)
    except Exception:
        pass


def normalize_search_text(text):
    """Normalize text for fuzzy title/artist comparisons."""
    text = str(text or "").lower()
    text = text.replace("&", " and ")
    text = text.replace("_", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def clean_title_for_search(title):
    """Remove noisy versioning text that hurts Deezer matching."""
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


def clean_artist_for_search(artist):
    """Strip featured-artist noise and parenthetical notes from artist strings."""
    artist = str(artist or "").replace("_", " ")
    artist = re.sub(r"\b(feat(?:uring)?|ft\.?|with)\b.*$", " ", artist, flags=re.IGNORECASE)
    artist = re.sub(r"\([^)]*\)", " ", artist)
    artist = re.sub(r"\s+", " ", artist)
    return artist.strip(" -")


def token_overlap_ratio(expected, actual):
    """Measure how many expected tokens appear in the candidate."""
    expected_tokens = set(normalize_search_text(expected).split())
    actual_tokens = set(normalize_search_text(actual).split())
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & actual_tokens) / len(expected_tokens)


def score_candidate(song_name, artist_name, candidate):
    """Rank Deezer candidates by title/artist agreement and preview availability."""
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
        and (expected_title_norm == candidate_title_norm
             or expected_title_norm in candidate_title_norm
             or candidate_title_norm in expected_title_norm)
    )
    artist_exactish = (
        expected_artist_norm
        and candidate_artist_norm
        and (expected_artist_norm == candidate_artist_norm
             or expected_artist_norm in candidate_artist_norm
             or candidate_artist_norm in expected_artist_norm)
    )

    score = 0.0
    score += 4.0 if title_exactish else 0.0
    score += 3.0 if artist_exactish else 0.0
    score += title_overlap * 3.0
    score += artist_overlap * 2.5
    score += 1.0 if candidate.get("preview") else -5.0

    return {
        "score": score,
        "title_overlap": title_overlap,
        "artist_overlap": artist_overlap,
        "title_exactish": title_exactish,
        "artist_exactish": artist_exactish,
    }


def is_confident_candidate(metrics, artist_name=""):
    """Reject loose search hits so we do not download the wrong song."""
    if not metrics:
        return False
    if metrics["title_exactish"] and (metrics["artist_exactish"] or metrics["artist_overlap"] >= 0.5):
        return True
    if metrics["title_overlap"] >= 0.7 and (not artist_name or metrics["artist_overlap"] >= 0.6):
        return True
    return False


def fetch_track_by_id(track_id):
    """Refresh a cached Deezer track to get a fresh preview URL."""
    if not track_id:
        return None
    try:
        response = requests.get(DEEZE_TRACK_URL.format(track_id), timeout=SEARCH_TIMEOUT_SEC)
        if response.status_code != 200:
            return None
        data = response.json()
        if not isinstance(data, dict) or data.get("error"):
            return None
        return data if data.get("preview") else None
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def run_search_query(query):
    """Run a Deezer search query and return the top candidate list."""
    if not query:
        return []
    try:
        response = requests.get(
            DEEZE_SEARCH_URL,
            params={"q": query, "limit": SEARCH_RESULT_LIMIT},
            timeout=SEARCH_TIMEOUT_SEC,
        )
        if response.status_code != 200:
            return []
        data = response.json()
        if not isinstance(data, dict):
            return []
        return data.get("data", []) or []
    except requests.exceptions.Timeout:
        return []
    except Exception:
        return []


def build_search_queries(song_name, artist_name):
    """Generate strict and fallback query variants for Deezer."""
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


def search_song(song_name, artist_name="", use_cache=True):
    """
    Search for a song on Deezer using song name + artist name.
    Uses caching to avoid redundant API calls.
    """
    # Check cache first
    cache_key = f"{song_name}|{artist_name}"
    if use_cache and cache_key in _search_cache:
        cached = _search_cache.get(cache_key)
        cached_track_id = None
        if isinstance(cached, dict):
            cached_track_id = cached.get("track_id") or cached.get("id")
        refreshed = fetch_track_by_id(cached_track_id)
        if refreshed:
            return refreshed
        _search_cache.pop(cache_key, None)

    candidates = {}
    for query in build_search_queries(song_name, artist_name):
        for candidate in run_search_query(query):
            track_id = candidate.get("id")
            if not track_id:
                continue
            metrics = score_candidate(song_name, artist_name, candidate)
            existing = candidates.get(track_id)
            if existing is None or metrics["score"] > existing["metrics"]["score"]:
                candidates[track_id] = {"song": candidate, "metrics": metrics}

    if not candidates:
        return None

    ranked = sorted(
        candidates.values(),
        key=lambda item: item["metrics"]["score"],
        reverse=True,
    )

    best = ranked[0]
    if not is_confident_candidate(best["metrics"], artist_name):
        return None

    if use_cache:
        _search_cache[cache_key] = {"track_id": best["song"].get("id")}
    return best["song"]


def download_preview(preview_url, output_file):
    """Download the audio preview from the given URL"""
    try:
        response = requests.get(preview_url, stream=True, timeout=PREVIEW_TIMEOUT_SEC)

        if response.status_code != 200:
            return False

        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return True
    except requests.exceptions.Timeout:
        return False
    except Exception as e:
        return False


def sanitize_filename(filename):
    # Remove or replace invalid characters for Windows filenames
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def checkpoint_song_key(song):
    """Build a stable deduplication key for checkpoint song records."""
    if not isinstance(song, dict):
        return ""

    msd_track_id = clean_optional_text(song.get("msd_track_id", ""))
    if msd_track_id:
        return f"msd:{msd_track_id}"

    deezer_track_id = clean_optional_text(song.get("deezer_track_id", ""))
    if deezer_track_id:
        return f"deezer:{deezer_track_id}"

    filename = clean_optional_text(song.get("filename", ""))
    if filename:
        return f"file:{filename.lower()}"

    artist = normalize_search_text(
        song.get("msd_artist")
        or song.get("deezer_artist")
        or song.get("artist")
        or ""
    )
    title = normalize_search_text(
        song.get("msd_title")
        or song.get("deezer_title")
        or song.get("title")
        or ""
    )
    if artist or title:
        return f"meta:{artist}|{title}"
    return ""


def dedupe_downloaded_songs(downloaded_songs):
    """Keep only the latest checkpoint entry for each song."""
    unique = {}
    ordered_keys = []
    for song in downloaded_songs or []:
        if not song:
            continue
        key = checkpoint_song_key(song)
        if not key:
            continue
        if key not in unique:
            ordered_keys.append(key)
        unique[key] = song
    return [unique[key] for key in ordered_keys]


def normalize_index_collection(values, max_index=None):
    """Normalize a list-like index collection to a clean integer set."""
    normalized = set()
    for value in values or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index < 0:
            continue
        if max_index is not None and index >= max_index:
            continue
        normalized.add(index)
    return normalized


def normalize_checkpoint_payload(checkpoint_data, max_index=None):
    """Normalize checkpoint fields and drop malformed entries."""
    checkpoint_data = checkpoint_data or {}
    processed_indices = normalize_index_collection(
        checkpoint_data.get("processed_indices", []),
        max_index=max_index,
    )
    failed_indices = normalize_index_collection(
        checkpoint_data.get("failed_indices", []),
        max_index=max_index,
    )
    failed_indices.difference_update(processed_indices)

    return {
        "processed_indices": sorted(processed_indices),
        "downloaded_songs": dedupe_downloaded_songs(checkpoint_data.get("downloaded_songs", [])),
        "failed_indices": sorted(failed_indices),
    }


def rebuild_download_registry(downloaded_songs):
    """Index existing downloads so retries do not overwrite unrelated songs."""
    global _download_registry_by_track_id, _download_registry_by_filename
    _download_registry_by_track_id = {}
    _download_registry_by_filename = {}

    for song in dedupe_downloaded_songs(downloaded_songs):
        filename = clean_optional_text(song.get("filename", ""))
        if not filename:
            continue

        msd_track_id = clean_optional_text(song.get("msd_track_id", ""))
        if msd_track_id:
            _download_registry_by_track_id[msd_track_id] = filename
            _download_registry_by_filename[filename.lower()] = msd_track_id
        else:
            _download_registry_by_filename[filename.lower()] = filename.lower()


def register_downloaded_song(song):
    """Register a downloaded song result for collision-free future saves."""
    if not song:
        return

    filename = clean_optional_text(song.get("filename", ""))
    if not filename:
        return

    msd_track_id = clean_optional_text(song.get("msd_track_id", ""))
    if msd_track_id:
        _download_registry_by_track_id[msd_track_id] = filename
        _download_registry_by_filename[filename.lower()] = msd_track_id
    else:
        _download_registry_by_filename[filename.lower()] = filename.lower()


def build_base_filename(song):
    """Create the legacy artist-title basename."""
    title = song.get("title", "Unknown")
    artist = song.get("artist", {}).get("name", "Unknown Artist")
    return sanitize_filename(f"{artist} - {title}")


def build_track_safe_filename(song, msd_track_id=None):
    """Create a deterministic filename that is unique per track."""
    base_filename = build_base_filename(song)
    safe_track_id = sanitize_filename(clean_optional_text(msd_track_id))
    if safe_track_id:
        return f"{base_filename} [{safe_track_id}].mp3"

    deezer_track_id = song.get("id")
    if deezer_track_id:
        return f"{base_filename} [deezer-{deezer_track_id}].mp3"

    return f"{base_filename}.mp3"


def choose_output_filename(song, msd_track_id=None):
    """Reuse the known filename for a track or pick a collision-safe new one."""
    track_id = clean_optional_text(msd_track_id)
    existing_filename = _download_registry_by_track_id.get(track_id) if track_id else None
    if existing_filename:
        return existing_filename

    legacy_filename = f"{build_base_filename(song)}.mp3"
    legacy_owner = _download_registry_by_filename.get(legacy_filename.lower())
    legacy_path = os.path.join(DEEZE_AUDIO_FOLDER, legacy_filename)

    if legacy_owner and (not track_id or legacy_owner == track_id):
        return legacy_filename

    if not legacy_owner and not os.path.exists(legacy_path):
        return legacy_filename

    return build_track_safe_filename(song, msd_track_id)


def upsert_downloaded_song(downloaded_songs, new_song):
    """Replace an existing checkpoint record for a song or append a new one."""
    new_key = checkpoint_song_key(new_song)
    if not new_key:
        downloaded_songs.append(new_song)
        return

    for index, existing_song in enumerate(downloaded_songs):
        if checkpoint_song_key(existing_song) == new_key:
            downloaded_songs[index] = new_song
            return

    downloaded_songs.append(new_song)


def download_and_save_song(song, artist_name="", msd_track_id=None):
    """Download and save a song preview, returning song metadata."""
    title = song.get('title', 'Unknown')
    artist = song.get('artist', {}).get('name', 'Unknown Artist')
    preview_url = song.get('preview', None)

    if not preview_url:
        return None

    # Ensure the output folder exists
    if not os.path.exists(DEEZE_AUDIO_FOLDER):
        os.makedirs(DEEZE_AUDIO_FOLDER)

    output_filename = choose_output_filename(song, msd_track_id=msd_track_id)
    output_file = os.path.join(DEEZE_AUDIO_FOLDER, output_filename)

    if os.path.exists(output_file):
        try:
            if os.path.getsize(output_file) > 0:
                return {
                    "title": title,
                    "artist": artist,
                    "filename": output_filename,
                    "deezer_track_id": song.get("id"),
                }
            os.remove(output_file)
        except OSError:
            pass

    success = download_preview(preview_url, output_file)

    if success:
        return {
            "title": title,
            "artist": artist,
            "filename": output_filename,
            "deezer_track_id": song.get("id"),
        }
    else:
        # Clean up partially downloaded file if it exists
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception:
                pass
        return None


def process_song(song_data):
    """Process a single song from CSV data"""
    try:
        # Add delay to avoid rate limiting (0.5s is safer for Deezer API)
        time.sleep(0.5)
        
        # Extract data from dict (for CSV) or parse string (for text file)
        if isinstance(song_data, dict):
            song_name = clean_optional_text(song_data.get('title', ''))
            artist_name = clean_optional_text(song_data.get('artist', ''))
            index = song_data.get('index', -1)
            msd_track_id = clean_optional_text(song_data.get('track_id', None))
        else:
            # Fallback for text file format
            song_line = song_data
            song_name = ""
            artist_name = ""
            index = -1
            msd_track_id = None
            
            if '–' in song_line:
                parts = song_line.strip().split('–')
                if len(parts) == 2:
                    song_name = parts[0].strip()
                    artist_name = parts[1].strip()
            elif '-' in song_line:
                parts = song_line.strip().split('-')
                if len(parts) == 2:
                    song_name = parts[0].strip()
                    artist_name = parts[1].strip()
            
            # If no separator or invalid format, treat entire line as song name
            if not song_name:
                song_name = song_line.strip()

        #  Add genre validation BEFORE search/download
        genre = clean_optional_text(song_data.get('genre', '')) if isinstance(song_data, dict) else ''
        
        if not genre or genre.lower() == 'unknown':
            return None

        if not song_name:
            return None

        # Search and download using simple song + artist search
        song = search_song(song_name, artist_name, use_cache=True)
        if not song:
            return None

        result = download_and_save_song(song, artist_name, msd_track_id=msd_track_id)
        if result:
            # Store both MSD (original) and Deezer (downloaded) info for the unified CSV
            result['genre'] = genre
            result['msd_artist'] = artist_name  # Original MSD artist used for search
            result['msd_title'] = song_name     # Original MSD title used for search
            result['msd_track_id'] = msd_track_id  # MSD track ID for direct matching
            result['deezer_artist'] = result.pop('artist')  # Rename to clarify source
            result['deezer_title'] = result.pop('title')    # Rename to clarify source
        return result

    except Exception as e:
        print(f"Error processing song: {str(e)}")
        return None


def load_checkpoint():
    """Load checkpoint data from file"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return normalize_checkpoint_payload(json.load(f))
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return {'processed_indices': [], 'downloaded_songs': [], 'failed_indices': []}


def save_checkpoint(checkpoint_data):
    """Save checkpoint data to file"""
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(normalize_checkpoint_payload(checkpoint_data), f, indent=2)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def reconcile_checkpoint_with_audio(checkpoint, songs_list):
    """
    Keep only checkpoint entries that still have audio files on disk.

    This prevents a stale checkpoint from skipping downloads after the
    `audio_files/` directory has been deleted or partially cleaned.
    """
    original_downloaded_count = len((checkpoint or {}).get('downloaded_songs', []))
    checkpoint = normalize_checkpoint_payload(checkpoint, max_index=len(songs_list))
    failed_indices = normalize_index_collection(
        checkpoint.get('failed_indices', []),
        max_index=len(songs_list),
    )

    duplicate_entries = original_downloaded_count - len(checkpoint.get('downloaded_songs', []))
    if duplicate_entries > 0:
        print(f"Checkpoint cleanup removed {duplicate_entries} duplicate download entries.")

    audio_dir = Path(DEEZE_AUDIO_FOLDER)
    if not audio_dir.exists():
        print("Audio directory is missing; ignoring previous checkpoint state.")
        return {'processed_indices': [], 'downloaded_songs': [], 'failed_indices': []}

    existing_basenames = {
        path.stem
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"}
    }
    if not existing_basenames:
        print("Audio directory is empty; ignoring previous checkpoint state.")
        return {'processed_indices': [], 'downloaded_songs': [], 'failed_indices': []}

    track_id_to_index = {
        song.get('track_id'): song.get('index', -1)
        for song in songs_list
        if song.get('track_id')
    }
    title_artist_to_index = {
        (song.get('title', '').strip().lower(), song.get('artist', '').strip().lower()): song.get('index', -1)
        for song in songs_list
    }

    valid_downloaded = []
    valid_processed = set()

    for song in checkpoint.get('downloaded_songs', []):
        filename = song.get('filename')
        if not filename or Path(filename).stem not in existing_basenames:
            continue

        valid_downloaded.append(song)
        msd_track_id = song.get('msd_track_id')
        idx = track_id_to_index.get(msd_track_id)
        if idx is None:
            lookup_key = (
                str(song.get('msd_title', '')).strip().lower(),
                str(song.get('msd_artist', '')).strip().lower(),
            )
            idx = title_artist_to_index.get(lookup_key)
        if idx is not None and idx >= 0:
            valid_processed.add(idx)

    dropped = len(checkpoint.get('downloaded_songs', [])) - len(valid_downloaded)
    if dropped > 0:
        print(
            f"Checkpoint/audio mismatch detected: dropped {dropped} stale checkpoint entries "
            f"that no longer exist in {DEEZE_AUDIO_FOLDER}/"
        )

    return {
        'processed_indices': sorted(valid_processed),
        'downloaded_songs': dedupe_downloaded_songs(valid_downloaded),
        'failed_indices': sorted(failed_indices.difference(valid_processed)),
    }


def create_download_statistics_graphs(total_songs, downloaded_count, failed_count, elapsed_time):
    """Create visualization graphs for download statistics"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "download_stats")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    downloaded_count = max(0, downloaded_count)
    failed_count = max(0, failed_count)
    not_found = max(0, total_songs - downloaded_count - failed_count)
    success_rate = (downloaded_count / total_songs * 100) if total_songs > 0 else 0
    failed_rate = (failed_count / total_songs * 100) if total_songs > 0 else 0
    not_found_rate = (not_found / total_songs * 100) if total_songs > 0 else 0
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Graph 1: Pie chart of download results
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sizes = [downloaded_count, failed_count, not_found]
    labels = [f'Downloaded\n({downloaded_count})', 
              f'Failed/No Preview\n({failed_count})', 
              f'Not Found\n({not_found})']
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, explode=explode, startangle=90,
                                        textprops={'fontsize': 11, 'weight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
    
    ax1.set_title(f'Download Results Summary\nTotal Songs: {total_songs}', 
                 fontsize=14, weight='bold', pad=20)
    
    fig1.tight_layout()
    pie_path = os.path.join(output_dir, f'download_pie_chart_{timestamp}.png')
    fig1.savefig(pie_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Pie chart saved: {pie_path}")
    plt.close(fig1)
    
    # Graph 2: Bar chart comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    categories = ['Downloaded', 'Failed/No Preview', 'Not Found']
    values = [downloaded_count, failed_count, not_found]
    percentages = [success_rate, failed_rate, not_found_rate]
    
    bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value, pct in zip(bars, values, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax2.set_ylabel('Number of Songs', fontsize=12, weight='bold')
    ax2.set_title('Download Statistics by Category', fontsize=14, weight='bold', pad=20)
    ax2.set_ylim(0, max(values) * 1.15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig2.tight_layout()
    bar_path = os.path.join(output_dir, f'download_bar_chart_{timestamp}.png')
    fig2.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Bar chart saved: {bar_path}")
    plt.close(fig2)
    
    # Graph 3: Summary statistics text visualization
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.axis('off')
    
    # Calculate minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    
    # Songs per minute
    songs_per_minute = (total_songs / elapsed_time * 60) if elapsed_time > 0 else 0
    
    summary_text = f"""
    DOWNLOAD SESSION SUMMARY
    {'='*50}
    
    Total Songs Processed:        {total_songs}
    Successfully Downloaded:      {downloaded_count} ({success_rate:.1f}%)
    Failed/No Preview Available:  {failed_count} ({failed_rate:.1f}%)
    Not Found on Deezer:          {not_found} ({not_found_rate:.1f}%)
    
    {'='*50}
    
    Session Duration:             {time_str}
    Average Speed:                {songs_per_minute:.1f} songs/min
    
    Output Files:
    • CSV Results: songs_data.csv
    • Checkpoint: download_checkpoint.json
    • Audio Files: audio_files/ folder
    """
    
    ax3.text(0.5, 0.5, summary_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig3.tight_layout()
    summary_path = os.path.join(output_dir, f'download_summary_{timestamp}.png')
    fig3.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Summary statistics saved: {summary_path}")
    plt.close(fig3)
    
    print(f"\n[INFO] All graphs saved to: {output_dir}")
    return output_dir


def update_unified_csv(downloaded_songs):
    """
    Update the unified songs.csv with download results.
    
    This updates existing rows with Deezer info (artist, title, filename)
    rather than creating a separate CSV.
    """
    import pandas as pd

    downloaded_songs = dedupe_downloaded_songs(downloaded_songs)
    
    if not downloaded_songs:
        print("No songs to update in unified CSV.")
        return
    
    if not os.path.exists(UNIFIED_CSV):
        print(f"Warning: Unified CSV not found at {UNIFIED_CSV}")
        print("Creating new unified CSV from download data...")
        df = pd.DataFrame(downloaded_songs)
        if "deezer_title" not in df.columns and "title" in df.columns:
            df["deezer_title"] = df["title"]
        if "deezer_artist" not in df.columns and "artist" in df.columns:
            df["deezer_artist"] = df["artist"]
        if "msd_track_id" not in df.columns:
            df["msd_track_id"] = ""
        if "msd_artist" not in df.columns:
            df["msd_artist"] = ""
        if "msd_title" not in df.columns:
            df["msd_title"] = ""
        df = ensure_unified_songs_schema(df)
        df["metadata_origin"] = "deezer_download_bootstrap"
        df["genre_source"] = np.where(df["genre"].astype(str).str.strip().ne(""), "deezer_download", "")
        df["audio_match_source"] = "deezer_download"
        df["has_audio"] = True
        df.to_csv(UNIFIED_CSV, index=False)
        return
    
    try:
        # Load existing unified CSV
        unified = ensure_unified_songs_schema(pd.read_csv(UNIFIED_CSV))
        
        def normalize(s):
            if pd.isna(s):
                return ""
            return str(s).lower().strip()
        
        updated_count = 0
        appended_count = 0
        bootstrap_rows = []
        for song in downloaded_songs:
            if song is None:
                continue
            
            # Match by MSD track_id if available, else by artist+title
            msd_track_id = song.get('msd_track_id')
            msd_artist = song.get('msd_artist', song.get('artist', ''))
            msd_title = song.get('msd_title', song.get('title', ''))
            
            if msd_track_id:
                mask = unified['msd_track_id'] == msd_track_id
            else:
                mask = (unified['msd_artist'].apply(normalize) == normalize(msd_artist)) & \
                       (unified['msd_title'].apply(normalize) == normalize(msd_title))
            
            if mask.any():
                idx = unified[mask].index[0]
                filename = str(song.get('filename', '') or '').strip()
                unified.loc[idx, 'deezer_artist'] = song.get('deezer_artist', song.get('artist'))
                unified.loc[idx, 'deezer_title'] = song.get('deezer_title', song.get('title'))
                unified.loc[idx, 'filename'] = filename
                unified.loc[idx, 'audio_basename'] = Path(filename).stem if filename else ''
                unified.loc[idx, 'audio_extension'] = Path(filename).suffix.lower() if filename else ''
                unified.loc[idx, 'has_audio'] = True
                if not str(unified.loc[idx, 'audio_match_source']).strip():
                    unified.loc[idx, 'audio_match_source'] = 'deezer_download'
                if not str(unified.loc[idx, 'metadata_origin']).strip():
                    unified.loc[idx, 'metadata_origin'] = 'deezer_download_bootstrap'
                updated_count += 1
                continue

            filename = str(song.get('filename', '') or '').strip()
            bootstrap_rows.append(
                {
                    'msd_track_id': str(song.get('msd_track_id', '') or '').strip(),
                    'msd_artist': str(song.get('msd_artist', song.get('artist', '')) or '').strip(),
                    'msd_title': str(song.get('msd_title', song.get('title', '')) or '').strip(),
                    'deezer_artist': str(song.get('deezer_artist', song.get('artist', '')) or '').strip(),
                    'deezer_title': str(song.get('deezer_title', song.get('title', '')) or '').strip(),
                    'filename': filename,
                    'audio_basename': Path(filename).stem if filename else '',
                    'audio_extension': Path(filename).suffix.lower() if filename else '',
                    'has_audio': True,
                    'genre': str(song.get('genre', '') or '').strip(),
                    'metadata_origin': 'deezer_download_bootstrap',
                    'genre_source': 'deezer_download' if str(song.get('genre', '') or '').strip() else '',
                    'audio_match_source': 'deezer_download',
                }
            )
            appended_count += 1

        if bootstrap_rows:
            unified = pd.concat([unified, pd.DataFrame(bootstrap_rows)], ignore_index=True)

        # Save updated unified CSV
        unified = ensure_unified_songs_schema(unified)
        unified.to_csv(UNIFIED_CSV, index=False)
        print(
            f"Updated {updated_count} songs and appended {appended_count} bootstrap rows "
            f"in unified CSV: {UNIFIED_CSV}"
        )
        
    except Exception as e:
        print(f"Error updating unified CSV: {str(e)}")
        # Fallback to old method
        save_songs_to_csv_legacy(downloaded_songs, "data/songs_data_with_genre.csv")


def save_songs_to_csv_legacy(songs_data, filename):
    """Legacy method: Save the songs data to a separate CSV file"""
    # Filter out None values
    songs_data = [song for song in songs_data if song]

    if not songs_data:
        print("No song data to save.")
        return

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'artist', 'filename', 'genre']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for song in songs_data:
                # Map to legacy format
                row = {
                    'title': song.get('deezer_title', song.get('title')),
                    'artist': song.get('deezer_artist', song.get('artist')),
                    'filename': song.get('filename'),
                    'genre': song.get('genre')
                }
                writer.writerow(row)

        print(f"Successfully saved {len(songs_data)} songs to {filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Download song previews from Deezer")
    parser.add_argument("--limit", type=int, help="Limit number of songs to download this session")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry songs that were previously marked as failed instead of taking new rows.",
    )
    args = parser.parse_args()

    start_time = time.time()
    
    # Load search cache from previous runs
    load_search_cache()
    if _search_cache:
        print(f"[CACHE] Loaded {len(_search_cache)} cached search results\n")
    
    # Use path relative to project root (since we chdir'd there)
    csv_path = INPUT_CSV
    
    # Check if Million Song CSV exists, fallback to names.txt
    if os.path.exists(csv_path):
        print(f"Loading songs from {INPUT_CSV}...")
        songs_list = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for idx, row in enumerate(reader):
                    row = normalize_input_row(row)
                    row['index'] = idx
                    songs_list.append(row)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
    else:
        # Fallback to old text file format
        file_path = "config/names.txt"
        print(f"CSV not found, falling back to {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"Error: Neither '{INPUT_CSV}' nor '{file_path}' found.")
            return

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if not lines:
            print(f"Error: '{file_path}' is empty.")
            return
        
        # Filter out empty lines and convert to dict format
        songs_list = [{'title': line.strip(), 'artist': '', 'index': idx} 
                     for idx, line in enumerate(lines) if line.strip()]

    # Load checkpoint
    checkpoint = load_checkpoint()
    checkpoint = reconcile_checkpoint_with_audio(checkpoint, songs_list)
    processed_indices = set(checkpoint.get('processed_indices', []))
    failed_indices = set(checkpoint.get('failed_indices', []))
    rebuild_download_registry(checkpoint.get('downloaded_songs', []))

    if args.retry_failed:
        songs_to_process = [
            song for song in songs_list
            if song.get('index', -1) in failed_indices
        ]
    else:
        songs_to_process = [
            song for song in songs_list
            if song.get('index', -1) not in processed_indices
            and song.get('index', -1) not in failed_indices
        ]

    # Apply limit if specified
    if args.limit:
        print(f"[INFO] Limit applied: Processing only first {args.limit} available songs")
        songs_to_process = songs_to_process[:args.limit]
    
    total_count = len(songs_list)
    already_processed = len(processed_indices)
    already_failed = len(failed_indices)
    remaining = len(songs_to_process)
    
    print(f"\nTotal songs: {total_count}")
    print(f"Already downloaded: {already_processed}")
    print(f"Already marked failed: {already_failed}")
    print(f"Remaining to process: {remaining}")
    
    if remaining == 0:
        if args.retry_failed:
            print("No failed songs queued for retry.")
        elif already_failed > 0:
            print(
                f"All new songs are finalized. Run again with --retry-failed to retry "
                f"{already_failed} failed songs."
            )
        else:
            print("All songs already processed!")
        return

    # Ensure the output folder exists
    os.makedirs(DEEZE_AUDIO_FOLDER, exist_ok=True)

    # Determine number of worker threads (reduced to 8 to avoid API rate limits)
    num_workers = min(8, remaining)
    print(f"Starting parallel processing with {num_workers} workers...\n")

    songs_data = dedupe_downloaded_songs(checkpoint.get('downloaded_songs', []))
    total_completed = 0
    current_failed_indices = set(failed_indices)
    
    # Track which songs failed in the current run (separate from checkpoint)
    failed_indices_this_run = set()
    
    # Run download attempts - retry as long as we get successful downloads
    attempt = 1
    while True:
        if attempt > 1:
            # On retry attempts, only process songs that failed in this run
            print(f"\n{'='*60}")
            print(f"RETRY ATTEMPT {attempt - 1} - Re-processing failed songs")
            print(f"{'='*60}\n")
            
            # Filter to only songs that failed (not in processed_indices but in failed_indices_this_run)
            songs_to_process = [song for song in songs_list 
                               if song.get('index', -1) in failed_indices_this_run]
            
            if not songs_to_process:
                print("No more songs to retry!")
                break
            
            print(f"Retrying {len(songs_to_process)} previously failed songs...\n")
        
        completed_in_attempt = 0
        failed_in_attempt = 0
        failed_indices_this_attempt = set()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_song = {executor.submit(
                process_song, song): song for song in songs_to_process}

            # Process results as they complete with progress bar
            pbar = tqdm(as_completed(future_to_song), total=len(future_to_song), 
                        desc=f"Downloading (attempt {attempt})", unit="song")
            for future in pbar:
                song_data = future_to_song[future]
                song_index = song_data.get('index', -1)
                
                try:
                    result = future.result(timeout=30)
                    
                    if result:
                        # Success - mark as processed and remove from failed set
                        processed_indices.add(song_index)
                        current_failed_indices.discard(song_index)
                        failed_indices_this_run.discard(song_index)
                        register_downloaded_song(result)
                        upsert_downloaded_song(songs_data, result)
                        completed_in_attempt += 1
                        total_completed += 1
                    else:
                        # Failed - track for retry but don't mark as processed
                        failed_indices_this_attempt.add(song_index)
                        failed_in_attempt += 1
                    
                    # Update progress bar description
                    pbar.set_postfix(success=completed_in_attempt, failed=failed_in_attempt)
                    
                    # Save checkpoint every 10 successful downloads
                    if total_completed % 10 == 0 and total_completed > 0:
                        save_checkpoint(
                            {
                                'processed_indices': list(processed_indices),
                                'downloaded_songs': songs_data,
                                'failed_indices': list(current_failed_indices),
                            }
                        )
                except Exception as e:
                    # Exception - track for retry but don't mark as processed
                    failed_indices_this_attempt.add(song_index)
                    failed_in_attempt += 1
                    pbar.set_postfix(success=completed_in_attempt, failed=failed_in_attempt)
        
        # Keep only the songs that still failed after this attempt.
        failed_indices_this_run = failed_indices_this_attempt
        
        print(f"\nAttempt {attempt} complete: {completed_in_attempt} success, {failed_in_attempt} failed")
        
        # Stop if no successes in this attempt (no point retrying)
        if completed_in_attempt == 0:
            print("No successful downloads in this attempt - stopping retries.")
            break
        
        # Stop if no failures to retry
        if not failed_indices_this_run:
            print("All songs downloaded successfully!")
            break
        
        # Continue to next retry with exponential backoff
        attempt += 1
        backoff_time = 5 * (2 ** (attempt - 2))  # 5s, 10s, 20s, 40s...
        backoff_time = min(backoff_time, 60)  # Cap at 60 seconds
        print(f"\nWaiting {backoff_time} seconds before retry (exponential backoff)...\n")
        time.sleep(backoff_time)

    current_failed_indices.update(failed_indices_this_run)
    current_failed_indices.difference_update(processed_indices)
    songs_data = dedupe_downloaded_songs(songs_data)

    # Final checkpoint save
    save_checkpoint(
        {
            'processed_indices': list(processed_indices),
            'downloaded_songs': songs_data,
            'failed_indices': list(current_failed_indices),
        }
    )

    # Count successful downloads
    success_count = len(songs_data)
    finalized_count = len(processed_indices) + len(current_failed_indices)
    remaining_unattempted = max(0, total_count - finalized_count)

    # Update unified CSV with download results
    update_unified_csv(songs_data)
    save_songs_to_csv_legacy(songs_data, "data/songs_data_with_genre.csv")

    elapsed = time.time() - start_time
    print(
        f"\nSession completed: {total_completed} downloaded, "
        f"{len(failed_indices_this_run)} still failed in this run"
    )
    print(f"Total successful downloads so far: {success_count} of {total_count} songs")
    print(f"Session time: {elapsed:.2f} seconds")
    
    # Save search cache for future runs
    save_search_cache()
    print(f"\n[CACHE] Saved {len(_search_cache)} search results to cache for next run")
    
    # Create visualization graphs
    print("\n" + "="*50)
    print("Generating download statistics graphs...")
    print("="*50)
    
    # Calculate total failed (including previous sessions)
    total_failed = len(current_failed_indices)
    create_download_statistics_graphs(total_count, success_count, total_failed, elapsed)
    
    if remaining_unattempted > 0:
        print(f"\nNote: {remaining_unattempted} songs remaining. Run script again to continue.")
    elif current_failed_indices and not args.retry_failed:
        print(
            f"\nFirst pass complete. {len(current_failed_indices)} songs are marked failed. "
            "Run again with --retry-failed to retry only those rows."
        )
    else:
        print(f"\nAll songs processed! You can delete '{CHECKPOINT_FILE}' to start fresh.")
    print(f"Song data saved to {UNIFIED_CSV}")


if __name__ == "__main__":
    main()
