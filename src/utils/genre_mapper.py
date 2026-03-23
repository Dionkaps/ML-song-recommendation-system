"""
Genre Mapper Utility
Handles multi-label genre mapping from unified songs.csv to audio files.
"""

import os
import csv
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, List, Tuple
import sys

from src.utils.genre_taxonomy import (
    PRIMARY_TAGS_COLUMN,
    SECONDARY_TAGS_COLUMN,
    coerce_bool,
    load_taxonomy_songs,
    resolve_pipeline_songs_csv,
    split_tag_list,
)

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Unified CSV path
UNIFIED_CSV = os.path.join("data", "songs.csv")


def load_genre_mapping(csv_path: str = None) -> Dict[str, List[str]]:
    """
    Load taxonomy-aware genre mapping from the active songs CSV.
    
    Args:
        csv_path: Path to a songs CSV. When pointed at the canonical
            ``data/songs.csv`` source, the merged-genre assignment CSV is
            auto-generated and used instead.
        
    Returns:
        Dictionary mapping filename to list of genres
        Example: {'Artist - Title.mp3': ['rock', 'alternative_indie_rock', 'non_genre_region_scene']}
    """
    genre_mapping = {}

    try:
        resolved_csv_path = resolve_pipeline_songs_csv(csv_path or UNIFIED_CSV)

        if resolved_csv_path.name == "songs_with_merged_genres.csv" or resolved_csv_path.name == "songs.csv":
            unified = load_taxonomy_songs(
                csv_path=str(resolved_csv_path),
                audio_only=True,
                eligible_only=True,
            )
            for _, row in unified.iterrows():
                genre_str = str(row.get("genre", "")).strip()
                if not genre_str:
                    continue
                key = str(row.get("filename", "")).strip() or str(
                    row.get("audio_basename", "")
                ).strip()
                if not key or key in genre_mapping:
                    continue
                genres = split_tag_list(genre_str, delimiter=",")
                if genres:
                    genre_mapping[key] = genres
        else:
            with open(resolved_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('filename', '').strip()
                    genre_str = row.get('genre', '').strip()
                    include_in_mrs_raw = row.get("include_in_mrs", "")

                    if filename and genre_str and (
                        include_in_mrs_raw == "" or coerce_bool(include_in_mrs_raw)
                    ):
                        genres = split_tag_list(genre_str, delimiter=",")
                        genre_mapping[filename] = genres

        print(
            f"Loaded genre mapping for {len(genre_mapping)} songs from "
            f"{resolved_csv_path}"
        )
        
        # Print genre statistics
        all_genres = set()
        for genres in genre_mapping.values():
            all_genres.update(genres)
        print(f"  - Total unique genres: {len(all_genres)}")
        print(f"  - Average genres per song: {np.mean([len(g) for g in genre_mapping.values()]):.2f}")
        
    except Exception as e:
        print(f"Error loading genre mapping: {e}")
    
    return genre_mapping


def get_multi_label_encoding(genre_mapping: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str], MultiLabelBinarizer]:
    """
    Create multi-label binary encoding for genres.
    
    Args:
        genre_mapping: Dictionary from load_genre_mapping()
        
    Returns:
        - Binary encoded array (n_songs, n_genres)
        - List of all unique genre labels
        - Fitted MultiLabelBinarizer instance
    """
    filenames = list(genre_mapping.keys())
    genre_lists = [genre_mapping[f] for f in filenames]
    
    # Fit and transform
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genre_lists)
    
    print(f"Loaded multi-label encoding: {genre_matrix.shape[0]} songs x {genre_matrix.shape[1]} genres")
    
    return genre_matrix, mlb.classes_.tolist(), mlb


def get_primary_genre(filename: str, genre_mapping: Dict[str, List[str]]) -> str:
    """
    Get the primary (first) genre for a song.
    Used for backward compatibility and single-label evaluation metrics.
    
    Args:
        filename: Audio filename (e.g., 'Artist - Title.mp3')
        genre_mapping: Dictionary from load_genre_mapping()
        
    Returns:
        Primary genre string, or 'unknown' if not found
    """
    genres = genre_mapping.get(filename, [])
    return genres[0] if genres else 'unknown'


def load_secondary_tag_mapping(csv_path: str = None) -> Dict[str, List[str]]:
    """Load secondary ``non_genre_*`` tags for eligible audio-backed songs."""

    secondary_mapping: Dict[str, List[str]] = {}
    resolved_csv_path = resolve_pipeline_songs_csv(csv_path or UNIFIED_CSV)
    unified = load_taxonomy_songs(
        csv_path=str(resolved_csv_path),
        audio_only=True,
        eligible_only=True,
    )
    for _, row in unified.iterrows():
        key = str(row.get("filename", "")).strip() or str(row.get("audio_basename", "")).strip()
        if not key or key in secondary_mapping:
            continue
        secondary_mapping[key] = split_tag_list(row.get(SECONDARY_TAGS_COLUMN, ""), delimiter=",")
    return secondary_mapping


def load_primary_tag_mapping(csv_path: str = None) -> Dict[str, List[str]]:
    """Load multi-label primary taxonomy tags for eligible audio-backed songs."""

    primary_mapping: Dict[str, List[str]] = {}
    resolved_csv_path = resolve_pipeline_songs_csv(csv_path or UNIFIED_CSV)
    unified = load_taxonomy_songs(
        csv_path=str(resolved_csv_path),
        audio_only=True,
        eligible_only=True,
    )
    for _, row in unified.iterrows():
        key = str(row.get("filename", "")).strip() or str(row.get("audio_basename", "")).strip()
        if not key or key in primary_mapping:
            continue
        tags = split_tag_list(row.get(PRIMARY_TAGS_COLUMN, ""), delimiter=",")
        primary_mapping[key] = tags
    return primary_mapping


def create_genre_map_for_audio_dir(audio_dir: str, genre_mapping: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Create a simple genre map (filename -> primary genre) for files in audio_dir.
    This is for backward compatibility with existing code that expects single genres.
    
    Args:
        audio_dir: Path to directory containing audio files
        genre_mapping: Dictionary from load_genre_mapping()
        
    Returns:
        Dictionary mapping audio file paths to primary genres
    """
    genre_map = {}
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a'}
    
    if not os.path.exists(audio_dir):
        print(f"Warning: Audio directory {audio_dir} not found")
        return genre_map
    
    for filename in os.listdir(audio_dir):
        if Path(filename).suffix.lower() in audio_extensions:
            filepath = os.path.join(audio_dir, filename)
            primary_genre = get_primary_genre(filename, genre_mapping)
            genre_map[filepath] = primary_genre
    
    # Statistics
    known_count = sum(1 for g in genre_map.values() if g != 'unknown')
    unknown_count = len(genre_map) - known_count
    
    print(f"Genre map created for {len(genre_map)} audio files")
    print(f"  - Known genres: {known_count}")
    print(f"  - Unknown: {unknown_count}")
    
    return genre_map


def get_genre_statistics(genre_mapping: Dict[str, List[str]]) -> Dict:
    """
    Calculate statistics about the genre distribution.
    
    Args:
        genre_mapping: Dictionary from load_genre_mapping()
        
    Returns:
        Dictionary with various statistics
    """
    if not genre_mapping:
        return {}
    
    all_genres = []
    for genres in genre_mapping.values():
        all_genres.extend(genres)
    
    from collections import Counter
    genre_counts = Counter(all_genres)
    
    stats = {
        'total_songs': len(genre_mapping),
        'unique_genres': len(genre_counts),
        'avg_genres_per_song': np.mean([len(g) for g in genre_mapping.values()]),
        'max_genres_per_song': max(len(g) for g in genre_mapping.values()),
        'min_genres_per_song': min(len(g) for g in genre_mapping.values()),
        'most_common_genres': genre_counts.most_common(10),
    }
    
    return stats


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("Genre Mapper Utility - Test")
    print("=" * 60)
    
    # Load mapping (will use unified songs.csv by default)
    print("\n1. Loading genre mapping...")
    genre_mapping = load_genre_mapping()
    
    if genre_mapping:
        # Show sample
        print("\n2. Sample entries:")
        for filename, genres in list(genre_mapping.items())[:3]:
            print(f"   {filename}: {genres}")
        
        # Multi-label encoding
        print("\n3. Creating multi-label encoding...")
        genre_matrix, genre_labels, mlb = get_multi_label_encoding(genre_mapping)
        print(f"   Genre labels (first 10): {genre_labels[:10]}")
        
        # Statistics
        print("\n4. Genre statistics:")
        stats = get_genre_statistics(genre_mapping)
        for key, value in stats.items():
            if key != 'most_common_genres':
                print(f"   {key}: {value}")
        print(f"   Most common genres:")
        for genre, count in stats['most_common_genres']:
            print(f"      {genre}: {count}")
        
        # Test audio dir mapping
        print("\n5. Testing audio directory mapping...")
        audio_dir = 'audio_files'
        genre_map = create_genre_map_for_audio_dir(audio_dir, genre_mapping)
        
        print("\nGenre mapper test complete!")
    else:
        print(f"\nError: Could not load genre mapping from {csv_path}")
