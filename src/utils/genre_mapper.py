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

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Unified CSV path
UNIFIED_CSV = os.path.join("data", "songs.csv")


def load_genre_mapping(csv_path: str = None) -> Dict[str, List[str]]:
    """
    Load genre mapping from unified songs.csv.
    
    Args:
        csv_path: Path to songs.csv (or legacy songs_data_with_genre.csv)
        
    Returns:
        Dictionary mapping filename to list of genres
        Example: {'Artist - Title.mp3': ['hip hop', 'underground rap', 'g funk']}
    """
    genre_mapping = {}
    
    # Use unified CSV by default
    if csv_path is None:
        csv_path = UNIFIED_CSV
    
    # Also check legacy path for backward compatibility
    if not os.path.exists(csv_path):
        legacy_path = os.path.join("data", "songs_data_with_genre.csv")
        if os.path.exists(legacy_path):
            print(f"Note: Using legacy CSV at {legacy_path}")
            csv_path = legacy_path
        else:
            print(f"Warning: {csv_path} not found. Returning empty mapping.")
            return genre_mapping
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', '').strip()
                genre_str = row.get('genre', '').strip()
                
                if filename and genre_str:
                    # Split comma-separated genres and clean whitespace
                    genres = [g.strip() for g in genre_str.split(',') if g.strip()]
                    genre_mapping[filename] = genres
                    
        print(f"✓ Loaded genre mapping for {len(genre_mapping)} songs from {csv_path}")
        
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
    
    print(f"✓ Multi-label encoding: {genre_matrix.shape[0]} songs × {genre_matrix.shape[1]} genres")
    
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
    
    print(f"✓ Genre map created for {len(genre_map)} audio files")
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
        
        print("\n✓ Genre mapper test complete!")
    else:
        print(f"\nError: Could not load genre mapping from {csv_path}")
