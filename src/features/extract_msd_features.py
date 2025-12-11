"""
Extract metadata features from Million Song Dataset (MSD) HDF5 files.

Features extracted:
  - Key: 12-dim one-hot (0=C, 1=C#, ..., 11=B)
  - Mode: 2-dim one-hot (major/minor)
  - Loudness: 1-dim (normalized to 0-1 range)
  - Tempo: 1-dim (BPM, normalized)

Total: 16 dimensions

Note: Time signature, segments_timbre aggregates, and segments_pitches aggregates
are intentionally excluded to keep only the most reliable metadata features.
"""

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Feature dimensions
KEY_DIM = 12          # One-hot encoding
MODE_DIM = 2          # Major/Minor one-hot
LOUDNESS_DIM = 1      # Normalized loudness
TEMPO_DIM = 1         # BPM

TOTAL_MSD_DIM = KEY_DIM + MODE_DIM + LOUDNESS_DIM + TEMPO_DIM  # 16 dimensions


def extract_single_msd_file(h5_path: str) -> dict:
    """
    Extract metadata features from a single MSD HDF5 file.
    
    Args:
        h5_path: Path to .h5 file
        
    Returns:
        Dictionary with track_id, artist, title, and feature arrays
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get metadata
            metadata = f['metadata']['songs'][0]
            analysis = f['analysis']['songs'][0]
            
            # track_id is in analysis section
            track_id = analysis['track_id'].decode('utf-8') if isinstance(analysis['track_id'], bytes) else str(analysis['track_id'])
            artist = metadata['artist_name'].decode('utf-8') if isinstance(metadata['artist_name'], bytes) else str(metadata['artist_name'])
            title = metadata['title'].decode('utf-8') if isinstance(metadata['title'], bytes) else str(metadata['title'])
            
            # Extract features
            key = int(analysis['key'])  # 0-11
            mode = int(analysis['mode'])  # 0=minor, 1=major
            loudness = float(analysis['loudness'])  # typically -60 to 0 dB
            tempo = float(analysis['tempo'])  # BPM
            
            # Key confidence (optional, for filtering unreliable keys)
            key_confidence = float(analysis['key_confidence'])
            mode_confidence = float(analysis['mode_confidence'])
            
            return {
                'track_id': track_id,
                'artist': artist,
                'title': title,
                'key': key,
                'mode': mode,
                'loudness': loudness,
                'tempo': tempo,
                'key_confidence': key_confidence,
                'mode_confidence': mode_confidence,
                'h5_path': h5_path
            }
    except Exception as e:
        return {'error': str(e), 'h5_path': h5_path}


def find_all_h5_files(msd_root: str) -> list:
    """
    Recursively find all .h5 files in MSD directory.
    
    Args:
        msd_root: Root directory of Million Song Dataset
        
    Returns:
        List of paths to .h5 files
    """
    h5_files = []
    for root, dirs, files in os.walk(msd_root):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files


def process_all_msd_files(msd_root: str, output_dir: str = None, n_workers: int = 8) -> pd.DataFrame:
    """
    Extract features from all MSD files in parallel and update unified CSV.
    
    Args:
        msd_root: Root directory of MSD subset
        output_dir: Directory to save output files (default: data/)
        n_workers: Number of parallel workers
        
    Returns:
        DataFrame with all extracted features
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all H5 files
    print(f"Scanning for .h5 files in {msd_root}...")
    h5_files = find_all_h5_files(msd_root)
    print(f"Found {len(h5_files)} .h5 files")
    
    # Process in parallel
    results = []
    errors = []
    
    print(f"Extracting features using {n_workers} workers...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(extract_single_msd_file, path): path for path in h5_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing MSD files"):
            result = future.result()
            if 'error' in result:
                errors.append(result)
            else:
                results.append(result)
    
    print(f"Successfully extracted: {len(results)}")
    print(f"Errors: {len(errors)}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Update unified songs.csv if it exists
    unified_path = os.path.join(output_dir, 'songs.csv')
    if os.path.exists(unified_path):
        update_unified_csv_with_msd_features(df, unified_path)
    else:
        # Create initial unified CSV from MSD features
        print(f"Creating initial unified CSV at {unified_path}")
        create_initial_unified_csv(df, unified_path)
    
    return df


def update_unified_csv_with_msd_features(msd_df: pd.DataFrame, unified_path: str):
    """Update existing unified CSV with MSD features."""
    unified = pd.read_csv(unified_path)
    
    # Create lookup by track_id
    msd_lookup = msd_df.set_index('track_id').to_dict('index')
    
    updated = 0
    for idx, row in unified.iterrows():
        track_id = row.get('msd_track_id')
        if pd.notna(track_id) and track_id in msd_lookup:
            msd_data = msd_lookup[track_id]
            unified.loc[idx, 'key'] = msd_data['key']
            unified.loc[idx, 'mode'] = msd_data['mode']
            unified.loc[idx, 'loudness'] = msd_data['loudness']
            unified.loc[idx, 'tempo'] = msd_data['tempo']
            unified.loc[idx, 'key_confidence'] = msd_data['key_confidence']
            unified.loc[idx, 'mode_confidence'] = msd_data['mode_confidence']
            updated += 1
    
    unified.to_csv(unified_path, index=False)
    print(f"Updated {updated} rows in unified CSV with MSD features")


def create_initial_unified_csv(msd_df: pd.DataFrame, unified_path: str):
    """Create initial unified CSV from MSD features."""
    unified = pd.DataFrame({
        'msd_track_id': msd_df['track_id'],
        'msd_artist': msd_df['artist'],
        'msd_title': msd_df['title'],
        'deezer_artist': None,
        'deezer_title': None,
        'filename': None,
        'genre': None,
        'key': msd_df['key'],
        'mode': msd_df['mode'],
        'loudness': msd_df['loudness'],
        'tempo': msd_df['tempo'],
        'key_confidence': msd_df['key_confidence'],
        'mode_confidence': msd_df['mode_confidence'],
        'has_audio': False
    })
    unified.to_csv(unified_path, index=False)
    print(f"Created initial unified CSV with {len(unified)} MSD tracks")


def create_feature_vectors(df: pd.DataFrame, normalize: bool = True) -> tuple:
    """
    Convert raw features to feature vectors suitable for clustering.
    
    Args:
        df: DataFrame from process_all_msd_files
        normalize: Whether to normalize features
        
    Returns:
        Tuple of (feature_matrix, track_ids)
    """
    n_tracks = len(df)
    feature_matrix = np.zeros((n_tracks, TOTAL_MSD_DIM), dtype=np.float32)
    
    # Normalization parameters (computed from data or use reasonable defaults)
    if normalize:
        # Loudness: typically -60 to 0 dB, normalize to 0-1
        loudness_min = df['loudness'].min()
        loudness_max = df['loudness'].max()
        
        # Tempo: typically 50-200 BPM, normalize to 0-1
        tempo_min = df['tempo'].min()
        tempo_max = df['tempo'].max()
    
    for i, row in df.iterrows():
        idx = 0
        
        # Key: 12-dim one-hot
        key_onehot = np.zeros(KEY_DIM, dtype=np.float32)
        key = int(row['key'])
        if 0 <= key < 12:
            key_onehot[key] = 1.0
        feature_matrix[i, idx:idx+KEY_DIM] = key_onehot
        idx += KEY_DIM
        
        # Mode: 2-dim one-hot (minor=0, major=1)
        mode_onehot = np.zeros(MODE_DIM, dtype=np.float32)
        mode = int(row['mode'])
        if mode == 0:  # minor
            mode_onehot[0] = 1.0
        else:  # major
            mode_onehot[1] = 1.0
        feature_matrix[i, idx:idx+MODE_DIM] = mode_onehot
        idx += MODE_DIM
        
        # Loudness: normalized
        if normalize:
            loudness_norm = (row['loudness'] - loudness_min) / (loudness_max - loudness_min + 1e-8)
        else:
            loudness_norm = row['loudness']
        feature_matrix[i, idx] = loudness_norm
        idx += LOUDNESS_DIM
        
        # Tempo: normalized
        if normalize:
            tempo_norm = (row['tempo'] - tempo_min) / (tempo_max - tempo_min + 1e-8)
        else:
            tempo_norm = row['tempo']
        feature_matrix[i, idx] = tempo_norm
        idx += TEMPO_DIM
    
    return feature_matrix, df['track_id'].tolist() if 'track_id' in df.columns else df['msd_track_id'].tolist()


def load_msd_features(data_dir: str = None) -> tuple:
    """
    Load MSD features from unified songs.csv.
    
    Args:
        data_dir: Directory containing songs.csv
        
    Returns:
        Tuple of (feature_df, feature_matrix, track_ids)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    
    # Try unified CSV first
    unified_path = os.path.join(data_dir, 'songs.csv')
    if os.path.exists(unified_path):
        df = pd.read_csv(unified_path)
        # Rename columns to match expected format
        df = df.rename(columns={'msd_track_id': 'track_id'})
    else:
        # Fallback to legacy msd_features.csv
        csv_path = os.path.join(data_dir, 'msd_features.csv')
        df = pd.read_csv(csv_path)
    
    feature_matrix, track_ids = create_feature_vectors(df)
    
    return df, feature_matrix, track_ids


def get_msd_feature_vector(track_id: str, msd_df: pd.DataFrame, feature_matrix: np.ndarray, track_ids: list) -> np.ndarray:
    """
    Get MSD feature vector for a specific track.
    
    Args:
        track_id: MSD track ID
        msd_df: DataFrame with MSD features
        feature_matrix: Pre-computed feature matrix
        track_ids: List of track IDs corresponding to feature_matrix rows
        
    Returns:
        Feature vector (16 dimensions) or None if not found
    """
    try:
        idx = track_ids.index(track_id)
        return feature_matrix[idx]
    except ValueError:
        return None


def get_msd_features_by_name(artist: str, title: str, msd_df: pd.DataFrame, 
                              feature_matrix: np.ndarray, track_ids: list) -> np.ndarray:
    """
    Get MSD features by artist and title match.
    
    Args:
        artist: Artist name to match
        title: Song title to match
        msd_df: DataFrame with MSD features
        feature_matrix: Pre-computed feature matrix
        track_ids: List of track IDs
        
    Returns:
        Feature vector (16 dims) or None if not found
    """
    # Normalize for matching
    artist_lower = artist.lower().strip()
    title_lower = title.lower().strip()
    
    # Try exact match first
    mask = (msd_df['artist'].str.lower().str.strip() == artist_lower) & \
           (msd_df['title'].str.lower().str.strip() == title_lower)
    
    matches = msd_df[mask]
    if len(matches) > 0:
        idx = matches.index[0]
        return feature_matrix[idx]
    
    return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract MSD metadata features')
    parser.add_argument('--msd-root', type=str, 
                        default='data/millionsongsubset_extracted/MillionSongSubset',
                        help='Root directory of Million Song Dataset')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for feature files')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Extract all features
    df = process_all_msd_files(args.msd_root, args.output_dir, args.workers)
    
    # Create and save feature vectors
    feature_matrix, track_ids = create_feature_vectors(df)
    
    npz_path = os.path.join(args.output_dir, 'msd_feature_vectors.npz')
    np.savez(npz_path, features=feature_matrix, track_ids=track_ids)
    print(f"Saved feature vectors to {npz_path}")
    print(f"Feature matrix shape: {feature_matrix.shape}")
