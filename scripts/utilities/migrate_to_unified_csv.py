"""
Migration script to create unified songs.csv from existing separate CSVs.

This merges:
- millionsong_dataset.csv (MSD title, artist, genre)
- songs_data_with_genre.csv (Deezer title, artist, filename, genre)
- msd_features.csv (track_id, key, mode, loudness, tempo)

Into a single unified songs.csv with all information.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def migrate_to_unified_csv(project_root: str = None):
    """
    Migrate existing CSVs to unified songs.csv format.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    
    data_dir = os.path.join(project_root, 'data')
    
    # Load existing CSVs
    msd_input = os.path.join(data_dir, 'millionsong_dataset.csv')
    songs_data = os.path.join(data_dir, 'songs_data_with_genre.csv')
    msd_features = os.path.join(data_dir, 'msd_features.csv')
    
    print("Loading existing CSVs...")
    
    # Load MSD input (original dataset)
    msd_df = pd.read_csv(msd_input)
    msd_df.columns = ['msd_title', 'msd_artist', 'genre']
    print(f"  MSD input: {len(msd_df)} tracks")
    
    # Load downloaded songs data
    if os.path.exists(songs_data):
        downloaded_df = pd.read_csv(songs_data)
        downloaded_df.columns = ['deezer_title', 'deezer_artist', 'filename', 'genre_downloaded']
        print(f"  Downloaded songs: {len(downloaded_df)} tracks")
    else:
        downloaded_df = pd.DataFrame(columns=['deezer_title', 'deezer_artist', 'filename', 'genre_downloaded'])
        print("  Downloaded songs: 0 tracks (file not found)")
    
    # Load MSD features (extracted from HDF5)
    if os.path.exists(msd_features):
        features_df = pd.read_csv(msd_features)
        print(f"  MSD features: {len(features_df)} tracks")
    else:
        features_df = pd.DataFrame(columns=['track_id', 'artist', 'title', 'key', 'mode', 
                                            'loudness', 'tempo', 'key_confidence', 'mode_confidence'])
        print("  MSD features: 0 tracks (file not found)")
    
    # Create unified dataframe starting from MSD features (has track_id)
    print("\nCreating unified CSV...")
    
    # Start with MSD features as base (has track_id which is critical)
    unified = features_df[['track_id', 'artist', 'title', 'key', 'mode', 
                           'loudness', 'tempo', 'key_confidence', 'mode_confidence']].copy()
    unified.columns = ['msd_track_id', 'msd_artist', 'msd_title', 'key', 'mode',
                       'loudness', 'tempo', 'key_confidence', 'mode_confidence']
    
    # Add genre from MSD input (matching on artist+title)
    def normalize_for_match(s):
        if pd.isna(s):
            return ""
        return str(s).lower().strip()
    
    # Create match keys
    unified['match_key'] = unified.apply(
        lambda r: normalize_for_match(r['msd_artist']) + '|' + normalize_for_match(r['msd_title']), 
        axis=1
    )
    msd_df['match_key'] = msd_df.apply(
        lambda r: normalize_for_match(r['msd_artist']) + '|' + normalize_for_match(r['msd_title']), 
        axis=1
    )
    
    # Merge genre
    genre_map = dict(zip(msd_df['match_key'], msd_df['genre']))
    unified['genre'] = unified['match_key'].map(genre_map)
    
    # Add Deezer columns (will be filled during download)
    unified['deezer_artist'] = None
    unified['deezer_title'] = None
    unified['filename'] = None
    unified['has_audio'] = False
    
    # Match downloaded songs to MSD tracks
    if len(downloaded_df) > 0:
        # Create match key for downloaded songs
        downloaded_df['match_key'] = downloaded_df.apply(
            lambda r: normalize_for_match(r['deezer_artist']) + '|' + normalize_for_match(r['deezer_title']),
            axis=1
        )
        
        # Try to match downloaded songs to unified data
        # First, try exact match on normalized artist+title
        matched_count = 0
        audio_dir = os.path.join(project_root, 'audio_files')
        
        for idx, row in downloaded_df.iterrows():
            # Check if audio file exists
            filename = row['filename']
            file_exists = os.path.exists(os.path.join(audio_dir, filename))
            
            # Find matching row in unified (fuzzy match on artist+title)
            deezer_artist_norm = normalize_for_match(row['deezer_artist'])
            deezer_title_norm = normalize_for_match(row['deezer_title'])
            
            # Try exact match first
            mask = (unified['msd_artist'].apply(normalize_for_match) == deezer_artist_norm) & \
                   (unified['msd_title'].apply(normalize_for_match) == deezer_title_norm)
            
            matches = unified[mask]
            
            if len(matches) > 0:
                match_idx = matches.index[0]
                unified.loc[match_idx, 'deezer_artist'] = row['deezer_artist']
                unified.loc[match_idx, 'deezer_title'] = row['deezer_title']
                unified.loc[match_idx, 'filename'] = filename
                unified.loc[match_idx, 'has_audio'] = file_exists
                matched_count += 1
        
        print(f"  Matched {matched_count} downloaded songs to MSD tracks")
    
    # Also check for audio files that exist but weren't in downloaded list
    audio_dir = os.path.join(project_root, 'audio_files')
    if os.path.exists(audio_dir):
        existing_files = set(f for f in os.listdir(audio_dir) if f.endswith('.mp3'))
        matched_files = set(unified[unified['has_audio']]['filename'].dropna())
        unmatched_files = existing_files - matched_files
        
        if unmatched_files:
            print(f"  Found {len(unmatched_files)} audio files not yet matched")
            # Try to match these by parsing filename
            for filename in unmatched_files:
                # Parse "Artist - Title.mp3"
                name = filename.replace('.mp3', '')
                if ' - ' in name:
                    parts = name.split(' - ', 1)
                    file_artist = normalize_for_match(parts[0])
                    file_title = normalize_for_match(parts[1])
                    
                    mask = (unified['msd_artist'].apply(normalize_for_match) == file_artist) & \
                           (unified['msd_title'].apply(normalize_for_match) == file_title)
                    
                    matches = unified[mask]
                    if len(matches) > 0:
                        match_idx = matches.index[0]
                        if pd.isna(unified.loc[match_idx, 'filename']):
                            unified.loc[match_idx, 'deezer_artist'] = parts[0]
                            unified.loc[match_idx, 'deezer_title'] = parts[1]
                            unified.loc[match_idx, 'filename'] = filename
                            unified.loc[match_idx, 'has_audio'] = True
    
    # Drop the match_key column
    unified = unified.drop(columns=['match_key'])
    
    # Reorder columns for clarity
    column_order = [
        'msd_track_id', 'msd_artist', 'msd_title',  # MSD identification
        'deezer_artist', 'deezer_title', 'filename',  # Deezer download info
        'genre',  # Genre tags
        'key', 'mode', 'loudness', 'tempo',  # MSD features
        'key_confidence', 'mode_confidence',  # Confidence scores
        'has_audio'  # Status
    ]
    unified = unified[column_order]
    
    # Summary
    print("\n" + "=" * 60)
    print("UNIFIED CSV SUMMARY")
    print("=" * 60)
    print(f"Total MSD tracks: {len(unified)}")
    print(f"Tracks with audio: {unified['has_audio'].sum()}")
    print(f"Tracks with genre: {unified['genre'].notna().sum()}")
    print(f"Tracks with MSD features: {unified['key'].notna().sum()}")
    
    # Save unified CSV
    output_path = os.path.join(data_dir, 'songs.csv')
    unified.to_csv(output_path, index=False)
    print(f"\nSaved unified CSV to: {output_path}")
    
    # Backup old files
    backup_dir = os.path.join(data_dir, 'backup_old_csvs')
    os.makedirs(backup_dir, exist_ok=True)
    
    for old_file in ['songs_data_with_genre.csv', 'msd_features.csv', 'msd_matches.csv', 'msd_unmatched.csv']:
        old_path = os.path.join(data_dir, old_file)
        if os.path.exists(old_path):
            backup_path = os.path.join(backup_dir, old_file)
            os.rename(old_path, backup_path)
            print(f"Backed up {old_file} to backup_old_csvs/")
    
    return unified


if __name__ == '__main__':
    migrate_to_unified_csv()
