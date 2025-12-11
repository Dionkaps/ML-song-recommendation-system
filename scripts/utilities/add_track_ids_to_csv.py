"""
Add MSD track_id to millionsong_dataset.csv by extracting from H5 files.
"""
import os
import sys
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup paths
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

MSD_ROOT = "data/millionsongsubset_extracted/MillionSongSubset"
INPUT_CSV = "data/millionsong_dataset.csv"
OUTPUT_CSV = "data/millionsong_dataset.csv"  # Overwrite


def extract_track_info(h5_path):
    """Extract track_id, artist, title from H5 file."""
    try:
        with h5py.File(h5_path, 'r') as f:
            metadata = f['metadata']['songs'][0]
            analysis = f['analysis']['songs'][0]
            
            track_id = analysis['track_id'].decode('utf-8') if isinstance(analysis['track_id'], bytes) else str(analysis['track_id'])
            artist = metadata['artist_name'].decode('utf-8') if isinstance(metadata['artist_name'], bytes) else str(metadata['artist_name'])
            title = metadata['title'].decode('utf-8') if isinstance(metadata['title'], bytes) else str(metadata['title'])
            
            return {
                'track_id': track_id,
                'artist': artist,
                'title': title
            }
    except Exception as e:
        return None


def find_all_h5_files(msd_root):
    """Find all H5 files recursively."""
    h5_files = []
    for root, dirs, files in os.walk(msd_root):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files


def main():
    print("Adding track_id to millionsong_dataset.csv...\n")
    
    # Load existing CSV
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows")
    
    # Find all H5 files
    print(f"\nScanning {MSD_ROOT} for H5 files...")
    h5_files = find_all_h5_files(MSD_ROOT)
    print(f"Found {len(h5_files)} H5 files")
    
    # Extract track info from all H5 files
    print("\nExtracting track info from H5 files...")
    h5_data = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(extract_track_info, path): path for path in h5_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                h5_data.append(result)
    
    print(f"Extracted info from {len(h5_data)} files")
    
    # Create lookup by normalized (artist, title)
    def normalize(s):
        return str(s).lower().strip()
    
    h5_lookup = {}
    for item in h5_data:
        key = (normalize(item['artist']), normalize(item['title']))
        h5_lookup[key] = item['track_id']
    
    # Match and add track_id to CSV
    print("\nMatching tracks...")
    matched = 0
    track_ids = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Matching"):
        key = (normalize(row['artist']), normalize(row['title']))
        track_id = h5_lookup.get(key)
        track_ids.append(track_id)
        if track_id:
            matched += 1
    
    df['track_id'] = track_ids
    
    print(f"\nMatched {matched}/{len(df)} tracks ({100*matched/len(df):.1f}%)")
    
    # Reorder columns: track_id first
    cols = ['track_id', 'title', 'artist', 'genre']
    df = df[cols]
    
    # Save
    print(f"\nSaving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")
    
    # Show sample
    print("\nSample of updated CSV:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
