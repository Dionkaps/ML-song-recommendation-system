import os
import tarfile
import h5py
import csv
from pathlib import Path
import sys

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

def extract_tar_gz(tar_path, extract_to):
    """Extract tar.gz file"""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete!")

def get_song_info_from_h5(h5_file_path):
    """Extract song name and genre from HDF5 file"""
    try:
        with h5py.File(h5_file_path, 'r') as h5:
            # Get song title
            title = h5['metadata']['songs']['title'][0]
            if isinstance(title, bytes):
                title = title.decode('utf-8', errors='ignore')
            
            # Get artist name (useful context)
            artist = h5['metadata']['songs']['artist_name'][0]
            if isinstance(artist, bytes):
                artist = artist.decode('utf-8', errors='ignore')
            
            # Try to get genre/tags - the Million Song Dataset doesn't have direct genre info
            # but we can get artist terms which are tag-like descriptors
            terms = []
            if 'metadata' in h5 and 'artist_terms' in h5['metadata']:
                terms_data = h5['metadata']['artist_terms'][:]
                terms = [t.decode('utf-8', errors='ignore') if isinstance(t, bytes) else str(t) 
                        for t in terms_data[:5]]  # Get top 5 terms
            
            genre = ', '.join(terms) if terms else 'Unknown'
            
            return {
                'title': title,
                'artist': artist,
                'genre': genre
            }
    except Exception as e:
        print(f"Error processing {h5_file_path}: {e}")
        return None

def find_h5_files(directory):
    """Recursively find all .h5 files"""
    h5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files

def main():
    # Paths
    data_dir = os.path.join(project_root, 'data')
    tar_path = os.path.join(data_dir, 'millionsongsubset.tar.gz')
    extract_dir = os.path.join(data_dir, 'millionsongsubset_extracted')
    output_csv = os.path.join(data_dir, 'millionsong_dataset.csv')
    
    # Extract tar.gz if not already extracted
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        extract_tar_gz(tar_path, extract_dir)
    else:
        print(f"Directory {extract_dir} already exists, skipping extraction...")
    
    # Find all HDF5 files
    print("Searching for HDF5 files...")
    h5_files = find_h5_files(extract_dir)
    print(f"Found {len(h5_files)} HDF5 files")
    
    # Extract song information
    songs_data = []
    total = len(h5_files)
    
    for idx, h5_file in enumerate(h5_files, 1):
        if idx % 100 == 0:
            print(f"Processing file {idx}/{total}...")
        
        song_info = get_song_info_from_h5(h5_file)
        if song_info:
            songs_data.append(song_info)
    
    # Write to CSV
    print(f"\nWriting {len(songs_data)} songs to CSV...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'artist', 'genre']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(songs_data)
    
    print(f"Done! CSV file created at: {output_csv}")
    print(f"Total songs extracted: {len(songs_data)}")

if __name__ == "__main__":
    main()
