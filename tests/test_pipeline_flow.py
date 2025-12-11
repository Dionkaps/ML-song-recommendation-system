#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the pipeline flow:
1. Check unified songs.csv format
2. Check audio files match CSV entries
3. Check feature files match audio files
4. Test genre mapping for clustering
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import csv
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_csv_structure():
    """Test 1: Verify CSV structure"""
    print("="*60)
    print("TEST 1: CSV Structure")
    print("="*60)
    
    # Check unified CSV first, then fallback to legacy
    csv_path = os.path.join("data", "songs.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "songs_data_with_genre.csv")
    
    if not os.path.exists(csv_path):
        print("❌ CSV file not found!")
        return None
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"✓ Using CSV: {csv_path}")
    print(f"✓ Total CSV entries: {len(rows)}")
    print(f"✓ Columns: {', '.join(rows[0].keys())}")
    
    # Check for empty genres
    empty_genres = sum(1 for r in rows if not r.get('genre', '').strip())
    print(f"✓ Entries with genres: {len(rows) - empty_genres}")
    print(f"✓ Entries without genres: {empty_genres}")
    
    # Sample entries
    print("\nSample entries:")
    for i, row in enumerate(rows[:3], 1):
        genre = row.get('genre', '')
        genre_preview = genre[:40] + "..." if len(genre) > 40 else genre
        filename = row.get('filename', 'N/A')
        print(f"  {i}. {filename}")
        print(f"     Genre: {genre_preview}")
    
    return rows

def test_audio_files(csv_rows):
    """Test 2: Check audio files match CSV"""
    print("\n" + "="*60)
    print("TEST 2: Audio Files vs CSV")
    print("="*60)
    
    audio_dir = "audio_files"
    if not os.path.exists(audio_dir):
        print("❌ Audio directory not found!")
        return None
    
    # Get all mp3 files
    audio_files = set([f for f in os.listdir(audio_dir) if f.endswith('.mp3')])
    print(f"✓ Audio files in directory: {len(audio_files)}")
    
    # Get filenames from CSV
    csv_files = set([row['filename'] for row in csv_rows])
    print(f"✓ Files listed in CSV: {len(csv_files)}")
    
    # Check matches
    in_both = audio_files & csv_files
    only_audio = audio_files - csv_files
    only_csv = csv_files - audio_files
    
    print(f"\n✓ Files in both: {len(in_both)}")
    if only_audio:
        print(f"⚠️  Audio files NOT in CSV: {len(only_audio)}")
        print(f"   First 3: {list(only_audio)[:3]}")
    if only_csv:
        print(f"⚠️  CSV entries WITHOUT audio file: {len(only_csv)}")
        print(f"   First 3: {list(only_csv)[:3]}")
    
    return audio_files

def test_feature_files(audio_files):
    """Test 3: Check feature files match audio files"""
    print("\n" + "="*60)
    print("TEST 3: Feature Files vs Audio Files")
    print("="*60)
    
    results_dir = "output/results"
    if not os.path.exists(results_dir):
        print("❌ Results directory not found!")
        return None
    
    # Get feature files
    mfcc_files = [f.replace('_mfcc.npy', '') for f in os.listdir(results_dir) if f.endswith('_mfcc.npy')]
    print(f"✓ Feature files (mfcc): {len(mfcc_files)}")
    
    # Convert audio filenames to basenames (without .mp3)
    audio_basenames = set([Path(f).stem for f in audio_files])
    mfcc_set = set(mfcc_files)
    
    in_both = audio_basenames & mfcc_set
    only_audio = audio_basenames - mfcc_set
    only_features = mfcc_set - audio_basenames
    
    print(f"\n✓ Audio files with features: {len(in_both)}")
    if only_audio:
        print(f"⚠️  Audio files WITHOUT features: {len(only_audio)}")
        print(f"   First 3: {list(only_audio)[:3]}")
    if only_features:
        print(f"⚠️  Feature files WITHOUT audio: {len(only_features)}")
        print(f"   First 3: {list(only_features)[:3]}")
    
    return mfcc_files

def test_genre_mapping(csv_rows):
    """Test 4: Test genre mapping for clustering"""
    print("\n" + "="*60)
    print("TEST 4: Genre Mapping for Clustering")
    print("="*60)
    
    # Simulate what kmeans.py does
    from src.utils import genre_mapper
    
    # Load genre mapping (will use unified songs.csv by default)
    multi_label_mapping = genre_mapper.load_genre_mapping()
    print(f"\n✓ Loaded multi-label mapping: {len(multi_label_mapping)} songs")
    
    # Check sample keys
    print("\nSample keys from genre mapping:")
    for i, (filename, genres) in enumerate(list(multi_label_mapping.items())[:3], 1):
        print(f"  {i}. Key: '{filename}'")
        print(f"     Genres: {genres}")
    
    # Simulate kmeans.py's conversion
    print("\n" + "-"*60)
    print("Simulating kmeans.py genre map creation:")
    print("-"*60)
    
    genre_map = {}
    for filename, genres_list in multi_label_mapping.items():
        # Remove .mp3 extension (as kmeans does)
        base = Path(filename).stem
        # Get primary genre
        primary_genre = genre_mapper.get_primary_genre(filename, multi_label_mapping)
        genre_map[base] = primary_genre
    
    print(f"✓ Created genre_map with {len(genre_map)} entries")
    print("\nSample genre_map entries:")
    for i, (base, genre) in enumerate(list(genre_map.items())[:3], 1):
        print(f"  {i}. Base: '{base}'")
        print(f"     Primary Genre: {genre}")
    
    # Now check if feature files can be mapped
    print("\n" + "-"*60)
    print("Testing feature file to genre mapping:")
    print("-"*60)
    
    results_dir = "output/results"
    if os.path.exists(results_dir):
        feature_files = [f.replace('_mfcc.npy', '') for f in os.listdir(results_dir) if f.endswith('_mfcc.npy')]
        
        # Check how many feature files have genres
        mapped = 0
        unmapped = 0
        unmapped_samples = []
        
        for base in feature_files[:100]:  # Test first 100
            if base in genre_map:
                mapped += 1
            else:
                unmapped += 1
                if len(unmapped_samples) < 3:
                    unmapped_samples.append(base)
        
        print(f"Tested {len(feature_files[:100])} feature files:")
        print(f"  ✓ Mapped to genre: {mapped}")
        print(f"  ✗ NOT mapped: {unmapped}")
        
        if unmapped_samples:
            print(f"\nSample unmapped feature files:")
            for s in unmapped_samples:
                print(f"    - '{s}'")
    
    return genre_map

def main():
    print("\n" + "="*60)
    print("PIPELINE FLOW VERIFICATION TEST")
    print("="*60 + "\n")
    
    # Run tests
    csv_rows = test_csv_structure()
    if csv_rows:
        audio_files = test_audio_files(csv_rows)
        if audio_files:
            feature_files = test_feature_files(audio_files)
            genre_map = test_genre_mapping(csv_rows)
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n✓ Pipeline verification complete!")
    print("\nKey findings:")
    print("1. CSV contains songs with genres (as required)")
    print("2. Audio files should match CSV entries")
    print("3. Feature files are extracted from audio files")
    print("4. Genre mapping connects CSV to clustering")
    print("\nIf all tests show ✓, the pipeline is working correctly!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
