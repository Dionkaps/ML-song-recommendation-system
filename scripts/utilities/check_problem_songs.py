#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check if specific songs have genres in CSV"""
import csv
import os
import sys
from pathlib import Path

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Songs that are showing warnings
problem_songs = [
    '-M- - Je suis une cigarette',
    '2 Chainz - Good Drank',
    '21 Savage - Bank Account',
    '21 Savage - No Heart',
    '21 Savage - out for the night (part 2)',
    '21 Savage - Savage Mode',
    '2Pac - Dear Mama',
    '2Pac - All Eyez On Me',
]

print("="*60)
print("Checking Problem Songs")
print("="*60)

# Load CSV (try unified first, then legacy)
csv_path = os.path.join('data', 'songs.csv')
if not os.path.exists(csv_path):
    csv_path = os.path.join('data', 'songs_data_with_genre.csv')

with open(csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"\nUsing CSV: {csv_path}")
print(f"Total songs in CSV: {len(rows)}")

print("\n" + "="*60)
print("Results:")
print("="*60)

for song in problem_songs:
    print(f"\nSong: {song}")
    
    # Check if in CSV (with .mp3 extension) - filename column works for both unified and legacy CSV
    matches = [r for r in rows if r.get('filename') and song in r['filename']]
    
    if matches:
        print(f"  [YES] Found in CSV: {matches[0]['filename']}")
        genre = matches[0].get('genre', 'N/A')
        print(f"        Genre: {genre[:60] if genre else 'N/A'}...")
    else:
        print(f"  [NO]  NOT in CSV")
    
    # Check if audio file exists
    audio_path = f"audio_files\\{song}.mp3"
    if os.path.exists(audio_path):
        print(f"  [YES] Audio file EXISTS")
    else:
        print(f"  [NO]  Audio file MISSING")
    
    # Check if feature file exists
    feature_path = f"output\\features\\{song}_mfcc.npy"
    if os.path.exists(feature_path):
        print(f"  [YES] Feature file EXISTS")
    else:
        print(f"  [NO]  Feature file MISSING")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("\nThese warnings occur when:")
print("  - Feature files exist BUT")
print("  - Audio files are missing OR")
print("  - Songs are not in the current CSV")
print("\nThese are ORPHANED feature files from previous runs.")
print("\nSolution: Run cleanup script to remove them")
print("  python cleanup_orphaned_features.py --confirm")
print("="*60)
