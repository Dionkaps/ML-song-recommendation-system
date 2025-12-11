#!/usr/bin/env python3
"""
Extract features from a sample of 20 audio files for testing.
"""
import os
import glob
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features.extract_features import process_file
from config import feature_vars as fv

def extract_sample(audio_dir='audio_files', results_dir='output/features', sample_size=20):
    """Extract features from a sample of audio files."""
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found.")
        return
    
    # Collect all audio files (support multiple formats)
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        found = glob.glob(os.path.join(audio_dir, ext))
        audio_files.extend(found)
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}.")
        return
    
    # Take only the sample
    audio_files = audio_files[:sample_size]
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction (Sample)")
    print(f"{'='*60}")
    print(f"Audio directory: {audio_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Sample size: {len(audio_files)} files (out of {len(glob.glob(os.path.join(audio_dir, '*.*')))} total)")
    print(f"{'='*60}\n")
    
    processed_count = 0
    failed_count = 0
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] ", end="")
        result = process_file(
            audio_path,
            results_dir,
            fv.n_mfcc,
            fv.n_fft,
            fv.hop_length,
            fv.n_mels
        )
        if result:
            processed_count += 1
        else:
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction Complete")
    print(f"{'='*60}")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    extract_sample(sample_size=20)
