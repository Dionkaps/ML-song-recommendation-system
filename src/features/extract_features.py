import os
import glob
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import feature_vars as fv

import librosa
import numpy as np
import contextlib
import io

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to temporarily redirect stderr to suppress warnings."""
    stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = stderr


def extract_mfcc(y, sr, n_mfcc=fv.n_mfcc):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def extract_melspectrogram(y, sr, n_fft=fv.n_fft, hop_length=fv.hop_length, n_mels=fv.n_mels):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def extract_spectral_centroid(y, sr, hop_length=fv.hop_length, n_fft=fv.n_fft):
    return librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft
    )


def extract_spectral_flatness(y, hop_length=fv.hop_length, n_fft=fv.n_fft):
    return librosa.feature.spectral_flatness(
        y=y, hop_length=hop_length, n_fft=n_fft
    )


def extract_zero_crossing_rate(y, hop_length=fv.hop_length):
    return librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)


def process_file(audio_path, results_dir, n_mfcc, n_fft, hop_length, n_mels):
    """
    Load an audio file, extract features, and save them to .npy files.
    Works with flat directory structure (audio_files).
    """
    filename = os.path.basename(audio_path)
    base_filename = os.path.splitext(filename)[0]
    
    try:
        print(f"Loading {filename}...")
        
        # Suppress stderr output during audio loading to hide mpg123 warnings
        with suppress_stderr():
            y, sr = librosa.load(audio_path, sr=None)
            
        # Extract features
        mfccs = extract_mfcc(y, sr, n_mfcc)
        mel_spec = extract_melspectrogram(y, sr, n_fft, hop_length, n_mels)
        spec_centroid = extract_spectral_centroid(y, sr, hop_length, n_fft)
        spec_flatness = extract_spectral_flatness(y, hop_length, n_fft)
        zcr = extract_zero_crossing_rate(y, hop_length)

        # Save features
        np.save(os.path.join(results_dir, f"{base_filename}_mfcc.npy"), mfccs)
        np.save(os.path.join(results_dir,
                f"{base_filename}_melspectrogram.npy"), mel_spec)
        np.save(os.path.join(results_dir,
                f"{base_filename}_spectral_centroid.npy"), spec_centroid)
        np.save(os.path.join(results_dir,
                f"{base_filename}_spectral_flatness.npy"), spec_flatness)
        np.save(os.path.join(results_dir,
                f"{base_filename}_zero_crossing_rate.npy"), zcr)

        print(f"✓ Finished processing {filename}")
        return base_filename
    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")
        return None


def run_feature_extraction(audio_dir='audio_files', results_dir='output/features'):
    """
    Finds all audio files (.wav, .mp3, .flac, .m4a) in audio_dir and processes them in parallel.
    Works with flat directory structure - genre info comes from songs_data_with_genre.csv.
    """
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
        print(f"Searched for extensions: {', '.join(audio_extensions)}")
        return
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction Started")
    print(f"{'='*60}")
    print(f"Audio directory: {audio_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Found {len(audio_files)} audio files")
    print(f"{'='*60}\n")

    # Determine number of worker processes
    num_workers = min(len(audio_files), multiprocessing.cpu_count())
    print(f"Starting parallel processing with {num_workers} workers...\n")

    # Use ProcessPoolExecutor for CPU-bound tasks
    processed_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_file,
                audio_path,
                results_dir,
                fv.n_mfcc,
                fv.n_fft,
                fv.hop_length,
                fv.n_mels
            )
            for audio_path in audio_files
        ]
        
        # Wait for all to complete
        for future in futures:
            result = future.result()
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
    
    # Note about genre mapping
    print("NOTE: Genre information should be loaded from data/songs_data_with_genre.csv")
    print("      Use genre_mapper.py utility to map features to genres.")


def main():
    start_time = time.time()
    run_feature_extraction()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
