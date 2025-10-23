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
    Also extracts and saves the genre information.
    """
    filename = os.path.basename(audio_path)
    base_filename = os.path.splitext(filename)[0]
    
    # Extract genre from the parent folder name
    genre = os.path.basename(os.path.dirname(audio_path))
    
    try:
        print(f"Loading {filename} (genre: {genre})...")
        
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
        
        # Save genre information
        np.save(os.path.join(results_dir, f"{base_filename}_genre.npy"), np.array([genre]))

        print(f"Finished processing {filename}")
        return base_filename, genre
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def run_feature_extraction(audio_dir='genres_original', results_dir='output/results'):
    """
    Finds all .wav files in the genre folders under audio_dir and processes them in parallel.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all genre directories
    genre_dirs = [d for d in glob.glob(os.path.join(audio_dir, "*")) if os.path.isdir(d)]
    if not genre_dirs:
        print(f"No genre directories found in {audio_dir}.")
        return
        
    # Collect all .wav files from all genre directories
    audio_files = []
    genre_map = {}  # To store mapping of filenames to genres
    
    for genre_dir in genre_dirs:
        genre_name = os.path.basename(genre_dir)
        wav_files = glob.glob(os.path.join(genre_dir, "*.wav"))
        
        for wav_file in wav_files:
            audio_files.append(wav_file)
            base_filename = os.path.splitext(os.path.basename(wav_file))[0]
            genre_map[base_filename] = genre_name
            
        print(f"Found {len(wav_files)} .wav files in {genre_name} genre.")
    
    if not audio_files:
        print(f"No .wav files found in genre directories under {audio_dir}.")
        return
        
    # Save the genre mapping for later use
    np.save(os.path.join(results_dir, "genre_map.npy"), genre_map)
    print(f"Saved genre mapping for {len(genre_map)} files")

    # Determine number of worker processes
    num_workers = min(len(audio_files), multiprocessing.cpu_count())
    print(f"Starting parallel processing with {num_workers} workers...")

    # Use ProcessPoolExecutor for CPU-bound tasks
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
        
        # Wait for all to complete and collect genre information
        for future in futures:
            result = future.result()
            if result:
                base_filename, genre = result
                genre_map[base_filename] = genre
    
    # Save the genre mapping to a file
    np.save(os.path.join(results_dir, "genre_map.npy"), genre_map)
    print(f"Genre mapping saved to {os.path.join(results_dir, 'genre_map.npy')}")
    
    # Also save as CSV for easy inspection
    with open(os.path.join(results_dir, "genre_map.csv"), "w") as f:
        f.write("filename,genre\n")
        for filename, genre in genre_map.items():
            f.write(f"{filename},{genre}\n")
    print(f"Genre mapping also saved to CSV: {os.path.join(results_dir, 'genre_map.csv')}")


def main():
    start_time = time.time()
    run_feature_extraction()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
