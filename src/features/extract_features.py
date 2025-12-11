import os
import glob
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from tqdm import tqdm

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
    """Extract MFCC (Mel-frequency cepstral coefficients)."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def extract_delta_mfcc(mfcc):
    """Extract ΔMFCC (first derivatives of MFCC)."""
    return librosa.feature.delta(mfcc)


def extract_delta2_mfcc(mfcc):
    """Extract ΔΔMFCC (second derivatives of MFCC)."""
    return librosa.feature.delta(mfcc, order=2)


def extract_spectral_centroid(y, sr, hop_length=fv.hop_length, n_fft=fv.n_fft):
    """Extract spectral centroid - brightness of sound."""
    return librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft
    )


def extract_spectral_rolloff(y, sr, hop_length=fv.hop_length, n_fft=fv.n_fft, roll_percent=0.85):
    """Extract spectral rolloff - frequency below which roll_percent of energy is contained."""
    return librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, roll_percent=roll_percent
    )


def extract_spectral_flux(y, sr, hop_length=fv.hop_length, n_fft=fv.n_fft):
    """
    Extract spectral flux - measure of how quickly the spectrum changes.
    Computed as the onset strength envelope which captures spectral change.
    """
    # Onset strength is essentially spectral flux - measures spectral change over time
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    # Return as 2D array to match other features [1, n_frames]
    return onset_env.reshape(1, -1)


def extract_spectral_flatness(y, hop_length=fv.hop_length, n_fft=fv.n_fft):
    """Extract spectral flatness - tonal vs noisy characteristics."""
    return librosa.feature.spectral_flatness(
        y=y, hop_length=hop_length, n_fft=n_fft
    )


def extract_zero_crossing_rate(y, hop_length=fv.hop_length):
    """Extract zero crossing rate - noisiness/percussiveness."""
    return librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)


def extract_chroma(y, sr, hop_length=fv.hop_length, n_fft=fv.n_fft, n_chroma=12):
    """Extract chroma features - 12-dimensional pitch class profile."""
    return librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_chroma=n_chroma
    )


def extract_beat_strength(y, sr, hop_length=fv.hop_length):
    """
    Extract beat strength / onset rate features.
    Returns tempo (BPM) and mean onset strength as a combined feature.
    """
    # Get tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    
    # Get onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Calculate statistics
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    
    mean_onset_strength = float(np.mean(onset_env))
    std_onset_strength = float(np.std(onset_env))
    
    # Calculate onset rate (onsets per second)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    duration = len(y) / sr
    onset_rate = len(onset_frames) / duration if duration > 0 else 0.0
    
    # Return as 2D array [4 features, 1 frame] for consistency
    # Features: tempo, mean_onset_strength, std_onset_strength, onset_rate
    return np.array([[tempo], [mean_onset_strength], [std_onset_strength], [onset_rate]])


def process_file(audio_path, results_dir, n_mfcc, n_fft, hop_length, n_mels):
    """
    Load an audio file, extract features, and save them to .npy files.
    
    Extracts the following features:
    - MFCC (Mel-frequency cepstral coefficients)
    - ΔMFCC (first derivatives)
    - ΔΔMFCC (second derivatives)
    - Spectral centroid
    - Spectral rolloff
    - Spectral flux
    - Spectral flatness
    - Zero Crossing Rate (ZCR)
    - Chroma (12-dim pitch class profile)
    - Beat strength / onset rate
    """
    filename = os.path.basename(audio_path)
    base_filename = os.path.splitext(filename)[0]
    
    try:
        # Suppress stderr output during audio loading to hide mpg123 warnings
        with suppress_stderr():
            y, sr = librosa.load(audio_path, sr=None)
            
        # Extract MFCC and derivatives
        mfccs = extract_mfcc(y, sr, n_mfcc)
        delta_mfcc = extract_delta_mfcc(mfccs)
        delta2_mfcc = extract_delta2_mfcc(mfccs)
        
        # Extract spectral features
        spec_centroid = extract_spectral_centroid(y, sr, hop_length, n_fft)
        spec_rolloff = extract_spectral_rolloff(y, sr, hop_length, n_fft)
        spec_flux = extract_spectral_flux(y, sr, hop_length, n_fft)
        spec_flatness = extract_spectral_flatness(y, hop_length, n_fft)
        
        # Extract other features
        zcr = extract_zero_crossing_rate(y, hop_length)
        chroma = extract_chroma(y, sr, hop_length, n_fft)
        beat_strength = extract_beat_strength(y, sr, hop_length)

        # Save all features
        np.save(os.path.join(results_dir, f"{base_filename}_mfcc.npy"), mfccs)
        np.save(os.path.join(results_dir, f"{base_filename}_delta_mfcc.npy"), delta_mfcc)
        np.save(os.path.join(results_dir, f"{base_filename}_delta2_mfcc.npy"), delta2_mfcc)
        np.save(os.path.join(results_dir, f"{base_filename}_spectral_centroid.npy"), spec_centroid)
        np.save(os.path.join(results_dir, f"{base_filename}_spectral_rolloff.npy"), spec_rolloff)
        np.save(os.path.join(results_dir, f"{base_filename}_spectral_flux.npy"), spec_flux)
        np.save(os.path.join(results_dir, f"{base_filename}_spectral_flatness.npy"), spec_flatness)
        np.save(os.path.join(results_dir, f"{base_filename}_zero_crossing_rate.npy"), zcr)
        np.save(os.path.join(results_dir, f"{base_filename}_chroma.npy"), chroma)
        np.save(os.path.join(results_dir, f"{base_filename}_beat_strength.npy"), beat_strength)

        return base_filename
    except Exception as e:
        return None


def run_feature_extraction(audio_dir='audio_files', results_dir='output/features'):
    """
    Finds all audio files (.wav, .mp3, .flac, .m4a) in audio_dir and processes them in parallel.
    Works with flat directory structure - genre info comes from unified songs.csv.
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
        futures = {
            executor.submit(
                process_file,
                audio_path,
                results_dir,
                fv.n_mfcc,
                fv.n_fft,
                fv.hop_length,
                fv.n_mels
            ): audio_path
            for audio_path in audio_files
        }
        
        # Wait for all to complete with progress bar
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="Extracting features", unit="file")
        for future in pbar:
            result = future.result()
            if result:
                processed_count += 1
            else:
                failed_count += 1
            pbar.set_postfix(success=processed_count, failed=failed_count)
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction Complete")
    print(f"{'='*60}")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    # Note about genre mapping
    print("NOTE: Genre and MSD metadata are loaded from unified data/songs.csv")
    print("      Use genre_mapper.py utility to map features to genres.")


def main():
    start_time = time.time()
    run_feature_extraction()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
