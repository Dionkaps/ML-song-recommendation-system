import os
import glob
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import feature_vars as fv
import librosa
import numpy as np
import contextlib
import io
import sys

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

        print(f"Finished processing {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")


def run_feature_extraction(audio_dir='audio_files', results_dir='results'):
    """
    Finds all .mp3 files in the audio_dir and processes them in parallel.
    """
    os.makedirs(results_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(audio_dir, "*.mp3"))
    if not audio_files:
        print(f"No .mp3 files found in {audio_dir}.")
        return


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
        # Wait for all to complete
        for future in futures:
            future.result()


def main():
    start_time = time.time()
    run_feature_extraction()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
