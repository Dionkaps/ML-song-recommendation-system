import os
import glob
import multiprocessing
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
import soundfile as sf

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

FEATURE_KEYS = tuple(fv.AUDIO_FEATURE_KEYS)
DEFAULT_MAX_WORKERS = 8

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to temporarily redirect stderr to suppress warnings."""
    # Open a pair of null files
    null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
    # Save the actual stdout (1) and stderr (2) file descriptors.
    save_fds = [os.dup(1), os.dup(2)]

    try:
        # Assign the null pointers to stdout and stderr.
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        # Close the null files and the temporary fds
        for fd in null_fds + save_fds:
            os.close(fd)


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
    # Build rhythmic descriptors from the onset envelope directly.
    # This avoids librosa.beat.beat_track(), which is unstable in this environment.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo_values = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
    )
    
    # Calculate statistics
    tempo = float(tempo_values[0]) if len(tempo_values) > 0 else 0.0
    
    mean_onset_strength = float(np.mean(onset_env))
    std_onset_strength = float(np.std(onset_env))
    
    # Calculate onset rate (onsets per second)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    duration = len(y) / sr
    onset_rate = len(onset_frames) / duration if duration > 0 else 0.0
    
    # Return as 2D array [4 features, 1 frame] for consistency
    # Features: tempo, mean_onset_strength, std_onset_strength, onset_rate
    return np.array([[tempo], [mean_onset_strength], [std_onset_strength], [onset_rate]])


def get_feature_output_paths(base_filename, results_dir):
    """Return the expected feature bundle paths for one song basename."""
    return {
        key: os.path.join(results_dir, f"{base_filename}_{key}.npy")
        for key in FEATURE_KEYS
    }


def has_complete_feature_bundle(base_filename, results_dir):
    """Check whether all expected feature files already exist for one song."""
    return all(
        os.path.isfile(path)
        for path in get_feature_output_paths(base_filename, results_dir).values()
    )


def save_array_atomic(output_path, array):
    """Write a .npy file via a temporary path so interrupted runs can resume safely."""
    file_descriptor, temp_path = tempfile.mkstemp(
        dir=os.path.dirname(output_path),
        suffix=".npy",
    )
    os.close(file_descriptor)
    try:
        np.save(temp_path, array)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_audio(audio_path):
    """Load audio as mono float32 while preferring direct WAV reads after preprocessing."""
    if Path(audio_path).suffix.lower() == ".wav":
        y, sr = sf.read(audio_path, always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1)
        return np.asarray(y, dtype=np.float32), sr

    y, sr = librosa.load(audio_path, sr=None)
    return y, sr


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
    output_paths = get_feature_output_paths(base_filename, results_dir)

    if has_complete_feature_bundle(base_filename, results_dir):
        return {"base_name": base_filename, "status": "skipped"}
    
    try:
        y, sr = load_audio(audio_path)
            
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

        feature_arrays = {
            "mfcc": mfccs,
            "delta_mfcc": delta_mfcc,
            "delta2_mfcc": delta2_mfcc,
            "spectral_centroid": spec_centroid,
            "spectral_rolloff": spec_rolloff,
            "spectral_flux": spec_flux,
            "spectral_flatness": spec_flatness,
            "zero_crossing_rate": zcr,
            "chroma": chroma,
            "beat_strength": beat_strength,
        }

        for key, array in feature_arrays.items():
            save_array_atomic(output_paths[key], array)

        return {"base_name": base_filename, "status": "processed"}
    except Exception as e:
        return {
            "base_name": base_filename,
            "status": "failed",
            "error": str(e),
        }


def run_feature_extraction(
    audio_dir='audio_files',
    results_dir='output/features',
    workers=None,
    executor_type='auto',
    resume=True,
):
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
    audio_files = sorted(set(audio_files))
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}.")
        print(f"Searched for extensions: {', '.join(audio_extensions)}")
        return {
            "total_audio_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "results_dir": results_dir,
        }

    pending_audio_files = []
    skipped_existing = 0
    for audio_path in audio_files:
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        if resume and has_complete_feature_bundle(base_filename, results_dir):
            skipped_existing += 1
            continue
        pending_audio_files.append(audio_path)

    print(f"\n{'='*60}")
    print(f"Feature Extraction Started")
    print(f"{'='*60}")
    print(f"Audio directory: {audio_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Found {len(audio_files)} audio files")
    print(f"Already complete: {skipped_existing}")
    print(f"Pending extraction: {len(pending_audio_files)}")
    print(f"{'='*60}\n")

    if not pending_audio_files:
        print("All audio files already have complete feature bundles.\n")
        print("NOTE: Genre and MSD metadata are loaded from unified data/songs.csv")
        print("      Use genre_mapper.py utility to map features to genres.")
        return {
            "total_audio_files": len(audio_files),
            "processed": 0,
            "skipped": skipped_existing,
            "failed": 0,
            "results_dir": results_dir,
        }

    if executor_type == 'auto':
        executor_type = 'thread' if os.name == 'nt' else 'process'
    executor_type = executor_type.lower()
    if executor_type not in {'thread', 'process'}:
        raise ValueError("executor_type must be one of: auto, thread, process")

    cpu_count = multiprocessing.cpu_count()
    default_workers = min(
        len(pending_audio_files),
        DEFAULT_MAX_WORKERS if executor_type == 'thread' else cpu_count,
    )
    num_workers = default_workers if workers is None else int(workers)
    num_workers = max(1, min(num_workers, len(pending_audio_files)))
    executor_cls = ThreadPoolExecutor if executor_type == 'thread' else ProcessPoolExecutor

    print(
        f"Starting parallel processing with {num_workers} {executor_type} "
        f"{'worker' if num_workers == 1 else 'workers'}...\n"
    )

    processed_count = 0
    failed_count = 0
    failed_files = []

    with executor_cls(max_workers=num_workers) as executor:
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
            for audio_path in pending_audio_files
        }
        
        # Wait for all to complete with progress bar
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extracting features",
            unit="file",
            file=sys.stdout,
        )
        for future in pbar:
            result = future.result()
            if result["status"] == "processed":
                processed_count += 1
            elif result["status"] == "skipped":
                skipped_existing += 1
            else:
                failed_count += 1
                failed_files.append(result)
            pbar.set_postfix(
                success=processed_count,
                skipped=skipped_existing,
                failed=failed_count,
            )
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction Complete")
    print(f"{'='*60}")
    print(f"Total audio files: {len(audio_files)}")
    print(f"Successfully processed: {processed_count} files")
    print(f"Skipped existing: {skipped_existing} files")
    print(f"Failed: {failed_count} files")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")

    if failed_files:
        failure_log_path = os.path.join(results_dir, "feature_extraction_failures.txt")
        with open(failure_log_path, "w", encoding="utf-8") as failure_log:
            for item in failed_files:
                failure_log.write(
                    f"{item['base_name']}\t{item['error']}\n"
                )
        print(f"Failure log saved to: {failure_log_path}\n")
    
    # Note about genre mapping
    print("NOTE: Genre and MSD metadata are loaded from unified data/songs.csv")
    print("      Use genre_mapper.py utility to map features to genres.")
    return {
        "total_audio_files": len(audio_files),
        "processed": processed_count,
        "skipped": skipped_existing,
        "failed": failed_count,
        "results_dir": results_dir,
    }


def main():
    start_time = time.time()
    run_feature_extraction()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
