"""
Audio Preprocessor - Main orchestrator for audio preprocessing pipeline.

This module provides a unified interface for preprocessing audio files
to ensure consistency and fairness across tracks for Music Information
Retrieval (MIR) tasks.

Pipeline Steps:
1. Load audio file (resampled to target sample rate, converted to mono)
2. Duration validation and cropping (default: 29 seconds)
3. Loudness normalization (ITU-R BS.1770, default: -14 LUFS)
4. True Peak limiting (EBU R128 compliant, default: -1.0 dBTP)
5. Save processed audio

Standards Compliance:
- ITU-R BS.1770-4: Loudness measurement
- EBU R128: Loudness normalization and maximum level

Usage:
    from src.audio_preprocessing import AudioPreprocessor
    
    processor = AudioPreprocessor(
        target_duration=29.0,
        target_lufs=-14.0,
        max_true_peak=-1.0
    )
    
    # Process single file
    result = processor.process_file("path/to/audio.mp3")
    
    # Process directory
    stats = processor.process_directory("path/to/audio_dir")
"""

import os
import logging
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Optional

import librosa
import soundfile as sf
from tqdm import tqdm

from .duration_handler import DurationHandler
from .loudness_normalizer import LoudnessNormalizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings from librosa/audioread/mpg123
warnings.filterwarnings("ignore")
logging.getLogger("audioread").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.ERROR)

# Redirect stderr to suppress C-level mpg123 warnings
import sys
import os
import io

class _SuppressStderr:
    """Context manager to suppress stderr (catches C-level warnings from mpg123)"""
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class AudioPreprocessor:
    def __init__(
        self,
        target_duration: float = 29.0,
        target_lufs: float = -14.0,
        max_true_peak: float = -1.0,
        sample_rate: int = 22050
    ):
        """
        Initialize the AudioPreprocessor.

        Args:
            target_duration: Target duration in seconds (default: 29.0)
                            Deezer previews are typically 30s, so 29s allows margin.
            target_lufs: Target Integrated Loudness in LUFS (default: -14.0)
                        -14 LUFS is standard for streaming platforms.
            max_true_peak: Maximum True Peak in dBTP (default: -1.0)
                          -1.0 dBTP provides headroom and prevents clipping.
            sample_rate: Sample rate for processing (default: 22050)
                        22050 Hz is standard for MIR feature extraction.
        
        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
            
        self.sample_rate = sample_rate
        self.duration_handler = DurationHandler(target_duration=target_duration)
        self.loudness_normalizer = LoudnessNormalizer(target_lufs=target_lufs, max_true_peak=max_true_peak)

    def process_file(self, file_path: str) -> Dict:
        """
        Process a single audio file through the complete pipeline.
        
        Pipeline:
        1. Load audio (resample to target SR, convert to mono)
        2. Check duration and crop if needed
        3. Normalize loudness (LUFS) with True Peak protection
        4. Save processed audio back to disk (overwrites original)

        Args:
            file_path: Path to the audio file (MP3, WAV, FLAC, etc.)

        Returns:
            Dictionary with processing statistics:
                - file: Filename
                - status: 'success', 'removed', or 'error'
                - actions: List of actions performed
                - original_duration: Original duration in seconds
                - final_duration: Final duration in seconds
                - original_lufs: Original loudness in LUFS
                - final_lufs: Final loudness in LUFS
                - original_peak_db: Original peak in dBFS
                - final_peak_db: Final peak in dBFS
                - gain_applied_db: Total gain applied in dB
                - error: Error message if status is 'error'
        """
        path = Path(file_path)
        result = {
            'file': path.name,
            'status': 'success',
            'actions': [],
            'original_duration': 0.0,
            'final_duration': 0.0,
            'original_lufs': None,
            'final_lufs': None,
            'original_peak_db': None,
            'final_peak_db': None,
            'gain_applied_db': 0.0,
            'error': None
        }

        try:
            # 1. Load Audio (suppress C-level warnings from mpg123)
            with _SuppressStderr():
                y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            result['original_duration'] = round(duration, 3)

            # 2. Duration Check & Cropping
            y, action = self.duration_handler.process(y, sr)
            
            if action == 'too_short':
                os.remove(path)
                result['status'] = 'removed'
                result['actions'].append('removed_too_short')
                return result
            
            if action == 'cropped':
                result['actions'].append('cropped')
            
            result['final_duration'] = round(librosa.get_duration(y=y, sr=sr), 3)

            # 3. Loudness Normalization
            y, norm_stats = self.loudness_normalizer.process(y, sr)
            
            # Copy loudness stats
            result['original_lufs'] = round(norm_stats['original_lufs'], 3) if norm_stats['original_lufs'] is not None else None
            result['final_lufs'] = round(norm_stats['final_lufs'], 3) if norm_stats['final_lufs'] is not None else None
            result['original_peak_db'] = round(norm_stats.get('original_peak_db', 0), 3) if norm_stats.get('original_peak_db') is not None else None
            result['final_peak_db'] = round(norm_stats.get('final_peak_db', 0), 3) if norm_stats.get('final_peak_db') is not None else None
            result['gain_applied_db'] = round(norm_stats.get('gain_applied_db', 0), 3)
            result['actions'].extend(norm_stats['actions'])

            # 4. Save File
            sf.write(path, y, sr)
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"Error processing {path.name}: {e}")

        return result

    def process_directory(self, directory: str, max_workers: int = None, return_details: bool = False) -> Dict:
        """
        Process all audio files in a directory in parallel.

        Args:
            directory: Path to directory containing audio files
            max_workers: Number of parallel workers (default: min(cpu_count, 8))
            return_details: If True, include per-file results in output

        Returns:
            Statistics dictionary containing:
                - total: Total files found
                - processed: Successfully processed files
                - removed: Files removed (too short)
                - errors: Files that failed to process
                - cropped: Files that were cropped
                - normalized: Files that were normalized
                - peak_limited: Files that required peak limiting
                - removed_files: List of removed filenames
                - details: (if return_details=True) List of per-file results
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return {'error': 'Directory not found'}

        # Find all audio files
        files = list(dir_path.glob("*.mp3")) + list(dir_path.glob("*.wav"))
        
        if not files:
            logger.warning(f"No audio files found in {directory}")
            return {'total': 0}

        if max_workers is None:
            max_workers = min(cpu_count(), 8)

        stats = {
            'total': len(files),
            'processed': 0,
            'removed': 0,
            'errors': 0,
            'cropped': 0,
            'normalized': 0,
            'peak_limited': 0,
            'removed_files': [],
            'details': [] if return_details else None
        }

        print(f"\n{'='*60}")
        print(f"AUDIO PREPROCESSING")
        print(f"{'='*60}")
        print(f"Files to process: {len(files)}")
        print(f"Workers: {max_workers}")
        print(f"Target Duration: {self.duration_handler.target_duration}s")
        print(f"Target Loudness: {self.loudness_normalizer.target_lufs} LUFS")
        print(f"Max True Peak: {self.loudness_normalizer.max_true_peak} dBTP")
        print(f"{'='*60}\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, str(f)): f 
                for f in files
            }

            for future in tqdm(as_completed(future_to_file), total=len(files), desc="Preprocessing"):
                res = future.result()
                
                if return_details:
                    stats['details'].append(res)
                
                if res['status'] == 'success':
                    stats['processed'] += 1
                    if 'cropped' in res['actions']:
                        stats['cropped'] += 1
                    if 'normalized' in res['actions']:
                        stats['normalized'] += 1
                    if 'peak_limited' in res['actions']:
                        stats['peak_limited'] += 1
                elif res['status'] == 'removed':
                    stats['removed'] += 1
                    stats['removed_files'].append(res['file'])
                elif res['status'] == 'error':
                    stats['errors'] += 1

        # Print summary
        print(f"\n{'='*60}")
        print(f"PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files:      {stats['total']}")
        print(f"Processed:        {stats['processed']}")
        print(f"  - Cropped:      {stats['cropped']}")
        print(f"  - Normalized:   {stats['normalized']}")
        print(f"  - Peak Limited: {stats['peak_limited']}")
        print(f"Removed (short):  {stats['removed']}")
        print(f"Errors:           {stats['errors']}")
        print(f"{'='*60}\n")

        return stats

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Audio Preprocessing: Crop & Normalize")
    parser.add_argument("--dir", required=True, help="Directory containing audio files")
    parser.add_argument("--lufs", type=float, default=-14.0, help="Target LUFS")
    parser.add_argument("--duration", type=float, default=29.0, help="Target duration in seconds")
    
    args = parser.parse_args()
    
    processor = AudioPreprocessor(
        target_duration=args.duration,
        target_lufs=args.lufs
    )
    processor.process_directory(args.dir)
