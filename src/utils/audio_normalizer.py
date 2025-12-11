"""
Audio Normalizer - Normalize all audio files to exactly 29 seconds.

This script:
1. Removes songs shorter than 29 seconds
2. Crops songs longer than 29 seconds to exactly 29 seconds (from the end)

Uses parallel processing for faster execution.
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pydub import AudioSegment
from tqdm import tqdm


TARGET_DURATION_MS = 29 * 1000  # 29 seconds in milliseconds


def process_single_file(mp3_file_path: str) -> dict:
    """
    Process a single audio file - normalize to 29 seconds.
    
    Args:
        mp3_file_path: Path to the MP3 file
        
    Returns:
        Dictionary with result info
    """
    mp3_file = Path(mp3_file_path)
    result = {
        'file': mp3_file.name,
        'action': None,
        'original_duration': 0,
        'error': None
    }
    
    try:
        # Load audio file
        audio = AudioSegment.from_mp3(str(mp3_file))
        duration_ms = len(audio)
        duration_sec = duration_ms / 1000
        result['original_duration'] = duration_sec
        
        if duration_ms < TARGET_DURATION_MS:
            # Remove files shorter than 29 seconds
            os.remove(mp3_file)
            result['action'] = 'removed'
            
        elif duration_ms > TARGET_DURATION_MS:
            # Crop files longer than 29 seconds (keep first 29 seconds)
            cropped_audio = audio[:TARGET_DURATION_MS]
            cropped_audio.export(str(mp3_file), format="mp3")
            result['action'] = 'cropped'
            
        else:
            # File is exactly 29 seconds
            result['action'] = 'unchanged'
            
    except Exception as e:
        result['action'] = 'error'
        result['error'] = str(e)
    
    return result


def normalize_audio_files(audio_dir: str, verbose: bool = True, max_workers: int = None) -> dict:
    """
    Normalize all audio files in the directory to exactly 29 seconds using parallel processing.
    
    Args:
        audio_dir: Path to the directory containing audio files
        verbose: Whether to print progress information
        max_workers: Number of parallel workers (default: CPU count)
        
    Returns:
        Dictionary with statistics about the normalization process
    """
    audio_path = Path(audio_dir)
    
    if not audio_path.exists():
        print(f"Error: Directory '{audio_dir}' does not exist!")
        return {'error': 'Directory not found'}
    
    # Get all mp3 files
    mp3_files = list(audio_path.glob("*.mp3"))
    
    if not mp3_files:
        print(f"No MP3 files found in '{audio_dir}'")
        return {'total': 0, 'removed': 0, 'cropped': 0, 'unchanged': 0}
    
    # Set number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), 8)  # Cap at 8 workers
    
    stats = {
        'total': len(mp3_files),
        'removed': 0,
        'cropped': 0,
        'unchanged': 0,
        'errors': 0,
        'removed_files': [],
        'cropped_files': [],
        'error_files': []
    }
    
    if verbose:
        print(f"Processing {len(mp3_files)} audio files with {max_workers} workers...")
        print(f"Target duration: 29 seconds")
        print("-" * 60)
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, str(mp3_file)): mp3_file 
            for mp3_file in mp3_files
        }
        
        # Process results as they complete with progress bar
        pbar = tqdm(as_completed(future_to_file), total=len(future_to_file),
                    desc="Normalizing audio", unit="file")
        for future in pbar:
            result = future.result()
            
            if result['action'] == 'removed':
                stats['removed'] += 1
                stats['removed_files'].append((result['file'], result['original_duration']))
                    
            elif result['action'] == 'cropped':
                stats['cropped'] += 1
                stats['cropped_files'].append((result['file'], result['original_duration']))
                    
            elif result['action'] == 'unchanged':
                stats['unchanged'] += 1
                    
            elif result['action'] == 'error':
                stats['errors'] += 1
                stats['error_files'].append((result['file'], result['error']))
            
            pbar.set_postfix(removed=stats['removed'], cropped=stats['cropped'], ok=stats['unchanged'])
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("NORMALIZATION SUMMARY")
        print("=" * 60)
        print(f"Total files processed: {stats['total']}")
        print(f"Files removed (< 29s): {stats['removed']}")
        print(f"Files cropped (> 29s): {stats['cropped']}")
        print(f"Files unchanged (= 29s): {stats['unchanged']}")
        print(f"Files with errors: {stats['errors']}")
        print(f"Remaining files: {stats['total'] - stats['removed'] - stats['errors']}")
        print("=" * 60)
    
    return stats


def main():
    """Main entry point for standalone execution."""
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    audio_dir = project_root / "audio_files"
    
    print("\n" + "=" * 60)
    print("AUDIO NORMALIZER - Standardizing to 29 seconds")
    print("=" * 60)
    print(f"Audio directory: {audio_dir}")
    print()
    
    stats = normalize_audio_files(str(audio_dir))
    
    return stats


if __name__ == "__main__":
    main()
