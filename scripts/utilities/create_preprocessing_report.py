"""
Create a comprehensive before/after CSV report for audio preprocessing.

This script:
1. Measures original audio characteristics (LUFS, duration, peak)
2. Runs preprocessing
3. Measures processed audio characteristics
4. Creates a CSV with before/after comparison
"""

import sys
import os
from pathlib import Path
import csv
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import librosa
import numpy as np
from tqdm import tqdm
import pyloudnorm as pyln

from src.audio_preprocessing import AudioPreprocessor


def measure_audio_stats(file_path, sample_rate=22050):
    """Measure audio characteristics without modifying the file."""
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        
        # Duration
        duration = len(audio) / sr
        
        # LUFS
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(audio)
        
        # Peak
        peak_linear = np.abs(audio).max()
        peak_db = 20 * np.log10(peak_linear) if peak_linear > 0 else -np.inf
        
        # True Peak (simple approximation)
        true_peak_db = peak_db + 0.5  # Approximate overhead
        
        return {
            'duration': round(duration, 3),
            'lufs': round(lufs, 2) if np.isfinite(lufs) else None,
            'peak_db': round(peak_db, 2) if np.isfinite(peak_db) else None,
            'true_peak_db': round(true_peak_db, 2) if np.isfinite(true_peak_db) else None,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'duration': None,
            'lufs': None,
            'peak_db': None,
            'true_peak_db': None,
            'success': False,
            'error': str(e)
        }


def create_preprocessing_report(audio_dir, output_csv):
    """Create before/after preprocessing report."""
    
    audio_dir = Path(audio_dir)
    output_csv = Path(output_csv)
    
    print("=" * 80)
    print("AUDIO PREPROCESSING REPORT GENERATOR")
    print("=" * 80)
    print(f"\nAudio directory: {audio_dir}")
    print(f"Output CSV: {output_csv}")
    print()
    
    # Get all audio files
    audio_files = sorted(audio_dir.glob("*.mp3"))
    
    if not audio_files:
        print("❌ No MP3 files found!")
        return
    
    print(f"📂 Found {len(audio_files)} audio files")
    print()
    
    # Step 1: Measure BEFORE preprocessing
    print("📊 Step 1/3: Measuring original audio characteristics...")
    before_stats = {}
    
    for file_path in tqdm(audio_files, desc="Analyzing"):
        stats = measure_audio_stats(file_path)
        before_stats[file_path.name] = stats
    
    print(f"✅ Measured {len(before_stats)} files")
    print()
    
    # Step 2: Run preprocessing
    print("🔧 Step 2/3: Running audio preprocessing...")
    print("   Target: 29s duration, -14 LUFS, -1.0 dBTP max peak")
    print()
    
    preprocessor = AudioPreprocessor(
        target_duration=29.0,
        target_lufs=-14.0,
        max_true_peak=-1.0,
        sample_rate=22050
    )
    
    processing_results = preprocessor.process_directory(str(audio_dir))
    
    print()
    print(f"✅ Processed: {processing_results.get('successful', 0)} files")
    print(f"❌ Errors: {processing_results.get('failed', 0)} files")
    print(f"🗑️  Removed (too short): {processing_results.get('removed', 0)} files")
    print()
    
    # Step 3: Measure AFTER preprocessing
    print("📊 Step 3/3: Measuring processed audio characteristics...")
    after_stats = {}
    
    # Only measure files that still exist (not removed)
    remaining_files = sorted(audio_dir.glob("*.mp3"))
    
    for file_path in tqdm(remaining_files, desc="Analyzing"):
        stats = measure_audio_stats(file_path)
        after_stats[file_path.name] = stats
    
    print(f"✅ Measured {len(after_stats)} files")
    print()
    
    # Step 4: Create CSV report
    print("💾 Creating CSV report...")
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'filename',
            'status',
            'before_duration_s',
            'after_duration_s',
            'duration_change_s',
            'before_lufs',
            'after_lufs',
            'lufs_change',
            'before_peak_db',
            'after_peak_db',
            'peak_change_db',
            'before_true_peak_db',
            'after_true_peak_db',
            'gain_applied_db',
            'was_cropped',
            'was_removed',
            'error'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write data for all original files
        for filename in sorted(before_stats.keys()):
            before = before_stats[filename]
            after = after_stats.get(filename, {})
            
            # Determine status
            if filename not in after_stats:
                status = 'REMOVED'
                was_removed = True
            elif not before.get('success'):
                status = 'ERROR_BEFORE'
                was_removed = False
            elif not after.get('success'):
                status = 'ERROR_AFTER'
                was_removed = False
            else:
                status = 'SUCCESS'
                was_removed = False
            
            # Calculate changes
            duration_change = None
            if before.get('duration') and after.get('duration'):
                duration_change = round(after['duration'] - before['duration'], 3)
            
            lufs_change = None
            if before.get('lufs') and after.get('lufs'):
                lufs_change = round(after['lufs'] - before['lufs'], 2)
            
            peak_change = None
            if before.get('peak_db') and after.get('peak_db'):
                peak_change = round(after['peak_db'] - before['peak_db'], 2)
            
            gain_applied = lufs_change  # Gain applied is approximately the LUFS change
            
            was_cropped = abs(duration_change) > 0.1 if duration_change is not None else False
            
            row = {
                'filename': filename,
                'status': status,
                'before_duration_s': before.get('duration'),
                'after_duration_s': after.get('duration'),
                'duration_change_s': duration_change,
                'before_lufs': before.get('lufs'),
                'after_lufs': after.get('lufs'),
                'lufs_change': lufs_change,
                'before_peak_db': before.get('peak_db'),
                'after_peak_db': after.get('peak_db'),
                'peak_change_db': peak_change,
                'before_true_peak_db': before.get('true_peak_db'),
                'after_true_peak_db': after.get('true_peak_db'),
                'gain_applied_db': gain_applied,
                'was_cropped': was_cropped,
                'was_removed': was_removed,
                'error': before.get('error') or after.get('error')
            }
            
            writer.writerow(row)
    
    print(f"✅ CSV report saved to: {output_csv}")
    print()
    
    # Print summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    successful_files = [f for f in before_stats.keys() if f in after_stats]
    
    if successful_files:
        # Duration stats
        durations_before = [before_stats[f]['duration'] for f in successful_files if before_stats[f].get('duration')]
        durations_after = [after_stats[f]['duration'] for f in successful_files if after_stats[f].get('duration')]
        
        print(f"\n📏 Duration:")
        print(f"   Before: {np.mean(durations_before):.1f}s ± {np.std(durations_before):.1f}s (range: {np.min(durations_before):.1f}-{np.max(durations_before):.1f}s)")
        print(f"   After:  {np.mean(durations_after):.1f}s ± {np.std(durations_after):.1f}s (range: {np.min(durations_after):.1f}-{np.max(durations_after):.1f}s)")
        
        # LUFS stats
        lufs_before = [before_stats[f]['lufs'] for f in successful_files if before_stats[f].get('lufs')]
        lufs_after = [after_stats[f]['lufs'] for f in successful_files if after_stats[f].get('lufs')]
        
        print(f"\n🔊 Loudness (LUFS):")
        print(f"   Before: {np.mean(lufs_before):.1f} ± {np.std(lufs_before):.1f} LUFS (range: {np.min(lufs_before):.1f} to {np.max(lufs_before):.1f} LUFS)")
        print(f"   After:  {np.mean(lufs_after):.1f} ± {np.std(lufs_after):.1f} LUFS (range: {np.min(lufs_after):.1f} to {np.max(lufs_after):.1f} LUFS)")
        print(f"   Target: -14.0 LUFS")
        
        # Files at target
        at_target = sum(1 for lufs in lufs_after if abs(lufs - (-14.0)) < 0.5)
        print(f"   Files within ±0.5 LUFS of target: {at_target}/{len(lufs_after)} ({100*at_target/len(lufs_after):.1f}%)")
    
    print()
    print("=" * 80)
    print("✅ Report generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create before/after preprocessing CSV report")
    parser.add_argument("--audio-dir", default="audio_files", help="Audio files directory")
    parser.add_argument("--output", default="output/preprocessing_report.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    audio_dir = project_root / args.audio_dir
    output_csv = project_root / args.output
    
    create_preprocessing_report(audio_dir, output_csv)
