import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Helper to execute a command and exit if it fails

def run(cmd: List[str]):
    print("\n$", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full audio analysis pipeline in one go.")
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=["download", "preprocess", "extract", "process", "plot", "cluster"],
        default=[],
        help="Names of steps to skip (space‑separated).",
    )
    parser.add_argument(
        "--clustering-method",
        choices=["kmeans", "hdbscan", "gmm", "vade"],
        default="kmeans",
        help="The clustering method to use (K-Means, HDBSCAN, GMM, or VaDE).",
    )
    args = parser.parse_args()

    py = sys.executable 

    here = Path(__file__).resolve().parent

    if "download" not in args.skip:
        print("\n" + "="*60)
        print("DOWNLOAD STEP: Songs with Genre Information")
        print("="*60)
        
        # Check for Million Song Dataset CSV
        million_song_csv = here / "data" / "millionsong_dataset.csv"
        if not million_song_csv.exists():
            # Check for tar.gz
            tar_path = here / "data" / "millionsongsubset.tar.gz"
            if tar_path.exists():
                print(f"✓ Found Million Song Dataset archive: {tar_path}")
                print("   Extracting dataset and generating CSV...")
                run([py, "src/data_collection/extract_millionsong_dataset.py"])
                
                # Re-check
                if not million_song_csv.exists():
                     print("❌ ERROR: Extraction failed to create CSV!")
                     sys.exit(1)
            else:
                print("❌ ERROR: Million Song Dataset CSV not found!")
                print(f"   Expected: {million_song_csv}")
                print("   OR")
                print(f"   Archive: {tar_path}")
                print("   Please ensure the dataset exists before downloading.")
                sys.exit(1)
        
        print(f"✓ Found Million Song Dataset: {million_song_csv}")
        
        # Run download
        print("\n📥 Downloading songs from Deezer...")
        print("   - Only songs WITH genres will be downloaded")
        print("   - Failed downloads will be cleaned up automatically")
        print("   - Results saved to unified: data/songs.csv")
        print("   - Audio files saved to: audio_files/")
        print()
        
        run([py, "src/data_collection/deezer-song.py"])
        
        # Validate output
        print("\n" + "="*60)
        print("Validating download results...")
        print("="*60)
        
        # Check unified CSV (primary) or legacy CSV
        csv_path = here / "data" / "songs.csv"
        legacy_csv_path = here / "data" / "songs_data_with_genre.csv"
        audio_dir = here / "audio_files"
        
        active_csv = csv_path if csv_path.exists() else legacy_csv_path
        
        if not active_csv.exists():
            print("⚠️  WARNING: songs.csv was not created!")
            print("   Download may have failed. Please check the logs above.")
        else:
            # Count CSV entries
            import csv as csv_module
            csv_count = 0
            try:
                with open(active_csv, 'r', encoding='utf-8') as f:
                    csv_count = sum(1 for _ in csv_module.DictReader(f))
                print(f"✓ CSV created: {csv_count} songs in {active_csv.name}")
            except Exception as e:
                print(f"⚠️  Error reading CSV: {e}")
        
        if not audio_dir.exists():
            print("⚠️  WARNING: audio_files/ directory not found!")
        else:
            # Count MP3 files
            mp3_files = list(audio_dir.glob("*.mp3"))
            mp3_count = len(mp3_files)
            print(f"✓ Audio files: {mp3_count} MP3 files")
            
            # Check for mismatch
            if csv_path.exists() and csv_count > 0:
                diff = abs(mp3_count - csv_count)
                if diff > 0:
                    print(f"\n⚠️  MISMATCH DETECTED: {diff} orphaned files")
                    print(f"   MP3 files: {mp3_count}")
                    print(f"   CSV entries: {csv_count}")
                    print("\n🧹 Running automatic cleanup...")
                    cleanup_script = Path("scripts/utilities/cleanup_orphaned_files.py")
                    run([py, str(cleanup_script), "--auto-confirm"])
                else:
                    print(f"✓ Perfect match! All {mp3_count} files have CSV entries")
        
        print("="*60)
        print("Download step complete!")
        print("="*60 + "\n")
        
    if "preprocess" not in args.skip:
        # Normalize audio files to exactly 29 seconds
        print("\n" + "="*60)
        print("AUDIO PREPROCESSING: Cropping & Loudness Normalization")
        print("="*60)
        print("   - Removing songs shorter than 29 seconds")
        print("   - Cropping songs longer than 29 seconds")
        print("   - Normalizing loudness to -14 LUFS (ITU-R BS.1770)")
        print("   - Limiting True Peak to -1.0 dBTP")
        print("   - Generating before/after CSV report")
        print()
        
        import csv as csv_module
        from tqdm import tqdm
        import librosa
        import pyloudnorm as pyln
        import numpy as np
        from src.audio_preprocessing.processor import AudioPreprocessor
        
        audio_dir = here / "audio_files"
        csv_path = here / "data" / "songs.csv"
        report_path = here / "output" / "preprocessing_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Measure BEFORE preprocessing
        print("📊 Step 1/3: Measuring original audio characteristics...")
        audio_files = sorted(audio_dir.glob("*.mp3"))
        before_measurements = {}
        
        meter = pyln.Meter(22050)  # Sample rate for measurement
        
        for audio_file in tqdm(audio_files, desc="Analyzing"):
            try:
                # Load audio
                y, sr = librosa.load(str(audio_file), sr=22050, mono=True)
                duration = len(y) / sr
                
                # Measure loudness
                if len(y) > 0:
                    loudness = meter.integrated_loudness(y)
                    peak_db = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else -np.inf
                else:
                    loudness = -np.inf
                    peak_db = -np.inf
                
                before_measurements[audio_file.name] = {
                    'duration': duration,
                    'lufs': loudness,
                    'peak_db': peak_db
                }
            except Exception as e:
                before_measurements[audio_file.name] = {
                    'duration': None,
                    'lufs': None,
                    'peak_db': None,
                    'error': str(e)
                }
        
        print(f"✅ Measured {len(before_measurements)} files")
        
        # Step 2: Run preprocessing
        print("\n🔧 Step 2/3: Running preprocessing pipeline...")
        processor = AudioPreprocessor(
            target_duration=29.0,
            target_lufs=-14.0,
            max_true_peak=-1.0
        )
        
        normalize_stats = processor.process_directory(str(audio_dir))
        
        # Step 3: Measure AFTER preprocessing and create report
        print("\n📊 Step 3/3: Measuring processed audio and generating report...")
        audio_files_after = sorted(audio_dir.glob("*.mp3"))
        after_measurements = {}
        
        for audio_file in tqdm(audio_files_after, desc="Analyzing"):
            try:
                # Load audio
                y, sr = librosa.load(str(audio_file), sr=22050, mono=True)
                duration = len(y) / sr
                
                # Measure loudness
                if len(y) > 0:
                    loudness = meter.integrated_loudness(y)
                    peak_db = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else -np.inf
                else:
                    loudness = -np.inf
                    peak_db = -np.inf
                
                after_measurements[audio_file.name] = {
                    'duration': duration,
                    'lufs': loudness,
                    'peak_db': peak_db
                }
            except Exception as e:
                after_measurements[audio_file.name] = {
                    'duration': None,
                    'lufs': None,
                    'peak_db': None,
                    'error': str(e)
                }
        
        print(f"✅ Measured {len(after_measurements)} files")
        
        # Create CSV report
        print("\n💾 Creating CSV report...")
        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv_module.writer(f)
            writer.writerow([
                'filename', 'status', 
                'before_duration_s', 'after_duration_s', 'duration_change_s',
                'before_lufs', 'after_lufs', 'lufs_change',
                'before_peak_db', 'after_peak_db', 'peak_change_db',
                'gain_applied_db', 'was_cropped', 'was_removed', 'error'
            ])
            
            # Combine all filenames
            all_files = set(before_measurements.keys()) | set(after_measurements.keys())
            removed_files = set(normalize_stats.get('removed_files', []))
            
            for filename in sorted(all_files):
                before = before_measurements.get(filename, {})
                after = after_measurements.get(filename, {})
                
                # Determine status
                if filename in removed_files:
                    status = 'removed'
                elif after:
                    status = 'processed'
                else:
                    status = 'error'
                
                # Calculate changes
                duration_change = (after.get('duration', 0) or 0) - (before.get('duration', 0) or 0)
                lufs_change = (after.get('lufs', 0) or 0) - (before.get('lufs', 0) or 0)
                peak_change = (after.get('peak_db', 0) or 0) - (before.get('peak_db', 0) or 0)
                
                # Gain applied is approximately the LUFS change
                gain_applied = lufs_change if status == 'processed' else None
                
                # Was cropped if duration decreased significantly
                was_cropped = duration_change < -0.1 if status == 'processed' else False
                
                writer.writerow([
                    filename, status,
                    f"{before.get('duration', 0):.2f}" if before.get('duration') else '',
                    f"{after.get('duration', 0):.2f}" if after.get('duration') else '',
                    f"{duration_change:.2f}" if status == 'processed' else '',
                    f"{before.get('lufs', 0):.1f}" if before.get('lufs') and before.get('lufs') > -np.inf else '',
                    f"{after.get('lufs', 0):.1f}" if after.get('lufs') and after.get('lufs') > -np.inf else '',
                    f"{lufs_change:.1f}" if status == 'processed' and gain_applied else '',
                    f"{before.get('peak_db', 0):.2f}" if before.get('peak_db') and before.get('peak_db') > -np.inf else '',
                    f"{after.get('peak_db', 0):.2f}" if after.get('peak_db') and after.get('peak_db') > -np.inf else '',
                    f"{peak_change:.2f}" if status == 'processed' else '',
                    f"{gain_applied:.1f}" if gain_applied else '',
                    'yes' if was_cropped else 'no',
                    'yes' if filename in removed_files else 'no',
                    before.get('error', '') or after.get('error', '')
                ])
        
        print(f"✅ CSV report saved to: {report_path}")
        
        # Print summary statistics
        processed_files = [f for f in all_files if f in after_measurements and after_measurements[f].get('lufs')]
        if processed_files:
            after_lufs = [after_measurements[f]['lufs'] for f in processed_files if after_measurements[f]['lufs'] > -np.inf]
            target_lufs = -14.0
            within_target = sum(1 for l in after_lufs if abs(l - target_lufs) <= 0.5)
            
            print("\n" + "="*80)
            print("SUMMARY STATISTICS")
            print("="*80)
            print(f"\n🔊 Loudness (LUFS):")
            if after_lufs:
                print(f"   After:  {np.mean(after_lufs):.1f} ± {np.std(after_lufs):.1f} LUFS")
                print(f"   Target: {target_lufs} LUFS")
                print(f"   Files within ±0.5 LUFS of target: {within_target}/{len(after_lufs)} ({100*within_target/len(after_lufs):.1f}%)")
            print("\n" + "="*80)
        
        # Update CSV to remove entries for deleted files
        if normalize_stats.get('removed', 0) > 0 and csv_path.exists():
            print("\n🧹 Updating CSV to remove entries for deleted audio files...")
            # removed_files is a list of filenames
            removed_names = set(normalize_stats.get('removed_files', []))
            
            # Read existing CSV
            rows_to_keep = []
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv_module.DictReader(f)
                    fieldnames = reader.fieldnames
                    for row in reader:
                        filename_from_row = f"{row.get('artist', '')} - {row.get('title', '')}.mp3"
                        # If there is a 'filename' column, use it.
                        if 'filename' in row and row['filename']:
                             filename_from_row = row['filename']
                        
                        if filename_from_row not in removed_names:
                            rows_to_keep.append(row)
                
                # Write updated CSV
                with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv_module.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows_to_keep)
                    
                print(f"✓ Removed {len(removed_names)} entries from CSV")
            except Exception as e:
                print(f"⚠️  Error updating CSV: {e}")
        
        print("="*60)
        print("Audio normalization complete!")
        print("="*60 + "\n")

    if "extract" not in args.skip:
        print("\nExtracting audio features (MFCC, mel-spectrogram, etc.)...")
        run([py, "src/features/extract_features.py"])
        print("\nNOTE: For embedding extraction (OpenL3, CREPE, MERT), run separately:")
        print("      python run_extraction.py")

    if "plot" not in args.skip:
        print("\nGenerating plots from extracted features...")
        run([py, "scripts/visualization/ploting.py", "--features_dir", "output/features", "--plots_dir", "output/plots"])

    if "cluster" not in args.skip:
        if args.clustering_method == "hdbscan":
            print("\nUsing HDBSCAN clustering method")
            run([py, "src/clustering/hdbscan.py"])
        elif args.clustering_method == "gmm":
            print("\nUsing GMM clustering method")
            run([py, "src/clustering/gmm.py"])
        elif args.clustering_method == "vade":
            print("\nUsing VaDE (Variational Deep Embedding) clustering method")
            run([py, "src/clustering/vade.py"])
        else:
            print("\nUsing K-Means clustering method")
            run([py, "src/clustering/kmeans.py"])


if __name__ == "__main__":
    main()
