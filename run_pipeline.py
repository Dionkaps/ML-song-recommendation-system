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
        # Adaptive loudness normalization with scan pass
        print("\n" + "="*60)
        print("AUDIO PREPROCESSING: Adaptive Loudness Normalization")
        print("="*60)
        print("   - Step 1: Scan all files for LUFS and true peak")
        print("   - Step 2: Analyze distribution and select optimal target")
        print("   - Step 3: Apply consistent normalization with peak limiting")
        print("   - Step 4: Verify and generate report")
        print()
        
        import csv as csv_module
        import os
        import warnings
        import logging
        from tqdm import tqdm
        import librosa
        import pyloudnorm as pyln
        import numpy as np
        from src.audio_preprocessing.processor import AudioPreprocessor
        from src.audio_preprocessing.loudness_scanner import LoudnessScanner, save_scan_results
        
        # Suppress all audio-related warnings (ID3v2, mpg123, audioread, etc.)
        warnings.filterwarnings("ignore")
        logging.getLogger("audioread").setLevel(logging.ERROR)
        logging.getLogger("librosa").setLevel(logging.ERROR)
        
        # Suppress C-level warnings from mpg123 by redirecting stderr temporarily
        import io
        
        class SuppressStderr:
            """Context manager to suppress stderr (catches C-level warnings from mpg123)"""
            def __enter__(self):
                self._stderr = sys.stderr
                sys.stderr = io.StringIO()
                return self
            def __exit__(self, *args):
                sys.stderr = self._stderr
        
        audio_dir = here / "audio_files"
        csv_path = here / "data" / "songs.csv"
        report_path = here / "output" / "preprocessing_report.csv"
        scan_path = here / "output" / "loudness_scan.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ===== STEP 1: SCAN PASS =====
        print("📊 Step 1/4: Scanning all files for LUFS and true peak...")
        scanner = LoudnessScanner(sample_rate=22050)
        scan_results = scanner.scan_directory(str(audio_dir))
        
        if len(scan_results) == 0:
            print("❌ No audio files found. Skipping preprocessing.")
        else:
            # ===== STEP 2: ANALYZE AND SELECT TARGET =====
            print("\n🎯 Step 2/4: Analyzing distribution and selecting target...")
            analysis = scanner.analyze_distribution(
                scan_results,
                headroom_db=-1.0,  # EBU R128 compliant
                target_coverage=0.95  # 95% of tracks reach target without limiting
            )
            scanner.print_analysis(analysis)
            
            # Save scan results for reference
            save_scan_results(scan_results, analysis, str(scan_path))
            
            target_lufs = analysis.suggested_target_lufs
            
            # Store before measurements from scan
            before_measurements = {}
            for _, row in scan_results.iterrows():
                before_measurements[row['filename']] = {
                    'duration': row['duration_seconds'],
                    'lufs': row['integrated_lufs'],
                    'peak_db': row['true_peak_dbfs'],
                    'error': row.get('error')
                }
            
            # ===== STEP 3: NORMALIZE =====
            print(f"\n🔧 Step 3/4: Normalizing to {target_lufs} LUFS (adaptive target)...")
            processor = AudioPreprocessor(
                target_duration=29.0,
                target_lufs=target_lufs,
                max_true_peak=-1.0
            )
            
            normalize_stats = processor.process_directory(str(audio_dir))
            
            # ===== STEP 4: VERIFY AND REPORT =====
            print("\n📊 Step 4/4: Verifying results and generating report...")
            audio_files_after = sorted(audio_dir.glob("*.mp3"))
            after_measurements = {}
            meter = pyln.Meter(22050)  # Sample rate for measurement
            
            for audio_file in tqdm(audio_files_after, desc="Verifying", leave=False):
                try:
                    # Load audio (suppress C-level warnings from mpg123)
                    with SuppressStderr():
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
            
            print(f"✅ Verified {len(after_measurements)} files")
            
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
            # Collect all measurements
            before_lufs_list = []
            after_lufs_list = []
            before_peak_list = []
            after_peak_list = []
            
            for f in processed_files:
                if f in after_measurements and after_measurements[f].get('lufs') and after_measurements[f]['lufs'] > -np.inf:
                    after_lufs_list.append(after_measurements[f]['lufs'])
                    after_peak_list.append(after_measurements[f].get('peak_db', 0))
                if f in before_measurements and before_measurements[f].get('lufs') and before_measurements[f]['lufs'] > -np.inf:
                    before_lufs_list.append(before_measurements[f]['lufs'])
                    before_peak_list.append(before_measurements[f].get('peak_db', 0))
            
            # target_lufs already set from analysis.suggested_target_lufs
            within_target = sum(1 for l in after_lufs_list if abs(l - target_lufs) <= 0.5)
            
            print("\n")
            print("╔" + "═"*78 + "╗")
            print("║" + " "*28 + "📊 PREPROCESSING STATISTICS" + " "*23 + "║")
            print("╚" + "═"*78 + "╝")
            
            # ===== BEFORE vs AFTER COMPARISON =====
            print("\n┌" + "─"*76 + "┐")
            print("│  📈 BEFORE vs AFTER COMPARISON" + " "*44 + "│")
            print("├" + "─"*76 + "┤")
            print("│" + " "*76 + "│")
            print(f"│  {'Metric':<25}  {'BEFORE':<20}  {'AFTER':<20}    │")
            print("│  " + "─"*25 + "  " + "─"*20 + "  " + "─"*20 + "    │")
            
            if before_lufs_list and after_lufs_list:
                before_mean = np.mean(before_lufs_list)
                before_std = np.std(before_lufs_list)
                after_mean = np.mean(after_lufs_list)
                after_std = np.std(after_lufs_list)
                
                print(f"│  {'Loudness (Mean)':<25}  {before_mean:>6.1f} LUFS          {after_mean:>6.1f} LUFS          │")
                print(f"│  {'Loudness (Std Dev)':<25}  {before_std:>6.1f} LUFS          {after_std:>6.1f} LUFS          │")
                print(f"│  {'Loudness (Min)':<25}  {np.min(before_lufs_list):>6.1f} LUFS          {np.min(after_lufs_list):>6.1f} LUFS          │")
                print(f"│  {'Loudness (Max)':<25}  {np.max(before_lufs_list):>6.1f} LUFS          {np.max(after_lufs_list):>6.1f} LUFS          │")
                
                before_peak_valid = [p for p in before_peak_list if p and p > -np.inf]
                after_peak_valid = [p for p in after_peak_list if p and p > -np.inf]
                if before_peak_valid and after_peak_valid:
                    print("│" + " "*76 + "│")
                    print(f"│  {'Peak Level (Mean)':<25}  {np.mean(before_peak_valid):>6.1f} dBFS         {np.mean(after_peak_valid):>6.1f} dBFS         │")
                    print(f"│  {'Peak Level (Max)':<25}  {np.max(before_peak_valid):>6.1f} dBFS         {np.max(after_peak_valid):>6.1f} dBFS         │")
            
            print("│" + " "*76 + "│")
            print("└" + "─"*76 + "┘")
            
            # ===== NORMALIZATION SUMMARY =====
            print("\n┌" + "─"*76 + "┐")
            print(f"│  🎯 NORMALIZATION SUMMARY (Target: {target_lufs} LUFS)" + " "*(45-len(f"(Target: {target_lufs} LUFS)")) + "│")
            print("├" + "─"*76 + "┤")
            
            total_processed = len(after_lufs_list)
            at_target = sum(1 for l in after_lufs_list if abs(l - target_lufs) <= 0.5)
            above_target = sum(1 for l in after_lufs_list if l > target_lufs + 0.5)
            below_target = sum(1 for l in after_lufs_list if l < target_lufs - 0.5)
            
            # Visual bar chart
            at_pct = at_target / total_processed * 100 if total_processed > 0 else 0
            above_pct = above_target / total_processed * 100 if total_processed > 0 else 0
            below_pct = below_target / total_processed * 100 if total_processed > 0 else 0
            
            print("│" + " "*76 + "│")
            print(f"│  Total processed:  {total_processed:,}" + " "*(56-len(f"{total_processed:,}")) + "│")
            print("│" + " "*76 + "│")
            
            bar_width = 40
            at_bar = "█" * int(at_pct / 100 * bar_width)
            below_bar = "█" * int(below_pct / 100 * bar_width)
            
            print(f"│  ✅ At target (±0.5 dB):    {at_bar:<{bar_width}}  {at_target:>5} ({at_pct:>5.1f}%)  │")
            print(f"│  ⬇️  Below target:          {below_bar:<{bar_width}}  {below_target:>5} ({below_pct:>5.1f}%)  │")
            if above_target > 0:
                above_bar = "█" * int(above_pct / 100 * bar_width)
                print(f"│  ⬆️  Above target:          {above_bar:<{bar_width}}  {above_target:>5} ({above_pct:>5.1f}%)  │")
            print("│" + " "*76 + "│")
            print("└" + "─"*76 + "┘")
            
            # ===== PEAK-LIMITED TRACKS (GAIN CAPPING) =====
            lufs_deviations = []
            peak_limited_files = []
            
            for f in processed_files:
                if f in after_measurements and after_measurements[f].get('lufs') and after_measurements[f]['lufs'] > -np.inf:
                    final_lufs = after_measurements[f]['lufs']
                    deviation = final_lufs - target_lufs
                    
                    if deviation < -0.5:
                        lufs_deviations.append(deviation)
                        peak_limited_files.append({
                            'filename': f,
                            'final_lufs': final_lufs,
                            'deviation': deviation,
                            'before_lufs': before_measurements.get(f, {}).get('lufs', None),
                            'before_peak': before_measurements.get(f, {}).get('peak_db', None)
                        })
            
            peak_limited_count = len(peak_limited_files)
            peak_limited_pct = (peak_limited_count / total_processed * 100) if total_processed > 0 else 0
            
            print("\n┌" + "─"*76 + "┐")
            print("│  🎚️  GAIN CAPPING ANALYSIS (Tracks that couldn't reach target)" + " "*13 + "│")
            print("├" + "─"*76 + "┤")
            print("│" + " "*76 + "│")
            print(f"│  Peak-limited tracks:  {peak_limited_count:,} / {total_processed:,} ({peak_limited_pct:.1f}%)" + " "*(43-len(f"{peak_limited_count:,} / {total_processed:,} ({peak_limited_pct:.1f}%)")) + "│")
            
            if peak_limited_count > 0:
                avg_deviation = np.mean(lufs_deviations)
                min_deviation = np.min(lufs_deviations)  # Most negative = furthest from target
                max_deviation = np.max(lufs_deviations)  # Least negative = closest to target
                median_deviation = np.median(lufs_deviations)
                
                print("│" + " "*76 + "│")
                print(f"│  Average final LUFS:   {target_lufs + avg_deviation:.1f} LUFS (deviation: {avg_deviation:+.2f} dB)" + " "*24 + "│")
                print(f"│  Median final LUFS:    {target_lufs + median_deviation:.1f} LUFS (deviation: {median_deviation:+.2f} dB)" + " "*24 + "│")
                print(f"│  Quietest track:       {target_lufs + min_deviation:.1f} LUFS (deviation: {min_deviation:+.2f} dB)" + " "*24 + "│")
                
                # Distribution by deviation ranges
                print("│" + " "*76 + "│")
                print("│  📊 Distribution by LUFS deviation from target:" + " "*27 + "│")
                print("│" + " "*76 + "│")
                
                deviation_ranges = [
                    (-0.5, -1.0, "-0.5 to -1.0 dB", "Very close"),
                    (-1.0, -2.0, "-1.0 to -2.0 dB", "Slightly below"),
                    (-2.0, -3.0, "-2.0 to -3.0 dB", "Moderately below"),
                    (-3.0, -5.0, "-3.0 to -5.0 dB", "Noticeably below"),
                    (-5.0, -10.0, "-5.0 to -10.0 dB", "Significantly below"),
                    (-10.0, -np.inf, "< -10.0 dB", "Very loud originals")
                ]
                
                for lower, upper, label, desc in deviation_ranges:
                    if upper == -np.inf:
                        count = sum(1 for d in lufs_deviations if d < lower)
                    else:
                        count = sum(1 for d in lufs_deviations if upper <= d < lower)
                    if count > 0:
                        pct = count / peak_limited_count * 100
                        bar_len = int(pct / 100 * 25)
                        bar = "█" * bar_len + "░" * (25 - bar_len)
                        print(f"│     {label:<18}  {bar}  {count:>4} ({pct:>5.1f}%)     │")
                
                # Top 5 most affected tracks (more concise)
                print("│" + " "*76 + "│")
                print("│  🔝 Top 5 most affected tracks:" + " "*43 + "│")
                sorted_limited = sorted(peak_limited_files, key=lambda x: x['deviation'])[:5]
                for i, track in enumerate(sorted_limited, 1):
                    name = track['filename'][:45] + "..." if len(track['filename']) > 48 else track['filename']
                    print(f"│     {i}. {name:<50} {track['final_lufs']:>5.1f} LUFS  │")
            else:
                print("│" + " "*76 + "│")
                print("│  ✅ All tracks successfully reached the target loudness!" + " "*18 + "│")
            
            print("│" + " "*76 + "│")
            print("└" + "─"*76 + "┘")
            print()
        
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
