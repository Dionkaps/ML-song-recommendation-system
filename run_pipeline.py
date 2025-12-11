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
        choices=["download", "extract", "process", "plot", "cluster"],
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
        
        # Normalize audio files to exactly 29 seconds
        print("\n" + "="*60)
        print("AUDIO NORMALIZATION: Standardizing to 29 seconds")
        print("="*60)
        print("   - Removing songs shorter than 29 seconds")
        print("   - Cropping songs longer than 29 seconds")
        print()
        
        from src.utils.audio_normalizer import normalize_audio_files
        normalize_stats = normalize_audio_files(str(audio_dir))
        
        # Update CSV to remove entries for deleted files
        if normalize_stats.get('removed', 0) > 0 and csv_path.exists():
            print("\n🧹 Updating CSV to remove entries for deleted audio files...")
            removed_names = {name for name, _ in normalize_stats.get('removed_files', [])}
            
            # Read existing CSV
            import csv as csv_module
            rows_to_keep = []
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv_module.DictReader(f)
                    fieldnames = reader.fieldnames
                    for row in reader:
                        # Check if the audio file still exists
                        filename = f"{row.get('artist', '')} - {row.get('title', '')}.mp3"
                        if filename not in removed_names:
                            rows_to_keep.append(row)
                
                # Write updated CSV
                with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv_module.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows_to_keep)
                
                print(f"✓ CSV updated: {len(rows_to_keep)} entries remaining")
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
