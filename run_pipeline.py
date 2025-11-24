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
        million_song_csv = here / "src" / "data_collection" / "millionsong_dataset.csv"
        if not million_song_csv.exists():
            print("❌ ERROR: Million Song Dataset CSV not found!")
            print(f"   Expected: {million_song_csv}")
            print("   Please ensure the dataset CSV exists before downloading.")
            sys.exit(1)
        
        print(f"✓ Found Million Song Dataset: {million_song_csv}")
        
        # Run download
        print("\n📥 Downloading songs from Deezer...")
        print("   - Only songs WITH genres will be downloaded")
        print("   - Failed downloads will be cleaned up automatically")
        print("   - Results saved to: songs_data_with_genre.csv")
        print("   - Audio files saved to: audio_files/")
        print()
        
        run([py, "src/data_collection/deezer-song.py"])
        
        # Validate output
        print("\n" + "="*60)
        print("Validating download results...")
        print("="*60)
        
        csv_path = here / "songs_data_with_genre.csv"
        audio_dir = here / "audio_files"
        
        if not csv_path.exists():
            print("⚠️  WARNING: songs_data_with_genre.csv was not created!")
            print("   Download may have failed. Please check the logs above.")
        else:
            # Count CSV entries
            import csv as csv_module
            csv_count = 0
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    csv_count = sum(1 for _ in csv_module.DictReader(f))
                print(f"✓ CSV created: {csv_count} songs with genres")
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
                    run([py, "cleanup_orphaned_files.py", "--auto-confirm"])
                else:
                    print(f"✓ Perfect match! All {mp3_count} files have CSV entries")
        
        print("="*60)
        print("Download step complete!")
        print("="*60 + "\n")

    if "extract" not in args.skip:
        print("\nExtracting audio features (MFCC, mel-spectrogram, etc.)...")
        run([py, "src/features/extract_features.py"])
        print("\nNOTE: For embedding extraction (OpenL3, CREPE, MERT), run separately:")
        print("      python run_extraction.py")

    if "plot" not in args.skip:
        plot_targets = ["output/results"]
        processed_dir = here / "output" / "processed_results"
        if processed_dir.exists():
            plot_targets.append("output/processed_results")
        else:
            print(f"Skipping plotting for {processed_dir} (directory not found).")

        for target in plot_targets:
            run([py, "scripts/ploting.py", target])

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
