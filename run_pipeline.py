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
        choices=["kmeans", "hierarchical", "dbscan"],
        help="The clustering method to use.",
    )
    args = parser.parse_args()

    py = sys.executable 

    here = Path(__file__).resolve().parent
    
    subprocess.run(["python", "-c", "import os,sys; os.chdir(sys.argv[1])" , str(here)])

    if "download" not in args.skip:
        print("Using pre-existing genre data from genres_original folder.")
        # The download step is skipped since we're using pre-existing data
        # run([py, "playlist_audio_download.py"])
        # run([py, "deezer-song.py"])

    if "extract" not in args.skip:
        run([py, "extract_features.py"])

    if "plot" not in args.skip:
        run([py, "ploting.py", "results"])
        run([py, "ploting.py", "processed_results"])

    if "cluster" not in args.skip:
        # Interactive selection if not specified in command line
        if args.clustering_method is None:
            print("\nPlease select a clustering method:")
            print("1. K-Means Clustering")
            print("2. Hierarchical Clustering")
            print("3. DBSCAN Clustering")
            
            while True:
                choice = input("Enter your choice (1, 2, or 3): ")
                if choice == "1":
                    args.clustering_method = "kmeans"
                    break
                elif choice == "2":
                    args.clustering_method = "hierarchical"
                    break
                elif choice == "3":
                    args.clustering_method = "dbscan"
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
        
        if args.clustering_method == "kmeans":
            print("\nUsing K-Means clustering method")
            run([py, "kmeans.py"])
        elif args.clustering_method == "hierarchical":
            print("\nUsing Hierarchical clustering method")
            run([py, "hierarchical_clustering.py"])
        elif args.clustering_method == "dbscan":
            print("\nUsing DBSCAN clustering method")
            run([py, "dbscan_clustering.py"])
        else:
            print(f"\nPlease select a valid clustering method: kmeans, hierarchical, or dbscan")
            print("Using default K-Means clustering method")
            run([py, "kmeans.py"])


if __name__ == "__main__":
    main()
