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
        print("Using pre-existing genre data from genres_original folder.")
        # The download step is skipped since we're using pre-existing data
        # run([py, "src/data_collection/playlist_audio_download.py"])
        # run([py, "src/data_collection/deezer-song.py"])

    if "extract" not in args.skip:
        run([py, "src/features/extract_features.py"])

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
