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
    args = parser.parse_args()

    py = sys.executable 

    here = Path(__file__).resolve().parent
    
    subprocess.run(["python", "-c", "import os,sys; os.chdir(sys.argv[1])" , str(here)])

    if "download" not in args.skip:
        run([py, "playlist_audio_download.py"])

    if "extract" not in args.skip:
        run([py, "extract_features.py"])

    if "process" not in args.skip:
        run([py, "process_features.py"])

    if "plot" not in args.skip:
        run([py, "ploting.py", "results"])
        run([py, "ploting.py", "processed_results"])

    if "cluster" not in args.skip:
        run([py, "kmeans.py"])


if __name__ == "__main__":
    main()
