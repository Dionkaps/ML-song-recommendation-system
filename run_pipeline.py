#!/usr/bin/env python3
"""
run_pipeline.py — One‑command orchestrator for your audio analysis workflow.

Place this file in the same directory as the other scripts and run:
    python run_pipeline.py

It will perform, in sequence:
  1. Download audio from the URLs in links.txt → audio_files/
  2. Extract audio features → results/
  3. Post‑process (outlier removal, scaling, log) → processed_results/
  4. Plot features for both raw and processed results → results/ & processed_results/
  5. K‑means clustering + visualisation → audio_clustering_results.csv

You can skip any step with the --skip flag, e.g.:
    python run_pipeline.py --skip download plot

Dependencies: yt_dlp, librosa, numpy, scikit‑learn, matplotlib, pandas, scipy.
"""

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

    py = sys.executable  # Absolute path to the current Python interpreter

    here = Path(__file__).resolve().parent
    # Change working dir to script location so relative paths work
    # (Useful if you call this from another folder)
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
