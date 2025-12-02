"""
Audio Feature Extraction Pipeline - Main Entry Point

This script coordinates feature extraction using three pretrained audio models:
    1. EnCodecMAE - Self-supervised audio encoder (modern, Python 3.12)
    2. MusiCNN    - Music tagging CNN (legacy, Python 3.7)
    3. MERT       - Music understanding transformer (modern, Python 3.12)

IMPORTANT: Due to dependency conflicts, the models require different Python environments:

    EnCodecMAE & MERT: Use .venv-encodecmae (Python 3.12)
    MusiCNN:          Use .venv-musicnn (Python 3.7)

USAGE:
======

Option 1: Run individual extraction scripts (RECOMMENDED)
---------------------------------------------------------
# For EnCodecMAE (from encodecmae folder to avoid import conflicts):
.\\.venv-encodecmae\\Scripts\\Activate.ps1
cd encodecmae
python ../scripts/extract_encodecmae.py --audio_dir ../audio_files --output_dir ../output/embeddings/encodecmae --skip_existing

# For MERT:
.\\.venv-encodecmae\\Scripts\\Activate.ps1
python scripts/extract_mert.py --audio_dir audio_files --output_dir output/embeddings/mert --skip_existing

# For MusiCNN:
.\\.venv-musicnn\\Scripts\\Activate.ps1
python scripts/extract_musicnn.py --audio_dir audio_files --output_dir output/embeddings/musicnn --skip_existing

Option 2: Run this orchestrator script
--------------------------------------
This script will print the commands you need to run manually, since it cannot
switch Python environments automatically.

python run_extraction.py --models encodecmae mert musicnn --audio_dir audio_files
"""

import os
import sys
import glob
import argparse
from pathlib import Path


def get_audio_files(audio_dir, limit=None):
    """Find all audio files in directory."""
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    
    if limit:
        audio_files = audio_files[:limit]
    
    return audio_files


def print_extraction_commands(args):
    """Print the commands needed to run extraction for each model."""
    
    project_root = Path(__file__).resolve().parent
    audio_dir = args.audio_dir
    output_base = args.output_base
    
    skip_flag = "--skip_existing" if args.skip_existing else ""
    verbose_flag = "--verbose" if args.verbose else ""
    limit_flag = f"--limit {args.limit}" if args.limit else ""
    
    print("\n" + "=" * 70)
    print("AUDIO FEATURE EXTRACTION - COMMAND REFERENCE")
    print("=" * 70)
    
    audio_files = get_audio_files(audio_dir)
    print(f"\nFound {len(audio_files)} audio files in '{audio_dir}'")
    
    if args.limit:
        print(f"(Limited to {args.limit} files)")
    
    print("\n" + "-" * 70)
    print("Run these commands in order (each in its appropriate environment):")
    print("-" * 70)
    
    if 'encodecmae' in args.models:
        print("\n[1] EnCodecMAE Extraction (.venv-encodecmae environment)")
        print("    " + "-" * 50)
        print(f"""
    # Activate the EnCodecMAE environment
    .\\.venv-encodecmae\\Scripts\\Activate.ps1
    
    # Change to encodecmae folder (to avoid import conflicts)
    cd encodecmae
    
    # Run extraction
    python ../scripts/extract_encodecmae.py ^
        --audio_dir "../{audio_dir}" ^
        --output_dir "../{output_base}/encodecmae" ^
        {skip_flag} {verbose_flag} {limit_flag}
    
    # Return to project root
    cd ..
""")
    
    if 'mert' in args.models:
        print("\n[2] MERT Extraction (.venv-encodecmae environment)")
        print("    " + "-" * 50)
        print(f"""
    # Activate the EnCodecMAE environment (same env works for MERT)
    .\\.venv-encodecmae\\Scripts\\Activate.ps1
    
    # Run extraction
    python scripts/extract_mert.py ^
        --audio_dir "{audio_dir}" ^
        --output_dir "{output_base}/mert" ^
        {skip_flag} {verbose_flag} {limit_flag}
""")
    
    if 'musicnn' in args.models:
        print("\n[3] MusiCNN Extraction (.venv-musicnn environment)")
        print("    " + "-" * 50)
        print(f"""
    # Activate the MusiCNN environment (Python 3.7)
    .\\.venv-musicnn\\Scripts\\Activate.ps1
    
    # Run extraction  
    python scripts/extract_musicnn.py ^
        --audio_dir "{audio_dir}" ^
        --output_dir "{output_base}/musicnn" ^
        {skip_flag} {verbose_flag} {limit_flag}
""")
    
    print("\n" + "=" * 70)
    print("OUTPUT STRUCTURE")
    print("=" * 70)
    print(f"""
    {output_base}/
    ├── encodecmae/     # EnCodecMAE embeddings (~768-dim vectors)
    │   └── <song_name>.npy
    ├── mert/           # MERT embeddings (~768-dim vectors)
    │   └── <song_name>.npy
    └── musicnn/        # MusiCNN embeddings (~200-dim vectors)
        └── <song_name>.npy
""")
    
    print("=" * 70)
    print("\nTIP: Use --skip_existing to resume interrupted extractions")
    print("=" * 70 + "\n")


def check_existing_embeddings(output_base, models):
    """Check how many embeddings already exist for each model."""
    print("\n" + "-" * 50)
    print("Existing Embeddings Status:")
    print("-" * 50)
    
    for model in models:
        model_dir = os.path.join(output_base, model)
        if os.path.exists(model_dir):
            npy_files = glob.glob(os.path.join(model_dir, "*.npy"))
            print(f"  {model.upper():12s}: {len(npy_files):4d} embeddings")
        else:
            print(f"  {model.upper():12s}:    0 embeddings (directory not found)")


def main():
    parser = argparse.ArgumentParser(
        description="Audio Feature Extraction Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--audio_dir", type=str, default="audio_files",
                        help="Directory containing audio files")
    parser.add_argument("--output_base", type=str, default="output/embeddings",
                        help="Base directory for output embeddings")
    parser.add_argument("--models", type=str, nargs='+',
                        default=['encodecmae', 'mert', 'musicnn'],
                        choices=['encodecmae', 'mert', 'musicnn'],
                        help="Models to use for extraction")
    parser.add_argument("--skip_existing", action='store_true',
                        help="Skip files that already have embeddings")
    parser.add_argument("--verbose", action='store_true',
                        help="Print verbose output")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process")
    parser.add_argument("--status", action='store_true',
                        help="Just show status of existing embeddings")
    args = parser.parse_args()
    
    # Show header
    print("\n" + "=" * 70)
    print("   AUDIO FEATURE EXTRACTION PIPELINE")
    print("   Using: EnCodecMAE, MERT, and MusiCNN")
    print("=" * 70)
    
    # Check audio files exist
    audio_files = get_audio_files(args.audio_dir, args.limit)
    if len(audio_files) == 0:
        print(f"\n❌ No audio files found in '{args.audio_dir}'")
        print("   Supported formats: .mp3, .wav, .flac, .m4a")
        return
    
    # Show existing embeddings status
    check_existing_embeddings(args.output_base, args.models)
    
    if args.status:
        return
    
    # Print the commands to run
    print_extraction_commands(args)


if __name__ == "__main__":
    main()
