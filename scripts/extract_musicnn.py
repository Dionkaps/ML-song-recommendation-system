#!/usr/bin/env python
"""
MusiCNN Feature Extraction Script

This script must be run in the .venv-musicnn environment (Python 3.7).

Usage:
    .\\.venv-musicnn\\Scripts\\Activate.ps1
    python scripts/extract_musicnn.py --audio_dir audio_files --output_dir output/embeddings/musicnn

Models available:
    - MTT_musicnn: MagnaTagATune trained MusiCNN
    - MTT_vgg: MagnaTagATune trained VGG-like model
    - MSD_musicnn: Million Song Dataset trained MusiCNN
    - MSD_vgg: Million Song Dataset trained VGG-like model
"""

import os
import sys
import glob
import argparse
import time
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def extract_musicnn_features(audio_path, output_dir, model_name='MTT_musicnn', verbose=False):
    """
    Extract MusiCNN embeddings from an audio file.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save embeddings
        model_name: One of 'MTT_musicnn', 'MTT_vgg', 'MSD_musicnn', 'MSD_vgg'
        verbose: Print verbose output
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from musicnn_keras.extractor import extractor
        
        # Extract taggram and features
        # extractor returns: taggram, tags, features
        taggram, tags, features = extractor(
            audio_path, 
            model=model_name,
            extract_features=True,
            input_length=3,  # 3 second patches
            input_overlap=None  # No overlap
        )
        
        # features dict contains:
        # - 'max_pool': max pooling of penultimate layer
        # - 'mean_pool': mean pooling of penultimate layer  
        # - 'penultimate': raw penultimate layer output (temporal)
        # - 'taggram': probability of each tag over time
        
        # Use mean_pool as the main embedding (compact, fixed-size)
        # Shape: (n_frames, embedding_dim) -> mean across frames -> (embedding_dim,)
        embedding = np.mean(features['mean_pool'], axis=0)
        
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.npy")
        np.save(output_path, embedding)
        
        if verbose:
            print(f"  Saved MusiCNN embedding for {basename} (shape: {embedding.shape})")
        
        return True
        
    except Exception as e:
        print(f"  Error extracting MusiCNN for {audio_path}: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract MusiCNN embeddings from audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--audio_dir", type=str, default="audio_files",
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="output/embeddings/musicnn",
                        help="Directory to save embeddings")
    parser.add_argument("--model", type=str, default="MTT_musicnn",
                        choices=['MTT_musicnn', 'MTT_vgg', 'MSD_musicnn', 'MSD_vgg'],
                        help="MusiCNN model variant to use")
    parser.add_argument("--skip_existing", action='store_true',
                        help="Skip files that already have embeddings")
    parser.add_argument("--verbose", action='store_true',
                        help="Print verbose output")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process")
    args = parser.parse_args()
    
    # Find audio files
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))
    
    print(f"MusiCNN Feature Extraction")
    print(f"=" * 50)
    print(f"Model: {args.model}")
    print(f"Found {len(audio_files)} audio files")
    
    if args.limit:
        print(f"Limiting to {args.limit} files")
        audio_files = audio_files[:args.limit]
    
    if len(audio_files) == 0:
        print(f"No audio files found in {args.audio_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files
    start_time = time.time()
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for idx, audio_path in enumerate(audio_files, 1):
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(args.output_dir, f"{basename}.npy")
        
        print(f"\n[{idx}/{len(audio_files)}] Processing: {basename}")
        
        if args.skip_existing and os.path.exists(output_path):
            print(f"  ✓ Skipped (already exists)")
            skipped_count += 1
            continue
        
        success = extract_musicnn_features(
            audio_path, 
            args.output_dir,
            model_name=args.model,
            verbose=args.verbose
        )
        
        if success:
            success_count += 1
            print(f"  ✓ Success")
        else:
            failed_count += 1
            print(f"  ✗ Failed")
        
        # Progress info
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (len(audio_files) - idx)
        print(f"  Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m")
    
    # Summary
    print("\n" + "=" * 50)
    print("MUSICNN EXTRACTION COMPLETE")
    print("=" * 50)
    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
