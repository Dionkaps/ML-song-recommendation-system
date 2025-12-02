#!/usr/bin/env python
"""
EnCodecMAE Feature Extraction Script

This script must be run in the .venv-encodecmae environment (Python 3.12).

IMPORTANT: Run this script from a directory OTHER than the project root,
OR cd into the encodecmae folder first, to avoid import conflicts with the
local encodecmae folder.

Usage:
    .\\.venv-encodecmae\\Scripts\\Activate.ps1
    cd encodecmae
    python ../scripts/extract_encodecmae.py --audio_dir ../audio_files --output_dir ../output/embeddings/encodecmae

Available models (from HuggingFace):
    - mel256-ec-base_st (default): Base model, semantic tokens
    - mel256-ec-large_st: Large model, semantic tokens
    - And others available via encodecmae.hub.get_available_models()
"""

import os
import sys
import glob
import argparse
import time
import numpy as np


def extract_encodecmae_features(audio_path, output_dir, model=None, device='cpu', verbose=False):
    """
    Extract EnCodecMAE embeddings from an audio file.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save embeddings
        model: Pre-loaded EnCodecMAE model (for efficiency)
        device: 'cpu' or 'cuda:0'
        verbose: Print verbose output
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import torch
        import librosa
        
        # Load audio at the model's expected sample rate (24kHz for EnCodecMAE)
        audio, sr = librosa.load(audio_path, sr=24000, mono=True)
        
        # Convert to torch tensor and add batch dimension
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            # The model has extract_features_from_array method
            # Returns features as numpy array of shape (batch, time, hidden_dim)
            features = model.extract_features_from_array(audio_tensor, layer=-1)
            
            # Mean pool across time to get a single embedding vector
            # features is a numpy array, not a torch tensor
            embedding = np.mean(features, axis=1).squeeze(0)  # Shape: (hidden_dim,)
        
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.npy")
        np.save(output_path, embedding)
        
        if verbose:
            print(f"  Saved EnCodecMAE embedding for {basename} (shape: {embedding.shape})")
        
        return True
        
    except Exception as e:
        print(f"  Error extracting EnCodecMAE for {audio_path}: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False


def load_encodecmae_model(model_name='mel256-ec-base_st', device='cpu'):
    """
    Load EnCodecMAE model from HuggingFace hub.
    
    Args:
        model_name: Name of the model variant
        device: 'cpu' or 'cuda:0'
    
    Returns:
        Loaded model
    """
    from encodecmae import load_model
    
    print(f"Loading EnCodecMAE model: {model_name}")
    model = load_model(model_name, device=device)
    print(f"Model loaded on {device}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Extract EnCodecMAE embeddings from audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--audio_dir", type=str, default="audio_files",
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="output/embeddings/encodecmae",
                        help="Directory to save embeddings")
    parser.add_argument("--model", type=str, default="mel256-ec-base_st",
                        help="EnCodecMAE model variant to use")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=['cpu', 'cuda:0', 'cuda'],
                        help="Device to run model on")
    parser.add_argument("--skip_existing", action='store_true',
                        help="Skip files that already have embeddings")
    parser.add_argument("--verbose", action='store_true',
                        help="Print verbose output")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process")
    args = parser.parse_args()
    
    # Resolve paths (handle relative paths)
    audio_dir = os.path.abspath(args.audio_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Find audio files
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    
    print(f"EnCodecMAE Feature Extraction")
    print(f"=" * 50)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Found {len(audio_files)} audio files")
    
    if args.limit:
        print(f"Limiting to {args.limit} files")
        audio_files = audio_files[:args.limit]
    
    if len(audio_files) == 0:
        print(f"No audio files found in {audio_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model once
    print("\n=== Loading model ===")
    device = args.device
    if device == 'cuda':
        device = 'cuda:0'
    
    model = load_encodecmae_model(args.model, device=device)
    print("Model loaded!\n")
    
    # Process files
    start_time = time.time()
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for idx, audio_path in enumerate(audio_files, 1):
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.npy")
        
        print(f"\n[{idx}/{len(audio_files)}] Processing: {basename}")
        
        if args.skip_existing and os.path.exists(output_path):
            print(f"  ✓ Skipped (already exists)")
            skipped_count += 1
            continue
        
        success = extract_encodecmae_features(
            audio_path, 
            output_dir,
            model=model,
            device=device,
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
    print("ENCODECMAE EXTRACTION COMPLETE")
    print("=" * 50)
    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
