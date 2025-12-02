#!/usr/bin/env python
"""
MERT Feature Extraction Script

This script can be run in the .venv-encodecmae environment (Python 3.12)
since MERT uses PyTorch and transformers which are compatible.

Usage:
    .\\.venv-encodecmae\\Scripts\\Activate.ps1
    python scripts/extract_mert.py --audio_dir audio_files --output_dir output/embeddings/mert

Models available:
    - m-a-p/MERT-v1-95M (default): 95M parameter model
    - m-a-p/MERT-v1-330M: 330M parameter model (larger, better quality)
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


# Global model cache
_MERT_MODEL = None
_MERT_FEATURE_EXTRACTOR = None
_MERT_DEVICE = None


def get_mert_model(model_id="m-a-p/MERT-v1-95M"):
    """Load and cache MERT model and feature extractor."""
    global _MERT_MODEL, _MERT_FEATURE_EXTRACTOR, _MERT_DEVICE
    
    if _MERT_MODEL is None:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, AutoModel
        
        print(f"Loading MERT model: {model_id}")
        _MERT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {_MERT_DEVICE}")
        
        _MERT_FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id, trust_remote_code=True
        )
        _MERT_MODEL = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        ).to(_MERT_DEVICE)
        _MERT_MODEL.eval()
        print("MERT model loaded!")
        
    return _MERT_MODEL, _MERT_FEATURE_EXTRACTOR, _MERT_DEVICE


def extract_mert_features(audio_path, output_dir, model=None, feature_extractor=None, 
                          device=None, verbose=False, use_mean_pooling=True):
    """
    Extract MERT embeddings from an audio file.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save embeddings
        model: Pre-loaded MERT model
        feature_extractor: Pre-loaded feature extractor
        device: torch device
        verbose: Print verbose output
        use_mean_pooling: If True, average across time for compact representation
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import torch
        import librosa
        
        if model is None or feature_extractor is None:
            model, feature_extractor, device = get_mert_model()
        
        # Load audio at 24kHz (MERT requirement)
        audio, sr = librosa.load(audio_path, sr=24000)
        
        # Prepare inputs
        inputs = feature_extractor(
            audio, 
            sampling_rate=24000, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs.input_values.to(device)
        
        # Extract features
        with torch.no_grad():
            outputs = model(input_values)
            last_hidden_state = outputs.last_hidden_state
            
            if use_mean_pooling:
                # Mean pooling: (batch, time, 768) -> (768,)
                embedding = last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            else:
                # Full sequence: (time, 768)
                embedding = last_hidden_state.squeeze(0).cpu().numpy()
        
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.npy")
        np.save(output_path, embedding)
        
        if verbose:
            pooling_info = "mean-pooled" if use_mean_pooling else "full sequence"
            print(f"  Saved MERT embedding for {basename} (shape: {embedding.shape}, {pooling_info})")
        
        return True
        
    except Exception as e:
        print(f"  Error extracting MERT for {audio_path}: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract MERT embeddings from audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--audio_dir", type=str, default="audio_files",
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="output/embeddings/mert",
                        help="Directory to save embeddings")
    parser.add_argument("--model", type=str, default="m-a-p/MERT-v1-95M",
                        help="MERT model variant to use")
    parser.add_argument("--skip_existing", action='store_true',
                        help="Skip files that already have embeddings")
    parser.add_argument("--verbose", action='store_true',
                        help="Print verbose output")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process")
    parser.add_argument("--full_sequence", action='store_true',
                        help="Keep full temporal sequence instead of mean pooling")
    args = parser.parse_args()
    
    # Find audio files
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))
    
    print(f"MERT Feature Extraction")
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
    
    # Load model once
    print("\n=== Loading model ===")
    model, feature_extractor, device = get_mert_model(args.model)
    print()
    
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
        
        success = extract_mert_features(
            audio_path, 
            args.output_dir,
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            verbose=args.verbose,
            use_mean_pooling=not args.full_sequence
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
    print("MERT EXTRACTION COMPLETE")
    print("=" * 50)
    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
