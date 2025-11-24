import os
import glob
import argparse
import time
from pathlib import Path
from src.extract_embeddings import (
    extract_openl3, extract_crepe, extract_madmom, extract_mert,
    get_openl3_model, get_mert_model, get_madmom_processor
)


def main():
    parser = argparse.ArgumentParser(description="Extract audio embeddings.")
    parser.add_argument("--audio_dir", type=str, default="audio_files", 
                        help="Directory containing audio files.")
    parser.add_argument("--output_base", type=str, default="output/embeddings", 
                        help="Base directory for output.")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=['openl3', 'crepe', 'madmom', 'mert'],
                        help="Models to use (space-separated). Options: openl3, crepe, madmom, mert")
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
    
    print(f"Found {len(audio_files)} audio files.")
    
    if args.limit:
        print(f"Limiting to {args.limit} files.")
        audio_files = audio_files[:args.limit]
    
    if len(audio_files) == 0:
        print(f"No audio files found in {args.audio_dir}")
        return

    # Create output directories
    for model in args.models:
        os.makedirs(os.path.join(args.output_base, model), exist_ok=True)
    
    # Pre-load models once (this is the key optimization!)
    print("\n=== Loading models ===")
    models_cache = {}
    
    if 'openl3' in args.models:
        models_cache['openl3_model'] = get_openl3_model()
    
    if 'mert' in args.models:
        mert_model, mert_extractor, mert_device = get_mert_model()
        models_cache['mert_model'] = mert_model
        models_cache['mert_extractor'] = mert_extractor
        models_cache['mert_device'] = mert_device
    
    if 'madmom' in args.models:
        models_cache['madmom_processor'] = get_madmom_processor()
    
    print("All models loaded!\n")
    
    # Process each audio file
    start_time = time.time()
    success_counts = {model: 0 for model in args.models}
    failed_counts = {model: 0 for model in args.models}
    skipped_counts = {model: 0 for model in args.models}
    
    for idx, audio_path in enumerate(audio_files, 1):
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"\n[{idx}/{len(audio_files)}] Processing: {basename}")
        
        # OpenL3
        if 'openl3' in args.models:
            output_path = os.path.join(args.output_base, 'openl3', f"{basename}.npy")
            if args.skip_existing and os.path.exists(output_path):
                print(f"  ✓ OpenL3: Skipped (already exists)")
                skipped_counts['openl3'] += 1
            else:
                success = extract_openl3(
                    audio_path, 
                    os.path.join(args.output_base, 'openl3'),
                    model=models_cache.get('openl3_model'),
                    verbose=args.verbose
                )
                if success:
                    success_counts['openl3'] += 1
                    print(f"  ✓ OpenL3: Success")
                else:
                    failed_counts['openl3'] += 1
                    print(f"  ✗ OpenL3: Failed")
        
        # CREPE
        if 'crepe' in args.models:
            output_path = os.path.join(args.output_base, 'crepe', f"{basename}.npy")
            if args.skip_existing and os.path.exists(output_path):
                print(f"  ✓ CREPE: Skipped (already exists)")
                skipped_counts['crepe'] += 1
            else:
                success = extract_crepe(
                    audio_path, 
                    os.path.join(args.output_base, 'crepe'),
                    verbose=args.verbose
                )
                if success:
                    success_counts['crepe'] += 1
                    print(f"  ✓ CREPE: Success")
                else:
                    failed_counts['crepe'] += 1
                    print(f"  ✗ CREPE: Failed")
        
        # madmom
        if 'madmom' in args.models:
            output_path = os.path.join(args.output_base, 'madmom', f"{basename}.npy")
            if args.skip_existing and os.path.exists(output_path):
                print(f"  ✓ madmom: Skipped (already exists)")
                skipped_counts['madmom'] += 1
            else:
                success = extract_madmom(
                    audio_path, 
                    os.path.join(args.output_base, 'madmom'),
                    processor=models_cache.get('madmom_processor'),
                    verbose=args.verbose
                )
                if success:
                    success_counts['madmom'] += 1
                    print(f"  ✓ madmom: Success")
                else:
                    failed_counts['madmom'] += 1
                    print(f"  ✗ madmom: Failed")
        
        # MERT
        if 'mert' in args.models:
            output_path = os.path.join(args.output_base, 'mert', f"{basename}.npy")
            if args.skip_existing and os.path.exists(output_path):
                print(f"  ✓ MERT: Skipped (already exists)")
                skipped_counts['mert'] += 1
            else:
                success = extract_mert(
                    audio_path, 
                    os.path.join(args.output_base, 'mert'),
                    model=models_cache.get('mert_model'),
                    feature_extractor=models_cache.get('mert_extractor'),
                    device=models_cache.get('mert_device'),
                    verbose=args.verbose
                )
                if success:
                    success_counts['mert'] += 1
                    print(f"  ✓ MERT: Success")
                else:
                    failed_counts['mert'] += 1
                    print(f"  ✗ MERT: Failed")
        
        # Show progress
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (len(audio_files) - idx)
        print(f"  Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m")
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Processed {len(audio_files)} audio files")
    print("\nResults by model:")
    for model in args.models:
        print(f"  {model.upper()}:")
        print(f"    Success: {success_counts[model]}")
        print(f"    Failed: {failed_counts[model]}")
        print(f"    Skipped: {skipped_counts[model]}")
    print("="*60)


if __name__ == "__main__":
    main()
