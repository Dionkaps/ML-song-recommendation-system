#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Files Cleanup Script

This script removes orphaned feature files from output/results that don't have
corresponding audio files in audio_files directory.

This ensures KMeans clustering only processes songs that:
1. Have audio files
2. Have genre information
3. Have complete feature sets
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

def get_audio_basenames(audio_dir='audio_files'):
    """Get set of basenames from audio files (without extensions)"""
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found!")
        return set()
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
    basenames = {Path(f).stem for f in audio_files}
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Unique basenames: {len(basenames)}")
    
    return basenames

def analyze_feature_files(results_dir='output/results', audio_basenames=None):
    """Analyze feature files and identify orphaned ones"""
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return {}, []
    
    # Group feature files by basename
    feature_groups = defaultdict(list)
    
    for feature_file in os.listdir(results_dir):
        if not feature_file.endswith('.npy'):
            continue
        
        # Extract basename from feature filename
        # e.g., "Artist - Title_mfcc.npy" -> "Artist - Title"
        base = feature_file
        for suffix in ['_mfcc.npy', '_melspectrogram.npy', '_spectral_centroid.npy', 
                      '_spectral_flatness.npy', '_zero_crossing_rate.npy']:
            if feature_file.endswith(suffix):
                base = feature_file[:-len(suffix)]
                break
        
        feature_groups[base].append(feature_file)
    
    print(f"\nTotal feature file groups: {len(feature_groups)}")
    print(f"Total feature files: {sum(len(files) for files in feature_groups.values())}")
    
    # Identify orphaned feature groups
    orphaned = {}
    valid = {}
    
    for base, files in feature_groups.items():
        if audio_basenames and base not in audio_basenames:
            orphaned[base] = files
        else:
            valid[base] = files
    
    print(f"\nAnalysis:")
    print(f"  Valid feature groups (have audio): {len(valid)}")
    print(f"  Orphaned feature groups (no audio): {len(orphaned)}")
    
    # Show sample orphaned files
    if orphaned:
        print(f"\nSample orphaned feature bases:")
        for i, base in enumerate(list(orphaned.keys())[:10], 1):
            print(f"  {i}. '{base}' ({len(orphaned[base])} files)")
    
    return orphaned, list(valid.keys())

def cleanup_orphaned_files(orphaned, results_dir='output/results', dry_run=True):
    """Remove orphaned feature files"""
    total_files = sum(len(files) for files in orphaned.values())
    
    print(f"\n{'='*60}")
    if dry_run:
        print("DRY RUN - No files will be deleted")
    else:
        print("CLEANUP MODE - Files WILL BE DELETED")
    print(f"{'='*60}")
    
    print(f"\nOrphaned feature groups to remove: {len(orphaned)}")
    print(f"Total orphaned files to remove: {total_files}")
    
    if not orphaned:
        print("\nNo orphaned files found. Nothing to clean up!")
        return
    
    if dry_run:
        print("\nFiles that WOULD be deleted:")
        count = 0
        for base, files in orphaned.items():
            for file in files:
                print(f"  - {file}")
                count += 1
                if count >= 20:  # Show max 20 examples
                    remaining = total_files - count
                    if remaining > 0:
                        print(f"  ... and {remaining} more files")
                    break
            if count >= 20:
                break
        
        print(f"\nTo actually delete these files, run:")
        print(f"  python {sys.argv[0]} --confirm")
    else:
        print("\nDeleting orphaned files...")
        deleted_count = 0
        for base, files in orphaned.items():
            for file in files:
                file_path = os.path.join(results_dir, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    if deleted_count % 100 == 0:
                        print(f"  Deleted {deleted_count}/{total_files} files...")
                except Exception as e:
                    print(f"  Error deleting {file}: {e}")
        
        print(f"\n✓ Successfully deleted {deleted_count} orphaned feature files")

def verify_cleanup(audio_basenames, results_dir='output/results'):
    """Verify that cleanup was successful"""
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    orphaned, valid = analyze_feature_files(results_dir, audio_basenames)
    
    if not orphaned:
        print("\n✓ All feature files have corresponding audio files!")
        print(f"✓ {len(valid)} valid feature groups remain")
    else:
        print(f"\n⚠️  Warning: {len(orphaned)} orphaned feature groups still exist")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up orphaned feature files')
    parser.add_argument('--confirm', action='store_true', 
                       help='Actually delete files (default is dry-run)')
    parser.add_argument('--audio-dir', default='audio_files',
                       help='Path to audio files directory')
    parser.add_argument('--results-dir', default='output/results',
                       help='Path to feature results directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Feature Files Cleanup Script")
    print("="*60)
    
    # Step 1: Get audio basenames
    print("\n[Step 1] Loading audio file list...")
    audio_basenames = get_audio_basenames(args.audio_dir)
    
    if not audio_basenames:
        print("Error: No audio files found. Cannot proceed.")
        return
    
    # Step 2: Analyze feature files
    print(f"\n[Step 2] Analyzing feature files in '{args.results_dir}'...")
    orphaned, valid = analyze_feature_files(args.results_dir, audio_basenames)
    
    # Step 3: Cleanup (or dry-run)
    print(f"\n[Step 3] Cleanup{'  (DRY RUN)' if not args.confirm else ''}...")
    cleanup_orphaned_files(orphaned, args.results_dir, dry_run=not args.confirm)
    
    # Step 4: Verify (if actual cleanup was done)
    if args.confirm and orphaned:
        verify_cleanup(audio_basenames, args.results_dir)
    
    print("\n" + "="*60)
    print("Cleanup script complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
