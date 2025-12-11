#!/usr/bin/env python3
"""
Explore the Million Song Dataset HDF5 structure to identify available features.
"""
import h5py
import os
import numpy as np

# Find a sample h5 file
base = 'data/millionsongsubset_extracted/MillionSongSubset'
sample = None
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.h5'):
            sample = os.path.join(root, f)
            break
    if sample:
        break

print(f'Sample file: {sample}')
print()

def print_h5_structure(h5, prefix=''):
    for key in h5.keys():
        item = h5[key]
        if isinstance(item, h5py.Group):
            print(f'{prefix}{key}/')
            print_h5_structure(item, prefix + '  ')
        else:
            print(f'{prefix}{key}: shape={item.shape}, dtype={item.dtype}')

with h5py.File(sample, 'r') as h5:
    print('='*70)
    print('HDF5 STRUCTURE')
    print('='*70)
    print_h5_structure(h5)
    
    print()
    print('='*70)
    print('KEY FEATURES FOR EXTRACTION')
    print('='*70)
    
    # Analysis group - contains audio features
    if 'analysis' in h5:
        analysis = h5['analysis']
        songs = analysis['songs']
        
        print('\n--- analysis/songs (track-level features) ---')
        for field in songs.dtype.names:
            val = songs[field][0]
            if isinstance(val, bytes):
                val = val.decode('utf-8', errors='ignore')
            print(f'  {field}: {val}')
    
    # Segments data
    if 'analysis' in h5:
        print('\n--- Segments (time-varying Echo Nest features) ---')
        
        if 'segments_timbre' in h5['analysis']:
            timbre = h5['analysis']['segments_timbre'][:]
            print(f'  segments_timbre: shape={timbre.shape} (segments × 12 timbre coeffs)')
            print(f'    Example values: {timbre[0, :3]}...')
        
        if 'segments_pitches' in h5['analysis']:
            pitches = h5['analysis']['segments_pitches'][:]
            print(f'  segments_pitches: shape={pitches.shape} (segments × 12 pitch classes)')
            print(f'    Example values: {pitches[0, :3]}...')
        
        if 'segments_start' in h5['analysis']:
            starts = h5['analysis']['segments_start'][:]
            print(f'  segments_start: {len(starts)} segments')
    
    # Musicbrainz for additional info
    if 'musicbrainz' in h5:
        print('\n--- musicbrainz (metadata) ---')
        mb_songs = h5['musicbrainz']['songs']
        for field in mb_songs.dtype.names:
            val = mb_songs[field][0]
            if isinstance(val, bytes):
                val = val.decode('utf-8', errors='ignore')
            print(f'  {field}: {val}')
