import numpy as np
import os

# Get features directory
features_dir = 'output/features'
files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]

# Load and check dimensions of each feature type
test_files = {
    'mfcc': [f for f in files if '_mfcc.npy' in f][0],
    'delta_mfcc': [f for f in files if '_delta_mfcc.npy' in f][0],
    'delta2_mfcc': [f for f in files if '_delta2_mfcc.npy' in f][0],
    'spectral_centroid': [f for f in files if '_spectral_centroid.npy' in f][0],
    'spectral_rolloff': [f for f in files if '_spectral_rolloff.npy' in f][0],
    'spectral_flux': [f for f in files if '_spectral_flux.npy' in f][0],
    'spectral_flatness': [f for f in files if '_spectral_flatness.npy' in f][0],
    'zero_crossing_rate': [f for f in files if '_zero_crossing_rate.npy' in f][0],
    'chroma': [f for f in files if '_chroma.npy' in f][0],
    'beat_strength': [f for f in files if '_beat_strength.npy' in f][0],
}

print("=" * 80)
print("AUDIO FEATURES DIMENSIONS (as stored in .npy files)")
print("=" * 80)

for name, filename in test_files.items():
    path = os.path.join(features_dir, filename)
    data = np.load(path)
    if len(data.shape) > 1:
        print(f"{name:25s}: shape={str(data.shape):20s} -> {data.shape[0]:4d} time steps x {data.shape[1]:4d} dims")
    else:
        print(f"{name:25s}: shape={str(data.shape):20s} -> {data.shape[0]:4d} values (1D array)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

total_dims = 0
feature_summary = {}

for name, filename in test_files.items():
    path = os.path.join(features_dir, filename)
    data = np.load(path)
    if len(data.shape) > 1:
        feature_summary[name] = data.shape[1]
        total_dims += data.shape[1]
    else:
        feature_summary[name] = 1
        total_dims += 1

for name, dims in feature_summary.items():
    print(f"  {name:25s}: {dims:4d} dimensions")

print(f"\n  Total audio features: {total_dims} dimensions (before summary stats)")
print(f"  With mean+std summary: {total_dims * 2} dimensions (time-varying features)")
