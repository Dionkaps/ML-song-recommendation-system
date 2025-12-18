# Audio Preprocessing Module

This module provides a robust audio preprocessing pipeline for Music Information Retrieval (MIR) tasks, ensuring **consistency and fairness** across all audio tracks before feature extraction and clustering.

## Overview

The preprocessing pipeline normalizes audio files to ensure:
- **Consistent duration**: All files are exactly 29 seconds
- **Consistent loudness**: All files normalized to -14 LUFS (ITU-R BS.1770)
- **No clipping**: True Peak limited to -1.0 dBTP (EBU R128 compliant)

## Why is Audio Preprocessing Important?

In MIR tasks, especially clustering-based recommendation systems, **inconsistent audio levels and durations can bias feature extraction**:

1. **Loudness Variations**: A quiet jazz track and a loud rock track will have different MFCC magnitudes even if they have similar timbral characteristics
2. **Duration Differences**: Longer tracks may have more varied features, affecting clustering
3. **Clipping/Distortion**: Over-driven audio can introduce artifacts in spectral features

## Standards Compliance

| Standard | Description | Our Implementation |
|----------|-------------|-------------------|
| **ITU-R BS.1770-4** | Algorithm for measuring audio loudness | Used via `pyloudnorm` library |
| **EBU R128** | Loudness normalization standard | Target: -14 LUFS, True Peak: -1.0 dBTP |

### Target Loudness: -14 LUFS

We use -14 LUFS as the target because:
- It's the standard for major streaming platforms (Spotify, YouTube)
- Provides good dynamic range for feature extraction
- Prevents over-compression of dynamics

### True Peak Limiting: -1.0 dBTP

True Peak limiting at -1.0 dBTP:
- Prevents inter-sample peaks from causing distortion
- Provides headroom for codec artifacts
- Is the EBU R128 recommendation for broadcast

## Module Structure

```
src/audio_preprocessing/
├── __init__.py              # Package exports
├── processor.py             # Main AudioPreprocessor class
├── duration_handler.py      # Duration validation & cropping
└── loudness_normalizer.py   # LUFS normalization & peak limiting
```

## Quick Start

### Basic Usage

```python
from src.audio_preprocessing import AudioPreprocessor

# Initialize with default settings
processor = AudioPreprocessor()

# Process a single file
result = processor.process_file("path/to/audio.mp3")
print(f"Status: {result['status']}")
print(f"Original LUFS: {result['original_lufs']}")
print(f"Final LUFS: {result['final_lufs']}")

# Process entire directory
stats = processor.process_directory("audio_files/")
print(f"Processed: {stats['processed']} files")
```

### Custom Settings

```python
processor = AudioPreprocessor(
    target_duration=29.0,    # Target duration in seconds
    target_lufs=-14.0,       # Target loudness in LUFS
    max_true_peak=-1.0,      # Maximum true peak in dBTP
    sample_rate=22050        # Sample rate for processing
)
```

### Getting Detailed Results

```python
stats = processor.process_directory("audio_files/", return_details=True)

# Access per-file results
for detail in stats['details']:
    print(f"{detail['file']}: {detail['status']}")
    print(f"  Original: {detail['original_lufs']} LUFS, {detail['original_duration']}s")
    print(f"  Final: {detail['final_lufs']} LUFS, {detail['final_duration']}s")
    print(f"  Actions: {detail['actions']}")
```

## Pipeline Steps

### 1. Audio Loading
- Loads audio file using `librosa`
- Resamples to target sample rate (default: 22050 Hz)
- Converts to mono

### 2. Duration Handling
- **< 29s**: File is **removed** (too short for consistent features)
- **= 29s**: No change
- **> 29s**: **Cropped** to first 29 seconds

### 3. Loudness Normalization
1. Measure integrated loudness using ITU-R BS.1770
2. Calculate gain needed to reach -14 LUFS
3. Apply gain
4. Check if peak exceeds -1.0 dBTP
5. If so, reduce gain (may result in slightly lower loudness)

### 4. Save
- Overwrites original file with processed audio
- Saved as WAV format at the target sample rate

## Processing Results

Each processed file returns a result dictionary:

```python
{
    'file': 'song.mp3',
    'status': 'success',  # or 'removed', 'error'
    'actions': ['cropped', 'normalized'],
    'original_duration': 35.0,
    'final_duration': 29.0,
    'original_lufs': -18.5,
    'final_lufs': -14.0,
    'original_peak_db': -6.2,
    'final_peak_db': -1.7,
    'gain_applied_db': 4.5,
    'error': None
}
```

### Possible Actions

| Action | Description |
|--------|-------------|
| `cropped` | Duration was reduced to target |
| `normalized` | Loudness was adjusted to target LUFS |
| `peak_limited` | Gain was reduced to meet True Peak limit |
| `skipped_silence` | File was silent, no normalization applied |
| `removed_too_short` | File was deleted (below minimum duration) |

## Integration with Pipeline

The preprocessing step is integrated into `run_pipeline.py`:

```bash
# Run full pipeline including preprocessing
python run_pipeline.py

# Skip download, only run preprocessing
python run_pipeline.py --skip download extract process plot cluster

# Skip preprocessing if already done
python run_pipeline.py --skip preprocess
```

## Testing

Run the demo test to see the pipeline in action:

```bash
python -m tests.test_preprocessing_demo
```

Run unit tests:

```bash
python -m tests.test_preprocessing_small
```

## Example Output

```
============================================================
AUDIO PREPROCESSING
============================================================
Files to process: 6
Workers: 4
Target Duration: 29.0s
Target Loudness: -14.0 LUFS
Max True Peak: -1.0 dBTP
============================================================

Preprocessing: 100%|████████████████████| 6/6 [00:03<00:00,  1.93it/s]

============================================================
PREPROCESSING SUMMARY
============================================================
Total files:      6
Processed:        5
  - Cropped:      2
  - Normalized:   5
  - Peak Limited: 0
Removed (short):  1
Errors:           0
============================================================
```

## Dependencies

- `librosa>=0.9.0`: Audio loading and duration calculation
- `soundfile>=0.10.0`: Audio file writing
- `pyloudnorm>=0.1.0`: ITU-R BS.1770 loudness measurement
- `numpy>=1.21.0`: Numerical operations
- `tqdm>=4.0.0`: Progress bars

## References

1. **ITU-R BS.1770-4**: [Algorithms to measure audio programme loudness and true-peak audio level](https://www.itu.int/rec/R-REC-BS.1770)
2. **EBU R128**: [Loudness normalisation and permitted maximum level of audio signals](https://tech.ebu.ch/docs/r/r128.pdf)
3. **pyloudnorm**: [Python implementation of ITU-R BS.1770-4](https://github.com/csteinmetz1/pyloudnorm)
