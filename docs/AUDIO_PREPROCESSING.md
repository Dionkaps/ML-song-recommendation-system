# Audio Preprocessing Module

This module provides the supported preprocessing pipeline for the workspace's
handcrafted-audio baseline.

## Overview

The preprocessing pipeline enforces these baseline invariants:

- consistent duration: all retained files are exactly `29` seconds
- consistent loudness: files are normalized toward `-14 LUFS`
- consistent format: output is mono `22050 Hz` `PCM_16` WAV
- peak safety: a `-1.0 dBFS` sample-peak ceiling is applied after loudness gain

## Important Accuracy Note

The current implementation measures integrated loudness with `pyloudnorm`
following ITU-R BS.1770, but peak protection is based on sample peak. It does
not perform oversampled true-peak measurement or limiting.

Why this matters:

- the historical parameter name `max_true_peak` is still present in code and
  configs for backward compatibility
- documentation should read that value as a sample-peak ceiling, not a formal
  dBTP true-peak guarantee

## Why Preprocessing Matters

In MIR and clustering workflows, inconsistent audio levels and durations can
distort the feature space:

1. Loud tracks can dominate magnitude-sensitive summaries.
2. Longer clips can contain more variation than short previews.
3. Mixed sample rates, channels, and codecs create avoidable extraction drift.

## Current Module Layout

```text
src/audio_preprocessing/
|- __init__.py
|- processor.py
|- duration_handler.py
`- loudness_normalizer.py
```

## Quick Start

### Supported runner

```bash
python scripts/run_audio_preprocessing.py --audio-dir audio_files
```

### Direct usage

```python
from src.audio_preprocessing import AudioPreprocessor

processor = AudioPreprocessor(
    target_duration=29.0,
    target_lufs=-14.0,
    max_true_peak=-1.0,  # legacy name; acts as a sample-peak ceiling
    sample_rate=22050,
)

result = processor.process_file("path/to/audio.wav")
print(result["status"])
```

### Detailed per-file output

```python
stats = processor.process_directory("audio_files", return_details=True)
for detail in stats["details"]:
    print(detail["file"], detail["actions"])
```

## Pipeline Steps

### 1. Load audio

- loads `.wav`, `.mp3`, `.flac`, and `.m4a`
- resamples to `22050 Hz`
- converts to mono

### 2. Enforce duration

- `< 29s`: remove the file from the processed library
- `= 29s`: keep as-is
- `> 29s`: crop to the first `29s`

### 3. Normalize loudness

1. measure integrated loudness with ITU-R BS.1770
2. compute gain toward `-14 LUFS`
3. apply gain
4. if the resulting sample peak exceeds the configured ceiling, reduce gain

Tracks that hit the peak ceiling may finish slightly below the loudness target.

### 4. Save normalized audio

- output format is WAV
- output subtype is `PCM_16`
- non-WAV sources are converted to `.wav`

## Result Structure

Each processed file returns a dictionary like:

```python
{
    "file": "song.wav",
    "status": "success",
    "actions": ["cropped", "normalized"],
    "original_duration": 35.0,
    "final_duration": 29.0,
    "original_lufs": -18.5,
    "final_lufs": -14.0,
    "original_peak_db": -6.2,
    "final_peak_db": -1.7,
    "gain_applied_db": 4.5,
    "error": None,
}
```

`original_peak_db` and `final_peak_db` are sample-peak measurements in dBFS.

## Actions

| Action | Meaning |
|---|---|
| `cropped` | Duration was reduced to the target |
| `normalized` | Loudness gain was applied |
| `peak_limited` | Gain was reduced to satisfy the sample-peak ceiling |
| `skipped_silence` | File was silent, so no gain change was applied |
| `removed_too_short` | File was deleted for being below the minimum duration |
| `converted_to_wav` | Source format was converted to WAV on write-back |

## Integration

The preprocessing step is part of [run_pipeline.py](/c:/Users/vpddk/Desktop/Me/Github/ML-song-recommendation-system/run_pipeline.py).

Examples:

```bash
python run_pipeline.py
python run_pipeline.py --skip download extract plot cluster
python run_pipeline.py --skip preprocess
```

## Legacy Paths

- `src/utils/audio_normalizer.py` now acts as a compatibility wrapper around the
  supported preprocessing pipeline
- old docs/scripts that described "true peak" behavior should be interpreted as
  sample-peak protection unless they explicitly describe oversampling

## Dependencies

- `librosa`
- `soundfile`
- `pyloudnorm`
- `numpy`
- `tqdm`

## References

1. ITU-R BS.1770-4
2. EBU R128
3. `pyloudnorm`
