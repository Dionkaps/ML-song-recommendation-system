# ML Song Recommendation System Documentation

This document gives a current high-level overview of the workspace and its
supported baseline.

## Project Overview

The project builds a content-based music recommendation system from local audio
files. The active workflow is:

1. collect or refresh metadata and local audio
2. preprocess audio to a consistent baseline
3. extract handcrafted audio features
4. cluster songs in prepared feature space
5. surface recommendations from clustering outputs in the UI

## Supported Baseline

The current supported baseline is documented in
[SUPPORTED_BASELINE.md](SUPPORTED_BASELINE.md). In short:

- preprocessing invariants: mono, `22050 Hz`, `29s`, `PCM_16`, loudness-normalized
- clustering input: audio-only handcrafted `spectral_plus_beat`
- preparation: per-group scaling plus `pca_per_group_5`
- default clustering method: `GMM`
- genre and other metadata remain metadata/evaluation signals, not clustering input

The current explicit decision policy is documented in
[DECISION_POLICY.md](DECISION_POLICY.md).

## Main Components

### Data and metadata

- `src/data_collection/deezer-song.py`
- `src/data_collection/extract_millionsong_dataset.py`
- `src/utils/song_metadata.py`

These scripts maintain the unified `data/songs.csv` catalog and align local
audio with metadata rows.

### Audio preprocessing

- `scripts/run_audio_preprocessing.py`
- `src/audio_preprocessing/processor.py`
- `src/audio_preprocessing/loudness_normalizer.py`

This stage enforces the supported audio baseline before feature extraction.
Integrated loudness is measured with BS.1770 logic, and a sample-peak ceiling
is applied after gain. The implementation does not perform oversampled
true-peak limiting.

### Feature extraction

- `src/features/extract_features.py`
- `scripts/visualization/ploting.py`

The active handcrafted extractor writes MFCC-family features, spectral
descriptors, chroma, and beat/rhythm summaries. The clustering baseline uses
the `spectral_plus_beat` subset, while plotting can still visualize legacy
mel-spectrogram arrays if they already exist on disk.

### Clustering and recommendation

- `src/clustering/kmeans.py`
- `src/clustering/gmm.py`
- `src/clustering/hdbscan.py`
- `src/clustering/vade.py`
- `src/ui/modern_ui.py`

The workspace supports multiple clustering methods for comparison, but the
default baseline currently points to `GMM`.

## Pipeline Entry Points

### Full pipeline

```bash
python run_pipeline.py
```

### Preprocess audio only

```bash
python scripts/run_audio_preprocessing.py --audio-dir audio_files
```

### Extract handcrafted features only

```bash
python src/features/extract_features.py
```

### Visualize extracted features

```bash
python scripts/visualization/ploting.py --features_dir output/features --plots_dir output/plots
```

### Run clustering directly

```bash
python src/clustering/gmm.py
python src/clustering/kmeans.py
python src/clustering/hdbscan.py
python src/clustering/vade.py
```

## Notes on Historical Paths

The repository still contains historical or compatibility-oriented code paths,
but they should not override the supported baseline. In particular:

- the old duration-only normalizer has been reduced to a compatibility wrapper
- plotting keeps a legacy mel-spectrogram branch only when those arrays exist
- helper scripts are now aligned to the processed `.wav` library rather than an
  `.mp3`-only assumption

## Related Docs

- [DECISION_POLICY.md](DECISION_POLICY.md)
- [RECOMMENDED_PRODUCTION_BASELINE.md](RECOMMENDED_PRODUCTION_BASELINE.md)
- [SUPPORTED_BASELINE.md](SUPPORTED_BASELINE.md)
- [AUDIO_PREPROCESSING.md](AUDIO_PREPROCESSING.md)
- [AUDIO_EMBEDDING_EXTRACTION.md](AUDIO_EMBEDDING_EXTRACTION.md)
- [QUICKSTART_EMBEDDINGS.md](QUICKSTART_EMBEDDINGS.md)
- [reports/](reports)
