# Supported Baseline

Updated: 2026-03-15

Machine-readable contract:

- `config/experiment_profiles.py`
- short production summary: `docs/RECOMMENDED_PRODUCTION_BASELINE.md`
- explicit decision summary: `docs/DECISION_POLICY.md`

## Supported operating mode

The currently supported clustering mode is `audio_only_spectral_plus_beat`.

That means:

- clustering uses audio features only
- the active clustering subset is `spectral_plus_beat`
- genre stays metadata-only and is not appended to clustering inputs
- unified metadata now lives at `data/songs.csv`
- MSD numeric metadata remains disabled in the supported baseline
- MSD numeric metadata may be restored later only as an explicit experiment after unified metadata coverage is judged acceptable
- the default clustering method is `GMM`

## Active clustering subset

The clustering subset is:

- `spectral_centroid`
- `spectral_rolloff`
- `spectral_flux`
- `spectral_flatness`
- `zero_crossing_rate`
- `beat_strength`

The following extracted features remain available on disk for analysis, but are excluded from clustering in the supported baseline:

- `mfcc`
- `delta_mfcc`
- `delta2_mfcc`
- `chroma`

## Representation contract

- time-varying features are summarized as `mean + std`
- `beat_strength` is kept as its raw 4-scalar block
- preprocessing uses per-group `StandardScaler`
- equalization uses `pca_per_group_5`
- with the active 6-group subset, the prepared clustering representation is `30` dimensions

## Audio preprocessing invariants

- mono audio
- sample rate `22050 Hz`
- duration `29.0` seconds
- output subtype `PCM_16`
- loudness normalization enabled
- baseline target loudness `-14 LUFS`
- baseline max peak target `-1.0 dBFS` sample peak

## Feature extraction reproducibility

- use `python scripts/run_feature_extraction.py` to regenerate handcrafted features
- every extraction run writes `output/features/feature_extraction_manifest.json`
- the manifest records sorted audio-file ordering, executor/worker settings, extracted feature keys, fixed DSP parameters, audio preprocessing invariants, and library versions
- the active extraction contract is deterministic as long as the preprocessed audio library and dependency versions stay fixed

## QC artifacts

- the shared clustering loader validates selected feature arrays before clustering and writes:
  - `output/metrics/clustering_dataset_qc_latest.csv`
  - `output/metrics/clustering_dataset_qc_summary_latest.json`
- the full library QC pass runs via `python scripts/analysis/run_feature_qc.py`
- that pass writes timestamped reports for per-track status, group variance diagnostics, and a human-readable summary under `output/metrics/`

## Default entry points

- clustering default: `python run_pipeline.py`
- direct GMM baseline: `python src/clustering/gmm.py`
- preprocessing wrapper: `python scripts/run_audio_preprocessing.py`
- experiment suite runner: `python scripts/run_all_clustering.py`

## Explicit decisions

The currently accepted decision policy is:

- broad macro-style clusters, with the supported GMM baseline expected to stay in the `4..8` cluster range
- a production-default GMM must clear:
  - median subsample ARI `>= 0.90`
  - mean subsample ARI `>= 0.75`
  - reference median ARI `>= 0.90`
  - per-cluster median Jaccard `>= 0.90`
- uncertain GMM assignments remain visible under normal distance ranking by default
- posterior-weighted ranking and hard confidence/posterior filtering remain optional, not default
- MSD numeric metadata returns only as an explicit experiment after near-complete coverage and a clean metadata audit

## Explicit comparison baselines

Documented comparison-only profiles now live in `config/experiment_profiles.py`:

- `all_audio_pca_comparison`
- `all_audio_zscore_comparison`

Comparison-only methods remain explicit:

- `K-Means`
- `HDBSCAN`
- optional `VaDE`

## Current limitations

- offline validation is still proxy-driven until listening-test or human-judgment checks are added
- recommendation quality is still estimated with metadata proxies rather than human similarity judgments
