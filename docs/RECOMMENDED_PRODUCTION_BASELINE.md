# Recommended Production Baseline

This is the short, durable reference for the baseline that should drive the
default clustering outputs, recommendation UI, and final offline reports.

## Canonical source of truth

The machine-readable source of truth now lives in:

- `config/experiment_profiles.py`

That file defines:

- the recommended production profile
- the documented comparison-only profiles
- the explicit decision policy
- the metadata-proxy evaluation limitation
- the future human-validation backlog

## Recommended production profile

Profile id: `recommended_production`

Use this profile for the default product path:

- audio-only clustering
- handcrafted subset: `spectral_plus_beat`
- selected groups:
  - `spectral_centroid`
  - `spectral_rolloff`
  - `spectral_flux`
  - `spectral_flatness`
  - `zero_crossing_rate`
  - `beat_strength`
- equalization: `pca_per_group`
- PCA components per group: `5`
- expected raw dimension: `14`
- expected prepared dimension: `30`
- genre: metadata/evaluation only
- MSD numeric metadata: disabled in the supported baseline
- default clustering method for product outputs: `GMM`

## Explicit decisions

The current explicit decision policy lives in:

- `docs/DECISION_POLICY.md`

In short:

- cluster granularity: broad macro-style clusters, target `4..8`, current reference target `4`
- GMM stability gate: median ARI `>= 0.90`, mean ARI `>= 0.75`, reference median ARI `>= 0.90`, per-cluster median Jaccard `>= 0.90`
- uncertain GMM assignments: show normally by default, keep posterior-weighted ranking and hard filters as optional controls
- MSD numeric metadata: restore only as an explicit experiment after `>= 98%` live audio-backed coverage and a clean audit

## Audio contract

- mono audio
- `22050 Hz`
- `29.0` seconds
- `PCM_16`
- target loudness `-14 LUFS`
- peak ceiling `-1.0 dBFS` sample peak

## Explicit comparison baselines

These are documented on purpose so they do not quietly become defaults again:

- `all_audio_pca_comparison`
  - full handcrafted audio set
  - `pca_per_group`
  - expected prepared dimension `50`
- `all_audio_zscore_comparison`
  - full handcrafted audio set
  - plain `zscore` standardization
  - expected prepared dimension `116`
- comparison methods on any profile:
  - `K-Means`
  - `HDBSCAN`
  - optional `VaDE`

They are comparison-only baselines, not the production default path.

## Evaluation limitation

Recommendation quality is still judged offline with metadata proxies:

- genre overlap
- artist overlap
- coverage/diversity checks

That is useful for model selection, but it is not the same thing as human
similarity judgment.

## Future validation backlog

- run a small blind listening study on top-K recommendations from the production profile
- collect human judgments for edge cases such as uncertain GMM assignments
- revisit MSD numeric metadata only as an explicitly logged experiment after coverage improves
