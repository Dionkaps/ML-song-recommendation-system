# Decision Policy

This document records the currently explicit product and pipeline decisions for the
supported baseline.

The machine-readable source of truth remains:

- `config/experiment_profiles.py`
- `config/feature_vars.py`

## 1. Cluster granularity

Decision:

- prefer broad macro-style clusters, not many micro-style clusters
- target occupied-cluster range for the supported GMM baseline: `4..8`
- current reference target: `4`

Why:

- the current supported product path favors navigable, stable recommendation
  neighborhoods over highly fragmented stylistic micro-clusters
- the selected production GMM currently lands at `4` clusters, which fits this target

## 2. Minimum acceptable GMM stability gate

Decision:

- a production-default GMM must clear all of the following:
  - subsample pairwise median ARI >= `0.90`
  - subsample pairwise mean ARI >= `0.75`
  - reference median ARI >= `0.90`
  - per-cluster median best-match Jaccard >= `0.90`

Why:

- median-only stability can hide a weak tail
- mean-only stability can over-penalize otherwise healthy models
- the combined gate keeps the supported GMM baseline in the range of "usable but not fragile"

## 3. Uncertain GMM assignments

Decision:

- show uncertain assignments normally by default
- do not hard-filter uncertain assignments by default
- keep posterior-weighted ranking and hard confidence/posterior thresholds available as operator controls

Operational default:

- ranking method: `distance`
- minimum assignment confidence: `0.0`
- minimum selected-cluster posterior: `0.0`

Why:

- hard filtering hurts coverage too early
- posterior-weighted ranking is still useful as an exploratory/operator mode
- but the current offline proxy evaluation showed that making posterior-weighting the default reduced GMM recommendation quality and catalog breadth enough to worsen its overall ranking
- the supported baseline therefore keeps normal distance ranking as the product default while still exposing the probabilistic controls

## 4. MSD numeric metadata return gate

Decision:

- MSD numeric metadata returns only as an explicit experiment, not as a silent default

Required gate:

- live audio-backed MSD numeric coverage >= `98%`
- missing live audio-backed rows <= `100`
- clean metadata/schema audit
- no silent fallback or silent imputation path
- fresh comparison rerun under the explicit experiment profile

Current status:

- not ready
- latest summary shows `4783 / 5535` live audio-backed rows with numeric MSD features
- coverage is about `86.41%`
- missing live audio-backed rows: `752`

Source:

- `data/songs_schema_summary.json`
