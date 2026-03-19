# P0 Implementation Report

Generated: 2026-03-14

## Request handled

This report documents the implementation of:

- `## P0 - Immediate blockers and baseline alignment`

from:

- `docs/reports/implementation_todo_20260314.md`

## Executive summary

P0 has been implemented for the currently supported clustering baseline.

The workspace is now aligned around one explicit operating mode:

- audio-only clustering
- `spectral_plus_beat` as the clustering feature subset
- per-group `StandardScaler` + `pca_per_group_5`
- genre kept as metadata only
- MSD numeric metadata disabled by default
- `GMM` set as the default clustering method for the main pipeline entrypoint

The most important code change is in the shared clustering data-preparation path in `src/clustering/kmeans.py`. That path now builds clustering vectors from the configured subset instead of always forcing the full 10 handcrafted groups into clustering.

## P0 items completed

All items in the P0 section of `docs/reports/implementation_todo_20260314.md` were marked complete:

- genre remains metadata-only
- explicit `spectral_plus_beat` clustering subset added to config
- clustering dataset assembly now excludes `mfcc`, `delta_mfcc`, `delta2_mfcc`, and `chroma`
- summarization contract is preserved
- prepared representation is verified to reach 30 dimensions under `pca_per_group_5`
- preprocessing invariants were centralized and documented
- loudness normalization was made explicit as a baseline invariant
- MSD metadata is disabled by default
- one supported operating mode is documented clearly
- GMM is now the default clustering method in the main runner

## Design decision

The supported mode chosen for P0 is:

- `audio_only_spectral_plus_beat`

This matches the research recommendation and the observed workspace evidence in `docs/reports/workspace_technical_report_20260314.md`.

The alternative choice of “finish metadata migration first” was not adopted for P0 because:

- `data/songs.csv` is still absent in the checked workspace
- current experiment evidence already supports a strong audio-only baseline
- the metadata path is a migration task, not a blocker for establishing the recommended clustering baseline

## Files changed

### 1. `config/feature_vars.py`

Purpose:

- define the supported clustering baseline explicitly
- disable MSD metadata by default
- centralize preprocessing invariants used by wrappers and docs

Changes made:

- added `supported_clustering_mode = "audio_only_spectral_plus_beat"`
- added `default_clustering_method = "gmm"`
- changed `include_msd_features` from `True` to `False`
- added `clustering_feature_subset_name = "spectral_plus_beat"`
- added `clustering_audio_feature_keys` with the 6 recommended groups
- added preprocessing baseline constants:
  - `baseline_target_duration_seconds = 29.0`
  - `baseline_target_lufs = -14.0`
  - `baseline_max_true_peak_dbtp = -1.0`
  - `baseline_sample_rate = 22050`
  - `baseline_output_subtype = "PCM_16"`
  - `baseline_force_mono = True`

Why this matters:

- the recommended baseline is no longer implicit or scattered
- callers can now use one configuration source instead of repeating magic numbers

### 2. `src/clustering/kmeans.py`

Purpose:

- make the shared clustering dataset/prep path honor the configured subset
- preserve the summarization contract
- guarantee the expected prepared dimension under the active baseline

What changed structurally:

- added `_resolve_selected_audio_feature_keys()`
- added `_get_audio_group_specs()`
- added `_get_full_group_specs()`
- added `_summarize_feature_array()`
- added new canonical implementations for:
  - `build_group_weights()`
  - `equalize_features_pca()`
  - `prepare_features()`
  - `_collect_feature_vectors()`
  - `load_clustering_dataset()`
- added `_expected_prepared_dimension()`

Important implementation detail:

- the file already contained older shared prep helpers
- instead of rewriting every legacy block in place, new canonical versions were added later in the module using the same function names
- in Python, the later definitions become the active ones at import time
- this let P0 land safely without doing a risky full rewrite of the older legacy helper section in one pass

Behavior changes:

- clustering input is now driven by `fv.clustering_audio_feature_keys`
- the active subset is:
  - `spectral_centroid`
  - `spectral_rolloff`
  - `spectral_flux`
  - `spectral_flatness`
  - `zero_crossing_rate`
  - `beat_strength`
- excluded audio groups remain on disk and available for later analysis, but are no longer forced into clustering
- time-varying features still summarize to `mean + std`
- `beat_strength` still stays as the raw 4-scalar block
- feature arrays are now validated before summarization:
  - empty arrays are rejected
  - non-finite values are rejected
  - wrong dimensionality is rejected
  - malformed `beat_strength` shapes are rejected
- the prepared dimension is validated against the expected dimension computed from:
  - active group count
  - equalization mode
  - PCA components per group
  - MSD/genre inclusion flags

Why this matters:

- before P0, clustering always assembled the full handcrafted feature stack
- after P0, clustering actually follows the research-backed subset instead of only documenting it

### 3. `scripts/run_all_clustering.py`

Purpose:

- stop passing a metadata CSV path when MSD features are disabled

Change made:

- `songs_csv_path` is now only passed when `fv.include_msd_features` is `True`

Why this matters:

- the runner now matches the supported audio-only baseline instead of acting like `songs.csv` is required in all modes

### 4. `scripts/analysis/evaluate_clustering.py`

Purpose:

- keep evaluation aligned with the same default feature assembly mode

Change made:

- `songs_csv_path` is now only passed when MSD features are enabled

Why this matters:

- evaluation now follows the same default baseline assumptions as clustering

### 5. `scripts/run_audio_preprocessing.py`

Purpose:

- make preprocessing defaults come from the baseline contract instead of duplicated literals

Changes made:

- imports `config.feature_vars as fv`
- default CLI values now come from:
  - `fv.baseline_target_duration_seconds`
  - `fv.baseline_target_lufs`
  - `fv.baseline_max_true_peak_dbtp`

Why this matters:

- preprocessing invariants are now centralized instead of repeated

### 6. `run_pipeline.py`

Purpose:

- make the pipeline default clustering method GMM
- align preprocessing defaults with the supported baseline constants

Changes made:

- imports `config.feature_vars as fv`
- changed default `--clustering-method` from `kmeans` to `fv.default_clustering_method`
- changed loudness scan sample rate to `fv.baseline_sample_rate`
- changed preprocessing `headroom_db` to `fv.baseline_max_true_peak_dbtp`
- changed `AudioPreprocessor` call to use:
  - `fv.baseline_target_duration_seconds`
  - `fv.baseline_max_true_peak_dbtp`
  - `fv.baseline_sample_rate`
- changed verification meter/load path to use `fv.baseline_sample_rate`
- changed verification load to use `fv.baseline_force_mono`

Why this matters:

- GMM is now the default clustering choice in the main runner
- preprocessing defaults are explicit and consistent with the baseline contract

### 7. `README.md`

Purpose:

- document the supported operating mode clearly

Changes made:

- added a `Current Supported Baseline` section near the top
- documented the current supported mode as:
  - audio-only
  - `spectral_plus_beat`
  - `pca_per_group_5`
  - GMM default
  - genre metadata only
  - MSD disabled by default
- added link to `docs/SUPPORTED_BASELINE.md`
- changed the quick-start clustering text from weighted K-Means language to the GMM default baseline
- changed the clustering-method example ordering so GMM is presented as the default supported baseline

Why this matters:

- the repo now states the current supported baseline in one obvious place instead of leaving readers to infer it from experiment reports

### 8. `docs/SUPPORTED_BASELINE.md`

Purpose:

- provide a short durable reference for the supported baseline contract

Contents added:

- supported operating mode
- active clustering subset
- representation contract
- preprocessing invariants
- default entry points
- current limitations

Why this matters:

- it separates “current supported baseline” from older exploratory documentation

### 9. `docs/reports/implementation_todo_20260314.md`

Purpose:

- reflect actual progress in the implementation checklist

Changes made:

- all P0 checkboxes were marked complete

## Verification performed

### 1. Static compilation check

I ran Python bytecode compilation against all modified runtime files:

- `config/feature_vars.py`
- `src/clustering/kmeans.py`
- `src/clustering/gmm.py`
- `src/clustering/hdbscan.py`
- `src/clustering/vade.py`
- `scripts/run_all_clustering.py`
- `scripts/analysis/evaluate_clustering.py`
- `scripts/run_audio_preprocessing.py`
- `run_pipeline.py`

Result:

- all compiled successfully

### 2. Runtime verification of the new clustering subset path

Full 5.5k-track dataset preparation was too heavy for an interactive verification pass, so I ran a targeted runtime check through the actual shared prep code on a real sample of feature files using the project `.venv`.

What was verified:

- import path works with the current baseline config
- active subset resolves correctly
- summarization logic works on actual stored feature arrays
- `prepare_features()` transforms the new subset to the expected dimension

Sample verification result:

- active subset:
  - `spectral_centroid`
  - `spectral_rolloff`
  - `spectral_flux`
  - `spectral_flatness`
  - `zero_crossing_rate`
  - `beat_strength`
- sample raw summarized shape: `(12, 14)`
- sample prepared shape: `(12, 30)`
- expected prepared dimension: `30`

Observed runtime output:

```text
Feature equalization method: pca_per_group
Active clustering feature subset: ['spectral_centroid', 'spectral_rolloff', 'spectral_flux', 'spectral_flatness', 'zero_crossing_rate', 'beat_strength']
  SpectralCentroid: 2 dims -> 5 dims (padded with 3 zeros)
  SpectralRolloff: 2 dims -> 5 dims (padded with 3 zeros)
  SpectralFlux: 2 dims -> 5 dims (padded with 3 zeros)
  SpectralFlatness: 2 dims -> 5 dims (padded with 3 zeros)
  ZCR: 2 dims -> 5 dims (padded with 3 zeros)
  BeatStrength: 4 dims -> 5 dims (padded with 1 zeros)
  Total: 14 dims -> 30 dims (6 groups x 5 dims)
```

Interpretation:

- the active clustering subset now truly produces the intended `30-D` prepared representation
- this directly satisfies the most important P0 representation-alignment requirement

## What was intentionally not implemented yet

The following were left untouched because they belong to later todo phases:

- UI recommendation distance still uses later-phase logic and is not part of P0
- recommendation quality evaluation upgrades are still P1
- GMM hyperparameter tuning and model-selection upgrades are still P1
- `songs.csv` migration is still P1
- `run_pipeline.py` still contains older orchestration debt outside the P0 scope
- stale legacy docs outside the baseline section remain for later cleanup

## Important residual caveats

### 1. `src/clustering/kmeans.py` still contains older helper implementations earlier in the file

Current state:

- the new canonical shared prep helpers were added later in the module and now override the older ones

Why this is acceptable for now:

- Python resolves the later function definitions at runtime
- this kept P0 implementation risk lower

Why it should still be cleaned up later:

- the file is now harder to read than it should be
- a later cleanup pass should remove the superseded older helper bodies once the new path is fully settled

### 2. `run_pipeline.py` still has legacy behavior outside the P0 scope

Examples:

- it still references older orchestration patterns
- its preprocessing verification block still has legacy assumptions outside the baseline work
- the missing `loudness_scanner` module issue remains outside P0

This was intentionally not expanded into P0 because it belongs to the later technical-debt item already listed in the todo.

## Outcome against the original P0 goals

### Achieved

- one supported operating mode is now explicit
- clustering no longer defaults to the full handcrafted feature vector
- MSD is disabled by default
- GMM is the default clustering method in the main runner
- the `spectral_plus_beat` subset is real code behavior now, not just report language
- the 30-D prepared representation is verified on real feature data

### Not claimed

- no claim is made here that recommendation quality is improved yet
- no claim is made here that GMM is fully tuned yet
- no claim is made here that metadata migration is fixed yet
- no claim is made here that the UI is already aligned with the full research recommendation

## Recommended next step

The next implementation target should be:

- `## P1 - Recommendation quality fixes`

That is the next place where the code still materially disagrees with the research recommendation at product-behavior level.
