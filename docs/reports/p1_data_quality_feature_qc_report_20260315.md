# P1 Data Quality And Feature QC Report

Date: 2026-03-15

## Scope

This report covers implementation of the `P1 - Data quality and feature QC` section from `docs/reports/implementation_todo_20260314.md`.

Target outcomes:

- deterministic feature-extraction settings are enforced and documented
- clustering input validation is strict for missing arrays, wrong shapes, `NaN`, and `Inf`
- broken tracks are never silently imputed into clustering
- the workspace can re-extract or drop broken tracks and log the result
- QC artifacts are written for dropped, stale, or re-extracted tracks
- stale cached genre metadata is cleaned
- the active `spectral_plus_beat` clustering groups are verified to retain sane variance after scaling/equalization

## Executive Summary

The `P1 - Data quality and feature QC` slice is complete.

The main implementation result is that the project now has one shared validation contract for handcrafted feature bundles and one reproducible QC workflow built on top of the real clustering loader. Feature extraction now emits a reproducibility manifest, the clustering loader now validates feature bundles before building the dataset and writes a QC summary every time it runs, and a new standalone script can inspect the full library, optionally re-extract broken tracks, clean stale genre cache entries, and measure variance retention for the supported `spectral_plus_beat` baseline.

The actual library state on 2026-03-15 is healthy:

- `5535` audio tracks were scanned
- `5535` tracks were already complete and valid before any repair step
- `0` incomplete bundles were found
- `0` invalid bundles were found
- `0` re-extractions were needed
- `0` stale feature bundles were found
- `39` stale cached genre-map entries were removed
- `5535` tracks were loaded into the clustering dataset after QC
- `0` tracks were dropped from clustering

The variance check also passed cleanly. All six active `spectral_plus_beat` groups preserved their expected active dimensions in prepared space, and the rebuilt per-group equalization matched the runtime prepared representation exactly for every group.

## Files Added Or Updated

### New shared QC utility module

File: `src/features/feature_qc.py`

Added a single source of truth for feature-bundle validation and inventory:

- `FEATURE_KEYS` mirrors the configured handcrafted extraction set
- `get_feature_output_paths()` centralizes filename generation for bundle members
- `split_feature_filename()` safely parses feature bundle filenames without confusing `mfcc` with `delta_mfcc` or `delta2_mfcc`
- `collect_feature_bundle_inventory()` builds exact bundle coverage from disk
- `audio_library_basenames()` enumerates the actual audio library
- `validate_feature_array()` enforces structural rules:
  - no empty arrays
  - no `NaN`
  - no `Inf`
  - expected dimensionality per feature group
  - `beat_strength` must flatten to exactly `4` scalars
- `load_validated_feature_bundle()` returns either a fully validated bundle or a structured validation result describing missing/invalid keys
- `remove_feature_bundle()` supports clean re-extraction of broken tracks

This module is the foundation for both clustering-time validation and the standalone QC script.

### Deterministic feature-extraction manifest

File: `src/features/extract_features.py`

Added:

- `build_feature_extraction_manifest()` at the top of the module
- `write_feature_extraction_manifest()` to persist the run contract

Updated behavior:

- extraction now validates every array before writing it to disk
- the extraction run writes `output/features/feature_extraction_manifest.json`
- the manifest is written for:
  - normal extraction runs
  - already-complete resume runs
  - empty-library runs

The manifest captures:

- audio directory and results directory
- sorted lexical audio-file ordering
- executor type and worker count
- whether resume mode was enabled
- extracted feature keys
- audio preprocessing contract
  - `22050 Hz`
  - `29.0 s`
  - mono
  - `PCM_16`
  - `-14 LUFS`
  - `-1.0 dBTP`
- feature settings
  - `n_mfcc = 13`
  - `n_fft = 2048`
  - `hop_length = 512`
  - `n_chroma = 12`
- library versions
  - Python
  - platform
  - NumPy
  - librosa
  - soundfile

### Shared clustering-loader QC

File: `src/clustering/kmeans.py`

Implemented and/or extended:

- `_clean_genre_mapping_for_audio()`
- `_write_dataset_qc_report()`
- the active later `_collect_feature_vectors()` implementation
- new `load_clustering_dataset_bundle()`
- `load_clustering_dataset()` now wraps the bundle loader

Important behavior changes:

- the loader now resolves the active clustering feature subset explicitly
- the loader validates every selected feature array before summarizing it
- incomplete or invalid tracks are dropped instead of silently imputed
- dropped/incomplete/invalid/stale bundle information is collected into QC rows
- stale cached genre-map entries are cleaned against the current audio library
- every dataset load writes:
  - `output/metrics/clustering_dataset_qc_latest.csv`
  - `output/metrics/clustering_dataset_qc_summary_latest.json`

The loader also now exposes a richer bundle for analysis:

- `file_names`
- `genres`
- `unique_genres`
- `raw_features`
- `prepared_features`
- `qc_rows`
- `qc_summary`
- QC output paths

This made it possible to build QC reporting on top of the actual runtime dataset path rather than duplicating clustering assembly logic in a separate script.

### Full-library QC analysis workflow

File: `scripts/analysis/run_feature_qc.py`

Added a standalone script that performs the complete P1 workflow:

1. scan the real audio library
2. validate the full extracted feature bundle for every track
3. optionally re-extract incomplete or invalid bundles
4. report stale feature bundles
5. clean stale `genre_map.npy` entries
6. load the supported clustering dataset through `load_clustering_dataset_bundle()`
7. compute per-group raw, scaled, and prepared-space variance diagnostics
8. write timestamped CSV, JSON, and Markdown outputs

Outputs produced by this script:

- `output/metrics/feature_qc_20260315_track_status.csv`
- `output/metrics/feature_qc_20260315_group_variance.csv`
- `output/metrics/feature_qc_20260315_summary.json`
- `output/metrics/feature_qc_20260315_summary.md`

### Documentation updates

Files:

- `docs/SUPPORTED_BASELINE.md`
- `docs/reports/implementation_todo_20260314.md`

Changes:

- documented the feature-extraction manifest and reproducibility contract
- documented the QC artifacts produced by the clustering loader and full QC run
- marked the `P1 - Data quality and feature QC` checklist complete

## Commands Run

### 1. Feature extraction manifest generation

Command:

```powershell
.\.venv\Scripts\python.exe scripts/run_feature_extraction.py --audio-dir audio_files --results-dir output/features
```

Observed result:

- `5535` audio files found
- `5535` already complete
- `0` pending extraction
- manifest written successfully to `output/features/feature_extraction_manifest.json`

### 2. Full QC run

Command:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/run_feature_qc.py --audio-dir audio_files --results-dir output/features --songs-csv data/songs.csv
```

Observed result:

- full track scan completed successfully
- clustering dataset was rebuilt through the shared loader
- timestamped QC outputs were written
- runtime was `1721.8055` seconds, approximately `28.7` minutes

### 3. Python syntax verification

Command:

```powershell
.\.venv\Scripts\python.exe -m py_compile src/features/feature_qc.py src/features/extract_features.py src/clustering/kmeans.py scripts/analysis/run_feature_qc.py
```

Observed result:

- syntax verification passed for all touched Python files

## Measured Results

### Feature extraction reproducibility

Manifest file:

- `output/features/feature_extraction_manifest.json`

Key recorded values from the manifest:

- `manifest_version: 1`
- `audio_file_order: sorted_lexicographically`
- `resume_enabled: true`
- `executor_type: thread`
- `workers: 8`
- `feature_keys`: all `10` handcrafted audio feature groups
- `run_summary.total_audio_files: 5535`
- `run_summary.processed: 0`
- `run_summary.skipped: 5535`
- `run_summary.failed: 0`

This means the feature library can now be tied to an explicit extraction contract instead of relying on undocumented defaults.

### Library-wide track validation

Source file:

- `output/metrics/feature_qc_20260315_summary.json`

Measured totals:

- `audio_tracks_scanned = 5535`
- `complete_before_repair = 5535`
- `incomplete_before_repair = 0`
- `invalid_before_repair = 0`
- `reextract_attempted = 0`
- `reextract_succeeded = 0`
- `reextract_failed = 0`
- `stale_feature_bundles = 0`

Interpretation:

- there were no broken handcrafted feature bundles in the current library
- the repair path is implemented and available, but this run did not need it
- because validation is now strict at clustering load time, future broken bundles would be dropped instead of silently passing through

### Stale genre cache cleanup

Source files:

- `output/metrics/feature_qc_20260315_summary.json`
- `output/features/genre_map.npy`

Measured result:

- `39` stale cached genre-map entries were removed
- cached `genre_map.npy` size after cleanup: `5535`

This resolves the earlier mismatch between cached metadata and the actual audio library.

### Clustering dataset QC

Source file:

- `output/metrics/clustering_dataset_qc_summary_latest.json`

Measured result:

- `candidate_audio_tracks = 5535`
- `loaded_tracks = 5535`
- `dropped_incomplete_tracks = 0`
- `dropped_invalid_tracks = 0`
- `missing_all_feature_bundles = 0`
- `stale_feature_bundles = 0`
- `raw_feature_dimension = 14`
- `prepared_feature_dimension = 30`
- `include_genre = false`
- `include_msd_requested = false`
- `include_msd_effective = false`
- `unique_genre_count = 419`

Interpretation:

- the supported baseline loader now reaches the expected `14 -> 30` dimensional transformation cleanly
- no tracks were lost to data-quality issues in the current library
- genre metadata remains available for reporting/evaluation while staying excluded from clustering input

## Variance Retention Verification

### Why this check matters

The active baseline uses the `spectral_plus_beat` subset:

- `spectral_centroid`
- `spectral_rolloff`
- `spectral_flux`
- `spectral_flatness`
- `zero_crossing_rate`
- `beat_strength`

Each group must still carry meaningful variation after the scaling/equalization step. If a group collapses to near-zero variance, it effectively stops contributing to clustering.

### Important implementation detail

For this baseline, each selected group has dimensionality less than or equal to `pca_components_per_group = 5`:

- the five spectral groups each contribute `2` summarized dimensions
- `beat_strength` contributes `4` summarized dimensions

That means the current `pca_per_group_5` baseline does not discard variance for these groups. Instead:

- groups are standardized
- smaller groups are zero-padded up to `5` dimensions
- the padded group is normalized to equalize contribution

So for the supported `spectral_plus_beat` baseline, the variance question is really:

- do the expected active dimensions remain active after equalization?
- do the padded dimensions stay zero by design?
- does the rebuilt group transformation match the runtime prepared representation?

All three checks passed.

### Prepared-space results

Source file:

- `output/metrics/feature_qc_20260315_group_variance.csv`

Prepared-space runtime results:

| Feature group | Expected active dims | Expected zero-padded dims | Observed active dims | Observed near-zero dims | Mean active variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| SpectralCentroid | 2 | 3 | 2 | 3 | 0.499999 |
| SpectralRolloff | 2 | 3 | 2 | 3 | 0.500000 |
| SpectralFlux | 2 | 3 | 2 | 3 | 0.500000 |
| SpectralFlatness | 2 | 3 | 2 | 3 | 0.500000 |
| ZCR | 2 | 3 | 2 | 3 | 0.500000 |
| BeatStrength | 4 | 1 | 4 | 1 | 0.250000 |

Interpretation:

- every group preserved exactly the number of active dimensions it was expected to preserve
- every group preserved only the expected padded near-zero dimensions
- no unexpected dimensional collapse was observed
- the lower `0.25` mean active variance for `BeatStrength` is expected because the group has `4` active dimensions after equalization rather than `2`

### Runtime consistency check

The script rebuilt the per-group scaling/equalization transformation locally and compared it to the actual runtime slices from `load_clustering_dataset_bundle()`.

Result:

- `PreparedMatchesRuntime = True` for all six feature groups
- `ExplainedVarianceRatioSum = 1.0` for all six feature groups

Interpretation:

- the variance diagnostics match the actual clustering representation, not just an approximate reimplementation
- because all selected groups fit within `5` prepared dimensions, no variance was truncated by PCA in the supported baseline

## QC Artifacts Produced

### Loader-level QC

- `output/metrics/clustering_dataset_qc_latest.csv`
- `output/metrics/clustering_dataset_qc_summary_latest.json`

### Full run QC

- `output/metrics/feature_qc_20260315_track_status.csv`
- `output/metrics/feature_qc_20260315_group_variance.csv`
- `output/metrics/feature_qc_20260315_summary.json`
- `output/metrics/feature_qc_20260315_summary.md`

### Reproducibility artifact

- `output/features/feature_extraction_manifest.json`

## Outcome Against The Todo

Status for `P1 - Data quality and feature QC`:

- deterministic feature extraction: complete
- strict clustering-time validation: complete
- no silent imputation of broken tracks: complete
- QC report for dropped/stale/re-extracted tracks: complete
- stale cached genre metadata cleanup: complete
- variance-retention verification for `spectral_plus_beat`: complete

## Residual Notes

- The full-library QC pass is I/O heavy. On the current workspace it took about `28.7` minutes because it validates the entire handcrafted feature library from disk.
- Importing `src.clustering.kmeans` currently emits a `pygame-ce` banner in this environment because of an upstream import side effect from the UI path. This is noisy but did not block QC execution.
- The current library is clean, so the repair path was implemented but not exercised by a real broken track in this run. The logic is still present and would be used automatically on future invalid or incomplete bundles when `run_feature_qc.py` is executed with repair enabled.

## Final Conclusion

The project now has a reproducible, auditable data-quality gate in front of clustering.

The supported audio-only `spectral_plus_beat` baseline is currently in a clean state:

- feature extraction is documented and reproducible
- the handcrafted feature library validates cleanly
- stale genre cache drift has been corrected
- clustering input assembly rejects broken bundles instead of masking them
- the active equalization path preserves the intended group variance structure

This closes the `P1 - Data quality and feature QC` section with both code changes and verified workspace outputs.
