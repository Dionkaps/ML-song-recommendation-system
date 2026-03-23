# Full Pipeline Fresh Rerun Report

- Generated at: `2026-03-19T23:12:06`
- Project root: `.`
- Rerun log root: `docs/reports/run_logs/full_pipeline_rerun_20260319_225913_codex_fresh_retry`
- Rerun summary present: `no`

## Scope

This report summarizes a fresh end-to-end execution that removes generated artifacts, rebuilds metadata, reruns `run_pipeline.py` from the download stage, rebuilds the taxonomy-aware merged-genre dataset, executes the clustering profile suite, and then runs the thesis clustering benchmark.

## Step Execution

| Step | DurationSec | ReturnCode | Command | Log |
| --- | --- | --- | --- | --- |
| 01_rebuild_unified_metadata | n/a | n/a |  | docs/reports/run_logs/full_pipeline_rerun_20260319_225913_codex_fresh_retry/01_rebuild_unified_metadata.log |
| 02_run_pipeline | n/a | n/a |  | docs/reports/run_logs/full_pipeline_rerun_20260319_225913_codex_fresh_retry/02_run_pipeline.log |

## Download Stage

- Catalog rows: `10000`
- Processed rows: `n/a`
- Downloaded rows recorded in checkpoint: `n/a`
- Audio files on disk after download stage: `n/a`
- Remaining rows after the resilient download loop: `n/a`
- Forced stall restarts: `0`
- Download summary artifact: `docs/reports/run_logs/latest_resilient_download/run_summary.json`

### Download Sessions

| Session | DurationSec | Limit | ProcessedGain | DownloadedGain | AudioGain | Stalled | ExitCode | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 165.80 | 500 | 409 | 409 | 409 | no | 0 | docs/reports/run_logs/latest_resilient_download/01_download_session.log |
| 2 | 161.40 | 500 | 338 | 338 | 329 | no | 0 | docs/reports/run_logs/latest_resilient_download/02_download_session.log |
| 3 | 136.59 | 500 | 263 | 272 | 261 | no | 0 | docs/reports/run_logs/latest_resilient_download/03_download_session.log |
| 4 | 116.66 | 500 | 207 | 218 | 205 | no | 0 | docs/reports/run_logs/latest_resilient_download/04_download_session.log |
| 5 | 112.00 | 500 | 162 | 175 | 160 | no | 0 | docs/reports/run_logs/latest_resilient_download/05_download_session.log |

## Metadata Outputs

- Unified songs CSV: `data/songs.csv`
- Schema summary JSON: `data/songs_schema_summary.json`
- Taxonomy-aware merged CSV: `data/songs_with_merged_genres.csv`
- Unified rows: `9939`
- Unified audio-backed rows: `1364`
- Unified unique raw genre values: `3822`
- Merged rows: `n/a`
- Merged audio-backed rows: `n/a`
- Merged rows included in MRS: `n/a`
- Merged unique primary genres: `n/a`
- Merged unique primary tag sets: `n/a`
- Merged unique secondary tags: `n/a`

### Schema Summary Snapshot

| Metric | Value |
| --- | --- |
| audio_rows_with_msd_track_id | 0 |
| audio_rows_with_numeric_msd_features | 0 |
| audio_rows_without_msd_track_id | 0 |
| audio_rows_without_numeric_msd_features | 0 |
| current_audio_assigned_via_legacy_audio_mapping | 0 |
| current_audio_assigned_via_legacy_download | 0 |
| current_audio_assigned_via_normalized_msd_match | 0 |
| current_audio_audio_only_rows | 0 |
| current_audio_rows | 0 |
| legacy_audio_mapping_rows | 5254 |
| legacy_download_rows | 8016 |
| legacy_match_rows | 7797 |
| msd_catalog_conflicting_track_id_groups | 4 |
| msd_catalog_exact_duplicates_removed | 57 |
| msd_catalog_rows_after_exact_dedup | 9943 |
| msd_catalog_rows_final | 9939 |
| msd_catalog_rows_missing_track_id | 1 |
| msd_catalog_rows_original | 10000 |
| msd_catalog_track_id_duplicates_removed | 4 |
| msd_feature_rows | 10000 |
| unified_rows_final | 9939 |
| unified_rows_with_audio | 0 |
| unified_rows_with_msd_track_id | 9938 |

## Feature Extraction

- Feature output directory: `output/features`
- Feature manifest: `output/features/feature_extraction_manifest.json`
- Audio files seen by extraction: `n/a`
- Newly processed files: `n/a`
- Skipped existing bundles: `n/a`
- Extraction failures reported by manifest: `n/a`
- Detected feature bundles in output folder: `n/a`
- Complete feature bundles: `n/a`
- Incomplete feature bundles: `n/a`
- `.npy` files present: `n/a`

## Clustering Profile Suite

_Experiment suite manifest is not available yet._

## Thesis Benchmark

_Benchmark dataset summary is not available yet._

## Key Artifact Paths

| Artifact | Path |
| --- | --- |
| Rerun log root | docs/reports/run_logs/full_pipeline_rerun_20260319_225913_codex_fresh_retry |
| Rerun summary JSON | docs/reports/run_logs/full_pipeline_rerun_20260319_225913_codex_fresh_retry/rerun_summary.json |
| Latest resilient download summary | docs/reports/run_logs/latest_resilient_download/run_summary.json |
| Unified songs CSV | data/songs.csv |
| Schema summary JSON | data/songs_schema_summary.json |
| Merged taxonomy CSV | data/songs_with_merged_genres.csv |
| Feature manifest JSON | output/features/feature_extraction_manifest.json |
| Experiment run manifest |  |
| Benchmark directory |  |
| Benchmark report |  |

## Residual Notes

- `rerun_summary.json` is missing, which typically means the full rerun has not finished yet or stopped before the wrapper wrote its final summary.
