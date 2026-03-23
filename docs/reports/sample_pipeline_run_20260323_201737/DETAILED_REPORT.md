# Detailed Sample Pipeline Report

## Run Identity

- Run label: `sample_pipeline_run_20260323_201737`
- Date: `2026-03-23`
- Repository: `C:\Users\vpddk\Desktop\Me\Github\ML-song-recommendation-system`
- Execution mode: system `python`, no virtual environment
- Python version: `3.11.9`
- Input catalog: `data/millionsong_dataset.csv`
- Input catalog size at run start: `500` rows plus header
- Input catalog header at run start: `track_id,title,artist,genre`
- Report directory: `docs/reports/sample_pipeline_run_20260323_201737`
- Step logs: `docs/reports/sample_pipeline_run_20260323_201737/step_logs`

## Objective

This pass reran the full 500-row sample pipeline from scratch inside the main project folders and documented every major stage:

1. Reset generated state
2. Rebuild unified metadata before download
3. Download previews
4. Preprocess audio
5. Rebuild unified metadata after preprocessing
6. Extract features
7. Run feature QC
8. Audit metadata schema
9. Build merged genre dataset
10. Run standalone GMM clustering
11. Run the multi-profile clustering suite
12. Run the thesis benchmark
13. Validate the generated CSV outputs
14. Compare the run to previous passes

## Executive Summary

- All tracked stages completed successfully with exit code `0`.
- The run reproduced the same key counts as the previous two clean passes.
- Download stage: `439` successful downloads, `61` failed downloads, `0` forced restarts.
- Preprocessing stage: `437` retained audio files, `2` removed as too short.
- Unified metadata after preprocessing: `500` rows total, `437` audio-backed rows, `0` blank `msd_track_id` values.
- Clustering-ready dataset: `434` audio tracks included in MRS, `3` excluded by taxonomy.
- Thesis benchmark output: `434` aligned rows, `0` blank `MSDTrackID` values, `3,738` full-grid rows, `126` native-best rows.
- Recommended production ranking remained stable: `K-Means` first, `GMM` second, `HDBSCAN` third.

## Stage Timing Summary

| Step | Stage | Runtime (sec) | Runtime (approx) | Status |
| --- | --- | ---: | --- | --- |
| 1 | Reset generated state | 0.10 | 0m 0.1s | Success |
| 2 | Initial unified metadata rebuild | 0.75 | 0m 0.8s | Success |
| 3 | Resilient download | 286.09 | 4m 46.1s | Success |
| 4 | Audio preprocessing | 11.68 | 0m 11.7s | Success |
| 5 | Post-preprocessing metadata rebuild | 0.90 | 0m 0.9s | Success |
| 6 | Feature extraction | 531.22 | 8m 51.2s | Success |
| 7 | Feature QC | 2.90 | 0m 2.9s | Success |
| 8 | Metadata schema audit | 0.72 | 0m 0.7s | Success |
| 9 | Merged genre dataset build | 0.47 | 0m 0.5s | Success |
| 10 | Standalone GMM | 89.01 | 1m 29.0s | Success |
| 11 | Clustering profile suite | 209.18 | 3m 29.2s | Success |
| 12 | Thesis benchmark | 451.72 | 7m 31.7s | Success |
| Total | Full tracked run | 1584.74 | 26m 24.7s | Success |

## Important File-Lifecycle Note

Some outputs in `output/metrics` are rolling "latest" files and can be overwritten by later clustering steps in the same run.

- `output/metrics/clustering_dataset_qc_latest.csv`
- `output/metrics/clustering_dataset_qc_summary_latest.json`
- `output/metrics/gmm_selection_criteria.csv`
- `output/metrics/hdbscan_selection_criteria.csv`
- `output/metrics/kmeans_selection_criteria.csv`

For audit purposes, the safest step-specific evidence is:

- the stage log in `docs/reports/sample_pipeline_run_20260323_201737/step_logs`
- the dated reports such as `feature_qc_20260323_summary.json`
- the run-scoped outputs inside `output/experiment_runs_taxonomy/run_20260323_183514Z`
- the unique benchmark directory `output/metrics/thesis_benchmark_inplace_sample_500_report_run_20260323_201737`

## Step 1: Reset Generated State

**Purpose**

Return the workspace to a clean generated state without modifying the sampled reference catalog.

**Command**

```powershell
python scripts/utilities/reset_full_pipeline_state.py --apply
```

**Runtime**

- `0.10` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/01_reset.log`

**Observed Result**

- The reset script completed successfully.
- The log reported `Targets currently present: 0`.
- The log reported `Targets removed: 0`.
- The preserved inputs section confirmed that the sampled catalog and reference files were not targeted for deletion.

**Interpretation**

- This specific pass started from a state the reset script considered already clean for generated artifacts.
- That is acceptable because the run still rebuilt all downstream artifacts from the sampled catalog.

## Step 2: Initial Unified Metadata Rebuild

**Purpose**

Create a fresh unified `songs.csv` before any new downloads occur.

**Command**

```powershell
python scripts/utilities/migrate_to_unified_csv.py
```

**Runtime**

- `0.75` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/02_migrate_initial.log`

**Observed Result**

- `Rows written: 500`
- `Audio-backed rows: 0`
- `Rows with MSD track id: 500`
- `Audio-backed rows with numeric MSD features: 0`

**Outputs**

- `data/songs.csv`
- `data/songs_schema_summary.json`

**Interpretation**

- The pre-download unified metadata state is correct.
- Every sampled row has an MSD track ID before download begins.
- No audio-backed rows exist yet, which is expected at this stage.

## Step 3: Resilient Download

**Purpose**

Download preview audio for the sampled rows and rebuild download-derived metadata.

**Command**

```powershell
python scripts/utilities/run_resilient_download.py --chunk-size 500 --idle-timeout-sec 240 --max-stall-restarts 20 --max-no-progress-sessions 2 --log-dir docs/reports/sample_pipeline_run_20260323_201737/download_session_logs
```

**Runtime**

- `286.09` seconds

**Logs**

- Stage log: `docs/reports/sample_pipeline_run_20260323_201737/step_logs/03_resilient_download.log`
- Session logs: `docs/reports/sample_pipeline_run_20260323_201737/download_session_logs`

**Observed Result**

- `catalog_rows: 500`
- `processed_rows: 500`
- `downloaded_rows: 439`
- `audio_files: 439`
- `remaining_rows: 0`
- `forced_restart_count: 0`
- Attempt 1 finished with `439` success and `61` failed.
- Attempt 2 finished with `0` additional success and `61` still failed.
- The unified CSV updater reported `Updated 439 songs and appended 0 bootstrap rows`.
- The download metadata writer reported `Successfully saved 439 songs to data/songs_data_with_genre.csv`.

**Key Outputs**

- `audio_files/`
- `download_checkpoint_with_genre.json`
- `deezer_search_cache.json`
- `data/songs_data_with_genre.csv`
- `data/songs.csv`
- `src/data_collection/download_stats/`

**Interpretation**

- The downloader behaved deterministically again on the sampled catalog.
- `61` rows had no successful preview download in this pass.
- `0` forced restarts means the resilient session controller never had to recover a stall.
- This stage completed faster than the previous reruns. That likely reflects normal cache and local runtime variability rather than a logic change.

## Step 4: Audio Preprocessing

**Purpose**

Crop, loudness-normalize, and peak-limit downloaded previews, while removing files too short to support the downstream pipeline.

**Command**

```powershell
python scripts/run_audio_preprocessing.py --audio-dir audio_files
```

**Runtime**

- `11.68` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/04_audio_preprocessing.log`

**Observed Result**

- `Total files: 439`
- `Processed: 437`
- `Peak Limited: 334`
- `Removed (short): 2`
- `Errors: 0`

**Removed Files**

- `Swami - Give It What U Got (feat. Asuivre).mp3`
- `The Pharcyde - Italian For Goodbye (skit).mp3`

**Interpretation**

- The same two short previews were removed again, which is a strong reproducibility signal.
- The absence of preprocessing errors indicates the downloaded audio set was structurally healthy.

## Step 5: Post-Preprocessing Unified Metadata Rebuild

**Purpose**

Reconcile metadata against the post-preprocessing audio state so `songs.csv` reflects the final usable audio inventory.

**Command**

```powershell
python scripts/utilities/migrate_to_unified_csv.py
```

**Runtime**

- `0.90` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/05_migrate_post_preprocessing.log`

**Observed Result**

- `Rows written: 500`
- `Audio-backed rows: 437`
- `Rows with MSD track id: 500`
- `Audio-backed rows with numeric MSD features: 437`

**Interpretation**

- The corrected rebuild logic is working as intended.
- No bogus metadata-only audio rows appeared.
- Every retained audio-backed row still has an MSD track ID and numeric metadata fields where expected.

## Step 6: Feature Extraction

**Purpose**

Extract audio feature bundles for each retained audio file.

**Command**

```powershell
python src/features/extract_features.py
```

**Runtime**

- `531.22` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/06_extract_features.log`

**Observed Result**

- `Found 437 audio files`
- `Successfully processed: 437 files`
- `Failed: 0 files`
- Feature extraction manifest written to `output/features/feature_extraction_manifest.json`

**Key Outputs**

- `output/features/`
- `output/features/feature_extraction_manifest.json`

**Interpretation**

- This was the longest stage in this pass.
- The stage was slower than the previous reruns while still succeeding on every file. That suggests runtime variability driven by machine load, disk throughput, or thread scheduling rather than by correctness problems.
- The important correctness result is that every retained audio file received a feature bundle.

## Step 7: Feature QC

**Purpose**

Validate the extracted feature bundles and prepare the clustering-ready subset.

**Command**

```powershell
python scripts/analysis/run_feature_qc.py
```

**Runtime**

- `2.90` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/07_run_feature_qc.log`

**Observed Result**

- Loaded genre mapping for `434` songs from `data/songs_with_merged_genres.csv`
- Excluded `3` audio tracks without a taxonomy-backed primary genre
- Prepared clustering dataset: `434` tracks, raw dims `14`, prepared dims `30`
- Wrote `output/metrics/feature_qc_20260323_track_status.csv`
- Wrote `output/metrics/feature_qc_20260323_group_variance.csv`
- Wrote `output/metrics/feature_qc_20260323_summary.json`

**Interpretation**

- No incomplete or invalid feature bundles were found before repair.
- No re-extraction was necessary.
- The only reduction from `437` retained audio files to `434` clustering rows came from the taxonomy gate, not from feature corruption.

## Step 8: Metadata Schema Audit

**Purpose**

Audit the rebuilt unified metadata and generate distribution summaries.

**Command**

```powershell
python scripts/analysis/audit_metadata_schema.py --output-dir output/metrics
```

**Runtime**

- `0.72` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/08_audit_metadata_schema.log`

**Key Audit Results**

- `unified_rows: 500`
- `audio_backed_rows: 437`
- `current_audio_files: 437`
- `audio_basename_gap: 0`
- `audio_rows_with_numeric_msd_features: 437`
- `audio_primary_genre_unique_count: 197`
- `audio_exploded_genre_label_unique_count: 465`
- `audio_primary_genre_top_10_share: 0.19221967963386727`
- `audio_rows_mean_genre_count: 4.935926773455377`

**Key Outputs**

- `output/metrics/metadata_schema_audit_20260323_summary.json`
- `output/metrics/metadata_schema_audit_20260323.md`
- `output/metrics/metadata_primary_genre_distribution_20260323.csv`
- `output/metrics/metadata_label_genre_distribution_20260323.csv`

**Interpretation**

- `audio_basename_gap: 0` confirms that the metadata and the actual audio directory agree exactly on retained audio basenames.
- `437` audio-backed rows with numeric MSD features confirms there is no hidden metadata dropout among retained audio rows.
- The label space is broad and long-tailed, which is expected for this taxonomy-expanded sample.

## Step 9: Merged Genre Dataset Build

**Purpose**

Produce the merged-genre dataset used by recommendation and clustering workflows.

**Command**

```powershell
python scripts/utilities/build_merged_genre_dataset.py
```

**Runtime**

- `0.47` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/09_build_merged_genre_dataset.log`

**Observed Result**

- `rows: 500`
- `rows_with_audio: 437`
- `rows_included_in_mrs: 497`
- `audio_rows_included_in_mrs: 434`
- `excluded_rows: 3`
- `excluded_non_genre_only_rows: 3`
- `unique_primary_genres: 128`
- `unique_primary_tag_sets: 456`
- `unique_secondary_tags: 7`

**Excluded Audio Rows**

- `Dakis - Perasmena Mesanihta`
- `George Lopez - Only For The Young`
- `Thomas Battenstein - Sympathy`

**Interpretation**

- The merged dataset is internally coherent and matches prior runs exactly.
- The three exclusions are consistent with the taxonomy gate seen in feature QC and clustering QC.

## Step 10: Standalone GMM Run

**Purpose**

Run the standalone GMM clustering entrypoint and inspect the selected model.

**Command**

```powershell
python src/clustering/gmm.py --no-ui
```

**Runtime**

- `89.01` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/10_gmm_no_ui.log`

**Observed Result**

- Prepared clustering dataset: `434` tracks, raw dims `14`, prepared dims `30`
- Selected GMM: `components=4`, `covariance_type=full`, `reg_covar=1e-06`
- Selection metrics:
  - `BIC=-79949.12`
  - `AIC=-88025.97`
  - `avg_log_likelihood=105.9815`
  - `silhouette=0.1337`
  - `avg_confidence=0.9781`
  - `stability_ari=0.9567`
- Cluster sizes:
  - `0:219`
  - `1:31`
  - `2:117`
  - `3:67`

**Interpretation**

- The standalone GMM result reproduced the same selected model family as earlier passes.
- The clustering solution remains stable and high-confidence, even though it is not the top-ranked production recommender.

## Step 11: Clustering Profile Suite

**Purpose**

Run the three configured clustering profiles and generate run-scoped recommendation comparison outputs.

**Command**

```powershell
python scripts/run_all_clustering.py --profiles recommended_production all_audio_pca_comparison all_audio_zscore_comparison --publish-ui-snapshot --run-root output/experiment_runs_taxonomy
```

**Runtime**

- `209.18` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/11_run_all_clustering.log`

**Run Output Root**

- `output/experiment_runs_taxonomy/run_20260323_183514Z`

**Run-Scoped Comparison CSVs**

- `output/experiment_runs_taxonomy/run_20260323_183514Z/recommended_production/metrics/recommended_production_comparison.csv`
- `output/experiment_runs_taxonomy/run_20260323_183514Z/all_audio_pca_comparison/metrics/all_audio_pca_comparison_comparison.csv`
- `output/experiment_runs_taxonomy/run_20260323_183514Z/all_audio_zscore_comparison/metrics/all_audio_zscore_comparison_comparison.csv`

**Recommended Production Outcome**

| Rank | Method | SupportedQueryFraction | CatalogCoverage | NoiseFraction | MeanReturned |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | K-Means | 1.0000 | 0.9931 | 0.0000 | 10.0000 |
| 2 | GMM | 1.0000 | 0.9908 | 0.0000 | 10.0000 |
| 3 | HDBSCAN | 0.3502 | 0.3502 | 0.6498 | 3.4378 |

**Interpretation**

- The ranking matched the earlier passes exactly.
- K-Means and GMM both support the full query set, while HDBSCAN loses much of the catalog to noise.
- HDBSCAN's lower recommendation quality is consistent with its high noise fraction and lower supported-query coverage.
- The run-scoped outputs in `run_20260323_183514Z` should be preferred over rolling files in `output/metrics` when auditing this pass.

## Step 12: Thesis Benchmark

**Purpose**

Sweep the representation and preprocessing combinations used for the thesis-style benchmark and save the outputs in a unique directory.

**Command**

```powershell
python scripts/analysis/thesis_clustering_benchmark.py --output-dir output/metrics/thesis_benchmark_inplace_sample_500_report_run_20260323_201737
```

**Runtime**

- `451.72` seconds

**Log**

- `docs/reports/sample_pipeline_run_20260323_201737/step_logs/12_thesis_benchmark.log`

**Benchmark Output Directory**

- `output/metrics/thesis_benchmark_inplace_sample_500_report_run_20260323_201737`

**Observed Result**

- The benchmark evaluated the full representation list from `mfcc_only / raw_zscore` through `all_audio / pca_per_group_5`.
- Benchmark completion was successful.
- Benchmark report path:
  - `output/metrics/thesis_benchmark_inplace_sample_500_report_run_20260323_201737/benchmark_report.md`

**Key Table Sizes**

- `aligned_metadata.csv`: `434` rows
- Blank `MSDTrackID` values in `aligned_metadata.csv`: `0`
- `full_grid_results.csv`: `3,738` rows
- `native_best_results.csv`: `126` rows
- Unique groups in `native_best_results.csv`: `14`
- Unique preprocess modes in `native_best_results.csv`: `3`
- Unique methods in `native_best_results.csv`: `3`

**Interpretation**

- The benchmark output shape matches the expected combinatorics.
- The metadata alignment bug found in the earliest isolated audit is not present here.
- The benchmark log line about loading `0` MSD-feature songs remains expected for the supported baseline because MSD metadata is intentionally disabled there.

## CSV Validation

### `data/songs.csv`

- Rows: `500`
- `has_audio=true`: `437`
- `has_audio=false`: `63`
- Blank `msd_track_id`: `0`
- Metadata origin counts:
  - `msd_catalog+legacy_download`: `439`
  - `msd_catalog`: `61`
- Audio match source counts:
  - `legacy_download`: `437`
  - blank: `63`

**Interpretation**

- The `63` non-audio rows make sense as `61` download misses plus `2` short-preview removals.

### `data/songs_data_with_genre.csv`

- Rows: `439`
- Role: download-stage record of successful preview acquisitions
- Design note: this file contains download metadata and does not carry `msd_track_id` as a column in its current shape

**Interpretation**

- The row count exactly matches successful downloads.
- This confirms the earlier checkpoint-key bug is not resurfacing.

### `data/songs_with_merged_genres.csv`

- Rows: `500`
- Audio rows: `437`
- Audio rows included in MRS: `434`
- Excluded audio rows: `3`

**Interpretation**

- The merged genre dataset is consistent with both feature QC and clustering.

### `output/metrics/feature_qc_20260323_track_status.csv`

- Rows: `437`
- Meaning: one QC row per retained audio track

### `output/metrics/clustering_dataset_qc_latest.csv`

- Rows after the full run: `3`
- Status values: all `excluded_by_taxonomy`

**Interpretation**

- The small size is correct.
- This file only contains the exclusion cases for the final clustering dataset snapshot.

### Metadata Distribution CSVs

- `output/metrics/metadata_primary_genre_distribution_20260323.csv`
  - Unique primary genres in audio-backed rows: `197`
  - Fraction sum: `0.9999999999999997`
- `output/metrics/metadata_label_genre_distribution_20260323.csv`
  - Unique exploded labels in audio-backed rows: `465`
  - Fraction sum: `0.9999999999999701`

**Top Primary Genres**

- `blues-rock`: `13`
- `hip hop`: `11`
- `chanson`: `9`
- `rock`: `9`
- `pop rock`: `8`

**Top Exploded Labels**

- `rock`: `77`
- `pop rock`: `42`
- `hip hop`: `36`
- `blues-rock`: `35`
- `pop`: `32`

**Interpretation**

- Both distribution tables are numerically coherent because their fractions sum to approximately `1.0`.

### Benchmark CSVs

- `output/metrics/thesis_benchmark_inplace_sample_500_report_run_20260323_201737/aligned_metadata.csv`
  - `434` rows
  - `0` blank `MSDTrackID`
- `output/metrics/thesis_benchmark_inplace_sample_500_report_run_20260323_201737/full_grid_results.csv`
  - `3,738` rows
- `output/metrics/thesis_benchmark_inplace_sample_500_report_run_20260323_201737/native_best_results.csv`
  - `126` rows

**Interpretation**

- The benchmark tables are structurally sound and match the expected search space.

## Cross-Run Reproducibility

This was the third clean in-project sample pass. The following key values matched the previous two passes:

| Metric | Pass 1 | Pass 2 | This Pass |
| --- | ---: | ---: | ---: |
| Successful downloads | 439 | 439 | 439 |
| Retained audio files after preprocessing | 437 | 437 | 437 |
| Clustering-ready audio rows | 434 | 434 | 434 |
| Taxonomy exclusions | 3 | 3 | 3 |
| Blank `msd_track_id` in `songs.csv` | 0 | 0 | 0 |
| Blank `MSDTrackID` in benchmark alignment | 0 | 0 | 0 |
| `full_grid_results.csv` rows | 3738 | 3738 | 3738 |
| `native_best_results.csv` rows | 126 | 126 | 126 |
| Recommended production ranking | K-Means > GMM > HDBSCAN | K-Means > GMM > HDBSCAN | K-Means > GMM > HDBSCAN |

**Interpretation**

- The pipeline is reproducible on this 500-row sample in its current project-folder configuration.
- Runtime is not perfectly constant across passes, but the data integrity and benchmark shape are stable.

## Main Conclusions

- The third rerun passed end-to-end without requiring manual intervention.
- The earlier downloader checkpoint-key issue remains fixed.
- The earlier metadata rebuild issue remains fixed.
- The generated CSVs are internally consistent and match the actual audio inventory and benchmark population.
- The only audio reduction after download continues to come from the same two short previews.
- The only clustering reduction after feature extraction continues to come from the same three taxonomy exclusions.
- The recommendation and benchmark outputs remain stable across repeated clean runs.

## Recommended Next Actions

- If the goal is to return the repository to full-catalog mode, restore `data/millionsong_dataset.csv` from `data/millionsong_dataset.full_backup_20260323_184958.csv`.
- If the goal is to keep using the 500-row sample for repeated audits, preserve this report directory and the run-scoped outputs as the current reference baseline.
- If you want the same level of reporting for a larger sample, the simplest next step is to automate this report generation into a reusable script so future runs produce the same artifact set with less manual orchestration.
