# Full Pipeline Fresh Rerun Report

- Generated at: `2026-03-20T02:16:34`
- Project root: `.`
- Rerun log root: `docs/reports/run_logs/full_pipeline_rerun_20260319_235915_sync_fresh_start`
- Rerun summary present: `yes`

## Scope

This report summarizes a fresh end-to-end execution that removes generated artifacts, rebuilds metadata, reruns `run_pipeline.py` from the download stage, rebuilds the taxonomy-aware merged-genre dataset, executes the clustering profile suite, and then runs the thesis clustering benchmark.

## Step Execution

- Note: `run_pipeline.py was executed with --skip plot to avoid the non-essential bulk visualization stage during the fresh benchmark rerun.`
- Note: `Step 04 was rerun after patching GMM evaluation stability fits to retry with stronger regularization on singular subsamples.`

| Step | DurationSec | ReturnCode | Command | Log |
| --- | --- | --- | --- | --- |
| 01_rebuild_unified_metadata | 10.78 | 0 | .\.venv\Scripts\python.exe scripts/utilities/migrate_to_unified_csv.py | docs/reports/run_logs/full_pipeline_rerun_20260319_235915_sync_fresh_start/01_rebuild_unified_metadata.log |
| 02_run_pipeline | 4954.61 | 0 | .\.venv\Scripts\python.exe run_pipeline.py --skip plot --clustering-method gmm | docs/reports/run_logs/full_pipeline_rerun_20260319_235915_sync_fresh_start/02_run_pipeline.log |
| 03_build_merged_genre_dataset | 0.79 | 0 | .\.venv\Scripts\python.exe scripts/utilities/build_merged_genre_dataset.py | docs/reports/run_logs/full_pipeline_rerun_20260319_235915_sync_fresh_start/03_build_merged_genre_dataset.log |
| 04_run_all_clustering_profiles | 715.70 | 0 | .\.venv\Scripts\python.exe scripts/run_all_clustering.py --profiles recommended_production all_audio_pca_comparison all_audio_zscore_comparison --run-root c:\Users\vpddk\Desktop\Me\Github\ML-song-recommendation-system\output\experiment_runs_taxonomy | docs/reports/run_logs/full_pipeline_rerun_20260319_235915_sync_fresh_start/04_run_all_clustering_profiles.log |
| 05_run_thesis_benchmark | 1297.90 | 0 | .\.venv\Scripts\python.exe scripts/analysis/thesis_clustering_benchmark.py --output-dir c:\Users\vpddk\Desktop\Me\Github\ML-song-recommendation-system\output\metrics\thesis_benchmark_full_rerun_20260319_235915 | docs/reports/run_logs/full_pipeline_rerun_20260319_235915_sync_fresh_start/05_run_thesis_benchmark.log |

## Cleanup Targets

| Path | Type | ExistedBeforeCleanup |
| --- | --- | --- |
| audio_files | directory | yes |
| metrics | missing | no |
| output | directory | yes |
| src/data_collection/download_stats | missing | no |
| docs/reports/run_logs/latest_resilient_download | directory | yes |
| data/songs.csv | file | yes |
| data/songs_schema_summary.json | file | yes |
| data/songs_with_merged_genres.csv | missing | no |
| data/songs_genre_list.csv | missing | no |
| data/unique_genres.csv | missing | no |
| download_checkpoint_with_genre.json | file | yes |
| deezer_search_cache.json | missing | no |

## Download Stage

- Catalog rows: `10000`
- Processed rows: `1924`
- Downloaded rows recorded in checkpoint: `1935`
- Audio files on disk after download stage: `1923`
- Remaining rows after the resilient download loop: `8076`
- Forced stall restarts: `0`
- Catalog-to-audio yield: `19.23%`
- Processed-to-audio yield: `99.95%`
- Download summary artifact: `docs/reports/run_logs/latest_resilient_download/run_summary.json`

### Download Sessions

| Session | DurationSec | Limit | ProcessedGain | DownloadedGain | AudioGain | Stalled | ExitCode | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 170.95 | 500 | 409 | 409 | 409 | no | 0 | docs/reports/run_logs/latest_resilient_download/01_download_session.log |
| 2 | 151.90 | 500 | 338 | 338 | 329 | no | 0 | docs/reports/run_logs/latest_resilient_download/02_download_session.log |
| 3 | 137.89 | 500 | 263 | 272 | 261 | no | 0 | docs/reports/run_logs/latest_resilient_download/03_download_session.log |
| 4 | 117.63 | 500 | 207 | 218 | 205 | no | 0 | docs/reports/run_logs/latest_resilient_download/04_download_session.log |
| 5 | 113.52 | 500 | 162 | 175 | 160 | no | 0 | docs/reports/run_logs/latest_resilient_download/05_download_session.log |
| 6 | 109.45 | 500 | 135 | 150 | 133 | no | 0 | docs/reports/run_logs/latest_resilient_download/06_download_session.log |
| 7 | 99.32 | 500 | 105 | 122 | 104 | no | 0 | docs/reports/run_logs/latest_resilient_download/07_download_session.log |
| 8 | 93.41 | 500 | 78 | 96 | 77 | no | 0 | docs/reports/run_logs/latest_resilient_download/08_download_session.log |
| 9 | 89.26 | 500 | 54 | 73 | 54 | no | 0 | docs/reports/run_logs/latest_resilient_download/09_download_session.log |
| 10 | 88.31 | 500 | 37 | 8 | 42 | no | 0 | docs/reports/run_logs/latest_resilient_download/10_download_session.log |
| 11 | 83.36 | 500 | 30 | 17 | 32 | no | 0 | docs/reports/run_logs/latest_resilient_download/11_download_session.log |
| 12 | 83.63 | 500 | 27 | 21 | 29 | no | 0 | docs/reports/run_logs/latest_resilient_download/12_download_session.log |
| 13 | 78.85 | 500 | 22 | 32 | 21 | no | 0 | docs/reports/run_logs/latest_resilient_download/13_download_session.log |
| 14 | 84.39 | 500 | 18 | 9 | 20 | no | 0 | docs/reports/run_logs/latest_resilient_download/14_download_session.log |
| 15 | 78.41 | 500 | 12 | 1 | 13 | no | 0 | docs/reports/run_logs/latest_resilient_download/15_download_session.log |
| 16 | 83.17 | 500 | 8 | 16 | 8 | no | 0 | docs/reports/run_logs/latest_resilient_download/16_download_session.log |
| 17 | 78.27 | 500 | 6 | -8 | 8 | no | 0 | docs/reports/run_logs/latest_resilient_download/17_download_session.log |
| 18 | 80.63 | 500 | 4 | -1 | 5 | no | 0 | docs/reports/run_logs/latest_resilient_download/18_download_session.log |
| 19 | 78.61 | 500 | 5 | -1 | 6 | no | 0 | docs/reports/run_logs/latest_resilient_download/19_download_session.log |
| 20 | 79.23 | 500 | 3 | 7 | 3 | no | 0 | docs/reports/run_logs/latest_resilient_download/20_download_session.log |
| 21 | 78.87 | 500 | 2 | -5 | 3 | no | 0 | docs/reports/run_logs/latest_resilient_download/21_download_session.log |
| 22 | 78.63 | 500 | 1 | 4 | 1 | no | 0 | docs/reports/run_logs/latest_resilient_download/22_download_session.log |
| 23 | 78.71 | 500 | 1 | -8 | 2 | no | 0 | docs/reports/run_logs/latest_resilient_download/23_download_session.log |
| 24 | 74.56 | 500 | -1 | 1 | -1 | no | 0 | docs/reports/run_logs/latest_resilient_download/24_download_session.log |
| 25 | 73.58 | 500 | -1 | -11 | 0 | no | 0 | docs/reports/run_logs/latest_resilient_download/25_download_session.log |
| 26 | 36.87 | 500 | -1 | 0 | -1 | no | 0 | docs/reports/run_logs/latest_resilient_download/26_download_session.log |

## Metadata Outputs

- Unified songs CSV: `data/songs.csv`
- Schema summary JSON: `data/songs_schema_summary.json`
- Taxonomy-aware merged CSV: `data/songs_with_merged_genres.csv`
- Audio files currently on disk: `1734`
- Unified rows: `9939`
- Unified audio-backed rows: `1946`
- Unified unique raw genre values: `3822`
- Merged rows: `9939`
- Merged audio-backed rows: `1946`
- Merged rows included in MRS: `9816`
- Merged unique primary genres: `194`
- Merged unique primary tag sets: `0`
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
- Audio files seen by extraction: `1734`
- Newly processed files: `1734`
- Skipped existing bundles: `0`
- Extraction failures reported by manifest: `0`
- Detected feature bundles in output folder: `1734`
- Complete feature bundles: `1734`
- Incomplete feature bundles: `0`
- `.npy` files present: `17342`

### Extraction Contract

| Field | Value |
| --- | --- |
| audio_contract.expected_duration_seconds | 29.0 |
| audio_contract.expected_sample_rate_hz | 22050 |
| audio_contract.force_mono | True |
| audio_contract.max_true_peak_dbtp | -1.0 |
| audio_contract.output_subtype | PCM_16 |
| audio_contract.target_lufs | -14.0 |
| feature_settings.hop_length | 512 |
| feature_settings.n_chroma | 12 |
| feature_settings.n_fft | 2048 |
| feature_settings.n_mfcc | 13 |
| library_versions.librosa | 0.11.0 |
| library_versions.numpy | 2.4.2 |
| library_versions.platform | Windows-10-10.0.26200-SP0 |
| library_versions.python | 3.11.9 |
| library_versions.soundfile | 0.13.1 |

## Clustering Profile Suite

- Experiment run manifest: `output/experiment_runs_taxonomy/run_20260319_234100Z/run_manifest.json`
- Requested profiles: `['recommended_production', 'all_audio_pca_comparison', 'all_audio_zscore_comparison']`
- Methods override: `None`
- Include VaDE: `False`
- Skip evaluation: `False`

### Profile `recommended_production`

- Title: `Recommended Production Baseline`
- Artifact dir: `output/experiment_runs_taxonomy/run_20260319_234100Z/recommended_production/clustering_results`
- Metrics dir: `output/experiment_runs_taxonomy/run_20260319_234100Z/recommended_production/metrics`
- Methods run: `['kmeans', 'gmm', 'hdbscan']`
- Evaluation summary JSON: `output/experiment_runs_taxonomy/run_20260319_234100Z/recommended_production/metrics/recommended_production_summary.json`
- Comparison CSV: `output/experiment_runs_taxonomy/run_20260319_234100Z/recommended_production/metrics/recommended_production_comparison.csv`
- Comparison report: `output/experiment_runs_taxonomy/run_20260319_234100Z/recommended_production/metrics/recommended_production_comparison_report.md`

Representation contract:

| Field | Value |
| --- | --- |
| artifact_version | 2 |
| equalization_method | pca_per_group |
| feature_subset_name | spectral_plus_beat |
| mismatch_methods | [] |
| pca_components_per_group | 5 |
| prepared_feature_dimension | 30 |
| profile_id | recommended_production |
| raw_feature_dimension | 14 |
| selected_audio_feature_keys | ['spectral_centroid', 'spectral_rolloff', 'spectral_flux', 'spectral_flatness', 'zero_crossing_rate', 'beat_strength'] |

Top-ranked methods:

| Rank | Method | PrimaryTagJaccard@10 | AllTagJaccard@10 | CatalogCoverage | SubsampleMedianARI | ClusterCount |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | K-Means | 0.0641 | 0.0632 | 0.9848 | 0.9600 | 2 |
| 2 | GMM | 0.0626 | 0.0620 | 0.9883 | 0.8270 | 4 |
| 3 | HDBSCAN | 0.0621 | 0.0614 | 0.9644 | 0.7159 | 3 |

Decision-policy assessment:

- `uncertain_assignments`: `{'status': 'aligned', 'policy': 'show_normally_with_optional_controls', 'active_ranking_method': 'distance', 'active_min_confidence': 0.0, 'active_min_posterior': 0.0}`
- `msd_restore_gate`: `{'status': 'not_ready', 'coverage_fraction': 0.0, 'missing_audio_rows': 0, 'audio_rows_with_numeric_msd_features': 0, 'current_audio_rows': 0, 'required_coverage_fraction': 0.98, 'required_max_missing_audio_rows': 100, 'summary_path': 'data\\songs_schema_summary.json'}`
- `cluster_granularity`: `{'status': 'pass', 'policy': 'broad_macro_clusters', 'selected_cluster_count': 4, 'target_cluster_range': [4, 8]}`
- `gmm_stability_gate`: `{'status': 'fail', 'subsample_median_ari': 0.8269856579348559, 'subsample_mean_ari': 0.7918280543950649, 'reference_median_ari': 0.8515090728611012, 'per_cluster_median_jaccard': 0.8952042986577806, 'required_subsample_median_ari': 0.9, 'required_subsample_mean_ari': 0.75, 'required_reference_median_ari': 0.9, 'required_per_cluster_median_jaccard': 0.9}`

### Profile `all_audio_pca_comparison`

- Title: `All Audio PCA Comparison`
- Artifact dir: `output/experiment_runs_taxonomy/run_20260319_234100Z/all_audio_pca_comparison/clustering_results`
- Metrics dir: `output/experiment_runs_taxonomy/run_20260319_234100Z/all_audio_pca_comparison/metrics`
- Methods run: `['kmeans', 'gmm', 'hdbscan']`
- Evaluation summary JSON: `output/experiment_runs_taxonomy/run_20260319_234100Z/all_audio_pca_comparison/metrics/all_audio_pca_comparison_summary.json`
- Comparison CSV: `output/experiment_runs_taxonomy/run_20260319_234100Z/all_audio_pca_comparison/metrics/all_audio_pca_comparison_comparison.csv`
- Comparison report: `output/experiment_runs_taxonomy/run_20260319_234100Z/all_audio_pca_comparison/metrics/all_audio_pca_comparison_comparison_report.md`

Representation contract:

| Field | Value |
| --- | --- |
| artifact_version | 2 |
| equalization_method | pca_per_group |
| feature_subset_name | all_audio |
| mismatch_methods | [] |
| pca_components_per_group | 5 |
| prepared_feature_dimension | 50 |
| profile_id | all_audio_pca_comparison |
| raw_feature_dimension | 116 |
| selected_audio_feature_keys | ['mfcc', 'delta_mfcc', 'delta2_mfcc', 'spectral_centroid', 'spectral_rolloff', 'spectral_flux', 'spectral_flatness', 'zero_crossing_rate', 'chroma', 'beat_strength'] |

Top-ranked methods:

| Rank | Method | PrimaryTagJaccard@10 | AllTagJaccard@10 | CatalogCoverage | SubsampleMedianARI | ClusterCount |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | GMM | 0.0697 | 0.0687 | 0.9661 | 0.7706 | 4 |
| 2 | K-Means | 0.0694 | 0.0685 | 0.9550 | 0.9665 | 2 |
| 3 | HDBSCAN | 0.0281 | 0.0278 | 0.3754 | 0.3981 | 2 |

Decision-policy assessment:

- `uncertain_assignments`: `{'status': 'aligned', 'policy': 'show_normally_with_optional_controls', 'active_ranking_method': 'distance', 'active_min_confidence': 0.0, 'active_min_posterior': 0.0}`
- `msd_restore_gate`: `{'status': 'not_ready', 'coverage_fraction': 0.0, 'missing_audio_rows': 0, 'audio_rows_with_numeric_msd_features': 0, 'current_audio_rows': 0, 'required_coverage_fraction': 0.98, 'required_max_missing_audio_rows': 100, 'summary_path': 'data\\songs_schema_summary.json'}`
- `cluster_granularity`: `{'status': 'pass', 'policy': 'broad_macro_clusters', 'selected_cluster_count': 4, 'target_cluster_range': [4, 8]}`
- `gmm_stability_gate`: `{'status': 'fail', 'subsample_median_ari': 0.7705628572056369, 'subsample_mean_ari': 0.7710581117826366, 'reference_median_ari': 0.7650044077916298, 'per_cluster_median_jaccard': 0.8423093651064688, 'required_subsample_median_ari': 0.9, 'required_subsample_mean_ari': 0.75, 'required_reference_median_ari': 0.9, 'required_per_cluster_median_jaccard': 0.9}`

### Profile `all_audio_zscore_comparison`

- Title: `All Audio Raw Z-Score Comparison`
- Artifact dir: `output/experiment_runs_taxonomy/run_20260319_234100Z/all_audio_zscore_comparison/clustering_results`
- Metrics dir: `output/experiment_runs_taxonomy/run_20260319_234100Z/all_audio_zscore_comparison/metrics`
- Methods run: `['kmeans', 'gmm', 'hdbscan']`
- Evaluation summary JSON: `output/experiment_runs_taxonomy/run_20260319_234100Z/all_audio_zscore_comparison/metrics/all_audio_zscore_comparison_summary.json`
- Comparison CSV: `output/experiment_runs_taxonomy/run_20260319_234100Z/all_audio_zscore_comparison/metrics/all_audio_zscore_comparison_comparison.csv`
- Comparison report: `output/experiment_runs_taxonomy/run_20260319_234100Z/all_audio_zscore_comparison/metrics/all_audio_zscore_comparison_comparison_report.md`

Representation contract:

| Field | Value |
| --- | --- |
| artifact_version | 2 |
| equalization_method | zscore |
| feature_subset_name | all_audio |
| mismatch_methods | [] |
| pca_components_per_group |  |
| prepared_feature_dimension | 116 |
| profile_id | all_audio_zscore_comparison |
| raw_feature_dimension | 116 |
| selected_audio_feature_keys | ['mfcc', 'delta_mfcc', 'delta2_mfcc', 'spectral_centroid', 'spectral_rolloff', 'spectral_flux', 'spectral_flatness', 'zero_crossing_rate', 'chroma', 'beat_strength'] |

Top-ranked methods:

| Rank | Method | PrimaryTagJaccard@10 | AllTagJaccard@10 | CatalogCoverage | SubsampleMedianARI | ClusterCount |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | GMM | 0.0693 | 0.0687 | 0.8669 | 0.8697 | 4 |
| 2 | K-Means | 0.0691 | 0.0687 | 0.8739 | 0.9703 | 2 |
| 3 | HDBSCAN | 0.0284 | 0.0285 | 0.3508 | 0.4443 | 2 |

Decision-policy assessment:

- `uncertain_assignments`: `{'status': 'aligned', 'policy': 'show_normally_with_optional_controls', 'active_ranking_method': 'distance', 'active_min_confidence': 0.0, 'active_min_posterior': 0.0}`
- `msd_restore_gate`: `{'status': 'not_ready', 'coverage_fraction': 0.0, 'missing_audio_rows': 0, 'audio_rows_with_numeric_msd_features': 0, 'current_audio_rows': 0, 'required_coverage_fraction': 0.98, 'required_max_missing_audio_rows': 100, 'summary_path': 'data\\songs_schema_summary.json'}`
- `cluster_granularity`: `{'status': 'pass', 'policy': 'broad_macro_clusters', 'selected_cluster_count': 4, 'target_cluster_range': [4, 8]}`
- `gmm_stability_gate`: `{'status': 'fail', 'subsample_median_ari': 0.8696678899128529, 'subsample_mean_ari': 0.7984121815290461, 'reference_median_ari': 0.8924853296463031, 'per_cluster_median_jaccard': 0.924314686413712, 'required_subsample_median_ari': 0.9, 'required_subsample_mean_ari': 0.75, 'required_reference_median_ari': 0.9, 'required_per_cluster_median_jaccard': 0.9}`

## Thesis Benchmark

- Benchmark directory: `output/metrics/thesis_benchmark_full_rerun_20260319_235915`
- Benchmark report: `output/metrics/thesis_benchmark_full_rerun_20260319_235915/benchmark_report.md`
- Songs evaluated: `1713`
- Unique genres: `163`
- Unique artists: `1365`
- Raw feature dimensions: `116`
- Representation families: `14`
- Preprocess modes: `['raw_zscore', 'pca_per_group_2', 'pca_per_group_5']`
- Matched cluster targets: `[4, 8, 12, 16, 20]`
- Raw cache path: `output/metrics/thesis_benchmark_full_rerun_20260319_235915/raw_audio_feature_cache.npz`

### Representation Catalog

| Combo | Groups | NGroups | RawDims | Rationale |
| --- | --- | --- | --- | --- |
| mfcc_only | mfcc | 1 | 26 | Canonical timbre baseline built from static cepstral coefficients. |
| delta_mfcc_only | delta_mfcc | 1 | 26 | First-order timbral dynamics without static cepstral context. |
| delta2_mfcc_only | delta2_mfcc | 1 | 26 | Second-order timbral dynamics to test acceleration-style temporal structure. |
| timbre_full | mfcc+delta_mfcc+delta2_mfcc | 3 | 78 | Full classical timbre stack combining static, delta, and delta-delta MFCC information. |
| spectral_shape | spectral_centroid+spectral_rolloff+spectral_flux+spectral_flatness+zero_crossing_rate | 5 | 10 | Brightness and spectral-shape family using scalar spectral descriptors. |
| pitch_only | chroma | 1 | 24 | Pitch-class profile baseline using chroma only. |
| rhythm_only | beat_strength | 1 | 4 | Pulse-strength baseline using beat-strength descriptors only. |
| pitch_rhythm | chroma+beat_strength | 2 | 28 | Joint harmonic-rhythmic view without timbral descriptors. |
| timbre_pitch | mfcc+delta_mfcc+delta2_mfcc+chroma | 4 | 102 | Timbre with pitch information to test whether harmonic context strengthens timbral organization. |
| timbre_spectral | mfcc+delta_mfcc+delta2_mfcc+spectral_centroid+spectral_rolloff+spectral_flux+spectral_flatness+zero_crossing_rate | 8 | 88 | Timbre plus spectral-shape descriptors for a broad timbral-textural representation. |
| spectral_pitch | spectral_centroid+spectral_rolloff+spectral_flux+spectral_flatness+zero_crossing_rate+chroma | 6 | 34 | Spectral-shape plus pitch-class energy for brightness-harmonic interactions. |
| spectral_rhythm | spectral_centroid+spectral_rolloff+spectral_flux+spectral_flatness+zero_crossing_rate+beat_strength | 6 | 14 | Spectral-shape plus rhythm to test textural-pulse structure. |
| timbre_pitch_rhythm | mfcc+delta_mfcc+delta2_mfcc+chroma+beat_strength | 5 | 106 | Multifamily combination covering timbre, pitch, and rhythm without explicit spectral scalars. |
| all_audio | mfcc+delta_mfcc+delta2_mfcc+spectral_centroid+spectral_rolloff+spectral_flux+spectral_flatness+zero_crossing_rate+chroma+beat_strength | 10 | 116 | Full engineered audio stack used as the omnibus reference representation. |

### Global Native Leaders

| method | combo | preprocess_mode | n_clusters | silhouette | nmi | stability_ari | fit_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kmeans | mfcc_only | pca_per_group_5 | 2.0 | 0.2936407625675201 | 0.0507708350785762 | 1.0 | 0.0878453000914305 |
| gmm | spectral_pitch | pca_per_group_5 | 2.0 | 0.2288158535957336 | 0.0639565593070699 | 0.9981329323602504 | 0.009274800075218 |
| hdbscan | rhythm_only | pca_per_group_2 | 170.0 | 0.4875356693331801 | 0.5020860848529286 | 0.8765062602701591 | 0.0355642999056726 |

### Global Matched-Granularity Leaders

| matched_target_clusters | method | combo | preprocess_mode | n_clusters | cluster_gap | silhouette | nmi | stability_ari |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | kmeans | spectral_shape | pca_per_group_5 | 4.0 | 0.0 | 0.2564255893230438 | 0.1190669373374974 | 1.0 |
| 8 | kmeans | pitch_rhythm | raw_zscore | 8.0 | 0.0 | 0.108121857047081 | 0.1695834335871509 | 0.716434511813453 |
| 12 | kmeans | rhythm_only | pca_per_group_2 | 12.0 | 0.0 | 0.3454001247882843 | 0.1806171559612587 | 0.7482688510709394 |
| 16 | kmeans | pitch_only | pca_per_group_2 | 16.0 | 0.0 | 0.3442985117435455 | 0.218719940044398 | 0.8377689889612098 |
| 20 | kmeans | pitch_only | pca_per_group_2 | 20.0 | 0.0 | 0.3490083515644073 | 0.2387233542634895 | 0.7911702198531506 |
| 4 | gmm | pitch_only | pca_per_group_5 | 4.0 | 0.0 | 0.1634308099746704 | 0.0901284666854149 | 0.5628068309996592 |
| 8 | gmm | pitch_rhythm | pca_per_group_2 | 8.0 | 0.0 | 0.1930252611637115 | 0.1804918188201819 | 0.5606270639245212 |
| 12 | gmm | pitch_only | pca_per_group_5 | 12.0 | 0.0 | 0.14699387550354 | 0.1989673759822783 | 0.608329245193701 |
| 16 | gmm | pitch_only | pca_per_group_2 | 16.0 | 0.0 | 0.3343261778354645 | 0.2168176775844144 | 0.5629059710405143 |
| 20 | gmm | pitch_only | pca_per_group_2 | 20.0 | 0.0 | 0.3328544199466705 | 0.2399730082654917 | 0.6545068487454944 |
| 4 | hdbscan | timbre_full | pca_per_group_5 | 4.0 | 0.0 | 0.2027502778985623 | 0.0407164872934394 | 0.5314231625439192 |
| 8 | hdbscan | rhythm_only | pca_per_group_2 | 8.0 | 0.0 | 0.0452431864336349 | 0.1234872660796912 | 0.9938684985064749 |
| 12 | hdbscan | timbre_full | pca_per_group_2 | 12.0 | 0.0 | 0.1402920462130442 | 0.1971778384513619 | -0.0070664286641775 |
| 16 | hdbscan | rhythm_only | pca_per_group_2 | 16.0 | 0.0 | 0.1634574361148496 | 0.2672096037027804 | 0.3002517928211022 |
| 20 | hdbscan | rhythm_only | raw_zscore | 20.0 | 0.0 | 0.0468739245615809 | 0.3200065228204247 | 0.5109199087659139 |

### Top Native Operating Points Across All Representations

| method | combo | preprocess_mode | n_clusters | silhouette | nmi | stability_ari | internal_selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hdbscan | pitch_only | pca_per_group_2 | 148.0 | 0.3922895710727521 | 0.5077771527158347 | 0.8177087836192182 | 0.9015546850242524 |
| hdbscan | delta_mfcc_only | pca_per_group_2 | 145.0 | 0.3962590304040824 | 0.5066075922747807 | 0.8900800866632471 | 0.877391796704744 |
| hdbscan | rhythm_only | pca_per_group_2 | 170.0 | 0.4875356693331801 | 0.5020860848529286 | 0.8765062602701591 | 0.9882950530498552 |
| hdbscan | mfcc_only | pca_per_group_2 | 143.0 | 0.3928858200068669 | 0.4979761878957303 | 0.867274207887064 | 0.893459444324409 |
| hdbscan | delta2_mfcc_only | pca_per_group_2 | 140.0 | 0.3961207625237778 | 0.4950853250939278 | 0.8999209056729714 | 0.8811805986563285 |
| hdbscan | spectral_pitch | pca_per_group_2 | 2.0 | 0.493341858663897 | 0.3548669283161461 | nan | 0.71042503630184 |
| hdbscan | pitch_only | pca_per_group_5 | 3.0 | 0.4175736926812511 | 0.3267500092869116 | 0.7138991229500179 | 0.7402665977857623 |
| hdbscan | timbre_pitch_rhythm | pca_per_group_2 | 2.0 | 0.3972894063159617 | 0.2594218564424074 | nan | 0.74378808703337 |
| hdbscan | pitch_rhythm | pca_per_group_5 | 2.0 | 0.4391422971227591 | 0.2428190077691513 | nan | 0.7458667969364138 |
| hdbscan | mfcc_only | raw_zscore | 2.0 | 0.2277150041726668 | 0.2366458208285301 | 1.0 | 0.6727874041275567 |

### Top Matched-Granularity Operating Points

| matched_target_clusters | method | combo | preprocess_mode | n_clusters | cluster_gap | nmi | stability_ari |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | hdbscan | pitch_rhythm | pca_per_group_2 | 4.0 | 0.0 | 0.1803845353381778 | 0.5506765466872162 |
| 4 | hdbscan | delta_mfcc_only | pca_per_group_2 | 4.0 | 0.0 | 0.1803193551626037 | 0.3153454570468654 |
| 4 | gmm | spectral_pitch | raw_zscore | 4.0 | 0.0 | 0.1479692262607726 | 0.670846865218415 |
| 4 | gmm | timbre_pitch_rhythm | pca_per_group_5 | 4.0 | 0.0 | 0.1451007783185721 | 0.9954792916748704 |
| 4 | kmeans | all_audio | pca_per_group_2 | 4.0 | 0.0 | 0.1445110674716612 | 0.9652551785707332 |
| 4 | gmm | timbre_pitch | pca_per_group_5 | 4.0 | 0.0 | 0.1429254643673042 | 0.9696042148438656 |
| 4 | kmeans | timbre_pitch_rhythm | pca_per_group_2 | 4.0 | 0.0 | 0.1425884981681695 | 0.9979838738575976 |
| 4 | kmeans | pitch_rhythm | pca_per_group_5 | 4.0 | 0.0 | 0.141738836011824 | 0.97998081174944 |
| 4 | gmm | all_audio | pca_per_group_5 | 4.0 | 0.0 | 0.1411331441956582 | 0.9842254243692308 |
| 4 | gmm | all_audio | raw_zscore | 4.0 | 0.0 | 0.1396969331779712 | 0.9143875627634046 |
| 4 | kmeans | timbre_pitch_rhythm | pca_per_group_5 | 4.0 | 0.0 | 0.1396141346039697 | 0.9954923320901972 |
| 4 | kmeans | all_audio | pca_per_group_5 | 4.0 | 0.0 | 0.1395385338870884 | 1.0 |

## Key Artifact Paths

| Artifact | Path |
| --- | --- |
| Rerun log root | docs/reports/run_logs/full_pipeline_rerun_20260319_235915_sync_fresh_start |
| Rerun summary JSON | docs/reports/run_logs/full_pipeline_rerun_20260319_235915_sync_fresh_start/rerun_summary.json |
| Latest resilient download summary | docs/reports/run_logs/latest_resilient_download/run_summary.json |
| Unified songs CSV | data/songs.csv |
| Schema summary JSON | data/songs_schema_summary.json |
| Merged taxonomy CSV | data/songs_with_merged_genres.csv |
| Feature manifest JSON | output/features/feature_extraction_manifest.json |
| Experiment run manifest | output/experiment_runs_taxonomy/run_20260319_234100Z/run_manifest.json |
| Benchmark directory | output/metrics/thesis_benchmark_full_rerun_20260319_235915 |
| Benchmark report | output/metrics/thesis_benchmark_full_rerun_20260319_235915/benchmark_report.md |

## Residual Notes

- The resilient downloader still reports `8076` remaining catalog rows after its stop condition. That means the library is a fresh rerun, but not a 100% successful audio acquisition across the full catalog.
- Metadata/audio mismatch after the rerun: songs.csv marks `1946` rows as audio-backed, while only `1734` audio files are currently on disk (gap `212`).
- `songs_schema_summary.json` still reports zero current audio rows, so its audio counters appear stale relative to the rebuilt `songs.csv`.
