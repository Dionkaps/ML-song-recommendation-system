# Command Cheat Sheet

All commands below assume you are running from the project root:

```powershell
cd C:\Users\vpddk\Desktop\Me\Github\ML-song-recommendation-system
```

These examples are written for PowerShell on Windows.

## Reset And Cleanup

Preview what would be deleted without deleting anything:

```powershell
python scripts/utilities/reset_full_pipeline_state.py
```

Fully reset generated state, including `audio_files/`, cached JSONs, outputs, and derived metadata:

```powershell
python scripts/utilities/reset_full_pipeline_state.py --apply
```

Reset only downstream pipeline outputs, while keeping downloaded audio and download caches:

```powershell
python scripts/utilities/reset_full_pipeline_state.py --apply --keep-download-state
```

Fully reset but keep run logs:

```powershell
python scripts/utilities/reset_full_pipeline_state.py --apply --keep-run-logs
```

Print the reset summary as JSON:

```powershell
python scripts/utilities/reset_full_pipeline_state.py --json
```

Write the reset summary to a file:

```powershell
python scripts/utilities/reset_full_pipeline_state.py --output-json output\reset_summary.json
```

Run the reset against another copy of the repo:

```powershell
python scripts/utilities/reset_full_pipeline_state.py --project-root C:\path\to\repo-copy --apply
```

Clean known junk CSV files only:

```powershell
python scripts/utilities/cleanup_project_junk.py --apply
```

## End-To-End Pipeline

Run the full supported pipeline from download through clustering:

```powershell
python run_pipeline.py
```

Run the full pipeline but choose a specific clustering method:

```powershell
python run_pipeline.py --clustering-method gmm
```

Run the full pipeline but skip selected stages:

```powershell
python run_pipeline.py --skip download preprocess
```

Run the full post-download pipeline from preprocessing through thesis benchmark:

```powershell
python run_post_download_pipeline.py
```

Run the post-download pipeline and include the optional plotting stage:

```powershell
python run_post_download_pipeline.py --include-plot
```

Run the post-download pipeline and print the final summary as JSON:

```powershell
python run_post_download_pipeline.py --json
```

## Benchmark Reruns And Sample Audits

Clean generated artifacts, preserve completed downloads, and rerun benchmark-related stages:

```powershell
python scripts/utilities/run_full_benchmark_rerun.py --preserve-download-state
```

Run the benchmark rerun and include the optional plotting stage:

```powershell
python scripts/utilities/run_full_benchmark_rerun.py --preserve-download-state --include-plot
```

Run a careful isolated sample audit on 500 songs:

```powershell
python scripts/utilities/run_sample_pipeline_audit.py --sample-size 500 --passes 2 --keep-workspace
```

Run a smaller one-pass sample audit:

```powershell
python scripts/utilities/run_sample_pipeline_audit.py --sample-size 100 --passes 1
```

Run the thesis clustering benchmark directly:

```powershell
python scripts/analysis/thesis_clustering_benchmark.py
```

Write the thesis benchmark summary as JSON:

```powershell
python scripts/analysis/thesis_clustering_benchmark.py --json
```

## Clustering Runs

Run the recommended production clustering profile:

```powershell
python scripts/run_all_clustering.py
```

Run multiple clustering methods explicitly:

```powershell
python scripts/run_all_clustering.py --methods kmeans gmm hdbscan
```

Run clustering and publish the stable UI snapshot:

```powershell
python scripts/run_all_clustering.py --publish-ui-snapshot
```

Run K-Means only:

```powershell
python src/clustering/kmeans.py
```

Run GMM only:

```powershell
python src/clustering/gmm.py
```

Run HDBSCAN only:

```powershell
python src/clustering/hdbscan.py
```

Launch a clustering script and open the UI afterward:

```powershell
python src/clustering/kmeans.py --ui
python src/clustering/gmm.py --ui
python src/clustering/hdbscan.py --ui
```

## UI

Launch the UI using the latest benchmark-linked snapshot:

```powershell
python scripts/run_ui.py
```

Force the UI to open a specific method:

```powershell
python scripts/run_ui.py --method gmm
python scripts/run_ui.py --method kmeans
python scripts/run_ui.py --method hdbscan
```

Point the UI at a specific benchmark directory:

```powershell
python scripts/run_ui.py --benchmark-dir output\metrics\thesis_benchmark_20260323_120000
```

## Preprocessing And Feature Extraction

Run audio preprocessing on the local audio library:

```powershell
python scripts/run_audio_preprocessing.py
```

Run preprocessing and export detailed stats:

```powershell
python scripts/run_audio_preprocessing.py --details --details-output output\metrics\preprocessing_details.json
```

Run handcrafted feature extraction:

```powershell
python scripts/run_feature_extraction.py
```

Force feature extraction to recompute instead of resuming:

```powershell
python scripts/run_feature_extraction.py --no-resume
```

Run feature QC and repair broken feature bundles:

```powershell
python scripts/analysis/run_feature_qc.py --repair
```

Run feature QC in report-only mode:

```powershell
python scripts/analysis/run_feature_qc.py --no-repair
```

## Metadata And Taxonomy

Rebuild unified `songs.csv` from the preserved source inputs:

```powershell
python scripts/utilities/migrate_to_unified_csv.py
```

Build the merged-genre taxonomy-aware dataset:

```powershell
python scripts/utilities/build_merged_genre_dataset.py
```

Build the merged-genre dataset from explicit paths:

```powershell
python scripts/utilities/build_merged_genre_dataset.py --source-songs-csv data\songs.csv --mapping-csv data\acoustically_coherent_merged_genres_corrected.csv --output-csv data\songs_with_merged_genres.csv
```

Audit metadata schema and coverage:

```powershell
python scripts/analysis/audit_metadata_schema.py
```

## Quick Checks

Check whether deleted items still exist on disk:

```powershell
Test-Path .\audio_files
Test-Path .\deezer_search_cache.json
Test-Path .\download_checkpoint_with_genre.json
```

List the latest benchmark output folders:

```powershell
Get-ChildItem .\output\metrics -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name,LastWriteTime
```

List the latest experiment run folders:

```powershell
Get-ChildItem .\output\experiment_runs_taxonomy -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name,LastWriteTime
```

See all options for a script:

```powershell
python scripts/run_ui.py --help
python scripts/utilities/reset_full_pipeline_state.py --help
python scripts/utilities/run_sample_pipeline_audit.py --help
```

## Suggested Workflows

Start completely fresh:

```powershell
python scripts/utilities/reset_full_pipeline_state.py --apply
python run_pipeline.py
```

Keep downloads but rerun everything after download:

```powershell
python scripts/utilities/reset_full_pipeline_state.py --apply --keep-download-state
python run_post_download_pipeline.py
```

Run clustering again, publish the UI snapshot, and open the UI:

```powershell
python scripts/run_all_clustering.py --publish-ui-snapshot
python scripts/run_ui.py
```

Run a careful validation pass on a 500-song sample:

```powershell
python scripts/utilities/run_sample_pipeline_audit.py --sample-size 500 --passes 2 --keep-workspace
```
