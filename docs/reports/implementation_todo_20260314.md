# Implementation TODO Based on Deep Research Report

Generated: 2026-03-14

This checklist is based primarily on `C:\Users\vpddk\Downloads\deep-research-report.md`, cross-checked against the current workspace report in `docs/reports/workspace_technical_report_20260314.md`.

## Target baseline to implement

- Audio-only clustering for now
- Feature subset: `spectral_plus_beat`
- Preprocessing/equalization: per-group `StandardScaler` + `pca_per_group_5` + equal group contribution normalization
- Main clustering model: GMM
- Recommendation distance: full prepared feature space, not PCA-2
- Evaluation priority: recommendation proxies + stability first, internal clustering metrics second

## P0 - Immediate blockers and baseline alignment

- [x] Keep genre as metadata/evaluation only; do not reintroduce genre one-hot into clustering input.
  Target files: `config/feature_vars.py`, clustering dataset assembly
- [x] Add an explicit clustering feature-subset config for `spectral_plus_beat`.
  Target files: `config/feature_vars.py`, `src/clustering/kmeans.py`
- [x] Update dataset assembly so clustering can exclude `mfcc`, `delta_mfcc`, `delta2_mfcc`, and `chroma` while still keeping them available for other analyses if needed.
  Target files: `src/clustering/kmeans.py`
- [x] Preserve the current time-to-fixed summarization contract:
  mean + std for time-varying features, and raw 4-scalar `beat_strength` block as-is.
  Target files: feature loading and vector assembly code
- [x] Make sure the prepared clustering representation is the expected 30-D output for the recommended subset under `pca_per_group_5`.
  Target files: `src/clustering/kmeans.py`
- [x] Keep the current audio preprocessing invariants from regressing:
  mono, `22050 Hz`, `PCM_16`, `29s` fixed duration.
  Target files: preprocessing scripts, validation checks, reports
- [x] Keep loudness normalization behavior as an explicit invariant of the baseline pipeline.
  Target files: preprocessing scripts, reports, pipeline docs
- [x] Disable MSD metadata by default until `data/songs.csv` is restored and validated.
  Why: current config says MSD is on, but the active workspace does not have the required unified CSV.
  Target files: `config/feature_vars.py`, `scripts/run_all_clustering.py`, `scripts/analysis/evaluate_clustering.py`
- [x] Decide on one supported operating mode and document it clearly:
  either `audio-only baseline now` or `finish metadata migration first`.
  Target files: `README.md`, `docs/`, pipeline wrappers
- [x] Make GMM the default clustering path for the recommended pipeline instead of treating K-Means as the practical default.
  Target files: `run_pipeline.py`, `scripts/run_all_clustering.py`

## P1 - Recommendation quality fixes

- [x] Stop using PCA-2 coordinates for nearest-neighbor recommendations in the UI.
  Target files: `src/ui/modern_ui.py`
- [x] Use full prepared feature-space distances for neighbor ranking inside the selected cluster.
  Target files: `src/ui/modern_ui.py`, clustering result loading path
- [x] Preserve PCA-2 only for visualization.
  Target files: `src/ui/modern_ui.py`, clustering modules
- [x] Expose the prepared feature matrix or a saved embedding artifact so the UI can do retrieval without reconstructing the wrong space.
  Target files: `src/clustering/kmeans.py`, `src/clustering/gmm.py`, output artifact format
- [x] For GMM results, add optional recommendation filtering by assignment confidence or posterior threshold.
  Target files: `src/clustering/gmm.py`, `src/ui/modern_ui.py`
- [x] Consider posterior-weighted neighbor ranking for GMM as a follow-up to hard cluster filtering.
  Target files: `src/clustering/gmm.py`, `src/ui/modern_ui.py`, evaluation scripts
- [x] Show recommendation confidence and/or distance in the UI so results are explainable.
  Target files: `src/ui/modern_ui.py`

## P1 - Clustering model improvements

- [x] Make the recommended GMM starting config the default:
  `covariance_type='diag'`, `reg_covar=1e-5`, `n_init=10`, `max_iter=300`, `tol=1e-3`.
  Target files: `src/clustering/gmm.py`
- [x] Expand GMM component search to the recommended working range of roughly `4..40`.
  Target files: `src/clustering/gmm.py`
- [x] Compare `diag` vs `full` covariance as part of model selection instead of locking into one option without evidence.
  Target files: `src/clustering/gmm.py`, evaluation scripts
- [x] Add an explicit tuning sweep for `reg_covar` across a reasonable range such as `1e-6..1e-3`.
  Target files: `src/clustering/gmm.py`, evaluation scripts
- [x] Raise `n_init` into the recommended tuning range when comparing candidate models, not just the minimum viable default.
  Target files: `src/clustering/gmm.py`
- [x] Allow `max_iter` to extend into the `300..500` range when convergence warnings appear, and log those cases.
  Target files: `src/clustering/gmm.py`
- [x] Add guards against degenerate solutions such as tiny effective clusters or unstable covariance fits.
  Target files: `src/clustering/gmm.py`
- [x] Save richer GMM selection diagnostics for each run: BIC, AIC, average log-likelihood, silhouette, average confidence, and cluster-size distribution.
  Target files: `src/clustering/gmm.py`
- [x] Make the selection procedure explicit:
  BIC as first-pass filter, then recommendation quality and stability to choose the final model from near-best BIC candidates.
  Target files: `src/clustering/gmm.py`, evaluation scripts, experiment docs
- [x] Re-run clustering with the recommended baseline and store fresh artifacts for K-Means, GMM, and HDBSCAN comparison.
  Target files: `scripts/run_all_clustering.py`, `output/`

## P1 - Evaluation upgrades

- [x] Add offline recommendation metrics that reflect the actual product behavior, not just clustering geometry.
  Target files: `scripts/analysis/evaluate_clustering.py`
- [x] Implement per-query and averaged `Precision@K` alongside hit-rate style metrics.
  Target files: `scripts/analysis/evaluate_clustering.py`
- [x] Implement genre hit rate@K using within-cluster recommendations ranked in full prepared space.
  Target files: `scripts/analysis/evaluate_clustering.py`
- [x] Implement artist hit rate@K once artist metadata is surfaced cleanly through the evaluation pipeline.
  Target files: evaluation scripts, metadata loading path
- [x] Add catalog coverage or recommendation diversity metrics to detect collapse into a few popular clusters.
  Target files: `scripts/analysis/evaluate_clustering.py`
- [x] Keep silhouette, Calinski-Harabasz, and Davies-Bouldin as diagnostics only, not the main selection objective.
  Target files: `scripts/analysis/evaluate_clustering.py`, experiment docs
- [x] Increase stability testing rigor:
  more bootstrap/subsample runs for HDBSCAN and repeated-seed checks for GMM under the final baseline.
  Target files: `scripts/analysis/evaluate_clustering.py`
- [x] Make the recommended stability protocol explicit for the final GMM baseline:
  start with about `50` resampling runs at `80%` subsample, then report mean and median ARI.
  Target files: `scripts/analysis/evaluate_clustering.py`, experiment docs
- [x] Add per-cluster stability reporting, not just one global stability number.
  Target files: `scripts/analysis/evaluate_clustering.py`
- [x] Re-check internal metrics together with cluster-size distribution as a final sanity check after model selection.
  Target files: `scripts/analysis/evaluate_clustering.py`, experiment reports
- [x] Produce one final comparison report that ranks candidate models by recommendation quality first, stability second.
  Target files: `scripts/analysis/evaluate_clustering.py`, `docs/reports/`

## P1 - Data quality and feature QC

- [x] Enforce deterministic feature-extraction settings and document them so regenerated features are reproducible.
  Target files: feature extraction scripts, config, docs
- [x] Add strict validation for missing feature arrays, wrong shapes, NaN values, and Inf values before clustering starts.
  Target files: `src/clustering/kmeans.py`, feature-loading helpers
- [x] Do not silently impute broken tracks in the clustering set; re-extract or drop them and log what happened.
  Target files: clustering data loader, feature extraction scripts
- [x] Write a QC report listing dropped, stale, or re-extracted tracks.
  Target files: `scripts/analysis/`, `docs/reports/`
- [x] Clean stale entries from cached genre metadata.
  Why: current cache contains entries without matching audio files.
  Target files: `output/features/genre_map.npy`, cache-generation path, cleanup utilities
- [x] Verify that the selected `spectral_plus_beat` groups retain sane variance after scaling and equalization.
  Target files: metrics scripts, experiment reports

## P1 - Metadata and schema cleanup

- [x] Resolve the `songs.csv` migration instead of leaving the project in a half-migrated state.
  Target files: `scripts/utilities/migrate_to_unified_csv.py`, `data/`, clustering wrappers
- [x] Reconcile schema expectations between `extract_millionsong_dataset.py`, the current `millionsong_dataset.csv`, and unified CSV tooling.
  Target files: `src/data_collection/extract_millionsong_dataset.py`, migration utilities
- [x] Audit current genre distribution and imbalance so recommendation-proxy metrics can be interpreted correctly.
  Target files: analysis scripts, reports
- [x] Decide whether MSD numeric metadata will remain disabled for the baseline or be restored later as an explicit experiment.
  Target files: `config/feature_vars.py`, experiment docs
- [x] If artist-based evaluation is required, make artist metadata available in clustering/evaluation outputs without leaking it into clustering features.
  Target files: metadata loading path, result CSV generation, evaluation scripts

## P2 - Pipeline and technical debt cleanup

- [x] Fix `run_pipeline.py` so it no longer imports the missing `src.audio_preprocessing.loudness_scanner`.
  Choose one path: implement the missing module or simplify the pipeline to the existing `AudioPreprocessor`.
  Target files: `run_pipeline.py`, `src/audio_preprocessing/`
- [x] Either implement true peak measurement/limiting properly or update docs/comments so they do not overclaim true-peak support.
  Target files: preprocessing code and preprocessing docs
- [x] Remove or mark stale references that no longer match the active workspace:
  WKBSC mentions, mel-spectrogram assumptions, old `.mp3`-only cleanup assumptions, legacy normalizer paths.
  Target files: `README.md`, `scripts/visualization/ploting.py`, `scripts/utilities/`, `src/utils/audio_normalizer.py`
- [x] Make helper and maintenance scripts consistent with the current all-`.wav` processed library.
  Target files: `scripts/utilities/`, checks and cleanup scripts

## P2 - Experiment and reporting hygiene

- [x] Save the final chosen configuration in one canonical place so defaults, reports, and scripts stay aligned.
  Target files: config, runner scripts, report docs
- [x] Record the exact feature subset, equalization mode, dimensions, model hyperparameters, and evaluation outputs for every comparison run.
  Target files: experiment output format, metrics scripts
- [x] Keep alternative pipelines such as `all_audio`, raw-zscore, HDBSCAN, and VaDE as explicit comparison baselines rather than silent defaults.
  Target files: runner scripts, experiment docs, reports
- [x] Add one short "recommended production baseline" doc so future work does not drift back to the full 116-D all-audio baseline by accident.
  Target files: `docs/`, `README.md`
- [x] Document the limitation that recommendation quality is still being judged with metadata proxies rather than human similarity judgments.
  Target files: evaluation docs, final reports
- [x] Add a future-validation backlog item for small-scale listening tests or human judgment checks once the offline baseline is stable.
  Target files: `docs/`, roadmap/report docs

## Decisions To Make Explicitly

- [x] Decide target cluster granularity for the product:
  broad macro-style clusters, with the supported GMM baseline targeting `4..8` occupied clusters and `4` as the current reference target.
- [x] Define the minimum acceptable stability threshold for selecting the final GMM model.
  Decision: require median subsample ARI `>= 0.90`, mean subsample ARI `>= 0.75`, reference median ARI `>= 0.90`, and per-cluster median Jaccard `>= 0.90`.
- [x] Decide whether uncertain GMM assignments should be filtered out, down-weighted, or still shown normally in recommendations.
  Decision: show them normally by default under full prepared-space distance; keep posterior-weighted ranking and hard thresholds as optional controls.
- [x] Decide when MSD numeric metadata should come back as a controlled experiment rather than silently returning to the default path.
  Decision: only after `>= 98%` live audio-backed numeric coverage, `<= 100` missing audio-backed rows, a clean schema audit, and a fresh explicit comparison rerun.

## Suggested execution order

- [x] 1. Turn off MSD by default and lock in `spectral_plus_beat`
- [x] 2. Fix feature loading so the recommended 30-D representation is the real clustering input
- [x] 3. Switch UI recommendation distance from PCA-2 to full prepared space
- [x] 4. Upgrade GMM defaults and component-selection search
- [x] 5. Add recommendation-centered evaluation metrics
- [x] 6. Re-run experiments and save fresh artifacts
- [x] 7. Clean metadata migration and remaining technical debt

## Definition of done

- [x] Clustering runs on the recommended `spectral_plus_beat` subset
- [x] Default prepared representation is 30-D with `pca_per_group_5`
- [x] GMM is the main baseline and is selected with an evidence-based search
- [x] UI recommendations use full prepared-space distance, not PCA-2 distance
- [x] Evaluation includes top-K recommendation proxies and stability
- [x] Metadata/CSV expectations are no longer contradictory
- [x] Pipeline docs and defaults match the actual implemented system
