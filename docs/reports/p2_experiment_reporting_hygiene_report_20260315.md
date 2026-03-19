# P2 Implementation Report - Experiment and Reporting Hygiene

Date: 2026-03-15

## Scope

This report covers the implementation of `## P2 - Experiment and reporting hygiene`
from `docs/reports/implementation_todo_20260314.md`.

The goal of this slice was to stop the experiment layer from drifting away from the
actual supported baseline. In practical terms, that meant:

- one canonical machine-readable place for the supported profile and comparison profiles
- run manifests that capture exactly what was executed
- method summaries and retrieval artifacts that carry representation metadata
- evaluation outputs that state both the representation contract and the current proxy-metric limitation
- short docs that make the recommended production baseline easy to find
- explicit comparison baselines so `all_audio`, raw-zscore, `HDBSCAN`, and `VaDE` do not silently become defaults again

## Final Status

This P2 slice is complete.

The final system now has:

- a canonical experiment-profile registry in `config/experiment_profiles.py`
- a timestamped experiment-suite runner in `scripts/run_all_clustering.py`
- run-specific clustering summaries for `K-Means`, `GMM`, `HDBSCAN`, and `VaDE`
- retrieval artifacts that record representation metadata and profile identity
- evaluation summaries that include representation contract, proxy-evaluation limitation, and future validation backlog
- a short durable production-baseline doc in `docs/RECOMMENDED_PRODUCTION_BASELINE.md`
- refreshed default promoted outputs in `output/clustering_results/` and `output/metrics/`

## Files Added Or Updated

### New files

- `config/experiment_profiles.py`
- `docs/RECOMMENDED_PRODUCTION_BASELINE.md`
- `docs/reports/p2_experiment_reporting_hygiene_report_20260315.md`

### Updated files

- `scripts/run_all_clustering.py`
- `scripts/analysis/evaluate_clustering.py`
- `src/clustering/kmeans.py`
- `src/clustering/gmm.py`
- `src/clustering/hdbscan.py`
- `src/clustering/vade.py`
- `docs/SUPPORTED_BASELINE.md`
- `docs/README.md`
- `README.md`
- `docs/reports/implementation_todo_20260314.md`

## What Was Implemented

### 1. Canonical experiment-profile registry

`config/experiment_profiles.py` is now the machine-readable source of truth for the
baseline and comparison profiles.

It defines:

- the canonical contract version
- the recommended production profile id: `recommended_production`
- explicit comparison profile ids:
  - `all_audio_pca_comparison`
  - `all_audio_zscore_comparison`
- the default comparison methods:
  - `kmeans`
  - `gmm`
  - `hdbscan`
- the optional comparison method:
  - `vade`
- the metadata-proxy evaluation limitation text
- the future validation backlog

The recommended production profile encodes:

- feature subset name: `spectral_plus_beat`
- selected audio groups:
  - `spectral_centroid`
  - `spectral_rolloff`
  - `spectral_flux`
  - `spectral_flatness`
  - `zero_crossing_rate`
  - `beat_strength`
- equalization method: `pca_per_group`
- PCA components per group: `5`
- expected raw dimension: `14`
- expected prepared dimension: `30`
- genre disabled in clustering
- MSD numeric metadata disabled in the supported baseline

The comparison profiles are now explicit by name instead of being implicit side
effects of older defaults.

### 2. Timestamped experiment-suite runner

`scripts/run_all_clustering.py` was upgraded from a simple batch runner into a
profile-aware experiment runner.

It now:

- accepts `--profiles`
- accepts optional `--methods`
- supports `--include-vade`
- supports `--skip-evaluation`
- writes all results under timestamped run roots:
  - `output/experiment_runs/run_<UTCSTAMP>/...`
- writes a top-level canonical baseline contract:
  - `canonical_baseline_contract.json`
- writes one `profile_manifest.json` per executed profile
- writes one top-level `run_manifest.json`
- promotes the recommended production profile outputs back into:
  - `output/clustering_results`
  - `output/metrics`

This means the workspace now has both:

- immutable experiment-run folders for reproducibility
- refreshed default output folders for the rest of the application

### 3. Per-method run summaries now record representation contract

The clustering runners now attach the active representation contract directly to
their method summaries and retrieval artifacts.

The affected methods are:

- `K-Means`
- `GMM`
- `HDBSCAN`
- `VaDE`

For each run, the method summary now records:

- `method_id`
- `profile_id`
- `feature_subset_name`
- `selected_audio_feature_keys`
- `equalization_method`
- `pca_components_per_group`
- `raw_feature_dimension`
- `prepared_feature_dimension`
- clustering hyperparameters
- selection-diagnostics output paths
- retrieval-artifact path
- dataset-QC paths

This closed one of the main reproducibility gaps from the earlier state of the
project, where downstream reports could infer the representation but did not store
it as a first-class run contract.

### 4. Retrieval artifacts now carry representation metadata

The saved `.npz` retrieval artifacts now include:

- `artifact_version = 2`
- `profile_id`
- `feature_subset_name`
- `selected_audio_feature_keys`
- `feature_equalization_method`
- `pca_components_per_group`
- `raw_feature_dimension`
- `prepared_feature_dimension`

That matters because the UI and evaluation code no longer need to assume how the
prepared space was built. They can load the artifact and know exactly which
representation it came from.

### 5. Evaluation output now states its contract and limitations

`scripts/analysis/evaluate_clustering.py` was extended so that evaluation is no
longer just a metric dump.

It now records:

- the shared representation contract across loaded artifacts
- any representation mismatches between methods
- the current evaluation limitation:
  - recommendation quality is still being estimated with metadata proxies
- the future validation backlog

It also writes cleaner output names when called from the experiment suite. The
production run now writes:

- `recommended_production_comparison.csv`
- `recommended_production_comparison_report.md`
- `recommended_production_summary.json`

instead of the earlier duplicated `*_comparison_comparison.*` naming.

### 6. Comparison baselines are explicit, not silent defaults

The runner and docs now treat alternative pipelines as named experiments rather than
ambient defaults.

The comparison-only paths are:

- `all_audio_pca_comparison`
- `all_audio_zscore_comparison`

The comparison methods are explicit too:

- `K-Means`
- `GMM`
- `HDBSCAN`
- optional `VaDE`

This is important because older project drift made it too easy for the full
116-dimensional `all_audio` representation to reappear as if it were the supported
baseline.

### 7. Short production-baseline documentation

`docs/RECOMMENDED_PRODUCTION_BASELINE.md` was added as the short durable reference.

Its purpose is deliberately narrow:

- make the supported production profile obvious
- prevent accidental drift back to the full `all_audio` baseline
- point engineers to the machine-readable contract in `config/experiment_profiles.py`
- document the proxy-evaluation limitation and the future listening-test backlog

`docs/SUPPORTED_BASELINE.md`, `docs/README.md`, and `README.md` were updated to
point to that short baseline doc.

## Late Verification Findings And Final Hardening

Two hygiene issues surfaced during verification and were fixed before finalizing
this slice.

### Finding 1. Evaluation filenames were awkward

Initial experiment-suite runs used the prefix `recommended_production_comparison`,
and `evaluate_clustering.py` also appends `_comparison.*`.

That produced filenames like:

- `recommended_production_comparison_comparison.csv`
- `recommended_production_comparison_comparison_report.md`

This was not functionally broken, but it was bad reporting hygiene. The runner now
passes the profile id itself as the prefix, which produces:

- `recommended_production_comparison.csv`
- `recommended_production_comparison_report.md`
- `recommended_production_summary.json`

### Finding 2. Method summaries pointed at mutable global QC files

The shared clustering loader writes convenience files:

- `output/metrics/clustering_dataset_qc_latest.csv`
- `output/metrics/clustering_dataset_qc_summary_latest.json`

During the first implementation pass, method summaries pointed directly at those
global `latest` files. That meant a later run could overwrite the QC evidence for an
earlier run.

This was fixed by snapshotting the QC files into each run-specific metrics folder.

Method summaries now point to stable run-local paths such as:

- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/clustering_dataset_qc.csv`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/clustering_dataset_qc_summary.json`

This makes the run manifest self-contained enough for later audit.

## Final Verification

## Static verification

The following command succeeded:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  src/clustering/kmeans.py `
  src/clustering/gmm.py `
  src/clustering/hdbscan.py `
  src/clustering/vade.py `
  scripts/run_all_clustering.py
```

## Runtime verification

### A. Full recommended production suite

Command:

```powershell
.\.venv\Scripts\python.exe scripts/run_all_clustering.py --profiles recommended_production --stability-jobs 1
```

Final verified run id:

- `run_20260315_175616Z`

Elapsed wall-clock time:

- about `623.7` seconds

Outputs written under:

- `output/experiment_runs/run_20260315_175616Z/recommended_production/clustering_results/`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/`

Promoted default outputs refreshed under:

- `output/clustering_results/`
- `output/metrics/`

### B. Comparison-profile smoke run

Command:

```powershell
.\.venv\Scripts\python.exe scripts/run_all_clustering.py --profiles all_audio_zscore_comparison --methods kmeans --skip-evaluation
```

Final verified run id:

- `run_20260315_180651Z`

Elapsed wall-clock time:

- about `311.8` seconds

This verified that the explicit comparison-profile path is alive after the final
hygiene fixes, not just the recommended production path.

## Final Verified Outputs

### Production profile manifests

- `output/experiment_runs/run_20260315_175616Z/canonical_baseline_contract.json`
- `output/experiment_runs/run_20260315_175616Z/run_manifest.json`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/profile_manifest.json`

### Production profile metrics

- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/kmeans_run_summary.json`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/gmm_selection_summary.json`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/hdbscan_run_summary.json`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/clustering_dataset_qc.csv`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/clustering_dataset_qc_summary.json`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/recommended_production_comparison.csv`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/recommended_production_comparison_report.md`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/metrics/recommended_production_summary.json`

### Production profile artifacts

- `output/experiment_runs/run_20260315_175616Z/recommended_production/clustering_results/audio_clustering_artifact_kmeans.npz`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/clustering_results/audio_clustering_artifact_gmm.npz`
- `output/experiment_runs/run_20260315_175616Z/recommended_production/clustering_results/audio_clustering_artifact_hdbscan.npz`

### Comparison-profile smoke outputs

- `output/experiment_runs/run_20260315_180651Z/all_audio_zscore_comparison/profile_manifest.json`
- `output/experiment_runs/run_20260315_180651Z/all_audio_zscore_comparison/metrics/kmeans_run_summary.json`
- `output/experiment_runs/run_20260315_180651Z/all_audio_zscore_comparison/metrics/clustering_dataset_qc.csv`
- `output/experiment_runs/run_20260315_180651Z/all_audio_zscore_comparison/metrics/clustering_dataset_qc_summary.json`
- `output/experiment_runs/run_20260315_180651Z/all_audio_zscore_comparison/clustering_results/audio_clustering_artifact_kmeans.npz`

## Verified Production Results Snapshot

The final verified production run used the expected contract:

- profile id: `recommended_production`
- feature subset name: `spectral_plus_beat`
- equalization method: `pca_per_group`
- PCA components per group: `5`
- raw dimension: `14`
- prepared dimension: `30`
- loaded tracks: `5535`
- mismatch methods in evaluation contract: none

The final verified production comparison ranked methods as:

1. `K-Means`
2. `GMM`
3. `HDBSCAN`

The stored GMM selection summary for the production profile reports:

- selected components: `4`
- covariance type: `full`
- `reg_covar = 1e-6`
- `n_init = 10`
- effective `max_iter = 300`
- average log-likelihood: `105.4716796875`
- stability ARI: `0.9777796145564507`

The production evaluation summary reports:

- evaluation still uses metadata proxies
- the representation contract is shared across all compared methods
- per-method recommendation, coverage, exposure, and stability outputs were written with profile-specific names

## Impact On The Workspace

After this work, the workspace is in a much healthier state operationally:

- the supported baseline is now encoded in code, not just prose
- experiment outputs can be traced back to a concrete profile and representation
- future reruns can compare named baselines instead of relying on memory or old defaults
- reports now tell the truth about the limitation of proxy-based recommendation evaluation
- future human validation is now documented as backlog, not forgotten context

## Remaining Caveats

These are not blockers for this P2 slice, but they are worth stating explicitly.

- Recommendation quality is still being evaluated with metadata proxies rather than human judgments.
- The current canonical production baseline remains `GMM` by policy/documentation, even though the latest proxy-based comparison run ranked `K-Means` first. Changing the product default would be a separate explicit decision.
- `VaDE` was wired into the same experiment/reporting framework and passed static verification, but it was not executed in this specific P2 verification pass.

## Conclusion

`P2 - Experiment and reporting hygiene` is implemented and verified.

The baseline now has a canonical profile registry, immutable experiment manifests,
run-local QC evidence, explicit comparison profiles, clearer docs, and evaluation
reports that record both the representation contract and the fact that offline
quality judgment is still proxy-based.
