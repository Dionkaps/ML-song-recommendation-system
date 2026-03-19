# P1 Clustering Model Improvements Report

Generated: 2026-03-14

## Request handled

This report documents the implementation of:

- `## P1 - Clustering model improvements`

from:

- `docs/reports/implementation_todo_20260314.md`

## Executive summary

The GMM baseline is no longer a simple one-axis BIC sweep over component count.

It is now a staged model-selection procedure that:

- starts from the recommended baseline defaults
- searches the recommended `4..40` component range
- compares `diag` and `full` covariance structures
- sweeps `reg_covar` over `1e-6..1e-3`
- extends iteration budget to `500` when convergence needs it
- filters obviously bad solutions
- records richer diagnostics
- uses repeated-seed stability on near-best BIC candidates

This phase also regenerated fresh baseline outputs for:

- K-Means
- GMM
- HDBSCAN

and confirmed that the new retrieval artifact files exist for all three methods.

## Scope completed

All items under `## P1 - Clustering model improvements` in `docs/reports/implementation_todo_20260314.md` were completed and marked done:

- recommended GMM defaults made explicit
- GMM component search expanded to `4..40`
- `diag` vs `full` covariance compared explicitly
- `reg_covar` sweep added across `1e-6..1e-3`
- `n_init` raised to the recommended baseline level during selection
- `max_iter` retry path added up to `500`
- degeneracy guards added
- richer diagnostics saved
- staged selection procedure made explicit
- fresh K-Means / GMM / HDBSCAN outputs regenerated

## Main code changes

### 1. `src/clustering/gmm.py`

This file was substantially rewritten.

The old behavior was:

- select `n_components` only
- use one fixed covariance type per run
- use one fixed `reg_covar`
- pick best BIC with silhouette tie-break
- save only minimal diagnostics

The new behavior is:

- use the recommended default fixed config when not doing dynamic selection:
  - `covariance_type='diag'`
  - `reg_covar=1e-5`
  - `n_init=10`
  - `max_iter=300`
  - `tol=1e-3`
- when dynamic selection is enabled:
  - stage 1: search components `4..40`
  - compare covariance types `diag` and `full`
  - use baseline `reg_covar=1e-5`
  - shortlist the strongest component counts per covariance type
  - stage 2: sweep `reg_covar` across `[1e-6, 1e-5, 1e-4, 1e-3]`
  - keep near-best BIC candidates
  - estimate repeated-seed stability
  - choose final model using explicit ranking rules

### New data structure

Added:

- `GMMCandidateResult`

Purpose:

- hold one fitted candidate and its diagnostics in a structured way
- preserve the fitted model object for the final selected candidate
- serialize diagnostics to CSV cleanly

Important fields tracked per candidate:

- stage
- components
- covariance type
- `reg_covar`
- `n_init`
- requested vs effective max iterations
- whether extended-iteration retry was used
- convergence status
- warning summary
- BIC
- AIC
- average log-likelihood
- silhouette
- average confidence
- occupied component count
- min / median / max cluster size
- tiny cluster count
- dominant cluster fraction
- cluster-size distribution
- covariance condition information
- whether the covariance looked unstable
- whether the candidate was marked degenerate
- whether it passed the validity filters
- stability ARI
- whether it was near-best by BIC
- whether it was selected

### New helper functions

Added:

- `_resolve_component_search_range()`
- `_fit_model_with_retry()`
- `_cluster_stats()`
- `_covariance_health_stats()`
- `_fit_candidate()`
- `_run_candidate_grid()`
- `_shortlist_component_pairs()`
- `_estimate_stability_ari()`
- `_selection_key()`
- `_select_model()`
- `_selection_diagnostics_frame()`

### Convergence handling

The new code explicitly supports an iteration-budget retry path.

Behavior:

- fit with `max_iter=300`
- if convergence warnings appear, or `model.converged_` is false:
  - retry with `max_iter=500`
- log whether extended iteration was used

This satisfies the todo item about allowing the model to extend into the `300..500` range when convergence needs it.

### Degeneracy guards

The new code checks candidate quality beyond raw BIC.

Cluster-shape guards:

- require at least 2 occupied components
- compute tiny-cluster threshold as `max(5, ceil(0.002 * n_samples))`
- track tiny cluster count
- track dominant cluster fraction
- mark a candidate degenerate if:
  - too many tiny occupied clusters exist, or
  - one cluster dominates too much

Covariance-health guards:

- compute minimum eigenvalue / variance
- compute maximum eigenvalue / variance
- compute condition number
- track how much covariance mass is pushed close to the regularization floor
- mark covariance unstable if:
  - eigenvalues are invalid / non-positive
  - condition number is extreme
  - floor-saturation and conditioning together indicate a likely numerical issue

Important implementation note:

- during development, the original covariance-floor rule was too strict and caused all full-covariance candidates in the real dataset to be flagged invalid
- that rule was relaxed so it now flags genuinely unstable fits instead of simply regularized ones
- this changed the final run from a fallback selection to a valid selection

### New selection procedure

The selection procedure is now explicit and persisted to disk.

Stage 1:

- search components from `4` to `40`
- compare `diag` and `full`
- hold `reg_covar` at the baseline `1e-5`

Stage 2:

- take the top component counts per covariance type from stage 1
- sweep `reg_covar` across:
  - `1e-6`
  - `1e-5`
  - `1e-4`
  - `1e-3`

First-pass filter:

- keep candidates within `10.0` BIC points of the best valid BIC

Second-pass selection:

- among those near-best BIC candidates, rank by:
  1. higher average confidence
  2. higher silhouette
  3. higher repeated-seed stability ARI
  4. lower BIC
  5. fewer components
  6. prefer `diag` on exact ties

Important limitation:

- this still uses proxy quality terms (`avg_confidence`, `silhouette`) rather than true recommendation metrics
- that is intentional for this phase
- the later `P1 - Evaluation upgrades` section is where recommendation-centered metrics will become the primary judge

### Diagnostics outputs added / expanded

Updated / added outputs:

- `output/metrics/gmm_selection_criteria.csv`
- `output/metrics/gmm_selection_summary.json`

`gmm_selection_criteria.csv` now stores:

- stage
- hyperparameters
- convergence metadata
- BIC / AIC
- average log-likelihood
- silhouette
- average confidence
- cluster-size distribution
- covariance diagnostics
- validity flags
- stability ARI
- selection flags

`gmm_selection_summary.json` now stores:

- the human-readable selection procedure
- selected candidate config
- selected candidate metrics
- candidate-count summary
- whether a validity fallback was needed

### Final GMM result for the fresh run

After the final rerun, the selected GMM configuration was:

- components: `4`
- covariance type: `full`
- `reg_covar = 1e-6`
- `n_init = 10`
- effective `max_iter = 300`

Selected metrics:

- `BIC = -1,150,480.3208467069`
- `AIC = -1,163,605.494140625`
- average log-likelihood: `105.4716796875`
- silhouette: `0.15275557339191437`
- average confidence: `0.9376877546310425`
- repeated-seed stability ARI: `0.9777796145564507`

Selected cluster sizes:

- cluster 0: `1346`
- cluster 1: `576`
- cluster 2: `2976`
- cluster 3: `637`

Validity outcome:

- `passed_validity_filters = true`
- `used_validity_fallback = false`

That is important.

The final chosen model is not a fallback.

It is now a candidate that genuinely passed the selection guards.

### 2. `scripts/run_all_clustering.py`

This script was updated to match the baseline comparison workflow more closely.

Changes made:

- added CLI parsing
- baseline default now runs:
  - K-Means
  - GMM
  - HDBSCAN
- VaDE is now optional via:
  - `--include-vade`
- added summary generation via:
  - `output/metrics/clustering_run_summary.json`

The summary now records:

- row count
- result CSV path
- retrieval artifact path
- whether the retrieval artifact exists
- cluster count
- HDBSCAN noise count where applicable

This makes the “fresh baseline rerun” auditable rather than just implied by console output.

### 3. `scripts/analysis/evaluate_clustering.py`

This file was lightly aligned to the new GMM baseline defaults.

Changes made:

- `stability_ari_gmm()` now defaults to:
  - `covariance_type='diag'`
  - `max_iter=300`
  - `reg_covar=1e-5`
  - `n_init=10`
- the GMM stability call in `main()` now uses those same values

Why this matters:

- evaluation no longer assumes the old `full / 200 / default n_init` configuration
- the evaluation helper is now better aligned with the new intended baseline

### 4. `docs/reports/implementation_todo_20260314.md`

All items under `## P1 - Clustering model improvements` were marked complete.

## Fresh baseline rerun results

### K-Means rerun

Fresh result:

- rows: `5535`
- selected clusters: `2`
- retrieval artifact exists: `true`

Best saved K-Means selection metric:

- `K = 2`
- silhouette: `0.26433`
- inertia: `23996.691406`

Outputs regenerated:

- `output/clustering_results/audio_clustering_results_kmeans.csv`
- `output/clustering_results/audio_clustering_artifact_kmeans.npz`

### GMM rerun

Fresh result:

- rows: `5535`
- selected clusters: `4`
- retrieval artifact exists: `true`

Key stage-1 best rows by covariance family:

- best `diag` stage-1 row:
  - components: `35`
  - `reg_covar = 1e-5`
  - BIC: `-799605.484012397`
  - silhouette: `0.0674530938267707`
  - average confidence: `0.9234671592712402`
- best `full` stage-1 row:
  - components: `4`
  - `reg_covar = 1e-5`
  - BIC: `-920848.8495179716`
  - silhouette: `0.152951031923294`
  - average confidence: `0.9379485249519348`

Stage-2 shortlist in the final successful rerun:

- `diag`: `35`, `36`, `40`
- `full`: `4`, `5`, `6`

Near-best BIC count in final run:

- `1`

Interpretation:

- in this actual rerun, BIC strongly preferred one candidate rather than leaving a crowded near-best set
- stability therefore acted more as a confirmation metric than as a tie-break

Outputs regenerated:

- `output/clustering_results/audio_clustering_results_gmm.csv`
- `output/clustering_results/audio_clustering_artifact_gmm.npz`
- `output/metrics/gmm_selection_criteria.csv`
- `output/metrics/gmm_selection_summary.json`

### HDBSCAN rerun

Fresh result:

- rows: `5535`
- selected clusters: `2`
- noise points: `258`
- noise percentage: about `4.66%`
- retrieval artifact exists: `true`

Best saved HDBSCAN selection row:

- `min_cluster_size = 5`
- `min_samples = 5`
- clusters: `2`
- `noise_fraction = 0.046612`
- `dbcv = 0.199366`
- silhouette: `0.199434`
- score: `0.200016`

Outputs regenerated:

- `output/clustering_results/audio_clustering_results_hdbscan.csv`
- `output/clustering_results/audio_clustering_artifact_hdbscan.npz`

### Comparison summary artifact

The rerun also produced:

- `output/metrics/clustering_run_summary.json`

Summary contents:

- K-Means: `5535` rows, `2` clusters, artifact present
- GMM: `5535` rows, `4` clusters, artifact present
- HDBSCAN: `5535` rows, `2` clusters, `258` noise points, artifact present

## Verification performed

### 1. Static compilation

The following files were compiled successfully with `py_compile`:

- `src/clustering/gmm.py`
- `scripts/run_all_clustering.py`
- `scripts/analysis/evaluate_clustering.py`

### 2. GMM selector smoke test

Before the full rerun, the new selector was exercised on a restricted small sample.

Why this was done:

- validate the new staged search logic cheaply
- catch selection-logic failures before spending a full rerun

What it validated:

- stage-1 candidate generation
- stage-2 `reg_covar` sweep
- shortlist generation
- summary-object construction
- near-best candidate handling

Important outcome:

- the first smoke test exposed that the covariance-floor guard was too strict
- this was corrected before the final full rerun

### 3. Full baseline rerun

The full baseline comparison run was executed via:

- `scripts/run_all_clustering.py`

Runtime observed for the fresh baseline trio rerun:

- about `1089.9` seconds total
- approximately `18.2` minutes

This produced fresh:

- K-Means results
- GMM results
- HDBSCAN results
- retrieval artifacts
- metrics summaries

### 4. Final targeted GMM rerun after guard adjustment

After relaxing the covariance guard, GMM was rerun again directly to refresh:

- `output/clustering_results/audio_clustering_results_gmm.csv`
- `output/clustering_results/audio_clustering_artifact_gmm.npz`
- `output/metrics/gmm_selection_criteria.csv`
- `output/metrics/gmm_selection_summary.json`

Observed runtime:

- about `367.6` seconds
- about `6.1` minutes

This final rerun confirmed:

- selected candidate passes validity filters
- stability ARI is finite
- no fallback selection was needed

## Important implementation details and decisions

### Default config vs selected config

There is an important distinction now:

- default fixed GMM baseline config:
  - starts from `diag`, `reg_covar=1e-5`, `n_init=10`, `max_iter=300`
- dynamic selected config in the actual rerun:
  - ended up choosing `full`, `4 components`, `reg_covar=1e-6`

This is correct and intentional.

Reason:

- the TODO asked for `diag` to be the recommended starting/default config
- it also asked for evidence-based comparison against `full`
- the code now does both

So:

- `diag` is the starting/default operating assumption
- `full` is allowed to win if the evidence supports it

### Why the shortlist contained large `diag` component counts

Stage 1 found the strongest `diag` BIC rows at high component counts such as `35`, `36`, and `40`.

This means:

- under diagonal covariance, the model needed many more components to compete on raw fit
- under full covariance, strong candidates appeared with far fewer components

That pattern is useful information in itself because it shows:

- `full` is currently modeling this prepared-space geometry much more efficiently than `diag`

### Why stability was not the decisive factor in the final rerun

The final rerun produced:

- only one near-best BIC candidate under the `10.0` BIC tolerance

Therefore:

- stability was computed and recorded
- but it did not need to break a tie between multiple near-best BIC candidates in this run

That still satisfies the requirement to make stability part of the explicit selection procedure.

It simply means:

- in this particular dataset/run, BIC already separated the winner clearly

## Caveats

### 1. Recommendation-quality selection still uses proxies

The final selection ranking still uses:

- average confidence
- silhouette
- stability ARI

These are not yet the final product-facing recommendation metrics.

That is expected.

The later evaluation section is where:

- `Precision@K`
- hit-rate style metrics
- genre/artist retrieval proxies
- coverage/diversity metrics

should become the real decision layer.

### 2. The fresh output files are runtime artifacts, not necessarily git-tracked assets

The reruns did regenerate fresh output files under `output/`, but those runtime files may be ignored by git depending on repo settings.

That does not change the implementation status.

It only means the workspace state should be understood through the actual files on disk, not just `git status`.

### 3. K-Means and HDBSCAN still produce only 2 clusters in the fresh rerun

That is a result, not an implementation bug.

It is also a useful signal:

- the current baseline geometry still tends toward coarse partitions for those methods

This is one reason the GMM improvements and later evaluation upgrades matter.

## Outcome against the original P1 goals

### Achieved

- GMM defaults now match the recommended baseline starting config
- component search now spans the intended `4..40` range
- covariance type is evidence-driven rather than fixed
- `reg_covar` sweep is implemented
- `n_init=10` is used during selection
- iteration extension is implemented and logged
- degeneracy/covariance guards are in place
- richer diagnostics are saved
- selection procedure is explicit and persisted
- fresh K-Means / GMM / HDBSCAN artifacts were regenerated

### Not claimed

- no claim is made that the final recommendation metric stack is finished
- no claim is made that metadata-based evaluation limitations are solved
- no claim is made that VaDE is part of the default comparison trio
- no claim is made that the final product cluster granularity decision has been made

## Recommended next steps

The most logical next implementation target is:

- `## P1 - Evaluation upgrades`

Reason:

- the clustering model-selection machinery is now much more mature
- the next bottleneck is that final model choice still relies on proxy quality terms rather than explicit recommendation-centered evaluation

## Final status

`P1 - Clustering model improvements` has been implemented.

The code now performs a materially stronger, evidence-based GMM search and the workspace has fresh baseline comparison outputs and retrieval artifacts for:

- K-Means
- GMM
- HDBSCAN
