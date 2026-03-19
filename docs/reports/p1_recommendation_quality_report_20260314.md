# P1 Recommendation Quality Fixes Report

Generated: 2026-03-14

## Request handled

This report documents the implementation of:

- `## P1 - Recommendation quality fixes`

from:

- `docs/reports/implementation_todo_20260314.md`

## Executive summary

The recommendation path has been moved off the PCA-2 visualization space and onto the real prepared clustering space.

That is the core product-level fix in this phase.

Before this implementation:

- the UI ranked nearest neighbors by Euclidean distance in 2D PCA coordinates
- those 2D coordinates were only a visualization projection
- recommendation quality could therefore be distorted by projection artifacts

After this implementation:

- recommendations are ranked in the full prepared feature space
- PCA-2 is now used only for plotting and zoomed visualization
- the UI can load a saved retrieval artifact instead of reconstructing the wrong space
- GMM/VaDE-style probabilistic results can optionally filter and rank recommendations using confidence/posterior information
- recommendation entries now show distance and confidence-style evidence so the UI is more explainable

## Scope completed

All items under `## P1 - Recommendation quality fixes` in `docs/reports/implementation_todo_20260314.md` were completed and marked done:

- stop using PCA-2 coordinates for nearest-neighbor recommendations in the UI
- use full prepared feature-space distances for ranking within the selected cluster
- preserve PCA-2 only for visualization
- expose the prepared feature matrix through a saved retrieval artifact
- add optional confidence/posterior filtering for GMM-style probabilistic outputs
- add optional posterior-weighted ranking as a follow-up mode to hard cluster filtering
- show recommendation distance/confidence information in the UI

## Design goals for this phase

The implementation was designed around five product-quality goals:

1. Recommendation ranking must operate in the same feature space the clustering methods actually use.
2. The UI must not silently reconstruct or approximate that space incorrectly.
3. Visualization and retrieval must be separated conceptually and technically.
4. Probabilistic clustering methods must be able to expose uncertainty to the user.
5. The retrieval path must be reusable by future offline evaluation code.

## Files changed

### 1. `src/clustering/kmeans.py`

This file now contains the reusable retrieval-artifact API used by the UI.

New functions added:

- `get_retrieval_artifact_path()`
- `save_retrieval_artifact()`
- `load_retrieval_artifact()`

Purpose of the artifact:

- persist the exact prepared feature matrix used by clustering
- persist row alignment via song names
- persist cluster labels and visualization coordinates
- optionally persist method-specific recommendation metadata such as confidence, posterior probabilities, distance-to-cluster, and log-likelihood

Artifact filename format:

- `output/clustering_results/audio_clustering_artifact_<method>.npz`

Artifact payload fields:

- required:
  - `artifact_version`
  - `method_id`
  - `songs`
  - `prepared_features`
  - `labels`
  - `coords`
  - `feature_subset_name`
  - `feature_equalization_method`
  - `pca_components_per_group`
- optional:
  - `assignment_confidence`
  - `posterior_probabilities`
  - `distance_to_cluster`
  - `log_likelihood`

Validation added when saving:

- `prepared_features` must be 2D
- sample count must match `file_names`
- `coords` must be exactly `(n_samples, 2)`
- optional vectors must align to sample count
- optional posterior matrix must be 2D with aligned row count

Behavior change in K-Means runner:

- K-Means now saves a prepared-space retrieval artifact after writing the CSV
- the saved artifact includes:
  - `prepared_features`
  - `labels`
  - `coords`
  - `distance_to_cluster`

UI launch behavior change:

- K-Means now launches the UI with `retrieval_method_id="kmeans"`
- this forces the UI to load the saved prepared-space artifact instead of using PCA-2 as a fallback retrieval space

### 2. `src/clustering/gmm.py`

This file now exports the extra data the UI needs for confidence-aware retrieval.

Changes made:

- imported `save_retrieval_artifact`
- computed `posterior_probabilities = model.predict_proba(X_prepared)`
- used those posteriors directly to derive:
  - `labels`
  - `probabilities` (max posterior / assignment confidence)
- saved a retrieval artifact after CSV export

GMM artifact fields now stored:

- `prepared_features`
- `labels`
- `coords`
- `assignment_confidence`
- `posterior_probabilities`
- `log_likelihood`

Why this matters:

- the UI can now use the real prepared-space embedding for distance
- the UI can also expose probabilistic controls without recomputing GMM outputs

UI launch behavior change:

- GMM now launches the UI with `retrieval_method_id="gmm"`

### 3. `src/clustering/hdbscan.py`

Although the TODO explicitly called out GMM, the retrieval artifact format was extended to HDBSCAN too so the UI behavior stays consistent across clustering methods.

Changes made:

- imported `save_retrieval_artifact`
- saved a retrieval artifact after CSV export

HDBSCAN artifact fields now stored:

- `prepared_features`
- `labels`
- `coords`
- `assignment_confidence` (HDBSCAN membership probability)
- `distance_to_cluster`

Behavior refinement in the UI enabled by this:

- when the selected song is labeled as HDBSCAN noise (`-1`), the UI now deliberately avoids generating within-cluster recommendations
- this is preferable to treating the noise set as a meaningful semantic cluster

### 4. `src/clustering/vade.py`

VaDE was updated to save the same retrieval artifact style as GMM.

Changes made:

- imported `save_retrieval_artifact`
- saved a retrieval artifact after CSV export

VaDE artifact fields now stored:

- `prepared_features`
- `labels`
- `coords`
- `assignment_confidence`
- `posterior_probabilities` (`GAMMA`)

Why this matters:

- VaDE can now use the same explainable UI controls as GMM
- the retrieval path is standardized across both hard and probabilistic methods

### 5. `src/ui/modern_ui.py`

This is the main implementation file for the P1 recommendation fixes.

The file was substantially reworked.

Core architectural change:

- the UI now resolves a retrieval payload first
- that retrieval payload must come from either:
  - `retrieval_features` passed directly, or
  - a saved artifact loaded via `retrieval_method_id`

If neither is provided:

- the UI raises an explicit error
- it does not silently fall back to PCA-2 for retrieval

New internal helpers:

- `_parse_bounded_int()`
- `_parse_bounded_float()`
- `_align_artifact_rows()`
- `_resolve_retrieval_payload()`

What `_align_artifact_rows()` does:

- aligns artifact arrays to the current DataFrame row order using song name
- validates that the artifact labels match the currently loaded clustering results
- fails fast if the artifact is stale or mismatched

What `_resolve_retrieval_payload()` does:

- loads the saved retrieval artifact when `retrieval_method_id` is supplied
- extracts:
  - `prepared_features`
  - optional `assignment_confidence`
  - optional `posterior_probabilities`
- validates all row counts

### UI behavior changes in detail

#### A. Recommendation ranking space

Old behavior:

- precomputed `distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)`
- recommendations were ranked in 2D PCA space

New behavior:

- recommendations are computed on demand in the full prepared feature space
- for a selected song `s`, candidate distance is:
  - `||prepared_features[candidate] - prepared_features[s]||_2`

This is the correct retrieval space for the supported baseline.

#### B. Hard cluster filtering remains in place

The UI still restricts recommendations to the selected song's assigned cluster first.

This preserves the product behavior described in the TODO:

- rank neighbors inside the selected cluster

So the retrieval pipeline is now:

1. choose selected song
2. find same-cluster candidates
3. optionally filter those candidates by confidence/posterior thresholds
4. rank the remaining candidates in prepared space

#### C. PCA-2 is now visualization-only

The PCA coordinates are still used for:

- full scatter plot
- zoomed plot after selecting a song
- drawing connection lines between the selected song and recommendations

The PCA coordinates are no longer used for:

- nearest-neighbor retrieval
- ranking decisions
- recommendation filtering

This cleanly separates:

- visualization geometry
- recommendation geometry

#### D. Explainability controls were added

The UI now exposes:

- `Top N`
- `Ranking`
- `Min confidence`
- `Min selected-cluster posterior`

Control behavior:

- `Top N`: limits the number of recommendations shown
- `Ranking`: chooses between pure distance ranking and posterior-weighted ranking
- `Min confidence`: filters low-confidence assignments when confidence values are available
- `Min selected-cluster posterior`: filters candidates by their probability mass on the selected song's cluster when posterior data is available

Controls are automatically disabled when a method does not provide the required data.

Examples:

- K-Means:
  - no confidence control
  - no posterior control
  - only pure distance ranking is available
- GMM:
  - confidence filter enabled
  - posterior filter enabled
  - posterior-weighted ranking enabled
- VaDE:
  - same style of probabilistic controls as GMM
- HDBSCAN:
  - confidence filter enabled
  - posterior-specific controls disabled

#### E. Posterior-weighted ranking mode

This was implemented as an optional follow-up mode, not as the default.

Default mode:

- `Distance`

Optional mode when posterior data exists:

- `Posterior-weighted`

Formula used:

- after hard cluster filtering, for each candidate:
  - `distance = ||x_candidate - x_selected||_2`
  - `p_cluster = posterior_probabilities[candidate, selected_cluster]`
  - `weighted_score = distance / max(p_cluster, 1e-3)`

Ranking then sorts by:

- `weighted_score` first
- raw `distance` second as a tie-breaker

Interpretation:

- candidates that are close in prepared space and strongly associated with the selected cluster rank highest
- candidates that are close but weakly attached to the selected cluster are pushed down

Why this was kept optional:

- the TODO phrased this as a follow-up consideration rather than the universal default
- pure distance remains the safest baseline

#### F. Explainable recommendation rows

Recommendation entries now display a compact evidence summary.

Possible fields shown:

- song name
- prepared-space distance
- assignment confidence
- selected-cluster posterior
- posterior-weighted score when that ranking mode is active

This means the UI now tells the user something about why a song appeared.

#### G. HDBSCAN noise handling

Special case added:

- if the selected song has label `-1`, the UI does not produce within-cluster recommendations

Reason:

- the HDBSCAN noise set is not a real cluster
- showing “similar songs” from the noise bucket would likely be misleading

#### H. Fail-fast behavior for stale artifacts

The UI now validates that:

- the artifact song set covers the current result rows
- the artifact labels match the loaded clustering result labels

If those validations fail:

- the UI raises an error rather than silently mixing old clustering outputs with new prepared features

That is important because this repo currently contains older clustering CSVs generated before the latest baseline alignment.

### 6. `docs/reports/implementation_todo_20260314.md`

The `P1 - Recommendation quality fixes` checklist items were marked complete.

## Technical behavior after the implementation

### Recommendation flow

The effective recommendation flow is now:

1. A clustering method runs and saves:
   - result CSV
   - prepared-space retrieval artifact
2. The UI opens with `retrieval_method_id=<method>`
3. The UI loads the saved artifact
4. The UI aligns artifact rows by song name
5. The user selects a song
6. Candidates are restricted to the assigned cluster
7. Optional confidence/posterior thresholds are applied
8. Candidates are ranked in the full prepared space
9. PCA-2 is used only to visualize the selected result set

### What is now persisted for future evaluation work

The saved artifact format now gives future scripts a stable input for:

- offline recommendation metrics
- full-space distance reuse
- confidence-aware retrieval experiments
- posterior-aware ranking experiments

This is an important enabling step for the later `P1 - Evaluation upgrades` section.

## Verification performed

### 1. Static compilation

The following files compiled successfully with `py_compile`:

- `src/ui/modern_ui.py`
- `src/clustering/kmeans.py`
- `src/clustering/gmm.py`
- `src/clustering/hdbscan.py`
- `src/clustering/vade.py`

Result:

- all compiled without syntax errors

### 2. Prepared-space artifact smoke test

A targeted runtime smoke test was run using the project virtual environment with:

- `SDL_AUDIODRIVER=dummy`
- `MPLBACKEND=Agg`

This avoided needing a real interactive audio/GUI session while still exercising the real code paths.

Smoke-test method:

- create a temporary 12-track restricted sample by exposing only 12 audio basenames
- build the real prepared feature matrix through `load_clustering_dataset()`
- compute PCA coordinates
- save a synthetic retrieval artifact with:
  - labels
  - confidence
  - posterior probabilities
- load that artifact back
- resolve it through the same UI retrieval-loader path

Observed results:

- used tracks: `12`
- prepared shape: `(12, 30)`
- artifact prepared shape: `(12, 30)`
- resolved prepared shape through UI loader: `(12, 30)`
- posterior shape: `(12, 3)`
- confidence shape: `(12,)`

Interpretation:

- the artifact save path works
- the artifact load path works
- the UI alignment logic works
- the prepared-space retrieval payload reaches the UI in the expected dimensions

### 3. What was not verified interactively

The following were not claimed as verified:

- no full manual GUI click-through session was run in a desktop environment
- no full end-to-end clustering rerun was performed yet for all methods
- no fresh production artifact files were generated for the existing historical output CSVs

Those omissions are intentional and important.

## Important caveats

### 1. Existing saved clustering CSVs are older than the new recommendation artifact format

Current workspace state:

- old CSV result files already exist in `output/clustering_results/`
- those were generated before this P1 implementation
- they do not yet have matching saved retrieval artifacts produced by the new code

Why this matters:

- the new UI path expects a matching artifact to be saved by a fresh run of the clustering method

Practical consequence:

- to use the new retrieval path with fresh saved outputs, the clustering methods should be re-run under the current baseline

This is not a bug in the P1 implementation.

It is just the natural consequence of changing the output format and retrieval contract.

### 2. No attempt was made to backfill artifacts for stale historical results

This was intentional.

Reason:

- the older output CSVs may reflect older clustering runs and potentially older feature/preprocessing assumptions
- generating new prepared-space artifacts and pairing them with stale labels would risk mixing incompatible states

The safer path is:

- run clustering again under the current supported baseline
- let each method save its own artifact at run time

### 3. `src/clustering/kmeans.py` still contains earlier legacy helper definitions

This caveat from P0 still applies.

Current state:

- later canonical helper definitions are still the active runtime versions
- the file remains functionally correct
- the file is still more cluttered than ideal

This did not block the P1 implementation, but it remains a cleanup target for a later technical-debt pass.

## Outcome against the original P1 goals

### Achieved

- UI recommendation ranking no longer uses PCA-2
- full prepared-space distance is now the ranking baseline
- PCA-2 has been reduced to visualization only
- saved retrieval artifacts now exist in the code path
- GMM/VaDE uncertainty can now affect filtering/ranking in the UI
- recommendation entries are now explainable via distance/confidence text

### Not claimed

- no claim is made that recommendation metrics have been upgraded yet
- no claim is made that the final GMM model-selection logic is finished yet
- no claim is made that fresh clustering artifacts have been regenerated for all methods yet
- no claim is made that human-evaluated recommendation quality has been validated yet

## Recommended next steps

The most sensible next implementation targets are:

- `## P1 - Clustering model improvements`
- `## P1 - Evaluation upgrades`

In practical order, the next workflow should be:

1. re-run clustering methods under the current supported baseline so fresh artifacts exist
2. upgrade evaluation to consume the same prepared-space retrieval logic
3. use those metrics to tune GMM selection and recommendation behavior

## Final status

`P1 - Recommendation quality fixes` has been implemented in code.

The main product-level issue identified in the research report has been corrected:

- recommendation ranking is now based on the real prepared feature space rather than the PCA-2 visualization projection
