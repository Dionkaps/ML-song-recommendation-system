# P1 Metadata And Schema Cleanup Report

Date: 2026-03-15

## Scope

This report covers the full `P1 - Metadata and schema cleanup` section from `docs/reports/implementation_todo_20260314.md`.

The targeted outcomes were:

- resolve the missing `data/songs.csv` migration
- reconcile schema expectations between `extract_millionsong_dataset.py`, the current `data/millionsong_dataset.csv`, and the unified CSV tooling
- audit genre distribution and imbalance for recommendation-proxy interpretation
- make an explicit decision on MSD numeric metadata for the supported baseline
- make artist metadata available in clustering and evaluation outputs without leaking it into clustering features

## Executive Summary

This section is complete.

The workspace now has a real, canonical unified metadata file at `data/songs.csv`, backed by shared schema utilities instead of ad hoc per-script assumptions. The migration path no longer assumes a three-column `millionsong_dataset.csv`, the extraction and update utilities now write or normalize the same schema, genre loading now uses live audio-backed unified metadata, clustering outputs now include explicit artist/title/filename/track-id columns, retrieval artifacts now carry the same metadata arrays, and the evaluation outputs now include explicit query filename and MSD track-id fields.

The most important data decision from this work is now evidence-based:

- MSD numeric metadata remains disabled for the supported baseline
- restoring it later is now clearly an experiment-only path, not a baseline assumption
- the reason is concrete coverage, not guesswork: only `4783 / 5535` live audio rows currently have numeric MSD metadata, leaving `752` audio-backed tracks without that metadata

The genre audit also now quantifies the proxy-metric limitations:

- the audio-backed subset has `419` unique primary genres
- the top 10 primary genres account for only `19.73%` of audio-backed rows
- `44.87%` of primary genres have `5` tracks or fewer
- `17.66%` of primary genres are singletons
- the primary-genre Gini coefficient is `0.5991`

That means the genre proxies remain useful, but they must be interpreted as long-tail and imbalanced.

## What Changed

### 1. Canonical unified metadata utilities

New file:

- `src/utils/song_metadata.py`

This file is now the shared metadata/schema contract for the workspace.

It provides:

- canonical unified song columns
- schema normalization for unified CSV data
- robust text normalization for metadata matching
- legacy loader helpers for:
  - `millionsong_dataset.csv`
  - `songs_data_with_genre.csv`
  - `msd_features.csv`
  - `msd_matches.csv`
  - `audio_msd_mapping.csv`
- current audio-library enumeration
- cached genre-map loading
- unified metadata rebuilding logic
- aligned per-audio metadata lookup for clustering outputs

This removed the earlier problem where multiple scripts each guessed their own interpretation of `track_id`, `artist`, `title`, `filename`, and `genre`.

### 2. Unified migration is now real and reproducible

Updated file:

- `scripts/utilities/migrate_to_unified_csv.py`

The old migration utility assumed a stale schema and built `songs.csv` with incorrect column expectations for the current `millionsong_dataset.csv`.

The new utility now:

- rebuilds `data/songs.csv` from the actual local metadata sources
- uses the shared schema utility
- writes `data/songs_schema_summary.json`
- preserves the supported unified-column names used by downstream code

This is no longer a speculative migration path. It is the actual current workspace metadata path.

### 3. `extract_millionsong_dataset.py` now matches the live schema

Updated file:

- `src/data_collection/extract_millionsong_dataset.py`

The extractor previously wrote:

- `title`
- `artist`
- `genre`

But the current workspace `data/millionsong_dataset.csv` already has:

- `track_id`
- `title`
- `artist`
- `genre`

That mismatch was one of the root causes of the broken migration logic.

The extractor now writes `track_id` explicitly, so future regenerations of `millionsong_dataset.csv` match the schema that the migration path expects.

### 4. Unified CSV creation/update from MSD features is schema-aware

Updated file:

- `src/features/extract_msd_features.py`

Changes:

- existing unified CSVs are normalized before MSD numeric metadata is updated
- initial unified CSV creation now uses the canonical schema
- loading MSD features from unified CSV normalizes the schema before reuse

This keeps the MSD extraction path consistent with the new unified metadata contract.

### 5. Deezer metadata updates now respect the canonical schema

Updated file:

- `src/data_collection/deezer-song.py`

Changes:

- bootstrap creation of `songs.csv` from downloaded songs now normalizes to the canonical schema
- updates to existing unified CSVs now normalize first and save back in canonical form

This closes another schema drift path that previously could have created a structurally different `songs.csv`.

### 6. Genre loading now uses live unified audio rows

Updated file:

- `src/utils/genre_mapper.py`

Changes:

- unified genre loading now reads `data/songs.csv` via the canonical loader
- when using the unified CSV, only live `has_audio == True` rows are used for the audio-backed genre map

This fixed the earlier issue where unified metadata included legacy/non-live rows and the clustering layer had to clean them back out later.

### 7. Clustering outputs now carry explicit metadata

Updated files:

- `src/clustering/kmeans.py`
- `src/clustering/gmm.py`
- `src/clustering/hdbscan.py`
- `src/clustering/vade.py`

Changes:

- the clustering dataset bundle now carries a metadata frame aligned to the loaded audio rows
- clustering result CSVs now include:
  - `Artist`
  - `Title`
  - `Filename`
  - `MSDTrackID`
  - `GenreList`
  - `Genre`
- retrieval artifacts now store metadata arrays:
  - `artists`
  - `titles`
  - `filenames`
  - `msd_track_ids`

Critically, this metadata stays output-only. It is not appended to the clustering feature matrix, so it does not leak into clustering behavior.

### 8. Evaluation now consumes explicit metadata fields

Updated file:

- `scripts/analysis/evaluate_clustering.py`

Changes:

- method loading now prefers explicit `Artist`, `Title`, `Filename`, and `MSDTrackID` metadata when present
- string-splitting fallback remains only for backward compatibility
- per-query evaluation outputs now include:
  - `QueryFilename`
  - `QueryMSDTrackID`

This removes the previous dependency on parsing `"Artist - Title"` strings when explicit metadata is available.

### 9. Metadata audit/reporting was added

New file:

- `scripts/analysis/audit_metadata_schema.py`

Outputs written by this script:

- `output/metrics/metadata_schema_audit_20260315_summary.json`
- `output/metrics/metadata_primary_genre_distribution_20260315.csv`
- `output/metrics/metadata_label_genre_distribution_20260315.csv`
- `output/metrics/metadata_schema_audit_20260315.md`

This is the evidence base for the genre-imbalance and MSD-policy conclusions.

### 10. Baseline MSD policy is now explicit

Updated files:

- `config/feature_vars.py`
- `docs/SUPPORTED_BASELINE.md`

Added explicit config policy fields:

- `msd_metadata_policy = "disabled_in_supported_baseline"`
- `msd_metadata_restore_policy = "restore_only_as_an_explicit_experiment_after_unified_metadata_audit_passes"`

This converts an implied status into an explicit workspace decision.

## Unified Metadata Rebuild Results

### Generated files

Created or refreshed:

- `data/songs.csv`
- `data/songs_schema_summary.json`

### Rebuild summary

Source: `data/songs_schema_summary.json`

Measured results:

- original MSD catalog rows: `10000`
- MSD rows after exact dedup: `9943`
- exact duplicate rows removed: `57`
- duplicate-track-id rows removed during canonical collapse: `4`
- conflicting duplicate track-id groups: `4`
- MSD rows missing track id: `1`
- final canonical MSD catalog rows: `9939`
- final unified rows: `10691`
- final audio-backed rows: `5535`
- final unified rows with MSD track id: `9938`
- audio-backed rows with MSD track id: `4783`
- audio-backed rows without MSD track id: `752`
- audio-backed rows with numeric MSD features: `4783`
- audio-backed rows without numeric MSD features: `752`

Interpretation:

- the canonical unified table is larger than the raw MSD catalog because it intentionally includes `752` audio-only rows that exist in the current workspace but cannot be linked back to a numeric-MSD row
- this is by design and prevents the workspace from silently dropping live audio files from unified metadata

### How current audio rows were resolved

Source: `data/songs_schema_summary.json`

Audio assignment counts:

- via legacy downloaded-song metadata: `4306`
- via legacy filename-to-track-id mapping only: `2`
- via normalized audio-to-MSD name matching: `496`
- audio-only cache fallback rows: `752`

Interpretation:

- most current audio rows were recovered from the old unified/download path
- a meaningful extra block could be recovered by conservative name matching
- `752` rows still have no safe MSD linkage, which is the main reason MSD numeric metadata should not be restored into the baseline

## Genre Imbalance Audit

### Generated files

Created:

- `output/metrics/metadata_schema_audit_20260315_summary.json`
- `output/metrics/metadata_primary_genre_distribution_20260315.csv`
- `output/metrics/metadata_label_genre_distribution_20260315.csv`
- `output/metrics/metadata_schema_audit_20260315.md`

### Audio-backed genre summary

Source: `output/metrics/metadata_schema_audit_20260315_summary.json`

Measured results:

- audio-backed rows: `5535`
- current audio files: `5535`
- audio coverage gap: `0`
- unique primary genres: `419`
- unique exploded genre labels: `814`
- rows with unknown primary genre: `0`
- rows with multi-label genres: `4723`
- mean genre count per audio row: `4.3799`

### Imbalance metrics

Measured results:

- primary-genre Gini: `0.599116`
- primary-genre HHI: `0.007794`
- effective primary-genre count: `128.30`
- top-10 primary genre share: `0.197290`
- singleton primary genre fraction: `0.176611`
- primary genres with `<= 5` tracks: `0.448687`

Interpretation:

- the audio-backed genre distribution is broad, but strongly long-tail
- genre proxy metrics are valid as rough metadata consistency checks, not as balanced class-performance measures
- comparisons between clustering methods should continue to treat genre precision/hit-rate as proxy evidence, not a clean class-balanced accuracy signal

### Top primary genres

Source: `output/metrics/metadata_primary_genre_distribution_20260315.csv`

Top primary genres in the live audio set:

- `blues-rock`: `185`
- `hip hop`: `159`
- `chanson`: `132`
- `ccm`: `126`
- `country rock`: `88`
- `roots reggae`: `87`
- `latin jazz`: `86`
- `dance pop`: `78`
- `pop rock`: `77`
- `post-grunge`: `74`

These counts reinforce the “broad but thin” distribution shape seen in the long-tail metrics.

## MSD Policy Decision

### Decision

The supported baseline keeps MSD numeric metadata disabled.

### Why

The decision is now grounded in measured coverage:

- live audio-backed rows: `5535`
- live audio-backed rows with numeric MSD metadata: `4783`
- live audio-backed rows without numeric MSD metadata: `752`

Coverage fraction:

- `4783 / 5535 = 86.41%`

Meaning:

- enabling MSD numeric metadata today would either:
  - drop `752` live audio rows from the baseline, or
  - require imputation or fallback behavior that would blur the experiment

That is exactly the kind of hidden condition the earlier todo wanted to avoid. So the baseline stays audio-only, and MSD restoration is explicitly moved into a future experiment path.

## Artist Metadata In Outputs

### Clustering result CSVs

Regenerated files:

- `output/clustering_results/audio_clustering_results_kmeans.csv`
- `output/clustering_results/audio_clustering_results_gmm.csv`
- `output/clustering_results/audio_clustering_results_hdbscan.csv`

Verified output columns now include:

- `Song`
- `Artist`
- `Title`
- `Filename`
- `MSDTrackID`
- `GenreList`
- `Genre`

### Retrieval artifacts

Regenerated files:

- `output/clustering_results/audio_clustering_artifact_kmeans.npz`
- `output/clustering_results/audio_clustering_artifact_gmm.npz`
- `output/clustering_results/audio_clustering_artifact_hdbscan.npz`

Verified artifact metadata keys now include:

- `artists`
- `titles`
- `filenames`
- `msd_track_ids`

### Evaluation outputs

Regenerated files:

- `output/metrics/evaluation_upgrades_per_query_kmeans.csv`
- `output/metrics/evaluation_upgrades_per_query_gmm.csv`
- `output/metrics/evaluation_upgrades_per_query_hdbscan.csv`

Verified new explicit per-query fields:

- `QueryFilename`
- `QueryMSDTrackID`

This closes the last todo item for this section: artist-based evaluation now has explicit metadata support, while clustering itself remains feature-only.

## Commands Run

### Syntax verification

```powershell
.\.venv\Scripts\python.exe -m py_compile src/utils/song_metadata.py scripts/utilities/migrate_to_unified_csv.py src/data_collection/extract_millionsong_dataset.py src/features/extract_msd_features.py src/data_collection/deezer-song.py src/utils/genre_mapper.py src/clustering/kmeans.py src/clustering/gmm.py src/clustering/hdbscan.py src/clustering/vade.py scripts/analysis/evaluate_clustering.py scripts/analysis/audit_metadata_schema.py
```

Result:

- passed

### Unified metadata rebuild

```powershell
.\.venv\Scripts\python.exe scripts/utilities/migrate_to_unified_csv.py
```

Result:

- `data/songs.csv` rebuilt successfully
- `data/songs_schema_summary.json` written
- runtime about `7.0` seconds

### Metadata audit

```powershell
.\.venv\Scripts\python.exe scripts/analysis/audit_metadata_schema.py
```

Result:

- summary, CSV distributions, and Markdown audit written
- runtime about `7.1` seconds

### Clustering regeneration

```powershell
.\.venv\Scripts\python.exe scripts/run_all_clustering.py
```

Result:

- K-Means, GMM, and HDBSCAN outputs regenerated with explicit metadata fields
- retrieval artifacts regenerated with metadata arrays
- final successful rerun runtime about `378.4` seconds

### Evaluation regeneration

```powershell
.\.venv\Scripts\python.exe scripts/analysis/evaluate_clustering.py
```

Result:

- evaluation outputs regenerated successfully
- runtime about `151.5` seconds

## Important Implementation Notes

### Why `songs.csv` has more than `10000` rows

The unified CSV now contains both:

- the canonical MSD catalog rows
- supplemental audio-only rows for live audio files that do not have a safe MSD track-id match

That is why `data/songs.csv` has `10691` rows instead of `10000`.

This is intentional. It is better than silently pretending the audio library is fully covered by MSD metadata when it is not.

### Why the baseline still excludes MSD metadata

The metadata cleanup solved the missing-file and schema problems, but it did not magically create missing MSD matches for every current audio row.

The remaining `752` unmatched audio-backed rows are a real data boundary. Keeping MSD disabled in the supported baseline is the correct technical choice until a future explicit experiment decides how to handle that gap.

### Why artist metadata is safe now

Artist/title/track-id fields are now:

- loaded from unified metadata
- written to result CSVs and artifacts
- consumed by evaluation outputs

But they are still not part of the clustering feature matrix. So they improve observability and evaluation without contaminating the unsupervised representation.

## Files Touched

Code:

- `src/utils/song_metadata.py`
- `scripts/utilities/migrate_to_unified_csv.py`
- `src/data_collection/extract_millionsong_dataset.py`
- `src/features/extract_msd_features.py`
- `src/data_collection/deezer-song.py`
- `src/utils/genre_mapper.py`
- `src/clustering/kmeans.py`
- `src/clustering/gmm.py`
- `src/clustering/hdbscan.py`
- `src/clustering/vade.py`
- `scripts/analysis/evaluate_clustering.py`
- `scripts/analysis/audit_metadata_schema.py`
- `config/feature_vars.py`

Docs:

- `docs/SUPPORTED_BASELINE.md`
- `docs/reports/implementation_todo_20260314.md`

Generated outputs:

- `data/songs.csv`
- `data/songs_schema_summary.json`
- metadata audit outputs under `output/metrics/`
- regenerated clustering results/artifacts under `output/clustering_results/`
- regenerated evaluation outputs under `output/metrics/`

## Final Outcome

The metadata layer is no longer half-migrated.

The workspace now has:

- a real unified `songs.csv`
- a shared metadata/schema contract
- a corrected MSD catalog extraction schema
- a measurable genre-imbalance audit
- an explicit MSD baseline policy
- explicit artist/title/track-id metadata in clustering and evaluation outputs

This fully closes the `P1 - Metadata and schema cleanup` section with implementation, regenerated outputs, and measured evidence for the remaining baseline decisions.
