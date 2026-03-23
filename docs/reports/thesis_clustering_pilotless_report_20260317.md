# Pilot-Free Thesis Clustering Report (Taxonomy-Aware Update)

## 1. Purpose of This Report

This report replaces the earlier pilot-shaped thesis narrative with a new full-space benchmark that does
not use any shortlist, subset-stage promotion logic, or pilot-driven representation selection.

The work in this report was done in direct response to the user requirement that the thesis experiment be
rebuilt without being biased by the earlier pilot. The benchmark described here therefore treats the
thesis experiment itself as the main study.

Everything in this report is based on:

- a prespecified representation space derived from standard Music Information Retrieval feature families
- full downloaded-audio evaluation after taxonomy-aware genre reassignment
- direct comparison of KMeans, Gaussian Mixture Models, and HDBSCAN
- multiple evaluation views rather than a single headline score

This report is intentionally detailed. It explains:

- why the pilot was removed from the thesis flow
- why the new benchmark was designed the way it was
- what was implemented
- what was run
- what the results were
- how those results should be interpreted scientifically
- what limitations still remain
- which sources support the design choices

## 2. Executive Summary

### 2.1 One-paragraph summary

This updated report reran the pilot-free thesis benchmark after replacing the raw genre labels with the
merged taxonomy in `data/acoustically_coherent_merged_genres_corrected.csv`. The new preprocessing step
creates `data/songs_with_merged_genres.csv`, promotes only non-`non_genre_*` mapped tags to primary genre
status, keeps `non_genre_*` tags as secondary metadata, and excludes songs that have no mapped primary
genre at all. That left `5,456` audio-backed songs for benchmarking out of `5,535` downloaded tracks,
with `187` taxonomy-backed primary genres and `3,038` artists. The pilot-free benchmark still evaluated
`14` prespecified audio representations across `3` preprocessing modes and `3` clustering algorithms,
yielding `42` representation/preprocessing scenarios and `3,738` fitted models. The strongest practical
native result remains `HDBSCAN + rhythm_only + pca_per_group_2`, now with `563` clusters, `8.5%` noise,
`silhouette=0.569`, `NMI=0.453`, and `stability_ari=0.887`. The highest raw NMI overall is now
`HDBSCAN + pitch_rhythm + pca_per_group_2` with `NMI=0.483`, but that solution discards `46.0%` of the
collection as noise, so it is not the best thesis headline. Under matched-granularity comparison, KMeans
remains the strongest balanced method overall, GMM remains highly competitive and sometimes slightly
stronger on raw NMI at higher target counts, and HDBSCAN remains the most interesting native discovery
method but not the strongest full-coverage matched method. In the downstream taxonomy-aware recommendation
rerun with multivector primary/secondary tags, KMeans ranked first, GMM second by a very small margin,
and HDBSCAN third.

### 2.2 High-level findings

1. The merged-genre taxonomy changed the evaluation regime materially by compressing the benchmark label
   space from noisy raw genres to `187` curated primary taxonomy labels.
2. `79` downloaded songs were excluded from the MRS benchmark because every mapped tag fell under a
   `non_genre_*` bucket and no real primary genre remained.
3. HDBSCAN is still the strongest method for native fine-grained unsupervised discovery, and
   `rhythm_only + pca_per_group_2` remains its most defensible native thesis result.
4. The old `delta2_mfcc` headline no longer survives the taxonomy-aware rerun: the top raw-NMI row is now
   `pitch_rhythm + pca_per_group_2`, and the best practical native leader is still rhythmic rather than
   delta-delta timbral.
5. KMeans remains the strongest balanced method when cluster granularity is matched and full coverage
   matters.
6. GMM remains a legitimate soft-clustering alternative and is nearly tied with KMeans in the production
   recommendation rerun, but it still fails the explicit stability gate.
7. Taxonomy-aware multivector evaluation is more informative than exact single-label genre agreement for
   downstream recommendation quality because it rewards shared primary tags and shared secondary context
   rather than only exact one-label equality.

## 3. Why the Pilot Was Removed

The user explicitly requested that the thesis process be rebuilt so that it was not based on the earlier
pilot. That change is scientifically reasonable.

The main reasons are:

1. A pilot shortlist can bias the main study toward the structures that happened to look strong in a
   smaller exploratory run.
2. If the main benchmark only evaluates pilot-promoted candidates, then the thesis is no longer
   answering "which method and representation perform best in the defined search space?" It is answering
   "which method performs best among the candidates that survived an earlier screen?"
3. In thesis work, exploratory and confirmatory stages should be separated clearly.
4. The earlier pilot used an exploratory scenario score to promote representations. That was acceptable
   for exploration, but weaker as a final comparative methodology.

From a research-design perspective, the pilot was therefore removed because:

- it introduced selection bias risk
- it made the final benchmark conditional on exploratory promotion
- it weakened the defensibility of the final thesis claims

This new study addresses that by evaluating the full prespecified representation space directly.

## 4. Scientific Basis for the New Design

The redesign was not arbitrary. It was anchored in four methodological ideas from the literature.

### 4.1 Audio representations should reflect established MIR feature families

The feature space in this repository is based on classical MIR descriptors: MFCC-derived timbre features,
spectral descriptors, chroma, and beat-strength information. That is a defensible foundation for a thesis
about clustering music tracks from audio features because these families are well established in MIR
research and genre-related audio analysis (Tzanetakis and Cook, 2002; Aucouturier and Pachet, 2003).

### 4.2 Clustering should not be judged by one metric

Clustering-validation literature is clear that no single metric is sufficient. Internal metrics can reward
trivial partitions, external metrics inherit label imperfections, and stability can favor repeatable but
uninformative solutions (Halkidi et al., 2001; Rousseeuw, 1987; von Luxburg, 2010; von Luxburg et al.,
2012). That is why this benchmark keeps:

- internal structure metrics
- external semantic alignment metrics
- stability metrics
- runtime
- coverage/noise diagnostics for HDBSCAN

### 4.3 MIR label alignment must be treated carefully

Music labels are useful but imperfect. MIR evaluation literature warns repeatedly about annotation noise,
collection effects, and confounds such as artist leakage or dataset-specific regularities (Urbano et al.,
2014; Flexer, 2007; Bogdanov et al., 2019; Sturm, 2021). That is why genre alignment is used here as an
evaluation output rather than the direct optimization target.

### 4.4 Native and matched comparisons answer different scientific questions

The literature on clustering comparison and model-based clustering supports the idea that method behavior
depends on operating regime and cluster-count assumptions (Fraley and Raftery, 1998; Fraley and Raftery,
2002; von Luxburg et al., 2012). Because of that, the benchmark was deliberately split into:

- native operating-point analysis
- matched-granularity analysis

These answer different questions:

- native mode asks how each method behaves when allowed to select what its own internal criteria prefer
- matched mode asks how methods compare when cluster granularity is constrained to be comparable

## 5. Final Experimental Design

## 5.1 Dataset

The benchmark was run on the full current downloaded-audio subset after taxonomy reassignment:

- songs evaluated: `5,456`
- downloaded songs before taxonomy filtering: `5,535`
- unique primary genres: `187`
- unique artists: `3,038`
- raw audio feature dimensions: `116`

No subset-stage shortlist was used.

The audio-backed benchmark subset now comes from a larger taxonomy-building step over the full song table:

- total rows in `songs.csv`: `10,691`
- rows retained for MRS after taxonomy reassignment: `10,544`
- rows excluded because they only mapped to `non_genre_*` tags: `147`
- audio-backed excluded rows: `79`

### 5.1.1 Taxonomy reassignment and eligibility rule

This update introduces a new preprocessing stage driven by
`data/acoustically_coherent_merged_genres_corrected.csv`.

For every song, the pipeline now:

1. maps each source genre tag into the merged taxonomy
2. stores mapped non-`non_genre_*` tags as primary-genre candidates
3. stores mapped `non_genre_*` tags as secondary descriptive tags
4. writes both sets plus their union into `data/songs_with_merged_genres.csv`
5. excludes the song from the MRS benchmark if no mapped primary genre remains

That means the benchmark and recommender now use:

- `mapped_primary_genres` for primary taxonomy identity
- `mapped_secondary_tags` for non-genre context such as scene, region, instrument, or spoken/comedy labels
- `mapped_all_tags` for multivector overlap evaluation downstream

This matters scientifically because the benchmark is no longer evaluating against the raw source-label
space. It is evaluating against a smaller, more acoustically coherent taxonomy while preserving
non-genre metadata separately instead of letting it masquerade as a primary genre.

## 5.2 Representation space

The new benchmark used `14` prespecified representations.

They were chosen to cover the major MIR families and their meaningful combinations without using earlier
pilot results to decide which ones were worth testing.

| Representation | Raw dims | Why it was included |
|---|---:|---|
| `mfcc_only` | `26` | Static timbre baseline |
| `delta_mfcc_only` | `26` | First-order timbral dynamics |
| `delta2_mfcc_only` | `26` | Second-order timbral dynamics |
| `timbre_full` | `78` | Full classical timbre stack |
| `spectral_shape` | `10` | Brightness and spectral-shape family |
| `pitch_only` | `24` | Chroma-only harmonic baseline |
| `rhythm_only` | `4` | Beat-strength-only rhythmic baseline |
| `pitch_rhythm` | `28` | Harmonic-rhythmic joint baseline |
| `timbre_pitch` | `102` | Timbre plus harmony |
| `timbre_spectral` | `88` | Timbre plus spectral texture/shape |
| `spectral_pitch` | `34` | Spectral shape plus harmony |
| `spectral_rhythm` | `14` | Spectral shape plus rhythm |
| `timbre_pitch_rhythm` | `106` | Timbre plus harmony plus rhythm |
| `all_audio` | `116` | Full engineered audio stack |

### 5.2.1 Exact raw feature groups used by the benchmark

The `14` thesis representations were built from `10` lower-level raw feature groups defined in the code.

Those groups and their raw dimensionalities were:

| Raw feature group | Dims |
|---|---:|
| `mfcc` | `26` |
| `delta_mfcc` | `26` |
| `delta2_mfcc` | `26` |
| `spectral_centroid` | `2` |
| `spectral_rolloff` | `2` |
| `spectral_flux` | `2` |
| `spectral_flatness` | `2` |
| `zero_crossing_rate` | `2` |
| `chroma` | `24` |
| `beat_strength` | `4` |

This matters because the benchmark is not comparing abstract names like "timbre" and "rhythm" in a
hand-wavy way. It is comparing concrete numeric blocks that are actually present in the repository cache.

The most important structural consequence is:

- MFCC-derived timbral families are relatively high-dimensional
- chroma is medium-dimensional
- beat-strength is very compact
- the scalar spectral descriptors are extremely low-dimensional on their own

That dimensional asymmetry is one reason preprocessing was so important in the benchmark.

## 5.3 Preprocessing modes

Each representation was evaluated under three preprocessing modes:

1. `raw_zscore`
2. `pca_per_group_2`
3. `pca_per_group_5`

The purpose of this design was not to "pre-prune" the search space using the pilot, but to test whether
compact group-preserving projections behave differently from standardized raw family stacks.

### 5.3.1 What `raw_zscore` does step by step

For `raw_zscore`, the code does the following:

1. Select the requested raw feature groups.
2. Concatenate those groups into one matrix.
3. Standardize every selected column using `StandardScaler`.
4. Pass the standardized matrix directly to the clustering algorithm.

This mode preserves all selected raw dimensions after standardization.

It is the closest thing in the benchmark to a "plain classical feature-engineering baseline".

### 5.3.2 What `pca_per_group_2` and `pca_per_group_5` do step by step

For grouped PCA modes, the code does not run one giant PCA over the entire representation. Instead, it
processes each feature family separately.

For every group inside the representation, the code:

1. extracts the raw block for that group
2. standardizes that group internally
3. applies PCA only to that group, if the group has more dimensions than the target component count
4. keeps up to `2` components for `pca_per_group_2` or up to `5` components for `pca_per_group_5`
5. pads with zeros if the group has fewer dimensions than the target component count
6. normalizes the transformed group by its average energy scale
7. concatenates all transformed groups back together

This grouped procedure was chosen for a reason.

It retains the high-level identity of each feature family while reducing dimensional dominance by large
groups. That is more interpretable than flattening everything into one global PCA space, because the
representation still preserves the distinction between timbre, pitch, rhythm, and spectral families.

### 5.3.3 Why grouped PCA was scientifically useful here

Grouped PCA served three research purposes:

1. It tested whether lower-dimensional family-level summaries perform better than raw standardized stacks.
2. It reduced the risk that large groups such as MFCC-derived families overwhelm smaller groups simply by
   dimension count.
3. It preserved family-level interpretability, which is useful in a thesis because the conclusions can be
   tied back to concrete audio-feature families rather than to a single opaque embedding.

## 5.4 Algorithms and search spaces

The benchmark compared three clustering methods:

1. KMeans
2. GMM
3. HDBSCAN

Parameter spaces used in the full run:

- KMeans: `k = 2..20`
- GMM: `n_components = 2..20`, covariance types `{full, diag}`
- HDBSCAN: `32` parameter pairs from the project search-space function for this dataset size

### 5.4.1 KMeans in this benchmark

KMeans is the centroid-based baseline.

In this implementation, for each candidate `k`, the code:

1. initializes a `KMeans` model with the requested number of clusters
2. uses `random_state=42`
3. uses `n_init=20` to reduce poor local minima from a single initialization
4. fits the model on the prepared feature matrix
5. records cluster labels and evaluation metrics

Why this method matters in the thesis:

- it is the classical baseline that many readers will expect
- it gives a clean partition of the whole dataset
- it is computationally efficient relative to more flexible alternatives
- it tells us what a simple partitioning method can achieve with the same feature input

### 5.4.2 GMM in this benchmark

GMM is the probabilistic parametric alternative.

For each candidate component count and covariance type, the code:

1. initializes a `GaussianMixture` model
2. uses `random_state=42`
3. uses `max_iter=200`, `tol=1e-3`, `reg_covar=1e-5`
4. uses `init_params="kmeans"` so the optimization begins from a structured initialization
5. fits the model
6. predicts hard labels from the fitted mixture
7. records metrics, plus `AIC`, `BIC`, and mean max posterior confidence

Why this method matters in the thesis:

- it allows non-spherical cluster geometry through covariance modeling
- it offers a probabilistic interpretation rather than purely geometric partitioning
- it is the natural soft-clustering comparator to KMeans

### 5.4.3 HDBSCAN in this benchmark

HDBSCAN is the density-based method in the study.

For each parameter pair, the code:

1. builds an `HDBSCAN` model with the given `min_cluster_size` and `min_samples`
2. fits the model
3. obtains labels, including possible noise labels `-1`
4. computes `coverage = 1 - noise_fraction`
5. evaluates clustering quality only on the non-noise subset for metrics that require assigned clusters
6. tries to compute DBCV as a density-validity metric

The search space for this dataset size (`5,456` songs) is generated from:

- `min_cluster_size` values: `5`, `8`, `10`, `15`, `25`, `109`, `218`
- `min_samples` values derived from `{1, 2, 5, min_cluster_size // 2, min_cluster_size}`

After removing invalid duplicates, this gives `32` HDBSCAN settings in the full benchmark.

Why this method matters in the thesis:

- it can discover non-convex or irregular density structure
- it can reject ambiguous points as noise
- it behaves differently enough from KMeans and GMM to make the comparison scientifically informative

## 5.5 Evaluation outputs

The benchmark computed a broad set of metrics, but the most important ones for interpretation were:

- `silhouette` for geometry/separation
- `nmi` for alignment with primary genre labels
- `stability_ari` for repeatability under resampling/restarts
- `coverage` and `noise_fraction` for HDBSCAN practicality
- `fit_time_sec` for computational cost

### 5.5.1 Exact meaning of the core metrics

The report uses several terms repeatedly. This subsection explains exactly what they mean in the
benchmark.

#### `n_clusters`

This is the number of unique cluster labels produced by the model.

For HDBSCAN, noise label `-1` is not counted as a cluster.

#### `cluster_balance`

This is a normalized entropy score over cluster sizes.

Interpretation:

- closer to `1`: cluster sizes are more evenly distributed
- closer to `0`: one or a few clusters dominate strongly

This is not a "goodness" metric by itself, but it helps penalize pathological cases where nearly all
points fall into one cluster.

#### `silhouette`

This is the classical silhouette coefficient.

Interpretation:

- higher positive values usually indicate cleaner separation
- values near `0` indicate overlapping structure
- negative values indicate poor separation or misassignment

In this benchmark, silhouette is computed with a sample size cap of `5,000` songs for efficiency when the
dataset is larger than that.

#### `nmi`

This is normalized mutual information between cluster labels and the taxonomy-aware primary genre label
used for benchmarking.

Interpretation:

- higher values mean stronger alignment between cluster structure and the metadata genre partition
- it does not mean the clustering is "correct" in an absolute musical sense

#### `ari`, `ami`, `homogeneity`, `completeness`, `v_measure`

These are additional external semantic-alignment metrics.

They were logged so the benchmark would not rely on one external score only, but the report mainly uses
`NMI` because it is easier to communicate and was stable enough as the main semantic-alignment reference.

#### `coverage` and `noise_fraction`

These are especially important for HDBSCAN.

- `coverage` = fraction of songs assigned to non-noise clusters
- `noise_fraction` = fraction of songs labeled as noise

These metrics are critical because HDBSCAN can achieve high semantic alignment partly by excluding difficult
points. A high-NMI HDBSCAN result is much less impressive if it only clusters a tiny minority of the
dataset.

#### `singleton_fraction`

This is the fraction of non-noise clusters that contain exactly one point.

Interpretation:

- high values can indicate fragmentation
- low values mean the clustering is not dominated by singleton groups

#### `largest_cluster_fraction`

This is the fraction of assigned points belonging to the single largest cluster.

Interpretation:

- high values can indicate excessive dominance by one cluster
- lower values suggest more distributed structure

#### `AIC` and `BIC`

These apply to GMM only.

They measure model fit penalized by complexity.

In the benchmark:

- lower values are better
- `BIC` is used inside the internal selection score for GMM

#### `avg_confidence`

This is the mean of the maximum posterior cluster probability for each point under GMM.

Interpretation:

- higher values mean the model tends to make more decisive assignments

#### `DBCV`

This is a density-based clustering validity metric used for HDBSCAN.

It is included because density-based clustering needs a validation signal that is more appropriate than
centroid-oriented measures alone.

### 5.5.2 Exactly how external metrics are computed for HDBSCAN

This point is easy to miss but very important.

For HDBSCAN:

- when noise exists, external metrics such as `NMI` are computed on the non-noise subset only

This is why coverage and noise have to be reported next to HDBSCAN semantic scores. Otherwise a result can
look extremely strong semantically while only describing a very small, easy subset of the collection.

### 5.5.3 Exactly how `stability_ari` is computed

The stability calculation is not identical for all three algorithms.

#### KMeans stability

For KMeans, the script:

1. reruns the chosen `k` with seeds `0, 1, 2, 3, 4`
2. collects the five labelings
3. computes the pairwise Adjusted Rand Index between every pair of runs
4. reports the mean pairwise ARI

Interpretation:

- higher means the clustering is reproducible across random initializations

#### GMM stability

For GMM, the script:

1. reruns the chosen component count and covariance type with seeds `0, 1, 2, 3, 4`
2. fits each model from that seed
3. predicts hard labels for each run
4. computes the mean pairwise ARI across those runs

Interpretation:

- higher means the fitted mixture solution is less sensitive to initialization

#### HDBSCAN stability

For HDBSCAN, the script cannot use the same restart-only procedure because the method is deterministic for
fixed data and parameters. Instead, it uses a subsampling-based robustness check.

The script:

1. draws `3` random subsamples
2. each subsample keeps approximately `90%` of the songs
3. fits HDBSCAN on each subsample with the same parameters
4. intersects the song indices shared by each pair of subsamples
5. removes points that are noise in either run for that pairwise comparison
6. computes ARI on the remaining common non-noise points
7. averages the valid pairwise ARI scores

Interpretation:

- higher means the discovered dense structure persists under mild data perturbation

This is an important nuance in the report because a HDBSCAN stability score is not directly the same kind
of object as a KMeans or GMM seed-stability score. It is closer to a subsampling robustness estimate.

### 5.5.4 Taxonomy-aware multivector recommendation metrics

The downstream production rerun does not rely only on exact single-label genre agreement.

After taxonomy reassignment, each song has three tag views:

- `mapped_primary_genres`
- `mapped_secondary_tags`
- `mapped_all_tags`

For the recommendation evaluation, those comma-separated tag lists are binarized into multivector form.
At each `K`, the evaluation then computes:

- `PrimaryTagPrecision@K`: fraction of returned songs that share at least one mapped primary tag with the
  query
- `PrimaryTagHitRate@K`: whether the query receives at least one primary-tag-overlap recommendation
- `PrimaryTagJaccard@K`: average Jaccard overlap between the query primary-tag set and each returned song
- `AllTagPrecision@K`, `AllTagHitRate@K`, `AllTagJaccard@K`: the same metrics on the union of primary and
  secondary tags
- `GenrePrecision@K` and `GenreHitRate@K`: exact single-primary-label continuity diagnostics only

This is a materially better downstream evaluation design for the merged taxonomy because it rewards shared
primary genre identity and shared secondary context instead of treating every song as if it had only one
meaningful label.

## 5.6 Selection regimes

Two evaluation regimes were used.

### Native operating point

Each method/representation/preprocessing group was allowed to select its own best internal configuration.

The internal selection score was method-specific:

- KMeans: silhouette + Calinski-Harabasz + Davies-Bouldin + cluster balance
- GMM: the same family, plus BIC
- HDBSCAN: DBCV + coverage + silhouette + cluster balance

### Matched granularity

Matched targets were:

- `4`
- `8`
- `12`
- `16`
- `20`

For each method/representation/preprocessing setting, the benchmark selected the best candidate within a
band around the target cluster count. The band width was `25%`, with a minimum tolerance of `1`.

That means the matched analysis is not asking "what does the method naturally choose?" It is asking:

- "how does the method perform when we compare it at approximately the same granularity?"

### 5.6.1 Exact native-selection scoring logic

The benchmark does not use one universal internal score for all methods. It uses method-specific weighted
scores because the methods do not produce the same kinds of diagnostics.

Before weighting, each metric is min-max normalized within each:

- representation
- preprocessing mode
- method

That means the internal selection score answers:

- "which setting is best for this method on this specific prepared representation?"

The exact weights were:

#### KMeans native internal score

- `silhouette`: `0.40`
- `calinski_harabasz`: `0.25`
- `davies_bouldin`: `0.20` with lower treated as better
- `cluster_balance`: `0.15`

Tie-breaking order:

- internal score
- silhouette
- Calinski-Harabasz
- cluster balance

#### GMM native internal score

- `silhouette`: `0.30`
- `calinski_harabasz`: `0.20`
- `davies_bouldin`: `0.15` with lower treated as better
- `cluster_balance`: `0.15`
- `bic`: `0.20` with lower treated as better

Tie-breaking order:

- internal score
- silhouette
- average confidence
- BIC

#### HDBSCAN native internal score

- `dbcv`: `0.40`
- `coverage`: `0.25`
- `silhouette`: `0.20`
- `cluster_balance`: `0.15`

Tie-breaking order:

- internal score
- DBCV
- coverage
- silhouette

This weighted design matters because it explains why:

- KMeans and GMM often select very coarse native solutions
- HDBSCAN can prefer richer native solutions when density validity and coverage support them

### 5.6.2 Exact matched-selection logic

The matched-granularity stage happens after the full grid has already been evaluated.

For each:

- representation
- preprocessing mode
- method
- target cluster count

the code:

1. computes `cluster_gap = abs(n_clusters - target)`
2. keeps only rows within the allowed tolerance
3. sorts those rows by:
   - smallest cluster gap first
   - highest internal selection score second
   - highest silhouette third
4. selects the top row as the scenario-level matched best

The tolerance rule is:

- `max(1, round(target * 0.25))`

So the exact tolerances were:

| Target | Tolerance |
|---:|---:|
| `4` | `1` |
| `8` | `2` |
| `12` | `3` |
| `16` | `4` |
| `20` | `5` |

This means the matched stage is intentionally approximate rather than perfectly rigid.

That is especially important for HDBSCAN, because density-based clustering cannot be forced directly to a
requested cluster count the way KMeans and GMM can.

## 6. What Was Implemented

The script [thesis_clustering_benchmark.py](c:/Users/vpddk/Desktop/Me/Github/ML-song-recommendation-system/scripts/analysis/thesis_clustering_benchmark.py) was reworked so that:

- it no longer performs subset-stage or pilot-stage shortlisting
- it no longer uses promotion from exploratory scenarios
- it resolves the taxonomy-aware song table automatically
- it evaluates the full prespecified representation space directly
- it writes a `representation_catalog.csv`
- it writes full-grid, per-scenario native, per-scenario matched, and global-leader CSVs
- it records taxonomy-aware metadata paths in the benchmark bundle

In addition, the repository now contains:

- a taxonomy-builder utility that creates `data/songs_with_merged_genres.csv`
- a taxonomy helper module that exposes mapped primary genres, mapped secondary tags, and MRS inclusion
  flags
- an updated recommendation evaluator that scores retrieval quality with multivector primary-tag and
  all-tag overlap metrics

The key output bundle from the final run is:

- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19`

The downstream production rerun bundle is:

- `output/experiment_runs_taxonomy/run_20260319_140333Z/recommended_production`

### 6.1 End-to-end program flow

The benchmark script runs in the following order.

1. Parse command-line arguments.
2. Create or resolve the output directory.
3. Load a cached raw feature matrix if one already exists.
4. If no suitable cache exists, rebuild the raw matrix from project feature files and save a fresh cache.
5. Build aligned metadata for all audio files.
6. Encode the genre labels numerically for metric computation.
7. Compute descriptive summaries:
   - feature-group variance summary
   - feature-group correlation summary
8. Build the prespecified `representation_catalog`.
9. Loop over every representation.
10. Inside that loop, iterate over every preprocessing mode.
11. Prepare the feature matrix for that representation/mode pair.
12. Run the KMeans parameter grid.
13. Run the GMM parameter grid.
14. Run the HDBSCAN parameter grid.
15. Append all those rows into the global full-grid table.
16. After the full grid is complete, compute internal selection scores.
17. Select one native-best row per method/representation/preprocessing scenario.
18. Select one matched-best row per method/representation/preprocessing/target scenario.
19. Run stability estimation for every selected native and matched row.
20. Derive the global native leaders.
21. Derive the global matched leaders.
22. Write all CSV, JSON, and markdown artifacts.

This explicit flow matters because it shows where the study becomes confirmatory:

- not at the start
- but only after the full representation-by-preprocessing-by-method grid has been evaluated

### 6.2 Exact role of the metadata alignment step

The metadata alignment step is easy to overlook, but it is essential.

The script uses [song_metadata.py](c:/Users/vpddk/Desktop/Me/Github/ML-song-recommendation-system/src/utils/song_metadata.py)
to build an aligned metadata table for the audio files. That table gives the benchmark:

- artist names
- original primary genre labels
- mapped primary genre labels
- mapped secondary tags
- mapped full tag lists
- MRS inclusion status
- normalized artist keys

Even though the pilot-free benchmark does not use artist-aware shortlisting anymore, aligned metadata still
matters because:

- taxonomy-aware primary genre labels are needed for external benchmark metrics
- secondary tags are needed for multivector recommendation evaluation
- artist and genre counts are part of dataset characterization
- excluded non-genre-only songs must be filtered consistently before any clustering step
- the output bundle becomes easier to audit and interpret later

### 6.3 Exact outputs written by the benchmark

The benchmark writes artifacts in a deliberate order.

1. `feature_group_variance_summary.csv`
2. `feature_group_correlation_summary.csv`
3. `representation_catalog.csv`
4. `full_grid_results.csv`
5. `native_best_results.csv`
6. `matched_granularity_results.csv`
7. `global_native_leaders.csv`
8. `global_matched_leaders.csv`
9. `aligned_metadata.csv`
10. `dataset_summary.json`
11. `benchmark_report.md`

This is useful when auditing or reproducing the study because each file corresponds to a specific stage of
the pipeline rather than being a redundant export.

## 7. Execution Log

## 7.1 Taxonomy dataset build

Before rerunning the benchmark, the song metadata was rebuilt with the merged taxonomy so that raw genres,
taxonomy-backed primary genres, and secondary `non_genre_*` tags were separated explicitly.

Command:

```powershell
.venv\Scripts\python.exe scripts\utilities\build_merged_genre_dataset.py
```

Output:

- `data/songs_with_merged_genres.csv`

Observed dataset outcome:

- `10,691` total songs in the source table
- `10,544` rows retained for MRS
- `147` rows excluded because they had no mapped primary genre
- `5,456` downloaded songs retained for clustering and benchmarking

## 7.2 Full taxonomy-aware benchmark rerun

The full benchmark was then rerun against the taxonomy-aware song table with the same pilot-free search
space.

Command:

```powershell
.venv\Scripts\python.exe scripts\analysis\thesis_clustering_benchmark.py `
  --output-dir output\metrics\thesis_benchmark_taxonomy_full_2026-03-19
```

Final bundle:

- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19`

Key dataset summary written by that run:

- songs evaluated: `5,456`
- unique primary genres: `187`
- unique artists: `3,038`
- raw audio dimensions: `116`
- representations: `14`
- preprocessing modes: `3`
- matched targets: `4 8 12 16 20`

## 7.3 Targeted validation and production rerun

During implementation, a targeted taxonomy-aware smoke validation was used to verify that the raw cache,
feature-collection path, and benchmark scorer were compatible with the new metadata contract. After that,
the full production clustering and recommendation comparison were rerun.

Command:

```powershell
.venv\Scripts\python.exe scripts\run_all_clustering.py `
  --profiles recommended_production `
  --run-root output\experiment_runs_taxonomy
```

Final production bundle:

- `output/experiment_runs_taxonomy/run_20260319_140333Z/recommended_production`

This rerun is the source of the taxonomy-aware recommendation metrics reported later in this document.
- very small floating-point drift in a small minority of rows
- tiny internal-score differences that did not change the thesis conclusions

This matters because the benchmark should not be presented as a one-off lucky run. It has now been
rechecked and shown to be stable in all important respects.

## 8. Descriptive Feature Evidence

The new benchmark did not use descriptive statistics to pre-prune the search space, but those statistics
were still computed because they are useful context for interpreting outcomes.

### 8.1 Variance summary

Highest-variance feature groups in the full dataset were:

| Group | Mean variance |
|---|---:|
| `spectral_rolloff` | `1.163887e+06` |
| `spectral_centroid` | `2.427934e+05` |
| `mfcc` | `3.079833e+02` |
| `beat_strength` | `1.140045e+02` |

Lowest-variance groups were:

| Group | Mean variance |
|---|---:|
| `spectral_flatness` | `4.449623e-04` |
| `zero_crossing_rate` | `1.006246e-03` |
| `chroma` | `8.585255e-03` |

Interpretation:

- the spectral-scalar descriptors operate at very different numeric scales
- beat-strength features carry much more raw variance than chroma
- z-scoring is necessary for fair algorithm comparison across mixed families

### 8.2 Correlation summary

Strongest average cross-group correlations were:

| Group A | Group B | Mean absolute correlation |
|---|---|---:|
| `spectral_flux` | `beat_strength` | `0.668851` |
| `spectral_centroid` | `zero_crossing_rate` | `0.601268` |
| `spectral_centroid` | `spectral_rolloff` | `0.582089` |
| `spectral_flatness` | `zero_crossing_rate` | `0.524990` |
| `spectral_rolloff` | `zero_crossing_rate` | `0.500764` |

Interpretation:

- strong rhythmic-energy linkage exists between `spectral_flux` and `beat_strength`
- spectral scalars are meaningfully redundant in places
- the descriptive evidence supports using PCA variants as one benchmark condition
- this evidence was **not** used to remove representations from the full benchmark

## 9. Result Inventory

The final full run produced:

- `42` representation/preprocessing scenarios
- `3,738` fitted models in `full_grid_results.csv`
- `126` native best rows in `native_best_results.csv`
- `503` matched-granularity rows in `matched_granularity_results.csv`
- `3` global native leaders in `global_native_leaders.csv`
- `15` global matched leaders in `global_matched_leaders.csv`

The reduced number of matched rows relative to the theoretical maximum comes from the fact that some
method/representation combinations did not produce valid candidates inside some target bands.

### 9.1 How to read the output bundle file by file

This subsection is included so that the report is not just a narrative summary. It tells you how to use
the output files directly.

#### `representation_catalog.csv`

Use this file to answer:

- which representations were evaluated?
- which raw groups belong to each representation?
- how many raw dimensions does each representation contain?
- what was the intended rationale for each representation?

This file is the design manifest for the benchmark.

#### `feature_group_variance_summary.csv`

Use this file to understand:

- which low-level feature families have the largest raw spread
- which families are numerically tiny and therefore most dependent on standardization

This file is descriptive context, not a model-selection file.

#### `feature_group_correlation_summary.csv`

Use this file to understand:

- which feature families overlap strongly
- where grouped PCA may help by compressing correlated blocks

Again, this is contextual evidence, not a direct winner table.

#### `full_grid_results.csv`

This is the most important raw result file.

Every row corresponds to:

- one representation
- one preprocessing mode
- one algorithm
- one hyperparameter setting

This file should be used when you want to ask questions like:

- what is the highest NMI achieved anywhere?
- what is the highest-silhouette KMeans result at target 12?
- which HDBSCAN settings produce high NMI but poor coverage?
- which preprocessing mode gives the best all-audio GMM result?

If a result claim cannot be traced back to this file, it is not well grounded.

#### `native_best_results.csv`

This file contains one selected native row per:

- representation
- preprocessing mode
- method

It is the scenario-level answer to:

- what does this method prefer on this prepared representation if we let its internal criteria choose?

This file is the correct place to study native method behavior.

#### `matched_granularity_results.csv`

This file contains one selected row per:

- representation
- preprocessing mode
- method
- matched target

It is the scenario-level answer to:

- what is the best approximately target-sized clustering for this method on this representation?

This file is the correct place to study medium-granularity fairness across methods.

#### `global_native_leaders.csv`

This file reduces `native_best_results.csv` even further.

It contains one native leader per method across the whole representation space.

This file is useful for concise reporting, but it should always be interpreted together with:

- `native_best_results.csv`

because a single global leader can hide interesting second-place patterns.

#### `global_matched_leaders.csv`

This file contains one row per:

- method
- matched target

It is the most compact summary of the matched-granularity benchmark.

It is useful for thesis tables, but it is not enough on its own because the raw best NMI row inside a
target band is not always the same as the internally selected balanced leader.

#### `aligned_metadata.csv`

This file links the feature-ordering used in the benchmark back to interpretable metadata.

It is useful for later analysis such as:

- artist inspection
- genre inspection
- cluster-content auditing

#### `dataset_summary.json`

This is the benchmark manifest.

It records:

- dataset counts
- preprocessing modes
- matched targets
- cache path
- timestamp

This file is useful for quick sanity checks and for proving what configuration was actually run.

#### `benchmark_report.md`

This is the lightweight machine-generated bundle summary.

It is useful for a fast glance, but the detailed thesis interpretation should come from this report and the
CSV files rather than from that compact markdown summary alone.

### 9.2 How to read one row from `full_grid_results.csv`

To understand the benchmark deeply, it helps to know how to read a single row correctly.

A row in `full_grid_results.csv` should be interpreted in this order:

1. `method`
   This tells you whether the row came from KMeans, GMM, or HDBSCAN.
2. `combo`
   This tells you which representation was used.
3. `preprocess_mode`
   This tells you how the raw groups were transformed before clustering.
4. `param_1_name`, `param_1_value`, `param_2_name`, `param_2_value`
   These specify the exact model setting.
5. `n_clusters`
   This tells you the granularity of the produced solution.
6. `coverage` and `noise_fraction`
   These are crucial for HDBSCAN.
7. `silhouette`
   This tells you how geometrically separated the solution is.
8. `nmi`
   This tells you how strongly the result aligns with primary genre labels.
9. `cluster_balance`, `singleton_fraction`, `largest_cluster_fraction`
   These tell you whether the clustering is dominated by giant or tiny clusters.
10. `internal_selection_score`
    This tells you how attractive the row was to the benchmark's internal scenario-level selector.

If you skip this order, it becomes very easy to overinterpret a single attractive number.

## 10. Main Results

### How the result sections should be read

The rest of the report uses several recurring labels that need to be interpreted carefully.

When the report says:

- `raw best`

it means:

- the numerically highest value on one metric, without necessarily considering whether the solution is
  practical, full-coverage, or balanced across other criteria

When the report says:

- `native leader`

it means:

- the row selected by the benchmark's internal native-selection logic for that method

When the report says:

- `matched leader`

it means:

- the row selected inside a target cluster-count band using the matched-granularity selection procedure

When the report says:

- `practical`

it means:

- a result that is not only numerically strong, but also more usable as a clustering of the collection as
  a whole, especially with respect to coverage, fragmentation, and interpretability

When the report says:

- `balanced`

it means:

- a result that does not win only one metric, but remains defensible across geometry, semantic alignment,
  and stability

This distinction is essential for understanding why the report sometimes does **not** choose the highest
single NMI row as the best thesis-level conclusion.

## 10.1 Native operating-point leaders

Global native leaders selected by the benchmark were:

| Method | Representation | Preprocess | Clusters | Coverage | Noise | Silhouette | NMI | Stability |
|---|---|---|---:|---:|---:|---:|---:|---:|
| KMeans | `mfcc_only` | `raw_zscore` | `2` | `1.000` | `0.000` | `0.194` | `0.040` | `0.999` |
| GMM | `delta_mfcc_only` | `pca_per_group_5` | `2` | `1.000` | `0.000` | `0.290` | `0.043` | `0.995` |
| HDBSCAN | `rhythm_only` | `pca_per_group_2` | `563` | `0.915` | `0.085` | `0.569` | `0.453` | `0.887` |

This updated native table changes the thesis story in two important ways.

First, the taxonomy-aware label space compresses the semantic evaluation problem. As a result, the native
parametric leaders still prefer extremely coarse solutions and therefore post very small `NMI` values.
Second, HDBSCAN remains the only method that discovers rich native fine-grained structure at a practical
coverage level.

The strongest practical native thesis result is still:

- `HDBSCAN + rhythm_only + pca_per_group_2`
- `563` clusters
- `91.5%` coverage
- `8.5%` noise
- `silhouette=0.569`
- `NMI=0.453`
- `stability_ari=0.887`

## 10.2 Highest raw NMI overall

The highest raw `NMI` anywhere in the full benchmark grid is now:

- method: `HDBSCAN`
- representation: `pitch_rhythm`
- preprocess: `pca_per_group_2`
- parameters: `min_cluster_size=5`, `min_samples=1`
- clusters: `333`
- coverage: `0.540`
- noise fraction: `0.460`
- silhouette: `0.220`
- NMI: `0.483`

This row is scientifically interesting, but it is still not the best thesis headline result because it
rejects almost half of the collection as noise. The taxonomy-aware rerun therefore keeps the same
interpretive principle as before:

- the single highest semantic score is not automatically the most defensible thesis conclusion

It also changes the old narrative materially:

- `delta2_mfcc_only` is no longer the raw-NMI headline representation
- the best raw semantic row is now a joint harmonic-rhythmic representation

## 10.3 Native cluster-count behavior by method

The native cluster-count summaries still reveal a strong method effect.

### KMeans

- native selections with more than `2` clusters: `2` out of `42`
- maximum native cluster count: `3`

Interpretation:

- KMeans almost always collapses to a very coarse solution when judged only by its internal geometry

### GMM

- native selections with more than `2` clusters: `9` out of `42`
- maximum native cluster count: `10`

Interpretation:

- GMM is somewhat more flexible than KMeans in native mode
- but it still mostly prefers very coarse partitions

### HDBSCAN

- native selections with more than `2` clusters: `12` out of `42`
- maximum native cluster count: `563`

Interpretation:

- HDBSCAN is still the only method in this study that repeatedly discovers genuinely rich native
  structure
- but whether it does so depends strongly on the representation

## 10.4 Matched-granularity leaders selected by internal criteria

The benchmark also computed global matched-granularity leaders by method and target, using the benchmark's
matched selection logic rather than raw `NMI` alone.

| Target | Method | Representation | Preprocess | Clusters | Coverage | Silhouette | NMI | Stability |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 4 | KMeans | `pitch_rhythm` | `pca_per_group_2` | `4` | `1.000` | `0.240` | `0.104` | `0.989` |
| 4 | GMM | `mfcc_only` | `pca_per_group_2` | `4` | `1.000` | `0.345` | `0.075` | `0.923` |
| 4 | HDBSCAN | `spectral_pitch` | `pca_per_group_2` | `4` | `0.959` | `0.162` | `0.005` | `0.876` |
| 8 | KMeans | `pitch_rhythm` | `pca_per_group_5` | `8` | `1.000` | `0.152` | `0.121` | `0.973` |
| 8 | GMM | `pitch_rhythm` | `pca_per_group_2` | `8` | `1.000` | `0.188` | `0.127` | `0.577` |
| 8 | HDBSCAN | `spectral_rhythm` | `raw_zscore` | `8` | `0.207` | `0.110` | `0.132` | `0.949` |
| 12 | KMeans | `pitch_only` | `pca_per_group_2` | `12` | `1.000` | `0.337` | `0.123` | `0.921` |
| 12 | GMM | `mfcc_only` | `pca_per_group_5` | `12` | `1.000` | `0.125` | `0.143` | `0.458` |
| 12 | HDBSCAN | `rhythm_only` | `pca_per_group_2` | `12` | `0.914` | `-0.035` | `0.068` | `0.896` |
| 16 | KMeans | `pitch_only` | `pca_per_group_2` | `16` | `1.000` | `0.335` | `0.134` | `0.878` |
| 16 | GMM | `pitch_only` | `pca_per_group_5` | `16` | `1.000` | `0.124` | `0.145` | `0.502` |
| 16 | HDBSCAN | `rhythm_only` | `pca_per_group_5` | `16` | `0.518` | `-0.027` | `0.120` | `0.697` |
| 20 | KMeans | `rhythm_only` | `pca_per_group_2` | `20` | `1.000` | `0.341` | `0.133` | `0.707` |
| 20 | GMM | `mfcc_only` | `pca_per_group_2` | `20` | `1.000` | `0.308` | `0.152` | `0.517` |
| 20 | HDBSCAN | `rhythm_only` | `pca_per_group_2` | `20` | `0.844` | `-0.141` | `0.097` | `0.947` |

These internal-selection matched leaders support three thesis-level conclusions.

1. KMeans remains the strongest balanced matched method overall.
2. GMM is competitive and occasionally stronger on `NMI`, but its stability degrades as target count grows.
3. HDBSCAN is still not the strongest balanced medium-granularity method even though it is the strongest
   native fine-grained discovery method.

## 10.5 Matched semantic leaders and why they must be read carefully

If we ignore the internal matched-selection logic and simply ask for the highest `NMI` row inside each
matched-target and method group, the story changes again.

### Best matched NMI by target and method

| Target | Method | Representation | Preprocess | Clusters | Gap | Coverage | NMI | Silhouette | Stability |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 | KMeans | `timbre_pitch_rhythm` | `pca_per_group_2` | `4` | `0` | `1.000` | `0.113` | `0.188` | `0.972` |
| 4 | GMM | `all_audio` | `raw_zscore` | `4` | `0` | `1.000` | `0.116` | `0.046` | `0.887` |
| 4 | HDBSCAN | `spectral_pitch` | `raw_zscore` | `4` | `0` | `0.021` | `0.371` | `0.180` | `0.645` |
| 8 | KMeans | `timbre_pitch_rhythm` | `raw_zscore` | `8` | `0` | `1.000` | `0.146` | `0.035` | `0.919` |
| 8 | GMM | `all_audio` | `raw_zscore` | `8` | `0` | `1.000` | `0.152` | `0.037` | `0.440` |
| 8 | HDBSCAN | `mfcc_only` | `pca_per_group_5` | `6` | `2` | `0.132` | `0.152` | `0.179` | `0.869` |
| 12 | KMeans | `timbre_pitch` | `raw_zscore` | `12` | `0` | `1.000` | `0.165` | `0.026` | `0.612` |
| 12 | GMM | `all_audio` | `raw_zscore` | `12` | `0` | `1.000` | `0.164` | `0.025` | `0.563` |
| 12 | HDBSCAN | `rhythm_only` | `raw_zscore` | `13` | `1` | `0.518` | `0.110` | `-0.014` | `0.992` |
| 16 | KMeans | `all_audio` | `raw_zscore` | `16` | `0` | `1.000` | `0.180` | `0.025` | `0.518` |
| 16 | GMM | `all_audio` | `raw_zscore` | `16` | `0` | `1.000` | `0.179` | `0.020` | `0.395` |
| 16 | HDBSCAN | `rhythm_only` | `pca_per_group_5` | `16` | `0` | `0.518` | `0.120` | `-0.027` | `0.697` |
| 20 | KMeans | `all_audio` | `raw_zscore` | `20` | `0` | `1.000` | `0.195` | `0.019` | `0.415` |
| 20 | GMM | `all_audio` | `raw_zscore` | `20` | `0` | `1.000` | `0.197` | `-0.009` | `0.581` |
| 20 | HDBSCAN | `rhythm_only` | `pca_per_group_5` | `20` | `0` | `0.698` | `0.115` | `-0.171` | `0.934` |

This table is essential for honest interpretation.

It shows that the most spectacular HDBSCAN matched `NMI` values at low target counts are still not
full-collection solutions. For example:

- target `4`, `spectral_pitch / raw_zscore`: coverage `0.021`
- target `8`, `mfcc_only / pca_per_group_5`: coverage `0.132`

That means those rows are interesting density discoveries, but weak candidates for a practical
full-dataset medium-granularity clustering.

For the parametric methods, the raw matched-`NMI` story is now very close:

- GMM slightly exceeds KMeans at targets `4`, `8`, and `20`
- KMeans slightly exceeds GMM at targets `12` and `16`
- the gaps are small enough that geometry and stability still matter for the thesis conclusion

## 10.6 Taxonomy-aware production recommendation ranking

The downstream production rerun used the repository's recommended production profile:

- feature subset: `spectral_plus_beat`
- equalization: `pca_per_group`
- prepared dimension: `30`
- recommendation ranking depth: `K=10`

Under the new multivector evaluation, the production ranking was:

| Rank | Method | Clusters | PrimaryTagJaccard@10 | AllTagJaccard@10 | PrimaryTagHitRate@10 | AllTagHitRate@10 | CatalogCoverage | Subsample Median ARI |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | KMeans | `2` | `0.0620` | `0.0609` | `0.8171` | `0.8371` | `0.9853` | `0.9714` |
| 2 | GMM | `4` | `0.0617` | `0.0604` | `0.8180` | `0.8372` | `0.9870` | `0.8120` |
| 3 | HDBSCAN | `2` | `0.0596` | `0.0586` | `0.7784` | `0.7971` | `0.9395` | `0.4717` |

This production rerun matters because it evaluates the user-facing recommendation behavior after the
taxonomy change, not just the thesis benchmark grids.

The practical interpretation is:

- KMeans and GMM are nearly tied semantically under the new primary-tag and all-tag overlap metrics
- KMeans ranks first because it combines that near-tie semantic performance with much stronger stability
  and cleaner geometry
- GMM remains attractive because it provides soft assignments, but it still fails the explicit stability
  gate in the production report
- HDBSCAN is clearly weaker on the production profile because it collapses into one dominant cluster of
  `5,198` tracks plus a tiny `7`-track cluster and `4.6%` noise

## 10.7 Geometry-focused matched leaders

If the target is compact separation rather than taxonomy alignment, KMeans dominates from target `8`
upward.

Global best matched silhouette at each target was:

| Target | Best row | Coverage | NMI |
|---:|---|---:|---:|
| 4 | `HDBSCAN + delta_mfcc_only + pca_per_group_5` with `3` clusters and silhouette `0.419` | `0.024` | `0.205` |
| 8 | `KMeans + rhythm_only + pca_per_group_2`, silhouette `0.336` | `1.000` | `0.092` |
| 12 | `KMeans + rhythm_only + pca_per_group_2`, silhouette `0.345` | `1.000` | `0.107` |
| 16 | `KMeans + rhythm_only + pca_per_group_2`, silhouette `0.335` | `1.000` | `0.120` |
| 20 | `KMeans + rhythm_only + pca_per_group_2`, silhouette `0.341` | `1.000` | `0.133` |

This reveals a strong rhythmic geometry effect that survives the taxonomy change:

- `rhythm_only + pca_per_group_2` is repeatedly the cleanest medium-granularity partition for KMeans

That does not make rhythm-only the universally best representation. It means the beat-strength family
organizes this collection very cleanly for centroid-based clustering.

## 10.8 Computational cost

Full-run fit-time totals were:

| Method | Fits | Total fit time (s) | Mean fit time (s) | Median fit time (s) | Max fit time (s) |
|---|---:|---:|---:|---:|---:|
| KMeans | `798` | `297.258` | `0.373` | `0.377` | `1.437` |
| GMM | `1,596` | `815.482` | `0.511` | `0.113` | `18.251` |
| HDBSCAN | `1,344` | `1,187.644` | `0.884` | `0.465` | `5.323` |

Interpretation:

- KMeans remains the cheapest method on mean runtime
- GMM still has a heavy tail because some covariance settings are expensive
- HDBSCAN remains the most expensive method overall, but it also remains the only method that discovers
  highly granular native structure

## 10.9 Representation and preprocessing patterns

Several representation-level patterns were stable in the taxonomy-aware rerun.

### Pattern 1: HDBSCAN still favors compact, coherent spaces

The strongest HDBSCAN semantic rows are now concentrated in representations such as:

- `pitch_rhythm / pca_per_group_2`
- `rhythm_only / pca_per_group_2`
- `delta_mfcc_only / pca_per_group_2`
- `pitch_only / pca_per_group_2`
- `mfcc_only / pca_per_group_2`

This again suggests that density discovery is strongest when the representation is focused and
structurally coherent rather than maximally wide.

### Pattern 2: `all_audio / raw_zscore` still helps raw matched semantic alignment for parametric models

Even though the internal matched leaders moved toward more compact families, the best raw matched `NMI`
rows for KMeans and GMM at higher targets still often come from:

- `all_audio / raw_zscore`

That means the wider mixed-family stack still helps parametric semantic alignment once the models are
forced to produce many clusters.

### Pattern 3: `pca_per_group_2` remains especially strong for rhythmic organization

Both the practical native HDBSCAN leader and the best matched-geometry KMeans rows repeatedly use:

- `rhythm_only / pca_per_group_2`

So light grouped compression still appears to sharpen rhythmic structure rather than damage it.

### Pattern 4: The taxonomy-aware rerun changes the headline representation story

The old raw-label rerun gave the most dramatic headline to `delta2_mfcc_only`.

The taxonomy-aware rerun changes that:

- the highest raw `NMI` row is now `pitch_rhythm / pca_per_group_2`
- the best practical native row is still `rhythm_only / pca_per_group_2`
- compact harmonic-rhythmic and rhythmic spaces matter more than the old headline suggested

## 11. What Changed Relative to the Earlier Pilot-Shaped Story

Removing the pilot changed the thesis once. The taxonomy-aware rerun changes it again.

### 11.1 The old `delta2_mfcc` headline no longer survives

Earlier pilot-shaped and raw-label narratives emphasized:

- `HDBSCAN + delta2_mfcc + grouped PCA`

In the taxonomy-aware rerun, that is no longer the headline story:

- the highest raw `NMI` overall is now `HDBSCAN + pitch_rhythm + pca_per_group_2`
- the strongest practical native leader is still `HDBSCAN + rhythm_only + pca_per_group_2`
- `delta2_mfcc_only` remains relevant, but it is no longer the best single summary of the study

### 11.2 The medium-granularity story is now split between internal balance and raw semantic maxima

Under the benchmark's internal matched-selection logic, the best balanced solutions come from relatively
compact spaces such as:

- `pitch_rhythm`
- `pitch_only`
- `rhythm_only`
- `mfcc_only`

But when ranking purely by raw matched `NMI`, higher-target parametric rows still often come from:

- `all_audio / raw_zscore`

That means the thesis should not claim that one representation family dominates unconditionally. The
answer depends on whether the emphasis is balanced clustering, raw semantic alignment, or production
recommendation behavior.

### 11.3 Taxonomy-aware evaluation changed the downstream recommendation story too

The new metadata contract introduced three important evaluation changes:

- `79` downloaded songs with only `non_genre_*` mappings are now excluded from the MRS benchmark
- the benchmark label space shrank to `187` taxonomy-backed primary genres
- the production comparison now ranks methods by multivector primary-tag and all-tag overlap, not just by
  exact single-label genre equality

That makes the downstream recommendation comparison more faithful to the user's intended genre taxonomy.

## 12. Final Answer to the Thesis Question

The thesis subject is:

> A study and comparison of the performance of machine learning algorithms for clustering music tracks
> based on their audio features.

The most defensible answer, based on this pilot-free and taxonomy-aware rerun, is the following.

### 12.1 There is no single absolute best algorithm

The benchmark still does not support a simple statement like:

- "KMeans is best"
- "GMM is best"
- "HDBSCAN is best"

The result depends on what "performance" means and on how that performance is evaluated.

### 12.2 If performance means native fine-grained unsupervised discovery, HDBSCAN is strongest

HDBSCAN is still the only method that repeatedly discovers rich native fine-grained structure.

The strongest practical native result is:

- `HDBSCAN + rhythm_only + pca_per_group_2`
- `563` clusters
- `8.5%` noise
- `silhouette=0.569`
- `NMI=0.453`
- `stability_ari=0.887`

The highest raw `NMI` overall is also HDBSCAN:

- `HDBSCAN + pitch_rhythm + pca_per_group_2`
- `NMI=0.483`
- `46.0%` noise

But that row is not the best thesis headline because of its large noise fraction.

### 12.3 If performance means balanced medium-granularity clustering and production recommendation quality, KMeans is strongest overall

Under matched-granularity comparison and under the downstream taxonomy-aware recommendation rerun:

- KMeans is the most balanced method overall
- it combines near-best semantic overlap with clearly stronger stability than GMM
- it avoids the severe coverage and dominance problems seen in HDBSCAN on the production profile

### 12.4 GMM is the closest competitor, especially when soft assignments are useful

The thesis should still acknowledge that:

- GMM slightly exceeds KMeans on raw matched `NMI` at some targets
- GMM is almost tied with KMeans in the production multivector tag-overlap metrics
- GMM offers a legitimate probabilistic alternative when soft membership matters

But it should also state that:

- GMM fails the explicit production stability gate in this rerun

### 12.5 Representation choice is as important as algorithm choice

This remains one of the strongest conclusions of the whole study.

Different evaluation questions favor different representations:

- HDBSCAN native discovery: `rhythm_only`, `pitch_rhythm`, and other compact coherent spaces
- matched geometry: `rhythm_only / pca_per_group_2`
- high-target parametric semantic alignment: `all_audio / raw_zscore`
- production recommendation rerun: `spectral_plus_beat / pca_per_group`

So the thesis should not frame the problem as algorithm-only. It should frame it as:

- how algorithm behavior changes with representation choice, coverage requirements, and evaluation regime

## 13. Why the New Conclusions Are More Defensible Than the Old Ones

The new conclusions are more defensible for five reasons.

1. No representation was promoted from a smaller exploratory screen.
2. The full downloaded-audio dataset was benchmarked directly after a transparent taxonomy reassignment
   step.
3. Non-genre-only songs were excluded explicitly instead of being forced into misleading primary genres.
4. Coverage and noise were reported and used in the interpretation of HDBSCAN.
5. The downstream evaluation now uses multivector primary-tag and all-tag overlap metrics rather than
   relying only on exact one-label agreement.

That is a stronger research posture than reporting only a single dramatic metric value on a noisier label
space.

## 14. Limitations and Threats to Validity

Even after removing the pilot and curating the taxonomy, the study still has important limitations.

### 14.1 The taxonomy is curated, not ground truth

The merged taxonomy is a defensible engineering choice, but it is still a human-designed label system.
Another curation scheme could move some songs between primary and secondary label sets.

### 14.2 Even `187` primary genres remain imperfect

The new label space is smaller and more coherent than the raw one, but `NMI` against those labels is still
not the same thing as proving musical or perceptual correctness.

### 14.3 Full-coverage and partial-coverage clusterings are inherently different

HDBSCAN can leave points unlabeled as noise, while KMeans and GMM cannot. That makes direct metric
comparison difficult unless coverage is reported alongside semantic scores.

### 14.4 Internal criteria still influence native selections

Native-mode conclusions depend on the internal selection rules used for each method. This is unavoidable,
but it should be stated transparently.

### 14.5 The study uses engineered audio descriptors, not deep embeddings

That is appropriate for the stated thesis topic, but the conclusions should not be generalized beyond this
feature family without further work.

### 14.6 No listening study was performed

The study is quantitative. It does not yet include human listening validation, playlist-quality judgment,
or blind preference testing.

## 15. Reports Removed or Superseded

Because the user explicitly asked for the thesis benchmark to stop depending on the pilot and then to move
to the merged taxonomy, two earlier narratives are now superseded as primary references:

- the earlier pilot-shaped thesis reports
- the March 17 raw-genre pilot-free benchmark bundles

They may still be useful for audit history, but the taxonomy-aware update in this file is the current
primary thesis reference.

## 16. Artifact Inventory

### 16.1 Main code artifacts

- `scripts/utilities/build_merged_genre_dataset.py`
- `src/utils/genre_taxonomy.py`
- `scripts/analysis/thesis_clustering_benchmark.py`
- `scripts/analysis/evaluate_clustering.py`

### 16.2 Derived taxonomy dataset

- `data/songs_with_merged_genres.csv`

Important added columns in that file include:

- `mapped_primary_genres`
- `mapped_secondary_tags`
- `mapped_all_tags`
- `include_in_mrs`
- `mrs_exclusion_reason`

### 16.3 Main thesis benchmark bundle

- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19`

Important files inside that bundle:

- `representation_catalog.csv`
- `full_grid_results.csv`
- `native_best_results.csv`
- `matched_granularity_results.csv`
- `global_native_leaders.csv`
- `global_matched_leaders.csv`
- `feature_group_variance_summary.csv`
- `feature_group_correlation_summary.csv`
- `dataset_summary.json`
- `benchmark_report.md`

### 16.4 Production clustering and evaluation bundle

- `output/experiment_runs_taxonomy/run_20260319_140333Z/recommended_production`

Important files inside that bundle:

- `clustering_results/audio_clustering_results_kmeans.csv`
- `clustering_results/audio_clustering_results_gmm.csv`
- `clustering_results/audio_clustering_results_hdbscan.csv`
- `metrics/recommended_production_comparison.csv`
- `metrics/recommended_production_comparison_report.md`
- `metrics/recommended_production_summary.json`

## 17. Sources Used

### 17.1 Local project sources

These were the direct local sources used for this taxonomy-aware thesis reconstruction:

- `data/acoustically_coherent_merged_genres_corrected.csv`
- `data/songs_with_merged_genres.csv`
- `scripts/utilities/build_merged_genre_dataset.py`
- `src/utils/genre_taxonomy.py`
- `src/utils/song_metadata.py`
- `src/utils/genre_mapper.py`
- `scripts/analysis/thesis_clustering_benchmark.py`
- `scripts/analysis/evaluate_clustering.py`
- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19/representation_catalog.csv`
- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19/full_grid_results.csv`
- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19/native_best_results.csv`
- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19/matched_granularity_results.csv`
- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19/global_native_leaders.csv`
- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19/global_matched_leaders.csv`
- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19/feature_group_variance_summary.csv`
- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19/feature_group_correlation_summary.csv`
- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19/dataset_summary.json`
- `output/metrics/thesis_benchmark_taxonomy_full_2026-03-19/benchmark_report.md`
- `output/experiment_runs_taxonomy/run_20260319_140333Z/recommended_production/metrics/recommended_production_comparison.csv`
- `output/experiment_runs_taxonomy/run_20260319_140333Z/recommended_production/metrics/recommended_production_comparison_report.md`
- `output/experiment_runs_taxonomy/run_20260319_140333Z/recommended_production/metrics/recommended_production_summary.json`

### 17.2 External scientific sources

Music Information Retrieval and dataset-evaluation context:

1. Tzanetakis, G., & Cook, P. (2002). *Musical Genre Classification of Audio Signals*.
   https://doi.org/10.1109/TSA.2002.800560
2. Aucouturier, J.-J., & Pachet, F. (2003). *Representing Musical Genre: A State of the Art*.
   https://www.francoispachet.fr/wp-content/uploads/2021/01/pachet-02c.pdf
3. Urbano, J., Schedl, M., Serra, X., & Gomez, E. (2014). *Evaluation in Music Information Retrieval*.
   https://julian-urbano.info/files/publications/051-evaluation-music-information-retrieval.pdf
4. Flexer, A. (2007). *A Closer Look on Artist Filters for Musical Genre Classification*.
   https://dblp.org/rec/conf/ismir/Flexer07
5. Bogdanov, D., Won, M., Tovstogan, P., Porter, A., Serra, X., & Herrera, P. (2019). *The MTG-Jamendo
   Dataset for Automatic Music Tagging*. https://transactions.ismir.net/articles/10.5334/tismir.16
6. Sturm, B. L. (2021). *Characterising Confounding Effects in Music Classification Experiments through
   Interventions*. https://transactions.ismir.net/articles/10.5334/tismir.24

Clustering algorithms and validation:

7. MacQueen, J. (1967). *Some Methods for classification and Analysis of Multivariate Observations*.
   https://digicoll.lib.berkeley.edu/record/113015?v=pdf
8. Fraley, C., & Raftery, A. E. (1998). *How Many Clusters? Which Clustering Method? Answers Via
   Model-Based Cluster Analysis*. https://sites.stat.washington.edu/raftery/Research/PDF/fraley1998.pdf
9. Fraley, C., & Raftery, A. E. (2002). *Model-Based Clustering, Discriminant Analysis, and Density
   Estimation*. https://sites.stat.washington.edu/raftery/Research/PDF/fraley2002.pdf
10. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A Density-Based Algorithm for Discovering
    Clusters in Large Spatial Databases with Noise*.
    https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/
11. McInnes, L., Healy, J., & Astels, S. (2017). *hdbscan: Hierarchical density based clustering*.
    https://joss.theoj.org/papers/10.21105/joss.00205
12. Halkidi, M., Batistakis, Y., & Vazirgiannis, M. (2001). *On Clustering Validation Techniques*.
    https://doi.org/10.1023/A:1012801612483
13. Rousseeuw, P. J. (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of
    Cluster Analysis*. https://doi.org/10.1016/0377-0427(87)90125-7
14. Rosenberg, A., & Hirschberg, J. (2007). *V-Measure: A conditional entropy-based external cluster
    evaluation measure*. https://aclanthology.org/D07-1043/
15. Lange, T., Roth, V., Braun, M., & Buhmann, J. M. (2004). *Stability-based validation of clustering
    solutions*. https://pubmed.ncbi.nlm.nih.gov/15130251/
16. von Luxburg, U. (2010). *Clustering Stability: An Overview*. https://arxiv.org/abs/1007.1075
17. von Luxburg, U., Williamson, R. C., & Guyon, I. (2012). *Clustering: Science or Art?*
    https://proceedings.mlr.press/v27/luxburg12a/luxburg12a.pdf

## 18. Final Bottom Line

The taxonomy-aware pilot-free rerun changed the empirical story again, and it did so in a more defensible
way than either the earlier pilot-shaped flow or the raw-genre pilot-free rerun.

The strongest concise conclusion I would now stand behind is:

- HDBSCAN is the strongest method for native fine-grained unsupervised discovery
- KMeans is the strongest balanced method for matched medium-granularity clustering and for the current
  production recommendation profile
- GMM is a very close second and a legitimate soft-clustering alternative, but it still has stability
  concerns
- secondary `non_genre_*` tags should remain contextual descriptors rather than stand-alone primary genres
- representation choice and evaluation regime matter at least as much as algorithm choice

That is the thesis answer supported by the taxonomy-aware benchmark and production rerun.
