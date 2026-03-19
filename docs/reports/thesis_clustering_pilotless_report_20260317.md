# Pilot-Free Thesis Clustering Report

## 1. Purpose of This Report

This report replaces the earlier pilot-shaped thesis narrative with a new full-space benchmark that does
not use any shortlist, subset-stage promotion logic, or pilot-driven representation selection.

The work in this report was done in direct response to the user requirement that the thesis experiment be
rebuilt without being biased by the earlier pilot. The benchmark described here therefore treats the
thesis experiment itself as the main study.

Everything in this report is based on:

- a prespecified representation space derived from standard Music Information Retrieval feature families
- full-dataset evaluation on the current repository dataset
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

The pilot-free thesis benchmark evaluated `14` prespecified audio representations across `3`
preprocessing modes and `3` clustering algorithms on the full dataset of `5,535` songs. This produced
`42` representation/preprocessing scenarios and `3,738` fitted clustering models. The results show that
there is still no single universal winner. KMeans and GMM usually collapse to very coarse native
solutions, while HDBSCAN is the only method that consistently discovers rich fine-grained native
structure. The strongest practical native result in the new benchmark is `HDBSCAN + rhythm_only +
pca_per_group_2`, which produced `576` clusters with only `8.3%` noise, `silhouette=0.566`,
`NMI=0.547`, and `stability_ari=0.875`. The highest raw NMI overall was
`HDBSCAN + delta2_mfcc_only + pca_per_group_5` with `NMI=0.588`, but that solution marked almost
`60%` of the dataset as noise, so it is not the most defensible headline result. Under matched
medium-granularity comparison, KMeans was the strongest balanced method overall, GMM was competitive and
occasionally slightly stronger on NMI at high target counts, and HDBSCAN produced some very high-NMI
matched candidates only by discarding most songs as noise. The thesis conclusion is therefore conditional:
HDBSCAN is strongest for native fine-grained discovery, while KMeans is the strongest balanced method for
medium-granularity full-coverage clustering.

### 2.2 High-level findings

1. Removing the pilot changed the story materially.
2. The new full-space benchmark exposed strong representations that the pilot flow did not privilege,
   especially `rhythm_only`, `pitch_only`, and `mfcc_only`.
3. HDBSCAN is still the most interesting method scientifically, but not for the same reason as before.
4. The old pilot narrative around `delta2_mfcc` remains partly true, but it is no longer the only or
   clearly dominant story once the search space is widened and no pilot promotion is used.
5. KMeans remains a very strong benchmark method when the comparison is made at matched cluster
   granularity.
6. GMM remains a legitimate soft-clustering alternative and slightly exceeds KMeans on NMI in some
   high-target full-audio settings.
7. Raw best-score claims can be misleading if coverage and noise are ignored, especially for HDBSCAN.

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

The benchmark was run on the full current dataset:

- songs: `5,535`
- unique primary genres: `419`
- unique artists: `3,094`
- raw audio feature dimensions: `116`

No subset-stage shortlist was used.

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

The search space for this dataset size (`5,535` songs) is generated from:

- `min_cluster_size` values: `5`, `8`, `10`, `15`, `25`, `111`, `221`
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

This is normalized mutual information between cluster labels and primary genre labels.

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
- it evaluates the full prespecified representation space directly
- it writes a `representation_catalog.csv`
- it writes full-grid, per-scenario native, per-scenario matched, and global-leader CSVs

The key output bundle from the final run is:

- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17`

The smoke-validation bundle is:

- `output/metrics/thesis_benchmark_pilotless_smoke_2026-03-17`

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
- primary genre labels
- genre lists
- normalized artist keys

Even though the pilot-free benchmark does not use artist-aware shortlisting anymore, aligned metadata still
matters because:

- genre labels are needed for external evaluation metrics
- artist and genre counts are part of dataset characterization
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

## 7.1 Smoke run

Purpose:

- validate the new pilot-free code path on the full dataset before the expensive full run

Command logic:

- reduced `max_k` and `max_components` to `6`
- matched targets reduced to `4` and `8`
- kept the full representation space

Observed runtime:

- approximately `3,811.3` seconds
- approximately `63.5` minutes

Outcome:

- completed successfully
- produced the expected artifact set
- confirmed that the new representation catalog and direct full-space evaluation path worked

## 7.2 Full run

The full benchmark was then executed with the full target list and full KMeans/GMM parameter ranges.

Observed runtime:

- approximately `5,508.6` seconds
- approximately `91.8` minutes

Final bundle:

- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17`

### 7.3 Exact command used for the main benchmark

The pilot-free full run was executed with:

```powershell
.venv\Scripts\python.exe scripts\analysis\thesis_clustering_benchmark.py `
  --output-dir output\metrics\thesis_benchmark_pilotless_full_2026-03-17
```

The benchmark used the script defaults for:

- matched targets: `4 8 12 16 20`
- matched band fraction: `0.25`
- KMeans maximum `k`: `20`
- GMM maximum components: `20`

### 7.4 Independent revalidation run

After the first full pilot-free report was written, the benchmark was run again independently in order to
check whether the thesis conclusions were stable.

Revalidation bundle:

- `output/metrics/thesis_benchmark_pilotless_revalidation_2026-03-17`

Command:

```powershell
.venv\Scripts\python.exe scripts\analysis\thesis_clustering_benchmark.py `
  --output-dir output\metrics\thesis_benchmark_pilotless_revalidation_2026-03-17
```

Observed runtime:

- approximately `5,492.7` seconds
- approximately `91.5` minutes

Key revalidation finding:

- the substantive conclusions held

What remained exactly the same:

- representation catalog
- variance summary
- correlation summary
- global native leader identities
- the top-10 NMI rows in the full grid

What changed slightly:

- fit times
- timestamps
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

| Method | Representation | Preprocess | Clusters | Silhouette | NMI | Stability |
|---|---|---|---:|---:|---:|---:|
| KMeans | `mfcc_only` | `raw_zscore` | `2` | `0.193` | `0.047` | `1.000` |
| GMM | `mfcc_only` | `pca_per_group_2` | `3` | `0.354` | `0.081` | `0.988` |
| HDBSCAN | `rhythm_only` | `pca_per_group_2` | `576` | `0.566` | `0.547` | `0.875` |

This already shows that removing the pilot changed the thesis story in a major way.

Under the new design:

- the strongest practical native HDBSCAN result is no longer the earlier `delta2_mfcc` story
- it is `rhythm_only + pca_per_group_2`

That result also has strong practicality diagnostics:

- coverage: `0.917`
- noise fraction: `0.083`
- largest cluster fraction: `0.0055`

This is a much stronger practical native result than a high-NMI solution that leaves most of the dataset
unassigned.

## 10.2 Highest raw NMI overall

The highest raw NMI in the entire full grid was:

- method: `HDBSCAN`
- representation: `delta2_mfcc_only`
- preprocess: `pca_per_group_5`
- parameters: `min_cluster_size=5`, `min_samples=1`
- clusters: `281`
- silhouette: `0.215`
- NMI: `0.588`
- noise fraction: `0.600`

This is scientifically interesting, but it is not the best headline conclusion for the thesis because:

- almost `60%` of songs were labeled as noise
- its native stability was only `0.293`
- the solution is much less practical as a full-collection clustering than the HDBSCAN `rhythm_only`
  native leader

The key research lesson is:

- highest raw semantic score is not automatically the best scientific conclusion

## 10.3 Native cluster-count behavior by method

The native cluster-count summaries reveal a very strong method effect.

### KMeans

- native selections with more than 2 clusters: `2` out of `42`
- maximum native cluster count: `3`

Interpretation:

- KMeans almost always collapsed to a very coarse 2-cluster solution when allowed to optimize only its
  internal criteria

### GMM

- native selections with more than 2 clusters: `9` out of `42`
- maximum native cluster count: `12`

Interpretation:

- GMM was somewhat more willing than KMeans to accept richer native structure
- but it still mostly preferred very coarse partitions

### HDBSCAN

- native selections with more than 2 clusters: `8` out of `42`
- maximum native cluster count: `576`

Interpretation:

- HDBSCAN is the only method that repeatedly discovered rich native fine-grained structure
- but it also produced many 2-cluster native solutions on other representations

This means the thesis should not say:

- "HDBSCAN always finds many clusters"

It should say:

- "HDBSCAN is the only method in this study that can discover very rich native structure, but whether it
  does so depends strongly on the representation."

## 10.4 Matched-granularity leaders selected by internal criteria

The benchmark also computed global matched-granularity leaders by method and target, using the benchmark's
matched selection logic rather than raw NMI alone.

| Target | Method | Representation | Preprocess | Clusters | Silhouette | NMI | Stability |
|---:|---|---|---|---:|---:|---:|---:|
| 4 | KMeans | `pitch_rhythm` | `pca_per_group_2` | `4` | `0.237` | `0.118` | `0.987` |
| 4 | GMM | `mfcc_only` | `pca_per_group_2` | `4` | `0.357` | `0.090` | `0.845` |
| 4 | HDBSCAN | `spectral_pitch` | `pca_per_group_2` | `4` | `0.162` | `0.006` | `1.000` |
| 8 | KMeans | `pitch_rhythm` | `pca_per_group_5` | `8` | `0.149` | `0.148` | `0.925` |
| 8 | GMM | `pitch_rhythm` | `pca_per_group_2` | `8` | `0.185` | `0.153` | `0.608` |
| 8 | HDBSCAN | `spectral_rhythm` | `pca_per_group_2` | `8` | `0.146` | `0.015` | `1.000` |
| 12 | KMeans | `pitch_only` | `pca_per_group_2` | `12` | `0.335` | `0.160` | `0.942` |
| 12 | GMM | `rhythm_only` | `pca_per_group_2` | `12` | `0.342` | `0.144` | `0.732` |
| 12 | HDBSCAN | `rhythm_only` | `pca_per_group_2` | `12` | `-0.015` | `0.115` | `0.953` |
| 16 | KMeans | `pitch_only` | `pca_per_group_2` | `16` | `0.337` | `0.180` | `0.907` |
| 16 | GMM | `rhythm_only` | `pca_per_group_2` | `16` | `0.328` | `0.167` | `0.551` |
| 16 | HDBSCAN | `rhythm_only` | `raw_zscore` | `15` | `0.002` | `0.186` | `0.819` |
| 20 | KMeans | `rhythm_only` | `pca_per_group_2` | `20` | `0.337` | `0.183` | `0.582` |
| 20 | GMM | `mfcc_only` | `pca_per_group_2` | `20` | `0.315` | `0.200` | `0.516` |
| 20 | HDBSCAN | `rhythm_only` | `pca_per_group_2` | `20` | `-0.208` | `0.158` | `0.944` |

These internal-selection matched leaders support three important conclusions.

1. KMeans is the strongest balanced method overall.
2. GMM is competitive, especially at target `8` and target `20`.
3. HDBSCAN is not the strongest balanced medium-granularity method even though it is the strongest
   native fine-grained discovery method.

## 10.5 Matched semantic leaders and why they must be read carefully

If we ignore practicality and simply ask for the highest NMI row inside each matched-target/method group,
the story changes.

### Best matched NMI by target and method

| Target | Method | Representation | Preprocess | Clusters | Gap | Coverage | NMI | Silhouette | Stability |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 | KMeans | `all_audio` | `pca_per_group_2` | `4` | `0` | `1.000` | `0.128` | `0.179` | `0.996` |
| 4 | GMM | `all_audio` | `raw_zscore` | `4` | `0` | `1.000` | `0.132` | `0.047` | `0.981` |
| 4 | HDBSCAN | `pitch_rhythm` | `raw_zscore` | `4` | `0` | `0.020` | `0.416` | `0.289` | `0.716` |
| 8 | KMeans | `timbre_pitch` | `raw_zscore` | `8` | `0` | `1.000` | `0.175` | `0.027` | `0.928` |
| 8 | GMM | `timbre_pitch` | `raw_zscore` | `8` | `0` | `1.000` | `0.175` | `0.020` | `0.454` |
| 8 | HDBSCAN | `spectral_pitch` | `raw_zscore` | `10` | `2` | `0.023` | `0.585` | `0.175` | `0.268` |
| 12 | KMeans | `timbre_pitch_rhythm` | `raw_zscore` | `12` | `0` | `1.000` | `0.199` | `0.024` | `0.765` |
| 12 | GMM | `timbre_pitch_rhythm` | `pca_per_group_5` | `12` | `0` | `1.000` | `0.199` | `0.029` | `0.618` |
| 12 | HDBSCAN | `spectral_pitch` | `raw_zscore` | `10` | `2` | `0.023` | `0.585` | `0.175` | `0.268` |
| 16 | KMeans | `all_audio` | `raw_zscore` | `16` | `0` | `1.000` | `0.225` | `0.024` | `0.542` |
| 16 | GMM | `all_audio` | `raw_zscore` | `16` | `0` | `1.000` | `0.222` | `0.016` | `0.387` |
| 16 | HDBSCAN | `rhythm_only` | `raw_zscore` | `15` | `1` | `0.433` | `0.186` | `0.002` | `0.819` |
| 20 | KMeans | `all_audio` | `raw_zscore` | `20` | `0` | `1.000` | `0.234` | `0.020` | `0.400` |
| 20 | GMM | `all_audio` | `raw_zscore` | `20` | `0` | `1.000` | `0.243` | `-0.010` | `0.532` |
| 20 | HDBSCAN | `rhythm_only` | `raw_zscore` | `19` | `1` | `0.706` | `0.164` | `-0.087` | `0.884` |

This table is crucial for honest interpretation.

It shows that some of the most spectacular HDBSCAN matched NMI values at low and medium targets are not
full-collection solutions at all. They are partial-coverage solutions:

- target `4`, `pitch_rhythm / raw_zscore`: coverage `0.020`
- target `8`, `spectral_pitch / raw_zscore`: coverage `0.023`
- target `12`, `spectral_pitch / raw_zscore`: coverage `0.023`

In other words:

- those solutions discard roughly `98%` of the dataset as noise

That makes them very interesting as density discoveries, but weak as full-dataset medium-granularity
clusterings.

This is one of the most important scientific outcomes of the whole study:

- if coverage is ignored, HDBSCAN appears to dominate matched semantic alignment at low and medium targets
- once coverage is taken seriously, that dominance is much less convincing

## 10.6 Direct KMeans vs GMM comparison on the full audio stack

Because `all_audio / raw_zscore` emerged as the strongest parametric semantic representation at higher
matched targets, it is useful to compare KMeans and GMM directly on that exact feature space.

| Target | KMeans NMI | KMeans Silhouette | GMM NMI | GMM Silhouette |
|---:|---:|---:|---:|---:|
| 4 | `0.127` | `0.083` | `0.128` | `0.068` |
| 8 | `0.170` | `0.041` | `0.170` | `0.020` |
| 12 | `0.199` | `0.031` | `0.200` | `-0.002` |
| 16 | `0.225` | `0.024` | `0.223` | `-0.007` |
| 20 | `0.234` | `0.020` | `0.243` | `-0.010` |

Interpretation:

- at low target counts the two methods are very close
- as target count grows, GMM slightly exceeds KMeans on NMI
- but GMM loses geometric compactness faster and becomes negative on silhouette at `12+`
- KMeans retains positive silhouette throughout

So the fairest reading is:

- GMM can edge KMeans on semantic alignment in richer high-target parametric settings
- KMeans remains more geometrically coherent in those same settings

## 10.7 Geometry-focused matched leaders

If the target is compact separation rather than label alignment, KMeans dominates the matched comparison
from target `8` onward.

Global best matched silhouette at each target was:

| Target | Best row |
|---:|---|
| 4 | `HDBSCAN + pitch_only + pca_per_group_5` with silhouette `0.371` but only approximate target match |
| 8 | `KMeans + rhythm_only + pca_per_group_2`, silhouette `0.336` |
| 12 | `KMeans + rhythm_only + pca_per_group_2`, silhouette `0.344` |
| 16 | `KMeans + rhythm_only + pca_per_group_2`, silhouette `0.338` |
| 20 | `KMeans + rhythm_only + pca_per_group_2`, silhouette `0.337` |

This reveals a very strong rhythmic geometry effect:

- `rhythm_only + pca_per_group_2` is repeatedly the cleanest medium-granularity partition for KMeans

That does not mean rhythm-only is the universally best representation. It means:

- the beat-strength family organizes the collection very cleanly for centroid-based clustering

## 10.8 Computational cost

Full-run fit-time totals were:

| Method | Fits | Total fit time (s) | Mean fit time (s) | Median fit time (s) | Max fit time (s) |
|---|---:|---:|---:|---:|---:|
| KMeans | `798` | `298.782` | `0.374` | `0.379` | `1.574` |
| GMM | `1,596` | `798.562` | `0.500` | `0.109` | `15.137` |
| HDBSCAN | `1,344` | `1,238.071` | `0.921` | `0.482` | `5.508` |

Interpretation:

- KMeans was the cheapest method on mean time
- GMM had a low median fit time but a heavy tail due to expensive configurations
- HDBSCAN was the most expensive method overall

This matters because thesis-level "performance" should not be interpreted only as final NMI.

Under a broader performance view:

- KMeans is strong on quality/cost balance
- GMM is a reasonable tradeoff when soft assignments are valuable
- HDBSCAN earns its cost by discovering structures that the parametric methods do not, not by being cheap

## 10.9 Representation and preprocessing patterns

Several representation-level patterns were stable across the benchmark.

### Pattern 1: HDBSCAN likes compact or low-dimensional structured spaces

The top raw-NMI representations for HDBSCAN were:

- `delta2_mfcc_only / pca_per_group_5`
- `delta2_mfcc_only / pca_per_group_2`
- `rhythm_only / pca_per_group_2`
- `delta_mfcc_only / pca_per_group_2`
- `mfcc_only / pca_per_group_2`
- `pitch_only / pca_per_group_2`

This suggests that density discovery is strongest when the representation is:

- focused
- low- to moderate-dimensional
- structurally coherent

### Pattern 2: All-audio raw stacks help parametric semantic alignment at higher target counts

The top KMeans and GMM NMI rows at higher matched targets were on:

- `all_audio / raw_zscore`

This means the richer mixed-family representation was helpful for parametric semantic alignment once the
 algorithms were forced to produce more than a handful of clusters.

### Pattern 3: `pca_per_group_2` was especially strong for clean rhythmic organization

The strongest matched-geometry KMeans results repeatedly used:

- `rhythm_only / pca_per_group_2`

This suggests that light grouped compression sharpened the rhythmic structure rather than destroying it.

### Pattern 4: `pca_per_group_5` helped some fine-grained HDBSCAN timbral discovery, but with cost

The strongest raw HDBSCAN NMI used:

- `delta2_mfcc_only / pca_per_group_5`

But that came with:

- `59.96%` noise
- weaker stability than the practical `rhythm_only / pca_per_group_2` native leader

So `pca_per_group_5` can help fine-grained density separation, but not always in the most practical way.

## 11. What Changed Relative to the Earlier Pilot-Shaped Story

Removing the pilot did not merely make the study cleaner. It changed the empirical conclusions.

### 11.1 The earlier `delta2_mfcc` HDBSCAN story was only part of the picture

The earlier pilot-based thesis flow emphasized:

- `HDBSCAN + delta2_mfcc + grouped PCA`

That phenomenon still exists in the pilot-free full benchmark:

- `delta2_mfcc_only / pca_per_group_5` achieved the highest raw NMI overall
- `delta2_mfcc_only / pca_per_group_2` remained a very strong practical density result

But the widened unbiased search space also showed something the pilot-shaped path underemphasized:

- `rhythm_only / pca_per_group_2` is an even stronger practical native HDBSCAN solution

### 11.2 The parametric medium-granularity story changed too

The pilot-based flow had favored selected spectral/chroma/beat representations. In the new study, the
strongest medium-granularity semantic parametric solutions came from:

- `all_audio / raw_zscore`

That means the thesis should not keep the earlier narrower representation story.

### 11.3 Low-dimensional families mattered more than expected

The pilot-free benchmark showed unexpectedly strong behavior from:

- `rhythm_only`
- `pitch_only`
- `mfcc_only`

These would have been easy to underemphasize in a pilot-promoted process.

That is exactly why the user was right to ask for the pilot to be removed from the thesis flow.

## 12. Final Answer to the Thesis Question

The thesis subject is:

> A study and comparison of the performance of machine learning algorithms for clustering music tracks
> based on their audio features.

The most defensible answer, based only on this pilot-free benchmark, is the following.

### 12.1 There is no single absolute best algorithm

The benchmark does not support a simple statement like:

- "KMeans is best"
- "GMM is best"
- "HDBSCAN is best"

The result depends on what "performance" means.

### 12.2 If performance means native fine-grained unsupervised discovery, HDBSCAN is strongest

HDBSCAN is the only method that repeatedly discovered rich native fine-grained structure.

The strongest practical native result was:

- `HDBSCAN + rhythm_only + pca_per_group_2`
- `576` clusters
- `8.3%` noise
- `silhouette=0.566`
- `NMI=0.547`
- `stability_ari=0.875`

The highest raw NMI overall was also HDBSCAN:

- `HDBSCAN + delta2_mfcc_only + pca_per_group_5`
- `NMI=0.588`

But that solution is not the best thesis headline because of its very high noise rate.

### 12.3 If performance means balanced medium-granularity clustering, KMeans is strongest overall

Under matched-granularity comparison:

- KMeans was the most balanced method overall
- it combined strong or near-strong NMI with consistently better silhouette and stability than GMM in
  many target regimes
- it avoided the extreme coverage problems seen in some HDBSCAN matched semantic leaders

### 12.4 If performance means highest semantic alignment for high-target parametric models, GMM can win

On `all_audio / raw_zscore` at `20` components:

- GMM reached `NMI=0.243`
- KMeans reached `NMI=0.234`

So the thesis should acknowledge that:

- GMM can slightly outperform KMeans on semantic alignment in some richer high-target parametric settings

### 12.5 Representation choice is as important as algorithm choice

This may be the most important scientific conclusion of the whole study.

Different questions favored different representations:

- HDBSCAN native discovery: `rhythm_only`, `delta2_mfcc_only`, `delta_mfcc_only`, `mfcc_only`
- matched geometry: `rhythm_only / pca_per_group_2`
- higher-target parametric semantic alignment: `all_audio / raw_zscore`

That means the thesis should not frame the question as algorithm-only.

It should frame it as:

- how algorithm performance changes with the audio representation and evaluation regime

## 13. Why the New Conclusions Are More Defensible Than the Old Ones

The new pilot-free conclusions are more defensible for four reasons.

1. No representation was promoted from a smaller exploratory screen.
2. The full dataset was used directly for the main benchmark.
3. Coverage and noise were explicitly considered when interpreting HDBSCAN.
4. The report distinguishes raw score maxima from practical thesis conclusions.

That is a better research posture than reporting only the single most dramatic metric value.

## 14. Limitations and Threats to Validity

Even after removing the pilot, the study still has important limitations.

### 14.1 Genre labels are imperfect

`419` genres is a large and potentially noisy label space. NMI against those labels is useful, but it is
not the same thing as proving musical or perceptual correctness.

### 14.2 Full-coverage and partial-coverage clusterings are inherently different

HDBSCAN can leave points unlabeled as noise, while KMeans and GMM cannot. That makes direct metric
comparison difficult unless coverage is reported alongside semantic scores.

### 14.3 Internal criteria still influence native selections

Native-mode conclusions depend on the internal selection rules used for each method. This is unavoidable,
but it should be stated transparently.

### 14.4 The study uses engineered audio descriptors, not deep embeddings

That is appropriate for the stated thesis topic, but the conclusions should not be generalized beyond this
feature family without further work.

### 14.5 No listening study was performed

The study is quantitative. It does not yet include human listening validation or playlist-quality
judgment.

## 15. Reports Removed or Superseded

Because the user explicitly asked that the thesis process stop depending on the pilot, earlier
pilot-shaped thesis reports were treated as superseded material.

The new report in this file is intended to replace them as the main thesis reference.

## 16. Artifact Inventory

### 16.1 Main code artifact

- `scripts/analysis/thesis_clustering_benchmark.py`

### 16.2 Main output bundle

- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17`

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

### 16.3 Smoke-validation bundle

- `output/metrics/thesis_benchmark_pilotless_smoke_2026-03-17`

### 16.4 Revalidation bundle

- `output/metrics/thesis_benchmark_pilotless_revalidation_2026-03-17`

This bundle exists so the main thesis result is not tied to a single benchmark execution.

## 17. Sources Used

### 17.1 Local project sources

These were the direct local sources used for the new thesis reconstruction:

- `scripts/analysis/thesis_clustering_benchmark.py`
- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17/representation_catalog.csv`
- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17/full_grid_results.csv`
- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17/native_best_results.csv`
- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17/matched_granularity_results.csv`
- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17/global_native_leaders.csv`
- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17/global_matched_leaders.csv`
- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17/feature_group_variance_summary.csv`
- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17/feature_group_correlation_summary.csv`
- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17/dataset_summary.json`
- `output/metrics/thesis_benchmark_pilotless_full_2026-03-17/benchmark_report.md`
- `output/metrics/thesis_benchmark_pilotless_smoke_2026-03-17/benchmark_report.md`
- `output/metrics/thesis_benchmark_pilotless_revalidation_2026-03-17/full_grid_results.csv`
- `output/metrics/thesis_benchmark_pilotless_revalidation_2026-03-17/native_best_results.csv`
- `output/metrics/thesis_benchmark_pilotless_revalidation_2026-03-17/matched_granularity_results.csv`
- `output/metrics/thesis_benchmark_pilotless_revalidation_2026-03-17/global_native_leaders.csv`
- `output/metrics/thesis_benchmark_pilotless_revalidation_2026-03-17/global_matched_leaders.csv`
- `output/metrics/thesis_benchmark_pilotless_revalidation_2026-03-17/benchmark_report.md`

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

The pilot-free thesis reconstruction changed the empirical story in a meaningful way, which confirms that
removing the pilot was the right methodological decision.

The strongest concise conclusion I would now stand behind is:

- HDBSCAN is the strongest method for native fine-grained unsupervised discovery
- KMeans is the strongest balanced method for matched medium-granularity clustering
- GMM is a competitive soft-clustering alternative and slightly stronger than KMeans on some high-target
  semantic comparisons
- representation choice is at least as important as algorithm choice
- the earlier pilot should not be used as the main thesis benchmark anymore

That is the thesis answer supported by the new pilot-free benchmark.
