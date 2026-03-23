# Thesis Benchmark Conclusions

Date: 2026-03-22

## Purpose

This note summarizes the main conclusions I would carry into the thesis based on the benchmark artifacts currently present in the repository.

It separates two evaluation contexts so they do not get mixed together:

1. the main thesis benchmark report
2. the later production-profile comparison artifact

That distinction matters because they were run in different local data states and should not be treated as one merged result.

## Benchmark contexts used

### 1. Main thesis benchmark

Primary source:

- `docs/reports/thesis_clustering_pilotless_report_20260317.md`

This is the main thesis-grade benchmark narrative. It reports:

- taxonomy-aware evaluation
- `5,456` audio-backed songs retained from `5,535` downloaded tracks
- `187` primary taxonomy genres
- `14` prespecified audio representations
- `3` preprocessing modes
- `3` clustering algorithms
- `3,738` fitted models

### 2. Later production-profile rerun

Consistency-check sources:

- `output/metrics/recommended_production_summary.json`
- `output/metrics/recommended_production_comparison_report.md`
- `docs/reports/explicit_decisions_report_20260315.md`

This later artifact reflects the current product-oriented profile in the present local state and evaluates a smaller current dataset slice:

- `849` loaded audio-backed tracks
- active profile: `spectral_plus_beat`
- active methods compared: `KMeans`, `GMM`, `HDBSCAN`

I use this second context as a product-facing cross-check, not as a replacement for the thesis benchmark.

## Main conclusion for the thesis

There is not one universal best clustering method in this project.

The strongest conclusion is:

- if the thesis question is native unsupervised discovery of fine-grained musical structure, `HDBSCAN` is the strongest method
- if the thesis question is balanced medium-granularity clustering with full coverage and cleaner recommendation behavior, `KMeans` is the strongest overall method
- `GMM` is the closest competitor to `KMeans` and is the best soft-clustering alternative, but it is less stable

That is the most defensible high-level thesis claim supported by the current benchmark evidence.

## What the main thesis benchmark shows

### 1. HDBSCAN is strongest for native fine-grained discovery

The strongest practical native result in the thesis benchmark is:

- method: `HDBSCAN`
- representation: `rhythm_only`
- preprocessing: `pca_per_group_2`
- clusters: `563`
- coverage: `91.5%`
- noise: `8.5%`
- silhouette: `0.569`
- NMI: `0.453`
- stability ARI: `0.887`

This is the clearest evidence that density-based clustering is the most interesting method when the goal is to let the data reveal detailed structure without forcing every item into a parametric partition.

### 2. The single highest raw semantic score is also HDBSCAN, but it is not the best headline result

The highest raw NMI anywhere in the thesis benchmark is:

- method: `HDBSCAN`
- representation: `pitch_rhythm`
- preprocessing: `pca_per_group_2`
- clusters: `333`
- coverage: `54.0%`
- noise: `46.0%`
- silhouette: `0.220`
- NMI: `0.483`

This result is scientifically interesting, but it is not the best thesis headline because it discards almost half the collection as noise.

So the thesis should not say only "the best result was the highest NMI." It should say that coverage and interpretability matter alongside semantic alignment.

### 3. KMeans is strongest for balanced medium-granularity clustering

Under matched-granularity comparison and downstream recommendation-oriented evaluation, the thesis report concludes that `KMeans` is the strongest balanced method overall.

Why:

- it stays full-coverage
- it gives stable medium-granularity partitions
- it remains competitive on semantic alignment
- it avoids the severe coverage and dominance problems that appear with `HDBSCAN` in product-style settings

This is the best method to emphasize if your thesis wants a practical clustering baseline rather than only a discovery-oriented one.

### 4. GMM is a serious competitor, but stability is the main weakness

The thesis benchmark shows that `GMM`:

- is highly competitive with `KMeans`
- sometimes slightly exceeds `KMeans` on raw matched `NMI` at some target cluster counts
- is attractive when soft membership matters

But it also shows that:

- `GMM` is less stable than `KMeans`
- this weaker stability is the main reason it is not the strongest overall balanced conclusion

So the correct thesis framing is not that `GMM` failed completely. The correct framing is that it is a credible soft-clustering alternative whose main tradeoff is stability.

### 5. Representation choice matters as much as algorithm choice

This is one of the strongest study-wide conclusions.

Different benchmark goals favored different feature spaces:

- native HDBSCAN discovery favored compact rhythmic and rhythm-plus-pitch spaces
- high-target parametric semantic alignment favored `all_audio / raw_zscore`
- the current product profile uses `spectral_plus_beat / pca_per_group`

So the thesis should not present the problem as only "which algorithm is best?"

A better statement is:

- clustering behavior depends jointly on algorithm, representation, coverage requirements, and evaluation regime

## What the later production-profile rerun confirms

The later current production-profile comparison on the smaller `849`-track local state ranks:

1. `KMeans`
2. `GMM`
3. `HDBSCAN`

At `K=10`, the current comparison report shows:

- `KMeans`: `PrimaryTagJaccard@10 = 0.0646`, `AllTagJaccard@10 = 0.0630`, `SubsampleMedianARI = 0.9418`
- `GMM`: `PrimaryTagJaccard@10 = 0.0645`, `AllTagJaccard@10 = 0.0634`, `SubsampleMedianARI = 0.7601`
- `HDBSCAN`: `PrimaryTagJaccard@10 = 0.0624`, `AllTagJaccard@10 = 0.0609`, `SubsampleMedianARI = 0.2270`

Interpretation:

- `KMeans` and `GMM` are very close semantically
- `KMeans` ranks first because its stability is much stronger
- `HDBSCAN` is clearly weaker in the current production-style profile

This later rerun is consistent with the main thesis story:

- `HDBSCAN` is the interesting native discovery method
- `KMeans` is the strongest balanced practical method
- `GMM` is close, but less stable

## Recommended thesis wording

If you want a short conclusion paragraph for the thesis, this is a defensible version:

The benchmark results do not support a single universally best clustering algorithm. Instead, performance depends on the evaluation objective. For native unsupervised discovery of fine-grained structure, HDBSCAN produced the strongest practical result, especially with compact rhythm-oriented representations. For balanced medium-granularity clustering and recommendation-oriented use, KMeans was the strongest overall method because it combined full coverage, competitive semantic alignment, and clearly stronger stability. GMM was the closest competitor and remained valuable when soft assignments were desired, but its weaker stability prevented it from becoming the strongest overall practical choice.

## What I would claim explicitly in the thesis

Safe claims:

- the thesis benchmark favors `HDBSCAN` for discovery-oriented analysis
- the thesis benchmark favors `KMeans` for balanced practical clustering
- `GMM` is competitive but less stable
- representation choice is a first-class experimental factor, not a secondary detail

Claims I would avoid:

- "HDBSCAN is simply the best overall"
- "KMeans is best in every setting"
- "GMM is worse in every meaningful way"
- "the highest NMI result is automatically the best thesis result"

## Limitations that should stay in the thesis

The benchmark evidence is still limited by:

- taxonomy-based evaluation rather than perceptual ground truth
- partial comparability between full-coverage and noise-allowing methods
- engineered handcrafted features rather than deep embeddings
- no listening study or human preference validation

These limitations do not invalidate the results, but they should remain explicit in the final thesis.

## Source files

- `docs/reports/thesis_clustering_pilotless_report_20260317.md`
- `output/metrics/recommended_production_summary.json`
- `output/metrics/recommended_production_comparison_report.md`
- `docs/reports/explicit_decisions_report_20260315.md`
