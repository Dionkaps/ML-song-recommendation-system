# P1 Evaluation Upgrades Report

Generated: 2026-03-14

## Request handled

This report documents the implementation of:

- `## P1 - Evaluation upgrades`

from:

- `docs/reports/implementation_todo_20260314.md`

## Executive summary

The evaluation layer has now been rebuilt around the actual saved clustering artifacts and the actual recommendation behavior of the product.

This phase introduced a new `scripts/analysis/evaluate_clustering.py` that:

- evaluates recommendations in the true prepared feature space
- mirrors the UI's cluster-constrained retrieval behavior
- computes per-query and averaged `Precision@K` and hit-rate style metrics
- computes genre and artist proxy metrics
- adds coverage and exposure-diversity checks
- treats silhouette / Calinski-Harabasz / Davies-Bouldin as diagnostics only
- runs a stronger stability protocol with `50` subsample runs at `80%`
- adds repeated-seed checks for seed-based models
- adds per-cluster stability reporting
- emits one final comparison report ranked by recommendation quality first and stability second

The most important outcome from the full-strength evaluation is:

- `K-Means` ranked `#1`
- `GMM` ranked `#2`
- `HDBSCAN` ranked `#3`

under the new recommendation-centered ranking rule.

That does not automatically force a baseline change by itself, but it is now strong evidence that the project should explicitly revisit whether `GMM` should remain the recommended operational default once evaluation, not just modeling flexibility, is treated as the decision driver.

## Scope completed

All items under `## P1 - Evaluation upgrades` in `docs/reports/implementation_todo_20260314.md` were completed and marked done:

- offline recommendation metrics now reflect the actual product behavior
- per-query and averaged `Precision@K` are implemented
- genre hit rate@K uses within-cluster full prepared-space ranking
- artist hit rate@K is implemented through explicit artist parsing in the evaluation pipeline
- catalog coverage and recommendation diversity checks were added
- silhouette / Calinski-Harabasz / Davies-Bouldin are diagnostics only
- stability rigor was increased
- the `50` runs at `80%` protocol is explicit in code and outputs
- per-cluster stability reporting was added
- internal diagnostics and cluster-size sanity checks are reported together
- a final recommendation-first comparison report is generated

## Why this required a new script

The todo referenced `scripts/analysis/evaluate_clustering.py`, but the checked workspace did not contain the source file anymore.

What existed:

- an old compiled artifact in `scripts/analysis/__pycache__/evaluate_clustering...pyc`

What did not exist:

- the editable source file itself

Because of that, this phase recreated the evaluation source from scratch instead of trying to patch a missing file.

## Main implementation

### 1. New evaluation entrypoint

Created:

- `scripts/analysis/evaluate_clustering.py`

The script now works directly from the current saved outputs:

- `output/clustering_results/audio_clustering_results_kmeans.csv`
- `output/clustering_results/audio_clustering_results_gmm.csv`
- `output/clustering_results/audio_clustering_results_hdbscan.csv`
- `output/clustering_results/audio_clustering_artifact_kmeans.npz`
- `output/clustering_results/audio_clustering_artifact_gmm.npz`
- `output/clustering_results/audio_clustering_artifact_hdbscan.npz`

This avoids reconstructing the wrong retrieval space and guarantees evaluation uses the same `30`-D prepared representation the UI now relies on.

### 2. Recommendation evaluation now mirrors product behavior

The new evaluation path intentionally matches the current UI logic:

- recommendations are constrained to the selected cluster
- distances are computed in the saved prepared feature space
- PCA-2 is not used for ranking
- HDBSCAN noise queries are treated as recommendation-disabled
- optional confidence / posterior filters are supported, but the default evaluation run uses:
  - `ranking_method=distance`
  - `min_confidence=0.0`
  - `min_posterior=0.0`

This means the evaluation is finally aligned to the actual product behavior instead of a geometry-only proxy.

### 3. Artist metadata is surfaced cleanly inside evaluation

The clustering features still do not use artist metadata.

For evaluation only, the script now explicitly parses:

- `QueryArtist`
- `QueryTitle`

from the saved `Song` field, which is consistently stored as:

- `Artist - Title`

This makes artist-based proxy metrics available without leaking artist information into clustering input.

### 4. Per-query and averaged recommendation metrics

For each evaluated method and each query track, the script now records:

- returned recommendation count
- genre hits at each `K`
- genre precision at each `K`
- genre hit rate at each `K`
- artist hits at each `K`
- artist precision at each `K`
- artist hit rate at each `K`
- whether the method returned a full list of size `K`
- the actual recommended item indices

The full run used:

- `K = 5`
- `K = 10`
- `K = 20`

Per-query outputs were written to:

- `output/metrics/evaluation_upgrades_20260314_per_query_kmeans.csv`
- `output/metrics/evaluation_upgrades_20260314_per_query_gmm.csv`
- `output/metrics/evaluation_upgrades_20260314_per_query_hdbscan.csv`

### 5. Coverage and diversity metrics added

To catch recommendation collapse, the script now reports:

- catalog coverage
- supported query fraction
- full-list fraction
- mean returned recommendations
- item exposure HHI
- item exposure diversity (`1 - HHI`)
- cluster exposure top share
- cluster exposure entropy

These are especially important for spotting cases where one huge cluster dominates exposure even if basic recommendation precision looks superficially acceptable.

### 6. Internal clustering metrics demoted to diagnostics

The script now computes:

- silhouette
- Calinski-Harabasz
- Davies-Bouldin

but keeps them in the output as diagnostics only.

They are not used as the primary ranking key for the final comparison.

This directly satisfies the todo item that internal indices should support diagnosis rather than drive final model selection on their own.

### 7. Stability protocol upgraded substantially

The final run used the explicit recommended protocol:

- `50` subsample runs
- `80%` sample fraction
- pairwise overlap comparison via `Adjusted Rand Index`

For seed-based methods, the script also added full-data repeated-seed reruns:

- `K-Means`: `10` seed runs
- `GMM`: `10` seed runs

For every method, the script now saves:

- run-level subsample summaries
- pairwise overlap ARI tables
- per-cluster stability tables

Generated outputs:

- `output/metrics/evaluation_upgrades_20260314_stability_runs_kmeans.csv`
- `output/metrics/evaluation_upgrades_20260314_stability_runs_gmm.csv`
- `output/metrics/evaluation_upgrades_20260314_stability_runs_hdbscan.csv`
- `output/metrics/evaluation_upgrades_20260314_stability_pairwise_kmeans.csv`
- `output/metrics/evaluation_upgrades_20260314_stability_pairwise_gmm.csv`
- `output/metrics/evaluation_upgrades_20260314_stability_pairwise_hdbscan.csv`
- `output/metrics/evaluation_upgrades_20260314_cluster_stability_kmeans.csv`
- `output/metrics/evaluation_upgrades_20260314_cluster_stability_gmm.csv`
- `output/metrics/evaluation_upgrades_20260314_cluster_stability_hdbscan.csv`
- `output/metrics/evaluation_upgrades_20260314_seed_stability_kmeans.csv`
- `output/metrics/evaluation_upgrades_20260314_seed_stability_gmm.csv`

### 8. Per-cluster stability is now explicit

Per-cluster stability is reported as best-match Jaccard against the reference clustering, plus the corresponding best-match precision and recall.

This is useful because global ARI can hide the fact that some clusters are stable while others are fragile.

### 9. Final comparison report is now recommendation-first

The comparison ranking is now ordered by:

1. genre precision@`10`
2. genre hit rate@`10`
3. artist precision@`10`
4. artist hit rate@`10`
5. catalog coverage
6. item exposure diversity
7. median subsample ARI
8. mean subsample ARI

Outputs:

- `output/metrics/evaluation_upgrades_20260314_comparison.csv`
- `output/metrics/evaluation_upgrades_20260314_comparison_report.md`
- `output/metrics/evaluation_upgrades_20260314_summary.json`

## Verification

Code verification:

- `py_compile` succeeded for `scripts/analysis/evaluate_clustering.py`

Functional verification:

- ran a smoke pass with reduced stability settings first
- removed the temporary smoke outputs after validation
- ran the full intended protocol successfully

Full run command:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/evaluate_clustering.py `
  --methods kmeans gmm hdbscan `
  --ks 5 10 20 `
  --prefix evaluation_upgrades_20260314 `
  --subsample-runs 50 `
  --subsample-fraction 0.8 `
  --seed-runs 10 `
  --stability-jobs 4
```

Observed wall time for the full run:

- about `131.5` seconds

## Measured results

### Final ranking at `K=10`

| Rank | Method | Genre Precision@10 | Genre Hit Rate@10 | Artist Precision@10 | Artist Hit Rate@10 | Catalog Coverage | Median Subsample ARI | Mean Subsample ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `K-Means` | 0.02748 | 0.20325 | 0.00381 | 0.03487 | 0.98591 | 0.97529 | 0.97151 |
| 2 | `GMM` | 0.02688 | 0.20163 | 0.00412 | 0.03758 | 0.98790 | 0.91390 | 0.79520 |
| 3 | `HDBSCAN` | 0.02649 | 0.19621 | 0.00367 | 0.03397 | 0.93984 | 0.24214 | 0.41923 |

### Recommendation-quality interpretation

`K-Means`:

- best overall rank
- best genre precision@`10`
- best genre hit rate@`10`
- perfect supported-query fraction
- perfect full-list fraction
- slightly lower artist metrics than `GMM`

`GMM`:

- second overall rank
- slightly worse genre metrics than `K-Means`
- best artist precision@`10`
- best artist hit rate@`10`
- best catalog coverage
- still lower recommendation-first score overall because genre metrics and stability were weaker

`HDBSCAN`:

- lowest rank
- worse recommendation quality than both `K-Means` and `GMM`
- only `95.34%` of queries receive any recommendations because noise queries are disabled
- cluster exposure is almost fully dominated by one giant cluster

### Recommendation breadth and operational behavior

At `K=10`:

| Method | Supported Query Fraction | Full List Fraction | Mean Returned | Cluster Exposure Top Share |
|---|---:|---:|---:|---:|
| `K-Means` | 1.00000 | 1.00000 | 10.00000 | 0.60939 |
| `GMM` | 1.00000 | 1.00000 | 10.00000 | 0.53767 |
| `HDBSCAN` | 0.95339 | 0.95194 | 9.52954 | 0.99894 |

Important interpretation:

- `K-Means` and `GMM` both serve full recommendation lists reliably
- `HDBSCAN` does not, because noise labeling disables recommendations for those queries
- `HDBSCAN` also collapses exposure almost entirely into one cluster, which is exactly the kind of failure the new coverage/diversity metrics were meant to reveal

### Internal diagnostics

| Method | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---:|---:|---:|
| `K-Means` | 0.26433 | 2124.34 | 1.43477 |
| `GMM` | 0.15276 | 919.92 | 2.12577 |
| `HDBSCAN` | 0.19943 | 11.85 | 1.19204 |

Important note:

- these metrics were intentionally not used as the primary ranking objective
- if they were used naively, they could obscure product-facing failures like HDBSCAN's recommendation disablement and exposure collapse

### Stability distribution details

Pairwise overlap ARI distribution across the `50` subsample runs:

| Method | Mean | Median | Min | P10 | P25 | P75 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `K-Means` | 0.97151 | 0.97529 | 0.89120 | 0.94734 | 0.95983 | 0.98531 | 0.99206 | 1.00000 |
| `GMM` | 0.79520 | 0.91390 | 0.32801 | 0.53402 | 0.59986 | 0.94134 | 0.95088 | 0.96969 |
| `HDBSCAN` | 0.41923 | 0.24214 | -0.04018 | -0.03020 | -0.00776 | 0.91183 | 0.94765 | 0.97794 |

Interpretation:

- `K-Means` is consistently stable with a tight high-ARI distribution
- `GMM` has a respectable median but a much weaker mean, which shows a long unstable tail across resamples
- `HDBSCAN` is highly unstable: some resamples look acceptable, but many diverge badly enough to drag the median down near `0.24`

### Reference-vs-full stability details

Subsample runs compared against the full saved clustering:

| Method | Mean Reference ARI | Median Reference ARI | Min Reference ARI |
|---|---:|---:|---:|
| `K-Means` | 0.98045 | 0.98287 | 0.93756 |
| `GMM` | 0.86504 | 0.94580 | 0.40128 |
| `HDBSCAN` | 0.57852 | 0.91612 | -0.03573 |

Interpretation:

- `GMM` again shows a wide tail even though its median still looks fairly good
- `HDBSCAN` is extremely inconsistent from run to run despite occasionally reproducing the large dominant cluster

## Per-cluster stability findings

### K-Means

| Cluster | Size | Mean Jaccard | Median Jaccard | Min Jaccard |
|---|---:|---:|---:|---:|
| `0` | 3373 | 0.99198 | 0.99300 | 0.97477 |
| `1` | 2162 | 0.98758 | 0.98905 | 0.95940 |

Interpretation:

- both `K-Means` clusters are very stable
- no cluster-specific weakness stood out

### GMM

| Cluster | Size | Mean Jaccard | Median Jaccard | Min Jaccard |
|---|---:|---:|---:|---:|
| `0` | 1346 | 0.94483 | 0.95966 | 0.81172 |
| `1` | 576 | 0.89463 | 0.93840 | 0.46787 |
| `2` | 2976 | 0.90720 | 0.96895 | 0.52261 |
| `3` | 637 | 0.84887 | 0.94721 | 0.35135 |

Interpretation:

- cluster `3` is the weakest cluster under the new stability protocol
- cluster `1` is also meaningfully less stable than the best GMM cluster
- the median values are better than the means, which again signals that instability is concentrated in a subset of resample runs rather than being evenly spread

### HDBSCAN

| Cluster | Size | Mean Jaccard | Median Jaccard | Min Jaccard |
|---|---:|---:|---:|---:|
| `0` | 8 | 0.37845 | 0.36364 | 0.00000 |
| `1` | 5269 | 0.80410 | 0.99305 | 0.36538 |

Interpretation:

- the tiny `8`-item cluster is not stable enough to trust
- even the dominant cluster shows a split personality:
  - many runs recover it very well
  - a meaningful subset of runs degrade badly enough to crater the global median pairwise ARI

## What changed operationally

Before this phase:

- the evaluation source file was missing
- there was no active recommendation-centered evaluation entrypoint in source control
- there was no final comparison report ranking methods by recommendation quality first
- there was no current `50 x 80%` stability protocol wired into the repo

After this phase:

- the evaluation source exists again
- the repo can now evaluate the current saved artifacts directly
- per-query recommendation metrics are persisted
- artist metrics are available without contaminating clustering features
- stability reporting is materially stronger
- comparison output is decision-ready

## Important conclusion for next decision point

The project's earlier baseline direction favored `GMM` for flexibility, soft assignment, and recommendation confidence.

The new evaluation evidence shows:

- `GMM` still has strengths:
  - best artist metrics
  - best catalog coverage
  - soft assignment information remains valuable
- but under the current recommendation-first proxy evaluation:
  - `K-Means` is slightly better on genre-oriented recommendation quality
  - `K-Means` is substantially more stable
  - `K-Means` now wins the overall comparison

That means the repo is now in a better position than before:

- the decision is no longer based mainly on modeling preference
- it is now based on saved, reproducible evaluation outputs

The next explicit choice should be:

- keep `GMM` as the preferred baseline because soft membership still matters enough to outweigh the observed stability gap, or
- promote `K-Means` as the new operational baseline under the current proxy-based evaluation regime

This report does not make that configuration change automatically.

It does make the tradeoff visible and measurable.
