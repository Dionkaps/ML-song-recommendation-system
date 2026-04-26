# Dim-Reduction Benchmark — Comparison Report

Each row is one (feature source, reduction, algorithm) clustering run.
Within every (feature source, algorithm) group the three reductions are ranked per metric; `overall_rank` is the mean of those per-metric ranks (lower is better).

## Winners per algorithm (lower overall_rank = better)

| feature_source | algorithm | best_reduction | overall_rank | silhouette | davies_bouldin | dunn | trustworthiness | stability_ari | k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| features | gmm | pca_only | 1.3330 | 0.0989 |  | 0.1160 | 0.9999 | 0.9751 | 2 |
| features | hdbscan | pca_then_umap | 1.5000 | 0.5673 |  | 0.0524 | 0.9049 | 0.4916 | 7 |
| features | kmeans | pca_then_umap | 1.5000 | 0.4614 |  | 0.0155 | 0.9049 | 0.9873 | 3 |
| pretrained_embeddings | gmm | pca_only | 1.3330 | 0.0834 |  | 0.1457 | 0.9998 | 0.9268 | 2 |
| pretrained_embeddings | hdbscan | pca_then_umap | 1.3330 | 0.5879 |  | 0.0554 | 0.9504 | 0.8429 | 17 |
| pretrained_embeddings | kmeans | pca_then_umap | 1.6670 | 0.4712 |  | 0.0062 | 0.9504 | 0.9964 | 3 |


## Full results

| feature_source | reduction | algorithm | cluster_count | silhouette_score | calinski_harabasz_score | davies_bouldin_score | dunn_index | trustworthiness | stability_mean_ari | overall_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| features | pca_only | gmm | 2 | 0.0989 |  |  | 0.1160 | 0.9999 | 0.9751 | 1.3333 |
| features | pca_then_umap | gmm | 10 | 0.2402 |  |  | 0.0122 | 0.9049 | 0.8174 | 2.0000 |
| features | umap_only | gmm | 10 | 0.2454 |  |  | 0.0128 | 0.9040 | 0.8267 | 1.6667 |
| features | pca_only | hdbscan | 0 |  |  |  |  | 0.9999 |  | 2.0000 |
| features | pca_then_umap | hdbscan | 7 | 0.5673 |  |  | 0.0524 | 0.9049 | 0.4916 | 1.5000 |
| features | umap_only | hdbscan | 14 | 0.5232 |  |  | 0.0780 | 0.9040 | 0.6444 | 1.5000 |
| features | pca_only | kmeans | 2 | 0.1542 |  |  | 0.1782 | 0.9999 | 0.9820 | 1.6667 |
| features | pca_then_umap | kmeans | 3 | 0.4614 |  |  | 0.0155 | 0.9049 | 0.9873 | 1.5000 |
| features | umap_only | kmeans | 3 | 0.4591 |  |  | 0.0123 | 0.9040 | 0.9925 | 1.8333 |
| pretrained_embeddings | pca_only | gmm | 2 | 0.0834 |  |  | 0.1457 | 0.9998 | 0.9268 | 1.3333 |
| pretrained_embeddings | pca_then_umap | gmm | 15 | 0.3866 |  |  | 0.0111 | 0.9504 | 0.8460 | 1.5000 |
| pretrained_embeddings | umap_only | gmm | 15 | 0.3865 |  |  | 0.0107 | 0.9472 | 0.8305 | 2.1667 |
| pretrained_embeddings | pca_only | hdbscan | 7 | 0.3044 |  |  | 0.4000 | 0.9998 | 0.7807 | 1.5000 |
| pretrained_embeddings | pca_then_umap | hdbscan | 17 | 0.5879 |  |  | 0.0554 | 0.9504 | 0.8429 | 1.3333 |
| pretrained_embeddings | umap_only | hdbscan | 33 | 0.5834 |  |  | 0.0466 | 0.9472 | 0.7601 | 2.1667 |
| pretrained_embeddings | pca_only | kmeans | 3 | 0.1032 |  |  | 0.1995 | 0.9998 | 0.9281 | 1.6667 |
| pretrained_embeddings | pca_then_umap | kmeans | 3 | 0.4712 |  |  | 0.0062 | 0.9504 | 0.9964 | 1.6667 |
| pretrained_embeddings | umap_only | kmeans | 5 | 0.4893 |  |  | 0.0103 | 0.9472 | 0.9922 | 1.6667 |


## Diagnostic: trustworthiness vs internal-CVI agreement

| feature_source | algorithm | reduction | rank_silhouette | rank_trustworthiness | agreement |
| --- | --- | --- | --- | --- | --- |
| features | gmm | pca_then_umap | 2 | 2 | agree |
| features | gmm | umap_only | 1 | 3 | disagree |
| features | gmm | pca_only | 3 | 1 | disagree |
| features | hdbscan | pca_then_umap | 1 | 2 | disagree |
| features | hdbscan | umap_only | 2 | 3 | disagree |
| features | hdbscan | pca_only | 3 | 1 | disagree |
| features | kmeans | pca_then_umap | 1 | 2 | disagree |
| features | kmeans | umap_only | 2 | 3 | disagree |
| features | kmeans | pca_only | 3 | 1 | disagree |
| pretrained_embeddings | gmm | pca_then_umap | 1 | 2 | disagree |
| pretrained_embeddings | gmm | umap_only | 2 | 3 | disagree |
| pretrained_embeddings | gmm | pca_only | 3 | 1 | disagree |
| pretrained_embeddings | hdbscan | pca_then_umap | 1 | 2 | disagree |
| pretrained_embeddings | hdbscan | umap_only | 2 | 3 | disagree |
| pretrained_embeddings | hdbscan | pca_only | 3 | 1 | disagree |
| pretrained_embeddings | kmeans | pca_then_umap | 2 | 2 | agree |
| pretrained_embeddings | kmeans | umap_only | 1 | 3 | disagree |
| pretrained_embeddings | kmeans | pca_only | 3 | 1 | disagree |
