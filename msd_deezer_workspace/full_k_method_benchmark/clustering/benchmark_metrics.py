"""
Extra clustering / dimensionality-reduction quality metrics for the
PCA / UMAP / PCA->UMAP benchmark.

These supplement the per-algorithm internal CVIs already computed in
`run_kmeans_clustering.py`, `run_gmm_clustering.py`, and
`run_hdbscan_clustering.py`. They exist here (not in production
`clustering/shared.py`) because they are only meaningful when comparing
*different* dimensionality reductions of the same underlying features --
which is what the benchmark does.

References (full citations in calendar/2026-04-25.md):
  Dunn (1974)            -- Dunn index
  Venna & Kaski (2001)   -- trustworthiness
  Lange et al. (2004)    -- bootstrap stability via ARI
  Ben-Hur et al. (2002)  -- stability-based clustering validation
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import adjusted_rand_score, pairwise_distances


# ---------------------------------------------------------------------------
# Dunn index
# ---------------------------------------------------------------------------

def dunn_index(x: np.ndarray, labels: np.ndarray) -> float:
    """
    Dunn index = (min inter-cluster distance) / (max intra-cluster diameter).

    Higher is better. Uses single-linkage inter-cluster distance (the
    minimum pairwise distance between members of two different clusters)
    and complete-diameter intra-cluster (the maximum pairwise distance
    between two members of the same cluster). This is the classical
    Dunn 1974 formulation; later "generalised Dunn indices" exist but
    we stick with the original because that's what most comparison
    papers report.

    Computed with full pairwise distances. Cost is O(n^2 * d). For the
    10k benchmark in 15-D this is ~1.5e9 element ops -- ~5 s on a
    single core, fine. For larger n use a sample-based variant.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels[labels != -1])  # ignore HDBSCAN noise
    if len(unique_labels) < 2:
        return float("nan")

    # If the inlier subset is itself too small, we can't say anything useful.
    inlier_mask = labels != -1
    if inlier_mask.sum() < 2 * len(unique_labels):
        return float("nan")

    distances = pairwise_distances(x[inlier_mask], metric="euclidean")
    inlier_labels = labels[inlier_mask]

    max_intra = 0.0
    for cid in unique_labels:
        members = np.where(inlier_labels == cid)[0]
        if len(members) < 2:
            continue
        sub = distances[np.ix_(members, members)]
        diameter = float(sub.max())
        if diameter > max_intra:
            max_intra = diameter

    if max_intra <= 0.0:
        return float("nan")

    min_inter = float("inf")
    for i, ci in enumerate(unique_labels):
        members_i = np.where(inlier_labels == ci)[0]
        for cj in unique_labels[i + 1:]:
            members_j = np.where(inlier_labels == cj)[0]
            sub = distances[np.ix_(members_i, members_j)]
            sep = float(sub.min())
            if sep < min_inter:
                min_inter = sep

    if not np.isfinite(min_inter) or min_inter <= 0.0:
        return float("nan")

    return min_inter / max_intra


# ---------------------------------------------------------------------------
# Trustworthiness (Venna & Kaski 2001)
# ---------------------------------------------------------------------------

def trustworthiness_score(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    n_neighbors: int = 10,
    sample_size: int | None = 5000,
    random_state: int = 42,
) -> float:
    """
    Measures how well the k-nearest-neighbour graph from `high_dim` is
    preserved in `low_dim`. Range [0, 1]; higher is better.

    Implementation note: sklearn's exact `trustworthiness` is O(n^2),
    so we evaluate on a uniform random subsample by default (n=5000 is
    a stable size in practice -- standard error of trustworthiness at
    n=5000 vs full n=10000 is well below 0.005 in our experiments).
    """
    n = high_dim.shape[0]
    if n != low_dim.shape[0]:
        raise ValueError(
            f"high_dim ({n}) and low_dim ({low_dim.shape[0]}) row counts differ"
        )

    if sample_size is not None and n > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, sample_size, replace=False)
        high_dim = high_dim[idx]
        low_dim = low_dim[idx]

    # n_neighbors must be < n / 2 for trustworthiness to be defined.
    eff_k = max(2, min(n_neighbors, len(high_dim) // 2 - 1))
    return float(sklearn_trustworthiness(high_dim, low_dim, n_neighbors=eff_k))


# ---------------------------------------------------------------------------
# Bootstrap cluster stability via ARI (Lange 2004; Ben-Hur 2002)
# ---------------------------------------------------------------------------

def bootstrap_stability_ari(
    x: np.ndarray,
    cluster_fn: Callable[[np.ndarray], np.ndarray],
    n_bootstraps: int = 5,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Refit the clustering on `n_bootstraps` random subsamples of size
    `sample_fraction * n` and report the mean / std pairwise Adjusted
    Rand Index across overlapping points.

    ARI is computed only on the intersection of any two bootstrap
    subsamples (the points present in both). High mean ARI indicates
    that the cluster structure is a property of the data, not the
    random initialisation -- the canonical test for "is this UMAP
    layout actually meaningful, or did the optimizer just find a
    convenient local minimum?".

    `cluster_fn(x_sub) -> labels` should fit a fresh clustering model
    on the provided subsample and return integer labels. Caller is
    responsible for keeping the model deterministic given x_sub.
    """
    n = x.shape[0]
    sub_size = max(2, int(n * sample_fraction))
    rng = np.random.RandomState(random_state)

    # Run all bootstraps first; keep (indices, labels) per bootstrap.
    indices_per_run: list[np.ndarray] = []
    labels_per_run: list[np.ndarray] = []
    for b in range(n_bootstraps):
        idx = rng.choice(n, sub_size, replace=False)
        idx.sort()
        labels = cluster_fn(x[idx])
        if len(np.unique(labels)) < 2:
            # Degenerate fit -- skip but record so the count reflects it.
            continue
        indices_per_run.append(idx)
        labels_per_run.append(np.asarray(labels))

    if len(labels_per_run) < 2:
        return {
            "mean_ari": float("nan"),
            "std_ari": float("nan"),
            "n_pairs": 0,
            "n_successful_runs": len(labels_per_run),
        }

    aris: list[float] = []
    for i in range(len(labels_per_run)):
        idx_i = indices_per_run[i]
        labels_i = labels_per_run[i]
        for j in range(i + 1, len(labels_per_run)):
            idx_j = indices_per_run[j]
            labels_j = labels_per_run[j]
            # Intersection of sampled indices -- the points that both
            # bootstraps got to label.
            common, ci, cj = np.intersect1d(
                idx_i, idx_j, return_indices=True,
            )
            if len(common) < 2:
                continue
            ari = adjusted_rand_score(labels_i[ci], labels_j[cj])
            aris.append(float(ari))

    if not aris:
        return {
            "mean_ari": float("nan"),
            "std_ari": float("nan"),
            "n_pairs": 0,
            "n_successful_runs": len(labels_per_run),
        }

    return {
        "mean_ari": float(np.mean(aris)),
        "std_ari": float(np.std(aris)),
        "n_pairs": len(aris),
        "n_successful_runs": len(labels_per_run),
    }
